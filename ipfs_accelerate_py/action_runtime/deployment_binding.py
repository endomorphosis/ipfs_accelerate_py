"""Signed deployment binding gate for production action execute.

Authority-plane production execute is admitted only when the live catalog
digest and adapter identities match an operator-signed deployment binding.
A confirmation flag cannot override a digest or identity mismatch.

Content-plane artifacts never carry binding payloads, signing keys, or
executable locators. Importing this module starts no processes and loads no
credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .catalog import ActionCatalog, ActionDescriptor
from .catalog_211ai import (
    CATALOG_ID,
    catalog_digest as pilot_catalog_digest,
    pilot_descriptors,
)
from .contracts import content_digest

SCHEMA: str = "voice-action/deployment-binding@1"
SIGNATURE_ALGORITHM: str = "hmac-sha256"
BINDING_VERSION: str = "1"

# Production execute is the only environment that requires a live signed match
# before side effects. Pilot/test may still use the same verifier for CI.
PRODUCTION_ENVIRONMENT: str = "production"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NONCE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_IDENTITY_TOKEN = re.compile(r"^[A-Za-z0-9_.:/=+-]{1,512}$")

# Stable interface-identity templates for the offline pilot adapter surface.
# Product/wallet bindings may pin different identities; the signed binding is
# the source of truth at execute time.
_PILOT_INTERFACE_FAMILY: Mapping[str, str] = {
    "handoff_live_agent": "human_handoff",
    "escalate_safety": "human_handoff",
    "open_app_surface": "app_surface",
    "open_wallet_documents": "app_surface",
    "read_calendar": "calendar",
    "create_calendar_reminder": "calendar",
    "read_provider_messages": "messaging",
    "leave_provider_message": "messaging",
    "open_service_detail": "service_interaction",
    "schedule_service_callback": "service_interaction",
}


@dataclass(frozen=True)
class AdapterIdentity:
    """Operator-reviewed identity of an admitted adapter for one descriptor.

    Identities are content-addressable and contain no executable locators,
    credentials, or network endpoints. The ``interface_identity`` string is
    the stable handle adapters publish on receipts.
    """

    descriptor_id: str
    logical_action: str
    adapter: str
    interface_identity: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("descriptor_id", self.descriptor_id),
            ("logical_action", self.logical_action),
            ("adapter", self.adapter),
            ("interface_identity", self.interface_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
            if not _IDENTITY_TOKEN.fullmatch(value):
                raise ValueError(f"{field_name} has disallowed characters: {value!r}")
            lowered = value.lower()
            # Fail closed: binding identities never smuggle executable locators
            # or credential material (content plane must not supply these).
            banned_fragments = (
                "command",
                "argv",
                "executable",
                "import_path",
                "credential",
                "secret",
            )
            if lowered.endswith("_path") or any(m in lowered for m in banned_fragments):
                raise ValueError(
                    f"{field_name} rejects locator/credential token: {value!r}"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter": self.adapter,
            "descriptor_id": self.descriptor_id,
            "interface_identity": self.interface_identity,
            "logical_action": self.logical_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterIdentity":
        if not isinstance(payload, Mapping):
            raise TypeError("adapter identity must be a mapping")
        return cls(
            descriptor_id=str(payload.get("descriptor_id") or ""),
            logical_action=str(payload.get("logical_action") or ""),
            adapter=str(payload.get("adapter") or ""),
            interface_identity=str(payload.get("interface_identity") or ""),
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())


def normalize_adapter_identities(
    identities: Sequence[AdapterIdentity] | Iterable[AdapterIdentity],
) -> tuple[AdapterIdentity, ...]:
    """Return identities sorted by descriptor_id; reject duplicates."""

    rows = list(identities)
    by_id: dict[str, AdapterIdentity] = {}
    for identity in rows:
        if not isinstance(identity, AdapterIdentity):
            raise TypeError("adapter identities must be AdapterIdentity instances")
        if identity.descriptor_id in by_id:
            raise ValueError(
                f"duplicate adapter identity for {identity.descriptor_id!r}"
            )
        by_id[identity.descriptor_id] = identity
    ordered = tuple(by_id[key] for key in sorted(by_id))
    return ordered


def adapter_identities_digest(
    identities: Sequence[AdapterIdentity] | Iterable[AdapterIdentity],
) -> str:
    """Stable digest over the normalized adapter-identity set."""

    ordered = normalize_adapter_identities(identities)
    return content_digest([row.to_dict() for row in ordered])


def pilot_interface_identity(
    logical_action: str,
    *,
    descriptor_id: str,
) -> str:
    """Build the default offline pilot interface identity for a logical action."""

    family = _PILOT_INTERFACE_FAMILY.get(logical_action)
    if family is None:
        raise KeyError(f"no pilot interface family for logical_action {logical_action!r}")
    return f"{family}:{logical_action}:{descriptor_id}"


def adapter_identity_from_descriptor(
    descriptor: ActionDescriptor,
    *,
    interface_identity: str | None = None,
) -> AdapterIdentity:
    """Project a catalog descriptor into a bindable adapter identity."""

    identity = interface_identity
    if identity is None:
        identity = pilot_interface_identity(
            descriptor.logical_action,
            descriptor_id=descriptor.descriptor_id,
        )
    return AdapterIdentity(
        descriptor_id=descriptor.descriptor_id,
        logical_action=descriptor.logical_action,
        adapter=descriptor.adapter,
        interface_identity=identity,
    )


def adapter_identities_from_catalog(
    catalog: ActionCatalog | Sequence[ActionDescriptor],
    *,
    interface_identities: Mapping[str, str] | None = None,
) -> tuple[AdapterIdentity, ...]:
    """Build adapter identities for every descriptor in a catalog."""

    if isinstance(catalog, ActionCatalog):
        descriptors = [catalog.require(descriptor_id) for descriptor_id in catalog.list_ids()]
    else:
        descriptors = list(catalog)

    overrides = dict(interface_identities or {})
    rows: list[AdapterIdentity] = []
    for descriptor in descriptors:
        override = overrides.get(descriptor.descriptor_id)
        rows.append(
            adapter_identity_from_descriptor(
                descriptor,
                interface_identity=override,
            )
        )
    return normalize_adapter_identities(rows)


def pilot_adapter_identities() -> tuple[AdapterIdentity, ...]:
    """Adapter identities for the reviewed 211-AI pilot catalog."""

    return adapter_identities_from_catalog(pilot_descriptors())


@dataclass(frozen=True)
class SignedDeploymentBinding:
    """Operator-signed pin of catalog digest + adapter identities.

    The signature covers every public binding field except ``signature`` itself.
    Unsigned or invalid bindings never admit production execute.
    """

    binding_id: str
    catalog_id: str
    catalog_digest: str
    adapter_identities: tuple[AdapterIdentity, ...]
    environment: str = PRODUCTION_ENVIRONMENT
    schema: str = SCHEMA
    version: str = BINDING_VERSION
    issuer: str = "operator"
    signature_algorithm: str = SIGNATURE_ALGORITHM
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    signature: str | None = None
    issued_at_epoch_s: float | None = None
    expires_at_epoch_s: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.binding_id or not isinstance(self.binding_id, str):
            raise ValueError("binding_id is required")
        if not self.catalog_id:
            raise ValueError("catalog_id is required")
        if not isinstance(self.catalog_digest, str) or not _HEX_SHA256.fullmatch(
            self.catalog_digest
        ):
            raise ValueError("catalog_digest must be a 64-char lowercase sha256 hex digest")
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported binding schema {self.schema!r}")
        if self.signature_algorithm != SIGNATURE_ALGORITHM:
            raise ValueError(
                f"unsupported signature algorithm {self.signature_algorithm!r}"
            )
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise ValueError("nonce is invalid")
        if self.environment not in {"production", "pilot", "test"}:
            raise ValueError(f"unsupported environment {self.environment!r}")
        ordered = normalize_adapter_identities(self.adapter_identities)
        object.__setattr__(self, "adapter_identities", ordered)
        if not ordered:
            raise ValueError("adapter_identities must be non-empty")
        meta = {str(k): str(v) for k, v in dict(self.metadata).items()}
        object.__setattr__(self, "metadata", meta)
        if self.signature is not None:
            if not isinstance(self.signature, str) or not _HEX_SHA256.fullmatch(
                self.signature
            ):
                raise ValueError("signature must be a 64-char lowercase sha256 hex digest")

    @property
    def adapter_identities_digest(self) -> str:
        return adapter_identities_digest(self.adapter_identities)

    @property
    def binding_digest(self) -> str:
        return content_digest(self.signing_dict())

    def signing_dict(self) -> dict[str, Any]:
        """Return every field covered by the signature (no signature field)."""

        return {
            "adapter_identities": [row.to_dict() for row in self.adapter_identities],
            "adapter_identities_digest": self.adapter_identities_digest,
            "binding_id": self.binding_id,
            "catalog_digest": self.catalog_digest,
            "catalog_id": self.catalog_id,
            "environment": self.environment,
            "expires_at_epoch_s": self.expires_at_epoch_s,
            "issued_at_epoch_s": self.issued_at_epoch_s,
            "issuer": self.issuer,
            "metadata": dict(sorted(self.metadata.items())),
            "nonce": self.nonce,
            "schema": self.schema,
            "signature_algorithm": self.signature_algorithm,
            "version": self.version,
        }

    def signing_payload(self) -> bytes:
        # Mirror contracts.content_digest canonicalization for HMAC input.
        import json

        return json.dumps(
            self.signing_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        payload = self.signing_dict()
        payload["signature"] = self.signature
        payload["binding_digest"] = self.binding_digest
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SignedDeploymentBinding":
        if not isinstance(payload, Mapping):
            raise TypeError("deployment binding payload must be a mapping")
        raw_identities = payload.get("adapter_identities")
        if not isinstance(raw_identities, list) or not raw_identities:
            raise ValueError("adapter_identities must be a non-empty list")
        identities = tuple(AdapterIdentity.from_dict(row) for row in raw_identities)
        expected_id_digest = payload.get("adapter_identities_digest")
        computed_id_digest = adapter_identities_digest(identities)
        if (
            expected_id_digest is not None
            and str(expected_id_digest) not in {"", "PENDING"}
            and expected_id_digest != computed_id_digest
        ):
            raise ValueError(
                "adapter_identities_digest mismatch: "
                f"expected {expected_id_digest}, got {computed_id_digest}"
            )
        metadata_raw = payload.get("metadata") or {}
        if not isinstance(metadata_raw, Mapping):
            raise TypeError("metadata must be a mapping")
        return cls(
            binding_id=str(payload.get("binding_id") or ""),
            catalog_id=str(payload.get("catalog_id") or ""),
            catalog_digest=str(payload.get("catalog_digest") or ""),
            adapter_identities=identities,
            environment=str(payload.get("environment") or PRODUCTION_ENVIRONMENT),
            schema=str(payload.get("schema") or SCHEMA),
            version=str(payload.get("version") or BINDING_VERSION),
            issuer=str(payload.get("issuer") or "operator"),
            signature_algorithm=str(
                payload.get("signature_algorithm") or SIGNATURE_ALGORITHM
            ),
            nonce=str(payload.get("nonce") or secrets.token_urlsafe(24)),
            signature=(
                str(payload["signature"])
                if payload.get("signature") is not None
                else None
            ),
            issued_at_epoch_s=_optional_float(payload.get("issued_at_epoch_s")),
            expires_at_epoch_s=_optional_float(payload.get("expires_at_epoch_s")),
            metadata={str(k): str(v) for k, v in metadata_raw.items()},
        )

    def is_expired_at(self, now_epoch_s: float) -> bool:
        if self.expires_at_epoch_s is None:
            return False
        return float(now_epoch_s) >= float(self.expires_at_epoch_s)

    def with_signature(self, signature: str) -> "SignedDeploymentBinding":
        return SignedDeploymentBinding(
            binding_id=self.binding_id,
            catalog_id=self.catalog_id,
            catalog_digest=self.catalog_digest,
            adapter_identities=self.adapter_identities,
            environment=self.environment,
            schema=self.schema,
            version=self.version,
            issuer=self.issuer,
            signature_algorithm=self.signature_algorithm,
            nonce=self.nonce,
            signature=signature,
            issued_at_epoch_s=self.issued_at_epoch_s,
            expires_at_epoch_s=self.expires_at_epoch_s,
            metadata=self.metadata,
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_key(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        material = key
    elif isinstance(key, str):
        material = key.encode("utf-8")
    else:
        raise TypeError("operator key must be bytes or str")
    if not material:
        raise ValueError("operator key must not be empty")
    return material


def sign_deployment_binding(
    binding: SignedDeploymentBinding,
    operator_key: bytes | str,
    *,
    rotate_nonce: bool = False,
) -> SignedDeploymentBinding:
    """Return a copy of ``binding`` with a fresh HMAC-SHA256 signature."""

    working = binding
    if rotate_nonce:
        working = SignedDeploymentBinding(
            binding_id=binding.binding_id,
            catalog_id=binding.catalog_id,
            catalog_digest=binding.catalog_digest,
            adapter_identities=binding.adapter_identities,
            environment=binding.environment,
            schema=binding.schema,
            version=binding.version,
            issuer=binding.issuer,
            signature_algorithm=binding.signature_algorithm,
            nonce=secrets.token_urlsafe(24),
            signature=None,
            issued_at_epoch_s=binding.issued_at_epoch_s,
            expires_at_epoch_s=binding.expires_at_epoch_s,
            metadata=binding.metadata,
        )
    signature = hmac.new(
        _coerce_key(operator_key),
        working.signing_payload(),
        hashlib.sha256,
    ).hexdigest()
    return working.with_signature(signature)


def verify_deployment_binding_signature(
    binding: SignedDeploymentBinding,
    operator_key: bytes | str,
) -> bool:
    """Constant-time verify of the binding HMAC. Missing/invalid → False."""

    if not isinstance(binding.signature, str) or not _HEX_SHA256.fullmatch(
        binding.signature
    ):
        return False
    expected = hmac.new(
        _coerce_key(operator_key),
        binding.signing_payload(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(binding.signature, expected)


def build_signed_pilot_binding(
    operator_key: bytes | str,
    *,
    binding_id: str = "voice-action-pilot-binding-v1",
    environment: str = PRODUCTION_ENVIRONMENT,
    issuer: str = "operator",
    issued_at_epoch_s: float | None = None,
    expires_at_epoch_s: float | None = None,
    interface_identities: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> SignedDeploymentBinding:
    """Construct and sign a binding for the current pilot catalog digest."""

    identities = adapter_identities_from_catalog(
        pilot_descriptors(),
        interface_identities=interface_identities,
    )
    now = time.time() if issued_at_epoch_s is None else float(issued_at_epoch_s)
    unsigned = SignedDeploymentBinding(
        binding_id=binding_id,
        catalog_id=CATALOG_ID,
        catalog_digest=pilot_catalog_digest(),
        adapter_identities=identities,
        environment=environment,
        issuer=issuer,
        issued_at_epoch_s=now,
        expires_at_epoch_s=expires_at_epoch_s,
        metadata=dict(metadata or {}),
    )
    return sign_deployment_binding(unsigned, operator_key)


@dataclass(frozen=True)
class DeploymentBindingVerdict:
    """Fail-closed result of a production execute binding check."""

    admitted: bool
    reason: str
    binding_id: str | None = None
    environment: str | None = None
    expected_catalog_digest: str | None = None
    actual_catalog_digest: str | None = None
    expected_adapter_identities_digest: str | None = None
    actual_adapter_identities_digest: str | None = None
    confirmed: bool = False
    signature_valid: bool = False

    @property
    def permits_execution(self) -> bool:
        return self.admitted

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "actual_adapter_identities_digest": self.actual_adapter_identities_digest,
            "actual_catalog_digest": self.actual_catalog_digest,
            "binding_id": self.binding_id,
            "confirmed": self.confirmed,
            "environment": self.environment,
            "expected_adapter_identities_digest": self.expected_adapter_identities_digest,
            "expected_catalog_digest": self.expected_catalog_digest,
            "permits_execution": self.permits_execution,
            "reason": self.reason,
            "signature_valid": self.signature_valid,
        }


def _identity_map(
    identities: Sequence[AdapterIdentity] | Mapping[str, AdapterIdentity],
) -> dict[str, AdapterIdentity]:
    if isinstance(identities, Mapping):
        rows = list(identities.values())
    else:
        rows = list(identities)
    ordered = normalize_adapter_identities(rows)
    return {row.descriptor_id: row for row in ordered}


def compare_adapter_identities(
    expected: Sequence[AdapterIdentity] | Mapping[str, AdapterIdentity],
    actual: Sequence[AdapterIdentity] | Mapping[str, AdapterIdentity],
) -> str | None:
    """Return a mismatch reason, or ``None`` when sets are identical."""

    expected_map = _identity_map(expected)
    actual_map = _identity_map(actual)
    if set(expected_map) != set(actual_map):
        missing = sorted(set(expected_map) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected_map))
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        return "adapter_identity_set_mismatch:" + ",".join(parts)
    for descriptor_id in sorted(expected_map):
        exp = expected_map[descriptor_id]
        act = actual_map[descriptor_id]
        if exp.to_dict() != act.to_dict():
            return f"adapter_identity_mismatch:{descriptor_id}"
    return None


def gate_production_execute(
    binding: SignedDeploymentBinding,
    *,
    operator_key: bytes | str,
    runtime_catalog_digest: str,
    runtime_adapter_identities: Sequence[AdapterIdentity]
    | Mapping[str, AdapterIdentity],
    confirmed: bool = False,
    now_epoch_s: float | None = None,
    require_production_environment: bool = True,
) -> DeploymentBindingVerdict:
    """Admit production execute only when the signed binding matches runtime.

    Confirmation is intentionally non-authoritative here: a confirmed caller
    still cannot execute when the catalog digest or any adapter identity
    diverges from the operator-signed binding.
    """

    now = time.time() if now_epoch_s is None else float(now_epoch_s)
    actual_ids = normalize_adapter_identities(
        list(_identity_map(runtime_adapter_identities).values())
    )
    actual_id_digest = adapter_identities_digest(actual_ids)
    base_kwargs: dict[str, Any] = {
        "binding_id": binding.binding_id,
        "environment": binding.environment,
        "expected_catalog_digest": binding.catalog_digest,
        "actual_catalog_digest": runtime_catalog_digest,
        "expected_adapter_identities_digest": binding.adapter_identities_digest,
        "actual_adapter_identities_digest": actual_id_digest,
        "confirmed": bool(confirmed),
    }

    signature_valid = verify_deployment_binding_signature(binding, operator_key)
    if not signature_valid:
        return DeploymentBindingVerdict(
            admitted=False,
            reason="invalid_or_missing_binding_signature",
            signature_valid=False,
            **base_kwargs,
        )

    if require_production_environment and binding.environment != PRODUCTION_ENVIRONMENT:
        return DeploymentBindingVerdict(
            admitted=False,
            reason="binding_environment_not_production",
            signature_valid=True,
            **base_kwargs,
        )

    if binding.is_expired_at(now):
        return DeploymentBindingVerdict(
            admitted=False,
            reason="binding_expired",
            signature_valid=True,
            **base_kwargs,
        )

    if not isinstance(runtime_catalog_digest, str) or not _HEX_SHA256.fullmatch(
        runtime_catalog_digest
    ):
        return DeploymentBindingVerdict(
            admitted=False,
            reason="runtime_catalog_digest_invalid",
            signature_valid=True,
            **base_kwargs,
        )

    if runtime_catalog_digest != binding.catalog_digest:
        # Confirm cannot override catalog pin.
        return DeploymentBindingVerdict(
            admitted=False,
            reason="catalog_digest_mismatch",
            signature_valid=True,
            **base_kwargs,
        )

    mismatch = compare_adapter_identities(binding.adapter_identities, actual_ids)
    if mismatch is not None:
        # Confirm cannot override adapter identity pin.
        return DeploymentBindingVerdict(
            admitted=False,
            reason=mismatch,
            signature_valid=True,
            **base_kwargs,
        )

    return DeploymentBindingVerdict(
        admitted=True,
        reason="binding_match_confirmed" if confirmed else "binding_match",
        signature_valid=True,
        **base_kwargs,
    )


def require_production_execute(
    binding: SignedDeploymentBinding,
    *,
    operator_key: bytes | str,
    runtime_catalog_digest: str,
    runtime_adapter_identities: Sequence[AdapterIdentity]
    | Mapping[str, AdapterIdentity],
    confirmed: bool = False,
    now_epoch_s: float | None = None,
) -> DeploymentBindingVerdict:
    """Same as :func:`gate_production_execute` but raises on denial.

    Useful for executor entry points that prefer exceptions over soft deny
    receipts. Confirmation still cannot override a binding mismatch.
    """

    verdict = gate_production_execute(
        binding,
        operator_key=operator_key,
        runtime_catalog_digest=runtime_catalog_digest,
        runtime_adapter_identities=runtime_adapter_identities,
        confirmed=confirmed,
        now_epoch_s=now_epoch_s,
    )
    if not verdict.admitted:
        raise DeploymentBindingError(verdict)
    return verdict


class DeploymentBindingError(PermissionError):
    """Raised when production execute is denied by the deployment binding gate."""

    def __init__(self, verdict: DeploymentBindingVerdict) -> None:
        self.verdict = verdict
        super().__init__(verdict.reason)


__all__ = [
    "BINDING_VERSION",
    "AdapterIdentity",
    "DeploymentBindingError",
    "DeploymentBindingVerdict",
    "PRODUCTION_ENVIRONMENT",
    "SCHEMA",
    "SIGNATURE_ALGORITHM",
    "SignedDeploymentBinding",
    "adapter_identities_digest",
    "adapter_identities_from_catalog",
    "adapter_identity_from_descriptor",
    "build_signed_pilot_binding",
    "compare_adapter_identities",
    "gate_production_execute",
    "normalize_adapter_identities",
    "pilot_adapter_identities",
    "pilot_interface_identity",
    "require_production_execute",
    "sign_deployment_binding",
    "verify_deployment_binding_signature",
]
