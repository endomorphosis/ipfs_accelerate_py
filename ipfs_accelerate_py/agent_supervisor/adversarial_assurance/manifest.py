"""Create sealed assurance manifests from released authorities (AAE-039).

Interface surface:

* ``create_assurance_manifest`` — bind repository state and verification policy
  to exact released authority status mappings and produce a content-addressed
  ``AssuranceManifest@1``.

This module does **not** change production policy, open a store, mutate
worktrees, or invent missing sealer/capability success. Missing or drifted
authority is recorded as ``typed_unavailable``.

Cold import is side-effect free.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.adapters import (
    AAE_RUNTIME_ADAPTERS_EVIDENCE,
    AUTHORITY_KEYS,
    AssuranceCapabilityUnavailable,
    AssuranceRuntimeAdapters,
    AuthorityStatus,
    CapabilityReason,
    SealStatus,
    load_runtime_adapters,
    probe_all_authorities,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    AssuranceBaseError,
    AssuranceTerminalStatus,
    MODEL_AUTHORITY_FORBIDDEN_KEYS,
    PRIVATE_FIELD_MARKERS,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.detection import (
    DETECTION_ASSURANCE_MANIFEST_INTERFACE,
    DETECTION_ASSURANCE_MANIFEST_SCHEMA,
    DetectionAssuranceManifest,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

CREATE_ASSURANCE_MANIFEST_INTERFACE: Final[str] = "create_assurance_manifest@1"
ASSURANCE_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-manifest@1"
)
ASSURANCE_MANIFEST_INTERFACE: Final[str] = "AssuranceManifest@1"
REPOSITORY_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-repository-state@1"
)
VERIFICATION_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-verification-policy@1"
)

GENERATOR_ID: Final[str] = "assurance_manifest"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_DETECTORS: Final[int] = 4_096
MAX_EDGES: Final[int] = 16_384
MAX_CLAIMS: Final[int] = 4_096

_REPOSITORY_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}$"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AssuranceManifestError(ValueError):
    """Raised when manifest inputs are malformed before a sealed result."""

    def __init__(self, message: str, *, reason_code: str = "malformed_input") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "malformed_input")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AssuranceManifestError(f"{name} must be a string")
    text = unicodedata.normalize("NFC", value)
    if not empty and not text.strip():
        raise AssuranceManifestError(f"{name} must not be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise AssuranceManifestError(f"{name} exceeds maximum length")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=True) or None


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise AssuranceManifestError(f"{name} must be a boolean")
    return value


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise AssuranceManifestError(f"{name} must be a valid CID") from exc


def _repository_id(value: Any, name: str = "repository_id") -> str:
    text = _text(value, name)
    if _REPOSITORY_ID_RE.fullmatch(text) is None:
        raise AssuranceManifestError(
            f"{name} must be a repository identity matching "
            f"{_REPOSITORY_ID_RE.pattern}"
        )
    return text


def _key_is_private_or_model_authority(key: str) -> bool:
    lowered = key.lower()
    if lowered in PRIVATE_FIELD_MARKERS or lowered in MODEL_AUTHORITY_FORBIDDEN_KEYS:
        return True
    for marker in PRIVATE_FIELD_MARKERS | MODEL_AUTHORITY_FORBIDDEN_KEYS:
        if marker in lowered:
            return True
    return False


def _reject_private_and_model_authority(value: Any, *, path: str) -> None:
    """Reject private material and model-authority claims.

    Host-fallback substring markers are intentionally not applied to sealed
    identity payloads: fields such as ``environment_cid`` are legitimate
    content-addressed bindings (not host env fallbacks).
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise AssuranceManifestError(
                    f"{path} map keys must be str, got {type(key).__name__}"
                )
            key_path = f"{path}.{key}"
            if _key_is_private_or_model_authority(key):
                raise AssuranceManifestError(
                    f"{key_path} rejects private or model-authority field {key!r}"
                )
            _reject_private_and_model_authority(item, path=key_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_and_model_authority(item, path=f"{path}[{index}]")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise AssuranceManifestError(f"{name} must be a mapping")
    raw = dict(value)
    _reject_private_and_model_authority(raw, path=name)
    return MappingProxyType(raw)


def _metadata_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Free-form metadata: also reject host-fallback markers."""

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise AssuranceManifestError(f"{name} must be a mapping")
    raw = dict(value)
    try:
        reject_private_and_model_authority(raw, path=name)
    except AssuranceBaseError as exc:
        raise AssuranceManifestError(str(exc)) from exc
    # Host-path markers remain forbidden on free-form metadata only.
    from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
        HOST_FALLBACK_MARKERS,
    )

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                lowered = str(key).lower()
                if lowered in HOST_FALLBACK_MARKERS or any(
                    marker in lowered for marker in HOST_FALLBACK_MARKERS
                ):
                    raise AssuranceManifestError(
                        f"{path}.{key} rejects host fallback field {key!r}"
                    )
                _walk(item, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                _walk(item, f"{path}[{index}]")

    _walk(raw, name)
    return MappingProxyType(raw)


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


# ---------------------------------------------------------------------------
# Repository state / verification policy views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepositoryStateBinding:
    """Normalized repository identity + state root for assurance manifests."""

    repository_id: str
    repository_state_cid: str
    revision: str = ""
    source_root_cid: str | None = None
    environment_cid: str | None = None
    dependency_lock_cid: str | None = None
    parent_revision_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "repository_id",
            "repository_state_cid",
            "revision",
            "source_root_cid",
            "environment_cid",
            "dependency_lock_cid",
            "parent_revision_ids",
            "metadata",
            "identity_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _repository_id(self.repository_id)
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self, "revision", _text(self.revision, "revision", empty=True)
        )
        object.__setattr__(
            self,
            "source_root_cid",
            _cid(self.source_root_cid, "source_root_cid")
            if self.source_root_cid is not None
            else None,
        )
        object.__setattr__(
            self,
            "environment_cid",
            _cid(self.environment_cid, "environment_cid")
            if self.environment_cid is not None
            else None,
        )
        object.__setattr__(
            self,
            "dependency_lock_cid",
            _cid(self.dependency_lock_cid, "dependency_lock_cid")
            if self.dependency_lock_cid is not None
            else None,
        )
        parents = tuple(str(item) for item in (self.parent_revision_ids or ()))
        object.__setattr__(self, "parent_revision_ids", parents)
        object.__setattr__(
            self, "metadata", _metadata_mapping(self.metadata, "metadata")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": REPOSITORY_STATE_SCHEMA,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "revision": self.revision,
            "source_root_cid": self.source_root_cid,
            "environment_cid": self.environment_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "parent_revision_ids": list(self.parent_revision_ids),
            "metadata": dict(self.metadata),
        }

    @property
    def identity_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["identity_cid"] = self.identity_cid
        return payload

    @classmethod
    def normalize(cls, value: Any) -> "RepositoryStateBinding":
        if isinstance(value, RepositoryStateBinding):
            return value
        if isinstance(value, str):
            # Bare state CID is insufficient without repository_id.
            raise AssuranceManifestError(
                "repository_state string must be provided as a mapping with "
                "repository_id and repository_state_cid"
            )
        if not isinstance(value, Mapping) and not hasattr(value, "to_canonical"):
            # Support RepositoryStateView-like objects.
            if not hasattr(value, "repository_id"):
                raise AssuranceManifestError(
                    "repository_state must be a mapping or RepositoryStateBinding"
                )
        if hasattr(value, "to_canonical") and callable(value.to_canonical):
            value = value.to_canonical()
        if not isinstance(value, Mapping):
            # Dataclass-like
            value = {
                "repository_id": _attr(value, "repository_id"),
                "repository_state_cid": _attr(value, "repository_state_cid")
                or _attr(value, "identity_cid"),
                "revision": _attr(value, "revision") or "",
                "source_root_cid": _attr(value, "source_root_cid"),
                "environment_cid": _attr(value, "environment_cid"),
                "dependency_lock_cid": _attr(value, "dependency_lock_cid"),
                "parent_revision_ids": _attr(value, "parent_revision_ids") or (),
                "metadata": _attr(value, "metadata") or {},
            }
        payload = dict(value)
        # Accept identity_cid as alias for repository_state_cid.
        if "repository_state_cid" not in payload and "identity_cid" in payload:
            payload["repository_state_cid"] = payload["identity_cid"]
        if "repository_id" not in payload:
            raise AssuranceManifestError("repository_state.repository_id is required")
        if "repository_state_cid" not in payload:
            raise AssuranceManifestError(
                "repository_state.repository_state_cid is required"
            )
        return cls(
            repository_id=payload["repository_id"],
            repository_state_cid=payload["repository_state_cid"],
            revision=payload.get("revision", ""),
            source_root_cid=payload.get("source_root_cid"),
            environment_cid=payload.get("environment_cid"),
            dependency_lock_cid=payload.get("dependency_lock_cid"),
            parent_revision_ids=payload.get("parent_revision_ids", ()),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class VerificationPolicyBinding:
    """Normalized verification policy identity for assurance manifests."""

    policy_cid: str
    policy_id: str = "default"
    proof_schema_version: str = "1"
    canonicalization_version: str = "1"
    dependency_graph_schema_version: str = "graph@1"
    circuit_id: str = "n/a"
    verification_key_id: str = "n/a"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_cid", _cid(self.policy_cid, "policy_cid"))
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "proof_schema_version",
            _text(self.proof_schema_version, "proof_schema_version"),
        )
        object.__setattr__(
            self,
            "canonicalization_version",
            _text(self.canonicalization_version, "canonicalization_version"),
        )
        object.__setattr__(
            self,
            "dependency_graph_schema_version",
            _text(
                self.dependency_graph_schema_version,
                "dependency_graph_schema_version",
            ),
        )
        object.__setattr__(
            self, "circuit_id", _text(self.circuit_id, "circuit_id")
        )
        object.__setattr__(
            self,
            "verification_key_id",
            _text(self.verification_key_id, "verification_key_id"),
        )
        object.__setattr__(
            self, "metadata", _metadata_mapping(self.metadata, "metadata")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VERIFICATION_POLICY_SCHEMA,
            "policy_cid": self.policy_cid,
            "policy_id": self.policy_id,
            "proof_schema_version": self.proof_schema_version,
            "canonicalization_version": self.canonicalization_version,
            "dependency_graph_schema_version": self.dependency_graph_schema_version,
            "circuit_id": self.circuit_id,
            "verification_key_id": self.verification_key_id,
            "metadata": dict(self.metadata),
        }

    @property
    def identity_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["identity_cid"] = self.identity_cid
        return payload

    @classmethod
    def normalize(cls, value: Any) -> "VerificationPolicyBinding":
        if isinstance(value, VerificationPolicyBinding):
            return value
        if isinstance(value, str):
            return cls(policy_cid=value)
        if hasattr(value, "to_canonical") and callable(value.to_canonical):
            value = value.to_canonical()
        if not isinstance(value, Mapping):
            if not hasattr(value, "policy_cid"):
                raise AssuranceManifestError(
                    "verification_policy must be a CID string, mapping, "
                    "or VerificationPolicyBinding"
                )
            value = {
                "policy_cid": _attr(value, "policy_cid"),
                "policy_id": _attr(value, "policy_id") or "default",
                "proof_schema_version": _attr(value, "proof_schema_version") or "1",
                "canonicalization_version": _attr(value, "canonicalization_version")
                or "1",
                "dependency_graph_schema_version": _attr(
                    value, "dependency_graph_schema_version"
                )
                or "graph@1",
                "circuit_id": _attr(value, "circuit_id") or "n/a",
                "verification_key_id": _attr(value, "verification_key_id") or "n/a",
                "metadata": _attr(value, "metadata") or {},
            }
        payload = dict(value)
        if "policy_cid" not in payload and "identity_cid" in payload:
            payload["policy_cid"] = payload["identity_cid"]
        if "policy_cid" not in payload:
            raise AssuranceManifestError(
                "verification_policy.policy_cid is required"
            )
        return cls(
            policy_cid=payload["policy_cid"],
            policy_id=payload.get("policy_id", "default"),
            proof_schema_version=payload.get("proof_schema_version", "1"),
            canonicalization_version=payload.get("canonicalization_version", "1"),
            dependency_graph_schema_version=payload.get(
                "dependency_graph_schema_version", "graph@1"
            ),
            circuit_id=payload.get("circuit_id", "n/a"),
            verification_key_id=payload.get("verification_key_id", "n/a"),
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# AssuranceManifest
# ---------------------------------------------------------------------------


def _normalize_authority_status_map(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Mapping[str, Any]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise AssuranceManifestError("authority_status must be a mapping")
    out: dict[str, Mapping[str, Any]] = {}
    for key in AUTHORITY_KEYS:
        if key not in value:
            out[key] = MappingProxyType(
                {
                    "authority": key,
                    "available": False,
                    "status": AuthorityStatus.TYPED_UNAVAILABLE.value,
                    "reason_code": CapabilityReason.MISSING.value,
                    "diagnostic": f"authority {key} not probed",
                }
            )
            continue
        entry = value[key]
        if not isinstance(entry, Mapping):
            raise AssuranceManifestError(
                f"authority_status[{key}] must be a mapping"
            )
        available = bool(entry.get("available", False))
        status = entry.get("status")
        if available:
            public_status = AuthorityStatus.AVAILABLE.value
        else:
            public_status = AuthorityStatus.TYPED_UNAVAILABLE.value
            if status == AuthorityStatus.AVAILABLE.value:
                public_status = AuthorityStatus.TYPED_UNAVAILABLE.value
            elif status in {
                AuthorityStatus.AVAILABLE.value,
                AuthorityStatus.TYPED_UNAVAILABLE.value,
            }:
                public_status = str(status)
            else:
                # Map drifted granular statuses to typed_unavailable.
                public_status = AuthorityStatus.TYPED_UNAVAILABLE.value
        out[key] = MappingProxyType(
            {
                "authority": key,
                "available": available,
                "status": public_status,
                "reason_code": entry.get("reason_code"),
                "diagnostic": _clip(str(entry.get("diagnostic") or "")),
                "adapter_id": entry.get("adapter_id"),
                "interface_id": entry.get("interface_id"),
                "schema": entry.get("schema"),
                "operations": list(entry.get("operations") or ()),
                "fingerprints": dict(entry.get("fingerprints") or {}),
                "seal_status": entry.get("seal_status"),
                "can_be_satisfied_by_ivp_commitment": False
                if key == "sealer"
                else entry.get("can_be_satisfied_by_ivp_commitment"),
                "retryable": bool(entry.get("retryable", False)),
            }
        )
    # Reject unknown authority keys (fail closed / no silent expansion).
    unknown = set(value) - set(AUTHORITY_KEYS)
    if unknown:
        raise AssuranceManifestError(
            f"authority_status contains unknown authorities: {sorted(unknown)}"
        )
    return MappingProxyType(out)


@dataclass(frozen=True, slots=True)
class AssuranceManifest:
    """Sealed campaign assurance manifest binding released authorities.

    Records repository state, verification policy identity, closed authority
    status mappings, optional detector catalog slice for detection prediction,
    and seal disposition without mutating production policy.
    """

    repository_id: str
    repository_state_cid: str
    verification_policy_cid: str
    authority_status: Mapping[str, Mapping[str, Any]]
    repository_state: Mapping[str, Any]
    verification_policy: Mapping[str, Any]
    detectors: tuple[Mapping[str, Any], ...] = ()
    dependency_edges: tuple[Mapping[str, Any], ...] = ()
    claims: tuple[Mapping[str, Any], ...] = ()
    seal_status: str = SealStatus.TYPED_UNAVAILABLE.value
    terminal_status: str = AssuranceTerminalStatus.COMPLETE.value
    observation_complete: bool = True
    production_policy_changed: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str = AAE_RUNTIME_ADAPTERS_EVIDENCE
    schema: str = ASSURANCE_MANIFEST_SCHEMA
    interface_id: str = ASSURANCE_MANIFEST_INTERFACE

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "evidence_id",
            "repository_id",
            "repository_state_cid",
            "verification_policy_cid",
            "authority_status",
            "repository_state",
            "verification_policy",
            "detectors",
            "dependency_edges",
            "claims",
            "seal_status",
            "terminal_status",
            "observation_complete",
            "production_policy_changed",
            "notes",
            "metadata",
            "manifest_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _repository_id(self.repository_id)
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self,
            "verification_policy_cid",
            _cid(self.verification_policy_cid, "verification_policy_cid"),
        )
        object.__setattr__(
            self,
            "authority_status",
            _normalize_authority_status_map(self.authority_status),
        )
        object.__setattr__(
            self,
            "repository_state",
            _mapping(self.repository_state, "repository_state"),
        )
        object.__setattr__(
            self,
            "verification_policy",
            _mapping(self.verification_policy, "verification_policy"),
        )
        if not isinstance(self.detectors, (list, tuple)):
            raise AssuranceManifestError("detectors must be a list")
        if len(self.detectors) > MAX_DETECTORS:
            raise AssuranceManifestError("detectors exceeds maximum length")
        object.__setattr__(
            self,
            "detectors",
            tuple(_mapping(item, "detectors") for item in self.detectors),
        )
        if not isinstance(self.dependency_edges, (list, tuple)):
            raise AssuranceManifestError("dependency_edges must be a list")
        if len(self.dependency_edges) > MAX_EDGES:
            raise AssuranceManifestError("dependency_edges exceeds maximum length")
        object.__setattr__(
            self,
            "dependency_edges",
            tuple(
                _mapping(item, "dependency_edges")
                for item in self.dependency_edges
            ),
        )
        if not isinstance(self.claims, (list, tuple)):
            raise AssuranceManifestError("claims must be a list")
        if len(self.claims) > MAX_CLAIMS:
            raise AssuranceManifestError("claims exceeds maximum length")
        object.__setattr__(
            self,
            "claims",
            tuple(_mapping(item, "claims") for item in self.claims),
        )
        try:
            object.__setattr__(
                self, "seal_status", SealStatus(self.seal_status).value
            )
        except ValueError as exc:
            raise AssuranceManifestError(
                f"seal_status has unsupported value {self.seal_status!r}"
            ) from exc
        try:
            object.__setattr__(
                self,
                "terminal_status",
                AssuranceTerminalStatus(self.terminal_status).value,
            )
        except ValueError as exc:
            raise AssuranceManifestError(
                f"terminal_status has unsupported value {self.terminal_status!r}"
            ) from exc
        object.__setattr__(
            self,
            "observation_complete",
            _bool(self.observation_complete, "observation_complete"),
        )
        object.__setattr__(
            self,
            "production_policy_changed",
            _bool(self.production_policy_changed, "production_policy_changed"),
        )
        if self.production_policy_changed:
            raise AssuranceManifestError(
                "create_assurance_manifest must not change production policy"
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(
            self, "metadata", _metadata_mapping(self.metadata, "metadata")
        )
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id")
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        if self.schema != ASSURANCE_MANIFEST_SCHEMA:
            raise AssuranceManifestError(
                "unsupported AssuranceManifest schema version"
            )
        if self.interface_id != ASSURANCE_MANIFEST_INTERFACE:
            raise AssuranceManifestError(
                "unsupported AssuranceManifest interface_id"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "evidence_id": self.evidence_id,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "verification_policy_cid": self.verification_policy_cid,
            "authority_status": {
                key: dict(value) for key, value in self.authority_status.items()
            },
            "repository_state": dict(self.repository_state),
            "verification_policy": dict(self.verification_policy),
            "detectors": [dict(item) for item in self.detectors],
            "dependency_edges": [dict(item) for item in self.dependency_edges],
            "claims": [dict(item) for item in self.claims],
            "seal_status": self.seal_status,
            "terminal_status": self.terminal_status,
            "observation_complete": self.observation_complete,
            "production_policy_changed": False,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @property
    def manifest_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["manifest_cid"] = self.manifest_cid
        return payload

    def authority_available(self, name: str) -> bool:
        entry = self.authority_status.get(name)
        if entry is None:
            return False
        return bool(entry.get("available"))

    def typed_unavailable_authorities(self) -> tuple[str, ...]:
        return tuple(
            key
            for key, entry in self.authority_status.items()
            if entry.get("status") == AuthorityStatus.TYPED_UNAVAILABLE.value
            or not entry.get("available")
        )

    def as_detection_manifest(self) -> DetectionAssuranceManifest:
        """Project the detection slice used by ``predict_detection_set@1``."""

        return DetectionAssuranceManifest(
            repository_id=self.repository_id,
            repository_state_cid=self.repository_state_cid,
            detectors=list(self.detectors),
            dependency_edges=list(self.dependency_edges),
            claims=list(self.claims),
            observation_complete=self.observation_complete,
            notes=self.notes,
            metadata={
                "source_manifest_interface": ASSURANCE_MANIFEST_INTERFACE,
                "source_manifest_schema": ASSURANCE_MANIFEST_SCHEMA,
                "source_manifest_cid": self.manifest_cid,
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssuranceManifest":
        if not isinstance(data, Mapping):
            raise AssuranceManifestError("AssuranceManifest data must be a mapping")
        payload = dict(data)
        claimed = payload.pop("manifest_cid", None)
        # Allow partial open construction via normalize path for non-strict callers.
        unknown = set(payload) - cls._FIELDS
        if unknown:
            raise AssuranceManifestError(
                f"AssuranceManifest contains unknown fields: {sorted(unknown)}"
            )
        result = cls(
            repository_id=payload["repository_id"],
            repository_state_cid=payload["repository_state_cid"],
            verification_policy_cid=payload["verification_policy_cid"],
            authority_status=payload.get("authority_status", {}),
            repository_state=payload.get("repository_state", {}),
            verification_policy=payload.get("verification_policy", {}),
            detectors=payload.get("detectors", ()),
            dependency_edges=payload.get("dependency_edges", ()),
            claims=payload.get("claims", ()),
            seal_status=payload.get(
                "seal_status", SealStatus.TYPED_UNAVAILABLE.value
            ),
            terminal_status=payload.get(
                "terminal_status", AssuranceTerminalStatus.COMPLETE.value
            ),
            observation_complete=payload.get("observation_complete", True),
            production_policy_changed=payload.get(
                "production_policy_changed", False
            ),
            notes=payload.get("notes"),
            metadata=payload.get("metadata", {}),
            evidence_id=payload.get("evidence_id", AAE_RUNTIME_ADAPTERS_EVIDENCE),
            schema=payload.get("schema", ASSURANCE_MANIFEST_SCHEMA),
            interface_id=payload.get(
                "interface_id", ASSURANCE_MANIFEST_INTERFACE
            ),
        )
        if claimed is not None and claimed != result.manifest_cid:
            raise AssuranceManifestError(
                "AssuranceManifest manifest_cid identity mismatch"
            )
        return result


def _seal_status_from_authorities(
    authority_status: Mapping[str, Mapping[str, Any]],
) -> str:
    sealer = authority_status.get("sealer") or {}
    if sealer.get("available") and sealer.get("status") == AuthorityStatus.AVAILABLE.value:
        return SealStatus.AVAILABLE.value
    seal_status = sealer.get("seal_status")
    if seal_status in {s.value for s in SealStatus}:
        return str(seal_status)
    return SealStatus.TYPED_UNAVAILABLE.value


def _terminal_status_from_authorities(
    authority_status: Mapping[str, Mapping[str, Any]],
    *,
    require_execution_authorities: bool,
) -> str:
    if not require_execution_authorities:
        return AssuranceTerminalStatus.COMPLETE.value
    required = (
        "index",
        "capsule",
        "context",
        "verification",
        "policy",
        "state",
        "storage",
    )
    for key in required:
        entry = authority_status.get(key) or {}
        if not entry.get("available"):
            return AssuranceTerminalStatus.UNAVAILABLE.value
    return AssuranceTerminalStatus.COMPLETE.value


def create_assurance_manifest(
    repository_state: Any,
    verification_policy: Any,
    *,
    detectors: Sequence[Mapping[str, Any]] | None = None,
    dependency_edges: Sequence[Mapping[str, Any]] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
    authority_status: Mapping[str, Mapping[str, Any]] | None = None,
    runtime: AssuranceRuntimeAdapters | None = None,
    probe_live_authorities: bool = False,
    require_execution_authorities: bool = False,
    require_sealer: bool = False,
    observation_complete: bool = True,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    index_surface: Any | None = None,
    capsule_surface: Any | None = None,
    context_surface: Any | None = None,
    verification_surface: Any | None = None,
    policy_surface: Any | None = None,
    state_surface: Any | None = None,
    storage_surface: Any | None = None,
    sealer_surface: Any | None = None,
) -> AssuranceManifest:
    """Create a sealed ``AssuranceManifest@1`` for the given repository state.

    Parameters
    ----------
    repository_state:
        Mapping or view with ``repository_id`` and ``repository_state_cid``.
    verification_policy:
        Policy CID string, mapping, or view with ``policy_cid``.
    authority_status:
        Optional precomputed authority status map (tests / injected probes).
    runtime:
        Optional already-loaded ``AssuranceRuntimeAdapters``.
    probe_live_authorities:
        When true and no status/runtime is supplied, probe live tree imports.
        Default false so unit tests and cold paths stay hermetic.
    require_execution_authorities / require_sealer:
        When true, fail closed if required authorities are typed unavailable.
    """

    repo = RepositoryStateBinding.normalize(repository_state)
    policy = VerificationPolicyBinding.normalize(verification_policy)

    if authority_status is not None:
        status_map = _normalize_authority_status_map(authority_status)
    elif runtime is not None:
        status_map = _normalize_authority_status_map(runtime.authority_status_map())
    elif probe_live_authorities or any(
        surface is not None
        for surface in (
            index_surface,
            capsule_surface,
            context_surface,
            verification_surface,
            policy_surface,
            state_surface,
            storage_surface,
            sealer_surface,
        )
    ):
        if any(
            surface is not None
            for surface in (
                index_surface,
                capsule_surface,
                context_surface,
                verification_surface,
                policy_surface,
                state_surface,
                storage_surface,
                sealer_surface,
            )
        ):
            try:
                loaded = load_runtime_adapters(
                    index_surface=index_surface,
                    capsule_surface=capsule_surface,
                    context_surface=context_surface,
                    verification_surface=verification_surface,
                    policy_surface=policy_surface,
                    state_surface=state_surface,
                    storage_surface=storage_surface,
                    sealer_surface=sealer_surface,
                    require_sealer=require_sealer,
                    require_execution=require_execution_authorities,
                )
                status_map = _normalize_authority_status_map(
                    loaded.authority_status_map()
                )
            except AssuranceCapabilityUnavailable as exc:
                # Capture partial typed unavailability when require_* is false.
                if require_execution_authorities or require_sealer:
                    raise AssuranceManifestError(
                        f"required authority typed unavailable: {exc.diagnostic}",
                        reason_code=exc.reason_code,
                    ) from exc
                status_map = _normalize_authority_status_map(
                    probe_all_authorities(
                        index_surface=index_surface,
                        capsule_surface=capsule_surface,
                        context_surface=context_surface,
                        verification_surface=verification_surface,
                        policy_surface=policy_surface,
                        state_surface=state_surface,
                        storage_surface=storage_surface,
                        sealer_surface=sealer_surface,
                    )
                )
        else:
            status_map = _normalize_authority_status_map(probe_all_authorities())
    else:
        # Hermetic default: all authorities typed unavailable until bound.
        status_map = _normalize_authority_status_map(
            {
                key: {
                    "authority": key,
                    "available": False,
                    "status": AuthorityStatus.TYPED_UNAVAILABLE.value,
                    "reason_code": CapabilityReason.MISSING.value,
                    "diagnostic": (
                        "authority not bound; pass surfaces, runtime, "
                        "authority_status, or probe_live_authorities=True"
                    ),
                }
                for key in AUTHORITY_KEYS
            }
        )

    if require_execution_authorities:
        for key in (
            "index",
            "capsule",
            "context",
            "verification",
            "policy",
            "state",
            "storage",
        ):
            entry = status_map.get(key) or {}
            if not entry.get("available"):
                raise AssuranceManifestError(
                    f"required authority {key!r} is typed unavailable",
                    reason_code=str(
                        entry.get("reason_code")
                        or CapabilityReason.CAPABILITY_UNAVAILABLE.value
                    ),
                )
    if require_sealer:
        sealer = status_map.get("sealer") or {}
        if not sealer.get("available"):
            raise AssuranceManifestError(
                "required sealer authority is typed unavailable",
                reason_code=str(
                    sealer.get("reason_code")
                    or CapabilityReason.SEALER_UNAVAILABLE.value
                ),
            )

    seal_status = _seal_status_from_authorities(status_map)
    terminal_status = _terminal_status_from_authorities(
        status_map,
        require_execution_authorities=require_execution_authorities,
    )
    # When hermetic/unbound, still allow a complete observational manifest that
    # honestly records typed_unavailable authorities.
    if not require_execution_authorities:
        terminal_status = AssuranceTerminalStatus.COMPLETE.value

    meta = dict(metadata or {})
    meta.setdefault("create_interface", CREATE_ASSURANCE_MANIFEST_INTERFACE)
    meta.setdefault("generator_id", GENERATOR_ID)
    meta.setdefault("generator_version", GENERATOR_VERSION)
    meta.setdefault("production_policy_changed", False)

    return AssuranceManifest(
        repository_id=repo.repository_id,
        repository_state_cid=repo.repository_state_cid,
        verification_policy_cid=policy.policy_cid,
        authority_status=status_map,
        repository_state=repo.identity_payload(),
        verification_policy=policy.identity_payload(),
        detectors=tuple(detectors or ()),
        dependency_edges=tuple(dependency_edges or ()),
        claims=tuple(claims or ()),
        seal_status=seal_status,
        terminal_status=terminal_status,
        observation_complete=observation_complete,
        production_policy_changed=False,
        notes=notes,
        metadata=meta,
        evidence_id=AAE_RUNTIME_ADAPTERS_EVIDENCE,
    )


__all__ = [
    "CREATE_ASSURANCE_MANIFEST_INTERFACE",
    "ASSURANCE_MANIFEST_SCHEMA",
    "ASSURANCE_MANIFEST_INTERFACE",
    "REPOSITORY_STATE_SCHEMA",
    "VERIFICATION_POLICY_SCHEMA",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "AssuranceManifestError",
    "RepositoryStateBinding",
    "VerificationPolicyBinding",
    "AssuranceManifest",
    "create_assurance_manifest",
    # Re-export detection schema constants used by projection consumers.
    "DETECTION_ASSURANCE_MANIFEST_SCHEMA",
    "DETECTION_ASSURANCE_MANIFEST_INTERFACE",
]
