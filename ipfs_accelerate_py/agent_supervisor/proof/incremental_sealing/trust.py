"""Allowlisted verification-key, proving-key, and signer trust policy (IPS-030).

Policy configuration — not an untrusted caller or model — selects
content-addressed verification keys, proving-key handles, circuits, and
signers.  Every key binds a setup origin, production/test-only designation,
circuit compatibility set, and epoch.  Signers bind scope and a revocation
epoch.

Proving-key *bytes* are never public API data: callers receive only a
:class:`ProvingKeyHandle`.  Production mode never generates or downloads key
material.

Interfaces: ``TrustedProofPolicy``, ``VerificationKeyRegistry``,
``SignerTrustRegistry``, ``ProvingKeyHandle``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

KEY_REGISTRY_EVIDENCE: Final[str] = "ips/key-registry@1"
SIGNER_TRUST_EVIDENCE: Final[str] = "ips/signer-trust@1"

KEY_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "verification-key-registry@1"
)
PROVING_KEY_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proving-key-handle@1"
)
SIGNER_TRUST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "signer-trust-registry@1"
)
TRUST_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "trust-decision@1"
)
TRUSTED_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "trusted-proof-policy@1"
)

# Closed setup-origin vocabulary.  Unknown origins always reject.
class SetupOrigin(str, Enum):
    """Documented origin of key material / ceremony."""

    OPERATOR_REVIEWED = "operator_reviewed"
    CEREMONY = "ceremony"
    TEST_FIXTURE = "test_fixture"


DEFAULT_PRODUCTION_SETUP_ORIGINS: Final[frozenset[str]] = frozenset(
    {
        SetupOrigin.OPERATOR_REVIEWED.value,
        SetupOrigin.CEREMONY.value,
    }
)
DEFAULT_TEST_SETUP_ORIGINS: Final[frozenset[str]] = frozenset(
    {
        SetupOrigin.OPERATOR_REVIEWED.value,
        SetupOrigin.CEREMONY.value,
        SetupOrigin.TEST_FIXTURE.value,
    }
)

# Sensitive field names that must never appear on public handle/export surfaces.
_SENSITIVE_KEY_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "proving_key",
        "proving_key_bytes",
        "proving_key_material",
        "private_key",
        "private_key_bytes",
        "witness",
        "witness_bytes",
        "trapdoor",
        "secret",
        "key_bytes",
        "raw_key",
        "download_url",
        "generated_bytes",
    }
)


class TrustError(ValueError):
    """Fail-closed trust-policy contract violation."""


class TrustOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class TrustRejectionReason(str, Enum):
    """Stable reason codes for rejected trust decisions."""

    UNALLOWLISTED_VERIFICATION_KEY = "unallowlisted_verification_key"
    SUBSTITUTED_VERIFICATION_KEY = "substituted_verification_key"
    OLD_VERIFICATION_KEY = "old_verification_key"
    REVOKED_VERIFICATION_KEY = "revoked_verification_key"
    TEST_ONLY_IN_PRODUCTION = "test_only_in_production"
    UNALLOWLISTED_PROVING_KEY = "unallowlisted_proving_key"
    SUBSTITUTED_PROVING_KEY = "substituted_proving_key"
    OLD_PROVING_KEY = "old_proving_key"
    REVOKED_PROVING_KEY = "revoked_proving_key"
    CIRCUIT_INCOMPATIBLE = "circuit_incompatible"
    UNTRUSTED_SIGNER = "untrusted_signer"
    REVOKED_SIGNER = "revoked_signer"
    OUT_OF_SCOPE_SIGNER = "out_of_scope_signer"
    KEY_GENERATION_FORBIDDEN = "key_generation_forbidden"
    KEY_DOWNLOAD_FORBIDDEN = "key_download_forbidden"
    DISALLOWED_SETUP_ORIGIN = "disallowed_setup_origin"
    MISSING_SETUP_ORIGIN = "missing_setup_origin"
    MALFORMED_REQUEST = "malformed_request"
    PRODUCTION_KEY_EXPORT_FORBIDDEN = "production_key_export_forbidden"


def closed_trust_rejection_reasons() -> frozenset[str]:
    return frozenset(item.value for item in TrustRejectionReason)


def closed_setup_origins() -> frozenset[str]:
    return frozenset(item.value for item in SetupOrigin)


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrustError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TrustError(f"{field_name} must be a boolean")
    return value


def _require_nonneg_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrustError(f"{field_name} must be a non-negative int")
    return value


def _require_str_set(value: Any, field_name: str, *, allow_empty: bool = False) -> frozenset[str]:
    if isinstance(value, frozenset):
        items = value
    elif isinstance(value, (set, list, tuple)):
        items = frozenset(value)
    else:
        raise TrustError(f"{field_name} must be a frozenset/set/list/tuple of strings")
    out: set[str] = set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise TrustError(f"{field_name} entries must be non-empty strings")
        out.add(item.strip())
    if not out and not allow_empty:
        raise TrustError(f"{field_name} must be non-empty")
    return frozenset(out)


def _coerce_setup_origin(value: SetupOrigin | str) -> SetupOrigin:
    if isinstance(value, SetupOrigin):
        return value
    if not isinstance(value, str) or not value.strip():
        raise TrustError("setup_origin must be a closed SetupOrigin string")
    try:
        return SetupOrigin(value.strip())
    except ValueError as exc:
        raise TrustError(f"unknown setup_origin {value!r}") from exc


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _cid_matches(left: str, right: str) -> bool:
    return hmac.compare_digest(str(left), str(right))


def _safe_subject_id(value: Any) -> str:
    """Return a non-empty subject id for rejection records."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    if value is None or value == "":
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


@dataclass(frozen=True, slots=True)
class VerificationKeyRecord:
    """Allowlisted verification key binding.

    Verification keys are content-addressed (``key_cid``).  Policy, not the
    caller, decides which CIDs are admitted.  ``test_only`` is machine-checked
    and cannot enter a production allowlist.
    """

    key_id: str
    key_cid: str
    circuit_ids: frozenset[str]
    setup_origin: SetupOrigin | str
    test_only: bool
    epoch: int = 0
    revoked: bool = False
    superseded_by: str | None = None
    digest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _require_nonempty_str(self.key_id, "key_id"))
        object.__setattr__(self, "key_cid", _require_nonempty_str(self.key_cid, "key_cid"))
        object.__setattr__(
            self,
            "circuit_ids",
            _require_str_set(self.circuit_ids, "circuit_ids"),
        )
        object.__setattr__(
            self, "setup_origin", _coerce_setup_origin(self.setup_origin)
        )
        object.__setattr__(self, "test_only", _require_bool(self.test_only, "test_only"))
        object.__setattr__(self, "epoch", _require_nonneg_int(self.epoch, "epoch"))
        object.__setattr__(self, "revoked", _require_bool(self.revoked, "revoked"))
        if self.superseded_by is not None:
            object.__setattr__(
                self,
                "superseded_by",
                _require_nonempty_str(self.superseded_by, "superseded_by"),
            )
        if self.digest is not None:
            object.__setattr__(
                self, "digest", _require_nonempty_str(self.digest, "digest")
            )
        if not isinstance(self.metadata, Mapping):
            raise TrustError("metadata must be a mapping")
        # Test-fixture origin forces test_only designation.
        if (
            self.setup_origin is SetupOrigin.TEST_FIXTURE
            and self.test_only is not True
        ):
            raise TrustError("test_fixture setup_origin requires test_only=True")

    @property
    def is_current(self) -> bool:
        return not self.revoked and self.superseded_by is None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key_cid": self.key_cid,
            "circuit_ids": sorted(self.circuit_ids),
            "setup_origin": self.setup_origin.value,
            "test_only": self.test_only,
            "epoch": self.epoch,
            "revoked": self.revoked,
            "superseded_by": self.superseded_by,
            "digest": self.digest,
            "is_current": self.is_current,
            # Public verification-key metadata only; never key bytes.
            "key_bytes_exported": False,
        }


@dataclass(frozen=True, slots=True)
class ProvingKeyRecord:
    """Private proving-key registry entry.

    Only nonexportable :class:`ProvingKeyHandle` references leave the registry.
    Raw proving-key bytes are intentionally absent from this record.
    """

    key_id: str
    key_cid: str
    circuit_ids: frozenset[str]
    setup_origin: SetupOrigin | str
    test_only: bool
    paired_verification_key_id: str
    epoch: int = 0
    revoked: bool = False
    superseded_by: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _require_nonempty_str(self.key_id, "key_id"))
        object.__setattr__(self, "key_cid", _require_nonempty_str(self.key_cid, "key_cid"))
        object.__setattr__(
            self,
            "circuit_ids",
            _require_str_set(self.circuit_ids, "circuit_ids"),
        )
        object.__setattr__(
            self, "setup_origin", _coerce_setup_origin(self.setup_origin)
        )
        object.__setattr__(self, "test_only", _require_bool(self.test_only, "test_only"))
        object.__setattr__(
            self,
            "paired_verification_key_id",
            _require_nonempty_str(
                self.paired_verification_key_id, "paired_verification_key_id"
            ),
        )
        object.__setattr__(self, "epoch", _require_nonneg_int(self.epoch, "epoch"))
        object.__setattr__(self, "revoked", _require_bool(self.revoked, "revoked"))
        if self.superseded_by is not None:
            object.__setattr__(
                self,
                "superseded_by",
                _require_nonempty_str(self.superseded_by, "superseded_by"),
            )
        if not isinstance(self.metadata, Mapping):
            raise TrustError("metadata must be a mapping")
        if (
            self.setup_origin is SetupOrigin.TEST_FIXTURE
            and self.test_only is not True
        ):
            raise TrustError("test_fixture setup_origin requires test_only=True")
        for forbidden in _SENSITIVE_KEY_FIELD_NAMES:
            if forbidden in self.metadata:
                raise TrustError(
                    f"proving-key metadata must not carry sensitive field {forbidden!r}"
                )

    @property
    def is_current(self) -> bool:
        return not self.revoked and self.superseded_by is None

    def to_handle(self) -> "ProvingKeyHandle":
        return ProvingKeyHandle(
            key_id=self.key_id,
            key_cid=self.key_cid,
            circuit_ids=self.circuit_ids,
            setup_origin=self.setup_origin,
            test_only=self.test_only,
            paired_verification_key_id=self.paired_verification_key_id,
            epoch=self.epoch,
            revoked=self.revoked,
            superseded_by=self.superseded_by,
        )


@dataclass(frozen=True, slots=True)
class ProvingKeyHandle:
    """Nonexportable public reference to a private proving key.

    Handles identify content-addressed proving keys without ever exposing
    proving-key bytes, trapdoors, or downloadable material.
    """

    key_id: str
    key_cid: str
    circuit_ids: frozenset[str]
    setup_origin: SetupOrigin | str
    test_only: bool
    paired_verification_key_id: str
    epoch: int = 0
    revoked: bool = False
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _require_nonempty_str(self.key_id, "key_id"))
        object.__setattr__(self, "key_cid", _require_nonempty_str(self.key_cid, "key_cid"))
        object.__setattr__(
            self,
            "circuit_ids",
            _require_str_set(self.circuit_ids, "circuit_ids"),
        )
        object.__setattr__(
            self, "setup_origin", _coerce_setup_origin(self.setup_origin)
        )
        object.__setattr__(self, "test_only", _require_bool(self.test_only, "test_only"))
        object.__setattr__(
            self,
            "paired_verification_key_id",
            _require_nonempty_str(
                self.paired_verification_key_id, "paired_verification_key_id"
            ),
        )
        object.__setattr__(self, "epoch", _require_nonneg_int(self.epoch, "epoch"))
        object.__setattr__(self, "revoked", _require_bool(self.revoked, "revoked"))
        if self.superseded_by is not None:
            object.__setattr__(
                self,
                "superseded_by",
                _require_nonempty_str(self.superseded_by, "superseded_by"),
            )

    @property
    def exportable(self) -> bool:
        return False

    @property
    def bytes_available(self) -> bool:
        return False

    def export_bytes(self) -> bytes:
        raise TrustError(
            "proving-key bytes are nonexportable; ProvingKeyHandle never returns key material"
        )

    def download(self) -> bytes:
        raise TrustError("proving-key download is forbidden via ProvingKeyHandle")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": PROVING_KEY_HANDLE_SCHEMA,
            "key_id": self.key_id,
            "key_cid": self.key_cid,
            "circuit_ids": sorted(self.circuit_ids),
            "setup_origin": self.setup_origin.value,
            "test_only": self.test_only,
            "paired_verification_key_id": self.paired_verification_key_id,
            "epoch": self.epoch,
            "revoked": self.revoked,
            "superseded_by": self.superseded_by,
            "exportable": False,
            "bytes_available": False,
            "proving_key_exported": False,
            "evidence_subset": KEY_REGISTRY_EVIDENCE,
        }

    def to_public_api(self) -> dict[str, Any]:
        """Public API projection — identity only, never key material."""
        payload = self.to_canonical()
        for name in _SENSITIVE_KEY_FIELD_NAMES:
            if name in payload:
                raise TrustError(f"public API leaked sensitive field {name!r}")
        return payload


@dataclass(frozen=True, slots=True)
class SignerTrustRecord:
    """Allowlisted signer with scope and revocation epoch."""

    signer_id: str
    scopes: frozenset[str]
    trusted: bool = True
    test_only: bool = False
    revocation_epoch: int | None = None
    public_key_cid: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "signer_id", _require_nonempty_str(self.signer_id, "signer_id")
        )
        object.__setattr__(
            self, "scopes", _require_str_set(self.scopes, "scopes", allow_empty=False)
        )
        object.__setattr__(self, "trusted", _require_bool(self.trusted, "trusted"))
        object.__setattr__(self, "test_only", _require_bool(self.test_only, "test_only"))
        if self.revocation_epoch is not None:
            object.__setattr__(
                self,
                "revocation_epoch",
                _require_nonneg_int(self.revocation_epoch, "revocation_epoch"),
            )
        if self.public_key_cid is not None:
            object.__setattr__(
                self,
                "public_key_cid",
                _require_nonempty_str(self.public_key_cid, "public_key_cid"),
            )
        if not isinstance(self.metadata, Mapping):
            raise TrustError("metadata must be a mapping")

    def is_revoked_at(self, current_epoch: int) -> bool:
        epoch = _require_nonneg_int(current_epoch, "current_epoch")
        if self.revocation_epoch is None:
            return False
        return epoch >= self.revocation_epoch

    def to_canonical(self) -> dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "scopes": sorted(self.scopes),
            "trusted": self.trusted,
            "test_only": self.test_only,
            "revocation_epoch": self.revocation_epoch,
            "public_key_cid": self.public_key_cid,
        }


@dataclass(frozen=True, slots=True)
class TrustDecision:
    """Typed accept/reject result for a key or signer trust check."""

    outcome: TrustOutcome
    accepted: bool
    reason_code: str | None
    message: str
    subject_kind: str
    subject_id: str
    evidence_subset: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            TrustOutcome(str(getattr(self.outcome, "value", self.outcome))),
        )
        if type(self.accepted) is not bool:
            raise TrustError("accepted must be a boolean")
        if self.accepted and self.outcome is not TrustOutcome.ACCEPTED:
            raise TrustError("accepted decisions require ACCEPTED outcome")
        if not self.accepted and self.outcome is not TrustOutcome.REJECTED:
            raise TrustError("rejected decisions require REJECTED outcome")
        if self.accepted and self.reason_code is not None:
            raise TrustError("accepted decisions must not carry a rejection reason")
        if not self.accepted and not self.reason_code:
            raise TrustError("rejected decisions require a reason_code")
        object.__setattr__(
            self,
            "subject_kind",
            _require_nonempty_str(self.subject_kind, "subject_kind"),
        )
        object.__setattr__(
            self, "subject_id", _require_nonempty_str(self.subject_id, "subject_id")
        )
        object.__setattr__(
            self,
            "evidence_subset",
            _require_nonempty_str(self.evidence_subset, "evidence_subset"),
        )
        object.__setattr__(self, "message", str(self.message))
        if not isinstance(self.details, Mapping):
            raise TrustError("details must be a mapping")
        # Never embed sensitive material in decision details.
        for name in _SENSITIVE_KEY_FIELD_NAMES:
            if name in self.details:
                raise TrustError(
                    f"trust decision details must not carry sensitive field {name!r}"
                )

    @property
    def rejected(self) -> bool:
        return not self.accepted

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": TRUST_DECISION_SCHEMA,
            "outcome": self.outcome.value,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "message": self.message,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "evidence_subset": self.evidence_subset,
            "details": dict(self.details),
            "proving_key_exported": False,
            "key_material_generated": False,
            "key_material_downloaded": False,
        }


def _accept(
    *,
    subject_kind: str,
    subject_id: str,
    evidence_subset: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> TrustDecision:
    return TrustDecision(
        outcome=TrustOutcome.ACCEPTED,
        accepted=True,
        reason_code=None,
        message=message,
        subject_kind=subject_kind,
        subject_id=subject_id,
        evidence_subset=evidence_subset,
        details=dict(details or {}),
    )


def _reject(
    *,
    reason: TrustRejectionReason,
    subject_kind: str,
    subject_id: str,
    evidence_subset: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> TrustDecision:
    return TrustDecision(
        outcome=TrustOutcome.REJECTED,
        accepted=False,
        reason_code=reason.value,
        message=message,
        subject_kind=subject_kind,
        subject_id=subject_id,
        evidence_subset=evidence_subset,
        details=dict(details or {}),
    )


class VerificationKeyRegistry:
    """Content-addressed allowlist of verification keys.

    Registration is a policy-configuration action.  Callers may only request
    evaluation against already-registered entries.
    """

    def __init__(
        self,
        entries: Iterable[VerificationKeyRecord] | None = None,
        *,
        production: bool = True,
        allowed_setup_origins: frozenset[str] | None = None,
        minimum_epoch: int = 0,
    ) -> None:
        self._production = _require_bool(production, "production")
        self._minimum_epoch = _require_nonneg_int(minimum_epoch, "minimum_epoch")
        if allowed_setup_origins is None:
            allowed_setup_origins = (
                DEFAULT_PRODUCTION_SETUP_ORIGINS
                if self._production
                else DEFAULT_TEST_SETUP_ORIGINS
            )
        self._allowed_setup_origins = _require_str_set(
            allowed_setup_origins, "allowed_setup_origins"
        )
        unknown = self._allowed_setup_origins - closed_setup_origins()
        if unknown:
            raise TrustError(
                f"allowed_setup_origins contains unknown origins: {sorted(unknown)}"
            )
        self._by_id: dict[str, VerificationKeyRecord] = {}
        self._by_cid: dict[str, str] = {}
        if entries is not None:
            for entry in entries:
                self.register(entry)

    @property
    def production(self) -> bool:
        return self._production

    @property
    def minimum_epoch(self) -> int:
        return self._minimum_epoch

    @property
    def allowed_setup_origins(self) -> frozenset[str]:
        return self._allowed_setup_origins

    def __contains__(self, key_id: object) -> bool:
        return isinstance(key_id, str) and key_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def get(self, key_id: str) -> VerificationKeyRecord | None:
        key_id = _require_nonempty_str(key_id, "key_id")
        return self._by_id.get(key_id)

    def get_by_cid(self, key_cid: str) -> VerificationKeyRecord | None:
        key_cid = _require_nonempty_str(key_cid, "key_cid")
        key_id = self._by_cid.get(key_cid)
        if key_id is None:
            return None
        return self._by_id.get(key_id)

    def register(self, record: VerificationKeyRecord) -> None:
        if not isinstance(record, VerificationKeyRecord):
            raise TrustError("record must be VerificationKeyRecord")
        # Production allowlist cannot admit test-only keys.
        if self._production and record.test_only:
            raise TrustError(
                "production VerificationKeyRegistry cannot register test_only keys"
            )
        if record.setup_origin.value not in self._allowed_setup_origins:
            raise TrustError(
                f"setup_origin {record.setup_origin.value!r} is not allowed by registry policy"
            )
        existing = self._by_id.get(record.key_id)
        if existing is not None:
            if existing.to_canonical() != record.to_canonical():
                raise TrustError(
                    f"verification key {record.key_id!r} already registered with different binding"
                )
            return
        cid_owner = self._by_cid.get(record.key_cid)
        if cid_owner is not None and cid_owner != record.key_id:
            raise TrustError(
                f"verification key CID {record.key_cid!r} already bound to {cid_owner!r}"
            )
        self._by_id[record.key_id] = record
        self._by_cid[record.key_cid] = record.key_id

    def evaluate(
        self,
        key_id: str,
        *,
        key_cid: str | None = None,
        circuit_id: str | None = None,
        current_epoch: int | None = None,
        production: bool | None = None,
    ) -> TrustDecision:
        """Evaluate a verification-key claim against the allowlist."""
        subject_id = _safe_subject_id(key_id)
        try:
            key_id = _require_nonempty_str(key_id, "key_id")
            if key_cid is not None:
                key_cid = _require_nonempty_str(key_cid, "key_cid")
            if circuit_id is not None:
                circuit_id = _require_nonempty_str(circuit_id, "circuit_id")
            epoch = (
                self._minimum_epoch
                if current_epoch is None
                else _require_nonneg_int(current_epoch, "current_epoch")
            )
            prod = self._production if production is None else _require_bool(
                production, "production"
            )
        except TrustError as exc:
            return _reject(
                reason=TrustRejectionReason.MALFORMED_REQUEST,
                subject_kind="verification_key",
                subject_id=subject_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=str(exc),
            )

        record = self._by_id.get(key_id)
        if record is None:
            return _reject(
                reason=TrustRejectionReason.UNALLOWLISTED_VERIFICATION_KEY,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=f"verification key {key_id!r} is not on the allowlist",
            )

        if key_cid is not None and not _cid_matches(key_cid, record.key_cid):
            return _reject(
                reason=TrustRejectionReason.SUBSTITUTED_VERIFICATION_KEY,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"verification key {key_id!r} CID mismatch: "
                    f"claimed {key_cid!r} != allowlisted {record.key_cid!r}"
                ),
                details={
                    "claimed_cid": key_cid,
                    "allowlisted_cid": record.key_cid,
                },
            )

        if record.revoked:
            return _reject(
                reason=TrustRejectionReason.REVOKED_VERIFICATION_KEY,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=f"verification key {key_id!r} is revoked",
            )

        # Superseded or epoch-stale keys are "old".
        if record.superseded_by is not None:
            return _reject(
                reason=TrustRejectionReason.OLD_VERIFICATION_KEY,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"verification key {key_id!r} is superseded by "
                    f"{record.superseded_by!r}"
                ),
                details={"superseded_by": record.superseded_by},
            )
        # Required floor is the registry minimum, optionally raised by the
        # caller's current_epoch when it asserts a higher floor.
        required_epoch = max(self._minimum_epoch, epoch)
        if record.epoch < required_epoch:
            return _reject(
                reason=TrustRejectionReason.OLD_VERIFICATION_KEY,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"verification key {key_id!r} epoch {record.epoch} is below "
                    f"required epoch {required_epoch}"
                ),
                details={
                    "key_epoch": record.epoch,
                    "required_epoch": required_epoch,
                },
            )

        if prod and record.test_only:
            return _reject(
                reason=TrustRejectionReason.TEST_ONLY_IN_PRODUCTION,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"verification key {key_id!r} is test_only and cannot be used "
                    "in production mode"
                ),
            )

        if record.setup_origin.value not in self._allowed_setup_origins:
            return _reject(
                reason=TrustRejectionReason.DISALLOWED_SETUP_ORIGIN,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"verification key {key_id!r} setup_origin "
                    f"{record.setup_origin.value!r} is not allowed"
                ),
            )

        if circuit_id is not None and circuit_id not in record.circuit_ids:
            return _reject(
                reason=TrustRejectionReason.CIRCUIT_INCOMPATIBLE,
                subject_kind="verification_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"verification key {key_id!r} is not compatible with circuit "
                    f"{circuit_id!r}; allowed={sorted(record.circuit_ids)}"
                ),
                details={
                    "circuit_id": circuit_id,
                    "compatible_circuits": sorted(record.circuit_ids),
                },
            )

        return _accept(
            subject_kind="verification_key",
            subject_id=key_id,
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            message=f"verification key {key_id!r} admitted by allowlist",
            details={
                "key_cid": record.key_cid,
                "circuit_ids": sorted(record.circuit_ids),
                "setup_origin": record.setup_origin.value,
                "test_only": record.test_only,
                "epoch": record.epoch,
            },
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": KEY_REGISTRY_SCHEMA,
            "evidence_subset": KEY_REGISTRY_EVIDENCE,
            "production": self._production,
            "minimum_epoch": self._minimum_epoch,
            "allowed_setup_origins": sorted(self._allowed_setup_origins),
            "keys": {
                key_id: record.to_canonical()
                for key_id, record in sorted(self._by_id.items())
            },
        }


class SignerTrustRegistry:
    """Allowlisted signers with scope and revocation epoch."""

    def __init__(
        self,
        entries: Iterable[SignerTrustRecord] | None = None,
        *,
        production: bool = True,
        current_epoch: int = 0,
    ) -> None:
        self._production = _require_bool(production, "production")
        self._current_epoch = _require_nonneg_int(current_epoch, "current_epoch")
        self._by_id: dict[str, SignerTrustRecord] = {}
        if entries is not None:
            for entry in entries:
                self.register(entry)

    @property
    def production(self) -> bool:
        return self._production

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def __contains__(self, signer_id: object) -> bool:
        return isinstance(signer_id, str) and signer_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def get(self, signer_id: str) -> SignerTrustRecord | None:
        signer_id = _require_nonempty_str(signer_id, "signer_id")
        return self._by_id.get(signer_id)

    def register(self, record: SignerTrustRecord) -> None:
        if not isinstance(record, SignerTrustRecord):
            raise TrustError("record must be SignerTrustRecord")
        if self._production and record.test_only:
            raise TrustError(
                "production SignerTrustRegistry cannot register test_only signers"
            )
        existing = self._by_id.get(record.signer_id)
        if existing is not None:
            if existing.to_canonical() != record.to_canonical():
                raise TrustError(
                    f"signer {record.signer_id!r} already registered with different binding"
                )
            return
        self._by_id[record.signer_id] = record

    def evaluate(
        self,
        signer_id: str,
        *,
        scope: str | None = None,
        current_epoch: int | None = None,
        production: bool | None = None,
        required_trusted: bool = True,
    ) -> TrustDecision:
        """Evaluate a signer claim against the trust allowlist."""
        subject_id = _safe_subject_id(signer_id)
        try:
            signer_id = _require_nonempty_str(signer_id, "signer_id")
            if scope is not None:
                scope = _require_nonempty_str(scope, "scope")
            epoch = (
                self._current_epoch
                if current_epoch is None
                else _require_nonneg_int(current_epoch, "current_epoch")
            )
            prod = self._production if production is None else _require_bool(
                production, "production"
            )
            required_trusted = _require_bool(required_trusted, "required_trusted")
        except TrustError as exc:
            return _reject(
                reason=TrustRejectionReason.MALFORMED_REQUEST,
                subject_kind="signer",
                subject_id=subject_id,
                evidence_subset=SIGNER_TRUST_EVIDENCE,
                message=str(exc),
            )

        record = self._by_id.get(signer_id)
        if record is None:
            return _reject(
                reason=TrustRejectionReason.UNTRUSTED_SIGNER,
                subject_kind="signer",
                subject_id=signer_id,
                evidence_subset=SIGNER_TRUST_EVIDENCE,
                message=f"signer {signer_id!r} is not on the trust allowlist",
            )

        if required_trusted and not record.trusted:
            return _reject(
                reason=TrustRejectionReason.UNTRUSTED_SIGNER,
                subject_kind="signer",
                subject_id=signer_id,
                evidence_subset=SIGNER_TRUST_EVIDENCE,
                message=f"signer {signer_id!r} is explicitly untrusted",
            )

        if record.is_revoked_at(epoch):
            return _reject(
                reason=TrustRejectionReason.REVOKED_SIGNER,
                subject_kind="signer",
                subject_id=signer_id,
                evidence_subset=SIGNER_TRUST_EVIDENCE,
                message=(
                    f"signer {signer_id!r} is revoked at epoch {epoch} "
                    f"(revocation_epoch={record.revocation_epoch})"
                ),
                details={
                    "current_epoch": epoch,
                    "revocation_epoch": record.revocation_epoch,
                },
            )

        if prod and record.test_only:
            return _reject(
                reason=TrustRejectionReason.TEST_ONLY_IN_PRODUCTION,
                subject_kind="signer",
                subject_id=signer_id,
                evidence_subset=SIGNER_TRUST_EVIDENCE,
                message=(
                    f"signer {signer_id!r} is test_only and cannot be used "
                    "in production mode"
                ),
            )

        if scope is not None and scope not in record.scopes:
            return _reject(
                reason=TrustRejectionReason.OUT_OF_SCOPE_SIGNER,
                subject_kind="signer",
                subject_id=signer_id,
                evidence_subset=SIGNER_TRUST_EVIDENCE,
                message=(
                    f"signer {signer_id!r} is out of scope for {scope!r}; "
                    f"allowed={sorted(record.scopes)}"
                ),
                details={
                    "requested_scope": scope,
                    "allowed_scopes": sorted(record.scopes),
                },
            )

        return _accept(
            subject_kind="signer",
            subject_id=signer_id,
            evidence_subset=SIGNER_TRUST_EVIDENCE,
            message=f"signer {signer_id!r} admitted by trust allowlist",
            details={
                "scopes": sorted(record.scopes),
                "trusted": record.trusted,
                "test_only": record.test_only,
                "revocation_epoch": record.revocation_epoch,
                "public_key_cid": record.public_key_cid,
            },
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": SIGNER_TRUST_SCHEMA,
            "evidence_subset": SIGNER_TRUST_EVIDENCE,
            "production": self._production,
            "current_epoch": self._current_epoch,
            "signers": {
                signer_id: record.to_canonical()
                for signer_id, record in sorted(self._by_id.items())
            },
        }


class ProvingKeyRegistry:
    """Private registry of proving-key identity bindings.

    Public consumers receive only :class:`ProvingKeyHandle` instances.  Raw
    key material is never stored or returned by this registry.
    """

    def __init__(
        self,
        entries: Iterable[ProvingKeyRecord] | None = None,
        *,
        production: bool = True,
        allowed_setup_origins: frozenset[str] | None = None,
        minimum_epoch: int = 0,
    ) -> None:
        self._production = _require_bool(production, "production")
        self._minimum_epoch = _require_nonneg_int(minimum_epoch, "minimum_epoch")
        if allowed_setup_origins is None:
            allowed_setup_origins = (
                DEFAULT_PRODUCTION_SETUP_ORIGINS
                if self._production
                else DEFAULT_TEST_SETUP_ORIGINS
            )
        self._allowed_setup_origins = _require_str_set(
            allowed_setup_origins, "allowed_setup_origins"
        )
        self._by_id: dict[str, ProvingKeyRecord] = {}
        self._by_cid: dict[str, str] = {}
        if entries is not None:
            for entry in entries:
                self.register(entry)

    @property
    def production(self) -> bool:
        return self._production

    def __contains__(self, key_id: object) -> bool:
        return isinstance(key_id, str) and key_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, key_id: str) -> ProvingKeyRecord | None:
        key_id = _require_nonempty_str(key_id, "key_id")
        return self._by_id.get(key_id)

    def get_handle(self, key_id: str) -> ProvingKeyHandle | None:
        record = self.get(key_id)
        if record is None:
            return None
        return record.to_handle()

    def register(self, record: ProvingKeyRecord) -> None:
        if not isinstance(record, ProvingKeyRecord):
            raise TrustError("record must be ProvingKeyRecord")
        if self._production and record.test_only:
            raise TrustError(
                "production ProvingKeyRegistry cannot register test_only keys"
            )
        if record.setup_origin.value not in self._allowed_setup_origins:
            raise TrustError(
                f"setup_origin {record.setup_origin.value!r} is not allowed by registry policy"
            )
        existing = self._by_id.get(record.key_id)
        if existing is not None:
            if (
                existing.key_cid != record.key_cid
                or existing.paired_verification_key_id
                != record.paired_verification_key_id
                or existing.circuit_ids != record.circuit_ids
                or existing.test_only != record.test_only
                or existing.epoch != record.epoch
            ):
                raise TrustError(
                    f"proving key {record.key_id!r} already registered with different binding"
                )
            return
        cid_owner = self._by_cid.get(record.key_cid)
        if cid_owner is not None and cid_owner != record.key_id:
            raise TrustError(
                f"proving key CID {record.key_cid!r} already bound to {cid_owner!r}"
            )
        self._by_id[record.key_id] = record
        self._by_cid[record.key_cid] = record.key_id

    def evaluate(
        self,
        key_id: str,
        *,
        key_cid: str | None = None,
        circuit_id: str | None = None,
        current_epoch: int | None = None,
        production: bool | None = None,
        paired_verification_key_id: str | None = None,
    ) -> TrustDecision:
        subject_id = _safe_subject_id(key_id)
        try:
            key_id = _require_nonempty_str(key_id, "key_id")
            if key_cid is not None:
                key_cid = _require_nonempty_str(key_cid, "key_cid")
            if circuit_id is not None:
                circuit_id = _require_nonempty_str(circuit_id, "circuit_id")
            if paired_verification_key_id is not None:
                paired_verification_key_id = _require_nonempty_str(
                    paired_verification_key_id, "paired_verification_key_id"
                )
            epoch = (
                self._minimum_epoch
                if current_epoch is None
                else _require_nonneg_int(current_epoch, "current_epoch")
            )
            prod = self._production if production is None else _require_bool(
                production, "production"
            )
        except TrustError as exc:
            return _reject(
                reason=TrustRejectionReason.MALFORMED_REQUEST,
                subject_kind="proving_key",
                subject_id=subject_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=str(exc),
            )

        record = self._by_id.get(key_id)
        if record is None:
            return _reject(
                reason=TrustRejectionReason.UNALLOWLISTED_PROVING_KEY,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=f"proving key {key_id!r} is not on the allowlist",
            )

        if key_cid is not None and not _cid_matches(key_cid, record.key_cid):
            return _reject(
                reason=TrustRejectionReason.SUBSTITUTED_PROVING_KEY,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"proving key {key_id!r} CID mismatch: "
                    f"claimed {key_cid!r} != allowlisted {record.key_cid!r}"
                ),
                details={
                    "claimed_cid": key_cid,
                    "allowlisted_cid": record.key_cid,
                },
            )

        if record.revoked:
            return _reject(
                reason=TrustRejectionReason.REVOKED_PROVING_KEY,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=f"proving key {key_id!r} is revoked",
            )

        if record.superseded_by is not None:
            return _reject(
                reason=TrustRejectionReason.OLD_PROVING_KEY,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"proving key {key_id!r} is superseded by "
                    f"{record.superseded_by!r}"
                ),
                details={"superseded_by": record.superseded_by},
            )
        required_epoch = max(self._minimum_epoch, epoch)
        if record.epoch < required_epoch:
            return _reject(
                reason=TrustRejectionReason.OLD_PROVING_KEY,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"proving key {key_id!r} is old "
                    f"(epoch={record.epoch}, required_epoch={required_epoch})"
                ),
                details={
                    "key_epoch": record.epoch,
                    "required_epoch": required_epoch,
                    "superseded_by": record.superseded_by,
                },
            )

        if prod and record.test_only:
            return _reject(
                reason=TrustRejectionReason.TEST_ONLY_IN_PRODUCTION,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"proving key {key_id!r} is test_only and cannot be used "
                    "in production mode"
                ),
            )

        if circuit_id is not None and circuit_id not in record.circuit_ids:
            return _reject(
                reason=TrustRejectionReason.CIRCUIT_INCOMPATIBLE,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"proving key {key_id!r} is not compatible with circuit "
                    f"{circuit_id!r}; allowed={sorted(record.circuit_ids)}"
                ),
                details={
                    "circuit_id": circuit_id,
                    "compatible_circuits": sorted(record.circuit_ids),
                },
            )

        if (
            paired_verification_key_id is not None
            and paired_verification_key_id != record.paired_verification_key_id
        ):
            return _reject(
                reason=TrustRejectionReason.SUBSTITUTED_PROVING_KEY,
                subject_kind="proving_key",
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    f"proving key {key_id!r} paired verification key mismatch: "
                    f"claimed {paired_verification_key_id!r} != "
                    f"{record.paired_verification_key_id!r}"
                ),
            )

        return _accept(
            subject_kind="proving_key",
            subject_id=key_id,
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            message=f"proving key {key_id!r} admitted by allowlist (handle only)",
            details={
                "key_cid": record.key_cid,
                "circuit_ids": sorted(record.circuit_ids),
                "setup_origin": record.setup_origin.value,
                "test_only": record.test_only,
                "epoch": record.epoch,
                "paired_verification_key_id": record.paired_verification_key_id,
                "handle_only": True,
            },
        )

    def to_canonical(self) -> dict[str, Any]:
        # Export only nonexportable handle projections.
        return {
            "evidence_subset": KEY_REGISTRY_EVIDENCE,
            "production": self._production,
            "minimum_epoch": self._minimum_epoch,
            "handles": {
                key_id: record.to_handle().to_canonical()
                for key_id, record in sorted(self._by_id.items())
            },
            "proving_key_exported": False,
        }


@dataclass(frozen=True, slots=True)
class TrustedProofPolicy:
    """Policy-selected allowlists for verification keys, proving keys, and signers.

    Construction freezes the registries.  Callers evaluate claims; they never
    inject unallowlisted keys, generate production key material, or download
    keys through this policy.
    """

    production: bool = True
    current_epoch: int = 0
    minimum_key_epoch: int = 0
    verification_keys: VerificationKeyRegistry | None = None
    proving_keys: ProvingKeyRegistry | None = None
    signers: SignerTrustRegistry | None = None
    allowed_setup_origins: frozenset[str] | None = None
    policy_id: str = "policy/ips-trusted-proof@1"
    allow_key_generation: bool = False
    allow_key_download: bool = False

    def __post_init__(self) -> None:
        production = _require_bool(self.production, "production")
        current_epoch = _require_nonneg_int(self.current_epoch, "current_epoch")
        minimum_key_epoch = _require_nonneg_int(
            self.minimum_key_epoch, "minimum_key_epoch"
        )
        object.__setattr__(self, "production", production)
        object.__setattr__(self, "current_epoch", current_epoch)
        object.__setattr__(self, "minimum_key_epoch", minimum_key_epoch)
        object.__setattr__(
            self, "policy_id", _require_nonempty_str(self.policy_id, "policy_id")
        )

        # Production hard-forbids generation and download regardless of flags.
        if production:
            object.__setattr__(self, "allow_key_generation", False)
            object.__setattr__(self, "allow_key_download", False)
        else:
            object.__setattr__(
                self,
                "allow_key_generation",
                _require_bool(self.allow_key_generation, "allow_key_generation"),
            )
            object.__setattr__(
                self,
                "allow_key_download",
                _require_bool(self.allow_key_download, "allow_key_download"),
            )

        if self.allowed_setup_origins is None:
            origins = (
                DEFAULT_PRODUCTION_SETUP_ORIGINS
                if production
                else DEFAULT_TEST_SETUP_ORIGINS
            )
        else:
            origins = _require_str_set(
                self.allowed_setup_origins, "allowed_setup_origins"
            )
        object.__setattr__(self, "allowed_setup_origins", origins)

        vk = self.verification_keys
        if vk is None:
            vk = VerificationKeyRegistry(
                production=production,
                allowed_setup_origins=origins,
                minimum_epoch=minimum_key_epoch,
            )
        elif not isinstance(vk, VerificationKeyRegistry):
            raise TrustError("verification_keys must be VerificationKeyRegistry")
        object.__setattr__(self, "verification_keys", vk)

        pk = self.proving_keys
        if pk is None:
            pk = ProvingKeyRegistry(
                production=production,
                allowed_setup_origins=origins,
                minimum_epoch=minimum_key_epoch,
            )
        elif not isinstance(pk, ProvingKeyRegistry):
            raise TrustError("proving_keys must be ProvingKeyRegistry")
        object.__setattr__(self, "proving_keys", pk)

        signers = self.signers
        if signers is None:
            signers = SignerTrustRegistry(
                production=production, current_epoch=current_epoch
            )
        elif not isinstance(signers, SignerTrustRegistry):
            raise TrustError("signers must be SignerTrustRegistry")
        object.__setattr__(self, "signers", signers)

    # ------------------------------------------------------------------
    # Selection (policy-owned; not caller-invented allowlists)
    # ------------------------------------------------------------------

    def select_verification_key(
        self,
        key_id: str,
        *,
        key_cid: str | None = None,
        circuit_id: str | None = None,
    ) -> TrustDecision:
        """Admit a verification key only if already allowlisted by policy."""
        return self.verification_keys.evaluate(
            key_id,
            key_cid=key_cid,
            circuit_id=circuit_id,
            current_epoch=self.minimum_key_epoch,
            production=self.production,
        )

    def select_proving_key_handle(
        self,
        key_id: str,
        *,
        key_cid: str | None = None,
        circuit_id: str | None = None,
        paired_verification_key_id: str | None = None,
    ) -> tuple[TrustDecision, ProvingKeyHandle | None]:
        """Return a nonexportable proving-key handle when allowlisted."""
        decision = self.proving_keys.evaluate(
            key_id,
            key_cid=key_cid,
            circuit_id=circuit_id,
            current_epoch=self.minimum_key_epoch,
            production=self.production,
            paired_verification_key_id=paired_verification_key_id,
        )
        if not decision.accepted:
            return decision, None
        handle = self.proving_keys.get_handle(key_id)
        if handle is None:
            # Should be unreachable after accept, but fail closed.
            return (
                _reject(
                    reason=TrustRejectionReason.UNALLOWLISTED_PROVING_KEY,
                    subject_kind="proving_key",
                    subject_id=key_id,
                    evidence_subset=KEY_REGISTRY_EVIDENCE,
                    message=f"proving key {key_id!r} handle vanished after admit",
                ),
                None,
            )
        return decision, handle

    def select_signer(
        self,
        signer_id: str,
        *,
        scope: str | None = None,
    ) -> TrustDecision:
        """Admit a signer only if allowlisted, in-scope, and not revoked."""
        return self.signers.evaluate(
            signer_id,
            scope=scope,
            current_epoch=self.current_epoch,
            production=self.production,
        )

    def evaluate_key_pair(
        self,
        *,
        verification_key_id: str,
        proving_key_id: str,
        circuit_id: str,
        verification_key_cid: str | None = None,
        proving_key_cid: str | None = None,
    ) -> TrustDecision:
        """Joint evaluation of paired verification + proving keys for a circuit."""
        vk_decision = self.select_verification_key(
            verification_key_id,
            key_cid=verification_key_cid,
            circuit_id=circuit_id,
        )
        if not vk_decision.accepted:
            return vk_decision
        pk_decision, _handle = self.select_proving_key_handle(
            proving_key_id,
            key_cid=proving_key_cid,
            circuit_id=circuit_id,
            paired_verification_key_id=verification_key_id,
        )
        if not pk_decision.accepted:
            return pk_decision
        return _accept(
            subject_kind="key_pair",
            subject_id=f"{verification_key_id}+{proving_key_id}",
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            message=(
                f"key pair verification_key={verification_key_id!r} "
                f"proving_key={proving_key_id!r} admitted for circuit {circuit_id!r}"
            ),
            details={
                "verification_key_id": verification_key_id,
                "proving_key_id": proving_key_id,
                "circuit_id": circuit_id,
                "handle_only": True,
            },
        )

    # ------------------------------------------------------------------
    # Production hard gates: never generate or download key material
    # ------------------------------------------------------------------

    def generate_key_material(
        self,
        *,
        kind: str = "verification_key",
        circuit_id: str | None = None,
    ) -> TrustDecision:
        """Reject production key generation; test mode still does not emit bytes."""
        kind = str(kind or "verification_key")
        if self.production or not self.allow_key_generation:
            return _reject(
                reason=TrustRejectionReason.KEY_GENERATION_FORBIDDEN,
                subject_kind=kind,
                subject_id=circuit_id or self.policy_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    "production mode never generates key material"
                    if self.production
                    else "key generation is disabled by TrustedProofPolicy"
                ),
                details={
                    "production": self.production,
                    "allow_key_generation": False if self.production else self.allow_key_generation,
                    "key_material_generated": False,
                },
            )
        # Even when explicitly allowed in non-production, this policy surface
        # never fabricates material; it only reports that generation is gated.
        return _reject(
            reason=TrustRejectionReason.KEY_GENERATION_FORBIDDEN,
            subject_kind=kind,
            subject_id=circuit_id or self.policy_id,
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            message=(
                "TrustedProofPolicy never generates key material; "
                "keys must be preconfigured and allowlisted"
            ),
            details={
                "production": self.production,
                "allow_key_generation": self.allow_key_generation,
                "key_material_generated": False,
            },
        )

    def download_key_material(
        self,
        *,
        key_id: str,
        kind: str = "verification_key",
        source: str | None = None,
    ) -> TrustDecision:
        """Reject production key download; no network fetch is performed."""
        subject_id = _safe_subject_id(key_id)
        try:
            key_id = _require_nonempty_str(key_id, "key_id")
        except TrustError as exc:
            return _reject(
                reason=TrustRejectionReason.MALFORMED_REQUEST,
                subject_kind=str(kind or "verification_key"),
                subject_id=subject_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=str(exc),
            )
        kind = str(kind or "verification_key")
        if self.production or not self.allow_key_download:
            return _reject(
                reason=TrustRejectionReason.KEY_DOWNLOAD_FORBIDDEN,
                subject_kind=kind,
                subject_id=key_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=(
                    "production mode never downloads key material"
                    if self.production
                    else "key download is disabled by TrustedProofPolicy"
                ),
                details={
                    "production": self.production,
                    "allow_key_download": False if self.production else self.allow_key_download,
                    "source": source,
                    "key_material_downloaded": False,
                },
            )
        return _reject(
            reason=TrustRejectionReason.KEY_DOWNLOAD_FORBIDDEN,
            subject_kind=kind,
            subject_id=key_id,
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            message=(
                "TrustedProofPolicy never downloads key material; "
                "keys must be preconfigured and allowlisted"
            ),
            details={
                "production": self.production,
                "allow_key_download": self.allow_key_download,
                "source": source,
                "key_material_downloaded": False,
            },
        )

    def export_proving_key_bytes(self, key_id: str) -> TrustDecision:
        """Always reject export of proving-key bytes."""
        subject_id = _safe_subject_id(key_id)
        try:
            key_id = _require_nonempty_str(key_id, "key_id")
        except TrustError as exc:
            return _reject(
                reason=TrustRejectionReason.MALFORMED_REQUEST,
                subject_kind="proving_key",
                subject_id=subject_id,
                evidence_subset=KEY_REGISTRY_EVIDENCE,
                message=str(exc),
            )
        return _reject(
            reason=TrustRejectionReason.PRODUCTION_KEY_EXPORT_FORBIDDEN,
            subject_kind="proving_key",
            subject_id=key_id,
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            message="proving-key bytes are nonexportable private handles",
            details={"proving_key_exported": False},
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": TRUSTED_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "production": self.production,
            "current_epoch": self.current_epoch,
            "minimum_key_epoch": self.minimum_key_epoch,
            "allowed_setup_origins": sorted(self.allowed_setup_origins),
            "allow_key_generation": self.allow_key_generation,
            "allow_key_download": self.allow_key_download,
            "verification_keys": self.verification_keys.to_canonical(),
            "proving_keys": self.proving_keys.to_canonical(),
            "signers": self.signers.to_canonical(),
            "evidence_subsets": [
                KEY_REGISTRY_EVIDENCE,
                SIGNER_TRUST_EVIDENCE,
            ],
            "key_material_generated": False,
            "key_material_downloaded": False,
            "proving_key_exported": False,
        }

    def policy_digest(self) -> str:
        return _sha256_hex(_canonical_json(self.to_canonical()))


def build_production_policy(
    *,
    verification_keys: Sequence[VerificationKeyRecord] = (),
    proving_keys: Sequence[ProvingKeyRecord] = (),
    signers: Sequence[SignerTrustRecord] = (),
    current_epoch: int = 0,
    minimum_key_epoch: int = 0,
    policy_id: str = "policy/ips-trusted-proof@1",
) -> TrustedProofPolicy:
    """Construct a production policy from preconfigured allowlist entries.

    Test-only entries are rejected at registration time.
    """
    origins = DEFAULT_PRODUCTION_SETUP_ORIGINS
    vk_registry = VerificationKeyRegistry(
        production=True,
        allowed_setup_origins=origins,
        minimum_epoch=minimum_key_epoch,
    )
    for record in verification_keys:
        vk_registry.register(record)

    pk_registry = ProvingKeyRegistry(
        production=True,
        allowed_setup_origins=origins,
        minimum_epoch=minimum_key_epoch,
    )
    for record in proving_keys:
        pk_registry.register(record)

    signer_registry = SignerTrustRegistry(
        production=True, current_epoch=current_epoch
    )
    for record in signers:
        signer_registry.register(record)

    return TrustedProofPolicy(
        production=True,
        current_epoch=current_epoch,
        minimum_key_epoch=minimum_key_epoch,
        verification_keys=vk_registry,
        proving_keys=pk_registry,
        signers=signer_registry,
        allowed_setup_origins=origins,
        policy_id=policy_id,
        allow_key_generation=False,
        allow_key_download=False,
    )


def build_test_policy(
    *,
    verification_keys: Sequence[VerificationKeyRecord] = (),
    proving_keys: Sequence[ProvingKeyRecord] = (),
    signers: Sequence[SignerTrustRecord] = (),
    current_epoch: int = 0,
    minimum_key_epoch: int = 0,
    policy_id: str = "policy/ips-trusted-proof-test@1",
) -> TrustedProofPolicy:
    """Construct a non-production policy that may include test-only keys/signers.

    Generation and download remain disabled; fixtures must be preconfigured.
    """
    origins = DEFAULT_TEST_SETUP_ORIGINS
    vk_registry = VerificationKeyRegistry(
        production=False,
        allowed_setup_origins=origins,
        minimum_epoch=minimum_key_epoch,
    )
    for record in verification_keys:
        vk_registry.register(record)

    pk_registry = ProvingKeyRegistry(
        production=False,
        allowed_setup_origins=origins,
        minimum_epoch=minimum_key_epoch,
    )
    for record in proving_keys:
        pk_registry.register(record)

    signer_registry = SignerTrustRegistry(
        production=False, current_epoch=current_epoch
    )
    for record in signers:
        signer_registry.register(record)

    return TrustedProofPolicy(
        production=False,
        current_epoch=current_epoch,
        minimum_key_epoch=minimum_key_epoch,
        verification_keys=vk_registry,
        proving_keys=pk_registry,
        signers=signer_registry,
        allowed_setup_origins=origins,
        policy_id=policy_id,
        allow_key_generation=False,
        allow_key_download=False,
    )
