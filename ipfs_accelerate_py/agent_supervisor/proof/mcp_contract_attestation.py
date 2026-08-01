"""Capability-checked attestation of MCP contract proof receipts.

This module is the narrow SwissKnife adapter around the generic
``proof_attestation`` contracts and optional datasets ZKP providers.  It does
not use ZK as a substitute for a property proof.  A statement can be built
only from a current, independently kernel-verified receipt which agrees with
its :class:`~.mcp_contract_proof_cache.ProofCacheKey`.

The public statement is deliberately closed.  Every content identity is a
``(CID, identity-profile-id)`` pair and the statement binds receipt, cache,
property, obligation, snapshot, policy, predicate, backend, setup, keys,
result-set root, capability report, verifier domain, challenge, and expiry.
The private witness is a separate single-use object.  It cannot be serialized
and its mutable buffers are overwritten after every proving attempt.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeVar

from ..analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    STRICT_ARTIFACT_PROFILE,
    identify_strict_artifact,
)
from .formal_verification_capabilities import CapabilityHealth
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    ProofReceipt,
    canonical_json_bytes,
)
from .mcp_contract_proof_cache import (
    IdentityBinding,
    ProofCacheKey,
    ProofCacheReason,
    TrustAwareProofCache,
)
from .proof_attestation import (
    AttestationBackendMode,
    ZkUseCaseDisposition,
)


MCP_CONTRACT_ATTESTATION_VERSION: Final = 1
MCP_CONTRACT_ATTESTATION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation-policy@1"
)
MCP_CONTRACT_ATTESTATION_PIN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation-pin@1"
)
MCP_CONTRACT_ATTESTATION_SETUP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation-setup@1"
)
MCP_CONTRACT_ATTESTATION_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation-capability@1"
)
MCP_CONTRACT_ATTESTATION_PUBLIC_INPUTS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation-inputs@1"
)
MCP_CONTRACT_ATTESTATION_ENVELOPE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation@1"
)
MCP_CONTRACT_ATTESTATION_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-attestation-verification@1"
)
MCP_CONTRACT_ATTESTATION_INTERFACE: Final = "ProofAttestation@1"

DATASETS_ZKP_BRIDGE_MODULE: Final = (
    "ipfs_datasets_py.logic.bridge.zkp_attestation"
)
MAX_PROOF_BYTES: Final = 1024 * 1024
MIN_CRYPTOGRAPHIC_PROOF_BYTES: Final = 8


class McpAttestationError(ValueError):
    """A fail-closed MCP attestation validation error."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class WitnessDisclosureError(McpAttestationError):
    """Private witness material reached a forbidden public boundary."""

    def __init__(self, message: str = "private witness cannot be serialized") -> None:
        super().__init__(message, reason_code="witness_disclosure")


class AttestationPredicateKind(str, Enum):
    """Closed predicate vocabulary approved by the SCA-080 policy."""

    RECEIPT_POSSESSION = "receipt_possession"
    RECEIPT_MEMBERSHIP = "receipt_membership"
    PRIVATE_REVIEWED_PREDICATE = "private_reviewed_predicate"


class AttestationStatus(str, Enum):
    """Typed outcome; only ``attested`` contributes ATTESTED assurance."""

    GENERATED = "generated"
    ATTESTED = "attested"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    REJECTED = "rejected"
    ERROR = "error"


class CapabilityFixture(str, Enum):
    """Evidence required before a real backend is production eligible."""

    GOLDEN = "golden"
    NEGATIVE_FALSE_WITNESS = "negative_false_witness"
    PUBLIC_INPUT_SUBSTITUTION = "public_input_substitution"
    STALE_OR_WRONG_KEY_SETUP = "stale_or_wrong_key_setup"
    MALFORMED_PROOF = "malformed_proof"
    REPLAY_FRESHNESS = "replay_freshness"
    CROSS_PROFILE_IDENTITY = "cross_profile_identity"
    WITNESS_NO_LEAK = "witness_no_leak"


REQUIRED_CAPABILITY_FIXTURES: Final = tuple(CapabilityFixture)


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise McpAttestationError(
            f"{field_name} must be a non-empty string",
            reason_code="invalid_schema",
        )
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise McpAttestationError(
            f"{field_name} must be a positive integer",
            reason_code="invalid_schema",
        )
    return value


def _timestamp(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise McpAttestationError(
            f"{field_name} must be an RFC3339 timestamp",
            reason_code="invalid_timestamp",
        ) from exc
    if parsed.tzinfo is None:
        raise McpAttestationError(
            f"{field_name} must include a timezone",
            reason_code="invalid_timestamp",
        )
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _time_value(value: str) -> datetime:
    return datetime.fromisoformat(
        value[:-1] + "+00:00" if value.endswith("Z") else value
    )


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise McpAttestationError(
            "canonical artifact must be an object",
            reason_code="invalid_schema",
        )
    if payload.get("schema") != expected:
        raise McpAttestationError(
            "canonical artifact has an unsupported schema",
            reason_code="invalid_schema",
        )


def _closed_fields(
    payload: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
) -> None:
    if set(payload).difference(allowed):
        raise McpAttestationError(
            "canonical artifact contains unsupported fields",
            reason_code="invalid_schema",
        )


def _enum(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise McpAttestationError(
            f"{field_name} has an unsupported value",
            reason_code="invalid_schema",
        ) from exc


@dataclass(frozen=True, slots=True)
class AttestationIdentityPin(CanonicalContract):
    """Public projection of one preimage-validated content identity."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_PIN_SCHEMA

    logical_id: str
    # ``CanonicalContract`` exposes a derived ``cid`` property.  ``field()``
    # explicitly replaces that descriptor with this pin's bound CID value.
    cid: str = field()
    identity_profile_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_id", _required_text(self.logical_id, "logical_id")
        )
        cid = _required_text(self.cid, "cid")
        if cid != cid.lower() or not cid.startswith("b") or len(cid) < 10:
            raise McpAttestationError(
                "cid must be a lowercase CIDv1 base32 string",
                reason_code="invalid_cid",
            )
        object.__setattr__(self, "cid", cid)
        profile = _required_text(
            self.identity_profile_id, "identity_profile_id"
        )
        if profile not in {STRICT_ARTIFACT_PROFILE, LOGIC_IR_PROFILE}:
            raise McpAttestationError(
                "identity profile is not supported",
                reason_code="identity_profile_mismatch",
            )
        object.__setattr__(self, "identity_profile_id", profile)

    @classmethod
    def from_binding(cls, binding: IdentityBinding) -> "AttestationIdentityPin":
        if not isinstance(binding, IdentityBinding):
            raise McpAttestationError(
                "identity pin requires a validated IdentityBinding",
                reason_code="identity_invalid",
            )
        # IdentityBinding construction has already recomputed the CID from the
        # retained canonical bytes.
        return cls(
            logical_id=binding.logical_id,
            cid=binding.cid,
            identity_profile_id=binding.profile,
        )

    @classmethod
    def for_artifact(
        cls, value: Any, *, logical_id: str
    ) -> "AttestationIdentityPin":
        identity = identify_strict_artifact(value)
        return cls(
            logical_id=logical_id,
            cid=identity.cid,
            identity_profile_id=identity.profile,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            "logical_id": self.logical_id,
            "cid": self.cid,
            "identity_profile_id": self.identity_profile_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationIdentityPin":
        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                "logical_id",
                "cid",
                "identity_profile_id",
                "content_id",
            },
        )
        result = cls(
            logical_id=payload.get("logical_id", ""),
            cid=payload.get("cid", ""),
            identity_profile_id=payload.get("identity_profile_id", ""),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise McpAttestationError(
                "identity pin content ID does not match its payload",
                reason_code="forged_root",
            )
        return result


def _pin(value: Any, field_name: str) -> AttestationIdentityPin:
    if isinstance(value, AttestationIdentityPin):
        return value
    if isinstance(value, IdentityBinding):
        return AttestationIdentityPin.from_binding(value)
    if isinstance(value, Mapping):
        return AttestationIdentityPin.from_dict(value)
    raise McpAttestationError(
        f"{field_name} must be an attestation identity pin",
        reason_code="identity_invalid",
    )


@dataclass(frozen=True, slots=True)
class ProofAttestationPolicy(CanonicalContract):
    """Immutable reviewed policy for one exact attestation predicate."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_POLICY_SCHEMA

    use_case_id: str
    disposition: ZkUseCaseDisposition
    predicate_kind: AttestationPredicateKind
    use_case_decision: AttestationIdentityPin
    predicate_manifest: AttestationIdentityPin
    verifier_domain: str
    reviewed_by: str
    reviewed_at: str
    expires_at: str
    qualifying_private_witness: bool
    qualifying_cross_trust_boundary: bool
    authorized_backend_families: tuple[str, ...] = ()
    required_base_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    max_proof_age_seconds: int = 900
    result_set_root_required: bool = False
    witness_isolation_mode: str = "local-ephemeral-zeroizing-v1"

    def __post_init__(self) -> None:
        for name in ("use_case_id", "verifier_domain", "reviewed_by"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ZkUseCaseDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "predicate_kind",
            _enum(
                self.predicate_kind,
                AttestationPredicateKind,
                "predicate_kind",
            ),
        )
        object.__setattr__(
            self,
            "use_case_decision",
            _pin(self.use_case_decision, "use_case_decision"),
        )
        object.__setattr__(
            self,
            "predicate_manifest",
            _pin(self.predicate_manifest, "predicate_manifest"),
        )
        object.__setattr__(
            self, "reviewed_at", _timestamp(self.reviewed_at, "reviewed_at")
        )
        object.__setattr__(
            self, "expires_at", _timestamp(self.expires_at, "expires_at")
        )
        if _time_value(self.expires_at) <= _time_value(self.reviewed_at):
            raise McpAttestationError(
                "policy expiry must follow review time",
                reason_code="policy_expired",
            )
        for name in (
            "qualifying_private_witness",
            "qualifying_cross_trust_boundary",
            "result_set_root_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise McpAttestationError(
                    f"{name} must be a boolean",
                    reason_code="invalid_schema",
                )
        try:
            required_assurance = AssuranceLevel(self.required_base_assurance)
        except ValueError as exc:
            raise McpAttestationError(
                "required_base_assurance is invalid",
                reason_code="invalid_schema",
            ) from exc
        if required_assurance.rank < AssuranceLevel.KERNEL_VERIFIED.rank:
            raise McpAttestationError(
                "attestation requires at least kernel-verified base assurance",
                reason_code="insufficient_base_assurance",
            )
        object.__setattr__(
            self, "required_base_assurance", required_assurance
        )
        object.__setattr__(
            self,
            "max_proof_age_seconds",
            _positive_int(self.max_proof_age_seconds, "max_proof_age_seconds"),
        )
        object.__setattr__(
            self,
            "witness_isolation_mode",
            _required_text(
                self.witness_isolation_mode, "witness_isolation_mode"
            ),
        )
        families = tuple(
            sorted(
                {
                    _required_text(item, "authorized_backend_families").lower()
                    for item in self.authorized_backend_families
                }
            )
        )
        object.__setattr__(self, "authorized_backend_families", families)
        if self.predicate_kind is AttestationPredicateKind.RECEIPT_MEMBERSHIP:
            if not self.result_set_root_required:
                raise McpAttestationError(
                    "receipt membership requires a result-set root",
                    reason_code="invalid_policy",
                )
        elif self.result_set_root_required:
            raise McpAttestationError(
                "only receipt membership may require a result-set root",
                reason_code="invalid_policy",
            )
        if self.disposition is ZkUseCaseDisposition.APPROVED:
            if not (
                self.qualifying_private_witness
                and self.qualifying_cross_trust_boundary
                and families
            ):
                raise McpAttestationError(
                    "approved policy requires witness, trust boundary, and backend",
                    reason_code="invalid_policy",
                )
        elif families:
            raise McpAttestationError(
                "non-approved policy cannot authorize a backend",
                reason_code="invalid_policy",
            )

    @property
    def policy_id(self) -> str:
        return self.content_id

    @property
    def terminal_without_backend(self) -> bool:
        return self.disposition in {
            ZkUseCaseDisposition.NOT_APPLICABLE,
            ZkUseCaseDisposition.REJECTED,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            "use_case_id": self.use_case_id,
            "disposition": self.disposition,
            "predicate_kind": self.predicate_kind,
            "use_case_decision": self.use_case_decision,
            "predicate_manifest": self.predicate_manifest,
            "verifier_domain": self.verifier_domain,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "expires_at": self.expires_at,
            "qualifying_private_witness": self.qualifying_private_witness,
            "qualifying_cross_trust_boundary": (
                self.qualifying_cross_trust_boundary
            ),
            "authorized_backend_families": self.authorized_backend_families,
            "required_base_assurance": self.required_base_assurance,
            "max_proof_age_seconds": self.max_proof_age_seconds,
            "result_set_root_required": self.result_set_root_required,
            "witness_isolation_mode": self.witness_isolation_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofAttestationPolicy":
        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                "use_case_id",
                "disposition",
                "predicate_kind",
                "use_case_decision",
                "predicate_manifest",
                "verifier_domain",
                "reviewed_by",
                "reviewed_at",
                "expires_at",
                "qualifying_private_witness",
                "qualifying_cross_trust_boundary",
                "authorized_backend_families",
                "required_base_assurance",
                "max_proof_age_seconds",
                "result_set_root_required",
                "witness_isolation_mode",
                "policy_id",
                "content_id",
            },
        )
        result = cls(
            use_case_id=payload.get("use_case_id", ""),
            disposition=payload.get("disposition", ""),
            predicate_kind=payload.get("predicate_kind", ""),
            use_case_decision=payload.get("use_case_decision"),
            predicate_manifest=payload.get("predicate_manifest"),
            verifier_domain=payload.get("verifier_domain", ""),
            reviewed_by=payload.get("reviewed_by", ""),
            reviewed_at=payload.get("reviewed_at", ""),
            expires_at=payload.get("expires_at", ""),
            qualifying_private_witness=payload.get(
                "qualifying_private_witness", False
            ),
            qualifying_cross_trust_boundary=payload.get(
                "qualifying_cross_trust_boundary", False
            ),
            authorized_backend_families=tuple(
                payload.get("authorized_backend_families") or ()
            ),
            required_base_assurance=payload.get(
                "required_base_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            max_proof_age_seconds=payload.get("max_proof_age_seconds", 900),
            result_set_root_required=payload.get(
                "result_set_root_required", False
            ),
            witness_isolation_mode=payload.get(
                "witness_isolation_mode", "local-ephemeral-zeroizing-v1"
            ),
        )
        claimed = payload.get("policy_id") or payload.get("content_id")
        if claimed not in (None, "", result.policy_id):
            raise McpAttestationError(
                "attestation policy identity does not match payload",
                reason_code="policy_mismatch",
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "policy_id": self.policy_id}

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


@dataclass(frozen=True, slots=True)
class AttestationBackendSetup(CanonicalContract):
    """All immutable backend, circuit, setup, and key pins."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_SETUP_SCHEMA

    backend_family: str
    backend_mode: AttestationBackendMode
    backend_policy: AttestationIdentityPin
    backend_implementation: AttestationIdentityPin
    setup_manifest: AttestationIdentityPin
    circuit: AttestationIdentityPin
    public_input_schema: AttestationIdentityPin
    proving_key: AttestationIdentityPin
    verification_key: AttestationIdentityPin
    backend_version: str
    circuit_version: str
    setup_version: str
    key_epoch: str
    verification_key_expires_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_family",
            _required_text(self.backend_family, "backend_family").lower(),
        )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(
                self.backend_mode, AttestationBackendMode, "backend_mode"
            ),
        )
        for name in (
            "backend_policy",
            "backend_implementation",
            "setup_manifest",
            "circuit",
            "public_input_schema",
            "proving_key",
            "verification_key",
        ):
            object.__setattr__(self, name, _pin(getattr(self, name), name))
        for name in (
            "backend_version",
            "circuit_version",
            "setup_version",
            "key_epoch",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "verification_key_expires_at",
            _timestamp(
                self.verification_key_expires_at,
                "verification_key_expires_at",
            ),
        )

    @property
    def setup_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            "backend_family": self.backend_family,
            "backend_mode": self.backend_mode,
            "backend_policy": self.backend_policy,
            "backend_implementation": self.backend_implementation,
            "setup_manifest": self.setup_manifest,
            "circuit": self.circuit,
            "public_input_schema": self.public_input_schema,
            "proving_key": self.proving_key,
            "verification_key": self.verification_key,
            "backend_version": self.backend_version,
            "circuit_version": self.circuit_version,
            "setup_version": self.setup_version,
            "key_epoch": self.key_epoch,
            "verification_key_expires_at": self.verification_key_expires_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationBackendSetup":
        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                "backend_family",
                "backend_mode",
                "backend_policy",
                "backend_implementation",
                "setup_manifest",
                "circuit",
                "public_input_schema",
                "proving_key",
                "verification_key",
                "backend_version",
                "circuit_version",
                "setup_version",
                "key_epoch",
                "verification_key_expires_at",
                "setup_id",
                "content_id",
            },
        )
        result = cls(
            backend_family=payload.get("backend_family", ""),
            backend_mode=payload.get("backend_mode", ""),
            backend_policy=payload.get("backend_policy"),
            backend_implementation=payload.get("backend_implementation"),
            setup_manifest=payload.get("setup_manifest"),
            circuit=payload.get("circuit"),
            public_input_schema=payload.get("public_input_schema"),
            proving_key=payload.get("proving_key"),
            verification_key=payload.get("verification_key"),
            backend_version=payload.get("backend_version", ""),
            circuit_version=payload.get("circuit_version", ""),
            setup_version=payload.get("setup_version", ""),
            key_epoch=payload.get("key_epoch", ""),
            verification_key_expires_at=payload.get(
                "verification_key_expires_at", ""
            ),
        )
        claimed = payload.get("setup_id") or payload.get("content_id")
        if claimed not in (None, "", result.setup_id):
            raise McpAttestationError(
                "backend setup identity does not match payload",
                reason_code="setup_mismatch",
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "setup_id": self.setup_id}

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


@dataclass(frozen=True, slots=True)
class AttestationCapabilityReport(CanonicalContract):
    """Current capability evidence for one exact backend setup."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_CAPABILITY_SCHEMA

    setup: AttestationBackendSetup
    health: CapabilityHealth
    configured: bool
    available: bool
    fixture_results: Mapping[str, bool]
    evaluated_at: str
    expires_at: str

    def __post_init__(self) -> None:
        setup = (
            self.setup
            if isinstance(self.setup, AttestationBackendSetup)
            else AttestationBackendSetup.from_dict(self.setup)
        )
        object.__setattr__(self, "setup", setup)
        object.__setattr__(
            self,
            "health",
            _enum(self.health, CapabilityHealth, "health"),
        )
        if not isinstance(self.configured, bool) or not isinstance(
            self.available, bool
        ):
            raise McpAttestationError(
                "configured and available must be booleans",
                reason_code="invalid_schema",
            )
        if self.available and not self.configured:
            raise McpAttestationError(
                "available backend must be configured",
                reason_code="capability_invalid",
            )
        if not isinstance(self.fixture_results, Mapping):
            raise McpAttestationError(
                "fixture_results must be an object",
                reason_code="capability_invalid",
            )
        results: dict[str, bool] = {}
        allowed = {fixture.value for fixture in REQUIRED_CAPABILITY_FIXTURES}
        if set(self.fixture_results).difference(allowed):
            raise McpAttestationError(
                "capability report contains an unknown fixture",
                reason_code="capability_invalid",
            )
        for fixture in REQUIRED_CAPABILITY_FIXTURES:
            value = self.fixture_results.get(fixture.value, False)
            if not isinstance(value, bool):
                raise McpAttestationError(
                    "capability fixture outcomes must be booleans",
                    reason_code="capability_invalid",
                )
            results[fixture.value] = value
        object.__setattr__(
            self, "fixture_results", MappingProxyType(results)
        )
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, "evaluated_at")
        )
        object.__setattr__(
            self, "expires_at", _timestamp(self.expires_at, "expires_at")
        )
        if _time_value(self.expires_at) <= _time_value(self.evaluated_at):
            raise McpAttestationError(
                "capability expiry must follow evaluation",
                reason_code="capability_invalid",
            )
        if self.health is CapabilityHealth.VERIFIED and not (
            self.configured
            and self.available
            and all(results.values())
            and setup.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
        ):
            raise McpAttestationError(
                "verified capability requires a real backend and every fixture",
                reason_code="capability_invalid",
            )
        if (
            setup.backend_mode is AttestationBackendMode.SIMULATED
            and self.health is not CapabilityHealth.SIMULATED
        ):
            raise McpAttestationError(
                "simulated setup must remain simulated in capability reports",
                reason_code="simulation_promotion",
            )

    @property
    def capability_id(self) -> str:
        return self.content_id

    @property
    def production_eligible(self) -> bool:
        return (
            self.health is CapabilityHealth.VERIFIED
            and self.configured
            and self.available
            and all(self.fixture_results.values())
            and self.setup.backend_mode
            is AttestationBackendMode.CRYPTOGRAPHIC
        )

    def current_at(self, timestamp: str) -> bool:
        checked = _time_value(_timestamp(timestamp, "timestamp"))
        return (
            _time_value(self.evaluated_at)
            <= checked
            < _time_value(self.expires_at)
            and checked < _time_value(self.setup.verification_key_expires_at)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            "setup": self.setup,
            "setup_id": self.setup.setup_id,
            "health": self.health,
            "configured": self.configured,
            "available": self.available,
            "fixture_results": dict(self.fixture_results),
            "evaluated_at": self.evaluated_at,
            "expires_at": self.expires_at,
            "production_eligible": self.production_eligible,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AttestationCapabilityReport":
        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                "setup",
                "setup_id",
                "health",
                "configured",
                "available",
                "fixture_results",
                "evaluated_at",
                "expires_at",
                "production_eligible",
                "capability_id",
                "content_id",
            },
        )
        result = cls(
            setup=payload.get("setup"),
            health=payload.get("health", ""),
            configured=payload.get("configured", False),
            available=payload.get("available", False),
            fixture_results=payload.get("fixture_results") or {},
            evaluated_at=payload.get("evaluated_at", ""),
            expires_at=payload.get("expires_at", ""),
        )
        derived = {
            "setup_id": result.setup.setup_id,
            "production_eligible": result.production_eligible,
        }
        for name, expected in derived.items():
            if payload.get(name) not in (None, "", expected):
                raise McpAttestationError(
                    "capability report contains a forged derived field",
                    reason_code="capability_drift",
                )
        claimed = payload.get("capability_id") or payload.get("content_id")
        if claimed not in (None, "", result.capability_id):
            raise McpAttestationError(
                "capability report identity does not match payload",
                reason_code="capability_drift",
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "capability_id": self.capability_id}

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


PUBLIC_IDENTITY_FIELDS: Final = (
    "receipt",
    "cache_key",
    "property",
    "obligation",
    "snapshot",
    "scope_root",
    "contract_proof_policy",
    "use_case_decision",
    "attestation_policy",
    "predicate_manifest",
    "backend_policy",
    "backend_implementation",
    "setup_manifest",
    "circuit",
    "public_input_schema",
    "proving_key",
    "verification_key",
    "result_set_root",
    "capability_report",
)


@dataclass(frozen=True, slots=True)
class AttestationPublicInputs(CanonicalContract):
    """Complete circuit-facing public statement with no ambient defaults."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_PUBLIC_INPUTS_SCHEMA

    receipt: AttestationIdentityPin
    cache_key: AttestationIdentityPin
    property: AttestationIdentityPin
    obligation: AttestationIdentityPin
    snapshot: AttestationIdentityPin
    scope_root: AttestationIdentityPin
    contract_proof_policy: AttestationIdentityPin
    use_case_decision: AttestationIdentityPin
    attestation_policy: AttestationIdentityPin
    predicate_manifest: AttestationIdentityPin
    backend_policy: AttestationIdentityPin
    backend_implementation: AttestationIdentityPin
    setup_manifest: AttestationIdentityPin
    circuit: AttestationIdentityPin
    public_input_schema: AttestationIdentityPin
    proving_key: AttestationIdentityPin
    verification_key: AttestationIdentityPin
    result_set_root: AttestationIdentityPin
    capability_report: AttestationIdentityPin
    repository_id: str
    use_case_id: str
    predicate_kind: AttestationPredicateKind
    backend_family: str
    backend_mode: AttestationBackendMode
    backend_version: str
    circuit_version: str
    setup_version: str
    key_epoch: str
    required_base_assurance: AssuranceLevel
    verifier_domain: str
    challenge: str
    issued_at: str
    expires_at: str
    revocation_epoch: str
    canonicalization_version: str = "strict-dag-json-v1"
    proof_schema_version: int = MCP_CONTRACT_ATTESTATION_VERSION
    envelope_schema_version: int = MCP_CONTRACT_ATTESTATION_VERSION

    def __post_init__(self) -> None:
        for name in PUBLIC_IDENTITY_FIELDS:
            object.__setattr__(self, name, _pin(getattr(self, name), name))
        for name in (
            "repository_id",
            "use_case_id",
            "backend_family",
            "backend_version",
            "circuit_version",
            "setup_version",
            "key_epoch",
            "verifier_domain",
            "challenge",
            "revocation_epoch",
            "canonicalization_version",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "predicate_kind",
            _enum(
                self.predicate_kind,
                AttestationPredicateKind,
                "predicate_kind",
            ),
        )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(
                self.backend_mode, AttestationBackendMode, "backend_mode"
            ),
        )
        object.__setattr__(
            self,
            "required_base_assurance",
            _enum(
                self.required_base_assurance,
                AssuranceLevel,
                "required_base_assurance",
            ),
        )
        object.__setattr__(
            self, "issued_at", _timestamp(self.issued_at, "issued_at")
        )
        object.__setattr__(
            self, "expires_at", _timestamp(self.expires_at, "expires_at")
        )
        if _time_value(self.expires_at) <= _time_value(self.issued_at):
            raise McpAttestationError(
                "attestation expiry must follow issue time",
                reason_code="freshness_invalid",
            )
        for name in ("proof_schema_version", "envelope_schema_version"):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name)
            )

    @property
    def statement_id(self) -> str:
        return self.content_id

    @property
    def public_input_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def identity_profile_ids(self) -> dict[str, str]:
        return {
            name: getattr(self, name).identity_profile_id
            for name in PUBLIC_IDENTITY_FIELDS
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            **{name: getattr(self, name) for name in PUBLIC_IDENTITY_FIELDS},
            "repository_id": self.repository_id,
            "use_case_id": self.use_case_id,
            "predicate_kind": self.predicate_kind,
            "backend_family": self.backend_family,
            "backend_mode": self.backend_mode,
            "backend_version": self.backend_version,
            "circuit_version": self.circuit_version,
            "setup_version": self.setup_version,
            "key_epoch": self.key_epoch,
            "required_base_assurance": self.required_base_assurance,
            "verifier_domain": self.verifier_domain,
            "challenge": self.challenge,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "revocation_epoch": self.revocation_epoch,
            "canonicalization_version": self.canonicalization_version,
            "proof_schema_version": self.proof_schema_version,
            "envelope_schema_version": self.envelope_schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationPublicInputs":
        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                *PUBLIC_IDENTITY_FIELDS,
                "repository_id",
                "use_case_id",
                "predicate_kind",
                "backend_family",
                "backend_mode",
                "backend_version",
                "circuit_version",
                "setup_version",
                "key_epoch",
                "required_base_assurance",
                "verifier_domain",
                "challenge",
                "issued_at",
                "expires_at",
                "revocation_epoch",
                "canonicalization_version",
                "proof_schema_version",
                "envelope_schema_version",
                "statement_id",
                "public_input_digest",
                "content_id",
            },
        )
        values = {name: payload.get(name) for name in PUBLIC_IDENTITY_FIELDS}
        result = cls(
            **values,
            repository_id=payload.get("repository_id", ""),
            use_case_id=payload.get("use_case_id", ""),
            predicate_kind=payload.get("predicate_kind", ""),
            backend_family=payload.get("backend_family", ""),
            backend_mode=payload.get("backend_mode", ""),
            backend_version=payload.get("backend_version", ""),
            circuit_version=payload.get("circuit_version", ""),
            setup_version=payload.get("setup_version", ""),
            key_epoch=payload.get("key_epoch", ""),
            required_base_assurance=payload.get(
                "required_base_assurance", ""
            ),
            verifier_domain=payload.get("verifier_domain", ""),
            challenge=payload.get("challenge", ""),
            issued_at=payload.get("issued_at", ""),
            expires_at=payload.get("expires_at", ""),
            revocation_epoch=payload.get("revocation_epoch", ""),
            canonicalization_version=payload.get(
                "canonicalization_version", ""
            ),
            proof_schema_version=payload.get("proof_schema_version", 0),
            envelope_schema_version=payload.get(
                "envelope_schema_version", 0
            ),
        )
        claimed = payload.get("statement_id") or payload.get("content_id")
        if claimed not in (None, "", result.statement_id):
            raise McpAttestationError(
                "public-input statement identity does not match payload",
                reason_code="forged_root",
            )
        digest = payload.get("public_input_digest")
        if digest not in (None, "", result.public_input_digest):
            raise McpAttestationError(
                "public-input digest does not match payload",
                reason_code="forged_root",
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "statement_id": self.statement_id,
            "public_input_digest": self.public_input_digest,
        }

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


def _artifact_pin(value: Any, logical_id: str) -> AttestationIdentityPin:
    # Some older canonical contracts expose Enum projections in ``to_dict``.
    # Normalize them through the shared canonical encoder before handing the
    # value to the stricter datasets DAG-JSON canonicalizer.
    normalized = json.loads(canonical_json_bytes(value))
    return AttestationIdentityPin.for_artifact(
        normalized, logical_id=logical_id
    )


def _policy_pin(policy: ProofAttestationPolicy) -> AttestationIdentityPin:
    return _artifact_pin(policy.to_dict(), policy.policy_id)


def _capability_pin(
    report: AttestationCapabilityReport,
) -> AttestationIdentityPin:
    return _artifact_pin(report.to_dict(), report.capability_id)


def build_attestation_public_inputs(
    receipt: ProofReceipt,
    cache_key: ProofCacheKey,
    *,
    policy: ProofAttestationPolicy,
    capability_report: AttestationCapabilityReport,
    challenge: str,
    issued_at: str,
    expires_at: str,
    revocation_epoch: str,
    result_set_root: IdentityBinding | AttestationIdentityPin | None = None,
) -> AttestationPublicInputs:
    """Reconstruct a closed public statement from validated trusted inputs."""

    if not isinstance(receipt, ProofReceipt):
        raise McpAttestationError(
            "receipt must be a ProofReceipt",
            reason_code="receipt_invalid",
        )
    if not isinstance(cache_key, ProofCacheKey):
        raise McpAttestationError(
            "cache_key must be a ProofCacheKey",
            reason_code="cache_binding_mismatch",
        )
    if not isinstance(policy, ProofAttestationPolicy):
        raise McpAttestationError(
            "policy must be a ProofAttestationPolicy",
            reason_code="policy_mismatch",
        )
    if not isinstance(capability_report, AttestationCapabilityReport):
        raise McpAttestationError(
            "capability_report must be an AttestationCapabilityReport",
            reason_code="capability_invalid",
        )
    reasons = TrustAwareProofCache._receipt_reasons(cache_key, receipt)
    if reasons:
        priority = (
            ProofCacheReason.WRONG_TREE.value,
            ProofCacheReason.BINDING_MISMATCH.value,
            ProofCacheReason.STALE.value,
            ProofCacheReason.CANDIDATE_ONLY.value,
            ProofCacheReason.REQUIRED_ASSURANCE.value,
        )
        reason = next(
            (candidate for candidate in priority if candidate in reasons),
            sorted(reasons)[0],
        )
        raise McpAttestationError(
            "receipt does not agree with its proof-cache key",
            reason_code=reason,
        )
    try:
        receipt.require_kernel_verified()
    except ContractValidationError as exc:
        raise McpAttestationError(
            "receipt is not independently kernel verified",
            reason_code="insufficient_base_assurance",
        ) from exc
    if receipt.authoritative_assurance.rank < policy.required_base_assurance.rank:
        raise McpAttestationError(
            "base receipt assurance does not satisfy attestation policy",
            reason_code="insufficient_base_assurance",
        )

    setup = capability_report.setup
    if policy.disposition is ZkUseCaseDisposition.APPROVED:
        if setup.backend_family not in policy.authorized_backend_families:
            raise McpAttestationError(
                "backend family is not authorized by policy",
                reason_code="backend_mismatch",
            )
    issued = _timestamp(issued_at, "issued_at")
    expires = _timestamp(expires_at, "expires_at")
    if _time_value(expires) <= _time_value(issued):
        raise McpAttestationError(
            "attestation expiry must follow issue time",
            reason_code="freshness_invalid",
        )
    if (
        (_time_value(expires) - _time_value(issued)).total_seconds()
        > policy.max_proof_age_seconds
        or _time_value(expires) > _time_value(policy.expires_at)
        or _time_value(expires)
        > _time_value(setup.verification_key_expires_at)
        or _time_value(expires) > _time_value(capability_report.expires_at)
    ):
        raise McpAttestationError(
            "attestation lifetime exceeds a policy, key, or capability pin",
            reason_code="freshness_invalid",
        )
    if _time_value(issued) < _time_value(capability_report.evaluated_at):
        raise McpAttestationError(
            "capability report was not current when statement was issued",
            reason_code="capability_drift",
        )

    if policy.result_set_root_required and result_set_root is None:
        raise McpAttestationError(
            "membership attestation requires result_set_root",
            reason_code="result_set_root_missing",
        )
    if not policy.result_set_root_required and result_set_root is not None:
        raise McpAttestationError(
            "non-membership predicate must use the policy sentinel",
            reason_code="result_set_root_unexpected",
        )
    result_pin = (
        _pin(result_set_root, "result_set_root")
        if result_set_root is not None
        else _artifact_pin(
            {
                "domain": "mcp-contract-attestation/result-set-root",
                "value": "not_applicable",
                "predicate_kind": policy.predicate_kind.value,
            },
            "result-set-root:not-applicable",
        )
    )
    receipt_pin = _artifact_pin(receipt.to_dict(), receipt.receipt_id)
    cache_pin = _artifact_pin(cache_key.to_dict(), cache_key.key_id)
    scope_pin = _artifact_pin(
        [AttestationIdentityPin.from_binding(item).to_dict() for item in cache_key.scope],
        "scope-root:" + cache_key.key_id,
    )
    return AttestationPublicInputs(
        receipt=receipt_pin,
        cache_key=cache_pin,
        property=AttestationIdentityPin.from_binding(
            cache_key.property_catalog
        ),
        obligation=AttestationIdentityPin.from_binding(cache_key.obligation),
        snapshot=AttestationIdentityPin.from_binding(cache_key.snapshot),
        scope_root=scope_pin,
        contract_proof_policy=AttestationIdentityPin.from_binding(
            cache_key.policy
        ),
        use_case_decision=policy.use_case_decision,
        attestation_policy=_policy_pin(policy),
        predicate_manifest=policy.predicate_manifest,
        backend_policy=setup.backend_policy,
        backend_implementation=setup.backend_implementation,
        setup_manifest=setup.setup_manifest,
        circuit=setup.circuit,
        public_input_schema=setup.public_input_schema,
        proving_key=setup.proving_key,
        verification_key=setup.verification_key,
        result_set_root=result_pin,
        capability_report=_capability_pin(capability_report),
        repository_id=receipt.repository_id,
        use_case_id=policy.use_case_id,
        predicate_kind=policy.predicate_kind,
        backend_family=setup.backend_family,
        backend_mode=setup.backend_mode,
        backend_version=setup.backend_version,
        circuit_version=setup.circuit_version,
        setup_version=setup.setup_version,
        key_epoch=setup.key_epoch,
        required_base_assurance=policy.required_base_assurance,
        verifier_domain=policy.verifier_domain,
        challenge=challenge,
        issued_at=issued,
        expires_at=expires,
        revocation_epoch=revocation_epoch,
    )


T = TypeVar("T")


class PrivateAttestationWitness:
    """Single-use, non-serializable mutable witness storage."""

    __slots__ = ("__buffers", "__zeroized", "__used")

    def __init__(self, values: Mapping[str, bytes | bytearray | memoryview]) -> None:
        if not isinstance(values, Mapping) or not values:
            raise McpAttestationError(
                "witness must be a non-empty mapping of bytes-like values",
                reason_code="witness_invalid",
            )
        buffers: dict[str, bytearray] = {}
        for name, value in values.items():
            key = _required_text(name, "witness field")
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise McpAttestationError(
                    "witness values must be bytes-like",
                    reason_code="witness_invalid",
                )
            buffers[key] = bytearray(value)
        self.__buffers = buffers
        self.__zeroized = False
        self.__used = False

    def __repr__(self) -> str:
        return "<PrivateAttestationWitness redacted>"

    __str__ = __repr__

    def __copy__(self) -> "PrivateAttestationWitness":
        raise WitnessDisclosureError("private witness cannot be copied")

    def __deepcopy__(self, memo: Any) -> "PrivateAttestationWitness":
        del memo
        raise WitnessDisclosureError("private witness cannot be copied")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise WitnessDisclosureError()

    def __getstate__(self) -> Any:
        raise WitnessDisclosureError()

    def to_dict(self) -> dict[str, Any]:
        raise WitnessDisclosureError()

    @property
    def zeroized(self) -> bool:
        return self.__zeroized

    def zeroize(self) -> None:
        for buffer in self.__buffers.values():
            for index in range(len(buffer)):
                buffer[index] = 0
        self.__zeroized = True

    def use(
        self,
        consumer: Callable[[Mapping[str, memoryview]], T],
    ) -> T:
        if self.__used:
            raise McpAttestationError(
                "private witness is single-use",
                reason_code="witness_reuse",
            )
        if not callable(consumer):
            raise McpAttestationError(
                "witness consumer must be callable",
                reason_code="witness_invalid",
            )
        self.__used = True
        views = MappingProxyType(
            {name: memoryview(value).toreadonly() for name, value in self.__buffers.items()}
        )
        try:
            return consumer(views)
        finally:
            for view in views.values():
                view.release()
            self.zeroize()

    @staticmethod
    def redacted_marker() -> dict[str, bool]:
        return {"private_witness_redacted": True}


def _proof_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        proof = bytes(value)
    elif isinstance(value, Mapping):
        if isinstance(value.get("proof_bytes"), (bytes, bytearray, memoryview)):
            proof = bytes(value["proof_bytes"])
        elif isinstance(value.get("proof_b64"), str):
            try:
                proof = base64.b64decode(value["proof_b64"], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise McpAttestationError(
                    "proof_b64 is malformed",
                    reason_code="malformed_proof",
                ) from exc
        elif isinstance(value.get("proof_data"), str):
            encoded = value["proof_data"]
            try:
                proof = bytes.fromhex(encoded)
            except ValueError:
                try:
                    proof = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise McpAttestationError(
                        "proof_data is malformed",
                        reason_code="malformed_proof",
                    ) from exc
        else:
            raise McpAttestationError(
                "prover omitted proof bytes",
                reason_code="malformed_proof",
            )
    else:
        raise McpAttestationError(
            "prover result has an unsupported shape",
            reason_code="malformed_proof",
        )
    if not proof or len(proof) > MAX_PROOF_BYTES:
        raise McpAttestationError(
            "proof size is outside the accepted bounds",
            reason_code="malformed_proof",
        )
    return proof


@dataclass(frozen=True, slots=True)
class ProofAttestation(CanonicalContract):
    """Public proof envelope; generation alone is never authoritative."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_ENVELOPE_SCHEMA

    public_inputs: AttestationPublicInputs
    status: AttestationStatus
    backend_mode: AttestationBackendMode
    capability_report_id: str
    proof: bytes = field(default=b"", repr=False)
    diagnostic_code: str = ""
    provider_verified: bool = False

    def __post_init__(self) -> None:
        inputs = (
            self.public_inputs
            if isinstance(self.public_inputs, AttestationPublicInputs)
            else AttestationPublicInputs.from_dict(self.public_inputs)
        )
        object.__setattr__(self, "public_inputs", inputs)
        object.__setattr__(
            self, "status", _enum(self.status, AttestationStatus, "status")
        )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(
                self.backend_mode, AttestationBackendMode, "backend_mode"
            ),
        )
        object.__setattr__(
            self,
            "capability_report_id",
            _required_text(
                self.capability_report_id, "capability_report_id"
            ),
        )
        if not isinstance(self.proof, (bytes, bytearray, memoryview)):
            raise McpAttestationError(
                "proof must be bytes-like",
                reason_code="malformed_proof",
            )
        proof = bytes(self.proof)
        object.__setattr__(self, "proof", proof)
        diagnostic = str(self.diagnostic_code or "").strip().lower()
        if len(diagnostic) > 96 or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in diagnostic
        ):
            raise McpAttestationError(
                "diagnostic_code must be a bounded machine code",
                reason_code="invalid_schema",
            )
        object.__setattr__(self, "diagnostic_code", diagnostic)
        if not isinstance(self.provider_verified, bool):
            raise McpAttestationError(
                "provider_verified must be boolean",
                reason_code="invalid_schema",
            )
        if self.status is AttestationStatus.ATTESTED:
            raise McpAttestationError(
                "a proof envelope cannot assert attested authority",
                reason_code="authority_injection",
            )
        if (
            self.backend_mode is AttestationBackendMode.SIMULATED
            and self.status is not AttestationStatus.SIMULATED
        ):
            raise McpAttestationError(
                "simulated proof cannot be promoted",
                reason_code="simulation_promotion",
            )
        if self.backend_mode is not inputs.backend_mode:
            raise McpAttestationError(
                "envelope backend mode does not match public inputs",
                reason_code="backend_mismatch",
            )
        has_proof = bool(proof)
        if has_proof != (
            self.status
            in {AttestationStatus.GENERATED, AttestationStatus.SIMULATED}
        ):
            raise McpAttestationError(
                "proof presence does not match envelope status",
                reason_code="malformed_proof",
            )
        if len(proof) > MAX_PROOF_BYTES:
            raise McpAttestationError(
                "proof exceeds maximum size",
                reason_code="malformed_proof",
            )

    @property
    def proof_digest(self) -> str:
        return (
            "sha256:" + hashlib.sha256(self.proof).hexdigest()
            if self.proof
            else ""
        )

    @property
    def attestation_id(self) -> str:
        return self.content_id

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def simulated(self) -> bool:
        return self.status is AttestationStatus.SIMULATED

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            "interface": MCP_CONTRACT_ATTESTATION_INTERFACE,
            "public_inputs": self.public_inputs,
            "statement_id": self.public_inputs.statement_id,
            "status": self.status,
            "backend_mode": self.backend_mode,
            "capability_report_id": self.capability_report_id,
            "proof_b64": base64.b64encode(self.proof).decode("ascii"),
            "proof_digest": self.proof_digest,
            "diagnostic_code": self.diagnostic_code,
            "provider_verified": self.provider_verified,
            "authoritative": False,
            "private_witness_redacted": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofAttestation":
        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "public_inputs",
                "statement_id",
                "status",
                "backend_mode",
                "capability_report_id",
                "proof_b64",
                "proof_digest",
                "diagnostic_code",
                "provider_verified",
                "authoritative",
                "private_witness_redacted",
                "attestation_id",
                "content_id",
            },
        )
        if payload.get("authoritative") not in (None, False):
            raise McpAttestationError(
                "provider authority claims are forbidden",
                reason_code="authority_injection",
            )
        encoded = payload.get("proof_b64", "")
        if not isinstance(encoded, str):
            raise McpAttestationError(
                "proof_b64 must be a string",
                reason_code="malformed_proof",
            )
        try:
            proof = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise McpAttestationError(
                "proof_b64 is malformed",
                reason_code="malformed_proof",
            ) from exc
        result = cls(
            public_inputs=payload.get("public_inputs"),
            status=payload.get("status", ""),
            backend_mode=payload.get("backend_mode", ""),
            capability_report_id=payload.get("capability_report_id", ""),
            proof=proof,
            diagnostic_code=payload.get("diagnostic_code", ""),
            provider_verified=payload.get("provider_verified", False),
        )
        derived = {
            "statement_id": result.public_inputs.statement_id,
            "proof_digest": result.proof_digest,
            "private_witness_redacted": True,
        }
        for name, expected in derived.items():
            if payload.get(name) not in (None, "", expected):
                raise McpAttestationError(
                    "proof envelope contains a forged derived field",
                    reason_code="forged_root",
                )
        claimed = payload.get("attestation_id") or payload.get("content_id")
        if claimed not in (None, "", result.attestation_id):
            raise McpAttestationError(
                "attestation identity does not match payload",
                reason_code="forged_root",
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "attestation_id": self.attestation_id}

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


@dataclass(frozen=True, slots=True)
class AttestationVerification(CanonicalContract):
    """Locally derived verification result; serialized claims have no authority."""

    SCHEMA: ClassVar[str] = MCP_CONTRACT_ATTESTATION_VERIFICATION_SCHEMA

    attestation_id: str
    statement_id: str
    status: AttestationStatus
    verifier_id: str
    diagnostic_code: str
    independent: bool
    simulated: bool

    def __post_init__(self) -> None:
        for name in (
            "attestation_id",
            "statement_id",
            "verifier_id",
            "diagnostic_code",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "status", _enum(self.status, AttestationStatus, "status")
        )
        if not isinstance(self.independent, bool) or not isinstance(
            self.simulated, bool
        ):
            raise McpAttestationError(
                "verification flags must be booleans",
                reason_code="invalid_schema",
            )
        if self.status is AttestationStatus.ATTESTED and (
            self.simulated or not self.independent
        ):
            raise McpAttestationError(
                "simulated or non-independent verification cannot be attested",
                reason_code="simulation_promotion",
            )

    @property
    def authoritative(self) -> bool:
        return (
            self.status is AttestationStatus.ATTESTED
            and self.independent
            and not self.simulated
        )

    @property
    def assurance(self) -> AssuranceLevel:
        return (
            AssuranceLevel.ATTESTED
            if self.authoritative
            else AssuranceLevel.UNVERIFIED
        )

    @property
    def verification_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MCP_CONTRACT_ATTESTATION_VERSION,
            "attestation_id": self.attestation_id,
            "statement_id": self.statement_id,
            "status": self.status,
            "verifier_id": self.verifier_id,
            "diagnostic_code": self.diagnostic_code,
            "independent": self.independent,
            "simulated": self.simulated,
            "authoritative": self.authoritative,
            "assurance": self.assurance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationVerification":
        """Parse only as an untrusted observation.

        An authoritative result must always be reproduced with
        :meth:`ZkpAttestationAdapter.verify`; it cannot be restored from flags.
        """

        _schema(payload, cls.SCHEMA)
        _closed_fields(
            payload,
            {
                "schema",
                "contract_version",
                "attestation_id",
                "statement_id",
                "status",
                "verifier_id",
                "diagnostic_code",
                "independent",
                "simulated",
                "authoritative",
                "assurance",
                "verification_id",
                "content_id",
            },
        )
        if (
            payload.get("status") == AttestationStatus.ATTESTED.value
            or payload.get("authoritative") is True
            or payload.get("assurance") == AssuranceLevel.ATTESTED.value
        ):
            raise McpAttestationError(
                "serialized verification cannot assert authority",
                reason_code="authority_injection",
            )
        return cls(
            attestation_id=payload.get("attestation_id", ""),
            statement_id=payload.get("statement_id", ""),
            status=payload.get("status", AttestationStatus.REJECTED),
            verifier_id=payload.get("verifier_id", ""),
            diagnostic_code=payload.get(
                "diagnostic_code", "serialized_observation"
            ),
            independent=False,
            simulated=payload.get("simulated", False),
        )

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "verification_id": self.verification_id}

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


class ReplayGuard:
    """Explicit verifier-owned replay state.

    Reproduction without a replay guard remains possible for cache validation;
    one-time authorization paths pass a guard and consume the exact domain,
    challenge, statement, and revocation epoch tuple.
    """

    def __init__(self) -> None:
        self._consumed: set[tuple[str, str, str, str]] = set()

    def consume(self, inputs: AttestationPublicInputs) -> bool:
        token = (
            inputs.verifier_domain,
            inputs.challenge,
            inputs.statement_id,
            inputs.revocation_epoch,
        )
        if token in self._consumed:
            return False
        self._consumed.add(token)
        return True


def _outcome(
    attestation: ProofAttestation,
    *,
    status: AttestationStatus,
    verifier_id: str,
    diagnostic_code: str,
    independent: bool = True,
) -> AttestationVerification:
    return AttestationVerification(
        attestation_id=attestation.attestation_id,
        statement_id=attestation.public_inputs.statement_id,
        status=status,
        verifier_id=verifier_id,
        diagnostic_code=diagnostic_code,
        independent=independent,
        simulated=attestation.simulated,
    )


def _matches_capability_setup(
    inputs: AttestationPublicInputs,
    report: AttestationCapabilityReport,
) -> bool:
    setup = report.setup
    pins_match = all(
        getattr(inputs, name) == getattr(setup, name)
        for name in (
            "backend_policy",
            "backend_implementation",
            "setup_manifest",
            "circuit",
            "public_input_schema",
            "proving_key",
            "verification_key",
        )
    )
    return pins_match and (
        inputs.backend_family,
        inputs.backend_mode,
        inputs.backend_version,
        inputs.circuit_version,
        inputs.setup_version,
        inputs.key_epoch,
    ) == (
        setup.backend_family,
        setup.backend_mode,
        setup.backend_version,
        setup.circuit_version,
        setup.setup_version,
        setup.key_epoch,
    )


class ZkpAttestationAdapter:
    """Lazy provider adapter with independent, fail-closed verification."""

    def __init__(
        self,
        *,
        prover: Callable[
            [bytes, Mapping[str, memoryview]], Any
        ]
        | None = None,
        verifier: Callable[[bytes, bytes], bool] | None = None,
        prover_id: str = "mcp-attestation-prover",
        verifier_id: str = "mcp-attestation-verifier",
    ) -> None:
        self._prover = prover
        self._verifier = verifier
        self.prover_id = _required_text(prover_id, "prover_id")
        self.verifier_id = _required_text(verifier_id, "verifier_id")

    @staticmethod
    def datasets_bridge_available() -> bool:
        """Probe the optional datasets bridge lazily."""

        try:
            importlib.import_module(DATASETS_ZKP_BRIDGE_MODULE)
        except (ImportError, ModuleNotFoundError):
            return False
        return True

    @staticmethod
    def not_applicable(
        public_inputs: AttestationPublicInputs,
        policy: ProofAttestationPolicy,
    ) -> ProofAttestation:
        status_by_disposition = {
            ZkUseCaseDisposition.PENDING_REVIEW: (
                AttestationStatus.PENDING_REVIEW
            ),
            ZkUseCaseDisposition.REJECTED: AttestationStatus.REJECTED,
            ZkUseCaseDisposition.NOT_APPLICABLE: (
                AttestationStatus.NOT_APPLICABLE
            ),
        }
        status = status_by_disposition.get(
            policy.disposition, AttestationStatus.NOT_APPLICABLE
        )
        return ProofAttestation(
            public_inputs=public_inputs,
            status=status,
            backend_mode=public_inputs.backend_mode,
            capability_report_id=public_inputs.capability_report.logical_id,
            diagnostic_code=status.value,
        )

    def attest(
        self,
        public_inputs: AttestationPublicInputs,
        *,
        policy: ProofAttestationPolicy,
        capability_report: AttestationCapabilityReport,
        witness: PrivateAttestationWitness | None = None,
        prover: Callable[[bytes, Mapping[str, memoryview]], Any] | None = None,
    ) -> ProofAttestation:
        """Generate an envelope after policy and capability gates pass."""

        if not isinstance(public_inputs, AttestationPublicInputs):
            raise McpAttestationError(
                "public_inputs must be AttestationPublicInputs",
                reason_code="invalid_schema",
            )
        expected_policy = _policy_pin(policy)
        if public_inputs.attestation_policy != expected_policy:
            if witness is not None:
                witness.zeroize()
            raise McpAttestationError(
                "public inputs bind a different attestation policy",
                reason_code="policy_mismatch",
            )
        if public_inputs.use_case_decision != policy.use_case_decision:
            if witness is not None:
                witness.zeroize()
            raise McpAttestationError(
                "public inputs bind a different use-case decision",
                reason_code="policy_mismatch",
            )
        if policy.disposition is not ZkUseCaseDisposition.APPROVED:
            if witness is not None:
                witness.zeroize()
            return self.not_applicable(public_inputs, policy)
        expected_capability = _capability_pin(capability_report)
        if public_inputs.capability_report != expected_capability:
            if witness is not None:
                witness.zeroize()
            return ProofAttestation(
                public_inputs=public_inputs,
                status=AttestationStatus.DEGRADED,
                backend_mode=public_inputs.backend_mode,
                capability_report_id=capability_report.capability_id,
                diagnostic_code="capability_drift",
            )
        if not _matches_capability_setup(public_inputs, capability_report):
            if witness is not None:
                witness.zeroize()
            return ProofAttestation(
                public_inputs=public_inputs,
                status=AttestationStatus.DEGRADED,
                backend_mode=public_inputs.backend_mode,
                capability_report_id=capability_report.capability_id,
                diagnostic_code="backend_mismatch",
            )
        if witness is None:
            return ProofAttestation(
                public_inputs=public_inputs,
                status=AttestationStatus.ERROR,
                backend_mode=public_inputs.backend_mode,
                capability_report_id=capability_report.capability_id,
                diagnostic_code="witness_missing",
            )
        selected = prover or self._prover
        if selected is None:
            witness.zeroize()
            return ProofAttestation(
                public_inputs=public_inputs,
                status=AttestationStatus.UNAVAILABLE,
                backend_mode=public_inputs.backend_mode,
                capability_report_id=capability_report.capability_id,
                diagnostic_code="backend_unavailable",
            )
        if (
            public_inputs.backend_mode
            is AttestationBackendMode.CRYPTOGRAPHIC
            and not capability_report.production_eligible
        ):
            witness.zeroize()
            return ProofAttestation(
                public_inputs=public_inputs,
                status=AttestationStatus.DEGRADED,
                backend_mode=public_inputs.backend_mode,
                capability_report_id=capability_report.capability_id,
                diagnostic_code="capability_not_verified",
            )

        try:
            generated = witness.use(
                lambda values: selected(
                    canonical_json_bytes(public_inputs.to_dict()), values
                )
            )
            proof = _proof_bytes(generated)
        except McpAttestationError:
            raise
        except Exception:
            # Never reflect provider output or exception text: either can carry
            # witness-derived data.
            return ProofAttestation(
                public_inputs=public_inputs,
                status=AttestationStatus.ERROR,
                backend_mode=public_inputs.backend_mode,
                capability_report_id=capability_report.capability_id,
                diagnostic_code="prover_error",
            )
        status = (
            AttestationStatus.SIMULATED
            if public_inputs.backend_mode is AttestationBackendMode.SIMULATED
            else AttestationStatus.GENERATED
        )
        provider_verified = bool(
            isinstance(generated, Mapping) and generated.get("verified") is True
        )
        return ProofAttestation(
            public_inputs=public_inputs,
            status=status,
            backend_mode=public_inputs.backend_mode,
            capability_report_id=capability_report.capability_id,
            proof=proof,
            provider_verified=provider_verified,
        )

    def ingest_datasets_result(
        self,
        public_inputs: AttestationPublicInputs,
        result: Mapping[str, Any],
        *,
        capability_report: AttestationCapabilityReport,
    ) -> ProofAttestation:
        """Map a datasets bridge result to a permanently simulated envelope."""

        if not isinstance(result, Mapping):
            raise McpAttestationError(
                "datasets result must be an object",
                reason_code="malformed_proof",
            )
        proof_record = result.get("proof")
        proof = _proof_bytes(
            proof_record if isinstance(proof_record, Mapping) else result
        )
        return ProofAttestation(
            public_inputs=public_inputs,
            status=AttestationStatus.SIMULATED,
            backend_mode=AttestationBackendMode.SIMULATED,
            capability_report_id=capability_report.capability_id,
            proof=proof,
            provider_verified=bool(result.get("verified", False)),
        )

    def verify(
        self,
        attestation: ProofAttestation,
        *,
        expected_public_inputs: AttestationPublicInputs,
        policy: ProofAttestationPolicy,
        current_capability_report: AttestationCapabilityReport,
        checked_at: str,
        verifier: Callable[[bytes, bytes], bool] | None = None,
        replay_guard: ReplayGuard | None = None,
    ) -> AttestationVerification:
        """Independently reconstruct bindings and verify the proof."""

        if not isinstance(attestation, ProofAttestation):
            raise McpAttestationError(
                "attestation must be a ProofAttestation",
                reason_code="invalid_schema",
            )
        verifier_id = self.verifier_id
        if attestation.simulated:
            return _outcome(
                attestation,
                status=AttestationStatus.SIMULATED,
                verifier_id=verifier_id,
                diagnostic_code="simulated_non_authoritative",
            )
        if attestation.status is not AttestationStatus.GENERATED:
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="proof_not_generated",
            )
        if attestation.public_inputs != expected_public_inputs:
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="public_input_mismatch",
            )
        if expected_public_inputs.attestation_policy != _policy_pin(policy):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="policy_mismatch",
            )
        current_pin = _capability_pin(current_capability_report)
        if (
            expected_public_inputs.capability_report != current_pin
            or attestation.capability_report_id
            != current_capability_report.capability_id
        ):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="capability_drift",
            )
        checked = _timestamp(checked_at, "checked_at")
        if not (
            _time_value(expected_public_inputs.issued_at)
            <= _time_value(checked)
            < _time_value(expected_public_inputs.expires_at)
            and current_capability_report.current_at(checked)
            and _time_value(checked) < _time_value(policy.expires_at)
        ):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="replay_or_expired",
            )
        setup = current_capability_report.setup
        if not _matches_capability_setup(
            expected_public_inputs, current_capability_report
        ):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="backend_or_setup_mismatch",
            )
        if (
            not current_capability_report.production_eligible
            or setup.backend_mode is not AttestationBackendMode.CRYPTOGRAPHIC
        ):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="capability_not_verified",
            )
        if not (
            MIN_CRYPTOGRAPHIC_PROOF_BYTES
            <= len(attestation.proof)
            <= MAX_PROOF_BYTES
        ):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="malformed_proof",
            )
        selected = verifier or self._verifier
        if selected is None:
            return _outcome(
                attestation,
                status=AttestationStatus.ERROR,
                verifier_id=verifier_id,
                diagnostic_code="verifier_unavailable",
            )
        try:
            verified = selected(
                attestation.proof,
                canonical_json_bytes(expected_public_inputs.to_dict()),
            )
        except Exception:
            return _outcome(
                attestation,
                status=AttestationStatus.ERROR,
                verifier_id=verifier_id,
                diagnostic_code="verifier_error",
            )
        if verified is not True:
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="proof_rejected",
            )
        if replay_guard is not None and not replay_guard.consume(
            expected_public_inputs
        ):
            return _outcome(
                attestation,
                status=AttestationStatus.REJECTED,
                verifier_id=verifier_id,
                diagnostic_code="replay_detected",
            )
        return _outcome(
            attestation,
            status=AttestationStatus.ATTESTED,
            verifier_id=verifier_id,
            diagnostic_code="verified",
        )


def public_attestation_artifact(value: Any) -> dict[str, Any]:
    """Return a witness-free public artifact or reject the value."""

    if isinstance(value, PrivateAttestationWitness):
        raise WitnessDisclosureError()
    if isinstance(value, ProofAttestation):
        artifact = value.to_public_artifact()
    elif isinstance(value, AttestationVerification):
        artifact = value.to_public_artifact()
    elif isinstance(value, AttestationPublicInputs):
        artifact = value.to_public_artifact()
    elif isinstance(value, ProofAttestationPolicy):
        artifact = value.to_public_artifact()
    elif isinstance(value, AttestationBackendSetup):
        artifact = value.to_public_artifact()
    elif isinstance(value, AttestationCapabilityReport):
        artifact = value.to_public_artifact()
    elif isinstance(value, AttestationIdentityPin):
        artifact = value.to_record()
    elif isinstance(value, Mapping):
        artifact = dict(value)
    else:
        raise McpAttestationError(
            "value has no public attestation representation",
            reason_code="invalid_schema",
        )
    _reject_witness_fields(artifact)
    return artifact


_SAFE_REDACTION_FIELDS: Final = frozenset({"private_witness_redacted"})
_WITNESS_FIELD_MARKERS: Final = (
    "witness",
    "private_key",
    "secret",
    "credential",
    "password",
    "token",
)


def _reject_witness_fields(value: Any) -> None:
    if isinstance(value, PrivateAttestationWitness):
        raise WitnessDisclosureError()
    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name).strip().casefold().replace("-", "_")
            if name not in _SAFE_REDACTION_FIELDS and any(
                marker in name for marker in _WITNESS_FIELD_MARKERS
            ):
                raise WitnessDisclosureError(
                    "public artifact contains a private field"
                )
            _reject_witness_fields(item)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _reject_witness_fields(item)


# Concise compatibility spellings for callers and the task's declared
# interfaces.
McpContractAttestation = ProofAttestation
McpContractAttestationPolicy = ProofAttestationPolicy
McpContractAttestationInputs = AttestationPublicInputs
McpContractAttestationCapability = AttestationCapabilityReport
ProofAttestationResult = AttestationVerification
ZKPAttestationAdapter = ZkpAttestationAdapter
ReceiptAttestationWitness = PrivateAttestationWitness


__all__ = [
    "AttestationBackendMode",
    "AttestationBackendSetup",
    "AttestationCapabilityReport",
    "AttestationIdentityPin",
    "AttestationPredicateKind",
    "AttestationPublicInputs",
    "AttestationStatus",
    "AttestationVerification",
    "CapabilityFixture",
    "DATASETS_ZKP_BRIDGE_MODULE",
    "MAX_PROOF_BYTES",
    "MCP_CONTRACT_ATTESTATION_CAPABILITY_SCHEMA",
    "MCP_CONTRACT_ATTESTATION_ENVELOPE_SCHEMA",
    "MCP_CONTRACT_ATTESTATION_INTERFACE",
    "MCP_CONTRACT_ATTESTATION_POLICY_SCHEMA",
    "MCP_CONTRACT_ATTESTATION_PUBLIC_INPUTS_SCHEMA",
    "MCP_CONTRACT_ATTESTATION_SETUP_SCHEMA",
    "MCP_CONTRACT_ATTESTATION_VERIFICATION_SCHEMA",
    "MCP_CONTRACT_ATTESTATION_VERSION",
    "McpAttestationError",
    "McpContractAttestation",
    "McpContractAttestationCapability",
    "McpContractAttestationInputs",
    "McpContractAttestationPolicy",
    "PrivateAttestationWitness",
    "ProofAttestation",
    "ProofAttestationPolicy",
    "ProofAttestationResult",
    "REQUIRED_CAPABILITY_FIXTURES",
    "ReceiptAttestationWitness",
    "ReplayGuard",
    "WitnessDisclosureError",
    "ZKPAttestationAdapter",
    "ZkpAttestationAdapter",
    "build_attestation_public_inputs",
    "public_attestation_artifact",
]
