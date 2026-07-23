"""Receipt-bound zero-knowledge attestation contracts.

This module defines the supervisor side of the ZKP trust boundary.  It does
not claim that a ZKP proves arbitrary Python correctness.  A statement may be
prepared only from an existing, current, independently kernel-verified
``ProofReceipt``.  The ZKP then attests that immutable receipt and its public
provenance identities.

Private witness material deliberately uses a separate, non-serializable
object.  It is not a field of any canonical contract, public artifact, cache
key, context capsule, log representation, or proof evidence record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Dict, TypeVar

from .formal_verification_capabilities import CapabilityHealth
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    _canonical_value,
    _enum,
    _schema,
    _text,
    canonical_json_bytes,
    content_identity,
)


PROOF_ATTESTATION_CONTRACT_VERSION = 1
PROOF_ATTESTATION_STATEMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-statement@1"
)
PROOF_ATTESTATION_ENVELOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-envelope@1"
)
PROOF_ATTESTATION_VERIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-verification@1"
)
PROOF_ATTESTATION_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-record@1"
)
PERSISTED_ATTESTATION_SCHEMA = PROOF_ATTESTATION_RECORD_SCHEMA
ATTESTATION_BACKEND_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/attestation-backend-policy@1"
)
ATTESTATION_BACKEND_TEST_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/attestation-backend-test-result@1"
)
ATTESTATION_BACKEND_HEALTH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/attestation-backend-health@1"
)

# Compatibility names used by callers which emphasize that this is a ZKP.
ZKP_RECEIPT_ATTESTATION_STATEMENT_SCHEMA = PROOF_ATTESTATION_STATEMENT_SCHEMA
ZKP_RECEIPT_ATTESTATION_ENVELOPE_SCHEMA = PROOF_ATTESTATION_ENVELOPE_SCHEMA
ZKP_RECEIPT_ATTESTATION_VERIFICATION_SCHEMA = PROOF_ATTESTATION_VERIFICATION_SCHEMA
RECEIPT_ATTESTATION_STATEMENT_SCHEMA = PROOF_ATTESTATION_STATEMENT_SCHEMA
RECEIPT_ATTESTATION_ENVELOPE_SCHEMA = PROOF_ATTESTATION_ENVELOPE_SCHEMA
ATTESTATION_VERIFICATION_SCHEMA = PROOF_ATTESTATION_VERIFICATION_SCHEMA
ZKP_RECEIPT_ATTESTATION_RECORD_SCHEMA = PROOF_ATTESTATION_RECORD_SCHEMA
RECEIPT_ATTESTATION_RECORD_SCHEMA = PROOF_ATTESTATION_RECORD_SCHEMA


class AttestationValidationError(ContractValidationError):
    """Raised when receipt-attestation data violates the trust contract."""


class WitnessDisclosureError(AttestationValidationError):
    """Raised when private witness material reaches a serialization boundary."""


class CryptographicBackendFailure(AttestationValidationError):
    """Raised when a production backend fails before a valid result exists."""


class AttestationBackendMode(str, Enum):
    """Trust class of the proof backend, independent of its product name."""

    CRYPTOGRAPHIC = "cryptographic"
    PRODUCTION = "cryptographic"  # compatibility spelling
    SIMULATED = "simulated"


ZKPBackendMode = AttestationBackendMode


class AttestationTrust(str, Enum):
    """Authority derived from verification, never asserted by a provider."""

    NON_AUTHORITATIVE = "non_authoritative"
    AUTHORITATIVE = "authoritative"


class AttestationGate(str, Enum):
    """Supervisor gates at which an attestation might be consumed."""

    SERIALIZATION = "serialization"
    TEST = "test"
    PRODUCTION = "production"
    COMPLETION = "completion"


class AttestationVerificationVerdict(str, Enum):
    """Result returned by the independent attestation verifier."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    ERROR = "error"


AttestationMode = AttestationBackendMode
AttestationBackendHealth = CapabilityHealth


class BackendTestCase(str, Enum):
    """Mandatory adversarial fixtures for a production cryptographic backend."""

    GOLDEN = "golden"
    NEGATIVE = "negative"
    STALE_KEY = "stale_key"
    MALFORMED_PROOF = "malformed_proof"
    WITNESS_NO_LEAK = "witness_no_leak"


class BackendTestVerdict(str, Enum):
    """Fail-closed outcome for one backend fixture."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    NOT_RUN = "not_run"


REQUIRED_BACKEND_TEST_CASES = tuple(BackendTestCase)


def _timestamp(value: Any, *, field_name: str, required: bool = True) -> str:
    text = _text(value, field_name=field_name, required=required)
    if not text:
        return ""
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AttestationValidationError(
            "%s must be an RFC3339 timestamp" % field_name
        ) from exc
    if parsed.tzinfo is None:
        raise AttestationValidationError(
            "%s must include a timezone" % field_name
        )
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(
        value[:-1] + "+00:00" if value.endswith("Z") else value
    )


@dataclass(frozen=True)
class AttestationBackendPolicy(CanonicalContract):
    """Reviewed, content-addressed pin set for one backend deployment.

    Product names are deliberately insufficient.  The backend implementation,
    circuit, public-input schema, and verification key all have independently
    pinned identifiers and versions.  The policy identity commits to the whole
    set and is itself included in managed receipt-attestation statements.
    """

    SCHEMA: ClassVar[str] = ATTESTATION_BACKEND_POLICY_SCHEMA

    backend_id: str
    backend_version: str
    circuit_id: str
    circuit_version: str
    public_input_schema_id: str
    public_input_schema_version: str
    verification_key_id: str
    verification_key_version: str
    backend_mode: AttestationBackendMode = AttestationBackendMode.CRYPTOGRAPHIC
    verification_key_expires_at: str = ""

    def __post_init__(self) -> None:
        for name in (
            "backend_id",
            "backend_version",
            "circuit_id",
            "circuit_version",
            "public_input_schema_id",
            "public_input_schema_version",
            "verification_key_id",
            "verification_key_version",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(
                self.backend_mode,
                AttestationBackendMode,
                field_name="backend_mode",
            ),
        )
        object.__setattr__(
            self,
            "verification_key_expires_at",
            _timestamp(
                self.verification_key_expires_at,
                field_name="verification_key_expires_at",
                required=False,
            ),
        )
        if (
            self.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
            and _backend_id_is_explicitly_simulated(self.backend_id)
        ):
            raise AttestationValidationError(
                "a simulated backend identity cannot be pinned as cryptographic"
            )

    @property
    def policy_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return self.backend_mode is AttestationBackendMode.SIMULATED

    def key_is_current_at(self, timestamp: str) -> bool:
        checked = _timestamp(timestamp, field_name="timestamp")
        return not self.verification_key_expires_at or (
            _timestamp_value(checked)
            < _timestamp_value(self.verification_key_expires_at)
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "circuit_id": self.circuit_id,
            "circuit_version": self.circuit_version,
            "public_input_schema_id": self.public_input_schema_id,
            "public_input_schema_version": self.public_input_schema_version,
            "verification_key_id": self.verification_key_id,
            "verification_key_version": self.verification_key_version,
            "backend_mode": self.backend_mode,
            "verification_key_expires_at": self.verification_key_expires_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationBackendPolicy":
        _schema(payload, cls.SCHEMA)
        result = cls(
            backend_id=payload.get("backend_id", ""),
            backend_version=payload.get("backend_version", ""),
            circuit_id=payload.get("circuit_id", ""),
            circuit_version=payload.get("circuit_version", ""),
            public_input_schema_id=payload.get("public_input_schema_id", ""),
            public_input_schema_version=payload.get(
                "public_input_schema_version", ""
            ),
            verification_key_id=payload.get("verification_key_id", ""),
            verification_key_version=payload.get(
                "verification_key_version", ""
            ),
            backend_mode=payload.get(
                "backend_mode", AttestationBackendMode.CRYPTOGRAPHIC
            ),
            verification_key_expires_at=payload.get(
                "verification_key_expires_at", ""
            ),
        )
        claimed = payload.get("policy_id") or payload.get("content_id")
        if claimed and claimed != result.policy_id:
            raise AttestationValidationError(
                "attestation backend policy identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "policy_id": self.policy_id}


CryptographicBackendPolicy = AttestationBackendPolicy
BackendPolicy = AttestationBackendPolicy


@dataclass(frozen=True)
class BackendTestResult(CanonicalContract):
    """Public, secret-free evidence for one mandatory backend fixture."""

    SCHEMA: ClassVar[str] = ATTESTATION_BACKEND_TEST_RESULT_SCHEMA

    case: BackendTestCase
    verdict: BackendTestVerdict
    backend_policy_id: str
    observed_at: str
    diagnostic_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case",
            _enum(self.case, BackendTestCase, field_name="case"),
        )
        object.__setattr__(
            self,
            "verdict",
            _enum(self.verdict, BackendTestVerdict, field_name="verdict"),
        )
        object.__setattr__(
            self,
            "backend_policy_id",
            _text(
                self.backend_policy_id,
                field_name="backend_policy_id",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(
            self,
            "diagnostic_code",
            _text(self.diagnostic_code, field_name="diagnostic_code"),
        )

    @property
    def passed(self) -> bool:
        return self.verdict is BackendTestVerdict.PASSED

    @property
    def result_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "case": self.case,
            "verdict": self.verdict,
            "backend_policy_id": self.backend_policy_id,
            "observed_at": self.observed_at,
            "diagnostic_code": self.diagnostic_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendTestResult":
        _schema(payload, cls.SCHEMA)
        result = cls(
            case=payload.get("case", ""),
            verdict=payload.get("verdict", BackendTestVerdict.NOT_RUN),
            backend_policy_id=payload.get("backend_policy_id", ""),
            observed_at=payload.get("observed_at", ""),
            diagnostic_code=payload.get("diagnostic_code", ""),
        )
        claimed = payload.get("result_id") or payload.get("content_id")
        if claimed and claimed != result.result_id:
            raise AttestationValidationError(
                "backend test-result identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "result_id": self.result_id}


BackendSelfTestResult = BackendTestResult
BackendConformanceEvidence = BackendTestResult


def _backend_policy(value: Any) -> AttestationBackendPolicy:
    if isinstance(value, AttestationBackendPolicy):
        return value
    if isinstance(value, Mapping):
        return AttestationBackendPolicy.from_dict(value)
    raise AttestationValidationError(
        "backend policy must be an AttestationBackendPolicy or mapping"
    )


def _backend_test_result(value: Any) -> BackendTestResult:
    if isinstance(value, BackendTestResult):
        return value
    if isinstance(value, Mapping):
        return BackendTestResult.from_dict(value)
    raise AttestationValidationError(
        "backend test result must be a BackendTestResult or mapping"
    )


@dataclass(frozen=True)
class BackendHealthReport(CanonicalContract):
    """Derived health and production eligibility for a pinned backend."""

    SCHEMA: ClassVar[str] = ATTESTATION_BACKEND_HEALTH_SCHEMA

    policy: AttestationBackendPolicy
    configured: bool
    available: bool
    test_results: tuple[BackendTestResult, ...]
    evaluated_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _backend_policy(self.policy))
        object.__setattr__(
            self, "configured", _bool(self.configured, field_name="configured")
        )
        object.__setattr__(
            self, "available", _bool(self.available, field_name="available")
        )
        if self.available and not self.configured:
            raise AttestationValidationError(
                "an available backend must also be configured"
            )
        results = tuple(_backend_test_result(item) for item in self.test_results)
        cases = tuple(result.case for result in results)
        if len(cases) != len(set(cases)):
            raise AttestationValidationError(
                "backend health cannot contain duplicate test cases"
            )
        if any(
            result.backend_policy_id != self.policy.policy_id
            for result in results
        ):
            raise AttestationValidationError(
                "backend test evidence is bound to a different policy"
            )
        object.__setattr__(
            self,
            "test_results",
            tuple(
                sorted(
                    results,
                    key=lambda result: tuple(BackendTestCase).index(result.case),
                )
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            _timestamp(self.evaluated_at, field_name="evaluated_at"),
        )

    @property
    def health_id(self) -> str:
        return self.content_id

    @property
    def results_by_case(self) -> Mapping[BackendTestCase, BackendTestResult]:
        return {result.case: result for result in self.test_results}

    @property
    def missing_cases(self) -> tuple[BackendTestCase, ...]:
        present = set(self.results_by_case)
        return tuple(
            case for case in REQUIRED_BACKEND_TEST_CASES if case not in present
        )

    @property
    def evidence_timestamps_valid(self) -> bool:
        """Return whether no fixture result claims to come from the future."""

        evaluated = _timestamp_value(self.evaluated_at)
        return all(
            _timestamp_value(result.observed_at) <= evaluated
            for result in self.test_results
        )

    @property
    def status(self) -> CapabilityHealth:
        if self.policy.simulated:
            return CapabilityHealth.SIMULATED
        if not self.configured:
            return CapabilityHealth.UNAVAILABLE
        if any(not result.passed for result in self.test_results):
            return CapabilityHealth.DEGRADED
        if not self.available:
            return CapabilityHealth.CONFIGURED
        if self.missing_cases:
            return CapabilityHealth.AVAILABLE
        if not self.evidence_timestamps_valid:
            return CapabilityHealth.DEGRADED
        if not self.policy.key_is_current_at(self.evaluated_at):
            return CapabilityHealth.DEGRADED
        return CapabilityHealth.VERIFIED

    @property
    def production_eligible(self) -> bool:
        return self.status is CapabilityHealth.VERIFIED

    @property
    def reason(self) -> str:
        if self.status is CapabilityHealth.SIMULATED:
            return "simulated backends are non-authoritative"
        if self.status is CapabilityHealth.UNAVAILABLE:
            return "backend is not configured"
        if self.status is CapabilityHealth.CONFIGURED:
            return "backend policy and artifacts are configured but backend is unavailable"
        failed = tuple(
            result.case.value for result in self.test_results if not result.passed
        )
        if failed:
            return "backend failed required cases: " + ", ".join(failed)
        if self.missing_cases:
            return "backend has not run required cases: " + ", ".join(
                case.value for case in self.missing_cases
            )
        if not self.evidence_timestamps_valid:
            return "backend test evidence is newer than the health evaluation"
        if not self.policy.key_is_current_at(self.evaluated_at):
            return "verification key is stale at health evaluation time"
        return "backend passed every production cryptographic and no-leak case"

    def require_production_eligible(self) -> None:
        if not self.production_eligible:
            raise CryptographicBackendFailure(
                "cryptographic backend is not production eligible: %s" % self.reason
            )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "policy": self.policy,
            "backend_policy_id": self.policy.policy_id,
            "configured": self.configured,
            "available": self.available,
            "test_results": self.test_results,
            "evaluated_at": self.evaluated_at,
            "status": self.status,
            "production_eligible": self.production_eligible,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendHealthReport":
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy=_backend_policy(payload.get("policy") or {}),
            configured=payload.get("configured", False),
            available=payload.get("available", False),
            test_results=tuple(payload.get("test_results") or ()),
            evaluated_at=payload.get("evaluated_at", ""),
        )
        derived = {
            "backend_policy_id": result.policy.policy_id,
            "status": result.status.value,
            "production_eligible": result.production_eligible,
            "reason": result.reason,
        }
        for name, expected in derived.items():
            if payload.get(name) not in (None, "", expected):
                raise AttestationValidationError(
                    "backend health %s does not match derived value" % name
                )
        claimed = payload.get("health_id") or payload.get("content_id")
        if claimed and claimed != result.health_id:
            raise AttestationValidationError(
                "backend health identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "health_id": self.health_id}


AttestationBackendHealthReport = BackendHealthReport
CryptographicBackendHealth = BackendHealthReport


def _backend_health(value: Any) -> BackendHealthReport:
    if isinstance(value, BackendHealthReport):
        return value
    if isinstance(value, Mapping):
        return BackendHealthReport.from_dict(value)
    raise AttestationValidationError(
        "backend health must be a BackendHealthReport or mapping"
    )


def _bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AttestationValidationError("%s must be a boolean" % field_name)
    return value


def _statement(value: Any) -> "ReceiptAttestationStatement":
    if isinstance(value, ReceiptAttestationStatement):
        return value
    if isinstance(value, Mapping):
        return ReceiptAttestationStatement.from_dict(value)
    raise AttestationValidationError(
        "statement must be a ReceiptAttestationStatement or mapping"
    )


def _backend_id_is_explicitly_simulated(backend_id: str) -> bool:
    """Recognize backend identities which can never be cryptographic."""

    normalized = (
        backend_id.strip()
        .lower()
        .replace("/", ":")
        .replace("@", ":")
        .replace("_", "-")
    )
    tokens = normalized.split(":")
    return any(
        token in {"sim", "simulated", "mock", "fake", "demo", "educational"}
        or token.startswith("simulated-")
        for token in tokens
    )


@dataclass(frozen=True)
class ReceiptAttestationStatement(CanonicalContract):
    """The complete public statement proven by a receipt attestation.

    The eight identity fields are intentionally explicit rather than hidden in
    an opaque digest.  ``statement_id`` and ``public_inputs_digest`` commit to
    their canonical encoding for circuit adapters that require a single field.
    No witness values or witness field names are accepted by this contract.
    """

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_STATEMENT_SCHEMA

    repository_tree_id: str
    obligation_id: str
    policy_id: str
    kernel_id: str
    receipt_id: str
    circuit_id: str
    backend_id: str
    verification_key_id: str
    backend_policy_id: str = ""
    backend_version: str = ""
    circuit_version: str = ""
    public_input_schema_id: str = ""
    public_input_schema_version: str = ""
    verification_key_version: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_tree_id",
            "obligation_id",
            "policy_id",
            "kernel_id",
            "receipt_id",
            "circuit_id",
            "backend_id",
            "verification_key_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        binding_fields = (
            "backend_policy_id",
            "backend_version",
            "circuit_version",
            "public_input_schema_id",
            "public_input_schema_version",
            "verification_key_version",
        )
        for name in binding_fields:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )
        populated = tuple(bool(getattr(self, name)) for name in binding_fields)
        if any(populated) and not all(populated):
            raise AttestationValidationError(
                "managed attestation statements must pin backend policy, backend, "
                "circuit, public-input schema, and verification-key versions"
            )

    @property
    def backend_managed(self) -> bool:
        """Whether this statement is bound to a version-pinned backend policy."""

        return bool(self.backend_policy_id)

    @property
    def tree_id(self) -> str:
        return self.repository_tree_id

    @property
    def statement_id(self) -> str:
        return self.content_id

    @property
    def public_input_digest(self) -> str:
        """Content identity of the circuit-facing public input object."""

        return content_identity(self.public_inputs)

    @property
    def public_inputs_digest(self) -> str:
        """Compatibility plural spelling for :attr:`public_input_digest`."""

        return self.public_input_digest

    @property
    def public_inputs(self) -> Dict[str, str]:
        """Return exactly the public identities supplied to the circuit."""

        result = {
            "backend_id": self.backend_id,
            "circuit_id": self.circuit_id,
            "kernel_id": self.kernel_id,
            "obligation_id": self.obligation_id,
            "policy_id": self.policy_id,
            "receipt_id": self.receipt_id,
            "repository_tree_id": self.repository_tree_id,
            "verification_key_id": self.verification_key_id,
        }
        if self.backend_managed:
            result.update(
                {
                    "backend_policy_id": self.backend_policy_id,
                    "backend_version": self.backend_version,
                    "circuit_version": self.circuit_version,
                    "public_input_schema_id": self.public_input_schema_id,
                    "public_input_schema_version": self.public_input_schema_version,
                    "verification_key_version": self.verification_key_version,
                }
            )
        return result

    def matches_backend_policy(self, policy: AttestationBackendPolicy) -> bool:
        """Return whether every managed public binding matches ``policy``."""

        checked = _backend_policy(policy)
        return self.backend_managed and {
            "backend_policy_id": self.backend_policy_id,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "circuit_id": self.circuit_id,
            "circuit_version": self.circuit_version,
            "public_input_schema_id": self.public_input_schema_id,
            "public_input_schema_version": self.public_input_schema_version,
            "verification_key_id": self.verification_key_id,
            "verification_key_version": self.verification_key_version,
        } == {
            "backend_policy_id": checked.policy_id,
            "backend_id": checked.backend_id,
            "backend_version": checked.backend_version,
            "circuit_id": checked.circuit_id,
            "circuit_version": checked.circuit_version,
            "public_input_schema_id": checked.public_input_schema_id,
            "public_input_schema_version": checked.public_input_schema_version,
            "verification_key_id": checked.verification_key_id,
            "verification_key_version": checked.verification_key_version,
        }

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            **self.public_inputs,
        }

    @classmethod
    def from_receipt(
        cls,
        receipt: ProofReceipt,
        *,
        circuit_id: str,
        backend_id: str,
        verification_key_id: str,
        backend_policy: AttestationBackendPolicy | None = None,
    ) -> "ReceiptAttestationStatement":
        """Build a public statement after enforcing receipt eligibility."""

        if not isinstance(receipt, ProofReceipt):
            raise AttestationValidationError(
                "attestation requires an existing ProofReceipt object"
            )
        receipt.require_kernel_verified()
        policy = (
            None if backend_policy is None else _backend_policy(backend_policy)
        )
        if policy is not None:
            supplied = (circuit_id, backend_id, verification_key_id)
            expected = (
                policy.circuit_id,
                policy.backend_id,
                policy.verification_key_id,
            )
            if supplied != expected:
                raise AttestationValidationError(
                    "statement backend, circuit, and verification key must match "
                    "the pinned backend policy"
                )
        return cls(
            repository_tree_id=receipt.repository_tree_id,
            obligation_id=receipt.obligation_id,
            policy_id=receipt.policy_id,
            kernel_id=receipt.kernel_id,
            receipt_id=receipt.receipt_id,
            circuit_id=circuit_id,
            backend_id=backend_id,
            verification_key_id=verification_key_id,
            backend_policy_id=policy.policy_id if policy is not None else "",
            backend_version=policy.backend_version if policy is not None else "",
            circuit_version=policy.circuit_version if policy is not None else "",
            public_input_schema_id=(
                policy.public_input_schema_id if policy is not None else ""
            ),
            public_input_schema_version=(
                policy.public_input_schema_version if policy is not None else ""
            ),
            verification_key_version=(
                policy.verification_key_version if policy is not None else ""
            ),
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ReceiptAttestationStatement":
        _schema(payload, cls.SCHEMA)
        result = cls(
            repository_tree_id=payload.get(
                "repository_tree_id", payload.get("tree_id", "")
            ),
            obligation_id=payload.get("obligation_id", ""),
            policy_id=payload.get("policy_id", ""),
            kernel_id=payload.get("kernel_id", ""),
            receipt_id=payload.get("receipt_id", ""),
            circuit_id=payload.get("circuit_id", ""),
            backend_id=payload.get("backend_id", ""),
            verification_key_id=payload.get("verification_key_id", ""),
            backend_policy_id=payload.get("backend_policy_id", ""),
            backend_version=payload.get("backend_version", ""),
            circuit_version=payload.get("circuit_version", ""),
            public_input_schema_id=payload.get("public_input_schema_id", ""),
            public_input_schema_version=payload.get(
                "public_input_schema_version", ""
            ),
            verification_key_version=payload.get(
                "verification_key_version", ""
            ),
        )
        claimed = payload.get("statement_id") or payload.get("content_id")
        if claimed and claimed != result.statement_id:
            raise AttestationValidationError(
                "attestation statement identity does not match payload"
            )
        claimed_digest = payload.get("public_input_digest") or payload.get(
            "public_inputs_digest"
        )
        if claimed_digest and claimed_digest != result.public_input_digest:
            raise AttestationValidationError(
                "attestation public-input digest does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        """Return the public statement plus its derived identities."""

        return {
            **self.to_dict(),
            "statement_id": self.statement_id,
            "public_input_digest": self.public_input_digest,
        }

    to_context_capsule = to_public_artifact
    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


ZKPReceiptAttestationStatement = ReceiptAttestationStatement
ProofAttestationStatement = ReceiptAttestationStatement


T = TypeVar("T")


class PrivateAttestationWitness:
    """Opaque, redacted, and non-serializable private proving inputs.

    A backend receives the values only inside :meth:`use`.  The wrapper has no
    mapping protocol, iterator, public value property, dataclass fields, JSON
    method, or pickle representation.  This keeps generic logging, context,
    artifact, and cache serializers from accidentally traversing its secrets.
    """

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise AttestationValidationError("witness values must be a mapping")
        normalized: Dict[str, Any] = {}
        for raw_name, value in values.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise AttestationValidationError(
                    "witness field names must be non-empty strings"
                )
            normalized[raw_name] = value
        if not normalized:
            raise AttestationValidationError("witness values must not be empty")
        # A private copy prevents later mutation through the caller's mapping.
        self.__values = dict(normalized)

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
        raise WitnessDisclosureError(
            "private witness cannot be serialized or cached"
        )

    def __getstate__(self) -> Any:
        raise WitnessDisclosureError(
            "private witness cannot be serialized or cached"
        )

    def to_dict(self) -> Dict[str, Any]:
        raise WitnessDisclosureError(
            "private witness has no public dictionary representation"
        )

    def use(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        """Invoke a local prover callback with a read-only witness view."""

        if not callable(consumer):
            raise AttestationValidationError("witness consumer must be callable")
        return consumer(MappingProxyType(self.__values))

    def redacted(self) -> Dict[str, bool]:
        """Return a constant safe marker which reveals no field names."""

        return {"private_witness_redacted": True}


ReceiptAttestationWitness = PrivateAttestationWitness
ZKPWitness = PrivateAttestationWitness


@dataclass(frozen=True, repr=False)
class ReceiptAttestationRequest:
    """Ephemeral proving request; only its public statement is serializable."""

    statement: ReceiptAttestationStatement
    _witness: PrivateAttestationWitness = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement", _statement(self.statement))
        if not isinstance(self._witness, PrivateAttestationWitness):
            raise AttestationValidationError(
                "_witness must be a PrivateAttestationWitness"
            )

    def __repr__(self) -> str:
        return (
            "ReceiptAttestationRequest("
            "statement_id=%r, witness=<redacted>)" % self.statement.statement_id
        )

    __str__ = __repr__

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise WitnessDisclosureError(
            "attestation proving requests cannot be serialized or cached"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return only public proving inputs, never the witness."""

        return {
            "statement": self.statement.to_public_artifact(),
            "private_witness_redacted": True,
        }

    to_public_artifact = to_dict
    to_context_capsule = to_dict
    to_log_record = to_dict

    def to_cache_record(self) -> Dict[str, Any]:
        raise WitnessDisclosureError(
            "attestation proving requests containing a witness cannot be cached"
        )

    def use_witness(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        """Pass the witness to an in-process proving callback."""

        return self._witness.use(consumer)


ZKPReceiptAttestationRequest = ReceiptAttestationRequest
ProofAttestationRequest = ReceiptAttestationRequest


@dataclass(frozen=True)
class ReceiptAttestationEnvelope(CanonicalContract):
    """Public proof envelope produced for one immutable statement.

    Generation is not verification: even a cryptographic envelope is
    non-authoritative until an independent verifier records a successful
    :class:`AttestationVerification`.
    """

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_ENVELOPE_SCHEMA

    statement: ReceiptAttestationStatement
    backend_mode: AttestationBackendMode
    proof_artifact_id: str
    proof_digest: str
    prover_id: str = ""
    backend_health: BackendHealthReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement", _statement(self.statement))
        object.__setattr__(
            self,
            "backend_mode",
            _enum(
                self.backend_mode,
                AttestationBackendMode,
                field_name="backend_mode",
            ),
        )
        for name in ("proof_artifact_id", "proof_digest"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self, "prover_id", _text(self.prover_id, field_name="prover_id")
        )
        if self.backend_health is not None:
            object.__setattr__(
                self,
                "backend_health",
                _backend_health(self.backend_health),
            )
        if (
            self.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
            and _backend_id_is_explicitly_simulated(self.statement.backend_id)
        ):
            raise AttestationValidationError(
                "a simulated backend identity cannot use cryptographic mode"
            )
        if self.statement.backend_managed:
            if self.backend_health is None:
                raise AttestationValidationError(
                    "a managed attestation envelope requires backend health evidence"
                )
            if not self.statement.matches_backend_policy(self.backend_health.policy):
                raise AttestationValidationError(
                    "backend health policy does not match attestation statement"
                )
            if self.backend_mode is not self.backend_health.policy.backend_mode:
                raise AttestationValidationError(
                    "envelope mode does not match backend policy"
                )
            if self.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC:
                self.backend_health.require_production_eligible()

    @property
    def envelope_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return self.backend_mode is AttestationBackendMode.SIMULATED

    @property
    def is_simulated(self) -> bool:
        return self.simulated

    @property
    def authoritative(self) -> bool:
        # Proof generation and provider claims never cross the verifier boundary.
        return False

    @property
    def production_eligible(self) -> bool:
        """Whether backend evidence permits independent production verification."""

        if self.backend_mode is not AttestationBackendMode.CRYPTOGRAPHIC:
            return False
        if not self.statement.backend_managed:
            # Version-1 legacy statements remain readable.  New managed
            # statements always take the evidence-backed path above.
            return True
        return bool(
            self.backend_health is not None
            and self.backend_health.production_eligible
            and self.statement.matches_backend_policy(self.backend_health.policy)
        )

    @property
    def backend_health_id(self) -> str:
        return self.backend_health.health_id if self.backend_health is not None else ""

    @property
    def is_authoritative(self) -> bool:
        return self.authoritative

    @property
    def trust(self) -> AttestationTrust:
        return AttestationTrust.NON_AUTHORITATIVE

    @property
    def non_authoritative_reason(self) -> str:
        if self.simulated:
            return "simulated_zkp_is_non_authoritative"
        return "independent_verification_required"

    def _payload(self) -> Dict[str, Any]:
        payload = {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "statement": self.statement,
            "statement_id": self.statement.statement_id,
            "public_input_digest": self.statement.public_input_digest,
            "backend_mode": self.backend_mode,
            "proof_artifact_id": self.proof_artifact_id,
            "proof_digest": self.proof_digest,
            "prover_id": self.prover_id,
            "simulated": self.simulated,
            "authoritative": False,
            "trust": self.trust,
            "non_authoritative_reason": self.non_authoritative_reason,
        }
        if self.backend_health is not None:
            payload["backend_health"] = self.backend_health
            payload["backend_health_id"] = self.backend_health.health_id
            payload["production_eligible"] = self.production_eligible
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ReceiptAttestationEnvelope":
        _schema(payload, cls.SCHEMA)
        result = cls(
            statement=_statement(payload.get("statement") or {}),
            backend_mode=payload.get(
                "backend_mode", AttestationBackendMode.SIMULATED
            ),
            proof_artifact_id=payload.get("proof_artifact_id", ""),
            proof_digest=payload.get("proof_digest", ""),
            prover_id=payload.get("prover_id", ""),
            backend_health=(
                _backend_health(payload["backend_health"])
                if payload.get("backend_health") is not None
                else None
            ),
        )
        if payload.get("authoritative") not in (None, False):
            raise AttestationValidationError(
                "a generated attestation envelope cannot assert authority"
            )
        claimed_trust = payload.get("trust")
        if claimed_trust not in (None, "", AttestationTrust.NON_AUTHORITATIVE.value):
            raise AttestationValidationError(
                "a generated attestation envelope must be non-authoritative"
            )
        if payload.get("simulated") not in (None, result.simulated):
            raise AttestationValidationError(
                "attestation simulation label does not match backend mode"
            )
        if payload.get("statement_id") not in (None, "", result.statement.statement_id):
            raise AttestationValidationError(
                "attestation statement identity does not match envelope"
            )
        if payload.get("public_input_digest") not in (
            None,
            "",
            result.statement.public_input_digest,
        ):
            raise AttestationValidationError(
                "attestation public-input digest does not match envelope"
            )
        if payload.get("backend_health_id") not in (
            None,
            "",
            result.backend_health_id,
        ):
            raise AttestationValidationError(
                "backend health identity does not match envelope"
            )
        if payload.get("production_eligible") not in (
            None,
            result.production_eligible,
        ):
            raise AttestationValidationError(
                "backend production eligibility does not match envelope"
            )
        claimed_id = payload.get("envelope_id") or payload.get("content_id")
        if claimed_id and claimed_id != result.envelope_id:
            raise AttestationValidationError(
                "attestation envelope identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "envelope_id": self.envelope_id}

    to_context_capsule = to_public_artifact
    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


ZKPReceiptAttestation = ReceiptAttestationEnvelope
ProofAttestationEnvelope = ReceiptAttestationEnvelope
AttestationEnvelope = ReceiptAttestationEnvelope


def _envelope(value: Any) -> ReceiptAttestationEnvelope:
    if isinstance(value, ReceiptAttestationEnvelope):
        return value
    if isinstance(value, Mapping):
        return ReceiptAttestationEnvelope.from_dict(value)
    raise AttestationValidationError(
        "envelope must be a ReceiptAttestationEnvelope or mapping"
    )


@dataclass(frozen=True)
class AttestationVerification(CanonicalContract):
    """Independent verification result and its derived gate authority."""

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_VERIFICATION_SCHEMA

    envelope: ReceiptAttestationEnvelope
    verdict: AttestationVerificationVerdict
    verifier_id: str
    independent: bool = True
    diagnostic_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope", _envelope(self.envelope))
        object.__setattr__(
            self,
            "verdict",
            _enum(
                self.verdict,
                AttestationVerificationVerdict,
                field_name="verdict",
            ),
        )
        object.__setattr__(
            self,
            "verifier_id",
            _text(self.verifier_id, field_name="verifier_id", required=True),
        )
        object.__setattr__(
            self,
            "diagnostic_code",
            _text(self.diagnostic_code, field_name="diagnostic_code"),
        )
        object.__setattr__(
            self, "independent", _bool(self.independent, field_name="independent")
        )

    @property
    def verification_id(self) -> str:
        return self.content_id

    @property
    def verified(self) -> bool:
        return self.verdict is AttestationVerificationVerdict.VERIFIED

    @property
    def simulated(self) -> bool:
        return self.envelope.simulated

    @property
    def is_simulated(self) -> bool:
        return self.simulated

    @property
    def authoritative(self) -> bool:
        return (
            self.verified
            and self.independent
            and self.envelope.production_eligible
            and self.envelope.backend_mode
            is AttestationBackendMode.CRYPTOGRAPHIC
        )

    @property
    def is_authoritative(self) -> bool:
        return self.authoritative

    @property
    def trust(self) -> AttestationTrust:
        if self.authoritative:
            return AttestationTrust.AUTHORITATIVE
        return AttestationTrust.NON_AUTHORITATIVE

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        """Assurance contributed by the envelope alone.

        An authoritative result contributes ``ATTESTED`` only because statement
        creation required the bound base receipt to already be kernel verified.
        A simulated or rejected result contributes no assurance.
        """

        if self.authoritative:
            return AssuranceLevel.ATTESTED
        return AssuranceLevel.UNVERIFIED

    def satisfies_gate(self, gate: AttestationGate) -> bool:
        normalized = _enum(gate, AttestationGate, field_name="gate")
        if normalized in (AttestationGate.SERIALIZATION, AttestationGate.TEST):
            return self.verified
        return self.authoritative

    def satisfies_production_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.PRODUCTION)

    def satisfies_completion_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.COMPLETION)

    def to_evidence(self) -> ProofEvidence:
        """Project the result into the shared proof-assurance vocabulary."""

        statement = self.envelope.statement
        verdict = (
            EvidenceVerdict.ACCEPTED
            if self.verified
            else (
                EvidenceVerdict.REJECTED
                if self.verdict is AttestationVerificationVerdict.REJECTED
                else EvidenceVerdict.ERROR
            )
        )
        metadata = {
            "attestation_contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "attestation_envelope_id": self.envelope.envelope_id,
            "attestation_statement_id": statement.statement_id,
            "backend_id": statement.backend_id,
            "circuit_id": statement.circuit_id,
            "kernel_id": statement.kernel_id,
            "obligation_id": statement.obligation_id,
            "policy_id": statement.policy_id,
            "public_input_digest": statement.public_input_digest,
            "repository_tree_id": statement.repository_tree_id,
            "verification_key_id": statement.verification_key_id,
        }
        if statement.backend_managed:
            metadata.update(
                {
                    "backend_policy_id": statement.backend_policy_id,
                    "backend_version": statement.backend_version,
                    "circuit_version": statement.circuit_version,
                    "public_input_schema_id": statement.public_input_schema_id,
                    "public_input_schema_version": (
                        statement.public_input_schema_version
                    ),
                    "verification_key_version": (
                        statement.verification_key_version
                    ),
                    "backend_health_id": self.envelope.backend_health_id,
                }
            )
        return ProofEvidence(
            kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
            authority=EvidenceAuthority.ATTESTATION_VERIFIER,
            verdict=verdict,
            artifact_id=self.envelope.proof_artifact_id,
            subject_id=statement.receipt_id,
            verifier_id=self.verifier_id,
            freshness=EvidenceFreshness.CURRENT,
            independent=self.independent,
            simulated=self.simulated,
            metadata=metadata,
        )

    as_evidence = to_evidence

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "envelope": self.envelope,
            "envelope_id": self.envelope.envelope_id,
            "verdict": self.verdict,
            "verifier_id": self.verifier_id,
            "independent": self.independent,
            "diagnostic_code": self.diagnostic_code,
            "simulated": self.simulated,
            "authoritative": self.authoritative,
            "trust": self.trust,
            "authoritative_assurance": self.authoritative_assurance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationVerification":
        _schema(payload, cls.SCHEMA)
        result = cls(
            envelope=_envelope(payload.get("envelope") or {}),
            verdict=payload.get(
                "verdict", AttestationVerificationVerdict.ERROR
            ),
            verifier_id=payload.get("verifier_id", ""),
            independent=payload.get("independent", True),
            diagnostic_code=payload.get("diagnostic_code", ""),
        )
        derived_claims = {
            "envelope_id": result.envelope.envelope_id,
            "simulated": result.simulated,
            "authoritative": result.authoritative,
            "trust": result.trust.value,
            "authoritative_assurance": result.authoritative_assurance.value,
        }
        for name, expected in derived_claims.items():
            supplied = payload.get(name)
            if supplied not in (None, "", expected):
                raise AttestationValidationError(
                    "attestation verification %s does not match derived value"
                    % name
                )
        claimed_id = payload.get("verification_id") or payload.get("content_id")
        if claimed_id and claimed_id != result.verification_id:
            raise AttestationValidationError(
                "attestation verification identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "verification_id": self.verification_id}

    to_context_capsule = to_public_artifact
    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


ReceiptAttestationVerification = AttestationVerification
ZKPReceiptAttestationVerification = AttestationVerification
ProofAttestationVerification = AttestationVerification
AttestationVerificationResult = AttestationVerification


def _receipt(value: Any) -> ProofReceipt:
    if isinstance(value, ProofReceipt):
        return value
    if isinstance(value, Mapping):
        try:
            return ProofReceipt.from_dict(value)
        except (TypeError, ValueError, ContractValidationError) as exc:
            raise AttestationValidationError(
                "receipt must be a valid immutable ProofReceipt"
            ) from exc
    raise AttestationValidationError(
        "receipt must be a ProofReceipt or mapping"
    )


def _verification(value: Any) -> AttestationVerification:
    if isinstance(value, AttestationVerification):
        return value
    if isinstance(value, Mapping):
        return AttestationVerification.from_dict(value)
    raise AttestationValidationError(
        "verification must be an AttestationVerification or mapping"
    )


@dataclass(frozen=True)
class PersistedAttestationRecord(CanonicalContract):
    """An independently expiring sidecar for an immutable proof receipt.

    The receipt remains the trust root for kernel assurance.  This record
    carries a complete public copy so every statement binding can be
    reproduced without a witness or mutable provider state.  Its repeated
    identity fields are derived in :meth:`_payload` and checked on load; callers
    cannot use serialized claims to change the effective assurance.
    """

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_RECORD_SCHEMA

    receipt: ProofReceipt
    verification: AttestationVerification
    created_at: str
    expires_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", _receipt(self.receipt))
        object.__setattr__(
            self, "verification", _verification(self.verification)
        )
        object.__setattr__(
            self,
            "created_at",
            _timestamp(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self,
            "expires_at",
            _timestamp(self.expires_at, field_name="expires_at"),
        )
        if _timestamp_value(self.expires_at) <= _timestamp_value(self.created_at):
            raise AttestationValidationError(
                "attestation expiration must be after creation"
            )

        self.receipt.require_kernel_verified()
        if not self.receipt.kernel_receipt_id:
            raise AttestationValidationError(
                "persisted attestations require an immutable kernel receipt identity"
            )
        verification = self.verification
        if not verification.authoritative:
            raise AttestationValidationError(
                "only authoritative independently verified attestations may be persisted"
            )
        statement = verification.envelope.statement
        if not statement.backend_managed:
            raise AttestationValidationError(
                "persisted attestations require a version-pinned backend policy"
            )
        health = verification.envelope.backend_health
        if health is None:
            raise AttestationValidationError(
                "persisted attestations require backend health evidence"
            )
        if not statement.matches_backend_policy(health.policy):
            raise AttestationValidationError(
                "persisted attestation does not match its backend policy"
            )
        expected = ReceiptAttestationStatement.from_receipt(
            self.receipt,
            circuit_id=health.policy.circuit_id,
            backend_id=health.policy.backend_id,
            verification_key_id=health.policy.verification_key_id,
            backend_policy=health.policy,
        )
        if statement != expected:
            raise AttestationValidationError(
                "attestation statement does not bind the immutable proof receipt"
            )
        if not health.policy.key_is_current_at(self.created_at):
            raise AttestationValidationError(
                "attestation was created with an expired verification key"
            )
        key_expiry = health.policy.verification_key_expires_at
        if key_expiry and _timestamp_value(self.expires_at) > _timestamp_value(
            key_expiry
        ):
            raise AttestationValidationError(
                "attestation expiration exceeds verification-key expiration"
            )

    @property
    def record_id(self) -> str:
        return self.content_id

    @property
    def proof_receipt_id(self) -> str:
        """Identity of the immutable receipt attested by the statement."""

        return self.receipt.receipt_id

    @property
    def base_receipt_id(self) -> str:
        return self.proof_receipt_id

    @property
    def kernel_receipt_id(self) -> str:
        """Immutable kernel reconstruction receipt referenced by the base receipt."""

        return self.receipt.kernel_receipt_id

    @property
    def envelope(self) -> ReceiptAttestationEnvelope:
        return self.verification.envelope

    @property
    def envelope_id(self) -> str:
        return self.envelope.envelope_id

    @property
    def verification_id(self) -> str:
        return self.verification.verification_id

    @property
    def statement_id(self) -> str:
        return self.envelope.statement.statement_id

    @property
    def public_input_digest(self) -> str:
        return self.envelope.statement.public_input_digest

    @property
    def backend_policy(self) -> AttestationBackendPolicy:
        health = self.envelope.backend_health
        if health is None:  # guarded during construction
            raise AttestationValidationError("backend health is unavailable")
        return health.policy

    def is_current_at(self, timestamp: str) -> bool:
        checked = _timestamp(timestamp, field_name="timestamp")
        value = _timestamp_value(checked)
        return (
            _timestamp_value(self.created_at) <= value
            and value < _timestamp_value(self.expires_at)
            and self.backend_policy.key_is_current_at(checked)
        )

    current_at = is_current_at

    def effective_assurance_at(self, timestamp: str) -> AssuranceLevel:
        """Return the sidecar view without mutating the underlying receipt."""

        if self.is_current_at(timestamp) and self.verification.authoritative:
            return AssuranceLevel.ATTESTED
        return self.receipt.authoritative_assurance

    def _payload(self) -> Dict[str, Any]:
        statement = self.envelope.statement
        policy = self.backend_policy
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "receipt": self.receipt,
            "proof_receipt_id": self.proof_receipt_id,
            "base_receipt_id": self.base_receipt_id,
            "kernel_receipt_id": self.kernel_receipt_id,
            "verification": self.verification,
            "verification_id": self.verification_id,
            "envelope_id": self.envelope_id,
            "statement_id": self.statement_id,
            "public_input_digest": self.public_input_digest,
            "backend_policy_id": statement.backend_policy_id,
            "formal_policy_id": statement.policy_id,
            "backend_id": statement.backend_id,
            "backend_version": statement.backend_version,
            "circuit_id": statement.circuit_id,
            "circuit_version": statement.circuit_version,
            "public_input_schema_id": statement.public_input_schema_id,
            "public_input_schema_version": statement.public_input_schema_version,
            "verification_key_id": statement.verification_key_id,
            "verification_key_version": statement.verification_key_version,
            "verification_key_expires_at": policy.verification_key_expires_at,
            "backend_health_id": self.envelope.backend_health_id,
            "proof_artifact_id": self.envelope.proof_artifact_id,
            "proof_digest": self.envelope.proof_digest,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersistedAttestationRecord":
        _schema(payload, cls.SCHEMA)
        result = cls(
            receipt=_receipt(payload.get("receipt") or {}),
            verification=_verification(payload.get("verification") or {}),
            created_at=payload.get("created_at", ""),
            expires_at=payload.get("expires_at", ""),
        )
        derived = result._payload()
        for name, expected in derived.items():
            if name in {"contract_version", "receipt", "verification"}:
                continue
            supplied = payload.get(name)
            if supplied not in (None, "", expected):
                raise AttestationValidationError(
                    "persisted attestation %s does not match derived binding" % name
                )
        claimed_id = payload.get("record_id") or payload.get("content_id")
        if claimed_id and claimed_id != result.record_id:
            raise AttestationValidationError(
                "persisted attestation record identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "record_id": self.record_id}

    to_cache_record = to_public_artifact
    to_context_capsule = to_public_artifact
    to_log_record = to_public_artifact


ProofAttestationRecord = PersistedAttestationRecord
ReceiptAttestationRecord = PersistedAttestationRecord
StoredProofAttestation = PersistedAttestationRecord
ZKPReceiptAttestationRecord = PersistedAttestationRecord


def build_persisted_attestation_record(
    receipt: ProofReceipt,
    verification: AttestationVerification,
    *,
    created_at: str,
    expires_at: str,
) -> PersistedAttestationRecord:
    """Bind a verified public envelope beside its immutable kernel receipt."""

    return PersistedAttestationRecord(
        receipt=receipt,
        verification=verification,
        created_at=created_at,
        expires_at=expires_at,
    )


bind_attestation_record = build_persisted_attestation_record
persisted_attestation_record = build_persisted_attestation_record


def reproduce_attestation_verification(
    record: PersistedAttestationRecord | Mapping[str, Any],
    *,
    verifier: Callable[[ReceiptAttestationEnvelope], bool],
    checked_at: str,
    receipt: ProofReceipt | Mapping[str, Any] | None = None,
    backend_policy: AttestationBackendPolicy | Mapping[str, Any] | None = None,
    verifier_id: str | None = None,
) -> AttestationVerification:
    """Re-run verification exclusively from persisted public contracts.

    Serialized ``verified`` and ``authoritative`` claims are never sufficient:
    all receipt and backend bindings are reconstructed first and the supplied
    independent verifier is invoked again.  An expired record fails closed
    before cryptographic verification.
    """

    checked = (
        record
        if isinstance(record, PersistedAttestationRecord)
        else PersistedAttestationRecord.from_dict(record)
    )
    if not callable(verifier):
        raise AttestationValidationError("verifier must be callable")
    if receipt is not None and _receipt(receipt).receipt_id != checked.proof_receipt_id:
        raise AttestationValidationError(
            "persisted attestation does not match the expected proof receipt"
        )
    if backend_policy is not None:
        expected_policy = _backend_policy(backend_policy)
        if expected_policy.policy_id != checked.backend_policy.policy_id:
            raise AttestationValidationError(
                "persisted attestation does not match the expected backend policy"
            )
    if not checked.is_current_at(checked_at):
        raise AttestationValidationError(
            "persisted attestation is not current at verification time"
        )
    try:
        verified = verifier(checked.envelope)
    except Exception:
        return AttestationVerification(
            envelope=checked.envelope,
            verdict=AttestationVerificationVerdict.ERROR,
            verifier_id=verifier_id or checked.verification.verifier_id,
            independent=True,
            diagnostic_code="persisted_attestation_verifier_error",
        )
    if not isinstance(verified, bool):
        return AttestationVerification(
            envelope=checked.envelope,
            verdict=AttestationVerificationVerdict.ERROR,
            verifier_id=verifier_id or checked.verification.verifier_id,
            independent=True,
            diagnostic_code="persisted_attestation_verifier_non_boolean",
        )
    return record_attestation_verification(
        checked.envelope,
        verified=verified,
        verifier_id=verifier_id or checked.verification.verifier_id,
        independent=True,
        diagnostic_code="" if verified else "persisted_attestation_rejected",
    )


verify_attestation_record = reproduce_attestation_verification
replay_attestation_verification = reproduce_attestation_verification


def build_receipt_attestation_statement(
    receipt: ProofReceipt,
    *,
    circuit_id: str | None = None,
    backend_id: str | None = None,
    verification_key_id: str | None = None,
    backend_policy: AttestationBackendPolicy | None = None,
) -> ReceiptAttestationStatement:
    """Prepare the canonical public statement for an eligible receipt."""

    if backend_policy is not None:
        policy = _backend_policy(backend_policy)
        circuit_id = circuit_id or policy.circuit_id
        backend_id = backend_id or policy.backend_id
        verification_key_id = (
            verification_key_id or policy.verification_key_id
        )
    if not circuit_id or not backend_id or not verification_key_id:
        raise AttestationValidationError(
            "circuit_id, backend_id, and verification_key_id are required"
        )
    return ReceiptAttestationStatement.from_receipt(
        receipt,
        circuit_id=circuit_id,
        backend_id=backend_id,
        verification_key_id=verification_key_id,
        backend_policy=backend_policy,
    )


def prepare_receipt_attestation(
    receipt: ProofReceipt,
    *,
    circuit_id: str | None = None,
    backend_id: str | None = None,
    verification_key_id: str | None = None,
    backend_policy: AttestationBackendPolicy | None = None,
    witness: PrivateAttestationWitness,
) -> ReceiptAttestationRequest:
    """Create an ephemeral proving request after the kernel receipt gate."""

    statement = build_receipt_attestation_statement(
        receipt,
        circuit_id=circuit_id,
        backend_id=backend_id,
        verification_key_id=verification_key_id,
        backend_policy=backend_policy,
    )
    return ReceiptAttestationRequest(statement=statement, _witness=witness)


prepare_proof_attestation = prepare_receipt_attestation
prepare_attestation = prepare_receipt_attestation
build_attestation_statement = build_receipt_attestation_statement


def create_attestation_envelope(
    request: ReceiptAttestationRequest,
    *,
    backend_mode: AttestationBackendMode,
    proof_artifact_id: str,
    proof_digest: str,
    prover_id: str = "",
    backend_health: BackendHealthReport | None = None,
) -> ReceiptAttestationEnvelope:
    """Record public prover output for an already-authorized request."""

    if not isinstance(request, ReceiptAttestationRequest):
        raise AttestationValidationError(
            "envelope generation requires a prepared receipt-attestation request"
        )
    return ReceiptAttestationEnvelope(
        statement=request.statement,
        backend_mode=backend_mode,
        proof_artifact_id=proof_artifact_id,
        proof_digest=proof_digest,
        prover_id=prover_id,
        backend_health=backend_health,
    )


def evaluate_backend_health(
    policy: AttestationBackendPolicy,
    *,
    configured: bool,
    available: bool,
    outcomes: Mapping[
        BackendTestCase | str, BackendTestVerdict | str | bool
    ] = MappingProxyType({}),
    evaluated_at: str,
    diagnostics: Mapping[BackendTestCase | str, str] = MappingProxyType({}),
) -> BackendHealthReport:
    """Build a derived backend report from secret-free fixture outcomes.

    A boolean outcome means that the *fixture expectation* passed.  Thus
    ``True`` for the negative, stale-key, and malformed-proof cases means the
    verifier correctly rejected those inputs; it never means the invalid proof
    was accepted.
    """

    checked_policy = _backend_policy(policy)
    normalized_diagnostics = {
        BackendTestCase(str(getattr(case, "value", case))): str(value)
        for case, value in diagnostics.items()
    }
    results = []
    for raw_case, raw_verdict in outcomes.items():
        case = BackendTestCase(str(getattr(raw_case, "value", raw_case)))
        if isinstance(raw_verdict, bool):
            verdict = (
                BackendTestVerdict.PASSED
                if raw_verdict
                else BackendTestVerdict.FAILED
            )
        else:
            verdict = BackendTestVerdict(
                str(getattr(raw_verdict, "value", raw_verdict))
            )
        results.append(
            BackendTestResult(
                case=case,
                verdict=verdict,
                backend_policy_id=checked_policy.policy_id,
                observed_at=evaluated_at,
                diagnostic_code=normalized_diagnostics.get(case, ""),
            )
        )
    return BackendHealthReport(
        policy=checked_policy,
        configured=configured,
        available=available,
        test_results=tuple(results),
        evaluated_at=evaluated_at,
    )


def run_backend_self_tests(
    policy: AttestationBackendPolicy,
    *,
    cases: Mapping[BackendTestCase | str, Callable[[], bool]],
    configured: bool,
    available: bool,
    evaluated_at: str,
) -> BackendHealthReport:
    """Run caller-supplied bounded fixture adapters and fail closed on errors.

    Backend-specific command execution belongs in adapters because ProveKit
    and Groth16 have different wire formats.  This runner supplies the common
    policy binding, complete-case requirement, exception handling, and
    production promotion semantics.
    """

    outcomes: Dict[BackendTestCase, BackendTestVerdict] = {}
    diagnostics: Dict[BackendTestCase, str] = {}
    for raw_case, callback in cases.items():
        case = BackendTestCase(str(getattr(raw_case, "value", raw_case)))
        if not callable(callback):
            outcomes[case] = BackendTestVerdict.ERROR
            diagnostics[case] = "fixture_not_callable"
            continue
        try:
            passed = callback()
        except Exception:
            outcomes[case] = BackendTestVerdict.ERROR
            diagnostics[case] = "fixture_raised"
            continue
        if not isinstance(passed, bool):
            outcomes[case] = BackendTestVerdict.ERROR
            diagnostics[case] = "fixture_returned_non_boolean"
            continue
        outcomes[case] = (
            BackendTestVerdict.PASSED
            if passed
            else BackendTestVerdict.FAILED
        )
    return evaluate_backend_health(
        policy,
        configured=configured,
        available=available,
        outcomes=outcomes,
        diagnostics=diagnostics,
        evaluated_at=evaluated_at,
    )


def witness_no_leak_test_result(
    policy: AttestationBackendPolicy,
    *,
    artifacts: Sequence[Any],
    secret_probes: Sequence[str | bytes],
    observed_at: str,
) -> BackendTestResult:
    """Evaluate public artifacts without publishing probes or witness names."""

    checked_policy = _backend_policy(policy)
    try:
        leaked = any(
            public_artifact_contains(artifact, probe)
            for artifact in artifacts
            for probe in secret_probes
        )
        verdict = (
            BackendTestVerdict.FAILED
            if leaked
            else BackendTestVerdict.PASSED
        )
        diagnostic = "witness_disclosure_detected" if leaked else ""
    except Exception:
        verdict = BackendTestVerdict.ERROR
        diagnostic = "no_leak_check_error"
    return BackendTestResult(
        case=BackendTestCase.WITNESS_NO_LEAK,
        verdict=verdict,
        backend_policy_id=checked_policy.policy_id,
        observed_at=observed_at,
        diagnostic_code=diagnostic,
    )


def execute_cryptographic_attestation(
    request: ReceiptAttestationRequest,
    *,
    backend_health: BackendHealthReport,
    prover: Callable[[ReceiptAttestationRequest], Mapping[str, Any]],
    verifier: Callable[[ReceiptAttestationEnvelope], bool],
    prover_id: str,
    verifier_id: str,
) -> AttestationVerification:
    """Execute one managed cryptographic attempt with no fallback path.

    ``prover`` returns only ``proof_artifact_id`` and ``proof_digest``.  Any
    proving exception or malformed output raises ``CryptographicBackendFailure``.
    A verifier rejection or exception returns a non-authoritative rejected or
    error result.  Simulation is never invoked after either failure.
    """

    if not isinstance(request, ReceiptAttestationRequest):
        raise AttestationValidationError(
            "cryptographic execution requires a prepared attestation request"
        )
    health = _backend_health(backend_health)
    health.require_production_eligible()
    if health.policy.backend_mode is not AttestationBackendMode.CRYPTOGRAPHIC:
        raise CryptographicBackendFailure(
            "managed cryptographic execution cannot use a simulated policy"
        )
    if not request.statement.matches_backend_policy(health.policy):
        raise CryptographicBackendFailure(
            "attestation request does not match backend health policy"
        )
    if not callable(prover) or not callable(verifier):
        raise AttestationValidationError("prover and verifier must be callable")

    try:
        output = prover(request)
    except Exception as exc:
        raise CryptographicBackendFailure(
            "cryptographic proof generation failed"
        ) from exc
    if not isinstance(output, Mapping):
        raise CryptographicBackendFailure(
            "cryptographic prover returned a malformed result"
        )
    proof_artifact_id = output.get("proof_artifact_id", "")
    proof_digest = output.get("proof_digest", "")
    try:
        envelope = create_attestation_envelope(
            request,
            backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
            proof_artifact_id=proof_artifact_id,
            proof_digest=proof_digest,
            prover_id=prover_id,
            backend_health=health,
        )
    except AttestationValidationError as exc:
        raise CryptographicBackendFailure(
            "cryptographic prover returned malformed proof metadata"
        ) from exc

    try:
        verified = verifier(envelope)
    except Exception:
        return AttestationVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.ERROR,
            verifier_id=verifier_id,
            independent=True,
            diagnostic_code="cryptographic_verifier_error",
        )
    if not isinstance(verified, bool):
        return AttestationVerification(
            envelope=envelope,
            verdict=AttestationVerificationVerdict.ERROR,
            verifier_id=verifier_id,
            independent=True,
            diagnostic_code="cryptographic_verifier_non_boolean",
        )
    return record_attestation_verification(
        envelope,
        verified=verified,
        verifier_id=verifier_id,
        independent=True,
        diagnostic_code="" if verified else "cryptographic_proof_rejected",
    )


gate_cryptographic_backend = execute_cryptographic_attestation


def record_attestation_verification(
    envelope: ReceiptAttestationEnvelope,
    *,
    verified: bool,
    verifier_id: str,
    independent: bool = True,
    diagnostic_code: str = "",
) -> AttestationVerification:
    """Create a fail-closed independent verification result."""

    checked = _bool(verified, field_name="verified")
    return AttestationVerification(
        envelope=envelope,
        verdict=(
            AttestationVerificationVerdict.VERIFIED
            if checked
            else AttestationVerificationVerdict.REJECTED
        ),
        verifier_id=verifier_id,
        independent=independent,
        diagnostic_code=diagnostic_code,
    )


def attestation_satisfies_gate(
    verification: AttestationVerification, gate: AttestationGate
) -> bool:
    """Return whether a verified envelope can satisfy ``gate``."""

    if not isinstance(verification, AttestationVerification):
        return False
    return verification.satisfies_gate(gate)


def public_attestation_artifact(value: Any) -> Any:
    """Return a canonical public value or reject witness-bearing objects.

    This helper is intended for log, context-capsule, artifact, and cache
    boundaries.  Public attestation contracts are supported directly.  A
    proving request is reduced to its explicitly redacted public view.
    """

    if isinstance(value, PrivateAttestationWitness):
        raise WitnessDisclosureError(
            "private witness cannot enter a public artifact"
        )
    if isinstance(value, ReceiptAttestationRequest):
        return value.to_public_artifact()
    if isinstance(
        value,
        (
            AttestationBackendPolicy,
            BackendTestResult,
            BackendHealthReport,
            ReceiptAttestationStatement,
            ReceiptAttestationEnvelope,
            AttestationVerification,
            PersistedAttestationRecord,
        ),
    ):
        return value.to_public_artifact()
    if isinstance(value, Mapping):
        return {
            str(key): public_attestation_artifact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [public_attestation_artifact(item) for item in value]
    return _canonical_value(value)


def public_artifact_contains(
    artifact: Any, secret: str | bytes
) -> bool:
    """Test helper for witness no-leak assertions."""

    needle = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if not needle:
        raise AttestationValidationError("secret probe must not be empty")
    encoded = canonical_json_bytes(public_attestation_artifact(artifact))
    return needle in encoded


__all__ = [
    "ATTESTATION_BACKEND_HEALTH_SCHEMA",
    "ATTESTATION_BACKEND_POLICY_SCHEMA",
    "ATTESTATION_BACKEND_TEST_RESULT_SCHEMA",
    "PROOF_ATTESTATION_CONTRACT_VERSION",
    "PROOF_ATTESTATION_ENVELOPE_SCHEMA",
    "PROOF_ATTESTATION_RECORD_SCHEMA",
    "PERSISTED_ATTESTATION_SCHEMA",
    "PROOF_ATTESTATION_STATEMENT_SCHEMA",
    "PROOF_ATTESTATION_VERIFICATION_SCHEMA",
    "RECEIPT_ATTESTATION_ENVELOPE_SCHEMA",
    "RECEIPT_ATTESTATION_RECORD_SCHEMA",
    "RECEIPT_ATTESTATION_STATEMENT_SCHEMA",
    "ATTESTATION_VERIFICATION_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_ENVELOPE_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_RECORD_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_STATEMENT_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_VERIFICATION_SCHEMA",
    "AttestationBackendHealth",
    "AttestationBackendHealthReport",
    "AttestationBackendMode",
    "AttestationBackendPolicy",
    "AttestationEnvelope",
    "AttestationGate",
    "AttestationMode",
    "AttestationTrust",
    "AttestationValidationError",
    "AttestationVerification",
    "AttestationVerificationResult",
    "AttestationVerificationVerdict",
    "BackendConformanceEvidence",
    "BackendHealthReport",
    "BackendPolicy",
    "BackendSelfTestResult",
    "BackendTestCase",
    "BackendTestResult",
    "BackendTestVerdict",
    "CryptographicBackendFailure",
    "CryptographicBackendHealth",
    "CryptographicBackendPolicy",
    "PrivateAttestationWitness",
    "ProofAttestationEnvelope",
    "ProofAttestationRequest",
    "ProofAttestationRecord",
    "ProofAttestationStatement",
    "ProofAttestationVerification",
    "ReceiptAttestationEnvelope",
    "ReceiptAttestationRequest",
    "ReceiptAttestationRecord",
    "ReceiptAttestationStatement",
    "ReceiptAttestationVerification",
    "ReceiptAttestationWitness",
    "WitnessDisclosureError",
    "ZKPBackendMode",
    "ZKPReceiptAttestation",
    "ZKPReceiptAttestationRequest",
    "ZKPReceiptAttestationRecord",
    "ZKPReceiptAttestationStatement",
    "ZKPReceiptAttestationVerification",
    "ZKPWitness",
    "REQUIRED_BACKEND_TEST_CASES",
    "attestation_satisfies_gate",
    "bind_attestation_record",
    "build_persisted_attestation_record",
    "build_receipt_attestation_statement",
    "build_attestation_statement",
    "create_attestation_envelope",
    "evaluate_backend_health",
    "execute_cryptographic_attestation",
    "gate_cryptographic_backend",
    "prepare_proof_attestation",
    "prepare_attestation",
    "prepare_receipt_attestation",
    "public_artifact_contains",
    "public_attestation_artifact",
    "record_attestation_verification",
    "replay_attestation_verification",
    "reproduce_attestation_verification",
    "run_backend_self_tests",
    "witness_no_leak_test_result",
    "PersistedAttestationRecord",
    "StoredProofAttestation",
    "verify_attestation_record",
]
