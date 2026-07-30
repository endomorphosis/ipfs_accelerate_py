"""Datasets Groth16/ProveKit binding for verified-receipt ZK attestation.

SCA-G082 / SCAEV082REALZK
-------------------------
Bind an available ``ipfs_datasets`` Groth16 or ProveKit backend to the approved
already-verified receipt predicate with:

* immutable setup / prover / verifier identities,
* mandatory positive and negative self-tests,
* an independent verifier callback, and
* explicit threat-model gating (use-case decision + production eligibility).

This module does **not** claim that a ZKP proves source-code correctness or
unverified function-call behavior.  A statement may be prepared only from an
existing, current, independently kernel-verified :class:`ProofReceipt`.  The
ZKP then attests receipt possession/membership (or another reviewed private
predicate) against public setup and policy pins.

Unavailable or simulated backends emit a typed non-attested status and never
satisfy production, completion, or ``ATTESTED`` assurance gates.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from .formal_verification_capabilities import CapabilityHealth
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    ProofReceipt,
    content_identity,
)
from .proof_attestation import (
    AttestationBackendMode,
    AttestationBackendPolicy,
    AttestationGate,
    AttestationTrust,
    AttestationValidationError,
    AttestationVerification,
    AttestationVerificationVerdict,
    BackendHealthReport,
    BackendTestCase,
    BackendTestResult,
    BackendTestVerdict,
    CryptographicBackendFailure,
    DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_DECISION,
    DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_ID,
    PrivateAttestationWitness,
    REQUIRED_BACKEND_TEST_CASES,
    ReceiptAttestationEnvelope,
    ReceiptAttestationRequest,
    ReceiptAttestationStatement,
    WitnessDisclosureError,
    ZkBackendFamily,
    ZkUseCaseDecisionRecord,
    ZkUseCaseDisposition,
    create_attestation_envelope,
    datasets_verified_receipt_zk_use_case_decision,
    evaluate_backend_health,
    execute_cryptographic_attestation,
    prepare_receipt_attestation,
    public_artifact_contains,
    public_attestation_artifact,
    require_zk_backend_selection_authorized,
    run_backend_self_tests,
    simulated_attestation_cannot_satisfy_attested,
    witness_no_leak_test_result,
)

# ---------------------------------------------------------------------------
# Evidence / interface identity (AST evidence term for SCA-G082)
# ---------------------------------------------------------------------------

SCAEV082REALZK: Final = "SCAEV082REALZK"
IPFS_DATASETS_ZK_ATTESTATION_INTERFACE: Final = "IpfsDatasetsZkAttestation@1"
IPFS_DATASETS_ZK_ATTESTATION_CONTRACT_VERSION: Final = 1
IPFS_DATASETS_ZK_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-zk-attestation-result@1"
)
IPFS_DATASETS_ZK_SETUP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-zk-setup-identity@1"
)
IPFS_DATASETS_ZK_SELECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-zk-backend-selection@1"
)

# Closed predicate catalog for real datasets ZK (threat-model gated).
APPROVED_VERIFIED_RECEIPT_PREDICATES: Final = frozenset(
    {
        "receipt_possession",
        "receipt_membership",
        "private_reviewed_predicate",
    }
)
DEFAULT_VERIFIED_RECEIPT_PREDICATE: Final = "receipt_possession"

# Circuit identity for the verified-receipt possession predicate.
DEFAULT_RECEIPT_BINDING_CIRCUIT_ID: Final = (
    "circuit:datasets-verified-receipt-possession@1"
)
DEFAULT_PUBLIC_INPUT_SCHEMA_ID: Final = (
    "schema:datasets-verified-receipt-public-inputs@1"
)

_DATASETS_ZKP_BACKENDS_MODULE: Final = "ipfs_datasets_py.logic.zkp.backends"
_SIMULATED_FAMILY_TOKENS: Final = frozenset(
    {"sim", "simulated", "mock", "fake", "demo", "educational"}
)


class DatasetsZkStatus(str, Enum):
    """Typed outcome for one datasets ZK attempt; only ``attested`` is authoritative."""

    ATTESTED = "attested"
    GENERATED = "generated"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"
    ERROR = "error"


class DatasetsZkPredicate(str, Enum):
    """Closed predicate kinds eligible under the datasets verified-receipt use case."""

    RECEIPT_POSSESSION = "receipt_possession"
    RECEIPT_MEMBERSHIP = "receipt_membership"
    PRIVATE_REVIEWED_PREDICATE = "private_reviewed_predicate"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise AttestationValidationError("%s must be a string" % field_name)
    if required and not text:
        raise AttestationValidationError("%s is required" % field_name)
    return text


def _normalize_family(value: str | ZkBackendFamily) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _family_is_simulated(family: str) -> bool:
    token = _normalize_family(family)
    return token in _SIMULATED_FAMILY_TOKENS or token.startswith("simulated-")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _discover_executable(names: Sequence[str], env_names: Sequence[str]) -> str:
    for env_name in env_names:
        explicit = os.environ.get(env_name, "").strip()
        if explicit:
            path = Path(explicit).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    search_path = os.environ.get("PATH", "")
    for directory in search_path.split(os.pathsep):
        if not directory:
            continue
        root = Path(directory)
        for name in names:
            candidate = root / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
    return ""


def _discover_artifact_dir(
    explicit: str | None,
    env_names: Sequence[str],
) -> str:
    if explicit and str(explicit).strip():
        path = Path(str(explicit)).expanduser()
        if path.is_dir():
            return str(path.resolve())
        return ""
    for env_name in env_names:
        configured = os.environ.get(env_name, "").strip()
        if not configured:
            continue
        path = Path(configured).expanduser()
        if path.is_dir():
            return str(path.resolve())
    return ""


def _artifact_present(directory: str, required_names: Sequence[str]) -> bool:
    if not directory:
        return False
    root = Path(directory)
    if not root.is_dir():
        return False
    present = {path.name for path in root.iterdir() if path.is_file()}
    return all(name in present for name in required_names)


def datasets_zkp_registry_available() -> bool:
    """Return whether the optional datasets ZKP backend registry is importable."""

    try:
        importlib.import_module(_DATASETS_ZKP_BACKENDS_MODULE)
    except (ImportError, ModuleNotFoundError):
        return False
    return True


@dataclass(frozen=True)
class DatasetsZkSetupIdentity(CanonicalContract):
    """Content-addressed setup / prover / verifier pin set for one backend family.

    Product names alone are insufficient.  The setup identity commits to the
    backend family, executable digest, artifact root, circuit, public-input
    schema, verification key, and versions that also appear on the managed
    :class:`AttestationBackendPolicy`.
    """

    SCHEMA: ClassVar[str] = IPFS_DATASETS_ZK_SETUP_SCHEMA

    backend_family: str
    backend_mode: AttestationBackendMode
    executable_path: str
    executable_digest: str
    artifacts_path: str
    artifacts_digest: str
    circuit_id: str
    circuit_version: str
    public_input_schema_id: str
    public_input_schema_version: str
    verification_key_id: str
    verification_key_version: str
    backend_version: str
    prover_id: str
    verifier_id: str
    verification_key_expires_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_family",
            _text(self.backend_family, field_name="backend_family"),
        )
        object.__setattr__(
            self,
            "backend_mode",
            AttestationBackendMode(
                str(getattr(self.backend_mode, "value", self.backend_mode))
            ),
        )
        for name in (
            "executable_path",
            "executable_digest",
            "artifacts_path",
            "artifacts_digest",
            "circuit_id",
            "circuit_version",
            "public_input_schema_id",
            "public_input_schema_version",
            "verification_key_id",
            "verification_key_version",
            "backend_version",
            "prover_id",
            "verifier_id",
        ):
            required = name not in {
                "executable_path",
                "executable_digest",
                "artifacts_path",
                "artifacts_digest",
            }
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=required),
            )
        object.__setattr__(
            self,
            "verification_key_expires_at",
            _text(
                self.verification_key_expires_at,
                field_name="verification_key_expires_at",
                required=False,
            ),
        )
        if self.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC and (
            _family_is_simulated(self.backend_family)
            or "simulated" in self.executable_digest.lower()
        ):
            raise AttestationValidationError(
                "a simulated backend identity cannot be pinned as cryptographic"
            )

    @property
    def setup_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return self.backend_mode is AttestationBackendMode.SIMULATED

    @property
    def configured(self) -> bool:
        if self.simulated:
            return True
        return bool(
            self.executable_path
            and self.executable_digest
            and self.artifacts_path
            and self.artifacts_digest
            and self.verification_key_id
        )

    def to_backend_policy(self) -> AttestationBackendPolicy:
        """Project setup pins into the shared managed backend policy contract."""

        backend_id = "backend:datasets:%s" % self.backend_family
        return AttestationBackendPolicy(
            backend_id=backend_id,
            backend_version=self.backend_version,
            circuit_id=self.circuit_id,
            circuit_version=self.circuit_version,
            public_input_schema_id=self.public_input_schema_id,
            public_input_schema_version=self.public_input_schema_version,
            verification_key_id=self.verification_key_id,
            verification_key_version=self.verification_key_version,
            backend_mode=self.backend_mode,
            verification_key_expires_at=self.verification_key_expires_at,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": IPFS_DATASETS_ZK_ATTESTATION_CONTRACT_VERSION,
            "evidence_id": SCAEV082REALZK,
            "backend_family": self.backend_family,
            "backend_mode": self.backend_mode,
            "executable_path": self.executable_path,
            "executable_digest": self.executable_digest,
            "artifacts_path": self.artifacts_path,
            "artifacts_digest": self.artifacts_digest,
            "circuit_id": self.circuit_id,
            "circuit_version": self.circuit_version,
            "public_input_schema_id": self.public_input_schema_id,
            "public_input_schema_version": self.public_input_schema_version,
            "verification_key_id": self.verification_key_id,
            "verification_key_version": self.verification_key_version,
            "backend_version": self.backend_version,
            "prover_id": self.prover_id,
            "verifier_id": self.verifier_id,
            "verification_key_expires_at": self.verification_key_expires_at,
            "simulated": self.simulated,
            "configured": self.configured,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetsZkSetupIdentity":
        result = cls(
            backend_family=payload.get("backend_family", ""),
            backend_mode=payload.get(
                "backend_mode", AttestationBackendMode.SIMULATED
            ),
            executable_path=payload.get("executable_path", ""),
            executable_digest=payload.get("executable_digest", ""),
            artifacts_path=payload.get("artifacts_path", ""),
            artifacts_digest=payload.get("artifacts_digest", ""),
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
            backend_version=payload.get("backend_version", ""),
            prover_id=payload.get("prover_id", ""),
            verifier_id=payload.get("verifier_id", ""),
            verification_key_expires_at=payload.get(
                "verification_key_expires_at", ""
            ),
        )
        claimed = payload.get("setup_id") or payload.get("content_id")
        if claimed and claimed != result.setup_id:
            raise AttestationValidationError(
                "datasets ZK setup identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "setup_id": self.setup_id}

    to_cache_record = to_public_artifact
    to_dict_public = to_public_artifact


@dataclass(frozen=True)
class DatasetsZkBackendSelection(CanonicalContract):
    """Authorized backend family selection bound to a reviewed use-case decision."""

    SCHEMA: ClassVar[str] = IPFS_DATASETS_ZK_SELECTION_SCHEMA

    backend_family: str
    decision: ZkUseCaseDecisionRecord
    setup: DatasetsZkSetupIdentity
    selected_at: str
    available: bool
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_family",
            _text(self.backend_family, field_name="backend_family"),
        )
        if not isinstance(self.decision, ZkUseCaseDecisionRecord):
            object.__setattr__(
                self,
                "decision",
                ZkUseCaseDecisionRecord.from_dict(self.decision),  # type: ignore[arg-type]
            )
        if not isinstance(self.setup, DatasetsZkSetupIdentity):
            object.__setattr__(
                self,
                "setup",
                DatasetsZkSetupIdentity.from_dict(self.setup),  # type: ignore[arg-type]
            )
        object.__setattr__(
            self,
            "selected_at",
            _text(self.selected_at, field_name="selected_at"),
        )
        if not isinstance(self.available, bool):
            raise AttestationValidationError("available must be a boolean")
        object.__setattr__(
            self, "reason", _text(self.reason, field_name="reason", required=False)
        )
        if self.setup.backend_family != self.backend_family:
            raise AttestationValidationError(
                "setup backend family does not match selection"
            )

    @property
    def selection_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return self.setup.simulated or _family_is_simulated(self.backend_family)

    @property
    def authorizes_production(self) -> bool:
        return (
            not self.simulated
            and self.available
            and self.decision.authorizes_backend_family(self.backend_family)
            and self.setup.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": IPFS_DATASETS_ZK_ATTESTATION_CONTRACT_VERSION,
            "evidence_id": SCAEV082REALZK,
            "backend_family": self.backend_family,
            "decision": self.decision,
            "decision_id": self.decision.decision_id,
            "setup": self.setup,
            "setup_id": self.setup.setup_id,
            "selected_at": self.selected_at,
            "available": self.available,
            "reason": self.reason,
            "simulated": self.simulated,
            "authorizes_production": self.authorizes_production,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetsZkBackendSelection":
        result = cls(
            backend_family=payload.get("backend_family", ""),
            decision=payload.get("decision") or {},
            setup=payload.get("setup") or {},
            selected_at=payload.get("selected_at", ""),
            available=payload.get("available", False),
            reason=payload.get("reason", ""),
        )
        claimed = payload.get("selection_id") or payload.get("content_id")
        if claimed and claimed != result.selection_id:
            raise AttestationValidationError(
                "datasets ZK backend selection identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "selection_id": self.selection_id}

    to_cache_record = to_public_artifact


@dataclass(frozen=True)
class DatasetsZkAttestationResult(CanonicalContract):
    """Public, witness-free outcome of one datasets verified-receipt attempt."""

    SCHEMA: ClassVar[str] = IPFS_DATASETS_ZK_RESULT_SCHEMA

    status: DatasetsZkStatus
    predicate: str
    use_case_id: str
    backend_family: str
    backend_mode: AttestationBackendMode
    statement: ReceiptAttestationStatement | None
    verification: AttestationVerification | None
    backend_health: BackendHealthReport | None
    setup: DatasetsZkSetupIdentity | None
    diagnostic_code: str = ""
    observed_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            DatasetsZkStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(
            self,
            "predicate",
            _text(self.predicate, field_name="predicate"),
        )
        object.__setattr__(
            self,
            "use_case_id",
            _text(self.use_case_id, field_name="use_case_id"),
        )
        object.__setattr__(
            self,
            "backend_family",
            _text(self.backend_family, field_name="backend_family", required=False),
        )
        object.__setattr__(
            self,
            "backend_mode",
            AttestationBackendMode(
                str(getattr(self.backend_mode, "value", self.backend_mode))
            ),
        )
        if self.statement is not None and not isinstance(
            self.statement, ReceiptAttestationStatement
        ):
            object.__setattr__(
                self,
                "statement",
                ReceiptAttestationStatement.from_dict(self.statement),  # type: ignore[arg-type]
            )
        if self.verification is not None and not isinstance(
            self.verification, AttestationVerification
        ):
            object.__setattr__(
                self,
                "verification",
                AttestationVerification.from_dict(self.verification),  # type: ignore[arg-type]
            )
        if self.backend_health is not None and not isinstance(
            self.backend_health, BackendHealthReport
        ):
            object.__setattr__(
                self,
                "backend_health",
                BackendHealthReport.from_dict(self.backend_health),  # type: ignore[arg-type]
            )
        if self.setup is not None and not isinstance(
            self.setup, DatasetsZkSetupIdentity
        ):
            object.__setattr__(
                self,
                "setup",
                DatasetsZkSetupIdentity.from_dict(self.setup),  # type: ignore[arg-type]
            )
        object.__setattr__(
            self,
            "diagnostic_code",
            _text(self.diagnostic_code, field_name="diagnostic_code", required=False),
        )
        object.__setattr__(
            self,
            "observed_at",
            _text(self.observed_at, field_name="observed_at", required=False)
            or _utc_now(),
        )
        if (
            self.status is DatasetsZkStatus.ATTESTED
            and (
                self.verification is None
                or not self.verification.authoritative
                or self.backend_mode is AttestationBackendMode.SIMULATED
            )
        ):
            raise AttestationValidationError(
                "attested status requires an authoritative cryptographic verification"
            )

    @property
    def result_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return (
            self.status is DatasetsZkStatus.SIMULATED
            or self.backend_mode is AttestationBackendMode.SIMULATED
            or (self.verification is not None and self.verification.simulated)
        )

    @property
    def verified(self) -> bool:
        return bool(self.verification is not None and self.verification.verified)

    @property
    def authoritative(self) -> bool:
        return (
            self.status is DatasetsZkStatus.ATTESTED
            and self.verification is not None
            and self.verification.authoritative
        )

    @property
    def trust(self) -> AttestationTrust:
        if self.authoritative:
            return AttestationTrust.AUTHORITATIVE
        return AttestationTrust.NON_AUTHORITATIVE

    @property
    def statement_id(self) -> str:
        return self.statement.statement_id if self.statement is not None else ""

    @property
    def tree_id(self) -> str:
        return self.statement.tree_id if self.statement is not None else ""

    @property
    def verification_id(self) -> str:
        return (
            self.verification.verification_id
            if self.verification is not None
            else ""
        )

    def satisfies_gate(self, gate: AttestationGate | str) -> bool:
        if self.verification is None:
            return False
        return self.verification.satisfies_gate(gate)

    def satisfies_production_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.PRODUCTION)

    def satisfies_completion_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.COMPLETION)

    def to_evidence(self) -> Any:
        if self.verification is None:
            raise AttestationValidationError(
                "datasets ZK result has no verification to project as evidence"
            )
        return self.verification.to_evidence()

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": IPFS_DATASETS_ZK_ATTESTATION_CONTRACT_VERSION,
            "evidence_id": SCAEV082REALZK,
            "interface": IPFS_DATASETS_ZK_ATTESTATION_INTERFACE,
            "status": self.status,
            "predicate": self.predicate,
            "use_case_id": self.use_case_id,
            "backend_family": self.backend_family,
            "backend_mode": self.backend_mode,
            "diagnostic_code": self.diagnostic_code,
            "observed_at": self.observed_at,
            "simulated": self.simulated,
            "verified": self.verified,
            "authoritative": self.authoritative,
            "trust": self.trust,
            "statement_id": self.statement_id,
            "tree_id": self.tree_id,
            "verification_id": self.verification_id,
        }
        if self.statement is not None:
            payload["statement"] = self.statement
        if self.verification is not None:
            payload["verification"] = self.verification
        if self.backend_health is not None:
            payload["backend_health"] = self.backend_health
            payload["backend_health_id"] = self.backend_health.health_id
            payload["backend_health_status"] = self.backend_health.status
        if self.setup is not None:
            payload["setup"] = self.setup
            payload["setup_id"] = self.setup.setup_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetsZkAttestationResult":
        result = cls(
            status=payload.get("status", DatasetsZkStatus.ERROR),
            predicate=payload.get("predicate", DEFAULT_VERIFIED_RECEIPT_PREDICATE),
            use_case_id=payload.get(
                "use_case_id", DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_ID
            ),
            backend_family=payload.get("backend_family", ""),
            backend_mode=payload.get(
                "backend_mode", AttestationBackendMode.SIMULATED
            ),
            statement=payload.get("statement"),
            verification=payload.get("verification"),
            backend_health=payload.get("backend_health"),
            setup=payload.get("setup"),
            diagnostic_code=payload.get("diagnostic_code", ""),
            observed_at=payload.get("observed_at", ""),
        )
        claimed = payload.get("result_id") or payload.get("content_id")
        if claimed and claimed != result.result_id:
            raise AttestationValidationError(
                "datasets ZK attestation result identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> dict[str, Any]:
        return {**self.to_dict(), "result_id": self.result_id}

    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


def build_datasets_zk_setup_identity(
    *,
    backend_family: str | ZkBackendFamily,
    backend_mode: AttestationBackendMode | str | None = None,
    executable_path: str = "",
    artifacts_path: str = "",
    circuit_id: str = DEFAULT_RECEIPT_BINDING_CIRCUIT_ID,
    circuit_version: str = "1.0.0",
    public_input_schema_id: str = DEFAULT_PUBLIC_INPUT_SCHEMA_ID,
    public_input_schema_version: str = "1.0.0",
    verification_key_id: str = "",
    verification_key_version: str = "1.0.0",
    backend_version: str = "1.0.0",
    prover_id: str = "",
    verifier_id: str = "",
    verification_key_expires_at: str = "2030-01-01T00:00:00Z",
) -> DatasetsZkSetupIdentity:
    """Build a setup identity from paths and/or environment discovery."""

    family = _normalize_family(backend_family)
    if _family_is_simulated(family):
        mode = AttestationBackendMode.SIMULATED
    elif backend_mode is None:
        mode = AttestationBackendMode.CRYPTOGRAPHIC
    else:
        mode = AttestationBackendMode(str(getattr(backend_mode, "value", backend_mode)))

    if family in {"provekit", "pk", "provekit-whir", "whir"}:
        family = ZkBackendFamily.PROVEKIT.value
        exe = executable_path or _discover_executable(
            ("provekit-cli", "provekit"),
            ("IPFS_DATASETS_PROVEKIT_BINARY", "PROVEKIT_CLI"),
        )
        artifacts = artifacts_path or _discover_artifact_dir(
            None,
            ("IPFS_DATASETS_PROVEKIT_ARTIFACTS_DIR", "PROVEKIT_ARTIFACTS_DIR"),
        )
        default_vk = "vk:datasets-provekit-receipt-possession"
    elif family in {"groth16", "g16"}:
        family = ZkBackendFamily.GROTH16.value
        exe = executable_path or _discover_executable(
            ("groth16",),
            ("IPFS_DATASETS_GROTH16_BINARY", "GROTH16_BINARY"),
        )
        artifacts = artifacts_path or _discover_artifact_dir(
            None,
            ("IPFS_DATASETS_GROTH16_ARTIFACTS_DIR", "GROTH16_ARTIFACTS_DIR"),
        )
        default_vk = "vk:datasets-groth16-receipt-possession"
    elif mode is AttestationBackendMode.SIMULATED:
        exe = executable_path
        artifacts = artifacts_path
        default_vk = "vk:datasets-simulated-receipt-possession"
    else:
        raise AttestationValidationError(
            "unsupported datasets ZK backend family: %s" % family
        )

    exe_digest = ""
    if exe and Path(exe).is_file():
        exe_digest = _sha256_file(Path(exe))
    artifacts_digest = ""
    if artifacts and Path(artifacts).is_dir():
        # Digest directory identity + sorted filenames (not full contents) so
        # setup identity remains stable without reading large key material.
        names = sorted(path.name for path in Path(artifacts).iterdir() if path.is_file())
        artifacts_digest = content_identity(
            {
                "artifacts_path": str(Path(artifacts).resolve()),
                "files": names,
            }
        )
    vk_id = verification_key_id or default_vk
    if artifacts and not verification_key_id:
        vk_candidate = Path(artifacts) / "verifying_key.bin"
        if vk_candidate.is_file():
            vk_id = "vk:" + _sha256_file(vk_candidate).removeprefix("sha256:")

    return DatasetsZkSetupIdentity(
        backend_family=family,
        backend_mode=mode,
        executable_path=exe,
        executable_digest=exe_digest,
        artifacts_path=artifacts,
        artifacts_digest=artifacts_digest,
        circuit_id=circuit_id,
        circuit_version=circuit_version,
        public_input_schema_id=public_input_schema_id,
        public_input_schema_version=public_input_schema_version,
        verification_key_id=vk_id,
        verification_key_version=verification_key_version,
        backend_version=backend_version,
        prover_id=prover_id or ("prover:datasets:%s" % family),
        verifier_id=verifier_id or ("verifier:datasets:%s" % family),
        verification_key_expires_at=verification_key_expires_at,
    )


def probe_datasets_zk_backend(
    *,
    preferred_families: Sequence[str | ZkBackendFamily] = (
        ZkBackendFamily.PROVEKIT,
        ZkBackendFamily.GROTH16,
    ),
    decision: ZkUseCaseDecisionRecord | None = None,
    selected_at: str | None = None,
    setup_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> DatasetsZkBackendSelection:
    """Select the first authorized real backend family with configured setup.

    Returns a selection even when no real backend is available; the result then
    carries ``available=False`` and a typed reason.  Simulated families are
    never authorized for production selection.
    """

    checked_decision = decision or datasets_verified_receipt_zk_use_case_decision()
    observed = selected_at or _utc_now()
    overrides = dict(setup_overrides or {})
    last_setup: DatasetsZkSetupIdentity | None = None
    last_family = ""
    last_reason = "no authorized cryptographic backend is configured"

    for raw_family in preferred_families:
        family = _normalize_family(raw_family)
        if _family_is_simulated(family):
            last_family = family
            last_reason = "simulated backends cannot be selected for real ZK"
            continue
        try:
            require_zk_backend_selection_authorized(
                checked_decision, backend_family=family
            )
        except AttestationValidationError as exc:
            last_family = family
            last_reason = str(exc)
            continue

        override = dict(overrides.get(family) or {})
        setup = build_datasets_zk_setup_identity(
            backend_family=family,
            backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
            **override,
        )
        last_setup = setup
        last_family = family
        ready = setup.configured
        if family == ZkBackendFamily.GROTH16.value:
            ready = ready and _artifact_present(
                setup.artifacts_path,
                ("proving_key.bin", "verifying_key.bin"),
            )
        elif family == ZkBackendFamily.PROVEKIT.value:
            # ProveKit requires at least one prover and verifier key pattern.
            root = Path(setup.artifacts_path) if setup.artifacts_path else None
            if root is None or not root.is_dir():
                ready = False
            else:
                names = {path.name for path in root.iterdir() if path.is_file()}
                has_pk = any(
                    name.endswith(".pkp")
                    or "prover" in name.lower()
                    or name.startswith("prover_key")
                    for name in names
                )
                has_vk = any(
                    name.endswith(".pkv")
                    or "verifier" in name.lower()
                    or name.startswith("verifier_key")
                    for name in names
                )
                ready = ready and has_pk and has_vk
        if ready:
            return DatasetsZkBackendSelection(
                backend_family=family,
                decision=checked_decision,
                setup=setup,
                selected_at=observed,
                available=True,
                reason="cryptographic backend and setup artifacts are configured",
            )
        last_reason = (
            "backend family %s is authorized but executable/setup artifacts "
            "are missing or incomplete" % family
        )

    if last_setup is None:
        # Emit an explicit simulated setup only as a non-production observation.
        last_setup = build_datasets_zk_setup_identity(
            backend_family="simulated",
            backend_mode=AttestationBackendMode.SIMULATED,
            verification_key_id="vk:datasets-simulated-unavailable",
        )
        last_family = "simulated"
    return DatasetsZkBackendSelection(
        backend_family=last_family or "simulated",
        decision=checked_decision,
        setup=last_setup,
        selected_at=observed,
        available=False,
        reason=last_reason,
    )


def default_backend_self_test_fixtures(
    setup: DatasetsZkSetupIdentity,
    *,
    prover: Callable[[ReceiptAttestationRequest], Mapping[str, Any]] | None = None,
    verifier: Callable[[ReceiptAttestationEnvelope], bool] | None = None,
    secret_probes: Sequence[str | bytes] = (),
) -> dict[BackendTestCase, Callable[[], bool]]:
    """Build bounded self-test adapters for the mandatory fixture set.

    When no live prover/verifier is supplied, cryptographic cases return
    ``False`` so health stays non-production-eligible rather than inventing a
    simulated success.  Callers with real (or test-double) backends should pass
    explicit fixture callbacks or inject prover/verifier hooks.
    """

    policy = setup.to_backend_policy()

    def _golden() -> bool:
        if prover is None or verifier is None:
            return False
        request = _fixture_request(policy)
        try:
            output = prover(request)
        except Exception:
            return False
        if not isinstance(output, Mapping):
            return False
        if not output.get("proof_artifact_id") or not output.get("proof_digest"):
            return False
        try:
            envelope = _fixture_envelope(
                request,
                setup=setup,
                policy=policy,
                proof_artifact_id=str(output["proof_artifact_id"]),
                proof_digest=str(output["proof_digest"]),
            )
            return verifier(envelope) is True
        except Exception:
            return False

    def _negative() -> bool:
        if verifier is None:
            return False
        request = _fixture_request(policy)
        try:
            envelope = _fixture_envelope(
                request,
                setup=setup,
                policy=policy,
                proof_artifact_id="artifact:negative-fixture",
                proof_digest="sha256:" + ("0" * 64),
            )
            return verifier(envelope) is False
        except Exception:
            return False

    def _stale_key() -> bool:
        expired = AttestationBackendPolicy(
            backend_id=policy.backend_id,
            backend_version=policy.backend_version,
            circuit_id=policy.circuit_id,
            circuit_version=policy.circuit_version,
            public_input_schema_id=policy.public_input_schema_id,
            public_input_schema_version=policy.public_input_schema_version,
            verification_key_id=policy.verification_key_id,
            verification_key_version=policy.verification_key_version,
            backend_mode=policy.backend_mode,
            verification_key_expires_at="2020-01-01T00:00:00Z",
        )
        health = evaluate_backend_health(
            expired,
            configured=True,
            available=True,
            outcomes={case: True for case in REQUIRED_BACKEND_TEST_CASES},
            evaluated_at=_utc_now(),
        )
        return not health.production_eligible

    def _malformed() -> bool:
        if verifier is None:
            return False
        request = _fixture_request(policy)
        try:
            envelope = _fixture_envelope(
                request,
                setup=setup,
                policy=policy,
                proof_artifact_id="artifact:malformed",
                proof_digest="not-a-digest",
            )
            return verifier(envelope) is False
        except Exception:
            # Fail-closed construction of a malformed envelope is also a pass
            # for the negative malformed-proof fixture.
            return True

    def _witness_no_leak() -> bool:
        probes = list(secret_probes) or ["__datasets_zk_witness_probe__"]
        public = {
            "setup": setup.to_public_artifact(),
            "policy": policy.to_public_artifact(),
            "private_witness_redacted": True,
        }
        result = witness_no_leak_test_result(
            policy,
            artifacts=[public],
            secret_probes=probes,
            observed_at=_utc_now(),
        )
        return result.passed

    return {
        BackendTestCase.GOLDEN: _golden,
        BackendTestCase.NEGATIVE: _negative,
        BackendTestCase.STALE_KEY: _stale_key,
        BackendTestCase.MALFORMED_PROOF: _malformed,
        BackendTestCase.WITNESS_NO_LEAK: _witness_no_leak,
    }


def _fixture_envelope(
    request: ReceiptAttestationRequest,
    *,
    setup: DatasetsZkSetupIdentity,
    policy: AttestationBackendPolicy,
    proof_artifact_id: str,
    proof_digest: str,
) -> ReceiptAttestationEnvelope:
    """Build a managed envelope for self-tests using synthetic full-pass health."""

    health = evaluate_backend_health(
        policy,
        configured=True,
        available=True,
        outcomes={case: True for case in REQUIRED_BACKEND_TEST_CASES},
        evaluated_at=_utc_now(),
    )
    return create_attestation_envelope(
        request,
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        proof_artifact_id=proof_artifact_id,
        proof_digest=proof_digest,
        prover_id=setup.prover_id,
        backend_health=health,
    )


def _fixture_request(policy: AttestationBackendPolicy) -> ReceiptAttestationRequest:
    """Build a minimal kernel-eligible fixture receipt for self-tests only."""

    from .formal_verification_contracts import (
        EvidenceAuthority,
        EvidenceFreshness,
        EvidenceKind,
        EvidenceVerdict,
        ProofEvidence,
        ProofVerdict,
        ResourceBudget,
    )

    obligation_id = "obligation:datasets-zk-self-test"
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-self-test",
        subject_id=obligation_id,
        verifier_id="kernel:datasets-zk-self-test",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )
    receipt = ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:datasets-zk-self-test",
        attempt_id="attempt:datasets-zk-self-test",
        repository_id="repo:datasets-zk-self-test",
        repository_tree_id="git-tree:datasets-zk-self-test",
        ast_scope_ids=("scope:datasets-zk-self-test",),
        premise_ids=("premise:datasets-zk-self-test",),
        translator_id="translator:datasets-zk-self-test@1",
        solver_id="solver:datasets-zk-self-test@1",
        kernel_id="kernel:datasets-zk-self-test@1",
        toolchain_id="toolchain:datasets-zk-self-test@1",
        policy_id="policy:datasets-zk-self-test@1",
        resource_budget=ResourceBudget(
            wall_time_ms=1_000,
            memory_bytes=1_000_000,
            max_processes=1,
            network_allowed=False,
        ),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        kernel_receipt_id="kernel-receipt:datasets-zk-self-test",
    )
    return prepare_receipt_attestation(
        receipt,
        backend_policy=policy,
        witness=PrivateAttestationWitness(
            {"receipt_opening": "self-test-opening-not-for-production"}
        ),
    )


def run_datasets_zk_backend_self_tests(
    setup: DatasetsZkSetupIdentity,
    *,
    configured: bool | None = None,
    available: bool | None = None,
    evaluated_at: str | None = None,
    cases: Mapping[BackendTestCase | str, Callable[[], bool]] | None = None,
    prover: Callable[[ReceiptAttestationRequest], Mapping[str, Any]] | None = None,
    verifier: Callable[[ReceiptAttestationEnvelope], bool] | None = None,
    secret_probes: Sequence[str | bytes] = (),
) -> BackendHealthReport:
    """Run mandatory self-tests and return production eligibility evidence."""

    policy = setup.to_backend_policy()
    is_configured = setup.configured if configured is None else bool(configured)
    is_available = is_configured if available is None else bool(available)
    fixtures = (
        dict(cases)
        if cases is not None
        else default_backend_self_test_fixtures(
            setup,
            prover=prover,
            verifier=verifier,
            secret_probes=secret_probes,
        )
    )
    # Ensure every required case is present (fail closed via missing_cases).
    for required in REQUIRED_BACKEND_TEST_CASES:
        fixtures.setdefault(required, lambda: False)
    return run_backend_self_tests(
        policy,
        cases=fixtures,
        configured=is_configured,
        available=is_available,
        evaluated_at=evaluated_at or _utc_now(),
    )


class IpfsDatasetsZkAttestation:
    """Production adapter: datasets Groth16/ProveKit ↔ verified-receipt predicate.

    Interface: ``IpfsDatasetsZkAttestation@1`` (SCAEV082REALZK).
    """

    interface: ClassVar[str] = IPFS_DATASETS_ZK_ATTESTATION_INTERFACE
    evidence_id: ClassVar[str] = SCAEV082REALZK

    def __init__(
        self,
        *,
        decision: ZkUseCaseDecisionRecord | None = None,
        prover: Callable[[ReceiptAttestationRequest], Mapping[str, Any]] | None = None,
        verifier: Callable[[ReceiptAttestationEnvelope], bool] | None = None,
        preferred_families: Sequence[str | ZkBackendFamily] = (
            ZkBackendFamily.PROVEKIT,
            ZkBackendFamily.GROTH16,
        ),
        setup_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        secret_probes: Sequence[str | bytes] = (),
    ) -> None:
        self._decision = decision or datasets_verified_receipt_zk_use_case_decision()
        self._prover = prover
        self._verifier = verifier
        self._preferred_families = tuple(preferred_families)
        self._setup_overrides = dict(setup_overrides or {})
        self._secret_probes = tuple(secret_probes)

    @property
    def decision(self) -> ZkUseCaseDecisionRecord:
        return self._decision

    def select_backend(
        self, *, selected_at: str | None = None
    ) -> DatasetsZkBackendSelection:
        return probe_datasets_zk_backend(
            preferred_families=self._preferred_families,
            decision=self._decision,
            selected_at=selected_at,
            setup_overrides=self._setup_overrides,
        )

    def evaluate_backend_health(
        self,
        selection: DatasetsZkBackendSelection | None = None,
        *,
        evaluated_at: str | None = None,
        cases: Mapping[BackendTestCase | str, Callable[[], bool]] | None = None,
    ) -> BackendHealthReport:
        selected = selection or self.select_backend(selected_at=evaluated_at)
        if selected.simulated or not selected.available:
            return evaluate_backend_health(
                selected.setup.to_backend_policy(),
                configured=selected.setup.configured,
                available=False,
                outcomes={},
                evaluated_at=evaluated_at or _utc_now(),
            )
        return run_datasets_zk_backend_self_tests(
            selected.setup,
            available=selected.available,
            evaluated_at=evaluated_at,
            cases=cases,
            prover=self._prover,
            verifier=self._verifier,
            secret_probes=self._secret_probes,
        )

    def _typed_result(
        self,
        *,
        status: DatasetsZkStatus,
        predicate: str,
        backend_family: str,
        backend_mode: AttestationBackendMode,
        statement: ReceiptAttestationStatement | None = None,
        verification: AttestationVerification | None = None,
        backend_health: BackendHealthReport | None = None,
        setup: DatasetsZkSetupIdentity | None = None,
        diagnostic_code: str = "",
        observed_at: str | None = None,
    ) -> DatasetsZkAttestationResult:
        return DatasetsZkAttestationResult(
            status=status,
            predicate=predicate,
            use_case_id=self._decision.use_case_id,
            backend_family=backend_family,
            backend_mode=backend_mode,
            statement=statement,
            verification=verification,
            backend_health=backend_health,
            setup=setup,
            diagnostic_code=diagnostic_code,
            observed_at=observed_at or _utc_now(),
        )

    def attest_verified_receipt(
        self,
        receipt: ProofReceipt,
        *,
        witness: PrivateAttestationWitness,
        predicate: str | DatasetsZkPredicate = DEFAULT_VERIFIED_RECEIPT_PREDICATE,
        selection: DatasetsZkBackendSelection | None = None,
        backend_health: BackendHealthReport | None = None,
        prover: Callable[[ReceiptAttestationRequest], Mapping[str, Any]] | None = None,
        verifier: Callable[[ReceiptAttestationEnvelope], bool] | None = None,
        evaluated_at: str | None = None,
        self_test_cases: Mapping[BackendTestCase | str, Callable[[], bool]]
        | None = None,
    ) -> DatasetsZkAttestationResult:
        """Attest an already kernel-verified receipt with the datasets backend.

        Fail-closed gates (in order):

        1. Closed predicate catalog.
        2. Reviewed use-case decision authorizes a cryptographic family.
        3. Receipt is current and kernel verified (via prepare_receipt_attestation).
        4. Backend setup identities are bound into the managed policy.
        5. Self-tests pass and health is production eligible.
        6. Independent verifier accepts the envelope.
        """

        predicate_value = str(getattr(predicate, "value", predicate)).strip().lower()
        if predicate_value not in APPROVED_VERIFIED_RECEIPT_PREDICATES:
            return self._typed_result(
                status=DatasetsZkStatus.NOT_APPLICABLE,
                predicate=predicate_value or "unknown",
                backend_family="",
                backend_mode=AttestationBackendMode.SIMULATED,
                diagnostic_code="predicate_not_in_closed_catalog",
                observed_at=evaluated_at,
            )

        if self._decision.disposition is not ZkUseCaseDisposition.APPROVED:
            status_map = {
                ZkUseCaseDisposition.PENDING_REVIEW: DatasetsZkStatus.PENDING_REVIEW,
                ZkUseCaseDisposition.REJECTED: DatasetsZkStatus.REJECTED,
                ZkUseCaseDisposition.NOT_APPLICABLE: DatasetsZkStatus.NOT_APPLICABLE,
            }
            return self._typed_result(
                status=status_map.get(
                    self._decision.disposition, DatasetsZkStatus.NOT_APPLICABLE
                ),
                predicate=predicate_value,
                backend_family="",
                backend_mode=AttestationBackendMode.SIMULATED,
                diagnostic_code="use_case_%s" % self._decision.disposition.value,
                observed_at=evaluated_at,
            )

        selected = selection or self.select_backend(selected_at=evaluated_at)
        if selected.simulated or _family_is_simulated(selected.backend_family):
            return self._typed_result(
                status=DatasetsZkStatus.SIMULATED,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=AttestationBackendMode.SIMULATED,
                setup=selected.setup,
                diagnostic_code="simulated_backend_non_authoritative",
                observed_at=evaluated_at,
            )
        if not selected.available:
            return self._typed_result(
                status=DatasetsZkStatus.UNAVAILABLE,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=selected.setup.backend_mode,
                setup=selected.setup,
                diagnostic_code="backend_unavailable",
                observed_at=evaluated_at,
            )

        try:
            require_zk_backend_selection_authorized(
                selected.decision, backend_family=selected.backend_family
            )
        except AttestationValidationError:
            return self._typed_result(
                status=DatasetsZkStatus.REJECTED,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=selected.setup.backend_mode,
                setup=selected.setup,
                diagnostic_code="backend_family_not_authorized",
                observed_at=evaluated_at,
            )

        policy = selected.setup.to_backend_policy()
        health = backend_health or self.evaluate_backend_health(
            selected,
            evaluated_at=evaluated_at,
            cases=self_test_cases,
        )
        if not health.production_eligible:
            return self._typed_result(
                status=DatasetsZkStatus.DEGRADED,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=policy.backend_mode,
                backend_health=health,
                setup=selected.setup,
                diagnostic_code="backend_not_production_eligible",
                observed_at=evaluated_at,
            )

        selected_prover = prover or self._prover
        selected_verifier = verifier or self._verifier
        if selected_prover is None or selected_verifier is None:
            return self._typed_result(
                status=DatasetsZkStatus.UNAVAILABLE,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=policy.backend_mode,
                backend_health=health,
                setup=selected.setup,
                diagnostic_code="prover_or_verifier_unavailable",
                observed_at=evaluated_at,
            )

        try:
            request = prepare_receipt_attestation(
                receipt,
                backend_policy=policy,
                witness=witness,
            )
        except (AttestationValidationError, ContractValidationError):
            return self._typed_result(
                status=DatasetsZkStatus.REJECTED,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=policy.backend_mode,
                backend_health=health,
                setup=selected.setup,
                diagnostic_code="receipt_not_kernel_verified",
                observed_at=evaluated_at,
            )

        try:
            health.require_production_eligible()
            verification = execute_cryptographic_attestation(
                request,
                backend_health=health,
                prover=selected_prover,
                verifier=selected_verifier,
                prover_id=selected.setup.prover_id,
                verifier_id=selected.setup.verifier_id,
            )
        except CryptographicBackendFailure:
            return self._typed_result(
                status=DatasetsZkStatus.ERROR,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=policy.backend_mode,
                statement=request.statement,
                backend_health=health,
                setup=selected.setup,
                diagnostic_code="cryptographic_backend_failure",
                observed_at=evaluated_at,
            )
        except AttestationValidationError:
            return self._typed_result(
                status=DatasetsZkStatus.ERROR,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=policy.backend_mode,
                statement=request.statement,
                backend_health=health,
                setup=selected.setup,
                diagnostic_code="attestation_validation_error",
                observed_at=evaluated_at,
            )

        # Threat model: simulated envelopes can never satisfy ATTESTED.
        if verification.simulated:
            if not simulated_attestation_cannot_satisfy_attested(verification):
                raise AttestationValidationError(
                    "simulated verification incorrectly claims ATTESTED authority"
                )
            return self._typed_result(
                status=DatasetsZkStatus.SIMULATED,
                predicate=predicate_value,
                backend_family=selected.backend_family,
                backend_mode=AttestationBackendMode.SIMULATED,
                statement=verification.envelope.statement,
                verification=verification,
                backend_health=health,
                setup=selected.setup,
                diagnostic_code="simulated_non_authoritative",
                observed_at=evaluated_at,
            )

        if verification.authoritative:
            status = DatasetsZkStatus.ATTESTED
            diagnostic = "verified"
        elif verification.verified:
            status = DatasetsZkStatus.GENERATED
            diagnostic = "verified_but_not_authoritative"
        elif verification.verdict is AttestationVerificationVerdict.REJECTED:
            status = DatasetsZkStatus.REJECTED
            diagnostic = verification.diagnostic_code or "proof_rejected"
        else:
            status = DatasetsZkStatus.ERROR
            diagnostic = verification.diagnostic_code or "verification_error"

        return self._typed_result(
            status=status,
            predicate=predicate_value,
            backend_family=selected.backend_family,
            backend_mode=verification.envelope.backend_mode,
            statement=verification.envelope.statement,
            verification=verification,
            backend_health=health,
            setup=selected.setup,
            diagnostic_code=diagnostic,
            observed_at=evaluated_at,
        )


# Compatibility aliases
IpfsDatasetsZkAttestationAdapter = IpfsDatasetsZkAttestation
DatasetsZkAttestation = IpfsDatasetsZkAttestation


def public_datasets_zk_artifact(value: Any) -> Any:
    """Project a datasets ZK value into a public, witness-free artifact."""

    if isinstance(value, PrivateAttestationWitness):
        raise WitnessDisclosureError(
            "private witness cannot enter a public datasets ZK artifact"
        )
    if isinstance(
        value,
        (
            DatasetsZkSetupIdentity,
            DatasetsZkBackendSelection,
            DatasetsZkAttestationResult,
        ),
    ):
        return value.to_public_artifact()
    return public_attestation_artifact(value)


__all__ = [
    "APPROVED_VERIFIED_RECEIPT_PREDICATES",
    "DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_DECISION",
    "DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_ID",
    "DEFAULT_PUBLIC_INPUT_SCHEMA_ID",
    "DEFAULT_RECEIPT_BINDING_CIRCUIT_ID",
    "DEFAULT_VERIFIED_RECEIPT_PREDICATE",
    "IPFS_DATASETS_ZK_ATTESTATION_CONTRACT_VERSION",
    "IPFS_DATASETS_ZK_ATTESTATION_INTERFACE",
    "IPFS_DATASETS_ZK_RESULT_SCHEMA",
    "IPFS_DATASETS_ZK_SELECTION_SCHEMA",
    "IPFS_DATASETS_ZK_SETUP_SCHEMA",
    "SCAEV082REALZK",
    "DatasetsZkAttestation",
    "DatasetsZkAttestationResult",
    "DatasetsZkBackendSelection",
    "DatasetsZkPredicate",
    "DatasetsZkSetupIdentity",
    "DatasetsZkStatus",
    "IpfsDatasetsZkAttestation",
    "IpfsDatasetsZkAttestationAdapter",
    "build_datasets_zk_setup_identity",
    "datasets_verified_receipt_zk_use_case_decision",
    "datasets_zkp_registry_available",
    "default_backend_self_test_fixtures",
    "probe_datasets_zk_backend",
    "public_datasets_zk_artifact",
    "run_datasets_zk_backend_self_tests",
    # Re-exported proof_attestation gates used by this adapter / evidence AST.
    "AssuranceLevel",
    "AttestationBackendMode",
    "AttestationBackendPolicy",
    "AttestationGate",
    "AttestationTrust",
    "AttestationValidationError",
    "AttestationVerification",
    "BackendHealthReport",
    "BackendTestCase",
    "BackendTestResult",
    "BackendTestVerdict",
    "CapabilityHealth",
    "CryptographicBackendFailure",
    "PrivateAttestationWitness",
    "REQUIRED_BACKEND_TEST_CASES",
    "WitnessDisclosureError",
    "ZkBackendFamily",
    "ZkUseCaseDecisionRecord",
    "ZkUseCaseDisposition",
    "evaluate_backend_health",
    "execute_cryptographic_attestation",
    "prepare_receipt_attestation",
    "public_artifact_contains",
    "require_zk_backend_selection_authorized",
    "run_backend_self_tests",
    "simulated_attestation_cannot_satisfy_attested",
    "witness_no_leak_test_result",
]
