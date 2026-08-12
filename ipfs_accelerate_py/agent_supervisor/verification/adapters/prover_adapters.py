"""Z3 and existing proof-assistant verification adapters (IVP-007).

These adapters are narrow wrappers over the admitted
:class:`~ipfs_accelerate_py.agent_supervisor.verification.process_runner.VerificationProcessRunner`
and the existing formal-proof / kernel-verification contracts.

Authority rules
---------------
* Z3 ``sat`` / ``unsat`` / ``unknown`` mappings bind the exact normalized or
  negated obligation, translator version, solver executable/version, and the
  existing assurance lattice.  Bare solver stdout text alone never proves.
* Absent Z3 and Z3 timeouts never project ``proved`` (or ``disproved``).
* The proof-assistant route selects only an offline, bounded,
  registry-admitted kernel probe (Lean / Coq-Rocq / Isabelle).  Anything else
  is typed ``unavailable``.
* ``sorry`` / ``admit`` / ``unsafe`` / model-generated drafts cannot prove.
* A wrapper ``proved`` / ``disproved`` status derives only from existing
  authoritative proof evidence or current direct execution at an independent
  boundary.
* Cancellation terminates descendants through the shared runner and fences
  late publication.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    AttemptStatus,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofAttempt,
    ProofEvidence,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ProofReceipt as FormalProofReceipt,
)
from ipfs_accelerate_py.agent_supervisor.proof.kernel_verification import (
    KernelFailureCode,
    KernelTarget,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    DirectExecutionObservation,
    ProofReceipt,
    TerminalStatus,
    VerificationContractError,
    VerificationIdentityError,
    VerificationReceiptKey,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    PROCESS_RUNNER_EVIDENCE,
    PROCESS_TREE_CANCELLATION_EVIDENCE,
    VerificationCancellation,
    VerificationCommand,
    VerificationProcessRunner,
    VerificationProcessRunnerError,
    VerificationRunDisposition,
    VerificationRunResult,
    VerificationSandboxIdentity,
    build_hermetic_environment,
)

# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

Z3_VERIFICATION_ADAPTER_INTERFACE: Final[str] = "Z3VerificationAdapter@1"
Z3_VERIFICATION_ADAPTER_SCHEMA: Final[str] = "z3-verification-adapter@1"
PROOF_ASSISTANT_ADAPTER_INTERFACE: Final[str] = "ExistingProofAssistantAdapter@1"
PROOF_ASSISTANT_ADAPTER_SCHEMA: Final[str] = (
    "existing-proof-assistant-adapter@1"
)
PROVER_ADAPTER_EVIDENCE: Final[str] = "ivp/prover-adapter@1"
Z3_SOLVER_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/z3-solver-report@1"
)
KERNEL_PROBE_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/kernel-probe-report@1"
)

_MAX_REASON_CODES: Final[int] = 64
_MAX_SMT_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_SOURCE_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_ARGV_ITEMS: Final[int] = 256

_REGISTRY_KERNEL_PROVERS: Final[frozenset[str]] = frozenset(
    {
        "lean",
        "coq",
        "rocq",
        "isabelle",
    }
)
_KERNEL_CAPABILITY_BY_PROVER: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lean": "lean_kernel_check",
        "coq": "coq_kernel_check",
        "rocq": "coq_kernel_check",
        "isabelle": "isabelle_kernel_check",
    }
)

_INCOMPLETE_PROOF_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?<![A-Za-z0-9_'])(sorry|admit|sorryAx|undefined|axiom\s+sorry)(?![A-Za-z0-9_'])"
)
_UNSAFE_PROOF_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?<![A-Za-z0-9_'])(unsafe|trusted|axiom|Admitted|admit\.)(?![A-Za-z0-9_'])"
)
_Z3_OUTCOME_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*(sat|unsat|unknown)\s*$"
)

_REASON_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_.:/+-]{0,127}$"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProverVerificationAdapterError(ValueError):
    """Fail-closed adapter contract violation (pre-execution)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_request",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class Z3SolverOutcome(str, Enum):
    """Closed solver outcomes observed from a structured Z3 check."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    MALFORMED = "malformed"
    ABSENT = "absent"


class ProofAuthoritySource(str, Enum):
    """Where a conclusive wrapper status is allowed to originate."""

    EXISTING_AUTHORITATIVE_EVIDENCE = "existing_authoritative_evidence"
    CURRENT_DIRECT_EXECUTION = "current_direct_execution"
    NONE = "none"


class KernelProbeTarget(str, Enum):
    """Registry-admitted interactive theorem prover kernels."""

    LEAN = "lean"
    COQ = "coq"
    ROCQ = "rocq"
    ISABELLE = "isabelle"

    @property
    def canonical_prover_id(self) -> str:
        if self is KernelProbeTarget.ROCQ:
            return "coq"
        return self.value

    @property
    def kernel_target(self) -> KernelTarget:
        if self in {KernelProbeTarget.COQ, KernelProbeTarget.ROCQ}:
            return KernelTarget.COQ
        if self is KernelProbeTarget.ISABELLE:
            return KernelTarget.ISABELLE
        return KernelTarget.LEAN

    @property
    def authority_capability(self) -> str:
        return _KERNEL_CAPABILITY_BY_PROVER[self.canonical_prover_id]


# ---------------------------------------------------------------------------
# Shared request / result shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryKernelAdmission:
    """Offline bounded registry admission for one kernel probe.

    The adapter does not invent provers.  Callers must present an already
    registry-admitted offline kernel probe identity.  Discovery alone is not
    admission.
    """

    prover_id: str
    offline: bool
    admitted: bool
    authority_capability: str
    smoke_tested: bool = False
    versioned: bool = False
    executable_path: str = ""
    executable_version: str = ""
    fixture_id: str = ""
    registry_entry_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prover = str(self.prover_id or "").strip().lower()
        if prover not in _REGISTRY_KERNEL_PROVERS:
            raise ProverVerificationAdapterError(
                "registry kernel admission must name lean, coq, rocq, or isabelle",
                reason_code="kernel_not_registry_admitted",
                details={"prover_id": prover},
            )
        object.__setattr__(self, "prover_id", prover)
        if not isinstance(self.offline, bool) or not self.offline:
            raise ProverVerificationAdapterError(
                "kernel probe must be offline",
                reason_code="kernel_not_offline",
            )
        if not isinstance(self.admitted, bool) or not self.admitted:
            raise ProverVerificationAdapterError(
                "kernel probe must be registry-admitted",
                reason_code="kernel_not_registry_admitted",
            )
        capability = str(self.authority_capability or "").strip()
        expected = _KERNEL_CAPABILITY_BY_PROVER[prover]
        if capability != expected:
            raise ProverVerificationAdapterError(
                "kernel authority capability does not match registry prover",
                reason_code="kernel_capability_mismatch",
                details={"expected": expected, "observed": capability},
            )
        object.__setattr__(self, "authority_capability", capability)
        for name in (
            "smoke_tested",
            "versioned",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProverVerificationAdapterError(
                    f"{name} must be a boolean",
                    reason_code="invalid_registry_admission",
                )
        for name in (
            "executable_path",
            "executable_version",
            "fixture_id",
            "registry_entry_id",
        ):
            object.__setattr__(
                self, name, str(getattr(self, name) or "").strip()
            )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in dict(self.metadata or {}).items()
                }
            ),
        )

    @property
    def operational(self) -> bool:
        return (
            self.offline
            and self.admitted
            and self.smoke_tested
            and bool(self.executable_path)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prover_id": self.prover_id,
            "offline": self.offline,
            "admitted": self.admitted,
            "authority_capability": self.authority_capability,
            "smoke_tested": self.smoke_tested,
            "versioned": self.versioned,
            "executable_path": self.executable_path,
            "executable_version": self.executable_version,
            "fixture_id": self.fixture_id,
            "registry_entry_id": self.registry_entry_id,
            "operational": self.operational,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Z3VerificationRequest:
    """One admitted Z3 verification request.

    *receipt_key* must already bind the proof obligation, translator, solver,
    and tool identities that this execution will observe.
    """

    receipt_key: VerificationReceiptKey
    z3_executable: str
    sandbox: VerificationSandboxIdentity
    cwd: str
    timeout_seconds: float
    # Exact SMT-LIB text of the normalized/negated obligation under check.
    smtlib_payload: str = ""
    smtlib_relpath: str = "obligation.smt2"
    # Translator identity (must match receipt_key.proof_backend_binding).
    translator_id: str = ""
    translator_version: str = ""
    # Optional explicit CodeProofObligation; defaults to receipt-key identity.
    proof_obligation: CodeProofObligation | None = None
    resource_budget: ResourceBudget | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    extra_z3_args: Sequence[str] = ()
    network_policy: str = NETWORK_POLICY_DENY_ALL
    max_stdout_bytes: int = 256 * 1024
    max_stderr_bytes: int = 256 * 1024
    lane_id: str = ""
    resource_class: str = "cpu-proof"
    stage: str = "validation"
    metadata: Mapping[str, str] = field(default_factory=dict)
    simulated: bool = False
    # Existing authoritative formal evidence (wrapper path without re-solve).
    existing_formal_proof_receipt: FormalProofReceipt | None = None
    existing_proof_attempt: ProofAttempt | None = None
    # Test / deterministic injection: structured outcome without invoking z3.
    injected_solver_outcome: Z3SolverOutcome | str | None = None
    injected_stdout: str = ""
    injected_stderr: str = ""
    injected_exit_code: int | None = 0
    counterexample_verified: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_key, VerificationReceiptKey):
            raise ProverVerificationAdapterError(
                "receipt_key must be a VerificationReceiptKey",
                reason_code="invalid_receipt_key",
            )
        if self.receipt_key.receipt_kind is not VerificationReceiptKind.PROOF:
            raise ProverVerificationAdapterError(
                "receipt_key must be receipt_kind=proof",
                reason_code="invalid_receipt_kind",
            )
        if self.receipt_key.tool_name != "z3":
            raise ProverVerificationAdapterError(
                "receipt_key.tool_name must be z3",
                reason_code="invalid_tool",
            )
        if self.receipt_key.adapter_schema != Z3_VERIFICATION_ADAPTER_SCHEMA:
            raise ProverVerificationAdapterError(
                "receipt_key.adapter_schema must be z3-verification-adapter@1",
                reason_code="invalid_adapter_schema",
            )
        if self.receipt_key.proof_backend_binding is None:
            raise ProverVerificationAdapterError(
                "receipt_key requires a proof backend binding",
                reason_code="missing_proof_backend",
            )
        executable = str(self.z3_executable or "").strip()
        if not executable:
            raise ProverVerificationAdapterError(
                "z3_executable is required",
                reason_code="invalid_executable",
            )
        if not (
            PurePosixPath(executable).is_absolute()
            or Path(executable).expanduser().is_absolute()
        ):
            raise ProverVerificationAdapterError(
                "z3_executable must be absolute",
                reason_code="invalid_executable",
            )
        object.__setattr__(self, "z3_executable", executable)
        if not isinstance(self.sandbox, VerificationSandboxIdentity):
            raise ProverVerificationAdapterError(
                "sandbox must be a VerificationSandboxIdentity",
                reason_code="sandbox_unavailable",
            )
        cwd = str(self.cwd or "").strip()
        if not cwd:
            raise ProverVerificationAdapterError(
                "cwd is required",
                reason_code="invalid_cwd",
            )
        object.__setattr__(self, "cwd", cwd)
        timeout = float(self.timeout_seconds)
        if not (timeout > 0.0):
            raise ProverVerificationAdapterError(
                "timeout_seconds must be positive",
                reason_code="invalid_timeout",
            )
        object.__setattr__(self, "timeout_seconds", timeout)
        smt = self.smtlib_payload if isinstance(self.smtlib_payload, str) else ""
        if len(smt.encode("utf-8")) > _MAX_SMT_BYTES:
            raise ProverVerificationAdapterError(
                "smtlib_payload exceeds bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "smtlib_payload", smt)
        rel = str(self.smtlib_relpath or "obligation.smt2").strip()
        if (
            not rel
            or rel.startswith("/")
            or ".." in PurePosixPath(rel).parts
            or "\x00" in rel
        ):
            raise ProverVerificationAdapterError(
                "smtlib_relpath must be a sandbox-relative path",
                reason_code="invalid_smt_path",
            )
        object.__setattr__(self, "smtlib_relpath", rel)
        object.__setattr__(
            self, "translator_id", str(self.translator_id or "").strip()
        )
        object.__setattr__(
            self,
            "translator_version",
            str(self.translator_version or "").strip(),
        )
        if self.proof_obligation is not None and not isinstance(
            self.proof_obligation, CodeProofObligation
        ):
            raise ProverVerificationAdapterError(
                "proof_obligation must be a CodeProofObligation",
                reason_code="invalid_obligation",
            )
        if self.resource_budget is not None and not isinstance(
            self.resource_budget, ResourceBudget
        ):
            raise ProverVerificationAdapterError(
                "resource_budget must be a ResourceBudget",
                reason_code="invalid_budget",
            )
        env = {
            str(key): str(value)
            for key, value in dict(self.environment or {}).items()
        }
        object.__setattr__(self, "environment", MappingProxyType(env))
        extra = _normalize_args(self.extra_z3_args, field_name="extra_z3_args")
        object.__setattr__(self, "extra_z3_args", extra)
        network = str(self.network_policy or "").strip() or NETWORK_POLICY_DENY_ALL
        if network != NETWORK_POLICY_DENY_ALL:
            raise ProverVerificationAdapterError(
                "network policy must be deny_all",
                reason_code="network_policy_denied",
            )
        object.__setattr__(self, "network_policy", network)
        object.__setattr__(self, "simulated", bool(self.simulated))
        # Attempt is optional when formal is present; formal is required when
        # an attempt is supplied.
        if (
            self.existing_proof_attempt is not None
            and self.existing_formal_proof_receipt is None
        ):
            raise ProverVerificationAdapterError(
                "existing proof attempt requires an existing formal proof receipt",
                reason_code="invalid_existing_evidence",
            )
        if self.injected_solver_outcome is not None:
            outcome = (
                self.injected_solver_outcome
                if isinstance(self.injected_solver_outcome, Z3SolverOutcome)
                else Z3SolverOutcome(str(self.injected_solver_outcome).strip().lower())
            )
            object.__setattr__(self, "injected_solver_outcome", outcome)
        object.__setattr__(
            self, "injected_stdout", str(self.injected_stdout or "")
        )
        object.__setattr__(
            self, "injected_stderr", str(self.injected_stderr or "")
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in dict(self.metadata or {}).items()
                }
            ),
        )


@dataclass(frozen=True)
class Z3VerificationResult:
    """Observed Z3 verification outcome with retained argv and artifacts."""

    terminal_status: TerminalStatus
    receipt: ProofReceipt | None
    command_argv: tuple[str, ...]
    solver_outcome: Z3SolverOutcome
    authority_source: ProofAuthoritySource
    artifact_cids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    production_admissible: bool
    simulated: bool
    run_result: VerificationRunResult | None
    formal_proof_receipt: FormalProofReceipt | None
    proof_attempt: ProofAttempt | None
    solver_report_cid: str
    evidence: tuple[str, ...] = (
        PROVER_ADAPTER_EVIDENCE,
        PROCESS_RUNNER_EVIDENCE,
        PROCESS_TREE_CANCELLATION_EVIDENCE,
    )
    duration_ms: int = 0
    exit_code: int | None = None
    publication_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "terminal_status", TerminalStatus(self.terminal_status)
        )
        object.__setattr__(
            self,
            "solver_outcome",
            self.solver_outcome
            if isinstance(self.solver_outcome, Z3SolverOutcome)
            else Z3SolverOutcome(str(self.solver_outcome)),
        )
        object.__setattr__(
            self,
            "authority_source",
            self.authority_source
            if isinstance(self.authority_source, ProofAuthoritySource)
            else ProofAuthoritySource(str(self.authority_source)),
        )
        object.__setattr__(
            self, "command_argv", tuple(str(item) for item in self.command_argv)
        )
        object.__setattr__(
            self,
            "artifact_cids",
            tuple(str(item) for item in self.artifact_cids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )
        object.__setattr__(
            self, "production_admissible", bool(self.production_admissible)
        )
        object.__setattr__(self, "simulated", bool(self.simulated))
        object.__setattr__(
            self, "publication_allowed", bool(self.publication_allowed)
        )
        object.__setattr__(
            self, "evidence", tuple(str(item) for item in self.evidence)
        )

    @property
    def ok(self) -> bool:
        return (
            self.production_admissible
            and self.terminal_status is TerminalStatus.PROVED
            and self.receipt is not None
            and self.receipt.terminal_success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": Z3_VERIFICATION_ADAPTER_SCHEMA,
            "interface": Z3_VERIFICATION_ADAPTER_INTERFACE,
            "evidence": list(self.evidence),
            "terminal_status": self.terminal_status.value,
            "receipt": self.receipt.to_record() if self.receipt is not None else None,
            "command_argv": list(self.command_argv),
            "solver_outcome": self.solver_outcome.value,
            "authority_source": self.authority_source.value,
            "artifact_cids": list(self.artifact_cids),
            "reason_codes": list(self.reason_codes),
            "production_admissible": self.production_admissible,
            "simulated": self.simulated,
            "solver_report_cid": self.solver_report_cid,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "publication_allowed": self.publication_allowed,
            "ok": self.ok,
            "formal_proof_receipt": (
                self.formal_proof_receipt.to_record()
                if self.formal_proof_receipt is not None
                else None
            ),
            "proof_attempt": (
                self.proof_attempt.to_record()
                if self.proof_attempt is not None
                else None
            ),
            "run_result": self.run_result.to_dict() if self.run_result else None,
        }


@dataclass(frozen=True)
class ProofAssistantVerificationRequest:
    """One existing proof-assistant verification request.

    Only offline, registry-admitted kernel probes may execute.  Model drafts
    and incomplete proofs are retained for audit but cannot prove.
    """

    receipt_key: VerificationReceiptKey
    sandbox: VerificationSandboxIdentity
    cwd: str
    timeout_seconds: float
    registry_admission: RegistryKernelAdmission | None = None
    kernel_executable: str = ""
    checked_source: str = ""
    source_relpath: str = "kernel_probe.lean"
    model_generated_draft: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)
    extra_kernel_args: Sequence[str] = ()
    resource_budget: ResourceBudget | None = None
    network_policy: str = NETWORK_POLICY_DENY_ALL
    max_stdout_bytes: int = 256 * 1024
    max_stderr_bytes: int = 256 * 1024
    lane_id: str = ""
    resource_class: str = "cpu-proof"
    stage: str = "validation"
    metadata: Mapping[str, str] = field(default_factory=dict)
    simulated: bool = False
    existing_formal_proof_receipt: FormalProofReceipt | None = None
    existing_proof_attempt: ProofAttempt | None = None
    # Deterministic injection for offline tests.
    injected_kernel_accepted: bool | None = None
    injected_stdout: str = ""
    injected_stderr: str = ""
    injected_exit_code: int | None = 0
    injected_failure_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_key, VerificationReceiptKey):
            raise ProverVerificationAdapterError(
                "receipt_key must be a VerificationReceiptKey",
                reason_code="invalid_receipt_key",
            )
        if self.receipt_key.receipt_kind is not VerificationReceiptKind.PROOF:
            raise ProverVerificationAdapterError(
                "receipt_key must be receipt_kind=proof",
                reason_code="invalid_receipt_kind",
            )
        if (
            self.receipt_key.adapter_schema
            != PROOF_ASSISTANT_ADAPTER_SCHEMA
        ):
            raise ProverVerificationAdapterError(
                "receipt_key.adapter_schema must be "
                "existing-proof-assistant-adapter@1",
                reason_code="invalid_adapter_schema",
            )
        if self.receipt_key.proof_backend_binding is None:
            raise ProverVerificationAdapterError(
                "receipt_key requires a proof backend binding",
                reason_code="missing_proof_backend",
            )
        if not isinstance(self.sandbox, VerificationSandboxIdentity):
            raise ProverVerificationAdapterError(
                "sandbox must be a VerificationSandboxIdentity",
                reason_code="sandbox_unavailable",
            )
        cwd = str(self.cwd or "").strip()
        if not cwd:
            raise ProverVerificationAdapterError(
                "cwd is required",
                reason_code="invalid_cwd",
            )
        object.__setattr__(self, "cwd", cwd)
        timeout = float(self.timeout_seconds)
        if not (timeout > 0.0):
            raise ProverVerificationAdapterError(
                "timeout_seconds must be positive",
                reason_code="invalid_timeout",
            )
        object.__setattr__(self, "timeout_seconds", timeout)
        if self.registry_admission is not None and not isinstance(
            self.registry_admission, RegistryKernelAdmission
        ):
            raise ProverVerificationAdapterError(
                "registry_admission must be a RegistryKernelAdmission",
                reason_code="invalid_registry_admission",
            )
        executable = str(self.kernel_executable or "").strip()
        if executable and not (
            PurePosixPath(executable).is_absolute()
            or Path(executable).expanduser().is_absolute()
        ):
            raise ProverVerificationAdapterError(
                "kernel_executable must be absolute when provided",
                reason_code="invalid_executable",
            )
        object.__setattr__(self, "kernel_executable", executable)
        source = self.checked_source if isinstance(self.checked_source, str) else ""
        if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            raise ProverVerificationAdapterError(
                "checked_source exceeds bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "checked_source", source)
        rel = str(self.source_relpath or "kernel_probe.lean").strip()
        if (
            not rel
            or rel.startswith("/")
            or ".." in PurePosixPath(rel).parts
            or "\x00" in rel
        ):
            raise ProverVerificationAdapterError(
                "source_relpath must be a sandbox-relative path",
                reason_code="invalid_source_path",
            )
        object.__setattr__(self, "source_relpath", rel)
        object.__setattr__(
            self, "model_generated_draft", bool(self.model_generated_draft)
        )
        env = {
            str(key): str(value)
            for key, value in dict(self.environment or {}).items()
        }
        object.__setattr__(self, "environment", MappingProxyType(env))
        extra = _normalize_args(
            self.extra_kernel_args, field_name="extra_kernel_args"
        )
        object.__setattr__(self, "extra_kernel_args", extra)
        network = str(self.network_policy or "").strip() or NETWORK_POLICY_DENY_ALL
        if network != NETWORK_POLICY_DENY_ALL:
            raise ProverVerificationAdapterError(
                "network policy must be deny_all",
                reason_code="network_policy_denied",
            )
        object.__setattr__(self, "network_policy", network)
        object.__setattr__(self, "simulated", bool(self.simulated))
        if (
            self.existing_proof_attempt is not None
            and self.existing_formal_proof_receipt is None
        ):
            raise ProverVerificationAdapterError(
                "existing proof attempt requires an existing formal proof receipt",
                reason_code="invalid_existing_evidence",
            )
        object.__setattr__(
            self, "injected_stdout", str(self.injected_stdout or "")
        )
        object.__setattr__(
            self, "injected_stderr", str(self.injected_stderr or "")
        )
        object.__setattr__(
            self,
            "injected_failure_code",
            str(self.injected_failure_code or "").strip(),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in dict(self.metadata or {}).items()
                }
            ),
        )


@dataclass(frozen=True)
class ProofAssistantVerificationResult:
    """Observed proof-assistant verification outcome."""

    terminal_status: TerminalStatus
    receipt: ProofReceipt | None
    command_argv: tuple[str, ...]
    authority_source: ProofAuthoritySource
    registry_admission: RegistryKernelAdmission | None
    artifact_cids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    production_admissible: bool
    simulated: bool
    run_result: VerificationRunResult | None
    formal_proof_receipt: FormalProofReceipt | None
    proof_attempt: ProofAttempt | None
    kernel_report_cid: str
    evidence: tuple[str, ...] = (
        PROVER_ADAPTER_EVIDENCE,
        PROCESS_RUNNER_EVIDENCE,
        PROCESS_TREE_CANCELLATION_EVIDENCE,
    )
    duration_ms: int = 0
    exit_code: int | None = None
    publication_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "terminal_status", TerminalStatus(self.terminal_status)
        )
        object.__setattr__(
            self,
            "authority_source",
            self.authority_source
            if isinstance(self.authority_source, ProofAuthoritySource)
            else ProofAuthoritySource(str(self.authority_source)),
        )
        object.__setattr__(
            self, "command_argv", tuple(str(item) for item in self.command_argv)
        )
        object.__setattr__(
            self,
            "artifact_cids",
            tuple(str(item) for item in self.artifact_cids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )
        object.__setattr__(
            self, "production_admissible", bool(self.production_admissible)
        )
        object.__setattr__(self, "simulated", bool(self.simulated))
        object.__setattr__(
            self, "publication_allowed", bool(self.publication_allowed)
        )
        object.__setattr__(
            self, "evidence", tuple(str(item) for item in self.evidence)
        )

    @property
    def ok(self) -> bool:
        return (
            self.production_admissible
            and self.terminal_status is TerminalStatus.PROVED
            and self.receipt is not None
            and self.receipt.terminal_success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_ASSISTANT_ADAPTER_SCHEMA,
            "interface": PROOF_ASSISTANT_ADAPTER_INTERFACE,
            "evidence": list(self.evidence),
            "terminal_status": self.terminal_status.value,
            "receipt": self.receipt.to_record() if self.receipt is not None else None,
            "command_argv": list(self.command_argv),
            "authority_source": self.authority_source.value,
            "registry_admission": (
                self.registry_admission.to_dict()
                if self.registry_admission is not None
                else None
            ),
            "artifact_cids": list(self.artifact_cids),
            "reason_codes": list(self.reason_codes),
            "production_admissible": self.production_admissible,
            "simulated": self.simulated,
            "kernel_report_cid": self.kernel_report_cid,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "publication_allowed": self.publication_allowed,
            "ok": self.ok,
            "formal_proof_receipt": (
                self.formal_proof_receipt.to_record()
                if self.formal_proof_receipt is not None
                else None
            ),
            "proof_attempt": (
                self.proof_attempt.to_record()
                if self.proof_attempt is not None
                else None
            ),
            "run_result": self.run_result.to_dict() if self.run_result else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_args(values: Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProverVerificationAdapterError(
            f"{field_name} must be a sequence of strings",
            reason_code="invalid_args",
        )
    if len(values) > _MAX_ARGV_ITEMS:
        raise ProverVerificationAdapterError(
            f"{field_name} exceeds {_MAX_ARGV_ITEMS} items",
            reason_code="bounds_exceeded",
        )
    ordered: list[str] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise ProverVerificationAdapterError(
                f"{field_name}[{index}] must be a string",
                reason_code="invalid_args",
            )
        item = raw.strip()
        if not item or "\x00" in item:
            raise ProverVerificationAdapterError(
                f"{field_name}[{index}] is empty or contains NUL",
                reason_code="invalid_args",
            )
        ordered.append(item)
    return tuple(ordered)


def _sanitize_reason_token(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if _REASON_TOKEN_RE.fullmatch(text) and len(text) <= 128:
        return text
    cleaned = re.sub(r"[^a-z0-9_.:/+-]+", "_", text)
    cleaned = cleaned.strip("._:-/+") or "reason"
    if not cleaned[0].isalpha():
        cleaned = "r_" + cleaned
    return cleaned[:128]


def _unique_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in reasons:
        text = _sanitize_reason_token(str(raw or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= _MAX_REASON_CODES:
            break
    return tuple(ordered)


def _default_budget(timeout_seconds: float) -> ResourceBudget:
    wall = max(1, int(float(timeout_seconds) * 1000))
    return ResourceBudget(
        wall_time_ms=wall,
        cpu_time_ms=wall,
        memory_bytes=512 * 1024 * 1024,
        disk_bytes=64 * 1024 * 1024,
        max_processes=4,
        max_premises=64,
        max_output_bytes=1_000_000,
        model_token_limit=0,
        provider_quota=1,
        network_allowed=False,
    )


def _artifact_label(label: str) -> str:
    return content_identity({"artifact": label, "schema": "prover-adapter-artifact@1"})


def parse_z3_solver_outcome(stdout: str, stderr: str = "") -> Z3SolverOutcome:
    """Parse a closed sat/unsat/unknown token from Z3 stdout.

    Only a whole-line token is accepted.  Free-form prose containing the words
    sat/unsat is treated as malformed so bare text cannot manufacture proof.
    """

    text = str(stdout or "")
    matches = _Z3_OUTCOME_RE.findall(text)
    if len(matches) == 1:
        token = matches[0].strip().lower()
        if token == "sat":
            return Z3SolverOutcome.SAT
        if token == "unsat":
            return Z3SolverOutcome.UNSAT
        if token == "unknown":
            return Z3SolverOutcome.UNKNOWN
    # stderr-only noise is never authoritative.
    if stderr and not matches:
        return Z3SolverOutcome.MALFORMED
    if not text.strip():
        return Z3SolverOutcome.MALFORMED
    return Z3SolverOutcome.MALFORMED


def source_contains_incomplete_or_unsafe_proof(source: str) -> tuple[bool, str]:
    """Return whether source text contains sorry/admit/unsafe escapes."""

    text = str(source or "")
    if not text.strip():
        return True, "empty_source"
    if _INCOMPLETE_PROOF_RE.search(text):
        return True, "incomplete_proof_sorry_or_admit"
    if _UNSAFE_PROOF_RE.search(text):
        return True, "unsafe_or_axiom_escape"
    # Lean admission helper is the strongest local check for Lean drafts.
    # Use a minimal template with one hole; if the source itself is a draft
    # containing sorry, admit_lean_proof_text rejects incomplete proofs.
    if "sorry" in text.lower() or "admit" in text.lower():
        return True, "incomplete_proof_sorry_or_admit"
    return False, ""


def build_z3_argv(
    *,
    z3_executable: str,
    smtlib_relpath: str,
    extra_z3_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return reproducible explicit Z3 argv (no shell)."""

    executable = str(z3_executable or "").strip()
    if not executable:
        raise ProverVerificationAdapterError(
            "z3_executable is required",
            reason_code="invalid_executable",
        )
    path = str(smtlib_relpath or "").strip()
    if not path:
        raise ProverVerificationAdapterError(
            "smtlib_relpath is required",
            reason_code="invalid_smt_path",
        )
    extra = _normalize_args(extra_z3_args, field_name="extra_z3_args")
    return (executable, "-smt2", *extra, path)


def _bindings_match_request(
    request: Z3VerificationRequest | ProofAssistantVerificationRequest,
) -> tuple[bool, tuple[str, ...]]:
    backend = request.receipt_key.proof_backend_binding
    assert backend is not None
    reasons: list[str] = []
    if isinstance(request, Z3VerificationRequest):
        if request.translator_id and request.translator_id != backend["translator_id"]:
            reasons.append("translator_id_mismatch")
        # Translator version is encoded into translator_id or tool_version.
        if request.translator_version:
            # Accept either explicit tool_version match or translator_id suffix.
            tool_version = str(request.receipt_key.tool_version)
            if (
                request.translator_version != tool_version
                and request.translator_version
                not in str(backend.get("translator_id") or "")
            ):
                reasons.append("translator_version_mismatch")
        if backend.get("tool_name") != request.receipt_key.tool_name:
            reasons.append("tool_name_mismatch")
        if backend.get("tool_version") != request.receipt_key.tool_version:
            reasons.append("tool_version_mismatch")
        if request.proof_obligation is not None:
            if (
                request.proof_obligation.obligation_id
                != request.receipt_key.proof_obligation_cid
            ):
                reasons.append("obligation_id_mismatch")
            if tuple(request.proof_obligation.ast_scope_ids) != tuple(
                backend["ast_scope_ids"]
            ):
                reasons.append("ast_scope_mismatch")
    return (not reasons), tuple(reasons)


def _build_solver_evidence(
    *,
    key: VerificationReceiptKey,
    outcome: Z3SolverOutcome,
    artifact_id: str,
    counterexample_verified: bool,
) -> ProofEvidence:
    backend = key.proof_backend_binding
    assert backend is not None
    if outcome is Z3SolverOutcome.UNSAT:
        verdict = EvidenceVerdict.ACCEPTED
        meta = {
            "solver_outcome": outcome.value,
            "counterexample_verified": False,
            "negated_obligation": True,
        }
    elif outcome is Z3SolverOutcome.SAT:
        verdict = EvidenceVerdict.REJECTED
        meta = {
            "solver_outcome": outcome.value,
            "counterexample_verified": bool(counterexample_verified),
            "negated_obligation": True,
        }
    else:
        verdict = EvidenceVerdict.INCONCLUSIVE
        meta = {
            "solver_outcome": outcome.value,
            "counterexample_verified": False,
            "negated_obligation": True,
        }
    return ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=verdict,
        artifact_id=artifact_id,
        subject_id=key.proof_obligation_cid,
        verifier_id=str(backend["solver_id"]),
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
        metadata=meta,
    )


def _build_kernel_evidence(
    *,
    key: VerificationReceiptKey,
    accepted: bool,
    artifact_id: str,
    failure_code: str = "",
) -> ProofEvidence:
    backend = key.proof_backend_binding
    assert backend is not None
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=(
            EvidenceVerdict.ACCEPTED if accepted else EvidenceVerdict.REJECTED
        ),
        artifact_id=artifact_id,
        subject_id=key.proof_obligation_cid,
        verifier_id=str(backend["kernel_id"]),
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
        metadata=(
            {"failure_code": failure_code}
            if (not accepted and failure_code)
            else {}
        ),
    )


def _build_formal_bundle(
    *,
    key: VerificationReceiptKey,
    evidence: tuple[ProofEvidence, ...],
    provider_verdict: ProofVerdict,
    budget: ResourceBudget,
    provider_claimed_assurance: AssuranceLevel = AssuranceLevel.ATTESTED,
) -> tuple[ProofAttempt, FormalProofReceipt]:
    backend = key.proof_backend_binding
    assert backend is not None
    # Attempt stage must match the pre-execution backend binding exactly.
    attempt_stage = ProofStage(backend["attempt_stage"])
    attempt_provider = (
        backend.get("attempt_provider_id") or backend.get("provider_id") or ""
    )
    formal_provider = backend.get("provider_id") or attempt_provider
    attempt = ProofAttempt(
        plan_id=backend["plan_id"],
        step_id=backend["step_id"],
        obligation_id=key.proof_obligation_cid,
        repository_tree_id=backend["repository_tree_identity"],
        provider_id=str(attempt_provider),
        stage=attempt_stage,
        status=AttemptStatus.SUCCEEDED,
        evidence=evidence,
        input_ids=(key.key_id,),
        output_ids=tuple(item.evidence_id for item in evidence),
    )
    formal = FormalProofReceipt(
        obligation_id=key.proof_obligation_cid,
        plan_id=backend["plan_id"],
        attempt_id=attempt.attempt_id,
        repository_id=backend["repository_id"],
        repository_tree_id=backend["repository_tree_identity"],
        ast_scope_ids=tuple(backend["ast_scope_ids"]),
        premise_ids=tuple(backend["premise_ids"]),
        translator_id=backend["translator_id"],
        solver_id=backend["solver_id"],
        kernel_id=backend["kernel_id"],
        toolchain_id=backend["toolchain_id"],
        theorem_registry_id=backend.get("theorem_registry_id") or "",
        policy_id=backend["policy_id"],
        resource_budget=budget,
        verdict=provider_verdict,
        evidence=evidence,
        provider_id=str(formal_provider),
        provider_claimed_assurance=provider_claimed_assurance,
        freshness=EvidenceFreshness.CURRENT,
    )
    return attempt, formal


def project_z3_terminal_status(
    *,
    run_result: VerificationRunResult | None,
    solver_outcome: Z3SolverOutcome,
    formal: FormalProofReceipt | None,
    required_assurance: AssuranceLevel | str,
    simulated: bool,
    bindings_ok: bool,
    bare_text_only: bool,
) -> tuple[TerminalStatus, tuple[str, ...], ProofAuthoritySource]:
    """Project closed terminal status for a Z3 observation."""

    reasons: list[str] = []
    if simulated:
        reasons.append("simulated_mode")
        return (
            TerminalStatus.SIMULATED,
            tuple(reasons),
            ProofAuthoritySource.NONE,
        )

    if run_result is not None:
        if run_result.timed_out or run_result.disposition is VerificationRunDisposition.TIMEOUT:
            reasons.append("timeout")
            reasons.extend(run_result.reason_codes)
            return (
                TerminalStatus.TIMEOUT,
                _unique_reasons(reasons),
                ProofAuthoritySource.NONE,
            )
        if run_result.cancelled or run_result.disposition is VerificationRunDisposition.CANCELLED:
            reasons.append("cancelled")
            reasons.extend(run_result.reason_codes)
            return (
                TerminalStatus.CANCELLED,
                _unique_reasons(reasons),
                ProofAuthoritySource.NONE,
            )
        if (
            run_result.unavailable
            or run_result.disposition is VerificationRunDisposition.UNAVAILABLE
        ):
            reasons.append("unavailable")
            reasons.extend(run_result.reason_codes)
            return (
                TerminalStatus.UNAVAILABLE,
                _unique_reasons(reasons),
                ProofAuthoritySource.NONE,
            )

    if solver_outcome is Z3SolverOutcome.ABSENT:
        reasons.append("z3_absent")
        return (
            TerminalStatus.UNAVAILABLE,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    if bare_text_only:
        reasons.append("bare_solver_text_not_authority")
        return (
            TerminalStatus.UNKNOWN,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    if not bindings_ok:
        reasons.append("obligation_or_tool_binding_mismatch")
        return (
            TerminalStatus.INVALID,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    if solver_outcome is Z3SolverOutcome.MALFORMED:
        reasons.append("malformed_solver_output")
        return (
            TerminalStatus.INVALID,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    if solver_outcome is Z3SolverOutcome.UNKNOWN:
        reasons.append("solver_unknown")
        return (
            TerminalStatus.UNKNOWN,
            _unique_reasons(reasons),
            ProofAuthoritySource.CURRENT_DIRECT_EXECUTION,
        )

    if formal is None:
        # Conclusive statuses require authoritative formal evidence.
        reasons.append("missing_authoritative_formal_evidence")
        return (
            TerminalStatus.UNKNOWN,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
        _formal_proof_status,
    )

    status = _formal_proof_status(formal, required_assurance)
    if status is TerminalStatus.PROVED:
        reasons.append("solver_unsat_authoritative")
        return (
            status,
            _unique_reasons(reasons),
            ProofAuthoritySource.CURRENT_DIRECT_EXECUTION,
        )
    if status is TerminalStatus.DISPROVED:
        reasons.append("solver_sat_counterexample")
        return (
            status,
            _unique_reasons(reasons),
            ProofAuthoritySource.CURRENT_DIRECT_EXECUTION,
        )
    reasons.append(f"formal_status:{status.value}")
    return (
        status,
        _unique_reasons(reasons),
        ProofAuthoritySource.CURRENT_DIRECT_EXECUTION,
    )


def project_proof_assistant_terminal_status(
    *,
    run_result: VerificationRunResult | None,
    formal: FormalProofReceipt | None,
    required_assurance: AssuranceLevel | str,
    simulated: bool,
    registry_admission: RegistryKernelAdmission | None,
    model_generated_draft: bool,
    incomplete_or_unsafe: bool,
    incomplete_reason: str,
    using_existing_evidence: bool,
) -> tuple[TerminalStatus, tuple[str, ...], ProofAuthoritySource]:
    """Project closed terminal status for a proof-assistant observation."""

    reasons: list[str] = []
    if simulated:
        reasons.append("simulated_mode")
        return (
            TerminalStatus.SIMULATED,
            tuple(reasons),
            ProofAuthoritySource.NONE,
        )

    if model_generated_draft:
        reasons.append("model_generated_draft_cannot_prove")
    if incomplete_or_unsafe:
        reasons.append(incomplete_reason or "incomplete_or_unsafe_proof")

    if using_existing_evidence and formal is not None:
        from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
            _formal_proof_status,
        )

        status = _formal_proof_status(formal, required_assurance)
        if status in {TerminalStatus.PROVED, TerminalStatus.DISPROVED}:
            if model_generated_draft or incomplete_or_unsafe:
                # Existing authoritative evidence still wins only when it is
                # independent kernel evidence; drafts never upgrade.
                reasons.append("draft_ignored_for_existing_evidence")
            reasons.append("existing_authoritative_evidence")
            return (
                status,
                _unique_reasons(reasons),
                ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE,
            )
        reasons.append(f"existing_evidence_status:{status.value}")
        return (
            status,
            _unique_reasons(reasons),
            ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE,
        )

    if registry_admission is None or not registry_admission.operational:
        reasons.append("kernel_probe_unavailable")
        if registry_admission is None:
            reasons.append("missing_registry_admission")
        elif not registry_admission.offline:
            reasons.append("kernel_not_offline")
        elif not registry_admission.admitted:
            reasons.append("kernel_not_registry_admitted")
        elif not registry_admission.smoke_tested:
            reasons.append("kernel_not_smoke_tested")
        return (
            TerminalStatus.UNAVAILABLE,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    if run_result is not None:
        if run_result.timed_out or run_result.disposition is VerificationRunDisposition.TIMEOUT:
            reasons.append("timeout")
            reasons.extend(run_result.reason_codes)
            return (
                TerminalStatus.TIMEOUT,
                _unique_reasons(reasons),
                ProofAuthoritySource.NONE,
            )
        if run_result.cancelled or run_result.disposition is VerificationRunDisposition.CANCELLED:
            reasons.append("cancelled")
            reasons.extend(run_result.reason_codes)
            return (
                TerminalStatus.CANCELLED,
                _unique_reasons(reasons),
                ProofAuthoritySource.NONE,
            )
        if (
            run_result.unavailable
            or run_result.disposition is VerificationRunDisposition.UNAVAILABLE
        ):
            reasons.append("unavailable")
            reasons.extend(run_result.reason_codes)
            return (
                TerminalStatus.UNAVAILABLE,
                _unique_reasons(reasons),
                ProofAuthoritySource.NONE,
            )

    if model_generated_draft or incomplete_or_unsafe:
        reasons.append("draft_or_escape_cannot_prove")
        return (
            TerminalStatus.UNKNOWN,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    if formal is None:
        reasons.append("missing_authoritative_formal_evidence")
        return (
            TerminalStatus.UNKNOWN,
            _unique_reasons(reasons),
            ProofAuthoritySource.NONE,
        )

    from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
        _formal_proof_status,
    )

    status = _formal_proof_status(formal, required_assurance)
    reasons.append(f"kernel_direct_execution:{status.value}")
    return (
        status,
        _unique_reasons(reasons),
        ProofAuthoritySource.CURRENT_DIRECT_EXECUTION,
    )


def _empty_stream_cids() -> tuple[str, str]:
    empty = cid_for_bytes(b"")
    return empty, empty


def _dedupe_cids(values: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _publication_allowed(
    *,
    status: TerminalStatus,
    run_result: VerificationRunResult | None,
    simulated: bool,
) -> bool:
    if simulated or status is TerminalStatus.SIMULATED:
        return False
    if status in {
        TerminalStatus.CANCELLED,
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
    }:
        if run_result is not None:
            return bool(run_result.publication_allowed)
        return False
    if run_result is not None:
        return bool(run_result.publication_allowed)
    return True


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class Z3VerificationAdapter:
    """Execute a bound Z3 SMT check through the shared process runner."""

    interface: Final[str] = Z3_VERIFICATION_ADAPTER_INTERFACE
    schema: Final[str] = Z3_VERIFICATION_ADAPTER_SCHEMA
    evidence: Final[str] = PROVER_ADAPTER_EVIDENCE

    def __init__(
        self,
        process_runner: VerificationProcessRunner | None = None,
        *,
        require_production: bool = True,
    ) -> None:
        self._runner = process_runner or VerificationProcessRunner()
        self._require_production = bool(require_production)

    @property
    def process_runner(self) -> VerificationProcessRunner:
        return self._runner

    def build_argv(self, request: Z3VerificationRequest) -> tuple[str, ...]:
        if not isinstance(request, Z3VerificationRequest):
            raise ProverVerificationAdapterError(
                "request must be a Z3VerificationRequest",
                reason_code="invalid_request",
            )
        return build_z3_argv(
            z3_executable=request.z3_executable,
            smtlib_relpath=request.smtlib_relpath,
            extra_z3_args=request.extra_z3_args,
        )

    def execute(
        self,
        request: Z3VerificationRequest,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> Z3VerificationResult:
        if not isinstance(request, Z3VerificationRequest):
            raise ProverVerificationAdapterError(
                "request must be a Z3VerificationRequest",
                reason_code="invalid_request",
            )
        argv = self.build_argv(request)
        self._validate_bindings(request, argv)

        if request.simulated:
            return self._finalize(
                request=request,
                argv=argv,
                run_result=None,
                solver_outcome=Z3SolverOutcome.UNKNOWN,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.SIMULATED,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=("simulated_mode",),
                bare_text_only=False,
                bindings_ok=True,
            )

        # Existing authoritative formal evidence path (no re-solve).
        if request.existing_formal_proof_receipt is not None:
            return self._from_existing_evidence(request, argv)

        # Injected structured outcome (tests / offline).
        if request.injected_solver_outcome is not None:
            outcome = request.injected_solver_outcome
            assert isinstance(outcome, Z3SolverOutcome)
            return self._from_solver_outcome(
                request=request,
                argv=argv,
                run_result=None,
                outcome=outcome,
                stdout=request.injected_stdout or outcome.value + "\n",
                stderr=request.injected_stderr,
                exit_code=request.injected_exit_code,
            )

        # Live execution via admitted runner.
        if not request.smtlib_payload.strip():
            return self._finalize(
                request=request,
                argv=argv,
                run_result=None,
                solver_outcome=Z3SolverOutcome.MALFORMED,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.INVALID,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=("empty_smtlib_payload",),
                bare_text_only=False,
                bindings_ok=True,
            )

        self._materialize_smt(request)
        command = self._build_command(request, argv)
        try:
            run_result = self._runner.run(command, cancellation=cancellation)
        except VerificationProcessRunnerError as exc:
            return self._finalize(
                request=request,
                argv=argv,
                run_result=None,
                solver_outcome=Z3SolverOutcome.ABSENT,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.UNAVAILABLE,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=(
                    getattr(exc, "reason_code", None) or "runner_error",
                    "unavailable",
                ),
                bare_text_only=False,
                bindings_ok=True,
            )

        if (
            run_result.timed_out
            or run_result.cancelled
            or run_result.unavailable
            or run_result.disposition
            in {
                VerificationRunDisposition.TIMEOUT,
                VerificationRunDisposition.CANCELLED,
                VerificationRunDisposition.UNAVAILABLE,
            }
        ):
            outcome = (
                Z3SolverOutcome.ABSENT
                if run_result.unavailable
                else Z3SolverOutcome.UNKNOWN
            )
            return self._finalize(
                request=request,
                argv=tuple(run_result.command_argv) or argv,
                run_result=run_result,
                solver_outcome=outcome,
                formal=None,
                attempt=None,
                forced_status=None,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=(),
                bare_text_only=False,
                bindings_ok=True,
            )

        stdout = (
            run_result.stdout.preview
            if run_result.stdout is not None
            else ""
        )
        stderr = (
            run_result.stderr.preview
            if run_result.stderr is not None
            else ""
        )
        outcome = parse_z3_solver_outcome(stdout, stderr)
        return self._from_solver_outcome(
            request=request,
            argv=tuple(run_result.command_argv) or argv,
            run_result=run_result,
            outcome=outcome,
            stdout=stdout,
            stderr=stderr,
            exit_code=run_result.exit_code,
        )

    # -- internals ---------------------------------------------------------

    def _validate_bindings(
        self,
        request: Z3VerificationRequest,
        argv: Sequence[str],
    ) -> None:
        key = request.receipt_key
        backend = key.proof_backend_binding
        assert backend is not None
        if list(argv[:2]) != [request.z3_executable, "-smt2"]:
            raise ProverVerificationAdapterError(
                "argv must be explicit z3 -smt2 ...",
                reason_code="invalid_argv_form",
            )
        env = key.environment_observation
        if env.get("tool_name") != "z3":
            raise ProverVerificationAdapterError(
                "environment tool_name must be z3",
                reason_code="environment_binding_mismatch",
            )
        if env.get("adapter_schema") != Z3_VERIFICATION_ADAPTER_SCHEMA:
            raise ProverVerificationAdapterError(
                "environment adapter_schema mismatch",
                reason_code="environment_binding_mismatch",
            )
        if env.get("network_policy") != request.network_policy:
            raise ProverVerificationAdapterError(
                "environment network_policy mismatch",
                reason_code="environment_binding_mismatch",
            )
        if backend.get("solver_id") and "z3" not in str(backend["solver_id"]).lower():
            raise ProverVerificationAdapterError(
                "proof backend solver_id must identify z3",
                reason_code="solver_binding_mismatch",
            )

    def _materialize_smt(self, request: Z3VerificationRequest) -> None:
        path = Path(request.sandbox.artifact_root) / request.smtlib_relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(request.smtlib_payload, encoding="utf-8")

    def _build_command(
        self,
        request: Z3VerificationRequest,
        argv: Sequence[str],
    ) -> VerificationCommand:
        env = dict(request.environment)
        if not env:
            env = build_hermetic_environment()
        return VerificationCommand(
            argv=list(argv),
            cwd=request.cwd,
            environment=env,
            timeout_seconds=request.timeout_seconds,
            sandbox=request.sandbox,
            network_policy=request.network_policy,
            max_stdout_bytes=request.max_stdout_bytes,
            max_stderr_bytes=request.max_stderr_bytes,
            lane_id=request.lane_id,
            resource_class=request.resource_class,
            stage=request.stage,
            metadata={
                **dict(request.metadata),
                "adapter": Z3_VERIFICATION_ADAPTER_SCHEMA,
            },
        )

    def _from_existing_evidence(
        self,
        request: Z3VerificationRequest,
        argv: Sequence[str],
    ) -> Z3VerificationResult:
        formal = request.existing_formal_proof_receipt
        attempt = request.existing_proof_attempt
        assert formal is not None
        backend = request.receipt_key.proof_backend_binding
        assert backend is not None
        from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
            _formal_proof_status,
        )

        status = _formal_proof_status(formal, backend["required_assurance"])
        reasons = _unique_reasons(
            ("existing_authoritative_evidence", f"formal_status:{status.value}")
        )
        # Infer solver outcome from formal evidence when possible.
        outcome = Z3SolverOutcome.UNKNOWN
        for item in formal.evidence:
            token = str(item.metadata.get("solver_outcome") or "").lower()
            if token in {"sat", "unsat", "unknown"}:
                outcome = Z3SolverOutcome(token)
                break
            if (
                item.kind is EvidenceKind.SOLVER_RESULT
                and item.verdict is EvidenceVerdict.ACCEPTED
            ):
                outcome = Z3SolverOutcome.UNSAT
            elif (
                item.kind is EvidenceKind.SOLVER_RESULT
                and item.verdict is EvidenceVerdict.REJECTED
                and item.metadata.get("counterexample_verified") is True
            ):
                outcome = Z3SolverOutcome.SAT
        return self._finalize(
            request=request,
            argv=argv,
            run_result=None,
            solver_outcome=outcome,
            formal=formal,
            attempt=attempt,
            forced_status=status,
            authority_source=ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE,
            extra_reasons=reasons,
            bare_text_only=False,
            bindings_ok=True,
        )

    def _from_solver_outcome(
        self,
        *,
        request: Z3VerificationRequest,
        argv: Sequence[str],
        run_result: VerificationRunResult | None,
        outcome: Z3SolverOutcome,
        stdout: str,
        stderr: str,
        exit_code: int | None,
    ) -> Z3VerificationResult:
        bindings_ok, binding_reasons = _bindings_match_request(request)
        # Bare text path: outcome parsed but no obligation/tool binding — never prove.
        bare_text_only = False
        if outcome in {Z3SolverOutcome.SAT, Z3SolverOutcome.UNSAT} and not bindings_ok:
            bare_text_only = True

        formal: FormalProofReceipt | None = None
        attempt: ProofAttempt | None = None
        budget = request.resource_budget or _default_budget(request.timeout_seconds)
        report_cid = content_identity(
            {
                "schema": Z3_SOLVER_REPORT_SCHEMA,
                "outcome": outcome.value,
                "stdout_preview": str(stdout)[:4096],
                "stderr_preview": str(stderr)[:1024],
                "obligation_id": request.receipt_key.proof_obligation_cid,
                "solver_id": (
                    request.receipt_key.proof_backend_binding or {}
                ).get("solver_id", ""),
                "translator_id": (
                    request.receipt_key.proof_backend_binding or {}
                ).get("translator_id", ""),
                "tool_version": request.receipt_key.tool_version,
            }
        )

        if (
            bindings_ok
            and outcome in {Z3SolverOutcome.SAT, Z3SolverOutcome.UNSAT}
            and not bare_text_only
        ):
            cex = request.counterexample_verified
            if cex is None:
                # For SAT, require explicit counterexample verification flag
                # before DISPROVED; otherwise leave non-conclusive.
                cex = outcome is Z3SolverOutcome.SAT
            evidence = (
                _build_solver_evidence(
                    key=request.receipt_key,
                    outcome=outcome,
                    artifact_id=report_cid,
                    counterexample_verified=bool(cex)
                    if outcome is Z3SolverOutcome.SAT
                    else False,
                ),
            )
            if outcome is Z3SolverOutcome.UNSAT:
                provider_verdict = ProofVerdict.PROVED
            elif bool(cex):
                provider_verdict = ProofVerdict.DISPROVED
            else:
                provider_verdict = ProofVerdict.INCONCLUSIVE
            if provider_verdict is not ProofVerdict.INCONCLUSIVE:
                attempt, formal = _build_formal_bundle(
                    key=request.receipt_key,
                    evidence=evidence,
                    provider_verdict=provider_verdict,
                    budget=budget,
                    provider_claimed_assurance=AssuranceLevel.ATTESTED,
                )
            else:
                # SAT without verified counterexample cannot disprove.
                outcome = Z3SolverOutcome.UNKNOWN

        return self._finalize(
            request=request,
            argv=argv,
            run_result=run_result,
            solver_outcome=outcome,
            formal=formal,
            attempt=attempt,
            forced_status=None,
            authority_source=ProofAuthoritySource.NONE,
            extra_reasons=binding_reasons,
            bare_text_only=bare_text_only,
            bindings_ok=bindings_ok,
            solver_report_cid=report_cid,
            forced_exit_code=exit_code,
        )

    def _finalize(
        self,
        *,
        request: Z3VerificationRequest,
        argv: Sequence[str],
        run_result: VerificationRunResult | None,
        solver_outcome: Z3SolverOutcome,
        formal: FormalProofReceipt | None,
        attempt: ProofAttempt | None,
        forced_status: TerminalStatus | None,
        authority_source: ProofAuthoritySource,
        extra_reasons: Sequence[str],
        bare_text_only: bool,
        bindings_ok: bool,
        solver_report_cid: str = "",
        forced_exit_code: int | None = None,
    ) -> Z3VerificationResult:
        backend = request.receipt_key.proof_backend_binding
        assert backend is not None
        if forced_status is not None:
            status = forced_status
            reasons = _unique_reasons(extra_reasons)
            auth = authority_source
        else:
            status, reasons, auth = project_z3_terminal_status(
                run_result=run_result,
                solver_outcome=solver_outcome,
                formal=formal,
                required_assurance=backend["required_assurance"],
                simulated=request.simulated,
                bindings_ok=bindings_ok,
                bare_text_only=bare_text_only,
            )
            if extra_reasons:
                reasons = _unique_reasons((*extra_reasons, *reasons))

        # Timeout/absent never prove even if a formal object was somehow present.
        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.CANCELLED,
        }:
            formal = None
            attempt = None

        duration_ms = int(run_result.duration_ms) if run_result is not None else 0
        stdout_cid = ""
        stderr_cid = ""
        artifact_cids: list[str] = []
        if run_result is not None:
            if run_result.stdout and run_result.stdout.cid:
                stdout_cid = run_result.stdout.cid
                artifact_cids.append(stdout_cid)
            if run_result.stderr and run_result.stderr.cid:
                stderr_cid = run_result.stderr.cid
                artifact_cids.append(stderr_cid)

        if not solver_report_cid:
            solver_report_cid = content_identity(
                {
                    "schema": Z3_SOLVER_REPORT_SCHEMA,
                    "outcome": solver_outcome.value,
                    "status": status.value,
                    "obligation_id": request.receipt_key.proof_obligation_cid,
                }
            )
        artifact_cids.append(solver_report_cid)

        if formal is not None:
            artifact_cids.append(formal.receipt_id)
            if attempt is not None:
                artifact_cids.append(attempt.attempt_id)

        empty_out, empty_err = _empty_stream_cids()
        exit_code = forced_exit_code
        if run_result is not None and exit_code is None:
            exit_code = run_result.exit_code

        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.CANCELLED,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.SIMULATED,
        }:
            if run_result is None or not run_result.process_started:
                exit_code = None
            stdout_cid = stdout_cid or empty_out
            stderr_cid = stderr_cid or empty_err
        else:
            if status in {TerminalStatus.PROVED, TerminalStatus.DISPROVED}:
                exit_code = 0
            elif exit_code is None:
                exit_code = 1 if status not in {TerminalStatus.UNKNOWN} else 0
            stdout_cid = stdout_cid or empty_out
            stderr_cid = stderr_cid or empty_err
        for cid in (stdout_cid, stderr_cid):
            if cid and cid not in artifact_cids:
                artifact_cids.append(cid)

        deduped = _dedupe_cids(artifact_cids)
        key = request.receipt_key
        # ProofReceipt forbids proved/disproved without formal evidence; also
        # forbids passed/failed on the direct path.
        execution = DirectExecutionObservation(
            receipt_key_cid=key.key_id,
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            repository_tree_observation=key.repository_tree_observation,
            environment_observation=key.environment_observation,
            terminal_status=status,
            command_argv=tuple(argv),
            duration_ms=duration_ms,
            exit_code=exit_code,
            stdout_artifact_cid=stdout_cid,
            stderr_artifact_cid=stderr_cid,
            artifact_cids=deduped,
            reason_codes=reasons,
        )

        receipt: ProofReceipt | None
        try:
            receipt = ProofReceipt(
                key=key,
                execution=execution,
                formal_proof_receipt=formal,
                proof_attempt=attempt,
                artifact_cids=deduped,
                reason_codes=reasons,
            )
        except (VerificationContractError, VerificationIdentityError) as exc:
            # Fail closed: never manufacture proved.
            drop_ids = {
                formal.receipt_id if formal is not None else "",
                attempt.attempt_id if attempt is not None else "",
            }
            status = TerminalStatus.INVALID
            reasons = _unique_reasons(
                (*reasons, "receipt_projection_failed", type(exc).__name__)
            )
            formal = None
            attempt = None
            filtered = tuple(cid for cid in deduped if cid not in drop_ids)
            if not filtered:
                filtered = deduped
            execution = DirectExecutionObservation(
                receipt_key_cid=key.key_id,
                repository_tree_cid=key.repository_tree_cid,
                environment_cid=key.environment_cid,
                repository_tree_observation=key.repository_tree_observation,
                environment_observation=key.environment_observation,
                terminal_status=status,
                command_argv=tuple(argv),
                duration_ms=duration_ms,
                exit_code=1,
                stdout_artifact_cid=stdout_cid,
                stderr_artifact_cid=stderr_cid,
                artifact_cids=filtered,
                reason_codes=reasons,
            )
            receipt = ProofReceipt(
                key=key,
                execution=execution,
                artifact_cids=filtered,
                reason_codes=reasons,
            )
            deduped = filtered

        production_admissible = (
            not request.simulated
            and status is TerminalStatus.PROVED
            and receipt is not None
            and receipt.terminal_success
            and receipt.status is TerminalStatus.PROVED
            and formal is not None
        )
        if self._require_production and request.simulated:
            production_admissible = False

        return Z3VerificationResult(
            terminal_status=status,
            receipt=receipt,
            command_argv=tuple(argv),
            solver_outcome=solver_outcome,
            authority_source=auth,
            artifact_cids=deduped,
            reason_codes=reasons,
            production_admissible=production_admissible,
            simulated=request.simulated,
            run_result=run_result,
            formal_proof_receipt=formal,
            proof_attempt=attempt,
            solver_report_cid=solver_report_cid,
            duration_ms=duration_ms,
            exit_code=exit_code,
            publication_allowed=_publication_allowed(
                status=status,
                run_result=run_result,
                simulated=request.simulated,
            ),
        )


class ExistingProofAssistantAdapter:
    """Wrap an offline registry-admitted Lean/Coq/Isabelle kernel probe."""

    interface: Final[str] = PROOF_ASSISTANT_ADAPTER_INTERFACE
    schema: Final[str] = PROOF_ASSISTANT_ADAPTER_SCHEMA
    evidence: Final[str] = PROVER_ADAPTER_EVIDENCE

    def __init__(
        self,
        process_runner: VerificationProcessRunner | None = None,
        *,
        require_production: bool = True,
    ) -> None:
        self._runner = process_runner or VerificationProcessRunner()
        self._require_production = bool(require_production)

    @property
    def process_runner(self) -> VerificationProcessRunner:
        return self._runner

    def select_registry_probe(
        self,
        admission: RegistryKernelAdmission | None,
    ) -> RegistryKernelAdmission | None:
        """Return *admission* only when it is an operational offline probe."""

        if admission is None:
            return None
        if not isinstance(admission, RegistryKernelAdmission):
            return None
        if not admission.operational:
            return None
        return admission

    def build_argv(
        self,
        request: ProofAssistantVerificationRequest,
    ) -> tuple[str, ...]:
        if not isinstance(request, ProofAssistantVerificationRequest):
            raise ProverVerificationAdapterError(
                "request must be a ProofAssistantVerificationRequest",
                reason_code="invalid_request",
            )
        admission = self.select_registry_probe(request.registry_admission)
        executable = (
            request.kernel_executable
            or (admission.executable_path if admission is not None else "")
        )
        if not executable:
            return ()
        extra = tuple(request.extra_kernel_args)
        # Lean default: `lean <file>`; Coq: `coqc <file>`; Isabelle: `isabelle process -T ...`
        prover = admission.prover_id if admission is not None else "lean"
        if prover in {"coq", "rocq"}:
            return (executable, *extra, request.source_relpath)
        if prover == "isabelle":
            return (executable, *extra, request.source_relpath)
        return (executable, *extra, request.source_relpath)

    def execute(
        self,
        request: ProofAssistantVerificationRequest,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> ProofAssistantVerificationResult:
        if not isinstance(request, ProofAssistantVerificationRequest):
            raise ProverVerificationAdapterError(
                "request must be a ProofAssistantVerificationRequest",
                reason_code="invalid_request",
            )

        if request.simulated:
            return self._finalize(
                request=request,
                argv=(),
                run_result=None,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.SIMULATED,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=("simulated_mode",),
                using_existing_evidence=False,
                incomplete_or_unsafe=False,
                incomplete_reason="",
            )

        # Existing authoritative evidence path.
        if request.existing_formal_proof_receipt is not None:
            return self._from_existing_evidence(request)

        admission = self.select_registry_probe(request.registry_admission)
        incomplete, incomplete_reason = source_contains_incomplete_or_unsafe_proof(
            request.checked_source
        )
        if request.model_generated_draft:
            incomplete = True
            incomplete_reason = incomplete_reason or "model_generated_draft_cannot_prove"

        if admission is None:
            return self._finalize(
                request=request,
                argv=(),
                run_result=None,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.UNAVAILABLE,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=(
                    "kernel_probe_unavailable",
                    "missing_or_non_operational_registry_admission",
                ),
                using_existing_evidence=False,
                incomplete_or_unsafe=incomplete,
                incomplete_reason=incomplete_reason,
            )

        # Drafts / sorry / admit / unsafe never prove even with an admitted kernel.
        if incomplete or request.model_generated_draft:
            return self._finalize(
                request=request,
                argv=self.build_argv(request),
                run_result=None,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.UNKNOWN,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=(
                    "draft_or_escape_cannot_prove",
                    incomplete_reason or "incomplete_or_unsafe_proof",
                ),
                using_existing_evidence=False,
                incomplete_or_unsafe=True,
                incomplete_reason=incomplete_reason,
            )

        # Injected kernel result (deterministic offline tests).
        if request.injected_kernel_accepted is not None:
            return self._from_kernel_result(
                request=request,
                admission=admission,
                argv=self.build_argv(request),
                run_result=None,
                accepted=bool(request.injected_kernel_accepted),
                failure_code=request.injected_failure_code,
                stdout=request.injected_stdout,
                stderr=request.injected_stderr,
                exit_code=request.injected_exit_code,
            )

        argv = self.build_argv(request)
        if not argv:
            return self._finalize(
                request=request,
                argv=(),
                run_result=None,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.UNAVAILABLE,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=("kernel_executable_missing",),
                using_existing_evidence=False,
                incomplete_or_unsafe=False,
                incomplete_reason="",
            )

        if request.checked_source.strip():
            path = Path(request.sandbox.artifact_root) / request.source_relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(request.checked_source, encoding="utf-8")

        command = self._build_command(request, argv)
        try:
            run_result = self._runner.run(command, cancellation=cancellation)
        except VerificationProcessRunnerError as exc:
            return self._finalize(
                request=request,
                argv=argv,
                run_result=None,
                formal=None,
                attempt=None,
                forced_status=TerminalStatus.UNAVAILABLE,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=(
                    getattr(exc, "reason_code", None) or "runner_error",
                    "unavailable",
                ),
                using_existing_evidence=False,
                incomplete_or_unsafe=False,
                incomplete_reason="",
            )

        if (
            run_result.timed_out
            or run_result.cancelled
            or run_result.unavailable
            or run_result.disposition
            in {
                VerificationRunDisposition.TIMEOUT,
                VerificationRunDisposition.CANCELLED,
                VerificationRunDisposition.UNAVAILABLE,
            }
        ):
            return self._finalize(
                request=request,
                argv=tuple(run_result.command_argv) or argv,
                run_result=run_result,
                formal=None,
                attempt=None,
                forced_status=None,
                authority_source=ProofAuthoritySource.NONE,
                extra_reasons=(),
                using_existing_evidence=False,
                incomplete_or_unsafe=False,
                incomplete_reason="",
            )

        stdout = run_result.stdout.preview if run_result.stdout else ""
        stderr = run_result.stderr.preview if run_result.stderr else ""
        accepted = run_result.exit_code == 0 and not _INCOMPLETE_PROOF_RE.search(
            stdout + "\n" + stderr
        )
        failure_code = ""
        if not accepted:
            failure_code = (
                KernelFailureCode.KERNEL_REJECTED.value
                if run_result.exit_code not in (0, None)
                else KernelFailureCode.INCOMPLETE_PROOF.value
            )
        return self._from_kernel_result(
            request=request,
            admission=admission,
            argv=tuple(run_result.command_argv) or argv,
            run_result=run_result,
            accepted=accepted,
            failure_code=failure_code,
            stdout=stdout,
            stderr=stderr,
            exit_code=run_result.exit_code,
        )

    def _build_command(
        self,
        request: ProofAssistantVerificationRequest,
        argv: Sequence[str],
    ) -> VerificationCommand:
        env = dict(request.environment)
        if not env:
            env = build_hermetic_environment()
        return VerificationCommand(
            argv=list(argv),
            cwd=request.cwd,
            environment=env,
            timeout_seconds=request.timeout_seconds,
            sandbox=request.sandbox,
            network_policy=request.network_policy,
            max_stdout_bytes=request.max_stdout_bytes,
            max_stderr_bytes=request.max_stderr_bytes,
            lane_id=request.lane_id,
            resource_class=request.resource_class,
            stage=request.stage,
            metadata={
                **dict(request.metadata),
                "adapter": PROOF_ASSISTANT_ADAPTER_SCHEMA,
            },
        )

    def _from_existing_evidence(
        self,
        request: ProofAssistantVerificationRequest,
    ) -> ProofAssistantVerificationResult:
        formal = request.existing_formal_proof_receipt
        attempt = request.existing_proof_attempt
        assert formal is not None
        backend = request.receipt_key.proof_backend_binding
        assert backend is not None
        from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
            _formal_proof_status,
        )

        status = _formal_proof_status(formal, backend["required_assurance"])
        incomplete, incomplete_reason = source_contains_incomplete_or_unsafe_proof(
            request.checked_source
        )
        return self._finalize(
            request=request,
            argv=(),
            run_result=None,
            formal=formal,
            attempt=attempt,
            forced_status=status,
            authority_source=ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE,
            extra_reasons=(
                "existing_authoritative_evidence",
                f"formal_status:{status.value}",
            ),
            using_existing_evidence=True,
            incomplete_or_unsafe=incomplete or request.model_generated_draft,
            incomplete_reason=incomplete_reason,
        )

    def _from_kernel_result(
        self,
        *,
        request: ProofAssistantVerificationRequest,
        admission: RegistryKernelAdmission,
        argv: Sequence[str],
        run_result: VerificationRunResult | None,
        accepted: bool,
        failure_code: str,
        stdout: str,
        stderr: str,
        exit_code: int | None,
    ) -> ProofAssistantVerificationResult:
        budget = request.resource_budget or _default_budget(request.timeout_seconds)
        report_cid = content_identity(
            {
                "schema": KERNEL_PROBE_REPORT_SCHEMA,
                "accepted": accepted,
                "failure_code": failure_code,
                "prover_id": admission.prover_id,
                "capability": admission.authority_capability,
                "stdout_preview": str(stdout)[:4096],
                "stderr_preview": str(stderr)[:1024],
                "obligation_id": request.receipt_key.proof_obligation_cid,
                "kernel_id": (
                    request.receipt_key.proof_backend_binding or {}
                ).get("kernel_id", ""),
            }
        )
        evidence = (
            _build_kernel_evidence(
                key=request.receipt_key,
                accepted=accepted,
                artifact_id=report_cid,
                failure_code=failure_code,
            ),
        )
        formal: FormalProofReceipt | None = None
        attempt: ProofAttempt | None = None
        if accepted:
            attempt, formal = _build_formal_bundle(
                key=request.receipt_key,
                evidence=evidence,
                provider_verdict=ProofVerdict.PROVED,
                budget=budget,
                provider_claimed_assurance=AssuranceLevel.ATTESTED,
            )
        else:
            # Rejection is inconclusive for the theorem (does not disprove).
            attempt, formal = _build_formal_bundle(
                key=request.receipt_key,
                evidence=evidence,
                provider_verdict=ProofVerdict.INCONCLUSIVE,
                budget=budget,
                provider_claimed_assurance=AssuranceLevel.UNVERIFIED,
            )

        return self._finalize(
            request=request,
            argv=argv,
            run_result=run_result,
            formal=formal,
            attempt=attempt,
            forced_status=None,
            authority_source=ProofAuthoritySource.CURRENT_DIRECT_EXECUTION,
            extra_reasons=(),
            using_existing_evidence=False,
            incomplete_or_unsafe=False,
            incomplete_reason="",
            kernel_report_cid=report_cid,
            forced_exit_code=exit_code,
        )

    def _finalize(
        self,
        *,
        request: ProofAssistantVerificationRequest,
        argv: Sequence[str],
        run_result: VerificationRunResult | None,
        formal: FormalProofReceipt | None,
        attempt: ProofAttempt | None,
        forced_status: TerminalStatus | None,
        authority_source: ProofAuthoritySource,
        extra_reasons: Sequence[str],
        using_existing_evidence: bool,
        incomplete_or_unsafe: bool,
        incomplete_reason: str,
        kernel_report_cid: str = "",
        forced_exit_code: int | None = None,
    ) -> ProofAssistantVerificationResult:
        backend = request.receipt_key.proof_backend_binding
        assert backend is not None
        admission = request.registry_admission

        if forced_status is not None and not using_existing_evidence:
            status = forced_status
            reasons = _unique_reasons(extra_reasons)
            auth = authority_source
        else:
            status, reasons, auth = project_proof_assistant_terminal_status(
                run_result=run_result,
                formal=formal,
                required_assurance=backend["required_assurance"],
                simulated=request.simulated,
                registry_admission=admission,
                model_generated_draft=request.model_generated_draft,
                incomplete_or_unsafe=incomplete_or_unsafe,
                incomplete_reason=incomplete_reason,
                using_existing_evidence=using_existing_evidence
                or authority_source
                is ProofAuthoritySource.EXISTING_AUTHORITATIVE_EVIDENCE,
            )
            if forced_status is not None and using_existing_evidence:
                status = forced_status
            if extra_reasons:
                reasons = _unique_reasons((*extra_reasons, *reasons))

        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.CANCELLED,
        } and not using_existing_evidence:
            # Fence conclusive formal material on non-execution terminals.
            formal = None
            attempt = None

        # Model drafts / sorry cannot keep a PROVED projection from direct exec.
        if (
            not using_existing_evidence
            and (incomplete_or_unsafe or request.model_generated_draft)
            and status is TerminalStatus.PROVED
        ):
            status = TerminalStatus.UNKNOWN
            formal = None
            attempt = None
            reasons = _unique_reasons(
                (*reasons, "draft_or_escape_cannot_prove")
            )

        duration_ms = int(run_result.duration_ms) if run_result is not None else 0
        stdout_cid = ""
        stderr_cid = ""
        artifact_cids: list[str] = []
        if run_result is not None:
            if run_result.stdout and run_result.stdout.cid:
                stdout_cid = run_result.stdout.cid
                artifact_cids.append(stdout_cid)
            if run_result.stderr and run_result.stderr.cid:
                stderr_cid = run_result.stderr.cid
                artifact_cids.append(stderr_cid)

        if not kernel_report_cid:
            kernel_report_cid = content_identity(
                {
                    "schema": KERNEL_PROBE_REPORT_SCHEMA,
                    "status": status.value,
                    "obligation_id": request.receipt_key.proof_obligation_cid,
                    "prover_id": (
                        admission.prover_id if admission is not None else ""
                    ),
                }
            )
        artifact_cids.append(kernel_report_cid)
        if formal is not None:
            artifact_cids.append(formal.receipt_id)
            if attempt is not None:
                artifact_cids.append(attempt.attempt_id)

        empty_out, empty_err = _empty_stream_cids()
        exit_code = forced_exit_code
        if run_result is not None and exit_code is None:
            exit_code = run_result.exit_code

        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.CANCELLED,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.SIMULATED,
        }:
            if run_result is None or not getattr(run_result, "process_started", False):
                exit_code = None
            stdout_cid = stdout_cid or empty_out
            stderr_cid = stderr_cid or empty_err
        else:
            if status in {TerminalStatus.PROVED, TerminalStatus.DISPROVED}:
                exit_code = 0
            elif exit_code is None:
                exit_code = 0 if status is TerminalStatus.UNKNOWN else 1
            stdout_cid = stdout_cid or empty_out
            stderr_cid = stderr_cid or empty_err
        for cid in (stdout_cid, stderr_cid):
            if cid and cid not in artifact_cids:
                artifact_cids.append(cid)

        # Observation argv must match the receipt-key selector identity. Prefer
        # the executed argv, then a rebuilt declared argv from the request.
        observed_argv = tuple(argv)
        if not observed_argv:
            observed_argv = self.build_argv(request)
        if not observed_argv and request.kernel_executable:
            observed_argv = (
                request.kernel_executable,
                *tuple(request.extra_kernel_args),
                request.source_relpath,
            )
        if not observed_argv:
            # Last resort: keep a non-empty argv only for non-receipt diagnostics.
            # Callers should supply kernel_executable bound to the receipt key.
            observed_argv = ("/usr/bin/false", "--proof-assistant-unavailable")

        deduped = _dedupe_cids(artifact_cids)
        key = request.receipt_key
        execution = DirectExecutionObservation(
            receipt_key_cid=key.key_id,
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            repository_tree_observation=key.repository_tree_observation,
            environment_observation=key.environment_observation,
            terminal_status=status,
            command_argv=observed_argv,
            duration_ms=duration_ms,
            exit_code=exit_code,
            stdout_artifact_cid=stdout_cid,
            stderr_artifact_cid=stderr_cid,
            artifact_cids=deduped,
            reason_codes=reasons,
        )

        receipt: ProofReceipt | None
        try:
            receipt = ProofReceipt(
                key=key,
                execution=execution,
                formal_proof_receipt=formal,
                proof_attempt=attempt,
                artifact_cids=deduped,
                reason_codes=reasons,
            )
        except (VerificationContractError, VerificationIdentityError) as exc:
            status = TerminalStatus.INVALID
            reasons = _unique_reasons(
                (*reasons, "receipt_projection_failed", type(exc).__name__)
            )
            formal = None
            attempt = None
            execution = DirectExecutionObservation(
                receipt_key_cid=key.key_id,
                repository_tree_cid=key.repository_tree_cid,
                environment_cid=key.environment_cid,
                repository_tree_observation=key.repository_tree_observation,
                environment_observation=key.environment_observation,
                terminal_status=status,
                command_argv=observed_argv,
                duration_ms=duration_ms,
                exit_code=1,
                stdout_artifact_cid=stdout_cid,
                stderr_artifact_cid=stderr_cid,
                artifact_cids=deduped,
                reason_codes=reasons,
            )
            receipt = ProofReceipt(
                key=key,
                execution=execution,
                artifact_cids=deduped,
                reason_codes=reasons,
            )

        production_admissible = (
            not request.simulated
            and status is TerminalStatus.PROVED
            and receipt is not None
            and receipt.terminal_success
            and receipt.status is TerminalStatus.PROVED
            and formal is not None
        )
        if self._require_production and request.simulated:
            production_admissible = False

        return ProofAssistantVerificationResult(
            terminal_status=status,
            receipt=receipt,
            command_argv=observed_argv,
            authority_source=auth,
            registry_admission=admission,
            artifact_cids=deduped,
            reason_codes=reasons,
            production_admissible=production_admissible,
            simulated=request.simulated,
            run_result=run_result,
            formal_proof_receipt=formal,
            proof_attempt=attempt,
            kernel_report_cid=kernel_report_cid,
            duration_ms=duration_ms,
            exit_code=exit_code,
            publication_allowed=_publication_allowed(
                status=status,
                run_result=run_result,
                simulated=request.simulated,
            ),
        )


def create_z3_verification_adapter(
    process_runner: VerificationProcessRunner | None = None,
    *,
    require_production: bool = True,
) -> Z3VerificationAdapter:
    return Z3VerificationAdapter(
        process_runner=process_runner,
        require_production=require_production,
    )


def create_existing_proof_assistant_adapter(
    process_runner: VerificationProcessRunner | None = None,
    *,
    require_production: bool = True,
) -> ExistingProofAssistantAdapter:
    return ExistingProofAssistantAdapter(
        process_runner=process_runner,
        require_production=require_production,
    )


__all__ = [
    "KERNEL_PROBE_REPORT_SCHEMA",
    "PROOF_ASSISTANT_ADAPTER_INTERFACE",
    "PROOF_ASSISTANT_ADAPTER_SCHEMA",
    "PROVER_ADAPTER_EVIDENCE",
    "Z3_SOLVER_REPORT_SCHEMA",
    "Z3_VERIFICATION_ADAPTER_INTERFACE",
    "Z3_VERIFICATION_ADAPTER_SCHEMA",
    "ExistingProofAssistantAdapter",
    "KernelProbeTarget",
    "ProofAssistantVerificationRequest",
    "ProofAssistantVerificationResult",
    "ProofAuthoritySource",
    "ProverVerificationAdapterError",
    "RegistryKernelAdmission",
    "Z3SolverOutcome",
    "Z3VerificationAdapter",
    "Z3VerificationRequest",
    "Z3VerificationResult",
    "build_z3_argv",
    "create_existing_proof_assistant_adapter",
    "create_z3_verification_adapter",
    "parse_z3_solver_outcome",
    "project_proof_assistant_terminal_status",
    "project_z3_terminal_status",
    "source_contains_incomplete_or_unsafe_proof",
]
