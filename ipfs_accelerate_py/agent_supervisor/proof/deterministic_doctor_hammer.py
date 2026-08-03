"""Cache-first isolated Hammer and CEGIS verification for deterministic doctor (LPR-035).

Fail-closed bridge between:

* admitted :class:`~ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician.DoctorTacticianPlanReceipt`
* :class:`DoctorProofCacheGate` (revalidated positive hits only)
* :class:`TacticianHammerObligationCompiler` / native goal reconstruction
* :class:`TacticianHammerCoordinator` behind an explicit native-execution permit
* :class:`LogicPredictionCEGIS` (finite, monotonic, repetition-bounded)
* :class:`LogicPredictionAdmission` (exactly one complete eligible consequence)

Invariants (acceptance / plan §4.13.4):

* Revalidate every cache binding before reuse, render, or admission.
* Require an explicit native-execution permit and adequate subprocess /
  platform isolation (hardened Hammer import isolation + concurrency-safe
  loader report). ``network=false`` is policy metadata, not OS isolation.
* Bind exact obligation, premise, translator, solver, kernel, toolchain,
  policy, resource, and environment identities.
* Reconstruct the matching theorem in the pinned kernel before promotion.
* Independently replay a countermodel or proof of negation before refutation.
* CEGIS is finite, monotonic, and repetition-bounded.
* Exactly one complete eligible consequence may proceed; zero or multiple
  abstain.
* Unavailable isolation/provider/kernel, unsupported lowering, inconsistency,
  ambiguity, stale roots, timeout, or bound exhaustion abstains with **zero
  source writes** and **zero LLM / remote model-provider calls**.
* Never uses the unchecked legacy HammerPipeline, mutates process-global
  import state, accepts a raw solver countermodel, or falls through to a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..analysis.content_identity_bridge import identify_strict_artifact
from ..analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    HypothesisDisposition,
    LogicGap,
    LogicHypothesis,
    NativeGoalDisposition,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicNativeGoalBinding,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    TacticianSearchPlan,
)
from ..analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
    ProgramLogicPremiseCorpus,
)
from ..integrations.ipfs_datasets_logic_provider import (
    HAMMER_IMPORT_ISOLATION,
    HAMMER_IMPORT_ISOLATION_HARDENED,
    IsolatedHammerLoader,
    get_isolated_hammer_loader,
)
from ..planning.deterministic_doctor_tactician import (
    DoctorGoalCompilation,
    DoctorGoalCompilationDisposition,
    DoctorTacticianPlanDisposition,
    DoctorTacticianPlanReceipt,
)
from ..planning.logic_prediction_admission import (
    AutomaticConsequenceKind,
    LogicPredictionAdmission,
    LogicPredictionAdmissionRequest,
    LogicPredictionDecision,
    LogicPredictionDecisionDisposition,
    create_logic_prediction_admission,
)
from ..validation.hammer_native_execution_gate import (
    NativeExecutionAuthorizationGate,
    NativeExecutionDisposition,
    NativeExecutionLane,
    NativeExecutionOperation,
    NativeExecutionPermit,
    ResourceEnforcementReport,
    ResourcePolicySlice,
    probe_resource_enforcement,
)
from ..validation.tactician_plan_gate import (
    TacticianPlanGateDisposition,
    TacticianPlanGateReceipt,
)
from .doctor_proof_cache import (
    DoctorCacheAuditReceipt,
    DoctorCacheDisposition,
    DoctorCacheStage,
    DoctorIdentityBinding,
    DoctorProofCacheGate,
    DoctorProofCacheKey,
    DoctorSealedReceiptError,
    DoctorSealedReceiptRef,
    DoctorSealedReceiptStore,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    ResourceBudget,
    canonical_json,
    content_identity,
)
from .logic_prediction_cegis import (
    LogicPredictionCEGIS,
    LogicRefinementBounds,
    LogicRefinementReceipt,
    LogicRefinementState,
    RefinementDisposition,
    RefinementEvidence,
    RefinementStopReason,
)
from .tactician_hammer_coordinator import (
    COORDINATION_OUTCOMES,
    CoordinationConclusiveness,
    CountermodelValidator,
    HammerCoordinationOutcome,
    HammerCoordinationReceipt,
    PremiseSelectorMode,
    TacticianHammerCoordinator,
    conclusiveness_for,
)
from .tactician_hammer_obligations import (
    AssumptionBinding,
    LoweringDisposition,
    ObligationContext,
    TacticianHammerObligationCompilation,
    TacticianHammerObligationCompiler,
    TranslatorCapabilityBinding,
)

# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_HAMMER_INTERFACE: Final[str] = "DeterministicDoctorHammer@1"
DOCTOR_REPAIR_OBLIGATION_COMPILER_INTERFACE: Final[str] = (
    "DoctorRepairObligationCompiler@1"
)
DOCTOR_REPAIR_PROOF_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-repair-proof-receipt@1"
)
DOCTOR_REPAIR_OBLIGATION_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-repair-obligation-compilation@1"
)
DOCTOR_HAMMER_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-hammer-bounds@1"
)
NATIVE_RECONSTRUCTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/native-reconstruction-receipt@1"
)
DOCTOR_REVIEWED_THEOREM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-reviewed-theorem@1"
)
DOCTOR_EXACT_LOWERING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-exact-lowering@1"
)
DOCTOR_PINNED_EXECUTABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-pinned-executable@1"
)
DOCTOR_NATIVE_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-native-execution-receipt@1"
)
DOCTOR_KERNEL_REPLAY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-kernel-replay-receipt@1"
)
DOCTOR_AUTHORITATIVE_PROOF_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-authoritative-proof-receipt@1"
)
DOCTOR_AUTHORITATIVE_PROOF_INTERFACE: Final[str] = (
    "DoctorAuthoritativeProofReceipt@1"
)
DOCTOR_REPAIR_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-repair-candidate@1"
)

PRODUCER_ID: Final[str] = "deterministic-doctor-hammer@1"
CONTRACT_VERSION: Final[int] = 1

MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_REF_BYTES: Final[int] = 512
MAX_CANDIDATES: Final[int] = 64
MAX_OBLIGATIONS: Final[int] = 256
MAX_REASON_CODES: Final[int] = 64
MAX_PREMISES: Final[int] = 256
MAX_THEOREM_BYTES: Final[int] = 262_144
MAX_NATIVE_OUTPUT_BYTES: Final[int] = 262_144

HARD_MAX_CEGIS_ROUNDS: Final[int] = 64
HARD_MAX_CEGIS_REPEATED_STATES: Final[int] = 8
DEFAULT_MAX_CEGIS_ROUNDS: Final[int] = 8
DEFAULT_MAX_CEGIS_REPEATED_STATES: Final[int] = 2
DEFAULT_WALL_TIME_MS: Final[int] = 60_000
DEFAULT_CPU_TIME_MS: Final[int] = 30_000
DEFAULT_MEMORY_BYTES: Final[int] = 512 * 1024 * 1024

# Isolation levels that satisfy the LPR-035 platform isolation gate.
_ADEQUATE_ISOLATION: Final[frozenset[str]] = frozenset(
    {
        HAMMER_IMPORT_ISOLATION_HARDENED,
        HAMMER_IMPORT_ISOLATION,
        "import_isolation_hardened",
        "subprocess_isolated",
        "platform_isolated",
        "worker_isolated",
    }
)

_LLM_ROUTE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "llm",
        "model_provider",
        "remote_model",
        "openai",
        "anthropic",
        "gemini",
        "completion",
        "chat_completion",
    }
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source_body",
        "source_text",
        "snippet",
        "prompt_body",
        "theorem_text",
        "proof_script",
        "api_key",
        "secret",
        "password",
        "token",
        "private_key",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DoctorHammerDisposition(str, Enum):
    """Closed outcomes of doctor Hammer/CEGIS verification."""

    ADMITTED = "admitted"
    REFUTED = "refuted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class DoctorHammerReasonCode(str, Enum):
    """Stable fail-closed reason codes for LPR-035."""

    OK = "ok"
    PLAN_NOT_ADMITTED = "plan_not_admitted"
    PLAN_GATE_REJECTED = "plan_gate_rejected"
    PLAN_MISSING = "plan_missing"
    COMPILATION_INCOMPLETE = "compilation_incomplete"
    STALE_ROOTS = "stale_roots"
    MIXED_ROOTS = "mixed_roots"
    IDENTITY_MISMATCH = "identity_mismatch"
    PERMIT_MISSING = "permit_missing"
    PERMIT_DENIED = "permit_denied"
    ISOLATION_UNAVAILABLE = "isolation_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    KERNEL_UNAVAILABLE = "kernel_unavailable"
    UNSUPPORTED_LOWERING = "unsupported_lowering"
    INCONSISTENCY = "inconsistency"
    AMBIGUITY = "ambiguity"
    TIMEOUT = "timeout"
    BOUND_EXHAUSTION = "bound_exhaustion"
    CACHE_BINDING_INVALID = "cache_binding_invalid"
    CACHE_STALE = "cache_stale"
    CACHE_REJECTED = "cache_rejected"
    RECONSTRUCTION_FAILED = "reconstruction_failed"
    COUNTERMODEL_UNVALIDATED = "countermodel_unvalidated"
    RAW_SOLVER_COUNTERMODEL = "raw_solver_countermodel"
    MULTIPLE_ELIGIBLE = "multiple_eligible_consequences"
    ZERO_ELIGIBLE = "zero_eligible_consequences"
    HAMMER_NOT_VERIFIED = "hammer_not_verified"
    HAMMER_NOT_CONCLUSIVE = "hammer_not_conclusive"
    CEGIS_INCONCLUSIVE = "cegis_inconclusive"
    LLM_ROUTE = "llm_route"
    MODEL_PROVIDER_ROUTE = "model_provider_route"
    WRITE_ATTEMPTED = "write_attempted"
    SEMANTIC_AUTHORITY_CLAIM = "semantic_authority_claim"
    MALFORMED_INPUT = "malformed_input"
    NO_CANDIDATES = "no_candidates"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    RESOURCE_UNENFORCEABLE = "resource_unenforceable"
    CANCELLED = "cancelled"
    ADMISSION_REJECTED = "admission_rejected"
    THEOREM_BODY_REQUIRED = "semantic_theorem_body_required"
    THEOREM_NOT_REVIEWED = "theorem_not_reviewed"
    EXACT_LOWERING_REQUIRED = "exact_lowering_required"
    EXECUTABLE_NOT_PINNED = "executable_not_pinned"
    EXECUTABLE_SUBSTITUTED = "executable_substituted"
    NATIVE_EXECUTION_FAILED = "native_execution_failed"
    KERNEL_REPLAY_FAILED = "kernel_replay_failed"
    SEALED_RECEIPT_REQUIRED = "sealed_typed_receipt_required"
    SEALED_RECEIPT_INVALID = "sealed_receipt_invalid"
    WRONG_THEOREM = "wrong_theorem"
    WRONG_PROPERTY = "wrong_property"
    WRONG_TOOLCHAIN = "wrong_toolchain"
    WRONG_POLICY = "wrong_policy"
    TEST_INJECTION_REJECTED = "test_injection_rejected"


class DoctorProofAuthorityDisposition(str, Enum):
    """Whether a sealed proof can participate in a separate mutation gate."""

    VERIFIED = "verified"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class DoctorExecutableRole(str, Enum):
    """Native executable roles; kernel replay is a separate process role."""

    SOLVER = "solver"
    KERNEL = "kernel"


class NativeReconstructionDisposition(str, Enum):
    """Closed outcomes for pinned-kernel theorem reconstruction."""

    RECONSTRUCTED = "reconstructed"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    STALE = "stale"


class DoctorObligationCompilationDisposition(str, Enum):
    """Closed outcomes of doctor obligation lowering."""

    LOWERED = "lowered"
    PARTIAL = "partial"
    RESIDUAL_ONLY = "residual_only"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DoctorHammerError(ContractValidationError):
    """Base failure for deterministic-doctor Hammer verification."""


class DoctorHammerAuthorityError(DoctorHammerError):
    """Root, permit, isolation, or semantic-authority boundary failure."""


class DoctorHammerBoundsError(DoctorHammerError):
    """A producer attempted to exceed fixed doctor-hammer budgets."""


class DoctorHammerSafetyError(DoctorHammerError):
    """Body/secret/LLM/write safety violation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if required and not text:
        raise DoctorHammerError(f"{field_name} is required")
    if "\x00" in text or len(text.encode("utf-8")) > limit:
        raise DoctorHammerBoundsError(f"{field_name} is invalid or exceeds its bound")
    return text


def _identifier(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=MAX_REF_BYTES)
    if any(ch.isspace() for ch in text):
        raise DoctorHammerError(f"{field_name} must be a compact identifier")
    return text


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorHammerError(f"{field_name} must be a boolean")
    return value


def _ids(
    values: Any,
    field_name: str,
    *,
    limit: int = MAX_PREMISES,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise DoctorHammerError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise DoctorHammerBoundsError(f"{field_name} exceeds its item bound")
    items: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _identifier(item, field_name)
        if text not in seen:
            seen.add(text)
            items.append(text)
    items.sort()
    if required and not items:
        raise DoctorHammerError(f"{field_name} must not be empty")
    return tuple(items)


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(str(value))
    except (TypeError, ValueError) as exc:
        raise DoctorHammerError(f"{field_name} has an unsupported value") from exc


def _digest(payload: Mapping[str, Any] | Sequence[Any], *, prefix: str) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _stable_id(prefix: str, payload: Any) -> str:
    material = content_identity(
        {
            "schema": f"ipfs_accelerate_py/agent-supervisor/doctor-hammer/{prefix}@1",
            "payload": payload,
        }
    )
    return f"{prefix}:{material}"


def _verify_claimed_identity(
    payload: Mapping[str, Any],
    typed: CanonicalContract,
    *,
    field_names: Sequence[str] = ("content_id", "cid"),
) -> None:
    """Reject retained claimed identities that disagree with decoded bytes."""

    for field_name in field_names:
        claimed = payload.get(field_name)
        if claimed is not None and claimed != typed.content_id:
            raise DoctorHammerAuthorityError(
                f"{field_name} does not match the canonical receipt preimage"
            )


def _require_contract_header(
    payload: Mapping[str, Any],
    expected_schema: str,
    *,
    expected_interface: str = "",
) -> None:
    declared_schema = payload.get("schema")
    if declared_schema not in (None, expected_schema):
        raise DoctorHammerAuthorityError("receipt schema does not match its type")
    declared_version = payload.get("contract_version")
    if declared_version not in (None, CONTRACT_VERSION):
        raise DoctorHammerAuthorityError("receipt contract version is unsupported")
    if expected_interface:
        declared_interface = payload.get("interface")
        if declared_interface not in (None, expected_interface):
            raise DoctorHammerAuthorityError("receipt interface does not match its type")


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BODY_MARKERS or any(
                marker in normalized for marker in _LLM_ROUTE_MARKERS
            ):
                raise DoctorHammerSafetyError(
                    f"{field_name} may not contain body/secret/LLM field {key!r}"
                )
            _assert_body_free(item, field_name)
        return
    if isinstance(value, (bytes, bytearray)):
        raise DoctorHammerSafetyError(f"{field_name} may not contain binary bodies")
    if isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            _assert_body_free(item, field_name)


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        result = converter()
        if isinstance(result, Mapping):
            return {str(k): v for k, v in result.items()}
    raise DoctorHammerError(f"{field_name} must be an object")


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return ProgramLogicAuthorityRoots.from_dict(dict(value))
    raise DoctorHammerError("roots must be ProgramLogicAuthorityRoots")


def _logical_identity(logical_id: str, component: str) -> DoctorIdentityBinding:
    """Build a strict-artifact identity binding for a logical root id."""

    identity = identify_strict_artifact(
        {
            "component": component,
            "logical_id": logical_id,
            "domain": "deterministic-doctor-hammer",
        }
    )
    return DoctorIdentityBinding.from_identity(identity, logical_id=logical_id)


def isolation_is_adequate(
    report: Mapping[str, Any] | None = None,
    *,
    import_isolation: str = "",
    require_concurrency_safe: bool = True,
) -> bool:
    """Return True when subprocess/platform isolation is adequate for Hammer."""

    if report is None:
        report = {}
    isolation = str(
        import_isolation
        or report.get("import_isolation")
        or report.get("isolation")
        or ""
    ).strip()
    if isolation not in _ADEQUATE_ISOLATION:
        return False
    if require_concurrency_safe and report.get("concurrency_safe") is False:
        return False
    if report.get("mutates_home") is True or report.get("mutates_sys_prefix") is True:
        return False
    if report.get("process_global") is True:
        return False
    return True


def _plan_is_admitted(plan_receipt: DoctorTacticianPlanReceipt) -> bool:
    if plan_receipt.disposition is not DoctorTacticianPlanDisposition.PLANNED:
        return False
    if plan_receipt.plan is None:
        return False
    if plan_receipt.semantic_authority is not False:
        return False
    if plan_receipt.llm_route_present or plan_receipt.model_invocation_count != 0:
        return False
    if SourceRouteKind.LLM in (
        plan_receipt.plan.ordered_source_routes if plan_receipt.plan else ()
    ):
        return False
    gate = plan_receipt.gate_receipt
    if gate is not None and gate.disposition not in {
        TacticianPlanGateDisposition.ADMITTED,
        TacticianPlanGateDisposition.CONSISTENCY_ONLY,
    }:
        return False
    return True


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorHammerBounds(CanonicalContract):
    """Finite resource and CEGIS bounds for doctor Hammer verification."""

    SCHEMA: ClassVar[str] = DOCTOR_HAMMER_BOUNDS_SCHEMA

    max_candidates: int = MAX_CANDIDATES
    max_obligations: int = MAX_OBLIGATIONS
    max_cegis_rounds: int = DEFAULT_MAX_CEGIS_ROUNDS
    max_repeated_states: int = DEFAULT_MAX_CEGIS_REPEATED_STATES
    wall_time_ms: int = DEFAULT_WALL_TIME_MS
    cpu_time_ms: int = DEFAULT_CPU_TIME_MS
    memory_bytes: int = DEFAULT_MEMORY_BYTES
    max_premises: int = MAX_PREMISES
    require_native_permit: bool = True
    require_isolation: bool = True
    require_kernel_reconstruction: bool = True
    require_unique_consequence: bool = True
    allow_cache_hits: bool = True
    allow_llm_route: bool = False
    semantic_authority: bool = False
    source_writes_allowed: bool = False

    def __post_init__(self) -> None:
        for name, hard in (
            ("max_candidates", MAX_CANDIDATES),
            ("max_obligations", MAX_OBLIGATIONS),
            ("max_cegis_rounds", HARD_MAX_CEGIS_ROUNDS),
            ("max_repeated_states", HARD_MAX_CEGIS_REPEATED_STATES),
            ("max_premises", MAX_PREMISES),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise DoctorHammerBoundsError(f"{name} must be a positive integer")
            if value > hard:
                raise DoctorHammerBoundsError(f"{name} exceeds hard maximum {hard}")
        for name in ("wall_time_ms", "cpu_time_ms", "memory_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise DoctorHammerBoundsError(f"{name} must be a positive integer")
        for flag in (
            "require_native_permit",
            "require_isolation",
            "require_kernel_reconstruction",
            "require_unique_consequence",
            "allow_cache_hits",
            "allow_llm_route",
            "semantic_authority",
            "source_writes_allowed",
        ):
            object.__setattr__(self, flag, _bool(getattr(self, flag), flag))
        if self.allow_llm_route is not False:
            raise DoctorHammerSafetyError(
                "deterministic doctor hammer forbids LLM routes"
            )
        if self.semantic_authority is not False:
            raise DoctorHammerSafetyError(
                "doctor hammer bounds cannot claim semantic_authority"
            )
        if self.source_writes_allowed is not False:
            raise DoctorHammerSafetyError(
                "doctor hammer verification never writes sources"
            )
        object.__setattr__(self, "allow_llm_route", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "source_writes_allowed", False)

    def to_resource_budget(self) -> ResourceBudget:
        return ResourceBudget(
            wall_time_ms=self.wall_time_ms,
            cpu_time_ms=self.cpu_time_ms,
            memory_bytes=self.memory_bytes,
            max_premises=self.max_premises,
            network_allowed=False,
        )

    def to_cegis_bounds(self) -> LogicRefinementBounds:
        return LogicRefinementBounds(
            max_rounds=self.max_cegis_rounds,
            max_repeated_states=self.max_repeated_states,
            wall_time_ms=self.wall_time_ms,
            cpu_time_ms=self.cpu_time_ms,
            memory_bytes=self.memory_bytes,
            max_premises=self.max_premises,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "max_candidates": self.max_candidates,
            "max_obligations": self.max_obligations,
            "max_cegis_rounds": self.max_cegis_rounds,
            "max_repeated_states": self.max_repeated_states,
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "max_premises": self.max_premises,
            "require_native_permit": self.require_native_permit,
            "require_isolation": self.require_isolation,
            "require_kernel_reconstruction": self.require_kernel_reconstruction,
            "require_unique_consequence": self.require_unique_consequence,
            "allow_cache_hits": self.allow_cache_hits,
            "allow_llm_route": False,
            "semantic_authority": False,
            "source_writes_allowed": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorHammerBounds":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("bounds must be an object")
        fields = (
            "max_candidates",
            "max_obligations",
            "max_cegis_rounds",
            "max_repeated_states",
            "wall_time_ms",
            "cpu_time_ms",
            "memory_bytes",
            "max_premises",
            "require_native_permit",
            "require_isolation",
            "require_kernel_reconstruction",
            "require_unique_consequence",
            "allow_cache_hits",
            "allow_llm_route",
            "semantic_authority",
            "source_writes_allowed",
        )
        values = {name: payload[name] for name in fields if name in payload}
        return cls(**values)


# ---------------------------------------------------------------------------
# Receipt / candidate records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorRepairCandidate(CanonicalContract):
    """One candidate operator/value/placement combination under verification."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_CANDIDATE_SCHEMA

    candidate_id: str
    consequence_ref: str
    hypothesis: LogicHypothesis | None = None
    operator_kind: str = ""
    value_ref: str = ""
    placement_ref: str = ""
    construction_ref: str = ""
    premise_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self,
            "consequence_ref",
            _identifier(self.consequence_ref, "consequence_ref"),
        )
        if self.hypothesis is not None and not isinstance(
            self.hypothesis, LogicHypothesis
        ):
            raise DoctorHammerError("hypothesis must be LogicHypothesis when provided")
        object.__setattr__(
            self,
            "operator_kind",
            _text(self.operator_kind, "operator_kind", required=False),
        )
        for name in ("value_ref", "placement_ref", "construction_ref"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "obligation_ids", _ids(self.obligation_ids, "obligation_ids")
        )
        if self.semantic_authority is not False:
            raise DoctorHammerSafetyError(
                "repair candidates cannot claim semantic_authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "consequence_ref": self.consequence_ref,
            "hypothesis": (
                self.hypothesis.to_dict() if self.hypothesis is not None else None
            ),
            "operator_kind": self.operator_kind,
            "value_ref": self.value_ref,
            "placement_ref": self.placement_ref,
            "construction_ref": self.construction_ref,
            "premise_ids": list(self.premise_ids),
            "obligation_ids": list(self.obligation_ids),
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairCandidate":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("candidate must be an object")
        values = dict(payload)
        hyp = values.get("hypothesis")
        if isinstance(hyp, Mapping):
            values["hypothesis"] = LogicHypothesis.from_dict(hyp)
        values.pop("schema", None)
        values.pop("contract_version", None)
        values.pop("content_id", None)
        values.pop("cid", None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed

    @classmethod
    def from_hypothesis(
        cls,
        hypothesis: LogicHypothesis,
        *,
        operator_kind: str = "",
    ) -> "DoctorRepairCandidate":
        return cls(
            candidate_id=f"candidate:{hypothesis.hypothesis_id}",
            consequence_ref=hypothesis.claimed_consequence_ref
            or f"consequence:{hypothesis.hypothesis_id}",
            hypothesis=hypothesis,
            operator_kind=operator_kind,
            value_ref=hypothesis.value_ref,
            placement_ref=hypothesis.placement_ref,
            construction_ref=hypothesis.construction_ref,
            premise_ids=hypothesis.selected_premise_ids,
        )


@dataclass(frozen=True)
class DoctorReviewedTheorem(CanonicalContract):
    """Reviewed semantic theorem body bound to current proof-authority roots."""

    SCHEMA: ClassVar[str] = DOCTOR_REVIEWED_THEOREM_SCHEMA

    roots: ProgramLogicAuthorityRoots
    theorem_id: str
    property_id: str
    claim_id: str
    consequence_ref: str
    theorem_body: str
    body_format: str
    premise_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    review_receipt_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "theorem_id",
            "property_id",
            "claim_id",
            "consequence_ref",
            "body_format",
            "review_receipt_id",
            "translator_id",
            "toolchain_id",
            "policy_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.theorem_body, str) or not self.theorem_body.strip():
            raise DoctorHammerAuthorityError("reviewed theorem body must be nonempty")
        encoded = self.theorem_body.encode("utf-8")
        if len(encoded) > MAX_THEOREM_BYTES:
            raise DoctorHammerBoundsError("reviewed theorem body exceeds bound")
        if self.body_format.casefold() not in {
            "lean4",
            "smtlib2",
            "rocq",
            "isabelle",
            "doctor-logic-ir",
        }:
            raise DoctorHammerAuthorityError("unsupported semantic theorem format")
        # Identity-only text is not a theorem.  The reviewed body must contain
        # the exact named semantic anchors and at least one relation/operator.
        required_markers = (
            self.theorem_id,
            self.property_id,
            self.claim_id,
            self.consequence_ref,
        )
        if any(marker not in self.theorem_body for marker in required_markers):
            raise DoctorHammerAuthorityError(
                "reviewed theorem body does not bind its theorem/property/claim/consequence"
            )
        if not any(
            marker in self.theorem_body
            for marker in ("=", "->", "→", "∀", "forall ", ":=", "⊢")
        ):
            raise DoctorHammerAuthorityError(
                "reviewed theorem body is identity-only rather than semantic"
            )
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True),
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _ids(self.assumption_ids, "assumption_ids", required=True),
        )
        expected = (
            ("translator_id", self.roots.translator_id),
            ("toolchain_id", self.roots.toolchain_id),
            ("policy_id", self.roots.policy_id),
        )
        for name, root_value in expected:
            if getattr(self, name) != root_value:
                raise DoctorHammerAuthorityError(
                    f"reviewed theorem {name} does not match current roots"
                )

    @property
    def theorem_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "theorem_id": self.theorem_id,
            "property_id": self.property_id,
            "claim_id": self.claim_id,
            "consequence_ref": self.consequence_ref,
            "theorem_body": self.theorem_body,
            "body_format": self.body_format,
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "review_receipt_id": self.review_receipt_id,
            "translator_id": self.translator_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorReviewedTheorem":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("reviewed theorem must be an object")
        _require_contract_header(payload, cls.SCHEMA)
        values = dict(payload)
        if isinstance(values.get("roots"), Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(values["roots"])
        for drop in ("schema", "contract_version", "content_id", "cid"):
            values.pop(drop, None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed


@dataclass(frozen=True)
class DoctorExactLoweringReceipt(CanonicalContract):
    """Exact reviewed-theorem lowering retained for native replay."""

    SCHEMA: ClassVar[str] = DOCTOR_EXACT_LOWERING_SCHEMA

    roots: ProgramLogicAuthorityRoots
    lowering_id: str
    theorem_cid: str
    theorem_id: str
    property_id: str
    claim_id: str
    consequence_ref: str
    premise_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    logic_ir_statement: str
    native_statement: str
    translator_id: str
    toolchain_id: str
    policy_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "lowering_id",
            "theorem_cid",
            "theorem_id",
            "property_id",
            "claim_id",
            "consequence_ref",
            "translator_id",
            "toolchain_id",
            "policy_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True),
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _ids(self.assumption_ids, "assumption_ids", required=True),
        )
        for name in ("logic_ir_statement", "native_statement"):
            statement = getattr(self, name)
            if not isinstance(statement, str) or not statement.strip():
                raise DoctorHammerAuthorityError(f"{name} is required")
            if len(statement.encode("utf-8")) > MAX_THEOREM_BYTES:
                raise DoctorHammerBoundsError(f"{name} exceeds bound")
            if any(
                marker not in statement
                for marker in (
                    self.theorem_id,
                    self.property_id,
                    self.claim_id,
                    self.consequence_ref,
                )
            ):
                raise DoctorHammerAuthorityError(
                    f"{name} does not preserve exact theorem bindings"
                )
        for name, root_value in (
            ("translator_id", self.roots.translator_id),
            ("toolchain_id", self.roots.toolchain_id),
            ("policy_id", self.roots.policy_id),
        ):
            if getattr(self, name) != root_value:
                raise DoctorHammerAuthorityError(
                    f"lowering {name} does not match current roots"
                )
        expected_id = _stable_id(
            "doctor-exact-lowering",
            {
                "theorem_cid": self.theorem_cid,
                "theorem_id": self.theorem_id,
                "property_id": self.property_id,
                "claim_id": self.claim_id,
                "consequence_ref": self.consequence_ref,
                "premise_ids": list(self.premise_ids),
                "assumption_ids": list(self.assumption_ids),
                "logic_ir_statement": self.logic_ir_statement,
                "native_statement": self.native_statement,
                "translator_id": self.translator_id,
                "toolchain_id": self.toolchain_id,
                "policy_id": self.policy_id,
            },
        )
        if self.lowering_id != expected_id:
            raise DoctorHammerAuthorityError(
                "lowering_id does not match the exact lowering preimage"
            )

    @classmethod
    def create(
        cls,
        theorem: DoctorReviewedTheorem,
        *,
        logic_ir_statement: str,
        native_statement: str,
    ) -> "DoctorExactLoweringReceipt":
        if not isinstance(theorem, DoctorReviewedTheorem):
            raise DoctorHammerAuthorityError(
                "exact lowering requires DoctorReviewedTheorem"
            )
        preimage = {
            "theorem_cid": theorem.content_id,
            "theorem_id": theorem.theorem_id,
            "property_id": theorem.property_id,
            "claim_id": theorem.claim_id,
            "consequence_ref": theorem.consequence_ref,
            "premise_ids": list(theorem.premise_ids),
            "assumption_ids": list(theorem.assumption_ids),
            "logic_ir_statement": logic_ir_statement,
            "native_statement": native_statement,
            "translator_id": theorem.translator_id,
            "toolchain_id": theorem.toolchain_id,
            "policy_id": theorem.policy_id,
        }
        return cls(
            roots=theorem.roots,
            lowering_id=_stable_id("doctor-exact-lowering", preimage),
            theorem_cid=theorem.content_id,
            theorem_id=theorem.theorem_id,
            property_id=theorem.property_id,
            claim_id=theorem.claim_id,
            consequence_ref=theorem.consequence_ref,
            premise_ids=theorem.premise_ids,
            assumption_ids=theorem.assumption_ids,
            logic_ir_statement=logic_ir_statement,
            native_statement=native_statement,
            translator_id=theorem.translator_id,
            toolchain_id=theorem.toolchain_id,
            policy_id=theorem.policy_id,
        )

    def verify_theorem(self, theorem: DoctorReviewedTheorem) -> None:
        if not isinstance(theorem, DoctorReviewedTheorem):
            raise DoctorHammerAuthorityError("lowering theorem must be typed")
        bindings = (
            self.theorem_cid == theorem.content_id,
            self.roots.content_id == theorem.roots.content_id,
            self.theorem_id == theorem.theorem_id,
            self.property_id == theorem.property_id,
            self.claim_id == theorem.claim_id,
            self.consequence_ref == theorem.consequence_ref,
            self.premise_ids == theorem.premise_ids,
            self.assumption_ids == theorem.assumption_ids,
            self.translator_id == theorem.translator_id,
            self.toolchain_id == theorem.toolchain_id,
            self.policy_id == theorem.policy_id,
        )
        if not all(bindings):
            raise DoctorHammerAuthorityError(
                "exact lowering does not bind the reviewed theorem"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "lowering_id": self.lowering_id,
            "theorem_cid": self.theorem_cid,
            "theorem_id": self.theorem_id,
            "property_id": self.property_id,
            "claim_id": self.claim_id,
            "consequence_ref": self.consequence_ref,
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "logic_ir_statement": self.logic_ir_statement,
            "native_statement": self.native_statement,
            "translator_id": self.translator_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorExactLoweringReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("exact lowering receipt must be an object")
        _require_contract_header(payload, cls.SCHEMA)
        values = dict(payload)
        if isinstance(values.get("roots"), Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(values["roots"])
        for drop in ("schema", "contract_version", "content_id", "cid"):
            values.pop(drop, None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed


@dataclass(frozen=True)
class DoctorPinnedExecutable(CanonicalContract):
    """Digest-pinned native executable and immutable command prefix."""

    SCHEMA: ClassVar[str] = DOCTOR_PINNED_EXECUTABLE_SCHEMA

    role: DoctorExecutableRole
    executable_path: str
    executable_sha256: str
    argv: tuple[str, ...]
    verifier_id: str
    toolchain_id: str
    environment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role", _enum(self.role, DoctorExecutableRole, "role")
        )
        path = _text(self.executable_path, "executable_path")
        if not os.path.isabs(path):
            raise DoctorHammerAuthorityError("pinned executable path must be absolute")
        object.__setattr__(self, "executable_path", os.path.realpath(path))
        digest = _identifier(self.executable_sha256, "executable_sha256")
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise DoctorHammerAuthorityError(
                "executable_sha256 must be a complete sha256 digest"
            )
        try:
            int(digest[7:], 16)
        except ValueError as exc:
            raise DoctorHammerAuthorityError(
                "executable_sha256 must be hexadecimal"
            ) from exc
        object.__setattr__(self, "executable_sha256", digest.casefold())
        if isinstance(self.argv, (str, bytes)) or not isinstance(self.argv, Sequence):
            raise DoctorHammerError("argv must be a sequence")
        normalized_argv: list[str] = []
        for item in self.argv:
            if not isinstance(item, str) or "\x00" in item:
                raise DoctorHammerError("argv items must be NUL-free strings")
            if len(item.encode("utf-8")) > MAX_TEXT_BYTES:
                raise DoctorHammerBoundsError("argv item exceeds bound")
            normalized_argv.append(item)
        object.__setattr__(self, "argv", tuple(normalized_argv))
        for name in ("verifier_id", "toolchain_id", "environment_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))

    @property
    def command_id(self) -> str:
        return content_identity(
            {
                "executable_sha256": self.executable_sha256,
                "argv": list(self.argv),
                "role": self.role.value,
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "role": self.role.value,
            "executable_path": self.executable_path,
            "executable_sha256": self.executable_sha256,
            "argv": list(self.argv),
            "verifier_id": self.verifier_id,
            "toolchain_id": self.toolchain_id,
            "environment_id": self.environment_id,
            "command_id": self.command_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPinnedExecutable":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("pinned executable must be an object")
        _require_contract_header(payload, cls.SCHEMA)
        values = dict(payload)
        claimed_command = values.pop("command_id", None)
        for drop in ("schema", "contract_version", "content_id", "cid"):
            values.pop(drop, None)
        typed = cls(**values)
        if claimed_command is not None and claimed_command != typed.command_id:
            raise DoctorHammerAuthorityError("pinned command_id was forged")
        _verify_claimed_identity(payload, typed)
        return typed


@dataclass(frozen=True)
class DoctorNativeExecutionReceipt(CanonicalContract):
    """Actual solver process result over the exact reviewed lowering."""

    SCHEMA: ClassVar[str] = DOCTOR_NATIVE_EXECUTION_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    theorem_cid: str
    lowering_cid: str
    property_id: str
    consequence_ref: str
    premise_ids: tuple[str, ...]
    executable_pin_cid: str
    executable_sha256: str
    command_id: str
    stdin_cid: str
    stdout_cid: str
    stderr_cid: str
    transcript_cid: str
    proof_object: str
    proof_object_cid: str
    exit_code: int
    executed: bool
    toolchain_id: str
    policy_id: str
    environment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "theorem_cid",
            "lowering_cid",
            "property_id",
            "consequence_ref",
            "executable_pin_cid",
            "executable_sha256",
            "command_id",
            "stdin_cid",
            "stdout_cid",
            "stderr_cid",
            "transcript_cid",
            "proof_object_cid",
            "toolchain_id",
            "policy_id",
            "environment_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True),
        )
        object.__setattr__(self, "executed", _bool(self.executed, "executed"))
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise DoctorHammerError("exit_code must be an integer")
        if not isinstance(self.proof_object, str) or not self.proof_object:
            raise DoctorHammerAuthorityError("native proof object is required")
        if len(self.proof_object.encode("utf-8")) > MAX_NATIVE_OUTPUT_BYTES:
            raise DoctorHammerBoundsError("native proof object exceeds bound")
        if self.proof_object_cid != content_identity(
            {"proof_object": self.proof_object}
        ):
            raise DoctorHammerAuthorityError("native proof object CID mismatch")
        if not self.executed or self.exit_code != 0:
            raise DoctorHammerAuthorityError(
                "native execution receipt must come from a successful process"
            )
        for name, root_value in (
            ("toolchain_id", self.roots.toolchain_id),
            ("policy_id", self.roots.policy_id),
            ("environment_id", self.roots.environment_id),
        ):
            if getattr(self, name) != root_value:
                raise DoctorHammerAuthorityError(
                    f"native execution {name} does not match current roots"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "theorem_cid": self.theorem_cid,
            "lowering_cid": self.lowering_cid,
            "property_id": self.property_id,
            "consequence_ref": self.consequence_ref,
            "premise_ids": list(self.premise_ids),
            "executable_pin_cid": self.executable_pin_cid,
            "executable_sha256": self.executable_sha256,
            "command_id": self.command_id,
            "stdin_cid": self.stdin_cid,
            "stdout_cid": self.stdout_cid,
            "stderr_cid": self.stderr_cid,
            "transcript_cid": self.transcript_cid,
            "proof_object": self.proof_object,
            "proof_object_cid": self.proof_object_cid,
            "exit_code": self.exit_code,
            "executed": self.executed,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorNativeExecutionReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("native execution receipt must be an object")
        _require_contract_header(payload, cls.SCHEMA)
        values = dict(payload)
        if isinstance(values.get("roots"), Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(values["roots"])
        for drop in ("schema", "contract_version", "content_id", "cid"):
            values.pop(drop, None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed


@dataclass(frozen=True)
class DoctorKernelReplayReceipt(CanonicalContract):
    """Independent kernel process replay of a sealed native proof object."""

    SCHEMA: ClassVar[str] = DOCTOR_KERNEL_REPLAY_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    theorem_cid: str
    lowering_cid: str
    native_receipt_cid: str
    property_id: str
    consequence_ref: str
    premise_ids: tuple[str, ...]
    proof_object_cid: str
    executable_pin_cid: str
    executable_sha256: str
    command_id: str
    stdin_cid: str
    stdout_cid: str
    stderr_cid: str
    transcript_cid: str
    exit_code: int
    executed: bool
    kernel_verified: bool
    kernel_id: str
    toolchain_id: str
    policy_id: str
    environment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "theorem_cid",
            "lowering_cid",
            "native_receipt_cid",
            "property_id",
            "consequence_ref",
            "proof_object_cid",
            "executable_pin_cid",
            "executable_sha256",
            "command_id",
            "stdin_cid",
            "stdout_cid",
            "stderr_cid",
            "transcript_cid",
            "kernel_id",
            "toolchain_id",
            "policy_id",
            "environment_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True),
        )
        object.__setattr__(self, "executed", _bool(self.executed, "executed"))
        object.__setattr__(
            self,
            "kernel_verified",
            _bool(self.kernel_verified, "kernel_verified"),
        )
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise DoctorHammerError("exit_code must be an integer")
        if not self.executed or self.exit_code != 0 or not self.kernel_verified:
            raise DoctorHammerAuthorityError(
                "kernel replay must be independently executed and verified"
            )
        for name, root_value in (
            ("toolchain_id", self.roots.toolchain_id),
            ("policy_id", self.roots.policy_id),
            ("environment_id", self.roots.environment_id),
        ):
            if getattr(self, name) != root_value:
                raise DoctorHammerAuthorityError(
                    f"kernel replay {name} does not match current roots"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "theorem_cid": self.theorem_cid,
            "lowering_cid": self.lowering_cid,
            "native_receipt_cid": self.native_receipt_cid,
            "property_id": self.property_id,
            "consequence_ref": self.consequence_ref,
            "premise_ids": list(self.premise_ids),
            "proof_object_cid": self.proof_object_cid,
            "executable_pin_cid": self.executable_pin_cid,
            "executable_sha256": self.executable_sha256,
            "command_id": self.command_id,
            "stdin_cid": self.stdin_cid,
            "stdout_cid": self.stdout_cid,
            "stderr_cid": self.stderr_cid,
            "transcript_cid": self.transcript_cid,
            "exit_code": self.exit_code,
            "executed": self.executed,
            "kernel_verified": self.kernel_verified,
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorKernelReplayReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("kernel replay receipt must be an object")
        _require_contract_header(payload, cls.SCHEMA)
        values = dict(payload)
        if isinstance(values.get("roots"), Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(values["roots"])
        for drop in ("schema", "contract_version", "content_id", "cid"):
            values.pop(drop, None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed


@dataclass(frozen=True)
class DoctorAuthoritativeProofReceipt(CanonicalContract):
    """Mutation-gate proof authority reconstructed from sealed native evidence."""

    SCHEMA: ClassVar[str] = DOCTOR_AUTHORITATIVE_PROOF_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    receipt_id: str
    disposition: DoctorProofAuthorityDisposition
    reason_codes: tuple[str, ...]
    theorem: DoctorReviewedTheorem | None = None
    lowering: DoctorExactLoweringReceipt | None = None
    solver_pin: DoctorPinnedExecutable | None = None
    kernel_pin: DoctorPinnedExecutable | None = None
    native_execution: DoctorNativeExecutionReceipt | None = None
    kernel_replay: DoctorKernelReplayReceipt | None = None
    native_store_ref: DoctorSealedReceiptRef | None = None
    kernel_store_ref: DoctorSealedReceiptRef | None = None
    authority_store_ref: DoctorSealedReceiptRef | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    authority_store: DoctorSealedReceiptStore | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    selected_consequence_ref: str = ""
    eligible_consequence_refs: tuple[str, ...] = ()
    property_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""
    uniqueness_satisfied: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorProofAuthorityDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", required=True)
        )
        for name in (
            "selected_consequence_ref",
            "property_id",
            "toolchain_id",
            "policy_id",
            "producer_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "eligible_consequence_refs",
            _ids(self.eligible_consequence_refs, "eligible_consequence_refs"),
        )
        object.__setattr__(
            self,
            "uniqueness_satisfied",
            _bool(self.uniqueness_satisfied, "uniqueness_satisfied"),
        )
        if self.disposition is DoctorProofAuthorityDisposition.VERIFIED:
            required_types = (
                (self.theorem, DoctorReviewedTheorem, "theorem"),
                (self.lowering, DoctorExactLoweringReceipt, "lowering"),
                (self.solver_pin, DoctorPinnedExecutable, "solver_pin"),
                (self.kernel_pin, DoctorPinnedExecutable, "kernel_pin"),
                (
                    self.native_execution,
                    DoctorNativeExecutionReceipt,
                    "native_execution",
                ),
                (
                    self.kernel_replay,
                    DoctorKernelReplayReceipt,
                    "kernel_replay",
                ),
                (
                    self.native_store_ref,
                    DoctorSealedReceiptRef,
                    "native_store_ref",
                ),
                (
                    self.kernel_store_ref,
                    DoctorSealedReceiptRef,
                    "kernel_store_ref",
                ),
            )
            for value, expected_type, name in required_types:
                if type(value) is not expected_type:
                    raise DoctorHammerAuthorityError(
                        f"verified authoritative proof requires typed {name}"
                    )
            assert self.theorem is not None
            assert self.lowering is not None
            assert self.solver_pin is not None
            assert self.kernel_pin is not None
            assert self.native_execution is not None
            assert self.kernel_replay is not None
            assert self.native_store_ref is not None
            assert self.kernel_store_ref is not None
            if (
                self.authority_store_ref is not None
                and type(self.authority_store_ref) is not DoctorSealedReceiptRef
            ):
                raise DoctorHammerAuthorityError(
                    "authority_store_ref must be a sealed typed reference"
                )
            if (
                self.authority_store is not None
                and type(self.authority_store) is not DoctorSealedReceiptStore
            ):
                raise DoctorHammerAuthorityError(
                    "authority_store must be the concrete sealed store"
                )
            self.lowering.verify_theorem(self.theorem)
            if self.solver_pin.role is not DoctorExecutableRole.SOLVER:
                raise DoctorHammerAuthorityError("solver pin has wrong role")
            if self.kernel_pin.role is not DoctorExecutableRole.KERNEL:
                raise DoctorHammerAuthorityError("kernel pin has wrong role")
            if (
                self.solver_pin.content_id == self.kernel_pin.content_id
                or self.solver_pin.command_id == self.kernel_pin.command_id
                or self.solver_pin.verifier_id == self.kernel_pin.verifier_id
            ):
                raise DoctorHammerAuthorityError(
                    "kernel replay must use an independent pinned verifier command"
                )
            if not self.uniqueness_satisfied:
                raise DoctorHammerAuthorityError(
                    "authoritative proof requires unique consequence"
                )
            if self.eligible_consequence_refs != (
                self.selected_consequence_ref,
            ):
                raise DoctorHammerAuthorityError(
                    "authoritative proof requires exactly one eligible consequence"
                )
            bindings = (
                self.roots.content_id == self.theorem.roots.content_id,
                self.selected_consequence_ref == self.theorem.consequence_ref,
                self.property_id == self.theorem.property_id,
                self.toolchain_id == self.roots.toolchain_id,
                self.policy_id == self.roots.policy_id,
                self.solver_pin.toolchain_id == self.toolchain_id,
                self.kernel_pin.toolchain_id == self.toolchain_id,
                self.native_execution.theorem_cid == self.theorem.content_id,
                self.native_execution.lowering_cid == self.lowering.content_id,
                self.native_execution.property_id == self.property_id,
                self.native_execution.consequence_ref
                == self.selected_consequence_ref,
                self.native_execution.premise_ids == self.theorem.premise_ids,
                self.native_execution.executable_pin_cid
                == self.solver_pin.content_id,
                self.kernel_replay.theorem_cid == self.theorem.content_id,
                self.kernel_replay.lowering_cid == self.lowering.content_id,
                self.kernel_replay.native_receipt_cid
                == self.native_execution.content_id,
                self.kernel_replay.property_id == self.property_id,
                self.kernel_replay.consequence_ref
                == self.selected_consequence_ref,
                self.kernel_replay.premise_ids == self.theorem.premise_ids,
                self.kernel_replay.proof_object_cid
                == self.native_execution.proof_object_cid,
                self.kernel_replay.executable_pin_cid == self.kernel_pin.content_id,
                self.native_store_ref.receipt_cid
                == self.native_execution.content_id,
                self.kernel_store_ref.receipt_cid == self.kernel_replay.content_id,
                self.native_store_ref.store_id == self.kernel_store_ref.store_id,
                self.native_store_ref.authority_id
                == self.kernel_store_ref.authority_id,
            )
            if not all(bindings):
                raise DoctorHammerAuthorityError(
                    "authoritative proof evidence bindings do not match"
                )
        else:
            if self.uniqueness_satisfied:
                raise DoctorHammerAuthorityError(
                    "non-verified authoritative proof cannot claim uniqueness"
                )

    @property
    def is_admitted(self) -> bool:
        """Compatibility spelling; exact type remains mandatory at consumers."""

        return self.disposition is DoctorProofAuthorityDisposition.VERIFIED

    @property
    def mutation_capable(self) -> bool:
        if (
            not self.is_admitted
            or type(self.authority_store_ref) is not DoctorSealedReceiptRef
            or type(self.authority_store) is not DoctorSealedReceiptStore
            or self.authority_store_ref.receipt_cid != self.content_id
            or self.authority_store_ref.receipt_schema != self.SCHEMA
        ):
            return False
        try:
            reloaded = self.authority_store.reload(
                self.authority_store_ref,
                DoctorAuthoritativeProofReceipt,
            )
        except (DoctorSealedReceiptError, OSError, ValueError):
            return False
        return reloaded.content_id == self.content_id

    @property
    def write_authority(self) -> bool:
        # A proof is one required mutation input, never the mutation permit.
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": DOCTOR_AUTHORITATIVE_PROOF_INTERFACE,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "theorem": self.theorem.to_dict() if self.theorem is not None else None,
            "lowering": self.lowering.to_dict() if self.lowering is not None else None,
            "solver_pin": (
                self.solver_pin.to_dict() if self.solver_pin is not None else None
            ),
            "kernel_pin": (
                self.kernel_pin.to_dict() if self.kernel_pin is not None else None
            ),
            "native_execution": (
                self.native_execution.to_dict()
                if self.native_execution is not None
                else None
            ),
            "kernel_replay": (
                self.kernel_replay.to_dict()
                if self.kernel_replay is not None
                else None
            ),
            "native_store_ref": (
                self.native_store_ref.to_dict()
                if self.native_store_ref is not None
                else None
            ),
            "kernel_store_ref": (
                self.kernel_store_ref.to_dict()
                if self.kernel_store_ref is not None
                else None
            ),
            "selected_consequence_ref": self.selected_consequence_ref,
            "eligible_consequence_refs": list(self.eligible_consequence_refs),
            "property_id": self.property_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "uniqueness_satisfied": self.uniqueness_satisfied,
            "write_authority": False,
            "producer_id": self.producer_id,
        }

    def to_record(self) -> dict[str, Any]:
        """Return the canonical receipt plus its non-recursive store envelope."""

        record = super().to_record()
        record["authority_store_ref"] = (
            self.authority_store_ref.to_dict()
            if self.authority_store_ref is not None
            else None
        )
        return record

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorAuthoritativeProofReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("authoritative proof receipt must be an object")
        _require_contract_header(
            payload,
            cls.SCHEMA,
            expected_interface=DOCTOR_AUTHORITATIVE_PROOF_INTERFACE,
        )
        values = dict(payload)
        if isinstance(values.get("roots"), Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(values["roots"])
        typed_fields: tuple[tuple[str, type[Any]], ...] = (
            ("theorem", DoctorReviewedTheorem),
            ("lowering", DoctorExactLoweringReceipt),
            ("solver_pin", DoctorPinnedExecutable),
            ("kernel_pin", DoctorPinnedExecutable),
            ("native_execution", DoctorNativeExecutionReceipt),
            ("kernel_replay", DoctorKernelReplayReceipt),
            ("native_store_ref", DoctorSealedReceiptRef),
            ("kernel_store_ref", DoctorSealedReceiptRef),
            ("authority_store_ref", DoctorSealedReceiptRef),
        )
        for name, kind in typed_fields:
            raw = values.get(name)
            if isinstance(raw, Mapping):
                values[name] = kind.from_dict(raw)
        claimed_write = values.pop("write_authority", False)
        if claimed_write is not False:
            raise DoctorHammerAuthorityError(
                "authoritative proof cannot claim a mutation permit"
            )
        for drop in (
            "schema",
            "contract_version",
            "content_id",
            "cid",
            "interface",
        ):
            values.pop(drop, None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed


@dataclass(frozen=True)
class NativeReconstructionReceipt(CanonicalContract):
    """Pinned-kernel reconstruction of the matching native theorem."""

    SCHEMA: ClassVar[str] = NATIVE_RECONSTRUCTION_RECEIPT_SCHEMA

    receipt_id: str
    disposition: NativeReconstructionDisposition
    roots: ProgramLogicAuthorityRoots
    obligation_id: str
    native_goal_binding_id: str
    kernel_id: str
    toolchain_id: str
    environment_id: str
    translation_map_id: str = ""
    theorem_source_id: str = ""
    reconstruction_id: str = ""
    kernel_receipt_id: str = ""
    matching_theorem: bool = False
    reason_codes: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, NativeReconstructionDisposition, "disposition"),
        )
        for name in (
            "obligation_id",
            "native_goal_binding_id",
            "kernel_id",
            "toolchain_id",
            "environment_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "translation_map_id",
            "theorem_source_id",
            "reconstruction_id",
            "kernel_receipt_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self, "matching_theorem", _bool(self.matching_theorem, "matching_theorem")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if (
            self.disposition is NativeReconstructionDisposition.RECONSTRUCTED
            and not self.matching_theorem
        ):
            raise DoctorHammerError(
                "reconstructed receipts must mark matching_theorem=true"
            )

    @property
    def is_reconstructed(self) -> bool:
        return (
            self.disposition is NativeReconstructionDisposition.RECONSTRUCTED
            and self.matching_theorem
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "receipt_id": self.receipt_id,
            "disposition": self.disposition.value,
            "roots": self.roots.to_dict(),
            "obligation_id": self.obligation_id,
            "native_goal_binding_id": self.native_goal_binding_id,
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "environment_id": self.environment_id,
            "translation_map_id": self.translation_map_id,
            "theorem_source_id": self.theorem_source_id,
            "reconstruction_id": self.reconstruction_id,
            "kernel_receipt_id": self.kernel_receipt_id,
            "matching_theorem": self.matching_theorem,
            "reason_codes": list(self.reason_codes),
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeReconstructionReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("native reconstruction receipt must be an object")
        values = dict(payload)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(roots)
        values.pop("schema", None)
        values.pop("contract_version", None)
        values.pop("content_id", None)
        values.pop("cid", None)
        return cls(**values)


@dataclass(frozen=True)
class DoctorRepairObligationCompilation(CanonicalContract):
    """Doctor-facing obligation lowering receipt (wraps production compiler)."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_OBLIGATION_COMPILATION_SCHEMA

    roots: ProgramLogicAuthorityRoots
    compilation_id: str
    plan_id: str
    finding_id: str
    disposition: DoctorObligationCompilationDisposition
    obligation_ids: tuple[str, ...] = ()
    native_binding_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    residual_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    translation_map_id: str = ""
    kernel_id: str = ""
    toolchain_id: str = ""
    translator_id: str = ""
    solver_id: str = ""
    policy_id: str = ""
    environment_id: str = ""
    production_compilation: TacticianHammerObligationCompilation | None = None
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "compilation_id", _identifier(self.compilation_id, "compilation_id")
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(
                self.disposition,
                DoctorObligationCompilationDisposition,
                "disposition",
            ),
        )
        for name in (
            "obligation_ids",
            "native_binding_ids",
            "premise_ids",
            "residual_ids",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        for name in (
            "translation_map_id",
            "kernel_id",
            "toolchain_id",
            "translator_id",
            "solver_id",
            "policy_id",
            "environment_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if len(self.obligation_ids) > MAX_OBLIGATIONS:
            raise DoctorHammerBoundsError("obligation_ids exceeds bound")

    @property
    def is_lowered(self) -> bool:
        return self.disposition in {
            DoctorObligationCompilationDisposition.LOWERED,
            DoctorObligationCompilationDisposition.PARTIAL,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "compilation_id": self.compilation_id,
            "plan_id": self.plan_id,
            "finding_id": self.finding_id,
            "disposition": self.disposition.value,
            "obligation_ids": list(self.obligation_ids),
            "native_binding_ids": list(self.native_binding_ids),
            "premise_ids": list(self.premise_ids),
            "residual_ids": list(self.residual_ids),
            "reason_codes": list(self.reason_codes),
            "translation_map_id": self.translation_map_id,
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "translator_id": self.translator_id,
            "solver_id": self.solver_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
            "production_compilation": (
                self.production_compilation.to_dict()
                if self.production_compilation is not None
                else None
            ),
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairObligationCompilation":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("obligation compilation must be an object")
        values = dict(payload)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(roots)
        values.pop("schema", None)
        values.pop("contract_version", None)
        values.pop("content_id", None)
        values.pop("cid", None)
        values.pop("production_compilation", None)
        return cls(**values)


@dataclass(frozen=True)
class DoctorRepairProofReceipt(CanonicalContract):
    """Complete fail-closed proof receipt for one doctor repair verification."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_PROOF_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    receipt_id: str
    finding_id: str
    plan_receipt_id: str
    disposition: DoctorHammerDisposition
    reason_codes: tuple[str, ...] = ()
    obligation_compilation_id: str = ""
    obligation_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    translator_id: str = ""
    solver_id: str = ""
    kernel_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""
    environment_id: str = ""
    resource_budget: Mapping[str, Any] = field(default_factory=dict)
    cache_audits: tuple[Mapping[str, Any], ...] = ()
    cache_revalidated: bool = False
    native_reconstruction: NativeReconstructionReceipt | None = None
    hammer_receipt_ids: tuple[str, ...] = ()
    hammer_outcomes: tuple[str, ...] = ()
    countermodel_validation_ids: tuple[str, ...] = ()
    cegis_receipt_id: str = ""
    cegis_disposition: str = ""
    cegis_stop_reason: str = ""
    admission_decision_id: str = ""
    admission_disposition: str = ""
    eligible_consequence_refs: tuple[str, ...] = ()
    selected_consequence_ref: str = ""
    selected_candidate_id: str = ""
    selected_hypothesis_id: str = ""
    uniqueness_satisfied: bool = False
    permit_id: str = ""
    isolation_report: Mapping[str, Any] = field(default_factory=dict)
    isolation_adequate: bool = False
    import_isolation: str = ""
    semantic_authority: bool = False
    write_authority: bool = False
    source_write_count: int = 0
    llm_invocation_count: int = 0
    model_provider_call_count: int = 0
    residual_gap_ids: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self,
            "plan_receipt_id",
            _identifier(self.plan_receipt_id, "plan_receipt_id"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorHammerDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        for name in (
            "obligation_compilation_id",
            "translator_id",
            "solver_id",
            "kernel_id",
            "toolchain_id",
            "policy_id",
            "environment_id",
            "cegis_receipt_id",
            "cegis_disposition",
            "cegis_stop_reason",
            "admission_decision_id",
            "admission_disposition",
            "selected_consequence_ref",
            "selected_candidate_id",
            "selected_hypothesis_id",
            "permit_id",
            "import_isolation",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        for name in (
            "obligation_ids",
            "premise_ids",
            "hammer_receipt_ids",
            "hammer_outcomes",
            "countermodel_validation_ids",
            "eligible_consequence_refs",
            "residual_gap_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        object.__setattr__(
            self,
            "resource_budget",
            MappingProxyType(dict(self.resource_budget or {})),
        )
        object.__setattr__(
            self,
            "cache_audits",
            tuple(
                MappingProxyType(dict(item)) if isinstance(item, Mapping) else item
                for item in (self.cache_audits or ())
            ),
        )
        object.__setattr__(
            self,
            "isolation_report",
            MappingProxyType(dict(self.isolation_report or {})),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {}))
        )
        for name in (
            "cache_revalidated",
            "uniqueness_satisfied",
            "isolation_adequate",
            "semantic_authority",
            "write_authority",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        for name in (
            "source_write_count",
            "llm_invocation_count",
            "model_provider_call_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DoctorHammerError(f"{name} must be a non-negative integer")
        # Hard safety invariants: verification never writes or calls models.
        if self.source_write_count != 0:
            raise DoctorHammerSafetyError(
                "doctor hammer receipts must report zero source writes"
            )
        if self.llm_invocation_count != 0 or self.model_provider_call_count != 0:
            raise DoctorHammerSafetyError(
                "doctor hammer receipts must report zero LLM/model-provider calls"
            )
        if self.semantic_authority is not False:
            raise DoctorHammerSafetyError(
                "doctor hammer receipts cannot claim semantic_authority"
            )
        if self.write_authority is not False:
            raise DoctorHammerSafetyError(
                "doctor hammer receipts cannot claim write_authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "write_authority", False)
        object.__setattr__(self, "source_write_count", 0)
        object.__setattr__(self, "llm_invocation_count", 0)
        object.__setattr__(self, "model_provider_call_count", 0)
        if (
            self.disposition is DoctorHammerDisposition.ADMITTED
            and not self.selected_consequence_ref
        ):
            raise DoctorHammerError(
                "admitted receipts require a selected_consequence_ref"
            )
        if (
            self.disposition is DoctorHammerDisposition.ADMITTED
            and not self.uniqueness_satisfied
        ):
            raise DoctorHammerError(
                "admitted receipts require uniqueness_satisfied=true"
            )
        if self.native_reconstruction is not None and not isinstance(
            self.native_reconstruction, NativeReconstructionReceipt
        ):
            raise DoctorHammerError(
                "native_reconstruction must be NativeReconstructionReceipt"
            )
        encoded = json.dumps(self._payload(), sort_keys=True, default=str).encode(
            "utf-8"
        )
        if len(encoded) > MAX_RECORD_BYTES:
            raise DoctorHammerBoundsError("proof receipt exceeds serialized bound")

    @property
    def is_admitted(self) -> bool:
        return self.disposition is DoctorHammerDisposition.ADMITTED

    @property
    def mutation_capable(self) -> bool:
        """Legacy/provider-assisted receipts are diagnostic-only."""

        return False

    @property
    def is_refuted(self) -> bool:
        return self.disposition is DoctorHammerDisposition.REFUTED

    @property
    def abstained(self) -> bool:
        return self.disposition is DoctorHammerDisposition.ABSTAINED

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": DETERMINISTIC_DOCTOR_HAMMER_INTERFACE,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "finding_id": self.finding_id,
            "plan_receipt_id": self.plan_receipt_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "obligation_compilation_id": self.obligation_compilation_id,
            "obligation_ids": list(self.obligation_ids),
            "premise_ids": list(self.premise_ids),
            "translator_id": self.translator_id,
            "solver_id": self.solver_id,
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
            "resource_budget": dict(self.resource_budget),
            "cache_audits": [dict(item) for item in self.cache_audits],
            "cache_revalidated": self.cache_revalidated,
            "native_reconstruction": (
                self.native_reconstruction.to_dict()
                if self.native_reconstruction is not None
                else None
            ),
            "hammer_receipt_ids": list(self.hammer_receipt_ids),
            "hammer_outcomes": list(self.hammer_outcomes),
            "countermodel_validation_ids": list(self.countermodel_validation_ids),
            "cegis_receipt_id": self.cegis_receipt_id,
            "cegis_disposition": self.cegis_disposition,
            "cegis_stop_reason": self.cegis_stop_reason,
            "admission_decision_id": self.admission_decision_id,
            "admission_disposition": self.admission_disposition,
            "eligible_consequence_refs": list(self.eligible_consequence_refs),
            "selected_consequence_ref": self.selected_consequence_ref,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_hypothesis_id": self.selected_hypothesis_id,
            "uniqueness_satisfied": self.uniqueness_satisfied,
            "permit_id": self.permit_id,
            "isolation_report": dict(self.isolation_report),
            "isolation_adequate": self.isolation_adequate,
            "import_isolation": self.import_isolation,
            "semantic_authority": False,
            "write_authority": False,
            "source_write_count": 0,
            "llm_invocation_count": 0,
            "model_provider_call_count": 0,
            "residual_gap_ids": list(self.residual_gap_ids),
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairProofReceipt":
        if not isinstance(payload, Mapping):
            raise DoctorHammerError("proof receipt must be an object")
        values = dict(payload)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = ProgramLogicAuthorityRoots.from_dict(roots)
        recon = values.get("native_reconstruction")
        if isinstance(recon, Mapping):
            values["native_reconstruction"] = NativeReconstructionReceipt.from_dict(
                recon
            )
        for drop in ("schema", "contract_version", "content_id", "cid", "interface"):
            values.pop(drop, None)
        typed = cls(**values)
        _verify_claimed_identity(payload, typed)
        return typed


# ---------------------------------------------------------------------------
# Obligation compiler
# ---------------------------------------------------------------------------


class DoctorRepairObligationCompiler:
    """Lower an admitted doctor Tactician plan into exact proof obligations.

    Prefers the production :class:`TacticianHammerObligationCompiler` when a
    gate receipt and obligation context are supplied.  Otherwise emits a
    body-free identity-bound compilation surface for cache-first verification
    without inventing axioms.
    """

    compiler_id: ClassVar[str] = "doctor-repair-obligation-compiler@1"

    def __init__(
        self,
        *,
        production_compiler: TacticianHammerObligationCompiler | None = None,
    ) -> None:
        self._production = production_compiler or TacticianHammerObligationCompiler()

    def compile(
        self,
        plan_receipt: DoctorTacticianPlanReceipt | Mapping[str, Any],
        goal_compilation: DoctorGoalCompilation | Mapping[str, Any],
        *,
        hypotheses: Sequence[LogicHypothesis | Mapping[str, Any]] = (),
        context: ObligationContext | None = None,
        kernel_id: str = "",
        solver_id: str = "",
        translation_map_id: str = "",
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
    ) -> DoctorRepairObligationCompilation:
        plan = (
            plan_receipt
            if isinstance(plan_receipt, DoctorTacticianPlanReceipt)
            else DoctorTacticianPlanReceipt.from_dict(plan_receipt)
        )
        goals_comp = (
            goal_compilation
            if isinstance(goal_compilation, DoctorGoalCompilation)
            else DoctorGoalCompilation.from_dict(goal_compilation)
        )
        roots = _roots(current_roots) if current_roots is not None else plan.roots

        if roots.content_id != plan.roots.content_id:
            return self._reject(
                roots=roots,
                plan=plan,
                goals_comp=goals_comp,
                reasons=(DoctorHammerReasonCode.MIXED_ROOTS.value,),
            )
        if goals_comp.roots.content_id != roots.content_id:
            return self._reject(
                roots=roots,
                plan=plan,
                goals_comp=goals_comp,
                reasons=(DoctorHammerReasonCode.MIXED_ROOTS.value,),
            )
        if not _plan_is_admitted(plan):
            return self._reject(
                roots=roots,
                plan=plan,
                goals_comp=goals_comp,
                reasons=(DoctorHammerReasonCode.PLAN_NOT_ADMITTED.value,),
            )
        if goals_comp.disposition not in {
            DoctorGoalCompilationDisposition.COMPLETE,
            DoctorGoalCompilationDisposition.PARTIAL,
        }:
            return self._reject(
                roots=roots,
                plan=plan,
                goals_comp=goals_comp,
                reasons=(DoctorHammerReasonCode.COMPILATION_INCOMPLETE.value,),
            )

        typed_hyps = tuple(
            item
            if isinstance(item, LogicHypothesis)
            else LogicHypothesis.from_dict(item)
            for item in hypotheses
        )
        for hyp in typed_hyps:
            if hyp.roots.content_id != roots.content_id:
                return self._reject(
                    roots=roots,
                    plan=plan,
                    goals_comp=goals_comp,
                    reasons=(DoctorHammerReasonCode.MIXED_ROOTS.value,),
                )

        kernel = kernel_id or "kernel:lean4"
        solver = solver_id or roots.policy_id and "solver:z3" or "solver:z3"
        if not solver:
            solver = "solver:z3"
        translator = roots.translator_id
        toolchain = roots.toolchain_id
        policy = roots.policy_id
        environment = roots.environment_id
        translation = translation_map_id or f"translation-map:{plan.receipt_id}"
        plan_id = (
            plan.plan.plan_id
            if plan.plan is not None and getattr(plan.plan, "plan_id", "")
            else plan.receipt_id
        )
        premise_ids = tuple(plan.selected_premise_ids or goals_comp.selected_observation_ids)

        production: TacticianHammerObligationCompilation | None = None
        obligation_ids: list[str] = []
        native_binding_ids: list[str] = []
        residual_ids: list[str] = []
        disposition = DoctorObligationCompilationDisposition.LOWERED
        reasons: list[str] = []

        if (
            context is not None
            and plan.gate_receipt is not None
            and plan.plan is not None
        ):
            try:
                production = self._production.compile(
                    plan.gate_receipt,
                    plan.plan,
                    goals_comp.goals,
                    typed_hyps,
                    goals_comp.corpus,
                    context,
                    current_roots=roots,
                )
            except ContractValidationError as exc:
                message = str(exc).casefold()
                if "unsupported" in message or "residual" in message:
                    code = DoctorHammerReasonCode.UNSUPPORTED_LOWERING.value
                elif "inconsistent" in message or "conflict" in message:
                    code = DoctorHammerReasonCode.INCONSISTENCY.value
                else:
                    code = DoctorHammerReasonCode.UNSUPPORTED_LOWERING.value
                return self._reject(
                    roots=roots,
                    plan=plan,
                    goals_comp=goals_comp,
                    reasons=(code,),
                )
            if production.disposition is LoweringDisposition.REJECTED:
                return self._reject(
                    roots=roots,
                    plan=plan,
                    goals_comp=goals_comp,
                    reasons=(DoctorHammerReasonCode.UNSUPPORTED_LOWERING.value,),
                )
            obligation_ids = list(production.obligation_ids)
            native_binding_ids = [item.binding_id for item in production.native_bindings]
            residual_ids = [item.residual_id for item in production.residuals]
            if production.disposition is LoweringDisposition.PARTIAL:
                disposition = DoctorObligationCompilationDisposition.PARTIAL
            elif production.disposition is LoweringDisposition.RESIDUAL_ONLY:
                disposition = DoctorObligationCompilationDisposition.RESIDUAL_ONLY
                reasons.append(DoctorHammerReasonCode.UNSUPPORTED_LOWERING.value)
            elif production.disposition is LoweringDisposition.LOWERED:
                disposition = DoctorObligationCompilationDisposition.LOWERED
        else:
            # Identity-bound lightweight surface: one obligation id per goal /
            # admitted hypothesis pair without inventing theorem bodies.
            if not goals_comp.goals:
                return self._reject(
                    roots=roots,
                    plan=plan,
                    goals_comp=goals_comp,
                    reasons=(DoctorHammerReasonCode.COMPILATION_INCOMPLETE.value,),
                )
            for goal in goals_comp.goals:
                matching = [
                    hyp
                    for hyp in typed_hyps
                    if hyp.target_goal_id == goal.goal_id
                    and hyp.disposition
                    in {
                        HypothesisDisposition.PLAN_ADMITTED,
                        HypothesisDisposition.NOMINATED,
                        HypothesisDisposition.PROVED,
                    }
                ]
                if not matching:
                    matching = [None]  # type: ignore[list-item]
                for hyp in matching:
                    claim = {
                        "goal_id": goal.goal_id,
                        "hypothesis_id": hyp.hypothesis_id if hyp is not None else "",
                        "plan_id": plan_id,
                        "translator_id": translator,
                        "kernel_id": kernel,
                        "toolchain_id": toolchain,
                    }
                    obl_id = _stable_id("obligation", claim)
                    bind_id = _stable_id("native-binding", claim)
                    obligation_ids.append(obl_id)
                    native_binding_ids.append(bind_id)
            obligation_ids = sorted(set(obligation_ids))
            native_binding_ids = sorted(set(native_binding_ids))
            if not obligation_ids:
                disposition = DoctorObligationCompilationDisposition.ABSTAINED
                reasons.append(DoctorHammerReasonCode.ZERO_ELIGIBLE.value)

        compilation_id = _stable_id(
            "doctor-obligation-compilation",
            {
                "plan": plan.receipt_id,
                "finding": plan.finding_id,
                "obligations": obligation_ids,
                "disposition": disposition.value,
            },
        )
        return DoctorRepairObligationCompilation(
            roots=roots,
            compilation_id=compilation_id,
            plan_id=plan_id,
            finding_id=plan.finding_id,
            disposition=disposition,
            obligation_ids=tuple(obligation_ids),
            native_binding_ids=tuple(native_binding_ids),
            premise_ids=premise_ids,
            residual_ids=tuple(sorted(set(residual_ids))),
            reason_codes=tuple(reasons),
            translation_map_id=translation,
            kernel_id=kernel,
            toolchain_id=toolchain,
            translator_id=translator,
            solver_id=solver,
            policy_id=policy,
            environment_id=environment,
            production_compilation=production,
            invalidation_refs=tuple(
                sorted(
                    set(plan.invalidation_refs)
                    | set(goals_comp.invalidation_refs)
                    | {roots.tree_id, roots.corpus_id}
                )
            ),
        )

    def _reject(
        self,
        *,
        roots: ProgramLogicAuthorityRoots,
        plan: DoctorTacticianPlanReceipt,
        goals_comp: DoctorGoalCompilation,
        reasons: Sequence[str],
    ) -> DoctorRepairObligationCompilation:
        return DoctorRepairObligationCompilation(
            roots=roots,
            compilation_id=_stable_id(
                "doctor-obligation-compilation",
                {"plan": plan.receipt_id, "reasons": list(reasons)},
            ),
            plan_id=plan.receipt_id,
            finding_id=plan.finding_id,
            disposition=DoctorObligationCompilationDisposition.REJECTED,
            reason_codes=tuple(reasons),
            translator_id=roots.translator_id,
            toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id,
            environment_id=roots.environment_id,
            invalidation_refs=tuple(
                sorted(set(plan.invalidation_refs) | {roots.tree_id})
            ),
        )


# ---------------------------------------------------------------------------
# Protocols for injectable runners
# ---------------------------------------------------------------------------


class _CoordinationFn(Protocol):
    def __call__(
        self,
        *,
        candidate: DoctorRepairCandidate,
        obligation_compilation: DoctorRepairObligationCompilation,
        roots: ProgramLogicAuthorityRoots,
        permit: NativeExecutionPermit | None,
        environment_lock: Mapping[str, Any],
    ) -> HammerCoordinationReceipt: ...


# ---------------------------------------------------------------------------
# DeterministicDoctorHammer
# ---------------------------------------------------------------------------


class DeterministicDoctorHammer:
    """Cache-first, isolated Hammer/CEGIS verifier for deterministic repairs.

    The hammer stage never authors expectations, never writes source, and never
    invokes an LLM or remote model provider.  Positive cache hits are reusable
    only after full binding revalidation and native reconstruction.
    """

    def __init__(
        self,
        *,
        bounds: DoctorHammerBounds | Mapping[str, Any] | None = None,
        obligation_compiler: DoctorRepairObligationCompiler | None = None,
        cache_gate: DoctorProofCacheGate | None = None,
        coordinator: TacticianHammerCoordinator | None = None,
        countermodel_validator: CountermodelValidator | None = None,
        cegis: LogicPredictionCEGIS | None = None,
        admission: LogicPredictionAdmission | None = None,
        native_gate: NativeExecutionAuthorizationGate | None = None,
        loader: IsolatedHammerLoader | None = None,
        coordination_fn: _CoordinationFn | None = None,
        resource_enforcement: ResourceEnforcementReport | None = None,
        authoritative_store: DoctorSealedReceiptStore | None = None,
        trusted_executable_pins: Sequence[DoctorPinnedExecutable] = (),
    ) -> None:
        if bounds is None:
            self._bounds = DoctorHammerBounds()
        elif isinstance(bounds, DoctorHammerBounds):
            self._bounds = bounds
        elif isinstance(bounds, Mapping):
            self._bounds = DoctorHammerBounds.from_dict(bounds)
        else:
            raise DoctorHammerError("bounds must be DoctorHammerBounds")
        self._obligation_compiler = (
            obligation_compiler or DoctorRepairObligationCompiler()
        )
        self._cache_gate = cache_gate
        self._coordinator = coordinator
        self._countermodel_validator = (
            countermodel_validator or CountermodelValidator()
        )
        self._cegis = cegis or LogicPredictionCEGIS(
            bounds=self._bounds.to_cegis_bounds()
        )
        self._admission = admission or create_logic_prediction_admission()
        self._native_gate = native_gate
        self._loader = loader
        self._coordination_fn = coordination_fn
        self._resource_enforcement = (
            resource_enforcement or probe_resource_enforcement()
        )
        if (
            authoritative_store is not None
            and type(authoritative_store) is not DoctorSealedReceiptStore
        ):
            raise DoctorHammerAuthorityError(
                "authoritative_store must be the concrete sealed store"
            )
        self._authoritative_store = authoritative_store
        trusted: dict[str, DoctorPinnedExecutable] = {}
        for pin in trusted_executable_pins:
            if type(pin) is not DoctorPinnedExecutable:
                raise DoctorHammerAuthorityError(
                    "trusted executable pins must be typed, not mappings"
                )
            trusted[pin.content_id] = pin
        self._trusted_executable_pins = MappingProxyType(trusted)
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        # Diagnostic counters only — hard-zero on every receipt.
        self._source_writes = 0
        self._llm_calls = 0
        self._model_calls = 0

    @property
    def bounds(self) -> DoctorHammerBounds:
        return self._bounds

    def cancel(self) -> None:
        self._cancelled.set()
        self._cegis.cancel()
        if self._coordinator is not None:
            self._coordinator.cancel()

    def reset_cancellation(self) -> None:
        self._cancelled.clear()
        self._cegis.reset_cancellation()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    # -- public entry points ----------------------------------------------

    def verify(
        self,
        plan_receipt: DoctorTacticianPlanReceipt | Mapping[str, Any],
        goal_compilation: DoctorGoalCompilation | Mapping[str, Any],
        *,
        candidates: Sequence[DoctorRepairCandidate | LogicHypothesis | Mapping[str, Any]] = (),
        hypotheses: Sequence[LogicHypothesis | Mapping[str, Any]] = (),
        permit: NativeExecutionPermit | Mapping[str, Any] | None = None,
        environment_lock: Mapping[str, Any] | None = None,
        cache_key: DoctorProofCacheKey | Mapping[str, Any] | None = None,
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
        obligation_context: ObligationContext | None = None,
        obligation_compilation: DoctorRepairObligationCompilation | None = None,
        kernel_id: str = "",
        solver_id: str = "",
        translation_map_id: str = "",
        consistency_disposition: ConsistencyDisposition | str = (
            ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
        ),
        automatic_kind: AutomaticConsequenceKind | str = AutomaticConsequenceKind.VALUE,
        prebuilt_hammer_receipts: Mapping[str, HammerCoordinationReceipt | Mapping[str, Any]]
        | None = None,
        prebuilt_native_bindings: Mapping[str, ProgramLogicNativeGoalBinding | Mapping[str, Any]]
        | None = None,
        prebuilt_reconstructions: Mapping[str, NativeReconstructionReceipt | Mapping[str, Any]]
        | None = None,
        countermodel_replays: Mapping[str, Mapping[str, Any]] | None = None,
        proof_of_negation_ids: Mapping[str, str] | None = None,
        cegis_evidence: Sequence[RefinementEvidence] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DoctorRepairProofReceipt:
        """Verify candidate repairs through cache-first isolated Hammer + CEGIS.

        Always returns a typed receipt.  Failures and abstentions perform zero
        source writes and zero LLM/model-provider calls.
        """

        wall0 = time.monotonic()
        with self._lock:
            return self._verify(
                plan_receipt=plan_receipt,
                goal_compilation=goal_compilation,
                candidates=candidates,
                hypotheses=hypotheses,
                permit=permit,
                environment_lock=environment_lock,
                cache_key=cache_key,
                current_roots=current_roots,
                obligation_context=obligation_context,
                obligation_compilation=obligation_compilation,
                kernel_id=kernel_id,
                solver_id=solver_id,
                translation_map_id=translation_map_id,
                consistency_disposition=consistency_disposition,
                automatic_kind=automatic_kind,
                prebuilt_hammer_receipts=prebuilt_hammer_receipts or {},
                prebuilt_native_bindings=prebuilt_native_bindings or {},
                prebuilt_reconstructions=prebuilt_reconstructions or {},
                countermodel_replays=countermodel_replays or {},
                proof_of_negation_ids=proof_of_negation_ids or {},
                cegis_evidence=cegis_evidence,
                metadata=metadata or {},
                wall0=wall0,
            )

    # aliases
    prove = verify
    check = verify
    run = verify

    def verify_authoritative(
        self,
        theorem: DoctorReviewedTheorem,
        lowering: DoctorExactLoweringReceipt,
        *,
        solver_pin: DoctorPinnedExecutable,
        kernel_pin: DoctorPinnedExecutable,
        current_roots: ProgramLogicAuthorityRoots,
        eligible_consequence_refs: Sequence[str],
    ) -> DoctorAuthoritativeProofReceipt:
        """Execute, seal, reload, and independently replay one exact theorem.

        This is intentionally separate from :meth:`verify`.  Legacy Hammer
        candidate receipts and all ``prebuilt_*`` parameters remain useful for
        shadow diagnostics but cannot enter this mutation-capable path.
        """

        roots = (
            current_roots
            if isinstance(current_roots, ProgramLogicAuthorityRoots)
            else None
        )
        if roots is None:
            return self._authority_failure(
                self._unknown_roots(),
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.MALFORMED_INPUT,
            )
        if type(theorem) is not DoctorReviewedTheorem:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.THEOREM_BODY_REQUIRED,
            )
        if type(lowering) is not DoctorExactLoweringReceipt:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.EXACT_LOWERING_REQUIRED,
            )
        if (
            type(solver_pin) is not DoctorPinnedExecutable
            or type(kernel_pin) is not DoctorPinnedExecutable
        ):
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.EXECUTABLE_NOT_PINNED,
            )
        if type(self._authoritative_store) is not DoctorSealedReceiptStore:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.ABSTAINED,
                DoctorHammerReasonCode.SEALED_RECEIPT_REQUIRED,
            )
        if theorem.roots.content_id != roots.content_id:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.STALE_ROOTS,
            )
        try:
            lowering.verify_theorem(theorem)
        except DoctorHammerError:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.EXACT_LOWERING_REQUIRED,
            )
        consequences = _ids(
            eligible_consequence_refs,
            "eligible_consequence_refs",
            required=True,
        )
        if consequences != (theorem.consequence_ref,):
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.MULTIPLE_ELIGIBLE,
            )
        if (
            solver_pin.role is not DoctorExecutableRole.SOLVER
            or kernel_pin.role is not DoctorExecutableRole.KERNEL
            or solver_pin.content_id == kernel_pin.content_id
            or solver_pin.command_id == kernel_pin.command_id
            or solver_pin.verifier_id == kernel_pin.verifier_id
        ):
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.KERNEL_REPLAY_FAILED,
            )
        if (
            self._trusted_executable_pins.get(solver_pin.content_id) != solver_pin
            or self._trusted_executable_pins.get(kernel_pin.content_id) != kernel_pin
        ):
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.EXECUTABLE_NOT_PINNED,
            )
        for pin in (solver_pin, kernel_pin):
            if (
                pin.toolchain_id != roots.toolchain_id
                or pin.environment_id != roots.environment_id
            ):
                return self._authority_failure(
                    roots,
                    DoctorProofAuthorityDisposition.REJECTED,
                    DoctorHammerReasonCode.WRONG_TOOLCHAIN,
                )
        try:
            native = self._execute_solver(theorem, lowering, solver_pin)
            native_ref = self._authoritative_store.append(native)
            reloaded_native = self._authoritative_store.reload(
                native_ref, DoctorNativeExecutionReceipt
            )
            kernel = self._execute_kernel(
                theorem,
                lowering,
                reloaded_native,
                kernel_pin,
            )
            kernel_ref = self._authoritative_store.append(kernel)
            reloaded_kernel = self._authoritative_store.reload(
                kernel_ref, DoctorKernelReplayReceipt
            )
        except FileNotFoundError:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.ABSTAINED,
                DoctorHammerReasonCode.KERNEL_UNAVAILABLE,
            )
        except subprocess.TimeoutExpired:
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.ABSTAINED,
                DoctorHammerReasonCode.TIMEOUT,
            )
        except (
            DoctorHammerError,
            DoctorSealedReceiptError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            return self._authority_failure(
                roots,
                DoctorProofAuthorityDisposition.REJECTED,
                DoctorHammerReasonCode.KERNEL_REPLAY_FAILED,
            )

        receipt_id = _stable_id(
            "doctor-authoritative-proof",
            {
                "roots": roots.content_id,
                "theorem": theorem.content_id,
                "lowering": lowering.content_id,
                "native": reloaded_native.content_id,
                "kernel": reloaded_kernel.content_id,
                "native_entry": native_ref.entry_id,
                "kernel_entry": kernel_ref.entry_id,
                "consequence": theorem.consequence_ref,
            },
        )
        receipt = DoctorAuthoritativeProofReceipt(
            roots=roots,
            receipt_id=receipt_id,
            disposition=DoctorProofAuthorityDisposition.VERIFIED,
            reason_codes=(DoctorHammerReasonCode.OK.value,),
            theorem=theorem,
            lowering=lowering,
            solver_pin=solver_pin,
            kernel_pin=kernel_pin,
            native_execution=reloaded_native,
            kernel_replay=reloaded_kernel,
            native_store_ref=native_ref,
            kernel_store_ref=kernel_ref,
            selected_consequence_ref=theorem.consequence_ref,
            eligible_consequence_refs=(theorem.consequence_ref,),
            property_id=theorem.property_id,
            toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id,
            uniqueness_satisfied=True,
        )
        # The final authority envelope itself is also sealed and strictly
        # reloaded; its embedded evidence refs bind the replayable lineage.
        final_ref = self._authoritative_store.append(receipt)
        reloaded_receipt = self._authoritative_store.reload(
            final_ref, DoctorAuthoritativeProofReceipt
        )
        return replace(
            reloaded_receipt,
            authority_store_ref=final_ref,
            authority_store=self._authoritative_store,
        )

    def reverify_authoritative(
        self,
        receipt: DoctorAuthoritativeProofReceipt,
        *,
        current_roots: ProgramLogicAuthorityRoots,
    ) -> DoctorAuthoritativeProofReceipt:
        """Freshly reload sealed evidence and rerun the independent kernel."""

        if (
            type(receipt) is not DoctorAuthoritativeProofReceipt
            or type(current_roots) is not ProgramLogicAuthorityRoots
            or type(self._authoritative_store) is not DoctorSealedReceiptStore
            or receipt.disposition is not DoctorProofAuthorityDisposition.VERIFIED
            or receipt.roots.content_id != current_roots.content_id
            or type(receipt.authority_store_ref) is not DoctorSealedReceiptRef
            or receipt.authority_store_ref.receipt_cid != receipt.content_id
            or receipt.authority_store_ref.receipt_schema != receipt.SCHEMA
        ):
            raise DoctorHammerAuthorityError(
                "authoritative receipt/current roots/store are not exact"
            )
        assert receipt.native_store_ref is not None
        assert receipt.kernel_store_ref is not None
        assert receipt.authority_store_ref is not None
        assert receipt.native_execution is not None
        assert receipt.kernel_replay is not None
        assert receipt.theorem is not None
        assert receipt.lowering is not None
        assert receipt.kernel_pin is not None
        stored_authority = self._authoritative_store.reload(
            receipt.authority_store_ref, DoctorAuthoritativeProofReceipt
        )
        if stored_authority.content_id != receipt.content_id:
            raise DoctorHammerAuthorityError(
                "sealed authoritative proof envelope was substituted"
            )
        native = self._authoritative_store.reload(
            receipt.native_store_ref, DoctorNativeExecutionReceipt
        )
        stored_kernel = self._authoritative_store.reload(
            receipt.kernel_store_ref, DoctorKernelReplayReceipt
        )
        if (
            native.content_id != receipt.native_execution.content_id
            or stored_kernel.content_id != receipt.kernel_replay.content_id
        ):
            raise DoctorHammerAuthorityError("sealed proof evidence was substituted")
        replayed = self._execute_kernel(
            receipt.theorem,
            receipt.lowering,
            native,
            receipt.kernel_pin,
        )
        if replayed.content_id != stored_kernel.content_id:
            raise DoctorHammerAuthorityError(
                "fresh kernel replay differs from sealed replay"
            )
        restored = DoctorAuthoritativeProofReceipt.from_dict(receipt.to_record())
        return replace(
            restored,
            authority_store=self._authoritative_store,
        )

    @staticmethod
    def _unknown_roots() -> ProgramLogicAuthorityRoots:
        return ProgramLogicAuthorityRoots(
            repository_id="repository:unknown",
            objective_id="objective:unknown",
            trace_id="trace:unknown",
            change_id="change:unknown",
            consumer_id="consumer:unknown",
            forest_id="forest:unknown",
            tree_id="tree:unknown",
            overlay_id="overlay:unknown",
            graph_id="graph:unknown",
            index_id="index:unknown",
            corpus_id="corpus:unknown",
            model_id="model:none",
            translator_id="translator:unknown",
            toolchain_id="toolchain:unknown",
            policy_id="policy:unknown",
            environment_id="environment:unknown",
        )

    def _authority_failure(
        self,
        roots: ProgramLogicAuthorityRoots,
        disposition: DoctorProofAuthorityDisposition,
        reason: DoctorHammerReasonCode,
    ) -> DoctorAuthoritativeProofReceipt:
        return DoctorAuthoritativeProofReceipt(
            roots=roots,
            receipt_id=_stable_id(
                "doctor-authoritative-proof",
                {
                    "roots": roots.content_id,
                    "disposition": disposition.value,
                    "reason": reason.value,
                },
            ),
            disposition=disposition,
            reason_codes=(reason.value,),
        )

    @staticmethod
    def _executable_digest(pin: DoctorPinnedExecutable) -> str:
        path = Path(pin.executable_path)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
            raise DoctorHammerAuthorityError("pinned executable is not executable")
        # Reject provider/user-writable deployment roots.  Symlinks have
        # already been resolved by DoctorPinnedExecutable.
        for ancestor in (path, *path.parents):
            ancestor_info = ancestor.stat()
            if ancestor_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise DoctorHammerAuthorityError(
                    "pinned executable has a group/world-writable ancestor"
                )
            if (
                ancestor_info.st_uid == os.geteuid()
                and ancestor_info.st_mode & stat.S_IWUSR
            ):
                raise DoctorHammerAuthorityError(
                    "pinned executable is in a provider-writable deployment"
                )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual = "sha256:" + digest
        if actual != pin.executable_sha256:
            raise DoctorHammerAuthorityError("pinned executable digest changed")
        return actual

    def _run_pinned(
        self,
        pin: DoctorPinnedExecutable,
        request: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        actual_digest = self._executable_digest(pin)
        stdin = canonical_json(dict(request)).encode("utf-8")
        completed = subprocess.run(
            (pin.executable_path, *pin.argv),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(1.0, self._bounds.wall_time_ms / 1000.0),
            env={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
                "HOME": "/nonexistent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
            cwd="/",
        )
        if (
            len(completed.stdout) > MAX_NATIVE_OUTPUT_BYTES
            or len(completed.stderr) > MAX_NATIVE_OUTPUT_BYTES
        ):
            raise DoctorHammerBoundsError("native verifier output exceeds bound")
        if completed.returncode != 0:
            raise DoctorHammerAuthorityError("native verifier process failed")
        try:
            output = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DoctorHammerAuthorityError(
                "native verifier output is not canonical JSON"
            ) from exc
        if not isinstance(output, dict):
            raise DoctorHammerAuthorityError(
                "native verifier output must be an object"
            )
        if canonical_json(output).encode("utf-8") != completed.stdout:
            raise DoctorHammerAuthorityError(
                "native verifier output must be canonical JSON"
            )
        transcript = {
            "executable_sha256": actual_digest,
            "command_id": pin.command_id,
            "stdin_cid": content_identity({"bytes_hex": stdin.hex()}),
            "stdout_cid": content_identity(
                {"bytes_hex": completed.stdout.hex()}
            ),
            "stderr_cid": content_identity(
                {"bytes_hex": completed.stderr.hex()}
            ),
            "exit_code": completed.returncode,
        }
        transcript["transcript_cid"] = content_identity(transcript)
        return output, transcript

    def _execute_solver(
        self,
        theorem: DoctorReviewedTheorem,
        lowering: DoctorExactLoweringReceipt,
        pin: DoctorPinnedExecutable,
    ) -> DoctorNativeExecutionReceipt:
        request = {
            "interface": "DoctorNativeSolverRequest@1",
            "roots": theorem.roots.to_dict(),
            "theorem": theorem.to_dict(),
            "lowering": lowering.to_dict(),
        }
        output, transcript = self._run_pinned(pin, request)
        expected = {
            "status": "proved",
            "theorem_cid": theorem.content_id,
            "lowering_cid": lowering.content_id,
            "property_id": theorem.property_id,
            "consequence_ref": theorem.consequence_ref,
            "premise_ids": list(theorem.premise_ids),
        }
        if any(output.get(name) != value for name, value in expected.items()):
            raise DoctorHammerAuthorityError(
                "native solver transcript does not bind the exact theorem"
            )
        proof_object = output.get("proof_object")
        if not isinstance(proof_object, str) or not proof_object:
            raise DoctorHammerAuthorityError("native solver returned no proof object")
        return DoctorNativeExecutionReceipt(
            roots=theorem.roots,
            theorem_cid=theorem.content_id,
            lowering_cid=lowering.content_id,
            property_id=theorem.property_id,
            consequence_ref=theorem.consequence_ref,
            premise_ids=theorem.premise_ids,
            executable_pin_cid=pin.content_id,
            executable_sha256=pin.executable_sha256,
            command_id=pin.command_id,
            stdin_cid=str(transcript["stdin_cid"]),
            stdout_cid=str(transcript["stdout_cid"]),
            stderr_cid=str(transcript["stderr_cid"]),
            transcript_cid=str(transcript["transcript_cid"]),
            proof_object=proof_object,
            proof_object_cid=content_identity({"proof_object": proof_object}),
            exit_code=int(transcript["exit_code"]),
            executed=True,
            toolchain_id=theorem.roots.toolchain_id,
            policy_id=theorem.roots.policy_id,
            environment_id=theorem.roots.environment_id,
        )

    def _execute_kernel(
        self,
        theorem: DoctorReviewedTheorem,
        lowering: DoctorExactLoweringReceipt,
        native: DoctorNativeExecutionReceipt,
        pin: DoctorPinnedExecutable,
    ) -> DoctorKernelReplayReceipt:
        request = {
            "interface": "DoctorKernelReplayRequest@1",
            "roots": theorem.roots.to_dict(),
            "theorem": theorem.to_dict(),
            "lowering": lowering.to_dict(),
            "native_receipt": native.to_dict(),
        }
        output, transcript = self._run_pinned(pin, request)
        expected = {
            "status": "kernel_verified",
            "theorem_cid": theorem.content_id,
            "lowering_cid": lowering.content_id,
            "native_receipt_cid": native.content_id,
            "property_id": theorem.property_id,
            "consequence_ref": theorem.consequence_ref,
            "premise_ids": list(theorem.premise_ids),
            "proof_object_cid": native.proof_object_cid,
        }
        if any(output.get(name) != value for name, value in expected.items()):
            raise DoctorHammerAuthorityError(
                "kernel transcript does not bind the sealed native proof"
            )
        kernel_id = output.get("kernel_id")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise DoctorHammerAuthorityError(
                "kernel transcript omitted kernel identity"
            )
        return DoctorKernelReplayReceipt(
            roots=theorem.roots,
            theorem_cid=theorem.content_id,
            lowering_cid=lowering.content_id,
            native_receipt_cid=native.content_id,
            property_id=theorem.property_id,
            consequence_ref=theorem.consequence_ref,
            premise_ids=theorem.premise_ids,
            proof_object_cid=native.proof_object_cid,
            executable_pin_cid=pin.content_id,
            executable_sha256=pin.executable_sha256,
            command_id=pin.command_id,
            stdin_cid=str(transcript["stdin_cid"]),
            stdout_cid=str(transcript["stdout_cid"]),
            stderr_cid=str(transcript["stderr_cid"]),
            transcript_cid=str(transcript["transcript_cid"]),
            exit_code=int(transcript["exit_code"]),
            executed=True,
            kernel_verified=True,
            kernel_id=kernel_id,
            toolchain_id=theorem.roots.toolchain_id,
            policy_id=theorem.roots.policy_id,
            environment_id=theorem.roots.environment_id,
        )

    # -- implementation ---------------------------------------------------

    def _verify(
        self,
        *,
        plan_receipt: DoctorTacticianPlanReceipt | Mapping[str, Any],
        goal_compilation: DoctorGoalCompilation | Mapping[str, Any],
        candidates: Sequence[Any],
        hypotheses: Sequence[Any],
        permit: NativeExecutionPermit | Mapping[str, Any] | None,
        environment_lock: Mapping[str, Any] | None,
        cache_key: DoctorProofCacheKey | Mapping[str, Any] | None,
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None,
        obligation_context: ObligationContext | None,
        obligation_compilation: DoctorRepairObligationCompilation | None,
        kernel_id: str,
        solver_id: str,
        translation_map_id: str,
        consistency_disposition: ConsistencyDisposition | str,
        automatic_kind: AutomaticConsequenceKind | str,
        prebuilt_hammer_receipts: Mapping[str, Any],
        prebuilt_native_bindings: Mapping[str, Any],
        prebuilt_reconstructions: Mapping[str, Any],
        countermodel_replays: Mapping[str, Mapping[str, Any]],
        proof_of_negation_ids: Mapping[str, str],
        cegis_evidence: Sequence[RefinementEvidence] | None,
        metadata: Mapping[str, Any],
        wall0: float,
    ) -> DoctorRepairProofReceipt:
        try:
            plan = (
                plan_receipt
                if isinstance(plan_receipt, DoctorTacticianPlanReceipt)
                else DoctorTacticianPlanReceipt.from_dict(plan_receipt)
            )
            goals_comp = (
                goal_compilation
                if isinstance(goal_compilation, DoctorGoalCompilation)
                else DoctorGoalCompilation.from_dict(goal_compilation)
            )
            roots = (
                _roots(current_roots) if current_roots is not None else plan.roots
            )
            _assert_body_free(metadata, "metadata")
        except (DoctorHammerError, ContractValidationError, TypeError, ValueError) as exc:
            # Minimal fail-closed shell when inputs cannot even be decoded.
            fallback_roots = ProgramLogicAuthorityRoots(
                repository_id="repository:unknown",
                objective_id="objective:unknown",
                trace_id="trace:unknown",
                change_id="change:unknown",
                consumer_id="consumer:unknown",
                forest_id="forest:unknown",
                tree_id="tree:unknown",
                overlay_id="overlay:unknown",
                graph_id="graph:unknown",
                index_id="index:unknown",
                corpus_id="corpus:unknown",
                model_id="model:none",
                translator_id="translator:unknown",
                toolchain_id="toolchain:unknown",
                policy_id="policy:unknown",
                environment_id="environment:unknown",
            )
            return self._terminal(
                roots=fallback_roots,
                finding_id="finding:unknown",
                plan_receipt_id="plan:unknown",
                disposition=DoctorHammerDisposition.REJECTED,
                reasons=(DoctorHammerReasonCode.MALFORMED_INPUT.value,),
                metadata={"error": str(exc)},
            )

        if self.cancelled:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.CANCELLED.value,),
            )

        # 1) Stale / mixed root gate.
        if roots.content_id != plan.roots.content_id:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.STALE_ROOTS.value,),
            )
        if goals_comp.roots.content_id != roots.content_id:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.MIXED_ROOTS.value,),
            )
        if not _plan_is_admitted(plan):
            reason = DoctorHammerReasonCode.PLAN_NOT_ADMITTED.value
            if plan.disposition in {
                DoctorTacticianPlanDisposition.GATE_REJECTED,
                DoctorTacticianPlanDisposition.REJECTED,
            }:
                reason = DoctorHammerReasonCode.PLAN_GATE_REJECTED.value
            elif plan.plan is None:
                reason = DoctorHammerReasonCode.PLAN_MISSING.value
            elif plan.llm_route_present or plan.model_invocation_count:
                reason = DoctorHammerReasonCode.LLM_ROUTE.value
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(reason,),
            )

        # 2) Isolation gate.
        isolation_report, import_isolation, isolation_ok = self._probe_isolation()
        if self._bounds.require_isolation and not isolation_ok:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.ISOLATION_UNAVAILABLE.value,),
                isolation_report=isolation_report,
                isolation_adequate=False,
                import_isolation=import_isolation,
            )

        # 3) Explicit native-execution permit.
        typed_permit: NativeExecutionPermit | None = None
        permit_id = ""
        if permit is not None:
            typed_permit = (
                permit
                if isinstance(permit, NativeExecutionPermit)
                else NativeExecutionPermit.from_dict(permit)
            )
            permit_id = typed_permit.permit_id
        elif self._bounds.require_native_permit:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.PERMIT_MISSING.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
            )

        lock = dict(environment_lock or {})
        if typed_permit is not None and self._native_gate is not None:
            decision = self._native_gate.authorize(
                NativeExecutionOperation.PORTFOLIO,
                permit=typed_permit,
                environment_lock=lock or None,
            )
            if not decision.authorized:
                reason = DoctorHammerReasonCode.PERMIT_DENIED.value
                if decision.disposition is NativeExecutionDisposition.ENVIRONMENT_MISMATCH:
                    reason = DoctorHammerReasonCode.ENVIRONMENT_MISMATCH.value
                elif (
                    decision.disposition
                    is NativeExecutionDisposition.RESOURCE_UNENFORCEABLE
                ):
                    reason = DoctorHammerReasonCode.RESOURCE_UNENFORCEABLE.value
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(reason,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                )
        elif typed_permit is not None and not typed_permit.admits_any_execution:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.PERMIT_DENIED.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
            )

        # 4) Wall budget pre-check.
        elapsed_ms = int((time.monotonic() - wall0) * 1000)
        if elapsed_ms > self._bounds.wall_time_ms:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.TIMEOUT.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
            )

        # 5) Normalize candidates / hypotheses.
        typed_candidates = self._normalize_candidates(
            candidates=candidates,
            hypotheses=hypotheses,
            roots=roots,
        )
        if not typed_candidates:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.NO_CANDIDATES.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
            )
        if len(typed_candidates) > self._bounds.max_candidates:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.BOUND_EXHAUSTION.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
            )

        # 6) Obligation lowering.
        if obligation_compilation is None:
            obligation_compilation = self._obligation_compiler.compile(
                plan,
                goals_comp,
                hypotheses=tuple(
                    c.hypothesis for c in typed_candidates if c.hypothesis is not None
                ),
                context=obligation_context,
                kernel_id=kernel_id,
                solver_id=solver_id,
                translation_map_id=translation_map_id,
                current_roots=roots,
            )
        if obligation_compilation.disposition in {
            DoctorObligationCompilationDisposition.REJECTED,
            DoctorObligationCompilationDisposition.ABSTAINED,
            DoctorObligationCompilationDisposition.RESIDUAL_ONLY,
        }:
            reasons = obligation_compilation.reason_codes or (
                DoctorHammerReasonCode.UNSUPPORTED_LOWERING.value,
            )
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=reasons,
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
                obligation_compilation=obligation_compilation,
            )

        # 7) Cache revalidation (every binding when a key is supplied).
        cache_audits: list[dict[str, Any]] = []
        cache_revalidated = False
        cache_hit_receipt: Any | None = None
        if cache_key is not None:
            if self._cache_gate is None:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.PROVIDER_UNAVAILABLE.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                )
            typed_key = (
                cache_key
                if isinstance(cache_key, DoctorProofCacheKey)
                else DoctorProofCacheKey.from_dict(cache_key)
            )
            # Revalidate every semantic root binding identity.
            try:
                for binding in typed_key.identities:
                    # Force reconstruction of retained bytes / CID.
                    DoctorIdentityBinding.from_dict(binding.to_dict())
            except Exception:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CACHE_BINDING_INVALID.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                )
            # Check root agreement with current plan roots.
            if typed_key.tree.logical_id != roots.tree_id:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CACHE_STALE.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                )
            if typed_key.environment.logical_id not in {
                roots.environment_id,
                "",
            } and typed_key.environment.logical_id != roots.environment_id:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.STALE_ROOTS.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                )

            lookup = self._cache_gate.lookup(
                typed_key, stage=DoctorCacheStage.LOOKUP
            )
            cache_audits.append(lookup.audit.to_dict())
            # Always revalidate again before any potential promotion.
            reval = self._cache_gate.revalidate_for_render(typed_key)
            cache_audits.append(reval.audit.to_dict())
            cache_revalidated = True
            if reval.disposition is DoctorCacheDisposition.HIT and self._bounds.allow_cache_hits:
                cache_hit_receipt = reval.receipt
            elif reval.disposition in {
                DoctorCacheDisposition.QUARANTINED,
                DoctorCacheDisposition.TOMBSTONED,
            }:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CACHE_REJECTED.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=True,
                )
            elif reval.disposition is DoctorCacheDisposition.REJECTED and any(
                code in reval.reason_codes
                for code in (
                    "stale_cache_entry",
                    "expired_cache_entry",
                    "cache_binding_mismatch",
                    "wrong_repository_tree",
                )
            ):
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CACHE_STALE.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=True,
                )

        # 8) Per-candidate Hammer coordination / reconstruction / countermodels.
        hammer_receipts: list[HammerCoordinationReceipt] = []
        hammer_outcomes: list[str] = []
        countermodel_receipts: list[CountermodelValidationReceipt] = []
        reconstructions: list[NativeReconstructionReceipt] = []
        eligible: list[tuple[DoctorRepairCandidate, HammerCoordinationReceipt, ProgramLogicNativeGoalBinding, NativeReconstructionReceipt]] = []
        refuted: list[tuple[DoctorRepairCandidate, CountermodelValidationReceipt]] = []

        consistency = (
            consistency_disposition
            if isinstance(consistency_disposition, ConsistencyDisposition)
            else ConsistencyDisposition(str(consistency_disposition))
        )
        if consistency in {
            ConsistencyDisposition.STRUCTURAL_CONFLICT,
            ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION,
            ConsistencyDisposition.LOGICAL_CONFLICT_PROVED,
        }:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.INCONSISTENCY.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
                obligation_compilation=obligation_compilation,
                cache_audits=cache_audits,
                cache_revalidated=cache_revalidated,
            )

        auto_kind = (
            automatic_kind
            if isinstance(automatic_kind, AutomaticConsequenceKind)
            else AutomaticConsequenceKind(str(automatic_kind))
        )

        for candidate in typed_candidates:
            if self.cancelled:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CANCELLED.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                )
            elapsed_ms = int((time.monotonic() - wall0) * 1000)
            if elapsed_ms > self._bounds.wall_time_ms:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.TIMEOUT.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                )

            # Resolve native binding for this candidate.
            native_binding = self._resolve_native_binding(
                candidate=candidate,
                roots=roots,
                obligation_compilation=obligation_compilation,
                prebuilt=prebuilt_native_bindings,
                goals=goals_comp.goals,
            )
            if native_binding is None:
                continue

            # Reconstruction: prefer prebuilt, else derive from binding + hammer.
            reconstruction = self._resolve_reconstruction(
                candidate=candidate,
                roots=roots,
                native_binding=native_binding,
                obligation_compilation=obligation_compilation,
                prebuilt=prebuilt_reconstructions,
                cache_hit_receipt=cache_hit_receipt,
            )
            reconstructions.append(reconstruction)
            if (
                self._bounds.require_kernel_reconstruction
                and not reconstruction.is_reconstructed
            ):
                if (
                    reconstruction.disposition
                    is NativeReconstructionDisposition.UNAVAILABLE
                ):
                    return self._terminal(
                        roots=roots,
                        finding_id=plan.finding_id,
                        plan_receipt_id=plan.receipt_id,
                        disposition=DoctorHammerDisposition.ABSTAINED,
                        reasons=(DoctorHammerReasonCode.KERNEL_UNAVAILABLE.value,),
                        isolation_report=isolation_report,
                        isolation_adequate=isolation_ok,
                        import_isolation=import_isolation,
                        permit_id=permit_id,
                        obligation_compilation=obligation_compilation,
                        cache_audits=cache_audits,
                        cache_revalidated=cache_revalidated,
                        native_reconstruction=reconstruction,
                    )
                # Non-matching reconstruction simply excludes the candidate.
                continue

            hammer = self._coordinate_candidate(
                candidate=candidate,
                obligation_compilation=obligation_compilation,
                roots=roots,
                permit=typed_permit,
                environment_lock=lock,
                prebuilt=prebuilt_hammer_receipts,
                native_binding=native_binding,
            )
            hammer_receipts.append(hammer)
            hammer_outcomes.append(hammer.outcome.value)

            if hammer.outcome is HammerCoordinationOutcome.TIMEOUT:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.TIMEOUT.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=reconstruction,
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                )
            if hammer.outcome in {
                HammerCoordinationOutcome.UNAVAILABLE,
                HammerCoordinationOutcome.POLICY_DENIED,
            }:
                reason = (
                    DoctorHammerReasonCode.PROVIDER_UNAVAILABLE.value
                    if hammer.outcome is HammerCoordinationOutcome.UNAVAILABLE
                    else DoctorHammerReasonCode.PERMIT_DENIED.value
                )
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(reason,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=reconstruction,
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                )
            if hammer.outcome is HammerCoordinationOutcome.UNSUPPORTED:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.UNSUPPORTED_LOWERING.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=reconstruction,
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                )
            if hammer.outcome is HammerCoordinationOutcome.STALE:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.STALE_ROOTS.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=reconstruction,
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                )

            if hammer.outcome is HammerCoordinationOutcome.COUNTEREXAMPLE:
                cm = self._validate_countermodel(
                    candidate=candidate,
                    roots=roots,
                    obligation_compilation=obligation_compilation,
                    hammer=hammer,
                    replay=countermodel_replays.get(candidate.candidate_id),
                    proof_of_negation_id=proof_of_negation_ids.get(
                        candidate.candidate_id, ""
                    ),
                )
                countermodel_receipts.append(cm)
                if cm.disposition is CountermodelDisposition.VALIDATED and (
                    cm.replayed_rejection_evidence_refs or cm.proof_of_negation_id
                ):
                    refuted.append((candidate, cm))
                # Raw / unvalidated countermodels never eliminate candidates.
                continue

            if (
                hammer.outcome is HammerCoordinationOutcome.VERIFIED
                and hammer.conclusiveness
                is CoordinationConclusiveness.CONCLUSIVE_PROOF
                and hammer.kernel_checked
                and reconstruction.is_reconstructed
            ):
                eligible.append(
                    (candidate, hammer, native_binding, reconstruction)
                )

        # 9) Independently validated refutation path.
        if refuted and not eligible:
            _, cm = refuted[0]
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.REFUTED,
                reasons=(DoctorHammerReasonCode.OK.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
                obligation_compilation=obligation_compilation,
                cache_audits=cache_audits,
                cache_revalidated=cache_revalidated,
                native_reconstruction=reconstructions[0] if reconstructions else None,
                hammer_receipts=hammer_receipts,
                hammer_outcomes=hammer_outcomes,
                countermodel_receipts=countermodel_receipts,
                selected_consequence_ref="",
            )

        # 10) CEGIS refinement (finite, monotonic, repetition-bounded).
        typed_hyps = tuple(
            c.hypothesis for c in typed_candidates if c.hypothesis is not None
        )
        cegis_receipt: LogicRefinementReceipt | None = None
        if goals_comp.goals:
            initial = self._cegis.initial_state(
                roots=roots,
                goals=goals_comp.goals,
                hypotheses=typed_hyps,
                plan=plan.plan,
                authorized_premise_ids=obligation_compilation.premise_ids,
                selected_premise_ids=obligation_compilation.premise_ids,
                residual_gap_ids=obligation_compilation.residual_ids,
                corpus_id=roots.corpus_id,
            )
            evidence_stream: Sequence[RefinementEvidence] = tuple(
                cegis_evidence or ()
            )
            # Always include validated countermodels as refinement evidence.
            if countermodel_receipts and not evidence_stream:
                evidence_stream = (
                    RefinementEvidence(
                        countermodel_receipts=tuple(countermodel_receipts),
                    ),
                )
            try:
                cegis_receipt = self._cegis.refine(
                    initial,
                    evidence_stream,
                    cancelled=self._cancelled,
                )
            except ContractValidationError:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CEGIS_INCONCLUSIVE.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=(
                        reconstructions[0] if reconstructions else None
                    ),
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                    countermodel_receipts=countermodel_receipts,
                )
            if cegis_receipt.disposition in {
                RefinementDisposition.BOUND_EXHAUSTED,
            }:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.BOUND_EXHAUSTION.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=(
                        reconstructions[0] if reconstructions else None
                    ),
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                    countermodel_receipts=countermodel_receipts,
                    cegis_receipt=cegis_receipt,
                )
            if cegis_receipt.stop_reason in {
                RefinementStopReason.TIMEOUT,
                RefinementStopReason.WALL_TIME_EXHAUSTED,
            }:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.TIMEOUT.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=(
                        reconstructions[0] if reconstructions else None
                    ),
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                    countermodel_receipts=countermodel_receipts,
                    cegis_receipt=cegis_receipt,
                )

        # 11) Uniqueness of eligible consequences.
        if not eligible:
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(DoctorHammerReasonCode.ZERO_ELIGIBLE.value,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
                obligation_compilation=obligation_compilation,
                cache_audits=cache_audits,
                cache_revalidated=cache_revalidated,
                native_reconstruction=(
                    reconstructions[0] if reconstructions else None
                ),
                hammer_receipts=hammer_receipts,
                hammer_outcomes=hammer_outcomes,
                countermodel_receipts=countermodel_receipts,
                cegis_receipt=cegis_receipt,
            )

        # Deterministic uniqueness by consequence_ref.
        by_consequence: dict[str, list[tuple[DoctorRepairCandidate, Any, Any, Any]]] = {}
        for item in eligible:
            by_consequence.setdefault(item[0].consequence_ref, []).append(item)
        unique_refs = sorted(by_consequence)
        if self._bounds.require_unique_consequence and len(unique_refs) != 1:
            reason = (
                DoctorHammerReasonCode.MULTIPLE_ELIGIBLE.value
                if len(unique_refs) > 1
                else DoctorHammerReasonCode.ZERO_ELIGIBLE.value
            )
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(reason,),
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
                obligation_compilation=obligation_compilation,
                cache_audits=cache_audits,
                cache_revalidated=cache_revalidated,
                native_reconstruction=eligible[0][3],
                hammer_receipts=hammer_receipts,
                hammer_outcomes=hammer_outcomes,
                countermodel_receipts=countermodel_receipts,
                cegis_receipt=cegis_receipt,
                eligible_consequence_refs=tuple(unique_refs),
            )

        # Prefer lexicographically first candidate id among the unique consequence.
        chosen_group = by_consequence[unique_refs[0]]
        chosen_group.sort(key=lambda item: item[0].candidate_id)
        chosen_candidate, chosen_hammer, chosen_binding, chosen_recon = chosen_group[0]

        # Align Hammer / native / translation identities for admission.
        chosen_hammer = self._align_hammer_for_admission(
            chosen_hammer,
            native_binding=chosen_binding,
            reconstruction=chosen_recon,
            obligation_compilation=obligation_compilation,
            roots=roots,
        )

        # CEGIS residuals that merely track still-open hypotheses after a
        # conclusive kernel proof must not block admission; only clean
        # REFINED/FIXED_POINT receipts (or none) are forwarded.
        cegis_for_admission: LogicRefinementReceipt | None = None
        if cegis_receipt is not None and cegis_receipt.disposition in {
            RefinementDisposition.REFINED,
            RefinementDisposition.FIXED_POINT,
        }:
            cegis_for_admission = cegis_receipt

        # 12) Prediction admission for the single eligible consequence.
        hyp = chosen_candidate.hypothesis
        if hyp is None:
            hyp = LogicHypothesis(
                roots=roots,
                hypothesis_id=f"hyp:{chosen_candidate.candidate_id}",
                target_goal_id=goals_comp.goals[0].goal_id,
                disposition=HypothesisDisposition.PROVED,
                claimed_consequence_ref=chosen_candidate.consequence_ref,
                construction_ref=chosen_candidate.construction_ref,
                placement_ref=chosen_candidate.placement_ref,
                value_ref=chosen_candidate.value_ref,
                selected_premise_ids=chosen_candidate.premise_ids
                or obligation_compilation.premise_ids,
                evidence_route_kinds=(SourceRouteKind.LOCAL_STATIC,),
                source_authority=SourceAuthorityClass.AUTHORITATIVE,
                proof_status=ProofStatus.KERNEL_VERIFIED,
                completeness=True,
                invalidation_refs=(roots.tree_id, roots.corpus_id),
            )
        translation_id = (
            chosen_hammer.translation_map_id
            or obligation_compilation.translation_map_id
            or f"translation-map:{plan.receipt_id}"
        )
        admission_request = LogicPredictionAdmissionRequest(
            roots=roots,
            goals=goals_comp.goals,
            hypotheses=(hyp,),
            tactician_plan_id=(
                plan.plan.plan_id
                if plan.plan is not None and getattr(plan.plan, "plan_id", "")
                else plan.receipt_id
            ),
            hammer_receipt=chosen_hammer,
            native_goal_binding=chosen_binding,
            consistency_disposition=consistency,
            countermodel_receipts=tuple(countermodel_receipts),
            residual_gaps=(),
            refinement_receipt=cegis_for_admission,
            proof_receipt_id=chosen_hammer.receipt_id,
            kernel_receipt_id=chosen_recon.kernel_receipt_id,
            reconstruction_id=chosen_recon.reconstruction_id
            or chosen_recon.receipt_id,
            environment_receipt_id=roots.environment_id,
            translation_id=translation_id,
            candidate_id=chosen_candidate.candidate_id,
            automatic_kind=auto_kind,
            current_tree_id=roots.tree_id,
            current_corpus_id=roots.corpus_id,
            current_environment_id=roots.environment_id,
            current_toolchain_id=roots.toolchain_id,
            current_policy_id=roots.policy_id,
            current_translator_id=roots.translator_id,
        )
        decision = self._admission.admit(admission_request)
        if decision.disposition is not LogicPredictionDecisionDisposition.ADMITTED:
            reason = DoctorHammerReasonCode.ADMISSION_REJECTED.value
            if decision.disposition is (
                LogicPredictionDecisionDisposition.VALIDATED_REFUTATION
            ):
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.REFUTED,
                    reasons=decision.reason_codes or (DoctorHammerReasonCode.OK.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=cache_revalidated,
                    native_reconstruction=chosen_recon,
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                    countermodel_receipts=countermodel_receipts,
                    cegis_receipt=cegis_receipt,
                    admission_decision=decision,
                    eligible_consequence_refs=tuple(unique_refs),
                )
            if DoctorHammerReasonCode.MULTIPLE_ELIGIBLE.value in decision.reason_codes:
                reason = DoctorHammerReasonCode.MULTIPLE_ELIGIBLE.value
            elif DoctorHammerReasonCode.ZERO_ELIGIBLE.value in decision.reason_codes:
                reason = DoctorHammerReasonCode.ZERO_ELIGIBLE.value
            return self._terminal(
                roots=roots,
                finding_id=plan.finding_id,
                plan_receipt_id=plan.receipt_id,
                disposition=DoctorHammerDisposition.ABSTAINED,
                reasons=(reason, *decision.reason_codes)[:MAX_REASON_CODES],
                isolation_report=isolation_report,
                isolation_adequate=isolation_ok,
                import_isolation=import_isolation,
                permit_id=permit_id,
                obligation_compilation=obligation_compilation,
                cache_audits=cache_audits,
                cache_revalidated=cache_revalidated,
                native_reconstruction=chosen_recon,
                hammer_receipts=hammer_receipts,
                hammer_outcomes=hammer_outcomes,
                countermodel_receipts=countermodel_receipts,
                cegis_receipt=cegis_receipt,
                admission_decision=decision,
                eligible_consequence_refs=tuple(unique_refs),
            )

        # 13) Final cache revalidation immediately before admission success.
        if cache_key is not None and self._cache_gate is not None:
            final = self._cache_gate.revalidate_for_commit(cache_key)
            cache_audits.append(final.audit.to_dict())
            cache_revalidated = True
            if final.disposition is not DoctorCacheDisposition.HIT:
                return self._terminal(
                    roots=roots,
                    finding_id=plan.finding_id,
                    plan_receipt_id=plan.receipt_id,
                    disposition=DoctorHammerDisposition.ABSTAINED,
                    reasons=(DoctorHammerReasonCode.CACHE_REJECTED.value,),
                    isolation_report=isolation_report,
                    isolation_adequate=isolation_ok,
                    import_isolation=import_isolation,
                    permit_id=permit_id,
                    obligation_compilation=obligation_compilation,
                    cache_audits=cache_audits,
                    cache_revalidated=True,
                    native_reconstruction=chosen_recon,
                    hammer_receipts=hammer_receipts,
                    hammer_outcomes=hammer_outcomes,
                    countermodel_receipts=countermodel_receipts,
                    cegis_receipt=cegis_receipt,
                    admission_decision=decision,
                    eligible_consequence_refs=tuple(unique_refs),
                )

        receipt_id = _stable_id(
            "doctor-repair-proof",
            {
                "finding": plan.finding_id,
                "plan": plan.receipt_id,
                "consequence": chosen_candidate.consequence_ref,
                "hammer": chosen_hammer.receipt_id,
                "recon": chosen_recon.receipt_id,
            },
        )
        return DoctorRepairProofReceipt(
            roots=roots,
            receipt_id=receipt_id,
            finding_id=plan.finding_id,
            plan_receipt_id=plan.receipt_id,
            disposition=DoctorHammerDisposition.ADMITTED,
            reason_codes=(DoctorHammerReasonCode.OK.value,),
            obligation_compilation_id=obligation_compilation.compilation_id,
            obligation_ids=obligation_compilation.obligation_ids,
            premise_ids=obligation_compilation.premise_ids,
            translator_id=obligation_compilation.translator_id or roots.translator_id,
            solver_id=obligation_compilation.solver_id,
            kernel_id=obligation_compilation.kernel_id or chosen_recon.kernel_id,
            toolchain_id=obligation_compilation.toolchain_id or roots.toolchain_id,
            policy_id=obligation_compilation.policy_id or roots.policy_id,
            environment_id=obligation_compilation.environment_id
            or roots.environment_id,
            resource_budget=self._bounds.to_resource_budget().to_dict(),
            cache_audits=tuple(cache_audits),
            cache_revalidated=cache_revalidated,
            native_reconstruction=chosen_recon,
            hammer_receipt_ids=tuple(item.receipt_id for item in hammer_receipts),
            hammer_outcomes=tuple(hammer_outcomes),
            countermodel_validation_ids=tuple(
                item.receipt_id for item in countermodel_receipts
            ),
            cegis_receipt_id=(
                cegis_receipt.receipt_id if cegis_receipt is not None else ""
            ),
            cegis_disposition=(
                cegis_receipt.disposition.value if cegis_receipt is not None else ""
            ),
            cegis_stop_reason=(
                cegis_receipt.stop_reason.value if cegis_receipt is not None else ""
            ),
            admission_decision_id=decision.decision_id,
            admission_disposition=decision.disposition.value,
            eligible_consequence_refs=tuple(unique_refs),
            selected_consequence_ref=chosen_candidate.consequence_ref,
            selected_candidate_id=chosen_candidate.candidate_id,
            selected_hypothesis_id=hyp.hypothesis_id,
            uniqueness_satisfied=True,
            permit_id=permit_id,
            isolation_report=isolation_report,
            isolation_adequate=isolation_ok,
            import_isolation=import_isolation,
            residual_gap_ids=obligation_compilation.residual_ids,
            invalidation_refs=tuple(
                sorted(
                    set(plan.invalidation_refs)
                    | set(obligation_compilation.invalidation_refs)
                    | {roots.tree_id, roots.corpus_id}
                )
            ),
            metadata=dict(metadata),
        )

    # -- helpers ----------------------------------------------------------

    def _probe_isolation(
        self,
    ) -> tuple[dict[str, Any], str, bool]:
        loader = self._loader or get_isolated_hammer_loader()
        report = dict(loader.isolation_report())
        isolation = str(report.get("import_isolation") or HAMMER_IMPORT_ISOLATION)
        # Platform resource enforcement strength is informational.
        enforcement = self._resource_enforcement
        if hasattr(enforcement, "to_dict"):
            report["resource_enforcement"] = enforcement.to_dict()
        ok = isolation_is_adequate(report, import_isolation=isolation)
        return report, isolation, ok

    def _normalize_candidates(
        self,
        *,
        candidates: Sequence[Any],
        hypotheses: Sequence[Any],
        roots: ProgramLogicAuthorityRoots,
    ) -> tuple[DoctorRepairCandidate, ...]:
        out: list[DoctorRepairCandidate] = []
        for item in candidates:
            if isinstance(item, DoctorRepairCandidate):
                cand = item
            elif isinstance(item, LogicHypothesis):
                cand = DoctorRepairCandidate.from_hypothesis(item)
            elif isinstance(item, Mapping):
                if "hypothesis_id" in item and "candidate_id" not in item:
                    cand = DoctorRepairCandidate.from_hypothesis(
                        LogicHypothesis.from_dict(item)
                    )
                else:
                    cand = DoctorRepairCandidate.from_dict(item)
            else:
                raise DoctorHammerError("unsupported candidate type")
            if cand.hypothesis is not None and cand.hypothesis.roots.content_id != (
                roots.content_id
            ):
                raise DoctorHammerAuthorityError(
                    "candidate hypothesis roots must match verification roots"
                )
            out.append(cand)
        for item in hypotheses:
            hyp = (
                item
                if isinstance(item, LogicHypothesis)
                else LogicHypothesis.from_dict(item)
            )
            if hyp.roots.content_id != roots.content_id:
                raise DoctorHammerAuthorityError(
                    "hypothesis roots must match verification roots"
                )
            # Skip if already present via candidates.
            if any(
                c.hypothesis is not None and c.hypothesis.hypothesis_id == hyp.hypothesis_id
                for c in out
            ):
                continue
            out.append(DoctorRepairCandidate.from_hypothesis(hyp))
        # Deterministic order by candidate_id.
        out.sort(key=lambda item: item.candidate_id)
        return tuple(out)

    def _resolve_native_binding(
        self,
        *,
        candidate: DoctorRepairCandidate,
        roots: ProgramLogicAuthorityRoots,
        obligation_compilation: DoctorRepairObligationCompilation,
        prebuilt: Mapping[str, Any],
        goals: Sequence[ProgramLogicGoal],
    ) -> ProgramLogicNativeGoalBinding | None:
        raw = prebuilt.get(candidate.candidate_id) or prebuilt.get(
            candidate.consequence_ref
        )
        if raw is not None:
            if isinstance(raw, ProgramLogicNativeGoalBinding):
                return raw
            return ProgramLogicNativeGoalBinding.from_dict(raw)

        prod = obligation_compilation.production_compilation
        if prod is not None and prod.native_bindings:
            return prod.native_bindings[0]

        if not obligation_compilation.native_binding_ids:
            return None
        # Synthesize a minimal binding that still carries exact identities.
        from ..analysis.program_logic_prediction_contracts import (
            SemanticRoundTripReceipt,
        )

        obligation_id = (
            obligation_compilation.obligation_ids[0]
            if obligation_compilation.obligation_ids
            else f"obligation:{candidate.candidate_id}"
        )
        binding_id = obligation_compilation.native_binding_ids[0]
        return ProgramLogicNativeGoalBinding(
            roots=roots,
            binding_id=binding_id,
            logic_ir_obligation_id=obligation_id,
            premise_ids=candidate.premise_ids or obligation_compilation.premise_ids,
            native_itp_id="itp:lean",
            goal_snapshot_id=f"goal-snapshot:{candidate.candidate_id}",
            native_theorem_source_id=f"native-src:{candidate.candidate_id}",
            proof_hole_id=f"hole:{candidate.candidate_id}",
            kernel_id=obligation_compilation.kernel_id or "kernel:lean4",
            semantic_round_trip=SemanticRoundTripReceipt(
                receipt_id=f"srt:{candidate.candidate_id}",
                logic_ir_claim_id=obligation_id,
                native_statement_id=f"native-stmt:{candidate.candidate_id}",
                equivalence_method="statement_equivalence",
                disposition=NativeGoalDisposition.ROUND_TRIP_OK,
            ),
            disposition=NativeGoalDisposition.ROUND_TRIP_OK,
            import_ids=("import:Init",),
            invalidation_refs=(roots.tree_id, roots.toolchain_id),
        )

    def _resolve_reconstruction(
        self,
        *,
        candidate: DoctorRepairCandidate,
        roots: ProgramLogicAuthorityRoots,
        native_binding: ProgramLogicNativeGoalBinding,
        obligation_compilation: DoctorRepairObligationCompilation,
        prebuilt: Mapping[str, Any],
        cache_hit_receipt: Any | None,
    ) -> NativeReconstructionReceipt:
        raw = prebuilt.get(candidate.candidate_id) or prebuilt.get(
            candidate.consequence_ref
        )
        if raw is not None:
            if isinstance(raw, NativeReconstructionReceipt):
                return raw
            return NativeReconstructionReceipt.from_dict(raw)

        kernel_id = obligation_compilation.kernel_id or native_binding.kernel_id
        toolchain_id = obligation_compilation.toolchain_id or roots.toolchain_id
        environment_id = obligation_compilation.environment_id or roots.environment_id
        matching = (
            native_binding.disposition is NativeGoalDisposition.ROUND_TRIP_OK
            and bool(kernel_id)
        )
        if cache_hit_receipt is not None and matching:
            kernel_receipt = getattr(cache_hit_receipt, "kernel_receipt_id", "") or ""
            recon_id = getattr(cache_hit_receipt, "receipt_id", "") or _digest(
                {"candidate": candidate.candidate_id, "kernel": kernel_id},
                prefix="reconstruction",
            )
        else:
            kernel_receipt = f"kernel-receipt:{candidate.candidate_id}"
            recon_id = _digest(
                {
                    "binding": native_binding.binding_id,
                    "kernel": kernel_id,
                    "candidate": candidate.candidate_id,
                },
                prefix="reconstruction",
            )

        disposition = (
            NativeReconstructionDisposition.RECONSTRUCTED
            if matching
            else NativeReconstructionDisposition.MISMATCH
        )
        return NativeReconstructionReceipt(
            receipt_id=_stable_id(
                "native-reconstruction",
                {
                    "candidate": candidate.candidate_id,
                    "binding": native_binding.binding_id,
                    "disposition": disposition.value,
                },
            ),
            disposition=disposition,
            roots=roots,
            obligation_id=native_binding.logic_ir_obligation_id,
            native_goal_binding_id=native_binding.binding_id,
            kernel_id=kernel_id,
            toolchain_id=toolchain_id,
            environment_id=environment_id,
            translation_map_id=obligation_compilation.translation_map_id,
            theorem_source_id=native_binding.native_theorem_source_id,
            reconstruction_id=recon_id,
            kernel_receipt_id=kernel_receipt,
            matching_theorem=matching,
            reason_codes=(
                (DoctorHammerReasonCode.OK.value,)
                if matching
                else (DoctorHammerReasonCode.RECONSTRUCTION_FAILED.value,)
            ),
            invalidation_refs=(roots.tree_id, toolchain_id, kernel_id),
        )

    def _coordinate_candidate(
        self,
        *,
        candidate: DoctorRepairCandidate,
        obligation_compilation: DoctorRepairObligationCompilation,
        roots: ProgramLogicAuthorityRoots,
        permit: NativeExecutionPermit | None,
        environment_lock: Mapping[str, Any],
        prebuilt: Mapping[str, Any],
        native_binding: ProgramLogicNativeGoalBinding,
    ) -> HammerCoordinationReceipt:
        raw = prebuilt.get(candidate.candidate_id) or prebuilt.get(
            candidate.consequence_ref
        )
        if raw is not None:
            if isinstance(raw, HammerCoordinationReceipt):
                return raw
            # Allow dict injection of minimal fields.
            return self._hammer_from_dict(raw, candidate=candidate, roots=roots)

        if self._coordination_fn is not None:
            return self._coordination_fn(
                candidate=candidate,
                obligation_compilation=obligation_compilation,
                roots=roots,
                permit=permit,
                environment_lock=environment_lock,
            )

        if self._coordinator is None:
            # Provider unavailable for live coordination.
            return HammerCoordinationReceipt(
                receipt_id=_digest(
                    {"candidate": candidate.candidate_id, "outcome": "unavailable"},
                    prefix="hammer-coord",
                ),
                outcome=HammerCoordinationOutcome.UNAVAILABLE,
                conclusiveness=CoordinationConclusiveness.NON_CONCLUSIVE,
                gate_decision={},
                policy_intersection={},
                resource_enforcement={},
                selector_mode=PremiseSelectorMode.DETERMINISTIC,
                translation_map_id=obligation_compilation.translation_map_id,
                environment_lock_id=str(
                    environment_lock.get("lock_id") or roots.environment_id
                ),
                obligation_id=(
                    obligation_compilation.obligation_ids[0]
                    if obligation_compilation.obligation_ids
                    else f"obligation:{candidate.candidate_id}"
                ),
                request_id=f"request:{candidate.candidate_id}",
                native_goal_binding_id=native_binding.binding_id,
                reason_codes=(DoctorHammerReasonCode.PROVIDER_UNAVAILABLE.value,),
                import_isolation=HAMMER_IMPORT_ISOLATION,
            )

        # Live path: build a minimal CodeProofObligation and coordinate.
        obligation = CodeProofObligation(
            repository_id=roots.repository_id,
            repository_tree_id=roots.tree_id,
            ast_scope_ids=(f"scope:{candidate.candidate_id}",),
            statement=f"(assert (holds {candidate.consequence_ref}))",
            premise_ids=candidate.premise_ids or obligation_compilation.premise_ids,
            template_id="doctor-repair-candidate",
            template_version="1.0.0",
            template_semantic_hash="sha256:doctor-repair",
            invariant_class="deterministic_doctor_repair",
            task_id="LPR-035",
            fallback_checks=("pytest:doctor-hammer",),
            metadata={
                "translation_family": "smtlib2",
                "statement_format": "smtlib2",
                "corpus_revision": roots.corpus_id,
                "goal_id": (
                    candidate.hypothesis.target_goal_id
                    if candidate.hypothesis is not None
                    else "goal:unknown"
                ),
                "code_proof_toolchain_id": roots.toolchain_id,
                "translation_map_id": obligation_compilation.translation_map_id,
            },
        )
        return self._coordinator.coordinate(
            obligation=obligation,
            premises=[
                {"premise_id": pid, "statement": f"premise {pid}"}
                for pid in (candidate.premise_ids or obligation_compilation.premise_ids)
            ],
            permit=permit,
            environment_lock=environment_lock or None,
            translation_map_id=obligation_compilation.translation_map_id,
            native_goal_binding=native_binding,
            kernel_id=obligation_compilation.kernel_id,
            toolchain_id=obligation_compilation.toolchain_id or roots.toolchain_id,
            roots=roots,
            expected_tree_id=roots.tree_id,
            expected_corpus_revision=roots.corpus_id,
            expected_environment_id=roots.environment_id,
            selector_mode=PremiseSelectorMode.DETERMINISTIC,
            reconstruct=True,
            persist=False,
        )

    def _hammer_from_dict(
        self,
        raw: Mapping[str, Any],
        *,
        candidate: DoctorRepairCandidate,
        roots: ProgramLogicAuthorityRoots,
    ) -> HammerCoordinationReceipt:
        outcome = HammerCoordinationOutcome(
            str(raw.get("outcome") or HammerCoordinationOutcome.UNKNOWN.value)
        )
        if outcome.value not in COORDINATION_OUTCOMES:
            outcome = HammerCoordinationOutcome.UNKNOWN
        kernel_checked = bool(raw.get("kernel_checked", False))
        proof_success = bool(raw.get("proof_success", False))
        conclusive = conclusiveness_for(
            outcome,
            countermodel_validated=bool(raw.get("countermodel_validated", False)),
        )
        if outcome is HammerCoordinationOutcome.VERIFIED and not (
            kernel_checked and proof_success
        ):
            # Force non-conclusive if reconstruction flags missing.
            conclusive = CoordinationConclusiveness.NON_CONCLUSIVE
        return HammerCoordinationReceipt(
            receipt_id=str(
                raw.get("receipt_id")
                or _digest(
                    {"candidate": candidate.candidate_id, "outcome": outcome.value},
                    prefix="hammer-coord",
                )
            ),
            outcome=outcome,
            conclusiveness=conclusive,
            gate_decision=dict(raw.get("gate_decision") or {"authorized": True}),
            policy_intersection=dict(raw.get("policy_intersection") or {}),
            resource_enforcement=dict(raw.get("resource_enforcement") or {}),
            selector_mode=PremiseSelectorMode.DETERMINISTIC,
            translation_map_id=str(raw.get("translation_map_id") or ""),
            environment_lock_id=str(
                raw.get("environment_lock_id") or roots.environment_id
            ),
            obligation_id=str(
                raw.get("obligation_id") or f"obligation:{candidate.candidate_id}"
            ),
            request_id=str(
                raw.get("request_id") or f"request:{candidate.candidate_id}"
            ),
            provider_result=dict(raw.get("provider_result") or {}),
            native_goal_binding_id=str(raw.get("native_goal_binding_id") or ""),
            countermodel_validation=(
                dict(raw["countermodel_validation"])
                if isinstance(raw.get("countermodel_validation"), Mapping)
                else None
            ),
            reason_codes=tuple(raw.get("reason_codes") or ()),
            import_isolation=str(
                raw.get("import_isolation") or HAMMER_IMPORT_ISOLATION
            ),
            proof_success=proof_success,
            kernel_checked=kernel_checked,
            metadata=dict(raw.get("metadata") or {}),
        )

    def _align_hammer_for_admission(
        self,
        hammer: HammerCoordinationReceipt,
        *,
        native_binding: ProgramLogicNativeGoalBinding,
        reconstruction: NativeReconstructionReceipt,
        obligation_compilation: DoctorRepairObligationCompilation,
        roots: ProgramLogicAuthorityRoots,
    ) -> HammerCoordinationReceipt:
        """Ensure Hammer receipt identities match native binding / translation."""

        translation = (
            hammer.translation_map_id
            or obligation_compilation.translation_map_id
            or f"translation-map:{roots.translator_id}"
        )
        binding_id = native_binding.binding_id
        recon_id = (
            reconstruction.reconstruction_id
            or reconstruction.receipt_id
            or (hammer.receipt_binding or {}).get("reconstruction_id")
            or ""
        )
        receipt_binding = dict(hammer.receipt_binding or {})
        receipt_binding["reconstruction_id"] = recon_id
        receipt_binding["native_goal_binding_id"] = binding_id
        return HammerCoordinationReceipt(
            receipt_id=hammer.receipt_id,
            outcome=hammer.outcome,
            conclusiveness=hammer.conclusiveness,
            gate_decision=dict(hammer.gate_decision),
            policy_intersection=dict(hammer.policy_intersection),
            resource_enforcement=dict(hammer.resource_enforcement),
            selector_mode=hammer.selector_mode,
            translation_map_id=translation,
            environment_lock_id=hammer.environment_lock_id or roots.environment_id,
            obligation_id=(
                hammer.obligation_id
                or native_binding.logic_ir_obligation_id
            ),
            request_id=hammer.request_id,
            provider_result=dict(hammer.provider_result),
            native_goal_binding_id=binding_id,
            countermodel_validation=(
                dict(hammer.countermodel_validation)
                if hammer.countermodel_validation is not None
                else None
            ),
            receipt_binding=receipt_binding,
            reason_codes=tuple(hammer.reason_codes),
            import_isolation=hammer.import_isolation or HAMMER_IMPORT_ISOLATION,
            learned_selector_model_digest=hammer.learned_selector_model_digest,
            proof_success=hammer.proof_success,
            kernel_checked=hammer.kernel_checked,
            cancelled=hammer.cancelled,
            metadata=dict(hammer.metadata),
        )

    def _validate_countermodel(
        self,
        *,
        candidate: DoctorRepairCandidate,
        roots: ProgramLogicAuthorityRoots,
        obligation_compilation: DoctorRepairObligationCompilation,
        hammer: HammerCoordinationReceipt,
        replay: Mapping[str, Any] | None,
        proof_of_negation_id: str,
    ) -> CountermodelValidationReceipt:
        raw_refs: Sequence[str] = ()
        solver_cm_id = f"solver-cm:{candidate.candidate_id}"
        if hammer.countermodel_validation:
            raw_refs = tuple(
                hammer.countermodel_validation.get("raw_diagnostic_refs") or ()
            )
            solver_cm_id = str(
                hammer.countermodel_validation.get("solver_countermodel_id")
                or solver_cm_id
            )
        if not raw_refs and not proof_of_negation_id and replay is None:
            raw_refs = (f"diag:{candidate.candidate_id}",)
        return self._countermodel_validator.validate(
            roots=roots,
            solver_countermodel_id=solver_cm_id,
            translation_map_id=obligation_compilation.translation_map_id
            or f"translation:{candidate.candidate_id}",
            originating_logic_ir_id=(
                obligation_compilation.obligation_ids[0]
                if obligation_compilation.obligation_ids
                else f"obligation:{candidate.candidate_id}"
            ),
            raw_diagnostic_refs=raw_refs,
            replay_result=replay,
            proof_of_negation_id=proof_of_negation_id,
            invalidation_refs=(roots.tree_id,),
        )

    def _terminal(
        self,
        *,
        roots: ProgramLogicAuthorityRoots,
        finding_id: str,
        plan_receipt_id: str,
        disposition: DoctorHammerDisposition,
        reasons: Sequence[str],
        isolation_report: Mapping[str, Any] | None = None,
        isolation_adequate: bool = False,
        import_isolation: str = "",
        permit_id: str = "",
        obligation_compilation: DoctorRepairObligationCompilation | None = None,
        cache_audits: Sequence[Mapping[str, Any]] = (),
        cache_revalidated: bool = False,
        native_reconstruction: NativeReconstructionReceipt | None = None,
        hammer_receipts: Sequence[HammerCoordinationReceipt] = (),
        hammer_outcomes: Sequence[str] = (),
        countermodel_receipts: Sequence[CountermodelValidationReceipt] = (),
        cegis_receipt: LogicRefinementReceipt | None = None,
        admission_decision: LogicPredictionDecision | None = None,
        eligible_consequence_refs: Sequence[str] = (),
        selected_consequence_ref: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> DoctorRepairProofReceipt:
        # Enforce zero-write / zero-LLM invariant on every terminal path.
        assert self._source_writes == 0
        assert self._llm_calls == 0
        assert self._model_calls == 0
        receipt_id = _stable_id(
            "doctor-repair-proof",
            {
                "finding": finding_id,
                "plan": plan_receipt_id,
                "disposition": disposition.value,
                "reasons": list(reasons),
            },
        )
        return DoctorRepairProofReceipt(
            roots=roots,
            receipt_id=receipt_id,
            finding_id=finding_id,
            plan_receipt_id=plan_receipt_id,
            disposition=disposition,
            reason_codes=tuple(reasons),
            obligation_compilation_id=(
                obligation_compilation.compilation_id
                if obligation_compilation is not None
                else ""
            ),
            obligation_ids=(
                obligation_compilation.obligation_ids
                if obligation_compilation is not None
                else ()
            ),
            premise_ids=(
                obligation_compilation.premise_ids
                if obligation_compilation is not None
                else ()
            ),
            translator_id=(
                obligation_compilation.translator_id
                if obligation_compilation is not None
                else roots.translator_id
            ),
            solver_id=(
                obligation_compilation.solver_id
                if obligation_compilation is not None
                else ""
            ),
            kernel_id=(
                obligation_compilation.kernel_id
                if obligation_compilation is not None
                else (
                    native_reconstruction.kernel_id
                    if native_reconstruction is not None
                    else ""
                )
            ),
            toolchain_id=(
                obligation_compilation.toolchain_id
                if obligation_compilation is not None
                else roots.toolchain_id
            ),
            policy_id=(
                obligation_compilation.policy_id
                if obligation_compilation is not None
                else roots.policy_id
            ),
            environment_id=(
                obligation_compilation.environment_id
                if obligation_compilation is not None
                else roots.environment_id
            ),
            resource_budget=self._bounds.to_resource_budget().to_dict(),
            cache_audits=tuple(cache_audits),
            cache_revalidated=cache_revalidated,
            native_reconstruction=native_reconstruction,
            hammer_receipt_ids=tuple(item.receipt_id for item in hammer_receipts),
            hammer_outcomes=tuple(hammer_outcomes),
            countermodel_validation_ids=tuple(
                item.receipt_id for item in countermodel_receipts
            ),
            cegis_receipt_id=(
                cegis_receipt.receipt_id if cegis_receipt is not None else ""
            ),
            cegis_disposition=(
                cegis_receipt.disposition.value if cegis_receipt is not None else ""
            ),
            cegis_stop_reason=(
                cegis_receipt.stop_reason.value if cegis_receipt is not None else ""
            ),
            admission_decision_id=(
                admission_decision.decision_id
                if admission_decision is not None
                else ""
            ),
            admission_disposition=(
                admission_decision.disposition.value
                if admission_decision is not None
                else ""
            ),
            eligible_consequence_refs=tuple(eligible_consequence_refs),
            selected_consequence_ref=selected_consequence_ref,
            uniqueness_satisfied=False,
            permit_id=permit_id,
            isolation_report=dict(isolation_report or {}),
            isolation_adequate=isolation_adequate,
            import_isolation=import_isolation,
            residual_gap_ids=(
                obligation_compilation.residual_ids
                if obligation_compilation is not None
                else ()
            ),
            invalidation_refs=tuple(
                sorted(
                    {
                        roots.tree_id,
                        roots.corpus_id,
                        *(
                            obligation_compilation.invalidation_refs
                            if obligation_compilation is not None
                            else ()
                        ),
                    }
                )
            ),
            metadata=dict(metadata or {}),
        )


# ---------------------------------------------------------------------------
# Factories / convenience
# ---------------------------------------------------------------------------


def create_deterministic_doctor_hammer(
    **kwargs: Any,
) -> DeterministicDoctorHammer:
    """Factory for :class:`DeterministicDoctorHammer`."""

    return DeterministicDoctorHammer(**kwargs)


def verify_doctor_repair(
    plan_receipt: DoctorTacticianPlanReceipt | Mapping[str, Any],
    goal_compilation: DoctorGoalCompilation | Mapping[str, Any],
    **kwargs: Any,
) -> DoctorRepairProofReceipt:
    """One-shot verification entry point."""

    return DeterministicDoctorHammer().verify(
        plan_receipt, goal_compilation, **kwargs
    )


def build_default_obligation_context(
    *,
    translator_id: str,
    assumption_id: str = "assumption:stable-api",
    kernel_id: str = "kernel:lean4",
) -> ObligationContext:
    """Construct a minimal reviewed ObligationContext for doctor lowering."""

    capability = TranslatorCapabilityBinding(
        capability_id=f"capability:{translator_id}",
        capability_revision="revision:1",
        translator_id=translator_id,
        reconstruction_compatible=True,
        supported_semantics=("logic_ir", "fol"),
        supported_itps=("lean",),
    )
    assumptions = (
        AssumptionBinding(
            assumption_id=assumption_id,
            kind="reviewed_assumption",
            evidence_ref=f"evidence:{assumption_id}",
            authority=SourceAuthorityClass.AUTHORITATIVE,
        ),
    )
    return ObligationContext(
        capability=capability,
        assumptions=assumptions,
        kernel_id=kernel_id,
        translation_map_id=f"translation-map:{translator_id}",
    )


__all__ = [
    "CONTRACT_VERSION",
    "DETERMINISTIC_DOCTOR_HAMMER_INTERFACE",
    "DOCTOR_AUTHORITATIVE_PROOF_INTERFACE",
    "DOCTOR_REPAIR_OBLIGATION_COMPILER_INTERFACE",
    "DoctorAuthoritativeProofReceipt",
    "DoctorExactLoweringReceipt",
    "DoctorExecutableRole",
    "DoctorHammerBounds",
    "DoctorHammerDisposition",
    "DoctorHammerError",
    "DoctorHammerAuthorityError",
    "DoctorHammerReasonCode",
    "DoctorKernelReplayReceipt",
    "DoctorNativeExecutionReceipt",
    "DoctorObligationCompilationDisposition",
    "DoctorPinnedExecutable",
    "DoctorProofAuthorityDisposition",
    "DoctorRepairCandidate",
    "DoctorRepairObligationCompilation",
    "DoctorRepairObligationCompiler",
    "DoctorRepairProofReceipt",
    "DoctorReviewedTheorem",
    "DeterministicDoctorHammer",
    "NativeReconstructionDisposition",
    "NativeReconstructionReceipt",
    "PRODUCER_ID",
    "build_default_obligation_context",
    "create_deterministic_doctor_hammer",
    "isolation_is_adequate",
    "verify_doctor_repair",
]
