"""Goal-directed proof tactician orchestration facade.

``GoalDirectedProofTactician@1`` composes existing supervisor utilities behind
one restartable orchestration surface:

* formalization (Leanstral only after a confirmed goal, via injectable route)
* proof-directed retrieval
* proof scheduler
* proof-carrying planner resume/checkpoint
* Hammer / kernel structural gates
* Leanstral goal development
* SymAI / autoencoder bounded guidance (never authority)
* legal evidence adapter compatibility
* exact formal-verification caches
* corpus identity binding
* ZKP receipt binding (attests without raising receipt assurance)
* supervisor admission (fail-closed)

Conflict policy (FVT-G036 / FVT-027): this module owns the parent orchestration
facade and its integration test. Canonical datasets contracts are reached only
through injectable provider boundaries; semantics are never duplicated here.

Acceptance invariants:

* exact cache keys include tree / target / assumptions / provider / version /
  policy / bounds;
* model drafts and cache hits cannot bypass independent validation;
* proof-carrying execution is resumable from a durable checkpoint;
* ZKP binds an existing trusted receipt without increasing its assurance; and
* legal compatibility remains intact when legal evidence is in scope.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from .formal_verification_cache import (
    FormalVerificationCache,
    ProofCacheKey,
    build_proof_cache_key,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .proof_directed_retrieval import (
    PremiseRankingReport,
    expand_bounded_branches,
    rank_proof_premises,
    recall_at_k,
    select_top_k_ids,
)


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE: Final = "GoalDirectedProofTactician@1"
GOAL_DIRECTED_PROOF_TACTICIAN_VERSION: Final = "1.0.0"
GOAL_DIRECTED_PROOF_TACTICIAN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-proof-tactician@1"
)
TACTICIAN_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-cache-key@1"
)
TACTICIAN_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-request@1"
)
TACTICIAN_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-result@1"
)
TACTICIAN_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-checkpoint@1"
)
TACTICIAN_PHASE_RECORD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-phase@1"
)
TACTICIAN_ZKP_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-zkp-binding@1"
)
TACTICIAN_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-admission@1"
)
TACTICIAN_UTILITY_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-directed-tactician-utility@1"
)
PROOF_STATE_CLASSIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-state-classification@1"
)
TACTIC_PREMISE_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactic-premise-trace@1"
)
GOAL_DECOMPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-decomposition@1"
)
BRANCH_COST_FAILURE_PREDICTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/branch-cost-failure-prediction@1"
)
CURRICULUM_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-curriculum-projection@1"
)

DEFAULT_CHECKPOINT_FILENAME: Final = "goal_directed_tactician_checkpoint.json"
DEFAULT_MAX_PHASES: Final = 16
ABSOLUTE_MAX_PHASES: Final = 64
DEFAULT_TOP_K: Final = 8
ABSOLUTE_MAX_TOP_K: Final = 64
DEFAULT_MAX_BRANCH_FACTOR: Final = 8
ABSOLUTE_MAX_BRANCH_FACTOR: Final = 32
RANKING_BASIS_POINTS: Final = 10_000

# Nested trust-claim names that model drafts / cache payloads must not use to
# bypass independent validation (aligned with formal_verification_cache).
_AUTHORITY_BOOLEAN_CLAIMS: Final = frozenset(
    {
        "admission_claimed",
        "admitted",
        "authoritative",
        "can_mark_complete",
        "can_satisfy_completion",
        "complete",
        "completion_evidence",
        "implementation_conformant",
        "kernel_checked",
        "proof_success",
        "trusted",
        "verified",
    }
)
_AUTHORITY_ASSURANCE_CLAIMS: Final = frozenset(
    {
        "assurance",
        "assurance_level",
        "authority",
        "authoritative_assurance",
        "trust",
        "trust_level",
        "verdict",
    }
)
_UNTRUSTED_CLAIM_VALUES: Final = frozenset(
    {
        "",
        "candidate",
        "draft",
        "inconclusive",
        "none",
        "not_proved",
        "proposed",
        "unknown",
        "untrusted",
        "unverified",
    }
)


# ---------------------------------------------------------------------------
# Errors and enums
# ---------------------------------------------------------------------------


class GoalDirectedTacticianError(ContractValidationError):
    """Raised when a goal-directed tactician request or artifact is invalid."""


class GoalDirectedTacticianCancelled(GoalDirectedTacticianError):
    """Cooperative cancellation stopped the run before a terminal outcome."""


class UtilityRole(str, Enum):
    """Roles of composed supervisor utilities behind the facade."""

    FORMALIZATION = "formalization"
    RETRIEVAL = "retrieval"
    PROOF_SCHEDULER = "proof_scheduler"
    PROOF_CARRYING_PLANNER = "proof_carrying_planner"
    HAMMER = "hammer"
    KERNEL = "kernel"
    LEANSTRAL = "leanstral"
    SYMAI = "symai"
    AUTOENCODER = "autoencoder"
    LEGAL_ADAPTER = "legal_adapter"
    CACHE = "cache"
    CORPUS = "corpus"
    ZKP_BINDING = "zkp_binding"
    SUPERVISOR_ADMISSION = "supervisor_admission"
    CURRICULUM = "curriculum"


class UtilityAuthority(str, Enum):
    """Whether a utility may contribute proof authority."""

    AUTHORITY = "authority"
    GUIDANCE = "guidance"
    BINDING = "binding"
    ORCHESTRATION = "orchestration"
    COMPATIBILITY = "compatibility"


class TacticianPhase(str, Enum):
    """Ordered phases recorded on each auditable run."""

    ADMIT_REQUEST = "admit_request"
    BUILD_CACHE_KEY = "build_cache_key"
    LOAD_CHECKPOINT = "load_checkpoint"
    FORMALIZE = "formalize"
    RETRIEVE = "retrieve"
    SCHEDULE = "schedule"
    PLAN = "plan"
    HAMMER = "hammer"
    KERNEL = "kernel"
    LEANSTRAL = "leanstral"
    GUIDANCE = "guidance"
    LEGAL = "legal"
    VALIDATE = "validate"
    CACHE_LOOKUP = "cache_lookup"
    PROVE = "prove"
    CACHE_STORE = "cache_store"
    ZKP_BIND = "zkp_bind"
    CLASSIFY = "classify"
    RANK = "rank"
    DECOMPOSE = "decompose"
    PREDICT = "predict"
    CURRICULUM = "curriculum"
    ADMISSION = "admission"
    CHECKPOINT = "checkpoint"
    COMPLETE = "complete"


class PhaseStatus(str, Enum):
    """Per-phase outcome; never confuses guidance success with authority."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"
    REJECTED = "rejected"
    RESUMED = "resumed"
    CANCELLED = "cancelled"


class TacticianStopReason(str, Enum):
    """Terminal stop reasons for one bounded tactician run."""

    ADMITTED = "admitted"
    VALIDATION_FAILED = "validation_failed"
    CACHE_BYPASS_REJECTED = "cache_bypass_rejected"
    MODEL_BYPASS_REJECTED = "model_bypass_rejected"
    LEGAL_INCOMPATIBLE = "legal_incompatible"
    FORMALIZATION_REQUIRED = "formalization_required"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    OPEN = "open"


class AdmissionDecision(str, Enum):
    """Supervisor admission outcomes (fail-closed)."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ProofStateClass(str, Enum):
    """Classification of the current proof-search state (not curriculum authority)."""

    OPEN = "open"
    PARSED = "parsed"
    TYPED = "typed"
    DECOMPOSED = "decomposed"
    BRANCHING = "branching"
    CLOSED = "closed"
    STUCK = "stuck"
    PARSE_ERROR = "parse_error"
    TYPE_ERROR = "type_error"
    TIMEOUT = "timeout"
    COUNTEREXAMPLE = "counterexample"


class CurriculumClass(str, Enum):
    """Typed curriculum projections produced from traces.

    High curriculum authority is reserved for independently validated
    ``verified_success`` and checked ``counterexample`` traces.  ``timeout``
    is never falsehood.  ``parse_type`` never upgrades proof authority.
    """

    VERIFIED_SUCCESS = "verified_success"
    PARSE_TYPE = "parse_type"
    COUNTEREXAMPLE = "counterexample"
    TIMEOUT = "timeout"


class CurriculumAuthority(str, Enum):
    """Whether a curriculum projection may enter high-authority training."""

    NONE = "none"
    CANDIDATE = "candidate"
    HIGH = "high"


class RankedKind(str, Enum):
    TACTIC = "tactic"
    PREMISE = "premise"


class EvidenceSource(str, Enum):
    """Origin of evidence presented for validation / admission."""

    MODEL_DRAFT = "model_draft"
    CACHE_HIT = "cache_hit"
    INDEPENDENT_VALIDATION = "independent_validation"
    KERNEL = "kernel"
    SOLVER = "solver"
    ZKP_BINDING = "zkp_binding"
    LEGAL = "legal"
    GUIDANCE = "guidance"
    CHECKPOINT = "checkpoint"
    CURRICULUM = "curriculum"
    TRACE = "trace"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise GoalDirectedTacticianError(f"{field_name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise GoalDirectedTacticianError(f"{field_name} is required")
    if "\x00" in result:
        raise GoalDirectedTacticianError(f"{field_name} must not contain NUL bytes")
    return result


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        values = value
    else:
        raise GoalDirectedTacticianError("expected a sequence of strings")
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _positive(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GoalDirectedTacticianError(f"{name} must be an integer of at least {minimum}")
    return value


def _non_negative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoalDirectedTacticianError(f"{name} must be a non-negative integer")
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoalDirectedTacticianError(f"{field_name} must be an object")
    return value


def _public_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return {str(key): item for key, item in value.items() if not str(key).startswith("_")}


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({item.value for item in kind}))
        raise GoalDirectedTacticianError(f"{name} must be one of: {allowed}") from exc


def _assurance(value: Any) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    raw = getattr(value, "value", value)
    try:
        return AssuranceLevel(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        raise GoalDirectedTacticianError(
            f"assurance must be one of: "
            f"{', '.join(sorted({item.value for item in AssuranceLevel}))}"
        ) from exc


def _sha256_hex(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    else:
        payload = canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _cancelled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if callable(value):
        return bool(value())
    checker = getattr(value, "is_set", None)
    if callable(checker):
        return bool(checker())
    checker = getattr(value, "is_cancelled", None)
    if callable(checker):
        return bool(checker())
    raise GoalDirectedTacticianError(
        "cancelled must be a boolean, predicate, event, token, or None"
    )


def claims_authority(value: Any) -> bool:
    """Return True when nested payload claims proof authority without validation."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name).strip().casefold().replace("-", "_")
            if name in _AUTHORITY_BOOLEAN_CLAIMS and item not in (
                False,
                None,
                0,
                "",
            ):
                return True
            if name in _AUTHORITY_ASSURANCE_CLAIMS:
                normalized = (
                    str(getattr(item, "value", item) or "").strip().casefold().replace("-", "_")
                )
                if normalized not in _UNTRUSTED_CLAIM_VALUES:
                    return True
            if claims_authority(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(claims_authority(item) for item in value)
    return False


def reject_authority_bypass(
    payload: Any,
    *,
    source: EvidenceSource | str,
    independently_validated: bool,
) -> None:
    """Fail closed when model/cache evidence claims authority without validation."""

    src = _enum(source, EvidenceSource, "source")
    if independently_validated:
        return
    if src not in {EvidenceSource.MODEL_DRAFT, EvidenceSource.CACHE_HIT}:
        return
    if claims_authority(payload):
        label = "model draft" if src is EvidenceSource.MODEL_DRAFT else "cache hit"
        raise GoalDirectedTacticianError(f"{label} evidence cannot bypass independent validation")


# ---------------------------------------------------------------------------
# Proof-state classification, ranking, decomposition, curriculum
# ---------------------------------------------------------------------------


def _stage_status(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    return str(payload.get("status") or payload.get("outcome") or "").strip().lower()


def _truthy_timeout(payload: Any) -> bool:
    if payload is True:
        return True
    if not isinstance(payload, Mapping):
        return False
    if payload.get("occurred") is True or payload.get("timed_out") is True:
        return True
    status = str(payload.get("status") or payload.get("outcome") or "").strip().lower()
    return status in {"timed_out", "timeout", "time_out"}


def _has_counterexamples(payload: Mapping[str, Any]) -> bool:
    examples = payload.get("counterexamples")
    if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes, bytearray)):
        return any(bool(item) for item in examples)
    if payload.get("counterexample"):
        return True
    for key in ("prover_outcome", "kernel_outcome", "elaboration_outcome"):
        if _stage_status(payload.get(key) if isinstance(payload.get(key), Mapping) else None) == (
            "counterexample"
        ):
            return True
    return False


@dataclass(frozen=True)
class ProofStateClassification:
    """Typed classification of one proof state. Never itself proof authority."""

    SCHEMA: ClassVar[str] = PROOF_STATE_CLASSIFICATION_SCHEMA

    state_class: ProofStateClass
    curriculum_class: CurriculumClass
    independently_validated: bool = False
    kernel_verified: bool = False
    timeout_is_falsehood: bool = False
    reason_code: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_class", _enum(self.state_class, ProofStateClass, "state_class")
        )
        object.__setattr__(
            self,
            "curriculum_class",
            _enum(self.curriculum_class, CurriculumClass, "curriculum_class"),
        )
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        object.__setattr__(self, "kernel_verified", bool(self.kernel_verified))
        # Timeout is observational; it never becomes a falsehood label.
        object.__setattr__(self, "timeout_is_falsehood", False)
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(self, "details", _public_mapping(dict(self.details or {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_STATE_CLASSIFICATION_SCHEMA,
            "state_class": self.state_class.value,
            "curriculum_class": self.curriculum_class.value,
            "independently_validated": self.independently_validated,
            "kernel_verified": self.kernel_verified,
            "timeout_is_falsehood": self.timeout_is_falsehood,
            "reason_code": self.reason_code,
            "details": dict(self.details),
            "proof_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofStateClassification":
        value = _mapping(payload, field_name="proof_state")
        return cls(
            state_class=value.get("state_class", ProofStateClass.OPEN),
            curriculum_class=value.get("curriculum_class", CurriculumClass.PARSE_TYPE),
            independently_validated=bool(value.get("independently_validated", False)),
            kernel_verified=bool(value.get("kernel_verified", False)),
            reason_code=str(value.get("reason_code") or ""),
            details=dict(value.get("details") or {}),
        )


def classify_proof_state(
    payload: Mapping[str, Any] | None,
    *,
    independently_validated: bool = False,
    kernel_verified: bool = False,
) -> ProofStateClassification:
    """Classify a J2-shaped attempt or tactician context into a proof state."""

    value = dict(payload or {})
    validated = bool(independently_validated or value.get("independently_validated"))
    kernel_ok = bool(kernel_verified or value.get("kernel_verified"))
    kernel_status = _stage_status(value.get("kernel_outcome") if isinstance(value.get("kernel_outcome"), Mapping) else None)
    if kernel_status == "accepted" and validated:
        kernel_ok = True
    parse_status = _stage_status(
        value.get("parse_outcome") if isinstance(value.get("parse_outcome"), Mapping) else None
    )
    elab_status = _stage_status(
        value.get("elaboration_outcome")
        if isinstance(value.get("elaboration_outcome"), Mapping)
        else None
    )
    prover_status = _stage_status(
        value.get("prover_outcome") if isinstance(value.get("prover_outcome"), Mapping) else None
    )

    if _truthy_timeout(value.get("timeout")) or prover_status in {"timed_out", "timeout"}:
        return ProofStateClassification(
            state_class=ProofStateClass.TIMEOUT,
            curriculum_class=CurriculumClass.TIMEOUT,
            independently_validated=validated,
            kernel_verified=False,
            reason_code="timeout_is_not_falsehood",
            details={"timeout_is_falsehood": False},
        )
    if _has_counterexamples(value) or prover_status == "counterexample" or kernel_status == (
        "counterexample"
    ):
        return ProofStateClassification(
            state_class=ProofStateClass.COUNTEREXAMPLE,
            curriculum_class=CurriculumClass.COUNTEREXAMPLE,
            independently_validated=validated,
            kernel_verified=kernel_ok,
            reason_code="checked_counterexample" if validated else "candidate_counterexample",
        )
    if parse_status in {"parse_failed", "parse_error", "error"}:
        return ProofStateClassification(
            state_class=ProofStateClass.PARSE_ERROR,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=validated,
            kernel_verified=False,
            reason_code="parse_error",
        )
    if elab_status in {"elaboration_failed", "type_error", "rejected"}:
        return ProofStateClassification(
            state_class=ProofStateClass.TYPE_ERROR,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=validated,
            kernel_verified=False,
            reason_code="type_error",
        )
    if kernel_ok and validated:
        return ProofStateClassification(
            state_class=ProofStateClass.CLOSED,
            curriculum_class=CurriculumClass.VERIFIED_SUCCESS,
            independently_validated=True,
            kernel_verified=True,
            reason_code="independently_validated_kernel_success",
        )
    if parse_status == "parsed" and elab_status in {"elaborated", "typed", ""}:
        if value.get("decomposition") or value.get("children"):
            return ProofStateClassification(
                state_class=ProofStateClass.DECOMPOSED,
                curriculum_class=CurriculumClass.PARSE_TYPE,
                independently_validated=validated,
                reason_code="decomposed_unverified",
            )
        return ProofStateClassification(
            state_class=ProofStateClass.TYPED if elab_status in {"elaborated", "typed"} else ProofStateClass.PARSED,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=validated,
            reason_code="parse_type_candidate",
        )
    if value.get("stuck") or prover_status in {"stuck", "saturated"}:
        return ProofStateClassification(
            state_class=ProofStateClass.STUCK,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=validated,
            reason_code="stuck",
        )
    return ProofStateClassification(
        state_class=ProofStateClass.OPEN,
        curriculum_class=CurriculumClass.PARSE_TYPE,
        independently_validated=validated,
        kernel_verified=kernel_ok,
        reason_code="open_unclassified",
    )


def curriculum_authority_for(
    curriculum_class: CurriculumClass | str,
    *,
    independently_validated: bool,
    kernel_verified: bool = False,
) -> CurriculumAuthority:
    """High authority only for validated verified-success or checked counterexamples."""

    cls = _enum(curriculum_class, CurriculumClass, "curriculum_class")
    if cls is CurriculumClass.TIMEOUT:
        return CurriculumAuthority.CANDIDATE
    if cls is CurriculumClass.PARSE_TYPE:
        return CurriculumAuthority.CANDIDATE
    if cls is CurriculumClass.VERIFIED_SUCCESS:
        if independently_validated and kernel_verified:
            return CurriculumAuthority.HIGH
        return CurriculumAuthority.CANDIDATE
    if cls is CurriculumClass.COUNTEREXAMPLE:
        if independently_validated:
            return CurriculumAuthority.HIGH
        return CurriculumAuthority.CANDIDATE
    return CurriculumAuthority.NONE


@dataclass(frozen=True)
class SubgoalNode:
    """One bounded child of a goal decomposition. Candidate only."""

    subgoal_id: str
    statement: str
    depends_on: tuple[str, ...] = ()
    predicted_cost_ms: int = 0
    predicted_failure_bps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "subgoal_id", _text(self.subgoal_id, field_name="subgoal_id"))
        object.__setattr__(self, "statement", _text(self.statement, field_name="statement"))
        object.__setattr__(self, "depends_on", _strings(self.depends_on))
        object.__setattr__(
            self,
            "predicted_cost_ms",
            _non_negative(self.predicted_cost_ms, "predicted_cost_ms"),
        )
        object.__setattr__(
            self,
            "predicted_failure_bps",
            _non_negative(self.predicted_failure_bps, "predicted_failure_bps"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "statement": self.statement,
            "depends_on": list(self.depends_on),
            "predicted_cost_ms": self.predicted_cost_ms,
            "predicted_failure_bps": self.predicted_failure_bps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubgoalNode":
        value = _mapping(payload, field_name="subgoal")
        return cls(
            subgoal_id=str(value.get("subgoal_id") or value.get("id") or ""),
            statement=str(value.get("statement") or value.get("text") or ""),
            depends_on=tuple(value.get("depends_on") or ()),
            predicted_cost_ms=int(value.get("predicted_cost_ms", 0)),
            predicted_failure_bps=int(value.get("predicted_failure_bps", 0)),
        )


@dataclass(frozen=True)
class GoalDecomposition:
    """Bounded goal decomposition. Never completion or proof authority."""

    SCHEMA: ClassVar[str] = GOAL_DECOMPOSITION_SCHEMA

    parent_goal_id: str
    children: tuple[SubgoalNode, ...]
    max_branch_factor: int = DEFAULT_MAX_BRANCH_FACTOR
    truncated: bool = False
    reason_code: str = "bounded_decomposition"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parent_goal_id", _text(self.parent_goal_id, field_name="parent_goal_id")
        )
        nodes: list[SubgoalNode] = []
        for item in self.children or ():
            if isinstance(item, SubgoalNode):
                nodes.append(item)
            else:
                nodes.append(SubgoalNode.from_dict(_mapping(item, field_name="child")))
        object.__setattr__(self, "children", tuple(nodes))
        object.__setattr__(
            self,
            "max_branch_factor",
            min(
                ABSOLUTE_MAX_BRANCH_FACTOR,
                _positive(self.max_branch_factor, "max_branch_factor"),
            ),
        )
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        ids = [item.subgoal_id for item in self.children]
        if len(ids) != len(set(ids)):
            raise GoalDirectedTacticianError("decomposition child ids must be unique")
        known = set(ids)
        for item in self.children:
            unknown = [dep for dep in item.depends_on if dep not in known]
            if unknown:
                raise GoalDirectedTacticianError(
                    "decomposition references unknown subgoal: " + ", ".join(unknown)
                )
        if len(self.children) > self.max_branch_factor:
            raise GoalDirectedTacticianError(
                f"decomposition exceeds max_branch_factor={self.max_branch_factor}"
            )

    @property
    def proof_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_DECOMPOSITION_SCHEMA,
            "parent_goal_id": self.parent_goal_id,
            "children": [item.to_dict() for item in self.children],
            "max_branch_factor": self.max_branch_factor,
            "truncated": self.truncated,
            "reason_code": self.reason_code,
            "proof_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDecomposition":
        value = _mapping(payload, field_name="decomposition")
        return cls(
            parent_goal_id=str(value.get("parent_goal_id") or value.get("goal_id") or ""),
            children=tuple(value.get("children") or ()),
            max_branch_factor=int(value.get("max_branch_factor", DEFAULT_MAX_BRANCH_FACTOR)),
            truncated=bool(value.get("truncated", False)),
            reason_code=str(value.get("reason_code") or "bounded_decomposition"),
        )


def decompose_goal(
    parent_goal_id: str,
    children: Sequence[SubgoalNode | Mapping[str, Any]],
    *,
    max_branch_factor: int = DEFAULT_MAX_BRANCH_FACTOR,
    statement: str = "",
) -> GoalDecomposition:
    """Build a fail-closed bounded decomposition. Over-branching is rejected."""

    del statement
    bound = min(ABSOLUTE_MAX_BRANCH_FACTOR, _positive(max_branch_factor, "max_branch_factor"))
    if len(tuple(children or ())) > bound:
        raise GoalDirectedTacticianError(f"decomposition exceeds max_branch_factor={bound}")
    return GoalDecomposition(
        parent_goal_id=parent_goal_id,
        children=tuple(children or ()),
        max_branch_factor=bound,
    )


def expand_goal_branches(
    adjacency: Mapping[str, Sequence[str]],
    roots: Sequence[str],
    *,
    max_branch_factor: int = DEFAULT_MAX_BRANCH_FACTOR,
    max_depth: int = 8,
    fail_closed: bool = True,
) -> Any:
    """Bounded branch expansion for goal graphs. Fail-closed by default."""

    return expand_bounded_branches(
        adjacency,
        roots,
        max_branch_factor=max_branch_factor,
        max_depth=max_depth,
        fail_closed=fail_closed,
    )


@dataclass(frozen=True)
class BranchCostFailurePrediction:
    """Guidance-only prediction of branching, cost, and failure. Never authority."""

    SCHEMA: ClassVar[str] = BRANCH_COST_FAILURE_PREDICTION_SCHEMA

    predicted_branch_factor: int
    predicted_cost_ms: int
    predicted_failure_bps: int
    bounded_branch_factor: int = DEFAULT_MAX_BRANCH_FACTOR
    reason_code: str = "guidance_prediction"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_branch_factor",
            _non_negative(self.predicted_branch_factor, "predicted_branch_factor"),
        )
        object.__setattr__(
            self, "predicted_cost_ms", _non_negative(self.predicted_cost_ms, "predicted_cost_ms")
        )
        object.__setattr__(
            self,
            "predicted_failure_bps",
            _non_negative(self.predicted_failure_bps, "predicted_failure_bps"),
        )
        if self.predicted_failure_bps > RANKING_BASIS_POINTS:
            raise GoalDirectedTacticianError("predicted_failure_bps cannot exceed 10000")
        object.__setattr__(
            self,
            "bounded_branch_factor",
            min(
                ABSOLUTE_MAX_BRANCH_FACTOR,
                _positive(self.bounded_branch_factor, "bounded_branch_factor"),
            ),
        )
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())

    @property
    def proof_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BRANCH_COST_FAILURE_PREDICTION_SCHEMA,
            "predicted_branch_factor": self.predicted_branch_factor,
            "predicted_cost_ms": self.predicted_cost_ms,
            "predicted_failure_bps": self.predicted_failure_bps,
            "bounded_branch_factor": self.bounded_branch_factor,
            "reason_code": self.reason_code,
            "authority": UtilityAuthority.GUIDANCE.value,
            "proof_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BranchCostFailurePrediction":
        value = _mapping(payload, field_name="prediction")
        return cls(
            predicted_branch_factor=int(value.get("predicted_branch_factor", 0)),
            predicted_cost_ms=int(value.get("predicted_cost_ms", 0)),
            predicted_failure_bps=int(value.get("predicted_failure_bps", 0)),
            bounded_branch_factor=int(value.get("bounded_branch_factor", DEFAULT_MAX_BRANCH_FACTOR)),
            reason_code=str(value.get("reason_code") or "guidance_prediction"),
        )


def predict_branch_cost_failure(
    *,
    ranking: PremiseRankingReport | None = None,
    decomposition: GoalDecomposition | None = None,
    max_branch_factor: int = DEFAULT_MAX_BRANCH_FACTOR,
) -> BranchCostFailurePrediction:
    """Predict branch width, cost, and failure from ranking and decomposition."""

    bound = min(ABSOLUTE_MAX_BRANCH_FACTOR, _positive(max_branch_factor, "max_branch_factor"))
    branch = 1
    cost = 0
    failures: list[int] = []
    if ranking is not None:
        cost += int(ranking.cost_ms)
        if ranking.items:
            branch = max(branch, max(item.predicted_branch_factor for item in ranking.items))
            failures.extend(item.predicted_failure_bps for item in ranking.items)
    if decomposition is not None:
        branch = max(branch, len(decomposition.children))
        cost += sum(item.predicted_cost_ms for item in decomposition.children)
        failures.extend(item.predicted_failure_bps for item in decomposition.children)
    failure = max(failures) if failures else 0
    return BranchCostFailurePrediction(
        predicted_branch_factor=min(branch, bound),
        predicted_cost_ms=cost,
        predicted_failure_bps=min(failure, RANKING_BASIS_POINTS),
        bounded_branch_factor=bound,
        reason_code="guidance_prediction",
    )


@dataclass(frozen=True)
class TacticPremiseTrace:
    """Content-addressed tactic or premise trace. Candidate unless independently checked."""

    SCHEMA: ClassVar[str] = TACTIC_PREMISE_TRACE_SCHEMA

    trace_id: str
    kind: RankedKind
    goal_id: str
    state_digest: str
    item_ids: tuple[str, ...]
    model_revision: str = ""
    tool_revision: str = ""
    outcome: str = "candidate"
    independently_validated: bool = False
    source_faithful: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _text(self.trace_id, field_name="trace_id"))
        object.__setattr__(self, "kind", _enum(self.kind, RankedKind, "kind"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, field_name="goal_id"))
        object.__setattr__(self, "state_digest", str(self.state_digest or "").strip())
        object.__setattr__(self, "item_ids", _strings(self.item_ids))
        object.__setattr__(self, "model_revision", str(self.model_revision or "").strip())
        object.__setattr__(self, "tool_revision", str(self.tool_revision or "").strip())
        object.__setattr__(self, "outcome", str(self.outcome or "candidate").strip() or "candidate")
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        # Tactic success is never source-faithfulness proof.
        object.__setattr__(self, "source_faithful", False)
        object.__setattr__(self, "metadata", _public_mapping(dict(self.metadata or {})))

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": TACTIC_PREMISE_TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "kind": self.kind.value,
            "goal_id": self.goal_id,
            "state_digest": self.state_digest,
            "item_ids": list(self.item_ids),
            "model_revision": self.model_revision,
            "tool_revision": self.tool_revision,
            "outcome": self.outcome,
            "independently_validated": self.independently_validated,
            "source_faithful": self.source_faithful,
            "metadata": dict(self.metadata),
            "proof_authority": False,
        }
        if include_identity:
            payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TacticPremiseTrace":
        value = _mapping(payload, field_name="trace")
        return cls(
            trace_id=str(value.get("trace_id") or ""),
            kind=value.get("kind", RankedKind.TACTIC),
            goal_id=str(value.get("goal_id") or ""),
            state_digest=str(value.get("state_digest") or ""),
            item_ids=tuple(value.get("item_ids") or ()),
            model_revision=str(value.get("model_revision") or ""),
            tool_revision=str(value.get("tool_revision") or ""),
            outcome=str(value.get("outcome") or "candidate"),
            independently_validated=bool(value.get("independently_validated", False)),
            metadata=dict(value.get("metadata") or {}),
        )


def build_tactic_premise_trace(
    *,
    kind: RankedKind | str,
    goal_id: str,
    item_ids: Sequence[str],
    state_digest: str = "",
    model_revision: str = "",
    tool_revision: str = "",
    outcome: str = "candidate",
    independently_validated: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> TacticPremiseTrace:
    kind_enum = _enum(kind, RankedKind, "kind")
    digest = state_digest or f"sha256:{_sha256_hex({'goal_id': goal_id, 'items': list(item_ids)})}"
    trace_id = f"trace:{kind_enum.value}:sha256:{_sha256_hex({'goal': goal_id, 'items': list(item_ids), 'state': digest})}"
    return TacticPremiseTrace(
        trace_id=trace_id,
        kind=kind_enum,
        goal_id=goal_id,
        state_digest=digest,
        item_ids=tuple(item_ids),
        model_revision=model_revision,
        tool_revision=tool_revision,
        outcome=outcome,
        independently_validated=independently_validated,
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class CurriculumProjection:
    """Typed curriculum projection. High authority is fail-closed."""

    SCHEMA: ClassVar[str] = CURRICULUM_PROJECTION_SCHEMA

    curriculum_class: CurriculumClass
    authority: CurriculumAuthority
    independently_validated: bool
    trace_ids: tuple[str, ...] = ()
    timeout_is_falsehood: bool = False
    source_faithful: bool = False
    reason_code: str = ""
    classification: ProofStateClassification | None = None
    ranking: PremiseRankingReport | Mapping[str, Any] | None = None
    decomposition: GoalDecomposition | None = None
    prediction: BranchCostFailurePrediction | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "curriculum_class",
            _enum(self.curriculum_class, CurriculumClass, "curriculum_class"),
        )
        object.__setattr__(self, "authority", _enum(self.authority, CurriculumAuthority, "authority"))
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        object.__setattr__(self, "trace_ids", _strings(self.trace_ids))
        object.__setattr__(self, "timeout_is_falsehood", False)
        object.__setattr__(self, "source_faithful", False)
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        if self.classification is not None and not isinstance(
            self.classification, ProofStateClassification
        ):
            object.__setattr__(
                self,
                "classification",
                ProofStateClassification.from_dict(
                    _mapping(self.classification, field_name="classification")
                ),
            )
        if self.decomposition is not None and not isinstance(self.decomposition, GoalDecomposition):
            object.__setattr__(
                self,
                "decomposition",
                GoalDecomposition.from_dict(_mapping(self.decomposition, field_name="decomposition")),
            )
        if self.prediction is not None and not isinstance(
            self.prediction, BranchCostFailurePrediction
        ):
            object.__setattr__(
                self,
                "prediction",
                BranchCostFailurePrediction.from_dict(
                    _mapping(self.prediction, field_name="prediction")
                ),
            )
        if self.authority is CurriculumAuthority.HIGH:
            if not self.independently_validated:
                raise GoalDirectedTacticianError(
                    "high curriculum authority requires independently validated traces"
                )
            if self.curriculum_class not in {
                CurriculumClass.VERIFIED_SUCCESS,
                CurriculumClass.COUNTEREXAMPLE,
            }:
                raise GoalDirectedTacticianError(
                    "timeout and parse_type traces cannot upgrade curriculum authority"
                )
            if (
                self.curriculum_class is CurriculumClass.VERIFIED_SUCCESS
                and self.classification is not None
                and not self.classification.kernel_verified
            ):
                raise GoalDirectedTacticianError(
                    "verified_success curriculum requires a kernel-verified classification"
                )

    @property
    def upgrades_curriculum_authority(self) -> bool:
        return self.authority is CurriculumAuthority.HIGH

    def to_dict(self) -> dict[str, Any]:
        ranking_payload: Any = None
        if self.ranking is not None:
            ranking_payload = (
                self.ranking.to_dict()
                if hasattr(self.ranking, "to_dict")
                else dict(self.ranking)
            )
        return {
            "schema": CURRICULUM_PROJECTION_SCHEMA,
            "curriculum_class": self.curriculum_class.value,
            "authority": self.authority.value,
            "independently_validated": self.independently_validated,
            "trace_ids": list(self.trace_ids),
            "timeout_is_falsehood": self.timeout_is_falsehood,
            "source_faithful": self.source_faithful,
            "reason_code": self.reason_code,
            "upgrades_curriculum_authority": self.upgrades_curriculum_authority,
            "classification": (
                self.classification.to_dict() if self.classification is not None else None
            ),
            "ranking": ranking_payload,
            "decomposition": (
                self.decomposition.to_dict() if self.decomposition is not None else None
            ),
            "prediction": (self.prediction.to_dict() if self.prediction is not None else None),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CurriculumProjection":
        value = _mapping(payload, field_name="curriculum")
        ranking = value.get("ranking")
        return cls(
            curriculum_class=value.get("curriculum_class", CurriculumClass.PARSE_TYPE),
            authority=value.get("authority", CurriculumAuthority.NONE),
            independently_validated=bool(value.get("independently_validated", False)),
            trace_ids=tuple(value.get("trace_ids") or ()),
            reason_code=str(value.get("reason_code") or ""),
            classification=value.get("classification"),
            ranking=dict(ranking) if isinstance(ranking, Mapping) else ranking,
            decomposition=value.get("decomposition"),
            prediction=value.get("prediction"),
        )


def project_curriculum(
    classification: ProofStateClassification,
    *,
    traces: Sequence[TacticPremiseTrace] = (),
    ranking: PremiseRankingReport | Mapping[str, Any] | None = None,
    decomposition: GoalDecomposition | None = None,
    prediction: BranchCostFailurePrediction | None = None,
    independently_validated: bool | None = None,
) -> CurriculumProjection:
    """Project a classified trace into a typed curriculum class.

    Only independently validated verified-success and checked counterexample
    traces receive high curriculum authority. Timeout is never falsehood.
    Tactic success never implies source-faithfulness.
    """

    validated = (
        classification.independently_validated
        if independently_validated is None
        else bool(independently_validated)
    )
    # Unvalidated traces cannot inherit a high-authority classification.
    if classification.curriculum_class is CurriculumClass.VERIFIED_SUCCESS and not (
        validated and classification.kernel_verified
    ):
        effective = ProofStateClassification(
            state_class=classification.state_class,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=validated,
            kernel_verified=False,
            reason_code="unvalidated_success_is_not_curriculum_authority",
            details=dict(classification.details),
        )
    else:
        effective = classification
        if not validated and classification.curriculum_class is CurriculumClass.COUNTEREXAMPLE:
            effective = ProofStateClassification(
                state_class=classification.state_class,
                curriculum_class=CurriculumClass.COUNTEREXAMPLE,
                independently_validated=False,
                kernel_verified=False,
                reason_code="candidate_counterexample",
                details=dict(classification.details),
            )
    authority = curriculum_authority_for(
        effective.curriculum_class,
        independently_validated=validated,
        kernel_verified=effective.kernel_verified,
    )
    reason = effective.reason_code
    if authority is CurriculumAuthority.HIGH:
        reason = reason or "validated_trace_upgrades_curriculum"
    elif effective.curriculum_class is CurriculumClass.TIMEOUT:
        reason = reason or "timeout_is_not_falsehood"
    else:
        reason = reason or "candidate_curriculum"
    return CurriculumProjection(
        curriculum_class=effective.curriculum_class,
        authority=authority,
        independently_validated=validated,
        trace_ids=tuple(item.trace_id for item in traces),
        reason_code=reason,
        classification=effective,
        ranking=ranking,
        decomposition=decomposition,
        prediction=prediction,
    )


def rank_tactics_and_premises(
    candidates: Sequence[Mapping[str, Any]],
    *,
    k: int = DEFAULT_TOP_K,
    relevant_ids: Sequence[str] = (),
    kind: RankedKind | str = RankedKind.TACTIC,
) -> PremiseRankingReport:
    """Rank tactic or premise candidates with top-k / Recall@k / cost."""

    bound_k = min(ABSOLUTE_MAX_TOP_K, _positive(k, "k"))
    rank_kind = _enum(kind, RankedKind, "kind")
    return rank_proof_premises(
        candidates,
        k=bound_k,
        relevant_ids=relevant_ids,
        kind=rank_kind.value,
    )


# ---------------------------------------------------------------------------
# Exact cache key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactTacticianCacheKey:
    """Exact cache identity for one goal-directed tactician invocation.

    Required components (acceptance): tree, target, assumptions, provider,
    version, policy, bounds. Optional corpus / toolchain fields tighten reuse
    without relaxing the mandatory set.
    """

    SCHEMA: ClassVar[str] = TACTICIAN_CACHE_KEY_SCHEMA

    tree_id: str
    target_id: str
    assumption_ids: tuple[str, ...]
    provider_id: str
    provider_version: str
    policy_id: str
    bounds: Mapping[str, Any]
    corpus_id: str = ""
    corpus_version: str = ""
    toolchain_id: str = ""
    obligation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _text(self.tree_id, field_name="tree_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, field_name="target_id"))
        object.__setattr__(self, "assumption_ids", _strings(self.assumption_ids))
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, field_name="provider_id"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(self.provider_version, field_name="provider_version"),
        )
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        if not isinstance(self.bounds, Mapping):
            raise GoalDirectedTacticianError("bounds must be an object")
        # Stable projection: reject empty bounds object — bounds must be explicit.
        if not dict(self.bounds):
            raise GoalDirectedTacticianError("bounds must be non-empty")
        object.__setattr__(self, "bounds", dict(self.bounds))
        object.__setattr__(self, "corpus_id", str(self.corpus_id or "").strip())
        object.__setattr__(self, "corpus_version", str(self.corpus_version or "").strip())
        object.__setattr__(self, "toolchain_id", str(self.toolchain_id or "").strip())
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "").strip())

    @property
    def bound_digest(self) -> str:
        return f"sha256:{_sha256_hex(self.bounds)}"

    @property
    def assumptions_digest(self) -> str:
        return f"sha256:{_sha256_hex(list(self.assumption_ids))}"

    def _identity_payload(self) -> dict[str, Any]:
        """Canonical fields that define cache identity (no derived digests)."""

        return {
            "tree_id": self.tree_id,
            "target_id": self.target_id,
            "assumption_ids": list(self.assumption_ids),
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "policy_id": self.policy_id,
            "bounds": dict(self.bounds),
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "toolchain_id": self.toolchain_id,
            "obligation_id": self.obligation_id,
        }

    @property
    def key_id(self) -> str:
        return f"tactician-cache-key:sha256:{_sha256_hex(self._identity_payload())}"

    @property
    def digest(self) -> str:
        return self.key_id

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self._identity_payload(),
            "assumptions_digest": self.assumptions_digest,
            "bound_digest": self.bound_digest,
            "key_id": self.key_id,
        }
        if include_schema:
            payload = {"schema": TACTICIAN_CACHE_KEY_SCHEMA, **payload}
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactTacticianCacheKey":
        value = _mapping(payload, field_name="cache_key")
        if value.get("schema") not in {None, TACTICIAN_CACHE_KEY_SCHEMA}:
            raise GoalDirectedTacticianError("unsupported tactician cache-key schema")
        return cls(
            tree_id=value.get("tree_id", ""),
            target_id=value.get("target_id", "")
            or value.get("obligation_id", "")
            or value.get("goal_id", ""),
            assumption_ids=tuple(value.get("assumption_ids") or ()),
            provider_id=value.get("provider_id", ""),
            provider_version=value.get("provider_version", "") or value.get("version", ""),
            policy_id=value.get("policy_id", ""),
            bounds=dict(value.get("bounds") or {}),
            corpus_id=value.get("corpus_id", ""),
            corpus_version=value.get("corpus_version", ""),
            toolchain_id=value.get("toolchain_id", ""),
            obligation_id=value.get("obligation_id", ""),
        )

    def to_proof_cache_key(self) -> ProofCacheKey:
        """Project into the shared formal-verification cache key surface."""

        return build_proof_cache_key(
            obligation=self.obligation_id or self.target_id,
            premises=list(self.assumption_ids),
            translator=self.provider_id,
            solver=self.provider_id,
            kernel="kernel:local",
            toolchain=self.toolchain_id or self.provider_version,
            theorem_registry=self.corpus_id or "corpus:none",
            policy=self.policy_id,
            resource_budget=dict(self.bounds),
            candidate_tree=self.tree_id,
        )

    def matches(self, other: "ExactTacticianCacheKey") -> bool:
        return self.key_id == other.key_id


def build_exact_tactician_cache_key(
    *,
    tree_id: str,
    target_id: str,
    assumption_ids: Sequence[str] | None = None,
    provider_id: str,
    provider_version: str,
    policy_id: str,
    bounds: Mapping[str, Any],
    corpus_id: str = "",
    corpus_version: str = "",
    toolchain_id: str = "",
    obligation_id: str = "",
) -> ExactTacticianCacheKey:
    """Build the exact cache key required by GoalDirectedProofTactician@1."""

    return ExactTacticianCacheKey(
        tree_id=tree_id,
        target_id=target_id,
        assumption_ids=tuple(assumption_ids or ()),
        provider_id=provider_id,
        provider_version=provider_version,
        policy_id=policy_id,
        bounds=dict(bounds),
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        toolchain_id=toolchain_id,
        obligation_id=obligation_id,
    )


# ---------------------------------------------------------------------------
# Utility bindings / phases / ZKP / admission / checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtilityBinding:
    """One composed utility declared behind the tactician facade."""

    SCHEMA: ClassVar[str] = TACTICIAN_UTILITY_BINDING_SCHEMA

    role: UtilityRole
    utility_id: str
    authority: UtilityAuthority
    version: str = "1"
    available: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum(self.role, UtilityRole, "role"))
        object.__setattr__(
            self,
            "utility_id",
            _text(self.utility_id, field_name="utility_id"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, UtilityAuthority, "authority"),
        )
        object.__setattr__(self, "version", _text(self.version, field_name="version"))
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "notes", str(self.notes or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_UTILITY_BINDING_SCHEMA,
            "role": self.role.value,
            "utility_id": self.utility_id,
            "authority": self.authority.value,
            "version": self.version,
            "available": self.available,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UtilityBinding":
        value = _mapping(payload, field_name="utility")
        return cls(
            role=value.get("role", UtilityRole.SUPERVISOR_ADMISSION),
            utility_id=value.get("utility_id", ""),
            authority=value.get("authority", UtilityAuthority.ORCHESTRATION),
            version=value.get("version", "1"),
            available=bool(value.get("available", True)),
            notes=value.get("notes", ""),
        )


def default_utility_bindings() -> tuple[UtilityBinding, ...]:
    """Declare the composed utility surface without importing package-private paths."""

    return (
        UtilityBinding(
            role=UtilityRole.FORMALIZATION,
            utility_id="formalized-goal-development-route@1",
            authority=UtilityAuthority.ORCHESTRATION,
            notes="Leanstral only after confirmed FormalGoal",
        ),
        UtilityBinding(
            role=UtilityRole.RETRIEVAL,
            utility_id="proof-directed-retrieval@1",
            authority=UtilityAuthority.GUIDANCE,
        ),
        UtilityBinding(
            role=UtilityRole.PROOF_SCHEDULER,
            utility_id="proof-scheduler@1",
            authority=UtilityAuthority.ORCHESTRATION,
        ),
        UtilityBinding(
            role=UtilityRole.PROOF_CARRYING_PLANNER,
            utility_id="proof-carrying-planner@1",
            authority=UtilityAuthority.ORCHESTRATION,
            notes="resumable paired workflow artifacts",
        ),
        UtilityBinding(
            role=UtilityRole.HAMMER,
            utility_id="hammer-structural-gate@1",
            authority=UtilityAuthority.GUIDANCE,
            notes="structural gate only; never completion authority",
        ),
        UtilityBinding(
            role=UtilityRole.KERNEL,
            utility_id="independent-kernel-verifier@1",
            authority=UtilityAuthority.AUTHORITY,
        ),
        UtilityBinding(
            role=UtilityRole.LEANSTRAL,
            utility_id="leanstral-goal-development@1",
            authority=UtilityAuthority.GUIDANCE,
        ),
        UtilityBinding(
            role=UtilityRole.SYMAI,
            utility_id="symai-bounded-guidance@1",
            authority=UtilityAuthority.GUIDANCE,
        ),
        UtilityBinding(
            role=UtilityRole.AUTOENCODER,
            utility_id="autoencoder-bounded-diagnostics@1",
            authority=UtilityAuthority.GUIDANCE,
        ),
        UtilityBinding(
            role=UtilityRole.LEGAL_ADAPTER,
            utility_id="supervisor-legal-constraint-adapter@1",
            authority=UtilityAuthority.COMPATIBILITY,
        ),
        UtilityBinding(
            role=UtilityRole.CACHE,
            utility_id="formal-verification-cache@1",
            authority=UtilityAuthority.BINDING,
        ),
        UtilityBinding(
            role=UtilityRole.CORPUS,
            utility_id="proof-tactician-corpus@1",
            authority=UtilityAuthority.BINDING,
        ),
        UtilityBinding(
            role=UtilityRole.ZKP_BINDING,
            utility_id="receipt-attestation@1",
            authority=UtilityAuthority.BINDING,
            notes="binds existing trusted receipt without raising assurance",
        ),
        UtilityBinding(
            role=UtilityRole.SUPERVISOR_ADMISSION,
            utility_id="supervisor-proof-admission@1",
            authority=UtilityAuthority.ORCHESTRATION,
        ),
        UtilityBinding(
            role=UtilityRole.CURRICULUM,
            utility_id="proof-state-curriculum-projection@1",
            authority=UtilityAuthority.GUIDANCE,
            notes="only independently validated traces upgrade curriculum authority",
        ),
    )


@dataclass(frozen=True)
class PhaseRecord:
    """Auditable record of one phase of a tactician run."""

    SCHEMA: ClassVar[str] = TACTICIAN_PHASE_RECORD_SCHEMA

    phase: TacticianPhase
    status: PhaseStatus
    utility_role: UtilityRole | None = None
    reason_code: str = ""
    evidence_source: EvidenceSource | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(self.phase, TacticianPhase, "phase"))
        object.__setattr__(self, "status", _enum(self.status, PhaseStatus, "status"))
        if self.utility_role is not None:
            object.__setattr__(
                self,
                "utility_role",
                _enum(self.utility_role, UtilityRole, "utility_role"),
            )
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        if self.evidence_source is not None:
            object.__setattr__(
                self,
                "evidence_source",
                _enum(self.evidence_source, EvidenceSource, "evidence_source"),
            )
        object.__setattr__(self, "details", _public_mapping(dict(self.details or {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_PHASE_RECORD_SCHEMA,
            "phase": self.phase.value,
            "status": self.status.value,
            "utility_role": (self.utility_role.value if self.utility_role is not None else ""),
            "reason_code": self.reason_code,
            "evidence_source": (
                self.evidence_source.value if self.evidence_source is not None else ""
            ),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseRecord":
        value = _mapping(payload, field_name="phase_record")
        role = value.get("utility_role") or None
        source = value.get("evidence_source") or None
        return cls(
            phase=value.get("phase", TacticianPhase.ADMIT_REQUEST),
            status=value.get("status", PhaseStatus.OK),
            utility_role=role,
            reason_code=value.get("reason_code", ""),
            evidence_source=source,
            details=dict(value.get("details") or {}),
        )


@dataclass(frozen=True)
class ZkpReceiptBinding:
    """Public binding of a ZKP attestation over an existing trusted receipt.

    The binding never upgrades :attr:`receipt_assurance`. Attestation is a
    separate binding layer; callers that require ``ATTESTED`` must already hold
    a kernel-trusted receipt and a successful cryptographic binding, but the
    bound receipt's own assurance projection remains unchanged.
    """

    SCHEMA: ClassVar[str] = TACTICIAN_ZKP_BINDING_SCHEMA

    receipt_id: str
    receipt_assurance: AssuranceLevel
    bound_assurance: AssuranceLevel
    circuit_id: str
    backend_id: str
    verification_key_id: str
    attestation_artifact_id: str = ""
    statement_id: str = ""
    assurance_increased: bool = False
    reason_code: str = "bound_existing_trusted_receipt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, field_name="receipt_id"))
        object.__setattr__(
            self,
            "receipt_assurance",
            _assurance(self.receipt_assurance),
        )
        object.__setattr__(self, "bound_assurance", _assurance(self.bound_assurance))
        object.__setattr__(self, "circuit_id", _text(self.circuit_id, field_name="circuit_id"))
        object.__setattr__(self, "backend_id", _text(self.backend_id, field_name="backend_id"))
        object.__setattr__(
            self,
            "verification_key_id",
            _text(self.verification_key_id, field_name="verification_key_id"),
        )
        object.__setattr__(
            self,
            "attestation_artifact_id",
            str(self.attestation_artifact_id or "").strip(),
        )
        object.__setattr__(self, "statement_id", str(self.statement_id or "").strip())
        # Fail closed: ZKP binding must not increase assurance of the receipt.
        if self.bound_assurance.rank > self.receipt_assurance.rank:
            raise GoalDirectedTacticianError(
                "ZKP binding must not increase the bound receipt's assurance"
            )
        object.__setattr__(
            self,
            "assurance_increased",
            self.bound_assurance.rank > self.receipt_assurance.rank,
        )
        object.__setattr__(
            self,
            "reason_code",
            str(self.reason_code or "bound_existing_trusted_receipt").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_ZKP_BINDING_SCHEMA,
            "receipt_id": self.receipt_id,
            "receipt_assurance": self.receipt_assurance.value,
            "bound_assurance": self.bound_assurance.value,
            "circuit_id": self.circuit_id,
            "backend_id": self.backend_id,
            "verification_key_id": self.verification_key_id,
            "attestation_artifact_id": self.attestation_artifact_id,
            "statement_id": self.statement_id,
            "assurance_increased": self.assurance_increased,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZkpReceiptBinding":
        value = _mapping(payload, field_name="zkp_binding")
        return cls(
            receipt_id=value.get("receipt_id", ""),
            receipt_assurance=value.get("receipt_assurance", AssuranceLevel.UNVERIFIED),
            bound_assurance=value.get("bound_assurance", AssuranceLevel.UNVERIFIED),
            circuit_id=value.get("circuit_id", ""),
            backend_id=value.get("backend_id", ""),
            verification_key_id=value.get("verification_key_id", ""),
            attestation_artifact_id=value.get("attestation_artifact_id", ""),
            statement_id=value.get("statement_id", ""),
            reason_code=value.get("reason_code", "bound_existing_trusted_receipt"),
        )


def bind_zkp_to_trusted_receipt(
    *,
    receipt_id: str,
    receipt_assurance: AssuranceLevel | str,
    circuit_id: str,
    backend_id: str,
    verification_key_id: str,
    attestation_artifact_id: str = "",
    statement_id: str = "",
) -> ZkpReceiptBinding:
    """Bind ZKP attestation to a trusted receipt without raising its assurance."""

    assurance = _assurance(receipt_assurance)
    if assurance.rank < AssuranceLevel.KERNEL_VERIFIED.rank:
        raise GoalDirectedTacticianError(
            "ZKP binding requires an existing trusted (kernel-verified) receipt"
        )
    return ZkpReceiptBinding(
        receipt_id=receipt_id,
        receipt_assurance=assurance,
        # Binding preserves the receipt's assurance projection.
        bound_assurance=assurance,
        circuit_id=circuit_id,
        backend_id=backend_id,
        verification_key_id=verification_key_id,
        attestation_artifact_id=attestation_artifact_id or f"attestation:{receipt_id}",
        statement_id=statement_id
        or f"statement:sha256:{_sha256_hex({'receipt_id': receipt_id, 'circuit_id': circuit_id})}",
        reason_code="bound_existing_trusted_receipt",
    )


@dataclass(frozen=True)
class AdmissionRecord:
    """Fail-closed supervisor admission decision for one tactician run."""

    SCHEMA: ClassVar[str] = TACTICIAN_ADMISSION_SCHEMA

    decision: AdmissionDecision
    required_assurance: AssuranceLevel
    authoritative_assurance: AssuranceLevel
    independently_validated: bool
    legal_compatible: bool
    reason_codes: tuple[str, ...] = ()
    cache_key_id: str = ""
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision",
            _enum(self.decision, AdmissionDecision, "decision"),
        )
        object.__setattr__(
            self,
            "required_assurance",
            _assurance(self.required_assurance),
        )
        object.__setattr__(
            self,
            "authoritative_assurance",
            _assurance(self.authoritative_assurance),
        )
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        object.__setattr__(self, "legal_compatible", bool(self.legal_compatible))
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        object.__setattr__(self, "cache_key_id", str(self.cache_key_id or "").strip())
        object.__setattr__(self, "receipt_id", str(self.receipt_id or "").strip())

    @property
    def admitted(self) -> bool:
        return self.decision is AdmissionDecision.ADMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_ADMISSION_SCHEMA,
            "decision": self.decision.value,
            "admitted": self.admitted,
            "required_assurance": self.required_assurance.value,
            "authoritative_assurance": self.authoritative_assurance.value,
            "independently_validated": self.independently_validated,
            "legal_compatible": self.legal_compatible,
            "reason_codes": list(self.reason_codes),
            "cache_key_id": self.cache_key_id,
            "receipt_id": self.receipt_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdmissionRecord":
        value = _mapping(payload, field_name="admission")
        return cls(
            decision=value.get("decision", AdmissionDecision.REJECTED),
            required_assurance=value.get("required_assurance", AssuranceLevel.KERNEL_VERIFIED),
            authoritative_assurance=value.get("authoritative_assurance", AssuranceLevel.UNVERIFIED),
            independently_validated=bool(value.get("independently_validated", False)),
            legal_compatible=bool(value.get("legal_compatible", True)),
            reason_codes=tuple(value.get("reason_codes") or ()),
            cache_key_id=value.get("cache_key_id", ""),
            receipt_id=value.get("receipt_id", ""),
        )


@dataclass(frozen=True)
class TacticianCheckpoint:
    """Durable, resumable proof-carrying checkpoint for the tactician."""

    SCHEMA: ClassVar[str] = TACTICIAN_CHECKPOINT_SCHEMA

    checkpoint_id: str
    cache_key: ExactTacticianCacheKey
    completed_phases: tuple[str, ...]
    workflow_id: str = ""
    receipt_id: str = ""
    authoritative_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    independently_validated: bool = False
    legal_compatible: bool = True
    phase_records: tuple[PhaseRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_id",
            _text(self.checkpoint_id, field_name="checkpoint_id"),
        )
        if not isinstance(self.cache_key, ExactTacticianCacheKey):
            object.__setattr__(
                self,
                "cache_key",
                ExactTacticianCacheKey.from_dict(_mapping(self.cache_key, field_name="cache_key")),
            )
        object.__setattr__(self, "completed_phases", _strings(self.completed_phases))
        object.__setattr__(self, "workflow_id", str(self.workflow_id or "").strip())
        object.__setattr__(self, "receipt_id", str(self.receipt_id or "").strip())
        object.__setattr__(
            self,
            "authoritative_assurance",
            _assurance(self.authoritative_assurance),
        )
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        object.__setattr__(self, "legal_compatible", bool(self.legal_compatible))
        records: list[PhaseRecord] = []
        for item in self.phase_records or ():
            if isinstance(item, PhaseRecord):
                records.append(item)
            else:
                records.append(PhaseRecord.from_dict(_mapping(item, field_name="phase_record")))
        object.__setattr__(self, "phase_records", tuple(records))
        object.__setattr__(self, "metadata", _public_mapping(dict(self.metadata or {})))

    @property
    def resumable(self) -> bool:
        return bool(self.checkpoint_id and self.cache_key.key_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_CHECKPOINT_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "cache_key": self.cache_key.to_dict(),
            "completed_phases": list(self.completed_phases),
            "workflow_id": self.workflow_id,
            "receipt_id": self.receipt_id,
            "authoritative_assurance": self.authoritative_assurance.value,
            "independently_validated": self.independently_validated,
            "legal_compatible": self.legal_compatible,
            "phase_records": [item.to_dict() for item in self.phase_records],
            "metadata": dict(self.metadata),
            "resumable": self.resumable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TacticianCheckpoint":
        value = _mapping(payload, field_name="checkpoint")
        if value.get("schema") not in {None, TACTICIAN_CHECKPOINT_SCHEMA}:
            raise GoalDirectedTacticianError("unsupported tactician checkpoint schema")
        return cls(
            checkpoint_id=value.get("checkpoint_id", ""),
            cache_key=ExactTacticianCacheKey.from_dict(value.get("cache_key") or {}),
            completed_phases=tuple(value.get("completed_phases") or ()),
            workflow_id=value.get("workflow_id", ""),
            receipt_id=value.get("receipt_id", ""),
            authoritative_assurance=value.get("authoritative_assurance", AssuranceLevel.UNVERIFIED),
            independently_validated=bool(value.get("independently_validated", False)),
            legal_compatible=bool(value.get("legal_compatible", True)),
            phase_records=tuple(value.get("phase_records") or ()),
            metadata=dict(value.get("metadata") or {}),
        )

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, separators=(",", ": "))
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "TacticianCheckpoint":
        target = Path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GoalDirectedTacticianError(f"checkpoint is unreadable: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise GoalDirectedTacticianError("checkpoint must be a JSON object")
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# Request / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalDirectedTacticianRequest:
    """Public request envelope for one goal-directed tactician run."""

    SCHEMA: ClassVar[str] = TACTICIAN_REQUEST_SCHEMA

    tree_id: str
    target_id: str
    provider_id: str
    provider_version: str
    policy_id: str
    bounds: Mapping[str, Any]
    assumption_ids: tuple[str, ...] = ()
    formal_goal_id: str = ""
    obligation_id: str = ""
    corpus_id: str = ""
    corpus_version: str = ""
    toolchain_id: str = ""
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    require_legal_compatibility: bool = False
    enable_zkp: bool = True
    model_draft: Mapping[str, Any] | None = None
    workflow_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _text(self.tree_id, field_name="tree_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, field_name="target_id"))
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, field_name="provider_id"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(self.provider_version, field_name="provider_version"),
        )
        object.__setattr__(self, "policy_id", _text(self.policy_id, field_name="policy_id"))
        if not isinstance(self.bounds, Mapping) or not dict(self.bounds):
            raise GoalDirectedTacticianError("bounds must be a non-empty object")
        object.__setattr__(self, "bounds", dict(self.bounds))
        object.__setattr__(self, "assumption_ids", _strings(self.assumption_ids))
        object.__setattr__(self, "formal_goal_id", str(self.formal_goal_id or "").strip())
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "").strip())
        object.__setattr__(self, "corpus_id", str(self.corpus_id or "").strip())
        object.__setattr__(self, "corpus_version", str(self.corpus_version or "").strip())
        object.__setattr__(self, "toolchain_id", str(self.toolchain_id or "").strip())
        object.__setattr__(
            self,
            "required_assurance",
            _assurance(self.required_assurance),
        )
        object.__setattr__(
            self,
            "require_legal_compatibility",
            bool(self.require_legal_compatibility),
        )
        object.__setattr__(self, "enable_zkp", bool(self.enable_zkp))
        if self.model_draft is not None:
            object.__setattr__(
                self,
                "model_draft",
                _public_mapping(dict(_mapping(self.model_draft, field_name="model_draft"))),
            )
        object.__setattr__(self, "workflow_id", str(self.workflow_id or "").strip())
        object.__setattr__(self, "metadata", _public_mapping(dict(self.metadata or {})))

    def cache_key(self) -> ExactTacticianCacheKey:
        return build_exact_tactician_cache_key(
            tree_id=self.tree_id,
            target_id=self.target_id,
            assumption_ids=self.assumption_ids,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            policy_id=self.policy_id,
            bounds=self.bounds,
            corpus_id=self.corpus_id,
            corpus_version=self.corpus_version,
            toolchain_id=self.toolchain_id,
            obligation_id=self.obligation_id or self.target_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_REQUEST_SCHEMA,
            "tree_id": self.tree_id,
            "target_id": self.target_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "policy_id": self.policy_id,
            "bounds": dict(self.bounds),
            "assumption_ids": list(self.assumption_ids),
            "formal_goal_id": self.formal_goal_id,
            "obligation_id": self.obligation_id,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "toolchain_id": self.toolchain_id,
            "required_assurance": self.required_assurance.value,
            "require_legal_compatibility": self.require_legal_compatibility,
            "enable_zkp": self.enable_zkp,
            "model_draft": dict(self.model_draft or {}),
            "workflow_id": self.workflow_id,
            "metadata": dict(self.metadata),
            "cache_key": self.cache_key().to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDirectedTacticianRequest":
        value = _mapping(payload, field_name="request")
        if value.get("schema") not in {None, TACTICIAN_REQUEST_SCHEMA}:
            raise GoalDirectedTacticianError("unsupported goal-directed tactician request schema")
        draft = value.get("model_draft")
        return cls(
            tree_id=value.get("tree_id", ""),
            target_id=value.get("target_id", "")
            or value.get("goal_id", "")
            or value.get("obligation_id", ""),
            provider_id=value.get("provider_id", ""),
            provider_version=value.get("provider_version", "") or value.get("version", ""),
            policy_id=value.get("policy_id", ""),
            bounds=dict(value.get("bounds") or {}),
            assumption_ids=tuple(value.get("assumption_ids") or ()),
            formal_goal_id=value.get("formal_goal_id", ""),
            obligation_id=value.get("obligation_id", ""),
            corpus_id=value.get("corpus_id", ""),
            corpus_version=value.get("corpus_version", ""),
            toolchain_id=value.get("toolchain_id", ""),
            required_assurance=value.get("required_assurance", AssuranceLevel.KERNEL_VERIFIED),
            require_legal_compatibility=bool(value.get("require_legal_compatibility", False)),
            enable_zkp=bool(value.get("enable_zkp", True)),
            model_draft=dict(draft) if isinstance(draft, Mapping) else None,
            workflow_id=value.get("workflow_id", ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class GoalDirectedTacticianResult:
    """Terminal result of one restartable goal-directed tactician run."""

    SCHEMA: ClassVar[str] = TACTICIAN_RESULT_SCHEMA

    stop_reason: TacticianStopReason
    cache_key: ExactTacticianCacheKey
    phases: tuple[PhaseRecord, ...]
    utilities: tuple[UtilityBinding, ...]
    admission: AdmissionRecord
    independently_validated: bool
    legal_compatible: bool
    resumable: bool
    checkpoint: TacticianCheckpoint | None = None
    zkp_binding: ZkpReceiptBinding | None = None
    receipt_id: str = ""
    authoritative_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    workflow_id: str = ""
    reason_code: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    proof_state: ProofStateClassification | None = None
    ranking: PremiseRankingReport | Mapping[str, Any] | None = None
    decomposition: GoalDecomposition | None = None
    prediction: BranchCostFailurePrediction | None = None
    traces: tuple[TacticPremiseTrace, ...] = ()
    curriculum: CurriculumProjection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stop_reason",
            _enum(self.stop_reason, TacticianStopReason, "stop_reason"),
        )
        if not isinstance(self.cache_key, ExactTacticianCacheKey):
            object.__setattr__(
                self,
                "cache_key",
                ExactTacticianCacheKey.from_dict(_mapping(self.cache_key, field_name="cache_key")),
            )
        object.__setattr__(self, "phases", tuple(self.phases or ()))
        object.__setattr__(self, "utilities", tuple(self.utilities or ()))
        if not isinstance(self.admission, AdmissionRecord):
            object.__setattr__(
                self,
                "admission",
                AdmissionRecord.from_dict(_mapping(self.admission, field_name="admission")),
            )
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        object.__setattr__(self, "legal_compatible", bool(self.legal_compatible))
        object.__setattr__(self, "resumable", bool(self.resumable))
        if self.checkpoint is not None and not isinstance(self.checkpoint, TacticianCheckpoint):
            object.__setattr__(
                self,
                "checkpoint",
                TacticianCheckpoint.from_dict(_mapping(self.checkpoint, field_name="checkpoint")),
            )
        if self.zkp_binding is not None and not isinstance(self.zkp_binding, ZkpReceiptBinding):
            object.__setattr__(
                self,
                "zkp_binding",
                ZkpReceiptBinding.from_dict(_mapping(self.zkp_binding, field_name="zkp_binding")),
            )
        object.__setattr__(self, "receipt_id", str(self.receipt_id or "").strip())
        object.__setattr__(
            self,
            "authoritative_assurance",
            _assurance(self.authoritative_assurance),
        )
        object.__setattr__(self, "workflow_id", str(self.workflow_id or "").strip())
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(self, "details", _public_mapping(dict(self.details or {})))
        if self.proof_state is not None and not isinstance(
            self.proof_state, ProofStateClassification
        ):
            object.__setattr__(
                self,
                "proof_state",
                ProofStateClassification.from_dict(
                    _mapping(self.proof_state, field_name="proof_state")
                ),
            )
        if self.decomposition is not None and not isinstance(self.decomposition, GoalDecomposition):
            object.__setattr__(
                self,
                "decomposition",
                GoalDecomposition.from_dict(
                    _mapping(self.decomposition, field_name="decomposition")
                ),
            )
        if self.prediction is not None and not isinstance(
            self.prediction, BranchCostFailurePrediction
        ):
            object.__setattr__(
                self,
                "prediction",
                BranchCostFailurePrediction.from_dict(
                    _mapping(self.prediction, field_name="prediction")
                ),
            )
        traces: list[TacticPremiseTrace] = []
        for item in self.traces or ():
            if isinstance(item, TacticPremiseTrace):
                traces.append(item)
            else:
                traces.append(TacticPremiseTrace.from_dict(_mapping(item, field_name="trace")))
        object.__setattr__(self, "traces", tuple(traces))
        if self.curriculum is not None and not isinstance(self.curriculum, CurriculumProjection):
            object.__setattr__(
                self,
                "curriculum",
                CurriculumProjection.from_dict(_mapping(self.curriculum, field_name="curriculum")),
            )

    @property
    def admitted(self) -> bool:
        return self.admission.admitted

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TACTICIAN_RESULT_SCHEMA,
            "interface": GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE,
            "version": GOAL_DIRECTED_PROOF_TACTICIAN_VERSION,
            "stop_reason": self.stop_reason.value,
            "cache_key": self.cache_key.to_dict(),
            "phases": [item.to_dict() for item in self.phases],
            "utilities": [item.to_dict() for item in self.utilities],
            "admission": self.admission.to_dict(),
            "independently_validated": self.independently_validated,
            "legal_compatible": self.legal_compatible,
            "resumable": self.resumable,
            "checkpoint": (self.checkpoint.to_dict() if self.checkpoint is not None else None),
            "zkp_binding": (self.zkp_binding.to_dict() if self.zkp_binding is not None else None),
            "receipt_id": self.receipt_id,
            "authoritative_assurance": self.authoritative_assurance.value,
            "workflow_id": self.workflow_id,
            "reason_code": self.reason_code,
            "details": dict(self.details),
            "admitted": self.admitted,
            "proof_state": (self.proof_state.to_dict() if self.proof_state is not None else None),
            "ranking": (
                self.ranking.to_dict()
                if self.ranking is not None and hasattr(self.ranking, "to_dict")
                else (dict(self.ranking) if isinstance(self.ranking, Mapping) else None)
            ),
            "decomposition": (
                self.decomposition.to_dict() if self.decomposition is not None else None
            ),
            "prediction": (self.prediction.to_dict() if self.prediction is not None else None),
            "traces": [item.to_dict() for item in self.traces],
            "curriculum": (self.curriculum.to_dict() if self.curriculum is not None else None),
        }
        if include_identity:
            payload["content_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "content_id"}
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDirectedTacticianResult":
        value = _mapping(payload, field_name="result")
        if value.get("schema") not in {None, TACTICIAN_RESULT_SCHEMA}:
            raise GoalDirectedTacticianError("unsupported goal-directed tactician result schema")
        checkpoint = value.get("checkpoint")
        zkp = value.get("zkp_binding")
        return cls(
            stop_reason=value.get("stop_reason", TacticianStopReason.OPEN),
            cache_key=ExactTacticianCacheKey.from_dict(value.get("cache_key") or {}),
            phases=tuple(PhaseRecord.from_dict(item) for item in (value.get("phases") or ())),
            utilities=tuple(
                UtilityBinding.from_dict(item) for item in (value.get("utilities") or ())
            ),
            admission=AdmissionRecord.from_dict(value.get("admission") or {}),
            independently_validated=bool(value.get("independently_validated", False)),
            legal_compatible=bool(value.get("legal_compatible", True)),
            resumable=bool(value.get("resumable", False)),
            checkpoint=(
                TacticianCheckpoint.from_dict(checkpoint)
                if isinstance(checkpoint, Mapping)
                else None
            ),
            zkp_binding=(ZkpReceiptBinding.from_dict(zkp) if isinstance(zkp, Mapping) else None),
            receipt_id=value.get("receipt_id", ""),
            authoritative_assurance=value.get("authoritative_assurance", AssuranceLevel.UNVERIFIED),
            workflow_id=value.get("workflow_id", ""),
            reason_code=value.get("reason_code", ""),
            details=dict(value.get("details") or {}),
            proof_state=value.get("proof_state"),
            ranking=value.get("ranking"),
            decomposition=value.get("decomposition"),
            prediction=value.get("prediction"),
            traces=tuple(value.get("traces") or ()),
            curriculum=value.get("curriculum"),
        )


# ---------------------------------------------------------------------------
# Provider callables
# ---------------------------------------------------------------------------

FormalizeProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
RetrieveProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ScheduleProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PlanProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
HammerProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
KernelProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
LeanstralProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
GuidanceProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
LegalProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ValidateProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ProveProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
CacheLookupProvider = Callable[[ExactTacticianCacheKey], Mapping[str, Any] | None]
CacheStoreProvider = Callable[[ExactTacticianCacheKey, Mapping[str, Any]], Mapping[str, Any] | None]
ZkpProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _default_formalize(context: Mapping[str, Any]) -> Mapping[str, Any]:
    goal_id = str(context.get("formal_goal_id") or "").strip()
    if not goal_id:
        return {
            "status": "rejected",
            "reason_code": "formalization_required",
            "formalized": False,
        }
    return {
        "status": "ok",
        "reason_code": "confirmed_formal_goal",
        "formalized": True,
        "formal_goal_id": goal_id,
    }


def _default_retrieve(context: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": "ok",
        "reason_code": "retrieval_skipped_no_index",
        "candidates": [],
        "target_id": context.get("target_id"),
    }


def _default_schedule(context: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": "ok",
        "reason_code": "schedule_materialized",
        "steps": [
            {
                "step_id": "prove",
                "target_id": context.get("target_id"),
            }
        ],
    }


def _default_plan(context: Mapping[str, Any]) -> Mapping[str, Any]:
    workflow_id = str(context.get("workflow_id") or "").strip()
    if not workflow_id:
        workflow_id = f"workflow:sha256:{_sha256_hex({'target': context.get('target_id')})}"
    return {
        "status": "ok",
        "reason_code": "proof_carrying_plan_ready",
        "workflow_id": workflow_id,
        "resumable": True,
    }


def _default_hammer(context: Mapping[str, Any]) -> Mapping[str, Any]:
    del context
    return {
        "status": "ok",
        "reason_code": "hammer_structural_gate",
        "authority": UtilityAuthority.GUIDANCE.value,
        "structural_ok": True,
    }


def _default_kernel(context: Mapping[str, Any]) -> Mapping[str, Any]:
    # Default kernel is not available; independent validation must supply authority.
    del context
    return {
        "status": "skipped",
        "reason_code": "kernel_not_configured",
        "assurance": AssuranceLevel.UNVERIFIED.value,
    }


def _default_leanstral(context: Mapping[str, Any]) -> Mapping[str, Any]:
    if not str(context.get("formal_goal_id") or "").strip():
        return {
            "status": "rejected",
            "reason_code": "leanstral_requires_formalization",
        }
    return {
        "status": "ok",
        "reason_code": "leanstral_guidance_only",
        "authority": UtilityAuthority.GUIDANCE.value,
    }


def _default_guidance(context: Mapping[str, Any]) -> Mapping[str, Any]:
    del context
    return {
        "status": "ok",
        "reason_code": "symai_autoencoder_guidance",
        "symai": {"authority": UtilityAuthority.GUIDANCE.value},
        "autoencoder": {"authority": UtilityAuthority.GUIDANCE.value},
    }


def _default_legal(context: Mapping[str, Any]) -> Mapping[str, Any]:
    if not context.get("require_legal_compatibility"):
        return {
            "status": "skipped",
            "reason_code": "legal_not_required",
            "legal_compatible": True,
        }
    # Without an injectable legal artifact, remain honest about unknown.
    return {
        "status": "ok",
        "reason_code": "legal_compatibility_assumed_absent_constraints",
        "legal_compatible": True,
    }


def _default_validate(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Independent validation gate.

    Model drafts and cache hits that claim authority are rejected unless a
    prior independent validation already succeeded. Authority only comes from
    kernel / independent validation evidence, never from guidance utilities.
    """

    draft = context.get("model_draft")
    if isinstance(draft, Mapping) and draft:
        try:
            reject_authority_bypass(
                draft,
                source=EvidenceSource.MODEL_DRAFT,
                independently_validated=False,
            )
        except GoalDirectedTacticianError as exc:
            return {
                "status": "rejected",
                "reason_code": "model_bypass_rejected",
                "independently_validated": False,
                "message": str(exc),
            }

    cache_payload = context.get("cache_payload")
    if isinstance(cache_payload, Mapping) and cache_payload:
        if cache_payload.get("independently_validated") is True:
            assurance = _assurance(
                cache_payload.get("authoritative_assurance", AssuranceLevel.UNVERIFIED)
            )
            return {
                "status": "ok",
                "reason_code": "validated_cache_hit",
                "independently_validated": True,
                "authoritative_assurance": assurance.value,
                "receipt_id": str(cache_payload.get("receipt_id") or ""),
                "evidence_source": EvidenceSource.CACHE_HIT.value,
            }
        try:
            reject_authority_bypass(
                cache_payload,
                source=EvidenceSource.CACHE_HIT,
                independently_validated=False,
            )
        except GoalDirectedTacticianError as exc:
            return {
                "status": "rejected",
                "reason_code": "cache_bypass_rejected",
                "independently_validated": False,
                "message": str(exc),
            }

    kernel = context.get("kernel_result") or {}
    if isinstance(kernel, Mapping):
        k_assurance = _assurance(kernel.get("assurance", AssuranceLevel.UNVERIFIED))
        if (
            str(kernel.get("status") or "") == "ok"
            and k_assurance.rank >= AssuranceLevel.KERNEL_VERIFIED.rank
        ):
            return {
                "status": "ok",
                "reason_code": "kernel_validated",
                "independently_validated": True,
                "authoritative_assurance": k_assurance.value,
                "receipt_id": str(
                    kernel.get("receipt_id")
                    or context.get("receipt_id")
                    or f"receipt:{context.get('target_id')}"
                ),
                "evidence_source": EvidenceSource.KERNEL.value,
            }

    prove = context.get("prove_result") or {}
    if isinstance(prove, Mapping) and str(prove.get("status") or "") == "ok":
        p_assurance = _assurance(prove.get("assurance", AssuranceLevel.UNVERIFIED))
        if prove.get("independently_validated") is True and p_assurance.rank > 0:
            return {
                "status": "ok",
                "reason_code": "independent_prove_validated",
                "independently_validated": True,
                "authoritative_assurance": p_assurance.value,
                "receipt_id": str(prove.get("receipt_id") or ""),
                "evidence_source": EvidenceSource.INDEPENDENT_VALIDATION.value,
            }

    return {
        "status": "rejected",
        "reason_code": "independent_validation_required",
        "independently_validated": False,
        "authoritative_assurance": AssuranceLevel.UNVERIFIED.value,
    }


def _default_prove(context: Mapping[str, Any]) -> Mapping[str, Any]:
    # Honest default: no live prover — leave open for injectable providers.
    del context
    return {
        "status": "skipped",
        "reason_code": "prove_provider_not_configured",
        "independently_validated": False,
        "assurance": AssuranceLevel.UNVERIFIED.value,
    }


def _default_zkp(context: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt_id = str(context.get("receipt_id") or "").strip()
    assurance = _assurance(context.get("authoritative_assurance", AssuranceLevel.UNVERIFIED))
    if not receipt_id:
        return {
            "status": "skipped",
            "reason_code": "no_receipt_to_bind",
        }
    if assurance.rank < AssuranceLevel.KERNEL_VERIFIED.rank:
        return {
            "status": "rejected",
            "reason_code": "receipt_not_trusted_for_zkp",
            "authoritative_assurance": assurance.value,
        }
    binding = bind_zkp_to_trusted_receipt(
        receipt_id=receipt_id,
        receipt_assurance=assurance,
        circuit_id=str(context.get("circuit_id") or "circuit:receipt-binding@1"),
        backend_id=str(context.get("backend_id") or "backend:simulated-public"),
        verification_key_id=str(context.get("verification_key_id") or "vk:receipt-binding@1"),
    )
    return {
        "status": "ok",
        "reason_code": "zkp_bound_without_assurance_increase",
        "binding": binding.to_dict(),
        "assurance_increased": False,
        "bound_assurance": binding.bound_assurance.value,
        "receipt_assurance": binding.receipt_assurance.value,
    }


# ---------------------------------------------------------------------------
# Tactician
# ---------------------------------------------------------------------------


class GoalDirectedProofTactician:
    """Restartable orchestration facade for goal-directed proof development.

    Inject providers for real formalization, retrieval, scheduling, planning,
    Hammer/kernel, Leanstral, SymAI/autoencoder guidance, legal compilation,
    validation, proving, cache, and ZKP. Defaults are fail-closed and never
    claim kernel or admission authority from guidance utilities.
    """

    interface: ClassVar[str] = GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE
    version: ClassVar[str] = GOAL_DIRECTED_PROOF_TACTICIAN_VERSION
    schema: ClassVar[str] = GOAL_DIRECTED_PROOF_TACTICIAN_SCHEMA

    def __init__(
        self,
        *,
        formalize: FormalizeProvider | None = None,
        retrieve: RetrieveProvider | None = None,
        schedule: ScheduleProvider | None = None,
        plan: PlanProvider | None = None,
        hammer: HammerProvider | None = None,
        kernel: KernelProvider | None = None,
        leanstral: LeanstralProvider | None = None,
        guidance: GuidanceProvider | None = None,
        legal: LegalProvider | None = None,
        validate: ValidateProvider | None = None,
        prove: ProveProvider | None = None,
        cache_lookup: CacheLookupProvider | None = None,
        cache_store: CacheStoreProvider | None = None,
        zkp: ZkpProvider | None = None,
        cache: FormalVerificationCache | None = None,
        utilities: Sequence[UtilityBinding] | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        self.formalize = formalize or _default_formalize
        self.retrieve = retrieve or _default_retrieve
        self.schedule = schedule or _default_schedule
        self.plan = plan or _default_plan
        self.hammer = hammer or _default_hammer
        self.kernel = kernel or _default_kernel
        self.leanstral = leanstral or _default_leanstral
        self.guidance = guidance or _default_guidance
        self.legal = legal or _default_legal
        self.validate = validate or _default_validate
        self.prove = prove or _default_prove
        self.zkp = zkp or _default_zkp
        self._cache = cache
        self.cache_lookup = cache_lookup
        self.cache_store = cache_store
        self.utilities = tuple(utilities or default_utility_bindings())
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    def run(
        self,
        request: GoalDirectedTacticianRequest | Mapping[str, Any],
        *,
        checkpoint: TacticianCheckpoint | Mapping[str, Any] | None = None,
        checkpoint_path: Path | str | None = None,
        cancelled: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> GoalDirectedTacticianResult:
        """Execute one restartable goal-directed tactician orchestration."""

        req = self._request(request)
        cache_key = req.cache_key()
        phases: list[PhaseRecord] = []
        extra = dict(context or {})
        workflow_id = req.workflow_id
        receipt_id = ""
        assurance = AssuranceLevel.UNVERIFIED
        independently_validated = False
        legal_compatible = True
        zkp_binding: ZkpReceiptBinding | None = None
        completed: list[str] = []
        cache_payload: Mapping[str, Any] | None = None
        kernel_result: Mapping[str, Any] = {}
        prove_result: Mapping[str, Any] = {}

        def record(
            phase: TacticianPhase,
            status: PhaseStatus,
            *,
            utility_role: UtilityRole | None = None,
            reason_code: str = "",
            evidence_source: EvidenceSource | None = None,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            phases.append(
                PhaseRecord(
                    phase=phase,
                    status=status,
                    utility_role=utility_role,
                    reason_code=reason_code,
                    evidence_source=evidence_source,
                    details=dict(details or {}),
                )
            )
            if status in {PhaseStatus.OK, PhaseStatus.SKIPPED, PhaseStatus.RESUMED}:
                completed.append(phase.value)

        if _cancelled(cancelled):
            return self._terminal(
                stop_reason=TacticianStopReason.CANCELLED,
                cache_key=cache_key,
                phases=phases,
                independently_validated=False,
                legal_compatible=True,
                resumable=False,
                required_assurance=req.required_assurance,
                authoritative_assurance=AssuranceLevel.UNVERIFIED,
                reason_code="cancelled_before_start",
            )

        record(
            TacticianPhase.ADMIT_REQUEST,
            PhaseStatus.OK,
            reason_code="request_accepted",
            details={"target_id": req.target_id, "tree_id": req.tree_id},
        )
        record(
            TacticianPhase.BUILD_CACHE_KEY,
            PhaseStatus.OK,
            utility_role=UtilityRole.CACHE,
            reason_code="exact_cache_key",
            details=cache_key.to_dict(),
        )

        # Resume path: load checkpoint and refuse identity mismatch.
        loaded = self._load_checkpoint(checkpoint, checkpoint_path, cache_key)
        if loaded is not None:
            if not loaded.cache_key.matches(cache_key):
                record(
                    TacticianPhase.LOAD_CHECKPOINT,
                    PhaseStatus.REJECTED,
                    reason_code="checkpoint_cache_key_mismatch",
                    details={
                        "checkpoint_key_id": loaded.cache_key.key_id,
                        "request_key_id": cache_key.key_id,
                    },
                )
                return self._terminal(
                    stop_reason=TacticianStopReason.CHECKPOINT_MISMATCH,
                    cache_key=cache_key,
                    phases=phases,
                    independently_validated=False,
                    legal_compatible=loaded.legal_compatible,
                    resumable=False,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=AssuranceLevel.UNVERIFIED,
                    reason_code="checkpoint_cache_key_mismatch",
                )
            record(
                TacticianPhase.LOAD_CHECKPOINT,
                PhaseStatus.RESUMED,
                utility_role=UtilityRole.PROOF_CARRYING_PLANNER,
                reason_code="checkpoint_resumed",
                evidence_source=EvidenceSource.CHECKPOINT,
                details={
                    "checkpoint_id": loaded.checkpoint_id,
                    "completed_phases": list(loaded.completed_phases),
                },
            )
            completed = list(loaded.completed_phases)
            workflow_id = loaded.workflow_id or workflow_id
            receipt_id = loaded.receipt_id or receipt_id
            assurance = loaded.authoritative_assurance
            independently_validated = loaded.independently_validated
            legal_compatible = loaded.legal_compatible
            for prior in loaded.phase_records:
                if prior.phase.value not in {p.phase.value for p in phases}:
                    phases.append(prior)

        base_context: dict[str, Any] = {
            "tree_id": req.tree_id,
            "target_id": req.target_id,
            "assumption_ids": list(req.assumption_ids),
            "provider_id": req.provider_id,
            "provider_version": req.provider_version,
            "policy_id": req.policy_id,
            "bounds": dict(req.bounds),
            "formal_goal_id": req.formal_goal_id,
            "obligation_id": req.obligation_id or req.target_id,
            "corpus_id": req.corpus_id,
            "corpus_version": req.corpus_version,
            "toolchain_id": req.toolchain_id,
            "required_assurance": req.required_assurance.value,
            "require_legal_compatibility": req.require_legal_compatibility,
            "enable_zkp": req.enable_zkp,
            "model_draft": dict(req.model_draft or {}),
            "workflow_id": workflow_id,
            "cache_key": cache_key.to_dict(),
            **extra,
        }

        # Formalization gate (prose cannot bypass).
        if TacticianPhase.FORMALIZE.value not in completed:
            if _cancelled(cancelled):
                return self._cancel(cache_key, phases, req, legal_compatible)
            formal = self._call(self.formalize, base_context, "formalize")
            if str(formal.get("status")) == "rejected" or not formal.get(
                "formalized", formal.get("status") == "ok"
            ):
                record(
                    TacticianPhase.FORMALIZE,
                    PhaseStatus.REJECTED,
                    utility_role=UtilityRole.FORMALIZATION,
                    reason_code=str(formal.get("reason_code") or "formalization_required"),
                )
                return self._terminal(
                    stop_reason=TacticianStopReason.FORMALIZATION_REQUIRED,
                    cache_key=cache_key,
                    phases=phases,
                    independently_validated=False,
                    legal_compatible=legal_compatible,
                    resumable=True,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=assurance,
                    reason_code=str(formal.get("reason_code") or "formalization_required"),
                    workflow_id=workflow_id,
                )
            record(
                TacticianPhase.FORMALIZE,
                PhaseStatus.OK,
                utility_role=UtilityRole.FORMALIZATION,
                reason_code=str(formal.get("reason_code") or "formalized"),
                details={"formal_goal_id": formal.get("formal_goal_id")},
            )
            base_context["formalize_result"] = dict(formal)

        # Retrieval / schedule / plan (orchestration + guidance).
        for phase, role, provider, key in (
            (
                TacticianPhase.RETRIEVE,
                UtilityRole.RETRIEVAL,
                self.retrieve,
                "retrieve_result",
            ),
            (
                TacticianPhase.SCHEDULE,
                UtilityRole.PROOF_SCHEDULER,
                self.schedule,
                "schedule_result",
            ),
            (
                TacticianPhase.PLAN,
                UtilityRole.PROOF_CARRYING_PLANNER,
                self.plan,
                "plan_result",
            ),
        ):
            if phase.value in completed:
                continue
            if _cancelled(cancelled):
                return self._cancel(cache_key, phases, req, legal_compatible)
            outcome = self._call(provider, base_context, phase.value)
            status = self._status_from_payload(outcome)
            record(
                phase,
                status,
                utility_role=role,
                reason_code=str(outcome.get("reason_code") or phase.value),
                details=_public_mapping(dict(outcome)),
            )
            base_context[key] = dict(outcome)
            if phase is TacticianPhase.PLAN:
                workflow_id = str(outcome.get("workflow_id") or workflow_id or "").strip()
                base_context["workflow_id"] = workflow_id
            if status is PhaseStatus.FAILED:
                return self._terminal(
                    stop_reason=TacticianStopReason.PROVIDER_UNAVAILABLE,
                    cache_key=cache_key,
                    phases=phases,
                    independently_validated=False,
                    legal_compatible=legal_compatible,
                    resumable=True,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=assurance,
                    reason_code=str(outcome.get("reason_code") or "provider_failed"),
                    workflow_id=workflow_id,
                )

        # Hammer structural gate (never authority).
        if TacticianPhase.HAMMER.value not in completed:
            hammer = self._call(self.hammer, base_context, "hammer")
            record(
                TacticianPhase.HAMMER,
                self._status_from_payload(hammer),
                utility_role=UtilityRole.HAMMER,
                reason_code=str(hammer.get("reason_code") or "hammer"),
                details=_public_mapping(dict(hammer)),
            )
            base_context["hammer_result"] = dict(hammer)

        # Kernel authority path when configured.
        if TacticianPhase.KERNEL.value not in completed:
            kernel_result = dict(self._call(self.kernel, base_context, "kernel"))
            record(
                TacticianPhase.KERNEL,
                self._status_from_payload(kernel_result),
                utility_role=UtilityRole.KERNEL,
                reason_code=str(kernel_result.get("reason_code") or "kernel"),
                evidence_source=EvidenceSource.KERNEL,
                details=_public_mapping(dict(kernel_result)),
            )
            base_context["kernel_result"] = dict(kernel_result)

        # Leanstral + SymAI/autoencoder guidance (never authority).
        if TacticianPhase.LEANSTRAL.value not in completed:
            lean = self._call(self.leanstral, base_context, "leanstral")
            if str(lean.get("status")) == "rejected":
                record(
                    TacticianPhase.LEANSTRAL,
                    PhaseStatus.REJECTED,
                    utility_role=UtilityRole.LEANSTRAL,
                    reason_code=str(lean.get("reason_code") or "leanstral_rejected"),
                )
            else:
                record(
                    TacticianPhase.LEANSTRAL,
                    self._status_from_payload(lean),
                    utility_role=UtilityRole.LEANSTRAL,
                    reason_code=str(lean.get("reason_code") or "leanstral"),
                    details=_public_mapping(dict(lean)),
                )
            base_context["leanstral_result"] = dict(lean)

        if TacticianPhase.GUIDANCE.value not in completed:
            guide = self._call(self.guidance, base_context, "guidance")
            record(
                TacticianPhase.GUIDANCE,
                self._status_from_payload(guide),
                utility_role=UtilityRole.SYMAI,
                reason_code=str(guide.get("reason_code") or "guidance"),
                evidence_source=EvidenceSource.GUIDANCE,
                details=_public_mapping(dict(guide)),
            )
            base_context["guidance_result"] = dict(guide)

        # Legal compatibility.
        if TacticianPhase.LEGAL.value not in completed:
            legal = self._call(self.legal, base_context, "legal")
            legal_compatible = bool(legal.get("legal_compatible", True))
            status = self._status_from_payload(legal)
            if req.require_legal_compatibility and not legal_compatible:
                record(
                    TacticianPhase.LEGAL,
                    PhaseStatus.REJECTED,
                    utility_role=UtilityRole.LEGAL_ADAPTER,
                    reason_code=str(legal.get("reason_code") or "legal_incompatible"),
                    evidence_source=EvidenceSource.LEGAL,
                    details=_public_mapping(dict(legal)),
                )
                return self._terminal(
                    stop_reason=TacticianStopReason.LEGAL_INCOMPATIBLE,
                    cache_key=cache_key,
                    phases=phases,
                    independently_validated=independently_validated,
                    legal_compatible=False,
                    resumable=True,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=assurance,
                    reason_code="legal_incompatible",
                    workflow_id=workflow_id,
                    receipt_id=receipt_id,
                )
            record(
                TacticianPhase.LEGAL,
                status,
                utility_role=UtilityRole.LEGAL_ADAPTER,
                reason_code=str(legal.get("reason_code") or "legal"),
                evidence_source=EvidenceSource.LEGAL,
                details=_public_mapping(dict(legal)),
            )
            base_context["legal_result"] = dict(legal)

        # Cache lookup (cannot bypass validation).
        if TacticianPhase.CACHE_LOOKUP.value not in completed:
            cache_payload = self._lookup_cache(cache_key)
            if cache_payload is not None:
                record(
                    TacticianPhase.CACHE_LOOKUP,
                    PhaseStatus.OK,
                    utility_role=UtilityRole.CACHE,
                    reason_code="cache_hit",
                    evidence_source=EvidenceSource.CACHE_HIT,
                    details={"key_id": cache_key.key_id},
                )
            else:
                record(
                    TacticianPhase.CACHE_LOOKUP,
                    PhaseStatus.SKIPPED,
                    utility_role=UtilityRole.CACHE,
                    reason_code="cache_miss",
                )
            base_context["cache_payload"] = dict(cache_payload) if cache_payload is not None else {}

        # Prove when no validated evidence yet.
        if TacticianPhase.PROVE.value not in completed:
            if _cancelled(cancelled):
                return self._cancel(cache_key, phases, req, legal_compatible)
            prove_result = dict(self._call(self.prove, base_context, "prove"))
            record(
                TacticianPhase.PROVE,
                self._status_from_payload(prove_result),
                reason_code=str(prove_result.get("reason_code") or "prove"),
                evidence_source=EvidenceSource.INDEPENDENT_VALIDATION,
                details=_public_mapping(dict(prove_result)),
            )
            base_context["prove_result"] = dict(prove_result)

        # Independent validation gate (model/cache cannot bypass).
        if TacticianPhase.VALIDATE.value not in completed:
            base_context["kernel_result"] = dict(kernel_result)
            base_context["prove_result"] = dict(prove_result)
            base_context["receipt_id"] = receipt_id
            validation = dict(self._call(self.validate, base_context, "validate"))
            status = self._status_from_payload(validation)
            reason = str(validation.get("reason_code") or "validation")
            if reason == "model_bypass_rejected" or (
                status is PhaseStatus.REJECTED and "model" in reason
            ):
                record(
                    TacticianPhase.VALIDATE,
                    PhaseStatus.REJECTED,
                    reason_code=reason,
                    evidence_source=EvidenceSource.MODEL_DRAFT,
                    details=_public_mapping(dict(validation)),
                )
                return self._terminal(
                    stop_reason=TacticianStopReason.MODEL_BYPASS_REJECTED,
                    cache_key=cache_key,
                    phases=phases,
                    independently_validated=False,
                    legal_compatible=legal_compatible,
                    resumable=True,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=AssuranceLevel.UNVERIFIED,
                    reason_code=reason,
                    workflow_id=workflow_id,
                )
            if reason == "cache_bypass_rejected" or (
                status is PhaseStatus.REJECTED and "cache" in reason
            ):
                record(
                    TacticianPhase.VALIDATE,
                    PhaseStatus.REJECTED,
                    reason_code=reason,
                    evidence_source=EvidenceSource.CACHE_HIT,
                    details=_public_mapping(dict(validation)),
                )
                return self._terminal(
                    stop_reason=TacticianStopReason.CACHE_BYPASS_REJECTED,
                    cache_key=cache_key,
                    phases=phases,
                    independently_validated=False,
                    legal_compatible=legal_compatible,
                    resumable=True,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=AssuranceLevel.UNVERIFIED,
                    reason_code=reason,
                    workflow_id=workflow_id,
                )
            if status is PhaseStatus.REJECTED or not validation.get("independently_validated"):
                record(
                    TacticianPhase.VALIDATE,
                    PhaseStatus.REJECTED,
                    reason_code=reason,
                    details=_public_mapping(dict(validation)),
                )
                checkpoint = self._write_checkpoint(
                    cache_key=cache_key,
                    completed_phases=completed + [TacticianPhase.VALIDATE.value],
                    phases=phases
                    + [
                        PhaseRecord(
                            phase=TacticianPhase.VALIDATE,
                            status=PhaseStatus.REJECTED,
                            reason_code=reason,
                        )
                    ],
                    workflow_id=workflow_id,
                    receipt_id=receipt_id,
                    assurance=assurance,
                    independently_validated=False,
                    legal_compatible=legal_compatible,
                    checkpoint_path=checkpoint_path,
                )
                return self._terminal(
                    stop_reason=TacticianStopReason.VALIDATION_FAILED,
                    cache_key=cache_key,
                    phases=phases
                    + [
                        PhaseRecord(
                            phase=TacticianPhase.VALIDATE,
                            status=PhaseStatus.REJECTED,
                            reason_code=reason,
                        )
                    ],
                    independently_validated=False,
                    legal_compatible=legal_compatible,
                    resumable=True,
                    required_assurance=req.required_assurance,
                    authoritative_assurance=assurance,
                    reason_code=reason,
                    workflow_id=workflow_id,
                    receipt_id=receipt_id,
                    checkpoint=checkpoint,
                )
            independently_validated = True
            assurance = _assurance(validation.get("authoritative_assurance", assurance))
            receipt_id = str(validation.get("receipt_id") or receipt_id)
            source = validation.get("evidence_source") or None
            record(
                TacticianPhase.VALIDATE,
                PhaseStatus.OK,
                reason_code=reason,
                evidence_source=source,
                details=_public_mapping(dict(validation)),
            )
            base_context["validate_result"] = dict(validation)
            base_context["receipt_id"] = receipt_id
            base_context["authoritative_assurance"] = assurance.value

        # Store validated evidence in cache (never unvalidated authority claims).
        if (
            TacticianPhase.CACHE_STORE.value not in completed
            and independently_validated
            and receipt_id
        ):
            store_payload = {
                "receipt_id": receipt_id,
                "authoritative_assurance": assurance.value,
                "independently_validated": True,
                "key_id": cache_key.key_id,
            }
            self._store_cache(cache_key, store_payload)
            record(
                TacticianPhase.CACHE_STORE,
                PhaseStatus.OK,
                utility_role=UtilityRole.CACHE,
                reason_code="validated_evidence_cached",
                details={"key_id": cache_key.key_id, "receipt_id": receipt_id},
            )

        # ZKP binds existing trusted receipt without increasing assurance.
        if TacticianPhase.ZKP_BIND.value not in completed:
            if req.enable_zkp and independently_validated and receipt_id:
                zkp_ctx = {
                    **base_context,
                    "receipt_id": receipt_id,
                    "authoritative_assurance": assurance.value,
                }
                zkp_result = dict(self._call(self.zkp, zkp_ctx, "zkp"))
                if str(zkp_result.get("status")) == "ok":
                    binding_payload = zkp_result.get("binding") or {}
                    zkp_binding = ZkpReceiptBinding.from_dict(binding_payload)
                    if zkp_binding.assurance_increased:
                        raise GoalDirectedTacticianError("ZKP binding increased receipt assurance")
                    # Bound assurance must equal receipt assurance projection.
                    assurance = zkp_binding.receipt_assurance
                    record(
                        TacticianPhase.ZKP_BIND,
                        PhaseStatus.OK,
                        utility_role=UtilityRole.ZKP_BINDING,
                        reason_code=str(
                            zkp_result.get("reason_code") or "zkp_bound_without_assurance_increase"
                        ),
                        evidence_source=EvidenceSource.ZKP_BINDING,
                        details=zkp_binding.to_dict(),
                    )
                else:
                    record(
                        TacticianPhase.ZKP_BIND,
                        self._status_from_payload(zkp_result),
                        utility_role=UtilityRole.ZKP_BINDING,
                        reason_code=str(zkp_result.get("reason_code") or "zkp_skipped"),
                        details=_public_mapping(dict(zkp_result)),
                    )
            else:
                record(
                    TacticianPhase.ZKP_BIND,
                    PhaseStatus.SKIPPED,
                    utility_role=UtilityRole.ZKP_BINDING,
                    reason_code="zkp_not_applicable",
                )

        proof_state, ranking, decomposition, prediction, traces, curriculum = (
            self._project_proof_state(
                request=req,
                context=base_context,
                independently_validated=independently_validated,
                kernel_verified=assurance.rank >= AssuranceLevel.KERNEL_VERIFIED.rank,
                record=record,
                completed=completed,
            )
        )

        # Supervisor admission (fail-closed).
        admission = self._admit(
            request=req,
            cache_key=cache_key,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            assurance=assurance,
            receipt_id=receipt_id,
        )
        record(
            TacticianPhase.ADMISSION,
            PhaseStatus.OK if admission.admitted else PhaseStatus.REJECTED,
            utility_role=UtilityRole.SUPERVISOR_ADMISSION,
            reason_code=(
                "admitted" if admission.admitted else ",".join(admission.reason_codes) or "rejected"
            ),
            details=admission.to_dict(),
        )

        checkpoint = self._write_checkpoint(
            cache_key=cache_key,
            completed_phases=completed + [TacticianPhase.ADMISSION.value],
            phases=phases,
            workflow_id=workflow_id,
            receipt_id=receipt_id,
            assurance=assurance,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            checkpoint_path=checkpoint_path,
        )
        record(
            TacticianPhase.CHECKPOINT,
            PhaseStatus.OK,
            utility_role=UtilityRole.PROOF_CARRYING_PLANNER,
            reason_code="checkpoint_persisted",
            details={
                "checkpoint_id": checkpoint.checkpoint_id if checkpoint else "",
                "resumable": bool(checkpoint and checkpoint.resumable),
            },
        )
        record(
            TacticianPhase.COMPLETE,
            PhaseStatus.OK if admission.admitted else PhaseStatus.REJECTED,
            reason_code="admitted" if admission.admitted else "not_admitted",
        )

        return GoalDirectedTacticianResult(
            stop_reason=(
                TacticianStopReason.ADMITTED
                if admission.admitted
                else TacticianStopReason.VALIDATION_FAILED
            ),
            cache_key=cache_key,
            phases=tuple(phases),
            utilities=self.utilities,
            admission=admission,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            resumable=True,
            checkpoint=checkpoint,
            zkp_binding=zkp_binding,
            receipt_id=receipt_id,
            authoritative_assurance=assurance,
            workflow_id=workflow_id,
            reason_code=("admitted" if admission.admitted else "admission_rejected"),
            details={
                "interface": GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE,
                "version": GOAL_DIRECTED_PROOF_TACTICIAN_VERSION,
            },
            proof_state=proof_state,
            ranking=ranking,
            decomposition=decomposition,
            prediction=prediction,
            traces=traces,
            curriculum=curriculum,
        )

    # -- internals -----------------------------------------------------------

    def _project_proof_state(
        self,
        *,
        request: GoalDirectedTacticianRequest,
        context: Mapping[str, Any],
        independently_validated: bool,
        kernel_verified: bool,
        record: Callable[..., None],
        completed: Sequence[str],
    ) -> tuple[
        ProofStateClassification,
        PremiseRankingReport | None,
        GoalDecomposition | None,
        BranchCostFailurePrediction | None,
        tuple[TacticPremiseTrace, ...],
        CurriculumProjection,
    ]:
        """Classify, rank, decompose, predict, and project curriculum (guidance)."""

        ranking: PremiseRankingReport | None = None
        decomposition: GoalDecomposition | None = None
        retrieve_result = context.get("retrieve_result") or {}
        candidates = []
        if isinstance(retrieve_result, Mapping):
            raw = retrieve_result.get("candidates") or retrieve_result.get("premises") or ()
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                candidates = [item for item in raw if isinstance(item, Mapping)]
        extra_candidates = context.get("ranking_candidates") or request.metadata.get(
            "ranking_candidates"
        )
        if isinstance(extra_candidates, Sequence) and not isinstance(
            extra_candidates, (str, bytes, bytearray)
        ):
            candidates.extend(item for item in extra_candidates if isinstance(item, Mapping))
        relevant = context.get("relevant_ids") or request.metadata.get("relevant_ids") or ()
        top_k = int(context.get("top_k") or request.bounds.get("top_k") or DEFAULT_TOP_K)
        if TacticianPhase.RANK.value not in completed:
            if candidates:
                ranking = rank_tactics_and_premises(
                    candidates,
                    k=top_k,
                    relevant_ids=tuple(relevant) if isinstance(relevant, Sequence) else (),
                    kind=context.get("rank_kind") or RankedKind.PREMISE,
                )
                record(
                    TacticianPhase.RANK,
                    PhaseStatus.OK,
                    utility_role=UtilityRole.RETRIEVAL,
                    reason_code="ranked_candidates",
                    details=ranking.to_dict(),
                )
            else:
                record(
                    TacticianPhase.RANK,
                    PhaseStatus.SKIPPED,
                    utility_role=UtilityRole.RETRIEVAL,
                    reason_code="no_rankable_candidates",
                )

        raw_decomp = (
            context.get("decomposition")
            or (context.get("leanstral_result") or {}).get("decomposition")
            or request.metadata.get("decomposition")
        )
        if TacticianPhase.DECOMPOSE.value not in completed:
            if isinstance(raw_decomp, Mapping) or isinstance(raw_decomp, Sequence):
                try:
                    if isinstance(raw_decomp, Mapping) and raw_decomp.get("children") is not None:
                        decomposition = decompose_goal(
                            str(raw_decomp.get("parent_goal_id") or request.target_id),
                            tuple(raw_decomp.get("children") or ()),
                            max_branch_factor=int(
                                raw_decomp.get("max_branch_factor")
                                or request.bounds.get("max_branch_factor")
                                or DEFAULT_MAX_BRANCH_FACTOR
                            ),
                        )
                    else:
                        decomposition = decompose_goal(
                            request.target_id,
                            tuple(raw_decomp or ()),
                            max_branch_factor=int(
                                request.bounds.get("max_branch_factor") or DEFAULT_MAX_BRANCH_FACTOR
                            ),
                        )
                    record(
                        TacticianPhase.DECOMPOSE,
                        PhaseStatus.OK,
                        utility_role=UtilityRole.LEANSTRAL,
                        reason_code="bounded_decomposition",
                        details=decomposition.to_dict(),
                    )
                except GoalDirectedTacticianError as exc:
                    record(
                        TacticianPhase.DECOMPOSE,
                        PhaseStatus.REJECTED,
                        utility_role=UtilityRole.LEANSTRAL,
                        reason_code="bounded_branching_rejected",
                        details={"message": str(exc)},
                    )
                    decomposition = None
            else:
                record(
                    TacticianPhase.DECOMPOSE,
                    PhaseStatus.SKIPPED,
                    utility_role=UtilityRole.LEANSTRAL,
                    reason_code="no_decomposition",
                )

        prediction = predict_branch_cost_failure(
            ranking=ranking,
            decomposition=decomposition,
            max_branch_factor=int(
                request.bounds.get("max_branch_factor") or DEFAULT_MAX_BRANCH_FACTOR
            ),
        )
        if TacticianPhase.PREDICT.value not in completed:
            record(
                TacticianPhase.PREDICT,
                PhaseStatus.OK,
                utility_role=UtilityRole.SYMAI,
                reason_code="guidance_prediction",
                evidence_source=EvidenceSource.GUIDANCE,
                details=prediction.to_dict(),
            )

        classify_payload = {
            **dict(context.get("attempt_trace") or request.metadata.get("attempt_trace") or {}),
            "kernel_outcome": context.get("kernel_result") or {},
            "prover_outcome": context.get("prove_result") or {},
            "independently_validated": independently_validated,
            "kernel_verified": kernel_verified,
            "timeout": context.get("timeout") or request.metadata.get("timeout") or {},
            "counterexamples": context.get("counterexamples")
            or request.metadata.get("counterexamples")
            or (),
            "decomposition": decomposition.to_dict() if decomposition is not None else None,
        }
        proof_state = classify_proof_state(
            classify_payload,
            independently_validated=independently_validated,
            kernel_verified=kernel_verified,
        )
        if TacticianPhase.CLASSIFY.value not in completed:
            record(
                TacticianPhase.CLASSIFY,
                PhaseStatus.OK,
                utility_role=UtilityRole.CURRICULUM,
                reason_code=proof_state.reason_code,
                details=proof_state.to_dict(),
            )

        traces: list[TacticPremiseTrace] = []
        if ranking is not None:
            traces.append(
                build_tactic_premise_trace(
                    kind=RankedKind.PREMISE,
                    goal_id=request.target_id,
                    item_ids=ranking.ranked_ids,
                    model_revision=request.provider_version,
                    tool_revision=request.toolchain_id,
                    outcome=proof_state.curriculum_class.value,
                    independently_validated=independently_validated,
                )
            )
        if decomposition is not None:
            traces.append(
                build_tactic_premise_trace(
                    kind=RankedKind.TACTIC,
                    goal_id=request.target_id,
                    item_ids=tuple(item.subgoal_id for item in decomposition.children),
                    model_revision=request.provider_version,
                    tool_revision=request.toolchain_id,
                    outcome=proof_state.curriculum_class.value,
                    independently_validated=independently_validated,
                    metadata={"decomposition": True},
                )
            )
        curriculum = project_curriculum(
            proof_state,
            traces=tuple(traces),
            ranking=ranking,
            decomposition=decomposition,
            prediction=prediction,
            independently_validated=independently_validated,
        )
        if TacticianPhase.CURRICULUM.value not in completed:
            record(
                TacticianPhase.CURRICULUM,
                PhaseStatus.OK,
                utility_role=UtilityRole.CURRICULUM,
                reason_code=curriculum.reason_code,
                evidence_source=EvidenceSource.CURRICULUM,
                details=curriculum.to_dict(),
            )
        return proof_state, ranking, decomposition, prediction, tuple(traces), curriculum

    def _request(
        self, value: GoalDirectedTacticianRequest | Mapping[str, Any]
    ) -> GoalDirectedTacticianRequest:
        if isinstance(value, GoalDirectedTacticianRequest):
            return value
        return GoalDirectedTacticianRequest.from_dict(_mapping(value, field_name="request"))

    def _call(
        self,
        provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        context: Mapping[str, Any],
        name: str,
    ) -> Mapping[str, Any]:
        try:
            result = provider(dict(context))
        except GoalDirectedTacticianError:
            raise
        except Exception as exc:  # honest failure surface
            return {
                "status": "failed",
                "reason_code": f"{name}_error:{type(exc).__name__}",
                "message": str(exc),
            }
        if not isinstance(result, Mapping):
            return {
                "status": "failed",
                "reason_code": f"{name}_invalid_result",
            }
        return result

    def _status_from_payload(self, payload: Mapping[str, Any]) -> PhaseStatus:
        status = str(payload.get("status") or "ok").strip().lower()
        if status in {"ok", "success", "validated"}:
            return PhaseStatus.OK
        if status in {"skipped", "skip"}:
            return PhaseStatus.SKIPPED
        if status in {"rejected", "deny", "denied"}:
            return PhaseStatus.REJECTED
        if status in {"cancelled", "canceled"}:
            return PhaseStatus.CANCELLED
        if status in {"resumed", "resume"}:
            return PhaseStatus.RESUMED
        return PhaseStatus.FAILED

    def _lookup_cache(self, cache_key: ExactTacticianCacheKey) -> Mapping[str, Any] | None:
        if self.cache_lookup is not None:
            try:
                return self.cache_lookup(cache_key)
            except Exception:
                return None
        # FormalVerificationCache is receipt-oriented; custom lookup preferred.
        # Keep self._cache available for future adapter wiring without claiming
        # hits when no tactician-level cache_lookup is configured.
        _ = self._cache
        return None

    def _store_cache(
        self,
        cache_key: ExactTacticianCacheKey,
        payload: Mapping[str, Any],
    ) -> None:
        if self.cache_store is not None:
            try:
                self.cache_store(cache_key, payload)
            except Exception:
                return

    def _load_checkpoint(
        self,
        checkpoint: TacticianCheckpoint | Mapping[str, Any] | None,
        checkpoint_path: Path | str | None,
        cache_key: ExactTacticianCacheKey,
    ) -> TacticianCheckpoint | None:
        del cache_key  # used by caller for mismatch check
        if checkpoint is not None:
            if isinstance(checkpoint, TacticianCheckpoint):
                return checkpoint
            return TacticianCheckpoint.from_dict(_mapping(checkpoint, field_name="checkpoint"))
        path: Path | None = None
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif self.checkpoint_dir is not None:
            path = self.checkpoint_dir / DEFAULT_CHECKPOINT_FILENAME
        if path is not None and path.is_file():
            return TacticianCheckpoint.load(path)
        return None

    def _write_checkpoint(
        self,
        *,
        cache_key: ExactTacticianCacheKey,
        completed_phases: Sequence[str],
        phases: Sequence[PhaseRecord],
        workflow_id: str,
        receipt_id: str,
        assurance: AssuranceLevel,
        independently_validated: bool,
        legal_compatible: bool,
        checkpoint_path: Path | str | None,
    ) -> TacticianCheckpoint | None:
        checkpoint = TacticianCheckpoint(
            checkpoint_id=(
                f"checkpoint:sha256:{_sha256_hex({'key': cache_key.key_id, 'phases': list(completed_phases)})}"
            ),
            cache_key=cache_key,
            completed_phases=tuple(completed_phases),
            workflow_id=workflow_id,
            receipt_id=receipt_id,
            authoritative_assurance=assurance,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            phase_records=tuple(phases),
        )
        path: Path | None = None
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif self.checkpoint_dir is not None:
            path = self.checkpoint_dir / DEFAULT_CHECKPOINT_FILENAME
        if path is not None:
            checkpoint.write(path)
        return checkpoint

    def _admit(
        self,
        *,
        request: GoalDirectedTacticianRequest,
        cache_key: ExactTacticianCacheKey,
        independently_validated: bool,
        legal_compatible: bool,
        assurance: AssuranceLevel,
        receipt_id: str,
    ) -> AdmissionRecord:
        reasons: list[str] = []
        if not independently_validated:
            reasons.append("independent_validation_required")
        if request.require_legal_compatibility and not legal_compatible:
            reasons.append("legal_incompatible")
        if not assurance.satisfies(request.required_assurance):
            reasons.append("required_assurance_not_satisfied")
        if not receipt_id:
            reasons.append("receipt_required")
        if reasons:
            return AdmissionRecord(
                decision=AdmissionDecision.REJECTED,
                required_assurance=request.required_assurance,
                authoritative_assurance=assurance,
                independently_validated=independently_validated,
                legal_compatible=legal_compatible,
                reason_codes=tuple(reasons),
                cache_key_id=cache_key.key_id,
                receipt_id=receipt_id,
            )
        return AdmissionRecord(
            decision=AdmissionDecision.ADMITTED,
            required_assurance=request.required_assurance,
            authoritative_assurance=assurance,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            reason_codes=("admitted",),
            cache_key_id=cache_key.key_id,
            receipt_id=receipt_id,
        )

    def _cancel(
        self,
        cache_key: ExactTacticianCacheKey,
        phases: list[PhaseRecord],
        request: GoalDirectedTacticianRequest,
        legal_compatible: bool,
    ) -> GoalDirectedTacticianResult:
        phases.append(
            PhaseRecord(
                phase=TacticianPhase.COMPLETE,
                status=PhaseStatus.CANCELLED,
                reason_code="cancelled",
            )
        )
        return self._terminal(
            stop_reason=TacticianStopReason.CANCELLED,
            cache_key=cache_key,
            phases=phases,
            independently_validated=False,
            legal_compatible=legal_compatible,
            resumable=True,
            required_assurance=request.required_assurance,
            authoritative_assurance=AssuranceLevel.UNVERIFIED,
            reason_code="cancelled",
        )

    def _terminal(
        self,
        *,
        stop_reason: TacticianStopReason,
        cache_key: ExactTacticianCacheKey,
        phases: Sequence[PhaseRecord],
        independently_validated: bool,
        legal_compatible: bool,
        resumable: bool,
        required_assurance: AssuranceLevel,
        authoritative_assurance: AssuranceLevel,
        reason_code: str,
        workflow_id: str = "",
        receipt_id: str = "",
        checkpoint: TacticianCheckpoint | None = None,
        zkp_binding: ZkpReceiptBinding | None = None,
    ) -> GoalDirectedTacticianResult:
        admission = AdmissionRecord(
            decision=(
                AdmissionDecision.ADMITTED
                if stop_reason is TacticianStopReason.ADMITTED
                else AdmissionDecision.REJECTED
            ),
            required_assurance=required_assurance,
            authoritative_assurance=authoritative_assurance,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            reason_codes=(reason_code,) if reason_code else (),
            cache_key_id=cache_key.key_id,
            receipt_id=receipt_id,
        )
        return GoalDirectedTacticianResult(
            stop_reason=stop_reason,
            cache_key=cache_key,
            phases=tuple(phases),
            utilities=self.utilities,
            admission=admission,
            independently_validated=independently_validated,
            legal_compatible=legal_compatible,
            resumable=resumable,
            checkpoint=checkpoint,
            zkp_binding=zkp_binding,
            receipt_id=receipt_id,
            authoritative_assurance=authoritative_assurance,
            workflow_id=workflow_id,
            reason_code=reason_code,
        )


def run_goal_directed_tactician(
    request: GoalDirectedTacticianRequest | Mapping[str, Any],
    **kwargs: Any,
) -> GoalDirectedTacticianResult:
    """Functional entry point for ``GoalDirectedProofTactician@1``."""

    tactician_kwargs = {
        key: kwargs.pop(key)
        for key in (
            "formalize",
            "retrieve",
            "schedule",
            "plan",
            "hammer",
            "kernel",
            "leanstral",
            "guidance",
            "legal",
            "validate",
            "prove",
            "cache_lookup",
            "cache_store",
            "zkp",
            "cache",
            "utilities",
            "checkpoint_dir",
        )
        if key in kwargs
    }
    return GoalDirectedProofTactician(**tactician_kwargs).run(request, **kwargs)


def create_goal_directed_proof_tactician(
    **kwargs: Any,
) -> GoalDirectedProofTactician:
    """Construct a ``GoalDirectedProofTactician@1`` orchestration facade."""

    return GoalDirectedProofTactician(**kwargs)


__all__ = [
    "ABSOLUTE_MAX_BRANCH_FACTOR",
    "ABSOLUTE_MAX_TOP_K",
    "CURRICULUM_PROJECTION_SCHEMA",
    "DEFAULT_MAX_BRANCH_FACTOR",
    "DEFAULT_TOP_K",
    "GOAL_DECOMPOSITION_SCHEMA",
    "GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE",
    "GOAL_DIRECTED_PROOF_TACTICIAN_SCHEMA",
    "GOAL_DIRECTED_PROOF_TACTICIAN_VERSION",
    "PROOF_STATE_CLASSIFICATION_SCHEMA",
    "RANKING_BASIS_POINTS",
    "TACTICIAN_ADMISSION_SCHEMA",
    "TACTICIAN_CACHE_KEY_SCHEMA",
    "TACTICIAN_CHECKPOINT_SCHEMA",
    "TACTICIAN_PHASE_RECORD_SCHEMA",
    "TACTICIAN_REQUEST_SCHEMA",
    "TACTICIAN_RESULT_SCHEMA",
    "TACTICIAN_UTILITY_BINDING_SCHEMA",
    "TACTICIAN_ZKP_BINDING_SCHEMA",
    "TACTIC_PREMISE_TRACE_SCHEMA",
    "AdmissionDecision",
    "AdmissionRecord",
    "BranchCostFailurePrediction",
    "CurriculumAuthority",
    "CurriculumClass",
    "CurriculumProjection",
    "EvidenceSource",
    "ExactTacticianCacheKey",
    "GoalDecomposition",
    "GoalDirectedProofTactician",
    "GoalDirectedTacticianCancelled",
    "GoalDirectedTacticianError",
    "GoalDirectedTacticianRequest",
    "GoalDirectedTacticianResult",
    "PhaseRecord",
    "PhaseStatus",
    "ProofStateClass",
    "ProofStateClassification",
    "RankedKind",
    "SubgoalNode",
    "TacticPremiseTrace",
    "TacticianCheckpoint",
    "TacticianPhase",
    "TacticianStopReason",
    "UtilityAuthority",
    "UtilityBinding",
    "UtilityRole",
    "ZkpReceiptBinding",
    "bind_zkp_to_trusted_receipt",
    "build_exact_tactician_cache_key",
    "build_tactic_premise_trace",
    "claims_authority",
    "classify_proof_state",
    "create_goal_directed_proof_tactician",
    "curriculum_authority_for",
    "decompose_goal",
    "default_utility_bindings",
    "expand_goal_branches",
    "predict_branch_cost_failure",
    "project_curriculum",
    "rank_tactics_and_premises",
    "recall_at_k",
    "reject_authority_bypass",
    "run_goal_directed_tactician",
    "select_top_k_ids",
]
