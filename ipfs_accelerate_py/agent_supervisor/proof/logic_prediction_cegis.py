"""Bounded counterexample-guided tactic refinement (LPR-013).

``LogicPredictionCEGIS`` is a monotonic, budgeted refinement state machine over
program-logic goals, hypotheses, premises, and residuals.  It sits between
Hammer coordination receipts and Tactician replanning:

* raw solver countermodels may guide *diagnostic* retrieval only — they cannot
  eliminate a hypothesis or influence admission authority;
* only a :class:`CountermodelValidationReceipt` with deterministic replay
  against the originating LogicIR semantics, or a kernel proof of negation,
  may narrow or reject a hypothesis;
* every original goal and required facet is preserved; subgoal conjunctions
  must refine (never weaken/delete) an original goal;
* model-promoted decompositions and unauthorized premises are rejected;
* state identity is content-addressed and monotonic; repeated states and
  cycles terminate;
* maximum rounds/goals/subgoals/premises/counterexamples plus wall/CPU/memory/
  context budgets are policy fields;
* cancellation, timeout, or bound exhaustion yields an *inconclusive* receipt
  with residual gaps for Tactician feedback;
* deterministic replay of a receipt is identity-equivalent.

The engine does not execute solvers, mutate source, or admit predictions.
Those authorities remain with the Hammer coordinator (LPR-012) and prediction
admission (LPR-014).
"""

from __future__ import annotations

import hashlib
import json
import resource
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    HypothesisDisposition,
    LogicGap,
    LogicHypothesis,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

LOGIC_PREDICTION_CEGIS_INTERFACE: Final = "LogicPredictionCEGIS@1"
LOGIC_REFINEMENT_BOUNDS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-refinement-bounds@1"
)
LOGIC_REFINEMENT_STATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-refinement-state@1"
)
LOGIC_REFINEMENT_ROUND_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-refinement-round@1"
)
LOGIC_REFINEMENT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-refinement-receipt@1"
)
SUBGOAL_REFINEMENT_PROOF_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/subgoal-refinement-proof@1"
)
TACTICIAN_RESIDUAL_FEEDBACK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-residual-feedback@1"
)

CEGIS_PRODUCER_ID: Final = "logic-prediction-cegis@1"
CEGIS_VERSION: Final = 1

# Hard ceilings (policy may only tighten these, never raise above).
HARD_MAX_ROUNDS: Final = 64
HARD_MAX_GOALS: Final = 256
HARD_MAX_SUBGOALS: Final = 256
HARD_MAX_PREMISES: Final = 512
HARD_MAX_COUNTEREXAMPLES: Final = 128
HARD_MAX_REPEATED_STATES: Final = 8
HARD_MAX_RESIDUAL_GAPS: Final = 256
HARD_MAX_CONTEXT_BYTES: Final = 262_144
HARD_MAX_WALL_MS: Final = 3_600_000
HARD_MAX_CPU_MS: Final = 3_600_000
HARD_MAX_MEMORY_BYTES: Final = 4 * 1024 * 1024 * 1024

DEFAULT_MAX_ROUNDS: Final = 8
DEFAULT_MAX_GOALS: Final = 32
DEFAULT_MAX_SUBGOALS: Final = 64
DEFAULT_MAX_PREMISES: Final = 64
DEFAULT_MAX_COUNTEREXAMPLES: Final = 16
DEFAULT_MAX_REPEATED_STATES: Final = 2
DEFAULT_MAX_RESIDUAL_GAPS: Final = 64
DEFAULT_MAX_CONTEXT_BYTES: Final = 65_536
DEFAULT_WALL_TIME_MS: Final = 60_000
DEFAULT_CPU_TIME_MS: Final = 30_000
DEFAULT_MEMORY_BYTES: Final = 512 * 1024 * 1024

# Nominating routes cannot authorize premises or model decompositions.
_NOMINATING_ROUTES: Final = frozenset(
    {
        SourceRouteKind.TACTICIAN,
        SourceRouteKind.VECTOR,
        SourceRouteKind.KNOWLEDGE_GRAPH,
        SourceRouteKind.LLM,
        SourceRouteKind.SOLVER,
    }
)

_AUTHORITATIVE_ROUTES: Final = frozenset(
    {
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.NORMATIVE_SPEC,
        SourceRouteKind.REVIEWED_TEST,
        SourceRouteKind.HISTORY,
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.RUNTIME_WITNESS,
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RefinementStopReason(str, Enum):
    """Closed terminal reasons for the refinement loop."""

    FIXED_POINT = "fixed_point"
    HYPOTHESES_RESOLVED = "hypotheses_resolved"
    REPEATED_STATE = "repeated_state"
    CYCLE_DETECTED = "cycle_detected"
    MAX_ROUNDS = "max_rounds"
    MAX_GOALS = "max_goals"
    MAX_SUBGOALS = "max_subgoals"
    MAX_PREMISES = "max_premises"
    MAX_COUNTEREXAMPLES = "max_counterexamples"
    MAX_RESIDUAL_GAPS = "max_residual_gaps"
    WALL_TIME_EXHAUSTED = "wall_time_exhausted"
    CPU_TIME_EXHAUSTED = "cpu_time_exhausted"
    MEMORY_EXHAUSTED = "memory_exhausted"
    CONTEXT_EXHAUSTED = "context_exhausted"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    NO_PROGRESS = "no_progress"
    AUTHORITY_VIOLATION = "authority_violation"
    INCONCLUSIVE = "inconclusive"


class RefinementDisposition(str, Enum):
    """Closed outcomes for a full refinement receipt."""

    REFINED = "refined"
    FIXED_POINT = "fixed_point"
    INCONCLUSIVE = "inconclusive"
    BOUND_EXHAUSTED = "bound_exhausted"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RoundDisposition(str, Enum):
    """Closed outcomes for one refinement round."""

    APPLIED = "applied"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    NO_OP = "no_op"
    BOUND_HIT = "bound_hit"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RefinementActionKind(str, Enum):
    """Closed action vocabulary applied during a round."""

    REJECT_HYPOTHESIS = "reject_hypothesis"
    NARROW_HYPOTHESIS = "narrow_hypothesis"
    EXCLUDE_PREMISE = "exclude_premise"
    ADD_AUTHORIZED_PREMISE = "add_authorized_premise"
    DECOMPOSE_SUBGOALS = "decompose_subgoals"
    RECORD_RESIDUAL = "record_residual"
    DIAGNOSTIC_RETRIEVAL = "diagnostic_retrieval"
    FEED_TACTICIAN = "feed_tactician"
    ABSTAIN = "abstain"


_BOUND_STOP_REASONS: Final = frozenset(
    {
        RefinementStopReason.MAX_ROUNDS,
        RefinementStopReason.MAX_GOALS,
        RefinementStopReason.MAX_SUBGOALS,
        RefinementStopReason.MAX_PREMISES,
        RefinementStopReason.MAX_COUNTEREXAMPLES,
        RefinementStopReason.MAX_RESIDUAL_GAPS,
        RefinementStopReason.WALL_TIME_EXHAUSTED,
        RefinementStopReason.CPU_TIME_EXHAUSTED,
        RefinementStopReason.MEMORY_EXHAUSTED,
        RefinementStopReason.CONTEXT_EXHAUSTED,
        RefinementStopReason.TIMEOUT,
    }
)

_TERMINAL_INCONCLUSIVE: Final = frozenset(
    {
        RefinementStopReason.REPEATED_STATE,
        RefinementStopReason.CYCLE_DETECTED,
        RefinementStopReason.NO_PROGRESS,
        RefinementStopReason.INCONCLUSIVE,
        RefinementStopReason.CANCELLED,
        *_BOUND_STOP_REASONS,
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LogicPredictionCegisError(ContractValidationError):
    """Base class for CEGIS contract failures."""


class LogicPredictionCegisBoundsError(LogicPredictionCegisError):
    """A refinement step would exceed declared policy bounds."""


class LogicPredictionCegisAuthorityError(LogicPredictionCegisError):
    """Authority, monotonicity, or admission-boundary violation."""


class LogicPredictionCegisMonotonicityError(LogicPredictionCegisAuthorityError):
    """State transition would weaken, delete, or non-monotonically rewrite goals."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        normalized = ""
    elif not isinstance(value, str):
        raise LogicPredictionCegisError(f"{field_name} must be a string")
    else:
        normalized = value.strip()
    if required and not normalized:
        raise LogicPredictionCegisError(f"{field_name} must not be empty")
    return normalized


def _positive_int(
    value: Any,
    *,
    field_name: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LogicPredictionCegisError(
            f"{field_name} must be an integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise LogicPredictionCegisBoundsError(
            f"{field_name}={value} exceeds hard ceiling {maximum}"
        )
    return value


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogicPredictionCegisError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise LogicPredictionCegisError(f"{field_name} must be a boolean")
    return value


def _ids(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    limit: int = HARD_MAX_PREMISES,
) -> tuple[str, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, str):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise LogicPredictionCegisError(f"{field_name} must be a sequence of ids")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item, field_name=field_name, required=True)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    if required and not out:
        raise LogicPredictionCegisError(f"{field_name} must not be empty")
    if len(out) > limit:
        raise LogicPredictionCegisBoundsError(
            f"{field_name} exceeds item bound {limit}"
        )
    return tuple(out)


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LogicPredictionCegisError(f"{field_name} must be an object")
    return {str(k): v for k, v in value.items()}


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return ProgramLogicAuthorityRoots.from_dict(dict(value))
    raise LogicPredictionCegisError("roots must be ProgramLogicAuthorityRoots")


def _enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise LogicPredictionCegisError(
            f"{field_name} must be one of "
            f"{sorted(item.value for item in enum_type)}"
        ) from exc


def _digest(payload: Mapping[str, Any] | Sequence[Any], *, prefix: str) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


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
    raise LogicPredictionCegisError(
        "cancelled must be a boolean, predicate, event, or None"
    )


def _cpu_time_ms() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int((usage.ru_utime + usage.ru_stime) * 1000)


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in kilobytes.
    return int(usage.ru_maxrss) * 1024


def _measured_rss_bytes() -> int | None:
    """Return a trustworthy process RSS high-water measurement, if available."""

    try:
        measured = _rss_bytes()
    except Exception:
        # Resource accounting is part of the admission boundary.  An
        # unavailable or malformed reading must never disable the bound.
        return None
    if isinstance(measured, bool) or not isinstance(measured, int) or measured <= 0:
        return None
    return measured


def _attributable_rss_bytes(baseline_bytes: int | None) -> int | None:
    """Measure RSS high-water growth attributable to the current campaign."""

    if (
        isinstance(baseline_bytes, bool)
        or not isinstance(baseline_bytes, int)
        or baseline_bytes <= 0
    ):
        return None
    measured = _measured_rss_bytes()
    if measured is None or measured < baseline_bytes:
        # ``ru_maxrss`` is monotonic.  A lower observation means the resource
        # measurement is not trustworthy, so fail closed rather than treating
        # the apparent decrease as free memory.
        return None
    return measured - baseline_bytes


def _goal_id(goal: ProgramLogicGoal | Mapping[str, Any] | str) -> str:
    if isinstance(goal, ProgramLogicGoal):
        return goal.goal_id
    if isinstance(goal, Mapping):
        return _text(goal.get("goal_id"), field_name="goal_id")
    return _text(goal, field_name="goal_id")


def _hypothesis_id(item: LogicHypothesis | Mapping[str, Any] | str) -> str:
    if isinstance(item, LogicHypothesis):
        return item.hypothesis_id
    if isinstance(item, Mapping):
        return _text(item.get("hypothesis_id"), field_name="hypothesis_id")
    return _text(item, field_name="hypothesis_id")


def _facet_ids(goal: ProgramLogicGoal) -> tuple[str, ...]:
    return tuple(sorted({facet.facet_id for facet in goal.required_facets}))


def _countermodel_may_reject(receipt: CountermodelValidationReceipt) -> bool:
    """Only independently validated receipts may narrow or reject."""

    if not receipt.may_reject_hypothesis:
        return False
    if receipt.disposition is not CountermodelDisposition.VALIDATED:
        return False
    # Deterministic LogicIR replay or proof of negation is mandatory.
    has_replay = bool(receipt.replayed_rejection_evidence_refs) and bool(
        receipt.replay_method
    )
    has_negation = bool(receipt.proof_of_negation_id)
    return has_replay or has_negation


# ---------------------------------------------------------------------------
# Bounds policy (explicit policy fields, not hidden constants)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRefinementBounds:
    """Finite refinement budgets declared as policy fields.

    Callers may only *tighten* these bounds; the hard ceilings above cannot be
    escalated through policy.
    """

    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_goals: int = DEFAULT_MAX_GOALS
    max_subgoals: int = DEFAULT_MAX_SUBGOALS
    max_premises: int = DEFAULT_MAX_PREMISES
    max_counterexamples: int = DEFAULT_MAX_COUNTEREXAMPLES
    max_repeated_states: int = DEFAULT_MAX_REPEATED_STATES
    max_residual_gaps: int = DEFAULT_MAX_RESIDUAL_GAPS
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES
    wall_time_ms: int = DEFAULT_WALL_TIME_MS
    cpu_time_ms: int = DEFAULT_CPU_TIME_MS
    memory_bytes: int = DEFAULT_MEMORY_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_rounds",
            _positive_int(
                self.max_rounds,
                field_name="max_rounds",
                maximum=HARD_MAX_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "max_goals",
            _positive_int(
                self.max_goals, field_name="max_goals", maximum=HARD_MAX_GOALS
            ),
        )
        object.__setattr__(
            self,
            "max_subgoals",
            _positive_int(
                self.max_subgoals,
                field_name="max_subgoals",
                maximum=HARD_MAX_SUBGOALS,
            ),
        )
        object.__setattr__(
            self,
            "max_premises",
            _positive_int(
                self.max_premises,
                field_name="max_premises",
                maximum=HARD_MAX_PREMISES,
            ),
        )
        object.__setattr__(
            self,
            "max_counterexamples",
            _positive_int(
                self.max_counterexamples,
                field_name="max_counterexamples",
                maximum=HARD_MAX_COUNTEREXAMPLES,
            ),
        )
        object.__setattr__(
            self,
            "max_repeated_states",
            _positive_int(
                self.max_repeated_states,
                field_name="max_repeated_states",
                maximum=HARD_MAX_REPEATED_STATES,
            ),
        )
        object.__setattr__(
            self,
            "max_residual_gaps",
            _positive_int(
                self.max_residual_gaps,
                field_name="max_residual_gaps",
                maximum=HARD_MAX_RESIDUAL_GAPS,
            ),
        )
        object.__setattr__(
            self,
            "max_context_bytes",
            _positive_int(
                self.max_context_bytes,
                field_name="max_context_bytes",
                minimum=1024,
                maximum=HARD_MAX_CONTEXT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "wall_time_ms",
            _positive_int(
                self.wall_time_ms,
                field_name="wall_time_ms",
                minimum=1,
                maximum=HARD_MAX_WALL_MS,
            ),
        )
        object.__setattr__(
            self,
            "cpu_time_ms",
            _positive_int(
                self.cpu_time_ms,
                field_name="cpu_time_ms",
                minimum=1,
                maximum=HARD_MAX_CPU_MS,
            ),
        )
        object.__setattr__(
            self,
            "memory_bytes",
            _positive_int(
                self.memory_bytes,
                field_name="memory_bytes",
                minimum=1024,
                maximum=HARD_MAX_MEMORY_BYTES,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "schema": LOGIC_REFINEMENT_BOUNDS_SCHEMA,
            "max_rounds": self.max_rounds,
            "max_goals": self.max_goals,
            "max_subgoals": self.max_subgoals,
            "max_premises": self.max_premises,
            "max_counterexamples": self.max_counterexamples,
            "max_repeated_states": self.max_repeated_states,
            "max_residual_gaps": self.max_residual_gaps,
            "max_context_bytes": self.max_context_bytes,
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRefinementBounds":
        if not isinstance(payload, Mapping):
            raise LogicPredictionCegisError("bounds must be an object")
        data = {k: v for k, v in payload.items() if k != "schema"}
        return cls(**data)

    def tighten(self, other: "LogicRefinementBounds") -> "LogicRefinementBounds":
        """Intersect two policies (component-wise min)."""

        return LogicRefinementBounds(
            max_rounds=min(self.max_rounds, other.max_rounds),
            max_goals=min(self.max_goals, other.max_goals),
            max_subgoals=min(self.max_subgoals, other.max_subgoals),
            max_premises=min(self.max_premises, other.max_premises),
            max_counterexamples=min(
                self.max_counterexamples, other.max_counterexamples
            ),
            max_repeated_states=min(
                self.max_repeated_states, other.max_repeated_states
            ),
            max_residual_gaps=min(self.max_residual_gaps, other.max_residual_gaps),
            max_context_bytes=min(self.max_context_bytes, other.max_context_bytes),
            wall_time_ms=min(self.wall_time_ms, other.wall_time_ms),
            cpu_time_ms=min(self.cpu_time_ms, other.cpu_time_ms),
            memory_bytes=min(self.memory_bytes, other.memory_bytes),
        )


# ---------------------------------------------------------------------------
# Subgoal refinement proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgoalRefinementProof:
    """Proof that a subgoal conjunction refines an original goal without weakening.

    The conjunction must:

    * cover every required facet of the parent;
    * reference only the original goal as parent (no goal deletion);
    * carry independent source authority for each clause when the decomposition
      is model-proposed (otherwise rejected as unauthorized model promotion).
    """

    proof_id: str
    parent_goal_id: str
    subgoal_ids: tuple[str, ...]
    covered_facet_ids: tuple[str, ...]
    required_facet_ids: tuple[str, ...]
    independent_source_authority: bool
    model_proposed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proof_id", _text(self.proof_id, field_name="proof_id")
        )
        object.__setattr__(
            self,
            "parent_goal_id",
            _text(self.parent_goal_id, field_name="parent_goal_id"),
        )
        object.__setattr__(
            self,
            "subgoal_ids",
            _ids(self.subgoal_ids, field_name="subgoal_ids", required=True),
        )
        object.__setattr__(
            self,
            "covered_facet_ids",
            _ids(self.covered_facet_ids, field_name="covered_facet_ids"),
        )
        object.__setattr__(
            self,
            "required_facet_ids",
            _ids(self.required_facet_ids, field_name="required_facet_ids"),
        )
        object.__setattr__(
            self,
            "independent_source_authority",
            _bool(
                self.independent_source_authority,
                field_name="independent_source_authority",
            ),
        )
        object.__setattr__(
            self,
            "model_proposed",
            _bool(self.model_proposed, field_name="model_proposed"),
        )
        required = set(self.required_facet_ids)
        covered = set(self.covered_facet_ids)
        if required - covered:
            raise LogicPredictionCegisMonotonicityError(
                "subgoal conjunction does not cover every original required facet: "
                f"{sorted(required - covered)}"
            )
        if self.model_proposed and not self.independent_source_authority:
            raise LogicPredictionCegisAuthorityError(
                "model-proposed decompositions require independent source authority "
                "on every resulting clause; cannot promote model decomposition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUBGOAL_REFINEMENT_PROOF_SCHEMA,
            "proof_id": self.proof_id,
            "parent_goal_id": self.parent_goal_id,
            "subgoal_ids": list(self.subgoal_ids),
            "covered_facet_ids": list(self.covered_facet_ids),
            "required_facet_ids": list(self.required_facet_ids),
            "independent_source_authority": self.independent_source_authority,
            "model_proposed": self.model_proposed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubgoalRefinementProof":
        if not isinstance(payload, Mapping):
            raise LogicPredictionCegisError("refinement proof must be an object")
        data = {k: v for k, v in payload.items() if k != "schema"}
        return cls(**data)


# ---------------------------------------------------------------------------
# Refinement actions / residual feedback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinementAction:
    """One typed, canonical refinement step applied inside a round."""

    kind: RefinementActionKind
    target_id: str
    evidence_ref: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, RefinementActionKind, "kind")
        )
        object.__setattr__(
            self, "target_id", _text(self.target_id, field_name="target_id")
        )
        object.__setattr__(
            self,
            "evidence_ref",
            _text(self.evidence_ref, field_name="evidence_ref", required=False),
        )
        object.__setattr__(
            self, "details", MappingProxyType(_mapping(self.details, field_name="details"))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_id": self.target_id,
            "evidence_ref": self.evidence_ref,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementAction":
        return cls(
            kind=payload.get("kind", RefinementActionKind.ABSTAIN),
            target_id=str(payload.get("target_id") or ""),
            evidence_ref=str(payload.get("evidence_ref") or ""),
            details=payload.get("details") or {},
        )


@dataclass(frozen=True)
class TacticianResidualFeedback:
    """Explicit residual packet fed back to Tactician (never silent drops)."""

    feedback_id: str
    residual_gap_ids: tuple[str, ...]
    diagnostic_countermodel_ids: tuple[str, ...]
    excluded_hypothesis_ids: tuple[str, ...]
    excluded_premise_ids: tuple[str, ...]
    query_hints: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    stop_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feedback_id", _text(self.feedback_id, field_name="feedback_id")
        )
        object.__setattr__(
            self,
            "residual_gap_ids",
            _ids(self.residual_gap_ids, field_name="residual_gap_ids"),
        )
        object.__setattr__(
            self,
            "diagnostic_countermodel_ids",
            _ids(
                self.diagnostic_countermodel_ids,
                field_name="diagnostic_countermodel_ids",
            ),
        )
        object.__setattr__(
            self,
            "excluded_hypothesis_ids",
            _ids(self.excluded_hypothesis_ids, field_name="excluded_hypothesis_ids"),
        )
        object.__setattr__(
            self,
            "excluded_premise_ids",
            _ids(self.excluded_premise_ids, field_name="excluded_premise_ids"),
        )
        object.__setattr__(
            self, "query_hints", _ids(self.query_hints, field_name="query_hints")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, field_name="reason_codes")
        )
        object.__setattr__(
            self,
            "stop_reason",
            _text(self.stop_reason, field_name="stop_reason", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TACTICIAN_RESIDUAL_FEEDBACK_SCHEMA,
            "feedback_id": self.feedback_id,
            "residual_gap_ids": list(self.residual_gap_ids),
            "diagnostic_countermodel_ids": list(self.diagnostic_countermodel_ids),
            "excluded_hypothesis_ids": list(self.excluded_hypothesis_ids),
            "excluded_premise_ids": list(self.excluded_premise_ids),
            "query_hints": list(self.query_hints),
            "reason_codes": list(self.reason_codes),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TacticianResidualFeedback":
        data = {k: v for k, v in dict(payload).items() if k != "schema"}
        return cls(**data)


# ---------------------------------------------------------------------------
# LogicRefinementState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRefinementState:
    """Monotonic, content-addressed refinement state.

    *original_goal_ids* and *original_facet_ids* are frozen at construction and
    may never shrink.  Active sets may only narrow (hypotheses/premises) or
    grow via authorized subgoals whose conjunction refines an original goal.
    """

    roots: ProgramLogicAuthorityRoots
    original_goal_ids: tuple[str, ...]
    original_facet_ids: tuple[str, ...]
    active_goal_ids: tuple[str, ...]
    active_hypothesis_ids: tuple[str, ...]
    excluded_hypothesis_ids: tuple[str, ...] = ()
    selected_premise_ids: tuple[str, ...] = ()
    excluded_premise_ids: tuple[str, ...] = ()
    authorized_premise_ids: tuple[str, ...] = ()
    subgoal_ids: tuple[str, ...] = ()
    residual_gap_ids: tuple[str, ...] = ()
    validated_countermodel_ids: tuple[str, ...] = ()
    diagnostic_countermodel_ids: tuple[str, ...] = ()
    refinement_proof_ids: tuple[str, ...] = ()
    lineage_state_ids: tuple[str, ...] = ()
    round_index: int = 0
    tactician_plan_id: str = ""
    corpus_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Derived; filled in __post_init__.
    state_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self,
            "original_goal_ids",
            _ids(self.original_goal_ids, field_name="original_goal_ids", required=True),
        )
        object.__setattr__(
            self,
            "original_facet_ids",
            _ids(self.original_facet_ids, field_name="original_facet_ids"),
        )
        object.__setattr__(
            self,
            "active_goal_ids",
            _ids(self.active_goal_ids, field_name="active_goal_ids", required=True),
        )
        # Monotonicity: every original goal must remain present (active or as
        # parent of subgoals).  Active set must be a superset of originals
        # union any refined children tracked separately; we require originals
        # ⊆ active ∪ covered-by-subgoals, enforced as originals ⊆ active here
        # (subgoals refine but do not replace identity of the original).
        original = set(self.original_goal_ids)
        active = set(self.active_goal_ids)
        if not original.issubset(active):
            missing = sorted(original - active)
            raise LogicPredictionCegisMonotonicityError(
                f"cannot delete or drop original goals: {missing}"
            )
        object.__setattr__(
            self,
            "active_hypothesis_ids",
            _ids(self.active_hypothesis_ids, field_name="active_hypothesis_ids"),
        )
        object.__setattr__(
            self,
            "excluded_hypothesis_ids",
            _ids(self.excluded_hypothesis_ids, field_name="excluded_hypothesis_ids"),
        )
        active_h = set(self.active_hypothesis_ids)
        excluded_h = set(self.excluded_hypothesis_ids)
        if active_h & excluded_h:
            raise LogicPredictionCegisError(
                "active and excluded hypotheses must be disjoint"
            )
        object.__setattr__(
            self,
            "selected_premise_ids",
            _ids(self.selected_premise_ids, field_name="selected_premise_ids"),
        )
        object.__setattr__(
            self,
            "excluded_premise_ids",
            _ids(self.excluded_premise_ids, field_name="excluded_premise_ids"),
        )
        selected = set(self.selected_premise_ids)
        excluded_p = set(self.excluded_premise_ids)
        if selected & excluded_p:
            raise LogicPredictionCegisError(
                "selected and excluded premises must be disjoint"
            )
        object.__setattr__(
            self,
            "authorized_premise_ids",
            _ids(self.authorized_premise_ids, field_name="authorized_premise_ids"),
        )
        object.__setattr__(
            self, "subgoal_ids", _ids(self.subgoal_ids, field_name="subgoal_ids")
        )
        object.__setattr__(
            self,
            "residual_gap_ids",
            _ids(self.residual_gap_ids, field_name="residual_gap_ids"),
        )
        object.__setattr__(
            self,
            "validated_countermodel_ids",
            _ids(
                self.validated_countermodel_ids,
                field_name="validated_countermodel_ids",
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_countermodel_ids",
            _ids(
                self.diagnostic_countermodel_ids,
                field_name="diagnostic_countermodel_ids",
            ),
        )
        # Validated and diagnostic channels must stay disjoint.
        if set(self.validated_countermodel_ids) & set(self.diagnostic_countermodel_ids):
            raise LogicPredictionCegisAuthorityError(
                "validated and diagnostic countermodel channels must be disjoint"
            )
        object.__setattr__(
            self,
            "refinement_proof_ids",
            _ids(self.refinement_proof_ids, field_name="refinement_proof_ids"),
        )
        object.__setattr__(
            self,
            "lineage_state_ids",
            _ids(self.lineage_state_ids, field_name="lineage_state_ids"),
        )
        object.__setattr__(
            self, "round_index", _nonnegative_int(self.round_index, field_name="round_index")
        )
        object.__setattr__(
            self,
            "tactician_plan_id",
            _text(self.tactician_plan_id, field_name="tactician_plan_id", required=False),
        )
        object.__setattr__(
            self,
            "corpus_id",
            _text(self.corpus_id, field_name="corpus_id", required=False),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, field_name="metadata"))
        )
        # Content-addressed state identity (excludes lineage to avoid circularity
        # but includes prior identity tip for monotonic chain).
        payload = self._identity_payload()
        derived = content_identity(payload)
        if self.state_id and self.state_id != derived:
            raise LogicPredictionCegisAuthorityError(
                "forged or stale state_id does not match canonical preimage"
            )
        object.__setattr__(self, "state_id", derived)

    def _semantic_payload(self) -> dict[str, Any]:
        """Semantic content used for cycle / repeated-state detection.

        Excludes ``round_index`` and lineage so re-applying equivalent evidence
        is recognized as a repeated state even though the monotonic lineage
        identity continues to advance.
        """

        return {
            "schema": LOGIC_REFINEMENT_STATE_SCHEMA,
            "roots": self.roots.to_dict(),
            "original_goal_ids": list(self.original_goal_ids),
            "original_facet_ids": list(self.original_facet_ids),
            "active_goal_ids": list(self.active_goal_ids),
            "active_hypothesis_ids": list(self.active_hypothesis_ids),
            "excluded_hypothesis_ids": list(self.excluded_hypothesis_ids),
            "selected_premise_ids": list(self.selected_premise_ids),
            "excluded_premise_ids": list(self.excluded_premise_ids),
            "authorized_premise_ids": list(self.authorized_premise_ids),
            "subgoal_ids": list(self.subgoal_ids),
            "residual_gap_ids": list(self.residual_gap_ids),
            "validated_countermodel_ids": list(self.validated_countermodel_ids),
            "diagnostic_countermodel_ids": list(self.diagnostic_countermodel_ids),
            "refinement_proof_ids": list(self.refinement_proof_ids),
            "tactician_plan_id": self.tactician_plan_id,
            "corpus_id": self.corpus_id,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "round_index": self.round_index,
            "lineage_tip": (
                self.lineage_state_ids[-1] if self.lineage_state_ids else ""
            ),
        }

    @property
    def semantic_id(self) -> str:
        """Content identity of the semantic refinement surface (cycle key)."""

        return content_identity(self._semantic_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "state_id": self.state_id,
            "semantic_id": self.semantic_id,
            "lineage_state_ids": list(self.lineage_state_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRefinementState":
        if not isinstance(payload, Mapping):
            raise LogicPredictionCegisError("state must be an object")
        data = dict(payload)
        data.pop("schema", None)
        roots = _roots(data.pop("roots"))
        state_id = str(data.pop("state_id", "") or "")
        lineage = data.pop("lineage_state_ids", ())
        metadata = data.pop("metadata", {})
        # Drop derived tip / semantic fields if present.
        data.pop("lineage_tip", None)
        data.pop("semantic_id", None)
        state = cls(
            roots=roots,
            lineage_state_ids=lineage,
            metadata=metadata,
            state_id=state_id,
            **data,
        )
        return state

    @property
    def open_hypothesis_count(self) -> int:
        return len(self.active_hypothesis_ids)

    def is_fixed_point(self) -> bool:
        return not self.active_hypothesis_ids and not self.residual_gap_ids


# ---------------------------------------------------------------------------
# LogicRefinementRound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRefinementRound:
    """One bounded refinement round with explicit actions and residuals."""

    round_id: str
    round_index: int
    input_state_id: str
    output_state_id: str
    disposition: RoundDisposition
    actions: tuple[RefinementAction, ...] = ()
    countermodel_receipt_ids: tuple[str, ...] = ()
    residual_feedback: Mapping[str, Any] | None = None
    stop_reason: str = ""
    reason_codes: tuple[str, ...] = ()
    wall_time_ms: int = 0
    cpu_time_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "round_id", _text(self.round_id, field_name="round_id")
        )
        object.__setattr__(
            self,
            "round_index",
            _nonnegative_int(self.round_index, field_name="round_index"),
        )
        object.__setattr__(
            self,
            "input_state_id",
            _text(self.input_state_id, field_name="input_state_id"),
        )
        object.__setattr__(
            self,
            "output_state_id",
            _text(self.output_state_id, field_name="output_state_id"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RoundDisposition, "disposition"),
        )
        actions: list[RefinementAction] = []
        if self.actions is None:
            raw_actions: Sequence[Any] = ()
        elif isinstance(self.actions, Sequence) and not isinstance(
            self.actions, (str, bytes, bytearray)
        ):
            raw_actions = self.actions
        else:
            raise LogicPredictionCegisError("actions must be a sequence")
        for item in raw_actions:
            if isinstance(item, RefinementAction):
                actions.append(item)
            elif isinstance(item, Mapping):
                actions.append(RefinementAction.from_dict(item))
            else:
                raise LogicPredictionCegisError("action entries must be objects")
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(
            self,
            "countermodel_receipt_ids",
            _ids(
                self.countermodel_receipt_ids,
                field_name="countermodel_receipt_ids",
            ),
        )
        if self.residual_feedback is not None:
            object.__setattr__(
                self,
                "residual_feedback",
                MappingProxyType(
                    _mapping(self.residual_feedback, field_name="residual_feedback")
                ),
            )
        object.__setattr__(
            self,
            "stop_reason",
            _text(self.stop_reason, field_name="stop_reason", required=False),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, field_name="reason_codes")
        )
        object.__setattr__(
            self,
            "wall_time_ms",
            _nonnegative_int(self.wall_time_ms, field_name="wall_time_ms"),
        )
        object.__setattr__(
            self,
            "cpu_time_ms",
            _nonnegative_int(self.cpu_time_ms, field_name="cpu_time_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LOGIC_REFINEMENT_ROUND_SCHEMA,
            "round_id": self.round_id,
            "round_index": self.round_index,
            "input_state_id": self.input_state_id,
            "output_state_id": self.output_state_id,
            "disposition": self.disposition.value,
            "actions": [item.to_dict() for item in self.actions],
            "countermodel_receipt_ids": list(self.countermodel_receipt_ids),
            "residual_feedback": (
                dict(self.residual_feedback)
                if self.residual_feedback is not None
                else None
            ),
            "stop_reason": self.stop_reason,
            "reason_codes": list(self.reason_codes),
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRefinementRound":
        data = {k: v for k, v in dict(payload).items() if k != "schema"}
        return cls(**data)


# ---------------------------------------------------------------------------
# LogicRefinementReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRefinementReceipt:
    """Terminal, content-addressed receipt for a refinement campaign."""

    receipt_id: str
    disposition: RefinementDisposition
    stop_reason: RefinementStopReason
    initial_state_id: str
    final_state: LogicRefinementState
    rounds: tuple[LogicRefinementRound, ...]
    residual_gap_ids: tuple[str, ...]
    tactician_feedback: TacticianResidualFeedback | None
    original_goal_ids: tuple[str, ...]
    original_facet_ids: tuple[str, ...]
    bounds: LogicRefinementBounds
    reason_codes: tuple[str, ...] = ()
    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    peak_memory_bytes: int = 0
    context_bytes: int = 0
    cancelled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, field_name="receipt_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RefinementDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "stop_reason",
            _enum(self.stop_reason, RefinementStopReason, "stop_reason"),
        )
        object.__setattr__(
            self,
            "initial_state_id",
            _text(self.initial_state_id, field_name="initial_state_id"),
        )
        if not isinstance(self.final_state, LogicRefinementState):
            if isinstance(self.final_state, Mapping):
                object.__setattr__(
                    self, "final_state", LogicRefinementState.from_dict(self.final_state)
                )
            else:
                raise LogicPredictionCegisError(
                    "final_state must be a LogicRefinementState"
                )
        # Preserve every original goal/facet on the terminal state.
        if set(self.original_goal_ids) != set(self.final_state.original_goal_ids):
            raise LogicPredictionCegisMonotonicityError(
                "receipt original goals must match final state originals"
            )
        if not set(self.original_goal_ids).issubset(
            set(self.final_state.active_goal_ids)
        ):
            raise LogicPredictionCegisMonotonicityError(
                "final state dropped original goals"
            )
        if set(self.original_facet_ids) != set(self.final_state.original_facet_ids):
            raise LogicPredictionCegisMonotonicityError(
                "receipt original facets must match final state originals"
            )
        rounds: list[LogicRefinementRound] = []
        for item in self.rounds or ():
            if isinstance(item, LogicRefinementRound):
                rounds.append(item)
            elif isinstance(item, Mapping):
                rounds.append(LogicRefinementRound.from_dict(item))
            else:
                raise LogicPredictionCegisError("rounds must contain round objects")
        object.__setattr__(self, "rounds", tuple(rounds))
        object.__setattr__(
            self,
            "residual_gap_ids",
            _ids(self.residual_gap_ids, field_name="residual_gap_ids"),
        )
        if self.tactician_feedback is not None and not isinstance(
            self.tactician_feedback, TacticianResidualFeedback
        ):
            object.__setattr__(
                self,
                "tactician_feedback",
                TacticianResidualFeedback.from_dict(
                    _mapping(self.tactician_feedback, field_name="tactician_feedback")
                ),
            )
        object.__setattr__(
            self,
            "original_goal_ids",
            _ids(self.original_goal_ids, field_name="original_goal_ids", required=True),
        )
        object.__setattr__(
            self,
            "original_facet_ids",
            _ids(self.original_facet_ids, field_name="original_facet_ids"),
        )
        if not isinstance(self.bounds, LogicRefinementBounds):
            object.__setattr__(
                self,
                "bounds",
                LogicRefinementBounds.from_dict(
                    _mapping(self.bounds, field_name="bounds")
                ),
            )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, field_name="reason_codes")
        )
        object.__setattr__(
            self,
            "wall_time_ms",
            _nonnegative_int(self.wall_time_ms, field_name="wall_time_ms"),
        )
        object.__setattr__(
            self,
            "cpu_time_ms",
            _nonnegative_int(self.cpu_time_ms, field_name="cpu_time_ms"),
        )
        object.__setattr__(
            self,
            "peak_memory_bytes",
            _nonnegative_int(self.peak_memory_bytes, field_name="peak_memory_bytes"),
        )
        object.__setattr__(
            self,
            "context_bytes",
            _nonnegative_int(self.context_bytes, field_name="context_bytes"),
        )
        object.__setattr__(
            self, "cancelled", _bool(self.cancelled, field_name="cancelled")
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, field_name="metadata"))
        )
        # Bound exhaustion / cancellation / cycle → inconclusive family.
        if self.stop_reason in _BOUND_STOP_REASONS:
            if self.disposition not in {
                RefinementDisposition.BOUND_EXHAUSTED,
                RefinementDisposition.INCONCLUSIVE,
            }:
                raise LogicPredictionCegisError(
                    "bound exhaustion must yield bound_exhausted or inconclusive"
                )
        if self.stop_reason is RefinementStopReason.CANCELLED:
            if self.disposition is not RefinementDisposition.CANCELLED:
                raise LogicPredictionCegisError(
                    "cancelled stop reason requires cancelled disposition"
                )
            if not self.cancelled:
                raise LogicPredictionCegisError(
                    "cancelled disposition requires cancelled=True"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LOGIC_REFINEMENT_RECEIPT_SCHEMA,
            "interface": LOGIC_PREDICTION_CEGIS_INTERFACE,
            "producer_id": CEGIS_PRODUCER_ID,
            "cegis_version": CEGIS_VERSION,
            "receipt_id": self.receipt_id,
            "disposition": self.disposition.value,
            "stop_reason": self.stop_reason.value,
            "initial_state_id": self.initial_state_id,
            "final_state": self.final_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "residual_gap_ids": list(self.residual_gap_ids),
            "tactician_feedback": (
                self.tactician_feedback.to_dict()
                if self.tactician_feedback is not None
                else None
            ),
            "original_goal_ids": list(self.original_goal_ids),
            "original_facet_ids": list(self.original_facet_ids),
            "bounds": self.bounds.to_dict(),
            "reason_codes": list(self.reason_codes),
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "context_bytes": self.context_bytes,
            "cancelled": self.cancelled,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRefinementReceipt":
        if not isinstance(payload, Mapping):
            raise LogicPredictionCegisError("receipt must be an object")
        data = dict(payload)
        for key in ("schema", "interface", "producer_id", "cegis_version"):
            data.pop(key, None)
        return cls(**data)

    @property
    def is_conclusive(self) -> bool:
        return self.disposition in {
            RefinementDisposition.REFINED,
            RefinementDisposition.FIXED_POINT,
        }

    @property
    def identity(self) -> str:
        return content_identity(self.to_dict())


# ---------------------------------------------------------------------------
# Evidence packet for one round
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinementEvidence:
    """Inputs observed for a single refinement round.

    Raw solver countermodel ids are diagnostic-only.  Only
    :class:`CountermodelValidationReceipt` instances with
    ``may_reject_hypothesis`` may reject or narrow.
    """

    countermodel_receipts: tuple[CountermodelValidationReceipt, ...] = ()
    raw_solver_countermodel_ids: tuple[str, ...] = ()
    residual_gaps: tuple[LogicGap | Mapping[str, Any] | str, ...] = ()
    authorized_premises_to_add: tuple[str, ...] = ()
    premises_to_exclude: tuple[str, ...] = ()
    subgoal_decomposition: tuple[LogicSubgoal | Mapping[str, Any], ...] = ()
    refinement_proof: SubgoalRefinementProof | Mapping[str, Any] | None = None
    hammer_coordination_receipt_id: str = ""
    hypothesis_narrowings: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    model_proposed_decomposition: bool = False
    query_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        receipts: list[CountermodelValidationReceipt] = []
        for item in self.countermodel_receipts or ():
            if isinstance(item, CountermodelValidationReceipt):
                receipts.append(item)
            elif isinstance(item, Mapping):
                receipts.append(CountermodelValidationReceipt.from_dict(item))
            else:
                raise LogicPredictionCegisError(
                    "countermodel_receipts must be CountermodelValidationReceipt"
                )
        object.__setattr__(self, "countermodel_receipts", tuple(receipts))
        object.__setattr__(
            self,
            "raw_solver_countermodel_ids",
            _ids(
                self.raw_solver_countermodel_ids,
                field_name="raw_solver_countermodel_ids",
            ),
        )
        object.__setattr__(
            self,
            "authorized_premises_to_add",
            _ids(
                self.authorized_premises_to_add,
                field_name="authorized_premises_to_add",
            ),
        )
        object.__setattr__(
            self,
            "premises_to_exclude",
            _ids(self.premises_to_exclude, field_name="premises_to_exclude"),
        )
        object.__setattr__(
            self,
            "query_hints",
            _ids(self.query_hints, field_name="query_hints"),
        )
        object.__setattr__(
            self,
            "hammer_coordination_receipt_id",
            _text(
                self.hammer_coordination_receipt_id,
                field_name="hammer_coordination_receipt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "model_proposed_decomposition",
            _bool(
                self.model_proposed_decomposition,
                field_name="model_proposed_decomposition",
            ),
        )
        object.__setattr__(
            self,
            "hypothesis_narrowings",
            MappingProxyType(
                {
                    str(k): dict(v) if isinstance(v, Mapping) else {"value": v}
                    for k, v in dict(self.hypothesis_narrowings or {}).items()
                }
            ),
        )
        if self.refinement_proof is not None and not isinstance(
            self.refinement_proof, SubgoalRefinementProof
        ):
            object.__setattr__(
                self,
                "refinement_proof",
                SubgoalRefinementProof.from_dict(
                    _mapping(self.refinement_proof, field_name="refinement_proof")
                ),
            )


# ---------------------------------------------------------------------------
# LogicPredictionCEGIS engine
# ---------------------------------------------------------------------------


@dataclass
class LogicPredictionCEGIS:
    """Bounded counterexample-guided tactic refinement engine.

    The engine is pure with respect to external provers: callers supply
    validated countermodel receipts, residual gaps, and authorized premises.
    Resource meters use process-local wall/CPU/RSS snapshots against policy.
    """

    bounds: LogicRefinementBounds = field(default_factory=LogicRefinementBounds)
    producer_id: str = CEGIS_PRODUCER_ID
    _cancelled: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, LogicRefinementBounds):
            self.bounds = LogicRefinementBounds.from_dict(
                _mapping(self.bounds, field_name="bounds")
            )
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id, field_name="producer_id"),
        )

    # -- lifecycle --------------------------------------------------------

    def cancel(self) -> None:
        """Signal cooperative cancellation for an in-flight refine() call."""

        self._cancelled.set()

    def reset_cancellation(self) -> None:
        self._cancelled.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    # -- state construction -----------------------------------------------

    def initial_state(
        self,
        *,
        roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any] | str],
        hypotheses: Sequence[LogicHypothesis | Mapping[str, Any] | str] = (),
        plan: TacticianSearchPlan | Mapping[str, Any] | None = None,
        authorized_premise_ids: Sequence[str] = (),
        selected_premise_ids: Sequence[str] = (),
        residual_gap_ids: Sequence[str] = (),
        corpus_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> LogicRefinementState:
        """Build the initial monotonic refinement state from goals/hypotheses."""

        roots_obj = _roots(roots)
        goal_ids: list[str] = []
        facet_ids: list[str] = []
        for goal in goals:
            if isinstance(goal, ProgramLogicGoal):
                if goal.roots != roots_obj and goal.roots.to_dict() != roots_obj.to_dict():
                    # Compare by content identity of roots.
                    if content_identity(goal.roots.to_dict()) != content_identity(
                        roots_obj.to_dict()
                    ):
                        raise LogicPredictionCegisAuthorityError(
                            f"goal {goal.goal_id} roots do not match refinement roots"
                        )
                goal_ids.append(goal.goal_id)
                facet_ids.extend(_facet_ids(goal))
            else:
                goal_ids.append(_goal_id(goal))
                if isinstance(goal, Mapping):
                    for facet in goal.get("required_facets") or ():
                        if isinstance(facet, Mapping) and facet.get("facet_id"):
                            facet_ids.append(str(facet["facet_id"]))
                        elif isinstance(facet, str):
                            facet_ids.append(facet)

        if not goal_ids:
            raise LogicPredictionCegisError("at least one original goal is required")
        if len(goal_ids) > self.bounds.max_goals:
            raise LogicPredictionCegisBoundsError(
                f"goal count {len(goal_ids)} exceeds max_goals={self.bounds.max_goals}"
            )

        hyp_ids: list[str] = []
        for hyp in hypotheses:
            hyp_ids.append(_hypothesis_id(hyp))
            if isinstance(hyp, LogicHypothesis):
                if hyp.target_goal_id not in goal_ids:
                    raise LogicPredictionCegisError(
                        f"hypothesis {hyp.hypothesis_id} targets unknown goal "
                        f"{hyp.target_goal_id}"
                    )

        plan_id = ""
        plan_selected: tuple[str, ...] = ()
        plan_subgoals: tuple[str, ...] = ()
        if plan is not None:
            if isinstance(plan, Mapping):
                plan = TacticianSearchPlan.from_dict(plan)
            if not isinstance(plan, TacticianSearchPlan):
                raise LogicPredictionCegisError("plan must be a TacticianSearchPlan")
            if plan.semantic_authority:
                raise LogicPredictionCegisAuthorityError(
                    "tactician plans cannot claim semantic authority"
                )
            plan_id = plan.plan_id
            plan_selected = plan.selected_premise_ids
            plan_subgoals = tuple(sg.subgoal_id for sg in plan.subgoals)
            for gid in plan.goal_ids:
                if gid not in goal_ids:
                    raise LogicPredictionCegisError(
                        f"plan goal {gid} is not among original goals"
                    )
            if len(plan.subgoals) > self.bounds.max_subgoals:
                raise LogicPredictionCegisBoundsError(
                    f"plan subgoals exceed max_subgoals={self.bounds.max_subgoals}"
                )

        selected = _ids(
            tuple(selected_premise_ids) or plan_selected,
            field_name="selected_premise_ids",
            limit=self.bounds.max_premises,
        )
        authorized = _ids(
            authorized_premise_ids,
            field_name="authorized_premise_ids",
            limit=HARD_MAX_PREMISES,
        )
        # Selected premises must be authorized when an authorization set is given.
        if authorized:
            unauthorized = set(selected) - set(authorized)
            if unauthorized:
                raise LogicPredictionCegisAuthorityError(
                    f"selected premises not authorized: {sorted(unauthorized)}"
                )

        if len(selected) > self.bounds.max_premises:
            raise LogicPredictionCegisBoundsError(
                f"selected premises exceed max_premises={self.bounds.max_premises}"
            )

        residuals = _ids(
            residual_gap_ids,
            field_name="residual_gap_ids",
            limit=self.bounds.max_residual_gaps,
        )

        return LogicRefinementState(
            roots=roots_obj,
            original_goal_ids=tuple(goal_ids),
            original_facet_ids=tuple(sorted(set(facet_ids))),
            active_goal_ids=tuple(goal_ids),
            active_hypothesis_ids=tuple(hyp_ids),
            selected_premise_ids=selected,
            authorized_premise_ids=authorized,
            subgoal_ids=plan_subgoals,
            residual_gap_ids=residuals,
            tactician_plan_id=plan_id,
            corpus_id=_text(corpus_id or roots_obj.corpus_id, field_name="corpus_id"),
            metadata=metadata or {},
        )

    # -- subgoal refinement -----------------------------------------------

    def prove_subgoal_refinement(
        self,
        *,
        parent_goal: ProgramLogicGoal | Mapping[str, Any],
        subgoals: Sequence[LogicSubgoal | Mapping[str, Any]],
        model_proposed: bool = False,
    ) -> SubgoalRefinementProof:
        """Prove a subgoal conjunction refines the original goal without weakening."""

        if isinstance(parent_goal, Mapping):
            parent_goal = ProgramLogicGoal.from_dict(parent_goal)
        if not isinstance(parent_goal, ProgramLogicGoal):
            raise LogicPredictionCegisError("parent_goal must be a ProgramLogicGoal")

        required = set(_facet_ids(parent_goal))
        covered: set[str] = set()
        subgoal_ids: list[str] = []
        independent = True

        if not subgoals:
            raise LogicPredictionCegisError(
                "subgoal decomposition requires at least one subgoal"
            )
        if len(subgoals) > self.bounds.max_subgoals:
            raise LogicPredictionCegisBoundsError(
                f"subgoal count exceeds max_subgoals={self.bounds.max_subgoals}"
            )

        for item in subgoals:
            if isinstance(item, Mapping):
                # Lightweight path for mapping subgoals (tests / overlays).
                sg_id = _text(item.get("subgoal_id"), field_name="subgoal_id")
                parent = str(item.get("parent_subgoal_id") or item.get("goal_id") or "")
                goal_id = str(item.get("goal_id") or "")
                source_route = item.get("source_route")
                source_authority = item.get("source_authority")
                claim_facets = item.get("covered_facet_ids") or item.get("facet_ids") or ()
            elif isinstance(item, LogicSubgoal):
                sg_id = item.subgoal_id
                parent = item.parent_subgoal_id
                goal_id = item.goal_id
                source_route = item.source_route
                source_authority = item.source_authority
                claim_facets = ()
            else:
                raise LogicPredictionCegisError("subgoals must be LogicSubgoal objects")

            if goal_id and goal_id != parent_goal.goal_id:
                raise LogicPredictionCegisMonotonicityError(
                    f"subgoal {sg_id} goal_id must equal parent goal "
                    f"{parent_goal.goal_id}"
                )
            subgoal_ids.append(sg_id)

            # Facet coverage: mapping may declare facets; otherwise inherit claim_ref.
            if claim_facets:
                covered.update(str(f) for f in claim_facets)
            else:
                # Without explicit facet claims, require independent authority and
                # treat claim_ref as an opaque cover token only when authoritative.
                pass

            route = (
                source_route
                if isinstance(source_route, SourceRouteKind)
                else (
                    SourceRouteKind(str(source_route))
                    if source_route
                    else SourceRouteKind.LOCAL_STATIC
                )
            )
            authority = (
                source_authority
                if isinstance(source_authority, SourceAuthorityClass)
                else (
                    SourceAuthorityClass(str(source_authority))
                    if source_authority
                    else SourceAuthorityClass.AUTHORITATIVE
                )
            )
            if route in _NOMINATING_ROUTES or authority in {
                SourceAuthorityClass.NOMINATING,
                SourceAuthorityClass.DIAGNOSTIC,
                SourceAuthorityClass.NONE,
            }:
                independent = False

        # If no explicit facet coverage was declared, require that the parent
        # had no required facets OR that every subgoal is independently
        # authoritative (caller must then re-bind facets via claim_ref).
        if required and not covered:
            # Allow refinement when parent facets are empty or when the caller
            # supplies coverage via a proof payload; otherwise fail closed.
            raise LogicPredictionCegisMonotonicityError(
                "subgoal conjunction must declare covered facets for every "
                "original required facet"
            )
        if required - covered:
            raise LogicPredictionCegisMonotonicityError(
                "subgoal conjunction omits original facets: "
                f"{sorted(required - covered)}"
            )

        proof_id = _digest(
            {
                "parent": parent_goal.goal_id,
                "subgoals": subgoal_ids,
                "covered": sorted(covered),
                "required": sorted(required),
            },
            prefix="subgoal-refinement",
        )
        return SubgoalRefinementProof(
            proof_id=proof_id,
            parent_goal_id=parent_goal.goal_id,
            subgoal_ids=tuple(subgoal_ids),
            covered_facet_ids=tuple(sorted(covered)),
            required_facet_ids=tuple(sorted(required)),
            independent_source_authority=independent,
            model_proposed=model_proposed,
        )

    # -- residual feedback ------------------------------------------------

    def residual_feedback_for_tactician(
        self,
        state: LogicRefinementState,
        *,
        query_hints: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
        stop_reason: str = "",
    ) -> TacticianResidualFeedback:
        """Build explicit residual packet for Tactician re-planning."""

        feedback_id = _digest(
            {
                "state_id": state.state_id,
                "residuals": list(state.residual_gap_ids),
                "diagnostics": list(state.diagnostic_countermodel_ids),
                "excluded_h": list(state.excluded_hypothesis_ids),
                "excluded_p": list(state.excluded_premise_ids),
            },
            prefix="tactician-residual",
        )
        return TacticianResidualFeedback(
            feedback_id=feedback_id,
            residual_gap_ids=state.residual_gap_ids,
            diagnostic_countermodel_ids=state.diagnostic_countermodel_ids,
            excluded_hypothesis_ids=state.excluded_hypothesis_ids,
            excluded_premise_ids=state.excluded_premise_ids,
            query_hints=tuple(query_hints),
            reason_codes=tuple(reason_codes),
            stop_reason=stop_reason,
        )

    # -- single round -----------------------------------------------------

    def apply_round(
        self,
        state: LogicRefinementState,
        evidence: RefinementEvidence,
        *,
        cancelled: Any = None,
        wall_started_ms: int | None = None,
        cpu_started_ms: int | None = None,
        memory_started_bytes: int | None = None,
    ) -> tuple[LogicRefinementState, LogicRefinementRound]:
        """Apply one refinement round.  Monotonic; never weakens original goals."""

        if not isinstance(state, LogicRefinementState):
            raise LogicPredictionCegisError("state must be a LogicRefinementState")
        if not isinstance(evidence, RefinementEvidence):
            raise LogicPredictionCegisError("evidence must be a RefinementEvidence")

        round_start_wall = time.monotonic()
        round_start_cpu = _cpu_time_ms()
        wall0 = wall_started_ms if wall_started_ms is not None else round_start_wall
        cpu0 = cpu_started_ms if cpu_started_ms is not None else round_start_cpu
        memory0 = (
            memory_started_bytes
            if memory_started_bytes is not None
            else _measured_rss_bytes()
        )

        actions: list[RefinementAction] = []
        reason_codes: list[str] = []
        receipt_ids: list[str] = []
        stop_reason = ""
        disposition = RoundDisposition.NO_OP

        if self.cancelled or _cancelled(cancelled):
            round_id = _digest(
                {"state": state.state_id, "round": state.round_index, "stop": "cancelled"},
                prefix="round",
            )
            rnd = LogicRefinementRound(
                round_id=round_id,
                round_index=state.round_index,
                input_state_id=state.state_id,
                output_state_id=state.state_id,
                disposition=RoundDisposition.CANCELLED,
                stop_reason=RefinementStopReason.CANCELLED.value,
                reason_codes=("cancelled",),
            )
            return state, rnd

        # Resource pre-checks.
        bound_hit = self._check_resources(
            state=state,
            wall0=wall0,
            cpu0=cpu0,
            memory0=memory0,
            projected_goals=len(state.active_goal_ids),
            projected_subgoals=len(state.subgoal_ids),
            projected_premises=len(state.selected_premise_ids),
            projected_counterexamples=len(state.validated_countermodel_ids)
            + len(state.diagnostic_countermodel_ids),
            projected_residuals=len(state.residual_gap_ids),
        )
        if bound_hit is not None:
            round_id = _digest(
                {
                    "state": state.state_id,
                    "round": state.round_index,
                    "stop": bound_hit.value,
                },
                prefix="round",
            )
            rnd = LogicRefinementRound(
                round_id=round_id,
                round_index=state.round_index,
                input_state_id=state.state_id,
                output_state_id=state.state_id,
                disposition=RoundDisposition.BOUND_HIT,
                stop_reason=bound_hit.value,
                reason_codes=(bound_hit.value,),
            )
            return state, rnd

        # Working copies (monotonic growth of exclusions / residuals).
        active_h = list(state.active_hypothesis_ids)
        excluded_h = list(state.excluded_hypothesis_ids)
        selected_p = list(state.selected_premise_ids)
        excluded_p = list(state.excluded_premise_ids)
        residuals = list(state.residual_gap_ids)
        validated_cm = list(state.validated_countermodel_ids)
        diagnostic_cm = list(state.diagnostic_countermodel_ids)
        subgoal_ids = list(state.subgoal_ids)
        proof_ids = list(state.refinement_proof_ids)
        active_goals = list(state.active_goal_ids)

        # 1) Raw solver countermodels → diagnostic only (never reject).
        for raw_id in evidence.raw_solver_countermodel_ids:
            if raw_id not in diagnostic_cm and raw_id not in validated_cm:
                diagnostic_cm.append(raw_id)
                actions.append(
                    RefinementAction(
                        kind=RefinementActionKind.DIAGNOSTIC_RETRIEVAL,
                        target_id=raw_id,
                        evidence_ref=raw_id,
                        details={"channel": "raw_solver", "may_reject": False},
                    )
                )
                reason_codes.append("countermodel_unvalidated")
                disposition = RoundDisposition.DIAGNOSTIC_ONLY

        # 2) CountermodelValidationReceipt processing.
        for receipt in evidence.countermodel_receipts:
            # Cross-root / stale receipts are non-authoritative.
            if content_identity(receipt.roots.to_dict()) != content_identity(
                state.roots.to_dict()
            ):
                reason_codes.append("countermodel_stale_roots")
                diagnostic_cm.append(receipt.receipt_id)
                actions.append(
                    RefinementAction(
                        kind=RefinementActionKind.DIAGNOSTIC_RETRIEVAL,
                        target_id=receipt.receipt_id,
                        evidence_ref=receipt.receipt_id,
                        details={"reason": "cross_root", "may_reject": False},
                    )
                )
                disposition = RoundDisposition.DIAGNOSTIC_ONLY
                continue

            receipt_ids.append(receipt.receipt_id)

            if _countermodel_may_reject(receipt):
                # Authoritative rejection path.
                if receipt.receipt_id not in validated_cm:
                    validated_cm.append(receipt.receipt_id)
                # Reject hypotheses that target this countermodel or are active
                # against the originating LogicIR.
                target_hyp = (
                    evidence.hypothesis_narrowings.get(receipt.receipt_id, {})
                    if evidence.hypothesis_narrowings
                    else {}
                )
                reject_ids = list(target_hyp.get("reject_hypothesis_ids") or ())
                narrow_ids = list(target_hyp.get("narrow_hypothesis_ids") or ())
                # Default: if receipt names a solver countermodel that matches
                # an active hypothesis id prefix, or if caller supplies none,
                # reject nothing silently — require explicit binding.
                if not reject_ids and not narrow_ids:
                    # Still record validated evidence; residual for Tactician.
                    reason_codes.append("validated_countermodel_unbound")
                    actions.append(
                        RefinementAction(
                            kind=RefinementActionKind.ABSTAIN,
                            target_id=receipt.receipt_id,
                            evidence_ref=receipt.receipt_id,
                            details={
                                "reason": "validated but no hypothesis binding",
                                "may_reject": True,
                            },
                        )
                    )
                for hid in reject_ids:
                    if hid in active_h:
                        active_h.remove(hid)
                        if hid not in excluded_h:
                            excluded_h.append(hid)
                        actions.append(
                            RefinementAction(
                                kind=RefinementActionKind.REJECT_HYPOTHESIS,
                                target_id=hid,
                                evidence_ref=receipt.receipt_id,
                                details={
                                    "replay_method": receipt.replay_method,
                                    "proof_of_negation_id": receipt.proof_of_negation_id,
                                    "originating_logic_ir_id": receipt.originating_logic_ir_id,
                                },
                            )
                        )
                        disposition = RoundDisposition.APPLIED
                        reason_codes.append("hypothesis_validated_refuted")
                for hid in narrow_ids:
                    if hid in active_h:
                        actions.append(
                            RefinementAction(
                                kind=RefinementActionKind.NARROW_HYPOTHESIS,
                                target_id=hid,
                                evidence_ref=receipt.receipt_id,
                                details=dict(target_hyp.get("narrow_details") or {}),
                            )
                        )
                        disposition = RoundDisposition.APPLIED
                        reason_codes.append("hypothesis_narrowed")
                    # Premise exclusions tied to validated countermodel.
                for prem in target_hyp.get("exclude_premise_ids") or ():
                    prem = str(prem)
                    if prem in selected_p:
                        selected_p.remove(prem)
                    if prem not in excluded_p:
                        excluded_p.append(prem)
                        actions.append(
                            RefinementAction(
                                kind=RefinementActionKind.EXCLUDE_PREMISE,
                                target_id=prem,
                                evidence_ref=receipt.receipt_id,
                            )
                        )
                        disposition = RoundDisposition.APPLIED
            else:
                # Diagnostic-only: may guide retrieval, never eliminate.
                if receipt.receipt_id not in diagnostic_cm:
                    diagnostic_cm.append(receipt.receipt_id)
                # Explicitly refuse any attempt to reject via this receipt.
                target_hyp = (
                    evidence.hypothesis_narrowings.get(receipt.receipt_id, {})
                    if evidence.hypothesis_narrowings
                    else {}
                )
                if target_hyp.get("reject_hypothesis_ids") or target_hyp.get(
                    "narrow_hypothesis_ids"
                ):
                    reason_codes.append("countermodel_unvalidated")
                    actions.append(
                        RefinementAction(
                            kind=RefinementActionKind.ABSTAIN,
                            target_id=receipt.receipt_id,
                            evidence_ref=receipt.receipt_id,
                            details={
                                "reason": (
                                    "raw/diagnostic countermodel cannot eliminate "
                                    "hypothesis or influence admission"
                                ),
                                "disposition": receipt.disposition.value,
                                "may_reject": False,
                                "refused_rejects": list(
                                    target_hyp.get("reject_hypothesis_ids") or ()
                                ),
                            },
                        )
                    )
                else:
                    actions.append(
                        RefinementAction(
                            kind=RefinementActionKind.DIAGNOSTIC_RETRIEVAL,
                            target_id=receipt.receipt_id,
                            evidence_ref=receipt.receipt_id,
                            details={
                                "disposition": receipt.disposition.value,
                                "may_reject": False,
                                "raw_diagnostic_refs": list(receipt.raw_diagnostic_refs),
                            },
                        )
                    )
                if disposition is RoundDisposition.NO_OP:
                    disposition = RoundDisposition.DIAGNOSTIC_ONLY

        # 3) Residual gaps (unsupported constructs, missing premises).
        for gap in evidence.residual_gaps:
            if isinstance(gap, LogicGap):
                gap_id = gap.gap_id
            elif isinstance(gap, Mapping):
                gap_id = _text(gap.get("gap_id"), field_name="gap_id")
            else:
                gap_id = _text(gap, field_name="gap_id")
            if gap_id not in residuals:
                residuals.append(gap_id)
                actions.append(
                    RefinementAction(
                        kind=RefinementActionKind.RECORD_RESIDUAL,
                        target_id=gap_id,
                        evidence_ref=gap_id,
                    )
                )
                disposition = RoundDisposition.APPLIED
                reason_codes.append("residual_gap")

        # 4) Authorized premise addition / exclusion.
        for prem in evidence.authorized_premises_to_add:
            if state.authorized_premise_ids and prem not in state.authorized_premise_ids:
                raise LogicPredictionCegisAuthorityError(
                    f"cannot add unauthorized premise {prem}"
                )
            if prem in excluded_p:
                raise LogicPredictionCegisMonotonicityError(
                    f"cannot re-add excluded premise {prem}"
                )
            if prem not in selected_p:
                selected_p.append(prem)
                actions.append(
                    RefinementAction(
                        kind=RefinementActionKind.ADD_AUTHORIZED_PREMISE,
                        target_id=prem,
                    )
                )
                disposition = RoundDisposition.APPLIED

        for prem in evidence.premises_to_exclude:
            if prem in selected_p:
                selected_p.remove(prem)
            if prem not in excluded_p:
                excluded_p.append(prem)
                actions.append(
                    RefinementAction(
                        kind=RefinementActionKind.EXCLUDE_PREMISE,
                        target_id=prem,
                    )
                )
                disposition = RoundDisposition.APPLIED

        # 5) Subgoal decomposition with refinement proof.
        if evidence.subgoal_decomposition:
            proof = evidence.refinement_proof
            if proof is None:
                raise LogicPredictionCegisAuthorityError(
                    "subgoal decomposition requires a SubgoalRefinementProof"
                )
            if not isinstance(proof, SubgoalRefinementProof):
                proof = SubgoalRefinementProof.from_dict(
                    _mapping(proof, field_name="refinement_proof")
                )
            if evidence.model_proposed_decomposition and not proof.independent_source_authority:
                raise LogicPredictionCegisAuthorityError(
                    "cannot promote a model decomposition without independent "
                    "source authority on every clause"
                )
            # Parent goal must be an original goal.
            if proof.parent_goal_id not in state.original_goal_ids:
                raise LogicPredictionCegisMonotonicityError(
                    "decomposition parent must be an original goal"
                )
            if proof.parent_goal_id not in active_goals:
                raise LogicPredictionCegisMonotonicityError(
                    "cannot decompose a deleted goal"
                )
            for sg in evidence.subgoal_decomposition:
                if isinstance(sg, LogicSubgoal):
                    sg_id = sg.subgoal_id
                    if sg.goal_id != proof.parent_goal_id:
                        raise LogicPredictionCegisMonotonicityError(
                            "subgoal goal_id must match parent"
                        )
                else:
                    sg_id = _text(sg.get("subgoal_id"), field_name="subgoal_id")
                if sg_id not in subgoal_ids:
                    subgoal_ids.append(sg_id)
            if proof.proof_id not in proof_ids:
                proof_ids.append(proof.proof_id)
            actions.append(
                RefinementAction(
                    kind=RefinementActionKind.DECOMPOSE_SUBGOALS,
                    target_id=proof.parent_goal_id,
                    evidence_ref=proof.proof_id,
                    details={
                        "subgoal_ids": list(proof.subgoal_ids),
                        "covered_facet_ids": list(proof.covered_facet_ids),
                    },
                )
            )
            disposition = RoundDisposition.APPLIED

        # Bound checks after mutation projections.
        if len(active_goals) > self.bounds.max_goals:
            raise LogicPredictionCegisBoundsError("max_goals exceeded")
        if len(subgoal_ids) > self.bounds.max_subgoals:
            raise LogicPredictionCegisBoundsError(
                f"max_subgoals={self.bounds.max_subgoals} exceeded"
            )
        if len(selected_p) > self.bounds.max_premises:
            raise LogicPredictionCegisBoundsError(
                f"max_premises={self.bounds.max_premises} exceeded"
            )
        if len(validated_cm) + len(diagnostic_cm) > self.bounds.max_counterexamples:
            raise LogicPredictionCegisBoundsError(
                f"max_counterexamples={self.bounds.max_counterexamples} exceeded"
            )
        if len(residuals) > self.bounds.max_residual_gaps:
            raise LogicPredictionCegisBoundsError(
                f"max_residual_gaps={self.bounds.max_residual_gaps} exceeded"
            )

        next_round_index = state.round_index + 1
        lineage = list(state.lineage_state_ids) + [state.state_id]

        new_state = LogicRefinementState(
            roots=state.roots,
            original_goal_ids=state.original_goal_ids,
            original_facet_ids=state.original_facet_ids,
            active_goal_ids=tuple(active_goals),
            active_hypothesis_ids=tuple(active_h),
            excluded_hypothesis_ids=tuple(excluded_h),
            selected_premise_ids=tuple(selected_p),
            excluded_premise_ids=tuple(excluded_p),
            authorized_premise_ids=state.authorized_premise_ids,
            subgoal_ids=tuple(subgoal_ids),
            residual_gap_ids=tuple(residuals),
            validated_countermodel_ids=tuple(validated_cm),
            diagnostic_countermodel_ids=tuple(diagnostic_cm),
            refinement_proof_ids=tuple(proof_ids),
            lineage_state_ids=tuple(lineage),
            round_index=next_round_index,
            tactician_plan_id=state.tactician_plan_id,
            corpus_id=state.corpus_id,
            metadata=dict(state.metadata),
        )

        # Semantic no-progress: lineage advances but the refinement surface is
        # unchanged (e.g. re-recording an already-known residual).
        if (
            new_state.semantic_id == state.semantic_id
            and disposition is not RoundDisposition.DIAGNOSTIC_ONLY
        ):
            disposition = RoundDisposition.NO_OP
            if "no_progress" not in reason_codes:
                reason_codes.append("no_progress")

        feedback = self.residual_feedback_for_tactician(
            new_state,
            query_hints=evidence.query_hints,
            reason_codes=reason_codes,
        )
        actions.append(
            RefinementAction(
                kind=RefinementActionKind.FEED_TACTICIAN,
                target_id=feedback.feedback_id,
                evidence_ref=feedback.feedback_id,
                details={"residual_count": len(feedback.residual_gap_ids)},
            )
        )

        wall_ms = int((time.monotonic() - round_start_wall) * 1000)
        cpu_ms = max(0, _cpu_time_ms() - round_start_cpu)
        round_id = _digest(
            {
                "input": state.state_id,
                "output": new_state.state_id,
                "round": state.round_index,
                "actions": [a.to_dict() for a in actions],
                "disposition": disposition.value,
            },
            prefix="round",
        )
        rnd = LogicRefinementRound(
            round_id=round_id,
            round_index=state.round_index,
            input_state_id=state.state_id,
            output_state_id=new_state.state_id,
            disposition=disposition,
            actions=tuple(actions),
            countermodel_receipt_ids=tuple(receipt_ids),
            residual_feedback=feedback.to_dict(),
            stop_reason=stop_reason,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            wall_time_ms=wall_ms,
            cpu_time_ms=cpu_ms,
        )
        return new_state, rnd

    def _check_resources(
        self,
        *,
        state: LogicRefinementState,
        wall0: float,
        cpu0: int,
        memory0: int | None,
        projected_goals: int,
        projected_subgoals: int,
        projected_premises: int,
        projected_counterexamples: int,
        projected_residuals: int,
    ) -> RefinementStopReason | None:
        if state.round_index >= self.bounds.max_rounds:
            return RefinementStopReason.MAX_ROUNDS
        if projected_goals > self.bounds.max_goals:
            return RefinementStopReason.MAX_GOALS
        if projected_subgoals > self.bounds.max_subgoals:
            return RefinementStopReason.MAX_SUBGOALS
        if projected_premises > self.bounds.max_premises:
            return RefinementStopReason.MAX_PREMISES
        if projected_counterexamples > self.bounds.max_counterexamples:
            return RefinementStopReason.MAX_COUNTEREXAMPLES
        if projected_residuals > self.bounds.max_residual_gaps:
            return RefinementStopReason.MAX_RESIDUAL_GAPS

        # wall0 may be monotonic seconds or ms depending on caller.
        now = time.monotonic()
        if wall0 > 1e12:
            # Treat as epoch ms (unlikely); fall back to elapsed 0.
            elapsed_wall_ms = 0
        elif wall0 > 1e6:
            elapsed_wall_ms = int(now * 1000 - wall0)
        else:
            elapsed_wall_ms = int((now - wall0) * 1000)
        if elapsed_wall_ms > self.bounds.wall_time_ms:
            return RefinementStopReason.WALL_TIME_EXHAUSTED

        elapsed_cpu = max(0, _cpu_time_ms() - cpu0)
        if elapsed_cpu > self.bounds.cpu_time_ms:
            return RefinementStopReason.CPU_TIME_EXHAUSTED

        attributable_rss = _attributable_rss_bytes(memory0)
        if (
            attributable_rss is None
            or attributable_rss > self.bounds.memory_bytes
        ):
            return RefinementStopReason.MEMORY_EXHAUSTED

        context_bytes = len(canonical_json(state.to_dict()).encode("utf-8"))
        if context_bytes > self.bounds.max_context_bytes:
            return RefinementStopReason.CONTEXT_EXHAUSTED
        return None

    # -- multi-round refine -----------------------------------------------

    def refine(
        self,
        initial: LogicRefinementState,
        evidence_stream: Sequence[RefinementEvidence] | Callable[[LogicRefinementState, int], RefinementEvidence | None],
        *,
        cancelled: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LogicRefinementReceipt:
        """Run the bounded CEGIS loop until fixed point, cycle, or bound exhaustion.

        ``evidence_stream`` may be a finite sequence of per-round evidence
        packets or a callable ``(state, round_index) -> evidence | None``.
        Returning ``None`` from the callable ends the loop (fixed point / no
        more evidence).
        """

        if not isinstance(initial, LogicRefinementState):
            raise LogicPredictionCegisError("initial must be a LogicRefinementState")

        wall0 = time.monotonic()
        cpu0 = _cpu_time_ms()
        memory0 = _measured_rss_bytes()
        peak_mem = 0
        state = initial
        rounds: list[LogicRefinementRound] = []
        # Cycle detection keys on *semantic* identity so re-applying equivalent
        # evidence terminates even while monotonic lineage identities advance.
        seen_semantic: list[str] = [initial.semantic_id]
        stop_reason = RefinementStopReason.INCONCLUSIVE
        disposition = RefinementDisposition.INCONCLUSIVE
        reason_codes: list[str] = []
        was_cancelled = False

        def _next_evidence(round_index: int) -> RefinementEvidence | None:
            if callable(evidence_stream):
                return evidence_stream(state, round_index)
            if round_index < len(evidence_stream):
                return evidence_stream[round_index]
            return None

        with self._lock:
            while True:
                if self.cancelled or _cancelled(cancelled):
                    stop_reason = RefinementStopReason.CANCELLED
                    disposition = RefinementDisposition.CANCELLED
                    was_cancelled = True
                    reason_codes.append("cancelled")
                    break

                attributable_rss = _attributable_rss_bytes(memory0)
                if attributable_rss is None:
                    stop_reason = RefinementStopReason.MEMORY_EXHAUSTED
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append("memory_measurement_invalid")
                    break
                peak_mem = max(peak_mem, attributable_rss)
                if attributable_rss > self.bounds.memory_bytes:
                    stop_reason = RefinementStopReason.MEMORY_EXHAUSTED
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append(stop_reason.value)
                    break

                if state.is_fixed_point():
                    stop_reason = RefinementStopReason.FIXED_POINT
                    disposition = RefinementDisposition.FIXED_POINT
                    reason_codes.append("fixed_point")
                    break

                if state.round_index >= self.bounds.max_rounds:
                    stop_reason = RefinementStopReason.MAX_ROUNDS
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append("max_rounds")
                    break

                bound_hit = self._check_resources(
                    state=state,
                    wall0=wall0,
                    cpu0=cpu0,
                    memory0=memory0,
                    projected_goals=len(state.active_goal_ids),
                    projected_subgoals=len(state.subgoal_ids),
                    projected_premises=len(state.selected_premise_ids),
                    projected_counterexamples=len(state.validated_countermodel_ids)
                    + len(state.diagnostic_countermodel_ids),
                    projected_residuals=len(state.residual_gap_ids),
                )
                if bound_hit is not None:
                    stop_reason = bound_hit
                    disposition = (
                        RefinementDisposition.CANCELLED
                        if bound_hit is RefinementStopReason.CANCELLED
                        else RefinementDisposition.BOUND_EXHAUSTED
                    )
                    reason_codes.append(bound_hit.value)
                    break

                evidence = _next_evidence(state.round_index)
                if evidence is None:
                    if state.active_hypothesis_ids:
                        stop_reason = RefinementStopReason.NO_PROGRESS
                        disposition = RefinementDisposition.INCONCLUSIVE
                        reason_codes.append("no_more_evidence")
                    else:
                        stop_reason = RefinementStopReason.HYPOTHESES_RESOLVED
                        disposition = RefinementDisposition.REFINED
                        reason_codes.append("hypotheses_resolved")
                    break

                try:
                    new_state, rnd = self.apply_round(
                        state,
                        evidence,
                        cancelled=cancelled,
                        wall_started_ms=wall0,
                        cpu_started_ms=cpu0,
                        memory_started_bytes=memory0,
                    )
                except LogicPredictionCegisAuthorityError as exc:
                    stop_reason = RefinementStopReason.AUTHORITY_VIOLATION
                    disposition = RefinementDisposition.REJECTED
                    reason_codes.append("authority_violation")
                    reason_codes.append(str(exc)[:200])
                    break
                except LogicPredictionCegisBoundsError as exc:
                    stop_reason = RefinementStopReason.MAX_ROUNDS
                    msg = str(exc)
                    if "max_subgoals" in msg:
                        stop_reason = RefinementStopReason.MAX_SUBGOALS
                    elif "max_premises" in msg:
                        stop_reason = RefinementStopReason.MAX_PREMISES
                    elif "max_counterexamples" in msg:
                        stop_reason = RefinementStopReason.MAX_COUNTEREXAMPLES
                    elif "max_residual" in msg:
                        stop_reason = RefinementStopReason.MAX_RESIDUAL_GAPS
                    elif "max_goals" in msg:
                        stop_reason = RefinementStopReason.MAX_GOALS
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append(stop_reason.value)
                    break

                rounds.append(rnd)
                attributable_rss = _attributable_rss_bytes(memory0)
                if attributable_rss is None:
                    state = new_state
                    stop_reason = RefinementStopReason.MEMORY_EXHAUSTED
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append("memory_measurement_invalid")
                    break
                peak_mem = max(peak_mem, attributable_rss)
                if attributable_rss > self.bounds.memory_bytes:
                    state = new_state
                    stop_reason = RefinementStopReason.MEMORY_EXHAUSTED
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append(stop_reason.value)
                    break

                if rnd.disposition is RoundDisposition.CANCELLED:
                    stop_reason = RefinementStopReason.CANCELLED
                    disposition = RefinementDisposition.CANCELLED
                    was_cancelled = True
                    reason_codes.append("cancelled")
                    break

                if rnd.disposition is RoundDisposition.BOUND_HIT:
                    stop_reason = RefinementStopReason(rnd.stop_reason or "max_rounds")
                    disposition = RefinementDisposition.BOUND_EXHAUSTED
                    reason_codes.append(stop_reason.value)
                    break

                # Semantic no-progress: surface unchanged after the round.
                if (
                    rnd.disposition is RoundDisposition.NO_OP
                    or new_state.semantic_id == state.semantic_id
                ):
                    # Count the repeated semantic surface (includes the prior).
                    repeat_count = seen_semantic.count(new_state.semantic_id) + 1
                    if repeat_count >= self.bounds.max_repeated_states:
                        stop_reason = (
                            RefinementStopReason.CYCLE_DETECTED
                            if repeat_count > self.bounds.max_repeated_states
                            else RefinementStopReason.REPEATED_STATE
                        )
                        # Prefer explicit no_progress when a single no-op repeats.
                        if rnd.disposition is RoundDisposition.NO_OP and repeat_count <= 2:
                            stop_reason = RefinementStopReason.NO_PROGRESS
                        disposition = RefinementDisposition.INCONCLUSIVE
                        reason_codes.append(stop_reason.value)
                        state = new_state
                        break
                    seen_semantic.append(new_state.semantic_id)
                    state = new_state
                    continue

                # Semantic progress: track and continue.
                if new_state.semantic_id in seen_semantic:
                    count = seen_semantic.count(new_state.semantic_id)
                    if count + 1 >= self.bounds.max_repeated_states:
                        stop_reason = (
                            RefinementStopReason.CYCLE_DETECTED
                            if count >= 1
                            else RefinementStopReason.REPEATED_STATE
                        )
                        disposition = RefinementDisposition.INCONCLUSIVE
                        reason_codes.append(stop_reason.value)
                        state = new_state
                        break
                seen_semantic.append(new_state.semantic_id)

                # Diagnostic-only rounds still advance state (diagnostic ids).
                state = new_state

                if state.is_fixed_point():
                    stop_reason = RefinementStopReason.FIXED_POINT
                    disposition = RefinementDisposition.FIXED_POINT
                    reason_codes.append("fixed_point")
                    break

                if (
                    not state.active_hypothesis_ids
                    and state.validated_countermodel_ids
                ):
                    stop_reason = RefinementStopReason.HYPOTHESES_RESOLVED
                    disposition = RefinementDisposition.REFINED
                    reason_codes.append("hypotheses_resolved")
                    break

        wall_ms = int((time.monotonic() - wall0) * 1000)
        cpu_ms = max(0, _cpu_time_ms() - cpu0)
        context_bytes = len(canonical_json(state.to_dict()).encode("utf-8"))

        # Inconclusive / bound exhaustion always expose residual gaps.
        residual_ids = state.residual_gap_ids
        if disposition in {
            RefinementDisposition.INCONCLUSIVE,
            RefinementDisposition.BOUND_EXHAUSTED,
            RefinementDisposition.CANCELLED,
        }:
            # Surface active hypotheses as residual work items when empty gaps.
            if not residual_ids and state.active_hypothesis_ids:
                residual_ids = tuple(
                    f"residual:hypothesis:{hid}" for hid in state.active_hypothesis_ids
                )

        feedback = self.residual_feedback_for_tactician(
            state,
            reason_codes=reason_codes,
            stop_reason=stop_reason.value,
        )

        receipt_id = _digest(
            {
                "initial": initial.state_id,
                "final": state.state_id,
                "disposition": disposition.value,
                "stop": stop_reason.value,
                "rounds": [r.round_id for r in rounds],
            },
            prefix="logic-refinement",
        )
        return LogicRefinementReceipt(
            receipt_id=receipt_id,
            disposition=disposition,
            stop_reason=stop_reason,
            initial_state_id=initial.state_id,
            final_state=state,
            rounds=tuple(rounds),
            residual_gap_ids=residual_ids,
            tactician_feedback=feedback,
            original_goal_ids=initial.original_goal_ids,
            original_facet_ids=initial.original_facet_ids,
            bounds=self.bounds,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            wall_time_ms=wall_ms,
            cpu_time_ms=cpu_ms,
            peak_memory_bytes=peak_mem,
            context_bytes=context_bytes,
            cancelled=was_cancelled,
            metadata=metadata or {},
        )

    # -- deterministic replay ---------------------------------------------

    def replay(self, receipt: LogicRefinementReceipt | Mapping[str, Any]) -> LogicRefinementReceipt:
        """Reconstruct a receipt from its canonical dict; identity-equivalent."""

        if isinstance(receipt, LogicRefinementReceipt):
            payload = receipt.to_dict()
            original_id = receipt.identity
        else:
            payload = dict(receipt)
            original_id = content_identity(payload)

        rebuilt = LogicRefinementReceipt.from_dict(payload)
        # Round-trip must be identity-equivalent.
        rebuilt_payload = rebuilt.to_dict()
        if content_identity(rebuilt_payload) != content_identity(payload):
            # Allow only if original carried non-canonical key order differences
            # that canonicalize equally — content_identity already canonicalizes.
            raise LogicPredictionCegisAuthorityError(
                "deterministic replay is not identity-equivalent"
            )
        if rebuilt.identity != content_identity(rebuilt_payload):
            raise LogicPredictionCegisAuthorityError(
                "receipt identity diverged from canonical payload"
            )
        # Re-hash of rebuilt must match original payload identity.
        if content_identity(rebuilt_payload) != original_id and isinstance(
            receipt, LogicRefinementReceipt
        ):
            # When starting from a live object, identities must match exactly.
            if rebuilt.identity != receipt.identity:
                raise LogicPredictionCegisAuthorityError(
                    "replayed receipt identity does not match original"
                )
        return rebuilt

    def state_identity(self, state: LogicRefinementState) -> str:
        """Return the content-addressed identity of a refinement state."""

        if not isinstance(state, LogicRefinementState):
            raise LogicPredictionCegisError("state must be a LogicRefinementState")
        return state.state_id


def create_logic_prediction_cegis(
    bounds: LogicRefinementBounds | Mapping[str, Any] | None = None,
) -> LogicPredictionCEGIS:
    """Factory for a default-bounded CEGIS engine."""

    if bounds is None:
        resolved = LogicRefinementBounds()
    elif isinstance(bounds, LogicRefinementBounds):
        resolved = bounds
    else:
        resolved = LogicRefinementBounds.from_dict(bounds)
    return LogicPredictionCEGIS(bounds=resolved)


__all__ = [
    "CEGIS_PRODUCER_ID",
    "CEGIS_VERSION",
    "HARD_MAX_CONTEXT_BYTES",
    "HARD_MAX_COUNTEREXAMPLES",
    "HARD_MAX_GOALS",
    "HARD_MAX_PREMISES",
    "HARD_MAX_REPEATED_STATES",
    "HARD_MAX_RESIDUAL_GAPS",
    "HARD_MAX_ROUNDS",
    "HARD_MAX_SUBGOALS",
    "LOGIC_PREDICTION_CEGIS_INTERFACE",
    "LOGIC_REFINEMENT_BOUNDS_SCHEMA",
    "LOGIC_REFINEMENT_RECEIPT_SCHEMA",
    "LOGIC_REFINEMENT_ROUND_SCHEMA",
    "LOGIC_REFINEMENT_STATE_SCHEMA",
    "LogicPredictionCEGIS",
    "LogicPredictionCegisAuthorityError",
    "LogicPredictionCegisBoundsError",
    "LogicPredictionCegisError",
    "LogicPredictionCegisMonotonicityError",
    "LogicRefinementBounds",
    "LogicRefinementReceipt",
    "LogicRefinementRound",
    "LogicRefinementState",
    "RefinementAction",
    "RefinementActionKind",
    "RefinementDisposition",
    "RefinementEvidence",
    "RefinementStopReason",
    "RoundDisposition",
    "SubgoalRefinementProof",
    "TacticianResidualFeedback",
    "create_logic_prediction_cegis",
]
