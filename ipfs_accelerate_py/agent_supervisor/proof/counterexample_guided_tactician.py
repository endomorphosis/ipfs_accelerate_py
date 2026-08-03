"""Bounded verifier-backed CEGIS/CEGAR loop for proof development.

``CounterexampleGuidedProofDevelopment@1`` owns the supervisor refinement loop:

1. normalize (and optionally replay) a verified counterexample witness;
2. refine the proof graph or a candidate invariant / contract / lemma / repair;
3. validate the candidate independently of the originating verifier;
4. rerun the *exact* originating verifier on the repaired tree / property;
5. close the counterexample only when a fresh bound receipt succeeds; and
6. keep disagreement, timeout, unavailable, or bound-change outcomes open or
   unknown.

Unchanged witnesses back off without another full candidate synthesis cycle.
Repeated identical failure terminates under policy. Structural admissibility
never alone reduces the open-witness count — that authority stays with
:func:`~ipfs_accelerate_py.agent_supervisor.planning.formal_replanner.evaluate_verifier_backed_closure`
so formal-replanner retry / fencing semantics remain intact.

Canonical datasets validation and replay are reached only through injectable
providers. This module never imports package-private datasets paths and never
claims completion, admission, or kernel authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ..planning.formal_replanner import (
    VERIFIER_BACKED_REPAIR_CLOSURE_SCHEMA,
    VerifierBackedRepairClosure,
    VerifierClosureReceipt,
    WitnessClosureStatus,
    _bound_digest,
    evaluate_verifier_backed_closure,
)
from .formal_counterexamples import (
    CounterexampleValidationError,
    FormalCounterexample,
    normalize_counterexample,
)
from .formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE: Final = (
    "CounterexampleGuidedProofDevelopment@1"
)
COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_VERSION: Final = "1.0.0"
COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-guided-proof-development@1"
)
CEGIS_BUDGET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cegis-budget@1"
)
CEGIS_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cegis-refinement-candidate@1"
)
CEGIS_ITERATION_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cegis-iteration-binding@1"
)
CEGIS_ITERATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cegis-iteration@1"
)
CEGIS_LOOP_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cegis-loop-result@1"
)

DEFAULT_MAX_ITERATIONS: Final = 8
DEFAULT_MAX_CANDIDATES_PER_ITERATION: Final = 4
DEFAULT_MAX_IDENTICAL_FAILURES: Final = 3
DEFAULT_BASE_BACKOFF_SECONDS: Final = 1
DEFAULT_MAX_BACKOFF_SECONDS: Final = 300
ABSOLUTE_MAX_ITERATIONS: Final = 64
ABSOLUTE_MAX_CANDIDATES_PER_ITERATION: Final = 32


# ---------------------------------------------------------------------------
# Errors and enums
# ---------------------------------------------------------------------------


class CegisValidationError(ContractValidationError):
    """Raised when a CEGIS/CEGAR request or artifact violates the contract."""


class CegisCancelled(CegisValidationError):
    """Cooperative cancellation stopped the loop before a terminal outcome."""


class CandidateKind(str, Enum):
    """Reviewed families of refinement artifacts produced by the loop."""

    INVARIANT = "invariant"
    CONTRACT = "contract"
    LEMMA = "lemma"
    REPAIR = "repair"
    PROOF_GRAPH_EDGE = "proof_graph_edge"
    PREMISE = "premise"
    BOUND_ADJUSTMENT = "bound_adjustment"


class CandidateValidationStatus(str, Enum):
    """Independent candidate-validation outcomes (never proof authority)."""

    VALID = "valid"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class IterationPhase(str, Enum):
    """Ordered phases recorded on each auditable iteration binding."""

    NORMALIZE = "normalize"
    REPLAY = "replay"
    REFINE = "refine"
    VALIDATE = "validate"
    VERIFY = "verify"
    CLOSE = "close"
    BACKOFF = "backoff"


class IterationOutcome(str, Enum):
    """Per-iteration outcome; never confuses transport success with closure."""

    CLOSED = "closed"
    STILL_OPEN = "still_open"
    UNKNOWN = "unknown"
    BACKED_OFF = "backed_off"
    CANDIDATE_REJECTED = "candidate_rejected"
    NO_CANDIDATE = "no_candidate"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CegisStopReason(str, Enum):
    """Terminal stop reasons for a bounded counterexample-guided run."""

    CLOSED = "closed"
    NO_ADMISSIBLE_CANDIDATE = "no_admissible_candidate"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    REFINEMENT_DEPTH_EXHAUSTED = "refinement_depth_exhausted"
    CANDIDATE_BUDGET_EXHAUSTED = "candidate_budget_exhausted"
    UNCHANGED_WITNESS_BACKOFF = "unchanged_witness_backoff"
    IDENTICAL_FAILURE_TERMINATED = "identical_failure_terminated"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_DISAGREEMENT = "verifier_disagreement"
    BOUND_CHANGED = "bound_changed"
    CANCELLED = "cancelled"
    OPEN_CONTINUED_FAILURE = "open_continued_failure"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise CegisValidationError(f"{field_name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise CegisValidationError(f"{field_name} is required")
    if "\x00" in result:
        raise CegisValidationError(f"{field_name} must not contain NUL bytes")
    return result


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        values = value
    else:
        raise CegisValidationError("expected a sequence of strings")
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _positive(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CegisValidationError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _non_negative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CegisValidationError(f"{name} must be a non-negative integer")
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CegisValidationError(f"{field_name} must be an object")
    return value


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
    raise CegisValidationError(
        "cancelled must be a boolean, predicate, event, token, or None"
    )


def _public_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({item.value for item in kind}))
        raise CegisValidationError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _counterexample(
    value: FormalCounterexample | Mapping[str, Any],
) -> FormalCounterexample:
    if isinstance(value, FormalCounterexample):
        return value
    if not isinstance(value, Mapping):
        raise CegisValidationError(
            "counterexample must be FormalCounterexample or a mapping"
        )
    try:
        return normalize_counterexample(value)
    except CounterexampleValidationError as exc:
        raise CegisValidationError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Budget / candidate / binding contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CegisBudget:
    """Finite resource and iteration budget bound into every loop step."""

    SCHEMA: ClassVar[str] = CEGIS_BUDGET_SCHEMA

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_candidates_per_iteration: int = DEFAULT_MAX_CANDIDATES_PER_ITERATION
    max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES
    base_backoff_seconds: int = DEFAULT_BASE_BACKOFF_SECONDS
    max_backoff_seconds: int = DEFAULT_MAX_BACKOFF_SECONDS
    finite_bounds: Mapping[str, Any] = field(default_factory=dict)
    iterations_used: int = 0
    candidates_tried: int = 0
    identical_failure_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_iterations",
            min(
                ABSOLUTE_MAX_ITERATIONS,
                _positive(self.max_iterations, "max_iterations"),
            ),
        )
        object.__setattr__(
            self,
            "max_candidates_per_iteration",
            min(
                ABSOLUTE_MAX_CANDIDATES_PER_ITERATION,
                _positive(
                    self.max_candidates_per_iteration,
                    "max_candidates_per_iteration",
                ),
            ),
        )
        object.__setattr__(
            self,
            "max_identical_failures",
            _positive(self.max_identical_failures, "max_identical_failures"),
        )
        object.__setattr__(
            self,
            "base_backoff_seconds",
            _positive(self.base_backoff_seconds, "base_backoff_seconds"),
        )
        object.__setattr__(
            self,
            "max_backoff_seconds",
            _positive(self.max_backoff_seconds, "max_backoff_seconds"),
        )
        if self.base_backoff_seconds > self.max_backoff_seconds:
            raise CegisValidationError(
                "base_backoff_seconds cannot exceed max_backoff_seconds"
            )
        bounds = dict(self.finite_bounds or {})
        object.__setattr__(self, "finite_bounds", bounds)
        object.__setattr__(
            self, "iterations_used", _non_negative(self.iterations_used, "iterations_used")
        )
        object.__setattr__(
            self,
            "candidates_tried",
            _non_negative(self.candidates_tried, "candidates_tried"),
        )
        object.__setattr__(
            self,
            "identical_failure_count",
            _non_negative(
                self.identical_failure_count, "identical_failure_count"
            ),
        )

    @property
    def bound_digest(self) -> str:
        return _bound_digest(self.finite_bounds)

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iterations_used)

    def with_usage(
        self,
        *,
        iterations_used: int | None = None,
        candidates_tried: int | None = None,
        identical_failure_count: int | None = None,
    ) -> "CegisBudget":
        return CegisBudget(
            max_iterations=self.max_iterations,
            max_candidates_per_iteration=self.max_candidates_per_iteration,
            max_identical_failures=self.max_identical_failures,
            base_backoff_seconds=self.base_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            finite_bounds=dict(self.finite_bounds),
            iterations_used=(
                self.iterations_used
                if iterations_used is None
                else iterations_used
            ),
            candidates_tried=(
                self.candidates_tried
                if candidates_tried is None
                else candidates_tried
            ),
            identical_failure_count=(
                self.identical_failure_count
                if identical_failure_count is None
                else identical_failure_count
            ),
        )

    def backoff_seconds(self, attempt: int) -> int:
        exponent = min(max(0, attempt), 30)
        return min(
            self.max_backoff_seconds,
            self.base_backoff_seconds * (2**exponent),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CEGIS_BUDGET_SCHEMA,
            "max_iterations": self.max_iterations,
            "max_candidates_per_iteration": self.max_candidates_per_iteration,
            "max_identical_failures": self.max_identical_failures,
            "base_backoff_seconds": self.base_backoff_seconds,
            "max_backoff_seconds": self.max_backoff_seconds,
            "finite_bounds": dict(self.finite_bounds),
            "bound_digest": self.bound_digest,
            "iterations_used": self.iterations_used,
            "candidates_tried": self.candidates_tried,
            "identical_failure_count": self.identical_failure_count,
            "remaining_iterations": self.remaining_iterations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CegisBudget":
        value = dict(payload or {})
        if value.get("schema") not in {None, CEGIS_BUDGET_SCHEMA}:
            raise CegisValidationError("unsupported cegis budget schema")
        return cls(
            max_iterations=value.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            max_candidates_per_iteration=value.get(
                "max_candidates_per_iteration",
                DEFAULT_MAX_CANDIDATES_PER_ITERATION,
            ),
            max_identical_failures=value.get(
                "max_identical_failures", DEFAULT_MAX_IDENTICAL_FAILURES
            ),
            base_backoff_seconds=value.get(
                "base_backoff_seconds", DEFAULT_BASE_BACKOFF_SECONDS
            ),
            max_backoff_seconds=value.get(
                "max_backoff_seconds", DEFAULT_MAX_BACKOFF_SECONDS
            ),
            finite_bounds=dict(value.get("finite_bounds") or {}),
            iterations_used=value.get("iterations_used", 0),
            candidates_tried=value.get("candidates_tried", 0),
            identical_failure_count=value.get("identical_failure_count", 0),
        )


@dataclass(frozen=True)
class RefinementCandidate:
    """One independently validated proof-gap or repair candidate."""

    SCHEMA: ClassVar[str] = CEGIS_CANDIDATE_SCHEMA

    candidate_id: str
    kind: CandidateKind
    goal_id: str
    repaired_tree_id: str
    repaired_plan_id: str = ""
    statement: str = ""
    addresses_witness: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)
    validation_status: CandidateValidationStatus = (
        CandidateValidationStatus.SKIPPED
    )
    validation_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _text(self.candidate_id, field_name="candidate_id"),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, CandidateKind, "kind")
        )
        object.__setattr__(
            self, "goal_id", _text(self.goal_id, field_name="goal_id")
        )
        object.__setattr__(
            self,
            "repaired_tree_id",
            _text(self.repaired_tree_id, field_name="repaired_tree_id"),
        )
        object.__setattr__(
            self,
            "repaired_plan_id",
            str(self.repaired_plan_id or "").strip(),
        )
        object.__setattr__(
            self, "statement", str(self.statement or "").strip()
        )
        object.__setattr__(
            self, "addresses_witness", bool(self.addresses_witness)
        )
        object.__setattr__(
            self, "parameters", _public_mapping(dict(self.parameters or {}))
        )
        object.__setattr__(
            self,
            "validation_status",
            _enum(
                self.validation_status,
                CandidateValidationStatus,
                "validation_status",
            ),
        )
        object.__setattr__(
            self,
            "validation_reason",
            str(self.validation_reason or "").strip(),
        )

    @property
    def semantic_id(self) -> str:
        return content_identity(self.to_dict(include_schema=False))

    @property
    def admissible(self) -> bool:
        return (
            self.validation_status is CandidateValidationStatus.VALID
            and self.addresses_witness
        )

    def with_validation(
        self,
        status: CandidateValidationStatus | str,
        *,
        reason: str = "",
    ) -> "RefinementCandidate":
        return RefinementCandidate(
            candidate_id=self.candidate_id,
            kind=self.kind,
            goal_id=self.goal_id,
            repaired_tree_id=self.repaired_tree_id,
            repaired_plan_id=self.repaired_plan_id,
            statement=self.statement,
            addresses_witness=self.addresses_witness,
            parameters=dict(self.parameters),
            validation_status=status,
            validation_reason=reason,
        )

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        payload = {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "goal_id": self.goal_id,
            "repaired_tree_id": self.repaired_tree_id,
            "repaired_plan_id": self.repaired_plan_id,
            "statement": self.statement,
            "addresses_witness": self.addresses_witness,
            "parameters": dict(self.parameters),
            "validation_status": self.validation_status.value,
            "validation_reason": self.validation_reason,
            "semantic_id": self.semantic_id if include_schema else None,
            "admissible": self.admissible,
        }
        if not include_schema:
            payload.pop("semantic_id", None)
            payload.pop("admissible", None)
            return payload
        payload["schema"] = CEGIS_CANDIDATE_SCHEMA
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementCandidate":
        value = _mapping(payload, field_name="candidate")
        if value.get("schema") not in {None, CEGIS_CANDIDATE_SCHEMA}:
            raise CegisValidationError("unsupported refinement candidate schema")
        return cls(
            candidate_id=str(value.get("candidate_id") or value.get("id") or ""),
            kind=value.get("kind", CandidateKind.REPAIR),
            goal_id=str(value.get("goal_id") or value.get("repaired_goal_id") or ""),
            repaired_tree_id=str(
                value.get("repaired_tree_id")
                or value.get("tree_id")
                or value.get("repository_tree_id")
                or ""
            ),
            repaired_plan_id=str(
                value.get("repaired_plan_id") or value.get("plan_id") or ""
            ),
            statement=str(value.get("statement") or value.get("text") or ""),
            addresses_witness=bool(value.get("addresses_witness", True)),
            parameters=dict(value.get("parameters") or {}),
            validation_status=value.get(
                "validation_status", CandidateValidationStatus.SKIPPED
            ),
            validation_reason=str(value.get("validation_reason") or ""),
        )


@dataclass(frozen=True)
class IterationBinding:
    """Exact audit binding for one CEGIS iteration.

    Acceptance requires every iteration to bind prior witness, candidate,
    repaired tree/goal, exact verifier, budget, and result.
    """

    SCHEMA: ClassVar[str] = CEGIS_ITERATION_BINDING_SCHEMA

    iteration_index: int
    prior_witness_id: str
    candidate_id: str
    repaired_tree_id: str
    repaired_goal_id: str
    exact_verifier_id: str
    budget: CegisBudget
    result_status: IterationOutcome
    property_id: str = ""
    assumption_ids: tuple[str, ...] = ()
    bound_digest: str = ""
    policy_id: str = ""
    repaired_plan_id: str = ""
    verifier_receipt_id: str = ""
    closure_status: WitnessClosureStatus = WitnessClosureStatus.OPEN
    reason_code: str = ""
    phase: IterationPhase = IterationPhase.VERIFY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iteration_index",
            _non_negative(self.iteration_index, "iteration_index"),
        )
        object.__setattr__(
            self,
            "prior_witness_id",
            _text(self.prior_witness_id, field_name="prior_witness_id"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            str(self.candidate_id or "").strip(),
        )
        object.__setattr__(
            self,
            "repaired_tree_id",
            _text(self.repaired_tree_id, field_name="repaired_tree_id"),
        )
        object.__setattr__(
            self,
            "repaired_goal_id",
            _text(self.repaired_goal_id, field_name="repaired_goal_id"),
        )
        object.__setattr__(
            self,
            "exact_verifier_id",
            _text(self.exact_verifier_id, field_name="exact_verifier_id"),
        )
        if not isinstance(self.budget, CegisBudget):
            raise CegisValidationError("budget must be CegisBudget")
        object.__setattr__(
            self,
            "result_status",
            _enum(self.result_status, IterationOutcome, "result_status"),
        )
        object.__setattr__(
            self, "property_id", str(self.property_id or "").strip()
        )
        object.__setattr__(self, "assumption_ids", _strings(self.assumption_ids))
        object.__setattr__(
            self,
            "bound_digest",
            str(self.bound_digest or self.budget.bound_digest or "").strip(),
        )
        object.__setattr__(self, "policy_id", str(self.policy_id or "").strip())
        object.__setattr__(
            self, "repaired_plan_id", str(self.repaired_plan_id or "").strip()
        )
        object.__setattr__(
            self,
            "verifier_receipt_id",
            str(self.verifier_receipt_id or "").strip(),
        )
        object.__setattr__(
            self,
            "closure_status",
            _enum(self.closure_status, WitnessClosureStatus, "closure_status"),
        )
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "").strip()
        )
        object.__setattr__(
            self, "phase", _enum(self.phase, IterationPhase, "phase")
        )

    @property
    def binding_id(self) -> str:
        return content_identity(self.to_dict(include_schema=False))

    @property
    def closed(self) -> bool:
        return (
            self.result_status is IterationOutcome.CLOSED
            and self.closure_status is WitnessClosureStatus.CLOSED
        )

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        payload = {
            "iteration_index": self.iteration_index,
            "prior_witness_id": self.prior_witness_id,
            "candidate_id": self.candidate_id,
            "repaired_tree_id": self.repaired_tree_id,
            "repaired_goal_id": self.repaired_goal_id,
            "exact_verifier_id": self.exact_verifier_id,
            "budget": self.budget.to_dict(),
            "result_status": self.result_status.value,
            "property_id": self.property_id,
            "assumption_ids": list(self.assumption_ids),
            "bound_digest": self.bound_digest,
            "policy_id": self.policy_id,
            "repaired_plan_id": self.repaired_plan_id,
            "verifier_receipt_id": self.verifier_receipt_id,
            "closure_status": self.closure_status.value,
            "reason_code": self.reason_code,
            "phase": self.phase.value,
            "closed": self.closed,
        }
        if include_schema:
            payload["schema"] = CEGIS_ITERATION_BINDING_SCHEMA
            payload["binding_id"] = self.binding_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IterationBinding":
        value = _mapping(payload, field_name="iteration_binding")
        if value.get("schema") not in {None, CEGIS_ITERATION_BINDING_SCHEMA}:
            raise CegisValidationError("unsupported iteration binding schema")
        return cls(
            iteration_index=value.get("iteration_index", 0),
            prior_witness_id=str(value.get("prior_witness_id") or ""),
            candidate_id=str(value.get("candidate_id") or ""),
            repaired_tree_id=str(value.get("repaired_tree_id") or ""),
            repaired_goal_id=str(value.get("repaired_goal_id") or ""),
            exact_verifier_id=str(value.get("exact_verifier_id") or ""),
            budget=CegisBudget.from_dict(value.get("budget")),
            result_status=value.get("result_status", IterationOutcome.STILL_OPEN),
            property_id=str(value.get("property_id") or ""),
            assumption_ids=tuple(value.get("assumption_ids") or ()),
            bound_digest=str(value.get("bound_digest") or ""),
            policy_id=str(value.get("policy_id") or ""),
            repaired_plan_id=str(value.get("repaired_plan_id") or ""),
            verifier_receipt_id=str(value.get("verifier_receipt_id") or ""),
            closure_status=value.get(
                "closure_status", WitnessClosureStatus.OPEN
            ),
            reason_code=str(value.get("reason_code") or ""),
            phase=value.get("phase", IterationPhase.VERIFY),
        )


@dataclass(frozen=True)
class CegisIteration:
    """Full auditable record of one loop turn."""

    SCHEMA: ClassVar[str] = CEGIS_ITERATION_SCHEMA

    binding: IterationBinding
    candidate: RefinementCandidate | None = None
    closure: VerifierBackedRepairClosure | None = None
    post_witness_id: str = ""
    backoff_seconds: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.binding, IterationBinding):
            raise CegisValidationError("binding must be IterationBinding")
        if self.candidate is not None and not isinstance(
            self.candidate, RefinementCandidate
        ):
            raise CegisValidationError(
                "candidate must be RefinementCandidate or None"
            )
        if self.closure is not None and not isinstance(
            self.closure, VerifierBackedRepairClosure
        ):
            raise CegisValidationError(
                "closure must be VerifierBackedRepairClosure or None"
            )
        object.__setattr__(
            self, "post_witness_id", str(self.post_witness_id or "").strip()
        )
        object.__setattr__(
            self, "backoff_seconds", _non_negative(self.backoff_seconds, "backoff_seconds")
        )
        object.__setattr__(self, "notes", _strings(self.notes))

    @property
    def closed(self) -> bool:
        return self.binding.closed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CEGIS_ITERATION_SCHEMA,
            "binding": self.binding.to_dict(),
            "candidate": (
                None if self.candidate is None else self.candidate.to_dict()
            ),
            "closure": None if self.closure is None else self.closure.to_dict(),
            "post_witness_id": self.post_witness_id,
            "backoff_seconds": self.backoff_seconds,
            "notes": list(self.notes),
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CegisIteration":
        value = _mapping(payload, field_name="cegis_iteration")
        if value.get("schema") not in {None, CEGIS_ITERATION_SCHEMA}:
            raise CegisValidationError("unsupported cegis iteration schema")
        candidate_payload = value.get("candidate")
        closure_payload = value.get("closure")
        return cls(
            binding=IterationBinding.from_dict(value.get("binding") or {}),
            candidate=(
                None
                if candidate_payload is None
                else RefinementCandidate.from_dict(candidate_payload)
            ),
            closure=(
                None
                if closure_payload is None
                else VerifierBackedRepairClosure.from_dict(closure_payload)
            ),
            post_witness_id=str(value.get("post_witness_id") or ""),
            backoff_seconds=value.get("backoff_seconds", 0),
            notes=tuple(value.get("notes") or ()),
        )


@dataclass(frozen=True)
class CegisLoopResult:
    """Terminal auditable outcome of a bounded CEGIS/CEGAR run."""

    SCHEMA: ClassVar[str] = CEGIS_LOOP_RESULT_SCHEMA
    INTERFACE: ClassVar[str] = COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE

    stop_reason: CegisStopReason
    initial_witness_id: str
    final_witness_id: str
    exact_verifier_id: str
    property_id: str
    iterations: tuple[CegisIteration, ...]
    budget: CegisBudget
    open_counterexamples: int
    closed: bool
    closure: VerifierBackedRepairClosure | None = None
    selected_candidate: RefinementCandidate | None = None
    repository_tree_id: str = ""
    policy_id: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stop_reason",
            _enum(self.stop_reason, CegisStopReason, "stop_reason"),
        )
        object.__setattr__(
            self,
            "initial_witness_id",
            _text(self.initial_witness_id, field_name="initial_witness_id"),
        )
        object.__setattr__(
            self,
            "final_witness_id",
            str(self.final_witness_id or "").strip(),
        )
        object.__setattr__(
            self,
            "exact_verifier_id",
            _text(self.exact_verifier_id, field_name="exact_verifier_id"),
        )
        object.__setattr__(
            self, "property_id", str(self.property_id or "").strip()
        )
        records = tuple(self.iterations or ())
        for item in records:
            if not isinstance(item, CegisIteration):
                raise CegisValidationError(
                    "iterations must contain CegisIteration records"
                )
        object.__setattr__(self, "iterations", records)
        if not isinstance(self.budget, CegisBudget):
            raise CegisValidationError("budget must be CegisBudget")
        object.__setattr__(
            self,
            "open_counterexamples",
            _non_negative(self.open_counterexamples, "open_counterexamples"),
        )
        object.__setattr__(self, "closed", bool(self.closed))
        if self.closed:
            if self.open_counterexamples != 0:
                raise CegisValidationError(
                    "closed loops must report zero open counterexamples"
                )
            if self.stop_reason is not CegisStopReason.CLOSED:
                raise CegisValidationError(
                    "closed loops must use stop_reason=closed"
                )
            if self.closure is None or not self.closure.closed:
                raise CegisValidationError(
                    "closed loops require a closed verifier-backed repair closure"
                )
        else:
            if self.open_counterexamples < 1 and self.stop_reason not in {
                CegisStopReason.CANCELLED,
            }:
                # Honest continued failure and terminal non-success keep the
                # witness open unless the run was cancelled before any work.
                if self.stop_reason is not CegisStopReason.CANCELLED:
                    raise CegisValidationError(
                        "open loops must keep a non-zero open-counterexample count"
                    )
        object.__setattr__(
            self, "repository_tree_id", str(self.repository_tree_id or "").strip()
        )
        object.__setattr__(self, "policy_id", str(self.policy_id or "").strip())
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "").strip()
        )

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CEGIS_LOOP_RESULT_SCHEMA,
            "interface": COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE,
            "version": COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_VERSION,
            "stop_reason": self.stop_reason.value,
            "initial_witness_id": self.initial_witness_id,
            "final_witness_id": self.final_witness_id,
            "exact_verifier_id": self.exact_verifier_id,
            "property_id": self.property_id,
            "iterations": [item.to_dict() for item in self.iterations],
            "budget": self.budget.to_dict(),
            "open_counterexamples": self.open_counterexamples,
            "closed": self.closed,
            "closure": None if self.closure is None else self.closure.to_dict(),
            "selected_candidate": (
                None
                if self.selected_candidate is None
                else self.selected_candidate.to_dict()
            ),
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "reason_code": self.reason_code,
            "iteration_count": self.iteration_count,
            "verifier_backed_repair_closure_schema": (
                VERIFIER_BACKED_REPAIR_CLOSURE_SCHEMA
            ),
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CegisLoopResult":
        value = _mapping(payload, field_name="cegis_loop_result")
        if value.get("schema") not in {None, CEGIS_LOOP_RESULT_SCHEMA}:
            raise CegisValidationError("unsupported cegis loop result schema")
        iterations = tuple(
            CegisIteration.from_dict(item)
            for item in (value.get("iterations") or ())
        )
        closure_payload = value.get("closure")
        candidate_payload = value.get("selected_candidate")
        return cls(
            stop_reason=value.get("stop_reason", CegisStopReason.OPEN_CONTINUED_FAILURE),
            initial_witness_id=str(value.get("initial_witness_id") or ""),
            final_witness_id=str(value.get("final_witness_id") or ""),
            exact_verifier_id=str(value.get("exact_verifier_id") or ""),
            property_id=str(value.get("property_id") or ""),
            iterations=iterations,
            budget=CegisBudget.from_dict(value.get("budget")),
            open_counterexamples=value.get("open_counterexamples", 1),
            closed=bool(value.get("closed", False)),
            closure=(
                None
                if closure_payload is None
                else VerifierBackedRepairClosure.from_dict(closure_payload)
            ),
            selected_candidate=(
                None
                if candidate_payload is None
                else RefinementCandidate.from_dict(candidate_payload)
            ),
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            policy_id=str(value.get("policy_id") or ""),
            reason_code=str(value.get("reason_code") or ""),
        )


# ---------------------------------------------------------------------------
# Provider protocols (callables)
# ---------------------------------------------------------------------------

# refine(witness, context) -> sequence of candidates / candidate mappings
RefineProvider = Callable[
    [FormalCounterexample, Mapping[str, Any]],
    Sequence[RefinementCandidate | Mapping[str, Any]],
]
# validate(candidate, context) -> status or (status, reason) or mapping
ValidateProvider = Callable[
    [RefinementCandidate, Mapping[str, Any]],
    Any,
]
# verify(binding_dict) -> VerifierClosureReceipt | mapping | None
VerifyProvider = Callable[[Mapping[str, Any]], Any]
# replay(counterexample, context) -> counterexample | mapping | None
ReplayProvider = Callable[
    [FormalCounterexample, Mapping[str, Any]],
    FormalCounterexample | Mapping[str, Any] | None,
]


def _default_refine(
    counterexample: FormalCounterexample,
    context: Mapping[str, Any],
) -> tuple[RefinementCandidate, ...]:
    """Deterministic single-candidate refinement when no provider is supplied.

    Emits one repair-class candidate that reuses the witness binding and marks
    the candidate as addressing the witness. Independent validation and the
    exact originating verifier still decide admissibility and closure.
    """

    tree = str(
        context.get("repository_tree_id")
        or (
            counterexample.bindings.tree_ids[0]
            if counterexample.bindings.tree_ids
            else "tree:unspecified"
        )
    )
    goal = str(
        context.get("goal_id")
        or (
            counterexample.bindings.obligation_ids[0]
            if counterexample.bindings.obligation_ids
            else counterexample.violated_property
            or "goal:unspecified"
        )
    )
    plan = str(
        context.get("repaired_plan_id")
        or (
            counterexample.bindings.plan_ids[0]
            if counterexample.bindings.plan_ids
            else ""
        )
    )
    kind = CandidateKind.REPAIR
    repair_classes = tuple(counterexample.repair_classes or ())
    if repair_classes:
        name = str(getattr(repair_classes[0], "value", repair_classes[0]))
        if "invariant" in name:
            kind = CandidateKind.INVARIANT
        elif "contract" in name:
            kind = CandidateKind.CONTRACT
        elif "premise" in name:
            kind = CandidateKind.PREMISE
        elif "resource" in name or "bound" in name:
            kind = CandidateKind.BOUND_ADJUSTMENT
    candidate_id = content_identity(
        {
            "witness": counterexample.semantic_id,
            "tree": tree,
            "goal": goal,
            "kind": kind.value,
            "depth": context.get("iteration_index", 0),
        }
    )
    return (
        RefinementCandidate(
            candidate_id=f"candidate:{candidate_id[-16:]}",
            kind=kind,
            goal_id=goal,
            repaired_tree_id=tree,
            repaired_plan_id=plan or f"plan:refined:{candidate_id[-12:]}",
            statement=(
                f"Refine against witness {counterexample.semantic_id[:24]}"
            ),
            addresses_witness=True,
            parameters={
                "source_witness_id": counterexample.semantic_id,
                "repair_classes": [
                    str(getattr(item, "value", item)) for item in repair_classes
                ],
            },
        ),
    )


def _default_validate(
    candidate: RefinementCandidate,
    context: Mapping[str, Any],
) -> tuple[CandidateValidationStatus, str]:
    del context
    if not candidate.addresses_witness:
        return CandidateValidationStatus.INVALID, "does_not_address_witness"
    if not candidate.repaired_tree_id or not candidate.goal_id:
        return CandidateValidationStatus.INVALID, "missing_tree_or_goal"
    return CandidateValidationStatus.VALID, "independent_validation_passed"


def _normalize_validation(raw: Any) -> tuple[CandidateValidationStatus, str]:
    if isinstance(raw, CandidateValidationStatus):
        return raw, ""
    if isinstance(raw, tuple) and raw:
        status = _enum(raw[0], CandidateValidationStatus, "validation_status")
        reason = str(raw[1] if len(raw) > 1 else "")
        return status, reason
    if isinstance(raw, Mapping):
        status = _enum(
            raw.get("status") or raw.get("validation_status") or "valid",
            CandidateValidationStatus,
            "validation_status",
        )
        reason = str(raw.get("reason") or raw.get("validation_reason") or "")
        return status, reason
    if isinstance(raw, bool):
        return (
            (
                CandidateValidationStatus.VALID
                if raw
                else CandidateValidationStatus.INVALID
            ),
            "boolean_validation",
        )
    if raw is None:
        return CandidateValidationStatus.UNAVAILABLE, "no_validation_result"
    return CandidateValidationStatus.VALID, str(raw)


def _originating_verifier_id(counterexample: FormalCounterexample) -> str:
    if counterexample.bindings.provider_ids:
        return counterexample.bindings.provider_ids[0]
    return "tool:unspecified"


def _property_id(counterexample: FormalCounterexample) -> str:
    property_id = str(counterexample.violated_property or "").strip()
    if property_id:
        return property_id
    if counterexample.bindings.obligation_ids:
        return counterexample.bindings.obligation_ids[0]
    return ""


def _policy_id(
    counterexample: FormalCounterexample,
    *,
    explicit: str = "",
) -> str:
    policy = str(explicit or "").strip()
    if policy:
        return policy
    policy = str(counterexample.observation_policy_id or "").strip()
    if policy:
        return policy
    if counterexample.bindings.policy_ids:
        return counterexample.bindings.policy_ids[0]
    return ""


def _assumption_ids(counterexample: FormalCounterexample) -> tuple[str, ...]:
    return _strings(
        tuple(counterexample.assumption_ids)
        + tuple(counterexample.bindings.assumption_ids)
    )


def _tree_id(
    counterexample: FormalCounterexample,
    *,
    explicit: str = "",
    candidate: RefinementCandidate | None = None,
) -> str:
    if candidate is not None and candidate.repaired_tree_id:
        return candidate.repaired_tree_id
    tree = str(explicit or "").strip()
    if tree:
        return tree
    if counterexample.bindings.tree_ids:
        return counterexample.bindings.tree_ids[0]
    return "tree:unspecified"


def _stop_reason_for_closure(
    closure: VerifierBackedRepairClosure,
) -> CegisStopReason | None:
    """Map a non-closed verifier outcome onto a terminal or continuing reason."""

    reason = closure.reason_code
    if closure.status is WitnessClosureStatus.CLOSED:
        return CegisStopReason.CLOSED
    if reason == "verifier_unavailable":
        return CegisStopReason.VERIFIER_UNAVAILABLE
    if reason == "verifier_timeout":
        return CegisStopReason.VERIFIER_TIMEOUT
    if reason == "verifier_disagreement":
        return CegisStopReason.VERIFIER_DISAGREEMENT
    if reason.startswith("binding_mismatch:") and "bound" in reason:
        return CegisStopReason.BOUND_CHANGED
    if reason == "stale_receipt":
        return None  # remain open and continue when budget allows
    return None


def _outcome_for_closure(
    closure: VerifierBackedRepairClosure,
) -> IterationOutcome:
    if closure.status is WitnessClosureStatus.CLOSED:
        return IterationOutcome.CLOSED
    if closure.status is WitnessClosureStatus.UNKNOWN:
        return IterationOutcome.UNKNOWN
    return IterationOutcome.STILL_OPEN


# ---------------------------------------------------------------------------
# Tactician
# ---------------------------------------------------------------------------


class CounterexampleGuidedTactician:
    """Bounded CEGIS/CEGAR orchestrator for verifier-backed proof development.

    Call :meth:`run` with a verified counterexample. Only a fresh matching
    receipt from the exact originating verifier may set ``closed=True``.
    """

    interface: ClassVar[str] = COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE
    version: ClassVar[str] = COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_VERSION
    schema: ClassVar[str] = COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_SCHEMA

    def __init__(
        self,
        *,
        refine: RefineProvider | None = None,
        validate: ValidateProvider | None = None,
        verify: VerifyProvider | None = None,
        replay: ReplayProvider | None = None,
        budget: CegisBudget | Mapping[str, Any] | None = None,
        verifier_available: bool | None = None,
    ) -> None:
        self.refine = refine or _default_refine
        self.validate = validate or _default_validate
        self.verify = verify
        self.replay = replay
        if budget is None:
            self.budget = CegisBudget()
        elif isinstance(budget, CegisBudget):
            self.budget = budget
        elif isinstance(budget, Mapping):
            self.budget = CegisBudget.from_dict(budget)
        else:
            raise CegisValidationError("budget must be CegisBudget or a mapping")
        self.verifier_available = verifier_available
        self._seen_candidate_ids: set[str] = set()

    def reset_history(self) -> None:
        self._seen_candidate_ids.clear()

    def run(
        self,
        counterexample: FormalCounterexample | Mapping[str, Any],
        *,
        repository_tree_id: str = "",
        goal_id: str = "",
        policy_id: str = "",
        exact_verifier_id: str = "",
        repaired_plan_id: str = "",
        previous_witness_id: str | None = None,
        budget: CegisBudget | Mapping[str, Any] | None = None,
        cancelled: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> CegisLoopResult:
        """Execute one bounded counterexample-guided synthesis loop."""

        witness = _counterexample(counterexample)
        active_budget = self._resolve_budget(budget)
        verifier_id = str(exact_verifier_id or "").strip() or _originating_verifier_id(
            witness
        )
        property_id = _property_id(witness)
        policy = _policy_id(witness, explicit=policy_id)
        tree = _tree_id(witness, explicit=repository_tree_id)
        assumptions = _assumption_ids(witness)
        bound_digest = (
            active_budget.bound_digest
            if active_budget.finite_bounds
            else _bound_digest(witness.finite_bounds)
        )
        # Keep finite bounds aligned with the witness when caller omitted them.
        if not active_budget.finite_bounds and witness.finite_bounds:
            active_budget = CegisBudget(
                max_iterations=active_budget.max_iterations,
                max_candidates_per_iteration=(
                    active_budget.max_candidates_per_iteration
                ),
                max_identical_failures=active_budget.max_identical_failures,
                base_backoff_seconds=active_budget.base_backoff_seconds,
                max_backoff_seconds=active_budget.max_backoff_seconds,
                finite_bounds=dict(witness.finite_bounds),
                iterations_used=active_budget.iterations_used,
                candidates_tried=active_budget.candidates_tried,
                identical_failure_count=active_budget.identical_failure_count,
            )
            bound_digest = active_budget.bound_digest

        base_context: dict[str, Any] = {
            "repository_tree_id": tree,
            "goal_id": str(goal_id or property_id or "goal:unspecified"),
            "policy_id": policy,
            "exact_verifier_id": verifier_id,
            "repaired_plan_id": str(repaired_plan_id or "").strip(),
            "property_id": property_id,
            "assumption_ids": assumptions,
            "bound_digest": bound_digest,
            **dict(context or {}),
        }

        if _cancelled(cancelled):
            return self._result(
                stop_reason=CegisStopReason.CANCELLED,
                initial_witness_id=witness.semantic_id,
                final_witness_id=witness.semantic_id,
                exact_verifier_id=verifier_id,
                property_id=property_id,
                iterations=(),
                budget=active_budget,
                open_counterexamples=1,
                closed=False,
                repository_tree_id=tree,
                policy_id=policy,
                reason_code="cancelled_before_start",
            )

        # Optional exact replay of the originating witness before refinement.
        if self.replay is not None:
            try:
                replayed = self.replay(witness, dict(base_context))
            except Exception as exc:  # honest failure — keep open
                return self._result(
                    stop_reason=CegisStopReason.OPEN_CONTINUED_FAILURE,
                    initial_witness_id=witness.semantic_id,
                    final_witness_id=witness.semantic_id,
                    exact_verifier_id=verifier_id,
                    property_id=property_id,
                    iterations=(),
                    budget=active_budget,
                    open_counterexamples=1,
                    closed=False,
                    repository_tree_id=tree,
                    policy_id=policy,
                    reason_code=f"replay_failed:{type(exc).__name__}",
                )
            if replayed is not None:
                witness = _counterexample(replayed)

        iterations: list[CegisIteration] = []
        current = witness
        previous = str(previous_witness_id or "").strip()
        identical_failures = active_budget.identical_failure_count
        candidates_tried = active_budget.candidates_tried
        last_closure: VerifierBackedRepairClosure | None = None
        selected: RefinementCandidate | None = None
        terminal_reason = CegisStopReason.OPEN_CONTINUED_FAILURE
        terminal_code = "budget_or_policy_exhausted"

        for index in range(active_budget.max_iterations):
            if _cancelled(cancelled):
                terminal_reason = CegisStopReason.CANCELLED
                terminal_code = "cancelled"
                break

            used_budget = active_budget.with_usage(
                iterations_used=index + 1,
                candidates_tried=candidates_tried,
                identical_failure_count=identical_failures,
            )

            # Unchanged witness → backoff (preserve formal-replanner semantics).
            if previous and previous == current.semantic_id:
                identical_failures += 1
                used_budget = used_budget.with_usage(
                    identical_failure_count=identical_failures
                )
                seconds = used_budget.backoff_seconds(identical_failures - 1)
                if identical_failures >= used_budget.max_identical_failures:
                    binding = IterationBinding(
                        iteration_index=index,
                        prior_witness_id=current.semantic_id,
                        candidate_id="",
                        repaired_tree_id=tree,
                        repaired_goal_id=str(
                            base_context.get("goal_id") or property_id or "goal:unspecified"
                        ),
                        exact_verifier_id=verifier_id,
                        budget=used_budget,
                        result_status=IterationOutcome.BUDGET_EXHAUSTED,
                        property_id=property_id,
                        assumption_ids=assumptions,
                        bound_digest=bound_digest,
                        policy_id=policy,
                        repaired_plan_id=str(
                            base_context.get("repaired_plan_id") or ""
                        ),
                        closure_status=WitnessClosureStatus.OPEN,
                        reason_code="identical_failure_terminated",
                        phase=IterationPhase.BACKOFF,
                    )
                    iterations.append(
                        CegisIteration(
                            binding=binding,
                            post_witness_id=current.semantic_id,
                            backoff_seconds=0,
                            notes=("identical_failure_terminated",),
                        )
                    )
                    terminal_reason = CegisStopReason.IDENTICAL_FAILURE_TERMINATED
                    terminal_code = "identical_failure_terminated"
                    break

                binding = IterationBinding(
                    iteration_index=index,
                    prior_witness_id=current.semantic_id,
                    candidate_id="",
                    repaired_tree_id=tree,
                    repaired_goal_id=str(
                        base_context.get("goal_id") or property_id or "goal:unspecified"
                    ),
                    exact_verifier_id=verifier_id,
                    budget=used_budget,
                    result_status=IterationOutcome.BACKED_OFF,
                    property_id=property_id,
                    assumption_ids=assumptions,
                    bound_digest=bound_digest,
                    policy_id=policy,
                    repaired_plan_id=str(
                        base_context.get("repaired_plan_id") or ""
                    ),
                    closure_status=WitnessClosureStatus.OPEN,
                    reason_code="unchanged_witness_backoff",
                    phase=IterationPhase.BACKOFF,
                )
                iterations.append(
                    CegisIteration(
                        binding=binding,
                        post_witness_id=current.semantic_id,
                        backoff_seconds=seconds,
                        notes=("unchanged_witness_backoff",),
                    )
                )
                terminal_reason = CegisStopReason.UNCHANGED_WITNESS_BACKOFF
                terminal_code = "unchanged_witness_backoff"
                # One backoff per unchanged observation; caller re-enters with
                # updated evidence. Do not spin the full remaining budget.
                break

            # Generate refinement candidates for this witness.
            refine_context = {
                **base_context,
                "iteration_index": index,
                "prior_witness_id": previous,
                "current_witness_id": current.semantic_id,
            }
            try:
                raw_candidates = self.refine(current, refine_context)
            except Exception as exc:
                binding = IterationBinding(
                    iteration_index=index,
                    prior_witness_id=current.semantic_id,
                    candidate_id="",
                    repaired_tree_id=tree,
                    repaired_goal_id=str(
                        base_context.get("goal_id") or property_id or "goal:unspecified"
                    ),
                    exact_verifier_id=verifier_id,
                    budget=used_budget,
                    result_status=IterationOutcome.NO_CANDIDATE,
                    property_id=property_id,
                    assumption_ids=assumptions,
                    bound_digest=bound_digest,
                    policy_id=policy,
                    closure_status=WitnessClosureStatus.UNKNOWN,
                    reason_code=f"refine_failed:{type(exc).__name__}",
                    phase=IterationPhase.REFINE,
                )
                iterations.append(
                    CegisIteration(
                        binding=binding,
                        post_witness_id=current.semantic_id,
                        notes=("refine_failed",),
                    )
                )
                terminal_reason = CegisStopReason.NO_ADMISSIBLE_CANDIDATE
                terminal_code = f"refine_failed:{type(exc).__name__}"
                break

            candidates = self._materialize_candidates(
                raw_candidates,
                default_tree=tree,
                default_goal=str(
                    base_context.get("goal_id") or property_id or "goal:unspecified"
                ),
                default_plan=str(base_context.get("repaired_plan_id") or ""),
            )
            if not candidates:
                binding = IterationBinding(
                    iteration_index=index,
                    prior_witness_id=current.semantic_id,
                    candidate_id="",
                    repaired_tree_id=tree,
                    repaired_goal_id=str(
                        base_context.get("goal_id") or property_id or "goal:unspecified"
                    ),
                    exact_verifier_id=verifier_id,
                    budget=used_budget,
                    result_status=IterationOutcome.NO_CANDIDATE,
                    property_id=property_id,
                    assumption_ids=assumptions,
                    bound_digest=bound_digest,
                    policy_id=policy,
                    closure_status=WitnessClosureStatus.OPEN,
                    reason_code="no_candidates_generated",
                    phase=IterationPhase.REFINE,
                )
                iterations.append(
                    CegisIteration(
                        binding=binding,
                        post_witness_id=current.semantic_id,
                        notes=("no_candidates_generated",),
                    )
                )
                terminal_reason = CegisStopReason.NO_ADMISSIBLE_CANDIDATE
                terminal_code = "no_candidates_generated"
                break

            admitted_this_round = False
            for candidate in candidates[: used_budget.max_candidates_per_iteration]:
                if _cancelled(cancelled):
                    terminal_reason = CegisStopReason.CANCELLED
                    terminal_code = "cancelled"
                    admitted_this_round = True  # stop outer loop cleanly
                    break
                if candidate.candidate_id in self._seen_candidate_ids:
                    continue
                self._seen_candidate_ids.add(candidate.candidate_id)
                candidates_tried += 1
                used_budget = used_budget.with_usage(
                    candidates_tried=candidates_tried,
                    identical_failure_count=identical_failures,
                )

                validated = self._validate_candidate(candidate, refine_context)
                if not validated.admissible:
                    binding = IterationBinding(
                        iteration_index=index,
                        prior_witness_id=current.semantic_id,
                        candidate_id=validated.candidate_id,
                        repaired_tree_id=validated.repaired_tree_id,
                        repaired_goal_id=validated.goal_id,
                        exact_verifier_id=verifier_id,
                        budget=used_budget,
                        result_status=IterationOutcome.CANDIDATE_REJECTED,
                        property_id=property_id,
                        assumption_ids=assumptions,
                        bound_digest=bound_digest,
                        policy_id=policy,
                        repaired_plan_id=validated.repaired_plan_id,
                        closure_status=WitnessClosureStatus.OPEN,
                        reason_code=(
                            validated.validation_reason
                            or validated.validation_status.value
                        ),
                        phase=IterationPhase.VALIDATE,
                    )
                    iterations.append(
                        CegisIteration(
                            binding=binding,
                            candidate=validated,
                            post_witness_id=current.semantic_id,
                            notes=("candidate_rejected",),
                        )
                    )
                    continue

                # Exact originating-verifier rerun with full binding.
                verify_binding = {
                    "counterexample_id": current.semantic_id,
                    "repository_tree_id": validated.repaired_tree_id,
                    "property_id": property_id,
                    "assumption_ids": list(assumptions),
                    "bound_digest": bound_digest,
                    "tool_id": verifier_id,
                    "policy_id": policy,
                    "repaired_plan_id": validated.repaired_plan_id,
                    "candidate_id": validated.candidate_id,
                    "goal_id": validated.goal_id,
                    "iteration_index": index,
                }
                closure = self._evaluate_closure(
                    verify_binding=verify_binding,
                    structural_addressed=validated.addresses_witness,
                )
                last_closure = closure
                outcome = _outcome_for_closure(closure)
                binding = IterationBinding(
                    iteration_index=index,
                    prior_witness_id=current.semantic_id,
                    candidate_id=validated.candidate_id,
                    repaired_tree_id=validated.repaired_tree_id,
                    repaired_goal_id=validated.goal_id,
                    exact_verifier_id=verifier_id,
                    budget=used_budget,
                    result_status=outcome,
                    property_id=property_id,
                    assumption_ids=assumptions,
                    bound_digest=bound_digest,
                    policy_id=policy,
                    repaired_plan_id=validated.repaired_plan_id,
                    verifier_receipt_id=closure.verifier_receipt_id,
                    closure_status=closure.status,
                    reason_code=closure.reason_code,
                    phase=IterationPhase.CLOSE
                    if closure.closed
                    else IterationPhase.VERIFY,
                )
                iterations.append(
                    CegisIteration(
                        binding=binding,
                        candidate=validated,
                        closure=closure,
                        post_witness_id=(
                            "" if closure.closed else current.semantic_id
                        ),
                        notes=(closure.reason_code,),
                    )
                )

                if closure.closed:
                    selected = validated
                    terminal_reason = CegisStopReason.CLOSED
                    terminal_code = closure.reason_code
                    previous = current.semantic_id
                    return self._result(
                        stop_reason=CegisStopReason.CLOSED,
                        initial_witness_id=witness.semantic_id,
                        final_witness_id=current.semantic_id,
                        exact_verifier_id=verifier_id,
                        property_id=property_id,
                        iterations=tuple(iterations),
                        budget=used_budget,
                        open_counterexamples=0,
                        closed=True,
                        closure=closure,
                        selected_candidate=selected,
                        repository_tree_id=validated.repaired_tree_id,
                        policy_id=policy,
                        reason_code=terminal_code,
                    )

                mapped = _stop_reason_for_closure(closure)
                if mapped in {
                    CegisStopReason.VERIFIER_UNAVAILABLE,
                    CegisStopReason.VERIFIER_TIMEOUT,
                    CegisStopReason.VERIFIER_DISAGREEMENT,
                    CegisStopReason.BOUND_CHANGED,
                }:
                    # Honest non-success: terminate this run open/unknown.
                    selected = validated
                    terminal_reason = mapped
                    terminal_code = closure.reason_code
                    previous = current.semantic_id
                    admitted_this_round = True
                    break

                # Still open: treat as attempted failure on this witness.
                previous = current.semantic_id
                selected = validated
                admitted_this_round = True
                terminal_reason = CegisStopReason.OPEN_CONTINUED_FAILURE
                terminal_code = closure.reason_code or "witness_still_open"
                # Continue outer loop; next turn sees previous == current and
                # will apply unchanged-witness backoff policy.
                break

            if terminal_reason is CegisStopReason.CANCELLED:
                break
            if terminal_reason in {
                CegisStopReason.VERIFIER_UNAVAILABLE,
                CegisStopReason.VERIFIER_TIMEOUT,
                CegisStopReason.VERIFIER_DISAGREEMENT,
                CegisStopReason.BOUND_CHANGED,
            }:
                break
            if not admitted_this_round:
                # All candidates rejected or duplicates.
                terminal_reason = CegisStopReason.NO_ADMISSIBLE_CANDIDATE
                terminal_code = "no_admissible_candidate"
                break
            # If we admitted a still-open attempt, the next outer iteration
            # hits the unchanged-witness branch.

        final_budget = active_budget.with_usage(
            iterations_used=len(iterations),
            candidates_tried=candidates_tried,
            identical_failure_count=identical_failures,
        )
        if (
            terminal_reason is CegisStopReason.OPEN_CONTINUED_FAILURE
            and len(iterations) >= active_budget.max_iterations
        ):
            terminal_reason = CegisStopReason.REFINEMENT_DEPTH_EXHAUSTED
            terminal_code = "refinement_depth_exhausted"

        open_count = 0 if terminal_reason is CegisStopReason.CLOSED else 1
        return self._result(
            stop_reason=terminal_reason,
            initial_witness_id=witness.semantic_id,
            final_witness_id=current.semantic_id,
            exact_verifier_id=verifier_id,
            property_id=property_id,
            iterations=tuple(iterations),
            budget=final_budget,
            open_counterexamples=open_count,
            closed=False,
            closure=last_closure,
            selected_candidate=selected,
            repository_tree_id=tree,
            policy_id=policy,
            reason_code=terminal_code,
        )

    # -- internals ---------------------------------------------------------

    def _resolve_budget(
        self, budget: CegisBudget | Mapping[str, Any] | None
    ) -> CegisBudget:
        if budget is None:
            return self.budget
        if isinstance(budget, CegisBudget):
            return budget
        if isinstance(budget, Mapping):
            return CegisBudget.from_dict(budget)
        raise CegisValidationError("budget must be CegisBudget or a mapping")

    def _materialize_candidates(
        self,
        raw: Sequence[RefinementCandidate | Mapping[str, Any]] | None,
        *,
        default_tree: str,
        default_goal: str,
        default_plan: str,
    ) -> list[RefinementCandidate]:
        if raw is None:
            return []
        if isinstance(raw, (str, bytes, bytearray, memoryview)):
            raise CegisValidationError("refine must return a sequence of candidates")
        if not isinstance(raw, Sequence):
            raise CegisValidationError("refine must return a sequence of candidates")
        result: list[RefinementCandidate] = []
        for item in raw:
            if isinstance(item, RefinementCandidate):
                candidate = item
            elif isinstance(item, Mapping):
                payload = dict(item)
                payload.setdefault("repaired_tree_id", default_tree)
                payload.setdefault("goal_id", default_goal)
                payload.setdefault("repaired_plan_id", default_plan)
                candidate = RefinementCandidate.from_dict(payload)
            else:
                raise CegisValidationError(
                    "each candidate must be RefinementCandidate or a mapping"
                )
            result.append(candidate)
        return result

    def _validate_candidate(
        self,
        candidate: RefinementCandidate,
        context: Mapping[str, Any],
    ) -> RefinementCandidate:
        try:
            raw = self.validate(candidate, context)
        except Exception as exc:
            return candidate.with_validation(
                CandidateValidationStatus.UNAVAILABLE,
                reason=f"validate_failed:{type(exc).__name__}",
            )
        status, reason = _normalize_validation(raw)
        return candidate.with_validation(status, reason=reason)

    def _evaluate_closure(
        self,
        *,
        verify_binding: Mapping[str, Any],
        structural_addressed: bool,
    ) -> VerifierBackedRepairClosure:
        receipt: VerifierClosureReceipt | Mapping[str, Any] | None = None
        available = self.verifier_available
        if self.verify is None:
            return evaluate_verifier_backed_closure(
                counterexample_id=str(verify_binding["counterexample_id"]),
                structural_addressed=structural_addressed,
                repository_tree_id=str(verify_binding.get("repository_tree_id") or ""),
                property_id=str(verify_binding.get("property_id") or ""),
                assumption_ids=tuple(verify_binding.get("assumption_ids") or ()),
                bound_digest=str(verify_binding.get("bound_digest") or ""),
                tool_id=str(verify_binding.get("tool_id") or ""),
                policy_id=str(verify_binding.get("policy_id") or ""),
                repaired_plan_id=str(verify_binding.get("repaired_plan_id") or ""),
                verifier_available=available,
                receipt=None,
            )
        if available is False:
            return evaluate_verifier_backed_closure(
                counterexample_id=str(verify_binding["counterexample_id"]),
                structural_addressed=structural_addressed,
                repository_tree_id=str(verify_binding.get("repository_tree_id") or ""),
                property_id=str(verify_binding.get("property_id") or ""),
                assumption_ids=tuple(verify_binding.get("assumption_ids") or ()),
                bound_digest=str(verify_binding.get("bound_digest") or ""),
                tool_id=str(verify_binding.get("tool_id") or ""),
                policy_id=str(verify_binding.get("policy_id") or ""),
                repaired_plan_id=str(verify_binding.get("repaired_plan_id") or ""),
                verifier_available=False,
                receipt=None,
            )
        try:
            raw = self.verify(dict(verify_binding))
        except Exception:
            return evaluate_verifier_backed_closure(
                counterexample_id=str(verify_binding["counterexample_id"]),
                structural_addressed=structural_addressed,
                repository_tree_id=str(verify_binding.get("repository_tree_id") or ""),
                property_id=str(verify_binding.get("property_id") or ""),
                assumption_ids=tuple(verify_binding.get("assumption_ids") or ()),
                bound_digest=str(verify_binding.get("bound_digest") or ""),
                tool_id=str(verify_binding.get("tool_id") or ""),
                policy_id=str(verify_binding.get("policy_id") or ""),
                repaired_plan_id=str(verify_binding.get("repaired_plan_id") or ""),
                verifier_available=False,
                receipt=None,
            )
        if raw is None:
            receipt = None
        elif isinstance(raw, VerifierClosureReceipt):
            receipt = raw
        elif isinstance(raw, Mapping):
            receipt = raw
        else:
            return evaluate_verifier_backed_closure(
                counterexample_id=str(verify_binding["counterexample_id"]),
                structural_addressed=structural_addressed,
                repository_tree_id=str(verify_binding.get("repository_tree_id") or ""),
                property_id=str(verify_binding.get("property_id") or ""),
                assumption_ids=tuple(verify_binding.get("assumption_ids") or ()),
                bound_digest=str(verify_binding.get("bound_digest") or ""),
                tool_id=str(verify_binding.get("tool_id") or ""),
                policy_id=str(verify_binding.get("policy_id") or ""),
                repaired_plan_id=str(verify_binding.get("repaired_plan_id") or ""),
                verifier_available=False,
                receipt=None,
            )
        return evaluate_verifier_backed_closure(
            counterexample_id=str(verify_binding["counterexample_id"]),
            structural_addressed=structural_addressed,
            repository_tree_id=str(verify_binding.get("repository_tree_id") or ""),
            property_id=str(verify_binding.get("property_id") or ""),
            assumption_ids=tuple(verify_binding.get("assumption_ids") or ()),
            bound_digest=str(verify_binding.get("bound_digest") or ""),
            tool_id=str(verify_binding.get("tool_id") or ""),
            policy_id=str(verify_binding.get("policy_id") or ""),
            repaired_plan_id=str(verify_binding.get("repaired_plan_id") or ""),
            verifier_available=available if available is not None else True,
            receipt=receipt,
        )

    @staticmethod
    def _result(
        *,
        stop_reason: CegisStopReason,
        initial_witness_id: str,
        final_witness_id: str,
        exact_verifier_id: str,
        property_id: str,
        iterations: tuple[CegisIteration, ...] | list[CegisIteration],
        budget: CegisBudget,
        open_counterexamples: int,
        closed: bool,
        closure: VerifierBackedRepairClosure | None = None,
        selected_candidate: RefinementCandidate | None = None,
        repository_tree_id: str = "",
        policy_id: str = "",
        reason_code: str = "",
    ) -> CegisLoopResult:
        return CegisLoopResult(
            stop_reason=stop_reason,
            initial_witness_id=initial_witness_id,
            final_witness_id=final_witness_id,
            exact_verifier_id=exact_verifier_id,
            property_id=property_id,
            iterations=tuple(iterations),
            budget=budget,
            open_counterexamples=open_counterexamples,
            closed=closed,
            closure=closure,
            selected_candidate=selected_candidate,
            repository_tree_id=repository_tree_id,
            policy_id=policy_id,
            reason_code=reason_code,
        )


def run_counterexample_guided_loop(
    counterexample: FormalCounterexample | Mapping[str, Any],
    *,
    refine: RefineProvider | None = None,
    validate: ValidateProvider | None = None,
    verify: VerifyProvider | None = None,
    replay: ReplayProvider | None = None,
    budget: CegisBudget | Mapping[str, Any] | None = None,
    verifier_available: bool | None = None,
    repository_tree_id: str = "",
    goal_id: str = "",
    policy_id: str = "",
    exact_verifier_id: str = "",
    repaired_plan_id: str = "",
    previous_witness_id: str | None = None,
    cancelled: Any = None,
    context: Mapping[str, Any] | None = None,
) -> CegisLoopResult:
    """Functional entry point for ``CounterexampleGuidedProofDevelopment@1``."""

    return CounterexampleGuidedTactician(
        refine=refine,
        validate=validate,
        verify=verify,
        replay=replay,
        budget=budget,
        verifier_available=verifier_available,
    ).run(
        counterexample,
        repository_tree_id=repository_tree_id,
        goal_id=goal_id,
        policy_id=policy_id,
        exact_verifier_id=exact_verifier_id,
        repaired_plan_id=repaired_plan_id,
        previous_witness_id=previous_witness_id,
        cancelled=cancelled,
        context=context,
    )


__all__ = [
    "ABSOLUTE_MAX_CANDIDATES_PER_ITERATION",
    "ABSOLUTE_MAX_ITERATIONS",
    "CEGIS_BUDGET_SCHEMA",
    "CEGIS_CANDIDATE_SCHEMA",
    "CEGIS_ITERATION_BINDING_SCHEMA",
    "CEGIS_ITERATION_SCHEMA",
    "CEGIS_LOOP_RESULT_SCHEMA",
    "COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE",
    "COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_SCHEMA",
    "COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_VERSION",
    "DEFAULT_BASE_BACKOFF_SECONDS",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "DEFAULT_MAX_CANDIDATES_PER_ITERATION",
    "DEFAULT_MAX_IDENTICAL_FAILURES",
    "DEFAULT_MAX_ITERATIONS",
    "CandidateKind",
    "CandidateValidationStatus",
    "CegisBudget",
    "CegisCancelled",
    "CegisIteration",
    "CegisLoopResult",
    "CegisStopReason",
    "CegisValidationError",
    "CounterexampleGuidedTactician",
    "IterationBinding",
    "IterationOutcome",
    "IterationPhase",
    "RefineProvider",
    "RefinementCandidate",
    "ReplayProvider",
    "ValidateProvider",
    "VerifyProvider",
    "run_counterexample_guided_loop",
]
