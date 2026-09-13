"""Bounded campaign refill scoring, curriculum priority, and loop limits.

Refill is a control surface, not an unbounded work generator.  Every trigger
has a hard count, every pass has a task ceiling, and a repeated no-progress
identity is refused instead of being replayed.  Scoring is a deterministic
total order so two supervisors observing the same residual set emit the same
admission decision.

Oscillation, runaway, and nonconvergence controls (DOEP-055) are a binding of
``CampaignRefillController``, not a second refill owner, planner, queue, or
event bus.  Cycling progress identities, repeated plan hashes, a stable
frontier, and generated-task/replan/objective runaway are quarantined as
nonconvergent instead of being replayed.  A worker or model assertion cannot skip these controls or treat an empty queue as completion.  This path never
writes DuckDB.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import content_identity
from ..runtime.learning_checkpoint import CAMPAIGN_DURABILITY_REQUIREMENT_ID


CAMPAIGN_REFILL_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-refill-policy@1"
)
CAMPAIGN_REFILL_POLICY_BINDING: Final = "CampaignRefillPolicy@1"
CAMPAIGN_REFILL_POLICY_INTERFACE: Final = CAMPAIGN_REFILL_POLICY_BINDING
CAMPAIGN_REFILL_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-refill-decision@1"
)
CAMPAIGN_REFILL_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-refill-candidate@1"
)
OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING: Final = (
    "OscillationRunawayNonconvergence@1"
)
OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE: Final = (
    OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING
)
OSCILLATION_RUNAWAY_NONCONVERGENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/oscillation-runaway-nonconvergence@1"
)
AUTOMATIC_BOUNDED_TASK_REFILL_BINDING: Final = "AutomaticBoundedTaskRefill@1"
TASK_SEMANTIC_DEDUPLICATION_BINDING: Final = "TaskSemanticDeduplication@1"
OSCILLATION_RUNAWAY_NONCONVERGENCE_CONSUMES: Final[tuple[str, ...]] = (
    CAMPAIGN_REFILL_POLICY_SCHEMA,
    AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
    TASK_SEMANTIC_DEDUPLICATION_BINDING,
)

# Hard safety ceilings.  Callers may tighten these but must not raise them.
MAX_REFILL_ROUNDS: Final = 8
MAX_NO_PROGRESS_ROUNDS: Final = 3
MAX_CURRICULUM_REPETITIONS: Final = 2
MAX_TASKS_PER_REFILL: Final = 8
MAX_OPEN_WORK: Final = 24
MAX_TRIGGER_FIRINGS: Final = 8
MAX_GENERATED_TASKS: Final = 24
MAX_REPLAN_EPOCHS: Final = 8
MAX_TASKS_PER_OBJECTIVE: Final = 109
MAX_OSCILLATION_WINDOW: Final = 4
MAX_REPEATED_PLAN_HASHES: Final = 2
MAX_STABLE_FRONTIER_ROUNDS: Final = 2
MIN_EVENT_DEBOUNCE_MS: Final = 5_000
MAX_EVENT_DEBOUNCE_MS: Final = 60_000


class CampaignRefillError(ValueError):
    """Malformed refill policy, candidate, or history."""


class RefillTrigger(str, Enum):
    NO_PROGRESS = "no_progress"
    CURRICULUM_GAP = "curriculum_gap"
    PROOF_RESIDUAL = "proof_residual"
    EVALUATION_RESIDUAL = "evaluation_residual"
    VALIDATION_FAILURE = "validation_failure"
    TOKEN_BUDGET = "token_budget"
    ROUND_BOUND = "round_bound"


# Lower rank is admitted first.  Proof and evaluation residuals outrank
# speculative curriculum expansion; no-progress is last and still bounded.
TRIGGER_PRIORITY: Final[Mapping[RefillTrigger, int]] = {
    RefillTrigger.PROOF_RESIDUAL: 0,
    RefillTrigger.EVALUATION_RESIDUAL: 1,
    RefillTrigger.CURRICULUM_GAP: 2,
    RefillTrigger.VALIDATION_FAILURE: 3,
    RefillTrigger.TOKEN_BUDGET: 4,
    RefillTrigger.NO_PROGRESS: 5,
    RefillTrigger.ROUND_BOUND: 6,
}

TRIGGER_BOUNDS: Final[Mapping[RefillTrigger, int]] = {
    RefillTrigger.NO_PROGRESS: MAX_NO_PROGRESS_ROUNDS,
    RefillTrigger.CURRICULUM_GAP: MAX_TRIGGER_FIRINGS,
    RefillTrigger.PROOF_RESIDUAL: MAX_TRIGGER_FIRINGS,
    RefillTrigger.EVALUATION_RESIDUAL: MAX_TRIGGER_FIRINGS,
    RefillTrigger.VALIDATION_FAILURE: 4,
    RefillTrigger.TOKEN_BUDGET: 2,
    RefillTrigger.ROUND_BOUND: 1,
}


class RefillDisposition(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    NO_PROGRESS_BOUNDED = "no_progress_bounded"
    TRIGGER_BOUNDED = "trigger_bounded"
    OPEN_WORK_BOUNDED = "open_work_bounded"
    ROUND_BOUNDED = "round_bounded"
    REPETITION_BOUNDED = "repetition_bounded"
    OSCILLATION_BOUNDED = "oscillation_bounded"
    RUNAWAY_BOUNDED = "runaway_bounded"
    STABLE_FRONTIER = "stable_frontier"
    EVENT_DEBOUNCED = "event_debounced"
    QUARANTINED_NONCONVERGENT = "quarantined_nonconvergent"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not str(value).strip():
        raise CampaignRefillError(f"{name} must be a non-empty string")
    text = str(value).strip()
    if "\x00" in text:
        raise CampaignRefillError(f"{name} must not contain NUL")
    return text


def _optional_text(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _required_text(value, name)


def _required_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignRefillError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise CampaignRefillError(f"{name} is outside its bound")
    return value


def _required_bool(value: Any, name: str, *, required_true: bool = False) -> bool:
    if not isinstance(value, bool):
        raise CampaignRefillError(f"{name} must be a boolean")
    if required_true and value is not True:
        raise CampaignRefillError(f"{name} cannot be disabled")
    return value


def _compact_identities(values: Sequence[Any], name: str) -> tuple[str, ...]:
    identities: list[str] = []
    for item in values:
        if item in (None, ""):
            continue
        identities.append(_required_text(item, name))
    return tuple(identities)


def _objective_id_of(candidate: CampaignRefillCandidate, fallback: str = "") -> str:
    metadata = candidate.metadata
    for key in ("objective_id", "objective id", "goal_id", "goal id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _required_text(value, "objective_id")
    return fallback


@dataclass(frozen=True)
class CampaignRefillPolicy:
    """Closed numeric bounds for one campaign refill controller."""

    max_refill_rounds: int = MAX_REFILL_ROUNDS
    max_no_progress_rounds: int = MAX_NO_PROGRESS_ROUNDS
    max_curriculum_repetitions: int = MAX_CURRICULUM_REPETITIONS
    max_tasks_per_refill: int = MAX_TASKS_PER_REFILL
    max_open_work: int = MAX_OPEN_WORK
    cooldown_ms: int = 0
    max_generated_tasks: int = MAX_GENERATED_TASKS
    max_replan_epochs: int = MAX_REPLAN_EPOCHS
    max_tasks_per_objective: int = MAX_TASKS_PER_OBJECTIVE
    oscillation_window: int = MAX_OSCILLATION_WINDOW
    max_repeated_plan_hashes: int = MAX_REPEATED_PLAN_HASHES
    max_stable_frontier_rounds: int = MAX_STABLE_FRONTIER_ROUNDS
    event_debounce_ms: int = MIN_EVENT_DEBOUNCE_MS
    oscillation_detection: bool = True
    runaway_detection: bool = True
    nonconvergence_quarantine: bool = True
    repeated_plan_hash_detection: bool = True
    stable_frontier_detection: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_refill_rounds",
            _required_int(
                self.max_refill_rounds,
                "max_refill_rounds",
                minimum=1,
                maximum=MAX_REFILL_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "max_no_progress_rounds",
            _required_int(
                self.max_no_progress_rounds,
                "max_no_progress_rounds",
                minimum=1,
                maximum=MAX_NO_PROGRESS_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "max_curriculum_repetitions",
            _required_int(
                self.max_curriculum_repetitions,
                "max_curriculum_repetitions",
                minimum=1,
                maximum=MAX_CURRICULUM_REPETITIONS,
            ),
        )
        object.__setattr__(
            self,
            "max_tasks_per_refill",
            _required_int(
                self.max_tasks_per_refill,
                "max_tasks_per_refill",
                minimum=1,
                maximum=MAX_TASKS_PER_REFILL,
            ),
        )
        object.__setattr__(
            self,
            "max_open_work",
            _required_int(self.max_open_work, "max_open_work", minimum=1, maximum=MAX_OPEN_WORK)
        )
        object.__setattr__(
            self,
            "cooldown_ms",
            _required_int(self.cooldown_ms, "cooldown_ms", minimum=0),
        )
        object.__setattr__(
            self,
            "max_generated_tasks",
            _required_int(
                self.max_generated_tasks,
                "max_generated_tasks",
                minimum=1,
                maximum=MAX_GENERATED_TASKS,
            ),
        )
        object.__setattr__(
            self,
            "max_replan_epochs",
            _required_int(
                self.max_replan_epochs,
                "max_replan_epochs",
                minimum=1,
                maximum=MAX_REPLAN_EPOCHS,
            ),
        )
        object.__setattr__(
            self,
            "max_tasks_per_objective",
            _required_int(
                self.max_tasks_per_objective,
                "max_tasks_per_objective",
                minimum=1,
                maximum=MAX_TASKS_PER_OBJECTIVE,
            ),
        )
        object.__setattr__(
            self,
            "oscillation_window",
            _required_int(
                self.oscillation_window,
                "oscillation_window",
                minimum=2,
                maximum=MAX_OSCILLATION_WINDOW,
            ),
        )
        object.__setattr__(
            self,
            "max_repeated_plan_hashes",
            _required_int(
                self.max_repeated_plan_hashes,
                "max_repeated_plan_hashes",
                minimum=2,
                maximum=MAX_REPEATED_PLAN_HASHES,
            ),
        )
        object.__setattr__(
            self,
            "max_stable_frontier_rounds",
            _required_int(
                self.max_stable_frontier_rounds,
                "max_stable_frontier_rounds",
                minimum=1,
                maximum=MAX_STABLE_FRONTIER_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "event_debounce_ms",
            _required_int(
                self.event_debounce_ms,
                "event_debounce_ms",
                minimum=MIN_EVENT_DEBOUNCE_MS,
                maximum=MAX_EVENT_DEBOUNCE_MS,
            ),
        )
        object.__setattr__(
            self,
            "oscillation_detection",
            _required_bool(
                self.oscillation_detection, "oscillation_detection", required_true=True
            ),
        )
        object.__setattr__(
            self,
            "runaway_detection",
            _required_bool(self.runaway_detection, "runaway_detection", required_true=True),
        )
        object.__setattr__(
            self,
            "nonconvergence_quarantine",
            _required_bool(
                self.nonconvergence_quarantine,
                "nonconvergence_quarantine",
                required_true=True,
            ),
        )
        object.__setattr__(
            self,
            "repeated_plan_hash_detection",
            _required_bool(
                self.repeated_plan_hash_detection,
                "repeated_plan_hash_detection",
                required_true=True,
            ),
        )
        object.__setattr__(
            self,
            "stable_frontier_detection",
            _required_bool(
                self.stable_frontier_detection,
                "stable_frontier_detection",
                required_true=True,
            ),
        )

    @property
    def policy_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_REFILL_POLICY_SCHEMA,
            "max_refill_rounds": self.max_refill_rounds,
            "max_no_progress_rounds": self.max_no_progress_rounds,
            "max_curriculum_repetitions": self.max_curriculum_repetitions,
            "max_tasks_per_refill": self.max_tasks_per_refill,
            "max_open_work": self.max_open_work,
            "cooldown_ms": self.cooldown_ms,
            "max_generated_tasks": self.max_generated_tasks,
            "max_replan_epochs": self.max_replan_epochs,
            "max_tasks_per_objective": self.max_tasks_per_objective,
            "oscillation_window": self.oscillation_window,
            "max_repeated_plan_hashes": self.max_repeated_plan_hashes,
            "max_stable_frontier_rounds": self.max_stable_frontier_rounds,
            "event_debounce_ms": self.event_debounce_ms,
            "oscillation_detection": self.oscillation_detection,
            "runaway_detection": self.runaway_detection,
            "nonconvergence_quarantine": self.nonconvergence_quarantine,
            "repeated_plan_hash_detection": self.repeated_plan_hash_detection,
            "stable_frontier_detection": self.stable_frontier_detection,
        }

    def trigger_bound(self, trigger: RefillTrigger) -> int:
        if trigger is RefillTrigger.NO_PROGRESS:
            return self.max_no_progress_rounds
        return TRIGGER_BOUNDS[trigger]


@dataclass(frozen=True)
class CampaignRefillCandidate:
    """One residual item considered for a refill pass."""

    candidate_id: str
    trigger: RefillTrigger
    residual_count: int = 1
    curriculum_key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _required_text(self.candidate_id, "candidate_id")
        )
        selected = (
            self.trigger if isinstance(self.trigger, RefillTrigger) else RefillTrigger(str(self.trigger))
        )
        object.__setattr__(self, "trigger", selected)
        object.__setattr__(
            self,
            "residual_count",
            _required_int(self.residual_count, "residual_count", minimum=1),
        )
        object.__setattr__(self, "curriculum_key", str(self.curriculum_key or "").strip())
        if not isinstance(self.metadata, Mapping):
            raise CampaignRefillError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def progress_identity(self) -> str:
        return self.curriculum_key or self.candidate_id

    def score(self) -> tuple[int, int, str]:
        """Deterministic total order: priority, residual, then identity."""

        return (
            TRIGGER_PRIORITY[self.trigger],
            -int(self.residual_count),
            self.candidate_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_REFILL_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "trigger": self.trigger.value,
            "residual_count": self.residual_count,
            "curriculum_key": self.curriculum_key,
            "progress_identity": self.progress_identity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CampaignRefillHistory:
    """Observed refill firings and progress identities for one campaign."""

    refill_rounds: int = 0
    open_work: int = 0
    trigger_counts: Mapping[str, int] = field(default_factory=dict)
    progress_identities: tuple[str, ...] = ()
    last_progress_identity: str = ""
    no_progress_streak: int = 0
    curriculum_repetitions: Mapping[str, int] = field(default_factory=dict)
    plan_hashes: tuple[str, ...] = ()
    last_plan_hash: str = ""
    frontier_identities: tuple[str, ...] = ()
    last_frontier_identity: str = ""
    stable_frontier_streak: int = 0
    generated_task_count: int = 0
    replan_epochs: int = 0
    tasks_per_objective: Mapping[str, int] = field(default_factory=dict)
    last_event_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "refill_rounds", _required_int(self.refill_rounds, "refill_rounds")
        )
        object.__setattr__(self, "open_work", _required_int(self.open_work, "open_work"))
        counts = {
            _required_text(key, "trigger"): _required_int(value, "trigger count")
            for key, value in dict(self.trigger_counts).items()
        }
        object.__setattr__(self, "trigger_counts", counts)
        identities = tuple(
            _required_text(item, "progress identity") for item in self.progress_identities
        )
        object.__setattr__(self, "progress_identities", identities)
        object.__setattr__(
            self, "last_progress_identity", str(self.last_progress_identity or "").strip()
        )
        object.__setattr__(
            self,
            "no_progress_streak",
            _required_int(self.no_progress_streak, "no_progress_streak"),
        )
        repetitions = {
            _required_text(key, "curriculum key"): _required_int(value, "curriculum repetition")
            for key, value in dict(self.curriculum_repetitions).items()
        }
        object.__setattr__(self, "curriculum_repetitions", repetitions)
        object.__setattr__(
            self, "plan_hashes", _compact_identities(self.plan_hashes, "plan hash")
        )
        object.__setattr__(self, "last_plan_hash", str(self.last_plan_hash or "").strip())
        object.__setattr__(
            self,
            "frontier_identities",
            _compact_identities(self.frontier_identities, "frontier identity"),
        )
        object.__setattr__(
            self,
            "last_frontier_identity",
            str(self.last_frontier_identity or "").strip(),
        )
        object.__setattr__(
            self,
            "stable_frontier_streak",
            _required_int(self.stable_frontier_streak, "stable_frontier_streak"),
        )
        object.__setattr__(
            self,
            "generated_task_count",
            _required_int(self.generated_task_count, "generated_task_count"),
        )
        object.__setattr__(
            self, "replan_epochs", _required_int(self.replan_epochs, "replan_epochs")
        )
        objectives = {
            _required_text(key, "objective_id"): _required_int(value, "tasks per objective")
            for key, value in dict(self.tasks_per_objective).items()
        }
        object.__setattr__(self, "tasks_per_objective", objectives)
        object.__setattr__(
            self, "last_event_ms", _required_int(self.last_event_ms, "last_event_ms")
        )

    def count_for(self, trigger: RefillTrigger) -> int:
        return int(self.trigger_counts.get(trigger.value, 0))


@dataclass(frozen=True)
class CampaignRefillDecision:
    """Exact, bounded outcome of one refill admission."""

    disposition: RefillDisposition
    policy_id: str
    admitted: tuple[CampaignRefillCandidate, ...] = ()
    rejected: tuple[CampaignRefillCandidate, ...] = ()
    reason_code: str = ""
    trigger_counts: Mapping[str, int] = field(default_factory=dict)
    control: str = ""
    quarantined: bool = False

    def __post_init__(self) -> None:
        selected = (
            self.disposition
            if isinstance(self.disposition, RefillDisposition)
            else RefillDisposition(str(self.disposition))
        )
        object.__setattr__(self, "disposition", selected)
        object.__setattr__(self, "policy_id", _required_text(self.policy_id, "policy_id"))
        object.__setattr__(self, "admitted", tuple(self.admitted))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "reason_code", str(self.reason_code or selected.value))
        object.__setattr__(
            self,
            "trigger_counts",
            {
                _required_text(key, "trigger"): _required_int(value, "trigger count")
                for key, value in dict(self.trigger_counts).items()
            },
        )
        object.__setattr__(self, "control", str(self.control or "").strip())
        quarantined = selected is RefillDisposition.QUARANTINED_NONCONVERGENT or bool(
            self.quarantined
        )
        if self.quarantined and selected is not RefillDisposition.QUARANTINED_NONCONVERGENT:
            raise CampaignRefillError("quarantined decisions must use quarantined_nonconvergent")
        object.__setattr__(self, "quarantined", quarantined)

    @property
    def changed(self) -> bool:
        return bool(self.admitted) and self.disposition is RefillDisposition.ADMITTED

    @property
    def bounded(self) -> bool:
        return self.disposition is not RefillDisposition.ADMITTED or not self.admitted

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CAMPAIGN_REFILL_DECISION_SCHEMA,
            "requirement_id": CAMPAIGN_DURABILITY_REQUIREMENT_ID,
            "disposition": self.disposition.value,
            "policy_id": self.policy_id,
            "reason_code": self.reason_code,
            "admitted": [item.to_dict() for item in self.admitted],
            "rejected": [item.to_dict() for item in self.rejected],
            "trigger_counts": dict(self.trigger_counts),
            "changed": self.changed,
            "control": self.control,
            "quarantined": self.quarantined,
            "binding": OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING,
            "interface": OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE,
            "carrier": "CampaignRefillController",
            "consumes": list(OSCILLATION_RUNAWAY_NONCONVERGENCE_CONSUMES),
            "model_free": True,
            "authorizes_append": False,
            "authorizes_completion": False,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
            "no_competing_subsystem_created": True,
            "empty_queue_is_completion": False,
        }
        payload["decision_id"] = content_identity(payload)
        return payload


def detect_oscillation(
    identities: Sequence[str],
    *,
    window: int = MAX_OSCILLATION_WINDOW,
) -> bool:
    """Return True when recent identities cycle instead of remaining a single stall."""

    bound = _required_int(
        window, "oscillation window", minimum=2, maximum=MAX_OSCILLATION_WINDOW
    )
    series = _compact_identities(identities, "oscillation identity")
    if len(series) < 3:
        return False
    recent = series[-bound:]
    if len(recent) < 3:
        return False
    unique = set(recent)
    if len(unique) == 1:
        return False
    return len(unique) < len(recent)


def detect_repeated_plan_hash(
    hashes: Sequence[str],
    current: str = "",
    *,
    max_repeats: int = MAX_REPEATED_PLAN_HASHES,
) -> bool:
    """Return True when the current plan hash has already been observed enough times."""

    bound = _required_int(
        max_repeats,
        "max_repeated_plan_hashes",
        minimum=2,
        maximum=MAX_REPEATED_PLAN_HASHES,
    )
    current_hash = _optional_text(current, "plan hash")
    if not current_hash:
        return False
    prior = _compact_identities(hashes, "plan hash")
    return prior.count(current_hash) + 1 >= bound


def detect_stable_frontier(
    identities: Sequence[str],
    current: str = "",
    *,
    max_stable_frontier_rounds: int = MAX_STABLE_FRONTIER_ROUNDS,
) -> bool:
    """Return True when the frontier identity has not changed across the bound."""

    bound = _required_int(
        max_stable_frontier_rounds,
        "max_stable_frontier_rounds",
        minimum=1,
        maximum=MAX_STABLE_FRONTIER_ROUNDS,
    )
    series = list(_compact_identities(identities, "frontier identity"))
    current_identity = _optional_text(current, "frontier identity")
    if current_identity:
        series.append(current_identity)
    if len(series) < bound:
        return False
    recent = tuple(series[-bound:])
    return len(set(recent)) == 1


def detect_runaway(
    *,
    generated_task_count: int,
    replan_epochs: int,
    tasks_for_objective: int,
    policy: CampaignRefillPolicy | None = None,
) -> str:
    """Return a runaway reason code, or empty when the campaign is still inside bounds."""

    selected = policy or CampaignRefillPolicy()
    generated = _required_int(generated_task_count, "generated_task_count")
    epochs = _required_int(replan_epochs, "replan_epochs")
    objective_count = _required_int(tasks_for_objective, "tasks_for_objective")
    if generated >= selected.max_generated_tasks:
        return "runaway_generated_tasks"
    if epochs >= selected.max_replan_epochs:
        return "runaway_replan_epochs"
    if objective_count >= selected.max_tasks_per_objective:
        return "runaway_tasks_per_objective"
    return ""


def _append_current(series: Sequence[str], current: str) -> tuple[str, ...]:
    items = list(series)
    if current:
        items.append(current)
    return tuple(items)


def _quarantine(
    *,
    policy: CampaignRefillPolicy,
    ranked: Sequence[CampaignRefillCandidate],
    reason_code: str,
    control: str,
    trigger_counts: Mapping[str, int],
) -> CampaignRefillDecision:
    return CampaignRefillDecision(
        disposition=RefillDisposition.QUARANTINED_NONCONVERGENT,
        policy_id=policy.policy_id,
        rejected=tuple(ranked),
        reason_code=reason_code,
        trigger_counts=trigger_counts,
        control=control,
        quarantined=True,
    )


def evaluate_convergence_controls(
    candidates: Sequence[CampaignRefillCandidate] = (),
    *,
    policy: CampaignRefillPolicy | None = None,
    history: CampaignRefillHistory | None = None,
    progress_identity: str = "",
    plan_hash: str = "",
    frontier_identity: str = "",
    generated_task_count: int | None = None,
    replan_epochs: int | None = None,
    objective_id: str = "",
    last_event_ms: int | None = None,
    now_ms: int | None = None,
    worker_assertion: bool = False,
) -> CampaignRefillDecision | None:
    """Return a blocking convergence decision, or None when refill may continue.

    A worker or model assertion cannot skip oscillation, runaway, or
    nonconvergence quarantine.
    """

    selected = policy or CampaignRefillPolicy()
    observed = history or CampaignRefillHistory()
    ranked = tuple(candidates)
    _required_bool(worker_assertion, "worker_assertion")
    current_progress = _optional_text(progress_identity, "progress_identity")
    current_hash = _optional_text(plan_hash, "plan_hash")
    current_frontier = _optional_text(frontier_identity, "frontier_identity")
    generated = (
        observed.generated_task_count
        if generated_task_count is None
        else _required_int(generated_task_count, "generated_task_count")
    )
    epochs = (
        observed.replan_epochs
        if replan_epochs is None
        else _required_int(replan_epochs, "replan_epochs")
    )
    current_objective = _optional_text(objective_id, "objective_id")
    if last_event_ms is None and now_ms is None:
        debounce_last = observed.last_event_ms or None
        debounce_now = None
    elif last_event_ms is None or now_ms is None:
        raise CampaignRefillError("event debounce requires both last_event_ms and now_ms")
    else:
        debounce_last = _required_int(last_event_ms, "last_event_ms")
        debounce_now = _required_int(now_ms, "now_ms")
        if debounce_now < debounce_last:
            raise CampaignRefillError("now_ms must not precede last_event_ms")

    if debounce_last is not None and debounce_now is not None:
        if debounce_now - debounce_last < selected.event_debounce_ms:
            return CampaignRefillDecision(
                disposition=RefillDisposition.EVENT_DEBOUNCED,
                policy_id=selected.policy_id,
                rejected=ranked,
                reason_code="event_debounced",
                trigger_counts=observed.trigger_counts,
                control="runaway",
            )

    progress_series = _append_current(observed.progress_identities, current_progress)
    if detect_oscillation(progress_series, window=selected.oscillation_window):
        return _quarantine(
            policy=selected,
            ranked=ranked,
            reason_code="oscillation_detected",
            control="oscillation",
            trigger_counts=observed.trigger_counts,
        )

    hash_series = observed.plan_hashes
    if not hash_series and observed.last_plan_hash:
        hash_series = (observed.last_plan_hash,)
    if detect_repeated_plan_hash(
        hash_series,
        current_hash,
        max_repeats=selected.max_repeated_plan_hashes,
    ):
        return _quarantine(
            policy=selected,
            ranked=ranked,
            reason_code="repeated_plan_hash",
            control="oscillation",
            trigger_counts=observed.trigger_counts,
        )

    frontier_series = observed.frontier_identities
    if not frontier_series and observed.last_frontier_identity:
        repeats = max(1, observed.stable_frontier_streak)
        frontier_series = tuple(observed.last_frontier_identity for _ in range(repeats))
    if detect_stable_frontier(
        frontier_series,
        current_frontier,
        max_stable_frontier_rounds=selected.max_stable_frontier_rounds,
    ):
        return _quarantine(
            policy=selected,
            ranked=ranked,
            reason_code="stable_frontier",
            control="nonconvergence",
            trigger_counts=observed.trigger_counts,
        )

    objective_counts = dict(observed.tasks_per_objective)
    tracked_objective = current_objective
    for candidate in ranked:
        candidate_objective = _objective_id_of(candidate, current_objective)
        if candidate_objective:
            tracked_objective = tracked_objective or candidate_objective
            if int(objective_counts.get(candidate_objective, 0)) >= selected.max_tasks_per_objective:
                return _quarantine(
                    policy=selected,
                    ranked=ranked,
                    reason_code="runaway_tasks_per_objective",
                    control="runaway",
                    trigger_counts=observed.trigger_counts,
                )
    tasks_for_objective = int(objective_counts.get(tracked_objective, 0)) if tracked_objective else 0
    runaway_reason = detect_runaway(
        generated_task_count=generated,
        replan_epochs=epochs,
        tasks_for_objective=tasks_for_objective,
        policy=selected,
    )
    if runaway_reason:
        return _quarantine(
            policy=selected,
            ranked=ranked,
            reason_code=runaway_reason,
            control="runaway",
            trigger_counts=observed.trigger_counts,
        )
    return None


class CampaignRefillController:
    """Score and admit a finite residual set under the campaign policy."""

    BINDING: ClassVar[str] = CAMPAIGN_REFILL_POLICY_BINDING
    INTERFACE: ClassVar[str] = CAMPAIGN_REFILL_POLICY_INTERFACE
    CONVERGENCE_BINDING: ClassVar[str] = OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING
    CONVERGENCE_INTERFACE: ClassVar[str] = OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE
    CONVERGENCE_SCHEMA: ClassVar[str] = OSCILLATION_RUNAWAY_NONCONVERGENCE_SCHEMA
    CONVERGENCE_CONSUMES: ClassVar[tuple[str, ...]] = (
        OSCILLATION_RUNAWAY_NONCONVERGENCE_CONSUMES
    )

    def __init__(self, policy: CampaignRefillPolicy | None = None) -> None:
        self.policy = policy or CampaignRefillPolicy()

    def rank(
        self, candidates: Iterable[CampaignRefillCandidate | Mapping[str, Any]]
    ) -> tuple[CampaignRefillCandidate, ...]:
        normalized = tuple(
            item if isinstance(item, CampaignRefillCandidate) else CampaignRefillCandidate(
                candidate_id=str(item.get("candidate_id") or ""),
                trigger=item.get("trigger") or RefillTrigger.CURRICULUM_GAP,  # type: ignore[arg-type]
                residual_count=int(item.get("residual_count") or 1),
                curriculum_key=str(item.get("curriculum_key") or ""),
                metadata=item.get("metadata") or {},
            )
            for item in candidates
        )
        return tuple(sorted(normalized, key=lambda item: item.score()))

    def evaluate_convergence_controls(
        self,
        candidates: Sequence[CampaignRefillCandidate | Mapping[str, Any]] = (),
        *,
        history: CampaignRefillHistory | None = None,
        progress_identity: str = "",
        plan_hash: str = "",
        frontier_identity: str = "",
        generated_task_count: int | None = None,
        replan_epochs: int | None = None,
        objective_id: str = "",
        last_event_ms: int | None = None,
        now_ms: int | None = None,
        worker_assertion: bool = False,
    ) -> CampaignRefillDecision | None:
        ranked = self.rank(candidates) if candidates else ()
        return evaluate_convergence_controls(
            ranked,
            policy=self.policy,
            history=history,
            progress_identity=progress_identity,
            plan_hash=plan_hash,
            frontier_identity=frontier_identity,
            generated_task_count=generated_task_count,
            replan_epochs=replan_epochs,
            objective_id=objective_id,
            last_event_ms=last_event_ms,
            now_ms=now_ms,
            worker_assertion=worker_assertion,
        )

    def decide(
        self,
        candidates: Sequence[CampaignRefillCandidate | Mapping[str, Any]],
        *,
        history: CampaignRefillHistory | None = None,
        cursor_advanced: bool = True,
        progress_identity: str = "",
        plan_hash: str = "",
        frontier_identity: str = "",
        generated_task_count: int | None = None,
        replan_epochs: int | None = None,
        objective_id: str = "",
        last_event_ms: int | None = None,
        now_ms: int | None = None,
        worker_assertion: bool = False,
    ) -> CampaignRefillDecision:
        policy = self.policy
        observed = history or CampaignRefillHistory()
        ranked = self.rank(candidates)
        blocked = evaluate_convergence_controls(
            ranked,
            policy=policy,
            history=observed,
            progress_identity=progress_identity,
            plan_hash=plan_hash,
            frontier_identity=frontier_identity,
            generated_task_count=generated_task_count,
            replan_epochs=replan_epochs,
            objective_id=objective_id,
            last_event_ms=last_event_ms,
            now_ms=now_ms,
            worker_assertion=worker_assertion,
        )
        if blocked is not None:
            return blocked
        rejected: list[CampaignRefillCandidate] = []
        if observed.refill_rounds >= policy.max_refill_rounds:
            return CampaignRefillDecision(
                disposition=RefillDisposition.ROUND_BOUNDED,
                policy_id=policy.policy_id,
                rejected=ranked,
                reason_code="max_refill_rounds",
                trigger_counts=observed.trigger_counts,
            )
        if observed.open_work >= policy.max_open_work:
            return CampaignRefillDecision(
                disposition=RefillDisposition.OPEN_WORK_BOUNDED,
                policy_id=policy.policy_id,
                rejected=ranked,
                reason_code="max_open_work",
                trigger_counts=observed.trigger_counts,
            )
        identity = str(progress_identity or "").strip()
        no_progress_streak = observed.no_progress_streak
        if identity and identity == observed.last_progress_identity and not cursor_advanced:
            no_progress_streak += 1
        elif not cursor_advanced:
            no_progress_streak += 1
        else:
            no_progress_streak = 0
        if no_progress_streak >= policy.max_no_progress_rounds:
            return CampaignRefillDecision(
                disposition=RefillDisposition.NO_PROGRESS_BOUNDED,
                policy_id=policy.policy_id,
                rejected=ranked,
                reason_code="no_progress_bound",
                trigger_counts=observed.trigger_counts,
            )

        admitted: list[CampaignRefillCandidate] = []
        local_counts = dict(observed.trigger_counts)
        local_repetitions = dict(observed.curriculum_repetitions)
        for candidate in ranked:
            if len(admitted) >= policy.max_tasks_per_refill:
                rejected.append(candidate)
                continue
            if observed.open_work + len(admitted) >= policy.max_open_work:
                rejected.append(candidate)
                continue
            trigger_count = int(local_counts.get(candidate.trigger.value, 0))
            if trigger_count >= policy.trigger_bound(candidate.trigger):
                rejected.append(candidate)
                continue
            curriculum_key = candidate.curriculum_key
            if curriculum_key:
                seen = int(local_repetitions.get(curriculum_key, 0))
                if seen >= policy.max_curriculum_repetitions:
                    rejected.append(candidate)
                    continue
                local_repetitions[curriculum_key] = seen + 1
            local_counts[candidate.trigger.value] = trigger_count + 1
            admitted.append(candidate)

        if not admitted:
            if rejected and any(
                int(local_counts.get(item.trigger.value, 0))
                >= policy.trigger_bound(item.trigger)
                or (
                    item.curriculum_key
                    and int(local_repetitions.get(item.curriculum_key, 0))
                    >= policy.max_curriculum_repetitions
                )
                for item in rejected
            ):
                if any(
                    item.curriculum_key
                    and int(observed.curriculum_repetitions.get(item.curriculum_key, 0))
                    >= policy.max_curriculum_repetitions
                    for item in rejected
                ):
                    disposition = RefillDisposition.REPETITION_BOUNDED
                    reason = "curriculum_repetition_bound"
                else:
                    disposition = RefillDisposition.TRIGGER_BOUNDED
                    reason = "trigger_bound"
            else:
                disposition = RefillDisposition.REJECTED
                reason = "no_admissible_candidates"
            return CampaignRefillDecision(
                disposition=disposition,
                policy_id=policy.policy_id,
                rejected=tuple(rejected),
                reason_code=reason,
                trigger_counts=local_counts,
            )
        return CampaignRefillDecision(
            disposition=RefillDisposition.ADMITTED,
            policy_id=policy.policy_id,
            admitted=tuple(admitted),
            rejected=tuple(rejected),
            reason_code="admitted",
            trigger_counts=local_counts,
        )


def apply_oscillation_runaway_nonconvergence_controls(
    candidates: Sequence[CampaignRefillCandidate | Mapping[str, Any]] = (),
    *,
    policy: CampaignRefillPolicy | None = None,
    history: CampaignRefillHistory | None = None,
    cursor_advanced: bool = True,
    progress_identity: str = "",
    plan_hash: str = "",
    frontier_identity: str = "",
    generated_task_count: int | None = None,
    replan_epochs: int | None = None,
    objective_id: str = "",
    last_event_ms: int | None = None,
    now_ms: int | None = None,
    worker_assertion: bool = False,
) -> CampaignRefillDecision:
    """Apply oscillation, runaway, and nonconvergence controls on the existing controller.

    This is not a second refill owner.  It reuses ``CampaignRefillController.decide``
    and never writes DuckDB or authorizes completion.
    """

    return CampaignRefillController(policy).decide(
        candidates,
        history=history,
        cursor_advanced=cursor_advanced,
        progress_identity=progress_identity,
        plan_hash=plan_hash,
        frontier_identity=frontier_identity,
        generated_task_count=generated_task_count,
        replan_epochs=replan_epochs,
        objective_id=objective_id,
        last_event_ms=last_event_ms,
        now_ms=now_ms,
        worker_assertion=worker_assertion,
    )


def all_refill_triggers_are_bounded(policy: CampaignRefillPolicy | None = None) -> bool:
    """Return True when every closed trigger has a positive finite bound."""

    selected = policy or CampaignRefillPolicy()
    return all(selected.trigger_bound(trigger) >= 1 for trigger in RefillTrigger)


def convergence_controls_are_armed(policy: CampaignRefillPolicy | None = None) -> bool:
    """Return True when oscillation, runaway, and nonconvergence gates are armed."""

    selected = policy or CampaignRefillPolicy()
    return bool(
        selected.oscillation_detection
        and selected.runaway_detection
        and selected.nonconvergence_quarantine
        and selected.repeated_plan_hash_detection
        and selected.stable_frontier_detection
        and selected.max_generated_tasks >= 1
        and selected.max_replan_epochs >= 1
        and selected.max_tasks_per_objective >= 1
        and selected.oscillation_window >= 2
        and selected.max_repeated_plan_hashes >= 2
        and selected.max_stable_frontier_rounds >= 1
        and selected.event_debounce_ms >= MIN_EVENT_DEBOUNCE_MS
    )


__all__ = (
    "AUTOMATIC_BOUNDED_TASK_REFILL_BINDING",
    "CAMPAIGN_REFILL_DECISION_SCHEMA",
    "CAMPAIGN_REFILL_POLICY_BINDING",
    "CAMPAIGN_REFILL_POLICY_INTERFACE",
    "CAMPAIGN_REFILL_POLICY_SCHEMA",
    "MAX_CURRICULUM_REPETITIONS",
    "MAX_GENERATED_TASKS",
    "MAX_NO_PROGRESS_ROUNDS",
    "MAX_OPEN_WORK",
    "MAX_OSCILLATION_WINDOW",
    "MAX_REFILL_ROUNDS",
    "MAX_REPEATED_PLAN_HASHES",
    "MAX_REPLAN_EPOCHS",
    "MAX_STABLE_FRONTIER_ROUNDS",
    "MAX_TASKS_PER_OBJECTIVE",
    "MAX_TASKS_PER_REFILL",
    "MIN_EVENT_DEBOUNCE_MS",
    "OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING",
    "OSCILLATION_RUNAWAY_NONCONVERGENCE_CONSUMES",
    "OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE",
    "OSCILLATION_RUNAWAY_NONCONVERGENCE_SCHEMA",
    "TASK_SEMANTIC_DEDUPLICATION_BINDING",
    "TRIGGER_BOUNDS",
    "TRIGGER_PRIORITY",
    "CampaignRefillCandidate",
    "CampaignRefillController",
    "CampaignRefillDecision",
    "CampaignRefillError",
    "CampaignRefillHistory",
    "CampaignRefillPolicy",
    "RefillDisposition",
    "RefillTrigger",
    "all_refill_triggers_are_bounded",
    "apply_oscillation_runaway_nonconvergence_controls",
    "convergence_controls_are_armed",
    "detect_oscillation",
    "detect_repeated_plan_hash",
    "detect_runaway",
    "detect_stable_frontier",
    "evaluate_convergence_controls",
)
