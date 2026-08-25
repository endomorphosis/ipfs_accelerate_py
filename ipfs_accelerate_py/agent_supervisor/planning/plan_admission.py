"""Deterministic scoring and single-plan admission (EAAEF-073).

Candidate work plans are ranked on integer axes only.  A closed comparison
gate admits exactly one plan; empty candidate sets fail closed.  No random
or wall-clock value participates in ranking.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.planning.external_work_plan import (
    ExternalWorkPlan,
    ExternalWorkTask,
    WorkPlanError,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


PLAN_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-plan-admission@1"
)
PLAN_ADMISSION_INTERFACE: Final[str] = "ExternalPlanAdmission@1"
SCORE_SCALE: Final[int] = 1_000_000
SCORE_AXES: Final[tuple[str, ...]] = (
    "critical_path",
    "safe_width",
    "model_proof_cost",
    "resources",
    "merge_risk",
    "uncertainty",
    "prior_success",
    "cache_locality",
)
_COST_AXES: Final[frozenset[str]] = frozenset(
    {
        "critical_path",
        "model_proof_cost",
        "resources",
        "merge_risk",
        "uncertainty",
    }
)
_BENEFIT_AXES: Final[frozenset[str]] = frozenset(
    {"safe_width", "prior_success", "cache_locality"}
)


class PlanAdmissionError(ValueError):
    """Empty, malformed, or unscored candidate set cannot be admitted."""


def _nonneg_int(value: object, name: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanAdmissionError(f"{name} must be a non-negative integer")
    return value


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise PlanAdmissionError(f"{name} is required")
    return text


def _tuple_text(values: object, name: str) -> tuple[str, ...]:
    if values is None:
        items: Sequence[object] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence):
        items = values
    else:
        raise PlanAdmissionError(f"{name} must be a list of strings")
    result = tuple(_text(item, name, required=True) for item in items)
    if len(set(result)) != len(result):
        raise PlanAdmissionError(f"{name} contains duplicates")
    return result


def _history_counts(history: object) -> Mapping[str, int]:
    if history is None:
        return MappingProxyType({})
    if not isinstance(history, Mapping):
        raise PlanAdmissionError("history must be an object")
    counts: dict[str, int] = {}
    for key, value in history.items():
        counts[_text(key, "history key")] = _nonneg_int(value, "history count")
    return MappingProxyType(counts)


def _invert(cost: int) -> int:
    return SCORE_SCALE // (1 + max(0, cost))


def _benefit(value: int) -> int:
    magnitude = max(0, value)
    return (SCORE_SCALE * magnitude) // (1 + magnitude)


def _task_levels(tasks: Sequence[ExternalWorkTask]) -> dict[str, int]:
    graph = {task.task_id: task.depends_on for task in tasks}
    levels: dict[str, int] = {}
    visiting: set[str] = set()

    def level(task_id: str) -> int:
        if task_id in levels:
            return levels[task_id]
        if task_id in visiting:
            raise PlanAdmissionError("task dependencies must be acyclic")
        visiting.add(task_id)
        current = 0
        for dependency in graph.get(task_id, ()):
            current = max(current, level(dependency) + 1)
        visiting.remove(task_id)
        levels[task_id] = current
        return current

    for task in tasks:
        level(task.task_id)
    return levels


def _critical_path(tasks: Sequence[ExternalWorkTask]) -> int:
    by_id = {task.task_id: task for task in tasks}
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def path(task_id: str) -> int:
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting:
            raise PlanAdmissionError("task dependencies must be acyclic")
        visiting.add(task_id)
        task = by_id[task_id]
        prior = 0
        for dependency in task.depends_on:
            prior = max(prior, path(dependency))
        visiting.remove(task_id)
        memo[task_id] = prior + task.timeout_seconds
        return memo[task_id]

    return max((path(task.task_id) for task in tasks), default=0)


def _safe_width(tasks: Sequence[ExternalWorkTask]) -> int:
    counts: dict[int, int] = defaultdict(int)
    levels = _task_levels(tasks)
    for task in tasks:
        counts[levels[task.task_id]] += 1
    return max(counts.values(), default=0)


def _peak_resources(tasks: Sequence[ExternalWorkTask]) -> int:
    levels = _task_levels(tasks)
    cpu: dict[int, int] = defaultdict(int)
    ram: dict[int, int] = defaultdict(int)
    for task in tasks:
        level = levels[task.task_id]
        cpu[level] += task.cpu_millicores
        ram[level] += task.ram_mib
    return max(cpu.values(), default=0) + max(ram.values(), default=0)


def _path_parent(path: str) -> str:
    if "/" not in path:
        return ""
    return path.rsplit("/", 1)[0]


def _merge_risk(tasks: Sequence[ExternalWorkTask]) -> int:
    risk = 0
    ordered = tuple(sorted(tasks, key=lambda item: item.task_id))
    for index, left in enumerate(ordered):
        left_parents = {_path_parent(path) for path in left.write_scope}
        for right in ordered[index + 1 :]:
            right_parents = {_path_parent(path) for path in right.write_scope}
            if left_parents.intersection(right_parents):
                risk += 1
    return risk


def _uncertainty(tasks: Sequence[ExternalWorkTask], extra: int) -> int:
    counts = Counter(cover for task in tasks for cover in task.covers)
    redundant = sum(count - 1 for count in counts.values() if count > 1)
    return extra + redundant


def _axis_score(name: str, value: int) -> int:
    if name in _COST_AXES:
        return _invert(value)
    if name in _BENEFIT_AXES:
        return _benefit(value)
    raise PlanAdmissionError(f"unknown score axis {name}")


@dataclass(frozen=True)
class PlanCandidate:
    """One scored alternative: a validated work plan plus closed extras."""

    plan: ExternalWorkPlan
    candidate_id: str = ""
    model_cost: int = 0
    proof_cost: int = 0
    uncertainty: int = 0
    prior_success: int = 0
    cache_locality: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExternalWorkPlan):
            raise PlanAdmissionError("plan must be an ExternalWorkPlan")
        ident = _text(self.candidate_id, "candidate_id", required=False) or self.plan.content_id
        object.__setattr__(self, "candidate_id", ident)
        object.__setattr__(self, "model_cost", _nonneg_int(self.model_cost, "model_cost"))
        object.__setattr__(self, "proof_cost", _nonneg_int(self.proof_cost, "proof_cost"))
        object.__setattr__(self, "uncertainty", _nonneg_int(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "prior_success", _nonneg_int(self.prior_success, "prior_success"))
        object.__setattr__(
            self, "cache_locality", _nonneg_int(self.cache_locality, "cache_locality")
        )


@dataclass(frozen=True)
class ScoredPlan:
    """Integer score vector and total for one candidate."""

    candidate_id: str
    content_id: str
    plan: ExternalWorkPlan
    components: Mapping[str, int]
    axis_scores: Mapping[str, int]
    total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "axis_scores", MappingProxyType(dict(self.axis_scores)))
        if tuple(self.components) != SCORE_AXES or tuple(self.axis_scores) != SCORE_AXES:
            raise PlanAdmissionError("score must include every named axis")
        if isinstance(self.total, bool) or not isinstance(self.total, int):
            raise PlanAdmissionError("total must be an integer")

    @property
    def rank_key(self) -> tuple[int, str, str]:
        return (-self.total, self.candidate_id, self.content_id)

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "candidate_id": self.candidate_id,
                "content_id": self.content_id,
                "total": int(self.total),
                "components": {name: int(self.components[name]) for name in SCORE_AXES},
                "axis_scores": {name: int(self.axis_scores[name]) for name in SCORE_AXES},
            }
        )


@dataclass(frozen=True)
class PlanAdmission:
    """Exactly one admitted plan and the deterministic ranking behind it."""

    admitted: ExternalWorkPlan
    admitted_id: str
    ranked: tuple[ScoredPlan, ...]

    def __post_init__(self) -> None:
        if not self.ranked:
            raise PlanAdmissionError("candidates are required")
        if self.ranked[0].candidate_id != self.admitted_id:
            raise PlanAdmissionError("admitted plan must be the first ranked candidate")
        if self.ranked[0].content_id != self.admitted.content_id:
            raise PlanAdmissionError("admitted plan identity mismatch")

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": PLAN_ADMISSION_SCHEMA,
                "interface": PLAN_ADMISSION_INTERFACE,
                "verdict": "admitted",
                "admitted_id": self.admitted_id,
                "admitted_plan_id": self.admitted.formal_plan.plan_id,
                "admitted_content_id": self.admitted.content_id,
                "ranked": [dict(item.to_dict()) for item in self.ranked],
            }
        )


def _compile_plan(payload: object) -> ExternalWorkPlan:
    if isinstance(payload, ExternalWorkPlan):
        return payload
    if not isinstance(payload, Mapping):
        raise PlanAdmissionError("plan must be an ExternalWorkPlan or object")
    try:
        return ExternalWorkPlan.from_dict(payload)
    except WorkPlanError as exc:
        raise PlanAdmissionError(str(exc)) from exc


def compile_candidate(
    payload: PlanCandidate | ExternalWorkPlan | Mapping[str, Any],
) -> PlanCandidate:
    """Normalize a candidate mapping or work plan into a PlanCandidate."""

    if isinstance(payload, PlanCandidate):
        return payload
    if isinstance(payload, ExternalWorkPlan):
        return PlanCandidate(plan=payload, candidate_id=payload.content_id)
    if not isinstance(payload, Mapping):
        raise PlanAdmissionError("candidate must be a plan or object")
    plan_payload = payload.get("plan") if "plan" in payload else payload
    return PlanCandidate(
        plan=_compile_plan(plan_payload),
        candidate_id=str(payload.get("candidate_id") or payload.get("id") or ""),
        model_cost=_nonneg_int(payload.get("model_cost"), "model_cost"),
        proof_cost=_nonneg_int(payload.get("proof_cost"), "proof_cost"),
        uncertainty=_nonneg_int(payload.get("uncertainty"), "uncertainty"),
        prior_success=_nonneg_int(payload.get("prior_success"), "prior_success"),
        cache_locality=_nonneg_int(payload.get("cache_locality"), "cache_locality"),
    )


def _cache_hits(plan: ExternalWorkPlan, cache_keys: Sequence[str]) -> int:
    keys = set(cache_keys)
    if not keys:
        return 0
    hits = 0
    for task in plan.tasks:
        hits += sum(1 for path in task.write_scope if path in keys)
        hits += sum(1 for cover in task.covers if cover in keys)
    return hits


def score_plan(
    candidate: PlanCandidate | ExternalWorkPlan | Mapping[str, Any],
    *,
    cache_keys: Sequence[str] = (),
    history: Mapping[str, int] | None = None,
) -> ScoredPlan:
    """Score one candidate on the closed integer axes."""

    compiled = compile_candidate(candidate)
    plan = compiled.plan
    tasks = plan.tasks
    cache = _tuple_text(cache_keys, "cache_keys")
    prior = _history_counts(history)
    history_hits = 0
    seen_history: set[str] = set()
    for key in (compiled.candidate_id, plan.content_id):
        if key in seen_history:
            continue
        seen_history.add(key)
        history_hits += int(prior.get(key, 0))
    components = {
        "critical_path": _critical_path(tasks),
        "safe_width": _safe_width(tasks),
        "model_proof_cost": (
            compiled.model_cost
            + compiled.proof_cost
            + len(plan.goal.proof_requirements) * len(tasks)
        ),
        "resources": _peak_resources(tasks),
        "merge_risk": _merge_risk(tasks),
        "uncertainty": _uncertainty(tasks, compiled.uncertainty),
        "prior_success": compiled.prior_success + history_hits,
        "cache_locality": compiled.cache_locality + _cache_hits(plan, cache),
    }
    ordered = {name: int(components[name]) for name in SCORE_AXES}
    axis_scores = {name: _axis_score(name, ordered[name]) for name in SCORE_AXES}
    return ScoredPlan(
        candidate_id=compiled.candidate_id,
        content_id=plan.content_id,
        plan=plan,
        components=ordered,
        axis_scores=axis_scores,
        total=sum(axis_scores.values()),
    )


def logic_gate(scored: Sequence[ScoredPlan]) -> ScoredPlan:
    """Admit the unique best score; ties break by candidate then content id."""

    if not scored:
        raise PlanAdmissionError("candidates are required")
    return tuple(sorted(scored, key=lambda item: item.rank_key))[0]


def rank_plans(
    candidates: Sequence[PlanCandidate | ExternalWorkPlan | Mapping[str, Any]],
    *,
    cache_keys: Sequence[str] = (),
    history: Mapping[str, int] | None = None,
) -> tuple[ScoredPlan, ...]:
    """Deterministically rank every candidate.  Empty input is rejected."""

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise PlanAdmissionError("candidates are required")
    if not candidates:
        raise PlanAdmissionError("candidates are required")
    scored = tuple(
        score_plan(item, cache_keys=cache_keys, history=history) for item in candidates
    )
    return tuple(sorted(scored, key=lambda item: item.rank_key))


def admit_plan(
    candidates: Sequence[PlanCandidate | ExternalWorkPlan | Mapping[str, Any]],
    *,
    cache_keys: Sequence[str] = (),
    history: Mapping[str, int] | None = None,
) -> PlanAdmission:
    """Score alternatives and admit exactly one plan through the logic gate."""

    ranked = rank_plans(candidates, cache_keys=cache_keys, history=history)
    winner = logic_gate(ranked)
    return PlanAdmission(admitted=winner.plan, admitted_id=winner.candidate_id, ranked=ranked)
