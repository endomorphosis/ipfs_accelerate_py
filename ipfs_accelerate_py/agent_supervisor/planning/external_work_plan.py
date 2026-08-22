"""Bounded task decompositions compiled from goal contracts (EAAEF-071).

Decompositions are admitted only after existing FormalWorkPlan machinery
accepts unique identifiers, acyclic dependencies, and non-conflicting
effects.  Coverage, write-scope, resource, merge/proof feasibility and
duplicate-effect checks are enforced against the parent goal contract.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
    GoalContractError,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    Actor,
    ActorKind,
    ContractValidationError,
    FormalWorkPlan,
    Goal,
    PlanTask,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_logic_vocabulary import (
    LOGIC_VOCABULARY_VERSION,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


WORK_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-work-plan@1"
)
WORK_PLAN_INTERFACE: Final[str] = "ExternalWorkPlan@1"
_SUPERVISOR_ACTOR: Final[str] = "actor:supervisor"
_AGENT_ACTOR: Final[str] = "actor:agent"


class WorkPlanError(ValueError):
    """Malformed, uncovered, cyclic, or infeasible bounded work plan."""


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise WorkPlanError(f"{name} is required")
    return text


def _tuple_text(values: object, name: str) -> tuple[str, ...]:
    if values is None:
        items: Sequence[object] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence):
        items = values
    else:
        raise WorkPlanError(f"{name} must be a list of strings")
    result = tuple(_text(item, name, required=True) for item in items)
    if len(set(result)) != len(result):
        raise WorkPlanError(f"{name} contains duplicates")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkPlanError(f"{name} must be positive")
    return value


def _flag(value: object, name: str, default: bool = True) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise WorkPlanError(f"{name} must be a boolean")
    return value


def _in_write_scope(path: str, scopes: Sequence[str]) -> bool:
    for scope in scopes:
        if path == scope:
            return True
        prefix = scope if scope.endswith("/") else f"{scope}/"
        if path.startswith(prefix):
            return True
    return False


def _depends_graph(tasks: Sequence["ExternalWorkTask"]) -> dict[str, tuple[str, ...]]:
    return {task.task_id: task.depends_on for task in tasks}


def _reaches(source: str, target: str, graph: Mapping[str, Sequence[str]]) -> bool:
    seen: set[str] = set()
    stack = [source]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()))
    return False


def _assert_acyclic(tasks: Sequence["ExternalWorkTask"]) -> None:
    graph = _depends_graph(tasks)
    known = set(graph)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise WorkPlanError("task dependencies must be acyclic")
        visiting.add(node)
        for dependency in graph.get(node, ()):
            if dependency not in known:
                raise WorkPlanError(f"task {node} has unknown dependency {dependency}")
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for task_id in sorted(known):
        visit(task_id)


def _longest_path_levels(tasks: Sequence["ExternalWorkTask"]) -> dict[str, int]:
    graph = _depends_graph(tasks)
    by_id = {task.task_id: task for task in tasks}
    levels: dict[str, int] = {}
    visiting: set[str] = set()

    def level(task_id: str) -> int:
        if task_id in levels:
            return levels[task_id]
        if task_id in visiting:
            raise WorkPlanError("task dependencies must be acyclic")
        visiting.add(task_id)
        current = 0
        for dependency in graph.get(task_id, ()):
            if dependency not in by_id:
                raise WorkPlanError(f"task {task_id} has unknown dependency {dependency}")
            current = max(current, level(dependency) + 1)
        visiting.remove(task_id)
        levels[task_id] = current
        return current

    for task in tasks:
        level(task.task_id)
    return levels


def _compile_goal(goal: ExternalGoalContract | Mapping[str, Any]) -> ExternalGoalContract:
    if isinstance(goal, ExternalGoalContract):
        return goal
    if not isinstance(goal, Mapping):
        raise WorkPlanError("goal must be an ExternalGoalContract or object")
    try:
        return ExternalGoalContract.compile(goal)
    except GoalContractError as exc:
        raise WorkPlanError(str(exc)) from exc


def _to_formal_work_plan(
    goal: ExternalGoalContract, tasks: Sequence["ExternalWorkTask"]
) -> FormalWorkPlan:
    try:
        return FormalWorkPlan(
            vocabulary_profile_id="supervisor-reviewed",
            vocabulary_version=LOGIC_VOCABULARY_VERSION,
            source_ids=(goal.objective_id, goal.content_id),
            repository_tree_id=goal.content_id,
            trace_bound=max(1, len(tuple(tasks))),
            actors=(
                Actor(
                    actor_id=_SUPERVISOR_ACTOR,
                    kind=ActorKind.SUPERVISOR,
                    capabilities=("delegate", "merge"),
                    authority_ids=(goal.authority_ceiling,),
                ),
                Actor(
                    actor_id=_AGENT_ACTOR,
                    kind=ActorKind.AGENT,
                    capabilities=("implement",),
                ),
            ),
            goals=(
                Goal(
                    goal_id=goal.objective_id,
                    owner_actor_id=_SUPERVISOR_ACTOR,
                    satisfaction_formula_id=f"formula:goal:{goal.objective_id}",
                    source_ids=(goal.content_id,),
                    metadata={
                        "desired_outcomes": list(goal.desired_outcomes),
                        "write_scope": list(goal.write_scope),
                    },
                ),
            ),
            subgoals=(),
            events=(),
            fluents=(),
            preconditions=(),
            effects=(),
            norms=(),
            temporal_constraints=(),
            evidence_requirements=(),
            tasks=tuple(
                PlanTask(
                    task_id=task.task_id,
                    goal_id=goal.objective_id,
                    actor_ids=(_AGENT_ACTOR,),
                    depends_on=task.depends_on,
                    metadata={
                        "covers": list(task.covers),
                        "write_scope": list(task.write_scope),
                        "timeout_seconds": task.timeout_seconds,
                        "cpu_millicores": task.cpu_millicores,
                        "ram_mib": task.ram_mib,
                        "merge_feasible": task.merge_feasible,
                        "proof_feasible": task.proof_feasible,
                    },
                )
                for task in tasks
            ),
            metadata={
                "schema": WORK_PLAN_SCHEMA,
                "goal_content_id": goal.content_id,
                "proof_requirements": list(goal.proof_requirements),
                "verification_requirements": list(goal.verification_requirements),
            },
        )
    except ContractValidationError as exc:
        raise WorkPlanError(str(exc)) from exc


@dataclass(frozen=True)
class ExternalWorkTask:
    """One bounded child of a goal contract."""

    task_id: str
    covers: tuple[str, ...]
    write_scope: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    timeout_seconds: int = 1
    cpu_millicores: int = 1
    ram_mib: int = 1
    merge_feasible: bool = True
    proof_feasible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "covers", _tuple_text(self.covers, "covers"))
        object.__setattr__(self, "write_scope", _tuple_text(self.write_scope, "write_scope"))
        object.__setattr__(self, "depends_on", _tuple_text(self.depends_on, "depends_on"))
        if self.task_id in self.depends_on:
            raise WorkPlanError("a task cannot depend on itself")
        object.__setattr__(
            self, "timeout_seconds", _positive_int(self.timeout_seconds, "timeout_seconds")
        )
        object.__setattr__(
            self, "cpu_millicores", _positive_int(self.cpu_millicores, "cpu_millicores")
        )
        object.__setattr__(self, "ram_mib", _positive_int(self.ram_mib, "ram_mib"))
        object.__setattr__(self, "merge_feasible", _flag(self.merge_feasible, "merge_feasible"))
        object.__setattr__(self, "proof_feasible", _flag(self.proof_feasible, "proof_feasible"))

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "task_id": self.task_id,
                "covers": list(self.covers),
                "write_scope": list(self.write_scope),
                "depends_on": list(self.depends_on),
                "timeout_seconds": int(self.timeout_seconds),
                "cpu_millicores": int(self.cpu_millicores),
                "ram_mib": int(self.ram_mib),
                "merge_feasible": bool(self.merge_feasible),
                "proof_feasible": bool(self.proof_feasible),
            }
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "ExternalWorkTask",
        *,
        goal: ExternalGoalContract | None = None,
    ) -> "ExternalWorkTask":
        if isinstance(payload, ExternalWorkTask):
            return payload
        if not isinstance(payload, Mapping):
            raise WorkPlanError("task must be an object")
        timeout = payload.get("timeout_seconds")
        cpu = payload.get("cpu_millicores")
        ram = payload.get("ram_mib")
        if goal is not None:
            if timeout is None:
                timeout = goal.timeout_seconds
            if cpu is None:
                cpu = goal.cpu_millicores
            if ram is None:
                ram = goal.ram_mib
        return cls(
            task_id=str(payload.get("task_id") or payload.get("id") or ""),
            covers=payload.get("covers") or (),
            write_scope=payload.get("write_scope") or (),
            depends_on=payload.get("depends_on") or (),
            timeout_seconds=int(timeout or 0),
            cpu_millicores=int(cpu or 0),
            ram_mib=int(ram or 0),
            merge_feasible=_flag(payload.get("merge_feasible"), "merge_feasible"),
            proof_feasible=_flag(payload.get("proof_feasible"), "proof_feasible"),
        )


@dataclass(frozen=True)
class ExternalWorkPlan:
    """Validated bounded decomposition of an ExternalGoalContract."""

    goal: ExternalGoalContract
    tasks: tuple[ExternalWorkTask, ...]
    formal_plan: FormalWorkPlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        goal = _compile_goal(self.goal)
        object.__setattr__(self, "goal", goal)
        raw_tasks = self.tasks
        if isinstance(raw_tasks, str) or not isinstance(raw_tasks, Sequence):
            raise WorkPlanError("tasks must be a list")
        compiled = tuple(
            ExternalWorkTask.from_mapping(item, goal=goal) for item in raw_tasks
        )
        if not compiled:
            raise WorkPlanError("tasks are required")
        ids = tuple(task.task_id for task in compiled)
        if len(ids) != len(set(ids)):
            raise WorkPlanError("duplicate task ids")
        compiled = tuple(sorted(compiled, key=lambda item: item.task_id))
        object.__setattr__(self, "tasks", compiled)
        _assert_acyclic(compiled)
        self._validate_against_goal(goal, compiled)
        object.__setattr__(self, "formal_plan", _to_formal_work_plan(goal, compiled))

    @staticmethod
    def _validate_against_goal(
        goal: ExternalGoalContract, tasks: Sequence[ExternalWorkTask]
    ) -> None:
        covered: set[str] = set()
        signatures: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
        for task in tasks:
            overlap = set(task.covers).intersection(goal.prohibited_outcomes)
            if overlap:
                raise WorkPlanError("task covers prohibited outcomes; contradiction")
            covered.update(task.covers)
            for path in task.write_scope:
                if not _in_write_scope(path, goal.write_scope):
                    raise WorkPlanError("write-scope outside the goal write_scope")
            if task.timeout_seconds > goal.timeout_seconds:
                raise WorkPlanError("timeout_seconds exceeds goal resource bound")
            if task.cpu_millicores > goal.cpu_millicores:
                raise WorkPlanError("cpu_millicores exceeds goal resource bound")
            if task.ram_mib > goal.ram_mib:
                raise WorkPlanError("ram_mib exceeds goal resource bound")
            if goal.proof_requirements and not task.proof_feasible:
                raise WorkPlanError("proof is not feasible for required proof obligations")
            signature = (tuple(sorted(task.covers)), tuple(sorted(task.write_scope)))
            prior = signatures.get(signature)
            if prior is not None:
                raise WorkPlanError("duplicate task semantics")
            signatures[signature] = task.task_id

        missing = [outcome for outcome in goal.desired_outcomes if outcome not in covered]
        if missing:
            raise WorkPlanError("missing coverage of desired outcomes")

        graph = _depends_graph(tasks)
        for left in tasks:
            for right in tasks:
                if left.task_id >= right.task_id:
                    continue
                shared = set(left.write_scope).intersection(right.write_scope)
                if not shared:
                    continue
                ordered = _reaches(left.task_id, right.task_id, graph) or _reaches(
                    right.task_id, left.task_id, graph
                )
                if ordered:
                    continue
                if not (left.merge_feasible and right.merge_feasible):
                    raise WorkPlanError("merge is not feasible for overlapping write-scope")
                raise WorkPlanError(
                    "overlapping write-scope without serialized order is duplicate semantics"
                )

        levels = _longest_path_levels(tasks)
        cpu_by_level: dict[int, int] = defaultdict(int)
        ram_by_level: dict[int, int] = defaultdict(int)
        timeout_by_id: dict[str, int] = {}
        visiting: set[str] = set()

        def critical_path(task_id: str) -> int:
            if task_id in timeout_by_id:
                return timeout_by_id[task_id]
            if task_id in visiting:
                raise WorkPlanError("task dependencies must be acyclic")
            visiting.add(task_id)
            task = next(item for item in tasks if item.task_id == task_id)
            prior = 0
            for dependency in task.depends_on:
                prior = max(prior, critical_path(dependency))
            visiting.remove(task_id)
            timeout_by_id[task_id] = prior + task.timeout_seconds
            return timeout_by_id[task_id]

        for task in tasks:
            cpu_by_level[levels[task.task_id]] += task.cpu_millicores
            ram_by_level[levels[task.task_id]] += task.ram_mib
            critical_path(task.task_id)
        if cpu_by_level and max(cpu_by_level.values()) > goal.cpu_millicores:
            raise WorkPlanError("parallel cpu_millicores exceed goal resource bound")
        if ram_by_level and max(ram_by_level.values()) > goal.ram_mib:
            raise WorkPlanError("parallel ram_mib exceed goal resource bound")
        if timeout_by_id and max(timeout_by_id.values()) > goal.timeout_seconds:
            raise WorkPlanError("critical-path timeout exceeds goal resource bound")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": WORK_PLAN_SCHEMA,
                "interface": WORK_PLAN_INTERFACE,
                "goal": dict(self.goal.to_dict()),
                "tasks": [dict(task.to_dict()) for task in self.tasks],
                "formal_plan_id": self.formal_plan.plan_id,
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))

    @classmethod
    def decompose(
        cls,
        goal: ExternalGoalContract | Mapping[str, Any],
        tasks: Sequence[ExternalWorkTask | Mapping[str, Any]],
    ) -> "ExternalWorkPlan":
        return cls(goal=_compile_goal(goal), tasks=tuple(tasks))

    compile = decompose

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalWorkPlan":
        if not isinstance(payload, Mapping):
            raise WorkPlanError("work plan must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", WORK_PLAN_SCHEMA):
            raise WorkPlanError(f"unsupported schema {schema!r}")
        return cls.decompose(payload.get("goal") or {}, payload.get("tasks") or ())
