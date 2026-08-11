"""Compile replayable conflict/resource/lease/merge execution plans.

``ParallelExecutionPlan@1`` is the provider-free join between an admitted task
DAG and the supervisor's conflict, resource, worktree, lease, and merge
contracts.  A parallel-lane label is deliberately not an authority input: the
compiler recomputes the dependency and conflict width from canonical task
content and intersects it with fresh host/provider capacity.

The compiler is pure.  It assigns *names* for worktrees, leases, and fencing
epochs, but does not create a worktree, reserve capacity, acquire a lease, or
mutate a merge queue.  The runtime must compare the plan's snapshot/replay
bindings with live state immediately before performing those operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..core.conflict_graph import materialize_task_conflict_graph

PARALLEL_PLAN_COMPILER_INTERFACE: Final[str] = "ParallelPlanCompiler@1"
PARALLEL_EXECUTION_PLAN_INTERFACE: Final[str] = "ParallelExecutionPlan@1"
PARALLEL_EXECUTION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/parallel-execution-plan@1"
)
PARALLEL_PLAN_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/parallel-plan-compilation-request@1"
)
MAX_TASKS: Final[int] = 4_096
MAX_WIDTH: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 8_192
DEFAULT_CAPACITY_MAX_AGE_MS: Final[int] = 60_000
_SPACE_RE = re.compile(r"\s+")


class ParallelPlanError(ValueError):
    """Base error for malformed or unsafe parallel-plan input."""


class ParallelPlanRejectedError(ParallelPlanError):
    """Raised when ``raise_on_rejection`` requests an exception result."""

    def __init__(self, plan: ParallelExecutionPlan) -> None:
        self.plan = plan
        super().__init__(
            "parallel plan rejected: "
            + ", ".join(issue.code.value for issue in plan.issues)
        )


class ParallelPlanOutcome(str, Enum):
    PARALLEL = "parallel"
    SERIAL = "serial"
    DEGRADED = "degraded"
    REVIEW_ONLY = "review_only"
    REJECTED = "rejected"


class ParallelPlanIssueCode(str, Enum):
    EMPTY_PLAN = "empty_plan"
    DUPLICATE_TASK_ID = "duplicate_task_id"
    UNKNOWN_DEPENDENCY = "unknown_dependency"
    DEPENDENCY_CYCLE = "dependency_cycle"
    MISSING_LEAF_PRODUCER = "missing_leaf_producer"
    OUTPUT_COLLISION = "output_collision"
    PROTECTED_BOTTLENECK = "protected_bottleneck"
    OVERLAPPING_SUBMODULES = "overlapping_submodules"
    STALE_CAPACITY = "stale_capacity"
    IMPOSSIBLE_DEADLINE = "impossible_deadline"
    FAKE_LANE_LABEL = "fake_lane_label"
    RESOURCE_INFEASIBLE = "resource_infeasible"
    PROVIDER_INFEASIBLE = "provider_infeasible"
    TOKEN_INFEASIBLE = "token_infeasible"
    COST_INFEASIBLE = "cost_infeasible"
    CONTEXT_INFEASIBLE = "context_infeasible"
    LEASE_INFEASIBLE = "lease_infeasible"
    WORKTREE_INFEASIBLE = "worktree_infeasible"
    INVALID_SNAPSHOT = "invalid_snapshot"


_HARD_REJECTION_CODES = frozenset(
    {
        ParallelPlanIssueCode.EMPTY_PLAN,
        ParallelPlanIssueCode.DUPLICATE_TASK_ID,
        ParallelPlanIssueCode.UNKNOWN_DEPENDENCY,
        ParallelPlanIssueCode.DEPENDENCY_CYCLE,
        ParallelPlanIssueCode.MISSING_LEAF_PRODUCER,
        ParallelPlanIssueCode.OUTPUT_COLLISION,
        ParallelPlanIssueCode.PROTECTED_BOTTLENECK,
        ParallelPlanIssueCode.OVERLAPPING_SUBMODULES,
        ParallelPlanIssueCode.STALE_CAPACITY,
        ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE,
        ParallelPlanIssueCode.FAKE_LANE_LABEL,
        ParallelPlanIssueCode.RESOURCE_INFEASIBLE,
        ParallelPlanIssueCode.PROVIDER_INFEASIBLE,
        ParallelPlanIssueCode.TOKEN_INFEASIBLE,
        ParallelPlanIssueCode.COST_INFEASIBLE,
        ParallelPlanIssueCode.CONTEXT_INFEASIBLE,
        ParallelPlanIssueCode.LEASE_INFEASIBLE,
        ParallelPlanIssueCode.WORKTREE_INFEASIBLE,
        ParallelPlanIssueCode.INVALID_SNAPSHOT,
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return {name: getattr(value, name) for name in fields}
    raise ParallelPlanError("records must be mappings or expose to_dict()")


def _sequence(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _strings(value: Any, *, preserve_order: bool = False) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in _sequence(value):
        text = _SPACE_RE.sub(" ", str(item)).strip()
        if not text or text in seen:
            continue
        if "\x00" in text or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ParallelPlanError("text value is invalid or exceeds the size bound")
        seen.add(text)
        result.append(text)
    return tuple(result if preserve_order else sorted(result))


def _paths(value: Any) -> tuple[str, ...]:
    result: set[str] = set()
    for raw in _strings(value):
        path = raw.replace("\\", "/").removeprefix("./").rstrip("/")
        pure = PurePosixPath(path)
        if not path or pure.is_absolute() or ".." in pure.parts:
            raise ParallelPlanError(f"non-canonical repository path: {raw!r}")
        result.add(pure.as_posix())
    return tuple(sorted(result))


def _integer(
    value: Any,
    name: str,
    *,
    default: int = 0,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ParallelPlanError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ParallelPlanError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ParallelPlanError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _boolean(value: Any, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ParallelPlanError("boolean field has a non-boolean value")


def _health(value: Any) -> bool:
    """Normalize the health vocabularies used by runtime provider snapshots."""

    if value in (None, ""):
        return True
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"ready", "available", "healthy", "up", "ok"}:
            return True
        if normalized in {
            "down",
            "failed",
            "error",
            "disabled",
            "offline",
            "unhealthy",
            "quota_exhausted",
            "rate_limited",
        }:
            return False
    return _boolean(value, default=True)


def _first(sources: Sequence[Mapping[str, Any]], *names: str, default: Any = None) -> Any:
    for source in sources:
        for name in names:
            if name in source and source[name] not in (None, "", (), [], {}):
                return source[name]
    return default


def _union(sources: Sequence[Mapping[str, Any]], *names: str) -> tuple[str, ...]:
    values: list[Any] = []
    for source in sources:
        for name in names:
            values.extend(_sequence(source.get(name)))
    return _strings(values)


def _union_paths(sources: Sequence[Mapping[str, Any]], *names: str) -> tuple[str, ...]:
    values: list[Any] = []
    for source in sources:
        for name in names:
            values.extend(_sequence(source.get(name)))
    return _paths(values)


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items.sort(key=lambda item: json.dumps(_canonical(item), sort_keys=True))
        return [_canonical(item) for item in items]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParallelPlanError("canonical replay values must be finite")
        return format(value, ".17g")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    return str(value)


def _digest(namespace: str, value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _path_overlaps(left: Iterable[str], right: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                shorter
                for a in left
                for b in right
                if _overlap(a, b)
                for shorter in (a if len(a) <= len(b) else b,)
            }
        )
    )


@dataclass(frozen=True)
class ParallelPlanIssue:
    code: ParallelPlanIssueCode
    message: str
    task_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "task_ids": list(self.task_ids),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class WidthProjection:
    requested: int
    graph: int
    conflict: int
    resource: int
    admitted: int

    def to_dict(self) -> dict[str, int]:
        return {
            "requested_width": self.requested,
            "graph_width": self.graph,
            "conflict_width": self.conflict,
            "resource_width": self.resource,
            "admitted_width": self.admitted,
        }


@dataclass(frozen=True)
class ResourceFeasibilityProjection:
    feasible: bool
    host_feasible: bool
    provider_feasible: bool
    token_feasible: bool
    cost_feasible: bool
    context_feasible: bool
    freshness_proved: bool
    required_totals: Mapping[str, int]
    available_host: Mapping[str, int]
    provider_by_task_id: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "host_feasible": self.host_feasible,
            "provider_feasible": self.provider_feasible,
            "token_feasible": self.token_feasible,
            "cost_feasible": self.cost_feasible,
            "context_feasible": self.context_feasible,
            "freshness_proved": self.freshness_proved,
            "required_totals": dict(sorted(self.required_totals.items())),
            "available_host": dict(sorted(self.available_host.items())),
            "provider_by_task_id": dict(sorted(self.provider_by_task_id.items())),
        }


@dataclass(frozen=True)
class LeafProducerClosure:
    required_leaf_ids: tuple[str, ...]
    producer_by_leaf_id: Mapping[str, str]
    terminal_task_ids: tuple[str, ...]
    closed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_leaf_ids": list(self.required_leaf_ids),
            "producer_by_leaf_id": dict(sorted(self.producer_by_leaf_id.items())),
            "terminal_task_ids": list(self.terminal_task_ids),
            "closed": self.closed,
        }


@dataclass(frozen=True)
class ConflictSurfaceRecord:
    left_task_id: str
    right_task_id: str
    paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    submodules: tuple[str, ...] = ()
    generated_artifacts: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    exclusive_paths: tuple[str, ...] = ()
    exclusive_groups: tuple[str, ...] = ()
    anti_affinity_keys: tuple[str, ...] = ()
    observed_receipts: tuple[str, ...] = ()
    blocking: bool = True

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "paths",
                "symbols",
                "interfaces",
                "submodules",
                "generated_artifacts",
                "protected_paths",
                "exclusive_paths",
                "exclusive_groups",
                "anti_affinity_keys",
                "observed_receipts",
            )
            if getattr(self, name)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_task_id": self.left_task_id,
            "right_task_id": self.right_task_id,
            "kinds": list(self.kinds),
            "paths": list(self.paths),
            "symbols": list(self.symbols),
            "interfaces": list(self.interfaces),
            "submodules": list(self.submodules),
            "generated_artifacts": list(self.generated_artifacts),
            "protected_paths": list(self.protected_paths),
            "exclusive_paths": list(self.exclusive_paths),
            "exclusive_groups": list(self.exclusive_groups),
            "anti_affinity_keys": list(self.anti_affinity_keys),
            "observed_receipts": list(self.observed_receipts),
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class TaskAssignment:
    task_id: str
    shard_id: str
    affinity_key: str
    exclusive_group: str
    exclusive_paths: tuple[str, ...]
    worktree_id: str
    worktree_path: str
    base_revision: str
    merge_target: str
    lease_id: str
    lease_scope: str
    lease_duration_ms: int
    heartbeat_interval_ms: int
    lease_owner_rule: str
    fence_epoch: int
    fence_token: str
    provider_id: str
    resource_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "shard_id": self.shard_id,
            "affinity_key": self.affinity_key,
            "exclusive_group": self.exclusive_group,
            "exclusive_paths": list(self.exclusive_paths),
            "worktree_id": self.worktree_id,
            "worktree_path": self.worktree_path,
            "base_revision": self.base_revision,
            "merge_target": self.merge_target,
            "lease_id": self.lease_id,
            "lease_scope": self.lease_scope,
            "lease_duration_ms": self.lease_duration_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "lease_owner_rule": self.lease_owner_rule,
            "fence_epoch": self.fence_epoch,
            "fence_token": self.fence_token,
            "provider_id": self.provider_id,
            "resource_class": self.resource_class,
        }


@dataclass(frozen=True)
class ReadyWave:
    dependency_wave: int
    graph_ready_task_ids: tuple[str, ...]
    conflict_free_lanes: tuple[tuple[str, ...], ...]
    graph_width: int
    conflict_width: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_wave": self.dependency_wave,
            "graph_ready_task_ids": list(self.graph_ready_task_ids),
            "conflict_free_lanes": [list(lane) for lane in self.conflict_free_lanes],
            "graph_width": self.graph_width,
            "conflict_width": self.conflict_width,
        }


@dataclass(frozen=True)
class ExecutionWave:
    execution_wave: int
    dependency_wave: int
    task_ids: tuple[str, ...]
    resource_usage: Mapping[str, int]
    provider_usage: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_wave": self.execution_wave,
            "dependency_wave": self.dependency_wave,
            "task_ids": list(self.task_ids),
            "width": len(self.task_ids),
            "resource_usage": dict(sorted(self.resource_usage.items())),
            "provider_usage": {
                key: dict(sorted(value.items()))
                for key, value in sorted(self.provider_usage.items())
            },
        }


@dataclass(frozen=True)
class MergeStep:
    order: int
    task_id: str
    merge_train_id: str
    depends_on: tuple[str, ...]
    rollback_boundary: str
    checkpoint_id: str
    post_merge_validation: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "task_id": self.task_id,
            "merge_train_id": self.merge_train_id,
            "depends_on": list(self.depends_on),
            "rollback_boundary": self.rollback_boundary,
            "checkpoint_id": self.checkpoint_id,
            "post_merge_validation": list(self.post_merge_validation),
        }


@dataclass(frozen=True)
class ParallelPlanCompilationRequest:
    tasks: tuple[Any, ...]
    requested_width: int = 1
    repository_snapshot: Any = field(default_factory=dict)
    capacity_snapshot: Any = field(default_factory=dict)
    provider_snapshots: Any = ()
    budget: Any = field(default_factory=dict)
    current_time_ms: int = 0
    deadline_ms: int = 0
    required_leaf_ids: tuple[str, ...] = ()
    completed_task_ids: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    submodule_paths: tuple[str, ...] = ()
    post_merge_validation: tuple[str, ...] = ()
    conflict_receipts: tuple[Any, ...] = ()
    base_fence_epoch: int = 0
    review_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        # Empty is represented as a typed rejection by the compiler; only the
        # upper bound is malformed input.
        if len(self.tasks) > MAX_TASKS:
            raise ParallelPlanError(f"tasks exceeds {MAX_TASKS}")
        object.__setattr__(
            self,
            "requested_width",
            _integer(self.requested_width, "requested_width", minimum=1, maximum=MAX_WIDTH),
        )
        for name in ("current_time_ms", "deadline_ms", "base_fence_epoch"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "required_leaf_ids", _strings(self.required_leaf_ids))
        object.__setattr__(self, "completed_task_ids", _strings(self.completed_task_ids))
        object.__setattr__(self, "protected_paths", _paths(self.protected_paths))
        object.__setattr__(self, "submodule_paths", _paths(self.submodule_paths))
        object.__setattr__(self, "post_merge_validation", _strings(self.post_merge_validation))
        object.__setattr__(
            self,
            "provider_snapshots",
            _provider_snapshot_records(self.provider_snapshots),
        )
        object.__setattr__(self, "conflict_receipts", tuple(self.conflict_receipts))
        object.__setattr__(self, "review_only", _boolean(self.review_only))

    def to_replay_dict(self) -> dict[str, Any]:
        return {
            "schema": PARALLEL_PLAN_REQUEST_SCHEMA,
            "tasks": [_mapping(task) for task in self.tasks],
            "requested_width": self.requested_width,
            "repository_snapshot": _mapping(self.repository_snapshot),
            "capacity_snapshot": _mapping(self.capacity_snapshot),
            "provider_snapshots": [dict(item) for item in self.provider_snapshots],
            "budget": _mapping(self.budget),
            "current_time_ms": self.current_time_ms,
            "deadline_ms": self.deadline_ms,
            "required_leaf_ids": list(self.required_leaf_ids),
            "completed_task_ids": list(self.completed_task_ids),
            "protected_paths": list(self.protected_paths),
            "submodule_paths": list(self.submodule_paths),
            "post_merge_validation": list(self.post_merge_validation),
            "conflict_receipts": [_mapping(item) for item in self.conflict_receipts],
            "base_fence_epoch": self.base_fence_epoch,
            "review_only": self.review_only,
        }


@dataclass(frozen=True)
class ParallelExecutionPlan:
    outcome: ParallelPlanOutcome
    input_digest: str
    repository_tree_id: str
    capacity_snapshot_id: str
    provider_snapshot_ids: tuple[str, ...]
    widths: WidthProjection
    resource_feasibility: ResourceFeasibilityProjection
    leaf_producer_closure: LeafProducerClosure
    dependency_edges: tuple[tuple[str, str], ...]
    critical_path: tuple[str, ...]
    critical_path_duration_ms: int
    estimated_makespan_ms: int
    ready_waves: tuple[ReadyWave, ...]
    execution_waves: tuple[ExecutionWave, ...]
    conflicts: tuple[ConflictSurfaceRecord, ...]
    assignments: tuple[TaskAssignment, ...]
    merge_order: tuple[MergeStep, ...]
    issues: tuple[ParallelPlanIssue, ...]
    replay_request: Mapping[str, Any] = field(repr=False, compare=False)
    plan_id: str = ""

    def __post_init__(self) -> None:
        material = self._material()
        computed = _digest("parallel-execution-plan", material)
        if self.plan_id and self.plan_id != computed:
            raise ParallelPlanError("parallel execution plan identity is invalid")
        object.__setattr__(self, "plan_id", computed)

    @property
    def admitted(self) -> bool:
        return self.outcome is not ParallelPlanOutcome.REJECTED

    @property
    def ready(self) -> bool:
        return self.admitted and bool(self.execution_waves or self.outcome is ParallelPlanOutcome.REVIEW_ONLY)

    @property
    def requested_width(self) -> int:
        return self.widths.requested

    @property
    def graph_width(self) -> int:
        return self.widths.graph

    @property
    def conflict_width(self) -> int:
        return self.widths.conflict

    @property
    def resource_width(self) -> int:
        return self.widths.resource

    @property
    def admitted_width(self) -> int:
        return self.widths.admitted

    @property
    def deterministic_replay(self) -> Mapping[str, Any]:
        return {
            "compiler_interface": PARALLEL_PLAN_COMPILER_INTERFACE,
            "input_digest": self.input_digest,
            "plan_id": self.plan_id,
            "repository_tree_id": self.repository_tree_id,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "provider_snapshot_ids": list(self.provider_snapshot_ids),
        }

    @property
    def dependency_graph(self) -> Mapping[str, Any]:
        return {
            "edges": [list(edge) for edge in self.dependency_edges],
            "ready_waves": [wave.to_dict() for wave in self.ready_waves],
            "critical_path": list(self.critical_path),
            "critical_path_duration_ms": self.critical_path_duration_ms,
            "estimated_makespan_ms": self.estimated_makespan_ms,
        }

    @property
    def conflict_graph(self) -> Mapping[str, Any]:
        task_ids = sorted(
            {
                task_id
                for wave in self.ready_waves
                for task_id in wave.graph_ready_task_ids
            }
        )
        return {
            "task_ids": task_ids,
            "edges": [conflict.to_dict() for conflict in self.conflicts],
            "blocking_pairs": [
                [conflict.left_task_id, conflict.right_task_id]
                for conflict in self.conflicts
                if conflict.blocking
            ],
        }

    def _material(self) -> dict[str, Any]:
        return {
            "schema": PARALLEL_EXECUTION_PLAN_SCHEMA,
            "interface": PARALLEL_EXECUTION_PLAN_INTERFACE,
            "outcome": self.outcome.value,
            "input_digest": self.input_digest,
            "repository_tree_id": self.repository_tree_id,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "provider_snapshot_ids": list(self.provider_snapshot_ids),
            "widths": self.widths.to_dict(),
            **self.widths.to_dict(),
            "resource_feasibility": self.resource_feasibility.to_dict(),
            "leaf_producer_closure": self.leaf_producer_closure.to_dict(),
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "dependency_graph": self.dependency_graph,
            "critical_path": list(self.critical_path),
            "critical_path_duration_ms": self.critical_path_duration_ms,
            "estimated_makespan_ms": self.estimated_makespan_ms,
            "ready_waves": [wave.to_dict() for wave in self.ready_waves],
            "execution_waves": [wave.to_dict() for wave in self.execution_waves],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "conflict_graph": self.conflict_graph,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "merge_order": [step.to_dict() for step in self.merge_order],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_dict(self, *, include_replay_request: bool = False) -> dict[str, Any]:
        payload = {
            **self._material(),
            "plan_id": self.plan_id,
            "admitted": self.admitted,
            "ready": self.ready,
            "deterministic_replay": dict(self.deterministic_replay),
        }
        if include_replay_request:
            payload["replay_request"] = _canonical(self.replay_request)
        return payload

    def to_json(self, *, include_replay_request: bool = False) -> str:
        return json.dumps(
            self.to_dict(include_replay_request=include_replay_request),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class _Task:
    task_id: str
    aliases: tuple[str, ...]
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]
    paths: tuple[str, ...]
    symbols: tuple[str, ...]
    interfaces: tuple[str, ...]
    submodules: tuple[str, ...]
    generated_artifacts: tuple[str, ...]
    protected_paths: tuple[str, ...]
    exclusive_paths: tuple[str, ...]
    produces: tuple[str, ...]
    required_leaf_ids: tuple[str, ...]
    validation: tuple[str, ...]
    resource_class: str
    stage: str
    cpu_slots: int
    process_slots: int
    memory_bytes: int
    gpu_memory_bytes: int
    disk_bytes: int
    duration_ms: int
    provider_requirement: str
    required_capabilities: tuple[str, ...]
    context_tokens: int
    output_tokens: int
    quota_units: int
    cost_micros: int
    max_provider_latency_ms: int
    affinity_key: str
    anti_affinity_key: str
    exclusive_group: str
    shard_key: str
    lane_label: str
    lane_authoritative: bool
    claimed_parallel_with: tuple[str, ...]
    worktree_policy: str
    worktree_bytes: int
    expected_base_revision: str
    expected_merge_target: str
    lease_scope: str
    lease_duration_ms: int
    heartbeat_interval_ms: int
    lease_owner_rule: str
    fence_epoch: int
    merge_train_id: str
    merge_after: tuple[str, ...]
    review_only: bool
    raw: Mapping[str, Any] = field(repr=False)


def _normalize_task(value: Any, index: int) -> _Task:
    root = _mapping(value)
    resource = _mapping(root.get("resource_contract"))
    provider = _mapping(root.get("provider_contract"))
    conflict = _mapping(root.get("conflict_contract"))
    lease = _mapping(root.get("lease_contract"))
    worktree = _mapping(root.get("worktree_contract"))
    merge = _mapping(root.get("merge_strategy"))
    sources = (root, conflict, resource, provider, lease, worktree, merge)
    task_id = str(
        _first(
            (root,),
            "task_cid",
            "canonical_task_cid",
            "task_id",
            "id",
            "canonical_task_key",
            default=f"task:{index}",
        )
    ).strip()
    if not task_id or len(task_id.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ParallelPlanError("task identity is empty or too large")
    aliases = _strings(
        (
            task_id,
            root.get("task_id", ""),
            root.get("id", ""),
            root.get("task_cid", ""),
            root.get("canonical_task_cid", ""),
            root.get("canonical_task_key", ""),
        )
    )
    outputs = _union_paths(sources, "outputs", "output_paths", "expected_outputs")
    predicted = _union_paths(
        sources,
        "predicted_paths",
        "predicted_files",
        "files",
        "predicted_directories",
    )
    changed = _union_paths(sources, "changed_paths")
    generated = _union_paths(sources, "generated_artifacts", "generated_outputs")
    submodules = _union_paths(sources, "submodules", "submodule_paths")
    protected = _union_paths(sources, "protected_paths")
    exclusive = _union_paths(sources, "exclusive_paths")
    duration = _integer(
        _first(
            sources,
            "duration_ms",
            "estimated_duration_ms",
            "wall_time_ms",
            default=0,
        ),
        "duration_ms",
    )
    if not duration:
        seconds = _integer(
            _first(sources, "estimated_validation_seconds", "validation_seconds", default=0),
            "estimated_validation_seconds",
        )
        duration = seconds * 1000
    return _Task(
        task_id=task_id,
        aliases=aliases,
        dependencies=_union(sources, "dependencies", "depends_on", "dependency_task_cids"),
        outputs=outputs,
        paths=tuple(sorted(set(outputs) | set(predicted) | set(changed))),
        symbols=_union(sources, "predicted_symbols", "ast_symbols", "symbols"),
        interfaces=_union(sources, "interfaces", "interface_ids"),
        submodules=submodules,
        generated_artifacts=generated,
        protected_paths=protected,
        exclusive_paths=exclusive,
        produces=_union(
            sources,
            "produces",
            "produced_leaf_ids",
            "producer_for",
            "output_ids",
            "effect_ids",
            "producer_leaf_ids",
            "leaf_ids",
            "effect_subset",
            "effects",
            "expected_effects",
        ),
        required_leaf_ids=_union(sources, "required_leaf_ids", "leaf_obligation_ids"),
        validation=_union(
            sources,
            "validation_commands",
            "post_merge_validation",
            "post_merge_validation_cids",
        ),
        resource_class=str(_first(sources, "resource_class", default="cpu-small")),
        stage=str(_first(sources, "resource_stage", "stage", default="execution")),
        cpu_slots=_integer(_first(sources, "cpu_slots", default=1), "cpu_slots", minimum=1),
        process_slots=_integer(
            _first(sources, "process_slots", default=1), "process_slots", minimum=1
        ),
        memory_bytes=_integer(_first(sources, "memory_bytes", default=0), "memory_bytes"),
        gpu_memory_bytes=_integer(
            _first(sources, "gpu_memory_bytes", default=0), "gpu_memory_bytes"
        ),
        disk_bytes=_integer(_first(sources, "disk_bytes", default=0), "disk_bytes"),
        duration_ms=duration,
        provider_requirement=str(
            _first(sources, "provider_requirement", "provider_id", "provider", default="")
        ).strip(),
        required_capabilities=_union(sources, "required_capabilities", "capabilities"),
        context_tokens=_integer(
            _first(sources, "context_tokens", "estimated_context_tokens", default=0),
            "context_tokens",
        ),
        output_tokens=_integer(
            _first(
                sources,
                "output_token_budget",
                "token_budget",
                "estimated_tokens",
                default=0,
            ),
            "output_tokens",
        ),
        quota_units=_integer(_first(sources, "quota_units", default=0), "quota_units"),
        cost_micros=_integer(
            _first(sources, "cost_limit_micros", "cost_micros", default=0),
            "cost_micros",
        ),
        max_provider_latency_ms=_integer(
            _first(sources, "max_provider_latency_ms", default=0),
            "max_provider_latency_ms",
        ),
        affinity_key=str(_first(sources, "affinity_key", default="")).strip(),
        anti_affinity_key=str(_first(sources, "anti_affinity_key", default="")).strip(),
        exclusive_group=str(_first(sources, "exclusive_group", default="")).strip(),
        shard_key=str(_first(sources, "shard_key", default="")).strip(),
        lane_label=str(_first(sources, "parallel_lane", "lane_label", default="")).strip(),
        lane_authoritative=_boolean(
            _first(sources, "lane_authoritative", "parallel_lane_authoritative", default=False)
        ),
        claimed_parallel_with=_union(sources, "claimed_parallel_with", "parallel_with"),
        worktree_policy=str(
            worktree.get("policy")
            or root.get("worktree_policy")
            or "isolated"
        ),
        worktree_bytes=_integer(
            _first(sources, "max_worktree_bytes", "worktree_bytes", default=0),
            "worktree_bytes",
        ),
        expected_base_revision=str(
            _first(sources, "expected_base_revision", default="")
        ).strip(),
        expected_merge_target=str(
            _first(sources, "expected_merge_target", default="")
        ).strip(),
        lease_scope=str(_first(sources, "lease_scope", default="task")),
        lease_duration_ms=_integer(
            _first(sources, "lease_duration_ms", default=max(duration * 2, 60_000)),
            "lease_duration_ms",
        ),
        heartbeat_interval_ms=_integer(
            _first(sources, "heartbeat_interval_ms", default=0),
            "heartbeat_interval_ms",
        ),
        lease_owner_rule=str(
            _first(sources, "owner_identity_rule", default="lane-owner")
        ).strip(),
        fence_epoch=_integer(_first(sources, "fencing_epoch", default=0), "fencing_epoch"),
        merge_train_id=str(_first(sources, "merge_train_id", default="merge-train:default")),
        merge_after=_union(
            sources,
            "merge_after",
            "merge_dependencies",
            "ordering_constraints",
        ),
        review_only=_boolean(_first(sources, "review_only", default=False)),
        raw=root,
    )


def _topological_waves(
    tasks: Sequence[_Task], completed: set[str]
) -> tuple[tuple[tuple[str, ...], ...], dict[str, tuple[str, ...]], list[ParallelPlanIssue]]:
    aliases: dict[str, str] = {}
    issues: list[ParallelPlanIssue] = []
    for task in tasks:
        for alias in task.aliases:
            prior = aliases.get(alias)
            if prior and prior != task.task_id:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.DUPLICATE_TASK_ID,
                        f"task alias {alias!r} identifies more than one task",
                        tuple(sorted({prior, task.task_id})),
                        (alias,),
                    )
                )
            aliases[alias] = task.task_id
    dependencies: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        resolved: set[str] = set()
        for dependency in task.dependencies:
            target = aliases.get(dependency)
            if target:
                resolved.add(target)
            elif dependency in completed:
                continue
            else:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.UNKNOWN_DEPENDENCY,
                        f"task {task.task_id} depends on unknown task {dependency}",
                        (task.task_id,),
                        (dependency,),
                    )
                )
        dependencies[task.task_id] = tuple(sorted(resolved))
    if issues:
        return (), dependencies, issues
    remaining = set(dependencies)
    completed_local: set[str] = set()
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(
                task_id
                for task_id in remaining
                if set(dependencies[task_id]).issubset(completed_local)
            )
        )
        if not ready:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.DEPENDENCY_CYCLE,
                    "task dependency graph contains a cycle",
                    tuple(sorted(remaining)),
                )
            )
            return (), dependencies, issues
        waves.append(ready)
        completed_local.update(ready)
        remaining.difference_update(ready)
    return tuple(waves), dependencies, issues


def _critical_path(
    tasks: Sequence[_Task],
    waves: Sequence[Sequence[str]],
    dependencies: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], int]:
    by_id = {task.task_id: task for task in tasks}
    best_duration: dict[str, int] = {}
    best_path: dict[str, tuple[str, ...]] = {}
    for wave in waves:
        for task_id in wave:
            candidates = [
                (best_duration[dep], best_path[dep])
                for dep in dependencies[task_id]
            ]
            prior_duration, prior_path = max(
                candidates, key=lambda item: (item[0], tuple(reversed(item[1])))
            ) if candidates else (0, ())
            best_duration[task_id] = prior_duration + by_id[task_id].duration_ms
            best_path[task_id] = (*prior_path, task_id)
    if not best_duration:
        return (), 0
    terminal = min(
        (
            task_id
            for task_id, duration in best_duration.items()
            if duration == max(best_duration.values())
        ),
        key=lambda task_id: best_path[task_id],
    )
    return best_path[terminal], best_duration[terminal]


def _merge_sequence(
    tasks: Sequence[_Task],
    dependencies: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]], list[ParallelPlanIssue]]:
    """Topologically order the serialized merge train, including merge hints."""

    aliases = {alias: task.task_id for task in tasks for alias in task.aliases}
    merge_dependencies: dict[str, set[str]] = {
        task.task_id: set(dependencies.get(task.task_id, ())) for task in tasks
    }
    for task in tasks:
        for raw in task.merge_after:
            target = aliases.get(raw)
            if target:
                merge_dependencies[task.task_id].add(target)
    remaining = set(merge_dependencies)
    merged: set[str] = set()
    order: list[str] = []
    while remaining:
        ready = sorted(
            task_id
            for task_id in remaining
            if merge_dependencies[task_id].issubset(merged)
        )
        if not ready:
            return (), {
                key: tuple(sorted(value))
                for key, value in merge_dependencies.items()
            }, [
                ParallelPlanIssue(
                    ParallelPlanIssueCode.DEPENDENCY_CYCLE,
                    "merge ordering constraints contain a cycle",
                    tuple(sorted(remaining)),
                )
            ]
        order.extend(ready)
        merged.update(ready)
        remaining.difference_update(ready)
    return tuple(order), {
        key: tuple(sorted(value)) for key, value in merge_dependencies.items()
    }, []


def _leaf_closure(
    tasks: Sequence[_Task],
    dependencies: Mapping[str, Sequence[str]],
    explicit_required: Sequence[str],
) -> tuple[LeafProducerClosure, list[ParallelPlanIssue]]:
    depended_on = {dependency for values in dependencies.values() for dependency in values}
    terminal = tuple(sorted(task.task_id for task in tasks if task.task_id not in depended_on))
    required = set(explicit_required)
    for task in tasks:
        required.update(task.required_leaf_ids)
    producers: dict[str, str] = {}
    duplicate: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for leaf in task.produces:
            duplicate[leaf].add(task.task_id)
            producers.setdefault(leaf, task.task_id)
    issues: list[ParallelPlanIssue] = []
    missing = sorted(required - set(producers))
    if missing:
        issues.append(
            ParallelPlanIssue(
                ParallelPlanIssueCode.MISSING_LEAF_PRODUCER,
                "required leaf obligations do not have a task producer",
                (),
                tuple(missing),
            )
        )
    for leaf, task_ids in sorted(duplicate.items()):
        if len(task_ids) > 1:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.OUTPUT_COLLISION,
                    f"leaf {leaf} has multiple producers",
                    tuple(sorted(task_ids)),
                    (leaf,),
                )
            )
    # When no explicit logical leaves exist, terminal mutation/validation tasks
    # themselves are the leaf producers.  An empty terminal contract is not a
    # proof of completion.
    if not required:
        by_id = {task.task_id: task for task in tasks}
        empty = [
            task_id
            for task_id in terminal
            if not (
                by_id[task_id].outputs
                or by_id[task_id].produces
                or by_id[task_id].validation
                or by_id[task_id].review_only
            )
        ]
        if empty:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.MISSING_LEAF_PRODUCER,
                    "terminal tasks must produce output, evidence, or validation",
                    tuple(empty),
                )
            )
        producers.update({f"terminal:{task_id}": task_id for task_id in terminal if task_id not in empty})
        required.update(f"terminal:{task_id}" for task_id in terminal)
    closure = LeafProducerClosure(
        required_leaf_ids=tuple(sorted(required)),
        producer_by_leaf_id=dict(sorted(producers.items())),
        terminal_task_ids=terminal,
        closed=not any(issue.code is ParallelPlanIssueCode.MISSING_LEAF_PRODUCER for issue in issues),
    )
    return closure, issues


def _conflicts(
    tasks: Sequence[_Task],
    *,
    protected_paths: Sequence[str],
    submodule_paths: Sequence[str],
    receipts: Sequence[Any],
) -> tuple[tuple[ConflictSurfaceRecord, ...], list[ParallelPlanIssue]]:
    graph_tasks = [
        {
            "task_id": task.task_id,
            "task_cid": task.task_id,
            "files": list(task.paths),
            "predicted_paths": list(task.paths),
            "predicted_symbols": list(task.symbols),
            "ast_symbols": list(task.symbols),
            "global_ast_symbols": list(task.symbols),
            "interfaces": list(task.interfaces),
            "submodules": list(task.submodules),
            "generated_artifacts": list(task.generated_artifacts),
        }
        for task in tasks
    ]
    graph = materialize_task_conflict_graph(
        graph_tasks,
        conflict_receipts=receipts,
    )
    edge_by_pair = {
        tuple(sorted((edge.left_task_cid, edge.right_task_cid))): edge
        for edge in graph.edges
        if edge.blocks_concurrency
    }
    global_protected = tuple(sorted(set(protected_paths)))
    global_submodules = tuple(sorted(set(submodule_paths)))
    records: list[ConflictSurfaceRecord] = []
    issues: list[ParallelPlanIssue] = []
    for index, left in enumerate(tasks):
        for right in tasks[index + 1 :]:
            pair = tuple(sorted((left.task_id, right.task_id)))
            edge = edge_by_pair.get(pair)
            left_all_paths = tuple(
                sorted(
                    set(left.paths)
                    | set(left.generated_artifacts)
                    | set(left.submodules)
                )
            )
            right_all_paths = tuple(
                sorted(
                    set(right.paths)
                    | set(right.generated_artifacts)
                    | set(right.submodules)
                )
            )
            paths = _path_overlaps(left_all_paths, right_all_paths)
            symbols = tuple(sorted(set(left.symbols) & set(right.symbols)))
            interfaces = tuple(sorted(set(left.interfaces) & set(right.interfaces)))
            left_submodules = tuple(
                sorted(set(left.submodules) | {p for p in global_submodules if any(_overlap(p, path) for path in left.paths)})
            )
            right_submodules = tuple(
                sorted(set(right.submodules) | {p for p in global_submodules if any(_overlap(p, path) for path in right.paths)})
            )
            submodules = _path_overlaps(left_submodules, right_submodules)
            generated = tuple(
                sorted(
                    set(
                        _path_overlaps(
                            left.generated_artifacts, right_all_paths
                        )
                    )
                    | set(
                        _path_overlaps(
                            right.generated_artifacts, left_all_paths
                        )
                    )
                )
            )
            left_protected = tuple(
                sorted(set(left.protected_paths) | {p for p in global_protected if any(_overlap(p, path) for path in left.paths)})
            )
            right_protected = tuple(
                sorted(set(right.protected_paths) | {p for p in global_protected if any(_overlap(p, path) for path in right.paths)})
            )
            protected = _path_overlaps(left_protected, right_protected)
            exclusive = tuple(
                sorted(
                    set(_path_overlaps(left.exclusive_paths, right_all_paths))
                    | set(_path_overlaps(right.exclusive_paths, left_all_paths))
                    | set(_path_overlaps(left.exclusive_paths, right.exclusive_paths))
                )
            )
            groups = (
                (left.exclusive_group,)
                if left.exclusive_group and left.exclusive_group == right.exclusive_group
                else ()
            )
            anti_affinity = (
                (left.anti_affinity_key,)
                if left.anti_affinity_key and left.anti_affinity_key == right.anti_affinity_key
                else ()
            )
            observed = tuple(
                sorted(
                    reason
                    for reason in (edge.reasons if edge else ())
                    if reason.startswith(("conflict_receipt", "observed "))
                )
            )
            blocking = bool(
                edge
                or paths
                or symbols
                or interfaces
                or submodules
                or generated
                or protected
                or exclusive
                or groups
                or anti_affinity
            )
            if blocking:
                records.append(
                    ConflictSurfaceRecord(
                        left_task_id=pair[0],
                        right_task_id=pair[1],
                        paths=paths,
                        symbols=symbols,
                        interfaces=interfaces,
                        submodules=submodules,
                        generated_artifacts=generated,
                        protected_paths=protected,
                        exclusive_paths=exclusive,
                        exclusive_groups=groups,
                        anti_affinity_keys=anti_affinity,
                        observed_receipts=observed,
                    )
                )
            if _path_overlaps(left.outputs, right.outputs):
                overlap = _path_overlaps(left.outputs, right.outputs)
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.OUTPUT_COLLISION,
                        "two tasks declare overlapping mutation outputs",
                        pair,
                        overlap,
                    )
                )
            if protected:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.PROTECTED_BOTTLENECK,
                        "multiple tasks require the same protected mutation surface",
                        pair,
                        protected,
                    )
                )
            if submodules:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.OVERLAPPING_SUBMODULES,
                        "tasks overlap the same recursive submodule surface",
                        pair,
                        submodules,
                    )
                )
    return tuple(sorted(records, key=lambda item: (item.left_task_id, item.right_task_id))), issues


def _color_wave(
    wave: Sequence[str], conflicts: Sequence[ConflictSurfaceRecord]
) -> tuple[tuple[str, ...], ...]:
    nodes = tuple(sorted(wave))
    adjacency = {task_id: set() for task_id in nodes}
    for conflict in conflicts:
        if conflict.left_task_id in adjacency and conflict.right_task_id in adjacency:
            adjacency[conflict.left_task_id].add(conflict.right_task_id)
            adjacency[conflict.right_task_id].add(conflict.left_task_id)
    colors: dict[str, int] = {}
    for task_id in sorted(nodes, key=lambda item: (-len(adjacency[item]), item)):
        unavailable = {colors[peer] for peer in adjacency[task_id] if peer in colors}
        color = 0
        while color in unavailable:
            color += 1
        colors[task_id] = color
    return tuple(
        tuple(sorted(task_id for task_id, assigned in colors.items() if assigned == color))
        for color in range(max(colors.values(), default=-1) + 1)
    )


def _snapshot_id(prefix: str, snapshot: Mapping[str, Any]) -> str:
    explicit = str(
        snapshot.get("snapshot_id")
        or snapshot.get("capacity_snapshot_id")
        or snapshot.get("provider_snapshot_id")
        or snapshot.get("tree_id")
        or snapshot.get("repository_tree_id")
        or ""
    ).strip()
    return explicit or _digest(prefix, snapshot)


def _provider_snapshot_records(value: Any) -> tuple[dict[str, Any], ...]:
    """Accept provider sequences and the runtime's provider-keyed projection."""

    if value in (None, "", (), [], {}):
        return ()
    if isinstance(value, Mapping):
        payload = dict(value)
        nested = payload.get("providers")
        if nested not in (None, "", (), [], {}):
            return _provider_snapshot_records(nested)
        identity_fields = {
            "provider_id",
            "provider",
            "id",
            "name",
            "effective_provider_name",
        }
        if identity_fields.intersection(payload):
            return (payload,)
        records: list[dict[str, Any]] = []
        for provider_id, raw in sorted(payload.items(), key=lambda item: str(item[0])):
            record = _mapping(raw)
            record.setdefault("provider_id", str(provider_id))
            records.append(record)
        return tuple(records)
    return tuple(_mapping(item) for item in _sequence(value))


def _provider_capacity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    active = _integer(
        snapshot.get("active_requests", snapshot.get("in_flight", 0)),
        "provider.active_requests",
    )
    available_value = snapshot.get(
        "available_slots",
        snapshot.get(
            "available_concurrency",
            snapshot.get("available_capacity", snapshot.get("free_slots")),
        ),
    )
    if available_value is None:
        maximum = _integer(
            snapshot.get(
                "max_concurrency",
                snapshot.get("concurrency_limit", snapshot.get("concurrency", 0)),
            ),
            "provider.max_concurrency",
        )
        available_slots = max(0, maximum - active)
    else:
        available_slots = _integer(available_value, "provider.available_slots")

    def capacity_value(*names: str) -> int:
        raw: Any = None
        for name in names:
            if name in snapshot:
                raw = snapshot[name]
                break
        # Runtime uses -1 for unknown.  It is deliberately projected to zero,
        # never to unlimited, at this planning boundary.
        if raw in (None, ""):
            return 0
        if not isinstance(raw, bool):
            try:
                if int(raw) < 0:
                    return 0
            except (TypeError, ValueError):
                pass
        return _integer(raw, f"provider.{names[0]}")

    return {
        "provider_id": str(
            snapshot.get("provider_id")
            or snapshot.get("provider")
            or snapshot.get("id")
            or snapshot.get("name")
            or snapshot.get("effective_provider_name")
            or ""
        ).strip().casefold(),
        "available_slots": available_slots,
        "context_limit": capacity_value(
            "context_limit", "context_window_tokens", "max_context_tokens", "context_tokens"
        ),
        "available_tokens": capacity_value(
            "available_tokens", "token_budget_remaining", "token_headroom", "token_limit"
        ),
        "available_quota": capacity_value(
            "available_quota", "quota_remaining", "quota_headroom", "quota_units"
        ),
        "available_cost_micros": capacity_value(
            "available_cost_micros", "cost_headroom_micros", "cost_limit_micros"
        ),
        "latency_ms": _integer(
            snapshot.get("latency_ms", snapshot.get("provider_latency_ms", 0)),
            "provider.latency_ms",
        ),
        "capabilities": set(
            _strings(snapshot.get("capabilities", snapshot.get("supported_capabilities", ())))
        ),
        "healthy": _health(
            snapshot.get("healthy", snapshot.get("status", snapshot.get("state", True)))
        ) and capacity_value("retry_after_ms") == 0,
    }


def _select_provider(
    task: _Task,
    providers: Sequence[Mapping[str, Any]],
) -> tuple[str, ParallelPlanIssue | None]:
    needs_provider = bool(
        task.provider_requirement
        or task.context_tokens
        or task.output_tokens
        or task.quota_units
        or task.cost_micros
        or any(capability.startswith("llm:") for capability in task.required_capabilities)
    )
    if not needs_provider:
        return "", None
    candidates: list[dict[str, Any]] = []
    failure_codes: set[ParallelPlanIssueCode] = set()
    evidence: set[str] = set()
    for raw in providers:
        provider = _provider_capacity(raw)
        provider_id = provider["provider_id"]
        if task.provider_requirement and provider_id != task.provider_requirement:
            continue
        if not provider["healthy"] or provider["available_slots"] < 1:
            failure_codes.add(ParallelPlanIssueCode.PROVIDER_INFEASIBLE)
            evidence.add(provider_id or "provider:unknown")
            continue
        if not set(task.required_capabilities).issubset(provider["capabilities"]):
            failure_codes.add(ParallelPlanIssueCode.PROVIDER_INFEASIBLE)
            evidence.add(f"{provider_id}:capability")
            continue
        if task.context_tokens > provider["context_limit"]:
            failure_codes.add(ParallelPlanIssueCode.CONTEXT_INFEASIBLE)
            evidence.add(f"{provider_id}:context")
            continue
        if task.output_tokens > provider["available_tokens"]:
            failure_codes.add(ParallelPlanIssueCode.TOKEN_INFEASIBLE)
            evidence.add(f"{provider_id}:tokens")
            continue
        if task.quota_units > provider["available_quota"]:
            failure_codes.add(ParallelPlanIssueCode.PROVIDER_INFEASIBLE)
            evidence.add(f"{provider_id}:quota")
            continue
        if task.cost_micros > provider["available_cost_micros"]:
            failure_codes.add(ParallelPlanIssueCode.COST_INFEASIBLE)
            evidence.add(f"{provider_id}:cost")
            continue
        if task.max_provider_latency_ms and provider["latency_ms"] > task.max_provider_latency_ms:
            failure_codes.add(ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE)
            evidence.add(f"{provider_id}:latency")
            continue
        candidates.append(provider)
    if not candidates:
        code = (
            min(failure_codes, key=lambda item: item.value)
            if failure_codes
            else ParallelPlanIssueCode.PROVIDER_INFEASIBLE
        )
        return "", ParallelPlanIssue(
            code,
            f"no provider can satisfy task {task.task_id}",
            (task.task_id,),
            tuple(sorted(evidence)),
        )
    selected = min(candidates, key=lambda item: item["provider_id"])
    return str(selected["provider_id"]), None


def _fits(
    tasks: Sequence[_Task],
    provider_by_task: Mapping[str, str],
    capacity: Mapping[str, Any],
    providers: Mapping[str, Mapping[str, Any]],
) -> bool:
    if sum(task.cpu_slots for task in tasks) > capacity["cpu_slots"]:
        return False
    if sum(task.process_slots for task in tasks) > capacity["process_slots"]:
        return False
    for name in ("memory_bytes", "gpu_memory_bytes", "disk_bytes"):
        limit = capacity[name]
        if sum(getattr(task, name) for task in tasks) > limit:
            return False
    advertised_classes = capacity.get("resource_classes", set())
    if advertised_classes and any(
        task.resource_class not in advertised_classes for task in tasks
    ):
        return False
    host_capabilities = capacity.get("capabilities", set())
    for task in tasks:
        if provider_by_task.get(task.task_id):
            continue
        local_requirements = {
            capability
            for capability in task.required_capabilities
            if not capability.startswith("llm:")
        }
        if local_requirements and not local_requirements.issubset(host_capabilities):
            return False
    class_slots = capacity.get("resource_class_slots", {})
    for resource_class in {task.resource_class for task in tasks}:
        if resource_class in class_slots:
            required = sum(
                task.process_slots
                for task in tasks
                if task.resource_class == resource_class
            )
            if required > int(class_slots[resource_class]):
                return False
    grouped: dict[str, list[_Task]] = defaultdict(list)
    for task in tasks:
        provider_id = provider_by_task.get(task.task_id, "")
        if provider_id:
            grouped[provider_id].append(task)
    for provider_id, members in grouped.items():
        provider = providers[provider_id]
        if len(members) > provider["available_slots"]:
            return False
        if sum(task.output_tokens for task in members) > provider["available_tokens"]:
            return False
        if sum(task.quota_units for task in members) > provider["available_quota"]:
            return False
        if sum(task.cost_micros for task in members) > provider["available_cost_micros"]:
            return False
    return True


def _wave_usage(
    tasks: Sequence[_Task], provider_by_task: Mapping[str, str]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    resource = {
        "cpu_slots": sum(task.cpu_slots for task in tasks),
        "process_slots": sum(task.process_slots for task in tasks),
        "memory_bytes": sum(task.memory_bytes for task in tasks),
        "gpu_memory_bytes": sum(task.gpu_memory_bytes for task in tasks),
        "disk_bytes": sum(task.disk_bytes for task in tasks),
        "context_tokens": sum(task.context_tokens for task in tasks),
        "output_tokens": sum(task.output_tokens for task in tasks),
        "cost_micros": sum(task.cost_micros for task in tasks),
    }
    provider: dict[str, dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "context_tokens": 0, "output_tokens": 0, "quota_units": 0, "cost_micros": 0}
    )
    for task in tasks:
        provider_id = provider_by_task.get(task.task_id, "")
        if not provider_id:
            continue
        usage = provider[provider_id]
        usage["requests"] += 1
        usage["context_tokens"] += task.context_tokens
        usage["output_tokens"] += task.output_tokens
        usage["quota_units"] += task.quota_units
        usage["cost_micros"] += task.cost_micros
    return resource, dict(provider)


class ParallelPlanCompiler:
    """Deterministically compile one admitted task population."""

    interface = PARALLEL_PLAN_COMPILER_INTERFACE

    def compile(
        self,
        request: ParallelPlanCompilationRequest | Sequence[Any] | None = None,
        *,
        tasks: Sequence[Any] | None = None,
        requested_width: int = 1,
        repository_snapshot: Any = None,
        capacity_snapshot: Any = None,
        provider_snapshots: Sequence[Any] = (),
        budget: Any = None,
        current_time_ms: int = 0,
        deadline_ms: int = 0,
        required_leaf_ids: Sequence[str] = (),
        completed_task_ids: Sequence[str] = (),
        protected_paths: Sequence[str] = (),
        submodule_paths: Sequence[str] = (),
        post_merge_validation: Sequence[str] = (),
        conflict_receipts: Sequence[Any] = (),
        base_fence_epoch: int = 0,
        review_only: bool = False,
        raise_on_rejection: bool = False,
    ) -> ParallelExecutionPlan:
        if isinstance(request, ParallelPlanCompilationRequest):
            compilation = request
        else:
            population = tasks if tasks is not None else (request or ())
            compilation = ParallelPlanCompilationRequest(
                tasks=tuple(population),
                requested_width=requested_width,
                repository_snapshot=repository_snapshot or {},
                capacity_snapshot=capacity_snapshot or {},
                provider_snapshots=_provider_snapshot_records(provider_snapshots),
                budget=budget or {},
                current_time_ms=current_time_ms,
                deadline_ms=deadline_ms,
                required_leaf_ids=tuple(required_leaf_ids),
                completed_task_ids=tuple(completed_task_ids),
                protected_paths=tuple(protected_paths),
                submodule_paths=tuple(submodule_paths),
                post_merge_validation=tuple(post_merge_validation),
                conflict_receipts=tuple(conflict_receipts),
                base_fence_epoch=base_fence_epoch,
                review_only=review_only,
            )
        plan = self._compile(compilation)
        if raise_on_rejection and not plan.admitted:
            raise ParallelPlanRejectedError(plan)
        return plan

    compile_plan = compile

    def _compile(self, request: ParallelPlanCompilationRequest) -> ParallelExecutionPlan:
        replay_request = request.to_replay_dict()
        input_digest = _digest("parallel-plan-input", replay_request)
        repository = _mapping(request.repository_snapshot)
        capacity_raw = _mapping(request.capacity_snapshot)
        # ``ResourceScheduleSnapshot`` nests the measured host while the
        # historical host-only contract exposes these fields at top level.
        # Flatten only for evaluation; replay continues to bind the complete
        # original snapshot.
        capacity_source = {
            **_mapping(capacity_raw.get("host")),
            **capacity_raw,
        }
        budget = _mapping(request.budget)
        providers_raw = list(
            _provider_snapshot_records(
                request.provider_snapshots or capacity_raw.get("providers")
            )
        )
        repository_tree_id = str(
            repository.get("tree_id")
            or repository.get("repository_tree_id")
            or repository.get("effective_tree_id")
            or repository.get("dirty_worktree_root")
            or repository.get("repository_root_cid")
            or ""
        ).strip()
        capacity_snapshot_id = _snapshot_id("capacity-snapshot", capacity_raw) if capacity_raw else ""
        provider_snapshot_ids = tuple(
            _snapshot_id("provider-snapshot", snapshot) for snapshot in providers_raw
        )
        issues: list[ParallelPlanIssue] = []
        if not request.tasks:
            issues.append(
                ParallelPlanIssue(ParallelPlanIssueCode.EMPTY_PLAN, "task population is empty")
            )
        provider_ids = [
            _provider_capacity(snapshot)["provider_id"]
            for snapshot in providers_raw
        ]
        if any(not provider_id for provider_id in provider_ids) or len(
            provider_ids
        ) != len(set(provider_ids)):
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.INVALID_SNAPSHOT,
                    "provider snapshot identities must be non-empty and unique",
                    evidence=tuple(provider_ids),
                )
            )
        normalized = tuple(_normalize_task(task, index) for index, task in enumerate(request.tasks))
        all_review = bool(normalized) and (
            request.review_only or all(task.review_only for task in normalized)
        )
        if normalized and not all_review and not repository_tree_id:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.INVALID_SNAPSHOT,
                    "an executable plan must bind an exact repository tree",
                )
            )
        ids = [task.task_id for task in normalized]
        if len(ids) != len(set(ids)):
            duplicates = tuple(sorted(task_id for task_id in set(ids) if ids.count(task_id) > 1))
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.DUPLICATE_TASK_ID,
                    "task identities must be unique",
                    duplicates,
                )
            )
        waves, dependencies, graph_issues = _topological_waves(
            normalized, set(request.completed_task_ids)
        )
        issues.extend(graph_issues)
        closure, closure_issues = _leaf_closure(
            normalized, dependencies, request.required_leaf_ids
        )
        issues.extend(closure_issues)
        repository_protected = _paths(repository.get("protected_paths", ()))
        repository_submodules = _paths(repository.get("submodule_paths", repository.get("submodules", ())))
        conflicts, conflict_issues = _conflicts(
            normalized,
            protected_paths=tuple(
                sorted(
                    set(request.protected_paths)
                    | set(repository_protected)
                    | {path for task in normalized for path in task.protected_paths}
                )
            ),
            submodule_paths=tuple(
                sorted(
                    set(request.submodule_paths)
                    | set(repository_submodules)
                    | {path for task in normalized for path in task.submodules}
                )
            ),
            receipts=request.conflict_receipts,
        )
        issues.extend(conflict_issues)

        conflict_pairs = {
            tuple(sorted((item.left_task_id, item.right_task_id))) for item in conflicts if item.blocking
        }
        aliases = {alias: task.task_id for task in normalized for alias in task.aliases}
        for task in normalized:
            if task.lane_authoritative:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.FAKE_LANE_LABEL,
                        "lane labels are hints and cannot be marked authoritative",
                        (task.task_id,),
                        (task.lane_label,) if task.lane_label else (),
                    )
                )
            for claimed in task.claimed_parallel_with:
                peer = aliases.get(claimed, claimed)
                if tuple(sorted((task.task_id, peer))) in conflict_pairs:
                    issues.append(
                        ParallelPlanIssue(
                            ParallelPlanIssueCode.FAKE_LANE_LABEL,
                            "claimed parallel peers have a compiled blocking conflict",
                            tuple(sorted((task.task_id, peer))),
                            (task.lane_label,) if task.lane_label else (),
                        )
                    )

        critical_path, critical_duration = _critical_path(normalized, waves, dependencies)
        merge_task_order, merge_dependencies, merge_issues = _merge_sequence(
            normalized, dependencies
        )
        issues.extend(merge_issues)
        now = request.current_time_ms or _integer(
            capacity_source.get(
                "observed_at_ms",
                capacity_source.get(
                    "measured_at_ms", capacity_source.get("timestamp_ms", 0)
                ),
            ),
            "observed_at_ms",
        )
        deadline = request.deadline_ms or _integer(budget.get("deadline_ms", 0), "deadline_ms")
        budget_latency = _integer(budget.get("max_latency_ms", 0), "max_latency_ms")
        if not deadline and budget_latency and now:
            deadline = now + budget_latency
        if deadline and (deadline <= now or now + critical_duration > deadline):
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE,
                    "critical path cannot complete before the bound deadline",
                    critical_path,
                    (f"now_ms={now}", f"deadline_ms={deadline}", f"critical_path_ms={critical_duration}"),
                )
            )
        elif deadline and any(task.duration_ms == 0 for task in normalized):
            unknown_duration_tasks = tuple(
                task.task_id for task in normalized if task.duration_ms == 0
            )
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE,
                    "deadline feasibility cannot be proved with unknown task durations",
                    unknown_duration_tasks,
                )
            )

        if capacity_raw:
            observed = _integer(
                capacity_source.get(
                    "observed_at_ms",
                    capacity_source.get(
                        "measured_at_ms", capacity_source.get("timestamp_ms", 0)
                    ),
                ),
                "capacity.observed_at_ms",
            )
            fresh_until = _integer(capacity_source.get("fresh_until_ms", 0), "capacity.fresh_until_ms")
            max_age = _integer(
                capacity_source.get("max_age_ms", DEFAULT_CAPACITY_MAX_AGE_MS),
                "capacity.max_age_ms",
            )
            explicitly_stale = _boolean(capacity_source.get("stale", False))
            if explicitly_stale or not observed or (now and observed > now) or (fresh_until and now >= fresh_until) or (max_age and now - observed > max_age):
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.STALE_CAPACITY,
                        "host capacity snapshot is missing freshness evidence or is stale",
                        evidence=(f"observed_at_ms={observed}", f"current_time_ms={now}", f"fresh_until_ms={fresh_until}"),
                    )
                )
        elif normalized and not all_review:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.INVALID_SNAPSHOT,
                    "an executable plan requires a current capacity snapshot",
                )
            )

        for snapshot in providers_raw:
            observed = _integer(
                snapshot.get(
                    "observed_at_ms",
                    snapshot.get(
                        "measured_at_ms",
                        snapshot.get(
                            "timestamp_ms", capacity_source.get("observed_at_ms", 0)
                        ),
                    ),
                ),
                "provider.observed_at_ms",
            )
            fresh_until = _integer(snapshot.get("fresh_until_ms", capacity_source.get("fresh_until_ms", 0)), "provider.fresh_until_ms")
            explicitly_stale = _boolean(snapshot.get("stale", False))
            max_age = _integer(
                snapshot.get(
                    "max_age_ms",
                    capacity_source.get("max_age_ms", DEFAULT_CAPACITY_MAX_AGE_MS),
                ),
                "provider.max_age_ms",
            )
            if explicitly_stale or not observed or (now and observed > now) or (fresh_until and now >= fresh_until) or (now and max_age and now - observed > max_age):
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.STALE_CAPACITY,
                        "provider capacity snapshot is missing freshness evidence or is stale",
                        evidence=(str(snapshot.get("provider_id") or snapshot.get("id") or "provider:unknown"),),
                    )
                )

        provider_by_task: dict[str, str] = {}
        for task in normalized:
            selected, provider_issue = _select_provider(task, providers_raw)
            provider_by_task[task.task_id] = selected
            if provider_issue:
                issues.append(provider_issue)

        provider_totals: dict[str, list[_Task]] = defaultdict(list)
        for task in normalized:
            if provider_by_task.get(task.task_id):
                provider_totals[provider_by_task[task.task_id]].append(task)
        provider_caps = {
            provider["provider_id"]: provider
            for provider in (_provider_capacity(item) for item in providers_raw)
            if provider["provider_id"]
        }
        for provider_id, members in sorted(provider_totals.items()):
            provider = provider_caps[provider_id]
            for required, limit, code, dimension in (
                (sum(task.output_tokens for task in members), provider["available_tokens"], ParallelPlanIssueCode.TOKEN_INFEASIBLE, "tokens"),
                (sum(task.quota_units for task in members), provider["available_quota"], ParallelPlanIssueCode.PROVIDER_INFEASIBLE, "quota"),
                (sum(task.cost_micros for task in members), provider["available_cost_micros"], ParallelPlanIssueCode.COST_INFEASIBLE, "cost"),
            ):
                if required > limit:
                    issues.append(
                        ParallelPlanIssue(
                            code,
                            f"provider {provider_id} lacks cumulative {dimension} headroom",
                            tuple(task.task_id for task in members),
                            (f"required={required}", f"available={limit}"),
                        )
                    )

        max_provider_tokens = _integer(budget.get("max_provider_tokens", 0), "max_provider_tokens")
        max_cost = _integer(budget.get("max_cost_micros", 0), "max_cost_micros")
        if "max_provider_tokens" in budget and sum(task.output_tokens for task in normalized) > max_provider_tokens:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.TOKEN_INFEASIBLE,
                    "task population exceeds the provider token budget",
                    evidence=(f"required={sum(task.output_tokens for task in normalized)}", f"limit={max_provider_tokens}"),
                )
            )
        if "max_cost_micros" in budget and sum(task.cost_micros for task in normalized) > max_cost:
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.COST_INFEASIBLE,
                    "task population exceeds the provider cost budget",
                    evidence=(f"required={sum(task.cost_micros for task in normalized)}", f"limit={max_cost}"),
                )
            )

        capacity = {
            "cpu_slots": _integer(
                capacity_source.get(
                    "cpu_slots",
                    capacity_source.get(
                        "available_cpu_slots",
                        capacity_source.get(
                            "available_slots",
                            capacity_source.get("available_worker_capacity", 0),
                        ),
                    ),
                ),
                "capacity.cpu_slots",
            ),
            "process_slots": _integer(
                capacity_source.get(
                    "process_slots",
                    capacity_source.get(
                        "available_process_slots",
                        capacity_source.get(
                            "available_slots",
                            capacity_source.get("available_worker_capacity", 0),
                        ),
                    ),
                ),
                "capacity.process_slots",
            ),
            "memory_bytes": _integer(
                capacity_source.get(
                    "memory_bytes",
                    capacity_source.get(
                        "available_memory_bytes",
                        capacity_source.get("memory_available_bytes", 0),
                    ),
                ),
                "capacity.memory_bytes",
            ),
            "gpu_memory_bytes": _integer(
                capacity_source.get(
                    "gpu_memory_bytes",
                    capacity_source.get(
                        "available_gpu_memory_bytes",
                        capacity_source.get("gpu_memory_available_bytes", 0),
                    ),
                ),
                "capacity.gpu_memory_bytes",
            ),
            "disk_bytes": _integer(
                capacity_source.get(
                    "disk_bytes",
                    capacity_source.get(
                        "available_disk_bytes",
                        capacity_source.get("disk_available_bytes", 0),
                    ),
                ),
                "capacity.disk_bytes",
            ),
            "resource_class_slots": {
                str(key): _integer(value, f"capacity.resource_class_slots.{key}")
                for key, value in sorted(
                    _mapping(capacity_source.get("resource_class_slots", {})).items()
                )
            },
            "capabilities": set(_strings(capacity_source.get("capabilities", ()))),
            "resource_classes": set(_strings(capacity_source.get("resource_classes", ()))),
        }
        providers = {
            provider["provider_id"]: provider
            for provider in (_provider_capacity(item) for item in providers_raw)
            if provider["provider_id"]
        }
        if normalized and capacity_raw and (capacity["cpu_slots"] < 1 or capacity["process_slots"] < 1):
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.RESOURCE_INFEASIBLE,
                    "capacity snapshot has no executable CPU/process slot",
                )
            )
        for task in normalized:
            if capacity_raw and not _fits((task,), provider_by_task, capacity, providers):
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.RESOURCE_INFEASIBLE,
                        f"task {task.task_id} cannot fit available host/provider resources",
                        (task.task_id,),
                    )
                )
            if task.worktree_policy in {"none", "shared"} and not task.review_only:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.WORKTREE_INFEASIBLE,
                        "mutating tasks require isolated worktrees",
                        (task.task_id,),
                        (task.worktree_policy,),
                    )
                )
            if task.heartbeat_interval_ms and task.heartbeat_interval_ms >= task.lease_duration_ms:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.LEASE_INFEASIBLE,
                        "lease heartbeat must occur before lease expiry",
                        (task.task_id,),
                    )
                )
            if not task.review_only and task.lease_duration_ms < 1:
                issues.append(
                    ParallelPlanIssue(
                        ParallelPlanIssueCode.LEASE_INFEASIBLE,
                        "mutating tasks require a positive lease duration",
                        (task.task_id,),
                    )
                )

        ready_waves: list[ReadyWave] = []
        graph_width = max((len(wave) for wave in waves), default=0)
        conflict_width = 0
        colored_by_wave: list[tuple[tuple[str, ...], ...]] = []
        for index, wave in enumerate(waves):
            lanes = _color_wave(wave, conflicts)
            colored_by_wave.append(lanes)
            projected = max((len(lane) for lane in lanes), default=0)
            conflict_width = max(conflict_width, projected)
            ready_waves.append(
                ReadyWave(index, tuple(wave), lanes, len(wave), projected)
            )

        by_id = {task.task_id: task for task in normalized}
        execution: list[ExecutionWave] = []
        resource_width = 0
        hard_rejected = any(issue.code in _HARD_REJECTION_CODES for issue in issues)
        if not hard_rejected and not all_review:
            # First project capacity without the caller's requested-width cap.
            # This keeps requested, graph, conflict, and resource surfaces
            # independently observable.
            for lanes in colored_by_wave:
                for lane in lanes:
                    packed: list[_Task] = []
                    for task_id in lane:
                        candidate = by_id[task_id]
                        if _fits(
                            (*packed, candidate),
                            provider_by_task,
                            capacity,
                            providers,
                        ):
                            packed.append(candidate)
                    resource_width = max(resource_width, len(packed))

            max_ready_width = _integer(
                budget.get("max_ready_width", request.requested_width),
                "max_ready_width",
                default=request.requested_width,
                minimum=1,
            )
            admitted_limit = min(
                request.requested_width,
                graph_width or 1,
                conflict_width or 1,
                resource_width or 1,
                max_ready_width,
            )
            next_wave = 0
            for dependency_wave, lanes in enumerate(colored_by_wave):
                for lane in lanes:
                    pending = list(lane)
                    while pending:
                        selected: list[_Task] = []
                        for task_id in tuple(pending):
                            candidate = by_id[task_id]
                            if len(selected) >= admitted_limit:
                                break
                            if _fits((*selected, candidate), provider_by_task, capacity, providers):
                                selected.append(candidate)
                        if not selected:
                            # Individual feasibility was checked above; this is
                            # defensive against an inconsistent capacity map.
                            issues.append(
                                ParallelPlanIssue(
                                    ParallelPlanIssueCode.RESOURCE_INFEASIBLE,
                                    "resource bin packing made no progress",
                                    tuple(pending),
                                )
                            )
                            hard_rejected = True
                            break
                        for task in selected:
                            pending.remove(task.task_id)
                        resource, provider_usage = _wave_usage(selected, provider_by_task)
                        execution.append(
                            ExecutionWave(
                                next_wave,
                                dependency_wave,
                                tuple(task.task_id for task in selected),
                                resource,
                                provider_usage,
                            )
                        )
                        next_wave += 1
                    if hard_rejected:
                        break
                if hard_rejected:
                    break

        estimated_makespan_ms = (
            sum(
                max(by_id[task_id].duration_ms for task_id in wave.task_ids)
                for wave in execution
            )
            if execution
            else critical_duration
        )
        if (
            deadline
            and ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE
            not in {issue.code for issue in issues}
            and now + estimated_makespan_ms > deadline
        ):
            issues.append(
                ParallelPlanIssue(
                    ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE,
                    "resource-constrained execution waves cannot meet the deadline",
                    tuple(
                        task_id
                        for wave in execution
                        for task_id in wave.task_ids
                    ),
                    (
                        f"now_ms={now}",
                        f"deadline_ms={deadline}",
                        f"estimated_makespan_ms={estimated_makespan_ms}",
                    ),
                )
            )
            hard_rejected = True
            execution = []

        admitted_width = 0 if hard_rejected or all_review else min(
            request.requested_width,
            graph_width or 1,
            conflict_width or 1,
            resource_width or 1,
            _integer(budget.get("max_ready_width", request.requested_width), "max_ready_width", default=request.requested_width, minimum=1),
        )
        if hard_rejected:
            outcome = ParallelPlanOutcome.REJECTED
            execution = []
        elif all_review:
            outcome = ParallelPlanOutcome.REVIEW_ONLY
        elif admitted_width <= 1:
            outcome = ParallelPlanOutcome.SERIAL
        elif admitted_width < request.requested_width:
            outcome = ParallelPlanOutcome.DEGRADED
        else:
            outcome = ParallelPlanOutcome.PARALLEL

        fence_base = max(
            request.base_fence_epoch,
            _integer(repository.get("fencing_epoch", 0), "repository.fencing_epoch"),
            max((task.fence_epoch for task in normalized), default=0),
        )
        assignments: list[TaskAssignment] = []
        for order, task_id in enumerate(merge_task_order, start=1):
            task = by_id[task_id]
            shard_material = task.shard_key or task.affinity_key or task.task_id
            shard_id = "shard-" + hashlib.sha256(shard_material.encode()).hexdigest()[:12]
            slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task.task_id).strip("-")[:48] or "task"
            worktree_id = f"worktree-{order:04d}-{slug}"
            lease_id = _digest("lease", {"input_digest": input_digest, "task_id": task.task_id, "order": order})
            epoch = fence_base + order
            fence_token = _digest("fence", {"lease_id": lease_id, "epoch": epoch, "tree": repository_tree_id})
            assignments.append(
                TaskAssignment(
                    task_id=task.task_id,
                    shard_id=shard_id,
                    affinity_key=task.affinity_key,
                    exclusive_group=task.exclusive_group,
                    exclusive_paths=task.exclusive_paths,
                    worktree_id=worktree_id,
                    worktree_path=f".agent-supervisor/worktrees/{worktree_id}",
                    base_revision=task.expected_base_revision
                    or repository_tree_id,
                    merge_target=task.expected_merge_target
                    or str(
                        repository.get("expected_merge_target")
                        or repository.get("merge_target")
                        or repository_tree_id
                    ),
                    lease_id=lease_id,
                    lease_scope=task.lease_scope,
                    lease_duration_ms=task.lease_duration_ms,
                    heartbeat_interval_ms=task.heartbeat_interval_ms,
                    lease_owner_rule=task.lease_owner_rule,
                    fence_epoch=epoch,
                    fence_token=fence_token,
                    provider_id=provider_by_task.get(task.task_id, ""),
                    resource_class=task.resource_class,
                )
            )

        merge_order: list[MergeStep] = []
        global_validation = tuple(sorted(set(request.post_merge_validation) | set(_strings(repository.get("post_merge_validation", ())))))
        for order, task_id in enumerate(merge_task_order, start=1):
            task = by_id[task_id]
            boundary = _digest("rollback-boundary", {"input_digest": input_digest, "order": order, "task_id": task_id})
            checkpoint = _digest("merge-checkpoint", {"tree": repository_tree_id, "order": order, "task_id": task_id})
            merge_order.append(
                MergeStep(
                    order=order,
                    task_id=task_id,
                    merge_train_id=task.merge_train_id,
                    depends_on=merge_dependencies.get(task_id, ()),
                    rollback_boundary=boundary,
                    checkpoint_id=checkpoint,
                    post_merge_validation=tuple(sorted(set(task.validation) | set(global_validation))),
                )
            )

        if outcome in {
            ParallelPlanOutcome.REJECTED,
            ParallelPlanOutcome.REVIEW_ONLY,
        }:
            assignments = []
            merge_order = []

        deduplicated_issues = tuple(
            sorted(
                {
                    (issue.code.value, issue.message, issue.task_ids, issue.evidence): issue
                    for issue in issues
                }.values(),
                key=lambda issue: (issue.code.value, issue.task_ids, issue.evidence, issue.message),
            )
        )
        widths = WidthProjection(
            requested=request.requested_width,
            graph=graph_width,
            conflict=conflict_width,
            resource=resource_width,
            admitted=admitted_width,
        )
        issue_codes = {issue.code for issue in issues}
        resource_feasibility = ResourceFeasibilityProjection(
            feasible=outcome is not ParallelPlanOutcome.REJECTED,
            host_feasible=not bool(
                issue_codes
                & {
                    ParallelPlanIssueCode.RESOURCE_INFEASIBLE,
                    ParallelPlanIssueCode.INVALID_SNAPSHOT,
                }
            ),
            provider_feasible=ParallelPlanIssueCode.PROVIDER_INFEASIBLE
            not in issue_codes,
            token_feasible=ParallelPlanIssueCode.TOKEN_INFEASIBLE
            not in issue_codes,
            cost_feasible=ParallelPlanIssueCode.COST_INFEASIBLE
            not in issue_codes,
            context_feasible=ParallelPlanIssueCode.CONTEXT_INFEASIBLE
            not in issue_codes,
            freshness_proved=ParallelPlanIssueCode.STALE_CAPACITY
            not in issue_codes,
            required_totals={
                "cpu_slots": sum(task.cpu_slots for task in normalized),
                "process_slots": sum(task.process_slots for task in normalized),
                "memory_bytes": sum(task.memory_bytes for task in normalized),
                "gpu_memory_bytes": sum(
                    task.gpu_memory_bytes for task in normalized
                ),
                "disk_bytes": sum(task.disk_bytes for task in normalized),
                "context_tokens": sum(task.context_tokens for task in normalized),
                "output_tokens": sum(task.output_tokens for task in normalized),
                "quota_units": sum(task.quota_units for task in normalized),
                "cost_micros": sum(task.cost_micros for task in normalized),
            },
            available_host={
                name: int(capacity[name])
                for name in (
                    "cpu_slots",
                    "process_slots",
                    "memory_bytes",
                    "gpu_memory_bytes",
                    "disk_bytes",
                )
            },
            provider_by_task_id=provider_by_task,
        )
        return ParallelExecutionPlan(
            outcome=outcome,
            input_digest=input_digest,
            repository_tree_id=repository_tree_id,
            capacity_snapshot_id=capacity_snapshot_id,
            provider_snapshot_ids=provider_snapshot_ids,
            widths=widths,
            resource_feasibility=resource_feasibility,
            leaf_producer_closure=closure,
            dependency_edges=tuple(
                sorted((dependency, task_id) for task_id, deps in dependencies.items() for dependency in deps)
            ),
            critical_path=critical_path,
            critical_path_duration_ms=critical_duration,
            estimated_makespan_ms=estimated_makespan_ms,
            ready_waves=tuple(ready_waves),
            execution_waves=tuple(execution),
            conflicts=conflicts,
            assignments=tuple(assignments),
            merge_order=tuple(merge_order),
            issues=deduplicated_issues,
            replay_request=replay_request,
        )

    def replay(
        self,
        plan: ParallelExecutionPlan,
        request: ParallelPlanCompilationRequest | None = None,
    ) -> ParallelExecutionPlan:
        """Recompile canonical inputs and reject snapshot or output drift."""

        if request is None:
            payload = dict(plan.replay_request)
            request = ParallelPlanCompilationRequest(
                tasks=tuple(payload.get("tasks", ())),
                requested_width=payload.get("requested_width", 1),
                repository_snapshot=payload.get("repository_snapshot", {}),
                capacity_snapshot=payload.get("capacity_snapshot", {}),
                provider_snapshots=tuple(payload.get("provider_snapshots", ())),
                budget=payload.get("budget", {}),
                current_time_ms=payload.get("current_time_ms", 0),
                deadline_ms=payload.get("deadline_ms", 0),
                required_leaf_ids=tuple(payload.get("required_leaf_ids", ())),
                completed_task_ids=tuple(payload.get("completed_task_ids", ())),
                protected_paths=tuple(payload.get("protected_paths", ())),
                submodule_paths=tuple(payload.get("submodule_paths", ())),
                post_merge_validation=tuple(payload.get("post_merge_validation", ())),
                conflict_receipts=tuple(payload.get("conflict_receipts", ())),
                base_fence_epoch=payload.get("base_fence_epoch", 0),
                review_only=payload.get("review_only", False),
            )
        replayed = self.compile(request)
        if replayed.input_digest != plan.input_digest or replayed.plan_id != plan.plan_id:
            raise ParallelPlanError("deterministic replay does not match the supplied plan")
        return replayed

    verify_replay = replay


def compile_parallel_execution_plan(
    tasks: Sequence[Any] | ParallelPlanCompilationRequest,
    **kwargs: Any,
) -> ParallelExecutionPlan:
    """Functional entry point for :class:`ParallelPlanCompiler`."""

    return ParallelPlanCompiler().compile(tasks, **kwargs)


def replay_parallel_execution_plan(
    plan: ParallelExecutionPlan,
    request: ParallelPlanCompilationRequest | None = None,
) -> ParallelExecutionPlan:
    """Replay a plan from its canonical body-free compilation request."""

    return ParallelPlanCompiler().replay(plan, request)


__all__ = [
    "PARALLEL_EXECUTION_PLAN_INTERFACE",
    "PARALLEL_EXECUTION_PLAN_SCHEMA",
    "PARALLEL_PLAN_COMPILER_INTERFACE",
    "ConflictSurfaceRecord",
    "ExecutionWave",
    "LeafProducerClosure",
    "MergeStep",
    "ParallelExecutionPlan",
    "ParallelPlanCompilationRequest",
    "ParallelPlanCompiler",
    "ParallelPlanError",
    "ParallelPlanIssue",
    "ParallelPlanIssueCode",
    "ParallelPlanOutcome",
    "ParallelPlanRejectedError",
    "ReadyWave",
    "ResourceFeasibilityProjection",
    "TaskAssignment",
    "WidthProjection",
    "compile_parallel_execution_plan",
    "replay_parallel_execution_plan",
]
