"""Durable adaptive dispatch for an immutable execution-plan revision.

The parallel plan compiler describes the whole admitted graph.  This module is
the deliberately smaller runtime join which chooses one *execution slice* at
a time.  It never treats a lane number as authority: a lane may dispatch only
the task CIDs sealed in its slice, and claims/effect reservations are committed
in one SQLite transaction shared by all local supervisors.

The entrypoint is intentionally provider- and worktree-agnostic.  Callers
claim first, create the already-selected isolated worktree, then report actual
paths through :meth:`AdaptiveExecutionScheduler.record_effect`.  This keeps a
late, undeclared overlap from becoming an accepted effect.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final


ADAPTIVE_EXECUTION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-execution-plan@1"
)
EXECUTION_SLICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/execution-slice@1"
)
TASK_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-task-claim@1"
)
MAX_TASKS: Final[int] = 4096


class ExecutionPlanError(RuntimeError):
    """Base class for inadmissible adaptive execution operations."""


class ExecutionSliceViolationError(ExecutionPlanError):
    """A lane attempted to select work outside its sealed slice."""


class ExecutionClaimConflictError(ExecutionPlanError):
    """A task/effect is already owned by another live fenced claim."""


class ExecutionReplanRequired(ExecutionPlanError):
    """Observed work invalidated the optimistic conflict plan."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionPlanError("execution-plan data must be canonical JSON") from exc


def _cid(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(_canonical(value)).hexdigest()}"


def _text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result or "\x00" in result:
        raise ExecutionPlanError(f"{field_name} must be a nonempty text value")
    return result


def _paths(value: Any, field_name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = (value,) if isinstance(value, str) else value
    try:
        iterable = tuple(values)
    except TypeError as exc:
        raise ExecutionPlanError(f"{field_name} must be a path sequence") from exc
    result: set[str] = set()
    for item in iterable:
        path = str(item or "").strip().replace("\\", "/").removeprefix("./").rstrip("/")
        pure = PurePosixPath(path)
        if not path or pure.is_absolute() or ".." in pure.parts:
            raise ExecutionPlanError(f"{field_name} has non-canonical path {item!r}")
        result.add(pure.as_posix())
    return tuple(sorted(result))


def _overlaps(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _any_path_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    return any(_overlaps(a, b) for a in left for b in right)


def _string_set(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = (value,) if isinstance(value, str) else value
    try:
        return tuple(sorted({_text(item, "metadata") for item in values}))
    except TypeError as exc:
        raise ExecutionPlanError("metadata must be a sequence") from exc


@dataclass(frozen=True)
class InvocationBudget:
    """Current invocation capacity, not a caller-authorized worker count."""

    max_lanes: int

    def __post_init__(self) -> None:
        if isinstance(self.max_lanes, bool) or not isinstance(self.max_lanes, int) or self.max_lanes < 0:
            raise ExecutionPlanError("max_lanes must be a nonnegative integer")


@dataclass(frozen=True)
class ExecutionTask:
    """Canonical scheduler fields extracted from an admitted task record."""

    task_cid: str
    dependencies: tuple[str, ...] = ()
    declared_paths: tuple[str, ...] = ()
    scope_paths: tuple[str, ...] = ()
    resource_class: str = ""
    provider_id: str = ""
    validation_keys: tuple[str, ...] = ()
    exclusive_keys: tuple[str, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(self, "dependencies", _string_set(self.dependencies))
        object.__setattr__(self, "declared_paths", _paths(self.declared_paths, "declared_paths"))
        object.__setattr__(self, "scope_paths", _paths(self.scope_paths, "scope_paths"))
        object.__setattr__(self, "validation_keys", _string_set(self.validation_keys))
        object.__setattr__(self, "exclusive_keys", _string_set(self.exclusive_keys))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ExecutionPlanError("priority must be an integer")

    @property
    def mutation_scope(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.declared_paths) | set(self.scope_paths)))

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "ExecutionTask":
        resource = value.get("resource_contract") if isinstance(value.get("resource_contract"), Mapping) else {}
        provider = value.get("provider_contract") if isinstance(value.get("provider_contract"), Mapping) else {}
        conflict = value.get("conflict_contract") if isinstance(value.get("conflict_contract"), Mapping) else {}
        task_cid = value.get("task_cid") or value.get("canonical_task_cid") or value.get("task_id") or value.get("id")
        dependencies = value.get("dependency_task_cids") or value.get("depends_on") or value.get("dependencies") or ()
        paths = value.get("outputs") or value.get("predicted_files") or value.get("predicted_paths") or ()
        scopes = value.get("scope_paths") or value.get("affected_paths") or conflict.get("scope_paths") or ()
        raw_priority = value.get("priority_rank", value.get("priority", 0))
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            # Repository boards commonly use P0..P4 while materialized plans
            # use a numeric rank.  Preserve their ordering without making a
            # presentation label a source of non-determinism.
            label = str(raw_priority or "").strip().upper()
            priority = 4 - int(label[1:]) if len(label) == 2 and label[0] == "P" and label[1:].isdigit() else 0
        return cls(
            task_cid=str(task_cid or ""), dependencies=tuple(dependencies),
            declared_paths=paths, scope_paths=scopes,
            resource_class=str(value.get("resource_class") or resource.get("resource_class") or ""),
            provider_id=str(value.get("provider_id") or provider.get("provider_id") or provider.get("provider") or ""),
            validation_keys=value.get("validation_keys") or value.get("validation_commands") or value.get("validations") or (),
            exclusive_keys=value.get("exclusive_keys") or conflict.get("exclusive_keys") or value.get("exclusive_group") or (),
            priority=priority,
        )


@dataclass(frozen=True)
class CapacitySnapshot:
    """Live capacity evidence consumed for exactly one scheduling pass."""

    snapshot_id: str
    host_lanes: int
    provider_lanes: int
    observed_at_ms: int
    fresh_until_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        for name in ("host_lanes", "provider_lanes", "observed_at_ms", "fresh_until_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutionPlanError(f"{name} must be a nonnegative integer")
        if self.fresh_until_ms and self.fresh_until_ms < self.observed_at_ms:
            raise ExecutionPlanError("fresh_until_ms precedes observed_at_ms")

    @property
    def lane_cap(self) -> int:
        return min(self.host_lanes, self.provider_lanes)

    def is_current(self, now_ms: int) -> bool:
        return bool(self.observed_at_ms and (not self.fresh_until_ms or now_ms < self.fresh_until_ms))


@dataclass(frozen=True)
class ExecutionSlice:
    """Exact task CIDs one lane may select for one immutable revision."""

    plan_revision: str
    lane_id: str
    task_cids: tuple[str, ...]
    capacity_snapshot_id: str
    slice_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_revision", _text(self.plan_revision, "plan_revision"))
        object.__setattr__(self, "lane_id", _text(self.lane_id, "lane_id"))
        object.__setattr__(self, "capacity_snapshot_id", _text(self.capacity_snapshot_id, "capacity_snapshot_id"))
        cids = _string_set(self.task_cids)
        object.__setattr__(self, "task_cids", cids)
        expected = _cid("execution-slice", self.to_dict(include_id=False))
        if self.slice_id and self.slice_id != expected:
            raise ExecutionPlanError("execution slice identity does not match its immutable contents")
        object.__setattr__(self, "slice_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {"schema": EXECUTION_SLICE_SCHEMA, "plan_revision": self.plan_revision, "lane_id": self.lane_id, "task_cids": list(self.task_cids), "capacity_snapshot_id": self.capacity_snapshot_id}
        if include_id:
            value["slice_id"] = self.slice_id
        return value


@dataclass(frozen=True)
class AdaptiveExecutionPlan:
    plan_revision: str
    capacity_snapshot_id: str
    requested_lanes: int
    admitted_lanes: int
    ready_task_cids: tuple[str, ...]
    selected_task_cids: tuple[str, ...]
    conflict_pairs: tuple[tuple[str, str], ...]
    slices: tuple[ExecutionSlice, ...]
    plan_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_revision", _text(self.plan_revision, "plan_revision"))
        object.__setattr__(self, "capacity_snapshot_id", _text(self.capacity_snapshot_id, "capacity_snapshot_id"))
        for name in ("requested_lanes", "admitted_lanes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutionPlanError(f"{name} must be a nonnegative integer")
        if self.admitted_lanes != len(self.selected_task_cids) or len(self.slices) != self.admitted_lanes:
            raise ExecutionPlanError("each admitted task must have one execution slice")
        expected = _cid("adaptive-execution-plan", self.to_dict(include_id=False))
        if self.plan_id and self.plan_id != expected:
            raise ExecutionPlanError("adaptive execution plan identity does not match")
        object.__setattr__(self, "plan_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {"schema": ADAPTIVE_EXECUTION_PLAN_SCHEMA, "plan_revision": self.plan_revision, "capacity_snapshot_id": self.capacity_snapshot_id, "requested_lanes": self.requested_lanes, "admitted_lanes": self.admitted_lanes, "ready_task_cids": list(self.ready_task_cids), "selected_task_cids": list(self.selected_task_cids), "conflict_pairs": [list(pair) for pair in self.conflict_pairs], "slices": [item.to_dict() for item in self.slices]}
        if include_id:
            value["plan_id"] = self.plan_id
        return value


@dataclass(frozen=True)
class TaskClaim:
    plan_revision: str
    task_cid: str
    lane_id: str
    slice_id: str
    fence: int
    claim_id: str
    claimed_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TASK_CLAIM_SCHEMA, "plan_revision": self.plan_revision, "task_cid": self.task_cid, "lane_id": self.lane_id, "slice_id": self.slice_id, "fence": self.fence, "claim_id": self.claim_id, "claimed_at_ms": self.claimed_at_ms}


@dataclass(frozen=True)
class EffectReceipt:
    task_cid: str
    accepted: bool
    replan_required: bool
    effect_id: str
    actual_paths: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class ExecutionAttempt:
    task_cid: str
    claim: TaskClaim
    started_at_ms: int
    finished_at_ms: int
    effect: EffectReceipt


class ExecutionLedger:
    """Cross-process claim/effect fence.  SQLite is the sole mutable owner."""

    _init_lock = threading.Lock()

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._init_lock:
            with self._connect() as connection:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS execution_claims (
                        plan_revision TEXT NOT NULL, task_cid TEXT NOT NULL,
                        lane_id TEXT NOT NULL, slice_id TEXT NOT NULL,
                        fence INTEGER NOT NULL, claim_id TEXT NOT NULL UNIQUE,
                        claimed_at_ms INTEGER NOT NULL,
                        PRIMARY KEY(plan_revision, task_cid)
                    );
                    CREATE TABLE IF NOT EXISTS execution_slice_tasks (
                        plan_revision TEXT NOT NULL, task_cid TEXT NOT NULL,
                        slice_id TEXT NOT NULL, lane_id TEXT NOT NULL,
                        capacity_snapshot_id TEXT NOT NULL,
                        PRIMARY KEY(plan_revision, task_cid),
                        UNIQUE(plan_revision, slice_id, task_cid)
                    );
                    CREATE TABLE IF NOT EXISTS execution_handoffs (
                        plan_revision TEXT NOT NULL, task_cid TEXT NOT NULL,
                        donor_slice_id TEXT NOT NULL, recipient_slice_id TEXT NOT NULL,
                        handoff_id TEXT NOT NULL UNIQUE,
                        PRIMARY KEY(plan_revision, task_cid)
                    );
                    CREATE TABLE IF NOT EXISTS execution_effects (
                        plan_revision TEXT NOT NULL, task_cid TEXT NOT NULL,
                        effect_id TEXT NOT NULL, actual_paths_json TEXT NOT NULL,
                        accepted INTEGER NOT NULL, replan_required INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        PRIMARY KEY(plan_revision, task_cid, effect_id),
                        UNIQUE(plan_revision, task_cid)
                    );
                    CREATE TABLE IF NOT EXISTS execution_replans (
                        plan_revision TEXT NOT NULL, task_cid TEXT NOT NULL,
                        reason TEXT NOT NULL, attempts INTEGER NOT NULL,
                        PRIMARY KEY(plan_revision, task_cid, reason)
                    );
                """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def register_slices(self, slices: Sequence[ExecutionSlice]) -> None:
        """Persist the only lanes authorized to claim this compilation pass.

        Registration is idempotent for byte-identical slices and rejects a
        competing compiler's ownership map.  Claims are not accepted until
        this transaction succeeds, so a restarted/empty lane cannot create a
        plausible-looking slice locally and steal another lane's task.
        """
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for execution_slice in slices:
                    for task_cid in execution_slice.task_cids:
                        row = connection.execute(
                            "SELECT slice_id, lane_id, capacity_snapshot_id FROM execution_slice_tasks WHERE plan_revision=? AND task_cid=?",
                            (execution_slice.plan_revision, task_cid),
                        ).fetchone()
                        expected = (execution_slice.slice_id, execution_slice.lane_id, execution_slice.capacity_snapshot_id)
                        if row is not None and row != expected:
                            raise ExecutionClaimConflictError(
                                f"task {task_cid!r} already belongs to a different immutable slice"
                            )
                        if row is None:
                            connection.execute(
                                "INSERT INTO execution_slice_tasks VALUES(?,?,?,?,?)",
                                (execution_slice.plan_revision, task_cid, *expected),
                            )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def steal(
        self,
        *,
        donor_slice: ExecutionSlice,
        recipient_lane_id: str,
        task_cid: str,
    ) -> ExecutionSlice:
        """Transfer one *unclaimed* task under the same revision fence.

        Work stealing is never inferred from an idle lane.  A supervisor must
        present the donor's exact immutable slice, and the durable ownership
        row moves once to a newly sealed recipient slice in the same plan
        revision and capacity snapshot.
        """
        task_cid = _text(task_cid, "task_cid")
        recipient_lane_id = _text(recipient_lane_id, "recipient_lane_id")
        if task_cid not in donor_slice.task_cids:
            raise ExecutionSliceViolationError("work-steal task is outside the donor slice")
        recipient = ExecutionSlice(
            donor_slice.plan_revision,
            recipient_lane_id,
            (task_cid,),
            donor_slice.capacity_snapshot_id,
        )
        handoff_id = _cid("execution-handoff", {"revision": donor_slice.plan_revision, "task": task_cid, "donor": donor_slice.slice_id, "recipient": recipient.slice_id})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute(
                    "SELECT slice_id, lane_id, capacity_snapshot_id FROM execution_slice_tasks WHERE plan_revision=? AND task_cid=?",
                    (donor_slice.plan_revision, task_cid),
                ).fetchone()
                if owner != (donor_slice.slice_id, donor_slice.lane_id, donor_slice.capacity_snapshot_id):
                    raise ExecutionSliceViolationError("donor slice no longer owns this task")
                claimed = connection.execute(
                    "SELECT 1 FROM execution_claims WHERE plan_revision=? AND task_cid=?",
                    (donor_slice.plan_revision, task_cid),
                ).fetchone()
                if claimed is not None:
                    raise ExecutionClaimConflictError("claimed work cannot be stolen")
                connection.execute(
                    "UPDATE execution_slice_tasks SET slice_id=?, lane_id=? WHERE plan_revision=? AND task_cid=?",
                    (recipient.slice_id, recipient_lane_id, donor_slice.plan_revision, task_cid),
                )
                connection.execute(
                    "INSERT INTO execution_handoffs VALUES(?,?,?,?,?)",
                    (donor_slice.plan_revision, task_cid, donor_slice.slice_id, recipient.slice_id, handoff_id),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return recipient

    def claim(self, execution_slice: ExecutionSlice, task_cid: str, *, now_ms: int) -> TaskClaim:
        task_cid = _text(task_cid, "task_cid")
        if task_cid not in execution_slice.task_cids:
            raise ExecutionSliceViolationError("task is outside this lane's immutable execution slice")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT slice_id, lane_id, capacity_snapshot_id FROM execution_slice_tasks WHERE plan_revision=? AND task_cid=?",
                (execution_slice.plan_revision, task_cid),
            ).fetchone()
            if owner != (
                execution_slice.slice_id,
                execution_slice.lane_id,
                execution_slice.capacity_snapshot_id,
            ):
                connection.execute("ROLLBACK")
                raise ExecutionSliceViolationError(
                    "execution slice is unregistered, stale, or owns another lane's task"
                )
            existing = connection.execute("SELECT lane_id, slice_id, fence, claim_id, claimed_at_ms FROM execution_claims WHERE plan_revision=? AND task_cid=?", (execution_slice.plan_revision, task_cid)).fetchone()
            if existing is not None:
                connection.execute("ROLLBACK")
                raise ExecutionClaimConflictError(f"task {task_cid!r} is already claimed by lane {existing[0]!r}")
            fence = int(connection.execute("SELECT COALESCE(MAX(fence), 0) FROM execution_claims WHERE plan_revision=?", (execution_slice.plan_revision,)).fetchone()[0]) + 1
            claim_id = _cid("adaptive-task-claim", {"revision": execution_slice.plan_revision, "task": task_cid, "slice": execution_slice.slice_id, "fence": fence})
            connection.execute("INSERT INTO execution_claims VALUES(?,?,?,?,?,?,?)", (execution_slice.plan_revision, task_cid, execution_slice.lane_id, execution_slice.slice_id, fence, claim_id, now_ms))
            connection.execute("COMMIT")
        return TaskClaim(execution_slice.plan_revision, task_cid, execution_slice.lane_id, execution_slice.slice_id, fence, claim_id, now_ms)

    def record_effect(self, claim: TaskClaim, *, actual_paths: Sequence[str], declared_scope: Sequence[str]) -> EffectReceipt:
        actual = _paths(actual_paths, "actual_paths")
        declared = _paths(declared_scope, "declared_scope")
        effect_id = _cid("execution-effect", {"claim_id": claim.claim_id, "actual_paths": actual})
        undeclared = tuple(path for path in actual if not any(_overlaps(path, scope) for scope in declared))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT lane_id, slice_id, fence, claim_id FROM execution_claims WHERE plan_revision=? AND task_cid=?", (claim.plan_revision, claim.task_cid)).fetchone()
            if current != (claim.lane_id, claim.slice_id, claim.fence, claim.claim_id):
                connection.execute("ROLLBACK")
                raise ExecutionClaimConflictError("claim lost its fence before effect publication")
            prior = connection.execute("SELECT effect_id, accepted, replan_required, reason FROM execution_effects WHERE plan_revision=? AND task_cid=?", (claim.plan_revision, claim.task_cid)).fetchone()
            if prior is not None:
                connection.execute("ROLLBACK")
                if prior[0] == effect_id:
                    return EffectReceipt(claim.task_cid, bool(prior[1]), bool(prior[2]), effect_id, actual, str(prior[3]))
                raise ExecutionClaimConflictError("task already has a distinct terminal effect")
            accepted, replan, reason = True, False, ""
            if undeclared:
                accepted, replan, reason = False, True, "undeclared_scope"
            # Existing accepted effects are the durable reservation surface.
            rows = connection.execute("SELECT task_cid, actual_paths_json FROM execution_effects WHERE plan_revision=? AND accepted=1", (claim.plan_revision,)).fetchall()
            if accepted:
                for other_task, paths_json in rows:
                    if other_task != claim.task_cid and _any_path_overlap(actual, json.loads(paths_json)):
                        accepted, replan, reason = False, True, "observed_scope_overlap"
                        break
            connection.execute("INSERT INTO execution_effects VALUES(?,?,?,?,?,?,?)", (claim.plan_revision, claim.task_cid, effect_id, json.dumps(actual), int(accepted), int(replan), reason))
            if replan:
                connection.execute("INSERT INTO execution_replans(plan_revision, task_cid, reason, attempts) VALUES(?,?,?,1) ON CONFLICT(plan_revision, task_cid, reason) DO UPDATE SET attempts=attempts+1", (claim.plan_revision, claim.task_cid, reason))
            connection.execute("COMMIT")
        return EffectReceipt(claim.task_cid, accepted, replan, effect_id, actual, reason)

    def replan_attempts(self, plan_revision: str, task_cid: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COALESCE(SUM(attempts), 0) FROM execution_replans WHERE plan_revision=? AND task_cid=?", (plan_revision, task_cid)).fetchone()[0])


class AdaptiveExecutionScheduler:
    """Compile and execute conflict-free slices with durable claim ownership."""

    def __init__(self, ledger: ExecutionLedger | Path | str) -> None:
        self.ledger = ledger if isinstance(ledger, ExecutionLedger) else ExecutionLedger(ledger)

    @staticmethod
    def _conflicts(left: ExecutionTask, right: ExecutionTask) -> bool:
        return bool(
            _any_path_overlap(left.mutation_scope, right.mutation_scope)
            or set(left.exclusive_keys) & set(right.exclusive_keys)
            or (left.resource_class and left.resource_class == right.resource_class and "exclusive" in left.resource_class)
            or (left.provider_id and left.provider_id == right.provider_id and "exclusive" in left.provider_id)
            or set(left.validation_keys) & set(right.validation_keys)
        )

    def compile(
        self, *, plan_revision: str, tasks: Sequence[ExecutionTask | Mapping[str, Any]], completed_task_cids: Iterable[str] = (), budget: InvocationBudget | int = InvocationBudget(1), capacity: CapacitySnapshot, now_ms: int | None = None,
    ) -> AdaptiveExecutionPlan:
        revision = _text(plan_revision, "plan_revision")
        normalized = tuple(task if isinstance(task, ExecutionTask) else ExecutionTask.from_record(task) for task in tasks)
        if len(normalized) > MAX_TASKS or len({task.task_cid for task in normalized}) != len(normalized):
            raise ExecutionPlanError("task population is too large or has duplicate canonical task CIDs")
        now = int(time.time() * 1000) if now_ms is None else now_ms
        requested = budget.max_lanes if isinstance(budget, InvocationBudget) else InvocationBudget(budget).max_lanes
        completed = set(_string_set(completed_task_cids))
        by_cid = {task.task_cid: task for task in normalized}
        unknown = sorted({dep for task in normalized for dep in task.dependencies if dep not in by_cid and dep not in completed})
        if unknown:
            raise ExecutionPlanError("ready closure contains unknown dependencies: " + ", ".join(unknown))
        ready = tuple(sorted(task.task_cid for task in normalized if task.task_cid not in completed and set(task.dependencies).issubset(completed)))
        conflicts = tuple(sorted((min(left.task_cid, right.task_cid), max(left.task_cid, right.task_cid)) for index, left in enumerate(normalized) for right in normalized[index + 1:] if self._conflicts(left, right)))
        admitted_cap = min(requested, capacity.lane_cap) if capacity.is_current(now) else 0
        selected: list[str] = []
        # Priority first, CID tie-breaker: deterministic maximal independent set
        # within the live capacity cap.
        for task in sorted((by_cid[cid] for cid in ready), key=lambda item: (-item.priority, item.task_cid)):
            if len(selected) >= admitted_cap:
                break
            if not any(tuple(sorted((task.task_cid, active))) in conflicts for active in selected):
                selected.append(task.task_cid)
        slices = tuple(ExecutionSlice(revision, f"lane-{index}", (task_cid,), capacity.snapshot_id) for index, task_cid in enumerate(selected))
        plan = AdaptiveExecutionPlan(revision, capacity.snapshot_id, requested, len(selected), ready, tuple(selected), conflicts, slices)
        self.ledger.register_slices(plan.slices)
        return plan

    def claim(self, execution_slice: ExecutionSlice, task_cid: str, *, now_ms: int | None = None) -> TaskClaim:
        return self.ledger.claim(execution_slice, task_cid, now_ms=int(time.time() * 1000) if now_ms is None else now_ms)

    def steal(self, *, donor_slice: ExecutionSlice, recipient_lane_id: str, task_cid: str) -> ExecutionSlice:
        """Explicitly transfer an unclaimed same-revision slice task."""
        return self.ledger.steal(
            donor_slice=donor_slice,
            recipient_lane_id=recipient_lane_id,
            task_cid=task_cid,
        )

    def record_effect(self, claim: TaskClaim, task: ExecutionTask, actual_paths: Sequence[str]) -> EffectReceipt:
        return self.ledger.record_effect(claim, actual_paths=actual_paths, declared_scope=task.mutation_scope)

    def execute(self, plan: AdaptiveExecutionPlan, tasks: Sequence[ExecutionTask | Mapping[str, Any]], runner: Callable[[ExecutionTask, TaskClaim], Sequence[str]], *, max_workers: int | None = None) -> tuple[ExecutionAttempt, ...]:
        """Run one selected slice set concurrently and return an overlap timeline."""
        by_cid = {task.task_cid: task if isinstance(task, ExecutionTask) else ExecutionTask.from_record(task) for task in tasks}
        selected = tuple(plan.selected_task_cids)
        sliced = tuple(task_cid for execution_slice in plan.slices for task_cid in execution_slice.task_cids)
        if set(selected) != set(sliced) or len(sliced) != len(set(sliced)):
            raise ExecutionSliceViolationError("execution plan does not seal exactly one slice per selected task")
        if any(task_cid not in by_cid for task_cid in selected):
            raise ExecutionPlanError("execution plan selected a task absent from its canonical population")
        for index, left_cid in enumerate(selected):
            left = by_cid[left_cid]
            if set(left.dependencies) & set(selected):
                raise ExecutionPlanError("dependency-related tasks cannot share one execution pass")
            if any(self._conflicts(left, by_cid[right_cid]) for right_cid in selected[index + 1:]):
                raise ExecutionPlanError("conflicting tasks cannot share one execution pass")
        def run(execution_slice: ExecutionSlice) -> ExecutionAttempt:
            task_cid = execution_slice.task_cids[0]  # slices are deliberately one task/lane
            task = by_cid[task_cid]
            claim = self.claim(execution_slice, task_cid)
            started = int(time.time() * 1000)
            paths = runner(task, claim)
            effect = self.record_effect(claim, task, paths)
            return ExecutionAttempt(task_cid, claim, started, int(time.time() * 1000), effect)
        if not plan.slices:
            return ()
        with ThreadPoolExecutor(max_workers=max_workers or len(plan.slices)) as pool:
            futures: list[Future[ExecutionAttempt]] = [pool.submit(run, item) for item in plan.slices]
            return tuple(sorted((future.result() for future in as_completed(futures)), key=lambda item: item.task_cid))


def compile_adaptive_execution_plan(**kwargs: Any) -> AdaptiveExecutionPlan:
    """Functional facade for one capacity-bound scheduling pass."""
    ledger = kwargs.pop("ledger", None)
    if ledger is None:
        raise ExecutionPlanError("compile_adaptive_execution_plan requires a durable ledger")
    return AdaptiveExecutionScheduler(ledger).compile(**kwargs)


__all__ = [
    "ADAPTIVE_EXECUTION_PLAN_SCHEMA", "EXECUTION_SLICE_SCHEMA", "TASK_CLAIM_SCHEMA",
    "AdaptiveExecutionPlan", "AdaptiveExecutionScheduler", "CapacitySnapshot",
    "EffectReceipt", "ExecutionAttempt", "ExecutionClaimConflictError", "ExecutionLedger",
    "ExecutionPlanError", "ExecutionReplanRequired", "ExecutionSlice", "ExecutionSliceViolationError",
    "ExecutionTask", "InvocationBudget", "TaskClaim", "compile_adaptive_execution_plan",
]
