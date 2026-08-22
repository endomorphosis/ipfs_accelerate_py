"""Database-backed task source adapter over the intent repository (DQP-012).

Interface: ``DatabaseTaskSource@1``

Presents the public task/plan/objective APIs used by scheduler and daemon
callers while retaining **canonical identities** (``task_cid``, ``plan_cid``,
``goal_cid``, ``objective_id``). Mutable display aliases never redefine those
keys.

This adapter is the control-plane cutover path for
:class:`~.duckdb_task_source.DuckDBTaskSource` consumers: status CAS, readiness,
snapshots, and completion all route through :class:`IntentRepository` so that
projections and domain events advance in one database transaction (no
cross-file saga).

Completion cannot be selected without current required evidence — the
intent-repository completion gate is enforced on every complete transition.
"""

from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import content_identity
from .control_plane_migrations import duckdb_available
from .duckdb_state import is_quack_transport_target, quack_transport_uri
from .intent_repository import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    IntentCompletionError,
    IntentReceipt,
    IntentRepository,
    IntentRepositoryBoundsError,
    IntentRepositoryConflictError,
    IntentRepositoryError,
    IntentRepositoryIntegrityError,
    IntentRepositoryTransitionError,
    IntentRepositoryUnknownOutcomeError,
    PlanRevisionRepository,
    QueueEntry,
    open_intent_repository,
    task_projection_spec_cid,
)

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

DATABASE_TASK_SOURCE_INTERFACE: Final[str] = "DatabaseTaskSource@1"
DATABASE_TASK_SOURCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-task-source@1"
)
DATABASE_TASK_SOURCE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-task-source-snapshot@1"
)
DATABASE_TASK_PAGE_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/database-task-page@1"
DATABASE_TASK_CAS_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/database-task-cas@1"

DEFAULT_QUERY_LIMIT: Final[int] = DEFAULT_PAGE_LIMIT
MAX_QUERY_LIMIT: Final[int] = MAX_PAGE_LIMIT


# ---------------------------------------------------------------------------
# Errors (mirror duckdb_task_source public vocabulary)
# ---------------------------------------------------------------------------


class DatabaseTaskSourceError(IntentRepositoryError):
    """Base fail-closed error for the database task source adapter."""


class TaskSourceIntegrityError(DatabaseTaskSourceError, IntentRepositoryIntegrityError):
    """Schema, identity, or projection integrity failure."""


class TaskSourceConflictError(DatabaseTaskSourceError, IntentRepositoryConflictError):
    """CAS head or expected-revision conflict."""


class TaskSourceTransitionError(DatabaseTaskSourceError, IntentRepositoryTransitionError):
    """Owner rejected a status transition outside the closed matrix."""


class TaskSourceUnknownOutcomeError(
    DatabaseTaskSourceError, IntentRepositoryUnknownOutcomeError
):
    """A remote owner effect requires exact post-restart reconciliation."""


class TaskSourceBoundsError(DatabaseTaskSourceError, IntentRepositoryBoundsError):
    """A query or population bound was exceeded."""


class TaskSourceCompletionError(DatabaseTaskSourceError, IntentCompletionError):
    """Completion refused without current required evidence."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskRecord:
    """Canonical task projection row returned by public APIs."""

    task_cid: str
    task_alias: str
    goal_cid: str
    ordinal: int
    status: str
    revision: int
    body: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    plan_cid: str = ""
    objective_id: str = ""
    priority: str = ""
    outputs: tuple[Mapping[str, Any], ...] = ()
    acceptance: tuple[Mapping[str, Any], ...] = ()
    validations: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "goal_cid": self.goal_cid,
            "plan_cid": self.plan_cid,
            "objective_id": self.objective_id,
            "ordinal": int(self.ordinal),
            "status": self.status,
            "revision": int(self.revision),
            "priority": self.priority,
            "body": dict(self.body),
            "dependencies": list(self.dependencies),
            "outputs": [dict(item) for item in self.outputs],
            "acceptance": [dict(item) for item in self.acceptance],
            "validations": [dict(item) for item in self.validations],
        }


@dataclass(frozen=True)
class TaskPage:
    """Bounded page of tasks with optional continuation cursor."""

    SCHEMA: ClassVar[str] = DATABASE_TASK_PAGE_SCHEMA

    tasks: tuple[TaskRecord, ...]
    revision: int
    next_cursor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tasks": [item.to_dict() for item in self.tasks],
            "revision": int(self.revision),
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class CASResult:
    """Result of a compare-and-set status transition."""

    SCHEMA: ClassVar[str] = DATABASE_TASK_CAS_SCHEMA

    task: TaskRecord
    previous_status: str
    revision: int
    event_cursor: int
    changed: bool
    receipt_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task": self.task.to_dict(),
            "previous_status": self.previous_status,
            "revision": int(self.revision),
            "event_cursor": int(self.event_cursor),
            "changed": bool(self.changed),
            "receipt_cid": self.receipt_cid,
        }


@dataclass(frozen=True)
class TaskSourceSnapshot:
    """Bounded snapshot of the database task-source projection."""

    SCHEMA: ClassVar[str] = DATABASE_TASK_SOURCE_SNAPSHOT_SCHEMA

    source_schema: str
    schema_version: int
    plan_root_cid: str
    repository_tree_id: str
    projection_cid: str
    formal_plan_id: str
    source_identity: str
    revision: int
    event_cursor: int
    goal_count: int
    task_count: int
    dependency_count: int
    terminal: bool
    objective_count: int = 0
    plan_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "source_schema": self.source_schema,
            "schema_version": int(self.schema_version),
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "projection_cid": self.projection_cid,
            "formal_plan_id": self.formal_plan_id,
            "source_identity": self.source_identity,
            "revision": int(self.revision),
            "event_cursor": int(self.event_cursor),
            "goal_count": int(self.goal_count),
            "task_count": int(self.task_count),
            "dependency_count": int(self.dependency_count),
            "terminal": bool(self.terminal),
            "objective_count": int(self.objective_count),
            "plan_count": int(self.plan_count),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cursor_encode(revision: int, offset: int) -> str:
    payload = json.dumps(
        {"v": 1, "revision": int(revision), "offset": int(offset)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _cursor_decode(cursor: str, *, revision: int) -> int:
    text = str(cursor or "").strip()
    if not text:
        return 0
    padded = text + ("=" * (-len(text) % 4))
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskSourceConflictError("task page cursor is malformed") from exc
    if not isinstance(payload, Mapping):
        raise TaskSourceConflictError("task page cursor is malformed")
    if int(payload.get("v") or 0) != 1:
        raise TaskSourceConflictError("task page cursor version is unsupported")
    if int(payload.get("revision") or -1) != int(revision):
        raise TaskSourceConflictError("task page cursor revision is stale")
    offset = int(payload.get("offset") or 0)
    if offset < 0:
        raise TaskSourceConflictError("task page cursor offset is invalid")
    return offset


def _task_key(value: str | TaskRecord | Mapping[str, Any]) -> str:
    if isinstance(value, TaskRecord):
        return value.task_cid
    if isinstance(value, Mapping):
        for key in ("task_cid", "cid", "id", "task_alias", "alias"):
            raw = value.get(key)
            if raw:
                return str(raw).strip()
        raise TaskSourceIntegrityError("task key mapping is missing identity")
    text = str(value or "").strip()
    if not text:
        raise TaskSourceIntegrityError("task key must not be empty")
    return text


def _as_task_record(row: Mapping[str, Any]) -> TaskRecord:
    body = row.get("body") if isinstance(row.get("body"), Mapping) else {}
    deps = row.get("dependencies") or ()
    if isinstance(deps, str):
        dep_tuple = (deps,)
    elif isinstance(deps, Sequence):
        dep_tuple = tuple(str(item) for item in deps)
    else:
        dep_tuple = ()
    outputs = row.get("outputs") or ()
    acceptance = row.get("acceptance") or ()
    validations = row.get("validations") or ()
    return TaskRecord(
        task_cid=str(row["task_cid"]),
        task_alias=str(row.get("task_alias") or row["task_cid"]),
        goal_cid=str(row.get("goal_cid") or ""),
        ordinal=int(row.get("ordinal") or 0),
        status=str(row.get("status") or "ready"),
        revision=int(row.get("revision") or 0),
        body=MappingProxyType(dict(body)),
        dependencies=dep_tuple,
        plan_cid=str(row.get("plan_cid") or ""),
        objective_id=str(row.get("objective_id") or ""),
        priority=str(row.get("priority") or ""),
        outputs=tuple(
            MappingProxyType(dict(item)) for item in outputs if isinstance(item, Mapping)
        ),
        acceptance=tuple(
            MappingProxyType(dict(item)) for item in acceptance if isinstance(item, Mapping)
        ),
        validations=tuple(
            MappingProxyType(dict(item)) for item in validations if isinstance(item, Mapping)
        ),
    )


# ---------------------------------------------------------------------------
# DatabaseTaskSource
# ---------------------------------------------------------------------------


class DatabaseTaskSource:
    """Public task-source adapter backed by :class:`IntentRepository`.

    Interface: ``DatabaseTaskSource@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_TASK_SOURCE_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_TASK_SOURCE_SCHEMA

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        intent: IntentRepository | None = None,
        owner_id: str = "database-task-source:local",
        repository_tree_id: str = "",
        plan_root_cid: str = "",
        install_schema: bool = True,
        evidence_freshness_seconds: int = 3600,
        clock_ms: Any | None = None,
    ) -> None:
        if intent is not None:
            self._intent = intent
            self.database_path = Path(intent.database_path)
        else:
            if database_path is None:
                raise ValueError("DatabaseTaskSource requires database_path or intent")
            if is_quack_transport_target(database_path):
                store = quack_transport_uri(database_path)
                self.database_path = store
            else:
                store = Path(database_path).absolute()
                self.database_path = store
            self._intent = open_intent_repository(
                store,
                owner_id=owner_id,
                install_schema=install_schema,
                evidence_freshness_seconds=evidence_freshness_seconds,
                clock_ms=clock_ms,
            )
        self.path = self.database_path
        self.repository_tree_id = str(repository_tree_id or "")
        self.plan_root_cid = str(plan_root_cid or "")
        self.owner_id = owner_id

    # -- lifecycle -----------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return duckdb_available()

    is_available = available

    @property
    def intent(self) -> IntentRepository:
        return self._intent

    @property
    def plans(self) -> PlanRevisionRepository:
        return self._intent.plan_revisions()

    def close(self) -> None:
        self._intent.close()

    def __enter__(self) -> DatabaseTaskSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- materialize / ingest ------------------------------------------------

    def materialize(
        self,
        population: Mapping[str, Any],
        *,
        repository_tree_id: str = "",
        plan_root_cid: str = "",
    ) -> Mapping[str, Any]:
        """Ingest a bounded population of objectives/goals/plans/tasks.

        Accepts a mapping shaped like formal-plan / prompt-graph fixtures and
        writes them through the intent repository in discrete transactions.
        Canonical CIDs supplied by the caller are retained.
        """

        if not isinstance(population, Mapping):
            raise TaskSourceIntegrityError("population must be a mapping")
        tree_id = str(
            repository_tree_id
            or population.get("repository_tree_id")
            or self.repository_tree_id
            or "tree:unknown"
        )
        self.repository_tree_id = tree_id
        root = str(plan_root_cid or population.get("plan_root_cid") or self.plan_root_cid or "")

        objectives = population.get("objectives") or population.get("goals") or ()
        if isinstance(objectives, Mapping):
            objectives = (objectives,)
        goal_cids: list[str] = []
        goal_cids_by_alias: dict[str, str] = {}
        for index, item in enumerate(objectives):
            if not isinstance(item, Mapping):
                continue
            goal_cid = str(item.get("goal_cid") or item.get("goal_id") or f"goal:cid:{index + 1}")
            goal_alias = str(
                item.get("goal_alias") or item.get("goal_id") or item.get("alias") or goal_cid
            )
            objective_id = str(item.get("objective_id") or item.get("owner_actor_id") or "")
            if objective_id:
                self._intent.upsert_objective(
                    objective_id=objective_id,
                    objective_alias=str(item.get("objective_alias") or objective_id),
                    title=str(item.get("title") or objective_id),
                    status=str(item.get("status") or "open"),
                    priority=str(item.get("priority") or "P2"),
                    body={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "objective_id",
                            "objective_alias",
                            "title",
                            "status",
                            "priority",
                        }
                    },
                )
            self._intent.upsert_goal(
                goal_cid=goal_cid,
                goal_alias=goal_alias,
                title=str(item.get("title") or goal_alias),
                objective_id=objective_id,
                parent_goal_cid=str(item.get("parent_goal_cid") or ""),
                ordinal=int(item.get("ordinal") or index + 1),
                status=str(item.get("status") or "open"),
                body={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "goal_cid",
                        "goal_id",
                        "goal_alias",
                        "title",
                        "status",
                        "ordinal",
                        "objective_id",
                    }
                },
            )
            goal_cids.append(goal_cid)
            goal_cids_by_alias[goal_alias] = goal_cid

        default_goal = goal_cids[0] if goal_cids else "goal:default"
        if not goal_cids:
            self._intent.upsert_goal(
                goal_cid=default_goal,
                goal_alias="G-DEFAULT",
                title="Default goal",
                ordinal=1,
            )
            goal_cids.append(default_goal)
            goal_cids_by_alias["G-DEFAULT"] = default_goal

        goal_edges = population.get("goal_edges") or ()
        if isinstance(goal_edges, Mapping):
            goal_edges = (goal_edges,)
        goal_edge_count = 0
        for item in goal_edges:
            if not isinstance(item, Mapping):
                continue
            parent_ref = str(item.get("parent_goal_cid") or item.get("parent") or "")
            child_ref = str(item.get("child_goal_cid") or item.get("child") or "")
            parent_cid = goal_cids_by_alias.get(parent_ref, parent_ref)
            child_cid = goal_cids_by_alias.get(child_ref, child_ref)
            self._intent.link_goal_edge(
                parent_goal_cid=parent_cid,
                child_goal_cid=child_cid,
                edge_kind=str(item.get("edge_kind") or "goal_dependency"),
            )
            goal_edge_count += 1

        plans = population.get("plans") or ()
        if isinstance(plans, Mapping):
            plans = (plans,)
        plan_cids: list[str] = []
        for index, item in enumerate(plans):
            if not isinstance(item, Mapping):
                continue
            plan_cid = str(item.get("plan_cid") or item.get("plan_id") or f"plan:{index + 1}")
            plan_alias = str(item.get("plan_alias") or item.get("alias") or plan_cid)
            goal_cid = str(item.get("goal_cid") or default_goal)
            self._intent.upsert_plan(
                plan_cid=plan_cid,
                goal_cid=goal_cid,
                plan_alias=plan_alias,
                status=str(item.get("status") or "active"),
                body=dict(item),
            )
            plan_cids.append(plan_cid)

        if root:
            self.plan_root_cid = root
        elif plan_cids:
            self.plan_root_cid = plan_cids[0]
        else:
            # Synthetic plan head so snapshot identity is non-empty.
            synthetic = content_identity({"repository_tree_id": tree_id, "goals": goal_cids})
            self._intent.upsert_plan(
                plan_cid=synthetic,
                goal_cid=default_goal,
                plan_alias="plan-root",
                status="active",
                body={"repository_tree_id": tree_id},
            )
            self.plan_root_cid = synthetic
            plan_cids.append(synthetic)

        taskboard = population.get("taskboard") or population.get("tasks") or ()
        if isinstance(taskboard, Mapping):
            taskboard = (taskboard,)
        task_cids: list[str] = []
        for index, item in enumerate(taskboard):
            if not isinstance(item, Mapping):
                continue
            task_cid = str(item.get("task_cid") or item.get("cid") or f"task:cid:{index + 1}")
            task_alias = str(
                item.get("task_id") or item.get("task_alias") or item.get("alias") or task_cid
            )
            goal_ref = str(item.get("goal_cid") or item.get("goal_id") or default_goal)
            # Resolve aliases to the durable goal_cid before task insert.
            existing_goal = self._intent.get_goal(goal_ref)
            if existing_goal is not None:
                goal_cid = str(existing_goal["goal_cid"])
            else:
                goal_cid = goal_ref
                self._intent.upsert_goal(
                    goal_cid=goal_cid,
                    goal_alias=str(item.get("goal_id") or goal_cid),
                    title=str(item.get("goal_id") or goal_cid),
                    ordinal=index + 1,
                )
            deps_raw = item.get("depends_on") or item.get("dependencies") or ()
            if isinstance(deps_raw, str):
                dependencies = [deps_raw]
            elif isinstance(deps_raw, Sequence):
                dependencies = [str(dep) for dep in deps_raw]
            else:
                dependencies = []
            # Resolve dependency aliases to durable CIDs before task insert so
            # readiness joins never depend on mutable display aliases.
            resolved_deps: list[str] = []
            for dep in dependencies:
                dep_text = str(dep or "").strip()
                if not dep_text:
                    continue
                prior = self._intent.get_task(dep_text)
                resolved_deps.append(str(prior["task_cid"]) if prior is not None else dep_text)
            outputs_raw = item.get("effects") or item.get("outputs") or ()
            outputs: list[Mapping[str, Any]] = []
            if isinstance(outputs_raw, Sequence):
                for effect in outputs_raw:
                    if isinstance(effect, Mapping):
                        outputs.append(dict(effect))
            acceptance_raw = item.get("acceptance_criteria") or item.get("acceptance") or ()
            acceptance: list[Any] = []
            if isinstance(acceptance_raw, (str, Mapping)):
                acceptance = [acceptance_raw]
            elif isinstance(acceptance_raw, Sequence):
                acceptance = list(acceptance_raw)
            validations_raw = item.get("validation_commands") or item.get("validations") or ()
            validations: list[Any] = []
            if isinstance(validations_raw, str):
                validations = [validations_raw]
            elif isinstance(validations_raw, Sequence):
                validations = list(validations_raw)
            self._intent.upsert_task(
                task_cid=task_cid,
                task_alias=task_alias,
                goal_cid=goal_cid,
                plan_cid=str(item.get("plan_cid") or self.plan_root_cid or ""),
                objective_id=str(item.get("objective_id") or ""),
                ordinal=int(item.get("ordinal") or index + 1),
                status=str(item.get("status") or "ready"),
                priority=str(item.get("priority") or "P2"),
                body={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "task_cid",
                        "task_id",
                        "task_alias",
                        "cid",
                        "goal_cid",
                        "goal_id",
                        "depends_on",
                        "dependencies",
                        "effects",
                        "outputs",
                        "acceptance_criteria",
                        "acceptance",
                        "validation_commands",
                        "validations",
                        "status",
                        "priority",
                        "ordinal",
                        "plan_cid",
                        "objective_id",
                    }
                },
                identity={
                    "task_cid": task_cid,
                    "task_alias": task_alias,
                    "repository_tree_id": tree_id,
                },
                dependencies=resolved_deps,
                outputs=outputs,
                acceptance=acceptance,
                validations=validations,
            )
            task_cids.append(task_cid)

        snap = self._intent.snapshot()
        receipt = {
            "schema": DATABASE_TASK_SOURCE_SCHEMA,
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": tree_id,
            "projection_cid": snap.projection_cid,
            "task_count": len(task_cids),
            "goal_count": len(goal_cids),
            "goal_edge_count": goal_edge_count,
            "plan_count": len(plan_cids),
            "event_watermark": snap.event_watermark,
            "task_cids": list(task_cids),
        }
        return MappingProxyType(receipt)

    # -- reads ---------------------------------------------------------------

    def snapshot(self) -> TaskSourceSnapshot:
        snap = self._intent.snapshot()
        terminal = True
        plan_root = self.plan_root_cid
        tasks = self._intent.list_tasks(limit=MAX_QUERY_LIMIT)
        task_plan_cids = {
            str(task.get("plan_cid") or "")
            for task in tasks
            if str(task.get("plan_cid") or "")
        }
        if not plan_root and len(task_plan_cids) == 1:
            # A reopened adapter has no in-memory materialization root.  The
            # task rows retain their admitted canonical plan binding, even
            # when their refinement-goal heads do not own the root plan.
            plan_root = next(iter(task_plan_cids))
        infer_from_goal_heads = not task_plan_cids
        for task in tasks:
            status = str(task.get("status") or "")
            if status not in {
                "completed",
                "skipped",
                "cancelled",
                "failed",
                "quarantined",
                "complete",
                "done",
            }:
                terminal = False
            if not plan_root and infer_from_goal_heads:
                head = self.plans.head(str(task.get("goal_cid") or ""))
                if head is not None:
                    plan_root = head.plan_cid
        return TaskSourceSnapshot(
            source_schema=DATABASE_TASK_SOURCE_SCHEMA,
            schema_version=1,
            plan_root_cid=plan_root,
            repository_tree_id=self.repository_tree_id,
            projection_cid=snap.projection_cid,
            formal_plan_id=plan_root,
            source_identity=content_identity(
                {
                    "plan_root_cid": plan_root,
                    "repository_tree_id": self.repository_tree_id,
                    "projection_cid": snap.projection_cid,
                }
            ),
            revision=max(1, snap.event_watermark),
            event_cursor=snap.event_watermark,
            goal_count=snap.goal_count,
            task_count=snap.task_count,
            dependency_count=snap.dependency_count,
            terminal=terminal and snap.task_count > 0,
            objective_count=snap.objective_count,
            plan_count=snap.plan_count,
        )

    def plan_projection(
        self, *, task_cids: Sequence[str] = ()
    ) -> Mapping[str, Any]:
        """Forward the full-fidelity intent plan projection."""

        return self._intent.plan_projection(task_cids=task_cids)

    def task_revision_history_projection(
        self, task_cid_or_alias: str
    ) -> Mapping[str, Any]:
        """Forward bounded lifecycle bodies used for legacy spec-CID replay."""

        return self._intent.task_revision_history_projection(task_cid_or_alias)

    def completion_evidence_projection(
        self, *, task_cids: Sequence[str] = ()
    ) -> Mapping[str, Any]:
        """Forward exact completion receipts without creating new authority."""

        return self._intent.completion_evidence_projection(task_cids=task_cids)

    def plan_revision_projection_cid(self) -> str:
        """Return the full plan projection CID used for revision verification."""

        return str(self.plan_projection().get("projection_cid") or "")

    def plan_revision_projection_paths(self) -> tuple[Path, ...]:
        """Declare local files mutated by a plan-revision apply."""

        if isinstance(self.database_path, Path):
            return (self.database_path.resolve(),)
        raise TaskSourceIntegrityError(
            "remote state-owner targets do not expose rollback file paths"
        )

    @staticmethod
    def _plan_population_mapping(source: Any) -> Mapping[str, Any]:
        if isinstance(source, Mapping):
            return source
        for method_name in ("to_dict", "to_record"):
            method = getattr(source, method_name, None)
            if callable(method):
                projected = method()
                if isinstance(projected, Mapping):
                    return projected
        raise TaskSourceIntegrityError(
            "plan revision input must expose a canonical mapping projection"
        )

    @staticmethod
    def _projection_tasks(
        projection: Mapping[str, Any], *, noun: str
    ) -> dict[str, Mapping[str, Any]]:
        raw_tasks = projection.get("tasks")
        if not isinstance(raw_tasks, list):
            raise TaskSourceIntegrityError(f"{noun} has no typed task population")
        tasks: dict[str, Mapping[str, Any]] = {}
        for item in raw_tasks:
            if not isinstance(item, Mapping):
                raise TaskSourceIntegrityError(f"{noun} task record is malformed")
            task_cid = str(item.get("task_cid") or "")
            if not task_cid or task_cid in tasks:
                raise TaskSourceIntegrityError(
                    f"{noun} task identities are missing or duplicated"
                )
            tasks[task_cid] = item
        return tasks

    @staticmethod
    def _lifecycle_for_status(status: str) -> Any:
        from ..planning.plan_revision_contracts import LifecycleState

        normalized = str(status or "").strip().lower()
        if normalized in {"proposed"}:
            return LifecycleState.PROPOSED
        if normalized in {"admitted"}:
            return LifecycleState.ADMITTED
        if normalized in {"ready", "todo", "pending", "queued", "retrying"}:
            return LifecycleState.UNSTARTED
        if normalized == "blocked":
            return LifecycleState.BLOCKED
        if normalized == "claimed":
            return LifecycleState.CLAIMED
        if normalized in {"in_progress", "running"}:
            return LifecycleState.RUNNING
        if normalized in {"completed", "complete", "done", "skipped"}:
            return LifecycleState.COMPLETED
        if normalized in {"failed", "cancelled", "quarantined", "rejected"}:
            return LifecycleState.FAILED
        raise TaskSourceIntegrityError(
            f"task status {status!r} has no plan-revision lifecycle mapping"
        )

    @staticmethod
    def _task_upsert_relations(
        task: Mapping[str, Any],
    ) -> tuple[list[str], list[Mapping[str, Any]], list[Any], list[Any]]:
        dependencies = [
            str(item.get("dependency_task_cid") or "")
            for item in task.get("dependencies", [])
            if isinstance(item, Mapping)
        ]
        if any(not dependency for dependency in dependencies):
            raise TaskSourceIntegrityError(
                "candidate task contains an empty dependency identity"
            )
        outputs: list[Mapping[str, Any]] = []
        for item in task.get("outputs", []):
            if not isinstance(item, Mapping) or not isinstance(
                item.get("effect"), Mapping
            ):
                raise TaskSourceIntegrityError(
                    "candidate task output effect must be a mapping"
                )
            outputs.append(dict(item["effect"]))
        acceptance: list[Any] = []
        for item in task.get("acceptance", []):
            if not isinstance(item, Mapping) or not isinstance(
                item.get("evidence_policy"), Mapping
            ):
                raise TaskSourceIntegrityError(
                    "candidate task acceptance policy must be a mapping"
                )
            acceptance.append(dict(item["evidence_policy"]))
        validations: list[Any] = []
        for item in task.get("validations", []):
            if not isinstance(item, Mapping) or not isinstance(
                item.get("policy"), Mapping
            ):
                raise TaskSourceIntegrityError(
                    "candidate task validation policy must be a mapping"
                )
            validations.append(
                {"argv": list(item.get("argv") or ()), **dict(item["policy"])}
            )
        return dependencies, outputs, acceptance, validations

    def apply_plan_revision(
        self,
        *,
        revision: Any = None,
        admission: Any = None,
        goal_graph: Any = None,
        aliases: Mapping[str, str] | None = None,
        repository_tree_id: str = "",
        retained_task_cids: Sequence[str] = (),
        claimed_task_cids: Sequence[str] = (),
        deferred_item_keys: Sequence[str] = (),
        origin: str = "create",
        delta: Any = None,
        store_continuation: Any | None = None,
        idempotency_key: str = "",
        fencing_token: int | None = None,
    ) -> Mapping[str, Any]:
        """Apply a checked create/steer revision without rewriting task history.

        Steer applies are intentionally narrow: additive tasks and exact-CAS
        amendments of unstarted/blocked task specifications.  Existing logical
        task CIDs, lifecycle states, completion receipts, attempts, and accepted
        history remain in the live IntentRepository.  PlanRevisionStore owns the
        enclosing byte backup/rollback transaction.
        """

        del aliases
        source = goal_graph if goal_graph is not None else admission
        population = self._plan_population_mapping(source)
        normalized_origin = str(origin or "").strip().lower()
        if normalized_origin.endswith("create"):
            if self.plan_projection().get("tasks"):
                raise TaskSourceConflictError(
                    "create cannot rewrite an existing task population; replay "
                    "must be resolved by PlanRevisionStore"
                )
            result = self.materialize(
                population,
                repository_tree_id=repository_tree_id,
                plan_root_cid=str(getattr(revision, "plan_root_cid", "") or ""),
            )
            return MappingProxyType(
                {
                    **dict(result),
                    "projection_cid": self.plan_revision_projection_cid(),
                    "deferred_item_keys": list(deferred_item_keys),
                }
            )

        if not normalized_origin.endswith("steer"):
            raise TaskSourceIntegrityError(f"unsupported plan origin: {origin!r}")
        if store_continuation is None or not idempotency_key:
            raise TaskSourceConflictError(
                "steer requires PlanRevisionStore rollback authority"
            )
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TaskSourceConflictError("steer requires a fencing token")
        if fencing_token < 1:
            raise TaskSourceConflictError("steer fencing token must be positive")
        if delta is None or revision is None:
            raise TaskSourceIntegrityError("steer requires a revision and closed delta")

        current_projection = self.plan_projection()
        current_tasks = self._projection_tasks(
            current_projection, noun="current plan projection"
        )
        source_root = str(self.plan_root_cid or "")
        if not source_root:
            active_plans = [
                item
                for item in current_projection.get("plans", [])
                if isinstance(item, Mapping) and item.get("status") == "active"
            ]
            if len(active_plans) != 1:
                raise TaskSourceIntegrityError(
                    "steer requires one exact active predecessor plan"
                )
            source_root = str(active_plans[0].get("plan_cid") or "")
        predecessor = self._intent.get_plan(source_root)
        if predecessor is None:
            raise TaskSourceIntegrityError(
                "steer predecessor plan is absent from the intent repository"
            )

        with tempfile.TemporaryDirectory(prefix="intent-plan-candidate-") as temp_dir:
            candidate_source = DatabaseTaskSource(
                Path(temp_dir) / "candidate.duckdb",
                repository_tree_id=repository_tree_id,
                plan_root_cid=str(getattr(revision, "plan_root_cid", "") or ""),
            )
            try:
                candidate_source.materialize(
                    population,
                    repository_tree_id=repository_tree_id,
                    plan_root_cid=str(
                        getattr(revision, "plan_root_cid", "") or ""
                    ),
                )
                candidate_projection = candidate_source.plan_projection()
            finally:
                candidate_source.close()
        candidate_tasks = self._projection_tasks(
            candidate_projection, noun="candidate plan projection"
        )

        current_cids = set(current_tasks)
        candidate_cids = set(candidate_tasks)
        if current_cids - candidate_cids:
            raise TaskSourceConflictError(
                "steer candidate would drop existing logical tasks: "
                + ", ".join(sorted(current_cids - candidate_cids))
            )
        retained = {str(task_cid) for task_cid in retained_task_cids}
        claimed = {str(task_cid) for task_cid in claimed_task_cids}
        if (retained | claimed) - current_cids:
            raise TaskSourceConflictError(
                "steer lifecycle population references an unknown live task"
            )

        from ..planning.plan_revision_contracts import (
            DeltaEffectClass,
            LifecycleState,
            PlanDeltaOperation,
        )

        amendments: dict[str, Any] = {}
        additions: dict[str, Any] = {}
        for item in tuple(getattr(delta, "items", ())):
            effect_class = getattr(item, "effect_class", None)
            if effect_class is not DeltaEffectClass.MATERIALIZABLE_NOW:
                continue
            operation = getattr(item, "operation", None)
            if operation in {
                PlanDeltaOperation.AMEND_UNSTARTED_TASK,
                PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
            }:
                target = str(getattr(item, "target_cid", "") or "")
                if not target or target in amendments:
                    raise TaskSourceIntegrityError(
                        "amend delta target is missing or duplicated"
                    )
                amendments[target] = item
            elif operation is PlanDeltaOperation.ADD_TASK:
                task_cid = str(getattr(item, "after_record_cid", "") or "")
                if not task_cid or task_cid in additions:
                    raise TaskSourceIntegrityError(
                        "add-task delta identity is missing or duplicated"
                    )
                additions[task_cid] = item
            elif operation in {
                PlanDeltaOperation.ATTACH_EVIDENCE,
                PlanDeltaOperation.RECORD_UNCERTAINTY,
            }:
                continue
            else:
                raise TaskSourceIntegrityError(
                    "DatabaseTaskSource cannot materialize operation "
                    f"{getattr(operation, 'value', operation)!r}"
                )

        new_cids = candidate_cids - current_cids
        if new_cids != set(additions):
            raise TaskSourceConflictError(
                "candidate task additions do not match the admitted delta"
            )
        changed_existing = {
            task_cid
            for task_cid in current_cids
            if task_projection_spec_cid(current_tasks[task_cid])
            != task_projection_spec_cid(candidate_tasks[task_cid])
        }
        if changed_existing != set(amendments):
            raise TaskSourceConflictError(
                "candidate task amendments do not match the admitted delta"
            )

        for task_cid, item in amendments.items():
            current = current_tasks[task_cid]
            candidate = candidate_tasks[task_cid]
            lifecycle = self._lifecycle_for_status(str(current.get("status") or ""))
            if lifecycle not in {
                LifecycleState.PROPOSED,
                LifecycleState.ADMITTED,
                LifecycleState.UNSTARTED,
                LifecycleState.READY,
                LifecycleState.BLOCKED,
            }:
                raise TaskSourceConflictError(
                    f"task {task_cid} is no longer amendable ({lifecycle.value})"
                )
            expected_lifecycle = getattr(item, "expected_target_lifecycle", None)
            if expected_lifecycle not in {lifecycle, LifecycleState.READY}:
                raise TaskSourceConflictError(
                    f"task {task_cid} lifecycle changed before steer apply"
                )
            live_spec = task_projection_spec_cid(current)
            if str(getattr(item, "expected_target_spec_revision", "")) != live_spec:
                raise TaskSourceConflictError(
                    f"task {task_cid} specification CAS is stale"
                )
            candidate_spec = task_projection_spec_cid(candidate)
            if str(getattr(item, "after_record_cid", "")) != candidate_spec:
                raise TaskSourceConflictError(
                    f"task {task_cid} replacement spec CID is not the candidate"
                )
        if claimed & changed_existing:
            raise TaskSourceConflictError("steer would amend claimed task history")

        for task_cid in sorted(amendments, key=lambda cid: (
            int(candidate_tasks[cid].get("ordinal") or 0), cid
        )):
            candidate = candidate_tasks[task_cid]
            live = current_tasks[task_cid]
            dependencies, outputs, acceptance, validations = (
                self._task_upsert_relations(candidate)
            )
            self._intent.upsert_task(
                task_cid=task_cid,
                task_alias=str(candidate["task_alias"]),
                goal_cid=str(candidate["goal_cid"]),
                plan_cid=str(live.get("plan_cid") or ""),
                objective_id=str(candidate.get("objective_id") or ""),
                ordinal=int(candidate.get("ordinal") or 0),
                status=str(live.get("status") or "ready"),
                priority=str(candidate.get("priority") or ""),
                body=dict(candidate.get("body") or {}),
                identity=dict(candidate.get("identity") or {}),
                dependencies=dependencies,
                outputs=outputs,
                acceptance=acceptance,
                validations=validations,
                expected_revision=int(live.get("revision") or 0),
            )

        candidate_root = str(getattr(revision, "plan_root_cid", "") or "")
        if not candidate_root:
            raise TaskSourceIntegrityError("steer revision has no plan root CID")
        for task_cid in sorted(additions, key=lambda cid: (
            int(candidate_tasks[cid].get("ordinal") or 0), cid
        )):
            candidate = candidate_tasks[task_cid]
            candidate_lifecycle = self._lifecycle_for_status(
                str(candidate.get("status") or "")
            )
            if candidate_lifecycle not in {
                LifecycleState.PROPOSED,
                LifecycleState.ADMITTED,
                LifecycleState.READY,
                LifecycleState.UNSTARTED,
                LifecycleState.BLOCKED,
            }:
                raise TaskSourceConflictError(
                    f"new task {task_cid} has a non-admissible initial lifecycle"
                )
            dependencies, outputs, acceptance, validations = (
                self._task_upsert_relations(candidate)
            )
            self._intent.upsert_task(
                task_cid=task_cid,
                task_alias=str(candidate["task_alias"]),
                goal_cid=str(candidate["goal_cid"]),
                plan_cid=candidate_root,
                objective_id=str(candidate.get("objective_id") or ""),
                ordinal=int(candidate.get("ordinal") or 0),
                status=str(candidate.get("status") or "ready"),
                priority=str(candidate.get("priority") or ""),
                body=dict(candidate.get("body") or {}),
                identity=dict(candidate.get("identity") or {}),
                dependencies=dependencies,
                outputs=outputs,
                acceptance=acceptance,
                validations=validations,
                expected_revision=0,
            )

        continuation = self.plans.continue_from(
            plan_cid=source_root,
            continuation_plan_cid=candidate_root,
            expected_revision=int(predecessor.get("revision") or 0),
            body={
                "revision_cid": str(getattr(revision, "revision_cid", "") or ""),
                "delta_cid": str(getattr(delta, "delta_cid", "") or ""),
                "idempotency_key": idempotency_key,
                "repository_tree_id": repository_tree_id,
            },
        )
        self.plan_root_cid = candidate_root
        projection = self.plan_projection()
        projected_tasks = self._projection_tasks(
            projection, noun="applied plan projection"
        )
        for task_cid in set(amendments) | set(additions):
            if task_projection_spec_cid(projected_tasks[task_cid]) != (
                task_projection_spec_cid(candidate_tasks[task_cid])
            ):
                raise TaskSourceIntegrityError(
                    f"task {task_cid} failed post-apply spec verification"
                )
        return MappingProxyType(
            {
                "projection_cid": str(projection.get("projection_cid") or ""),
                "receipt_cid": continuation.event_id,
                "plan_root_cid": candidate_root,
                "changed": bool(amendments or additions),
                "replayed": False,
                "amended_task_cids": sorted(amendments),
                "added_task_cids": sorted(additions),
                "deferred_item_keys": list(deferred_item_keys),
                "delta_cid": str(getattr(delta, "delta_cid", "") or ""),
            }
        )

    def get_task(
        self, task_cid_or_alias: str | TaskRecord | Mapping[str, Any]
    ) -> TaskRecord | None:
        key = _task_key(task_cid_or_alias)
        row = self._intent.get_task(key)
        if row is None:
            return None
        return _as_task_record(row)

    get = get_task

    def list_tasks(
        self,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
        snap = self._intent.snapshot()
        revision = max(1, snap.event_watermark)
        offset = _cursor_decode(cursor, revision=revision) if cursor else 0
        # Intent repository max page is MAX_QUERY_LIMIT; requesting limit+1 at
        # the ceiling raises. Cap the probe and treat a full max page as more.
        fetch_limit = min(limit + 1, MAX_QUERY_LIMIT)
        rows = self._intent.list_tasks(status=status, limit=fetch_limit, offset=offset)
        if limit >= MAX_QUERY_LIMIT:
            has_more = len(rows) >= MAX_QUERY_LIMIT
            page_rows = rows[:limit]
        else:
            has_more = len(rows) > limit
            page_rows = rows[:limit]
        tasks = tuple(_as_task_record(row) for row in page_rows)
        next_cursor = _cursor_encode(revision, offset + len(tasks)) if has_more else ""
        return TaskPage(tasks=tasks, revision=revision, next_cursor=next_cursor)

    def ready_tasks(
        self,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
        # completed_ids / blocked_ids are advisory overlays for callers that
        # track ephemeral state outside the durable projection.
        completed = {str(item).strip() for item in completed_ids if str(item).strip()}
        blocked = {str(item).strip() for item in blocked_ids if str(item).strip()}
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        selected = self._intent.select_ready_tasks(limit=MAX_QUERY_LIMIT)
        filtered: list[TaskRecord] = []
        for item in selected:
            tcid = str(item["task_cid"])
            alias = str(item.get("task_alias") or "")
            if tcid in blocked or alias in blocked:
                continue
            if tcid in completed or alias in completed:
                continue
            full = self._intent.get_task(tcid)
            if full is None:
                continue
            filtered.append(_as_task_record(full))
            if len(filtered) >= limit:
                break
        snap = self._intent.snapshot()
        return TaskPage(
            tasks=tuple(filtered),
            revision=max(1, snap.event_watermark),
        )

    readiness = ready_tasks

    def get_objective(self, objective_id: str) -> Mapping[str, Any] | None:
        return self._intent.get_objective(objective_id)

    def get_goal(self, goal_cid: str) -> Mapping[str, Any] | None:
        return self._intent.get_goal(goal_cid)

    def list_goal_edges(
        self,
        *,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return the bounded canonical goal-edge projection."""

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
        return self._intent.list_goal_edges(limit=limit)

    def get_plan(self, plan_cid: str) -> Mapping[str, Any] | None:
        return self._intent.get_plan(plan_cid)

    # -- mutations -----------------------------------------------------------

    def compare_and_set_status(
        self,
        task_cid_or_alias: str | TaskRecord | Mapping[str, Any],
        expected_revision: int,
        status: str,
        receipt: Mapping[str, Any] | None = None,
        *,
        evidence_digests: Sequence[str] | None = None,
    ) -> CASResult:
        key = _task_key(task_cid_or_alias)
        prior = self._intent.get_task(key)
        if prior is None:
            raise KeyError(key)
        previous_status = str(prior["status"])
        try:
            intent_receipt = self._intent.cas_task_status(
                task_cid=str(prior["task_cid"]),
                expected_revision=int(expected_revision),
                new_status=status,
                receipt=receipt,
                evidence_digests=evidence_digests,
            )
        except IntentCompletionError as exc:
            raise TaskSourceCompletionError(str(exc)) from exc
        except IntentRepositoryConflictError as exc:
            raise TaskSourceConflictError(str(exc)) from exc
        except IntentRepositoryTransitionError as exc:
            raise TaskSourceTransitionError(str(exc)) from exc
        except IntentRepositoryUnknownOutcomeError as exc:
            raise TaskSourceUnknownOutcomeError(str(exc)) from exc
        updated = self._intent.get_task(str(prior["task_cid"]))
        if updated is None:
            raise TaskSourceIntegrityError("task disappeared after CAS")
        task = _as_task_record(updated)
        receipt_cid = ""
        if intent_receipt.changed:
            details = dict(intent_receipt.details)
            receipt_cid = str(details.get("completion_receipt_cid") or "")
            if not receipt_cid and intent_receipt.event_id:
                receipt_cid = intent_receipt.event_id
        return CASResult(
            task=task,
            previous_status=previous_status,
            revision=int(intent_receipt.revision or task.revision),
            event_cursor=int(intent_receipt.global_sequence),
            changed=bool(intent_receipt.changed),
            receipt_cid=receipt_cid,
        )

    cas_status = compare_and_set_status

    def record_queue_backoff(
        self,
        *,
        task_cid: str,
        delay_ms: int,
        reason: str = "backoff",
        selection_penalty: int = 0,
    ) -> IntentReceipt:
        """Persist a task-selection cooldown through the intent authority.

        Database-authoritative executors must not rely on an attempt-local
        JSON queue: those projections are disposable and a replacement
        attempt receives a different state directory.  This checked adapter
        keeps the existing :class:`IntentRepository` as the sole queue
        authority while avoiding private repository access by callers.
        """

        return self._intent.record_queue_backoff(
            task_cid=task_cid,
            delay_ms=delay_ms,
            reason=reason,
            selection_penalty=selection_penalty,
        )

    def record_queue_retry(self, *, task_cid: str) -> IntentReceipt:
        """Clear one canonical task cooldown through the intent authority."""

        return self._intent.record_queue_retry(task_cid=task_cid)

    def get_queue_entry(self, task_cid: str) -> QueueEntry | None:
        """Return the canonical queue state without exposing raw SQL."""

        return self._intent.get_queue_entry(task_cid)

    def record_evidence(
        self,
        *,
        task_cid: str,
        evidence_kind: str,
        digest: str,
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        return self._intent.record_evidence(
            task_cid=task_cid,
            evidence_kind=evidence_kind,
            digest=digest,
            body=body,
        )

    def record_validation_result(
        self,
        *,
        task_cid: str,
        outcome: str,
        evidence_digest: str,
        argv: Sequence[str] | None = None,
        attempt_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        try:
            return self._intent.record_validation_result(
                task_cid=task_cid,
                outcome=outcome,
                evidence_digest=evidence_digest,
                argv=argv,
                attempt_id=attempt_id,
                body=body,
            )
        except IntentRepositoryUnknownOutcomeError as exc:
            raise TaskSourceUnknownOutcomeError(str(exc)) from exc

    def current_evidence_for_task(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return bounded current evidence through the canonical repository."""

        return self._intent.current_evidence_for_task(task_cid, now_ms=now_ms)

    def qualification_authority_for_task(
        self,
        task_cid: str,
    ) -> Mapping[str, Any]:
        """Return bounded rows that authorize qualification evidence."""

        return self._intent.qualification_authority_for_task(task_cid)

    def select_ready_tasks(self, *, limit: int = DEFAULT_QUERY_LIMIT) -> TaskPage:
        return self.ready_tasks(limit=limit)

    def rebuild_from_events(self) -> TaskSourceSnapshot:
        """Rebuild projections from admitted events and return a snapshot."""

        self._intent.rebuild_projections_from_events()
        return self.snapshot()

    def projection_matches_events(self) -> bool:
        """Return whether a rebuild yields the same projection CID."""

        before = self._intent.snapshot()
        after = self._intent.rebuild_projections_from_events()
        return before.projection_cid == after.projection_cid


__all__ = (
    "DATABASE_TASK_SOURCE_INTERFACE",
    "DATABASE_TASK_SOURCE_SCHEMA",
    "DatabaseTaskSource",
    "DatabaseTaskSourceError",
    "TaskSourceIntegrityError",
    "TaskSourceConflictError",
    "TaskSourceUnknownOutcomeError",
    "TaskSourceBoundsError",
    "TaskSourceCompletionError",
    "TaskRecord",
    "TaskPage",
    "CASResult",
    "TaskSourceSnapshot",
    "duckdb_available",
)
