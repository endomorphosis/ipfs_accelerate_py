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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import content_identity
from .control_plane_migrations import duckdb_available
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
    PlanRevisionRepository,
    open_intent_repository,
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
DATABASE_TASK_PAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-task-page@1"
)
DATABASE_TASK_CAS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-task-cas@1"
)

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
            MappingProxyType(dict(item))
            for item in outputs
            if isinstance(item, Mapping)
        ),
        acceptance=tuple(
            MappingProxyType(dict(item))
            for item in acceptance
            if isinstance(item, Mapping)
        ),
        validations=tuple(
            MappingProxyType(dict(item))
            for item in validations
            if isinstance(item, Mapping)
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
    ) -> None:
        if intent is not None:
            self._intent = intent
            self.database_path = Path(intent.database_path)
        else:
            if database_path is None:
                raise ValueError(
                    "DatabaseTaskSource requires database_path or intent"
                )
            self.database_path = Path(database_path).absolute()
            self._intent = open_intent_repository(
                self.database_path,
                owner_id=owner_id,
                install_schema=install_schema,
                evidence_freshness_seconds=evidence_freshness_seconds,
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
        root = str(
            plan_root_cid
            or population.get("plan_root_cid")
            or self.plan_root_cid
            or ""
        )

        objectives = population.get("objectives") or population.get("goals") or ()
        if isinstance(objectives, Mapping):
            objectives = (objectives,)
        goal_cids: list[str] = []
        for index, item in enumerate(objectives):
            if not isinstance(item, Mapping):
                continue
            goal_cid = str(
                item.get("goal_cid")
                or item.get("goal_id")
                or f"goal:cid:{index + 1}"
            )
            goal_alias = str(
                item.get("goal_alias")
                or item.get("goal_id")
                or item.get("alias")
                or goal_cid
            )
            objective_id = str(
                item.get("objective_id") or item.get("owner_actor_id") or ""
            )
            if objective_id:
                self._intent.upsert_objective(
                    objective_id=objective_id,
                    objective_alias=str(
                        item.get("objective_alias") or objective_id
                    ),
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

        default_goal = goal_cids[0] if goal_cids else "goal:default"
        if not goal_cids:
            self._intent.upsert_goal(
                goal_cid=default_goal,
                goal_alias="G-DEFAULT",
                title="Default goal",
                ordinal=1,
            )
            goal_cids.append(default_goal)

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
            synthetic = content_identity(
                {"repository_tree_id": tree_id, "goals": goal_cids}
            )
            self._intent.upsert_plan(
                plan_cid=synthetic,
                goal_cid=default_goal,
                plan_alias="plan-root",
                status="active",
                body={"repository_tree_id": tree_id},
            )
            self.plan_root_cid = synthetic
            plan_cids.append(synthetic)

        taskboard = (
            population.get("taskboard")
            or population.get("tasks")
            or ()
        )
        if isinstance(taskboard, Mapping):
            taskboard = (taskboard,)
        task_cids: list[str] = []
        for index, item in enumerate(taskboard):
            if not isinstance(item, Mapping):
                continue
            task_cid = str(
                item.get("task_cid")
                or item.get("cid")
                or f"task:cid:{index + 1}"
            )
            task_alias = str(
                item.get("task_id")
                or item.get("task_alias")
                or item.get("alias")
                or task_cid
            )
            goal_ref = str(
                item.get("goal_cid")
                or item.get("goal_id")
                or default_goal
            )
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
                resolved_deps.append(
                    str(prior["task_cid"]) if prior is not None else dep_text
                )
            outputs_raw = item.get("effects") or item.get("outputs") or ()
            outputs: list[Mapping[str, Any]] = []
            if isinstance(outputs_raw, Sequence):
                for effect in outputs_raw:
                    if isinstance(effect, Mapping):
                        outputs.append(dict(effect))
            acceptance_raw = (
                item.get("acceptance_criteria")
                or item.get("acceptance")
                or ()
            )
            acceptance: list[Any] = []
            if isinstance(acceptance_raw, (str, Mapping)):
                acceptance = [acceptance_raw]
            elif isinstance(acceptance_raw, Sequence):
                acceptance = list(acceptance_raw)
            validations_raw = item.get("validation_commands") or item.get(
                "validations"
            ) or ()
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
        for task in self._intent.list_tasks(limit=MAX_QUERY_LIMIT):
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
            if not plan_root:
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
            raise TaskSourceBoundsError(
                f"limit must be in [1, {MAX_QUERY_LIMIT}]"
            )
        snap = self._intent.snapshot()
        revision = max(1, snap.event_watermark)
        offset = _cursor_decode(cursor, revision=revision) if cursor else 0
        # Intent repository max page is MAX_QUERY_LIMIT; requesting limit+1 at
        # the ceiling raises. Cap the probe and treat a full max page as more.
        fetch_limit = min(limit + 1, MAX_QUERY_LIMIT)
        rows = self._intent.list_tasks(
            status=status, limit=fetch_limit, offset=offset
        )
        if limit >= MAX_QUERY_LIMIT:
            has_more = len(rows) >= MAX_QUERY_LIMIT
            page_rows = rows[:limit]
        else:
            has_more = len(rows) > limit
            page_rows = rows[:limit]
        tasks = tuple(_as_task_record(row) for row in page_rows)
        next_cursor = (
            _cursor_encode(revision, offset + len(tasks)) if has_more else ""
        )
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
            raise TaskSourceBoundsError(
                f"limit must be in [1, {MAX_QUERY_LIMIT}]"
            )
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
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        return self._intent.record_validation_result(
            task_cid=task_cid,
            outcome=outcome,
            evidence_digest=evidence_digest,
            argv=argv,
            body=body,
        )

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
    "TaskSourceBoundsError",
    "TaskSourceCompletionError",
    "TaskRecord",
    "TaskPage",
    "CASResult",
    "TaskSourceSnapshot",
    "duckdb_available",
)
