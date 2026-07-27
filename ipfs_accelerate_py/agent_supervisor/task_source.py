"""Backend-neutral task-source protocol for implementation daemons.

The Markdown and DuckDB projections intentionally retain their native storage
and concurrency models.  This module is the narrow integration boundary: it
normalizes immutable identities, bounded task pages, readiness, status CAS,
events, watches, and integrity results without translating either source into
the other.

Every :class:`CanonicalTaskSource` pins the immutable identity observed at
open time.  Replacing a database/board, changing its plan population, swapping
backend kinds, or presenting a foreign root/schema therefore fails before a
task can be returned or mutated.  Mutable revisions and event cursors may
advance, but stale query/watch/CAS inputs remain revision-bound.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Protocol, Sequence, runtime_checkable


TASK_SOURCE_PROTOCOL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-protocol@1"
)
TASK_SOURCE_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-identity@1"
)
TASK_SOURCE_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-task-source-snapshot@1"
)
TASK_SOURCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-operation-receipt@1"
)
TASK_SOURCE_SCHEMA_VERSION: Final = 1
DEFAULT_QUERY_LIMIT: Final = 100
MAX_QUERY_LIMIT: Final = 1_000
MAX_SNAPSHOT_TASKS: Final = 8_192
MAX_WATCH_SECONDS: Final = 30.0
SUPPORTED_SOURCE_KINDS: Final = frozenset({"markdown", "duckdb"})
COMPLETED_STATUSES: Final = frozenset({"completed", "complete", "done", "skipped"})
READY_STATUSES: Final = frozenset(
    {
        "todo",
        "queued",
        "proposed",
        "admitted",
        "pending",
        "ready",
        "retrying",
    }
)


class TaskSourceError(RuntimeError):
    """Base class for fail-closed common task-source failures."""


class TaskSourceIntegrityError(TaskSourceError):
    """The source is corrupt, foreign, unsupported, or changed identity."""


class TaskSourceConflictError(TaskSourceError):
    """A revision, cursor, status, lease, or writer fence is stale."""


class TaskSourceBoundsError(TaskSourceError, ValueError):
    """A bounded operation exceeded the common protocol limits."""


class UnsupportedTaskSourceError(TaskSourceError):
    """The configured object, backend kind, or schema is unsupported."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise TaskSourceIntegrityError("task-source value is not canonical JSON") from exc


def _operation_id(payload: Mapping[str, Any]) -> str:
    return "task-source:sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _bounded_limit(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > MAX_QUERY_LIMIT
    ):
        raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
    return limit


def _bounded_timeout(timeout: float) -> float:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
        or timeout > MAX_WATCH_SECONDS
    ):
        raise TaskSourceBoundsError(
            f"watch timeout must be between 0 and {MAX_WATCH_SECONDS:g} seconds"
        )
    return float(timeout)


def _status_values(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values or len(values) > 32:
        raise TaskSourceBoundsError("status filter is empty or exceeds its bound")
    selected = tuple(sorted({str(item).strip().lower() for item in values}))
    if any(not item or len(item) > 64 for item in selected):
        raise ValueError("status values must be bounded non-empty tokens")
    return selected


@dataclass(frozen=True)
class TaskSourceIdentity:
    """Immutable identity which must not change during one daemon run."""

    source_kind: str
    locator: str
    source_id: str
    root_id: str
    source_schema: str
    schema_version: int
    repository_root_id: str = ""
    protocol_schema: str = TASK_SOURCE_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.protocol_schema != TASK_SOURCE_PROTOCOL_SCHEMA:
            raise UnsupportedTaskSourceError("unsupported task-source protocol schema")
        if self.source_kind not in SUPPORTED_SOURCE_KINDS:
            raise UnsupportedTaskSourceError(
                f"unsupported task-source kind {self.source_kind!r}"
            )
        for name in ("locator", "source_id", "root_id", "source_schema"):
            value = str(getattr(self, name) or "").strip()
            if not value or "\x00" in value or "\n" in value or "\r" in value:
                raise TaskSourceIntegrityError(f"task-source {name} is missing or unsafe")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise UnsupportedTaskSourceError(
                "task-source schema version must be a positive integer"
            )

    @property
    def identity_id(self) -> str:
        return _operation_id(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": TASK_SOURCE_IDENTITY_SCHEMA,
            "protocol_schema": self.protocol_schema,
            "source_kind": self.source_kind,
            "locator": self.locator,
            "source_id": self.source_id,
            "root_id": self.root_id,
            "source_schema": self.source_schema,
            "schema_version": self.schema_version,
            "repository_root_id": self.repository_root_id,
        }
        if include_identity:
            value["identity_id"] = self.identity_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSourceIdentity":
        if value.get("schema") not in {None, TASK_SOURCE_IDENTITY_SCHEMA}:
            raise UnsupportedTaskSourceError("unsupported task-source identity schema")
        result = cls(
            source_kind=str(value.get("source_kind") or ""),
            locator=str(value.get("locator") or ""),
            source_id=str(value.get("source_id") or ""),
            root_id=str(value.get("root_id") or ""),
            source_schema=str(value.get("source_schema") or ""),
            schema_version=int(value.get("schema_version") or 0),
            repository_root_id=str(value.get("repository_root_id") or ""),
            protocol_schema=str(
                value.get("protocol_schema") or TASK_SOURCE_PROTOCOL_SCHEMA
            ),
        )
        claimed = str(value.get("identity_id") or "")
        if claimed and claimed != result.identity_id:
            raise TaskSourceIntegrityError("task-source identity digest does not match")
        return result


@dataclass(frozen=True)
class TaskSourceTask:
    """Canonical task record shared by Markdown and DuckDB consumers."""

    task_id: str
    task_cid: str
    goal_id: str
    goal_cid: str
    title: str
    status: str
    revision: str | int
    ordinal: int
    dependency_task_ids: tuple[str, ...] = ()
    dependency_task_cids: tuple[str, ...] = ()
    body: Mapping[str, Any] = field(default_factory=dict)
    board_namespace: str = ""
    source_line: int = 0

    @property
    def task_alias(self) -> str:
        return self.task_id

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.dependency_task_cids

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_alias": self.task_id,
            "task_cid": self.task_cid,
            "goal_id": self.goal_id,
            "goal_cid": self.goal_cid,
            "title": self.title,
            "status": self.status,
            "revision": self.revision,
            "ordinal": self.ordinal,
            "dependency_task_ids": list(self.dependency_task_ids),
            "dependency_task_cids": list(self.dependency_task_cids),
            "dependencies": list(self.dependency_task_cids),
            "body": dict(self.body),
            "board_namespace": self.board_namespace,
            "source_line": self.source_line,
        }


@dataclass(frozen=True)
class TaskSourceSnapshot:
    identity: TaskSourceIdentity
    revision: str | int
    event_cursor: Any
    task_count: int
    goal_count: int
    dependency_count: int
    terminal: bool
    tasks: tuple[TaskSourceTask, ...] = ()
    schema: str = TASK_SOURCE_SNAPSHOT_SCHEMA

    @property
    def source_id(self) -> str:
        return self.identity.source_id

    @property
    def root_id(self) -> str:
        return self.identity.root_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "identity": self.identity.to_dict(),
            "revision": self.revision,
            "event_cursor": self.event_cursor,
            "task_count": self.task_count,
            "goal_count": self.goal_count,
            "dependency_count": self.dependency_count,
            "terminal": self.terminal,
            "tasks": [item.to_dict() for item in self.tasks],
        }


@dataclass(frozen=True)
class TaskSourcePage:
    tasks: tuple[TaskSourceTask, ...]
    revision: str | int
    next_cursor: str = ""

    @property
    def records(self) -> tuple[TaskSourceTask, ...]:
        return self.tasks


@dataclass(frozen=True)
class TaskSourceCASResult:
    changed: bool
    task: TaskSourceTask
    previous_status: str
    revision: str | int
    event_cursor: Any
    receipt_id: str
    identity: TaskSourceIdentity

    @property
    def receipt_cid(self) -> str:
        return self.receipt_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_RECEIPT_SCHEMA,
            "operation": "compare_and_swap_status",
            "changed": self.changed,
            "task": self.task.to_dict(),
            "previous_status": self.previous_status,
            "revision": self.revision,
            "event_cursor": self.event_cursor,
            "receipt_id": self.receipt_id,
            "task_source_identity": self.identity.to_dict(),
        }


@dataclass(frozen=True)
class TaskSourceWatchResult:
    events: tuple[Mapping[str, Any], ...]
    cursor: str
    revision: str | int
    changed: bool
    timed_out: bool
    snapshot: TaskSourceSnapshot | None = None

    @property
    def next_cursor(self) -> str:
        return self.cursor


@dataclass(frozen=True)
class TaskSourceIntegrityReport:
    valid: bool
    identity: TaskSourceIdentity | None = None
    revision: str | int = ""
    event_cursor: Any = None
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.valid

    def require_valid(self) -> "TaskSourceIntegrityReport":
        if not self.valid:
            raise TaskSourceIntegrityError(
                "task-source integrity failed: " + ", ".join(self.issues)
            )
        return self


@runtime_checkable
class TaskSource(Protocol):
    """Minimal common protocol consumed by implementation daemons."""

    @property
    def identity(self) -> TaskSourceIdentity: ...

    @property
    def path(self) -> Path: ...

    def snapshot(self, *, include_tasks: bool = False) -> TaskSourceSnapshot: ...

    def query(
        self,
        *,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourcePage: ...

    def get(self, task_id: str) -> TaskSourceTask | None: ...

    def ready_set(
        self,
        *,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourcePage: ...

    def compare_and_swap_status(
        self,
        task_id: str,
        *,
        expected_status: str | Sequence[str],
        new_status: str,
        expected_revision: str | int,
        receipt: Mapping[str, Any] | None = None,
    ) -> TaskSourceCASResult: ...

    def append_event(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def watch(
        self,
        *,
        cursor: str = "",
        timeout: float = 0.0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourceWatchResult: ...

    def check_integrity(self) -> TaskSourceIntegrityReport: ...


def _encode_cursor(
    *,
    purpose: str,
    identity: TaskSourceIdentity,
    revision: str | int,
    backend_cursor: Any,
) -> str:
    payload = {
        "v": TASK_SOURCE_SCHEMA_VERSION,
        "purpose": purpose,
        "identity_id": identity.identity_id,
        "revision": revision,
        "backend_cursor": backend_cursor,
    }
    digest = hashlib.sha256(
        b"canonical-task-source-cursor-v1\0" + _canonical_bytes(payload)
    ).hexdigest()
    raw = _canonical_bytes({"payload": payload, "digest": digest})
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(
    cursor: str,
    *,
    purpose: str,
    identity: TaskSourceIdentity,
    revision: str | int,
) -> Any:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 16_384:
        raise TaskSourceConflictError("task-source cursor is malformed")
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        envelope = json.loads(raw)
        payload = envelope["payload"]
        claimed = envelope["digest"]
        expected = hashlib.sha256(
            b"canonical-task-source-cursor-v1\0" + _canonical_bytes(payload)
        ).hexdigest()
    except Exception as exc:
        raise TaskSourceConflictError("task-source cursor is malformed") from exc
    if claimed != expected:
        raise TaskSourceConflictError("task-source cursor digest does not match")
    if (
        payload.get("v") != TASK_SOURCE_SCHEMA_VERSION
        or payload.get("purpose") != purpose
        or payload.get("identity_id") != identity.identity_id
        or payload.get("revision") != revision
    ):
        raise TaskSourceConflictError("task-source cursor is stale or foreign")
    return payload.get("backend_cursor")


class CanonicalTaskSource:
    """Normalize one native Markdown or DuckDB task source."""

    def __init__(
        self,
        backend: Any,
        *,
        source_kind: str = "",
        expected_identity: TaskSourceIdentity | Mapping[str, Any] | None = None,
        expected_root_id: str = "",
        expected_repository_root_id: str = "",
    ) -> None:
        self.backend = backend
        self.source_kind = self._detect_kind(backend, source_kind)
        backend_path = getattr(backend, "path", None)
        if backend_path is None:
            backend_path = getattr(backend, "database_path", None)
        if backend_path is None:
            raise UnsupportedTaskSourceError("task-source backend has no path")
        self.path = Path(backend_path).absolute()
        self.events_path = Path(
            getattr(backend, "events_path", self.path)
        ).absolute()
        observed = self._observe_identity()
        expected = (
            TaskSourceIdentity.from_dict(expected_identity)
            if isinstance(expected_identity, Mapping)
            else expected_identity
        )
        if expected is not None and observed != expected:
            raise TaskSourceIntegrityError(
                "configured task source does not match its expected identity"
            )
        if expected_root_id and observed.root_id != expected_root_id:
            raise TaskSourceIntegrityError("task source has a foreign plan root")
        if (
            expected_repository_root_id
            and observed.repository_root_id != expected_repository_root_id
        ):
            raise TaskSourceIntegrityError("task source has a foreign repository root")
        self._identity = observed
        self.check_integrity().require_valid()

    @staticmethod
    def _detect_kind(backend: Any, selected: str) -> str:
        kind = str(selected or "").strip().lower()
        module = type(backend).__module__
        name = type(backend).__name__
        inferred = (
            "markdown"
            if module.endswith("markdown_task_source") and name == "MarkdownTaskSource"
            else (
                "duckdb"
                if module.endswith("duckdb_task_source") and name == "DuckDBTaskSource"
                else ""
            )
        )
        if kind and kind not in SUPPORTED_SOURCE_KINDS:
            raise UnsupportedTaskSourceError(f"unsupported task-source kind {kind!r}")
        if kind and inferred and kind != inferred:
            raise UnsupportedTaskSourceError(
                "configured task-source kind disagrees with the backend object"
            )
        if not (kind or inferred):
            raise UnsupportedTaskSourceError(
                "backend is not a supported Markdown or DuckDB task source"
            )
        return kind or inferred

    @property
    def identity(self) -> TaskSourceIdentity:
        self._require_pinned()
        return self._identity

    @property
    def pinned_identity(self) -> TaskSourceIdentity:
        """Return the identity accepted at open time without reading storage."""

        return self._identity

    def _observe_identity(self) -> TaskSourceIdentity:
        if self.source_kind == "markdown":
            from .markdown_task_source import MARKDOWN_TASK_SOURCE_VERSION

            snapshot = self.backend.snapshot()
            return TaskSourceIdentity(
                source_kind="markdown",
                locator=str(self.path),
                source_id=str(snapshot.projection_id or ""),
                root_id=str(snapshot.plan_root or ""),
                source_schema=str(snapshot.projection_schema or ""),
                schema_version=MARKDOWN_TASK_SOURCE_VERSION,
            )
        snapshot = self.backend.snapshot()
        return TaskSourceIdentity(
            source_kind="duckdb",
            locator=str(self.path),
            source_id=str(snapshot.projection_cid or ""),
            root_id=str(snapshot.plan_root_cid or ""),
            source_schema=str(snapshot.source_schema or ""),
            schema_version=int(snapshot.schema_version or 0),
            repository_root_id=str(snapshot.repository_tree_id or ""),
        )

    def _require_pinned(self) -> TaskSourceIdentity:
        try:
            current = self._observe_identity()
        except Exception as exc:
            raise TaskSourceIntegrityError(
                f"could not verify task-source identity: {exc}"
            ) from exc
        if current != self._identity:
            raise TaskSourceIntegrityError(
                "task-source identity changed during the daemon run"
            )
        return current

    @staticmethod
    def _markdown_task(record: Any, revision: str, ordinal: int) -> TaskSourceTask:
        metadata = dict(record.metadata)
        body = metadata.get("task_record")
        if not isinstance(body, Mapping):
            raise TaskSourceIntegrityError("Markdown task lacks its canonical body")
        return TaskSourceTask(
            task_id=str(record.task_id),
            task_cid=str(record.task_cid),
            goal_id=str(record.goal_id),
            goal_cid=str(record.goal_cid),
            title=str(record.title),
            status=str(record.status).lower(),
            revision=revision,
            ordinal=ordinal,
            dependency_task_ids=tuple(record.dependency_task_ids),
            dependency_task_cids=tuple(record.dependency_task_cids),
            body=dict(body),
            board_namespace=str(record.board_namespace),
            source_line=int(record.source_line),
        )

    @staticmethod
    def _duckdb_task(
        record: Any,
        aliases: Mapping[str, str] | None = None,
    ) -> TaskSourceTask:
        body = dict(record.body)
        dependency_cids = tuple(str(item) for item in record.dependencies)
        alias_map = dict(aliases or {})
        dependency_aliases = tuple(
            alias_map.get(item, item) for item in dependency_cids
        )
        title = str(
            body.get("objective")
            or body.get("title")
            or body.get("description")
            or ""
        )
        goal_id = str(
            body.get("goal_id")
            or body.get("goal_key")
            or record.goal_cid
        )
        return TaskSourceTask(
            task_id=str(record.task_alias),
            task_cid=str(record.task_cid),
            goal_id=goal_id,
            goal_cid=str(record.goal_cid),
            title=title,
            status=str(record.status).lower(),
            revision=int(record.revision),
            ordinal=int(record.ordinal),
            dependency_task_ids=dependency_aliases,
            dependency_task_cids=dependency_cids,
            body=body,
            board_namespace=str(
                body.get("board_namespace") or body.get("track") or "duckdb"
            ),
            source_line=int(record.ordinal) + 1,
        )

    def snapshot(self, *, include_tasks: bool = False) -> TaskSourceSnapshot:
        self._require_pinned()
        if self.source_kind == "markdown":
            raw = self.backend.snapshot()
            tasks = tuple(
                self._markdown_task(item, raw.board_revision, index)
                for index, item in enumerate(raw.tasks)
            )
            dependency_count = sum(
                len(item.dependency_task_cids) for item in tasks
            )
            terminal = bool(tasks) and all(
                item.status in COMPLETED_STATUSES
                or item.status
                in {"failed", "rejected", "cancelled", "quarantined"}
                for item in tasks
            )
            return TaskSourceSnapshot(
                identity=self._identity,
                revision=raw.board_revision,
                event_cursor=self._markdown_current_cursor(),
                task_count=len(tasks),
                goal_count=len({item.goal_cid for item in tasks}),
                dependency_count=dependency_count,
                terminal=terminal,
                tasks=tasks if include_tasks else (),
            )
        raw = self.backend.snapshot()
        tasks = self._all_duckdb_tasks() if include_tasks else ()
        return TaskSourceSnapshot(
            identity=self._identity,
            revision=int(raw.revision),
            event_cursor=int(raw.event_cursor),
            task_count=int(raw.task_count),
            goal_count=int(raw.goal_count),
            dependency_count=int(raw.dependency_count),
            terminal=bool(raw.terminal),
            tasks=tasks,
        )

    def _all_duckdb_tasks(self) -> tuple[TaskSourceTask, ...]:
        records: list[Any] = []
        cursor = ""
        while True:
            page = self.backend.list_tasks(cursor=cursor, limit=MAX_QUERY_LIMIT)
            records.extend(page.tasks)
            if len(records) > MAX_SNAPSHOT_TASKS:
                raise TaskSourceBoundsError(
                    "task population exceeds the common snapshot bound"
                )
            cursor = str(page.next_cursor or "")
            if not cursor:
                break
        aliases = {str(item.task_cid): str(item.task_alias) for item in records}
        tasks = tuple(self._duckdb_task(item, aliases) for item in records)
        return self._topological_tasks(tasks)

    @staticmethod
    def _topological_tasks(
        tasks: Sequence[TaskSourceTask],
    ) -> tuple[TaskSourceTask, ...]:
        by_cid = {item.task_cid: item for item in tasks}
        dependencies = {
            item.task_cid: set(item.dependency_task_cids) for item in tasks
        }
        dependents = {item.task_cid: set() for item in tasks}
        for task_cid, required in dependencies.items():
            unknown = required - set(by_cid)
            if unknown:
                raise TaskSourceIntegrityError(
                    f"task {task_cid!r} references unknown dependencies"
                )
            for dependency in required:
                dependents[dependency].add(task_cid)
        ready = sorted(
            task_cid
            for task_cid, required in dependencies.items()
            if not required
        )
        ordered: list[TaskSourceTask] = []
        while ready:
            task_cid = ready.pop(0)
            ordered.append(by_cid[task_cid])
            for dependent in sorted(dependents[task_cid]):
                dependencies[dependent].discard(task_cid)
                if (
                    not dependencies[dependent]
                    and dependent not in ready
                    and by_cid[dependent] not in ordered
                ):
                    ready.append(dependent)
                    ready.sort()
        if len(ordered) != len(tasks):
            raise TaskSourceIntegrityError("task dependency graph contains a cycle")
        return tuple(ordered)

    def query(
        self,
        *,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourcePage:
        selected_limit = _bounded_limit(limit)
        statuses = _status_values(status)
        current = self.snapshot(include_tasks=True)
        offset = (
            int(
                _decode_cursor(
                    cursor,
                    purpose="query",
                    identity=self._identity,
                    revision=current.revision,
                )
            )
            if cursor
            else 0
        )
        selected = tuple(
            item
            for item in current.tasks
            if not statuses or item.status in statuses
        )
        tasks = selected[offset : offset + selected_limit]
        next_offset = offset + len(tasks)
        next_cursor = (
            _encode_cursor(
                purpose="query",
                identity=self._identity,
                revision=current.revision,
                backend_cursor=next_offset,
            )
            if next_offset < len(selected)
            else ""
        )
        return TaskSourcePage(tasks=tasks, revision=current.revision, next_cursor=next_cursor)

    def _duckdb_aliases(self) -> dict[str, str]:
        return {
            item.task_cid: item.task_id for item in self._all_duckdb_tasks()
        }

    def get(self, task_id: str) -> TaskSourceTask | None:
        self._require_pinned()
        try:
            raw = self.backend.get(task_id)
        except Exception as exc:
            raise self._translated(exc) from exc
        if raw is None:
            return None
        if self.source_kind == "markdown":
            snapshot = self.backend.snapshot()
            ordinal = next(
                index
                for index, item in enumerate(snapshot.tasks)
                if item.task_cid == raw.task_cid
            )
            return self._markdown_task(raw, snapshot.board_revision, ordinal)
        return self._duckdb_task(raw, self._duckdb_aliases())

    get_task = get

    def ready_set(
        self,
        *,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourcePage:
        selected_limit = _bounded_limit(limit)
        completed = {str(item) for item in completed_ids}
        blocked = {str(item) for item in blocked_ids}
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        snapshot = self.snapshot(include_tasks=True)
        by_cid = {item.task_cid: item for item in snapshot.tasks}
        aliases = {item.task_id: item.task_cid for item in snapshot.tasks}
        resolved_completed = {aliases.get(item, item) for item in completed}
        resolved_blocked = {aliases.get(item, item) for item in blocked}
        unknown = (resolved_completed | resolved_blocked) - set(by_cid)
        if unknown:
            raise TaskSourceIntegrityError(
                "readiness input references unknown tasks: "
                + ", ".join(sorted(unknown))
            )
        satisfied = resolved_completed | {
            item.task_cid
            for item in snapshot.tasks
            if item.status in COMPLETED_STATUSES
        }
        unavailable = resolved_blocked | {
            item.task_cid for item in snapshot.tasks if item.status == "blocked"
        }
        ready = tuple(
            item
            for item in snapshot.tasks
            if item.status in READY_STATUSES
            and item.task_cid not in unavailable
            and all(dependency in satisfied for dependency in item.dependency_task_cids)
        )
        return TaskSourcePage(
            tasks=ready[:selected_limit],
            revision=snapshot.revision,
        )

    ready = ready_set
    ready_tasks = ready_set

    def compare_and_swap_status(
        self,
        task_id: str,
        *,
        expected_status: str | Sequence[str],
        new_status: str,
        expected_revision: str | int,
        receipt: Mapping[str, Any] | None = None,
    ) -> TaskSourceCASResult:
        self._require_pinned()
        expected = {
            str(item).strip().lower()
            for item in (
                (expected_status,)
                if isinstance(expected_status, str)
                else tuple(expected_status)
            )
        }
        if not expected:
            raise ValueError("expected_status must not be empty")
        current = self.get(task_id)
        if current is None:
            raise KeyError(task_id)
        if current.revision != expected_revision:
            raise TaskSourceConflictError("task revision CAS is stale")
        if current.status not in expected:
            raise TaskSourceConflictError("task status compare-and-swap conflict")
        payload = {
            **dict(receipt or {}),
            "task_source_identity_id": self._identity.identity_id,
        }
        try:
            if self.source_kind == "markdown":
                raw = self.backend.compare_and_swap_status(
                    task_id,
                    expected_status=tuple(sorted(expected)),
                    new_status=new_status,
                    expected_revision=str(expected_revision),
                    event_payload=payload,
                )
                task = self.get(task_id)
                assert task is not None
                event = dict(raw.event or {})
                event_cursor = event.get("sequence", "")
                receipt_id = str(event.get("event_id") or "")
                revision: str | int = raw.board_revision
                previous_status = current.status
            else:
                raw = self.backend.compare_and_set_status(
                    task_id,
                    int(expected_revision),
                    new_status,
                    payload,
                )
                task = self._duckdb_task(raw.task, self._duckdb_aliases())
                event_cursor = raw.event_cursor
                receipt_id = str(raw.receipt_cid or "")
                revision = raw.revision
                previous_status = str(raw.previous_status)
        except Exception as exc:
            raise self._translated(exc) from exc
        if not receipt_id:
            receipt_id = _operation_id(
                {
                    "schema": TASK_SOURCE_RECEIPT_SCHEMA,
                    "identity_id": self._identity.identity_id,
                    "task_cid": task.task_cid,
                    "previous_status": previous_status,
                    "status": task.status,
                    "revision": revision,
                    "changed": bool(raw.changed),
                }
            )
        return TaskSourceCASResult(
            changed=bool(raw.changed),
            task=task,
            previous_status=previous_status,
            revision=revision,
            event_cursor=event_cursor,
            receipt_id=receipt_id,
            identity=self._identity,
        )

    cas_status = compare_and_swap_status

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._require_pinned()
        enriched = {
            **dict(payload),
            "task_source_identity_id": self._identity.identity_id,
            "task_source_identity": self._identity.to_dict(),
        }
        try:
            if self.source_kind == "markdown":
                return self.backend.append_event(event_type, enriched)
            return self.backend.append_event(
                {**enriched, "event_type": event_type}
            )
        except Exception as exc:
            raise self._translated(exc) from exc

    def _markdown_initial_cursor(self) -> str:
        from .event_log import initial_event_cursor

        return initial_event_cursor(self.events_path).to_token()

    def _markdown_current_cursor(self) -> str:
        from .event_log import latest_event_cursor

        return latest_event_cursor(self.events_path).to_token()

    def watch(
        self,
        *,
        cursor: str = "",
        timeout: float = 0.0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourceWatchResult:
        selected_limit = _bounded_limit(limit)
        selected_timeout = _bounded_timeout(timeout)
        snapshot = self.snapshot()
        native_cursor = (
            _decode_cursor(
                cursor,
                purpose="watch",
                identity=self._identity,
                revision=snapshot.revision,
            )
            if cursor
            else (
                self._markdown_initial_cursor()
                if self.source_kind == "markdown"
                else 0
            )
        )
        try:
            if self.source_kind == "markdown":
                raw = self.backend.watch(
                    revision=str(snapshot.revision),
                    cursor=native_cursor,
                    timeout=selected_timeout,
                    event_limit=selected_limit,
                )
                next_native = (
                    raw.cursor.to_token()
                    if hasattr(raw.cursor, "to_token")
                    else str(raw.cursor or native_cursor)
                )
                revision = raw.snapshot.board_revision
                events = tuple(dict(item) for item in raw.events)
                changed = bool(raw.changed)
                timed_out = bool(raw.timed_out)
                next_snapshot = self.snapshot()
            else:
                raw = self.backend.watch(
                    cursor=int(native_cursor),
                    timeout=selected_timeout,
                    limit=selected_limit,
                )
                next_native = int(raw.cursor)
                revision = int(raw.revision)
                events = tuple(dict(item) for item in raw.events)
                changed = bool(events)
                timed_out = bool(raw.timed_out)
                next_snapshot = None
        except Exception as exc:
            raise self._translated(exc) from exc
        return TaskSourceWatchResult(
            events=events,
            cursor=_encode_cursor(
                purpose="watch",
                identity=self._identity,
                revision=revision,
                backend_cursor=next_native,
            ),
            revision=revision,
            changed=changed,
            timed_out=timed_out,
            snapshot=next_snapshot,
        )

    def check_integrity(self) -> TaskSourceIntegrityReport:
        try:
            current = self._observe_identity()
            if hasattr(self, "_identity") and current != self._identity:
                raise TaskSourceIntegrityError(
                    "task-source identity changed during the daemon run"
                )
            if self.source_kind == "markdown":
                native = self.backend.check_integrity()
                if not native.valid:
                    return TaskSourceIntegrityReport(
                        valid=False,
                        identity=current,
                        revision=str(native.board_revision or ""),
                        issues=tuple(native.reason_codes),
                    )
                snapshot = self.backend.snapshot()
                return TaskSourceIntegrityReport(
                    valid=True,
                    identity=current,
                    revision=snapshot.board_revision,
                    event_cursor=self._markdown_current_cursor(),
                )
            native = self.backend.validate_integrity()
            return TaskSourceIntegrityReport(
                valid=bool(native.valid),
                identity=current,
                revision=int(native.revision),
                event_cursor=int(native.event_cursor),
                issues=tuple(native.issues),
            )
        except Exception as exc:
            return TaskSourceIntegrityReport(
                valid=False,
                identity=getattr(self, "_identity", None),
                issues=(str(exc) or type(exc).__name__,),
            )

    integrity = check_integrity

    @staticmethod
    def _translated(exc: Exception) -> TaskSourceError:
        name = type(exc).__name__.lower()
        text = str(exc) or type(exc).__name__
        if isinstance(exc, TaskSourceError):
            return exc
        if "bound" in name or "limit" in text.lower():
            return TaskSourceBoundsError(text)
        if any(token in name for token in ("conflict", "stale")) or any(
            token in text.lower()
            for token in ("stale", "compare-and-swap", "fence", "conflict")
        ):
            return TaskSourceConflictError(text)
        return TaskSourceIntegrityError(text)


def open_task_source(
    source: Any,
    *,
    kind: str = "",
    root: Path | str | None = None,
    expected_identity: TaskSourceIdentity | Mapping[str, Any] | None = None,
    expected_root_id: str = "",
    expected_repository_root_id: str = "",
    **backend_options: Any,
) -> CanonicalTaskSource:
    """Open a native source or wrap an already-configured backend.

    Path inputs infer ``duckdb`` only from a ``.duckdb``/``.ddb`` suffix;
    every other path remains the backward-compatible Markdown default.
    """

    if isinstance(source, CanonicalTaskSource):
        if kind and source.source_kind != kind:
            raise UnsupportedTaskSourceError(
                "configured source kind disagrees with the open source"
            )
        source.check_integrity().require_valid()
        if expected_identity is not None:
            expected = (
                TaskSourceIdentity.from_dict(expected_identity)
                if isinstance(expected_identity, Mapping)
                else expected_identity
            )
            if source.identity != expected:
                raise TaskSourceIntegrityError(
                    "open source does not match expected identity"
                )
        if expected_root_id and source.identity.root_id != expected_root_id:
            raise TaskSourceIntegrityError("task source has a foreign plan root")
        if (
            expected_repository_root_id
            and source.identity.repository_root_id
            != expected_repository_root_id
        ):
            raise TaskSourceIntegrityError(
                "task source has a foreign repository root"
            )
        return source

    backend = source
    selected_kind = str(kind or "").strip().lower()
    if isinstance(source, (str, Path)):
        path = Path(source)
        selected_kind = selected_kind or (
            "duckdb" if path.suffix.lower() in {".duckdb", ".ddb"} else "markdown"
        )
        if selected_kind == "markdown":
            from .markdown_task_source import MarkdownTaskSource

            backend = MarkdownTaskSource(path, root=root, **backend_options)
        elif selected_kind == "duckdb":
            from .duckdb_task_source import DuckDBTaskSource

            backend = DuckDBTaskSource(path, **backend_options)
        else:
            raise UnsupportedTaskSourceError(
                f"unsupported task-source kind {selected_kind!r}"
            )
    return CanonicalTaskSource(
        backend,
        source_kind=selected_kind,
        expected_identity=expected_identity,
        expected_root_id=expected_root_id,
        expected_repository_root_id=expected_repository_root_id,
    )


adapt_task_source = open_task_source


__all__ = [
    "CanonicalTaskSource",
    "DEFAULT_QUERY_LIMIT",
    "MAX_QUERY_LIMIT",
    "MAX_SNAPSHOT_TASKS",
    "SUPPORTED_SOURCE_KINDS",
    "TASK_SOURCE_IDENTITY_SCHEMA",
    "TASK_SOURCE_PROTOCOL_SCHEMA",
    "TASK_SOURCE_SCHEMA_VERSION",
    "TaskSource",
    "TaskSourceBoundsError",
    "TaskSourceCASResult",
    "TaskSourceConflictError",
    "TaskSourceError",
    "TaskSourceIdentity",
    "TaskSourceIntegrityError",
    "TaskSourceIntegrityReport",
    "TaskSourcePage",
    "TaskSourceSnapshot",
    "TaskSourceTask",
    "TaskSourceWatchResult",
    "UnsupportedTaskSourceError",
    "adapt_task_source",
    "open_task_source",
]
