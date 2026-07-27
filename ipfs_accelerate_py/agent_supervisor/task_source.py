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
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
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
SUPPORTED_SOURCE_KINDS: Final = frozenset({"markdown", "duckdb", "dual"})
DUAL_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dual-task-source@1"
)
CANONICAL_PROJECTION_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-canonical-task-projection@1"
)
TASK_SOURCE_PARITY_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-parity-report@1"
)
DUAL_TASK_SOURCE_TRANSACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dual-task-source-transaction@1"
)
DUAL_TASK_SOURCE_JOURNAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dual-task-source-journal@1"
)
TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-projection-migration@1"
)
MAX_DUAL_EVENTS: Final = 100_000
MAX_DUAL_TRANSACTIONS: Final = 8_192
_MUTABLE_RECORD_FIELDS: Final = frozenset(
    {
        "status",
        "created_at",
        "created_at_ms",
        "updated_at",
        "updated_at_ms",
        "revision",
        "task_revision",
        "completion",
        "completion_receipt",
        "receipt",
    }
)
_READY_STATUS_EQUIVALENTS: Final = frozenset(
    {"todo", "queued", "proposed", "admitted", "pending", "ready", "retrying"}
)
_DUAL_MUTATION_STATUSES: Final = frozenset(
    {
        "proposed",
        "admitted",
        "ready",
        "in_progress",
        "blocked",
        "completed",
        "failed",
        "quarantined",
    }
)
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


class DualTaskSourcePartialError(TaskSourceConflictError):
    """One leg of a dual mutation committed and is durably recoverable."""

    def __init__(self, message: str, *, transaction_id: str = "") -> None:
        super().__init__(message)
        self.transaction_id = transaction_id


class TaskSourceQuarantinedError(TaskSourceIntegrityError):
    """A projection or dual transaction was quarantined after disagreement."""


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


def _semantic_status(value: Any) -> str:
    selected = str(value or "").strip().lower()
    if selected in _READY_STATUS_EQUIVALENTS:
        return "ready"
    if selected in {"complete", "done"}:
        return "completed"
    if selected == "running":
        return "in_progress"
    return selected


def _immutable_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the status-independent canonical record used by both projections."""

    return {
        str(key): member
        for key, member in json.loads(_canonical_bytes(dict(value))).items()
        if str(key) not in _MUTABLE_RECORD_FIELDS
    }


def _json_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(member) for key, member in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_json_plain(member) for member in value]
    return value


def _frozen_mapping(value: Mapping[str, Any], *, noun: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(_canonical_bytes(_json_plain(value)))
    except (TypeError, ValueError) as exc:
        raise TaskSourceIntegrityError(f"{noun} is not canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise TaskSourceIntegrityError(f"{noun} must be an object")
    return MappingProxyType(decoded)


def _pairs(
    value: Mapping[str, Any] | Iterable[Sequence[Any]],
    *,
    noun: str,
) -> tuple[tuple[str, Any], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    selected: list[tuple[str, Any]] = []
    for item in items:
        if isinstance(item, (str, bytes)) or len(item) != 2:
            raise TaskSourceIntegrityError(f"{noun} entries must be key/value pairs")
        key, member = item
        selected.append((str(key), member))
    selected.sort(key=lambda item: item[0])
    if len({key for key, _member in selected}) != len(selected):
        raise TaskSourceIntegrityError(f"{noun} contains duplicate keys")
    return tuple(selected)


@dataclass(frozen=True)
class CanonicalProjectionSnapshot:
    """Verified, backend-neutral migration and parity snapshot.

    ``graph_record`` is optional for parity but is required when rebuilding a
    Markdown projection.  It is carried through DuckDB migration receipts so a
    later reverse migration never has to infer or forge the original graph.
    """

    plan_root: str
    task_cids: tuple[str, ...]
    goal_cids: tuple[str, ...]
    task_aliases: tuple[tuple[str, str], ...]
    goal_aliases: tuple[tuple[str, str], ...]
    task_records: tuple[tuple[str, Mapping[str, Any]], ...]
    goal_records: tuple[tuple[str, Mapping[str, Any]], ...]
    dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    statuses: tuple[tuple[str, str], ...]
    task_revisions: tuple[tuple[str, int], ...]
    ready_task_cids: tuple[str, ...]
    revision: int
    events: tuple[Mapping[str, Any], ...]
    terminal: bool
    admitted_plan_root: str = ""
    repository_root_id: str = ""
    board_namespace: str = "prompt-workflow"
    graph_record: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CANONICAL_PROJECTION_SNAPSHOT_SCHEMA
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_PROJECTION_SNAPSHOT_SCHEMA:
            raise UnsupportedTaskSourceError(
                "unsupported canonical projection snapshot schema"
            )
        if not str(self.plan_root or "").strip():
            raise TaskSourceIntegrityError("canonical projection plan root is missing")
        task_cids = tuple(str(item) for item in self.task_cids)
        goal_cids = tuple(str(item) for item in self.goal_cids)
        if (
            not task_cids
            or len(task_cids) > MAX_SNAPSHOT_TASKS
            or len(task_cids) != len(set(task_cids))
            or not goal_cids
            or len(goal_cids) != len(set(goal_cids))
        ):
            raise TaskSourceIntegrityError(
                "canonical projection has an invalid task or goal population"
            )
        task_aliases = tuple(
            (str(key), str(value))
            for key, value in _pairs(self.task_aliases, noun="task aliases")
        )
        goal_aliases = tuple(
            (str(key), str(value))
            for key, value in _pairs(self.goal_aliases, noun="goal aliases")
        )
        task_records = tuple(
            (key, _frozen_mapping(value, noun=f"task record {key}"))
            for key, value in _pairs(self.task_records, noun="task records")
        )
        goal_records = tuple(
            (key, _frozen_mapping(value, noun=f"goal record {key}"))
            for key, value in _pairs(self.goal_records, noun="goal records")
        )
        dependencies = tuple(
            (key, tuple(str(member) for member in value))
            for key, value in _pairs(self.dependencies, noun="dependencies")
        )
        statuses = tuple(
            (key, _semantic_status(value))
            for key, value in _pairs(self.statuses, noun="statuses")
        )
        task_revisions = tuple(
            (key, int(value))
            for key, value in _pairs(self.task_revisions, noun="task revisions")
        )
        expected_tasks = set(task_cids)
        expected_goals = set(goal_cids)
        if any(
            {key for key, _value in population} != expected_tasks
            for population in (
                task_aliases,
                task_records,
                dependencies,
                statuses,
                task_revisions,
            )
        ):
            raise TaskSourceIntegrityError(
                "canonical projection task components disagree on population"
            )
        if (
            {key for key, _value in goal_aliases} != expected_goals
            or {key for key, _value in goal_records} != expected_goals
        ):
            raise TaskSourceIntegrityError(
                "canonical projection goal components disagree on population"
            )
        aliases = [value for _key, value in task_aliases]
        goal_alias_values = [value for _key, value in goal_aliases]
        if (
            len(aliases) != len(set(aliases))
            or len(goal_alias_values) != len(set(goal_alias_values))
        ):
            raise TaskSourceIntegrityError("canonical projection aliases are duplicated")
        for task_cid, required in dependencies:
            if task_cid in required or set(required) - expected_tasks:
                raise TaskSourceIntegrityError(
                    "canonical projection dependency graph is foreign or self-referential"
                )
        CanonicalTaskSource._topological_tasks(
            tuple(
                TaskSourceTask(
                    task_id=dict(task_aliases)[task_cid],
                    task_cid=task_cid,
                    goal_id=dict(goal_aliases)[
                        str(dict(task_records)[task_cid].get("goal_cid") or "")
                    ],
                    goal_cid=str(
                        dict(task_records)[task_cid].get("goal_cid") or ""
                    ),
                    title="canonical",
                    status=dict(statuses)[task_cid],
                    revision=dict(task_revisions)[task_cid],
                    ordinal=index,
                    dependency_task_ids=tuple(
                        dict(task_aliases)[item]
                        for item in dict(dependencies)[task_cid]
                    ),
                    dependency_task_cids=dict(dependencies)[task_cid],
                    body=dict(dict(task_records)[task_cid]),
                )
                for index, task_cid in enumerate(task_cids)
            )
        )
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise TaskSourceIntegrityError(
                "canonical projection revision must be positive"
            )
        events = tuple(
            _frozen_mapping(item, noun="canonical projection event")
            for item in self.events
        )
        if len(events) > MAX_DUAL_EVENTS:
            raise TaskSourceBoundsError("canonical event history exceeds its bound")
        if [int(item.get("sequence") or 0) for item in events] != list(
            range(1, len(events) + 1)
        ):
            raise TaskSourceIntegrityError(
                "canonical event sequence is not contiguous"
            )
        if self.revision != len(events) + 1:
            raise TaskSourceIntegrityError(
                "canonical revision does not match its event history"
            )
        event_revisions = {task_cid: 1 for task_cid in task_cids}
        for event in events:
            task_cid = str(event.get("task_cid") or "")
            if task_cid not in expected_tasks:
                raise TaskSourceIntegrityError(
                    "canonical event references an unknown task"
                )
            if event.get("event_type") == "status_changed":
                event_revisions[task_cid] += 1
                if int(event.get("task_revision") or 0) != event_revisions[task_cid]:
                    raise TaskSourceIntegrityError(
                        "canonical task revision history is not contiguous"
                    )
        if dict(task_revisions) != event_revisions:
            raise TaskSourceIntegrityError(
                "canonical task revisions do not match status events"
            )
        reconstructed_statuses = dict(statuses)
        for event in reversed(events):
            if event.get("event_type") != "status_changed":
                continue
            task_cid = str(event["task_cid"])
            if reconstructed_statuses[task_cid] != _semantic_status(
                event.get("status")
            ):
                raise TaskSourceIntegrityError(
                    "canonical status event outcome disagrees with current state"
                )
            reconstructed_statuses[task_cid] = _semantic_status(
                event.get("previous_status")
            )
        if any(value != "ready" for value in reconstructed_statuses.values()):
            raise TaskSourceIntegrityError(
                "canonical status history does not begin at the admitted ready state"
            )
        ready_task_cids = tuple(str(item) for item in self.ready_task_cids)
        if set(ready_task_cids) - expected_tasks:
            raise TaskSourceIntegrityError(
                "canonical ready set references an unknown task"
            )
        satisfied = {
            task_cid
            for task_cid, status in statuses
            if status in COMPLETED_STATUSES
        }
        expected_ready = tuple(
            task_cid
            for task_cid in task_cids
            if dict(statuses)[task_cid] == "ready"
            and all(
                dependency in satisfied
                for dependency in dict(dependencies)[task_cid]
            )
        )
        if ready_task_cids != expected_ready:
            raise TaskSourceIntegrityError(
                "canonical ready set does not match statuses and dependencies"
            )
        expected_terminal = all(
            status
            in {
                "completed",
                "skipped",
                "failed",
                "rejected",
                "cancelled",
                "quarantined",
            }
            for _task_cid, status in statuses
        )
        if bool(self.terminal) != expected_terminal:
            raise TaskSourceIntegrityError(
                "canonical terminal outcome does not match task statuses"
            )
        graph_record = _frozen_mapping(
            self.graph_record, noun="canonical graph record"
        )
        object.__setattr__(self, "task_cids", task_cids)
        object.__setattr__(self, "goal_cids", goal_cids)
        object.__setattr__(self, "task_aliases", task_aliases)
        object.__setattr__(self, "goal_aliases", goal_aliases)
        object.__setattr__(self, "task_records", task_records)
        object.__setattr__(self, "goal_records", goal_records)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "task_revisions", task_revisions)
        object.__setattr__(self, "ready_task_cids", ready_task_cids)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "graph_record", graph_record)
        expected_id = _operation_id(self.to_dict(include_snapshot_id=False))
        if self.snapshot_id and self.snapshot_id != expected_id:
            raise TaskSourceIntegrityError(
                "canonical projection snapshot digest does not match"
            )
        object.__setattr__(self, "snapshot_id", expected_id)

    def parity_dict(self) -> dict[str, Any]:
        """The exact cross-backend contract, excluding migration-only payload."""

        return {
            "plan_root": self.plan_root,
            "task_cids": list(self.task_cids),
            "goal_cids": list(self.goal_cids),
            "task_aliases": dict(self.task_aliases),
            "goal_aliases": dict(self.goal_aliases),
            "task_records": {
                key: dict(value) for key, value in self.task_records
            },
            "goal_records": {
                key: dict(value) for key, value in self.goal_records
            },
            "dependencies": {
                key: list(value) for key, value in self.dependencies
            },
            "statuses": dict(self.statuses),
            "task_revisions": dict(self.task_revisions),
            "ready_task_cids": list(self.ready_task_cids),
            "revision": self.revision,
            "events": [dict(item) for item in self.events],
            "terminal": self.terminal,
        }

    def to_dict(self, *, include_snapshot_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            **self.parity_dict(),
            "admitted_plan_root": self.admitted_plan_root,
            "repository_root_id": self.repository_root_id,
            "board_namespace": self.board_namespace,
            "graph_record": dict(self.graph_record),
        }
        if include_snapshot_id:
            value["snapshot_id"] = self.snapshot_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalProjectionSnapshot":
        allowed = {
            "schema",
            "plan_root",
            "task_cids",
            "goal_cids",
            "task_aliases",
            "goal_aliases",
            "task_records",
            "goal_records",
            "dependencies",
            "statuses",
            "task_revisions",
            "ready_task_cids",
            "revision",
            "events",
            "terminal",
            "admitted_plan_root",
            "repository_root_id",
            "board_namespace",
            "graph_record",
            "snapshot_id",
        }
        unknown = set(value) - allowed
        if unknown:
            raise TaskSourceIntegrityError(
                "canonical projection snapshot contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        claimed_snapshot_id = str(value.get("snapshot_id") or "")
        if claimed_snapshot_id:
            raw_material = {
                str(key): _json_plain(member)
                for key, member in value.items()
                if key != "snapshot_id"
            }
            if _operation_id(raw_material) != claimed_snapshot_id:
                raise TaskSourceIntegrityError(
                    "canonical projection snapshot digest does not match"
                )
        return cls(
            schema=str(value.get("schema") or ""),
            plan_root=str(value.get("plan_root") or ""),
            task_cids=tuple(value.get("task_cids") or ()),
            goal_cids=tuple(value.get("goal_cids") or ()),
            task_aliases=_pairs(
                value.get("task_aliases") or {}, noun="task aliases"
            ),
            goal_aliases=_pairs(
                value.get("goal_aliases") or {}, noun="goal aliases"
            ),
            task_records=_pairs(
                value.get("task_records") or {}, noun="task records"
            ),
            goal_records=_pairs(
                value.get("goal_records") or {}, noun="goal records"
            ),
            dependencies=tuple(
                (key, tuple(member))
                for key, member in _pairs(
                    value.get("dependencies") or {}, noun="dependencies"
                )
            ),
            statuses=_pairs(value.get("statuses") or {}, noun="statuses"),
            task_revisions=_pairs(
                value.get("task_revisions") or {}, noun="task revisions"
            ),
            ready_task_cids=tuple(value.get("ready_task_cids") or ()),
            revision=int(value.get("revision") or 0),
            events=tuple(value.get("events") or ()),
            terminal=bool(value.get("terminal")),
            admitted_plan_root=str(value.get("admitted_plan_root") or ""),
            repository_root_id=str(value.get("repository_root_id") or ""),
            board_namespace=str(value.get("board_namespace") or "prompt-workflow"),
            graph_record=dict(value.get("graph_record") or {}),
            snapshot_id=str(value.get("snapshot_id") or ""),
        )


@dataclass(frozen=True)
class TaskSourceParityReport:
    valid: bool
    left_snapshot_id: str
    right_snapshot_id: str
    parity_id: str
    mismatches: tuple[str, ...] = ()
    promotion_allowed: bool = False
    schema: str = TASK_SOURCE_PARITY_REPORT_SCHEMA

    @property
    def equivalent(self) -> bool:
        return self.valid

    def require_valid(self) -> "TaskSourceParityReport":
        if not self.valid:
            raise TaskSourceIntegrityError(
                "task-source parity disagreement: " + ", ".join(self.mismatches)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "equivalent": self.valid,
            "left_snapshot_id": self.left_snapshot_id,
            "right_snapshot_id": self.right_snapshot_id,
            "parity_id": self.parity_id,
            "mismatches": list(self.mismatches),
            "promotion_allowed": self.promotion_allowed,
        }


@dataclass(frozen=True)
class TaskSourceMigrationResult:
    source_snapshot_id: str
    target_snapshot_id: str
    target_kind: str
    target_path: Path
    changed: bool
    replayed: bool
    resumed: bool
    parity: TaskSourceParityReport
    receipt_id: str
    quarantine_path: Path | None = None
    schema: str = TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA

    @property
    def promotion_allowed(self) -> bool:
        return self.parity.promotion_allowed


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


def _canonical_event_receipt(
    value: Mapping[str, Any],
    *,
    markdown: bool,
) -> dict[str, Any]:
    generated = {
        "board_revision",
        "event_cid",
        "event_id",
        "event_type",
        "lease",
        "plan_root",
        "previous_board_revision",
        "previous_event_id",
        "previous_status",
        "revision",
        "schema",
        "sequence",
        "snapshot_id",
        "status",
        "stream_id",
        "task_cid",
        "task_id",
        "task_revision",
        "timestamp",
        "type",
        "task_source_identity",
        "task_source_identity_id",
        "dual_event",
        "logical_transaction_id",
    }
    receipt = dict(value) if markdown else dict(value.get("receipt") or {})
    return {
        str(key): member
        for key, member in receipt.items()
        if str(key) not in generated
    }


def _canonical_events(source: CanonicalTaskSource) -> tuple[Mapping[str, Any], ...]:
    raw_events: list[Mapping[str, Any]] = []
    if source.source_kind == "markdown":
        from .event_log import initial_event_cursor

        cursor: Any = initial_event_cursor(source.events_path)
        while True:
            page = source.backend.events(cursor, limit=MAX_QUERY_LIMIT)
            raw_events.extend(dict(item) for item in page.events)
            if len(raw_events) > MAX_DUAL_EVENTS:
                raise TaskSourceBoundsError("Markdown event history exceeds its bound")
            next_cursor = page.cursor
            if not page.events:
                break
            cursor = next_cursor
    else:
        cursor = 0
        while True:
            page = source.backend.events(cursor=cursor, limit=MAX_QUERY_LIMIT)
            raw_events.extend(dict(item) for item in page.events)
            if len(raw_events) > MAX_DUAL_EVENTS:
                raise TaskSourceBoundsError("DuckDB event history exceeds its bound")
            if not page.events:
                break
            cursor = int(page.cursor)

    selected: list[Mapping[str, Any]] = []
    task_revisions: dict[str, int] = {}
    for raw in raw_events:
        body = (
            dict(raw)
            if source.source_kind == "markdown"
            else dict(raw.get("body") or {})
        )
        receipt = (
            body
            if source.source_kind == "markdown"
            else dict(body.get("receipt") or {})
        )
        carried = body.get("dual_event") or receipt.get("dual_event")
        if isinstance(carried, Mapping):
            event = dict(carried)
            event["sequence"] = len(selected) + 1
            selected.append(_frozen_mapping(event, noun="carried canonical event"))
            if event.get("event_type") == "status_changed":
                task_revisions[str(event.get("task_cid") or "")] = int(
                    event.get("task_revision") or 0
                )
            continue
        raw_type = str(
            raw.get("type")
            or raw.get("event_type")
            or body.get("event_type")
            or ""
        )
        task_cid = str(raw.get("task_cid") or body.get("task_cid") or "")
        if raw_type in {"task_status_changed", "status_changed"}:
            task_revision = task_revisions.get(task_cid, 1) + 1
            task_revisions[task_cid] = task_revision
            event = {
                "sequence": len(selected) + 1,
                "event_type": "status_changed",
                "task_cid": task_cid,
                "previous_status": _semantic_status(body.get("previous_status")),
                "status": _semantic_status(body.get("status")),
                "task_revision": task_revision,
                "receipt": _canonical_event_receipt(
                    body, markdown=source.source_kind == "markdown"
                ),
            }
        else:
            payload = _canonical_event_receipt(
                body, markdown=source.source_kind == "markdown"
            )
            event = {
                "sequence": len(selected) + 1,
                "event_type": raw_type,
                "task_cid": task_cid,
                "payload": payload,
            }
        selected.append(_frozen_mapping(event, noun="canonical event"))
    return tuple(selected)


def _markdown_graph_record(raw: Any) -> dict[str, Any]:
    if not raw.tasks:
        return {}
    first = dict(raw.tasks[0].metadata)
    graph_core = first.get("graph_core")
    if not isinstance(graph_core, Mapping):
        raise TaskSourceIntegrityError("Markdown graph core is missing")
    goals: dict[str, Mapping[str, Any]] = {}
    tasks: dict[str, Mapping[str, Any]] = {}
    for item in raw.tasks:
        metadata = dict(item.metadata)
        task = metadata.get("task_record")
        if not isinstance(task, Mapping):
            raise TaskSourceIntegrityError("Markdown immutable task record is missing")
        tasks[str(item.task_cid)] = _immutable_record(task)
        for goal in metadata.get("goal_records") or ():
            if not isinstance(goal, Mapping):
                raise TaskSourceIntegrityError(
                    "Markdown immutable goal record is malformed"
                )
            goal_cid = str(goal.get("content_id") or "")
            goals[goal_cid] = _immutable_record(goal)
    return {
        **dict(graph_core),
        "goals": [dict(goals[key]) for key in sorted(goals)],
        "tasks": [dict(tasks[key]) for key in sorted(tasks)],
    }


def _duckdb_carried_metadata(source: CanonicalTaskSource) -> dict[str, Any]:
    try:
        rows = source.backend.query(
            "materialization_receipts", cursor=0, limit=MAX_QUERY_LIMIT
        )
    except Exception:
        return {}
    for row in rows:
        try:
            body = json.loads(str(row.get("body_json") or ""))
        except (TypeError, ValueError):
            continue
        receipt = body.get("receipt")
        if isinstance(receipt, Mapping) and isinstance(
            receipt.get("canonical_graph"), Mapping
        ):
            return json.loads(_canonical_bytes(dict(receipt)))
    return {}


def _duckdb_query_all(
    source: CanonicalTaskSource,
    table: str,
    *,
    maximum: int,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    cursor = 0
    while True:
        page = tuple(
            source.backend.query(
                table, cursor=cursor, limit=min(MAX_QUERY_LIMIT, maximum)
            )
        )
        records.extend(dict(item) for item in page)
        if len(records) > maximum:
            raise TaskSourceBoundsError(
                f"DuckDB {table} population exceeds the canonical snapshot bound"
            )
        if len(page) < min(MAX_QUERY_LIMIT, maximum):
            break
        cursor += len(page)
    return tuple(records)


def canonical_projection_snapshot(
    source: TaskSource | Any,
) -> CanonicalProjectionSnapshot:
    """Independently verify and export one projection into canonical form."""

    selected = (
        source
        if isinstance(source, CanonicalTaskSource)
        else open_task_source(source)
    )
    if isinstance(selected, DualTaskSource):
        return selected.canonical_snapshot()
    selected.check_integrity().require_valid()
    snapshot = selected.snapshot(include_tasks=True)
    events = _canonical_events(selected)
    task_event_revisions = {item.task_cid: 1 for item in snapshot.tasks}
    for event in events:
        if event.get("event_type") == "status_changed":
            task_event_revisions[str(event.get("task_cid") or "")] += 1

    if selected.source_kind == "markdown":
        raw = selected.backend.snapshot()
        first_metadata = dict(raw.tasks[0].metadata)
        plan_root = str(first_metadata.get("candidate_plan_root") or "")
        admitted_root = str(raw.plan_root or "")
        graph_record = _markdown_graph_record(raw)
        goal_records: dict[str, Mapping[str, Any]] = {}
        for item in raw.tasks:
            for goal in item.metadata.get("goal_records") or ():
                goal_records[str(goal.get("content_id") or "")] = _immutable_record(
                    goal
                )
        goal_aliases = {
            item.goal_cid: item.goal_id for item in snapshot.tasks
        }
        graph_goals = {
            str(item.get("content_id") or ""): item
            for item in graph_record.get("goals") or ()
        }
        for goal_cid, goal in graph_goals.items():
            goal_aliases.setdefault(goal_cid, str(goal.get("goal_key") or goal_cid))
        repository_root_id = str(graph_record.get("program_root") or "")
        board_namespace = str(raw.tasks[0].board_namespace or "prompt-workflow")
    else:
        plan_root = selected.identity.root_id
        carried = _duckdb_carried_metadata(selected)
        admitted_root = str(carried.get("admitted_plan_root") or "")
        graph_record = dict(carried.get("canonical_graph") or {})
        goal_records = {}
        goal_aliases = {}
        for row in _duckdb_query_all(
            selected, "goals", maximum=MAX_SNAPSHOT_TASKS
        ):
            goal_cid = str(row["goal_cid"])
            try:
                goal_body = json.loads(str(row["body_json"]))
            except (TypeError, ValueError) as exc:
                raise TaskSourceIntegrityError(
                    f"DuckDB goal {goal_cid!r} body is malformed"
                ) from exc
            goal_records[goal_cid] = _immutable_record(goal_body)
            goal_aliases[goal_cid] = str(row["goal_alias"])
        repository_root_id = selected.identity.repository_root_id
        board_namespace = str(
            carried.get("board_namespace")
            or (
                snapshot.tasks[0].board_namespace
                if snapshot.tasks
                else "prompt-workflow"
            )
        )

    tasks = tuple(snapshot.tasks)
    task_aliases = {item.task_cid: item.task_id for item in tasks}
    task_records = {
        item.task_cid: _immutable_record(item.body) for item in tasks
    }
    dependencies = {
        item.task_cid: tuple(item.dependency_task_cids) for item in tasks
    }
    statuses = {
        item.task_cid: _semantic_status(item.status) for item in tasks
    }
    satisfied = {
        task_cid
        for task_cid, status in statuses.items()
        if status in COMPLETED_STATUSES
    }
    ready_task_cids = tuple(
        item.task_cid
        for item in tasks
        if statuses[item.task_cid] == "ready"
        and all(dependency in satisfied for dependency in dependencies[item.task_cid])
    )
    terminal = bool(tasks) and all(
        status
        in {
            "completed",
            "skipped",
            "failed",
            "rejected",
            "cancelled",
            "quarantined",
        }
        for status in statuses.values()
    )
    result = CanonicalProjectionSnapshot(
        plan_root=plan_root,
        task_cids=tuple(item.task_cid for item in tasks),
        goal_cids=tuple(sorted(goal_records)),
        task_aliases=tuple(sorted(task_aliases.items())),
        goal_aliases=tuple(sorted(goal_aliases.items())),
        task_records=tuple(sorted(task_records.items())),
        goal_records=tuple(sorted(goal_records.items())),
        dependencies=tuple(sorted(dependencies.items())),
        statuses=tuple(sorted(statuses.items())),
        task_revisions=tuple(sorted(task_event_revisions.items())),
        ready_task_cids=ready_task_cids,
        revision=len(events) + 1,
        events=events,
        terminal=terminal,
        admitted_plan_root=admitted_root,
        repository_root_id=repository_root_id,
        board_namespace=board_namespace,
        graph_record=graph_record,
    )
    # A second read closes the snapshot race between the integrity check,
    # record queries, and event scan.
    final = selected.snapshot()
    if (
        final.revision != snapshot.revision
        or final.event_cursor != snapshot.event_cursor
        or final.task_count != snapshot.task_count
    ):
        raise TaskSourceConflictError(
            "task source changed while its canonical snapshot was captured"
        )
    return result


def compare_task_source_projections(
    left: CanonicalProjectionSnapshot | TaskSource | Any,
    right: CanonicalProjectionSnapshot | TaskSource | Any,
    *,
    promotion_blocked: bool = False,
) -> TaskSourceParityReport:
    left_snapshot = (
        left
        if isinstance(left, CanonicalProjectionSnapshot)
        else canonical_projection_snapshot(left)
    )
    right_snapshot = (
        right
        if isinstance(right, CanonicalProjectionSnapshot)
        else canonical_projection_snapshot(right)
    )
    left_value = left_snapshot.parity_dict()
    right_value = right_snapshot.parity_dict()
    component_names = (
        "plan_root",
        "task_cids",
        "goal_cids",
        "task_aliases",
        "goal_aliases",
        "task_records",
        "goal_records",
        "dependencies",
        "statuses",
        "task_revisions",
        "ready_task_cids",
        "revision",
        "events",
        "terminal",
    )
    mismatches = tuple(
        name for name in component_names if left_value[name] != right_value[name]
    )
    parity_id = _operation_id(
        {
            "schema": TASK_SOURCE_PARITY_REPORT_SCHEMA,
            "left": left_value,
            "right": right_value,
            "mismatches": list(mismatches),
        }
    )
    valid = not mismatches
    return TaskSourceParityReport(
        valid=valid,
        left_snapshot_id=left_snapshot.snapshot_id,
        right_snapshot_id=right_snapshot.snapshot_id,
        parity_id=parity_id,
        mismatches=mismatches,
        promotion_allowed=valid and not promotion_blocked,
    )


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class DualTaskSource:
    """Fenced logical task source over a primary and shadow projection.

    Cross-file atomic commit is impossible, so every mutation first persists a
    content-addressed transaction.  A crash after one native commit leaves a
    ``partial`` record which the next opener can finish exactly once.  Any
    state that cannot be explained by that record is quarantined and cannot be
    automatically promoted.
    """

    def __init__(
        self,
        primary: TaskSource | Any,
        shadow: TaskSource | Any,
        *,
        mode: str = "shadow",
        journal_path: Path | str | None = None,
        recover: bool = True,
        fault_injector: Any | None = None,
        auto_promote: bool = False,
    ) -> None:
        self.primary = (
            primary
            if isinstance(primary, CanonicalTaskSource)
            else open_task_source(primary)
        )
        self.shadow = (
            shadow
            if isinstance(shadow, CanonicalTaskSource)
            else open_task_source(shadow)
        )
        self.primary_source = self.primary
        self.shadow_source = self.shadow
        if isinstance(self.primary, DualTaskSource) or isinstance(
            self.shadow, DualTaskSource
        ):
            raise UnsupportedTaskSourceError("nested dual task sources are unsupported")
        if self.primary.source_kind == self.shadow.source_kind:
            raise UnsupportedTaskSourceError(
                "dual task source requires one Markdown and one DuckDB projection"
            )
        selected_mode = str(mode or "").strip().lower()
        if selected_mode not in {"shadow", "migration"}:
            raise ValueError("dual task-source mode must be shadow or migration")
        self.mode = selected_mode
        self.path = self.primary.path
        self.events_path = self.primary.events_path
        self.journal_path = Path(
            journal_path
            or self.primary.path.with_name(
                f".{self.primary.path.name}.dual-task-source.json"
            )
        ).absolute()
        self._lock_path = self.journal_path.with_name(
            f".{self.journal_path.name}.lock"
        )
        self._fault_injector = fault_injector
        self._promoted = False
        self._quarantined = False
        self._dual_id = _operation_id(
            {
                "schema": DUAL_TASK_SOURCE_SCHEMA,
                "primary": self.primary.pinned_identity.to_dict(),
                "shadow": self.shadow.pinned_identity.to_dict(),
            }
        )
        canonical_root = self._best_effort_root()
        self._identity = TaskSourceIdentity(
            source_kind="dual",
            locator=f"{self.primary.path}|{self.shadow.path}",
            source_id=self._dual_id,
            root_id=canonical_root,
            source_schema=DUAL_TASK_SOURCE_SCHEMA,
            schema_version=1,
            repository_root_id=(
                self.primary.identity.repository_root_id
                or self.shadow.identity.repository_root_id
            ),
        )
        self._ensure_journal()
        if recover:
            self.recover()
        if auto_promote:
            self.promote(automatic=True)

    def _best_effort_root(self) -> str:
        for source in (self.primary, self.shadow):
            try:
                if source.source_kind == "markdown":
                    raw = source.backend.snapshot()
                    if raw.tasks:
                        root = str(
                            raw.tasks[0].metadata.get("candidate_plan_root") or ""
                        )
                        if root:
                            return root
                elif source.identity.root_id:
                    return source.identity.root_id
            except Exception:
                continue
        return self.primary.pinned_identity.root_id

    @property
    def identity(self) -> TaskSourceIdentity:
        return self._identity

    @property
    def pinned_identity(self) -> TaskSourceIdentity:
        return self._identity

    @property
    def active(self) -> CanonicalTaskSource:
        return self.shadow if self._promoted else self.primary

    @property
    def promoted(self) -> bool:
        return self._promoted

    def _empty_journal(self) -> dict[str, Any]:
        return {
            "schema": DUAL_TASK_SOURCE_JOURNAL_SCHEMA,
            "dual_id": self._dual_id,
            "promoted": False,
            "quarantined": False,
            "operations": {},
        }

    def _journal_envelope(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "payload": dict(payload),
            "digest": hashlib.sha256(
                b"dual-task-source-journal-v1\0" + _canonical_bytes(payload)
            ).hexdigest(),
        }

    def _ensure_journal(self) -> None:
        if self.journal_path.exists():
            payload = self._read_journal()
            self._promoted = bool(payload.get("promoted"))
            self._quarantined = bool(payload.get("quarantined"))
            return
        _atomic_json_write(
            self.journal_path, self._journal_envelope(self._empty_journal())
        )

    def _read_journal(self) -> dict[str, Any]:
        try:
            envelope = json.loads(self.journal_path.read_bytes())
            payload = envelope["payload"]
            digest = envelope["digest"]
            expected = hashlib.sha256(
                b"dual-task-source-journal-v1\0" + _canonical_bytes(payload)
            ).hexdigest()
        except Exception as exc:
            raise TaskSourceQuarantinedError(
                "dual task-source journal is corrupt"
            ) from exc
        if (
            digest != expected
            or payload.get("schema") != DUAL_TASK_SOURCE_JOURNAL_SCHEMA
            or payload.get("dual_id") != self._dual_id
            or not isinstance(payload.get("operations"), Mapping)
        ):
            raise TaskSourceQuarantinedError(
                "dual task-source journal is corrupt or foreign"
            )
        if len(payload["operations"]) > MAX_DUAL_TRANSACTIONS:
            raise TaskSourceBoundsError(
                "dual task-source transaction journal exceeds its bound"
            )
        return dict(payload)

    def _write_journal(self, payload: Mapping[str, Any]) -> None:
        _atomic_json_write(self.journal_path, self._journal_envelope(payload))

    def _fault(self, point: str) -> None:
        if callable(self._fault_injector):
            self._fault_injector(point)

    def canonical_snapshot(
        self, source: str | TaskSource | None = None
    ) -> CanonicalProjectionSnapshot:
        if source is None:
            report, left, right = self._parity_triplet()
            report.require_valid()
            return right if self._promoted else left
        if source == "primary":
            return canonical_projection_snapshot(self.primary)
        if source == "shadow":
            return canonical_projection_snapshot(self.shadow)
        return canonical_projection_snapshot(source)

    snapshot_canonical = canonical_snapshot

    def _parity_triplet(
        self,
    ) -> tuple[
        TaskSourceParityReport,
        CanonicalProjectionSnapshot,
        CanonicalProjectionSnapshot,
    ]:
        left = canonical_projection_snapshot(self.primary)
        right = canonical_projection_snapshot(self.shadow)
        report = compare_task_source_projections(
            left,
            right,
            promotion_blocked=self._quarantined or self._has_pending(),
        )
        return report, left, right

    def parity(self) -> TaskSourceParityReport:
        return self._parity_triplet()[0]

    compare = parity
    check_parity = parity
    compare_parity = parity
    parity_report = parity

    def _has_pending(self) -> bool:
        try:
            operations = self._read_journal()["operations"]
        except TaskSourceError:
            return True
        return any(
            record.get("state") in {"prepared", "partial"}
            for record in operations.values()
        )

    def _quarantine_operation(
        self,
        payload: dict[str, Any],
        transaction_id: str,
        reason: str,
    ) -> None:
        operations = dict(payload["operations"])
        record = dict(operations[transaction_id])
        record.update({"state": "quarantined", "reason": reason})
        operations[transaction_id] = record
        payload.update({"operations": operations, "quarantined": True})
        self._write_journal(payload)
        self._quarantined = True

    def recover(self) -> tuple[str, ...]:
        """Resume, roll back, or quarantine every interrupted transaction."""

        from .duckdb_state import exclusive_file_lock

        recovered: list[str] = []
        with exclusive_file_lock(self._lock_path, timeout_seconds=30.0):
            payload = self._read_journal()
            operations = dict(payload["operations"])
            for transaction_id in sorted(operations):
                record = dict(operations[transaction_id])
                if record.get("state") not in {"prepared", "partial"}:
                    continue
                if record.get("operation") != "status":
                    self._quarantine_operation(
                        payload,
                        transaction_id,
                        "unsupported interrupted operation",
                    )
                    raise TaskSourceQuarantinedError(
                        "interrupted dual event append requires operator review"
                    )
                task_cid = str(record["task_cid"])
                target = str(record["new_status"])
                primary_task = self.primary.get(task_cid)
                shadow_task = self.shadow.get(task_cid)
                if primary_task is None or shadow_task is None:
                    self._quarantine_operation(
                        payload, transaction_id, "task population changed"
                    )
                    raise TaskSourceQuarantinedError(
                        "dual transaction task population changed during recovery"
                    )
                before_primary = str(record["primary_status"])
                before_shadow = str(record["shadow_status"])
                states = (primary_task.status, shadow_task.status)
                try:
                    if states == (before_primary, before_shadow):
                        record["state"] = "rolled_back"
                    elif states == (target, before_shadow):
                        self.shadow.compare_and_swap_status(
                            task_cid,
                            expected_status=before_shadow,
                            new_status=target,
                            expected_revision=shadow_task.revision,
                            receipt={
                                **dict(record.get("receipt") or {}),
                                "logical_transaction_id": transaction_id,
                                "dual_event": dict(record["canonical_event"]),
                            },
                        )
                        record["state"] = "committed"
                    elif states == (before_primary, target):
                        self.primary.compare_and_swap_status(
                            task_cid,
                            expected_status=before_primary,
                            new_status=target,
                            expected_revision=primary_task.revision,
                            receipt={
                                **dict(record.get("receipt") or {}),
                                "logical_transaction_id": transaction_id,
                                "dual_event": dict(record["canonical_event"]),
                            },
                        )
                        record["state"] = "committed"
                    elif states == (target, target):
                        record["state"] = "committed"
                    else:
                        raise TaskSourceQuarantinedError(
                            "projection statuses cannot be explained by transaction"
                        )
                    operations[transaction_id] = record
                    payload["operations"] = operations
                    self._write_journal(payload)
                    if record["state"] == "committed":
                        compare_task_source_projections(
                            self.primary, self.shadow
                        ).require_valid()
                    recovered.append(transaction_id)
                except Exception as exc:
                    self._quarantine_operation(
                        payload, transaction_id, str(exc) or type(exc).__name__
                    )
                    raise TaskSourceQuarantinedError(
                        f"dual transaction {transaction_id} was quarantined"
                    ) from exc
            self._promoted = bool(payload.get("promoted"))
            self._quarantined = bool(payload.get("quarantined"))
        return tuple(recovered)

    def _require_operable(
        self,
    ) -> tuple[
        TaskSourceParityReport,
        CanonicalProjectionSnapshot,
        CanonicalProjectionSnapshot,
    ]:
        if self._quarantined:
            raise TaskSourceQuarantinedError("dual task source is quarantined")
        if self._has_pending():
            self.recover()
        report, left, right = self._parity_triplet()
        report.require_valid()
        return report, left, right

    @staticmethod
    def _dual_tasks(
        source: CanonicalTaskSource,
        canonical: CanonicalProjectionSnapshot,
    ) -> tuple[TaskSourceTask, ...]:
        raw = source.snapshot(include_tasks=True).tasks
        revisions = dict(canonical.task_revisions)
        return tuple(
            TaskSourceTask(
                task_id=item.task_id,
                task_cid=item.task_cid,
                goal_id=item.goal_id,
                goal_cid=item.goal_cid,
                title=item.title,
                status=item.status,
                revision=revisions[item.task_cid],
                ordinal=item.ordinal,
                dependency_task_ids=item.dependency_task_ids,
                dependency_task_cids=item.dependency_task_cids,
                body=item.body,
                board_namespace=item.board_namespace,
                source_line=item.source_line,
            )
            for item in raw
        )

    def snapshot(self, *, include_tasks: bool = False) -> TaskSourceSnapshot:
        _report, left, right = self._require_operable()
        canonical = right if self._promoted else left
        tasks = (
            self._dual_tasks(self.active, canonical) if include_tasks else ()
        )
        return TaskSourceSnapshot(
            identity=self._identity,
            revision=canonical.revision,
            event_cursor=len(canonical.events),
            task_count=len(canonical.task_cids),
            goal_count=len(canonical.goal_cids),
            dependency_count=sum(
                len(value) for _key, value in canonical.dependencies
            ),
            terminal=canonical.terminal,
            tasks=tasks,
        )

    def query(
        self,
        *,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourcePage:
        selected_limit = _bounded_limit(limit)
        statuses = _status_values(status)
        snapshot = self.snapshot(include_tasks=True)
        offset = (
            int(
                _decode_cursor(
                    cursor,
                    purpose="query",
                    identity=self._identity,
                    revision=snapshot.revision,
                )
            )
            if cursor
            else 0
        )
        selected = tuple(
            task
            for task in snapshot.tasks
            if not statuses or task.status in statuses
        )
        tasks = selected[offset : offset + selected_limit]
        next_offset = offset + len(tasks)
        next_cursor = (
            _encode_cursor(
                purpose="query",
                identity=self._identity,
                revision=snapshot.revision,
                backend_cursor=next_offset,
            )
            if next_offset < len(selected)
            else ""
        )
        return TaskSourcePage(tasks, snapshot.revision, next_cursor)

    def get(self, task_id: str) -> TaskSourceTask | None:
        snapshot = self.snapshot(include_tasks=True)
        matches = [
            item
            for item in snapshot.tasks
            if item.task_id == task_id or item.task_cid == task_id
        ]
        if len(matches) > 1:
            raise TaskSourceIntegrityError("dual task lookup is ambiguous")
        return matches[0] if matches else None

    get_task = get

    def ready_set(
        self,
        *,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourcePage:
        selected_limit = _bounded_limit(limit)
        snapshot = self.snapshot(include_tasks=True)
        by_cid = {item.task_cid: item for item in snapshot.tasks}
        aliases = {item.task_id: item.task_cid for item in snapshot.tasks}
        completed = {aliases.get(str(item), str(item)) for item in completed_ids}
        blocked = {aliases.get(str(item), str(item)) for item in blocked_ids}
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        unknown = (completed | blocked) - set(by_cid)
        if unknown:
            raise TaskSourceIntegrityError(
                "readiness input references unknown tasks: "
                + ", ".join(sorted(unknown))
            )
        durable_completed = {
            item.task_cid
            for item in snapshot.tasks
            if _semantic_status(item.status) in COMPLETED_STATUSES
        }
        ready = tuple(
            item
            for item in snapshot.tasks
            if _semantic_status(item.status) == "ready"
            and item.task_cid not in blocked
            and all(
                dependency in durable_completed | completed
                for dependency in item.dependency_task_cids
            )
        )
        return TaskSourcePage(ready[:selected_limit], snapshot.revision)

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
        from .duckdb_state import exclusive_file_lock

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
        selected_new_status = str(new_status or "").strip().lower()
        if selected_new_status not in _DUAL_MUTATION_STATUSES:
            raise ValueError(
                "new_status is not supported by both task-source projections"
            )
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, (str, int))
        ):
            raise ValueError("expected_revision must be a task revision token")
        requested_receipt = json.loads(_canonical_bytes(dict(receipt or {})))
        with exclusive_file_lock(self._lock_path, timeout_seconds=30.0):
            _report, left, right = self._require_operable()
            canonical = right if self._promoted else left
            aliases = dict(canonical.task_aliases)
            reverse_aliases = {alias: cid for cid, alias in aliases.items()}
            task_cid = task_id if task_id in aliases else reverse_aliases.get(task_id)
            if task_cid is None:
                raise KeyError(task_id)
            current_task_revision = dict(canonical.task_revisions)[task_cid]
            if current_task_revision != expected_revision:
                # The exact original call may be a completed replay.
                transaction_id = _operation_id(
                    {
                        "schema": DUAL_TASK_SOURCE_TRANSACTION_SCHEMA,
                        "dual_id": self._dual_id,
                        "operation": "status",
                        "task_cid": task_cid,
                        "expected_status": sorted(expected),
                        "new_status": selected_new_status,
                        "expected_revision": expected_revision,
                        "receipt": requested_receipt,
                    }
                )
                record = self._read_journal()["operations"].get(transaction_id)
                if isinstance(record, Mapping) and record.get("state") == "committed":
                    task = self.get(task_cid)
                    assert task is not None
                    return TaskSourceCASResult(
                        changed=False,
                        task=task,
                        previous_status=str(record["primary_status"]),
                        revision=canonical.revision,
                        event_cursor=len(canonical.events),
                        receipt_id=transaction_id,
                        identity=self._identity,
                    )
                raise TaskSourceConflictError("task revision CAS is stale")
            primary_task = self.primary.get(task_cid)
            shadow_task = self.shadow.get(task_cid)
            if primary_task is None or shadow_task is None:
                raise TaskSourceIntegrityError(
                    "dual task population changed before mutation"
                )
            active_status = (
                shadow_task.status if self._promoted else primary_task.status
            )
            if active_status not in expected:
                raise TaskSourceConflictError(
                    "task status compare-and-swap conflict"
                )
            if _semantic_status(primary_task.status) != _semantic_status(
                shadow_task.status
            ):
                raise TaskSourceIntegrityError(
                    "dual task statuses disagree before mutation"
                )
            transaction_id = _operation_id(
                {
                    "schema": DUAL_TASK_SOURCE_TRANSACTION_SCHEMA,
                    "dual_id": self._dual_id,
                    "operation": "status",
                    "task_cid": task_cid,
                    "expected_status": sorted(expected),
                    "new_status": selected_new_status,
                    "expected_revision": expected_revision,
                    "receipt": requested_receipt,
                }
            )
            canonical_event = {
                "sequence": len(canonical.events) + 1,
                "event_type": "status_changed",
                "task_cid": task_cid,
                "previous_status": _semantic_status(active_status),
                "status": _semantic_status(selected_new_status),
                "task_revision": current_task_revision + 1,
                "receipt": requested_receipt,
            }
            payload = self._read_journal()
            operations = dict(payload["operations"])
            previous = operations.get(transaction_id)
            if isinstance(previous, Mapping) and previous.get("state") == "committed":
                task = self.get(task_cid)
                assert task is not None
                return TaskSourceCASResult(
                    changed=False,
                    task=task,
                    previous_status=str(previous["primary_status"]),
                    revision=canonical.revision,
                    event_cursor=len(canonical.events),
                    receipt_id=transaction_id,
                    identity=self._identity,
                )
            operations[transaction_id] = {
                "schema": DUAL_TASK_SOURCE_TRANSACTION_SCHEMA,
                "operation": "status",
                "state": "prepared",
                "task_cid": task_cid,
                "primary_status": primary_task.status,
                "shadow_status": shadow_task.status,
                "new_status": selected_new_status,
                "primary_revision": primary_task.revision,
                "shadow_revision": shadow_task.revision,
                "receipt": requested_receipt,
                "canonical_event": canonical_event,
            }
            payload["operations"] = operations
            self._write_journal(payload)
            self._fault("after_prepare")
            enriched = {
                **requested_receipt,
                "logical_transaction_id": transaction_id,
                "dual_event": canonical_event,
            }
            try:
                first_result = self.primary.compare_and_swap_status(
                    task_cid,
                    expected_status=primary_task.status,
                    new_status=selected_new_status,
                    expected_revision=primary_task.revision,
                    receipt=enriched,
                )
                operations[transaction_id]["state"] = "partial"
                payload["operations"] = operations
                self._write_journal(payload)
                self._fault("after_primary")
                second_result = self.shadow.compare_and_swap_status(
                    task_cid,
                    expected_status=shadow_task.status,
                    new_status=selected_new_status,
                    expected_revision=shadow_task.revision,
                    receipt=enriched,
                )
                self._fault("after_shadow")
                parity = compare_task_source_projections(
                    self.primary, self.shadow
                )
                parity.require_valid()
                operations[transaction_id]["state"] = "committed"
                operations[transaction_id]["primary_receipt_id"] = (
                    first_result.receipt_id
                )
                operations[transaction_id]["shadow_receipt_id"] = (
                    second_result.receipt_id
                )
                payload["operations"] = operations
                self._write_journal(payload)
            except BaseException as exc:
                # The prepared/partial record is already durable.  Do not
                # manufacture a compensating event; recovery will finish or
                # quarantine the exact transaction.
                raise DualTaskSourcePartialError(
                    "dual status mutation is recoverably partial",
                    transaction_id=transaction_id,
                ) from exc
            final = self.snapshot(include_tasks=True)
            task = next(item for item in final.tasks if item.task_cid == task_cid)
            return TaskSourceCASResult(
                changed=bool(first_result.changed or second_result.changed),
                task=task,
                previous_status=active_status,
                revision=final.revision,
                event_cursor=final.event_cursor,
                receipt_id=transaction_id,
                identity=self._identity,
            )

    cas_status = compare_and_swap_status

    def append_event(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Append a replay-safe event to both projections.

        Event appends use the same durable idempotency journal as status
        mutations.  They are intentionally not auto-recovered after a crash:
        unlike a status CAS, an arbitrary event has no independently visible
        before/after predicate.  Such an interruption is quarantined.
        """

        from .duckdb_state import exclusive_file_lock

        selected_payload = json.loads(_canonical_bytes(dict(payload)))
        with exclusive_file_lock(self._lock_path, timeout_seconds=30.0):
            _report, left, right = self._require_operable()
            canonical = right if self._promoted else left
            task_key = str(
                selected_payload.get("task_cid")
                or selected_payload.get("task_id")
                or ""
            )
            aliases = dict(canonical.task_aliases)
            reverse = {alias: cid for cid, alias in aliases.items()}
            task_cid = task_key if task_key in aliases else reverse.get(task_key, "")
            if not task_cid:
                raise TaskSourceIntegrityError(
                    "dual event must reference one canonical task"
                )
            transaction_id = _operation_id(
                {
                    "schema": DUAL_TASK_SOURCE_TRANSACTION_SCHEMA,
                    "dual_id": self._dual_id,
                    "operation": "event",
                    "event_type": event_type,
                    "payload": selected_payload,
                }
            )
            journal = self._read_journal()
            operations = dict(journal["operations"])
            existing = operations.get(transaction_id)
            if isinstance(existing, Mapping) and existing.get("state") == "committed":
                return {
                    "event_id": transaction_id,
                    "changed": False,
                    "revision": canonical.revision,
                }
            canonical_event = {
                "sequence": len(canonical.events) + 1,
                "event_type": str(event_type),
                "task_cid": task_cid,
                "payload": {
                    key: member
                    for key, member in selected_payload.items()
                    if key not in {"task_id", "task_cid"}
                },
            }
            operations[transaction_id] = {
                "schema": DUAL_TASK_SOURCE_TRANSACTION_SCHEMA,
                "operation": "event",
                "state": "prepared",
                "canonical_event": canonical_event,
            }
            journal["operations"] = operations
            self._write_journal(journal)
            enriched = {
                **selected_payload,
                "task_cid": task_cid,
                "event_cid": transaction_id,
                "logical_transaction_id": transaction_id,
                "dual_event": canonical_event,
            }
            try:
                primary_result = self.primary.append_event(event_type, enriched)
                operations[transaction_id]["state"] = "partial"
                journal["operations"] = operations
                self._write_journal(journal)
                shadow_result = self.shadow.append_event(event_type, enriched)
                compare_task_source_projections(
                    self.primary, self.shadow
                ).require_valid()
                operations[transaction_id]["state"] = "committed"
                journal["operations"] = operations
                self._write_journal(journal)
            except BaseException as exc:
                raise DualTaskSourcePartialError(
                    "dual event append is recoverably partial",
                    transaction_id=transaction_id,
                ) from exc
            return {
                "event_id": transaction_id,
                "changed": bool(
                    primary_result.get("changed", True)
                    or shadow_result.get("changed", True)
                ),
                "revision": canonical.revision + 1,
            }

    def watch(
        self,
        *,
        cursor: str = "",
        timeout: float = 0.0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskSourceWatchResult:
        selected_timeout = _bounded_timeout(timeout)
        selected_limit = _bounded_limit(limit)
        first = self.canonical_snapshot()
        offset = (
            int(
                _decode_cursor(
                    cursor,
                    purpose="watch",
                    identity=self._identity,
                    revision=first.revision,
                )
            )
            if cursor
            else 0
        )
        deadline = time.monotonic() + selected_timeout
        current = first
        while offset >= len(current.events) and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            current = self.canonical_snapshot()
        events = tuple(current.events[offset : offset + selected_limit])
        next_offset = offset + len(events)
        return TaskSourceWatchResult(
            events=events,
            cursor=_encode_cursor(
                purpose="watch",
                identity=self._identity,
                revision=current.revision,
                backend_cursor=next_offset,
            ),
            revision=current.revision,
            changed=bool(events),
            timed_out=not events,
            snapshot=self.snapshot(),
        )

    def check_integrity(self) -> TaskSourceIntegrityReport:
        issues: list[str] = []
        for name, source in (("primary", self.primary), ("shadow", self.shadow)):
            native = source.check_integrity()
            if not native.valid:
                issues.extend(f"{name}:{item}" for item in native.issues)
        try:
            report = self.parity()
            issues.extend(f"parity:{item}" for item in report.mismatches)
        except Exception as exc:
            issues.append(str(exc) or type(exc).__name__)
        if self._quarantined:
            issues.append("dual_source_quarantined")
        snapshot: TaskSourceSnapshot | None = None
        if not issues:
            try:
                snapshot = self.snapshot()
            except Exception as exc:
                issues.append(str(exc) or type(exc).__name__)
        return TaskSourceIntegrityReport(
            valid=not issues,
            identity=self._identity,
            revision="" if snapshot is None else snapshot.revision,
            event_cursor=None if snapshot is None else snapshot.event_cursor,
            issues=tuple(issues),
        )

    integrity = check_integrity

    def promote(self, *, automatic: bool = True) -> TaskSourceParityReport:
        """Promote the shadow only after a fresh exact parity verification."""

        from .duckdb_state import exclusive_file_lock

        with exclusive_file_lock(self._lock_path, timeout_seconds=30.0):
            report = self.parity()
            if not report.valid:
                raise TaskSourceIntegrityError(
                    "parity disagreement prevents projection promotion"
                )
            if automatic and not report.promotion_allowed:
                raise TaskSourceIntegrityError(
                    "automatic promotion is blocked by recovery or quarantine state"
                )
            payload = self._read_journal()
            payload["promoted"] = True
            self._write_journal(payload)
            self._promoted = True
            return report

    promote_shadow = promote

    def rebuild_projection(
        self,
        target: TaskSource | Any,
        *,
        snapshot: CanonicalProjectionSnapshot | Mapping[str, Any] | None = None,
        kind: str = "",
        repository_root_id: str = "",
        fault_injector: Any | None = None,
    ) -> TaskSourceMigrationResult:
        return rebuild_task_source_projection(
            snapshot or self.canonical_snapshot(),
            target,
            kind=kind,
            repository_root_id=repository_root_id,
            fault_injector=fault_injector,
        )

    migrate = rebuild_projection


def _graph_with_lifecycle(value: Mapping[str, Any]) -> dict[str, Any]:
    graph = json.loads(_canonical_bytes(dict(value)))
    for goal in graph.get("goals") or ():
        goal.update({"status": "proposed", "created_at_ms": 0, "updated_at_ms": 0})
    for task in graph.get("tasks") or ():
        task.update({"status": "proposed", "created_at_ms": 0, "updated_at_ms": 0})
    for evidence in graph.get("evidence") or ():
        evidence.update(
            {"status": "admitted", "created_at_ms": 0, "updated_at_ms": 0}
        )
    graph.update({"status": "proposed", "created_at_ms": 0, "updated_at_ms": 0})
    return graph


def _markdown_projection_from_snapshot(
    snapshot: CanonicalProjectionSnapshot,
) -> Any:
    if not snapshot.graph_record:
        raise TaskSourceIntegrityError(
            "a verified canonical graph payload is required to rebuild Markdown"
        )
    from .markdown_task_source import (
        MARKDOWN_TASK_SOURCE_SCHEMA,
        MarkdownTaskProjection,
        _projection_identity,
        _render_task_block,
        _semantic,
        _topological_tasks,
    )
    from .prompt_workflow import PromptGoalGraph

    try:
        graph = PromptGoalGraph.from_dict(
            _graph_with_lifecycle(snapshot.graph_record)
        )
    except Exception as exc:
        raise TaskSourceIntegrityError(
            "canonical graph payload cannot rebuild a verified Markdown projection"
        ) from exc
    if graph.plan_root_cid != snapshot.plan_root:
        raise TaskSourceIntegrityError(
            "canonical graph payload disagrees with the snapshot plan root"
        )
    task_aliases = dict(snapshot.task_aliases)
    goal_aliases = dict(snapshot.goal_aliases)
    ordered_tasks = _topological_tasks(graph)
    if tuple(item.task_cid for item in ordered_tasks) != snapshot.task_cids:
        raise TaskSourceIntegrityError(
            "canonical graph topology disagrees with the snapshot"
        )
    projection_root = snapshot.admitted_plan_root or snapshot.plan_root
    projection_id = _projection_identity(
        plan_root=projection_root,
        revision=1,
        task_aliases=task_aliases,
        goal_aliases=goal_aliases,
    )
    graph_semantic = _semantic(graph.to_dict())
    graph_core = {
        key: member
        for key, member in graph_semantic.items()
        if key not in {"goals", "tasks"}
    }
    goal_records = tuple(_semantic(goal.to_record()) for goal in graph.goals)
    assignments: list[list[Mapping[str, Any]]] = [
        [] for _item in ordered_tasks
    ]
    for index, record in enumerate(goal_records):
        assignments[index % len(assignments)].append(record)
    goals = {item.goal_cid: item for item in graph.goals}
    entries = []
    from .taskboard_store import TaskboardMaterializationEntry

    for index, task in enumerate(ordered_tasks):
        goal = goals[task.goal_cid]
        entries.append(
            TaskboardMaterializationEntry(
                task_id=task_aliases[task.task_cid],
                goal_id=goal_aliases[goal.goal_cid],
                rendered_block=_render_task_block(
                    task=task,
                    goal=goal,
                    task_alias=task_aliases[task.task_cid],
                    goal_alias=goal_aliases[goal.goal_cid],
                    dependency_aliases=tuple(
                        task_aliases[item]
                        for item in task.dependency_task_cids
                    ),
                    plan_root=projection_root,
                    candidate_plan_root=snapshot.plan_root,
                    projection_id=projection_id,
                    revision=1,
                    board_namespace=snapshot.board_namespace,
                    task_population_cids=snapshot.task_cids,
                    goal_population_cids=snapshot.goal_cids,
                    graph_core=graph_core,
                    assigned_goal_records=assignments[index],
                ),
            )
        )
    return MarkdownTaskProjection(
        plan_root=projection_root,
        projection_id=projection_id,
        entries=tuple(entries),
        task_cids=snapshot.task_cids,
        task_aliases=task_aliases,
        goal_cids=snapshot.goal_cids,
        goal_aliases=goal_aliases,
        board_namespace=snapshot.board_namespace,
        revision=1,
        schema=MARKDOWN_TASK_SOURCE_SCHEMA,
    )


def _replay_canonical_events(
    source: CanonicalTaskSource,
    snapshot: CanonicalProjectionSnapshot,
) -> None:
    for event in snapshot.events:
        event_type = str(event.get("event_type") or "")
        task_cid = str(event.get("task_cid") or "")
        if event_type == "status_changed":
            task = source.get(task_cid)
            if task is None:
                raise TaskSourceIntegrityError(
                    "migration event references an unknown task"
                )
            if _semantic_status(task.status) != str(event["previous_status"]):
                raise TaskSourceIntegrityError(
                    "migration event history does not match target status"
                )
            source.compare_and_swap_status(
                task_cid,
                expected_status=task.status,
                new_status=str(event["status"]),
                expected_revision=task.revision,
                receipt={
                    **dict(event.get("receipt") or {}),
                    "dual_event": dict(event),
                },
            )
        else:
            source.append_event(
                event_type,
                {
                    **dict(event.get("payload") or {}),
                    "task_cid": task_cid,
                    "event_cid": _operation_id(dict(event)),
                    "dual_event": dict(event),
                },
            )


def rebuild_task_source_projection(
    snapshot: CanonicalProjectionSnapshot | Mapping[str, Any],
    target: TaskSource | Any,
    *,
    kind: str = "",
    repository_root_id: str = "",
    fault_injector: Any | None = None,
) -> TaskSourceMigrationResult:
    """Rebuild one projection solely from a verified canonical snapshot.

    An existing disagreeing/corrupt target is moved aside, never overwritten.
    The phase journal makes an interrupted install deterministic to resume.
    """

    verified = (
        CanonicalProjectionSnapshot.from_dict(snapshot)
        if isinstance(snapshot, Mapping)
        else snapshot
    )
    if not isinstance(verified, CanonicalProjectionSnapshot):
        raise TypeError("snapshot must be a CanonicalProjectionSnapshot")
    # Reconstructing through the strict constructor verifies the digest even
    # when a caller retained and mutated an underlying mapping.
    verified = CanonicalProjectionSnapshot.from_dict(verified.to_dict())

    selected_kind = str(kind or "").strip().lower()
    native_target: Any | None = None
    if isinstance(target, CanonicalTaskSource):
        selected_kind = selected_kind or target.source_kind
        path = target.path
        native_target = target.backend
    elif isinstance(target, (str, Path)):
        path = Path(target).absolute()
        selected_kind = selected_kind or (
            "duckdb" if path.suffix.lower() in {".duckdb", ".ddb"} else "markdown"
        )
    else:
        native_target = target
        path_value = getattr(target, "path", None) or getattr(
            target, "database_path", None
        )
        if path_value is None:
            raise UnsupportedTaskSourceError("migration target has no path")
        path = Path(path_value).absolute()
        module = type(target).__module__
        selected_kind = selected_kind or (
            "markdown" if module.endswith("markdown_task_source") else "duckdb"
        )
    if selected_kind not in {"markdown", "duckdb"}:
        raise UnsupportedTaskSourceError("migration target kind is unsupported")
    journal_path = path.with_name(f".{path.name}.projection-migration.json")
    lock_path = journal_path.with_name(f".{journal_path.name}.lock")
    migration_id = _operation_id(
        {
            "schema": TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA,
            "snapshot_id": verified.snapshot_id,
            "target_kind": selected_kind,
            "target_path": str(path),
        }
    )

    def fault(point: str) -> None:
        if callable(fault_injector):
            fault_injector(point)

    def write_state(value: Mapping[str, Any]) -> None:
        payload = dict(value)
        _atomic_json_write(
            journal_path,
            {
                "payload": payload,
                "digest": hashlib.sha256(
                    b"task-source-projection-migration-v1\0"
                    + _canonical_bytes(payload)
                ).hexdigest(),
            },
        )

    def read_state() -> dict[str, Any]:
        try:
            envelope = json.loads(journal_path.read_bytes())
            payload = envelope["payload"]
            digest = str(envelope["digest"])
            expected = hashlib.sha256(
                b"task-source-projection-migration-v1\0"
                + _canonical_bytes(payload)
            ).hexdigest()
        except Exception as exc:
            raise TaskSourceQuarantinedError(
                "projection migration journal is corrupt"
            ) from exc
        if digest != expected or not isinstance(payload, Mapping):
            raise TaskSourceQuarantinedError(
                "projection migration journal digest does not match"
            )
        return dict(payload)

    from .duckdb_state import exclusive_file_lock

    with exclusive_file_lock(lock_path, timeout_seconds=30.0):
        resumed = journal_path.exists()
        if resumed:
            state = read_state()
            if (
                state.get("schema") != TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA
                or state.get("migration_id") != migration_id
                or state.get("snapshot_id") != verified.snapshot_id
            ):
                raise TaskSourceQuarantinedError(
                    "projection migration journal is stale or foreign"
                )
        else:
            state = {
                "schema": TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA,
                "migration_id": migration_id,
                "snapshot_id": verified.snapshot_id,
                "target_kind": selected_kind,
                "target_path": str(path),
                "phase": "snapshot_verified",
                "quarantine_path": "",
                "quarantine_generation": 0,
            }
            write_state(state)
        fault("after_snapshot_verified")

        if state["phase"] == "verified":
            try:
                rebuilt = open_task_source(
                    native_target or path, kind=selected_kind
                )
                target_snapshot = canonical_projection_snapshot(rebuilt)
                parity = compare_task_source_projections(
                    verified, target_snapshot
                )
                parity.require_valid()
            except Exception:
                # A projection can be corrupted after a completed migration.
                # Re-enter the verified-snapshot phase so the exact current
                # bytes and sidecars are quarantined before reconstruction.
                state["phase"] = "snapshot_verified"
                state.pop("target_snapshot_id", None)
                state["quarantine_path"] = ""
                state["quarantine_generation"] = (
                    int(state.get("quarantine_generation") or 0) + 1
                )
                write_state(state)
            else:
                return TaskSourceMigrationResult(
                    source_snapshot_id=verified.snapshot_id,
                    target_snapshot_id=target_snapshot.snapshot_id,
                    target_kind=selected_kind,
                    target_path=path,
                    changed=False,
                    replayed=True,
                    resumed=True,
                    parity=parity,
                    receipt_id=migration_id,
                    quarantine_path=(
                        Path(state["quarantine_path"])
                        if state.get("quarantine_path")
                        else None
                    ),
                )

        quarantine_path: Path | None = (
            Path(state["quarantine_path"])
            if state.get("quarantine_path")
            else None
        )
        if state["phase"] == "snapshot_verified" and path.exists():
            try:
                existing = open_task_source(
                    native_target or path, kind=selected_kind
                )
                existing_snapshot = canonical_projection_snapshot(existing)
                parity = compare_task_source_projections(verified, existing_snapshot)
                if parity.valid:
                    state["phase"] = "verified"
                    write_state(state)
                    return TaskSourceMigrationResult(
                        source_snapshot_id=verified.snapshot_id,
                        target_snapshot_id=existing_snapshot.snapshot_id,
                        target_kind=selected_kind,
                        target_path=path,
                        changed=False,
                        replayed=True,
                        resumed=resumed,
                        parity=parity,
                        receipt_id=migration_id,
                    )
            except Exception:
                pass
            quarantine_generation = int(
                state.get("quarantine_generation") or 0
            )
            generation_suffix = (
                "" if quarantine_generation == 0 else f".{quarantine_generation}"
            )
            quarantine_path = path.with_name(
                f"{path.name}.quarantine."
                f"{verified.snapshot_id.rsplit(':', 1)[-1][:16]}"
                f"{generation_suffix}"
            )
            if quarantine_path.exists():
                raise TaskSourceQuarantinedError(
                    "projection quarantine destination already exists"
                )
            associated_paths = [path]
            if selected_kind == "markdown":
                from .markdown_task_source import MarkdownTaskSource

                markdown_target = (
                    native_target
                    if type(native_target).__name__ == "MarkdownTaskSource"
                    else MarkdownTaskSource(path, root=path.parent)
                )
                associated_paths.extend(
                    (
                        Path(markdown_target.events_path),
                        Path(markdown_target.journal_path),
                    )
                )
            quarantine_suffix = (
                verified.snapshot_id.rsplit(":", 1)[-1][:16]
            )
            quarantined_artifacts: list[str] = []
            # Sidecars move first so a newly materialized Markdown board can
            # never inherit a foreign event stream or recovery journal.
            for artifact in (*associated_paths[1:], associated_paths[0]):
                if not artifact.exists():
                    continue
                destination = (
                    quarantine_path
                    if artifact == path
                    else artifact.with_name(
                        f"{artifact.name}.quarantine.{quarantine_suffix}"
                        f"{generation_suffix}"
                    )
                )
                if destination.exists():
                    raise TaskSourceQuarantinedError(
                        "projection sidecar quarantine destination already exists"
                    )
                os.replace(artifact, destination)
                quarantined_artifacts.append(str(destination))
            state.update(
                {
                    "phase": "target_quarantined",
                    "quarantine_path": str(quarantine_path),
                    "quarantined_artifacts": quarantined_artifacts,
                }
            )
            write_state(state)
        fault("after_target_quarantined")

        if state["phase"] in {"snapshot_verified", "target_quarantined"}:
            if selected_kind == "duckdb":
                from .duckdb_task_source import DuckDBTaskSource
                from .prompt_workflow import PromptGoalGraph

                if not verified.graph_record:
                    raise TaskSourceIntegrityError(
                        "canonical graph payload is required to rebuild DuckDB"
                    )
                graph = PromptGoalGraph.from_dict(
                    _graph_with_lifecycle(verified.graph_record)
                )
                tree_id = (
                    repository_root_id
                    or verified.repository_root_id
                    or str(graph.program_root)
                )
                backend = DuckDBTaskSource(path)
                backend.materialize(
                    graph,
                    repository_tree_id=tree_id,
                    plan_root_cid=verified.plan_root,
                    receipt={
                        "migration_id": migration_id,
                        "canonical_snapshot_id": verified.snapshot_id,
                        "canonical_graph": dict(verified.graph_record),
                        "admitted_plan_root": verified.admitted_plan_root,
                        "board_namespace": verified.board_namespace,
                    },
                )
            else:
                from .markdown_task_source import MarkdownTaskSource

                backend = MarkdownTaskSource(
                    path,
                    root=path.parent,
                    board_namespace=verified.board_namespace,
                )
                backend.materialize(
                    _markdown_projection_from_snapshot(verified),
                    epoch_id=migration_id,
                )
            native_target = backend
            state["phase"] = "installed"
            write_state(state)
        fault("after_install")

        rebuilt = open_task_source(native_target or path, kind=selected_kind)
        if state["phase"] == "installed":
            _replay_canonical_events(rebuilt, verified)
            state["phase"] = "events_replayed"
            write_state(state)
        fault("after_events_replayed")

        target_snapshot = canonical_projection_snapshot(rebuilt)
        parity = compare_task_source_projections(verified, target_snapshot)
        if not parity.valid:
            state.update(
                {
                    "phase": "quarantined",
                    "parity_mismatches": list(parity.mismatches),
                }
            )
            write_state(state)
            raise TaskSourceQuarantinedError(
                "rebuilt projection failed canonical parity: "
                + ", ".join(parity.mismatches)
            )
        state["phase"] = "verified"
        state["target_snapshot_id"] = target_snapshot.snapshot_id
        write_state(state)
        return TaskSourceMigrationResult(
            source_snapshot_id=verified.snapshot_id,
            target_snapshot_id=target_snapshot.snapshot_id,
            target_kind=selected_kind,
            target_path=path,
            changed=True,
            replayed=False,
            resumed=resumed,
            parity=parity,
            receipt_id=migration_id,
            quarantine_path=quarantine_path,
        )


VerifiedCanonicalTaskSourceSnapshot = CanonicalProjectionSnapshot
TaskSourceProjectionSnapshot = CanonicalProjectionSnapshot
compare_task_sources = compare_task_source_projections
migrate_task_source_projection = rebuild_task_source_projection


def open_task_source(
    source: Any,
    *,
    kind: str = "",
    root: Path | str | None = None,
    expected_identity: TaskSourceIdentity | Mapping[str, Any] | None = None,
    expected_root_id: str = "",
    expected_repository_root_id: str = "",
    **backend_options: Any,
) -> CanonicalTaskSource | DualTaskSource:
    """Open a native source or wrap an already-configured backend.

    Path inputs infer ``duckdb`` only from a ``.duckdb``/``.ddb`` suffix;
    every other path remains the backward-compatible Markdown default.
    """

    if isinstance(source, DualTaskSource):
        if kind and kind != "dual":
            raise UnsupportedTaskSourceError(
                "configured source kind disagrees with the open dual source"
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
                    "open dual source does not match expected identity"
                )
        if expected_root_id and source.identity.root_id != expected_root_id:
            raise TaskSourceIntegrityError("task source has a foreign plan root")
        return source

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
    "CANONICAL_PROJECTION_SNAPSHOT_SCHEMA",
    "CanonicalTaskSource",
    "CanonicalProjectionSnapshot",
    "DEFAULT_QUERY_LIMIT",
    "DUAL_TASK_SOURCE_SCHEMA",
    "DUAL_TASK_SOURCE_TRANSACTION_SCHEMA",
    "DualTaskSource",
    "DualTaskSourcePartialError",
    "MAX_QUERY_LIMIT",
    "MAX_SNAPSHOT_TASKS",
    "SUPPORTED_SOURCE_KINDS",
    "TASK_SOURCE_IDENTITY_SCHEMA",
    "TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA",
    "TASK_SOURCE_PARITY_REPORT_SCHEMA",
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
    "TaskSourceMigrationResult",
    "TaskSourcePage",
    "TaskSourceParityReport",
    "TaskSourceProjectionSnapshot",
    "TaskSourceQuarantinedError",
    "TaskSourceSnapshot",
    "TaskSourceTask",
    "TaskSourceWatchResult",
    "UnsupportedTaskSourceError",
    "VerifiedCanonicalTaskSourceSnapshot",
    "adapt_task_source",
    "canonical_projection_snapshot",
    "compare_task_source_projections",
    "compare_task_sources",
    "migrate_task_source_projection",
    "open_task_source",
    "rebuild_task_source_projection",
]
