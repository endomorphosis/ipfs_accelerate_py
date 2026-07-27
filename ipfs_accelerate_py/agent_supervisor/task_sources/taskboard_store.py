"""Durable taskboard updates and event-driven runtime wakeup primitives.

The watcher in this module is deliberately only a *hint* source.  Correctness
comes from comparing bounded, canonical metadata snapshots after every native
notification and after a low-frequency safety timeout.  This keeps duplicate
or coalesced filesystem notifications harmless and preserves liveness on
filesystems which do not implement native change notification.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import select
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, TextIO

from ..control_contracts import EventCursor


PATH_METADATA_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/path-metadata@1"
)
PROJECTION_DELTA_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/projection-delta-checkpoint@1"
)
TASKBOARD_MATERIALIZATION_JOURNAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-materialization-journal@1"
)
TASKBOARD_MATERIALIZATION_TRANSACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-materialization-transaction@1"
)
TASKBOARD_MATERIALIZATION_PREVIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-materialization-preview@1"
)
EVENT_DRIVEN_RUNTIME_REQUIREMENT_ID: Final = (
    "asi-117:event-driven-delta-checkpoint-runtime"
)
DEFAULT_METADATA_ENTRY_LIMIT: Final = 256
DEFAULT_METADATA_DEPTH_LIMIT: Final = 6
DEFAULT_SAFETY_INTERVAL_SECONDS: Final = 300.0
MAX_TASKBOARD_MATERIALIZATION_ENTRIES: Final = 24
_MAX_TASKBOARD_IDENTIFIER_BYTES: Final = 512
_MAX_TASKBOARD_ENTRY_BYTES: Final = 128 * 1024
_MAX_CHECKPOINT_BYTES: Final = 1024 * 1024
_MAX_TASKBOARD_JOURNAL_BYTES: Final = 4 * 1024 * 1024
TASKBOARD_STORE_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-store-snapshot@1"
)
TASKBOARD_STORE_EVENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/taskboard-store-event@1"
)
DEFAULT_TASKBOARD_QUERY_LIMIT: Final = 256
MAX_TASKBOARD_QUERY_LIMIT: Final = 1024
DEFAULT_TASKBOARD_BYTES_LIMIT: Final = 4 * 1024 * 1024
TASKBOARD_STATUSES: Final = frozenset(
    {
        "todo",
        "queued",
        "proposed",
        "admitted",
        "ready",
        "in_progress",
        "running",
        "blocked",
        "completed",
        "failed",
        "rejected",
        "quarantined",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("runtime metadata must contain canonical JSON values") from exc


def _content_id(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:{hashlib.sha256(_canonical_json_bytes(value)).hexdigest()}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def locked_taskboard(path: Path) -> Iterator[TextIO]:
    """Lock a taskboard's inode while a scanner performs read-modify-write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def replace_locked_taskboard(stream: TextIO, text: str) -> bool:
    """Replace a locked taskboard only when its exact text changed.

    ``False`` means the current bytes already matched.  That path deliberately
    performs neither a write nor an fsync, which makes idempotent refill and
    drained-board passes observable as true no-ops.
    """

    stream.seek(0)
    if stream.read() == text:
        return False
    stream.seek(0)
    stream.truncate()
    stream.write(text)
    stream.flush()
    os.fsync(stream.fileno())
    return True


def taskboard_revision(text: str | bytes) -> str:
    """Return an exact, content-addressed revision for taskboard bytes."""

    if isinstance(text, str):
        payload = text.encode("utf-8")
    elif isinstance(text, bytes):
        payload = text
    else:
        raise TypeError("taskboard content must be text or bytes")
    digest = hashlib.sha256()
    digest.update(b"agent-supervisor-taskboard-revision-v1\0")
    digest.update(payload)
    return f"taskboard:sha256:{digest.hexdigest()}"


def _taskboard_identifier(value: Any, *, name: str) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError(f"{name} must not be empty")
    if "\n" in selected or "\r" in selected or "\x00" in selected:
        raise ValueError(f"{name} must be a single-line identifier")
    if len(selected.encode("utf-8")) > _MAX_TASKBOARD_IDENTIFIER_BYTES:
        raise ValueError(f"{name} exceeds its persistence bound")
    return selected


def _task_heading_pattern(task_id: str) -> re.Pattern[str]:
    return re.compile(
        rf"^##[ \t]+{re.escape(task_id)}(?=[ \t]|$)",
        flags=re.MULTILINE,
    )


def _goal_metadata_pattern(goal_id: str) -> re.Pattern[str]:
    return re.compile(
        rf"^[ \t]*-[ \t]*Goal[ \t]+id:[ \t]*{re.escape(goal_id)}[ \t]*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )


@dataclass(frozen=True)
class TaskboardMaterializationEntry:
    """One exact task block and its unique admitted-goal ownership mapping."""

    task_id: str
    goal_id: str
    rendered_block: str

    def __post_init__(self) -> None:
        task_id = _taskboard_identifier(self.task_id, name="task_id")
        goal_id = _taskboard_identifier(self.goal_id, name="goal_id")
        if not isinstance(self.rendered_block, str):
            raise TypeError("rendered_block must be text")
        rendered_block = self.rendered_block.strip()
        if not rendered_block:
            raise ValueError("rendered_block must not be empty")
        if len(rendered_block.encode("utf-8")) > _MAX_TASKBOARD_ENTRY_BYTES:
            raise ValueError("rendered_block exceeds its persistence bound")
        if len(_task_heading_pattern(task_id).findall(rendered_block)) != 1:
            raise ValueError(
                "rendered_block must contain exactly one matching task heading"
            )
        if len(_goal_metadata_pattern(goal_id).findall(rendered_block)) != 1:
            raise ValueError(
                "rendered_block must contain exactly one matching goal metadata line"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "rendered_block", rendered_block)

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "rendered_block": self.rendered_block,
        }


def _append_taskboard_entries(
    board_text: str,
    entries: Iterable[TaskboardMaterializationEntry],
) -> str:
    result = board_text
    for entry in entries:
        prefix = result.rstrip()
        separator = "\n\n" if prefix else ""
        result = prefix + separator + entry.rendered_block + "\n"
    return result


@dataclass(frozen=True)
class TaskboardMaterializationPreview:
    """Immutable exact taskboard delta prepared without filesystem writes."""

    base_text: str
    candidate_text: str
    entries: tuple[TaskboardMaterializationEntry, ...]
    base_board_revision: str
    candidate_board_revision: str
    preview_id: str
    schema: str = TASKBOARD_MATERIALIZATION_PREVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TASKBOARD_MATERIALIZATION_PREVIEW_SCHEMA:
            raise ValueError("unsupported taskboard materialization preview schema")
        if not isinstance(self.base_text, str) or not isinstance(
            self.candidate_text, str
        ):
            raise TypeError("taskboard preview content must be text")
        entries = tuple(self.entries)
        if len(entries) > MAX_TASKBOARD_MATERIALIZATION_ENTRIES:
            raise ValueError(
                "taskboard materialization exceeds the 24-task epoch limit"
            )
        if any(not isinstance(item, TaskboardMaterializationEntry) for item in entries):
            raise TypeError(
                "taskboard preview entries must be TaskboardMaterializationEntry values"
            )
        task_ids = [item.task_id for item in entries]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("taskboard materialization task IDs must be unique")
        expected_base_revision = taskboard_revision(self.base_text)
        if self.base_board_revision != expected_base_revision:
            raise ValueError("taskboard preview base revision does not match")
        expected_candidate = _append_taskboard_entries(self.base_text, entries)
        if self.candidate_text != expected_candidate:
            raise ValueError("taskboard preview candidate is not the exact entry delta")
        expected_candidate_revision = taskboard_revision(self.candidate_text)
        if self.candidate_board_revision != expected_candidate_revision:
            raise ValueError("taskboard preview candidate revision does not match")
        entries_by_goal: dict[str, list[TaskboardMaterializationEntry]] = {}
        for entry in entries:
            entries_by_goal.setdefault(entry.goal_id, []).append(entry)
            if _task_heading_pattern(entry.task_id).search(self.base_text):
                raise ValueError(
                    f"taskboard already contains task ID {entry.task_id}"
                )
            if _goal_metadata_pattern(entry.goal_id).search(self.base_text):
                raise ValueError(
                    f"taskboard already maps admitted goal {entry.goal_id}"
                )
            if (
                len(
                    _task_heading_pattern(entry.task_id).findall(
                        self.candidate_text
                    )
                )
                != 1
            ):
                raise ValueError(
                    f"candidate taskboard must map task {entry.task_id} exactly once"
                )
        for goal_id, owned_entries in entries_by_goal.items():
            if len(
                _goal_metadata_pattern(goal_id).findall(self.candidate_text)
            ) != len(owned_entries):
                raise ValueError(
                    "candidate taskboard must map only the declared tasks for "
                    f"goal {goal_id}"
                )
        expected_preview_id = _content_id(
            "taskboard-materialization-preview",
            self._identity_payload(),
        )
        if self.preview_id != expected_preview_id:
            raise ValueError(
                "taskboard materialization preview identity does not match"
            )
        object.__setattr__(self, "entries", entries)

    @property
    def changed(self) -> bool:
        return self.base_board_revision != self.candidate_board_revision

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self.entries)

    @property
    def goal_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.goal_id for item in self.entries))

    @property
    def goal_task_mappings(self) -> dict[str, tuple[str, ...]]:
        mappings: dict[str, list[str]] = {}
        for item in self.entries:
            mappings.setdefault(item.goal_id, []).append(item.task_id)
        return {
            goal_id: tuple(task_ids)
            for goal_id, task_ids in mappings.items()
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_board_revision": self.base_board_revision,
            "candidate_board_revision": self.candidate_board_revision,
            "entries": [item.to_dict() for item in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "preview_id": self.preview_id,
            "changed": self.changed,
            "goal_task_mappings": self.goal_task_mappings,
        }


def preview_taskboard_materialization(
    board_text: str,
    entries: Iterable[TaskboardMaterializationEntry | Mapping[str, Any]],
    expected_board_revision: str = "",
) -> TaskboardMaterializationPreview:
    """Preview one exact bounded task delta without reading or writing files."""

    if not isinstance(board_text, str):
        raise TypeError("board_text must be text")
    normalized_items: list[TaskboardMaterializationEntry] = []
    for item in entries:
        if isinstance(item, TaskboardMaterializationEntry):
            normalized_items.append(item)
        elif isinstance(item, Mapping):
            normalized_items.append(
                TaskboardMaterializationEntry(
                    task_id=item.get("task_id", ""),
                    goal_id=item.get("goal_id", ""),
                    rendered_block=item.get("rendered_block", ""),
                )
            )
        else:
            raise TypeError(
                "entries must contain taskboard materialization entries"
            )
    normalized = tuple(normalized_items)
    if len(normalized) > MAX_TASKBOARD_MATERIALIZATION_ENTRIES:
        raise ValueError("taskboard materialization exceeds the 24-task epoch limit")
    base_revision = taskboard_revision(board_text)
    if expected_board_revision and expected_board_revision != base_revision:
        raise ValueError("expected taskboard revision does not match preview input")
    candidate_text = _append_taskboard_entries(board_text, normalized)
    candidate_revision = taskboard_revision(candidate_text)
    identity_payload = {
        "schema": TASKBOARD_MATERIALIZATION_PREVIEW_SCHEMA,
        "base_board_revision": base_revision,
        "candidate_board_revision": candidate_revision,
        "entries": [item.to_dict() for item in normalized],
    }
    return TaskboardMaterializationPreview(
        base_text=board_text,
        candidate_text=candidate_text,
        entries=normalized,
        base_board_revision=base_revision,
        candidate_board_revision=candidate_revision,
        preview_id=_content_id(
            "taskboard-materialization-preview", identity_payload
        ),
    )


class TaskboardMaterializationTransactionState(str, Enum):
    """Durable state of one taskboard compare-and-swap transaction."""

    PREPARED = "prepared"
    COMMITTED = "committed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TaskboardMaterializationTransactionResult:
    """Outcome of one journaled taskboard materialization transaction."""

    taskboard_path: Path
    journal_path: Path
    transaction_id: str
    state: TaskboardMaterializationTransactionState
    epoch_id: str = ""
    goal_task_mappings: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    base_board_revision: str = ""
    candidate_board_revision: str = ""
    changed: bool = False
    resumed: bool = False
    reason_codes: tuple[str, ...] = ()
    board_write_count: int = 0
    journal_write_count: int = 0

    @property
    def committed(self) -> bool:
        return self.state is TaskboardMaterializationTransactionState.COMMITTED

    @property
    def write_count(self) -> int:
        return self.board_write_count + self.journal_write_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "taskboard-materialization-transaction-result@1"
            ),
            "taskboard_path": str(self.taskboard_path),
            "journal_path": str(self.journal_path),
            "transaction_id": self.transaction_id,
            "state": self.state.value,
            "epoch_id": self.epoch_id,
            "goal_task_mappings": {
                goal_id: list(task_ids)
                for goal_id, task_ids in self.goal_task_mappings.items()
            },
            "base_board_revision": self.base_board_revision,
            "candidate_board_revision": self.candidate_board_revision,
            "changed": self.changed,
            "resumed": self.resumed,
            "reason_codes": list(self.reason_codes),
            "board_write_count": self.board_write_count,
            "journal_write_count": self.journal_write_count,
            "write_count": self.write_count,
            "committed": self.committed,
        }


def _load_taskboard_materialization_journal(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {
            "schema": TASKBOARD_MATERIALIZATION_JOURNAL_SCHEMA,
            "transactions": {},
            "latest_transaction_id": "",
        }
    except OSError as exc:
        raise ValueError(
            f"cannot read taskboard materialization journal: {exc}"
        ) from exc
    if len(raw) > _MAX_TASKBOARD_JOURNAL_BYTES:
        raise ValueError("taskboard materialization journal exceeds persistence bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("taskboard materialization journal is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError("taskboard materialization journal must be an object")
    if value.get("schema") != TASKBOARD_MATERIALIZATION_JOURNAL_SCHEMA:
        raise ValueError("unsupported taskboard materialization journal schema")
    transactions = value.get("transactions")
    if not isinstance(transactions, dict):
        raise ValueError("taskboard materialization transactions must be an object")
    return value


def _taskboard_transaction_id(
    preview: TaskboardMaterializationPreview,
    epoch_id: str,
) -> str:
    return _content_id(
        "taskboard-materialization-transaction",
        {
            "schema": TASKBOARD_MATERIALIZATION_TRANSACTION_SCHEMA,
            "epoch_id": epoch_id,
            "preview_id": preview.preview_id,
        },
    )


def _taskboard_transaction_matches(
    record: Mapping[str, Any],
    preview: TaskboardMaterializationPreview,
    epoch_id: str,
) -> bool:
    raw_mappings = record.get("goal_task_mappings")
    if not isinstance(raw_mappings, Mapping):
        return False
    try:
        stored_mappings = {
            str(goal_id): tuple(str(task_id) for task_id in task_ids)
            for goal_id, task_ids in raw_mappings.items()
            if isinstance(task_ids, list)
        }
    except (TypeError, ValueError):
        return False
    return (
        record.get("schema") == TASKBOARD_MATERIALIZATION_TRANSACTION_SCHEMA
        and str(record.get("epoch_id") or "") == epoch_id
        and str(record.get("preview_id") or "") == preview.preview_id
        and str(record.get("base_board_revision") or "")
        == preview.base_board_revision
        and str(record.get("candidate_board_revision") or "")
        == preview.candidate_board_revision
        and stored_mappings == preview.goal_task_mappings
    )


def _has_taskboard_materialization_transaction(
    journal_path: Path,
    preview: TaskboardMaterializationPreview,
    epoch_id: str,
) -> bool:
    """Return whether the journal contains this exact transaction.

    A complete board is an identical replay even if its journal was removed.
    Recovery is therefore attempted only when the existing journal proves
    that this exact preview was interrupted.  Malformed or conflicting
    records fail closed instead of being mistaken for unrelated history.
    """

    journal = _load_taskboard_materialization_journal(journal_path)
    transaction_id = _taskboard_transaction_id(preview, epoch_id)
    record = journal["transactions"].get(transaction_id)
    if record is None:
        return False
    if not isinstance(record, Mapping) or not _taskboard_transaction_matches(
        record, preview, epoch_id
    ):
        raise ValueError("taskboard materialization transaction identity conflict")
    return True


def _taskboard_prefix_count(
    current_revision: str,
    preview: TaskboardMaterializationPreview,
) -> int:
    for count in range(0, len(preview.entries) + 1):
        candidate = _append_taskboard_entries(
            preview.base_text, preview.entries[:count]
        )
        if taskboard_revision(candidate) == current_revision:
            return count
    return -1


def _taskboard_transaction_result(
    *,
    taskboard_path: Path,
    journal_path: Path,
    preview: TaskboardMaterializationPreview,
    transaction_id: str,
    state: TaskboardMaterializationTransactionState,
    epoch_id: str,
    changed: bool = False,
    resumed: bool = False,
    reason_codes: Iterable[str] = (),
    board_write_count: int = 0,
    journal_write_count: int = 0,
) -> TaskboardMaterializationTransactionResult:
    return TaskboardMaterializationTransactionResult(
        taskboard_path=taskboard_path,
        journal_path=journal_path,
        transaction_id=transaction_id,
        state=state,
        epoch_id=epoch_id,
        goal_task_mappings=preview.goal_task_mappings,
        base_board_revision=preview.base_board_revision,
        candidate_board_revision=preview.candidate_board_revision,
        changed=changed,
        resumed=resumed,
        reason_codes=tuple(dict.fromkeys(str(item) for item in reason_codes if item)),
        board_write_count=board_write_count,
        journal_write_count=journal_write_count,
    )


def commit_taskboard_materialization(
    taskboard_path: Path | str,
    journal_path: Path | str,
    preview: TaskboardMaterializationPreview,
    epoch_id: str = "",
    expected_board_revision: str = "",
) -> TaskboardMaterializationTransactionResult:
    """Journal and CAS-commit an exact task delta with crash-safe replay.

    A sidecar lock fences atomic board replacement (locking the board inode
    itself would be unsafe across ``os.replace``).  The durable journal is
    published in ``prepared`` state before the board write and in ``committed``
    state only after the exact candidate revision is observed.
    """

    if not isinstance(preview, TaskboardMaterializationPreview):
        raise TypeError("preview must be a TaskboardMaterializationPreview")
    board = Path(taskboard_path).resolve()
    journal_file = Path(journal_path).resolve()
    if board == journal_file:
        raise ValueError("journal_path must be separate from taskboard_path")
    epoch = str(epoch_id).strip()
    expected = str(expected_board_revision).strip()
    if (
        "\x00" in epoch
        or "\x00" in expected
        or len(epoch.encode("utf-8")) > _MAX_TASKBOARD_IDENTIFIER_BYTES
        or len(expected.encode("utf-8")) > _MAX_TASKBOARD_IDENTIFIER_BYTES
    ):
        raise ValueError("taskboard transaction fence exceeds its safe bound")
    transaction_id = _taskboard_transaction_id(preview, epoch)
    if expected and expected != preview.base_board_revision:
        return _taskboard_transaction_result(
            taskboard_path=board,
            journal_path=journal_file,
            preview=preview,
            transaction_id=transaction_id,
            state=TaskboardMaterializationTransactionState.BLOCKED,
            epoch_id=epoch,
            reason_codes=("expected_board_revision_conflict",),
        )
    if not preview.entries or not preview.changed:
        return _taskboard_transaction_result(
            taskboard_path=board,
            journal_path=journal_file,
            preview=preview,
            transaction_id=transaction_id,
            state=TaskboardMaterializationTransactionState.BLOCKED,
            epoch_id=epoch,
            reason_codes=("preview_not_changed",),
        )

    lock_path = board.with_name(f".{board.name}.materialization.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            journal = _load_taskboard_materialization_journal(journal_file)
            transactions = dict(journal["transactions"])
            prior = transactions.get(transaction_id)
            if prior is not None and not isinstance(prior, Mapping):
                return _taskboard_transaction_result(
                    taskboard_path=board,
                    journal_path=journal_file,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=TaskboardMaterializationTransactionState.BLOCKED,
                    epoch_id=epoch,
                    resumed=True,
                    reason_codes=("transaction_record_malformed",),
                )
            resumed = prior is not None
            if prior is not None and not _taskboard_transaction_matches(
                prior, preview, epoch
            ):
                return _taskboard_transaction_result(
                    taskboard_path=board,
                    journal_path=journal_file,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=TaskboardMaterializationTransactionState.BLOCKED,
                    epoch_id=epoch,
                    resumed=True,
                    reason_codes=("transaction_identity_conflict",),
                )

            try:
                current_bytes = board.read_bytes()
            except FileNotFoundError:
                current_bytes = b""
            current_revision = taskboard_revision(current_bytes)
            prefix_count = _taskboard_prefix_count(current_revision, preview)
            if prefix_count < 0:
                return _taskboard_transaction_result(
                    taskboard_path=board,
                    journal_path=journal_file,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=TaskboardMaterializationTransactionState.BLOCKED,
                    epoch_id=epoch,
                    resumed=resumed,
                    reason_codes=("stale_taskboard_revision",),
                )

            if (
                prior is not None
                and str(prior.get("state") or "") == "committed"
                and prefix_count == len(preview.entries)
            ):
                return _taskboard_transaction_result(
                    taskboard_path=board,
                    journal_path=journal_file,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=TaskboardMaterializationTransactionState.COMMITTED,
                    epoch_id=epoch,
                    resumed=True,
                )

            journal_write_count = 0
            if prior is None or str(prior.get("state") or "") != "prepared":
                prepared = {
                    "schema": TASKBOARD_MATERIALIZATION_TRANSACTION_SCHEMA,
                    "transaction_id": transaction_id,
                    "state": "prepared",
                    "epoch_id": epoch,
                    "preview_id": preview.preview_id,
                    "base_board_revision": preview.base_board_revision,
                    "candidate_board_revision": preview.candidate_board_revision,
                    "goal_task_mappings": {
                        goal_id: list(task_ids)
                        for goal_id, task_ids in preview.goal_task_mappings.items()
                    },
                    "task_ids": list(preview.task_ids),
                    "goal_ids": list(preview.goal_ids),
                    "rendered_block_digests": {
                        item.task_id: hashlib.sha256(
                            item.rendered_block.encode("utf-8")
                        ).hexdigest()
                        for item in preview.entries
                    },
                    "prepared_at_ns": time.time_ns(),
                }
                transactions[transaction_id] = prepared
                journal.update(
                    {
                        "transactions": transactions,
                        "latest_transaction_id": transaction_id,
                    }
                )
                encoded = _canonical_json_bytes(journal) + b"\n"
                if len(encoded) > _MAX_TASKBOARD_JOURNAL_BYTES:
                    raise ValueError(
                        "taskboard materialization journal exceeds persistence bound"
                    )
                _atomic_write(journal_file, encoded)
                journal_write_count += 1
            else:
                prepared = dict(prior)

            board_write_count = 0
            if prefix_count < len(preview.entries):
                _atomic_write(board, preview.candidate_text.encode("utf-8"))
                board_write_count = 1
            try:
                persisted = board.read_bytes()
            except FileNotFoundError:
                persisted = b""
            if taskboard_revision(persisted) != preview.candidate_board_revision:
                return _taskboard_transaction_result(
                    taskboard_path=board,
                    journal_path=journal_file,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=TaskboardMaterializationTransactionState.PREPARED,
                    epoch_id=epoch,
                    changed=bool(board_write_count),
                    resumed=resumed or prefix_count > 0,
                    reason_codes=("partial_taskboard_write",),
                    board_write_count=board_write_count,
                    journal_write_count=journal_write_count,
                )

            prepared.update(
                {
                    "state": "committed",
                    "committed_at_ns": time.time_ns(),
                }
            )
            transactions[transaction_id] = prepared
            journal.update(
                {
                    "transactions": transactions,
                    "latest_transaction_id": transaction_id,
                }
            )
            encoded = _canonical_json_bytes(journal) + b"\n"
            if len(encoded) > _MAX_TASKBOARD_JOURNAL_BYTES:
                raise ValueError(
                    "taskboard materialization journal exceeds persistence bound"
                )
            _atomic_write(journal_file, encoded)
            journal_write_count += 1
            return _taskboard_transaction_result(
                taskboard_path=board,
                journal_path=journal_file,
                preview=preview,
                transaction_id=transaction_id,
                state=TaskboardMaterializationTransactionState.COMMITTED,
                epoch_id=epoch,
                changed=bool(board_write_count),
                resumed=resumed or prefix_count > 0,
                board_write_count=board_write_count,
                journal_write_count=journal_write_count,
            )
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class TaskboardTaskRecord:
    """One lossless canonical task-source record recovered from Markdown."""

    task_id: str
    task_cid: str
    goal_id: str
    goal_cid: str
    plan_root: str
    title: str
    status: str
    dependency_task_ids: tuple[str, ...] = ()
    dependency_task_cids: tuple[str, ...] = ()
    schema: str = ""
    projection_revision: int = 1
    board_namespace: str = ""
    source_line: int = 0
    rendered_block: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "task_cid",
            "goal_id",
            "goal_cid",
            "plan_root",
            "schema",
        ):
            value = _taskboard_identifier(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        for name in ("title", "status", "board_namespace"):
            value = str(getattr(self, name) or "").strip()
            if "\x00" in value or "\n" in value or "\r" in value:
                raise ValueError(f"{name} must be single-line text")
            if name == "status" and not value:
                raise ValueError("status must not be empty")
            object.__setattr__(self, name, value)
        if self.status.lower() not in TASKBOARD_STATUSES:
            raise ValueError(f"unsupported taskboard status {self.status!r}")
        if (
            isinstance(self.projection_revision, bool)
            or not isinstance(self.projection_revision, int)
            or self.projection_revision < 1
        ):
            raise ValueError("projection_revision must be a positive integer")
        for name in ("dependency_task_ids", "dependency_task_cids"):
            values = tuple(
                _taskboard_identifier(item, name=name)
                for item in getattr(self, name)
            )
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
            object.__setattr__(self, name, values)
        if len(self.dependency_task_ids) != len(self.dependency_task_cids):
            raise ValueError("dependency aliases and CIDs must have equal population")
        if self.task_id in self.dependency_task_ids:
            raise ValueError("task cannot depend on itself")
        if self.task_cid in self.dependency_task_cids:
            raise ValueError("task CID cannot depend on itself")
        if (
            isinstance(self.source_line, bool)
            or not isinstance(self.source_line, int)
            or self.source_line < 0
        ):
            raise ValueError("source_line must be a non-negative integer")
        if not isinstance(self.rendered_block, str):
            raise TypeError("rendered_block must be text")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        canonical_metadata = json.loads(_canonical_json_bytes(dict(self.metadata)))
        object.__setattr__(self, "metadata", canonical_metadata)

    @property
    def alias(self) -> str:
        return self.task_id

    @property
    def depends_on(self) -> tuple[str, ...]:
        return self.dependency_task_ids

    def to_dict(self, *, include_block: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "task_id": self.task_id,
            "task_alias": self.task_id,
            "task_cid": self.task_cid,
            "goal_id": self.goal_id,
            "goal_cid": self.goal_cid,
            "plan_root": self.plan_root,
            "title": self.title,
            "status": self.status,
            "dependency_task_ids": list(self.dependency_task_ids),
            "dependency_task_cids": list(self.dependency_task_cids),
            "schema": self.schema,
            "projection_revision": self.projection_revision,
            "board_namespace": self.board_namespace,
            "source_line": self.source_line,
            "metadata": dict(self.metadata),
        }
        if include_block:
            value["rendered_block"] = self.rendered_block
        return value


def _assert_taskboard_acyclic(tasks: Sequence[TaskboardTaskRecord]) -> None:
    dependencies = {
        item.task_cid: tuple(item.dependency_task_cids) for item in tasks
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_cid: str) -> None:
        if task_cid in visiting:
            raise ValueError("taskboard task dependency graph contains a cycle")
        if task_cid in visited:
            return
        visiting.add(task_cid)
        for dependency in dependencies[task_cid]:
            if dependency not in dependencies:
                raise ValueError(
                    f"taskboard task references unknown dependency CID {dependency}"
                )
            visit(dependency)
        visiting.remove(task_cid)
        visited.add(task_cid)

    for task_cid in sorted(dependencies):
        visit(task_cid)


@dataclass(frozen=True)
class TaskboardSnapshot:
    """Bounded immutable view of one exact taskboard revision."""

    path: Path
    board_revision: str
    tasks: tuple[TaskboardTaskRecord, ...]
    byte_count: int
    plan_root: str = ""
    projection_schema: str = ""
    projection_revision: int = 0
    projection_id: str = ""
    schema: str = TASKBOARD_STORE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TASKBOARD_STORE_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported taskboard snapshot schema")
        tasks = tuple(self.tasks)
        if any(not isinstance(item, TaskboardTaskRecord) for item in tasks):
            raise TypeError("snapshot tasks must be TaskboardTaskRecord values")
        aliases = [item.task_id for item in tasks]
        cids = [item.task_cid for item in tasks]
        if len(aliases) != len(set(aliases)):
            raise ValueError("taskboard contains duplicate task aliases")
        if len(cids) != len(set(cids)):
            raise ValueError("taskboard contains duplicate task CIDs")
        cid_population = set(cids)
        if any(
            alias != cid and alias in cid_population
            for alias, cid in zip(aliases, cids, strict=True)
        ):
            raise ValueError("taskboard task alias collides with another task CID")
        _assert_taskboard_acyclic(tasks)
        if isinstance(self.byte_count, bool) or self.byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        expected_plan_roots = {item.plan_root for item in tasks}
        if tasks and expected_plan_roots != {self.plan_root}:
            raise ValueError("taskboard contains foreign or mixed plan roots")
        expected_schemas = {item.schema for item in tasks}
        if tasks and expected_schemas != {self.projection_schema}:
            raise ValueError("taskboard contains mixed projection schemas")
        expected_revisions = {item.projection_revision for item in tasks}
        if tasks and expected_revisions != {self.projection_revision}:
            raise ValueError("taskboard contains mixed projection revisions")
        object.__setattr__(self, "tasks", tasks)

    @property
    def revision(self) -> str:
        return self.board_revision

    @property
    def root_id(self) -> str:
        return self.plan_root

    @property
    def source_id(self) -> str:
        return self.projection_id or self.board_revision

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def task_cids(self) -> tuple[str, ...]:
        return tuple(item.task_cid for item in self.tasks)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": str(self.path),
            "board_revision": self.board_revision,
            "revision": self.board_revision,
            "byte_count": self.byte_count,
            "plan_root": self.plan_root,
            "projection_schema": self.projection_schema,
            "projection_revision": self.projection_revision,
            "projection_id": self.projection_id,
            "task_count": self.task_count,
            "task_cids": list(self.task_cids),
            "tasks": [item.to_dict() for item in self.tasks],
        }


@dataclass(frozen=True)
class TaskboardCASResult:
    """Result of one status compare-and-swap."""

    changed: bool
    task: TaskboardTaskRecord
    previous_revision: str
    board_revision: str
    event: Mapping[str, Any] = field(default_factory=dict)

    @property
    def revision(self) -> str:
        return self.board_revision

    @property
    def stale(self) -> bool:
        return False


@dataclass(frozen=True)
class TaskboardIntegrityReport:
    valid: bool
    board_revision: str = ""
    task_count: int = 0
    plan_root: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.valid

    def require_valid(self) -> "TaskboardIntegrityReport":
        if not self.valid:
            raise ValueError(
                "taskboard integrity failed: " + ", ".join(self.reason_codes)
            )
        return self


@dataclass(frozen=True)
class TaskboardWatchResult:
    changed: bool
    snapshot: TaskboardSnapshot
    events: tuple[Mapping[str, Any], ...] = ()
    cursor: Any = None
    timed_out: bool = False


class TaskboardStore:
    """Bounded canonical task-source operations over one Markdown taskboard.

    Parsing is imported lazily from :mod:`markdown_task_source`, keeping the
    long-standing low-level journal and watcher helpers in this module usable
    without importing prompt planning contracts.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        root: Path | str | None = None,
        journal_path: Path | str | None = None,
        events_path: Path | str | None = None,
        max_bytes: int = DEFAULT_TASKBOARD_BYTES_LIMIT,
        max_tasks: int = MAX_TASKBOARD_MATERIALIZATION_ENTRIES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(max_tasks, bool)
            or not isinstance(max_tasks, int)
            or max_tasks < 1
            or max_tasks > MAX_TASKBOARD_MATERIALIZATION_ENTRIES
        ):
            raise ValueError(
                "max_tasks must be within the taskboard population bound"
            )
        selected_path = Path(path).absolute()
        selected_root = (
            Path(root).resolve(strict=False)
            if root is not None
            else selected_path.parent.resolve(strict=False)
        )
        resolved_path = selected_path.resolve(strict=False)
        try:
            resolved_path.relative_to(selected_root)
        except ValueError as exc:
            raise ValueError("taskboard path escapes its configured root") from exc
        self.path = resolved_path
        self.root = selected_root
        self.journal_path = self._resolve_sidecar(
            journal_path,
            self.path.with_name(f".{self.path.name}.materialization.json"),
            "journal_path",
        )
        self.events_path = self._resolve_sidecar(
            events_path,
            self.path.with_name(f".{self.path.name}.events.jsonl"),
            "events_path",
        )
        if self.path in {self.journal_path, self.events_path}:
            raise ValueError("taskboard sidecar paths must be separate")
        if self.journal_path == self.events_path:
            raise ValueError("journal and event paths must be separate")
        self.max_bytes = max_bytes
        self.max_tasks = max_tasks
        self._thread_lock = threading.RLock()
        self._lock_path = self.path.with_name(f".{self.path.name}.store.lock")

    def _resolve_sidecar(
        self,
        supplied: Path | str | None,
        default: Path,
        name: str,
    ) -> Path:
        candidate = Path(supplied).absolute() if supplied is not None else default
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"{name} escapes the configured root") from exc
        return resolved

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._thread_lock:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read_bytes(self) -> bytes:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            payload = b""
        if len(payload) > self.max_bytes:
            raise ValueError("taskboard exceeds its configured byte bound")
        return payload

    def snapshot(self) -> TaskboardSnapshot:
        payload = self._read_bytes()
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("taskboard is not UTF-8") from exc
        from .markdown_task_source import parse_markdown_task_source

        result = parse_markdown_task_source(
            text,
            path=self.path,
            max_tasks=self.max_tasks,
            board_revision=taskboard_revision(payload),
        )
        if result.byte_count != len(payload):
            raise ValueError("taskboard snapshot byte population drift")
        return result

    load = snapshot

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_TASKBOARD_QUERY_LIMIT
        ):
            raise ValueError(
                f"limit must be in [1, {MAX_TASKBOARD_QUERY_LIMIT}]"
            )
        return limit

    def query(
        self,
        *,
        status: str | Sequence[str] | None = None,
        goal_id: str = "",
        task_cids: Iterable[str] = (),
        offset: int = 0,
        limit: int = DEFAULT_TASKBOARD_QUERY_LIMIT,
    ) -> tuple[TaskboardTaskRecord, ...]:
        selected_limit = self._bounded_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if status is None:
            statuses: set[str] | None = None
        elif isinstance(status, str):
            statuses = {status.strip().lower()}
        else:
            statuses = set()
            for index, item in enumerate(status):
                if index >= len(TASKBOARD_STATUSES):
                    raise ValueError("status filter exceeds its configured bound")
                statuses.add(str(item).strip().lower())
        if statuses is not None and (
            not statuses or not statuses.issubset(TASKBOARD_STATUSES)
        ):
            raise ValueError("status filter contains an unsupported status")
        if isinstance(task_cids, (str, bytes, bytearray, memoryview)):
            raise TypeError("task_cids must be an iterable of task CIDs")
        selected_cids: set[str] = set()
        for index, item in enumerate(task_cids):
            if index >= MAX_TASKBOARD_QUERY_LIMIT:
                raise ValueError("task_cids filter exceeds its configured bound")
            selected_cids.add(
                _taskboard_identifier(item, name="task_cids")
            )
        selected_goal = (
            _taskboard_identifier(goal_id, name="goal_id") if goal_id else ""
        )
        records = (
            item
            for item in self.snapshot().tasks
            if (statuses is None or item.status.lower() in statuses)
            and (
                not selected_goal
                or item.goal_id == selected_goal
                or item.goal_cid == selected_goal
            )
            and (not selected_cids or item.task_cid in selected_cids)
        )
        result: list[TaskboardTaskRecord] = []
        skipped = 0
        for item in records:
            if skipped < offset:
                skipped += 1
                continue
            result.append(item)
            if len(result) >= selected_limit:
                break
        return tuple(result)

    def get(self, task_id: str) -> TaskboardTaskRecord | None:
        selected = str(task_id).strip()
        if not selected:
            return None
        selected = _taskboard_identifier(selected, name="task_id")
        matches = [
            item
            for item in self.snapshot().tasks
            if item.task_id == selected or item.task_cid == selected
        ]
        if len(matches) > 1:
            raise ValueError("task identifier is ambiguous")
        return matches[0] if matches else None

    get_task = get
    find = get

    def ready_set(
        self,
        *,
        limit: int = DEFAULT_TASKBOARD_QUERY_LIMIT,
    ) -> tuple[TaskboardTaskRecord, ...]:
        selected_limit = self._bounded_limit(limit)
        snapshot = self.snapshot()
        by_cid = {item.task_cid: item for item in snapshot.tasks}
        completed = {
            item.task_cid
            for item in snapshot.tasks
            if item.status.lower() in {"completed", "complete", "done"}
        }
        ready = tuple(
            item
            for item in snapshot.tasks
            if item.status.lower()
            in {"todo", "ready", "queued", "proposed", "admitted"}
            and all(dependency in completed for dependency in item.dependency_task_cids)
            and all(dependency in by_cid for dependency in item.dependency_task_cids)
        )
        return ready[:selected_limit]

    ready = ready_set
    get_ready = ready_set
    ready_tasks = ready_set

    def compare_and_swap_status(
        self,
        task_id: str,
        *,
        expected_status: str | Sequence[str],
        new_status: str,
        expected_revision: str,
        event_payload: Mapping[str, Any] | None = None,
    ) -> TaskboardCASResult:
        if isinstance(expected_status, str):
            expected = {expected_status.strip().lower()}
        else:
            expected = set()
            for index, item in enumerate(expected_status):
                if index >= len(TASKBOARD_STATUSES):
                    raise ValueError("expected status set exceeds its bound")
                expected.add(str(item).strip().lower())
        replacement = str(new_status).strip().lower()
        if not expected or not replacement:
            raise ValueError("status CAS requires expected and replacement status")
        if any(
            not value
            or "\n" in value
            or "\r" in value
            or "\x00" in value
            for value in (*expected, replacement)
        ):
            raise ValueError("status values must be safe single-line tokens")
        if not expected.issubset(TASKBOARD_STATUSES) or replacement not in (
            TASKBOARD_STATUSES
        ):
            raise ValueError("status CAS contains an unsupported status")
        if not expected_revision:
            raise ValueError("expected_revision is required")
        selected_task_id = _taskboard_identifier(task_id, name="task_id")
        selected_revision = _taskboard_identifier(
            expected_revision, name="expected_revision"
        )
        with self._guard():
            snapshot = self.snapshot()
            if snapshot.board_revision != selected_revision:
                raise ValueError("stale taskboard revision")
            matches = [
                item
                for item in snapshot.tasks
                if item.task_id == selected_task_id
                or item.task_cid == selected_task_id
            ]
            if len(matches) != 1:
                raise KeyError(f"unknown or ambiguous task {selected_task_id!r}")
            task = matches[0]
            if task.status.lower() not in expected:
                raise ValueError("task status compare-and-swap conflict")
            if task.status.lower() == replacement:
                return TaskboardCASResult(
                    changed=False,
                    task=task,
                    previous_revision=snapshot.board_revision,
                    board_revision=snapshot.board_revision,
                )
            payload = self._read_bytes()
            text = payload.decode("utf-8")
            from .markdown_task_source import replace_markdown_task_status

            changed_text = replace_markdown_task_status(
                text,
                task_id=task.task_id,
                expected_status=task.status,
                new_status=replacement,
            )
            encoded = changed_text.encode("utf-8")
            if len(encoded) > self.max_bytes:
                raise ValueError("taskboard exceeds its configured byte bound")
            _atomic_write(self.path, encoded)
            next_snapshot = self.snapshot()
            changed_task = next(
                item for item in next_snapshot.tasks if item.task_cid == task.task_cid
            )
            event = self.append_event(
                "task_status_changed",
                {
                    **dict(event_payload or {}),
                    "schema": TASKBOARD_STORE_EVENT_SCHEMA,
                    "task_id": task.task_id,
                    "task_cid": task.task_cid,
                    "plan_root": task.plan_root,
                    "previous_status": task.status,
                    "status": replacement,
                    "previous_board_revision": snapshot.board_revision,
                    "board_revision": next_snapshot.board_revision,
                },
            )
            return TaskboardCASResult(
                changed=True,
                task=changed_task,
                previous_revision=snapshot.board_revision,
                board_revision=next_snapshot.board_revision,
                event=event,
            )

    cas_status = compare_and_swap_status
    update_status = compare_and_swap_status
    compare_and_set_status = compare_and_swap_status

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        selected_type = _taskboard_identifier(event_type, name="event_type")
        if not isinstance(payload, Mapping):
            raise TypeError("event payload must be a mapping")
        from ..event_log import append_jsonl_event

        return append_jsonl_event(
            self.events_path,
            selected_type,
            dict(payload),
            max_bytes=min(self.max_bytes, _MAX_CHECKPOINT_BYTES),
        )

    def event_cursor(self) -> Any:
        from ..event_log import latest_event_cursor

        return latest_event_cursor(self.events_path)

    def events(
        self,
        cursor: Any,
        *,
        limit: int = DEFAULT_TASKBOARD_QUERY_LIMIT,
    ) -> Any:
        selected_limit = self._bounded_limit(limit)
        from ..event_log import read_jsonl_event_page

        return read_jsonl_event_page(
            self.events_path, cursor, limit=selected_limit
        )

    read_events = events

    def watch(
        self,
        *,
        revision: str = "",
        cursor: Any = None,
        timeout: float = 0.0,
        event_limit: int = DEFAULT_TASKBOARD_QUERY_LIMIT,
    ) -> TaskboardWatchResult:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
            or timeout > DEFAULT_SAFETY_INTERVAL_SECONDS
        ):
            raise ValueError(
                "timeout must be between zero and the safety interval"
            )
        selected_event_limit = self._bounded_limit(event_limit)
        watcher = create_directory_watcher((self.path, self.events_path))
        try:
            first = self.snapshot()
            if revision and first.board_revision != revision:
                changed = True
            else:
                changed = False
                if timeout:
                    watcher.wait(timeout)
                first = self.snapshot()
                changed = bool(revision and first.board_revision != revision)
            selected_cursor = cursor
            events: tuple[Mapping[str, Any], ...] = ()
            next_cursor = cursor
            if selected_cursor is not None:
                page = self.events(selected_cursor, limit=selected_event_limit)
                events = tuple(page.events)
                next_cursor = page.next_cursor
            return TaskboardWatchResult(
                changed=changed or bool(events),
                snapshot=first,
                events=events,
                cursor=next_cursor,
                timed_out=bool(timeout and not changed and not events),
            )
        finally:
            watcher.close()

    def check_integrity(self) -> TaskboardIntegrityReport:
        try:
            snapshot = self.snapshot()
        except (OSError, TypeError, ValueError) as exc:
            reason = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(exc).strip().lower(),
            ).strip("_")
            return TaskboardIntegrityReport(
                valid=False,
                reason_codes=(reason or type(exc).__name__.lower(),),
            )
        return TaskboardIntegrityReport(
            valid=True,
            board_revision=snapshot.board_revision,
            task_count=snapshot.task_count,
            plan_root=snapshot.plan_root,
        )

    integrity = check_integrity
    verify_integrity = check_integrity
    integrity_check = check_integrity


@dataclass(frozen=True)
class PathMetadata:
    """A bounded canonical metadata snapshot for one file or directory."""

    path: str
    exists: bool
    kind: str
    device: int = 0
    inode: int = 0
    mode: int = 0
    size: int = 0
    mtime_ns: int = 0
    ctime_ns: int = 0
    entry_count: int = 0
    entries_scanned: int = 0
    entries_truncated: bool = False
    entries_digest: str = ""
    error: str = ""
    schema: str = PATH_METADATA_SCHEMA
    metadata_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != PATH_METADATA_SCHEMA:
            raise ValueError("unsupported path metadata schema")
        if not self.path:
            raise ValueError("path metadata requires a path")
        if self.kind not in {"missing", "file", "directory", "symlink", "other", "error"}:
            raise ValueError("unsupported path metadata kind")
        for name in (
            "device",
            "inode",
            "mode",
            "size",
            "mtime_ns",
            "ctime_ns",
            "entry_count",
            "entries_scanned",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected = _content_id("path-metadata", self._identity_payload())
        if self.metadata_id and self.metadata_id != expected:
            raise ValueError("path metadata identity does not match")
        object.__setattr__(self, "metadata_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "exists": self.exists,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "entry_count": self.entry_count,
            "entries_scanned": self.entries_scanned,
            "entries_truncated": self.entries_truncated,
            "entries_digest": self.entries_digest,
            "error": self.error,
        }

    @property
    def cursor(self) -> str:
        """Return the canonical cursor for change comparisons."""

        return self.metadata_id

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "metadata_id": self.metadata_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PathMetadata":
        return cls(
            schema=str(value.get("schema") or ""),
            path=str(value.get("path") or ""),
            exists=bool(value.get("exists")),
            kind=str(value.get("kind") or ""),
            device=int(value.get("device") or 0),
            inode=int(value.get("inode") or 0),
            mode=int(value.get("mode") or 0),
            size=int(value.get("size") or 0),
            mtime_ns=int(value.get("mtime_ns") or 0),
            ctime_ns=int(value.get("ctime_ns") or 0),
            entry_count=int(value.get("entry_count") or 0),
            entries_scanned=int(value.get("entries_scanned") or 0),
            entries_truncated=bool(value.get("entries_truncated")),
            entries_digest=str(value.get("entries_digest") or ""),
            error=str(value.get("error") or ""),
            metadata_id=str(value.get("metadata_id") or ""),
        )

    @classmethod
    def capture(
        cls,
        path: Path | str,
        *,
        max_entries: int = DEFAULT_METADATA_ENTRY_LIMIT,
        max_depth: int = DEFAULT_METADATA_DEPTH_LIMIT,
    ) -> "PathMetadata":
        """Capture stable stat metadata and a bounded directory-tree digest."""

        target = Path(path)
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        display_path = str(target.absolute())
        try:
            stat_result = target.lstat()
        except FileNotFoundError:
            return cls(path=display_path, exists=False, kind="missing")
        except OSError as exc:
            return cls(
                path=display_path,
                exists=False,
                kind="error",
                error=f"{type(exc).__name__}:{exc.errno or 0}",
            )

        if target.is_symlink():
            kind = "symlink"
        elif target.is_file():
            kind = "file"
        elif target.is_dir():
            kind = "directory"
        else:
            kind = "other"

        entries: list[tuple[str, int, int, int, int, int, str]] = []
        entry_count = 0
        truncated = False
        if kind == "directory":
            pending: deque[tuple[Path, str, int]] = deque([(target, "", 0)])
            while pending:
                directory, relative_parent, depth = pending.popleft()
                try:
                    child_stream = os.scandir(directory)
                except OSError as exc:
                    entries.append(
                        (
                            relative_parent,
                            0,
                            0,
                            0,
                            0,
                            0,
                            f"error:{exc.errno or 0}",
                        )
                    )
                    continue
                with child_stream:
                    for child in child_stream:
                        entry_count += 1
                        if len(entries) >= max_entries:
                            truncated = True
                            break
                        relative = (
                            f"{relative_parent}/{child.name}"
                            if relative_parent
                            else child.name
                        )
                        try:
                            child_stat = child.stat(follow_symlinks=False)
                            if child.is_symlink():
                                child_kind = "symlink"
                            elif child.is_dir(follow_symlinks=False):
                                child_kind = "directory"
                            elif child.is_file(follow_symlinks=False):
                                child_kind = "file"
                            else:
                                child_kind = "other"
                            entries.append(
                                (
                                    relative,
                                    int(child_stat.st_mode),
                                    int(child_stat.st_size),
                                    int(child_stat.st_mtime_ns),
                                    int(child_stat.st_ctime_ns),
                                    int(child_stat.st_ino),
                                    child_kind,
                                )
                            )
                            if (
                                child_kind == "directory"
                                and depth < max_depth
                                and len(entries) < max_entries
                            ):
                                pending.append(
                                    (Path(child.path), relative, depth + 1)
                                )
                        except OSError as exc:
                            entries.append(
                                (
                                    relative,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    f"error:{exc.errno or 0}",
                                )
                            )
                if truncated:
                    break
            entries.sort(key=lambda item: item[0])
        entries_digest = (
            "sha256:" + hashlib.sha256(_canonical_json_bytes(entries)).hexdigest()
            if entries
            else ""
        )
        return cls(
            path=display_path,
            exists=True,
            kind=kind,
            device=int(stat_result.st_dev),
            inode=int(stat_result.st_ino),
            mode=int(stat_result.st_mode),
            size=int(stat_result.st_size),
            mtime_ns=int(stat_result.st_mtime_ns),
            ctime_ns=int(stat_result.st_ctime_ns),
            entry_count=entry_count,
            entries_scanned=len(entries),
            entries_truncated=truncated,
            entries_digest=entries_digest,
        )


def path_metadata(
    path: Path | str,
    *,
    max_entries: int = DEFAULT_METADATA_ENTRY_LIMIT,
    max_depth: int = DEFAULT_METADATA_DEPTH_LIMIT,
) -> PathMetadata:
    """Compatibility-friendly function form of :meth:`PathMetadata.capture`."""

    return PathMetadata.capture(
        path,
        max_entries=max_entries,
        max_depth=max_depth,
    )


metadata_snapshot = path_metadata


def task_ids_from_artifact_names(
    directory: Path,
    *,
    task_prefix: str,
) -> set[str]:
    """Recover allocated display IDs from durable discovery filenames."""

    if not directory.exists():
        return set()
    normalized = task_prefix.rstrip("-") + "-"
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?P<number>\d+)(?!\d)",
        flags=re.IGNORECASE,
    )
    task_ids: set[str] = set()
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        for match in pattern.finditer(path.name):
            number = int(match.group("number"))
            task_ids.add(f"{normalized}{number:03d}")
    return task_ids


def _projection_delta(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    changed = {
        key: current[key]
        for key in sorted(current)
        if key not in previous or previous[key] != current[key]
    }
    removed = [key for key in sorted(previous) if key not in current]
    return {"set": changed, "remove": removed}


@dataclass(frozen=True)
class ProjectionCheckpointResult:
    """Result of one conditional projection/checkpoint materialization."""

    changed: bool
    projection_changed: bool
    cursor_changed: bool
    checkpoint_id: str
    projection_id: str
    delta: Mapping[str, Any] = field(default_factory=dict)
    write_count: int = 0

    @property
    def written(self) -> bool:
        return self.write_count > 0


class ProjectionDeltaCheckpointStore:
    """Atomically bind a projection and canonical event cursor.

    The durable record contains the recoverable current projection plus the
    exact top-level delta from the preceding checkpoint.  Identical projection
    and cursor inputs return without opening a temporary output or fsyncing.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int = _MAX_CHECKPOINT_BYTES,
    ) -> None:
        self.path = Path(path)
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._thread_lock = threading.RLock()
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._thread_lock:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any] | None:
        try:
            raw = self.path.read_bytes()
        except (FileNotFoundError, OSError):
            return None
        if len(raw) > self.max_bytes:
            raise ValueError("projection checkpoint exceeds persistence bound")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("projection checkpoint is malformed") from exc
        if not isinstance(value, dict):
            raise ValueError("projection checkpoint must be an object")
        if value.get("schema") != PROJECTION_DELTA_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported projection checkpoint schema")
        expected = _content_id(
            "projection-checkpoint",
            {key: item for key, item in value.items() if key != "checkpoint_id"},
        )
        if value.get("checkpoint_id") != expected:
            raise ValueError("projection checkpoint identity does not match")
        projection = value.get("projection")
        if not isinstance(projection, dict):
            raise ValueError("projection checkpoint projection must be an object")
        cursor = value.get("cursor")
        if not isinstance(cursor, Mapping):
            raise ValueError("projection checkpoint cursor must be an object")
        EventCursor.from_dict(cursor)
        return value

    def load(self) -> tuple[dict[str, Any], EventCursor] | None:
        # Writers publish complete checkpoints with ``os.replace``.  A reader
        # therefore sees either the previous complete file or the next one and
        # does not need to create/acquire the sidecar lock.  Keeping this path
        # lock-free is important for read-only daemon startup: merely
        # inspecting an absent checkpoint must not create repository state.
        record = self._read_unlocked()
        if record is None:
            return None
        return (
            dict(record["projection"]),
            EventCursor.from_dict(record["cursor"]),
        )

    def load_record(self) -> dict[str, Any] | None:
        record = self._read_unlocked()
        return None if record is None else dict(record)

    def materialize(
        self,
        projection: Mapping[str, Any],
        cursor: EventCursor | Mapping[str, Any] | str,
    ) -> ProjectionCheckpointResult:
        """Persist one changed projection/cursor pair and its compact delta."""

        if not isinstance(projection, Mapping):
            raise TypeError("projection must be a mapping")
        canonical_projection = json.loads(_canonical_json_bytes(projection))
        if not isinstance(canonical_projection, dict):
            raise TypeError("projection must be a JSON object")
        if isinstance(cursor, str):
            canonical_cursor = EventCursor.from_token(cursor)
        elif isinstance(cursor, EventCursor):
            canonical_cursor = cursor
        elif isinstance(cursor, Mapping):
            canonical_cursor = EventCursor.from_dict(cursor)
        else:
            raise TypeError("cursor must be an EventCursor, record, or token")

        with self._guard():
            previous = self._read_unlocked()
            previous_projection = (
                dict(previous.get("projection") or {}) if previous else {}
            )
            previous_cursor = (
                EventCursor.from_dict(previous["cursor"]) if previous else None
            )
            if previous_cursor is not None:
                if canonical_cursor.stream_id != previous_cursor.stream_id:
                    raise ValueError(
                        "projection checkpoint cursor belongs to a different stream"
                    )
                if canonical_cursor.position < previous_cursor.position:
                    raise ValueError(
                        "projection checkpoint cursor cannot move backwards"
                    )
                if (
                    canonical_cursor.position == previous_cursor.position
                    and canonical_cursor != previous_cursor
                ):
                    raise ValueError(
                        "projection checkpoint cursor anchor changed at the same position"
                    )
            projection_changed = (
                previous is None or previous_projection != canonical_projection
            )
            cursor_changed = (
                previous_cursor is None or previous_cursor != canonical_cursor
            )
            delta = _projection_delta(
                previous_projection,
                canonical_projection,
            )
            projection_id = _content_id(
                "runtime-projection", canonical_projection
            )
            if not projection_changed and not cursor_changed:
                return ProjectionCheckpointResult(
                    changed=False,
                    projection_changed=False,
                    cursor_changed=False,
                    checkpoint_id=str(previous.get("checkpoint_id") or ""),
                    projection_id=projection_id,
                    delta=delta,
                    write_count=0,
                )
            generation = int(previous.get("generation") or 0) + 1 if previous else 1
            record: dict[str, Any] = {
                "schema": PROJECTION_DELTA_CHECKPOINT_SCHEMA,
                "generation": generation,
                "projection_id": projection_id,
                "projection": canonical_projection,
                "cursor": canonical_cursor.to_record(),
                "delta": delta,
            }
            record["checkpoint_id"] = _content_id(
                "projection-checkpoint", record
            )
            encoded = _canonical_json_bytes(record) + b"\n"
            if len(encoded) > self.max_bytes:
                raise ValueError("projection checkpoint exceeds persistence bound")
            _atomic_write(self.path, encoded)
            return ProjectionCheckpointResult(
                changed=True,
                projection_changed=projection_changed,
                cursor_changed=cursor_changed,
                checkpoint_id=record["checkpoint_id"],
                projection_id=projection_id,
                delta=delta,
                write_count=1,
            )

    # Concise aliases for callers which model checkpoints as conditional saves.
    checkpoint = materialize
    store = materialize


class NativeWatcherUnavailable(OSError):
    """Native directory notification is unavailable for the selected targets."""


class BlockingTimerWatcher:
    """Blocking pipe/timer fallback with no periodic spin loop."""

    backend = "blocking_timer"
    native = False

    def __init__(self, _paths: Iterable[Path | str] = ()) -> None:
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._read_fd, False)
        os.set_blocking(self._write_fd, False)
        self._closed = False

    def notify(self) -> None:
        if self._closed:
            return
        try:
            os.write(self._write_fd, b"\0")
        except BlockingIOError:
            pass
        except OSError:
            if not self._closed:
                raise

    wake = notify

    def wait(self, timeout: float | None) -> bool:
        if self._closed:
            return False
        selected_timeout = None if timeout is None else max(0.0, float(timeout))
        try:
            ready, _writable, _errors = select.select(
                [self._read_fd], [], [], selected_timeout
            )
        except InterruptedError:
            return True
        if not ready:
            return False
        while True:
            try:
                if not os.read(self._read_fd, 4096):
                    break
            except BlockingIOError:
                break
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._read_fd)
        os.close(self._write_fd)

    def __enter__(self) -> "BlockingTimerWatcher":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class LinuxDirectoryWatcher:
    """Small stdlib/ctypes inotify adapter used only as a wakeup hint."""

    backend = "inotify"
    native = True

    _IN_NONBLOCK = getattr(os, "O_NONBLOCK", 0x800)
    _IN_CLOEXEC = getattr(os, "O_CLOEXEC", 0x80000)
    _IN_MASK = (
        0x00000002  # IN_MODIFY
        | 0x00000004  # IN_ATTRIB
        | 0x00000008  # IN_CLOSE_WRITE
        | 0x00000040  # IN_MOVED_FROM
        | 0x00000080  # IN_MOVED_TO
        | 0x00000100  # IN_CREATE
        | 0x00000200  # IN_DELETE
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
    )

    def __init__(
        self,
        paths: Iterable[Path | str],
        *,
        max_directories: int = DEFAULT_METADATA_ENTRY_LIMIT,
    ) -> None:
        if os.name != "posix" or not hasattr(ctypes, "CDLL"):
            raise NativeWatcherUnavailable(
                errno.ENOSYS, "native directory notification is unavailable"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        initializer = getattr(libc, "inotify_init1", None)
        add_watch = getattr(libc, "inotify_add_watch", None)
        if initializer is None or add_watch is None:
            raise NativeWatcherUnavailable(
                errno.ENOSYS, "inotify is unavailable"
            )
        initializer.argtypes = [ctypes.c_int]
        initializer.restype = ctypes.c_int
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        fd = initializer(self._IN_NONBLOCK | self._IN_CLOEXEC)
        if fd < 0:
            error_number = ctypes.get_errno()
            raise NativeWatcherUnavailable(
                error_number, os.strerror(error_number)
            )
        self._fd = fd
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._read_fd, False)
        os.set_blocking(self._write_fd, False)
        self._closed = False
        self._watch_descriptors: dict[int, Path] = {}
        try:
            directories = self._watch_directories(
                paths, max_directories=max_directories
            )
            if not directories:
                directories = [Path.cwd()]
            for directory in directories:
                descriptor = add_watch(
                    self._fd,
                    os.fsencode(directory),
                    self._IN_MASK,
                )
                if descriptor < 0:
                    error_number = ctypes.get_errno()
                    raise NativeWatcherUnavailable(
                        error_number,
                        f"cannot watch {directory}: {os.strerror(error_number)}",
                    )
                self._watch_descriptors[int(descriptor)] = directory
        except Exception:
            self.close()
            raise

    @staticmethod
    def _nearest_existing_directory(path: Path) -> Path:
        candidate = path if path.is_dir() else path.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    @classmethod
    def _watch_directories(
        cls,
        paths: Iterable[Path | str],
        *,
        max_directories: int,
    ) -> list[Path]:
        directories: list[Path] = []
        seen: set[Path] = set()
        pending: deque[Path] = deque(
            cls._nearest_existing_directory(Path(path).absolute())
            for path in paths
        )
        while pending and len(directories) < max_directories:
            directory = pending.popleft()
            try:
                key = directory.resolve()
            except OSError:
                key = directory.absolute()
            if key in seen or not directory.is_dir():
                continue
            seen.add(key)
            directories.append(directory)
            try:
                child_stream = os.scandir(directory)
            except OSError:
                continue
            with child_stream:
                for item in child_stream:
                    if len(directories) + len(pending) >= max_directories:
                        break
                    try:
                        is_directory = item.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_directory:
                        pending.append(Path(item.path))
        return directories

    def notify(self) -> None:
        if self._closed:
            return
        try:
            os.write(self._write_fd, b"\0")
        except BlockingIOError:
            pass

    wake = notify

    def wait(self, timeout: float | None) -> bool:
        if self._closed:
            return False
        selected_timeout = None if timeout is None else max(0.0, float(timeout))
        try:
            ready, _writable, _errors = select.select(
                [self._fd, self._read_fd], [], [], selected_timeout
            )
        except InterruptedError:
            return True
        if not ready:
            return False
        if self._read_fd in ready:
            try:
                while os.read(self._read_fd, 4096):
                    pass
            except BlockingIOError:
                pass
        if self._fd in ready:
            try:
                while os.read(self._fd, 64 * 1024):
                    pass
            except BlockingIOError:
                pass
        return True

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for descriptor in (self._fd, self._read_fd, self._write_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> "LinuxDirectoryWatcher":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def create_directory_watcher(
    paths: Iterable[Path | str],
    *,
    prefer_native: bool = True,
) -> LinuxDirectoryWatcher | BlockingTimerWatcher:
    """Create a native watcher, safely falling back to a blocking pipe/timer."""

    selected_paths = tuple(Path(path) for path in paths)
    if prefer_native and selected_paths:
        try:
            return LinuxDirectoryWatcher(selected_paths)
        except (NativeWatcherUnavailable, OSError, ValueError):
            pass
    return BlockingTimerWatcher(selected_paths)


# Public compatibility aliases use capability-oriented rather than OS names.
NativeDirectoryWatcher = LinuxDirectoryWatcher
FallbackDirectoryWatcher = BlockingTimerWatcher
DeltaCheckpointStore = ProjectionDeltaCheckpointStore
ProjectionDeltaStore = ProjectionDeltaCheckpointStore
DirectoryWatcher = LinuxDirectoryWatcher


class RuntimeWakeKind(str, Enum):
    TASK_BOARD = "task_board"
    OBJECTIVE = "objective"
    REPOSITORY = "repository"
    CHILD_PROCESS = "child_process"
    LEASE = "lease"
    VALIDATION = "validation"
    PROVIDER_CAPACITY = "provider_capacity"
    POLICY = "policy"
    OBSERVATION_WINDOW = "observation_window"


RUNTIME_WAKE_KINDS: Final = tuple(item.value for item in RuntimeWakeKind)


@dataclass(frozen=True)
class RuntimeWakeEvent:
    """One coalesced wake decision backed by canonical source cursors."""

    kinds: tuple[RuntimeWakeKind, ...]
    metadata: tuple[PathMetadata, ...] = ()
    semantic_cursors: Mapping[str, str] = field(default_factory=dict)
    reason: str = "notification"
    safety_timer: bool = False

    @property
    def kind(self) -> RuntimeWakeKind:
        return self.kinds[0]

    @property
    def cursor_ids(self) -> tuple[str, ...]:
        values = [item.cursor for item in self.metadata]
        values.extend(
            self.semantic_cursors[key] for key in sorted(self.semantic_cursors)
        )
        return tuple(values)


class RuntimeWakeCoordinator:
    """Block for semantic/native wakeups and reconcile them with metadata.

    ``wait`` is two-phase: it does not advance acknowledged cursors.
    Call :meth:`acknowledge` only after the corresponding projection delta is
    durable.  A failure before acknowledgement therefore replays the wake.
    """

    def __init__(
        self,
        targets: Mapping[
            RuntimeWakeKind | str,
            Path | str | Iterable[Path | str],
        ] | None = None,
        *,
        safety_interval_seconds: float = DEFAULT_SAFETY_INTERVAL_SECONDS,
        prefer_native: bool = True,
        watcher: Any | None = None,
        clock: Any = time.monotonic,
        metadata_entry_limit: int = DEFAULT_METADATA_ENTRY_LIMIT,
        metadata_depth_limit: int = DEFAULT_METADATA_DEPTH_LIMIT,
    ) -> None:
        if safety_interval_seconds <= 0:
            raise ValueError("safety_interval_seconds must be positive")
        self.safety_interval_seconds = float(safety_interval_seconds)
        self.clock = clock
        self.metadata_entry_limit = int(metadata_entry_limit)
        self.metadata_depth_limit = int(metadata_depth_limit)
        self._targets: dict[RuntimeWakeKind, tuple[Path, ...]] = {}
        for raw_kind, raw_paths in (targets or {}).items():
            kind = self._coerce_kind(raw_kind)
            if isinstance(raw_paths, (str, Path)):
                paths = (Path(raw_paths),)
            else:
                paths = tuple(Path(path) for path in raw_paths)
            self._targets[kind] = paths
        all_paths = tuple(
            path for paths in self._targets.values() for path in paths
        )
        self.watcher = watcher or create_directory_watcher(
            all_paths, prefer_native=prefer_native
        )
        self._acknowledged_metadata = self._capture_all()
        self._pending_hints: dict[RuntimeWakeKind, str] = {}
        self._acknowledged_semantic: dict[RuntimeWakeKind, str] = {}
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _coerce_kind(value: RuntimeWakeKind | str) -> RuntimeWakeKind:
        if isinstance(value, RuntimeWakeKind):
            return value
        return RuntimeWakeKind(str(value))

    @property
    def backend(self) -> str:
        return str(getattr(self.watcher, "backend", "custom"))

    @property
    def native(self) -> bool:
        return bool(getattr(self.watcher, "native", False))

    def _capture_all(
        self,
    ) -> dict[RuntimeWakeKind, tuple[PathMetadata, ...]]:
        return {
            kind: tuple(
                PathMetadata.capture(
                    path,
                    max_entries=self.metadata_entry_limit,
                    max_depth=self.metadata_depth_limit,
                )
                for path in paths
            )
            for kind, paths in self._targets.items()
        }

    def notify(
        self,
        kind: RuntimeWakeKind | str,
        *,
        revision: str = "",
    ) -> None:
        """Publish a coalescible hint and interrupt the blocking waiter."""

        selected = self._coerce_kind(kind)
        token = str(revision).strip()
        if not token:
            token = _content_id(
                "wake-hint",
                {
                    "kind": selected.value,
                    "ordinal": time.monotonic_ns(),
                },
            )
        with self._lock:
            self._pending_hints[selected] = token
        self.watcher.notify()

    publish = notify
    signal = notify

    def register_child_process(
        self,
        process: Any,
        *,
        revision: str = "",
    ) -> threading.Thread:
        """Wake when a ``subprocess.Popen``-compatible child exits."""

        def await_child() -> None:
            returncode = process.wait()
            token = revision or f"pid:{getattr(process, 'pid', 0)}:rc:{returncode}"
            self.notify(RuntimeWakeKind.CHILD_PROCESS, revision=token)

        thread = threading.Thread(
            target=await_child,
            name="agent-supervisor-child-wake",
            daemon=True,
        )
        thread.start()
        return thread

    def _candidate(
        self,
        *,
        safety_timer: bool,
    ) -> RuntimeWakeEvent | None:
        current = self._capture_all()
        changed_kinds = {
            kind
            for kind, snapshots in current.items()
            if snapshots != self._acknowledged_metadata.get(kind, ())
        }
        with self._lock:
            hints = dict(self._pending_hints)
        semantic: dict[str, str] = {}
        stale_semantic: list[RuntimeWakeKind] = []
        for kind, token in hints.items():
            # File-backed hints are verified against canonical metadata, so a
            # duplicate native notification cannot trigger an expensive pass.
            if kind in self._targets:
                continue
            if self._acknowledged_semantic.get(kind) != token:
                changed_kinds.add(kind)
                semantic[kind.value] = token
            else:
                stale_semantic.append(kind)
        if stale_semantic:
            with self._lock:
                for kind in stale_semantic:
                    if self._pending_hints.get(kind) == hints[kind]:
                        self._pending_hints.pop(kind, None)
        if safety_timer:
            changed_kinds.add(RuntimeWakeKind.OBSERVATION_WINDOW)
        if not changed_kinds:
            # Discard only proven-spurious file hints.  Semantic hints remain
            # pending until their wake has been durably acknowledged.
            with self._lock:
                for kind in tuple(self._pending_hints):
                    if kind in self._targets:
                        self._pending_hints.pop(kind, None)
            return None
        selected_metadata = tuple(
            snapshot
            for kind in sorted(changed_kinds, key=lambda item: item.value)
            for snapshot in current.get(kind, ())
        )
        return RuntimeWakeEvent(
            kinds=tuple(sorted(changed_kinds, key=lambda item: item.value)),
            metadata=selected_metadata,
            semantic_cursors=semantic,
            reason="safety_timer" if safety_timer else "notification",
            safety_timer=safety_timer,
        )

    def wait(self, timeout: float | None = None) -> RuntimeWakeEvent:
        """Block until a meaningful wake or the safety observation window."""

        if self._closed:
            raise RuntimeError("runtime wake coordinator is closed")
        maximum = (
            self.safety_interval_seconds
            if timeout is None
            else min(self.safety_interval_seconds, max(0.0, float(timeout)))
        )
        deadline = self.clock() + maximum
        while True:
            with self._lock:
                has_pending = bool(self._pending_hints)
            if has_pending:
                candidate = self._candidate(safety_timer=False)
                if candidate is not None:
                    return candidate
            remaining = max(0.0, deadline - self.clock())
            if remaining <= 0:
                candidate = self._candidate(safety_timer=True)
                assert candidate is not None
                return candidate
            notified = bool(self.watcher.wait(remaining))
            if not notified:
                candidate = self._candidate(safety_timer=True)
                assert candidate is not None
                return candidate
            candidate = self._candidate(safety_timer=False)
            if candidate is not None:
                return candidate

    wait_for_wake = wait

    def acknowledge(self, event: RuntimeWakeEvent) -> None:
        """Advance only cursors represented by a successfully applied wake."""

        if not isinstance(event, RuntimeWakeEvent):
            raise TypeError("event must be a RuntimeWakeEvent")
        by_path = {item.path: item for item in event.metadata}
        with self._lock:
            for kind in event.kinds:
                if kind in self._targets:
                    previous = self._acknowledged_metadata.get(kind, ())
                    self._acknowledged_metadata[kind] = tuple(
                        by_path.get(item.path, item) for item in previous
                    )
                    self._pending_hints.pop(kind, None)
                token = event.semantic_cursors.get(kind.value)
                if token:
                    self._acknowledged_semantic[kind] = token
                    if self._pending_hints.get(kind) == token:
                        self._pending_hints.pop(kind, None)

    ack = acknowledge

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.watcher.close()

    def __enter__(self) -> "RuntimeWakeCoordinator":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
