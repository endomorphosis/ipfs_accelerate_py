"""JSONL event-log helpers for agent supervisor runtimes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

from ..control.control_contracts import (
    CursorReplayError,
    EventCursor,
    EventCursorError,
    EventPage,
    replay_event_page,
)
from ..self_improvement.supervisor_v2_contracts import MAX_PROJECTION_BYTES, MAX_RECEIPT_BYTES


# Event log rotation: archive when file exceeds this size (default 50MB)
_EVENT_LOG_MAX_BYTES_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_MAX_BYTES"
_DEFAULT_EVENT_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB

# Keep only the most recent N events after rotation
_EVENT_LOG_RETAIN_RECENT_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_RETAIN_RECENT"
_DEFAULT_EVENT_LOG_RETAIN_RECENT = 500
_EVENT_LOG_MAX_ARCHIVES_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_MAX_ARCHIVES"
_DEFAULT_EVENT_LOG_MAX_ARCHIVES = 8

EVENT_LOG_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.event-log-manifest@2"
)
LEGACY_EVENT_LOG_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.event-log-manifest@1"
)
EVENT_CURSOR_CHECKPOINT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.event-cursor-checkpoint@1"
)
SEMANTIC_CHANGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/semantic-change@1"
)
SEMANTIC_CHANGE_EVENT_TYPE: Final = "decision_runtime_semantic_change"
_EVENT_OFFSET_INDEX_STRIDE = 256
_EVENT_OFFSET_INDEX_MAX_ITEMS = 4096
_EVENT_RECOVERY_TAIL_MAX_BYTES = 16 * MAX_PROJECTION_BYTES
_RESERVED_EVENT_FIELDS = frozenset(
    {
        "stream_id",
        "snapshot_id",
        "sequence",
        "position",
        "event_id",
        "previous_event_id",
    }
)


class EventPayloadTooLarge(ValueError):
    """An event exceeded its receipt or routine projection bound."""


class SemanticChangeIntegrityError(CursorReplayError):
    """A logical semantic-change event is malformed, duplicated, or reordered."""


class SemanticChangeKind(str, Enum):
    """Closed population of inputs which can invalidate runtime authority."""

    WORKTREE = "worktree"
    AST = "ast"
    EFFECT = "effect"
    INTENT_IR = "intent_ir"
    LEGAL_IR = "legal_ir"
    SECURITY_IR = "security_ir"
    POLICY = "policy"
    TOOL_CATALOG = "tool_catalog"
    CAPABILITY = "capability"
    PROOF = "proof"
    MONITOR = "monitor"
    LEASE = "lease"
    OBSERVED_EFFECT = "observed_effect"


@dataclass(frozen=True)
class SemanticChange:
    """Content-addressed logical change carried by the canonical event stream.

    The outer JSONL event supplies stream ordering and durability. ``change_id``
    supplies idempotency across replay and binds the semantic old-to-new
    transition independently from timestamps or physical log rotation.
    """

    kind: SemanticChangeKind | str
    subject_id: str
    previous_root_id: str
    current_root_id: str
    scope_kind: str
    scope_value: str
    repository_id: str = ""
    tree_id: str = ""
    semantic_dependency_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SEMANTIC_CHANGE_SCHEMA
    change_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_CHANGE_SCHEMA:
            raise SemanticChangeIntegrityError(
                "unsupported semantic change schema"
            )
        try:
            native_kind = (
                self.kind
                if isinstance(self.kind, SemanticChangeKind)
                else SemanticChangeKind(str(self.kind))
            )
        except ValueError as exc:
            raise SemanticChangeIntegrityError(
                f"unknown semantic change kind {self.kind!r}"
            ) from exc
        object.__setattr__(self, "kind", native_kind)
        for name in (
            "subject_id",
            "previous_root_id",
            "current_root_id",
            "scope_kind",
            "scope_value",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value or "\x00" in value:
                raise SemanticChangeIntegrityError(
                    f"semantic change {name} must be non-empty"
                )
            object.__setattr__(self, name, value)
        for name in ("repository_id", "tree_id"):
            value = str(getattr(self, name) or "").strip()
            if "\x00" in value:
                raise SemanticChangeIntegrityError(
                    f"semantic change {name} contains NUL"
                )
            object.__setattr__(self, name, value)
        if self.previous_root_id == self.current_root_id:
            raise SemanticChangeIntegrityError(
                "semantic change must advance its root"
            )
        dependency_ids = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in self.semantic_dependency_ids
                    if str(item).strip()
                }
            )
        )
        if len(dependency_ids) != len(tuple(self.semantic_dependency_ids)):
            raise SemanticChangeIntegrityError(
                "semantic change contains empty or duplicate dependency IDs"
            )
        object.__setattr__(self, "semantic_dependency_ids", dependency_ids)
        if not isinstance(self.metadata, Mapping):
            raise SemanticChangeIntegrityError(
                "semantic change metadata must be an object"
            )
        try:
            canonical_metadata = json.loads(
                _canonical_event_bytes(dict(self.metadata), MAX_PROJECTION_BYTES)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SemanticChangeIntegrityError(
                "semantic change metadata is not canonical JSON"
            ) from exc
        object.__setattr__(
            self, "metadata", MappingProxyType(canonical_metadata)
        )
        expected = _canonical_identity(self.to_dict(include_identity=False))
        if self.change_id and self.change_id != expected:
            raise SemanticChangeIntegrityError(
                "semantic change identity mismatch"
            )
        object.__setattr__(self, "change_id", expected)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "previous_root_id": self.previous_root_id,
            "current_root_id": self.current_root_id,
            "scope_kind": self.scope_kind,
            "scope_value": self.scope_value,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "semantic_dependency_ids": list(self.semantic_dependency_ids),
            "metadata": dict(self.metadata),
        }
        if include_identity:
            value["change_id"] = self.change_id
        return value

    to_record = to_dict

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticChange":
        if not isinstance(value, Mapping):
            raise SemanticChangeIntegrityError(
                "semantic change must be an object"
            )
        allowed = {
            "schema",
            "kind",
            "change_kind",
            "subject_id",
            "previous_root_id",
            "previous_revision",
            "current_root_id",
            "replacement_revision",
            "scope_kind",
            "scope_value",
            "repository_id",
            "tree_id",
            "semantic_dependency_ids",
            "metadata",
            "change_id",
        }
        if set(value).difference(allowed):
            raise SemanticChangeIntegrityError(
                "semantic change contains unknown fields"
            )
        return cls(
            schema=str(value.get("schema") or SEMANTIC_CHANGE_SCHEMA),
            kind=value.get("kind", value.get("change_kind", "")),
            subject_id=str(value.get("subject_id") or ""),
            previous_root_id=str(
                value.get("previous_root_id", value.get("previous_revision", ""))
                or ""
            ),
            current_root_id=str(
                value.get(
                    "current_root_id", value.get("replacement_revision", "")
                )
                or ""
            ),
            scope_kind=str(value.get("scope_kind") or ""),
            scope_value=str(value.get("scope_value") or ""),
            repository_id=str(value.get("repository_id") or ""),
            tree_id=str(value.get("tree_id") or ""),
            semantic_dependency_ids=tuple(
                value.get("semantic_dependency_ids") or ()
            ),
            metadata=value.get("metadata") or {},
            change_id=str(value.get("change_id") or ""),
        )


@dataclass(frozen=True)
class SemanticChangePage:
    """Verified logical changes and the exact physical cursor they consumed."""

    changes: tuple[SemanticChange, ...]
    next_cursor: EventCursor
    has_more: bool
    event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.next_cursor, EventCursor):
            raise TypeError("next_cursor must be an EventCursor")
        if len(self.changes) != len(self.event_ids):
            raise SemanticChangeIntegrityError(
                "semantic change page event bindings are incomplete"
            )


_EVENT_LOCKS: dict[str, threading.RLock] = {}
_EVENT_LOCKS_GUARD = threading.Lock()


def _event_thread_lock(path: Path) -> threading.RLock:
    identity = str(path.absolute())
    with _EVENT_LOCKS_GUARD:
        return _EVENT_LOCKS.setdefault(identity, threading.RLock())


class _EventLogLock:
    def __init__(self, path: Path) -> None:
        self._thread_lock = _event_thread_lock(path)
        self._path = path.with_name(f".{path.name}.lock")
        self._handle: Any = None

    def __enter__(self) -> None:
        self._thread_lock.acquire()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a+b")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
            raise

    def __exit__(self, *_args: Any) -> None:
        assert self._handle is not None
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._thread_lock.release()


def _canonical_event_bytes(value: Mapping[str, Any], maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("event payload must contain canonical JSON values") from exc
    if len(encoded) > maximum:
        raise EventPayloadTooLarge(
            f"event exceeds the {maximum}-byte persistence bound"
        )
    return encoded


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def _event_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _canonical_identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _event_identity(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("event_id", None)
    return _canonical_identity(body)


def _event_stream_binding(path: Path) -> tuple[str, str]:
    try:
        identity = str(path.resolve())
    except OSError:
        identity = str(path.absolute())
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (
        f"event-log:sha256:{digest}",
        f"event-log-snapshot:sha256:{digest}",
    )


def _source_paths(path: Path) -> list[Path]:
    sources: list[Path] = []
    if path.parent.exists():
        sources.extend(sorted(path.parent.glob(f"{path.name}.rotated-*")))
    if path.exists() and not path.is_dir():
        sources.append(path)
    return sources


def _stat_fields(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_record(path: Path) -> dict[str, Any]:
    """Return a compatibility record for one physical JSONL segment."""

    digest = hashlib.sha256()
    count = 0
    size = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            if line.strip():
                count += 1
    record = {
        "path": path.name,
        "size_bytes": size,
        "event_count": count,
        "sha256": digest.hexdigest(),
    }
    try:
        record.update(_stat_fields(path))
    except OSError:
        pass
    return record


def _scan_event_log(
    path: Path,
    *,
    generation: int = 0,
    stream_id: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Rebuild canonical segment metadata.

    This is deliberately the recovery path, not the append path. Legacy
    events are assigned deterministic virtual positions without rewriting the
    source files. Exact canonical duplicates left by a rotation crash are
    represented by overlapping segment ranges and deduplicated during replay.
    """

    default_stream, default_snapshot = _event_stream_binding(path)
    selected_stream = str(stream_id or default_stream)
    selected_snapshot = str(snapshot_id or default_snapshot)
    latest_sequence = 0
    latest_event_id = ""
    earliest_sequence = 0
    identities: dict[int, str] = {}
    records: list[dict[str, Any]] = []
    for source in _source_paths(path):
        digest = hashlib.sha256()
        physical_count = 0
        size = 0
        first_sequence = 0
        last_sequence = 0
        starting_previous_event_id = latest_event_id
        offsets: list[list[int]] = []
        all_canonical = True
        with source.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    raw_event = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(raw_event, dict):
                    continue
                physical_count += 1
                raw_sequence = raw_event.get(
                    "sequence", raw_event.get("position")
                )
                canonical = (
                    isinstance(raw_sequence, int)
                    and not isinstance(raw_sequence, bool)
                    and raw_sequence > 0
                    and str(raw_event.get("stream_id") or "") == selected_stream
                    and str(raw_event.get("snapshot_id") or "")
                    == selected_snapshot
                )
                sequence = int(raw_sequence) if canonical else latest_sequence + 1
                all_canonical = all_canonical and canonical
                event = dict(raw_event)
                if not canonical:
                    event.update(
                        {
                            "stream_id": selected_stream,
                            "snapshot_id": selected_snapshot,
                            "sequence": sequence,
                            "previous_event_id": latest_event_id,
                        }
                    )
                event_id = str(event.get("event_id") or "")
                if not event_id:
                    event_id = _event_identity(event)
                elif event_id != _event_identity(event):
                    raise CursorReplayError(
                        f"event {sequence} has a non-canonical event_id"
                    )
                known_identity = identities.get(sequence)
                if known_identity is not None:
                    if known_identity != event_id:
                        raise CursorReplayError(
                            f"event sequence {sequence} has conflicting identities"
                        )
                else:
                    if latest_sequence and sequence != latest_sequence + 1:
                        raise CursorReplayError(
                            "event recovery encountered a sequence gap"
                        )
                    if (
                        latest_sequence
                        and str(event.get("previous_event_id") or "")
                        != latest_event_id
                    ):
                        raise CursorReplayError(
                            "event recovery encountered a broken hash chain"
                        )
                    identities[sequence] = event_id
                    latest_sequence = sequence
                    latest_event_id = event_id
                    if not earliest_sequence:
                        earliest_sequence = sequence
                if not first_sequence:
                    first_sequence = sequence
                last_sequence = max(last_sequence, sequence)
                if (
                    not offsets
                    or (sequence - first_sequence) % _EVENT_OFFSET_INDEX_STRIDE
                    == 0
                ):
                    offsets.append([sequence, offset])
        record = {
            "path": source.name,
            "size_bytes": size,
            "event_count": physical_count,
            "sha256": digest.hexdigest(),
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "start_previous_event_id": starting_previous_event_id,
            "offset_index": offsets[-_EVENT_OFFSET_INDEX_MAX_ITEMS:],
            "canonical_events": all_canonical,
        }
        try:
            record.update(_stat_fields(source))
        except OSError:
            pass
        records.append(record)
    active_record = next(
        (item for item in records if item.get("path") == path.name),
        None,
    )
    value: dict[str, Any] = {
        "schema": EVENT_LOG_MANIFEST_SCHEMA,
        "generation": max(0, int(generation)),
        "updated_at": utc_now(),
        "active_path": path.name,
        "stream_id": selected_stream,
        "snapshot_id": selected_snapshot,
        "earliest_sequence": earliest_sequence,
        "latest_sequence": latest_sequence,
        "last_event_id": latest_event_id,
        "active_indexed_bytes": int(
            (active_record or {}).get("size_bytes", 0)
        ),
        "files": records,
    }
    value["manifest_digest"] = _event_manifest_digest(value)
    return value


def _event_manifest_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("manifest_digest", None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_event_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(
            _event_manifest_path(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema") != EVENT_LOG_MANIFEST_SCHEMA
        or value.get("manifest_digest") != _event_manifest_digest(value)
        or not str(value.get("stream_id") or "")
        or not str(value.get("snapshot_id") or "")
    ):
        return None
    return value


def _manifest_matches_metadata(path: Path, value: Mapping[str, Any]) -> bool:
    expected = {
        str(item.get("path")): item
        for item in value.get("files", ())
        if isinstance(item, Mapping)
    }
    actual_paths = _source_paths(path)
    if set(expected) != {item.name for item in actual_paths}:
        return False
    for source in actual_paths:
        record = expected[source.name]
        try:
            stat = source.stat()
        except OSError:
            return False
        if (
            int(record.get("size_bytes", -1)) != stat.st_size
            or int(record.get("device", -1)) != stat.st_dev
            or int(record.get("inode", -1)) != stat.st_ino
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
        ):
            return False
    return True


def _write_manifest_value(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        _event_manifest_path(path),
        json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )


def event_log_manifest(path: Path | str) -> dict[str, Any]:
    """Return the cheap active/archive head, rebuilding only after drift.

    A healthy v2 manifest is validated with bounded ``stat`` metadata. File
    bodies are scanned only when the manifest is absent, corrupt, or disagrees
    with the physical segments.
    """

    event_path = Path(path)
    value = _load_event_manifest(event_path)
    if value is not None and _manifest_matches_metadata(event_path, value):
        return value
    value = _scan_event_log(event_path, generation=0)
    _write_manifest_value(event_path, value)
    return value


def _write_event_manifest(
    path: Path,
    *,
    previous: Mapping[str, Any] | None = None,
    increment_generation: bool = True,
) -> dict[str, Any]:
    prior = dict(previous or _load_event_manifest(path) or {})
    value = _scan_event_log(
        path,
        generation=(
            int(prior.get("generation", 0))
            + (1 if increment_generation else 0)
        ),
        stream_id=str(prior.get("stream_id") or "") or None,
        snapshot_id=str(prior.get("snapshot_id") or "") or None,
    )
    _write_manifest_value(path, value)
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_backup_path(path: Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for index in range(1000):
        suffix = f"{label}-{stamp}" if index == 0 else f"{label}-{stamp}-{index}"
        candidate = path.with_name(f"{path.name}.{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.name}.{label}-{stamp}-overflow")


def repair_jsonl_event_log(path: Path) -> dict[str, Any]:
    """Repair event-log storage enough for later reads and appends to proceed."""

    result: dict[str, Any] = {
        "repaired": False,
        "reason": "valid",
        "path": str(path),
        "valid_count": 0,
        "invalid_count": 0,
    }
    if not path.exists():
        result["reason"] = "missing"
        return result
    if path.is_dir():
        backup_path = unique_backup_path(path, "directory-backup")
        path.rename(backup_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        result.update(
            {
                "repaired": True,
                "reason": "event_path_was_directory",
                "backup_path": str(backup_path),
            }
        )
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result.update({"reason": "unreadable", "error": str(exc)})
        return result

    valid_events: list[dict[str, Any]] = []
    invalid_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines.append(raw_line)
            continue
        if isinstance(event, dict):
            valid_events.append(event)
        else:
            invalid_lines.append(raw_line)

    result["valid_count"] = len(valid_events)
    result["invalid_count"] = len(invalid_lines)
    if not invalid_lines:
        return result

    quarantine_path = unique_backup_path(path, "invalid-jsonl")
    quarantine_path.write_text("\n".join(invalid_lines) + "\n", encoding="utf-8")
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in valid_events),
        encoding="utf-8",
    )
    result.update(
        {
            "repaired": True,
            "reason": "malformed_jsonl",
            "quarantine_path": str(quarantine_path),
        }
    )
    return result


def recover_jsonl_event_log_tail(
    path: Path | str,
    *,
    checkpoint: EventCursor | Mapping[str, Any] | str | None = None,
    max_quarantine_bytes: int = MAX_PROJECTION_BYTES,
) -> dict[str, Any]:
    """Atomically remove an invalid active-log suffix and quarantine it.

    Unlike :func:`repair_jsonl_event_log`, this recovery primitive is strict:
    the first malformed/non-object line terminates the valid prefix.  Bytes
    after that boundary are never interpreted as committed events.  Recovery
    also proves that an optional durable cursor is still represented by the
    retained hash chain before replacing the active file.

    The operation fails closed without changing the log when the corrupt
    suffix exceeds ``max_quarantine_bytes`` or the checkpoint cannot be
    proven.  Exact duplicate canonical events are tolerated because rotation
    recovery may leave them behind; conflicting identities are rejected.
    """

    if (
        isinstance(max_quarantine_bytes, bool)
        or not isinstance(max_quarantine_bytes, int)
        or max_quarantine_bytes < 1
    ):
        raise ValueError("max_quarantine_bytes must be a positive integer")
    event_path = Path(path)
    selected_checkpoint = (
        _coerce_event_cursor(checkpoint) if checkpoint is not None else None
    )
    result: dict[str, Any] = {
        "repaired": False,
        "failed_closed": False,
        "reason": "valid",
        "path": str(event_path),
        "valid_count": 0,
        "invalid_bytes": 0,
        "quarantine_path": "",
    }
    with _EventLogLock(event_path):
        if not event_path.exists():
            result["reason"] = "missing"
            return result
        if event_path.is_dir():
            result.update(
                {
                    "failed_closed": True,
                    "reason": "event_path_is_directory",
                }
            )
            return result
        try:
            payload = event_path.read_bytes()
        except OSError as exc:
            result.update(
                {
                    "failed_closed": True,
                    "reason": "event_log_unreadable",
                    "error": type(exc).__name__,
                }
            )
            return result

        retained = bytearray()
        invalid = b""
        valid_events: list[dict[str, Any]] = []
        offset = 0
        for raw_line in payload.splitlines(keepends=True):
            next_offset = offset + len(raw_line)
            if not raw_line.strip():
                retained.extend(raw_line)
                offset = next_offset
                continue
            # A final line without a newline has not crossed the append
            # durability boundary and is treated as a partial write.
            complete = raw_line.endswith((b"\n", b"\r"))
            try:
                value = json.loads(raw_line) if complete else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            if not complete or not isinstance(value, dict):
                invalid = payload[offset:]
                break
            retained.extend(raw_line)
            valid_events.append(value)
            offset = next_offset

        result["valid_count"] = len(valid_events)
        result["invalid_bytes"] = len(invalid)
        if len(invalid) > max_quarantine_bytes:
            result.update(
                {
                    "failed_closed": True,
                    "reason": "quarantine_bound_exceeded",
                }
            )
            return result

        # Rebuild/validate the logical chain using the retained bytes in a
        # sibling candidate.  This reuses the same canonical validation as
        # normal replay without exposing a partially repaired active path.
        candidate: Path | None = None
        try:
            descriptor, candidate_name = tempfile.mkstemp(
                prefix=f".{event_path.name}.recovery-",
                dir=event_path.parent,
            )
            candidate = Path(candidate_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(retained)
                stream.flush()
                os.fsync(stream.fileno())
            prior_manifest = _load_event_manifest(event_path) or {}
            candidate_manifest = _scan_event_log(
                candidate,
                generation=int(prior_manifest.get("generation") or 0) + 1,
                stream_id=str(prior_manifest.get("stream_id") or "") or None,
                snapshot_id=str(prior_manifest.get("snapshot_id") or "") or None,
            )
            if selected_checkpoint is not None:
                selected_stream = str(candidate_manifest["stream_id"])
                selected_snapshot = str(candidate_manifest["snapshot_id"])
                if (
                    selected_checkpoint.stream_id != selected_stream
                    or selected_checkpoint.snapshot_id != selected_snapshot
                ):
                    result.update(
                        {
                            "failed_closed": True,
                            "reason": "checkpoint_not_in_retained_log",
                        }
                    )
                    return result
                if selected_checkpoint.position:
                    identities: dict[int, str] = {}
                    # The checkpoint anchor may live in a rotated segment.
                    # Read only canonical identities; _scan_event_log below
                    # remains responsible for complete chain validation.
                    sources = [
                        source
                        for source in _source_paths(event_path)
                        if source != event_path
                    ]
                    source_events: list[dict[str, Any]] = []
                    for source in sources:
                        source_events.extend(read_jsonl_events(source))
                    source_events.extend(valid_events)
                    for value in source_events:
                        sequence = value.get(
                            "sequence", value.get("position")
                        )
                        if (
                            isinstance(sequence, int)
                            and not isinstance(sequence, bool)
                            and str(value.get("stream_id") or "")
                            == selected_stream
                            and str(value.get("snapshot_id") or "")
                            == selected_snapshot
                        ):
                            event_id = str(value.get("event_id") or "")
                            if event_id and event_id == _event_identity(value):
                                identities[int(sequence)] = event_id
                    if (
                        identities.get(selected_checkpoint.position)
                        != selected_checkpoint.last_event_id
                    ):
                        result.update(
                            {
                                "failed_closed": True,
                                "reason": "checkpoint_anchor_mismatch",
                            }
                        )
                        return result

            if invalid:
                quarantine_path = unique_backup_path(
                    event_path, "partial-tail"
                )
                _atomic_write_bytes(quarantine_path, invalid)
                _atomic_write_bytes(event_path, bytes(retained))
                result.update(
                    {
                        "repaired": True,
                        "reason": "partial_tail_quarantined",
                        "quarantine_path": str(quarantine_path),
                    }
                )
            _write_event_manifest(
                event_path,
                previous=prior_manifest,
                increment_generation=True,
            )
        except (CursorReplayError, OSError, ValueError) as exc:
            result.update(
                {
                    "failed_closed": True,
                    "reason": "event_chain_not_recoverable",
                    "error": type(exc).__name__,
                }
            )
            return result
        finally:
            if candidate is not None:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
    return result


def read_jsonl_events(path: Path, *, repair: bool = False) -> list[dict[str, Any]]:
    if repair:
        repair_jsonl_event_log(path)
    if not path.exists() or path.is_dir():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def event_log_sources(
    paths: Iterable[Path | str],
    *,
    include_rotated: bool = True,
) -> list[Path]:
    """Resolve active and rotated JSONL logs in deterministic archive order.

    Rotation archives are part of the lifecycle history.  Metrics readers need
    them to avoid resetting counters whenever the active log is compacted.
    Missing paths are retained only conceptually and therefore produce no
    source entry.
    """

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidates: list[Path] = []
        if include_rotated and path.parent.exists():
            candidates.extend(sorted(path.parent.glob(f"{path.name}.rotated-*")))
        candidates.append(path)
        for candidate in candidates:
            try:
                key = candidate.resolve()
            except OSError:
                key = candidate.absolute()
            if key in seen or not candidate.exists() or candidate.is_dir():
                continue
            seen.add(key)
            resolved.append(candidate)
    return resolved


def read_jsonl_event_sources(
    paths: Iterable[Path | str],
    *,
    repair: bool = False,
    include_rotated: bool = True,
) -> list[dict[str, Any]]:
    """Read and timestamp-order events from multiple supervisor logs.

    File order is used as a stable tie breaker.  Invalid or missing timestamps
    sort after timestamped events while preserving their source order.
    """

    indexed: list[tuple[int, dict[str, Any]]] = []
    seen_canonical: dict[
        tuple[str, str, int],
        str,
    ] = {}
    index = 0
    for source in event_log_sources(paths, include_rotated=include_rotated):
        source_repair = repair and ".rotated-" not in source.name
        for event in read_jsonl_events(source, repair=source_repair):
            sequence = event.get("sequence", event.get("position"))
            stream_id = str(event.get("stream_id") or "")
            snapshot_id = str(event.get("snapshot_id") or "")
            event_id = str(event.get("event_id") or "")
            canonical = (
                stream_id
                and snapshot_id
                and isinstance(sequence, int)
                and not isinstance(sequence, bool)
                and sequence > 0
                and event_id
            )
            if canonical:
                identity = (stream_id, snapshot_id, int(sequence))
                known = seen_canonical.get(identity)
                if known == event_id:
                    continue
                if known is not None and known != event_id:
                    # The compatibility reader remains best-effort. Strict
                    # cursor replay below fails closed on the same conflict.
                    continue
                seen_canonical[identity] = event_id
            indexed.append((index, event))
            index += 1

    def timestamp_key(item: tuple[int, dict[str, Any]]) -> tuple[int, str, int]:
        position, event = item
        timestamp = str(event.get("timestamp") or event.get("occurred_at") or "")
        return (0 if timestamp else 1, timestamp, position)

    indexed.sort(key=timestamp_key)
    return [event for _index, event in indexed]


def initial_event_cursor(path: Path | str) -> EventCursor:
    """Return the canonical position before the first event in ``path``."""

    event_path = Path(path)
    with _EventLogLock(event_path):
        manifest = _manifest_for_append(event_path)
    return EventCursor.initial(
        str(manifest["stream_id"]),
        snapshot_id=str(manifest["snapshot_id"]),
    )


event_log_initial_cursor = initial_event_cursor


def latest_event_cursor(path: Path | str) -> EventCursor:
    """Return a cursor bound to the exact durable event-log head."""

    event_path = Path(path)
    with _EventLogLock(event_path):
        manifest = _manifest_for_append(event_path)
    position = int(manifest.get("latest_sequence") or 0)
    if position == 0:
        return EventCursor.initial(
            str(manifest["stream_id"]),
            snapshot_id=str(manifest["snapshot_id"]),
        )
    return EventCursor(
        stream_id=str(manifest["stream_id"]),
        snapshot_id=str(manifest["snapshot_id"]),
        position=position,
        last_event_id=str(manifest["last_event_id"]),
    )


event_log_latest_cursor = latest_event_cursor


def _coerce_event_cursor(
    cursor: EventCursor | Mapping[str, Any] | str,
) -> EventCursor:
    if isinstance(cursor, EventCursor):
        return cursor
    if isinstance(cursor, str):
        return EventCursor.from_token(cursor)
    if isinstance(cursor, Mapping):
        return EventCursor.from_dict(cursor)
    raise EventCursorError(
        "cursor must be an EventCursor, canonical cursor record, or token"
    )


def _record_for_source(
    manifest: Mapping[str, Any],
    source: Path,
) -> Mapping[str, Any]:
    for item in manifest.get("files", ()):
        if isinstance(item, Mapping) and str(item.get("path")) == source.name:
            return item
    return {}


def _segment_seek_offset(
    record: Mapping[str, Any],
    wanted_sequence: int,
) -> tuple[int, int]:
    if not bool(record.get("canonical_events", False)):
        return 0, int(record.get("first_sequence") or 1)
    selected_offset = 0
    selected_sequence = int(record.get("first_sequence") or 1)
    for item in record.get("offset_index", ()):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 2
        ):
            continue
        try:
            sequence = int(item[0])
            offset = int(item[1])
        except (TypeError, ValueError):
            continue
        if sequence <= wanted_sequence and sequence >= selected_sequence:
            selected_sequence = sequence
            selected_offset = max(0, offset)
    return selected_offset, selected_sequence


def _read_segment_page_events(
    source: Path,
    record: Mapping[str, Any],
    *,
    stream_id: str,
    snapshot_id: str,
    wanted_sequence: int,
    maximum_events: int,
) -> list[dict[str, Any]]:
    """Read a bounded logical suffix from one segment.

    The sparse manifest offset avoids walking the segment prefix during
    steady-state replay. Legacy records are enriched in memory only.
    """

    if maximum_events <= 0:
        return []
    offset, inferred_sequence = _segment_seek_offset(record, wanted_sequence)
    previous_event_id = str(record.get("start_previous_event_id") or "")
    events: list[dict[str, Any]] = []
    with source.open("rb") as stream:
        stream.seek(offset)
        while len(events) < maximum_events:
            raw_line = stream.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                raw_event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CursorReplayError(
                    f"event segment {source.name!r} contains malformed JSON"
                ) from exc
            if not isinstance(raw_event, dict):
                raise CursorReplayError(
                    f"event segment {source.name!r} contains a non-object event"
                )
            raw_sequence = raw_event.get(
                "sequence", raw_event.get("position")
            )
            canonical = (
                isinstance(raw_sequence, int)
                and not isinstance(raw_sequence, bool)
                and raw_sequence > 0
                and str(raw_event.get("stream_id") or "") == stream_id
                and str(raw_event.get("snapshot_id") or "") == snapshot_id
            )
            if canonical:
                sequence = int(raw_sequence)
            else:
                sequence = inferred_sequence
                inferred_sequence += 1
            event = dict(raw_event)
            if not canonical:
                event.update(
                    {
                        "stream_id": stream_id,
                        "snapshot_id": snapshot_id,
                        "sequence": sequence,
                        "previous_event_id": previous_event_id,
                    }
                )
            event_id = str(event.get("event_id") or "")
            expected_event_id = _event_identity(event)
            if event_id and event_id != expected_event_id:
                raise CursorReplayError(
                    f"event {sequence} has a non-canonical event_id"
                )
            event["event_id"] = event_id or expected_event_id
            previous_event_id = event["event_id"]
            if sequence >= wanted_sequence:
                events.append(event)
    return events


def read_jsonl_event_page(
    path: Path | str,
    cursor: EventCursor | Mapping[str, Any] | str,
    *,
    limit: int = 256,
) -> EventPage:
    """Replay at most ``limit`` canonical events strictly after ``cursor``.

    Segment ranges and sparse byte offsets are consulted before bodies are
    opened. Exact physical duplicates are coalesced; gaps, foreign cursors,
    and conflicting identities fail closed.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    event_path = Path(path)
    manifest = event_log_manifest(event_path)
    selected_cursor = _coerce_event_cursor(cursor)
    stream_id = str(manifest["stream_id"])
    snapshot_id = str(manifest["snapshot_id"])
    selected_cursor.assert_replayable(
        stream_id=stream_id,
        earliest_position=int(manifest.get("earliest_sequence") or 0),
        latest_position=int(manifest.get("latest_sequence") or 0),
        snapshot_id=snapshot_id,
    )
    wanted_sequence = max(1, selected_cursor.position)
    population: dict[int, dict[str, Any]] = {}
    target_population = limit + 2
    for source in _source_paths(event_path):
        record = _record_for_source(manifest, source)
        first_sequence = int(record.get("first_sequence") or 0)
        last_sequence = int(record.get("last_sequence") or 0)
        if not first_sequence or last_sequence < wanted_sequence:
            continue
        segment_events = _read_segment_page_events(
            source,
            record,
            stream_id=stream_id,
            snapshot_id=snapshot_id,
            wanted_sequence=wanted_sequence,
            maximum_events=target_population,
        )
        for event in segment_events:
            sequence = int(event["sequence"])
            known = population.get(sequence)
            if known is not None:
                if known["event_id"] != event["event_id"]:
                    raise CursorReplayError(
                        f"event sequence {sequence} has conflicting identities"
                    )
                continue
            population[sequence] = event
        if len(population) >= target_population:
            break
    ordered = [population[key] for key in sorted(population)]
    page = replay_event_page(
        ordered,
        selected_cursor,
        limit=limit,
        stream_id=stream_id,
        snapshot_id=snapshot_id,
    )
    manifest_has_more = (
        page.next_cursor.position
        < int(manifest.get("latest_sequence") or 0)
    )
    if manifest_has_more == page.has_more:
        return page
    return EventPage(
        events=page.events,
        next_cursor=page.next_cursor,
        has_more=manifest_has_more,
    )


read_event_page = read_jsonl_event_page
read_event_log_page = read_jsonl_event_page


def write_event_cursor_checkpoint(
    path: Path | str,
    cursor: EventCursor | Mapping[str, Any] | str,
) -> bool:
    """Atomically persist a canonical cursor, returning ``False`` on no-op."""

    checkpoint_path = Path(path)
    selected = _coerce_event_cursor(cursor)
    value: dict[str, Any] = {
        "schema": EVENT_CURSOR_CHECKPOINT_SCHEMA,
        "cursor": selected.to_record(),
    }
    value["checkpoint_digest"] = _canonical_identity(value)
    payload = (
        json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    with _EventLogLock(checkpoint_path):
        try:
            if checkpoint_path.read_bytes() == payload:
                return False
        except OSError:
            pass
        _atomic_write_bytes(checkpoint_path, payload)
        return True


def read_event_cursor_checkpoint(
    path: Path | str,
    *,
    stream_id: str = "",
    snapshot_id: str = "",
) -> EventCursor | None:
    """Load and validate a durable canonical cursor checkpoint."""

    checkpoint_path = Path(path)
    try:
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventCursorError("event cursor checkpoint is malformed") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != EVENT_CURSOR_CHECKPOINT_SCHEMA
        or value.get("checkpoint_digest")
        != _canonical_identity(
            {
                key: item
                for key, item in value.items()
                if key != "checkpoint_digest"
            }
        )
        or not isinstance(value.get("cursor"), Mapping)
    ):
        raise EventCursorError("event cursor checkpoint is malformed")
    cursor = EventCursor.from_dict(value["cursor"])
    if stream_id and cursor.stream_id != stream_id:
        raise CursorReplayError(
            "event cursor checkpoint belongs to a different stream"
        )
    if snapshot_id and cursor.snapshot_id != snapshot_id:
        raise CursorReplayError(
            "event cursor checkpoint belongs to a different snapshot"
        )
    return cursor


persist_event_cursor = write_event_cursor_checkpoint
load_event_cursor = read_event_cursor_checkpoint


def _manifest_for_append(path: Path) -> dict[str, Any]:
    manifest = _load_event_manifest(path)
    if manifest is not None and _manifest_matches_metadata(path, manifest):
        return manifest
    if manifest is not None:
        reconciled = _reconcile_manifest_tail(path, manifest)
        if reconciled is not None:
            return reconciled
    manifest = _scan_event_log(
        path,
        generation=int((manifest or {}).get("generation", 0)),
        stream_id=str((manifest or {}).get("stream_id") or "") or None,
        snapshot_id=str((manifest or {}).get("snapshot_id") or "") or None,
    )
    try:
        _write_manifest_value(path, manifest)
    except OSError:
        # A manifest is an acceleration structure. The fsynced event stream
        # remains authoritative and will be rebuilt on the next append.
        pass
    return manifest


def _manifest_after_append(
    path: Path,
    manifest: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    offset: int,
) -> dict[str, Any]:
    value = dict(manifest)
    records = [
        dict(item)
        for item in manifest.get("files", ())
        if isinstance(item, Mapping)
    ]
    active = next(
        (item for item in records if item.get("path") == path.name),
        None,
    )
    if active is None:
        active = {
            "path": path.name,
            "size_bytes": 0,
            "event_count": 0,
            "sha256": "",
            "first_sequence": 0,
            "last_sequence": 0,
            "start_previous_event_id": str(
                event.get("previous_event_id") or ""
            ),
            "offset_index": [],
            "canonical_events": True,
        }
        records.append(active)
    sequence = int(event["sequence"])
    count = int(active.get("event_count") or 0) + 1
    first_sequence = int(active.get("first_sequence") or sequence)
    offsets = [
        list(item)
        for item in active.get("offset_index", ())
        if isinstance(item, Sequence)
        and not isinstance(item, (str, bytes, bytearray))
        and len(item) == 2
    ]
    if (
        not offsets
        or (sequence - first_sequence) % _EVENT_OFFSET_INDEX_STRIDE == 0
    ):
        offsets.append([sequence, offset])
    stat = path.stat()
    active.update(
        {
            "size_bytes": int(stat.st_size),
            "event_count": count,
            # The hash chain is authoritative for the mutable active segment.
            # A sealed archive receives a physical sha256 during rotation.
            "sha256": "",
            "first_sequence": first_sequence,
            "last_sequence": sequence,
            "offset_index": offsets[-_EVENT_OFFSET_INDEX_MAX_ITEMS:],
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    value.update(
        {
            "updated_at": utc_now(),
            "earliest_sequence": int(
                value.get("earliest_sequence") or sequence
            ),
            "latest_sequence": sequence,
            "last_event_id": str(event["event_id"]),
            "active_indexed_bytes": int(stat.st_size),
            "files": records,
        }
    )
    value["manifest_digest"] = _event_manifest_digest(value)
    return value


def _reconcile_manifest_tail(
    path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reconcile a bounded unindexed active tail after a metadata-write crash."""

    expected_records = {
        str(item.get("path")): item
        for item in manifest.get("files", ())
        if isinstance(item, Mapping)
    }
    sources = _source_paths(path)
    if set(expected_records) != {item.name for item in sources}:
        return None
    for source in sources:
        if source == path:
            continue
        record = expected_records[source.name]
        try:
            stat = source.stat()
        except OSError:
            return None
        if (
            int(record.get("size_bytes", -1)) != stat.st_size
            or int(record.get("device", -1)) != stat.st_dev
            or int(record.get("inode", -1)) != stat.st_ino
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
        ):
            return None
    active = expected_records.get(path.name)
    if active is None or not path.exists():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    indexed = int(manifest.get("active_indexed_bytes") or 0)
    if (
        int(active.get("device", -1)) != stat.st_dev
        or int(active.get("inode", -1)) != stat.st_ino
        or stat.st_size < indexed
        or stat.st_size - indexed > _EVENT_RECOVERY_TAIL_MAX_BYTES
    ):
        return None
    value = dict(manifest)
    if stat.st_size == indexed:
        records = [
            dict(item)
            for item in manifest.get("files", ())
            if isinstance(item, Mapping)
        ]
        for record in records:
            if record.get("path") == path.name:
                record.update(_stat_fields(path))
        value["files"] = records
        value["manifest_digest"] = _event_manifest_digest(value)
        return value
    try:
        with path.open("rb") as stream:
            stream.seek(indexed)
            while stream.tell() < stat.st_size:
                offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line.endswith(b"\n"):
                    return None
                raw_event = json.loads(raw_line)
                if not isinstance(raw_event, dict):
                    return None
                expected_sequence = int(value.get("latest_sequence") or 0) + 1
                if (
                    raw_event.get("sequence") != expected_sequence
                    or str(raw_event.get("stream_id") or "")
                    != str(value.get("stream_id") or "")
                    or str(raw_event.get("snapshot_id") or "")
                    != str(value.get("snapshot_id") or "")
                    or str(raw_event.get("previous_event_id") or "")
                    != str(value.get("last_event_id") or "")
                    or str(raw_event.get("event_id") or "")
                    != _event_identity(raw_event)
                ):
                    return None
                value = _manifest_after_append(
                    path,
                    value,
                    raw_event,
                    offset=offset,
                )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value


def append_jsonl_event(
    path: Path | str,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    max_bytes: int | None = None,
    fsync: bool = True,
    artifact_store: Any | None = None,
) -> dict[str, Any]:
    """Append one event and return the exact JSON object written.

    Returning the object is backward compatible with callers which ignored
    the former ``None`` return and lets receipt publishers reuse the exact
    compact projection which reached the durable log. Receipt-shaped events
    have a hard 256 KiB ceiling; every other routine event has a 1 MiB ceiling.
    """

    path = Path(path)
    if path.exists() and path.is_dir():
        repair_jsonl_event_log(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_payload: Mapping[str, Any] = payload
    if artifact_store is not None:
        projector = getattr(artifact_store, "project_payload", None)
        if not callable(projector):
            raise ValueError("artifact_store must provide project_payload")
        projected, _references = projector(payload)
        if not isinstance(projected, Mapping):
            raise ValueError("event projection must be an object")
        compact_payload = projected
    supplied_payload = dict(compact_payload)
    for field_name in _RESERVED_EVENT_FIELDS:
        supplied_payload.pop(field_name, None)
    selected_timestamp = supplied_payload.pop("timestamp", None)
    supplied_payload.pop("type", None)
    default_limit = (
        MAX_RECEIPT_BYTES
        if (
            "receipt" in str(event_type).casefold()
            or any("receipt" in str(name).casefold() for name in payload)
        )
        else MAX_PROJECTION_BYTES
    )
    if max_bytes is not None and (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise ValueError("max_bytes must be a positive integer or None")
    limit = min(max_bytes or default_limit, default_limit)
    event: dict[str, Any]
    with _EventLogLock(path):
        manifest = _manifest_for_append(path)
        previous_sequence = int(manifest.get("latest_sequence") or 0)
        previous_event_id = str(manifest.get("last_event_id") or "")
        event = {
            "type": event_type,
            "timestamp": (
                selected_timestamp
                if selected_timestamp is not None
                else utc_now()
            ),
            **supplied_payload,
            "stream_id": str(manifest["stream_id"]),
            "snapshot_id": str(manifest["snapshot_id"]),
            "sequence": previous_sequence + 1,
            "previous_event_id": previous_event_id,
        }
        event["event_id"] = _event_identity(event)
        encoded = _canonical_event_bytes(event, limit) + b"\n"
        offset = path.stat().st_size if path.exists() else 0
        with path.open("ab") as fh:
            fh.write(encoded)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        updated_manifest = _manifest_after_append(
            path,
            manifest,
            event,
            offset=offset,
        )
        try:
            _write_manifest_value(path, updated_manifest)
        except OSError:
            # The line has crossed its durability boundary. Do not report the
            # append as failed merely because its acceleration metadata could
            # not be refreshed; the next locked append reconciles the stream.
            pass
    # Auto-rotate if the log exceeds the size threshold
    try:
        rotate_event_log_if_needed(path)
    except (OSError, ValueError):
        # Rotation is post-commit maintenance. The acknowledged append remains
        # available in the active file and the next pass may rotate it.
        pass
    return event


def semantic_change_from_event(value: Mapping[str, Any]) -> SemanticChange:
    """Decode and verify a semantic change from its durable event envelope."""

    if not isinstance(value, Mapping):
        raise SemanticChangeIntegrityError(
            "semantic change event must be an object"
        )
    if str(value.get("type") or "") != SEMANTIC_CHANGE_EVENT_TYPE:
        raise SemanticChangeIntegrityError(
            "event is not a semantic change event"
        )
    payload = value.get("change")
    if not isinstance(payload, Mapping):
        # Compatibility with an early flat projection. Reserved physical
        # fields are excluded so the logical identity remains exact.
        payload = {
            key: item
            for key, item in value.items()
            if key
            not in {
                "type",
                "timestamp",
                "stream_id",
                "snapshot_id",
                "sequence",
                "position",
                "event_id",
                "previous_event_id",
            }
        }
    change = SemanticChange.from_dict(payload)
    claimed = str(value.get("change_id") or "")
    if claimed and claimed != change.change_id:
        raise SemanticChangeIntegrityError(
            "semantic change event has conflicting identities"
        )
    return change


def append_semantic_change_event(
    path: Path | str,
    change: SemanticChange | Mapping[str, Any],
    *,
    fsync: bool = True,
) -> dict[str, Any]:
    """Append one canonical semantic transition to the supervisor event log."""

    selected = (
        change
        if isinstance(change, SemanticChange)
        else SemanticChange.from_dict(change)
    )
    return append_jsonl_event(
        path,
        SEMANTIC_CHANGE_EVENT_TYPE,
        {
            "change_id": selected.change_id,
            "change": selected.to_dict(),
        },
        fsync=fsync,
    )


def read_semantic_change_page(
    path: Path | str,
    cursor: EventCursor | Mapping[str, Any] | str,
    *,
    limit: int = 256,
    known_change_ids: Iterable[str] = (),
    expected_roots: Mapping[str, str] | None = None,
) -> SemanticChangePage:
    """Replay a strictly ordered, idempotent page of logical root changes.

    Non-semantic events advance the returned physical cursor but are omitted
    from ``changes``. A repeated logical identity at a new physical position,
    a transition whose previous root does not match the replay state, or an
    event with a forged logical identity fails closed.
    """

    page = read_jsonl_event_page(path, cursor, limit=limit)
    seen = {str(item) for item in known_change_ids}
    roots = {
        str(key): str(item)
        for key, item in (expected_roots or {}).items()
    }
    changes: list[SemanticChange] = []
    event_ids: list[str] = []
    for event in page.events:
        if str(event.get("type") or "") != SEMANTIC_CHANGE_EVENT_TYPE:
            continue
        change = semantic_change_from_event(event)
        if change.change_id in seen:
            raise SemanticChangeIntegrityError(
                f"duplicate semantic change {change.change_id}"
            )
        seen.add(change.change_id)
        current = roots.get(change.subject_id)
        if (
            current is not None
            and change.previous_root_id != current
        ):
            raise SemanticChangeIntegrityError(
                "semantic changes are missing or reordered for "
                f"{change.subject_id!r}"
            )
        roots[change.subject_id] = change.current_root_id
        changes.append(change)
        event_ids.append(str(event.get("event_id") or ""))
    return SemanticChangePage(
        changes=tuple(changes),
        next_cursor=page.next_cursor,
        has_more=page.has_more,
        event_ids=tuple(event_ids),
    )


# Descriptive compatibility spellings for integration callers.
CanonicalSemanticChange = SemanticChange
CanonicalSemanticChangeEvent = SemanticChange
append_canonical_semantic_change = append_semantic_change_event
read_canonical_semantic_change_page = read_semantic_change_page


def append_scan_receipt_event(
    event_path: Path | str,
    result: Any,
    artifact_dir: Path | str,
    *,
    scan_kind: str,
    relative_to: Path | str | None = None,
) -> dict[str, Any]:
    """Persist one full scan receipt and append its compact event projection.

    Every invocation emits exactly one ``refill_scan_receipt`` event after the
    content-addressed artifact is durable.  No generated item, per-file path
    list, parser exception list, or arbitrary receipt metadata is copied into
    the event log.
    """

    # Local import avoids making the general-purpose event-log reader import
    # scan/git machinery at startup.
    from ..objectives.scan_receipts import persist_scan_receipt

    projection = persist_scan_receipt(
        result,
        artifact_dir,
        scan_kind=scan_kind,
        relative_to=relative_to,
    )
    append_jsonl_event(Path(event_path), "refill_scan_receipt", projection)
    return projection


def rotate_event_log_if_needed(
    path: Path | str,
    *,
    max_bytes: int | None = None,
    retain_recent: int | None = None,
    max_archives: int | None = None,
) -> dict[str, Any]:
    """Rotate the event log when it exceeds the configured size threshold.

    The input is streamed through a fixed-size deque, so compaction memory is
    proportional to the retained tail instead of the complete log. Archive,
    active-tail, and manifest files are individually fsynced and atomically
    installed. A crash can therefore cause a recoverable duplicate generation,
    but never an acknowledged event to disappear from both files.
    """
    path = Path(path)
    selected_max_bytes = (
        int(
            os.environ.get(
                _EVENT_LOG_MAX_BYTES_ENV, str(_DEFAULT_EVENT_LOG_MAX_BYTES)
            )
        )
        if max_bytes is None
        else max_bytes
    )
    selected_retain_recent = (
        int(
            os.environ.get(
                _EVENT_LOG_RETAIN_RECENT_ENV,
                str(_DEFAULT_EVENT_LOG_RETAIN_RECENT),
            )
        )
        if retain_recent is None
        else retain_recent
    )
    selected_max_archives = (
        int(
            os.environ.get(
                _EVENT_LOG_MAX_ARCHIVES_ENV,
                str(_DEFAULT_EVENT_LOG_MAX_ARCHIVES),
            )
        )
        if max_archives is None
        else max_archives
    )
    if selected_max_bytes <= 0:
        return {"rotated": False, "reason": "rotation_disabled"}
    if selected_retain_recent < 1:
        return {"rotated": False, "reason": "invalid_retain_recent"}
    if selected_max_archives < 1:
        return {"rotated": False, "reason": "invalid_max_archives"}
    if not path.exists():
        return {"rotated": False, "reason": "missing"}

    with _EventLogLock(path):
        try:
            file_size = path.stat().st_size
        except OSError:
            return {"rotated": False, "reason": "stat_failed"}
        if file_size < selected_max_bytes:
            return {
                "rotated": False,
                "reason": "under_threshold",
                "size": file_size,
            }

        archive_path = unique_backup_path(path, "rotated")
        archive_descriptor, archive_temporary = tempfile.mkstemp(
            prefix=f".{archive_path.name}.", dir=path.parent
        )
        retained: deque[bytes] = deque(maxlen=selected_retain_recent)
        total_count = 0
        archived_count = 0
        try:
            with os.fdopen(archive_descriptor, "wb") as archive_stream:
                try:
                    source = path.open("rb")
                except OSError as exc:
                    return {
                        "rotated": False,
                        "reason": "read_failed",
                        "error": str(exc),
                    }
                with source:
                    for raw_line in source:
                        if not raw_line.strip():
                            continue
                        total_count += 1
                        if len(retained) == selected_retain_recent:
                            archive_stream.write(retained.popleft())
                            archived_count += 1
                        retained.append(
                            raw_line
                            if raw_line.endswith(b"\n")
                            else raw_line + b"\n"
                        )
                archive_stream.flush()
                os.fsync(archive_stream.fileno())
            if total_count <= selected_retain_recent:
                return {
                    "rotated": False,
                    "reason": "too_few_events",
                    "count": total_count,
                }

            retained_payload = b"".join(retained)
            os.replace(archive_temporary, archive_path)
            archive_temporary = ""
            _atomic_write_bytes(path, retained_payload)

            removed_archives: list[str] = []
            archives = sorted(path.parent.glob(f"{path.name}.rotated-*"))
            while len(archives) > selected_max_archives:
                expired_archive = archives.pop(0)
                try:
                    expired_archive.unlink()
                except OSError:
                    break
                removed_archives.append(str(expired_archive))
            manifest = _write_event_manifest(path)
            return {
                "rotated": True,
                "archived_count": archived_count,
                "retained_count": len(retained),
                "archive_path": str(archive_path),
                "previous_size": file_size,
                "removed_archives": removed_archives,
                "manifest_generation": manifest["generation"],
            }
        except OSError as exc:
            return {
                "rotated": False,
                "reason": "write_failed",
                "error": str(exc),
            }
        finally:
            if archive_temporary:
                try:
                    os.unlink(archive_temporary)
                except FileNotFoundError:
                    pass


compact_event_log = rotate_event_log_if_needed
incremental_compact_event_log = rotate_event_log_if_needed
