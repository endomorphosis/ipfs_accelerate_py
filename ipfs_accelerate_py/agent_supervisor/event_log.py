"""JSONL event-log helpers for agent supervisor runtimes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .supervisor_v2_contracts import MAX_PROJECTION_BYTES, MAX_RECEIPT_BYTES


# Event log rotation: archive when file exceeds this size (default 50MB)
_EVENT_LOG_MAX_BYTES_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_MAX_BYTES"
_DEFAULT_EVENT_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB

# Keep only the most recent N events after rotation
_EVENT_LOG_RETAIN_RECENT_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_RETAIN_RECENT"
_DEFAULT_EVENT_LOG_RETAIN_RECENT = 500
_EVENT_LOG_MAX_ARCHIVES_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_MAX_ARCHIVES"
_DEFAULT_EVENT_LOG_MAX_ARCHIVES = 8

EVENT_LOG_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.event-log-manifest@1"
)


class EventPayloadTooLarge(ValueError):
    """An event exceeded its receipt or routine projection bound."""


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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+b")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

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


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            if line.strip():
                count += 1
    return {
        "path": path.name,
        "size_bytes": size,
        "event_count": count,
        "sha256": digest.hexdigest(),
    }


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


def event_log_manifest(path: Path | str) -> dict[str, Any]:
    """Return or reconstruct the crash-safe active/archive manifest."""

    event_path = Path(path)
    manifest_path = _event_manifest_path(event_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        value = None
    valid_manifest = (
        isinstance(value, dict)
        and value.get("schema") == EVENT_LOG_MANIFEST_SCHEMA
        and value.get("manifest_digest") == _event_manifest_digest(value)
    )
    if valid_manifest:
        expected = {
            str(item.get("path")): item
            for item in value.get("files", ())
            if isinstance(item, Mapping)
        }
        actual_paths = []
        if event_path.parent.exists():
            actual_paths.extend(
                sorted(event_path.parent.glob(f"{event_path.name}.rotated-*"))
            )
        if event_path.exists() and not event_path.is_dir():
            actual_paths.append(event_path)
        actual: dict[str, dict[str, Any]] = {}
        try:
            actual = {
                item.name: _file_record(item)
                for item in actual_paths
            }
        except OSError:
            valid_manifest = False
        else:
            valid_manifest = expected == actual
    if valid_manifest:
        return value
    sources = []
    if event_path.parent.exists():
        sources.extend(sorted(event_path.parent.glob(f"{event_path.name}.rotated-*")))
    if event_path.exists() and not event_path.is_dir():
        sources.append(event_path)
    records: list[dict[str, Any]] = []
    for source in sources:
        try:
            records.append(_file_record(source))
        except OSError:
            continue
    value = {
        "schema": EVENT_LOG_MANIFEST_SCHEMA,
        "generation": 0,
        "updated_at": utc_now(),
        "active_path": event_path.name,
        "files": records,
    }
    value["manifest_digest"] = _event_manifest_digest(value)
    _atomic_write_bytes(
        manifest_path,
        json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    return value


def _write_event_manifest(path: Path) -> dict[str, Any]:
    previous = event_log_manifest(path)
    sources = sorted(path.parent.glob(f"{path.name}.rotated-*"))
    if path.exists() and not path.is_dir():
        sources.append(path)
    records = [_file_record(source) for source in sources]
    value = {
        "schema": EVENT_LOG_MANIFEST_SCHEMA,
        "generation": int(previous.get("generation", 0)) + 1,
        "updated_at": utc_now(),
        "active_path": path.name,
        "files": records,
    }
    value["manifest_digest"] = _event_manifest_digest(value)
    _atomic_write_bytes(
        _event_manifest_path(path),
        json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
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
    index = 0
    for source in event_log_sources(paths, include_rotated=include_rotated):
        source_repair = repair and ".rotated-" not in source.name
        for event in read_jsonl_events(source, repair=source_repair):
            indexed.append((index, event))
            index += 1

    def timestamp_key(item: tuple[int, dict[str, Any]]) -> tuple[int, str, int]:
        position, event = item
        timestamp = str(event.get("timestamp") or event.get("occurred_at") or "")
        return (0 if timestamp else 1, timestamp, position)

    indexed.sort(key=timestamp_key)
    return [event for _index, event in indexed]


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
    event = {
        "type": event_type,
        "timestamp": utc_now(),
        **dict(compact_payload),
    }
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
    encoded = _canonical_event_bytes(event, limit) + b"\n"
    with _EventLogLock(path):
        with path.open("ab") as fh:
            fh.write(encoded)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        if _event_manifest_path(path).exists():
            _write_event_manifest(path)
    # Auto-rotate if the log exceeds the size threshold
    rotate_event_log_if_needed(path)
    return event


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
    from .scan_receipts import persist_scan_receipt

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
