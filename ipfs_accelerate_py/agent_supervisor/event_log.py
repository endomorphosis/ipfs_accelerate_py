"""JSONL event-log helpers for agent supervisor runtimes."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# Event log rotation: archive when file exceeds this size (default 50MB)
_EVENT_LOG_MAX_BYTES_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_MAX_BYTES"
_DEFAULT_EVENT_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB

# Keep only the most recent N events after rotation
_EVENT_LOG_RETAIN_RECENT_ENV = "IPFS_ACCELERATE_AGENT_EVENT_LOG_RETAIN_RECENT"
_DEFAULT_EVENT_LOG_RETAIN_RECENT = 500


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


def append_jsonl_event(path: Path, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append one event and return the exact JSON object written.

    Returning the object is backward compatible with callers which ignored
    the former ``None`` return and lets receipt publishers reuse the exact
    compact projection which reached the durable log.
    """

    if path.exists() and path.is_dir():
        repair_jsonl_event_log(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"type": event_type, "timestamp": utc_now(), **dict(payload)}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
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


def rotate_event_log_if_needed(path: Path) -> dict[str, Any]:
    """Rotate the event log when it exceeds the configured size threshold.

    Archives old events and retains only the most recent N events in the
    active log file. This prevents unbounded growth that degrades performance
    over days of continuous operation.
    """
    max_bytes = int(os.environ.get(_EVENT_LOG_MAX_BYTES_ENV, str(_DEFAULT_EVENT_LOG_MAX_BYTES)))
    retain_recent = int(os.environ.get(_EVENT_LOG_RETAIN_RECENT_ENV, str(_DEFAULT_EVENT_LOG_RETAIN_RECENT)))

    if max_bytes <= 0:
        return {"rotated": False, "reason": "rotation_disabled"}
    if not path.exists():
        return {"rotated": False, "reason": "missing"}

    try:
        file_size = path.stat().st_size
    except OSError:
        return {"rotated": False, "reason": "stat_failed"}

    if file_size < max_bytes:
        return {"rotated": False, "reason": "under_threshold", "size": file_size}

    # Read all events
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"rotated": False, "reason": "read_failed", "error": str(exc)}

    events: list[str] = [line for line in lines if line.strip()]
    total_count = len(events)

    if total_count <= retain_recent:
        return {"rotated": False, "reason": "too_few_events", "count": total_count}

    # Archive older events
    archive_events = events[:-retain_recent]
    retained_events = events[-retain_recent:]

    archive_path = unique_backup_path(path, "rotated")
    try:
        archive_path.write_text("\n".join(archive_events) + "\n", encoding="utf-8")
        path.write_text("\n".join(retained_events) + "\n", encoding="utf-8")
    except OSError as exc:
        return {"rotated": False, "reason": "write_failed", "error": str(exc)}

    return {
        "rotated": True,
        "archived_count": len(archive_events),
        "retained_count": len(retained_events),
        "archive_path": str(archive_path),
        "previous_size": file_size,
    }
