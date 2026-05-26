"""JSONL event-log helpers for agent supervisor runtimes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


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


def append_jsonl_event(path: Path, event_type: str, payload: Mapping[str, Any]) -> None:
    if path.exists() and path.is_dir():
        repair_jsonl_event_log(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"type": event_type, "timestamp": utc_now(), **dict(payload)}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
