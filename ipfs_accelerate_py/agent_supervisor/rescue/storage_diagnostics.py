"""Bounded, read-only checks for explicitly configured active storage.

These observations do not authorize cleanup, process control, Git repair or
publication. Git pointer files are read without invoking Git or opening stores.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import stat
from typing import Any

SCHEMA = "agent-supervisor/storage-diagnostics@1"
MAX_POINTER_BYTES = 4096


def _absolute_path(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and Path(value).is_absolute()
            and not any(char in value for char in "\0\r\n"))


def validate_storage_checks(config: Any) -> None:
    if not isinstance(config, dict) or set(config) - {"filesystems", "git_worktrees"}:
        raise ValueError("storage_checks must contain filesystems and/or git_worktrees")
    for key in ("filesystems", "git_worktrees"):
        entries = config.get(key, [])
        if not isinstance(entries, list) or len(entries) > 32:
            raise ValueError(f"storage_checks.{key} must be a list of at most 32 entries")
        for entry in entries:
            if key == "git_worktrees":
                if not _absolute_path(entry):
                    raise ValueError("storage_checks.git_worktrees requires absolute paths")
                continue
            if (not isinstance(entry, dict) or not _absolute_path(entry.get("path"))
                    or set(entry) - {"path", "min_available_bytes", "min_available_percent"}):
                raise ValueError("filesystem checks require an absolute path and space thresholds")
            minimum_bytes = entry.get("min_available_bytes", 0)
            minimum_percent = entry.get("min_available_percent", 0)
            if type(minimum_bytes) is not int or minimum_bytes < 0:
                raise ValueError("min_available_bytes must be a nonnegative integer")
            if (type(minimum_percent) not in (int, float)
                    or not math.isfinite(minimum_percent) or not 0 <= minimum_percent <= 100):
                raise ValueError("min_available_percent must be finite and between 0 and 100")


def _read_pointer(path: Path) -> str:
    # O_NONBLOCK and fstat prevent a replaced pointer becoming a FIFO hang.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_POINTER_BYTES:
            raise ValueError("invalid Git pointer file")
        value = os.read(descriptor, MAX_POINTER_BYTES + 1)
        if len(value) > MAX_POINTER_BYTES:
            raise ValueError("oversized Git pointer file")
        text = value.decode("utf-8").strip()
        if not text or any(char in text for char in "\0\r\n"):
            raise ValueError("invalid Git pointer contents")
        return text
    finally:
        os.close(descriptor)


def _pointer_target(parent: Path, value: str) -> Path:
    target = Path(value)
    return (target if target.is_absolute() else parent / target).resolve()


def _git_worktree(root: str) -> dict[str, Any]:
    marker = Path(root) / ".git"
    result: dict[str, Any] = {"worktree": root, "git_marker": str(marker), "reason_codes": []}
    reasons = result["reason_codes"]
    try:
        if not marker.exists():
            reasons.append("git_marker_missing")
        else:
            if marker.is_dir():
                git_dir = marker.resolve()
            else:
                pointer = _read_pointer(marker)
                if not pointer.startswith("gitdir: "):
                    raise ValueError("invalid Git directory pointer")
                target = pointer[len("gitdir: "):].strip()
                if not target:
                    raise ValueError("empty Git directory pointer")
                git_dir = _pointer_target(marker.parent, target)
            result["git_dir"] = str(git_dir)
            if not git_dir.is_dir():
                reasons.append("git_directory_missing")
            else:
                common_pointer = git_dir / "commondir"
                linked = os.path.lexists(common_pointer)
                common = (_pointer_target(git_dir, _read_pointer(common_pointer))
                          if linked else git_dir)
                result["common_dir"] = str(common)
                if not common.is_dir():
                    reasons.append("git_common_directory_missing")
                elif not (common / "objects").is_dir():
                    reasons.append("git_objects_directory_missing")
                if not (git_dir / "HEAD").is_file():
                    reasons.append("git_head_missing")
                if linked:
                    backlink = git_dir / "gitdir"
                    if not backlink.exists():
                        reasons.append("git_worktree_registration_missing")
                    elif _pointer_target(git_dir, _read_pointer(backlink)) != marker.resolve():
                        reasons.append("git_worktree_registration_mismatch")
    except (OSError, ValueError, RuntimeError) as exc:
        reasons.append("git_metadata_unavailable")
        result["error_type"] = type(exc).__name__
        if isinstance(exc, OSError):
            result["errno"] = exc.errno
    result["status"] = "attention_required" if reasons else "healthy"
    return result


def _filesystem(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": config["path"], "min_available_bytes": config.get("min_available_bytes", 0),
        "min_available_percent": config.get("min_available_percent", 0), "reason_codes": [],
    }
    reasons = result["reason_codes"]
    try:
        info = os.statvfs(config["path"])
        if info.f_frsize <= 0 or info.f_blocks <= 0 or info.f_bavail < 0:
            raise ValueError("invalid filesystem capacity sample")
        available = info.f_bavail * info.f_frsize
        capacity = info.f_blocks * info.f_frsize
        percent = 100 * info.f_bavail / info.f_blocks
        result.update(available_bytes=available, capacity_bytes=capacity, available_percent=percent)
        if available < result["min_available_bytes"]:
            reasons.append("filesystem_available_bytes_low")
        if percent < result["min_available_percent"]:
            reasons.append("filesystem_available_percent_low")
    except (OSError, ValueError) as exc:
        reasons.append("filesystem_sample_unavailable")
        result["error_type"] = type(exc).__name__
        if isinstance(exc, OSError):
            result["errno"] = exc.errno
    result["status"] = "attention_required" if reasons else "healthy"
    return result


def observe_storage(config: dict[str, Any]) -> dict[str, Any]:
    validate_storage_checks(config)
    filesystems = [_filesystem(entry) for entry in config.get("filesystems", [])]
    worktrees = [_git_worktree(root) for root in config.get("git_worktrees", [])]
    reasons = sorted({code for entry in [*filesystems, *worktrees] for code in entry["reason_codes"]})
    return {"schema": SCHEMA, "status": "attention_required" if reasons else "healthy",
            "reason_codes": reasons, "filesystems": filesystems, "git_worktrees": worktrees,
            "handling": "diagnostic_only"}
