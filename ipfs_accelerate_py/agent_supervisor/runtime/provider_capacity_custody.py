"""Deny-only evidence for quota failures whose task dispatch is not disproved.

These records preserve observations, not callback settlement. In particular a
completed runner is not proof that every tool/container lifetime has closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from ..proof.formal_verification_contracts import content_identity

SCHEMA = "ipfs_accelerate_py/agent-supervisor/retained-provider-capacity@1"
REASON = "provider_capacity_settlement_required"
MAX_RECORD_BYTES = 64 * 1024


class RetentionError(RuntimeError):
    """Custody publication failure is not a terminal callback outcome."""


def execution_observation(text: str) -> dict[str, Any]:
    """Positive log observations may deny a refund; absence proves nothing."""
    tool_events = 0
    model_calls = 0
    turns = 0
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        if value.get("type") in {"tool_call", "tool_call_update"}:
            tool_events += 1
        count = value.get("num_turns")
        if type(count) is int and count > 0:
            turns = max(turns, count)
        usage = value.get("modelUsage")
        if isinstance(usage, dict):
            for model in usage.values():
                count = model.get("modelCalls") if isinstance(model, dict) else None
                if type(count) is int and count > 0:
                    model_calls = max(model_calls, count)
    return {
        "task_execution_observed": bool(tool_events or model_calls or turns),
        "tool_event_count": tool_events,
        "model_calls": model_calls,
        "turns": turns,
        "terminal_callback_custody": "unknown",
    }


def path_for(state_path: Path) -> Path:
    return state_path.with_name(state_path.name + ".retained-provider-capacity.json")


def read(path: Path) -> dict[str, Any] | None:
    """A missing record is distinct from any malformed/inaccessible record."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or not 0 < before.st_size <= MAX_RECORD_BYTES
        ):
            raise ValueError("retained capacity record identity is invalid")
        raw = os.read(fd, MAX_RECORD_BYTES + 1)
        after = os.fstat(fd)
        final = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, k) != getattr(item, k)
            for item in (after, final)
            for k in fields
        ):
            raise ValueError("retained capacity record changed")
    finally:
        os.close(fd)
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("receipt_id")
        != content_identity({k: v for k, v in value.items() if k != "receipt_id"})
        or any(
            value.get(k) is not False
            for k in (
                "retry_authorized",
                "settlement_authority",
                "completion_authority",
                "cleanup_allowed",
            )
        )
    ):
        raise ValueError("retained capacity record is invalid")
    return value


def publish(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    """Durably append one record without replacing an earlier observation."""
    value = {**body, "schema": SCHEMA}
    value["receipt_id"] = content_identity(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("retained capacity record exceeds its bound")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".capacity-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(name, path, follow_symlinks=False)
        except FileExistsError:
            pass
        os.unlink(name)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    observed = read(path)
    if observed != value:
        raise ValueError("retained capacity observation conflicts")
    return value


def log_identity(path: Path) -> dict[str, Any]:
    """Hash available evidence without interpreting it as execution authority."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
        ):
            raise ValueError("retained capacity log is not an owned regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        # Oversized logs remain preserved in place; an unavailable digest
        # must never become a cleanup or replay permission.
        if remaining <= 256 * 1024 * 1024:
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    raise ValueError("retained capacity log shortened")
                digest.update(chunk)
                remaining -= len(chunk)
        after = os.fstat(fd)
        final = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, k) != getattr(item, k)
            for item in (after, final)
            for k in fields
        ):
            raise ValueError("retained capacity log changed")
        return {
            "path": str(path.resolve()),
            "device": before.st_dev,
            "inode": before.st_ino,
            "size": before.st_size,
            "sha256": digest.hexdigest() if remaining == 0 else "",
            "complete_digest": remaining == 0,
        }
    finally:
        os.close(fd)
