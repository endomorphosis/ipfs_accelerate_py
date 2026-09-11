"""Optional closed diagnostic references for a future coding-repair prompt.

Only the configured board's repair directory is read. Report bodies, arbitrary
prose and commands never enter the prompt. These observations grant no native
source, signal, callback, completion or replay authority.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

SCHEMA = "fleet-repair/diagnostic-handoff@1"
STAGES = frozenset(
    {
        "incident_observation",
        "source_qualification",
        "generic_fix_qualification",
        "native_fix_qualification",
        "graceful_closure",
        "preservation_archive",
        "source_transition",
        "native_observation",
        "service_recreation",
        "native_restart",
        "post_start_verification",
        "independent_review",
    }
)
_PRIVATE_WORDS = {
    "vault",
    "secret",
    "secrets",
    "token",
    "tokens",
    "credential",
    "credentials",
    "environment",
    "environ",
    "env",
}


class _Unavailable(ValueError):
    pass


def _require(value: bool, reason: str) -> None:
    if not value:
        raise _Unavailable(reason)


def _parts(value: Any) -> tuple[str, ...]:
    _require(isinstance(value, str) and 0 < len(value) <= 1024, "path_invalid")
    parts = tuple(value.split("/"))
    _require(
        1 <= len(parts) <= 24
        and all(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
            and not (_PRIVATE_WORDS & set(re.split(r"[_.-]", part.lower())))
            for part in parts
        ),
        "path_outside_diagnostic_scope",
    )
    _require(parts[-1].endswith(".json"), "path_not_json")
    return parts


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        *_identity(info),
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class _Files:
    """Pin nofollow directory chains and reobserve every read before return."""

    def __init__(self, root: Path, stack: contextlib.ExitStack, deadline: float):
        _require(
            root.is_absolute() and ".." not in root.parts, "repair_directory_invalid"
        )
        self.stack, self.deadline = stack, deadline
        self.directories: list[tuple[int, str, int, tuple[int, ...]]] = []
        self.files: list[tuple[int, str, int, tuple[int, ...]]] = []
        current = self._open("/", os.O_RDONLY | os.O_DIRECTORY)
        for part in root.parts[1:]:
            current = self._directory(current, part)
        root_info = os.fstat(current)
        _require(
            root_info.st_uid == os.geteuid()
            and stat.S_IMODE(root_info.st_mode) == 0o700,
            "repair_directory_not_private",
        )
        self.root_fd = current

    def _open(self, path: str, flags: int, *, parent: int | None = None) -> int:
        self.budget()
        fd = os.open(
            path, flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent
        )
        self.stack.callback(os.close, fd)
        return fd

    def _directory(self, parent: int, name: str) -> int:
        fd = self._open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
        self.directories.append((parent, name, fd, _identity(os.fstat(fd))))
        return fd

    def budget(self) -> None:
        _require(time.monotonic() < self.deadline, "read_budget_exceeded")

    def read(self, relative: Any, limit: int) -> tuple[bytes, str]:
        parts = _parts(relative)
        parent = self.root_fd
        for part in parts[:-1]:
            parent = self._directory(parent, part)
        fd = self._open(parts[-1], os.O_RDONLY, parent=parent)
        before = os.fstat(fd)
        # Native repair reports may be0664 under umask002. The pinned owned
        # 0700 board directory prevents group access; do not chmod old data.
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.geteuid()
            and before.st_nlink == 1
            and not before.st_mode & 0o002,
            "file_not_owned_regular",
        )
        _require(0 < before.st_size <= limit, "file_size_bound")
        chunks, size = [], 0
        while True:
            self.budget()
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            _require(size <= limit, "file_size_bound")
        _require(size == before.st_size, "file_changed")
        self.files.append((parent, parts[-1], fd, _file_identity(before)))
        self.verify()
        raw = b"".join(chunks)
        return raw, hashlib.sha256(raw).hexdigest()

    def verify(self) -> None:
        self.budget()
        for parent, name, fd, expected in self.directories:
            _require(
                _identity(os.fstat(fd)) == expected
                and _identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
                == expected,
                "directory_changed",
            )
        for parent, name, fd, expected in self.files:
            _require(
                _file_identity(os.fstat(fd)) == expected
                and _file_identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
                == expected,
                "file_changed",
            )


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_field")
        result[key] = value
    return result


def _timestamp(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def source_binding(observation: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Bind observed Git heads/cleanliness, not loaded code or dirty file bytes."""
    details = observation.get("details", {})
    _require(isinstance(details, dict), "source_observation_unavailable")
    heads, integrity = details.get("source_heads"), details.get("source_integrity")
    _require(
        isinstance(heads, dict)
        and 1 <= len(heads) <= 16
        and all(
            isinstance(k, str)
            and 0 < len(k) <= 512
            and isinstance(v, str)
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", v)
            for k, v in heads.items()
        )
        and isinstance(integrity, dict),
        "source_observation_unavailable",
    )
    raw = json.dumps(
        integrity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _require(len(raw) <= 32768, "source_observation_bound")
    return dict(heads), hashlib.sha256(raw).hexdigest()


def load_diagnostic_handoff(
    board: dict[str, Any],
    directory: Path,
    observation: dict[str, Any],
    *,
    observed_at: float,
    now: float | None = None,
) -> dict[str, Any]:
    configured = board.get("diagnostic_handoff")
    if configured is None:
        return {}
    unavailable = {"status": "unavailable", "diagnostic_only": True}
    try:
        current = time.time() if now is None else now
        _require(
            _timestamp(current)
            and _timestamp(observed_at)
            and 0 <= current - observed_at <= 180,
            "current_probe_stale",
        )
        _require(observation.get("board_id") == board["id"], "current_probe_foreign")
        heads, integrity = source_binding(observation)
        read_started = time.monotonic()
        with contextlib.ExitStack() as stack:
            files = _Files(directory, stack, read_started + 2)
            raw, handoff_digest = files.read(configured, 32768)
            value = json.loads(raw, object_pairs_hook=_object)
            _require(
                isinstance(value, dict)
                and set(value)
                == {
                    "schema",
                    "board_id",
                    "observed_at",
                    "expires_at",
                    "source_heads",
                    "source_integrity_sha256",
                    "completed_stages",
                    "reports",
                },
                "handoff_schema_invalid",
            )
            _require(
                value["schema"] == SCHEMA and value["board_id"] == board["id"],
                "handoff_scope_mismatch",
            )
            start, expires = value["observed_at"], value["expires_at"]
            _require(
                _timestamp(start)
                and _timestamp(expires)
                and start <= current <= expires
                and 0 < expires - start <= 86400,
                "handoff_stale",
            )
            _require(
                value["source_heads"] == heads
                and value["source_integrity_sha256"] == integrity,
                "handoff_source_mismatch",
            )
            stages, refs = value["completed_stages"], value["reports"]
            _require(
                isinstance(stages, list)
                and 1 <= len(stages) <= len(STAGES)
                and all(isinstance(s, str) and s in STAGES for s in stages)
                and len(set(stages)) == len(stages),
                "stage_invalid",
            )
            _require(
                isinstance(refs, list) and 1 <= len(refs) <= 8, "report_count_bound"
            )
            references, paths = [], set()
            for ref in refs:
                _require(
                    isinstance(ref, dict)
                    and set(ref) == {"path", "sha256", "stage"}
                    and ref["stage"] in stages
                    and isinstance(ref["path"], str)
                    and ref["path"] not in paths,
                    "report_reference_invalid",
                )
                paths.add(ref["path"])
                _, digest = files.read(ref["path"], 1024 * 1024)
                _require(ref["sha256"] == digest, "report_digest_mismatch")
                references.append(
                    {
                        "path": str(directory / ref["path"]),
                        "sha256": digest,
                        "stage": ref["stage"],
                    }
                )
            _require(
                set(stages) == {ref["stage"] for ref in references},
                "stage_report_missing",
            )
            files.verify()
        # Reading is optional; an expiry reached while reading cannot revive it.
        finished = current + max(0.0, time.monotonic() - read_started)
        _require(expires >= finished, "handoff_stale")
        _require(finished - observed_at <= 180, "current_probe_stale")
        return {
            "status": "available",
            "diagnostic_only": True,
            "handoff_sha256": handoff_digest,
            "observed_at": start,
            "expires_at": expires,
            "matched_probe_observed_at": observed_at,
            "source_heads": heads,
            "source_integrity_sha256": integrity,
            "completed_stages": stages,
            "report_references": references,
        }
    except _Unavailable as exc:
        return {**unavailable, "reason": str(exc)}
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        # Never expose file contents, exception strings or arbitrary paths.
        return {**unavailable, "reason": "handoff_unreadable"}
