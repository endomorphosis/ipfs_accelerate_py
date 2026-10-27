"""Observe the current Native execution start, after bounded checkout setup.

This read-only clock grants no process, retry or task authority. A private
authenticated event and both current dispatch leases must agree with the live
sealed daemon. Missing evidence preserves the caller's older clock fallback.
"""
from __future__ import annotations

import os
import re
import math
from pathlib import Path
import stat

from .current_incident_adjudication import _bytes, _object, _read, _sha
from . import supervisor_runtime as runtime

_STAT = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid", "st_nlink")


def _clock_rows(raw):
    """Verify a retained Native segment without assuming its first sequence."""
    if not raw.endswith(b"\n"):
        raise ValueError("execution_clock_partial_history")
    rows = [_object(line) for line in raw.splitlines() if line.strip()]
    previous, sequence, stream, snapshot = None, None, None, None
    for row in rows:
        index = row.get("sequence")
        prior = row.get("previous_event_id")
        if (type(index) is not int or index <= 0
                or row.get("event_id") != "sha256:" + _sha(_bytes({k: v for k, v in row.items() if k != "event_id"}))
                or sequence is not None and (index != sequence + 1 or prior != previous)
                or sequence is None and not (index == 1 and prior == "" or index > 1 and isinstance(prior, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", prior))
                or sequence is not None and (row.get("stream_id") != stream or row.get("snapshot_id") != snapshot)):
            raise ValueError("execution_clock_history_changed")
        previous, sequence, stream, snapshot = row["event_id"], index, row.get("stream_id"), row.get("snapshot_id")
    return rows


def current_execution_started_at(supervisor, state, *, now_ts):
    """Return only this exact unfinished, owned attempt's primary start."""
    try:
        from . import implementation_daemon as native
        from .supervisor import parse_timestamp
        from ..merge.checkout_lock import checkout_lock_repository_matches
        if type(now_ts) not in (int, float) or not math.isfinite(now_ts):
            return None
        if (not state.implementation_in_progress or not state.active_task_id
                or not state.active_task_cid or not state.active_task_key
                or type(state.active_attempt) is not int or state.active_attempt <= 0
                or state.last_implementation_task_id != state.active_task_id
                or state.last_implementation_task_cid != state.active_task_cid
                or state.last_implementation_worktree_path != state.active_worktree_path
                or state.last_implementation_log_path != state.active_log_path):
            return None
        workspace = Path(state.active_worktree_path)
        log = Path(state.active_log_path)
        workroot = supervisor.config.worktree_root
        if workroot is None:
            return None
        if not workroot.is_absolute():
            workroot = supervisor.config.repo_root / workroot
        if (not workspace.is_absolute() or workspace.resolve() != workspace
                or workspace == workroot.resolve() or not workspace.is_relative_to(workroot.resolve())
                or not log.is_absolute() or log.resolve() != log or not log.is_relative_to(supervisor.config.state_dir)):
            return None
        expected = {"task_id": state.active_task_id, "canonical_task_cid": state.active_task_cid,
            "canonical_task_key": state.active_task_key, "attempt": state.active_attempt}
        if (state.implementation_attempts.get(state.active_task_id) != state.active_attempt
                or state.implementation_attempts_by_cid.get(state.active_task_cid) != state.active_attempt):
            return None
        lease_path = supervisor._implementation_maintenance_lock_path()
        claim_path = supervisor._task_dispatch_claim_path(state)
        lease = supervisor._read_watchdog_authority(lease_path)
        claim = supervisor._read_watchdog_authority(claim_path)
        identity = runtime.load_supervised_child_identity(supervisor._managed_daemon_identity_path())
        if lease is None or claim is None or identity is None:
            return None
        birth = native.ProcessBirthIdentity.from_dict(claim.get("owner_process_birth"))
        if (dict(identity.owner_scope) != supervisor._managed_daemon_owner_scope()
                or tuple(identity.command) != tuple(supervisor._build_daemon_command())
                or birth != identity.process_birth or runtime.read_process_birth(birth.pid) != birth
                or runtime.read_process_command_argv(birth.pid) != identity.command
                or runtime.supervised_child_identity_liveness(identity) is not native.OwnerLiveness.ALIVE
                or supervisor._read_managed_daemon_pid() != birth.pid
                or lease.get("kind") != "implementation" or lease.get("lease_role") == "supervisor_maintenance"
                or claim.get("kind") != native.IMPLEMENTATION_TASK_CLAIM_LOCK_KIND
                or claim.get("lease_role") != native.IMPLEMENTATION_DISPATCH_INTENT_LEASE_ROLE
                or not lease.get("lease_id") or not claim.get("lease_id")
                or lease.get("pid") != birth.pid or claim.get("pid") != birth.pid
                or lease.get("owner_script") != claim.get("owner_script")
                or any(lease.get(key) != value or claim.get(key) != value for key, value in expected.items())
                or not claim.get("board_namespace") or lease.get("board_namespace") != claim["board_namespace"]
                or Path(str(lease.get("repo_root") or "")).resolve() != supervisor.config.repo_root.resolve()
                or Path(str(lease.get("state_dir") or "")).resolve() != supervisor.config.state_dir.resolve()
                or Path(str(claim.get("state_dir") or "")).resolve() != supervisor.config.state_dir.resolve()
                or Path(str(claim.get("state_path") or "")).resolve() != supervisor.config.state_path.resolve()
                or not checkout_lock_repository_matches(claim, supervisor.config.repo_root)
                or claim.get("dispatch_authority_cid") != native.PortalImplementationDaemon._implementation_dispatch_authority_cid(claim)):
            return None
        if "owner_process_birth" in lease and native.ProcessBirthIdentity.from_dict(lease["owner_process_birth"]) != birth:
            return None
        issued = parse_timestamp(claim.get("started_at"))
        selected = parse_timestamp(state.active_task_started_at)
        if (issued is None or selected is None or issued.tzinfo is None or selected.tzinfo is None
                or lease.get("started_at") != claim.get("started_at")
                or state.last_implementation_started_at != claim.get("started_at")
                or not selected.timestamp() <= issued.timestamp() <= now_ts):
            return None
        state_directory = supervisor.config.state_dir
        directory_info = state_directory.stat()
        if state_directory.resolve() != state_directory or directory_info.st_uid != os.getuid() or directory_info.st_mode & 0o077:
            return None
        path = state_directory / f"{supervisor.config.state_prefix}_events.jsonl"
        info = path.lstat()
        if (path.resolve(strict=True) != path.absolute() or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o002
                or not 0 < info.st_size <= 64 * 1024 * 1024):
            return None
        signature = tuple(getattr(info, field) for field in _STAT)
        binding = tuple(expected.items()) + (("worktree", state.active_worktree_path), ("log", state.active_log_path),
            ("lease", lease["lease_id"]), ("claim", claim["lease_id"]), ("birth", birth))
        cache = getattr(supervisor, "_current_execution_clock_cache", None)
        if cache and cache[0] == signature and cache[1] == binding:
            stamp = cache[2]
        else:
            rows = _clock_rows(_read(path))
            starts = [row for row in rows if row.get("type") == "implementation_started"
                and row.get("canonical_task_cid") == state.active_task_cid and row.get("attempt") == state.active_attempt]
            if len(starts) != 1:
                return None
            start = starts[0]
            if (any(start.get(key) != value for key, value in expected.items())
                    or start.get("board_namespace") != claim["board_namespace"]
                    or start.get("worktree_path") != state.active_worktree_path
                    or start.get("log_path") != state.active_log_path
                    or any(row.get("type") in {"implementation_finished", "implementation_terminated", "implementation_timeout"}
                        and row.get("canonical_task_cid") == state.active_task_cid and row.get("attempt") == state.active_attempt
                        for row in rows if row.get("sequence", 0) > start["sequence"])):
                return None
            parsed = parse_timestamp(start.get("timestamp"))
            if parsed is None or parsed.tzinfo is None:
                return None
            stamp = parsed.timestamp()
            if tuple(getattr(path.lstat(), field) for field in _STAT) != signature:
                return None
            supervisor._current_execution_clock_cache = (signature, binding, stamp)
        if (not issued.timestamp() <= stamp <= now_ts
                or supervisor._read_watchdog_authority(lease_path) != lease
                or supervisor._read_watchdog_authority(claim_path) != claim
                or runtime.read_process_birth(birth.pid) != birth):
            return None
        return stamp
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
