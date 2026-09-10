#!/usr/bin/env python3
"""Bounded, read-only health observations for existing DuckDB board operators.

This adapter never opens a DuckDB file or signals a process. Process identity,
fresh lane projections and existing operator status commands are observations;
they cannot authorize task completion or Git publication. A separate closeout
gate must admit completion, even when every observed task is terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import stat
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "ipfs_accelerate_py/taskboard-fleet-live-probe@1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_SOURCE_INDEX_BYTES = 2 * 1024 * 1024
COMPLETED = frozenset({"complete", "completed", "done", "accepted", "succeeded"})
BLOCKED = frozenset({"blocked", "quarantined", "failed", "error"})
PROVIDER_MODULES = frozenset({
    "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner",
    "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
    "ipfs_accelerate_py.agent_supervisor.runtime.provider_fallback_runner",
})


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one bounded regular-file object, preserving failures for strict callers."""
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
            raise ValueError(f"expected bounded regular JSON file: {path}")
        chunks = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if (len(raw) > MAX_JSON_BYTES or len(raw) != before.st_size
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError(f"JSON file changed or exceeded read bound: {path}")
    finally:
        os.close(descriptor)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    """Invalid or unavailable observation metadata cannot establish board health."""
    try:
        return read_json_object(path)
    except (OSError, ValueError, RecursionError):
        return {}


def _age(value: Any, now: float) -> float | None:
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timestamp = float(value)
        elif isinstance(value, str):
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        else:
            return None
        age = now - timestamp
        return age if age >= -5 else None
    except (ValueError, OverflowError):
        return None


def process_identity(pid: Any, *, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Capture Linux birth identity and reject zombies and racing PID reuse."""
    try:
        pid = int(pid)
        if pid <= 1:
            return {}
        root = proc_root / str(pid)
        before = (root / "stat").read_text()
        stat = before[before.rfind(")") + 2:].split()
        if stat[0] in {"Z", "X"}:
            return {}
        cmdline = (root / "cmdline").read_bytes()
        argv = [value.decode("utf-8", errors="replace") for value in cmdline.split(b"\0") if value]
        try:
            cwd = str((root / "cwd").resolve(strict=True))
        except PermissionError:
            # Hardened state owners intentionally disable dumpability, which
            # hides cwd even from their Unix user. Birth fields remain visible.
            cwd = ""
        try:
            wait_channel = (root / "wchan").read_text().strip()
        except OSError:
            # Hardened owners can hide this diagnostic without hiding birth.
            wait_channel = ""
        after = (root / "stat").read_text()
        latest = after[after.rfind(")") + 2:].split()
        if stat[19] != latest[19] or latest[0] in {"Z", "X"}:
            return {}
        return {
            "pid": pid, "parent_pid": int(stat[1]), "start_time_ticks": int(stat[19]),
            "boot_id": (proc_root / "sys/kernel/random/boot_id").read_text().strip(),
            "cwd": cwd, "argv": argv, "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
            "cpu_ticks": int(stat[11]) + int(stat[12]),
            "process_state": latest[0], "wait_channel": wait_channel,
        }
    except (OSError, ValueError, IndexError, TypeError):
        return {}


def birth_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return bool(actual and expected and all(
        str(actual.get(key, "")) == str(expected.get(key, ""))
        for key in ("pid", "start_time_ticks", "boot_id")
    ))


def _public_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    # Keep enough information for an operator to recheck identity, not argv (which
    # may contain long prompts or credentials in unrelated process descendants).
    return {key: identity[key] for key in (
        "pid", "parent_pid", "start_time_ticks", "boot_id", "cwd", "cmdline_sha256", "cpu_ticks",
        "process_state", "wait_channel",
    ) if key in identity}


def _process_condition(identity: Mapping[str, Any]) -> str:
    """Report a kernel observation, never infer death or permission to signal."""
    state = identity.get("process_state")
    if state in {"T", "t"}:
        return "stopped"
    if state == "D":
        return "uninterruptible"
    return ""


def _flag(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return ""


def _lane_process(pid: Any, lane_dir: Path, prefix: str, expected_cwd: str) -> dict[str, Any]:
    identity = process_identity(pid)
    argv = identity.get("argv", [])
    if (identity.get("cwd") not in ("", expected_cwd)
            or _flag(argv, "--state-dir") != str(lane_dir)
            or _flag(argv, "--state-prefix") != prefix):
        return {}
    return identity


def _status_command(board: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    argv = board.get("status_argv")
    if not isinstance(argv, list) or not argv:
        return {}, "native_status_unavailable"
    # Native status processes can fork. Give them their own process group so a
    # deadline cannot strand a probe child, or signal a live owner by accident.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            environment = dict(os.environ)
            # A packaged fleet release uses a deliberately minimal package
            # stub. Its PYTHONPATH must not shadow the board's sealed runtime.
            board_root = Path(board["cwd"])
            board_python_paths = [board_root, board_root / "external/ipfs_accelerate"]
            board_python_paths.extend(Path(value) for value in board.get("status_python_paths", []))
            environment["PYTHONPATH"] = os.pathsep.join(str(value) for value in board_python_paths if value.is_dir())
            child = subprocess.Popen(argv, cwd=board["cwd"], stdout=stdout, stderr=stderr,
                                     env=environment, start_new_session=True)
            try:
                code = child.wait(timeout=float(board.get("status_timeout_seconds", 45)))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                # A TERM-responsive leader can leave an ignoring descendant in
                # its dedicated group. Kill that group even after leader exit.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    return {}, "native_status_cleanup_timeout"
                return {}, "native_status_timeout"
            stdout.seek(0)
            raw = stdout.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES:
                return {}, "native_status_output_too_large"
            result = _object(json.loads(raw))
            if not result:
                return {}, "native_status_invalid_json"
            return result, "" if code == 0 else "native_status_nonzero"
        except (OSError, ValueError, TypeError):
            return {}, "native_status_failed"


def _status_with_receipt_retry(
    board: Mapping[str, Any], expected_birth: Mapping[str, Any],
) -> tuple[dict[str, Any], str, int]:
    """Re-read an unavailable native receipt within one bounded publication window.

    Slow authenticated queries can leave a gap between receipt expiry and the
    next publication. Only the native reader may admit the replacement. Never
    reuse a rejected sample, extend its TTL, or retry a blocked/stuck decision.
    Recheck the exact owner around every invocation: some native commands open
    offline authority if called after the owner exits.
    """
    def same_ready_owner() -> bool:
        status = read_json(Path(board["owner_status_path"]))
        birth = _object(_object(status.get("identity")).get("process_birth"))
        return bool(status.get("lifecycle") == "ready"
                    and birth_matches(birth, expected_birth)
                    and birth_matches(process_identity(expected_birth.get("pid")), expected_birth))

    def unavailable(native: Mapping[str, Any], error: str) -> bool:
        return bool(not error and native.get("healthy") is False
                    and native.get("owner_ready") is True
                    and native.get("broker_authenticated_receipt") is False
                    and native.get("receipt") == {}
                    and not native.get("blocked") and not native.get("stuck")
                    and _object(native.get("receipt_error")).get("reason")
                    == "live_status_receipt_unavailable_or_invalid")

    attempts = 0
    native: dict[str, Any] = {}
    error = ""
    deadline = None
    for index in range(4):
        command_board = board
        if index:
            remaining = deadline - time.monotonic()
            if remaining <= 5.0:
                break
            time.sleep(5.0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            command_board = {**board, "status_timeout_seconds": min(
                float(board.get("status_timeout_seconds", 45)), remaining)}
        if not same_ready_owner():
            return {}, "native_status_owner_changed", attempts
        native, error = _status_command(command_board)
        attempts += 1
        if not same_ready_owner():
            return {}, "native_status_owner_changed", attempts
        if not unavailable(native, error):
            return native, error, attempts
        if deadline is None:
            deadline = time.monotonic() + 20.0
    return native, "native_receipt_unavailable_after_retry", attempts


def _database_board_authority(native: Mapping[str, Any], board: Mapping[str, Any],
                              owner_status: Mapping[str, Any], now: float) -> dict[str, Any]:
    """Accept only a fresh native observation bound to the current ready owner.

    The configured, sealed operator performs the credential admission and
    completion snapshot validation. This adapter checks its exact owner and
    population again; it never turns a task projection into a closeout gate.
    """
    if native.get("schema") != "ipfs_accelerate_py/agent-supervisor/database-board-status@1":
        return {}
    age = _age(native.get("observed_at"), now)
    owner = _object(owner_status.get("identity"))
    admitted = _object(native.get("owner_identity"))
    snapshot = _object(native.get("completion_snapshot"))
    snapshot_owner = _object(snapshot.get("owner_identity"))
    keys = ("server_id", "process_birth_id", "database_uuid", "store_id", "generation", "fence_epoch")
    if (native.get("authoritative_task_observation") is not True
            or not board.get("task_namespace")
            or native.get("board_namespace") != board.get("task_namespace")
            or age is None or age > 30
            or any(owner.get(key) in (None, "") or admitted.get(key) != owner.get(key)
                   or snapshot_owner.get(key) != owner.get(key) for key in keys)
            or snapshot.get("schema") != "ipfs_accelerate_py/agent-supervisor/typed-completion-progress-snapshot@1"):
        return {}
    tasks = native.get("tasks")
    control = _object(native.get("control"))
    projection = _object(snapshot.get("completion_projection"))
    states = projection.get("task_states")
    if (not isinstance(tasks, list) or not 1 <= len(tasks) <= 512
            or not isinstance(states, list) or len(tasks) != len(states)
            or control.get("task_count") != len(tasks)):
        return {}
    try:
        identities = [(row["task_cid"], row["status"], row["revision"]) for row in tasks]
        expected = [(row["task_cid"], row["status"], row["revision"]) for row in states]
        aliases = [row["task_alias"] for row in tasks]
        goals = json.loads(control["goals_json"])
        if (any(not isinstance(alias, str) or not alias for alias in aliases)
                or len(set(aliases)) != len(tasks)
                or len({row[0] for row in identities}) != len(tasks)
                or sorted(identities) != sorted(expected)
                or not isinstance(goals, list) or len(goals) != control.get("goal_count")):
            return {}
        unsettled = [row["goal_cid"] for row in goals if row["status"] not in COMPLETED]
        receipts = projection["completion_receipts"]
        if not isinstance(receipts, list):
            return {}
    except (KeyError, TypeError, ValueError):
        return {}
    return {"authenticated_query": True, "task_count": len(tasks),
            "task_statuses": {row["task_alias"]: row["status"] for row in tasks},
            "event_cursor": control.get("event_watermark"),
            "unsettled_goal_count": len(unsettled), "unsettled_goal_ids": unsettled,
            "completion_receipt_count": len(receipts),
            "blocked": bool(unsettled) and all(row["status"] in COMPLETED for row in tasks)}


def _counts(authority: Mapping[str, Any]) -> dict[str, int]:
    counts = authority.get("status_counts")
    if isinstance(counts, dict) and counts and all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for key, value in counts.items()
    ):
        return dict(sorted(counts.items()))
    states = authority.get("task_statuses")
    if isinstance(states, dict) and states:
        values = [value if isinstance(value, str) else _object(value).get("status") for value in states.values()]
        if all(isinstance(value, str) and value for value in values):
            return dict(sorted(Counter(values).items()))
    return {}


def _blocked_task_ids(authority: Mapping[str, Any]) -> list[str]:
    """Normalize native operator blocker lists without treating text as an ID list."""
    identities = set()
    for key in ("blocked_task_ids", "failed_or_blocked_task_ids"):
        values = authority.get(key)
        if isinstance(values, list):
            identities.update(value for value in values if isinstance(value, str) and value.strip())
    return sorted(identities)


def _progress(authority: Mapping[str, Any]) -> str:
    """Track task/goal changes, excluding publication and maintenance activity.

    A database event cursor can advance for leases, observations or other
    control events without advancing any task. Keep that cursor in diagnostic
    details, but never use it to postpone the watchdog's task-stall deadline.
    """
    counts = _counts(authority)
    fields = {key: authority[key] for key in (
        "task_count", "unsettled_goal_count"
    ) if key in authority}
    for key in ("active_task_ids", "completed_task_ids"):
        values = authority.get(key)
        if isinstance(values, list):
            # Native result ordering and duplicate projections are not work.
            fields[key] = sorted({value for value in values
                                  if isinstance(value, str) and value.strip()})
    if any(key in authority for key in ("blocked_task_ids", "failed_or_blocked_task_ids")):
        fields["blocked_task_ids"] = _blocked_task_ids(authority)
    statuses = authority.get("task_statuses")
    if isinstance(statuses, dict) and statuses:
        fields["task_statuses"] = {key: value if isinstance(value, str)
            else _object(value).get("status") for key, value in statuses.items()}
    if counts:
        fields["status_counts"] = counts
    if not fields:
        return ""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _provider_command(argv: list[str]) -> bool:
    executable = Path(argv[0]).name if argv else ""
    return bool(argv and (executable in {"codex", "grok"}
        or executable.startswith(("grok-", "codex-"))
        or any(argument in PROVIDER_MODULES for argument in argv)))


def _provider_busy(lanes: list[dict[str, Any]], board: Mapping[str, Any], now: float) -> list[dict[str, Any]]:
    """Require a real provider descendant and recent attempt output, not a daemon PID."""
    started = time.monotonic()
    roots = {lane[key]["pid"] for lane in lanes for key in ("supervisor", "daemon") if lane.get(key)}
    if not roots:
        return []
    provider_candidates: list[dict[str, Any]] = []
    identities: dict[int, dict[str, Any]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        identity = process_identity(entry.name)
        if not identity:
            continue
        identities[identity["pid"]] = identity
        argv = identity["argv"]
        if _provider_command(argv):
            provider_candidates.append(identity)
    recent_lanes: set[int] = set()
    limit = float(board.get("provider_log_stall_seconds", 1200))

    def recent(path: Path) -> bool:
        modified = path.stat().st_mtime
        sampled_at = now + max(0.0, time.monotonic() - started)
        return 0 <= sampled_at - modified <= limit

    for index, lane in enumerate(lanes):
        lane_dir = Path(lane["state_dir"])
        # Database-backed Portal attempts keep output below a binding directory,
        # rather than directly under the lane. Keep discovery at that known depth.
        for pattern in (
            "implementation-logs/*attempt-*.log",
            "implementation-logs/**/*.log",
            "*_database_portal_attempts/*/implementation-logs/*attempt-*.log",
        ):
            if any(recent(path) for path in lane_dir.glob(pattern) if path.is_file()):
                recent_lanes.add(index)
                break
    busy = []
    for identity in provider_candidates:
        current = identity
        seen: set[int] = set()
        ancestors: set[int] = set()
        while current and current["pid"] not in seen:
            seen.add(current["pid"])
            ancestors.add(current["pid"])
            current = identities.get(current["parent_pid"], {})
        if not ancestors.intersection(roots):
            continue
        if any(index in recent_lanes and any(lane.get(key, {}).get("pid") in ancestors
               for key in ("supervisor", "daemon")) for index, lane in enumerate(lanes)):
            busy.append(_public_identity(identity))
    return busy


def _source_heads(board: Mapping[str, Any]) -> dict[str, str]:
    heads: dict[str, str] = {}
    for relative in [".", *board.get("submodule_paths", [])]:
        try:
            result = subprocess.run(["git", "-C", str(Path(board["cwd"]) / relative), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, timeout=5, check=False)
            head = result.stdout.strip()
            if result.returncode == 0 and len(head) in (40, 64):
                heads[relative] = head
        except (OSError, subprocess.TimeoutExpired):
            continue
    return heads


def _source_integrity(board: Mapping[str, Any]) -> dict[str, Any]:
    """Reject dirty configured control-plane paths without importing their code.

    This is a read-only cleanliness constraint, not native source qualification
    or proof of what an already running interpreter has loaded.
    """
    entries = board.get("source_integrity_paths")
    if entries is None:
        return {"configured": False, "valid": True}
    result: dict[str, Any] = {"configured": True, "valid": False,
                              "scope": "configured_control_plane_cleanliness"}
    if not isinstance(entries, list) or not 1 <= len(entries) <= 4:
        return {**result, "reason": "invalid_source_integrity_configuration"}
    deadline = time.monotonic() + 5
    checked = []
    try:
        for entry in entries:
            repository = entry.get("repository") if isinstance(entry, dict) else None
            paths = entry.get("paths") if isinstance(entry, dict) else None
            if (not isinstance(repository, str) or not Path(repository).is_absolute()
                    or not isinstance(paths, list) or not 1 <= len(paths) <= 64
                    or any(not isinstance(p, str) or not p or not Path(p).parts or Path(p).is_absolute()
                           or any(part in {"..", ".git"} for part in Path(p).parts)
                           or p in {".", "./"} for p in paths)):
                return {**result, "reason": "invalid_source_integrity_configuration"}
            root = Path(repository).resolve(strict=True)
            argv = ["git", "--literal-pathspecs", "-c", "core.fsmonitor=false",
                    "-c", "core.untrackedCache=false", "-c", "core.quotePath=true", "-C", str(root)]

            def git(*arguments: str, command=argv) -> tuple[int, bytes]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 5)
                with tempfile.TemporaryFile() as output:
                    completed = subprocess.run([*command, *arguments], stdout=output,
                                               stderr=subprocess.DEVNULL, timeout=remaining,
                                               env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, check=False)
                    output.seek(0)
                    return completed.returncode, output.read(MAX_SOURCE_INDEX_BYTES + 1)

            code, top = git("rev-parse", "--show-toplevel")
            if code or Path(os.fsdecode(top).strip()).resolve() != root:
                return {**result, "reason": "source_repository_unavailable"}
            code, tracked = git("ls-files", "-v", "--stage", "-z", "--error-unmatch", "--", *paths)
            if code or not tracked:
                return {**result, "reason": "source_paths_not_tracked"}
            if (len(tracked) > MAX_SOURCE_INDEX_BYTES or not tracked.endswith(b"\0")
                    or any(not record.startswith((b"H 100644 ", b"H 100755 "))
                           or b" 0\t" not in record
                           for record in tracked[:-1].split(b"\0"))):
                # Git status otherwise hides edits behind assume-unchanged or
                # skip-worktree flags. Symlinks/gitlinks can hide mutable code
                # outside the scope; nested repositories need explicit entries.
                return {**result, "reason": "source_index_not_verifiable"}
            code, status = git("status", "--porcelain=v1", "-z", "--untracked-files=all",
                               "--ignore-submodules=none", "--", *paths)
            if code:
                return {**result, "reason": "source_integrity_query_failed"}
            checked.append({"repository": str(root), "paths": paths})
            if status:
                return {**result, "reason": "configured_control_plane_dirty", "checked": checked,
                        "status_digest": hashlib.sha256(status).hexdigest(),
                        "status_truncated": len(status) > MAX_SOURCE_INDEX_BYTES}
        return {**result, "valid": True, "reason": "configured_control_plane_clean", "checked": checked}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {**result, "reason": "source_integrity_unavailable"}


def observe_board(board: Mapping[str, Any], *, now: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    now = time.time() if now is None else now
    board_id = str(board.get("board_id", board.get("id", ""))).lower()
    reasons: list[str] = []
    source_integrity = _source_integrity(board)
    owner_status = read_json(Path(board["owner_status_path"]))
    expected_birth = _object(_object(owner_status.get("identity")).get("process_birth"))
    owner = process_identity(expected_birth.get("pid"))
    owner_live = birth_matches(owner, expected_birth)
    owner_ready = owner_live and owner_status.get("lifecycle") == "ready"
    if not owner_live:
        reasons.append("owner_process_missing_or_birth_mismatch")
    elif not owner_ready:
        reasons.append("owner_not_ready")
    if owner_live and (condition := _process_condition(owner)):
        reasons.append(f"owner_process_{condition}")
    if owner_live:
        try:
            endpoint = str(board["quack_endpoint"]).removeprefix("quack:")
            host, port = endpoint.rsplit(":", 1)
            with socket.create_connection((host, int(port)), timeout=1):
                pass
        except (OSError, ValueError):
            reasons.append("owner_endpoint_unreachable")
    lanes = []
    fresh_projections = []
    for index in range(int(board["max_lanes"])):
        lane_dir = Path(board["state_root"]) / f"lane-{index}"
        prefix = f"{board_id}_lane_{index}"
        status = read_json(lane_dir / f"{prefix}_supervisor_status.json")
        supervisor = _lane_process(status.get("supervisor_pid"), lane_dir, prefix, board["cwd"])
        daemon = _lane_process(status.get("daemon_pid"), lane_dir, prefix, board["cwd"])
        if daemon and status.get("daemon_process_birth") and not birth_matches(daemon, _object(status["daemon_process_birth"])):
            daemon = {}
        age = _age(status.get("updated_at"), now)
        limit = max(float(board.get("supervisor_heartbeat_stale_seconds", 600)),
                    float(status.get("supervisor_heartbeat_seconds") or 20) * 6)
        lane = {"lane": index, "state_dir": str(lane_dir), "supervisor": _public_identity(supervisor),
                "daemon": _public_identity(daemon), "status_age_seconds": age,
                "status": status.get("status", "absent"), "last_exit_code": status.get("last_exit_code"),
                "last_recycle_reason": status.get("last_recycle_reason", ""),
                "stalled_without_active_worker": status.get("stalled_without_active_worker", False)}
        lanes.append(lane)
        if not supervisor:
            reasons.append(f"lane_{index}_supervisor_missing")
        elif age is None or age > limit:
            reasons.append(f"lane_{index}_supervisor_heartbeat_stale")
        if not daemon:
            reasons.append(f"lane_{index}_daemon_missing")
        # A supervisor heartbeat can advance while its daemon is stopped or
        # waiting in the kernel. Keep the exact live identity and route a
        # persistent condition through the watchdog's existing incident grace,
        # holds and native repair admission; one sample never authorizes a kill.
        for role, identity in (("supervisor", supervisor), ("daemon", daemon)):
            if condition := _process_condition(identity):
                reasons.append(f"lane_{index}_{role}_process_{condition}")
        if status.get("stalled_without_active_worker") is True:
            reasons.append(f"lane_{index}_reports_stalled")
        if status.get("last_exit_code") == 78 and not daemon:
            reasons.append(f"lane_{index}_typed_fail_closed_exit")
        projection = read_json(lane_dir / f"{prefix}_task_state.json")
        projection_age = _age(projection.get("heartbeat_at"), now)
        if (supervisor and daemon and projection.get("projection_complete") is True
                and projection_age is not None and projection_age <= 120):
            fresh_projections.append(projection)
    native, command_error = ({}, "")
    native_status_attempts = 0
    # Several legacy status commands open the authoritative DB directly when no
    # owner is present. Never execute them during an outage.
    if source_integrity["valid"] and owner_ready and board.get("status_argv"):
        native, command_error, native_status_attempts = _status_with_receipt_retry(board, expected_birth)
        if command_error:
            reasons.append(command_error)
        if any(native.get(key) is False for key in ("ready", "healthy", "operational_ready")):
            reasons.append("native_operator_reports_unhealthy")
    authority = _object(native.get("receipt")) if board_id == "aseh" else _object(native.get("task_authority"))
    if board_id == "aseh" and "samples" in authority:
        # ASEH v2 puts task observations inside its admitted two-sample
        # receipt, not on the health wrapper. Never extract a sample from an
        # expired or owner-mismatched receipt rejected by the native reader.
        samples = authority.get("samples")
        admitted = bool(
            native.get("broker_authenticated_receipt") is True
            and authority.get("broker_authenticated") is True
            and isinstance(samples, list) and len(samples) == 2
            and all(
                _object(_object(sample).get("authority")).get("available") is True
                and _object(_object(sample).get("authority")).get("transport") == "quack"
                and _object(_object(sample).get("authority")).get("credential_path") == "sealed_memfd_broker"
                for sample in samples
            )
        )
        authority = dict(samples[-1]["authority"]) if admitted else {}
        task_count = _object(authority.get("snapshot")).get("task_count")
        if type(task_count) is int and task_count >= 0:
            authority["task_count"] = task_count
    database_authority = _database_board_authority(native, board, owner_status, now)
    if native.get("schema") == "ipfs_accelerate_py/agent-supervisor/database-board-status@1":
        if database_authority:
            authority = database_authority
        else:
            authority = {}
            reasons.append("native_database_status_not_admitted")
    source = "native_operator_status" if authority else "unavailable"
    authenticated = bool(authority and (authority.get("authenticated_query") is True
        or (board_id == "aseh" and native.get("broker_authenticated_receipt") is True)
        or (board_id == "doep" and native.get("status_age_seconds", float("inf")) <= 30)
        or authority.get("transport") == "exclusive_owner_authenticated_quack_projection"))
    if (board_id == "spar" and not database_authority
            and authority.get("transport") == "exclusive_owner_authenticated_quack_projection"):
        # The legacy publisher can reuse its cached task snapshot after a
        # failed native read while retaining this transport label. Keep the
        # observation visible, but require a separate admitted native snapshot
        # for any claim of current authenticated state or completion authority.
        authenticated = False
        source = "native_cached_projection_non_authoritative"
    if fresh_projections and (not authority or (board_id == "pcpr" and not authenticated)):
        # PCPR's checkpoint replica can lag by dozens of completions. Fresh
        # daemon projections are preferable for observing activity, never gates.
        authority = max(fresh_projections, key=lambda value: _age(value.get("heartbeat_at"), now) * -1)
        source = "fresh_daemon_database_projection_non_authoritative"
        authenticated = False
    counts = _counts(authority)
    if not counts and board_id == "aseh":
        statuses = authority.get("task_statuses") or authority.get("task_status_by_alias")
        if isinstance(statuses, dict):
            counts = _counts({"task_statuses": statuses})
    blocked_task_ids = _blocked_task_ids(authority)
    blocked = (bool(blocked_task_ids) or native.get("blocked") is True or authority.get("blocked") is True
               or any(counts.get(status, 0) for status in BLOCKED)
               or int(authority.get("blocked_count") or 0) > 0)
    if blocked and not database_authority.get("blocked"):
        reasons.append("board_has_blocked_or_quarantined_tasks")
    if authority.get("unsettled_goal_count"):
        reasons.append("board_has_unsettled_goals")
    if authority and not authenticated:
        reasons.append("task_observation_not_completion_authority")
    # Native receipt reads can wait while workers keep writing. Advance the
    # provider sample clock so output during that wait is not future-dated.
    providers = _provider_busy(lanes, board, now + max(0.0, time.monotonic() - started))
    if source_integrity["configured"] and source_integrity["valid"]:
        source_integrity = _source_integrity(board)
    if not source_integrity["valid"]:
        reasons.append("source_integrity_not_verified")
        health = "degraded"
    elif not owner_live:
        health = "stopped"
    elif blocked:
        health = "blocked"
    elif any("reports_stalled" in reason for reason in reasons) or native.get("stuck") is True:
        health = "stalled"
    elif any(reason != "task_observation_not_completion_authority" for reason in reasons):
        health = "degraded"
    elif not authority:
        health = "unknown"
        reasons.append("task_observation_unavailable")
    else:
        health = "healthy"
    candidate = bool(source_integrity["valid"] and not blocked and counts and sum(counts.values()) > 0
        and authority.get("task_count") == sum(counts.values())
        and all(key in COMPLETED for key in counts))
    result: dict[str, Any] = {
        "schema": SCHEMA, "board_id": board_id, "health": health, "reason_codes": sorted(set(reasons)),
        "progress_token": _progress(authority), "busy": bool(providers), "complete": False,
        "completion_candidate": candidate,
        "details": {"observed_at": datetime.fromtimestamp(now, UTC).isoformat(),
            "owner": _public_identity(owner) if owner_live else {}, "owner_ready": owner_ready,
            "lanes": lanes, "providers": providers, "task_counts": counts, "progress_source": source,
            "authenticated_task_observation": authenticated,
            "unsettled_goal_count": authority.get("unsettled_goal_count"),
            "completion_receipt_count": authority.get("completion_receipt_count"),
            "task_count": authority.get("task_count", sum(counts.values()) if counts else None),
            "event_cursor": authority.get("event_cursor"),
            "native_status_attempts": native_status_attempts,
            "blocked_task_ids": blocked_task_ids,
            "source_heads": _source_heads(board),
            "source_integrity": source_integrity,
            "completion_gate": "separate_authoritative_closeout_verification_required"},
    }
    if health == "stopped" and board.get("ensure_argv"):
        result["recovery_action"] = "ensure"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--board", required=True)
    args = parser.parse_args(argv)
    inventory = read_json(args.inventory)
    board = next((item for item in inventory.get("boards", [])
                  if str(item.get("board_id", item.get("id", ""))).lower() == args.board.lower()), None)
    if board is None:
        parser.error("board is absent from inventory")
    try:
        result = observe_board(board)
    except Exception as exc:
        result = {"schema": SCHEMA, "board_id": args.board.lower(), "health": "unknown",
                  "reason_codes": ["probe_exception"], "progress_token": "", "busy": False,
                  "complete": False, "details": {"error_type": type(exc).__name__}}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
