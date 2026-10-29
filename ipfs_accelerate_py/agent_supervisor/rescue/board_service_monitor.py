"""Bounded external health monitor for one explicitly configured user service.

Run this file directly with the system Python; it has no package dependencies::

    python3 board_service_monitor.py --config /absolute/board-monitor.json --apply

Without ``--apply`` it only records observations and proposed actions. The only
repair authority is ``systemctl --user restart <exact configured .service>``.
Task state, dependencies, locks, and worker processes are never changed here.
An active worker may emit logs to prove liveness, but logs cannot extend its
absolute ``progress_timeout_seconds`` deadline. Polling is not task progress.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


_PROGRESS_EVENTS = frozenset({
    "task_completed", "task_marked_completed",
    "implementation_started", "implementation_finished", "implementation_completed",
    "merge_started", "merge_finished", "merge_completed",
    "validation_started", "validation_finished", "validation_completed",
})
_FINISH_EVENTS = frozenset({"implementation_finished", "implementation_completed"})
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_EVENT_BYTES = 4 * 1024 * 1024


def load_config(path: Path) -> dict[str, Any]:
    config, error = _read_json(str(path))
    if error:
        raise ValueError("monitor configuration unreadable: " + error)
    if not isinstance(config, dict):
        raise ValueError("monitor configuration must be a JSON object")
    for key in ("service", "state_dir", "state_prefix", "board_namespace"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*\.service", config["service"]):
        raise ValueError("service must name one exact systemd .service unit")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["state_prefix"]):
        raise ValueError("state_prefix must be a plain filename prefix")
    for key, default in {
        "startup_grace_seconds": 120, "heartbeat_timeout_seconds": 300,
        "progress_timeout_seconds": 7200, "restart_cooldown_seconds": 600,
        "max_restarts_per_hour": 3, "native_recovery_timeout_seconds": 600,
    }.items():
        value = config.setdefault(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be a positive finite number")
    if int(config["max_restarts_per_hour"]) != config["max_restarts_per_hour"]:
        raise ValueError("max_restarts_per_hour must be an integer")
    for key, default in (("task_shard_count", 1), ("task_shard_index", 0)):
        value = config.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if not 0 <= config["task_shard_index"] < config["task_shard_count"]:
        raise ValueError("task_shard_index must be within task_shard_count")
    state_dir = Path(config["state_dir"]).expanduser()
    if not state_dir.is_absolute():
        state_dir = path.resolve().parent / state_dir
    config["state_dir"] = str(state_dir.resolve())
    scope = hashlib.sha256((config["service"] + "\0" + config["board_namespace"]).encode()).hexdigest()[:12]
    stem = f"{config['state_prefix']}_service_monitor_{scope}"
    defaults = {
        "native_state_path": f"{config['state_prefix']}_task_state.json",
        "checkpoint_path": f"{config['state_prefix']}_task_state.event-driven-checkpoint.json",
        "status_path": f"{config['state_prefix']}_supervisor_status.json",
        "heartbeat_path": f"{config['state_prefix']}_database_daemon_pass_heartbeat.json",
        "event_path": f"{config['state_prefix']}_events.jsonl",
        "health_path": stem + "_health.json", "actions_path": stem + "_actions.jsonl",
        "monitor_state_path": stem + "_state.json", "monitor_lock_path": stem + ".lock",
    }
    for key, default in defaults.items():
        raw = Path(config.get(key) or default).expanduser()
        config[key] = str(raw if raw.is_absolute() else state_dir / raw)
    return config


def _open_regular(path: str | Path, *, append: bool = False):
    """Reject peer-dependent devices before reading, writing, or taking a lock."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT if append else os.O_RDONLY
    fd = os.open(path, flags | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "monitor path must be a regular file", os.fspath(path))
        return os.fdopen(fd, "ab" if append else "rb")
    except BaseException:
        os.close(fd)
        raise


def _read_json(path: str) -> tuple[dict[str, Any], str | None]:
    try:
        with _open_regular(path) as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("JSON exceeds monitor size limit")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value, None
    except FileNotFoundError:
        return {}, "missing"
    except (OSError, ValueError, RecursionError) as exc:
        return {}, str(exc)


def _timestamp(value: Any, now: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = (float(value) if isinstance(value, (float, int))
                  else datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
        return parsed if math.isfinite(parsed) and 0 < parsed <= now + 60 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _latest(payload: dict[str, Any], keys: tuple[str, ...], now: float) -> float | None:
    values = [_timestamp(payload.get(key), now) for key in keys]
    return max((value for value in values if value is not None), default=None)


def _mtime(path: str, now: float) -> float | None:
    try:
        value = Path(path).stat().st_mtime
        return value if 0 < value <= now + 60 else None
    except OSError:
        return None


def _event_tail(config: dict[str, Any], now: float) -> tuple[list[dict[str, Any]], str | None]:
    try:
        with _open_regular(config["event_path"]) as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            start = max(0, size - _MAX_EVENT_BYTES)
            stream.seek(start)
            raw = stream.read(size - start)
            if start:
                raw = raw.partition(b"\n")[2]  # a partial first record is not evidence
            lines = raw.splitlines()
    except OSError as exc:
        return [], str(exc)
    events = []
    error = None
    for line in lines:
        try:
            event = json.loads(line)
        except RecursionError:
            error = "event_json_nesting_limit"
            continue
        except (ValueError, UnicodeError):
            continue  # a writer's partial last line is harmless
        if not isinstance(event, dict):
            continue
        namespace = event.get("board_namespace")
        if namespace and namespace != config["board_namespace"]:
            continue
        timestamp = _latest(event, ("timestamp", "updated_at", "at"), now)
        if timestamp is not None:
            events.append({**event, "_timestamp": timestamp})
    return sorted(events, key=lambda event: event["_timestamp"]), error


def _unresolved_retry_deferrals(events: list[dict[str, Any]],
                                native: dict[str, Any], now: float) -> list[dict[str, Any]]:
    """Retain bounded typed no-dispatch fences, never model prose or selections."""
    pending: dict[tuple[str, str], dict[str, Any]] = {}

    def identity(event: dict[str, Any]) -> tuple[str, str] | None:
        for field in ("canonical_task_cid", "canonical_task_key", "task_id"):
            value = event.get(field)
            if isinstance(value, str) and value.strip():
                return field, value[:512]
        return None

    for event in events:
        key = identity(event)
        if key is None:
            continue
        event_type = event.get("type")
        if (event_type == "implementation_retry_deferred"
                and event.get("attempt_consumed") is False
                and isinstance(event.get("reason"), str) and event["reason"]):
            contract = event.get("provider_task_contract")
            contract = contract if isinstance(contract, dict) else {}
            missing = contract.get("missing_validation_executables")
            missing = [name for name in missing[:32] if isinstance(name, str)
                       and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]{0,127}", name)] if isinstance(missing, list) else []
            unavailable = contract.get("unavailable_validation_file_pins")
            unavailable = unavailable if isinstance(unavailable, list) else []
            file_inputs = [{"path": row["path"][:1024], "reason": row["reason"][:128]}
                           for row in unavailable[:8] if isinstance(row, dict)
                           and isinstance(row.get("path"), str) and isinstance(row.get("reason"), str)]
            backoff = event.get("backoff_seconds")
            backoff = backoff if (isinstance(backoff, (int, float))
                and not isinstance(backoff, bool) and math.isfinite(backoff) and backoff >= 0) else 0
            pending[key] = {
                "task_id": str(event.get("task_id") or "")[:256],
                "canonical_task_cid": str(event.get("canonical_task_cid") or "")[:512],
                "reason": event["reason"][:256],
                "contract_fingerprint": str(event.get("contract_fingerprint") or "")[:256],
                "missing_validation_executables": missing,
                "unavailable_validation_file_pins": file_inputs,
                "unavailable_validation_file_pin_count": len(unavailable),
                "file_pin_evidence_truncated": len(unavailable) > len(file_inputs),
                "deferred_at_epoch": event["_timestamp"],
                "retry_after_epoch": event["_timestamp"] + backoff,
            }
            # The event tail is byte-bounded; additionally bound retained tasks.
            if len(pending) > 128:
                del pending[min(pending, key=lambda item: pending[item]["deferred_at_epoch"])]
        elif (event_type in {"implementation_started", "implementation_completed",
                            "task_completed", "task_marked_completed"}
              or (event_type == "implementation_finished" and event.get("returncode") == 0)):
            pending.pop(key, None)

    completed = native.get("completed_task_ids")
    completed = set(completed) if isinstance(completed, list) and all(isinstance(x, str) for x in completed) else set()
    statuses = native.get("task_statuses")
    statuses = statuses if isinstance(statuses, dict) else {}
    native_at = _latest(native, ("heartbeat_at", "updated_at"), now) or 0
    return [item for item in pending.values() if not (
        native_at >= item["deferred_at_epoch"] and
        (item["task_id"] in completed or statuses.get(item["task_id"]) == "completed")
    )]


# Reuse the canonical bounded, birth-bound Linux process observer without
# importing the supervisor package or invoking its recovery mechanisms.
_PROCESS_SPEC = importlib.util.spec_from_file_location(
    "_board_monitor_process_owner", Path(__file__).resolve().parents[1] / "runtime/native_graceful_recovery.py")
_PROCESS_OWNER = importlib.util.module_from_spec(_PROCESS_SPEC)
sys.modules[_PROCESS_SPEC.name] = _PROCESS_OWNER
_PROCESS_SPEC.loader.exec_module(_PROCESS_OWNER)


def _thread_ids(root: Path) -> set[int]:
    tids: set[int] = set()
    with os.scandir(root / "task") as entries:
        for entry in entries:
            if not entry.name.isascii() or not entry.name.isdecimal():
                raise ValueError("process_thread_invalid")
            tids.add(int(entry.name))
            if len(tids) > 256:
                raise ValueError("process_thread_bound")
    if not tids:
        raise ValueError("process_threads_unavailable")
    return tids


def _thread_children(root: Path) -> list[int]:
    """Union every thread's children; partial observations never prove idle."""
    tids = _thread_ids(root)
    children: set[int] = set()
    for tid in sorted(tids):
        values = _PROCESS_OWNER._read_proc(root / "task" / str(tid) / "children", 8192).decode("ascii").split()
        for value in values:
            if not value.isdecimal() or int(value) <= 1:
                raise ValueError("process_child_invalid")
            children.add(int(value))
            if len(children) > 256:
                raise ValueError("process_child_bound")
    if tids != _thread_ids(root):
        raise ValueError("process_threads_changed")
    return sorted(children)


def _process_snapshot(pid: int) -> dict[str, Any] | None:
    """Read exact Linux birth and all-thread children; unknown grants no repair."""
    try:
        binding = _PROCESS_OWNER.observe_process(pid)
        children = _thread_children(Path("/proc") / str(pid))
        _PROCESS_OWNER.require_exact_process(binding)
        if not binding.boot_id:
            return None
        return {"pid": pid, "birth_ticks": binding.birth, "boot_id": binding.boot_id,
                "children": children}
    except (OSError, ValueError, IndexError, _PROCESS_OWNER.GracefulRecoveryUnverified):
        return None


def _owned_work_observation(service: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """A quiet validation child is work, without renewing any task deadline."""
    try:
        pid = int(service["MainPID"])
        daemon_pid = evidence["native_daemon_pid"]
        if (evidence.get("native_supervisor_pid") != pid
                or evidence.get("native_daemon_pid_alive") is not True
                or type(daemon_pid) is not int or daemon_pid <= 1):
            raise ValueError("owned_work_identity_unverified")
        parent = _PROCESS_OWNER.observe_process(pid)
        daemon = _PROCESS_OWNER.observe_process(daemon_pid)
        if daemon.parent != pid or daemon_pid not in _thread_children(Path("/proc") / str(pid)):
            raise ValueError("owned_daemon_binding_unverified")
        children = _thread_children(Path("/proc") / str(daemon_pid))
        pending = [(child, daemon_pid, 1) for child in children]
        samples = []
        bindings = [(daemon, children)]
        seen = {pid, daemon_pid}
        while pending:
            child, expected_parent, depth = pending.pop()
            if child in seen or len(seen) >= 50 or depth > 8:
                raise ValueError("owned_descendant_bound_or_cycle")
            seen.add(child)
            binding = _PROCESS_OWNER.observe_process(child)
            if binding.parent != expected_parent:
                raise ValueError("owned_child_binding_unverified")
            fields = _PROCESS_OWNER._stat(child)
            io = dict(line.split(":", 1) for line in _PROCESS_OWNER._read_proc(
                Path("/proc") / str(child) / "io", 8192).decode("ascii").splitlines())
            samples.append({"pid": child, "birth_ticks": binding.birth,
                            "parent_pid": binding.parent,
                            "cpu_ticks": int(fields[11]) + int(fields[12]),
                            "io_bytes": sum(int(io[key]) for key in ("rchar", "wchar", "read_bytes", "write_bytes"))})
            descendants = _thread_children(Path("/proc") / str(child))
            bindings.append((binding, descendants))
            pending.extend((descendant, child, depth + 1) for descendant in descendants)
        # A new child, a reparenting or a partial thread observation cannot
        # establish a stable tree for the final restart decision.
        for binding, descendants in bindings:
            if descendants != _thread_children(Path("/proc") / str(binding.pid)):
                raise ValueError("owned_descendants_changed")
            _PROCESS_OWNER.require_exact_process(binding)
        samples.sort(key=lambda item: item["pid"])
        children = [sample["pid"] for sample in samples]
        _PROCESS_OWNER.require_exact_process(daemon)
        _PROCESS_OWNER.require_exact_process(parent)
        return {"verified": True, "children": children, "samples": samples,
                "daemon_pid": daemon_pid, "daemon_birth_ticks": daemon.birth,
                "supervisor_pid": pid, "supervisor_birth_ticks": parent.birth, "boot_id": parent.boot_id}
    except (KeyError, TypeError, OSError, ValueError, IndexError, _PROCESS_OWNER.GracefulRecoveryUnverified):
        return {"verified": False, "reason": "owned_work_observation_unverified"}


def _native_recovery_observation(config: dict[str, Any], service: dict[str, Any],
                                 evidence: dict[str, Any], previous: dict[str, Any],
                                 now: float, monotonic: float) -> dict[str, Any]:
    """Bound the parent's own recovery, with positive quiescence before escalation."""
    result = {"tracked": False, "expired": False, "quiescent": False}
    if evidence.get("native_daemon_pid_alive") is not False:
        return result
    try:
        pid = int(service.get("MainPID", 0))
        started = float(service.get("ExecMainStartTimestampMonotonic", 0)) / 1_000_000
        status_at = evidence.get("native_status_at_epoch")
        run_id = evidence.get("native_run_id")
        retries = evidence.get("native_restart_count")
        if (pid <= 1 or evidence.get("native_supervisor_pid") != pid
                or service.get("ActiveState") != "active" or started <= 0
                or not 0 <= monotonic - started
                or type(status_at) not in (int, float) or not math.isfinite(status_at)
                or not max(now - config["heartbeat_timeout_seconds"], now - (monotonic - started) - 1) <= status_at <= now
                or type(run_id) is not str or not 0 < len(run_id) <= 256
                or type(retries) is not int or retries < 0):
            return {**result, "reason": "native_recovery_identity_unverified"}
        process = _process_snapshot(pid)
        if process is None:
            return {**result, "reason": "native_recovery_process_unverified"}
        identity = {key: process[key] for key in ("pid", "birth_ticks", "boot_id")}
        old = previous.get("native_recovery")
        old = old if type(old) is dict else {}
        since = old.get("since_at_epoch") if old.get("identity") == identity else now
        if type(since) not in (int, float) or not math.isfinite(since) or not 0 < since <= now:
            since = now
        work = evidence.get("last_work_event_at_epoch") or 0
        if type(work) not in (int, float) or not math.isfinite(work):
            return {**result, "reason": "native_recovery_progress_unverified"}
        if old.get("identity") == identity and work > old.get("work_event_at_epoch", work):
            since = now
        safety = evidence.get("native_unavailable_child_recovery")
        safety = safety if type(safety) is dict else {}
        birth = safety.get("supervisor_process_birth")
        birth = birth if type(birth) is dict else {}
        proof_current = (safety.get("schema") == "ipfs-accelerate/unavailable-child-recovery-safety@1"
            and safety.get("safe") is True and safety.get("blockers") == []
            and birth.get("pid") == pid and birth.get("start_time_ticks") == process["birth_ticks"]
            and birth.get("boot_id") == process["boot_id"]
            and _timestamp(safety.get("observed_at"), now) == status_at)
        first_deadline = old.get("maintenance_deadline_at_epoch") if old.get("identity") == identity and since != now else None
        window = evidence.get("native_maintenance_window")
        if first_deadline is None and evidence.get("native_managed_status") == "agentic_maintenance_started":
            if type(window) is not dict or window.get("schema") != "ipfs_accelerate_py/agent-supervisor/supervisor-maintenance@1" or window.get("phase") != "running" or window.get("finished_at") is not None:
                return {**result, "reason": "native_maintenance_window_unverified"}
            begin = _timestamp(window.get("started_at"), now)
            deadline_value = window.get("deadline_at")
            if type(deadline_value) is not str or not 0 < len(deadline_value) <= 128:
                return {**result, "reason": "native_maintenance_deadline_unverified"}
            deadline_value = datetime.fromisoformat(deadline_value.replace("Z", "+00:00"))
            if deadline_value.tzinfo is None or deadline_value.utcoffset() is None:
                return {**result, "reason": "native_maintenance_deadline_unverified"}
            first_deadline = deadline_value.timestamp()
            duration = window.get("timeout_seconds")
            if (begin is None or first_deadline is None or type(duration) not in (int, float)
                    or not math.isfinite(duration) or duration <= 0
                    or not math.isclose(first_deadline - begin, duration, abs_tol=0.000001)
                    or not begin <= status_at):
                return {**result, "reason": "native_maintenance_deadline_unverified"}
        if first_deadline is not None and (type(first_deadline) not in (int, float)
                or not math.isfinite(first_deadline) or first_deadline <= 0):
            return {**result, "reason": "native_maintenance_deadline_unverified"}
        checks = {
            "native_owner_quiescence_not_proven": proof_current,
            "native_worker_count_not_zero": type(safety.get("active_worker_count")) is int and safety["active_worker_count"] == 0,
            "native_worker_pids_not_empty": safety.get("active_worker_pids") == [],
            "native_worker_descendants_not_zero": type(safety.get("worker_descendant_count")) is int and safety["worker_descendant_count"] == 0,
            "native_phase_not_quiescent": safety.get("worker_phase") in {"", "idle", "selected"},
            "active_task_present": evidence.get("active") is False,
            "supervisor_has_children": process["children"] == [],
            "implementation_not_enabled": evidence.get("implement") is True,
        }
        timeout = config["native_recovery_timeout_seconds"]
        deadline = max(since + timeout, first_deadline or 0)
        return {"tracked": True, "identity": identity, "since_at_epoch": since,
                "work_event_at_epoch": work, "maintenance_deadline_at_epoch": first_deadline,
                "age_seconds": now - since, "timeout_seconds": timeout,
                "deadline_at_epoch": deadline, "run_id": run_id, "restart_count": retries,
                "expired": now >= deadline, "quiescent": all(checks.values()),
                "quiescence_blockers": [key for key, good in checks.items() if not good]}
    except (TypeError, ValueError, OverflowError):
        return {**result, "reason": "native_recovery_observation_invalid"}


def collect_evidence(config: dict[str, Any], now: float) -> dict[str, Any]:
    native, native_error = _read_json(config["native_state_path"])
    source = config["native_state_path"]
    if not native:
        native, checkpoint_error = _read_json(config["checkpoint_path"])
        if native:
            source = config["checkpoint_path"]
    else:
        checkpoint_error = None
    # Some legacy checkpoints wrap the state projection.
    if isinstance(native.get("projection"), dict):
        native = native["projection"]
    for key in ("state", "task_state"):
        if isinstance(native.get(key), dict):
            native = native[key]
            break
    status, status_error = _read_json(config["status_path"])
    heartbeat, heartbeat_error = _read_json(config["heartbeat_path"])
    events, event_error = _event_tail(config, now)
    # Preserve actual execution heartbeat evidence before projecting newer
    # daemon polls. A fresh queue pass proves the parent loop is alive, not
    # that a historical implementation_started event still has a live worker.
    native_worker_heartbeat = (
        _latest(native, ("heartbeat_at", "updated_at"), now)
        if native.get("implementation_in_progress") or native.get("active_phase") else None
    )
    passes = [event for event in events if event.get("type") == "daemon_pass"]
    if passes and passes[-1]["_timestamp"] > (_latest(native, ("heartbeat_at",), now) or 0):
        # Event-driven mode deliberately leaves the checkpoint unchanged until
        # its next checkpoint. The latest pass contains current queue counts.
        native = dict(native)
        for kind in ("ready", "selectable_ready", "eligible_ready", "completed", "blocked", "waiting"):
            key = kind + "_count"
            if key in passes[-1]:
                native[key] = passes[-1][key]
                native.pop(kind + "_task_ids", None)
        if "task_count" in passes[-1]:
            native["task_count"] = passes[-1]["task_count"]
        elif isinstance(passes[-1].get("execution_slice_task_statuses"), dict):
            native["task_count"] = len(passes[-1]["execution_slice_task_statuses"])
        if "selection_idle_reason" in passes[-1]:
            native["selection_idle_reason"] = passes[-1]["selection_idle_reason"]
        if isinstance(passes[-1].get("execution_slice_task_statuses"), dict):
            native["task_statuses"] = passes[-1]["execution_slice_task_statuses"]
            native["task_cids_by_id"] = passes[-1].get("execution_slice_task_cids_by_id", {})
        native["heartbeat_at"] = passes[-1]["_timestamp"]
    namespace = native.get("board_namespace")
    mismatch = bool(namespace and namespace != config["board_namespace"])
    if mismatch:
        native = {}
    progress = [_latest(native, ("last_progress_at", "last_implementation_finished_at",
                                "last_merge_finished_at"), now)]
    work_events = [event["_timestamp"] for event in events if event.get("type") in _PROGRESS_EVENTS]
    progress.extend(work_events)
    heartbeat_times = [_latest(payload, ("heartbeat_at", "updated_at"), now)
                       for payload in (native, status, heartbeat)]
    worker_heartbeat_times = [native_worker_heartbeat,
                             _latest(heartbeat, ("heartbeat_at", "updated_at"), now)]
    selected_id = str(passes[-1].get("active_task_id") or "") if passes else str(native.get("active_task_id") or "")
    # active_task_id alone often denotes a selected candidate while dispatch is
    # disabled or waiting. It is not proof that an attempt actually started.
    active = bool(native.get("implementation_in_progress") or native.get("active_phase"))
    active_id = str(native.get("active_task_id") or "") if active else ""
    active_started = (_latest(native, ("active_task_started_at", "last_implementation_started_at"), now)
                      if active else None)
    active_log = str(native.get("active_log_path") or "")
    # Fresh event evidence supersedes an old event-driven checkpoint.
    starts = [event for event in events if event.get("type") == "implementation_started"]
    if starts:
        start = starts[-1]
        finished = any(event.get("type") in (_FINISH_EVENTS | {"implementation_lock_cleared"})
                       and event["_timestamp"] >= start["_timestamp"]
                       and event.get("task_id") == start.get("task_id") for event in events)
        if not active_started or start["_timestamp"] >= active_started:
            active = not finished
            active_id = str(start.get("task_id") or "") if active else ""
            active_started = start["_timestamp"] if active else None
            active_log = str(start.get("log_path") or start.get("implementation_log_path") or active_log)
    if active_log and not Path(active_log).is_absolute():
        # Relative paths need an explicit repository root, never cwd inference.
        active_log = str(Path(config["repo_root"]) / active_log) if config.get("repo_root") else ""
    log_time = _mtime(active_log, now) if active and active_log else None
    statuses = native.get("task_statuses")
    counts = Counter(str(value).lower() for value in statuses.values()) if isinstance(statuses, dict) else Counter()

    def count(kind: str) -> int:
        ids = native.get(kind + "_task_ids")
        value = native.get(kind + "_count")
        if isinstance(ids, list):
            return len(ids)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value)
        return counts[kind]

    global_ready = count("ready")
    selectable = count("selectable_ready") if ("selectable_ready_count" in native
                 or "selectable_ready_task_ids" in native) else None
    sharded = config.get("task_shard_count", 1) > 1

    # Ready counts include retry-cooling tasks on some native projections.
    # Prove the exact lane identities before classifying all ready work fenced;
    # a partial fence must not conceal a genuine stalled ready task.
    ready_ids = []
    if isinstance(statuses, dict):
        ready_ids = sorted(task_id for task_id, task_status in statuses.items()
                           if isinstance(task_id, str) and str(task_status).lower() == "ready"
                           and (not sharded or int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
                                % config["task_shard_count"] == config.get("task_shard_index", 0)))
    lane_ready = selectable if sharded and selectable is not None else global_ready
    deferrals = _unresolved_retry_deferrals(events, native, now)
    cids = native.get("task_cids_by_id")
    cids = cids if isinstance(cids, dict) else {}
    fenced = {item["task_id"] for item in deferrals if item.get("task_id") in ready_ids
              and (not item.get("canonical_task_cid")
                   or cids.get(item["task_id"]) == item["canonical_task_cid"])}
    all_ready_fenced = bool(ready_ids) and len(ready_ids) == lane_ready and fenced == set(ready_ids)

    return {
        "state_source": source, "state_available": bool(native), "namespace_mismatch": mismatch,
        "native_state_error": native_error, "checkpoint_error": checkpoint_error,
        "status_error": status_error, "heartbeat_error": heartbeat_error, "event_error": event_error,
        "last_heartbeat_at_epoch": max((v for v in heartbeat_times if v is not None), default=None),
        "worker_heartbeat_at_epoch": max((v for v in worker_heartbeat_times if v is not None), default=None),
        "last_progress_at_epoch": max((v for v in progress if v is not None), default=None),
        "last_work_event_at_epoch": max(work_events, default=None),
        "active_task_id": active_id, "active": active, "active_started_at_epoch": active_started,
        "selected_task_id": selected_id,
        "active_log_path": active_log, "active_log_at_epoch": log_time,
        "ready_count": lane_ready,
        "global_ready_count": global_ready, "selectable_ready_count": selectable,
        "sharded": sharded, "task_shard_index": config.get("task_shard_index", 0),
        "task_shard_count": config.get("task_shard_count", 1), "waiting_count": count("waiting"),
        "blocked_count": count("blocked"), "completed_count": count("completed"),
        "task_count": native.get("task_count") or sum(counts.values()),
        "selection_idle_reason": native.get("selection_idle_reason", ""),
        "unresolved_retry_deferrals": deferrals,
        "all_ready_tasks_retry_fenced": all_ready_fenced,
        "retry_fenced_ready_task_ids": sorted(fenced),
        "native_progress_status": status.get("implementation_progress_status"),
        "native_managed_status": status.get("status"),
        "native_status_at_epoch": _latest(status, ("updated_at", "heartbeat_at"), now),
        "native_daemon_pid_alive": status.get("daemon_pid_alive"),
        "native_daemon_pid": status.get("daemon_pid"),
        "native_last_exit_code": status.get("last_exit_code"),
        "native_restart_count": status.get("restart_count"),
        "native_idle_ready_seconds": status.get("idle_ready_seconds"),
        "native_recycle_safe": status.get("idle_ready_recycle_safe"),
        "native_supervisor_pid": status.get("supervisor_pid"),
        "native_run_id": status.get("run_id"),
        "native_active_worker_count": status.get("active_worker_count"),
        "native_active_worker_pids": status.get("active_worker_pids"),
        "native_worker_descendant_count": status.get("worker_descendant_count"),
        "native_worker_phase": status.get("worker_phase"),
        "native_recycle_blockers": status.get("idle_ready_recycle_blockers"),
        "native_unavailable_child_recovery": status.get("unavailable_child_recovery"),
        "native_maintenance_window": status.get("supervisor_maintenance"),
        "implement": status.get("implementation_enabled", status.get("implement",
                     status.get("config", {}).get("implement", config.get("implement"))
                     if isinstance(status.get("config"), dict) else config.get("implement"))),
        "last_event_type": events[-1].get("type") if events else None,
        "last_event_at_epoch": events[-1]["_timestamp"] if events else None,
    }


def systemd_show(service: str) -> dict[str, Any]:
    command = ["systemctl", "--user", "show", service, "--no-pager",
               "--property=ActiveState,SubState,MainPID,ExecMainStartTimestampMonotonic,LoadState"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    if result.returncode:
        return {"error": result.stderr.strip() or f"systemctl exited {result.returncode}"}
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _service_identity(service: dict[str, Any]) -> tuple[str, int, int] | None:
    """Bind the exact systemd main PID to its monotonic service birth.

    A PID alone can be reused. A running service with no verified birth cannot
    authorize recovery from a previous observation. A never-started inactive
    service legitimately has both PID and birth zero.
    """
    if service.get("error") or service.get("LoadState") != "loaded":
        return None
    pid, birth = service.get("MainPID"), service.get("ExecMainStartTimestampMonotonic")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9]{1,24}", value) is None
           for value in (pid, birth)):
        return None
    pid, birth = int(pid), int(birth)
    if pid > 0 and birth == 0:
        return None
    state = service.get("ActiveState")
    return (state, pid, birth) if isinstance(state, str) and state else None


def evaluate_health(config: dict[str, Any], service: dict[str, Any], evidence: dict[str, Any],
                    previous: dict[str, Any], now: float, monotonic: float) -> dict[str, Any]:
    """Separate liveness, useful progress, intentional idle, and repairability."""
    result = {"healthy": False, "restart_eligible": False, "status": "unknown", "reason": ""}

    def decision(status: str, reason: str, healthy: bool = False, restart: bool = False) -> dict[str, Any]:
        return {**result, "status": status, "reason": reason, "healthy": healthy, "restart_eligible": restart}

    if service.get("error"):
        return decision("observation_failed", service["error"])
    if evidence.get("namespace_mismatch"):
        return decision("scope_mismatch", "Native state belongs to a different board; no repair attempted")
    if service.get("LoadState") == "not-found":
        return decision("service_missing", "Configured service is not installed")
    if service.get("ActiveState") not in {"active", "activating", "reloading"}:
        return decision("service_dead", "Configured service is inactive or failed", restart=True)
    try:
        started_mono = float(service.get("ExecMainStartTimestampMonotonic", 0)) / 1_000_000
        age = monotonic - started_mono if started_mono > 0 else None
        main_pid = int(service.get("MainPID", 0))
    except (TypeError, ValueError):
        return decision("observation_failed", "systemd returned invalid process metadata")
    in_grace = age is not None and 0 <= age < config["startup_grace_seconds"]
    native_status_at = evidence.get("native_status_at_epoch") or 0
    native_exit = evidence.get("native_last_exit_code")
    current_native_status = (
        native_status_at >= now - config["heartbeat_timeout_seconds"]
        and (age is None or native_status_at >= now - age - 1)
    )
    recovery = evidence.get("native_recovery") or {}
    if recovery.get("expired") is True:
        if recovery.get("quiescent") is True:
            return decision("native_recovery_stalled", "The same birth-bound parent exhausted its native recovery grace with no owned work", restart=True)
        return decision("native_recovery_blocked", "Native recovery exceeded its grace but positive quiescence is unproven; no restart authorized")
    if recovery.get("tracked") is True:
        return decision("native_child_restarting", "Birth-bound parent recovery is within its grace; no managed child is currently available")
    if (current_native_status and evidence.get("native_daemon_pid_alive") is False
            and evidence.get("native_managed_status") in {
                "max_restarts_reached", "child_exited", "launch_failed"}):
        return decision("native_child_restarting", "Native implementation child is unavailable; its parent owns bounded recovery, including clean child exits and exhausted inner restart loops")
    if (current_native_status and evidence.get("native_daemon_pid_alive") is False
            and isinstance(native_exit, int) and not isinstance(native_exit, bool) and native_exit != 0
            and evidence.get("native_managed_status") in {
                "restarting", "failed", "restart_budget_exhausted", "restart_limit_reached"}):
        return decision("native_child_restarting", "Native implementation child exited unsuccessfully and is unavailable; its bounded supervisor recovery is in progress")
    if in_grace:
        return decision("starting", "Service is within its configured startup grace", healthy=True)
    if main_pid <= 0:
        return decision("service_dead", "Active service has no main process after startup grace", restart=True)
    # Do not restart deliberately disabled implementation in an endless loop.
    if evidence.get("implement") is False:
        return decision("report_only", "Supervisor explicitly reports implement=false; task execution is disabled")
    fresh_after = now - config["heartbeat_timeout_seconds"]
    heartbeat = evidence.get("last_heartbeat_at_epoch") or 0
    log_time = evidence.get("active_log_at_epoch") or 0
    progress = evidence.get("last_progress_at_epoch") or 0
    first_seen = previous.get("first_seen_at_epoch", now)
    if evidence.get("active"):
        start = evidence.get("active_started_at_epoch")
        if start is not None and now - start > config["progress_timeout_seconds"]:
            return decision("worker_deadline_exceeded", "Active task exceeded its absolute progress deadline", restart=True)
        if max(evidence.get("worker_heartbeat_at_epoch") or 0, log_time, progress) < fresh_after:
            return decision("stale_worker", "Active worker has no fresh heartbeat, progress event, or worker log", restart=True)
        if start is None:
            return decision("active_start_unknown", "Active work has no start timestamp, so its deadline cannot be verified")
        return decision("working", "Active task has fresh liveness evidence and is within its absolute deadline", healthy=True)
    if heartbeat < fresh_after and progress < fresh_after:
        # A newly installed monitor without birth metadata cannot assume stale.
        if age is None and now - first_seen < config["startup_grace_seconds"]:
            return decision("observing", "Waiting through startup grace because systemd start time is unavailable", healthy=True)
        return decision("stale_heartbeat", "No fresh native heartbeat or meaningful work event", restart=True)
    if current_native_status and evidence.get("native_progress_status") == "backpressure":
        return decision("backpressure", "Native supervisor reports a retry, quota, or resource fence; waiting for its bounded recovery")
    idle_reason = str(evidence.get("selection_idle_reason") or "")
    if idle_reason.startswith(("implementation_retry_deferred:", "resource_claim_deferred:")) or idle_reason in {
        "provider_capacity_backoff", "all_selectable_ready_tasks_reached_max_task_attempts",
        "all_selectable_ready_tasks_blocked_on_attempt_accounting_evidence",
        "all_selectable_ready_tasks_deferred_by_resource_claim",
        "all_selectable_ready_tasks_deprioritized_as_off_mission",
        "no_eligible_ready_tasks_after_selection_filters", "virgin_transfer_rendezvous_pending",
    }:
        return decision("backpressure", "Native selection is fenced: " + idle_reason)
    if evidence.get("sharded") and evidence.get("selectable_ready_count") is None:
        return decision("shard_backlog_unknown", "Sharded lane has no native selectable-ready count; global backlog does not establish lane work")
    if evidence.get("all_ready_tasks_retry_fenced"):
        return decision("backpressure", "Every native ready task in this lane has an unresolved no-dispatch retry fence")
    if evidence.get("ready_count", 0):
        # Native last_progress_at can advance during a poll/selection pass.
        # Only a real attempt/work event can extend the queue deadline.
        baseline = max(evidence.get("last_work_event_at_epoch") or 0,
                       previous.get("ready_since_at_epoch", now - age if age is not None else now))
        if now - baseline > config["progress_timeout_seconds"]:
            return decision("ready_backlog_stalled", "Ready work has made no meaningful progress before its deadline", restart=True)
        return decision("ready", "Ready work remains and progress is within its deadline", healthy=True)
    if evidence.get("unresolved_retry_deferrals"):
        reasons = sorted({str(item.get("reason") or "unknown")
                          for item in evidence["unresolved_retry_deferrals"]})
        return decision("backpressure", "Lane has unresolved no-dispatch retry fences: " + ", ".join(reasons))
    if evidence.get("sharded") and evidence.get("global_ready_count", 0):
        if idle_reason == "no_shard_selectable_ready_tasks":
            return decision("shard_idle", "Native selection confirms this lane has no selectable work; remaining ready work belongs to other lanes", healthy=True)
        return decision("shard_backlog_unknown", "Lane has no selectable work but lacks a native reason proving peer-only backlog")
    if evidence.get("blocked_count", 0) or evidence.get("waiting_count", 0):
        return decision("dependencies_waiting", "Tasks report blocked or waiting dependencies; restart cannot resolve their contracts")
    if not evidence.get("state_available"):
        return decision("state_unavailable", "Service heartbeat exists but native task state is unavailable")
    if evidence.get("completed_count", 0) and evidence.get("completed_count") == evidence.get("task_count"):
        return decision("completed", "All observed native tasks are completed", healthy=True)
    return decision("idle", "Fresh supervisor evidence, no active or ready work; completion is not asserted", healthy=True)


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + target.name, dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_action(path: str, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _open_regular(target, append=True) as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def monitor(config: dict[str, Any], *, apply: bool = False) -> dict[str, Any]:
    """Record a health snapshot, and at most one scoped, budgeted restart."""
    Path(config["monitor_lock_path"]).parent.mkdir(parents=True, exist_ok=True)
    with _open_regular(config["monitor_lock_path"], append=True) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _monitor_locked(config, apply=apply)


def _monitor_locked(config: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    now = time.time()
    previous, previous_error = _read_json(config["monitor_state_path"])
    scope = {"service": config["service"], "board_namespace": config["board_namespace"]}
    if previous and any(previous.get(key) != value for key, value in scope.items()):
        raise ValueError("monitor state belongs to a different service or board")
    attempts = previous.get("restart_attempts_at_epoch")
    if previous_error is None and (not isinstance(attempts, list) or any(
            isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value <= 0 for value in attempts)):
        previous_error = "invalid durable restart budget"
        previous = {}
    evidence = collect_evidence(config, now)
    service = systemd_show(config["service"])
    monotonic = time.monotonic()
    evidence["native_recovery"] = _native_recovery_observation(config, service, evidence, previous, now, monotonic)
    health = evaluate_health(config, service, evidence, previous, now, monotonic)
    restarts = [value for value in previous.get("restart_attempts_at_epoch", [])
                if isinstance(value, (float, int)) and value > now - 3600]
    current = {**scope, "first_seen_at_epoch": previous.get("first_seen_at_epoch", now),
               "restart_attempts_at_epoch": restarts}
    if evidence["native_recovery"].get("tracked"):
        current["native_recovery"] = evidence["native_recovery"]
    elif evidence.get("native_daemon_pid_alive") is not True and type(previous.get("native_recovery")) is dict:
        current["native_recovery"] = previous["native_recovery"]
    if evidence.get("ready_count") and not evidence.get("active"):
        current["ready_since_at_epoch"] = previous.get("ready_since_at_epoch", now)
    summary = {"schema": "ipfs-accelerate/board-service-monitor@1", **scope,
               "checked_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
               "checked_at_epoch": now, "apply": apply, **health,
               "service_observation": service, "evidence": evidence,
               "action": "none", "restart_attempts_last_hour": len(restarts)}
    if health["restart_eligible"]:
        summary["action"] = "would_restart"
        if previous_error and previous_error != "missing":
            summary["action"] = "restart_suppressed_state_unreadable"
        elif restarts and now - max(restarts) < config["restart_cooldown_seconds"]:
            summary["action"] = "restart_suppressed_cooldown"
        elif len(restarts) >= config["max_restarts_per_hour"]:
            summary["action"] = "restart_suppressed_budget"
        elif apply:
            # The native owner can recover or start useful work while files
            # are being observed. Re-read before charging any restart, and
            # never apply an old diagnosis to a different service generation.
            observed_service = service
            observed_recovery = evidence.get("native_recovery") or {}
            observed_status = health["status"]
            now = time.time()
            evidence = collect_evidence(config, now)
            service = systemd_show(config["service"])
            monotonic = time.monotonic()
            evidence["native_recovery"] = _native_recovery_observation(config, service, evidence, previous, now, monotonic)
            health = evaluate_health(config, service, evidence, previous, now, monotonic)
            summary.update(**health, evidence=evidence, service_observation=service,
                           checked_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                           checked_at_epoch=now)
            observed_identity = _service_identity(observed_service)
            refreshed_identity = _service_identity(service)
            summary["restart_recheck"] = {
                "previous_status": observed_status,
                "service_identity_unchanged": (
                    observed_identity is not None and observed_identity == refreshed_identity),
            }
            rechecked_recovery = evidence["native_recovery"]
            if rechecked_recovery.get("tracked"):
                current["native_recovery"] = rechecked_recovery
            elif evidence.get("native_daemon_pid_alive") is True:
                current.pop("native_recovery", None)
            if not evidence.get("ready_count") or evidence.get("active"):
                current.pop("ready_since_at_epoch", None)
            if observed_identity is None or refreshed_identity is None:
                summary["action"] = "restart_suppressed_identity_unavailable"
            elif observed_identity != refreshed_identity:
                summary["action"] = "restart_suppressed_service_changed"
                if evidence.get("ready_count") and not evidence.get("active"):
                    current["ready_since_at_epoch"] = now
            elif observed_status == "native_recovery_stalled" and not (
                    rechecked_recovery.get("expired") is True and rechecked_recovery.get("quiescent") is True
                    and rechecked_recovery.get("identity") == observed_recovery.get("identity")):
                summary["action"] = "restart_suppressed_recovery_changed"
            elif not health["restart_eligible"]:
                summary["action"] = "restart_suppressed_health_changed"
            elif health["status"] == "stale_worker":
                # Provider output can stop while its native validation thread
                # owns a live subprocess. Keep the original absolute deadline;
                # neither a child nor uncertainty authorizes an early restart.
                owned = _owned_work_observation(service, evidence)
                owned["observed_at_epoch"] = now
                summary["owned_work_recheck"] = owned
                current["owned_work_observation"] = owned
                if owned.get("verified") is not True:
                    summary.update(action="restart_suppressed_owned_work_unverified", restart_eligible=False,
                                   status="owned_work_unverified", healthy=False,
                                   reason="Owned subprocess safety could not be verified; no early restart authorized")
                elif owned["children"]:
                    old = previous.get("owned_work_observation")
                    old = old if type(old) is dict else {}
                    keys = ("supervisor_pid", "supervisor_birth_ticks", "daemon_pid", "daemon_birth_ticks", "boot_id", "children")
                    old_samples = old.get("samples")
                    same = (old.get("verified") is True and all(old.get(key) == owned.get(key) for key in keys)
                            and type(old.get("observed_at_epoch")) in (int, float)
                            and now - config["heartbeat_timeout_seconds"] <= old["observed_at_epoch"] < now
                            and type(old_samples) is list and len(old_samples) == len(owned["samples"])
                            and all(a.get("pid") == b["pid"] and a.get("birth_ticks") == b["birth_ticks"]
                                    and a.get("parent_pid") == b.get("parent_pid") for a, b in zip(old_samples, owned["samples"])))
                    progressed = same and any(
                        b["cpu_ticks"] > a.get("cpu_ticks", b["cpu_ticks"])
                        or b["io_bytes"] > a.get("io_bytes", b["io_bytes"])
                        for a, b in zip(old_samples, owned["samples"]))
                    if progressed or not same:
                        summary.update(action="restart_suppressed_owned_work", restart_eligible=False,
                                       healthy=bool(progressed),
                                       status="quiet_owned_work" if progressed else "observing_owned_work",
                                       reason="Birth-bound owned subprocess progress observed; original absolute deadline remains in force"
                                              if progressed else "Sampling owned subprocess liveness before repair; original absolute deadline remains in force")
                    else:
                        summary["action"] = "restart_revalidated"
                else:
                    summary["action"] = "restart_revalidated"
            else:
                summary["action"] = "restart_revalidated"
        if apply and summary["action"] == "restart_revalidated":
            # Persist the charged attempt before its effect. A monitor crash
            # between these operations cannot erase the cooldown or budget.
            current["restart_attempts_at_epoch"].append(now)
            _atomic_json(config["monitor_state_path"], current)
            _append_action(config["actions_path"], {**scope, "at_epoch": now,
                           "action": "restart_requested", "reason": health["status"]})
            try:
                result = subprocess.run(["systemctl", "--user", "restart", config["service"]],
                                        text=True, capture_output=True, timeout=45, check=False)
                summary["action"] = "restart_requested" if result.returncode == 0 else "restart_failed"
                summary["restart_returncode"] = result.returncode
                summary["restart_stderr"] = result.stderr[-2000:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                summary["action"] = "restart_failed"
                summary["restart_error"] = str(exc)
            summary["restart_attempts_last_hour"] = len(current["restart_attempts_at_epoch"])
    # Keep corrupt budget state intact and require operator repair instead of
    # silently granting an empty restart budget.
    if not previous_error or previous_error == "missing":
        _atomic_json(config["monitor_state_path"], current)
    try:
        _append_action(config["actions_path"], {**scope, "at_epoch": now,
                       "action": summary["action"], "status": summary["status"], "apply": apply})
    except (OSError, ValueError) as exc:
        # This is an observation log, after any charged repair has completed.
        # Preserve the actual action and publish its outcome even if logging
        # failed. The earlier restart-intent append remains fail closed.
        summary.update(healthy=False, status="monitor_error", restart_eligible=False,
                       observed_health_status=health["status"],
                       observation_log_error=f"{type(exc).__name__}: {exc}"[:512])
    _atomic_json(config["health_path"], summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Allow bounded restart of the exact configured service")
    args = parser.parse_args(argv)
    try:
        summary = monitor(load_config(args.config), apply=args.apply)
    except (OSError, ValueError) as exc:
        print(json.dumps({"healthy": False, "status": "monitor_error", "error": str(exc)}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
