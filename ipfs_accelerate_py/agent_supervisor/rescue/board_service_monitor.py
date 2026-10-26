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
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
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
        "max_restarts_per_hour": 3,
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
        "native_last_exit_code": status.get("last_exit_code"),
        "native_restart_count": status.get("restart_count"),
        "native_idle_ready_seconds": status.get("idle_ready_seconds"),
        "native_recycle_safe": status.get("idle_ready_recycle_safe"),
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
    health = evaluate_health(config, service, evidence, previous, now, time.monotonic())
    restarts = [value for value in previous.get("restart_attempts_at_epoch", [])
                if isinstance(value, (float, int)) and value > now - 3600]
    current = {**scope, "first_seen_at_epoch": previous.get("first_seen_at_epoch", now),
               "restart_attempts_at_epoch": restarts}
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
                       "action": summary["action"], "status": health["status"], "apply": apply})
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
