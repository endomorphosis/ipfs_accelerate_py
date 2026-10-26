"""Observe independent board lanes and apply only their bounded local repairs.

This stdlib-only CLI runs directly, without importing the supervisor package.
Fleet configuration names ``lanes`` containing ``lane_id``, ``config`` (the
existing per-service monitor JSON), and optional ``role``. ``health_path`` is
required. Optional ``merge_queue_dir`` observes JSON stage receipts only; the
mutable authoritative DuckDB queue is never opened or modified.

Cross-lane conflicts, repeated failures, and stale merge receipt projections
are diagnostic only. They never authorize a fleet restart or task mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import ipfs_accelerate_py.agent_supervisor.rescue.board_service_monitor

_SOURCE = Path(__file__).with_name("board_service_monitor.py")
_SPEC = importlib.util.spec_from_file_location("board_service_monitor_standalone", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_SOURCE}")
lane_monitor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lane_monitor)


def load_config(path: Path) -> dict[str, Any]:
    config, error = lane_monitor._read_json(str(path))
    if error:
        raise ValueError("fleet configuration unreadable: " + error)
    if not isinstance(config, dict) or not isinstance(config.get("lanes"), list) or not config["lanes"]:
        raise ValueError("fleet configuration must contain a nonempty lanes list")
    if not isinstance(config.get("health_path"), str) or not config["health_path"].strip():
        raise ValueError("health_path is required")
    base = path.resolve().parent

    def resolve(value: str) -> str:
        target = Path(value).expanduser()
        return str(target.resolve() if target.is_absolute() else (base / target).resolve())

    config["health_path"] = resolve(config["health_path"])
    for key, default in {
        "actions_path": config["health_path"] + ".actions.jsonl",
        "lock_path": config["health_path"] + ".lock",
    }.items():
        config[key] = resolve(config.get(key) or default)
    if config.get("merge_queue_dir"):
        config["merge_queue_dir"] = resolve(config["merge_queue_dir"])
    # Optional pinned descriptions are observations, never executable hooks.
    # Leave malformed entries to the independent diagnostic below so they do
    # not prevent the existing lane observations and bounded repairs.
    attachment = config.get("prerequisite_diagnostics")
    if type(attachment) is dict:
        attachment = dict(attachment)
        if type(attachment.get("owner")) is dict:
            attachment["owner"] = dict(attachment["owner"])
        for entry in (attachment, attachment.get("owner")):
            if type(entry) is dict and type(entry.get("path")) is str:
                target = Path(entry["path"])
                entry["path"] = os.path.abspath(target if target.is_absolute() else base / target)
        config["prerequisite_diagnostics"] = attachment
    for key, default in {
        "error_window_seconds": 900, "repeated_error_threshold": 3,
        "merge_stale_seconds": 3600, "max_merge_receipts": 1024,
    }.items():
        value = config.setdefault(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be a positive finite number")
        if key in {"repeated_error_threshold", "max_merge_receipts"} and int(value) != value:
            raise ValueError(f"{key} must be an integer")
    lane_ids, services, health_paths, state_paths, lock_paths = set(), set(), set(), set(), set()
    loaded = []
    for item in config["lanes"]:
        if (not isinstance(item, dict) or not isinstance(item.get("lane_id"), str)
                or not item["lane_id"].strip() or not isinstance(item.get("config"), str)):
            raise ValueError("each lane requires lane_id and config strings")
        per_lane_path = resolve(item["config"])
        per_lane = lane_monitor.load_config(Path(per_lane_path))
        if (item["lane_id"] in lane_ids or per_lane["service"] in services or per_lane["health_path"] in health_paths
                or per_lane["monitor_state_path"] in state_paths or per_lane["monitor_lock_path"] in lock_paths):
            raise ValueError("lanes must have distinct lane IDs, exact services, and health paths")
        if per_lane["health_path"] == config["health_path"]:
            raise ValueError("fleet health_path must differ from lane health paths")
        lane_ids.add(item["lane_id"])
        services.add(per_lane["service"])
        health_paths.add(per_lane["health_path"])
        state_paths.add(per_lane["monitor_state_path"])
        lock_paths.add(per_lane["monitor_lock_path"])
        loaded.append({"lane_id": item["lane_id"], "role": str(item.get("role") or "implementation"),
                       "config_path": per_lane_path, "monitor_config": per_lane})
    config["lanes"] = loaded
    return config


def _failure_signature(event: dict[str, Any]) -> tuple[str, str, str] | None:
    kind = str(event.get("type") or "")
    returncode = event.get("returncode", event.get("exit_code"))
    failed = (isinstance(returncode, (int, float)) and not isinstance(returncode, bool) and returncode != 0)
    failed = failed or kind.endswith(("_failed", "_error"))
    # Text in an LLM worker log can discuss errors without encountering one.
    # Only structured failure events establish this diagnostic.
    if not failed:
        return None
    return kind, str(event.get("task_id") or ""), str(event.get("reason_code") or event.get("reason") or "")[:200]


def lane_diagnostics(lane: dict[str, Any], summary: dict[str, Any], config: dict[str, Any], now: float) -> dict[str, Any]:
    events, error = lane_monitor._event_tail(lane["monitor_config"], now)
    failures = Counter()
    for event in events:
        signature = _failure_signature(event)
        if signature and event["_timestamp"] >= now - config["error_window_seconds"]:
            failures[signature] += 1
    repeated = [{"event_type": signature[0], "task_id": signature[1], "reason": signature[2], "count": count}
                for signature, count in sorted(failures.items()) if count >= config["repeated_error_threshold"]]
    evidence = summary.get("evidence") or {}
    active_id = str(evidence.get("active_task_id") or "")
    starts = [event for event in events if event.get("type") == "implementation_started"
              and str(event.get("task_id") or "") == active_id]
    start = starts[-1] if starts else {}
    return {"event_read_error": error, "recent_failure_count": sum(failures.values()),
            "repeated_failures": repeated,
            "active_task_id": active_id,
            "active_task_cid": str(start.get("canonical_task_cid") or start.get("task_cid") or ""),
            "active_task_key": str(start.get("canonical_task_key") or ""),
            "active_log_path": evidence.get("active_log_path"),
            "active_log_at_epoch": evidence.get("active_log_at_epoch")}


def duplicate_active_tasks(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = defaultdict(list)
    for lane in lanes:
        summary, details = lane["health"], lane["diagnostics"]
        if (not summary.get("healthy") or summary.get("status") != "working"
                or summary.get("service_observation", {}).get("ActiveState") != "active"
                or not summary.get("evidence", {}).get("active") or not details["active_task_id"]):
            continue
        identity = details["active_task_cid"] or details["active_task_key"] or details["active_task_id"]
        basis = "canonical_cid" if details["active_task_cid"] else "canonical_key" if details["active_task_key"] else "display_id"
        active[(summary["board_namespace"], basis, identity)].append(lane)
    return [{"board_namespace": key[0], "identity_basis": key[1], "task_identity": key[2],
             "lane_ids": [lane["lane_id"] for lane in values],
             "task_ids": [lane["diagnostics"]["active_task_id"] for lane in values]}
            for key, values in sorted(active.items()) if len(values) > 1]


def merge_diagnostics(config: dict[str, Any], now: float) -> dict[str, Any]:
    directory = config.get("merge_queue_dir")
    if not directory:
        return {"configured": False, "issues": [], "authority": "observation_only"}
    root = Path(directory)
    result = {"configured": True, "path": str(root), "authority": "json_stage_receipt_projection",
              "database_opened": False, "counts": {}, "counts_truncated": [], "issues": [], "stale_requests": []}
    if not root.is_dir():
        result["issues"].append({"code": "merge_receipts_unavailable", "reason": "queue directory is missing"})
        return result
    for stage in ("pending", "processing", "completed", "failed", "quarantine", "cancelled"):
        count = 0
        for path in (root / stage).glob("*.json"):
            count += 1
            if count > config["max_merge_receipts"]:
                result["counts_truncated"].append(stage)
                if stage not in {"completed", "cancelled"}:
                    result["issues"].append({"code": "merge_receipt_scan_limit", "stage": stage})
                break
            if stage in {"completed", "cancelled"}:
                continue
            receipt, error = lane_monitor._read_json(str(path))
            if error:
                result["issues"].append({"code": "merge_receipt_unreadable", "stage": stage, "path": str(path)})
                continue
            if stage in {"quarantine", "failed"}:
                result["issues"].append({"code": "merge_quarantined" if stage == "quarantine" else "merge_failed",
                                         "request_id": receipt.get("request_id"),
                                         "task_id": receipt.get("task_id"), "lane_id": receipt.get("lane_id")})
                continue
            # Stage moves write the destination before unlinking the previous
            # projection. A terminal receipt supersedes a transient old file.
            if any((root / terminal / path.name).exists() for terminal in ("completed", "failed", "quarantine", "cancelled")):
                continue
            failures = receipt.get("failure_count", 0)
            if (isinstance(failures, (int, float)) and not isinstance(failures, bool)
                    and math.isfinite(failures) and failures > 0):
                # A pending retry can be recent and its worker can already
                # be on another task. Keep the integration failure visible
                # without treating it as authority to restart that worker.
                result["issues"].append({"code": "merge_retrying_failure",
                    "request_id": receipt.get("request_id"), "task_id": receipt.get("task_id"),
                    "lane_id": receipt.get("lane_id"), "failure_count": failures,
                    "reason": str(receipt.get("failure_reason") or "")[:200]})
            retry_at = lane_monitor._timestamp(receipt.get("retry_not_before"), float("inf"))
            if stage == "pending" and retry_at and retry_at > now:
                continue
            keys = ("claimed_at", "enqueued_at") if stage == "processing" else ("enqueued_at",)
            started = lane_monitor._latest(receipt, keys, now)
            if started is None:
                result["issues"].append({"code": "merge_receipt_timestamp_missing", "path": str(path)})
            elif now - started > config["merge_stale_seconds"]:
                result["stale_requests"].append({"request_id": receipt.get("request_id"),
                    "task_id": receipt.get("task_id"), "lane_id": receipt.get("lane_id"),
                    "stage": stage, "age_seconds": now - started, "receipt_path": str(path)})
        result["counts"][stage] = min(count, int(config["max_merge_receipts"]))
    if result["stale_requests"]:
        result["issues"].append({"code": "merge_receipts_stale", "count": len(result["stale_requests"])})
    return result


def _prerequisite_pin(reference: dict[str, Any], maximum: int) -> bytes:
    if (type(reference) is not dict or set(reference) != {"path", "bytes", "sha256"}
            or type(reference["path"]) is not str or not Path(reference["path"]).is_absolute()
            or type(reference["bytes"]) is not int or not 0 < reference["bytes"] <= maximum
            or type(reference["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None):
        raise ValueError("closed bounded prerequisite file pin required")
    descriptor = os.open(reference["path"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != reference["bytes"]:
            raise ValueError("prerequisite input is not an exact bounded ordinary file")
        pieces, count = [], 0
        while count <= maximum:
            piece = os.read(descriptor, min(65536, maximum + 1 - count))
            if not piece:
                break
            pieces.append(piece)
            count += len(piece)
        after = os.fstat(descriptor)
        current = os.stat(reference["path"], follow_symlinks=False)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                   value.st_mtime_ns, value.st_ctime_ns)
        raw = b"".join(pieces)
        if (identity(before) != identity(after) or identity(after) != identity(current)
                or len(raw) != reference["bytes"]
                or hashlib.sha256(raw).hexdigest() != reference["sha256"]):
            raise ValueError("prerequisite file changed or differs from its independent pin")
        return raw
    finally:
        os.close(descriptor)


def prerequisite_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Read a bounded pinned requirement description without running its owner.

    File pins establish captured bytes, not the presence or compatibility of
    any required vectors, producer or replay route. Source/report checks are
    sequential observations; no action, completion or atomic-snapshot claim
    follows from their agreement.
    """
    result = {"configured": True, "available": False, "authority": "declared_requirements_only",
              "completion_authority": False, "action_authority": False}
    try:
        reference = config["prerequisite_diagnostics"]
        if type(reference) is not dict or set(reference) != {"path", "bytes", "sha256", "owner"}:
            raise ValueError("closed prerequisite report and owner pins required")
        report_pin = {key: reference[key] for key in ("path", "bytes", "sha256")}
        if type(reference["owner"]) is not dict:
            raise ValueError("closed prerequisite owner pin required")
        owner_pin = dict(reference["owner"])
        frozen_reference = {**report_pin, "owner": dict(owner_pin)}
        _prerequisite_pin(owner_pin, 256 * 1024)
        raw = _prerequisite_pin(report_pin, 64 * 1024)

        def object_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate prerequisite JSON key")
                value[key] = item
            return value

        def invalid_constant(value):
            raise ValueError("nonfinite prerequisite JSON value: " + value)

        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("overflowing prerequisite JSON number")
            return number

        report = json.loads(raw, object_pairs_hook=object_pairs,
                            parse_constant=invalid_constant, parse_float=finite_float)
        if (type(report) is not dict
                or set(report) != {"schema", "diagnostic_scope", "diagnostics", "completion_allowed"}
                or report["schema"] != "legal-ir-width-replay-prerequisite-bundle/v1"
                or report["diagnostic_scope"] != "declared_requirements_only"
                or report["completion_allowed"] is not False
                or type(report["diagnostics"]) is not list or len(report["diagnostics"]) != 2):
            raise ValueError("closed observation-only prerequisite bundle required")
        item_fields = {"schema", "task_id", "dimension", "diagnostic_scope", "input_availability",
                       "input_files_checked", "executable_presence_checked", "reviewed_route_available",
                       "executable_presence_sufficient", "reviewed_local_dimensions", "source_contract",
                       "required_inputs", "task_constraints", "next_actions", "completion_allowed",
                       "resource_admission_granted", "encoder_executed", "training_executed",
                       "admitted", "formalized", "proof_authority"}
        expected_tasks = (("LIRAE-WIDTH-08", 8), ("LIRAE-WIDTH-4096", 4096))
        for item, (task_id, dimension) in zip(report["diagnostics"], expected_tasks):
            if (type(item) is not dict or set(item) != item_fields
                    or type(item.get("task_id")) is not str or item["task_id"] != task_id
                    or type(item.get("dimension")) is not int or item["dimension"] != dimension
                    or type(item.get("reviewed_local_dimensions")) is not list
                    or any(type(value) is not int for value in item["reviewed_local_dimensions"])
                    or item["reviewed_local_dimensions"] != [384, 768]
                    or item.get("schema") != "legal-ir-width-replay-prerequisites/v1"
                    or item.get("diagnostic_scope") != "declared_requirements_only"
                    or item.get("input_availability") != "not_inspected"
                    or any(item.get(key) is not False for key in (
                        "input_files_checked", "executable_presence_checked", "reviewed_route_available",
                        "executable_presence_sufficient", "completion_allowed", "encoder_executed",
                        "training_executed", "admitted", "formalized", "resource_admission_granted",
                        "proof_authority"))):
                raise ValueError("unique non-authoritative prerequisite descriptions required")
        _prerequisite_pin(owner_pin, 256 * 1024)
        _prerequisite_pin(report_pin, 64 * 1024)
        if reference != frozen_reference or config.get("prerequisite_diagnostics") != frozen_reference:
            raise ValueError("original prerequisite references changed during observation")
        result.update(available=True, report_pin=report_pin, owner_pin=owner_pin, report=report,
                      consistency_scope="sequential_pinned_file_observations",
                      payload_validation_scope="pinned_description_envelope_and_non_authority_flags_only")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:512]
    return result


def monitor_fleet(config: dict[str, Any], *, apply: bool = False) -> dict[str, Any]:
    Path(config["lock_path"]).parent.mkdir(parents=True, exist_ok=True)
    with lane_monitor._open_regular(config["lock_path"], append=True) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _monitor_locked(config, apply=apply)


def _observe_lane(lane: dict[str, Any], config: dict[str, Any], now: float, apply: bool) -> dict[str, Any]:
    try:
        health = lane_monitor.monitor(lane["monitor_config"], apply=apply)
    except Exception as exc:
        # One malformed native state or unexpected lane error must not hide
        # observations from the other independently managed services.
        health = {"healthy": False, "status": "monitor_error", "error": str(exc),
                  "service": lane["monitor_config"]["service"],
                  "board_namespace": lane["monitor_config"]["board_namespace"]}
    try:
        diagnostics = lane_diagnostics(lane, health, config, now)
        if diagnostics.get("event_read_error"):
            raise ValueError(str(diagnostics["event_read_error"]))
    except Exception as exc:
        # Diagnostics observe the same independently mutable files as health.
        # Their failure must publish an explicit unavailable observation while
        # preserving sibling observations and any action already performed.
        error = f"{type(exc).__name__}: {exc}"[:512]
        health = {**health, "healthy": False, "status": "monitor_error",
                  "diagnostics_error": error, "restart_eligible": False}
        diagnostics = {"event_read_error": error, "recent_failure_count": 0,
                       "repeated_failures": [], "active_task_id": "", "active_task_cid": "",
                       "active_task_key": "", "active_log_path": None, "active_log_at_epoch": None,
                       "diagnostics_error": error}
    return {"lane_id": lane["lane_id"], "role": lane["role"], "health": health,
            "diagnostics": diagnostics}


def _monitor_locked(config: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    now = time.time()
    issues = []
    # Separate lane locks, health files, and restart budgets make these calls
    # independent. Four bounded checks fit the systemd monitor's 90s deadline.
    with ThreadPoolExecutor(max_workers=min(4, len(config["lanes"]))) as pool:
        lanes = list(pool.map(lambda lane: _observe_lane(lane, config, now, apply), config["lanes"]))
    for observed in lanes:
        health, diagnostics = observed["health"], observed["diagnostics"]
        if not health.get("healthy"):
            issues.append({"code": "lane_unhealthy", "lane_id": observed["lane_id"], "status": health.get("status")})
        if diagnostics["repeated_failures"]:
            issues.append({"code": "repeated_lane_failures", "lane_id": observed["lane_id"],
                           "failures": diagnostics["repeated_failures"]})
    duplicates = duplicate_active_tasks(lanes)
    if duplicates:
        issues.append({"code": "duplicate_active_tasks", "conflicts": duplicates})
    merges = merge_diagnostics(config, now)
    issues.extend(merges["issues"])
    summary = {"schema": "ipfs-accelerate/board-service-fleet-monitor@1", "apply": apply,
               "checked_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
               "checked_at_epoch": now, "healthy": not issues,
               "status": "healthy" if not issues else "attention_required",
               "lane_count": len(lanes), "healthy_lane_count": sum(bool(lane["health"].get("healthy")) for lane in lanes),
               "lanes": lanes, "duplicate_active_tasks": duplicates, "merge_queue": merges, "issues": issues,
               "completion_authority": False, "task_counts_aggregated": False}
    if "prerequisite_diagnostics" in config:
        attachment = prerequisite_diagnostics(config)
        summary["prerequisite_diagnostics"] = attachment
        if not attachment["available"]:
            issues.append({"code": "prerequisite_diagnostics_unavailable", "error": attachment["error"]})
            summary.update(healthy=False, status="attention_required")
    try:
        lane_monitor._append_action(config["actions_path"], {"at_epoch": now, "action": "fleet_observed",
            "apply": apply, "healthy": summary["healthy"], "issue_codes": sorted({issue["code"] for issue in issues}),
            "lane_actions": {lane["lane_id"]: lane["health"].get("action", "none") for lane in lanes}})
    except (OSError, ValueError) as exc:
        issues.append({"code": "fleet_observation_log_unavailable",
                       "error": f"{type(exc).__name__}: {exc}"[:512]})
        summary.update(healthy=False, status="attention_required")
    lane_monitor._atomic_json(config["health_path"], summary)
    return summary


SERVICE_FLEET_PROBE_PROFILE = "board_service_fleet"
_PROBE_WAIT_STATES = frozenset({"dependencies_waiting", "backpressure", "report_only"})
_PROBE_HEALTHY_STATES = frozenset({"working", "ready", "shard_idle", "idle", "observing", "completed"})


def _probe_process_birth(pid: int) -> dict[str, Any]:
    """Observe one service process; this identity never authorizes signalling."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return {}
        fields = raw.rsplit(b")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(fields[19]), "state": fields[0].decode("ascii")}
    except (OSError, ValueError, IndexError, UnicodeError):
        return {}


def fleet_probe(config: dict[str, Any], *, board_id: str, max_age_seconds: float = 180,
                now: float | None = None) -> dict[str, Any]:
    """Read the original monitor's snapshot without running or repairing it.

    The service monitor owns progress deadlines and bounded lane restarts.
    This adapter provides only operational evidence to the coding-repair fleet;
    observed legacy task counts never become native completion authority.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", board_id):
        raise ValueError("safe fleet board identity required")
    if (type(max_age_seconds) not in (int, float) or not math.isfinite(max_age_seconds)
            or not 5 <= max_age_seconds <= 300):
        raise ValueError("health age must be between five and300 seconds")
    now = time.time() if now is None else now
    details: dict[str, Any] = {"probe_profile": SERVICE_FLEET_PROBE_PROFILE,
        "monitor_scope_verified": False, "authenticated_task_observation": False,
        "native_completion_authority": False, "task_counts_aggregated": False,
        "monitor_progress_authority": True, "lanes": []}
    result = {"schema": "ipfs_accelerate_py/taskboard-fleet-live-probe@1", "board_id": board_id,
        "health": "unknown", "complete": False, "completion_candidate": False,
        "busy": False, "progress_token": "", "reason_codes": [], "details": details}

    def refused(reason: str, *, ensure: bool = False) -> dict[str, Any]:
        result["reason_codes"] = [reason]
        if ensure:
            result["recovery_action"] = "ensure"
        return result

    report, error = lane_monitor._read_json(config["health_path"])
    if error:
        return refused("service_fleet_health_unavailable", ensure=True)
    stamp = report.get("checked_at_epoch")
    if (type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not 0 <= now - stamp <= max_age_seconds):
        return refused("service_fleet_health_stale", ensure=True)
    configured = {lane["lane_id"]: lane for lane in config["lanes"]}
    rows = report.get("lanes")
    if (report.get("schema") != "ipfs-accelerate/board-service-fleet-monitor@1"
            or report.get("completion_authority") is not False
            or report.get("task_counts_aggregated") is not False
            or type(rows) is not list or len(rows) != len(configured)
            or type(report.get("lane_count")) is not int or report["lane_count"] != len(configured)
            or any(type(row) is not dict for row in rows)
            or {row.get("lane_id") for row in rows} != set(configured)):
        return refused("service_fleet_scope_mismatch")
    states, active_births, process_fault = [], [], False
    for row in rows:
        per_lane = configured[row["lane_id"]]["monitor_config"]
        health = row.get("health")
        if (type(health) is not dict or health.get("service") != per_lane["service"]
                or health.get("board_namespace") != per_lane["board_namespace"]
                or type(health.get("checked_at_epoch")) not in (int, float)
                or not math.isfinite(health["checked_at_epoch"])
                or not 0 <= now - health["checked_at_epoch"] <= max_age_seconds):
            return refused("service_fleet_lane_scope_mismatch")
        observed = health.get("service_observation")
        current = lane_monitor.systemd_show(per_lane["service"])
        identity_keys = ("LoadState", "ActiveState", "SubState", "MainPID", "ExecMainStartTimestampMonotonic")
        if (type(observed) is not dict or "error" in current
                or any(key not in observed or observed[key] != current.get(key) for key in identity_keys)):
            return refused("service_fleet_generation_changed")
        pid_text = current.get("MainPID", "")
        if not isinstance(pid_text, str) or not pid_text.isdecimal():
            return refused("service_fleet_process_identity_unavailable")
        pid = int(pid_text)
        birth = _probe_process_birth(pid) if pid else {}
        if pid and (not birth or birth.get("state") in {"Z", "X"}):
            return refused("service_fleet_process_identity_unavailable")
        process_fault = process_fault or birth.get("state") in {"T", "t", "D"}
        rechecked = lane_monitor.systemd_show(per_lane["service"])
        rechecked_birth = _probe_process_birth(pid) if pid else {}
        if (any(rechecked.get(key) != current.get(key) for key in identity_keys)
                or (pid and any(rechecked_birth.get(key) != birth.get(key) for key in ("pid", "start_ticks")))):
            return refused("service_fleet_generation_changed")
        process_fault = process_fault or rechecked_birth.get("state") in {"T", "t", "D"}
        if pid and rechecked_birth.get("state") in {"Z", "X"}:
            return refused("service_fleet_process_identity_unavailable")
        state = health.get("status")
        if not isinstance(state, str) or type(health.get("healthy")) is not bool:
            return refused("service_fleet_lane_health_invalid")
        evidence = health.get("evidence") if isinstance(health.get("evidence"), dict) else {}
        counts = {key: evidence[key] for key in ("task_count", "completed_count", "ready_count", "blocked_count", "waiting_count")
                  if type(evidence.get(key)) is int and evidence[key] >= 0}
        if state == "completed" and (not counts.get("task_count")
                or counts.get("completed_count") != counts["task_count"]
                or any(counts.get(key, 0) for key in ("ready_count", "blocked_count", "waiting_count"))):
            return refused("service_fleet_completion_projection_invalid")
        details["lanes"].append({"lane_id": row["lane_id"], "service": per_lane["service"],
            "board_namespace": per_lane["board_namespace"], "status": state,
            "healthy": health["healthy"], "process_birth": birth,
            "service_start_monotonic": current["ExecMainStartTimestampMonotonic"],
            "observed_counts": counts})
        states.append(state)
        if state == "working" and health["healthy"] and pid:
            active_births.append(birth)
    details.update(monitor_scope_verified=True, checked_at_epoch=stamp,
        health_report_sha256=hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    codes = {item.get("code") for item in report.get("issues", []) if isinstance(item, dict)}
    only_waiting = (all(state in _PROBE_WAIT_STATES | _PROBE_HEALTHY_STATES for state in states)
                    and (any(state in _PROBE_WAIT_STATES for state in states)
                         or codes == {"prerequisite_diagnostics_unavailable"})
                    and not codes - {"lane_unhealthy", "prerequisite_diagnostics_unavailable"})
    if only_waiting and not process_fault:
        result.update(health="blocked", reason_codes=["service_fleet_waiting_on_contract"], recovery_action="none")
    elif (report.get("healthy") is True and not codes and not process_fault
            and all(row["healthy"] and row["status"] in _PROBE_HEALTHY_STATES for row in details["lanes"])):
        result.update(health="healthy", busy=bool(active_births),
            completion_candidate=all(state == "completed" for state in states))
    else:
        result.update(health="degraded", reason_codes=["service_fleet_monitor_requires_repair"], recovery_action="repair")
    return result


def _configured_fleet_probe(config_path: Path, inventory_path: Path, board_id: str,
                            max_age_seconds: float) -> dict[str, Any]:
    # Reuse the existing bounded inventory/source-cleanliness reader. This is
    # diagnostic cleanliness, not proof of an interpreter's loaded source.
    helper_path = Path(__file__).with_name("live_board_probe.py")
    spec = importlib.util.spec_from_file_location("service_fleet_source_probe", helper_path)
    if spec is None or spec.loader is None:
        raise ValueError("native source observation helper unavailable")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    inventory = helper.read_json_object(inventory_path)
    entries = [entry for entry in inventory.get("boards", []) if isinstance(entry, dict) and entry.get("id") == board_id]
    if (len(entries) != 1 or entries[0].get("probe_profile") != SERVICE_FLEET_PROBE_PROFILE
            or Path(entries[0].get("config_path", "")).resolve() != config_path.resolve()):
        raise ValueError("exact service-fleet inventory/configuration binding required")
    result = fleet_probe(load_config(config_path), board_id=board_id, max_age_seconds=max_age_seconds)
    integrity = helper._source_integrity(entries[0])
    result["details"].update(source_integrity=integrity, source_heads=helper._source_heads(entries[0]))
    if integrity.get("valid") is not True:
        result.update(health="degraded", busy=False, completion_candidate=False,
            reason_codes=["source_integrity_not_verified"], recovery_action="repair")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Allow only each lane's existing bounded repairs")
    parser.add_argument("--fleet-probe-board-id", help="Read-only operational adapter for the existing coding-repair fleet")
    parser.add_argument("--fleet-probe-inventory", type=Path)
    parser.add_argument("--max-health-age-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if args.fleet_probe_board_id or args.fleet_probe_inventory:
        if args.apply or not (args.fleet_probe_board_id and args.fleet_probe_inventory):
            parser.error("fleet probe requires board/inventory together and forbids --apply")
        try:
            summary = _configured_fleet_probe(args.config, args.fleet_probe_inventory,
                args.fleet_probe_board_id, args.max_health_age_seconds)
        except (OSError, ValueError, TypeError) as exc:
            summary = {"board_id": args.fleet_probe_board_id, "health": "unknown", "complete": False,
                "busy": False, "reason_codes": ["service_fleet_probe_unavailable"],
                "details": {"error_type": type(exc).__name__}}
        print(json.dumps(summary, sort_keys=True))
        return 0
    try:
        summary = monitor_fleet(load_config(args.config), apply=args.apply)
    except (OSError, ValueError) as exc:
        print(json.dumps({"healthy": False, "status": "monitor_error", "error": str(exc)}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
