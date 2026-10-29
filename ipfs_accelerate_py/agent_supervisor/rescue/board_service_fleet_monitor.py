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
    # Keep optional liveness diagnostics separate from lane repair policy.
    liveness = config.get("watchdog_liveness")
    if type(liveness) is dict:
        liveness = dict(liveness)
        if type(liveness.get("status_path")) is str:
            try:
                target = Path(liveness["status_path"]).expanduser()
                liveness["status_path"] = os.path.abspath(target if target.is_absolute() else base / target)
            except (OSError, ValueError, RuntimeError):
                # An optional invalid home/path must not hide sibling lanes.
                # Keep its original spelling for the diagnostic refusal.
                pass
        config["watchdog_liveness"] = liveness
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


def _read_watchdog_liveness_report(path: str) -> tuple[dict[str, Any], str | None]:
    """Use the existing bounded ordinary-file reader with strict JSON fields."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate watchdog report field")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite watchdog report value")

    try:
        with lane_monitor._open_regular(path) as stream:
            raw = stream.read(lane_monitor._MAX_JSON_BYTES + 1)
        if len(raw) > lane_monitor._MAX_JSON_BYTES:
            raise ValueError("watchdog report exceeds monitor size limit")
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        if type(value) is not dict:
            raise ValueError("expected watchdog report object")
        return value, None
    except FileNotFoundError:
        return {}, "missing"
    except (OSError, ValueError, RecursionError) as exc:
        return {}, str(exc)


def _watchdog_liveness_incident(result, binding, identity, kernel_identity):
    if result["attention_required"]:
        # Stable consecutive-incident identity, not a task CID or repair grant.
        safe_binding = binding if type(binding) is dict else {}
        def safe_text(key, limit):
            value = safe_binding.get(key)
            return value[:limit] if type(value) is str else ""

        signature = {"service": safe_text("service", 208),
                     "deployment_id": safe_text("deployment_id", 128),
                     "source_sha256": safe_text("source_sha256", 64),
                     "service_identity": identity, "process_identity": kernel_identity,
                     "status": result["status"],
                     "reason_codes": result["reason_codes"]}
        result["incident_id"] = "watchdog-liveness:" + hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return result


def watchdog_liveness_diagnostics(config: dict[str, Any], now: float, *,
                                  deadline_monotonic: float | None = None) -> dict[str, Any]:
    """Observe one declared watchdog generation, without authorizing recovery.

    The producer's profile is a declaration, not an attestation of imported
    bytes. Sequential unit, kernel-process and report observations must agree.
    This attachment must never alter lane health: existing fleet consumers use
    aggregate health as an infrastructure-repair trigger.
    """
    result = {
        "configured": "watchdog_liveness" in config, "available": False,
        "status": "unavailable", "attention_required": True,
        "action_authority": False, "restart_eligible": False,
        "completion_authority": False, "identity_verified": False,
        "source_binding": "declared_profile_only",
        "consistency_scope": "sequential_identity_and_report_observations",
        "reason_codes": [], "incident_id": None,
    }
    if not result["configured"]:
        result.update(status="not_configured", attention_required=False)
        return result
    identity = None
    kernel_identity = None
    binding = config.get("watchdog_liveness")

    def require_budget(seconds):
        if deadline_monotonic is not None:
            current = time.monotonic()
            if (type(deadline_monotonic) not in (int, float)
                    or not math.isfinite(deadline_monotonic)
                    or not math.isfinite(current)
                    or current + seconds > deadline_monotonic):
                raise ValueError("observation_budget_exhausted")

    try:
        required = {"schema", "service", "status_path", "deployment_id", "source_sha256",
                    "stale_after_seconds", "startup_grace_seconds"}
        if (type(binding) is not dict or not required <= set(binding)
                or set(binding) - required - {"operator_hold"}
                or binding["schema"] != "agent-supervisor/watchdog-liveness-binding@1"
                or type(binding["service"]) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,199}\.service", binding["service"]) is None
                or type(binding["status_path"]) is not str or "\0" in binding["status_path"]
                or not Path(binding["status_path"]).is_absolute()
                or type(binding["deployment_id"]) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", binding["deployment_id"]) is None
                or type(binding["source_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", binding["source_sha256"]) is None
                or type(binding.get("operator_hold", False)) is not bool):
            raise ValueError("invalid_binding")
        frozen_binding = dict(binding)
        for key, minimum in (("stale_after_seconds", 0), ("startup_grace_seconds", -1)):
            value = binding[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value <= minimum or (key == "startup_grace_seconds" and value < 0)):
                raise ValueError("invalid_binding")
        if type(now) not in (int, float) or not math.isfinite(now) or now <= 0:
            raise ValueError("invalid_clock")
        result["service"] = binding["service"]
        if binding.get("operator_hold", False):
            result.update(available=True, status="held", attention_required=False,
                          reason_codes=["operator_hold"])
            return result
        # Existing exact-unit queries have a 15s timeout. Reserve both queries
        # plus five seconds for local observations; slow lanes take priority.
        require_budget(35)
        opening_unit = lane_monitor.systemd_show(binding["service"])
        identity = lane_monitor._service_identity(opening_unit)
        if identity is None:
            raise ValueError("service_identity_unavailable")
        result["service_identity"] = {"state": identity[0], "pid": identity[1],
                                      "birth_monotonic_us": identity[2]}
        if identity[0] != "active":
            result.update(available=True, status="inactive", attention_required=True,
                          reason_codes=["service_not_active"])
            return _watchdog_liveness_incident(result, binding, identity, kernel_identity)
        if identity[1] <= 0:
            raise ValueError("service_identity_unavailable")
        monotonic_now = time.monotonic()
        service_age = monotonic_now - identity[2] / 1_000_000
        if not math.isfinite(service_age) or service_age < 0:
            raise ValueError("service_birth_unverified")

        def process_identity(snapshot):
            if (type(snapshot) is not dict
                    or any(type(snapshot.get(k)) is not int or snapshot[k] <= 0
                           for k in ("pid", "birth_ticks"))
                    or snapshot["pid"] != identity[1]
                    or type(snapshot.get("boot_id")) is not str
                    or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                                    snapshot["boot_id"]) is None):
                raise ValueError("process_identity_unavailable")
            return {k: snapshot[k] for k in ("pid", "birth_ticks", "boot_id")}

        opening_process = process_identity(lane_monitor._process_snapshot(identity[1]))
        kernel_identity = dict(opening_process)
        report, report_error = _read_watchdog_liveness_report(binding["status_path"])
        # Preserve a detached opening observation even when a test/adapter reuses
        # mutable objects; nonfinite values are not report evidence.
        frozen_report = json.dumps(report, sort_keys=True, allow_nan=False)
        closing_report, closing_error = _read_watchdog_liveness_report(binding["status_path"])
        if (report_error != closing_error
                or frozen_report != json.dumps(closing_report, sort_keys=True, allow_nan=False)):
            raise ValueError("report_changed")
        closing_process = process_identity(lane_monitor._process_snapshot(identity[1]))
        require_budget(20)
        closing_identity = lane_monitor._service_identity(lane_monitor.systemd_show(binding["service"]))
        if opening_process != closing_process or identity != closing_identity:
            raise ValueError("service_generation_changed")
        if config.get("watchdog_liveness") != frozen_binding:
            raise ValueError("binding_changed")
        # Lane checks can precede this attachment by many seconds. A closing
        # wall clock avoids calling a legitimately newer cycle report "future".
        closing_now = time.time()
        closing_monotonic = time.monotonic()
        if (type(closing_now) not in (int, float) or not math.isfinite(closing_now)
                or closing_now < now or not math.isfinite(closing_monotonic)
                or closing_monotonic < monotonic_now):
            raise ValueError("clock_changed")
        service_age = closing_monotonic - identity[2] / 1_000_000
        result["checked_at_epoch"] = closing_now
        result["service_age_seconds"] = service_age
        if report_error:
            if report_error == "missing" and service_age < binding["startup_grace_seconds"]:
                result.update(available=True, status="starting", attention_required=False,
                              reason_codes=["awaiting_first_report"])
                return result
            raise ValueError("report_unavailable")
        if (type(report) is not dict or report.get("schema") != "agent-supervisor/fleet-watchdog@1"
                or type(report.get("apply")) is not bool or type(report.get("boards")) is not dict):
            raise ValueError("report_invalid")
        declaration = report.get("liveness")
        if type(declaration) is not dict:
            raise ValueError("producer_identity_unbound")
        if declaration.get("available") is not True:
            raise ValueError("producer_identity_unavailable")
        if (set(declaration) != {"schema", "available", "process", "profile",
                                "source_binding", "action_authority"}
                or declaration.get("schema") != "agent-supervisor/watchdog-liveness@1"
                or declaration.get("source_binding") != "declared_profile_only"
                or declaration.get("action_authority") is not False
                or type(declaration.get("process")) is not dict
                or set(declaration["process"]) != {"pid", "birth_ticks", "boot_id"}
                or process_identity(declaration["process"]) != opening_process
                or type(declaration.get("profile")) is not dict
                or set(declaration["profile"]) != {"deployment_id", "source_sha256"}
                or declaration["profile"] != {k: binding[k] for k in ("deployment_id", "source_sha256")}):
            raise ValueError("producer_generation_mismatch")
        observed = report.get("observed_at")
        if (type(observed) not in (int, float) or not math.isfinite(observed)
                or observed <= 0 or observed > closing_now):
            raise ValueError("report_timestamp_invalid")
        age = closing_now - observed
        if age > service_age:
            raise ValueError("report_predates_service")
        stale = age > binding["stale_after_seconds"]
        result.update(available=True, status="stale" if stale else "healthy",
                      attention_required=stale, identity_verified=True,
                      declared_profile_matches=True, report_age_seconds=age,
                      reason_codes=["watchdog_report_stale"] if stale else [])
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        reason = str(exc)
        if re.fullmatch(r"[a-z_]{1,64}", reason) is None:
            reason = "observation_unavailable"
        result.update(available=False, status="unavailable", attention_required=True,
                      identity_verified=False, reason_codes=[reason],
                      error=f"{type(exc).__name__}: {exc}"[:512])
    return _watchdog_liveness_incident(result, binding, identity, kernel_identity)


def _attach_watchdog_liveness(config: dict[str, Any], summary: dict[str, Any], now: float, *,
                              deadline_monotonic: float | None = None) -> None:
    """Publish a separate deduplicated diagnostic, preserving action feedback."""
    try:
        attachment = watchdog_liveness_diagnostics(config, now, deadline_monotonic=deadline_monotonic)
    except Exception as exc:
        attachment = {"configured": True, "available": False, "status": "unavailable",
                      "attention_required": True, "action_authority": False,
                      "restart_eligible": False, "completion_authority": False,
                      "reason_codes": ["diagnostic_error"], "incident_id": None,
                      "error": f"{type(exc).__name__}: {exc}"[:512]}
    summary["watchdog_liveness"] = attachment
    previous, error = lane_monitor._read_json(config["health_path"])
    old = previous.get("watchdog_liveness", {}) if not error else {}
    old_id = old.get("incident_id") if type(old) is dict else None
    new_id = attachment.get("incident_id")
    if new_id != old_id and (new_id is not None or old_id is not None):
        try:
            lane_monitor._append_action(config["actions_path"], {
                "at_epoch": now, "action": "watchdog_liveness_observed",
                "status": attachment["status"], "incident_id": new_id,
                "previous_incident_id": old_id, "action_authority": False,
                "restart_eligible": False,
            })
        except (OSError, ValueError) as exc:
            # Optional diagnostic logging must not create a lane-health repair
            # trigger in consumers which do not understand this attachment.
            attachment["incident_log_error"] = f"{type(exc).__name__}: {exc}"[:512]


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
    # The optional observation must leave time to publish inside the existing
    # 90s oneshot. An absent binding adds no clock or watchdog I/O.
    liveness_deadline = time.monotonic() + 80 if "watchdog_liveness" in config else None
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
    if "watchdog_liveness" in config:
        _attach_watchdog_liveness(config, summary, now, deadline_monotonic=liveness_deadline)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Allow only each lane's existing bounded repairs")
    args = parser.parse_args(argv)
    try:
        summary = monitor_fleet(load_config(args.config), apply=args.apply)
    except (OSError, ValueError) as exc:
        print(json.dumps({"healthy": False, "status": "monitor_error", "error": str(exc)}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
