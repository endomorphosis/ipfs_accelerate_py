"""Read an existing service-fleet monitor report without controlling its services.

This is a diagnostic bridge for service-managed legacy boards. It never opens a
board database, probes processes, restarts services, or grants task completion or
publication authority. A fresh exhausted legacy board can be healthy and idle
without being the completion of the wider programme.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .live_board_probe import read_json_object

SCHEMA = "agent-supervisor/service-fleet-probe@1"
REPORT_SCHEMA = "ipfs-accelerate/board-service-fleet-monitor@1"
LANE_SCHEMA = "ipfs-accelerate/board-service-monitor@1"
DIAGNOSTIC_SCHEMA = "service-fleet-diagnostic/v1"
FUTURE_TOLERANCE_SECONDS = 5
COUNT_FIELDS = ("ready_count", "selectable_ready_count", "global_ready_count",
                "blocked_count", "waiting_count", "task_count", "completed_count")
NO_WORK_COUNTS = COUNT_FIELDS[:5]


class InvalidReport(ValueError):
    """A closed diagnostic code, never report text or an executable instruction."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and not any(
        ord(character) < 32 for character in value)


def _epoch(value: Any, now: float, *, fresh: float | None = None) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or value > now + FUTURE_TOLERANCE_SECONDS):
        raise InvalidReport("invalid_or_future_timestamp")
    if fresh is not None and now - value > fresh:
        raise InvalidReport("stale_report")
    return value


def _binding(board_id: str, namespace: str, units: Mapping[str, str], age: float) -> None:
    if (not _text(board_id) or not _text(namespace)
            or type(age) not in (int, float) or not math.isfinite(age) or age <= 0
            or not isinstance(units, Mapping) or not 1 <= len(units) <= 32):
        raise InvalidReport("invalid_probe_binding")
    if any(not _text(lane) or not _text(unit)
           or re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", unit) is None
           for lane, unit in units.items()) or len(set(units.values())) != len(units):
        raise InvalidReport("invalid_probe_binding")


BACKPRESSURE_IDLE_REASONS = frozenset({
    "provider_capacity_backoff", "all_selectable_ready_tasks_reached_max_task_attempts",
    "all_selectable_ready_tasks_blocked_on_attempt_accounting_evidence",
    "all_selectable_ready_tasks_deferred_by_resource_claim",
    "all_selectable_ready_tasks_deprioritized_as_off_mission",
    "no_eligible_ready_tasks_after_selection_filters", "virgin_transfer_rendezvous_pending",
})


def retry_backpressure_lane(lane: Any) -> bool:
    """Recognize an explicit live monitor pause, after scope/freshness checks.

    This is scheduling evidence, never task completion or permission to reset
    retries. A missing/dead worker, active task or restart-eligible diagnosis
    cannot become an intentional pause merely by supplying an idle string.
    """
    if (not isinstance(lane, dict) or lane.get("healthy") is not False
            or lane.get("status") != "backpressure" or lane.get("action") != "none"
            or lane.get("restart_eligible") is not False or lane.get("active") is not False
            or lane.get("native_daemon_pid_alive") is not True):
        return False
    idle = lane.get("selection_idle_reason")
    if (not isinstance(idle, str) or len(idle) > 256
            or any(ord(character) < 32 for character in idle)):
        return False
    return (lane.get("native_progress_status") == "backpressure"
            or idle in BACKPRESSURE_IDLE_REASONS
            or any(idle.startswith(prefix) and len(idle) > len(prefix)
                   for prefix in ("implementation_retry_deferred:", "resource_claim_deferred:")))


def backpressure_monitor_issues(healthy: Any, issues: Any, paused: set[str]) -> bool:
    """Do not conceal any fleet issue other than these exact paused lanes."""
    if healthy is not False or not paused or not isinstance(issues, list) or len(issues) != len(paused):
        return False
    seen = set()
    for issue in issues:
        if (not isinstance(issue, dict) or set(issue) != {"code", "lane_id", "status"}
                or issue.get("code") != "lane_unhealthy" or issue.get("status") != "backpressure"
                or not _text(issue.get("lane_id")) or issue["lane_id"] not in paused
                or issue["lane_id"] in seen):
            return False
        seen.add(issue["lane_id"])
    return seen == paused


def _lane(row: Any, units: Mapping[str, str], namespace: str,
          now: float, age: float) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("lane_id") not in units:
        raise InvalidReport("unexpected_lane")
    health = row.get("health")
    if not isinstance(health, dict) or health.get("schema") != LANE_SCHEMA:
        raise InvalidReport("invalid_lane_schema")
    lane_id = row["lane_id"]
    if health.get("service") != units[lane_id] or health.get("board_namespace") != namespace:
        raise InvalidReport("lane_scope_mismatch")
    if (type(health.get("healthy")) is not bool or not _text(health.get("status"))
            or not _text(health.get("action"))
            or health.get("completion_authority", False) is not False):
        raise InvalidReport("invalid_lane_health")
    evidence = health.get("evidence")
    if not isinstance(evidence, dict):
        raise InvalidReport("missing_lane_evidence")
    normalized = {"lane_id": lane_id, "service": units[lane_id],
                  "board_namespace": namespace,
                  "checked_at_epoch": _epoch(health.get("checked_at_epoch"), now, fresh=age),
                  "healthy": health["healthy"], "status": health["status"],
                  "action": health["action"]}
    for field in COUNT_FIELDS:
        value = evidence.get(field)
        if type(value) is not int or value < 0:
            raise InvalidReport("invalid_lane_count")
        normalized[field] = value
    if normalized["completed_count"] > normalized["task_count"]:
        raise InvalidReport("invalid_lane_count")
    for field in ("active", "native_daemon_pid_alive", "namespace_mismatch"):
        if type(evidence.get(field)) is not bool:
            raise InvalidReport("invalid_lane_boolean")
        normalized[field] = evidence[field]
    if normalized["namespace_mismatch"]:
        raise InvalidReport("lane_scope_mismatch")
    normalized["native_status_at_epoch"] = _epoch(
        evidence.get("native_status_at_epoch"), now, fresh=age)
    for field in ("last_progress_at_epoch", "last_work_event_at_epoch"):
        value = evidence.get(field)
        normalized[field] = None if value is None else _epoch(value, now)
    idle = evidence.get("selection_idle_reason")
    if not isinstance(idle, str) or len(idle) > 256 or any(ord(c) < 32 for c in idle):
        raise InvalidReport("invalid_idle_reason")
    normalized["selection_idle_reason"] = idle
    restart = health.get("restart_eligible")
    if "restart_eligible" in health and type(restart) is not bool:
        raise InvalidReport("invalid_restart_eligibility")
    normalized["restart_eligible"] = restart
    progress_status = evidence.get("native_progress_status")
    if progress_status is not None and not _text(progress_status):
        raise InvalidReport("invalid_native_progress_status")
    normalized["native_progress_status"] = progress_status
    return normalized


def observe_service_fleet(report_path: Path, *, board_id: str, board_namespace: str,
                          expected_units: Mapping[str, str], max_age_seconds: float = 180,
                          now: float | None = None) -> dict[str, Any]:
    """Map a fresh, exactly bound existing report to a non-authoritative probe.

    Each lane retains its own counts. Replicated whole-board counts are never
    added together. Timestamp/heartbeat refresh alone cannot change progress_token.
    Missing or stale evidence is unknown, rather than an invented zero or a
    request to start a competing owner.
    """
    now = time.time() if now is None else now
    service_fleet: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA, "board_namespace": board_namespace,
        "checked_at_epoch": None, "max_age_seconds": max_age_seconds,
        "work_availability": "unknown", "lanes": [], "completion_authority": False,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA, "board_id": board_id, "health": "unknown",
        "complete": False, "completion_candidate": False, "busy": False,
        "progress_token": "", "reason_codes": [],
        "details": {"service_fleet": service_fleet, "completion_authority": False},
    }
    try:
        _binding(board_id, board_namespace, expected_units, max_age_seconds)
        _epoch(now, now)
        report = read_json_object(Path(report_path))
        if report.get("schema") != REPORT_SCHEMA:
            raise InvalidReport("invalid_report_schema")
        if (report.get("completion_authority") is not False
                or type(report.get("healthy")) is not bool or not _text(report.get("status"))):
            raise InvalidReport("invalid_report_health")
        service_fleet["checked_at_epoch"] = _epoch(report.get("checked_at_epoch"), now,
                                                   fresh=max_age_seconds)
        rows = report.get("lanes")
        if (not isinstance(rows, list) or type(report.get("lane_count")) is not int
                or report["lane_count"] != len(expected_units) or len(rows) != len(expected_units)):
            raise InvalidReport("lane_population_mismatch")
        lanes = [_lane(row, expected_units, board_namespace, now, max_age_seconds) for row in rows]
        if len({lane["lane_id"] for lane in lanes}) != len(expected_units):
            raise InvalidReport("lane_population_mismatch")
        lanes.sort(key=lambda lane: lane["lane_id"])
        service_fleet["lanes"] = lanes
        paused = {lane["lane_id"] for lane in lanes if retry_backpressure_lane(lane)}
        paused_issues_only = (all(lane["healthy"] or lane["lane_id"] in paused for lane in lanes)
                             and backpressure_monitor_issues(report["healthy"], report.get("issues"), paused))
        service_fleet["monitor_healthy"] = report["healthy"]
        service_fleet["monitor_issues"] = ([{"code": "lane_unhealthy", "lane_id": lane,
                                             "status": "backpressure"} for lane in sorted(paused)]
                                            if paused_issues_only else [])
        if (not report["healthy"] or not all(lane["healthy"] for lane in lanes)) and not paused_issues_only:
            result.update(health="degraded", reason_codes=["service_fleet_unhealthy"])
            return result
        if not all(lane["native_daemon_pid_alive"] for lane in lanes):
            result["reason_codes"] = ["service_fleet_native_liveness_unavailable"]
            return result
        # Only declared work observations affect the token, never report/native
        # heartbeat timestamps, selection cursors, action labels or source heads.
        progress = [[lane["lane_id"], lane["completed_count"],
                     lane["last_progress_at_epoch"], lane["last_work_event_at_epoch"]]
                    for lane in lanes]
        result["progress_token"] = hashlib.sha256(
            json.dumps([board_namespace, progress], separators=(",", ":")).encode()).hexdigest()
        active = any(lane["active"] for lane in lanes)
        ready = any(any(lane[key] > 0 for key in NO_WORK_COUNTS[:3]) for lane in lanes)
        waiting = any(lane["blocked_count"] or lane["waiting_count"] for lane in lanes)
        independent_ready = any(lane["lane_id"] not in paused
                                and (lane["ready_count"] or lane["selectable_ready_count"]) for lane in lanes)
        if paused and not active and not independent_ready:
            service_fleet["work_availability"] = "backpressure_wait"
            result.update(health="blocked", reason_codes=["service_fleet_backpressure_wait"])
            result["details"]["wait"] = {"kind": "native_retry_or_resource_fence", "completion_authority": False}
            return result
        if not active and not ready and waiting:
            service_fleet["work_availability"] = "dependency_wait"
            result.update(health="blocked", reason_codes=["external_dependency_wait"])
            result["details"]["wait"] = {"kind": "dependency_or_review",
                                          "completion_authority": False}
            return result
        if not active and not any(any(lane[key] for key in NO_WORK_COUNTS) for lane in lanes):
            service_fleet["work_availability"] = "no_runnable_work"
            result.update(health="healthy", reason_codes=["service_fleet_no_runnable_work"])
        else:
            service_fleet["work_availability"] = "runnable_work"
            result.update(health="healthy", busy=active,
                          reason_codes=["service_fleet_work_observed"])
        return result
    except InvalidReport as error:
        result["reason_codes"] = ["service_fleet_report_invalid", str(error)]
    except (OSError, ValueError, TypeError, RecursionError):
        result["reason_codes"] = ["service_fleet_report_unavailable"]
    return result


def _expected_unit(value: str) -> tuple[str, str]:
    lane, separator, unit = value.partition("=")
    if not separator or not lane or not unit:
        raise argparse.ArgumentTypeError("expected lane-id=unit.service")
    return lane, unit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--max-age-seconds", type=float, default=180)
    parser.add_argument("--expected-unit", type=_expected_unit, action="append", required=True)
    args = parser.parse_args(argv)
    units = dict(args.expected_unit)
    if len(units) != len(args.expected_unit):
        parser.error("expected lane IDs must be unique")
    print(json.dumps(observe_service_fleet(
        args.report, board_id=args.board, board_namespace=args.namespace,
        expected_units=units, max_age_seconds=args.max_age_seconds), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
