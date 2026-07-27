"""Outer watchdog process that monitors and restarts the bundle supervisor.

This provides a second level of fault tolerance: even if the bundle supervisor
itself crashes or hangs, this watchdog will detect the failure and restart it.

Designed for systemd-style long-running operation (hours/days/weeks).

Usage:
    python -m ipfs_accelerate_py.agent_supervisor.supervisor_watchdog \
        --manifest-path data/agent_supervisor/bundle_lanes/bundle_lanes.json \
        --repo-root .

Environment variables:
    WATCHDOG_CHECK_INTERVAL_SECONDS: How often to check lane health (default: 120)
    WATCHDOG_LANE_TIMEOUT_SECONDS: Consider a lane dead if no heartbeat for this long (default: 600)
    WATCHDOG_MAX_CONSECUTIVE_RESTARTS: After this many rapid restarts, back off (default: 5)
    WATCHDOG_LOG_AGGREGATION_DIR: Directory for unified structured logs (default: state_root/logs/aggregated)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from .control_plane import (
    LIFECYCLE_STATUS_SCHEMA,
    SupervisorLifecycleState,
)
from .scheduler_metrics import scheduler_snapshot, scheduler_state_events

logger = logging.getLogger(__name__)

SUPERVISOR_LIFECYCLE_STATES: Final[tuple[str, ...]] = (
    tuple(state.value for state in SupervisorLifecycleState)
)
_LIFECYCLE_STATE_SET: Final[frozenset[str]] = frozenset(
    SUPERVISOR_LIFECYCLE_STATES
)
_TRANSITIONAL_STATES: Final[frozenset[str]] = frozenset(
    {"starting", "draining", "stopping"}
)
_INTENTIONAL_NON_RUNNING_STATES: Final[frozenset[str]] = frozenset(
    {"paused", "draining", "blocked", "stopping", "stopped", "failed"}
)
_STATE_ALIASES: Final[dict[str, str]] = {
    "alive": "healthy",
    "ok": "healthy",
    "running": "healthy",
    "ready": "healthy",
    "unhealthy": "degraded",
    "hung": "degraded",
    "error": "failed",
    "crashed": "failed",
    "shutdown": "stopped",
    "shutdown_complete": "stopped",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    """Check if a process with given PID is alive."""
    if isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Permission denial still proves that a process owns the PID.
        return True
    except OSError:
        return False


def _lifecycle_state(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = _STATE_ALIASES.get(text, text)
    return text if text in _LIFECYCLE_STATE_SET else ""


def _timestamp_seconds(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # LifecycleStatus exposes millisecond timestamps.
        return number / 1000.0 if number > 10_000_000_000 else number
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _nonnegative_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _timestamp_text_from_ms(value: int) -> str:
    if value <= 0:
        return ""
    return (
        datetime.fromtimestamp(value / 1000.0, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _status_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _active_leases(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("active_leases")
    if isinstance(value, (list, tuple)):
        return sorted(
            {str(item).strip() for item in value if str(item).strip()}
        )[:256]
    if isinstance(value, Mapping):
        return sorted(
            {str(item).strip() for item in value if str(item).strip()}
        )[:256]
    lease_id = str(payload.get("lease_id") or "").strip()
    return [lease_id] if lease_id else []


def lifecycle_status_projection(
    *,
    pid_check: Mapping[str, Any],
    heartbeat_check: Mapping[str, Any],
    state: str | None = None,
    phase: str | None = None,
    target_id: str = "supervisor:lane",
) -> dict[str, Any]:
    """Project lane observations into the shared lifecycle status schema.

    The control plane owns lifecycle mutation policy.  The watchdog only
    reconciles process/heartbeat observations and emits the same public shape,
    so a caller never has to interpret a second set of health fields.
    """

    observed_state = _lifecycle_state(state or heartbeat_check.get("state"))
    alive = bool(pid_check.get("alive"))
    stale = bool(heartbeat_check.get("stale"))
    if not observed_state:
        observed_state = "healthy" if alive and not stale else (
            "degraded" if alive else "stopped"
        )
    if alive and stale and observed_state not in _INTENTIONAL_NON_RUNNING_STATES:
        observed_state = "degraded"
    elif not alive and observed_state == "healthy":
        observed_state = "degraded"

    heartbeat_at = str(heartbeat_check.get("heartbeat_at") or "")
    heartbeat_at_ms = heartbeat_check.get("heartbeat_at_ms")
    if heartbeat_at_ms is None:
        timestamp = _timestamp_seconds(heartbeat_at)
        heartbeat_at_ms = None if timestamp is None else int(timestamp * 1000)
    heartbeat_at_ms = _nonnegative_int(heartbeat_at_ms)
    heartbeat_at = _timestamp_text_from_ms(heartbeat_at_ms)

    updated_at = str(heartbeat_check.get("updated_at") or heartbeat_at or "")
    updated_at_ms = heartbeat_check.get("updated_at_ms")
    if updated_at_ms is None:
        timestamp = _timestamp_seconds(updated_at)
        updated_at_ms = None if timestamp is None else int(timestamp * 1000)
    updated_at_ms = _nonnegative_int(updated_at_ms)
    updated_at = _timestamp_text_from_ms(updated_at_ms)

    reasons = heartbeat_check.get("backpressure_reasons")
    if not isinstance(reasons, (list, tuple)):
        reasons = []
    leases = sorted(
        {
            str(item).strip()
            for item in heartbeat_check.get("active_leases", ())
            if str(item).strip()
        }
    )[:256]
    result: dict[str, Any] = {
        "schema": LIFECYCLE_STATUS_SCHEMA,
        "target_id": str(target_id or "supervisor:lane"),
        "state": observed_state,
        "phase": str(phase or heartbeat_check.get("phase") or observed_state),
        "heartbeat_at_ms": heartbeat_at_ms,
        "heartbeat_at": heartbeat_at,
        "pid": (
            pid_check.get("pid")
            if isinstance(pid_check.get("pid"), int)
            and not isinstance(pid_check.get("pid"), bool)
            and pid_check.get("pid") > 0
            else None
        ),
        "active_leases": leases,
        "active_lease_count": len(leases),
        "refill_state": str(heartbeat_check.get("refill_state") or "idle"),
        "backpressure": bool(heartbeat_check.get("backpressure", False)),
        "backpressure_reasons": sorted(
            {str(item).strip() for item in reasons if str(item).strip()}
        )[:256],
        "terminal_reason": str(heartbeat_check.get("terminal_reason") or ""),
        "transition_id": str(heartbeat_check.get("transition_id") or ""),
        "generation": _nonnegative_int(heartbeat_check.get("generation")),
        "fencing_epoch": (
            None
            if heartbeat_check.get("fencing_epoch") in (None, "")
            else _nonnegative_int(heartbeat_check.get("fencing_epoch"))
        ),
        "updated_at_ms": updated_at_ms,
        "updated_at": updated_at,
    }
    return result


def watchdog_process_status(
    state: str,
    *,
    phase: str,
    terminal_reason: str = "",
    backpressure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a canonical status for watchdog-level outcomes."""

    now_ms = int(time.time() * 1000)
    running = state != "stopped"
    return lifecycle_status_projection(
        pid_check={
            "pid": os.getpid() if running else None,
            "alive": running,
        },
        heartbeat_check={
            "state": state,
            "phase": phase,
            "heartbeat_at_ms": now_ms,
            "updated_at_ms": now_ms,
            "active_leases": (),
            "refill_state": "idle",
            "backpressure": bool(backpressure_reasons),
            "backpressure_reasons": tuple(backpressure_reasons),
            "terminal_reason": terminal_reason,
            "generation": 0,
            "fencing_epoch": None,
        },
        target_id="supervisor:watchdog",
    )


def read_lane_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read the bundle lane manifest JSON."""
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read manifest %s: %s", manifest_path, exc)
        return {}


def read_scheduler_snapshot(manifest_path: Path) -> dict[str, Any]:
    """Return the exact operator snapshot embedded by the live scheduler."""

    manifest = read_lane_manifest(manifest_path)
    snapshot = manifest.get("scheduler_snapshot")
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def check_lane_pid(state_dir: Path, state_prefix: str) -> dict[str, Any]:
    """Check if a lane's supervisor process is alive by its PID file."""
    pid_path = state_dir / f"{state_prefix}_bundle_supervisor.pid"
    result: dict[str, Any] = {"pid_path": str(pid_path), "alive": False}

    if not pid_path.exists():
        result["reason"] = "no_pid_file"
        return result

    try:
        pid = int(pid_path.read_text().strip())
        result["pid"] = pid
        if pid <= 0:
            result["reason"] = "invalid_pid"
        elif pid_alive(pid):
            result["alive"] = True
        else:
            result["reason"] = "process_dead"
    except (ValueError, OSError) as exc:
        result["reason"] = f"pid_read_error: {exc}"

    return result


def check_lane_heartbeat(state_dir: Path, state_prefix: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Check if a lane has updated its status file recently."""
    status_path = state_dir / f"{state_prefix}_status.json"
    result: dict[str, Any] = {"status_path": str(status_path), "stale": False}

    if not status_path.exists():
        result["stale"] = True
        result["reason"] = "no_status_file"
        return result

    try:
        stat = status_path.stat()
        status = _status_payload(status_path)
        if status.get("heartbeat_at_ms") not in (None, ""):
            try:
                heartbeat_timestamp = float(status["heartbeat_at_ms"]) / 1000.0
            except (TypeError, ValueError):
                heartbeat_timestamp = None
        elif status.get("heartbeat_at") not in (None, ""):
            heartbeat_timestamp = _timestamp_seconds(status["heartbeat_at"])
        elif status.get("updated_at_ms") not in (None, ""):
            try:
                heartbeat_timestamp = float(status["updated_at_ms"]) / 1000.0
            except (TypeError, ValueError):
                heartbeat_timestamp = None
        else:
            heartbeat_timestamp = _timestamp_seconds(
                status.get("updated_at") or status.get("timestamp")
            )
        age_seconds = time.time() - (
            heartbeat_timestamp if heartbeat_timestamp is not None else stat.st_mtime
        )
        # Future timestamps can occur under small host clock skews.
        age_seconds = max(0.0, age_seconds)
        result["age_seconds"] = age_seconds
        if age_seconds > timeout_seconds:
            result["stale"] = True
            result["reason"] = "heartbeat_timeout"
        result["phase"] = str(status.get("active_phase") or status.get("phase") or "")
        result["schema"] = str(status.get("schema") or "")
        result["active_task_id"] = str(status.get("active_task_id") or "")
        result["heartbeat_at"] = str(status.get("heartbeat_at") or "")
        result["heartbeat_at_ms"] = status.get("heartbeat_at_ms")
        result["updated_at"] = str(status.get("updated_at") or "")
        result["updated_at_ms"] = status.get("updated_at_ms")
        result["state"] = _lifecycle_state(
            status.get("state")
            or status.get("lifecycle_state")
            or status.get("status")
        )
        result["active_leases"] = _active_leases(status)
        result["refill_state"] = status.get("refill_state")
        result["backpressure"] = bool(status.get("backpressure", False))
        raw_reasons = status.get("backpressure_reasons")
        result["backpressure_reasons"] = (
            list(raw_reasons[:256])
            if isinstance(raw_reasons, (list, tuple))
            else []
        )
        result["terminal_reason"] = str(
            status.get("terminal_reason") or status.get("reason") or ""
        )
        result["transition_id"] = str(status.get("transition_id") or "")
        result["generation"] = status.get("generation") or 0
        result["fencing_epoch"] = status.get("fencing_epoch")
        status_pid = status.get("pid")
        if status_pid not in (None, ""):
            try:
                result["status_pid"] = int(status_pid)
            except (TypeError, ValueError):
                result["state_inconsistent"] = True
                result["state_reason"] = "invalid_status_pid"
    except OSError as exc:
        result["stale"] = True
        result["reason"] = f"stat_error: {exc}"

    return result


def _replace_pid_file(pid_path: Path, pid: int) -> None:
    """Atomically repair a PID file from a live canonical status record."""

    temporary = pid_path.with_name(
        f".{pid_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        temporary.write_text(f"{pid}\n", encoding="utf-8")
        os.replace(temporary, pid_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def restart_lane(
    lane_info: dict[str, Any],
    *,
    repo_root: Path,
    lifecycle_transition: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Delegate restart to an authorized fenced lifecycle transition.

    A watchdog observation and a manifest command are not signal or launch
    authority.  In particular, a dead PID file cannot prove that detached
    descendants are absent.  The embedding control service supplies a callback
    which resolves the bound ``OperationRequest`` and invokes
    :class:`LifecycleOrchestrator`; without it the watchdog reports the missing
    authority instead of performing a raw ``Popen``.
    """

    del repo_root
    if lifecycle_transition is None:
        return {
            "restarted": False,
            "reason": "lifecycle_orchestrator_required",
            "control_recovery_required": True,
        }
    try:
        raw = lifecycle_transition(dict(lane_info))
    except Exception as exc:
        return {
            "restarted": False,
            "reason": f"lifecycle_transition_failed: {exc}",
            "control_recovery_required": True,
        }
    result = dict(raw) if isinstance(raw, Mapping) else {}
    transition = result.get("transition")
    if isinstance(transition, Mapping):
        result.setdefault("receipt_id", transition.get("receipt_id"))
        new_tree = transition.get("new_tree")
        if isinstance(new_tree, Mapping):
            members = new_tree.get("members")
            if isinstance(members, Sequence) and members:
                first = members[0]
                if isinstance(first, Mapping):
                    result.setdefault("new_pid", first.get("pid"))
    restarted = bool(
        result.get("restarted")
        or result.get("succeeded")
        or result.get("status") == "succeeded"
        or (
            isinstance(transition, Mapping)
            and transition.get("phase") == "committed"
        )
    )
    result["restarted"] = restarted
    result.setdefault("pid_persisted", False)
    if restarted and not isinstance(result.get("new_pid"), int):
        return {
            **result,
            "restarted": False,
            "reason": "lifecycle_receipt_missing_new_process_identity",
        }
    return result


def aggregate_logs(
    lanes: Sequence[dict[str, Any]],
    *,
    repo_root: Path,
    output_dir: Path,
    max_lines_per_lane: int = 100,
) -> dict[str, Any]:
    """Aggregate recent log lines from all lanes into a unified structured log.

    This provides a single view of what's happening across all parallel lanes
    without needing to tail multiple log files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated: list[dict[str, Any]] = []

    for lane in lanes:
        log_path_str = lane.get("log_path", "")
        if not log_path_str:
            continue
        log_path = Path(log_path_str) if Path(log_path_str).is_absolute() else repo_root / log_path_str
        if not log_path.exists():
            continue

        bundle_key = lane.get("bundle_key", "unknown")
        try:
            # Read last N lines efficiently
            with log_path.open("rb") as f:
                # Seek to end, read backward to find last N lines
                f.seek(0, 2)
                file_size = f.tell()
                # Read last 64KB or whole file
                read_size = min(file_size, 65536)
                f.seek(max(0, file_size - read_size))
                tail_bytes = f.read()

            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()[-max_lines_per_lane:]
            for line in lines:
                aggregated.append({
                    "lane": bundle_key,
                    "line": line,
                })
        except OSError:
            continue

    # Write aggregated log
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"aggregated_{timestamp}.jsonl"
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for entry in aggregated:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        return {"aggregated": False, "error": str(exc)}

    # Prune old aggregated logs (keep last 24)
    existing = sorted(output_dir.glob("aggregated_*.jsonl"))
    for old_file in existing[:-24]:
        try:
            old_file.unlink()
        except OSError:
            pass

    return {
        "aggregated": True,
        "output_path": str(output_path),
        "total_lines": len(aggregated),
        "lane_count": len(lanes),
    }


class SupervisorWatchdog:
    """Outer watchdog that monitors bundle supervisor lanes."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        repo_root: Path,
        check_interval: float = 120.0,
        lane_timeout: float = 600.0,
        max_consecutive_restarts: int = 5,
        log_aggregation_dir: Path | None = None,
        lifecycle_restart: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ]
        | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.repo_root = repo_root
        self.check_interval = check_interval
        self.lane_timeout = lane_timeout
        self.max_consecutive_restarts = max_consecutive_restarts
        self.log_aggregation_dir = log_aggregation_dir or (
            manifest_path.parent / "logs" / "aggregated"
        )
        self.lifecycle_restart = lifecycle_restart
        self._consecutive_restart_counts: dict[str, int] = {}
        self._recent_restarts: dict[str, tuple[int, float]] = {}
        self._generation = 0
        self._running = True

    def run(self) -> dict[str, Any]:
        """Run the watchdog loop indefinitely."""
        logger.info(
            "Watchdog started: manifest=%s, check_interval=%.0fs, lane_timeout=%.0fs",
            self.manifest_path,
            self.check_interval,
            self.lane_timeout,
        )

        # Handle SIGTERM gracefully
        def handle_signal(signum, frame):
            logger.info("Watchdog received signal %d, shutting down", signum)
            self._running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        total_checks = 0
        total_restarts = 0

        while self._running:
            try:
                report = self._check_cycle()
                total_checks += 1
                total_restarts += report.get("restarts", 0)

                if report.get("restarts", 0) > 0:
                    logger.warning(
                        "Watchdog cycle %d: restarted %d lanes",
                        total_checks,
                        report["restarts"],
                    )
                else:
                    logger.debug("Watchdog cycle %d: all lanes healthy", total_checks)

            except Exception as exc:
                logger.error("Watchdog cycle error: %s", exc, exc_info=True)

            # Sleep in small increments so we can respond to signals
            sleep_remaining = self.check_interval
            while sleep_remaining > 0 and self._running:
                chunk = min(sleep_remaining, 5.0)
                time.sleep(chunk)
                sleep_remaining -= chunk

        return {
            "status": watchdog_process_status(
                "stopped",
                phase="watchdog_stopped",
                terminal_reason="watchdog_shutdown",
            ),
            "legacy_status": "shutdown",
            "total_checks": total_checks,
            "total_restarts": total_restarts,
        }

    def _check_cycle(self) -> dict[str, Any]:
        """Run one health-check cycle across all lanes."""
        manifest = read_lane_manifest(self.manifest_path)
        if not manifest:
            return {
                "timestamp": utc_now(),
                "error": "manifest_empty",
                "restarts": 0,
                "status": watchdog_process_status(
                    "blocked",
                    phase="manifest_unavailable",
                    backpressure_reasons=("manifest_empty",),
                ),
            }

        lanes = manifest.get("lanes", [])
        started = manifest.get("started", [])
        dynamic_authority = bool(manifest.get("authoritative")) and str(
            manifest.get("schema") or ""
        ).startswith("ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler")

        restarts = 0
        reports: list[dict[str, Any]] = []

        for i, lane in enumerate(lanes):
            bundle_key = lane.get("bundle_key", f"lane_{i}")
            state_dir = Path(lane.get("state_dir", ""))
            state_prefix = lane.get("state_prefix", "")

            if not state_dir.is_absolute():
                state_dir = self.repo_root / state_dir

            # Find matching started info
            lane_started = next(
                (s for s in started if s.get("bundle_key") == bundle_key),
                {},
            )

            # Check PID
            pid_check = check_lane_pid(state_dir, state_prefix)

            # Check heartbeat
            heartbeat_check = check_lane_heartbeat(
                state_dir, state_prefix, timeout_seconds=self.lane_timeout
            )
            canonical_status = (
                heartbeat_check.get("schema") == LIFECYCLE_STATUS_SCHEMA
            )
            observed_state = _lifecycle_state(heartbeat_check.get("state"))
            recovery: dict[str, Any] = {}

            # The canonical status binds the process it describes.  Repair a
            # stale PID file from a live status PID instead of launching an
            # unfenced duplicate.
            status_pid = heartbeat_check.get("status_pid")
            pid_file_pid = pid_check.get("pid")
            untrusted_live_status_pid = False
            if (
                canonical_status
                and isinstance(status_pid, int)
                and status_pid > 0
                and status_pid != pid_file_pid
            ):
                status_pid_alive = pid_alive(status_pid)
                if status_pid_alive and not heartbeat_check.get("stale", False):
                    pid_path = Path(str(pid_check["pid_path"]))
                    try:
                        pid_path.parent.mkdir(parents=True, exist_ok=True)
                        _replace_pid_file(pid_path, status_pid)
                    except OSError as exc:
                        heartbeat_check["state_inconsistent"] = True
                        heartbeat_check["state_reason"] = (
                            f"pid_file_repair_error: {exc}"
                        )
                    else:
                        recovery = {
                            "kind": "stale_pid_file",
                            "previous_pid": pid_file_pid,
                            "recovered_pid": status_pid,
                        }
                        self._generation += 1
                        pid_check = {
                            **pid_check,
                            "pid": status_pid,
                            "alive": True,
                        }
                        pid_check.pop("reason", None)
                elif status_pid_alive:
                    untrusted_live_status_pid = True
                    heartbeat_check["state_inconsistent"] = True
                    heartbeat_check["state_reason"] = "stale_status_pid_alive"
                else:
                    heartbeat_check["state_inconsistent"] = True
                    heartbeat_check["state_reason"] = "status_pid_dead"

            report = {
                "bundle_key": bundle_key,
                "pid_check": pid_check,
                "heartbeat_check": heartbeat_check,
                "action": "none",
            }

            alive = bool(pid_check["alive"])
            heartbeat_stale = bool(heartbeat_check.get("stale", False))
            inconsistent = bool(heartbeat_check.get("state_inconsistent"))
            recent = self._recent_restarts.get(bundle_key)
            in_startup_grace = bool(
                recent
                and alive
                and pid_check.get("pid") == recent[0]
                and time.monotonic() - recent[1] <= self.lane_timeout
            )
            if in_startup_grace and heartbeat_stale:
                observed_state = "starting"
                recovery = {
                    "kind": "restart_heartbeat_pending",
                    "pid": pid_check.get("pid"),
                }
            elif recent and (
                not heartbeat_stale or pid_check.get("pid") != recent[0]
            ):
                self._recent_restarts.pop(bundle_key, None)

            terminal_process_conflict = (
                canonical_status
                and observed_state in {"stopped", "failed"}
                and alive
            )
            if terminal_process_conflict:
                inconsistent = True
                heartbeat_check["state_inconsistent"] = True
                heartbeat_check["state_reason"] = "terminal_state_pid_alive"
            suppressed_state = (
                canonical_status
                and observed_state in _INTENTIONAL_NON_RUNNING_STATES
                and not terminal_process_conflict
                and not (alive and heartbeat_stale)
            )
            needs_restart = (
                not in_startup_grace
                and not suppressed_state
                and (not alive or heartbeat_stale or inconsistent)
            )

            if needs_restart:
                if dynamic_authority:
                    # The persistent scheduler owns the fenced lease and is
                    # the only process allowed to replace its leased wrapper.
                    # Starting the raw lane command here would create an
                    # unfenced duplicate.
                    report["action"] = "scheduler_recovery_required"
                    report["reason"] = "dynamic_scheduler_owns_restart"
                    report["status"] = lifecycle_status_projection(
                        pid_check=pid_check,
                        heartbeat_check=heartbeat_check,
                        state="blocked",
                        target_id=str(bundle_key),
                    )
                    if recovery:
                        report["recovery"] = recovery
                    reports.append(report)
                    continue
                if canonical_status and (alive or untrusted_live_status_pid):
                    # Replacing a live canonical process requires the control
                    # plane's lease/fencing check.  The watchdog reports the
                    # requirement and never starts a second raw lane.
                    report["action"] = "fenced_stop_required"
                    report["reason"] = str(
                        heartbeat_check.get("state_reason")
                        or heartbeat_check.get("reason")
                        or "live_process_unhealthy"
                    )
                    observed_state = "blocked"
                # Check consecutive restart count for backoff
                count = self._consecutive_restart_counts.get(bundle_key, 0)
                if report["action"] == "fenced_stop_required":
                    pass
                elif count >= self.max_consecutive_restarts:
                    # Exponential backoff: skip this cycle
                    backoff_cycles = 2 ** min(count - self.max_consecutive_restarts, 5)
                    report["action"] = "backoff"
                    report["backoff_cycles"] = backoff_cycles
                    # Decrement so we'll try again eventually
                    self._consecutive_restart_counts[bundle_key] = count + 1
                else:
                    restart_info = dict(lane_started or lane)
                    restart_info.setdefault("pid_path", str(pid_check["pid_path"]))
                    if self.lifecycle_restart is None:
                        # Keep the call shape compatible with injected test and
                        # deployment shims.  The built-in helper fails closed.
                        restart_result = restart_lane(
                            restart_info, repo_root=self.repo_root
                        )
                    else:
                        restart_result = restart_lane(
                            restart_info,
                            repo_root=self.repo_root,
                            lifecycle_transition=self.lifecycle_restart,
                        )
                    report["action"] = "restarted" if restart_result.get("restarted") else "restart_failed"
                    report["restart_result"] = restart_result
                    if restart_result.get("restarted"):
                        restarts += 1
                        self._consecutive_restart_counts[bundle_key] = count + 1
                        new_pid = int(restart_result["new_pid"])
                        self._recent_restarts[bundle_key] = (
                            new_pid,
                            time.monotonic(),
                        )
                        self._generation += 1
                        if observed_state in _TRANSITIONAL_STATES:
                            recovery = {
                                "kind": "interrupted_transition",
                                "previous_state": observed_state,
                                "restarted_pid": new_pid,
                            }
                        observed_state = "starting"
                        pid_check = {
                            **pid_check,
                            "pid": new_pid,
                            "alive": True,
                        }
                        pid_check.pop("reason", None)
                    else:
                        self._consecutive_restart_counts[bundle_key] = count + 1
                        observed_state = "failed"
            else:
                # Healthy - reset consecutive restart counter
                self._consecutive_restart_counts[bundle_key] = 0
                if suppressed_state:
                    if (
                        not alive
                        and observed_state
                        in {"paused", "draining", "stopping"}
                    ):
                        interrupted_state = observed_state
                        if interrupted_state == "stopping":
                            observed_state = "stopped"
                            terminal_reason = (
                                "stop_completed_after_process_exit"
                            )
                        else:
                            observed_state = "failed"
                            terminal_reason = (
                                f"{interrupted_state}_process_exited"
                            )
                        heartbeat_check["terminal_reason"] = terminal_reason
                        report["action"] = "control_recovery_required"
                        report["reason"] = terminal_reason
                        recovery = {
                            "kind": "interrupted_transition",
                            "previous_state": interrupted_state,
                            "recovered_state": observed_state,
                        }
                    else:
                        report["action"] = "state_preserved"
                        report["reason"] = f"canonical_{observed_state}"

            report["status"] = lifecycle_status_projection(
                pid_check=pid_check,
                heartbeat_check=heartbeat_check,
                state=observed_state,
                target_id=str(bundle_key),
            )
            if recovery:
                report["recovery"] = recovery
            reports.append(report)

        # Aggregate logs periodically
        if started:
            try:
                aggregate_logs(
                    started,
                    repo_root=self.repo_root,
                    output_dir=self.log_aggregation_dir,
                )
            except Exception as exc:
                logger.debug("Log aggregation failed: %s", exc)

        embedded_snapshot = manifest.get("scheduler_snapshot")
        if isinstance(embedded_snapshot, dict):
            operator_snapshot = dict(embedded_snapshot)
        else:
            events = scheduler_state_events(
                [dict(lane) for lane in lanes if isinstance(lane, dict)],
                timestamp=utc_now(),
            )
            operator_snapshot = scheduler_snapshot(events).to_dict()

        timestamp = utc_now()
        lane_states = [
            str(item.get("status", {}).get("state") or "") for item in reports
        ]
        state_priority = (
            "failed",
            "blocked",
            "degraded",
            "starting",
            "stopping",
            "draining",
        )
        aggregate_state = next(
            (state for state in state_priority if state in lane_states),
            (
                "stopped"
                if not lane_states or all(state == "stopped" for state in lane_states)
                else (
                    "paused"
                    if all(state == "paused" for state in lane_states)
                    else "healthy"
                )
            ),
        )
        timestamp_seconds = _timestamp_seconds(timestamp)
        timestamp_ms = (
            None if timestamp_seconds is None else int(timestamp_seconds * 1000)
        )
        status_timestamp = _timestamp_text_from_ms(timestamp_ms or 0)
        aggregate_leases = sorted(
            {
                str(lease)
                for item in reports
                for lease in item.get("status", {}).get("active_leases", ())
                if str(lease)
            }
        )[:256]
        aggregate_status = {
            "schema": LIFECYCLE_STATUS_SCHEMA,
            "target_id": "supervisor:watchdog",
            "state": aggregate_state,
            "phase": "watchdog_check",
            "heartbeat_at_ms": timestamp_ms,
            "heartbeat_at": status_timestamp,
            "pid": os.getpid(),
            "active_leases": aggregate_leases,
            "active_lease_count": len(aggregate_leases),
            "refill_state": "idle",
            "backpressure": any(
                bool(item.get("status", {}).get("backpressure"))
                for item in reports
            ),
            "backpressure_reasons": sorted(
                {
                    str(reason)
                    for item in reports
                    for reason in item.get("status", {}).get(
                        "backpressure_reasons", ()
                    )
                }
            )[:256],
            "terminal_reason": "",
            "transition_id": "",
            "generation": self._generation,
            "fencing_epoch": 0,
            "updated_at_ms": timestamp_ms,
            "updated_at": status_timestamp,
        }
        return {
            "timestamp": timestamp,
            "lane_count": len(lanes),
            "restarts": restarts,
            "reports": reports,
            "status": aggregate_status,
            "scheduler_snapshot": operator_snapshot,
            "scheduler_snapshot_id": str(operator_snapshot.get("snapshot_id") or ""),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outer watchdog that monitors and restarts bundle supervisor lanes"
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check-interval",
        type=float,
        default=float(os.environ.get("WATCHDOG_CHECK_INTERVAL_SECONDS", "120")),
    )
    parser.add_argument(
        "--lane-timeout",
        type=float,
        default=float(os.environ.get("WATCHDOG_LANE_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--max-consecutive-restarts",
        type=int,
        default=int(os.environ.get("WATCHDOG_MAX_CONSECUTIVE_RESTARTS", "5")),
    )
    parser.add_argument("--log-aggregation-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watchdog = SupervisorWatchdog(
        manifest_path=args.manifest_path.resolve(),
        repo_root=args.repo_root.resolve(),
        check_interval=args.check_interval,
        lane_timeout=args.lane_timeout,
        max_consecutive_restarts=args.max_consecutive_restarts,
        log_aggregation_dir=args.log_aggregation_dir,
    )

    result = watchdog.run()
    logger.info("Watchdog exited: %s", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
