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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    """Check if a process with given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def read_lane_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read the bundle lane manifest JSON."""
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read manifest %s: %s", manifest_path, exc)
        return {}


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
        if pid_alive(pid):
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
        age_seconds = time.time() - stat.st_mtime
        result["age_seconds"] = age_seconds
        if age_seconds > timeout_seconds:
            result["stale"] = True
            result["reason"] = "heartbeat_timeout"
    except OSError as exc:
        result["stale"] = True
        result["reason"] = f"stat_error: {exc}"

    return result


def restart_lane(lane_info: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Restart a dead or stale lane process."""
    import subprocess

    command = lane_info.get("command", [])
    if not command:
        return {"restarted": False, "reason": "no_command"}

    log_path = lane_info.get("log_path", "")
    if log_path:
        full_log_path = Path(log_path) if Path(log_path).is_absolute() else repo_root / log_path
        full_log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = full_log_path.open("ab")
    else:
        handle = subprocess.DEVNULL

    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=handle if handle != subprocess.DEVNULL else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        return {"restarted": False, "reason": f"launch_error: {exc}"}
    finally:
        if handle != subprocess.DEVNULL:
            handle.close()

    # Update PID file
    pid_path_str = lane_info.get("pid_path", "")
    if pid_path_str:
        pid_path = Path(pid_path_str) if Path(pid_path_str).is_absolute() else repo_root / pid_path_str
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")

    return {"restarted": True, "new_pid": process.pid}


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
    ) -> None:
        self.manifest_path = manifest_path
        self.repo_root = repo_root
        self.check_interval = check_interval
        self.lane_timeout = lane_timeout
        self.max_consecutive_restarts = max_consecutive_restarts
        self.log_aggregation_dir = log_aggregation_dir or (
            manifest_path.parent / "logs" / "aggregated"
        )
        self._consecutive_restart_counts: dict[str, int] = {}
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
            "status": "shutdown",
            "total_checks": total_checks,
            "total_restarts": total_restarts,
        }

    def _check_cycle(self) -> dict[str, Any]:
        """Run one health-check cycle across all lanes."""
        manifest = read_lane_manifest(self.manifest_path)
        if not manifest:
            return {"error": "manifest_empty", "restarts": 0}

        lanes = manifest.get("lanes", [])
        started = manifest.get("started", [])

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

            report = {
                "bundle_key": bundle_key,
                "pid_check": pid_check,
                "heartbeat_check": heartbeat_check,
                "action": "none",
            }

            needs_restart = not pid_check["alive"] or heartbeat_check.get("stale", False)

            if needs_restart:
                # Check consecutive restart count for backoff
                count = self._consecutive_restart_counts.get(bundle_key, 0)
                if count >= self.max_consecutive_restarts:
                    # Exponential backoff: skip this cycle
                    backoff_cycles = 2 ** min(count - self.max_consecutive_restarts, 5)
                    report["action"] = "backoff"
                    report["backoff_cycles"] = backoff_cycles
                    # Decrement so we'll try again eventually
                    self._consecutive_restart_counts[bundle_key] = count + 1
                else:
                    restart_info = lane_started or lane
                    restart_result = restart_lane(restart_info, repo_root=self.repo_root)
                    report["action"] = "restarted" if restart_result.get("restarted") else "restart_failed"
                    report["restart_result"] = restart_result
                    if restart_result.get("restarted"):
                        restarts += 1
                        self._consecutive_restart_counts[bundle_key] = count + 1
                    else:
                        self._consecutive_restart_counts[bundle_key] = count + 1
            else:
                # Healthy - reset consecutive restart counter
                self._consecutive_restart_counts[bundle_key] = 0

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

        return {
            "timestamp": utc_now(),
            "lane_count": len(lanes),
            "restarts": restarts,
            "reports": reports,
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
