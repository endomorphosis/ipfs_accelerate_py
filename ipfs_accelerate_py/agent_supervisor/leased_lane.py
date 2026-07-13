"""Process guard that runs a daemon lane only while its lease is accepted."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from .lease_coordination import LeaseCoordinator, LeaseError, LeaseGrant

logger = logging.getLogger(__name__)


def run_leased_lane(
    *,
    coordination_path: Path,
    grant: LeaseGrant,
    command: Sequence[str],
    lease_ms: int,
    heartbeat_interval: float,
    capacity_millionths: int = 1_000_000,
) -> int:
    """Run ``command``, renewing it and fencing it immediately on lease loss."""

    if not command:
        raise ValueError("leased lane command is required")
    started_at_ms = int(time.time() * 1000)
    with LeaseCoordinator(coordination_path) as coordinator:
        grant = coordinator.validate(grant)
        coordinator.heartbeat(grant, capacity_millionths=capacity_millionths)
        process = subprocess.Popen(list(command))
        stopping = False

        def stop_child(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True
            if process.poll() is None:
                process.terminate()

        old_term = signal.signal(signal.SIGTERM, stop_child)
        old_int = signal.signal(signal.SIGINT, stop_child)
        try:
            while process.poll() is None:
                time.sleep(max(0.05, heartbeat_interval))
                try:
                    now = int(time.time() * 1000)
                    if grant.lease_expires_at_ms - now <= max(1_000, lease_ms // 2):
                        grant = coordinator.renew(grant, requested_lease_ms=lease_ms, now_ms=now)
                    coordinator.heartbeat(grant, capacity_millionths=capacity_millionths, now_ms=now)
                except LeaseError as exc:
                    logger.error("Fencing daemon lane after lease loss: %s", exc)
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    return 75
            return_code = int(process.returncode or 0)
            try:
                if return_code == 0 and not stopping:
                    coordinator.receipt(
                        grant,
                        status="succeeded",
                        output={"exit_code": return_code, "command": list(command)},
                        started_at_ms=started_at_ms,
                    )
                else:
                    coordinator.receipt(
                        grant,
                        status="cancelled" if stopping else "failed",
                        failure_class="retryable",
                        started_at_ms=started_at_ms,
                    )
            except LeaseError:
                # The takeover's fencing token is authoritative; the old lane
                # must not manufacture a receipt after losing ownership.
                logger.warning("Discarded terminal result from fenced lane %s", grant.task_cid)
                return 75
            return return_code
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a command under an accepted Profile G lease")
    parser.add_argument("--coordination-path", type=Path, required=True)
    parser.add_argument("--grant-json", required=True)
    parser.add_argument("--lease-ms", type=int, default=60_000)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--capacity-millionths", type=int, default=1_000_000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    grant = LeaseGrant(**json.loads(args.grant_json))
    return run_leased_lane(
        coordination_path=args.coordination_path,
        grant=grant,
        command=command,
        lease_ms=args.lease_ms,
        heartbeat_interval=args.heartbeat_interval,
        capacity_millionths=args.capacity_millionths,
    )


if __name__ == "__main__":
    raise SystemExit(main())
