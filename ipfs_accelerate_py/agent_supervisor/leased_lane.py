"""Run one daemon command while its scheduler lease remains accepted.

This module is the execution fence between the dynamic bundle scheduler and a
lane subprocess.  The scheduler is free to reuse a lane slot after this guard
exits because every exit path either writes a terminal receipt, releases the
lease after a bookkeeping failure, or proves that this worker has already been
fenced by a newer lease.

``run_leased_lane`` retains the original integer-returning API and command-line
interface.  ``run_leased_lane_result`` exposes the same lifecycle as a small,
immutable result for in-process schedulers and tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .lease_coordination import LeaseCoordinator, LeaseError, LeaseGrant
from .todo_daemon.core import terminate_pid_tree

logger = logging.getLogger(__name__)

FENCED_EXIT_CODE = 75
START_FAILED_EXIT_CODE = 70

LaneDisposition = Literal[
    "completed",
    "blocked",
    "failed",
    "cancelled",
    "fenced",
    "start_failed",
]


@dataclass(frozen=True)
class LeasedLaneResult:
    """Scheduler-facing terminal state for one leased command execution."""

    task_cid: str
    claim_cid: str
    claimant_did: str
    fencing_token: int
    disposition: LaneDisposition
    exit_code: int
    child_exit_code: int | None
    started_at_ms: int
    finished_at_ms: int
    receipt_cid: str | None = None
    lease_released: bool = False
    error: str = ""

    @property
    def reusable(self) -> bool:
        """Return whether the scheduler may immediately reuse the lane slot."""

        # A fenced result is reusable as well: the old process has been
        # terminated and authority belongs to another accepted fencing token.
        return self.disposition in {
            "completed",
            "blocked",
            "failed",
            "cancelled",
            "fenced",
            "start_failed",
        }

    @property
    def successful(self) -> bool:
        return self.disposition == "completed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable projection suitable for a live manifest."""

        return {**asdict(self), "reusable": self.reusable, "successful": self.successful}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _terminate_child(process: subprocess.Popen[Any], *, timeout: float = 5.0) -> None:
    """Terminate a child and do not return while it can still execute work."""

    if process.poll() is not None:
        return
    terminate_pid_tree(process.pid, grace_seconds=timeout)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _receipt_cid(receipt: Mapping[str, Any] | None) -> str | None:
    if not receipt:
        return None
    value = receipt.get("receipt_cid")
    return str(value) if value else None


def _release_after_bookkeeping_failure(
    coordinator: LeaseCoordinator,
    grant: LeaseGrant,
) -> bool:
    """Best-effort release when a non-fencing receipt error occurs."""

    try:
        coordinator.release(grant)
        return True
    except LeaseError:
        # A takeover or expiry has already removed this worker's authority.
        return False
    except Exception:
        logger.exception("Could not release lease after terminal receipt failure")
        return False


def run_leased_lane_result(
    *,
    coordination_path: Path,
    grant: LeaseGrant,
    command: Sequence[str],
    lease_ms: int,
    heartbeat_interval: float,
    capacity_millionths: int = 1_000_000,
) -> LeasedLaneResult:
    """Run ``command`` and return its fenced, terminal lease disposition.

    Successful children produce a successful receipt (which completes the
    task).  Non-zero children produce a retryable failed receipt, signals
    produce a cancelled receipt, and both cases release the task for another
    worker.  Exit code 75 is treated as a blocked/retryable child convention.
    If renewal or heartbeat proves the grant stale, the child is synchronously
    stopped and no stale receipt is manufactured.
    """

    if not command:
        raise ValueError("leased lane command is required")
    if float(heartbeat_interval) <= 0:
        raise ValueError("heartbeat_interval must be positive")
    if int(lease_ms) <= 0:
        raise ValueError("lease_ms must be positive")

    started_at_ms = _now_ms()
    with LeaseCoordinator(coordination_path) as coordinator:
        try:
            grant = coordinator.validate(grant)
            coordinator.heartbeat(grant, capacity_millionths=capacity_millionths)
        except LeaseError as exc:
            logger.warning("Refused to start fenced lane %s: %s", grant.task_cid, exc)
            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition="fenced",
                exit_code=FENCED_EXIT_CODE,
                child_exit_code=None,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                error=str(exc),
            )
        except Exception as exc:
            # Invalid capacity and local coordination failures must not strand
            # an accepted lease when no child was ever started.
            logger.exception("Could not initialize leased lane %s", grant.task_cid)
            released = _release_after_bookkeeping_failure(coordinator, grant)
            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition="start_failed",
                exit_code=START_FAILED_EXIT_CODE,
                child_exit_code=None,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                lease_released=released,
                error=str(exc),
            )

        try:
            process = subprocess.Popen(list(command))
        except Exception as exc:
            logger.error("Could not start leased lane %s: %s", grant.task_cid, exc)
            try:
                receipt = coordinator.receipt(
                    grant,
                    status="failed",
                    failure_class="retryable",
                    started_at_ms=started_at_ms,
                )
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="start_failed",
                    exit_code=START_FAILED_EXIT_CODE,
                    child_exit_code=None,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    receipt_cid=_receipt_cid(receipt),
                    lease_released=True,
                    error=str(exc),
                )
            except LeaseError as lease_exc:
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="fenced",
                    exit_code=FENCED_EXIT_CODE,
                    child_exit_code=None,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    error=str(lease_exc),
                )
            except Exception as receipt_exc:
                logger.exception("Could not record leased lane start failure")
                released = _release_after_bookkeeping_failure(coordinator, grant)
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="start_failed",
                    exit_code=START_FAILED_EXIT_CODE,
                    child_exit_code=None,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    lease_released=released,
                    error=f"{exc}; terminal bookkeeping failed: {receipt_exc}",
                )

        stopping_signal: int | None = None
        stop_event = threading.Event()

        def stop_child(signum: int, _frame: object) -> None:
            nonlocal stopping_signal
            stopping_signal = signum
            stop_event.set()
            # The polling loop must capture and terminate the whole descendant
            # tree before the immediate child can exit and orphan its daemon.

        handlers_installed = threading.current_thread() is threading.main_thread()
        old_term: Any = None
        old_int: Any = None
        if handlers_installed:
            old_term = signal.signal(signal.SIGTERM, stop_child)
            old_int = signal.signal(signal.SIGINT, stop_child)

        try:
            while process.poll() is None:
                now = _now_ms()
                renew_window_ms = max(1_000, int(lease_ms) // 2)
                until_renewal = max(0.0, (grant.lease_expires_at_ms - now - renew_window_ms) / 1000)
                # Never sleep through the renewal window, even when an
                # operator configured a heartbeat interval longer than it.
                delay = min(max(0.05, float(heartbeat_interval)), max(0.05, until_renewal))
                stop_event.wait(delay)
                if stopping_signal is not None:
                    _terminate_child(process)
                    break
                if process.poll() is not None:
                    break
                try:
                    now = _now_ms()
                    if grant.lease_expires_at_ms - now <= renew_window_ms:
                        grant = coordinator.renew(grant, requested_lease_ms=lease_ms, now_ms=now)
                    coordinator.heartbeat(
                        grant,
                        capacity_millionths=capacity_millionths,
                        now_ms=now,
                    )
                except LeaseError as exc:
                    logger.error("Fencing daemon lane after lease loss: %s", exc)
                    _terminate_child(process)
                    return LeasedLaneResult(
                        task_cid=grant.task_cid,
                        claim_cid=grant.claim_cid,
                        claimant_did=grant.claimant_did,
                        fencing_token=grant.fencing_token,
                        disposition="fenced",
                        exit_code=FENCED_EXIT_CODE,
                        child_exit_code=process.returncode,
                        started_at_ms=started_at_ms,
                        finished_at_ms=_now_ms(),
                        error=str(exc),
                    )
                except Exception as exc:
                    # Losing access to the coordination store is equivalent
                    # to losing proof of authority. Stop execution first; a
                    # best-effort release then makes recovery immediate when
                    # the store failure was transient.
                    logger.exception("Fencing daemon lane after coordination failure")
                    _terminate_child(process)
                    released = _release_after_bookkeeping_failure(coordinator, grant)
                    return LeasedLaneResult(
                        task_cid=grant.task_cid,
                        claim_cid=grant.claim_cid,
                        claimant_did=grant.claimant_did,
                        fencing_token=grant.fencing_token,
                        disposition="failed",
                        exit_code=START_FAILED_EXIT_CODE,
                        child_exit_code=process.returncode,
                        started_at_ms=started_at_ms,
                        finished_at_ms=_now_ms(),
                        lease_released=released,
                        error=f"coordination failure: {exc}",
                    )

            # A signal handler asks politely first; enforce the process fence
            # here in case the child ignored or delayed termination.
            if stopping_signal is not None:
                _terminate_child(process)
            child_exit_code = int(process.returncode or 0)
            receipt_status = (
                "cancelled"
                if stopping_signal is not None
                else "succeeded"
                if child_exit_code == 0
                else "failed"
            )
            disposition: LaneDisposition = (
                "cancelled"
                if stopping_signal is not None
                else "completed"
                if child_exit_code == 0
                else "blocked"
                if child_exit_code == FENCED_EXIT_CODE
                else "failed"
            )
            try:
                receipt = coordinator.receipt(
                    grant,
                    status=receipt_status,
                    output={"exit_code": child_exit_code, "command": list(command)}
                    if receipt_status == "succeeded"
                    else None,
                    failure_class="none" if receipt_status == "succeeded" else "retryable",
                    started_at_ms=started_at_ms,
                )
            except LeaseError as exc:
                # The takeover's fencing token is authoritative; the old lane
                # must not manufacture a receipt after losing ownership.
                logger.warning("Discarded terminal result from fenced lane %s", grant.task_cid)
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="fenced",
                    exit_code=FENCED_EXIT_CODE,
                    child_exit_code=child_exit_code,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception("Could not record terminal result for lane %s", grant.task_cid)
                released = _release_after_bookkeeping_failure(coordinator, grant)
                return LeasedLaneResult(
                    task_cid=grant.task_cid,
                    claim_cid=grant.claim_cid,
                    claimant_did=grant.claimant_did,
                    fencing_token=grant.fencing_token,
                    disposition="failed",
                    exit_code=START_FAILED_EXIT_CODE,
                    child_exit_code=child_exit_code,
                    started_at_ms=started_at_ms,
                    finished_at_ms=_now_ms(),
                    lease_released=released,
                    error=f"terminal bookkeeping failed: {exc}",
                )

            return LeasedLaneResult(
                task_cid=grant.task_cid,
                claim_cid=grant.claim_cid,
                claimant_did=grant.claimant_did,
                fencing_token=grant.fencing_token,
                disposition=disposition,
                exit_code=child_exit_code,
                child_exit_code=child_exit_code,
                started_at_ms=started_at_ms,
                finished_at_ms=_now_ms(),
                receipt_cid=_receipt_cid(receipt),
                lease_released=True,
            )
        finally:
            if handlers_installed:
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)


def run_leased_lane(
    *,
    coordination_path: Path,
    grant: LeaseGrant,
    command: Sequence[str],
    lease_ms: int,
    heartbeat_interval: float,
    capacity_millionths: int = 1_000_000,
) -> int:
    """Compatibility wrapper returning the guarded command's lane exit code."""

    return run_leased_lane_result(
        coordination_path=coordination_path,
        grant=grant,
        command=command,
        lease_ms=lease_ms,
        heartbeat_interval=heartbeat_interval,
        capacity_millionths=capacity_millionths,
    ).exit_code


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


__all__ = [
    "FENCED_EXIT_CODE",
    "START_FAILED_EXIT_CODE",
    "LeasedLaneResult",
    "build_parser",
    "main",
    "run_leased_lane",
    "run_leased_lane_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
