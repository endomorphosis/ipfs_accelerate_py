"""Retain a source-service owner in its existing synchronous monitor.

Each close request and backoff is bounded; total retirement time deliberately
is not. A pending receipt, interrupted close, or unknown admission outcome
cannot authorize the caller to exit and discard its in-process cleanup owner.
The existing outer process guardian or an operator must handle a permanently
stuck owner. This helper creates no guardian, thread, lease, or continuation.

These are cooperative bounds: the underlying native cleanup or scheduler I/O
can exceed the requested close timeout. The helper does not kill or reap the
registered worker and cannot establish isolated-worker or container cleanup.
"""
from __future__ import annotations

import os
import time

from .repository_source_owner_lifecycle import RepositorySourceOwnerLifecycle, SCHEMA

CLOSE_TIMEOUT_SECONDS = 1.0
INITIAL_BACKOFF_SECONDS = 0.1
MAX_BACKOFF_SECONDS = 1.0
RETENTION_SCHEMA = "repository-source-owner-retirement-retention@1"

# Narrow timing seams for authored monitor tests; no runtime policy overrides.
_sleep = time.sleep
_monotonic = time.monotonic


def _closed_receipt(receipt):
    """Require affirmative completion and a known native parent disposition."""
    if type(receipt) is not dict:
        return False
    if (receipt.get("schema") != SCHEMA or receipt.get("state") != "closed"
            or receipt.get("ready") is not False
            or receipt.get("cleanup_complete") is not True
            or receipt.get("cleanup_pending") is not False
            or receipt.get("observation_pending") is not False
            or receipt.get("admission_uncertain") is not False
            or receipt.get("retained_custody") is not False
            or "cleanup_error" not in receipt or receipt["cleanup_error"] is not None):
        return False
    acquired, released = receipt.get("native_parent_acquired"), receipt.get("native_parent_released")
    return ((acquired is True and released is True)
            or (acquired is False and released is None
                and "native_parent_released" in receipt
                and "native_parent" in receipt and receipt["native_parent"] is None))


def _backoff(seconds):
    """Finish one finite pause even if a cleanup-time signal interrupts sleep."""
    deadline = _monotonic() + seconds
    while True:
        try:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                return
            _sleep(remaining)
        except BaseException:
            # The monitor must retain custody through KeyboardInterrupt and
            # SystemExit too. Its established cleanup signal policy still owns
            # process-level termination; no replacement handler is installed.
            continue


def retire_repository_source_owner(owner):
    """Return only a validated closed receipt from this process's exact owner.

    Call from the established monitor's cleanup path after its worker/other
    service cleanup. The argument remains strongly reachable throughout all
    close passes. No new work is submitted or retried; only this existing
    owner's cleanup is polled. Wrong types and foreign-process owners are
    refused before any cleanup call. Persistent errors intentionally retain
    the calling monitor indefinitely rather than report successful cleanup.
    """
    if type(owner) is not RepositorySourceOwnerLifecycle:
        raise ValueError("exact_repository_source_owner_required")
    if type(owner._owner_pid) is not int or owner._owner_pid != os.getpid():
        raise ValueError("repository_source_owner_belongs_to_another_process")
    delay = INITIAL_BACKOFF_SECONDS
    while True:
        try:
            receipt = owner.close(timeout_seconds=CLOSE_TIMEOUT_SECONDS)
            if _closed_receipt(receipt):
                break
        except BaseException:
            # Exceptions are not evidence of release, including exceptions
            # raised while constructing or inspecting a cleanup receipt.
            pass
        _backoff(delay)
        delay = min(MAX_BACKOFF_SECONDS, delay * 2)
    # Retention is attempted only after cleanup is established, outside the
    # cleanup retry loop. Evidence failure cannot prolong released custody or
    # replace the established caller's worker/startup outcome.
    try:
        from .repository_source_owner_retirement import retain_repository_source_owner_retirement
        retained = retain_repository_source_owner_retirement(owner)
    except BaseException:
        retained = dict(schema=RETENTION_SCHEMA, status="unavailable", reason="retention_failed",
                        name=None, sha256=None, bytes=None)
    receipt["retained_lifecycle_receipt"] = retained
    return receipt
