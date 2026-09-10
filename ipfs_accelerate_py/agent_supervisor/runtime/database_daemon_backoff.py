"""Retry database daemon pressure without replacing its admitted process.

The caller owns admission, claims and cleanup. This helper neither changes the
owner transport nor interprets successful return as task completion.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

RETRYABLE_DATABASE_ERROR_TYPES = frozenset({
    "OutOfMemoryException", "MemoryError", "IOException",
})
DATABASE_RETRY_BACKOFF_SECONDS = (30.0, 60.0, 120.0, 300.0, 600.0)


def run_database_daemon_with_backoff(
    main: Callable[..., Any],
    argv: Sequence[str],
    *,
    sleep: Callable[[float], None] = time.sleep,
    backoff_seconds: Sequence[float] = DATABASE_RETRY_BACKOFF_SECONDS,
    logger: logging.Logger | None = None,
) -> int:
    """Keep transient DuckDB failures inside the existing admitted child.

    Retry with a capped delay, without consuming provider attempts or changing
    ownership. Admission failures and explicit process exits propagate. Error
    messages and local values are never logged because they may hold secrets.
    """
    log = logger or logging.getLogger(__name__)
    delays = tuple(float(value) for value in backoff_seconds) or (30.0,)
    attempt = 0
    while True:
        try:
            return int(main(list(argv)) or 0)
        except Exception as exc:
            if type(exc).__name__ not in RETRYABLE_DATABASE_ERROR_TYPES:
                frame = exc.__traceback__
                while frame is not None and frame.tb_next is not None:
                    frame = frame.tb_next
                log.error(
                    "Sealed daemon child non-retryable failure type=%s function=%s line=%s",
                    type(exc).__name__,
                    frame.tb_frame.f_code.co_name if frame is not None else "unknown",
                    frame.tb_lineno if frame is not None else 0,
                )
                raise
            delay = delays[min(attempt, len(delays) - 1)]
            attempt += 1
            log.warning(
                "Sealed daemon child retryable pressure (%s); retry in %.1fs",
                type(exc).__name__, delay,
            )
            sleep(delay)
