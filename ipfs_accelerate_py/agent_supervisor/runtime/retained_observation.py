"""Keep observation lifetime separate from the state being observed."""

from __future__ import annotations

import math
from collections.abc import Callable
from threading import Event


def run_retained_observation(
    observe_once: Callable[[], None],
    *,
    stop: Event,
    failed: Event,
    interval: float,
    on_error: Callable[[Exception], None],
) -> None:
    """Sample in the caller's retained authority until shutdown or failure.

    A normal callback return ends only that sample, including when its task
    portfolio is terminal. The callback retains its own progress and failure
    budgets across samples. It must set ``failed`` for an admitted failure;
    exceptions are reported once and never replayed. Neither callback results
    nor thread lifetime grant completion, source transition, or write authority.

    This runs synchronously: callers may put it in their existing owner thread.
    It creates no process, credential, connection, or replacement launch owner.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("observation interval must be finite and positive")
    while not failed.is_set() and not stop.wait(interval):
        # Failure may have been published by another owner thread while waiting.
        if failed.is_set() or stop.is_set():
            return
        try:
            observe_once()
        except Exception as exc:
            on_error(exc)
            return
