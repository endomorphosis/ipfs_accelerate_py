"""Keep Python process-control exceptions raised inside DuckDB native calls."""
from __future__ import annotations


def reraise_process_interrupt(error: BaseException) -> None:
    """Restore an actual chained stop, never infer one from query error text.

    DuckDB can translate a Python signal handler's SystemExit/KeyboardInterrupt
    into RuntimeError. Ordinary exception recovery must not swallow that stop.
    Follow the explicit cause (or unsuppressed context), with a cycle/bound guard.
    This supplies no task disposition, replay, or owner-replacement authority.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    for _ in range(32):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        if isinstance(current, (SystemExit, KeyboardInterrupt)):
            raise current
        current = (
            current.__cause__
            if current.__cause__ is not None
            else None if current.__suppress_context__ else current.__context__
        )
