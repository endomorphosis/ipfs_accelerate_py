"""Keep a native authority available after implementation lanes drain.

Task drain ends implementation dispatch. It does not retire the state owner:
the board's native acceptance adapter must still verify goals and roots.
This lifecycle helper never reads a replica or changes acceptance state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def retain_owner_for_closeout(
    *,
    observe: Callable[[], Mapping[str, Any]],
    wait: Callable[[float], bool],
    stopped: Callable[[], bool],
    check_owner: Callable[[], None],
    output: Callable[[str], None],
    interval_seconds: float = 10.0,
) -> str:
    """Wait for native acceptance or an explicit stop, preserving the owner.

    ``observe`` is the native board's trusted acceptance reader, not worker
    output or a configurable shell success code. Its completion_authority
    flag must mean the full board contract was verified against current
    receipts and source heads. An ordinary authenticated progress snapshot
    explicitly lacks that authority and therefore cannot end this phase.

    Owner faults propagate to the existing qualified restart path. Stop
    markers and signals end the phase without claiming completion.
    """
    if interval_seconds <= 0:
        raise ValueError("closeout interval must be positive")
    output("implementation lanes drained; retaining native owner for acceptance")
    while not stopped():
        check_owner()
        observation = observe()
        if (
            observation.get("completion_authority") is True
            and observation.get("complete") is True
        ):
            # Recheck stop and owner failure after a potentially slow read.
            if stopped():
                return "stopped"
            check_owner()
            return "accepted"
        if wait(interval_seconds):
            return "stopped"
    return "stopped"
