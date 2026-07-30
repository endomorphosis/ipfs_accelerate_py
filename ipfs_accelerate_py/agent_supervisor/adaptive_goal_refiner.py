"""Objective-heap bridge for evidence-backed symbolic backlog refill.

The general adaptive refiner lives in :mod:`agent_supervisor.objectives`.
This narrow supervisor bridge couples its frozen-root rule to symbolic finding
refill: callers provide one immutable view of the authoritative objective heap,
taskboard, and replay state, and receive a proposal without either source being
mutated.

Every proposal carries ``vfs/symbolic-refill-epoch@1`` and
``vfs/refill-idempotency@1`` evidence from the underlying planner.  New goals
and tasks remain bound to the snapshot's exact objective-forest revision.
"""

from __future__ import annotations

from .objectives.adaptive_goal_refiner import AdaptiveGoalRefiner
from .symbolic_finding_refill import (
    BacklogRefinery,
    SupervisorBacklogSnapshot,
)

__all__ = [
    "AdaptiveGoalRefiner",
    "BacklogRefinery",
    "SupervisorBacklogSnapshot",
]
