"""Objective-heap bridge for evidence-backed symbolic backlog refill.

The general adaptive refiner lives in :mod:`agent_supervisor.objectives`.
This narrow supervisor bridge couples its frozen-root rule to symbolic finding
refill: callers provide one immutable view of the authoritative objective heap,
taskboard, and replay state, and receive a proposal without either source being
mutated.

Every proposal carries ``vfs/symbolic-refill-epoch@1`` and
``vfs/refill-idempotency@1`` evidence from the underlying planner.  New goals
and tasks remain bound to the snapshot's exact objective-forest revision.

Objective-heap ownership for the autonomous-refill packet
(goal_packet/autonomous_refill/ipfs_accelerate_py/767f3cfd52ba):

* VFS-G160 / VFS-080 prove ``vfs/symbolic-refill-epoch@1``
* VFS-G161 / VFS-083 prove ``vfs/refill-idempotency@1``
* VFS-G120 remains the parent refill goal; proposals never authorize execution
"""

from __future__ import annotations

from .objectives.adaptive_goal_refiner import AdaptiveGoalRefiner
from .symbolic_finding_refill import (
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_G160_ID,
    OBJECTIVE_GOAL_G161_ID,
    OBJECTIVE_PACKET_EVIDENCE_TERMS,
    OBJECTIVE_PACKET_GOAL_IDS,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_G160_ID,
    OBJECTIVE_TASK_G161_ID,
    OBJECTIVE_TASK_PACKET_ID,
    REFILL_IDEMPOTENCY_EVIDENCE,
    REFILL_IDEMPOTENCY_SCHEMA,
    SYMBOLIC_REFILL_EPOCH_EVIDENCE,
    SYMBOLIC_REFILL_EPOCH_SCHEMA,
    SYMBOLIC_REFILL_EVIDENCE_SCHEMAS,
    BacklogRefinery,
    SupervisorBacklogSnapshot,
    all_covered_evidence_terms,
    covered_evidence_terms,
    packet_evidence_terms,
    prove_autonomous_refill_packet,
    prove_refill_idempotency,
    prove_symbolic_refill_epoch,
    refill_idempotency_evidence,
    refill_idempotency_evidence_terms,
    symbolic_refill_epoch_evidence,
    symbolic_refill_epoch_evidence_terms,
    verify_refill_idempotency,
    verify_symbolic_refill_epoch,
)

# Exact-text discovery anchors for adaptive-goal-refiner import paths.
assert SYMBOLIC_REFILL_EPOCH_EVIDENCE == "vfs/symbolic-refill-epoch@1"
assert REFILL_IDEMPOTENCY_EVIDENCE == "vfs/refill-idempotency@1"
assert OBJECTIVE_GOAL_G160_ID == "VFS-G160"
assert OBJECTIVE_GOAL_G161_ID == "VFS-G161"
assert OBJECTIVE_TASK_G160_ID == "VFS-080"
assert OBJECTIVE_TASK_G161_ID == "VFS-083"
assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G120"
assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
    "vfs/symbolic-refill-epoch@1",
    "vfs/refill-idempotency@1",
)

__all__ = [
    "OBJECTIVE_DOMAIN_EVIDENCE_TERMS",
    "OBJECTIVE_GOAL_G160_ID",
    "OBJECTIVE_GOAL_G161_ID",
    "OBJECTIVE_PACKET_EVIDENCE_TERMS",
    "OBJECTIVE_PACKET_GOAL_IDS",
    "OBJECTIVE_PARENT_GOAL_ID",
    "OBJECTIVE_TASK_G160_ID",
    "OBJECTIVE_TASK_G161_ID",
    "OBJECTIVE_TASK_PACKET_ID",
    "REFILL_IDEMPOTENCY_EVIDENCE",
    "REFILL_IDEMPOTENCY_SCHEMA",
    "SYMBOLIC_REFILL_EPOCH_EVIDENCE",
    "SYMBOLIC_REFILL_EPOCH_SCHEMA",
    "SYMBOLIC_REFILL_EVIDENCE_SCHEMAS",
    "AdaptiveGoalRefiner",
    "BacklogRefinery",
    "SupervisorBacklogSnapshot",
    "all_covered_evidence_terms",
    "covered_evidence_terms",
    "packet_evidence_terms",
    "prove_autonomous_refill_packet",
    "prove_refill_idempotency",
    "prove_symbolic_refill_epoch",
    "refill_idempotency_evidence",
    "refill_idempotency_evidence_terms",
    "symbolic_refill_epoch_evidence",
    "symbolic_refill_epoch_evidence_terms",
    "verify_refill_idempotency",
    "verify_symbolic_refill_epoch",
]
