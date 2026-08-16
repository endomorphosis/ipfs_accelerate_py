"""DCR-064 pure deterministic resource scheduling for repair DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from ..autonomous_repair.contracts import RepairAuthorityRoots, repair_evidence_cid
from .deterministic_candidate_portfolio import CandidatePortfolio
from .deterministic_failure_memory import (
    FailureAttempt,
    FailureMemoryReceipt,
    ReplanMemoryDecision,
    decide_replan,
)
from .proof_carrying_repair_dag import (
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
    RepairPlanNode,
)


DCR_RESOURCE_SCHEDULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-resource-schedule@1"
)


@dataclass(frozen=True)
class RepairResourcePolicy:
    lanes: tuple[str, ...]
    resource_budgets: Mapping[str, int]
    timeout_steps: int
    retry_budget: int
    epoch_cid: str

    def __post_init__(self) -> None:
        lanes = tuple(sorted(set(self.lanes)))
        if not lanes or any(not isinstance(item, str) or not item for item in lanes):
            raise ValueError("lanes must be non-empty closed identifiers")
        object.__setattr__(self, "lanes", lanes)
        budgets = dict(self.resource_budgets)
        if not budgets or any(type(value) is not int or value <= 0 for value in budgets.values()):
            raise ValueError("resource_budgets must be positive integers")
        object.__setattr__(self, "resource_budgets", dict(sorted(budgets.items())))
        for name in ("timeout_steps", "retry_budget"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.epoch_cid, str) or not self.epoch_cid:
            raise ValueError("epoch_cid is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lanes": list(self.lanes),
            "resource_budgets": dict(self.resource_budgets),
            "timeout_steps": self.timeout_steps,
            "retry_budget": self.retry_budget,
            "epoch_cid": self.epoch_cid,
        }


@dataclass(frozen=True)
class ScheduledRepairNode:
    node_id: str
    lane: str
    ordinal: int
    lease_cid: str
    fence_cid: str
    dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "lane": self.lane,
            "ordinal": self.ordinal,
            "lease_cid": self.lease_cid,
            "fence_cid": self.fence_cid,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class RepairResourceSchedule:
    disposition: str
    reason_codes: tuple[str, ...]
    schedule_cid: str = ""
    nodes: tuple[ScheduledRepairNode, ...] = ()
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def _overlap(left: RepairPlanNode, right: RepairPlanNode) -> bool:
    paths = (left.write_path, right.write_path)
    shared_resources = set(dict(left.resource_bounds)).intersection(dict(right.resource_bounds))
    return (
        left.owner_root == right.owner_root
        or bool(shared_resources)
        or (
            paths[0] == paths[1]
            or paths[0].startswith(paths[1] + "/")
            or paths[1].startswith(paths[0] + "/")
        )
    )


def schedule_repair_resources(
    plan: Any,
    result: Any,
    decision: Any,
    *,
    portfolio: Any,
    attempt: Any,
    history: Sequence[FailureMemoryReceipt],
    current_roots: Any,
    policy: Any,
) -> RepairResourceSchedule:
    """Create an immutable lane order without locking, executing, or waiting."""
    if not isinstance(plan, ProofCarryingRepairPlan) or not isinstance(result, RepairPlanDagResult):
        return RepairResourceSchedule("rejected", ("typed_dcr061_plan_result_required",))
    if (
        not isinstance(decision, ReplanMemoryDecision)
        or not isinstance(portfolio, CandidatePortfolio)
        or not isinstance(attempt, FailureAttempt)
        or not isinstance(current_roots, RepairAuthorityRoots)
        or not isinstance(policy, RepairResourcePolicy)
    ):
        return RepairResourceSchedule("rejected", ("typed_dcr063_roots_policy_required",))
    if plan.authority_roots != current_roots or result.plan_cid != plan.content_id:
        return RepairResourceSchedule("rejected", ("stale_dcr061_roots_or_plan",))
    previous = ""
    for receipt in history:
        if (
            not isinstance(receipt, FailureMemoryReceipt)
            or receipt.attempt.previous_receipt_cid != previous
        ):
            return RepairResourceSchedule("rejected", ("forged_failure_history",))
        previous = receipt.receipt_cid
    expected = decide_replan(portfolio, result, current_roots, attempt, history=history)
    if decision != expected or decision.receipt_cid != attempt.content_id:
        return RepairResourceSchedule("rejected", ("forged_or_unbound_replan_decision",))
    if result.disposition is RepairPlanDagDisposition.REJECTED:
        return RepairResourceSchedule("abstained", ("dcr061_plan_rejected",))
    if decision.disposition.value not in {"retry_pending", "no_work"}:
        return RepairResourceSchedule("abstained", ("dcr063_replan_not_schedulable",))
    if decision.disposition.value == "no_work":
        return RepairResourceSchedule("abstained", ("dcr063_no_work",))
    if len(plan.nodes) > policy.timeout_steps and policy.timeout_steps:
        return RepairResourceSchedule("abstained", ("timeout_capacity_unschedulable",))
    for node in plan.nodes:
        for resource, required in node.resource_bounds:
            if policy.resource_budgets.get(resource, 0) < required:
                return RepairResourceSchedule(
                    "abstained", ("named_resource_capacity_unschedulable",)
                )
    by_id = {node.node_id: node for node in plan.nodes}
    ordered: list[RepairPlanNode] = []
    pending = dict(by_id)
    while pending:
        ready = sorted(
            (
                node
                for node in pending.values()
                if set(node.dependencies).issubset({item.node_id for item in ordered})
            ),
            key=lambda item: item.node_id,
        )
        if not ready:
            return RepairResourceSchedule("abstained", ("dependency_cycle_or_deadlock",))
        ordered.extend(ready)
        for node in ready:
            pending.pop(node.node_id)
    scheduled: list[ScheduledRepairNode] = []
    lane_tail: dict[str, str] = {}
    for ordinal, node in enumerate(ordered):
        conflicting = [item for item in ordered[:ordinal] if _overlap(node, item)]
        lane = policy.lanes[ordinal % len(policy.lanes)]
        if conflicting:
            lane = next(
                (item.lane for item in scheduled if item.node_id == conflicting[-1].node_id), lane
            )
        lease_cid = repair_evidence_cid(
            {
                "plan": plan.content_id,
                "epoch": policy.epoch_cid,
                "node": node.node_id,
                "lane": lane,
                "ordinal": ordinal,
                "kind": "lease",
            }
        )
        fence_cid = repair_evidence_cid({"lease": lease_cid, "kind": "fence"})
        deps = set(node.dependencies)
        if lane in lane_tail:
            deps.add(lane_tail[lane])
        scheduled.append(
            ScheduledRepairNode(
                node.node_id, lane, ordinal, lease_cid, fence_cid, tuple(sorted(deps))
            )
        )
        lane_tail[lane] = node.node_id
    body = {
        "schema": DCR_RESOURCE_SCHEDULE_SCHEMA,
        "plan_cid": plan.content_id,
        "root_cid": current_roots.content_id,
        "policy": policy.to_dict(),
        "nodes": [item.to_dict() for item in scheduled],
    }
    return RepairResourceSchedule(
        "integration_pending",
        ("integration_pending_live_dcr061_dcr063_evidence",),
        repair_evidence_cid(body),
        tuple(scheduled),
    )


__all__ = [
    "DCR_RESOURCE_SCHEDULE_SCHEMA",
    "RepairResourcePolicy",
    "RepairResourceSchedule",
    "ScheduledRepairNode",
    "schedule_repair_resources",
]
