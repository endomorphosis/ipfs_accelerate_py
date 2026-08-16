"""Focused DCR-064 pure scheduler tests."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_failure_memory import (
    FailureAttempt,
    FailureClass,
    decide_replan,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CandidatePortfolio,
    CandidatePortfolioDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    DoctorTransformBinding,
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
    RepairPlanNode,
    RepairPlanNodeKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_resource_scheduler import (
    RepairResourcePolicy,
    schedule_repair_resources,
)


def _inputs():
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "op",
            "kind": "replace_exact_bytes",
            "owner_root": "root",
            "write_scope": ["a.py", "b.py"],
            "before_predicates": ["before"],
            "after_predicates": ["after"],
            "applicability_proofs": ["proof"],
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["python", "-m", "py_compile", "a.py"]],
        }
    )
    registry = OperatorRegistry((descriptor,), reviewed_manifest={"op": descriptor.descriptor_id})
    roots = RepairAuthorityRoots("repo", "forest", "tree", "policy", "rpr-plan", "rpr-packet")
    common = (
        RepairPlanNodeKind.REPAIR,
        "root",
        "a.py",
        "span",
        "digest",
        "after",
        descriptor,
        registry.report()["registry_cid"],
        "proof",
        "logic",
        "impact",
        "noninterference",
        (("python", "-m", "py_compile", "a.py"),),
        "inverse",
        "rollback",
    )
    first = RepairPlanNode("a", *common, resource_bounds=(("cpu", 1),))
    second = replace(first, node_id="b", dependencies=("a",))
    plan = ProofCarryingRepairPlan(
        DoctorTransformBinding("dcr051", "dcr052", "doctor"),
        roots,
        registry,
        registry.report()["registry_cid"],
        (first, second),
    )
    result = RepairPlanDagResult(RepairPlanDagDisposition.INTEGRATION_PENDING, (), plan.content_id)
    portfolio = CandidatePortfolio(
        CandidatePortfolioDisposition.INTEGRATION_PENDING, (), "portfolio", ("candidate",)
    )
    attempt = FailureAttempt(
        "portfolio",
        "candidate",
        plan.content_id,
        roots.content_id,
        FailureClass.VALIDATION,
        ("evidence",),
        (1,),
    )
    decision = decide_replan(portfolio, result, roots, attempt)
    return plan, result, decision, roots, portfolio, attempt


def test_schedule_is_deterministic_serializes_overlap_and_distinguishes_fence() -> None:
    plan, result, decision, roots, portfolio, attempt = _inputs()
    policy = RepairResourcePolicy(("lane-a", "lane-b"), {"cpu": 1}, 8, 1, "epoch")
    one = schedule_repair_resources(
        plan,
        result,
        decision,
        portfolio=portfolio,
        attempt=attempt,
        history=(),
        current_roots=roots,
        policy=policy,
    )
    two = schedule_repair_resources(
        plan,
        result,
        decision,
        portfolio=portfolio,
        attempt=attempt,
        history=(),
        current_roots=roots,
        policy=policy,
    )
    assert one == two
    assert one.nodes[1].lane == one.nodes[0].lane
    assert "a" in one.nodes[1].dependencies
    assert one.nodes[0].lease_cid != one.nodes[0].fence_cid
    assert one.execution_authorized is False


def test_capacity_and_stale_plan_abstain_or_reject() -> None:
    plan, result, decision, roots, portfolio, attempt = _inputs()
    no_cpu = RepairResourcePolicy(("lane",), {"other": 1}, 8, 1, "epoch")
    assert (
        schedule_repair_resources(
            plan,
            result,
            decision,
            portfolio=portfolio,
            attempt=attempt,
            history=(),
            current_roots=roots,
            policy=no_cpu,
        ).disposition
        == "abstained"
    )
    assert (
        schedule_repair_resources(
            plan,
            replace(result, plan_cid="old"),
            decision,
            portfolio=portfolio,
            attempt=attempt,
            history=(),
            current_roots=roots,
            policy=RepairResourcePolicy(("lane",), {"cpu": 1}, 8, 1, "epoch"),
        ).disposition
        == "rejected"
    )
