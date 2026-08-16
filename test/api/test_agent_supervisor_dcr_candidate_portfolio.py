"""Focused DCR-062 deterministic candidate portfolio tests."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CandidateFacts,
    CandidatePortfolioDisposition,
    CandidatePortfolioRequest,
    build_deterministic_candidate_portfolio,
)
from ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory import (
    DcrPlannerCompositionDisposition,
    DcrPlannerCompositionResult,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    DoctorTransformBinding,
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
    RepairPlanNode,
    RepairPlanNodeKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    IrLogicRequiredGateDisposition,
    IrLogicRequiredGateResult,
)


def _request(*, tied: bool = False) -> CandidatePortfolioRequest:
    descriptor = OperatorDescriptor.from_mapping({
        "operator_id": "candidate.exact", "kind": "replace_exact_bytes",
        "owner_root": "root", "write_scope": ["module.py"],
        "before_predicates": ["before"], "after_predicates": ["after"],
        "applicability_proofs": ["proof"],
        "input_schema": {
            "type": "object", "required": ["source_digest"],
            "properties": {"source_digest": "sha256"}, "additional_properties": False,
        },
        "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
        "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
        "validation_commands": [["python", "-m", "py_compile", "module.py"]],
    })
    registry = OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )
    roots = RepairAuthorityRoots("repo", "forest", "tree", "policy", "rpr-plan", "rpr-packet")
    node = RepairPlanNode(
        "node", RepairPlanNodeKind.REPAIR, "root", "module.py", "span", "sha256-before",
        "after", descriptor, registry.report()["registry_cid"], "proof", "logic", "impact",
        "noninterference", (("python", "-m", "py_compile", "module.py"),), "inverse", "rollback",
    )
    plan = ProofCarryingRepairPlan(
        DoctorTransformBinding("dcr051", "dcr052", "doctor"), roots, registry,
        registry.report()["registry_cid"], (node,),
    )
    gate = IrLogicRequiredGateResult(
        IrLogicRequiredGateDisposition.PASSING, (), {"dcr030": "x", "dcr034": "cache"}, ("receipt",)
    )
    facts = [CandidateFacts("candidate-a", ("node",), 2, 3, 1, "cache", roots.content_id)]
    facts.append(
        CandidateFacts(
            "candidate-b", ("node",), 2 if tied else 4, 3 if tied else 4,
            1 if tied else 2, "cache", roots.content_id,
        )
    )
    return CandidatePortfolioRequest(
        plan.transform, plan,
        RepairPlanDagResult(RepairPlanDagDisposition.INTEGRATION_PENDING, (), plan.content_id),
        gate, roots, "cache",
        DcrPlannerCompositionResult(
            DcrPlannerCompositionDisposition.INTEGRATION_PENDING, ("dcr060-pending",)
        ),
        tuple(facts),
    )


def test_dcr062_orders_closed_numeric_facts_but_stays_pending() -> None:
    portfolio = build_deterministic_candidate_portfolio(_request())
    assert portfolio.disposition is CandidatePortfolioDisposition.INTEGRATION_PENDING
    assert len(portfolio.candidate_cids) == 2
    assert portfolio.execution_authorized is False
    assert portfolio.completion_authorized is False


def test_dcr062_abstains_on_equal_numeric_top_score() -> None:
    portfolio = build_deterministic_candidate_portfolio(_request(tied=True))
    assert portfolio.disposition is CandidatePortfolioDisposition.ABSTAINED
    assert portfolio.reason_codes == ("top_candidate_score_ambiguous",)


def test_dcr062_rejects_raw_or_stale_candidate_facts() -> None:
    request = _request()
    raw = CandidatePortfolioRequest(**{**request.__dict__, "candidates": ({"bad": True},)})
    portfolio = build_deterministic_candidate_portfolio(raw)
    assert portfolio.disposition is CandidatePortfolioDisposition.REJECTED
    assert "closed_candidate_facts_required" in portfolio.reason_codes
