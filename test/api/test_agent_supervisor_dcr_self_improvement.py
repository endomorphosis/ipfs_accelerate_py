"""Focused DCR-084 bounded pending self-improvement tests."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AuthorityStage,
    DeterministicRepairDisposition,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.self_improvement import (
    BoundedParameterPolicy,
    ImprovementDisposition,
    ImprovementMetrics,
    ImprovementProposal,
    ShadowReceipt,
    evaluate_improvement_proposal,
)
from ipfs_accelerate_py.agent_supervisor.objectives.deterministic_repair_selection import (
    RepairSelectionDependencyBundle,
    RepairSelectionEvidence,
    SelectionState,
    select_and_refill_repairs,
)
from ipfs_accelerate_py.agent_supervisor.objectives.repair_authority_projection import (
    RepairAuthorityProjection,
    RepairAuthorityStatus,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CandidatePortfolio,
    CandidatePortfolioDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_failure_memory import (
    FailureAttempt,
    FailureClass,
    decide_replan,
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
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_composition import (
    DeterministicRepairCompositionDisposition,
    DeterministicRepairCompositionResult,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_recovery import (
    RecoveryDecision,
    RecoveryDisposition,
)


def _roots():
    return RepairAuthorityRoots("repo", "forest", "tree", "policy", "plan", "packet")


def _descriptor():
    return OperatorDescriptor.from_mapping(
        {
            "operator_id": "op",
            "kind": "replace_exact_bytes",
            "owner_root": "root",
            "write_scope": ["a.py"],
            "before_predicates": ["b"],
            "after_predicates": ["a"],
            "applicability_proofs": ["p"],
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["python", "a.py"]],
        }
    )


def _evidence():
    roots = _roots()
    d = _descriptor()
    r = OperatorRegistry((d,), reviewed_manifest={d.operator_id: d.descriptor_id})
    n = RepairPlanNode(
        "n",
        RepairPlanNodeKind.REPAIR,
        "root",
        "a.py",
        "s",
        "before",
        "after",
        d,
        r.report()["registry_cid"],
        "proof",
        "logic",
        "impact",
        "non",
        (("python", "a.py"),),
        "inverse",
        "rollback",
    )
    plan = ProofCarryingRepairPlan(
        DoctorTransformBinding("d51", "d52", "doctor"), roots, r, r.report()["registry_cid"], (n,)
    )
    result = RepairPlanDagResult(
        RepairPlanDagDisposition.INTEGRATION_PENDING, ("pending",), plan.content_id, (n.content_id,)
    )
    portfolio = CandidatePortfolio(
        CandidatePortfolioDisposition.INTEGRATION_PENDING, ("pending",), "portfolio", ("candidate",)
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
    policy = RepairResourcePolicy(("lane",), {"cpu": 1}, 4, 1, "epoch")
    schedule = schedule_repair_resources(
        plan,
        result,
        decision,
        portfolio=portfolio,
        attempt=attempt,
        history=(),
        current_roots=roots,
        policy=policy,
    )
    bundle = RepairSelectionDependencyBundle(
        roots, plan, result, portfolio, attempt, (), decision, policy, schedule
    )
    observed = RepairEvidenceEnvelope(
        "task",
        DeterministicRepairDisposition.REFUTED_REPAIRABLE,
        AuthorityStage.OBSERVED,
        roots,
        "obs",
    )
    envelope = RepairEvidenceEnvelope(
        "task",
        DeterministicRepairDisposition.REFUTED_REPAIRABLE,
        AuthorityStage.DERIVED,
        roots,
        "obs",
        AuthorityStage.OBSERVED,
        observed.content_id,
        "der",
    )
    return RepairSelectionEvidence(
        "task",
        SelectionState.DERIVED,
        envelope,
        bundle,
        "root",
        1,
        "available",
        DeterministicRepairCompositionResult(
            DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
            ("pending",),
            ("d50",),
            "task",
        ),
    ), roots


def _proposal():
    evidence, roots = _evidence()
    selection = select_and_refill_repairs((evidence,))
    baseline, candidate = ImprovementMetrics(5, 5, 5, 5), ImprovementMetrics(6, 5, 4, 5)
    base, cand = content_identity(baseline.__dict__), content_identity(candidate.__dict__)
    shadow = ShadowReceipt(
        base, cand, content_identity({"baseline": base, "candidate": cand, "passed": True}), True
    )
    d80 = evidence.transition.receipt_cid
    authority = RepairAuthorityProjection(
        "task",
        "goal",
        roots,
        AuthorityStage.DERIVED,
        RepairAuthorityStatus.BLOCKED,
        ("pending",),
        "base",
        "ready",
        "d10",
        d80,
        (),
        (),
    )
    recovery = RecoveryDecision(
        RecoveryDisposition.INTEGRATION_PENDING, ("pending",), "", "task", roots.content_id, d80
    )
    return ImprovementProposal(
        selection,
        (evidence,),
        recovery,
        authority,
        roots.content_id,
        baseline,
        candidate,
        {"safety": 5, "correctness": 5},
        {"max_candidates": 4},
        BoundedParameterPolicy({"max_candidates": (1, 8, 3)}),
        shadow,
        "inverse",
        "manual",
    )


def test_bounded_proposal_pending_and_zero_effects():
    result = evaluate_improvement_proposal(_proposal())
    assert result.disposition is ImprovementDisposition.PROPOSAL_PENDING
    assert result.execution_authorized is result.completion_authorized is False


def test_noop_and_security_errors_reject_before_noop():
    proposal = _proposal()
    base = content_identity(proposal.baseline.__dict__)
    stable_shadow = ShadowReceipt(
        base, base, content_identity({"baseline": base, "candidate": base, "passed": True}), True
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, candidate=proposal.baseline, shadow=stable_shadow)
        ).disposition
        is ImprovementDisposition.NO_OP
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, parameter_changes={"policy_root": 1}, candidate=proposal.baseline)
        ).disposition
        is ImprovementDisposition.REJECTED
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, parameter_changes={"unknown": 1})
        ).disposition
        is ImprovementDisposition.REJECTED
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, parameter_changes={"max_candidates": 9})
        ).disposition
        is ImprovementDisposition.REJECTED
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, safety_floors={"safety": 6, "correctness": 5})
        ).disposition
        is ImprovementDisposition.REJECTED
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, recovery=replace(proposal.recovery, task_id="forged-task"))
        ).disposition
        is ImprovementDisposition.REJECTED
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, recovery=replace(proposal.recovery, dcr080_receipt_cid="cid:foreign"))
        ).disposition
        is ImprovementDisposition.REJECTED
    )
    assert (
        evaluate_improvement_proposal(
            replace(proposal, authority=replace(proposal.authority, task_id="forged-task"))
        ).disposition
        is ImprovementDisposition.REJECTED
    )
