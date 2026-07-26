from __future__ import annotations

import copy
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adaptive_planner import (
    AUTHORITY_NON_COMPENSATION_ACCEPTANCE_CRITERIA,
    AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,
    AdaptivePlanCandidate,
    AdaptivePlanReceiptStore,
    AdaptivePlanSelectionReceipt,
    AdaptivePlanner,
    AdaptivePlanningRunReceipt,
    AdaptivePlanningRunStore,
    AdaptivePlannerValidationError,
    FrozenPlanningGoal,
    GateProducerKind,
    HardConstraintReceipt,
    HardPlanConstraint,
    adaptive_plan_candidate_snapshot_id,
    deterministic_hard_gate_receipts,
    plan_adaptively,
    select_adaptive_plan,
)
from ipfs_accelerate_py.agent_supervisor.formal_replanner import (
    RepairOperation,
    RepairProgress,
    RepairRuleKind,
    RepairTransition,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.plan_evaluator import (
    EvidenceAwarePlanCandidate,
    EvidenceAwarePlanPolicy,
    PlanBranch,
    PlanEvaluationDimension,
)
from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    AdaptiveCandidateProviderKind,
    CandidateGenerationBounds,
    CandidateProviderStatus,
)


def _branch(
    candidate_id: str,
    *,
    cost: float,
    objective_delta: float = 0.8,
) -> PlanBranch:
    return PlanBranch(
        branch_id=candidate_id,
        summary=f"Implement the {candidate_id} plan.",
        predicted_files=("src/planner.py", "tests/test_planner.py"),
        predicted_symbols=("AdaptivePlanner.select",),
        dependencies=("dependency:context",),
        validation_commands=("python -m pytest tests/test_planner.py -q",),
        validation_proof=("the test observes the selected candidate",),
        estimated_cost=cost,
        risk=0.1,
        expected_objective_delta=objective_delta,
        source="deterministic_baseline",
    )


def _plan(
    candidate_id: str,
    *,
    cost: float,
    authority_violations: tuple[str, ...] = (),
) -> EvidenceAwarePlanCandidate:
    return EvidenceAwarePlanCandidate(
        branch=_branch(candidate_id, cost=cost),
        covered_acceptance_criteria=("acceptance:tests-pass",),
        covered_evidence_terms=(AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,),
        assumptions=("assumption:frozen-tree",),
        validated_assumptions=("assumption:frozen-tree",),
        semantic_requirements=("semantics:typed-plan",),
        supported_semantics=("semantics:typed-plan",),
        dependencies=("dependency:context",),
        critical_path=("dependency:context",),
        unresolved_conflicts=(),
        changed_scopes=("scope:adaptive-planner",),
        authorized_scopes=("scope:adaptive-planner",),
        authority_violations=authority_violations,
        validation_feasible=True,
        proof_feasible=True,
        novelty=0.8,
        resource_classes=("cpu",),
        estimated_resource_cost=cost,
        estimated_tokens=int(cost * 100),
    )


def _goal() -> FrozenPlanningGoal:
    return FrozenPlanningGoal(
        goal_id="ASI-G097",
        goal_content_id="goal:adaptive-planning:v3",
        repository_tree_id="tree:abc123",
        policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=("acceptance:tests-pass",),
            evidence_terms=(AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,),
            trusted_assumptions=("assumption:frozen-tree",),
            supported_semantics=("semantics:typed-plan",),
            satisfied_dependencies=("dependency:context",),
            allowed_scopes=("scope:adaptive-planner",),
            available_resource_classes=("cpu",),
            max_estimated_resource_cost=100.0,
            max_estimated_tokens=100_000,
            require_validation=True,
            require_proof=True,
        ),
    )


def _producer(constraint: HardPlanConstraint) -> GateProducerKind:
    return {
        HardPlanConstraint.AUTHORITY: GateProducerKind.AUTHORIZATION_ENGINE,
        HardPlanConstraint.SCOPE: GateProducerKind.AUTHORIZATION_ENGINE,
        HardPlanConstraint.SAFETY: GateProducerKind.FORMAL_VALIDATOR,
        HardPlanConstraint.PROOF: GateProducerKind.PROOF_VERIFIER,
    }[constraint]


def _candidate(
    goal: FrozenPlanningGoal,
    candidate_id: str,
    *,
    cost: float,
    failed: HardPlanConstraint | None = None,
    binding_overrides: dict[str, str] | None = None,
    plan_overrides: dict[str, Any] | None = None,
    formal_plan_id: str = "",
    repair_transition: RepairTransition | None = None,
) -> AdaptivePlanCandidate:
    plan = _plan(candidate_id, cost=cost)
    if plan_overrides:
        plan = replace(plan, **plan_overrides)
    bindings = {
        "goal_content_id": goal.goal_content_id,
        "repository_tree_id": goal.repository_tree_id,
        "policy_digest": goal.policy_digest,
    }
    bindings.update(binding_overrides or {})
    snapshot_id = adaptive_plan_candidate_snapshot_id(
        plan,
        goal_content_id=bindings["goal_content_id"],
        repository_tree_id=bindings["repository_tree_id"],
        policy_digest=bindings["policy_digest"],
        formal_plan_id=formal_plan_id,
        repair_transition=repair_transition,
    )
    receipts = tuple(
        HardConstraintReceipt(
            constraint=constraint,
            candidate_id=candidate_id,
            candidate_snapshot_id=snapshot_id,
            goal_content_id=bindings["goal_content_id"],
            repository_tree_id=bindings["repository_tree_id"],
            policy_digest=bindings["policy_digest"],
            passed=constraint is not failed,
            producer_kind=_producer(constraint),
            producer_id=f"trusted:{constraint.value}:v1",
            evidence_ids=(f"evidence:{candidate_id}:{constraint.value}",),
            reason_codes=(
                ()
                if constraint is not failed
                else (f"{constraint.value}_policy_denied",)
            ),
        )
        for constraint in HardPlanConstraint
    )
    return AdaptivePlanCandidate(
        plan=plan,
        hard_constraint_receipts=receipts,
        formal_plan_id=formal_plan_id,
        repair_transition=repair_transition,
        **bindings,
    )


def _repair_transition(
    *,
    repaired_plan_id: str,
    counterexample_id: str = "counterexample:authority",
) -> RepairTransition:
    return RepairTransition(
        original_plan_id="formal-plan:original",
        repaired_plan_id=repaired_plan_id,
        counterexample_id=counterexample_id,
        repair=RepairOperation(
            kind=RepairRuleKind.TIGHTEN_AUTHORITY,
            target_task_id="ASI-059",
            parameters={
                "actor_ids": ["actor:planner"],
                "fencing_token": 7,
            },
            counterexample_id=counterexample_id,
        ),
        goal_ids=("ASI-G097",),
        taskboard_records=(
            {
                "task_id": "ASI-059",
                "status": "todo",
                "authority": "actor:planner",
            },
        ),
        refinement_depth=1,
        progress=RepairProgress(
            before_open_counterexamples=1,
            after_open_counterexamples=0,
            before_validation_findings=0,
            after_validation_findings=0,
            changed_records=1,
        ),
    )


def test_cheaper_authority_violating_plan_is_absolutely_rejected() -> None:
    """Prove objective evidence 173075880069453142914839090434430341799."""

    goal = _goal()
    cheap_invalid = _candidate(
        goal,
        "cheap-invalid",
        cost=0.01,
        failed=HardPlanConstraint.AUTHORITY,
    )
    valid = _candidate(goal, "valid", cost=12.0)

    receipt = select_adaptive_plan(goal, (cheap_invalid, valid))

    assert receipt.selected_candidate_id == "valid"
    assert [item.candidate_id for item in receipt.evaluation.rejected] == [
        "cheap-invalid"
    ]
    rejected = receipt.evaluation.rejected[0]
    assert PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY.value in (
        rejected.hard_gate_failures
    )
    assert receipt.proves_authority_non_compensation
    assert receipt.proved_requirement_ids == (
        "173075880069453142914839090434430341799",
    )
    evidence = receipt.authority_non_compensation_evidence
    assert evidence is not None
    assert evidence.requirement_id == AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID
    assert evidence.producer_kind == "adaptive_plan_selection"
    assert evidence.goal_content_id == goal.goal_content_id
    assert evidence.repository_tree_id == goal.repository_tree_id
    assert evidence.policy_digest == goal.policy_digest
    assert evidence.rejected_candidate_ids == ("cheap-invalid",)
    assert evidence.rejected_cost_millionths[0] < evidence.selected_cost_millionths
    assert evidence.authority_receipt_ids == (
        cheap_invalid.receipt_for(HardPlanConstraint.AUTHORITY).receipt_id,
    )
    assert {
        item.candidate_snapshot_id
        for item in receipt.hard_constraint_receipts
        if item.candidate_id == cheap_invalid.candidate_id
    } == {cheap_invalid.snapshot_id}
    assert receipt.to_dict()["planner_version"] == 2


def test_g097_completion_requires_fresh_complete_current_tree_proof() -> None:
    """ASI-059: runtime selection stays separate from goal completion."""

    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (
            _candidate(
                goal,
                "cheap-invalid",
                cost=0.1,
                failed=HardPlanConstraint.AUTHORITY,
            ),
            _candidate(goal, "safe", cost=2.0),
        ),
    )
    assert receipt.proves_authority_non_compensation

    tree_id = receipt.frozen_goal.repository_tree_id
    now = datetime(2026, 7, 24, 14, 0, tzinfo=timezone.utc)
    criteria = AUTHORITY_NON_COMPENSATION_ACCEPTANCE_CRITERIA
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-059",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": tree_id,
                "command": (
                    "python -m pytest "
                    "test/api/test_agent_supervisor_adaptive_planner.py -q"
                ),
            },
            validation_passed=True,
            repository_tree=tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-059:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(criteria, start=1)
    )
    coverage = {
        "repository_tree": tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "adaptive_planner.py"
                ),
                "validation": (
                    "test/api/test_agent_supervisor_adaptive_planner.py"
                ),
            }
            for criterion in criteria
        ],
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "exhaustive": True,
        "safe_for_completion_reasoning": True,
        "analyzer_version": "asi-059-completion-analyzer@1",
    }
    binding = {
        "tree_id": tree_id,
        "analyzer_version": "asi-059-completion-analyzer@1",
        "configuration_revision": "asi-059-completion-policy@1",
        "objective_revision": "ASI-G097@asi-059",
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-059-exhaustive-implementation",
                "evidence_channel": "implementation-validation",
                "receipt_cid": "scan:asi-059:implementation",
                "binding": binding,
                "scan_mode": "exhaustive",
                "analyzer_version": "asi-059-implementation-analyzer@1",
                "passed": True,
                "healthy": True,
                "exhaustive": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-059-exhaustive-receipt-audit",
                "evidence_channel": "receipt-replay-audit",
                "receipt_cid": "scan:asi-059:receipt-audit",
                "binding": binding,
                "scan_mode": "exhaustive",
                "analyzer_version": "asi-059-receipt-analyzer@1",
                "passed": True,
                "healthy": True,
                "exhaustive": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    # A qualifying runtime witness cannot certify its own objective.
    no_completion_proof = receipt.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        tasks_complete=True,
        now=now,
        freshness_seconds=300,
    )
    assert no_completion_proof.state is GoalState.PROVISIONALLY_COMPLETE
    assert not no_completion_proof.verified
    assert (
        no_completion_proof.gate is not None
        and not no_completion_proof.gate.passed
    )

    # A fully passing first evaluation still cannot jump directly from active
    # to verified; verification requires a later evaluation of the provisional
    # state with the complete proof population still valid.
    provisional = receipt.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert (
        provisional.gate is not None and provisional.gate.passed
    ), tuple(
        (check.name, check.reason_code, check.evidence)
        for check in provisional.gate.checks
        if not check.passed
    )
    assert provisional.acceptance_criteria == criteria
    assert "provisional_transition_required" in provisional.reason_codes
    assert provisional.gate.evaluated_evidence["analysis_result"] == {}

    verified = receipt.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified
    assert verified.gate is not None and verified.gate.passed

    # The closed criterion population cannot be narrowed, and every submitted
    # validation must itself be fresh and passing.
    missing = receipt.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": evidence[:-1]},
    )
    assert missing.state is GoalState.PROVISIONALLY_COMPLETE
    assert criteria[-1] in missing.missing_criteria
    assert "validation_evidence_incomplete" in missing.gate.fail_reason_codes

    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-059:failed",
        validation_passed=False,
        validation_receipt={"status": "failed", "tree_id": tree_id},
    )
    failed_submission = receipt.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (*evidence, failed)},
    )
    assert failed_submission.state is GoalState.PROVISIONALLY_COMPLETE
    assert "failed_validation" in failed_submission.reason_codes
    assert (
        "validation_evidence_incomplete"
        in failed_submission.gate.fail_reason_codes
    )

    stale = replace(
        evidence[0],
        provenance_cid="validation:asi-059:stale",
        observed_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    stale_submission = receipt.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (stale, *evidence[1:])},
    )
    assert stale_submission.state is GoalState.PROVISIONALLY_COMPLETE
    assert "stale_evidence" in stale_submission.reason_codes

    # A summary cannot claim coverage without mapping every criterion to both
    # implementation and validation.
    incomplete_coverage = copy.deepcopy(coverage)
    incomplete_coverage["criteria"][0]["implementation"] = ""
    for invalid_coverage in (
        incomplete_coverage,
        {**coverage, "criteria": coverage["criteria"][:-1]},
    ):
        coverage_gap = receipt.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "coverage": invalid_coverage},
        )
        assert coverage_gap.state is GoalState.PROVISIONALLY_COMPLETE
        assert any(
            code in coverage_gap.reason_codes
            for code in ("coverage_unverified", "coverage_missing")
        )

    # Analyzer health and completion safety must both be explicit.
    for invalid_health in (
        {"status": "healthy"},
        {**health, "safe_for_completion_reasoning": False},
        {**health, "healthy": False},
    ):
        unhealthy = receipt.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "analyzer_health": invalid_health},
        )
        assert unhealthy.state is GoalState.PROVISIONALLY_COMPLETE
        assert "analyzer_unhealthy" in unhealthy.reason_codes

    # The configured quorum requires independently derived scan channels and
    # receipts, and every fresh member must explicitly pass, be exhaustive,
    # healthy, conclusive, uncontradicted, completion-safe, and tree-bound.
    invalid_quorums = (
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "analyzer_version": (
                        quorum["members"][0]["analyzer_version"]
                    ),
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "receipt_cid": "scan:asi-059:implementation",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "exhaustive": False},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "healthy": False},
            ],
        },
        {
            **quorum,
            "member_count": 1,
            "members": [quorum["members"][0]],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "finished_at": "2026-07-24T12:00:00+00:00",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "binding": {**binding, "tree_id": "tree:foreign"},
                },
            ],
        },
    )
    for invalid_quorum in invalid_quorums:
        no_quorum = receipt.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "exhaustion_quorum": invalid_quorum},
        )
        assert no_quorum.state is GoalState.PROVISIONALLY_COMPLETE
        assert any(
            code.startswith("exhaustion_quorum")
            for code in no_quorum.reason_codes
        )

    foreign = replace(
        evidence[0],
        repository_tree="tree:foreign",
        tree_id="tree:foreign",
        provenance_cid="validation:asi-059:foreign",
    )
    wrong_tree = receipt.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (foreign, *evidence[1:])},
    )
    assert wrong_tree.state is GoalState.PROVISIONALLY_COMPLETE
    assert "repository_tree_mismatch" in wrong_tree.reason_codes


def test_candidate_self_report_cannot_manufacture_authority_evidence() -> None:
    goal = _goal()
    untrusted_claim = _candidate(
        goal,
        "untrusted-claim",
        cost=0.01,
        plan_overrides={"authority_violations": ("model says lease is missing",)},
    )
    valid = _candidate(goal, "valid", cost=12.0)

    receipt = AdaptivePlanner().select(goal, (untrusted_claim, valid))

    assert receipt.selected_candidate_id == "valid"
    assert receipt.evaluation.rejected[0].candidate_id == "untrusted-claim"
    assert untrusted_claim.receipt_for(HardPlanConstraint.AUTHORITY).passed
    assert receipt.evaluation.evidence_ids == ()
    assert receipt.to_dict()["evaluation"]["evidence_ids"] == []
    assert not receipt.proves_authority_non_compensation
    assert receipt.proved_requirement_ids == ()


@pytest.mark.parametrize(
    ("constraint", "dimension"),
    [
        (
            HardPlanConstraint.SCOPE,
            PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY,
        ),
        (
            HardPlanConstraint.SAFETY,
            PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY,
        ),
        (
            HardPlanConstraint.PROOF,
            PlanEvaluationDimension.VALIDATION_AND_PROOF,
        ),
    ],
)
def test_other_hard_failures_are_non_compensable_but_do_not_claim_authority_evidence(
    constraint: HardPlanConstraint,
    dimension: PlanEvaluationDimension,
) -> None:
    goal = _goal()
    invalid = _candidate(goal, "invalid", cost=0.001, failed=constraint)
    valid = _candidate(goal, "valid", cost=20.0)

    receipt = AdaptivePlanner().select(goal, (invalid, valid))

    assert receipt.selected_candidate_id == "valid"
    assert dimension.value in receipt.evaluation.rejected[0].hard_gate_failures
    assert not receipt.proves_authority_non_compensation
    assert receipt.proved_requirement_ids == ()


def test_every_quality_dimension_is_evaluated_and_weighting_is_deterministic() -> None:
    goal = _goal()
    alpha = _candidate(goal, "alpha", cost=4.0)
    beta = _candidate(goal, "beta", cost=4.0)

    forward = AdaptivePlanner().select(goal, (beta, alpha))
    reverse = AdaptivePlanner().select(goal, (alpha, beta))

    assert forward.selected_candidate_id == "alpha"
    assert reverse.selected_candidate_id == "alpha"
    assert forward.receipt_id == reverse.receipt_id
    assert {
        item.dimension for item in forward.evaluation.selected.dimensions
    } == set(PlanEvaluationDimension)
    assert forward.evaluation.covers_every_planning_dimension
    assert forward.evaluation.completion_dimension_population == tuple(
        item.value for item in PlanEvaluationDimension
    )
    assert all(
        {assessment.dimension for assessment in evaluated.dimensions}
        == set(PlanEvaluationDimension)
        for evaluated in forward.evaluation.ranked
    )
    assert forward.evaluation.selected.score_millionths == (
        reverse.evaluation.selected.score_millionths
    )


def test_stale_frozen_binding_fails_closed_before_ranking() -> None:
    goal = _goal()
    stale = _candidate(
        goal,
        "stale",
        cost=0.0,
        binding_overrides={"repository_tree_id": "tree:old"},
    )

    with pytest.raises(
        AdaptivePlannerValidationError, match="frozen goal bindings"
    ):
        AdaptivePlanner().select(goal, (stale,))


def test_candidate_boundary_requires_exact_trusted_gate_receipts() -> None:
    goal = _goal()
    candidate = _candidate(goal, "candidate", cost=1.0)

    assert FrozenPlanningGoal.from_dict(goal.to_dict()) == goal
    assert AdaptivePlanCandidate.from_dict(candidate.to_dict()) == candidate

    with pytest.raises(AdaptivePlannerValidationError, match="exactly one"):
        AdaptivePlanCandidate(
            plan=candidate.plan,
            goal_content_id=goal.goal_content_id,
            repository_tree_id=goal.repository_tree_id,
            policy_digest=goal.policy_digest,
            hard_constraint_receipts=candidate.hard_constraint_receipts[:-1],
        )

    with pytest.raises(AdaptivePlannerValidationError, match="cannot decide"):
        HardConstraintReceipt(
            constraint=HardPlanConstraint.AUTHORITY,
            candidate_id="candidate",
            candidate_snapshot_id=candidate.snapshot_id,
            goal_content_id=goal.goal_content_id,
            repository_tree_id=goal.repository_tree_id,
            policy_digest=goal.policy_digest,
            passed=True,
            producer_kind=GateProducerKind.PROOF_VERIFIER,
            producer_id="untrusted-for-authority",
            evidence_ids=("evidence:claim",),
            reason_codes=(),
        )


def test_selection_receipt_round_trips_persists_and_detects_tampering(
    tmp_path,
) -> None:
    goal = _goal()
    cheap_invalid = _candidate(
        goal, "cheap-invalid", cost=0.1, failed=HardPlanConstraint.AUTHORITY
    )
    valid = _candidate(goal, "valid", cost=2.0)
    receipt = AdaptivePlanner().select(goal, (cheap_invalid, valid))

    payload = receipt.to_dict()
    restored = AdaptivePlanSelectionReceipt.from_dict(payload)

    assert restored == receipt
    assert restored.receipt_id == receipt.receipt_id
    store = AdaptivePlanReceiptStore(tmp_path)
    path = store.persist(receipt)
    assert path.name == f"{receipt.receipt_id}.json"
    assert store.load(receipt.receipt_id) == receipt
    assert store.persist(receipt) == path

    payload["frozen_goal"]["repository_tree_id"] = "tree:tampered"
    with pytest.raises(
        AdaptivePlannerValidationError,
        match="frozen goal bindings|identity",
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)


def test_requirement_witness_is_recomputed_from_evaluation_and_gate_matrix() -> None:
    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (
            _candidate(
                goal,
                "cheap-invalid",
                cost=0.1,
                failed=HardPlanConstraint.AUTHORITY,
            ),
            _candidate(goal, "valid", cost=2.0),
        ),
    )

    def tampered_payload() -> dict[str, Any]:
        result = copy.deepcopy(receipt.to_dict())
        result.pop("receipt_id")
        result["authority_non_compensation_evidence"].pop("evidence_id")
        return result

    payload = tampered_payload()
    payload["authority_non_compensation_evidence"][
        "selected_candidate_id"
    ] = "cheap-invalid"
    with pytest.raises(
        AdaptivePlannerValidationError, match="selected candidate"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = tampered_payload()
    payload["authority_non_compensation_evidence"][
        "rejected_candidate_ids"
    ] = ["valid"]
    with pytest.raises(
        AdaptivePlannerValidationError, match="incomplete or inconsistent"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = tampered_payload()
    payload["authority_non_compensation_evidence"][
        "rejected_cost_millionths"
    ] = [1]
    with pytest.raises(
        AdaptivePlannerValidationError, match="incomplete or inconsistent"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = tampered_payload()
    payload["authority_non_compensation_evidence"][
        "authority_receipt_ids"
    ] = ["receipt:forged"]
    with pytest.raises(
        AdaptivePlannerValidationError, match="incomplete or inconsistent"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = tampered_payload()
    receipts = payload["hard_constraint_receipts"]
    cheap_authority = next(
        item
        for item in receipts
        if item["candidate_id"] == "cheap-invalid"
        and item["constraint"] == "authority"
    )
    cheap_scope_index = next(
        index
        for index, item in enumerate(receipts)
        if item["candidate_id"] == "cheap-invalid"
        and item["constraint"] == "scope"
    )
    receipts[cheap_scope_index] = copy.deepcopy(cheap_authority)
    with pytest.raises(
        AdaptivePlannerValidationError, match="candidate/constraint pair"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)


def test_candidate_budget_duplicate_identity_and_empty_input_are_bounded() -> None:
    goal = _goal()
    candidate = _candidate(goal, "only", cost=1.0)

    with pytest.raises(AdaptivePlannerValidationError, match="at least one"):
        AdaptivePlanner().select(goal, ())
    with pytest.raises(AdaptivePlannerValidationError, match="unique"):
        AdaptivePlanner().select(goal, (candidate, candidate))
    with pytest.raises(AdaptivePlannerValidationError, match="budget"):
        AdaptivePlanner(max_candidates=1).select(
            goal, (candidate, _candidate(goal, "second", cost=2.0))
        )


def test_hard_gate_receipt_cannot_be_replayed_onto_changed_plan_content() -> None:
    goal = _goal()
    inspected = _candidate(
        goal,
        "same-branch-id",
        cost=100.0,
        failed=HardPlanConstraint.AUTHORITY,
    )

    with pytest.raises(
        AdaptivePlannerValidationError, match="candidate content"
    ):
        replace(
            inspected,
            plan=_plan("same-branch-id", cost=0.01),
        )


def test_hard_gate_receipt_cannot_be_replayed_after_formal_provenance_change() -> None:
    goal = _goal()
    transition = _repair_transition(repaired_plan_id="formal-plan:repaired:v1")
    inspected = _candidate(
        goal,
        "formal-candidate",
        cost=1.0,
        formal_plan_id=transition.repaired_plan_id,
        repair_transition=transition,
    )

    # Changing only a formal plan identity changes the canonical candidate
    # snapshot even when no repair transition is attached.
    plan_only = _candidate(
        goal,
        "formal-plan-only",
        cost=1.0,
        formal_plan_id="formal-plan:v1",
    )
    with pytest.raises(
        AdaptivePlannerValidationError, match="candidate content"
    ):
        replace(plan_only, formal_plan_id="formal-plan:v2")

    # A coherent new formal plan plus transition also cannot reuse the four
    # hard-gate observations made against the previous transition.
    changed_transition = _repair_transition(
        repaired_plan_id="formal-plan:repaired:v2",
        counterexample_id="counterexample:authority:v2",
    )
    with pytest.raises(
        AdaptivePlannerValidationError, match="candidate content"
    ):
        replace(
            inspected,
            formal_plan_id=changed_transition.repaired_plan_id,
            repair_transition=changed_transition,
        )

    # Nested restored provenance uses the formal replanner's own fail-closed
    # schema and version boundary before candidate snapshot validation.
    payload = inspected.to_dict()
    payload["repair_transition"]["replanner_version"] = 999
    with pytest.raises(ValueError, match="formal replanner version"):
        AdaptivePlanCandidate.from_dict(payload)


def test_authority_witness_is_complete_for_every_qualifying_rejection() -> None:
    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (
            _candidate(
                goal,
                "cheap-b",
                cost=0.2,
                failed=HardPlanConstraint.AUTHORITY,
            ),
            _candidate(
                goal,
                "cheap-a",
                cost=0.1,
                failed=HardPlanConstraint.AUTHORITY,
            ),
            _candidate(goal, "safe", cost=2.0),
        ),
    )

    evidence = receipt.authority_non_compensation_evidence
    assert evidence is not None
    assert evidence.rejected_candidate_ids == ("cheap-a", "cheap-b")

    omitted = copy.deepcopy(receipt.to_dict())
    omitted.pop("receipt_id")
    witness = omitted["authority_non_compensation_evidence"]
    witness.pop("evidence_id")
    for field in (
        "rejected_candidate_ids",
        "rejected_cost_millionths",
        "authority_receipt_ids",
    ):
        witness[field].pop()
    with pytest.raises(
        AdaptivePlannerValidationError, match="incomplete or inconsistent"
    ):
        AdaptivePlanSelectionReceipt.from_dict(omitted)

    stripped = copy.deepcopy(receipt.to_dict())
    stripped.pop("receipt_id")
    stripped["authority_non_compensation_evidence"] = None
    stripped["proved_requirement_ids"] = []
    with pytest.raises(
        AdaptivePlannerValidationError, match="exactly cover"
    ):
        AdaptivePlanSelectionReceipt.from_dict(stripped)


@pytest.mark.parametrize("invalid_cost", [2.0, 3.0])
def test_non_cheaper_authority_failure_does_not_claim_requirement(
    invalid_cost: float,
) -> None:
    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (
            _candidate(
                goal,
                "authority-invalid",
                cost=invalid_cost,
                failed=HardPlanConstraint.AUTHORITY,
            ),
            _candidate(goal, "safe", cost=2.0),
        ),
    )

    assert receipt.selected_candidate_id == "safe"
    assert receipt.authority_non_compensation_evidence is None
    assert receipt.proved_requirement_ids == ()


def test_no_admissible_plan_emits_no_objective_evidence() -> None:
    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (
            _candidate(
                goal,
                "authority-invalid",
                cost=0.01,
                failed=HardPlanConstraint.AUTHORITY,
            ),
        ),
    )

    assert receipt.selected is None
    assert receipt.authority_non_compensation_evidence is None
    assert receipt.proved_requirement_ids == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("planner_version", 999, "planner version"),
        ("evaluator_version", "unsupported", "receipt evaluator version"),
    ],
)
def test_selection_receipt_rejects_unsupported_outer_versions(
    field: str,
    value: Any,
    message: str,
) -> None:
    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (_candidate(goal, "safe", cost=1.0),),
    )
    payload = receipt.to_dict()
    payload.pop("receipt_id")
    payload[field] = value

    with pytest.raises(AdaptivePlannerValidationError, match=message):
        AdaptivePlanSelectionReceipt.from_dict(payload)


def test_selection_receipt_recomputes_persisted_evaluation() -> None:
    goal = _goal()
    receipt = AdaptivePlanner().select(
        goal,
        (_candidate(goal, "safe", cost=1.0),),
    )
    payload = receipt.to_dict()
    payload.pop("receipt_id")
    payload["evaluation"]["selected"]["score_millionths"] += 1
    payload["evaluation"]["admissible"][0]["score_millionths"] += 1

    with pytest.raises(
        AdaptivePlannerValidationError, match="deterministic recomputation"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload.pop("receipt_id")
    payload["evaluation"]["evaluator_version"] = "unsupported"
    with pytest.raises(
        AdaptivePlannerValidationError, match="evaluator version"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)


def _provider_plan(
    candidate_id: str,
    provider: AdaptiveCandidateProviderKind,
    *,
    cost: float = 0.25,
) -> EvidenceAwarePlanCandidate:
    plan = _plan(candidate_id, cost=cost)
    return replace(
        plan,
        branch=replace(plan.branch, source=provider.value),
        estimated_runtime_seconds=0.25,
    )


def _planning_context() -> dict[str, Any]:
    return {
        "title": "Implement adaptive planning",
        "outputs": ["src/planner.py", "tests/test_planner.py"],
        "predicted_symbols": ["AdaptivePlanner.plan"],
        "dependencies": ["dependency:context"],
        "validation_commands": [
            "python -m pytest tests/test_planner.py -q"
        ],
        "estimated_tokens": 100,
        "estimated_runtime_seconds": 1.0,
        "estimated_resource_cost": 1.0,
        "resource_classes": ["cpu"],
    }


def test_full_pipeline_routes_all_optional_providers_over_one_frozen_context() -> None:
    observed_requests: list[Any] = []

    def provider(kind: AdaptiveCandidateProviderKind) -> Any:
        def generate(request: Any) -> dict[str, Any]:
            observed_requests.append(request)
            return {
                "candidates": [_provider_plan(f"{kind.value}:candidate", kind)],
                "input_tokens": 40,
                "output_tokens": 20,
                "runtime_milliseconds": 3,
                "resource_cost_millionths": 10,
            }

        return generate

    result = plan_adaptively(
        _goal(),
        _planning_context(),
        providers={
            kind: provider(kind)
            for kind in (
                AdaptiveCandidateProviderKind.LLM,
                AdaptiveCandidateProviderKind.LEANSTRAL,
                AdaptiveCandidateProviderKind.IPFS_DATASETS,
            )
        },
    )

    assert len(observed_requests) == 3
    assert observed_requests[0] is observed_requests[1] is observed_requests[2]
    with pytest.raises(TypeError):
        observed_requests[0].context["title"] = "provider mutation"
    assert {item.request_id for item in result.routing.outcomes} == {
        observed_requests[0].request_id
    }
    assert [item.provider_kind for item in result.routing.outcomes] == list(
        AdaptiveCandidateProviderKind
    )
    assert all(
        item.status is CandidateProviderStatus.SUCCEEDED
        for item in result.routing.outcomes
    )
    assert not result.fallback_used
    assert result.selected_candidate_id == "ipfs_datasets_py:candidate"
    payload = result.to_dict()
    assert len(payload["paired_quality_cost_metrics"]) == 4
    assert payload["non_selection_reasons"]["baseline:ASI-G097"] == [
        "lower_deterministic_quality_cost_score"
    ]
    assert all(
        isinstance(item["estimated_runtime_milliseconds"], int)
        for item in payload["paired_quality_cost_metrics"]
    )


def test_optional_provider_unavailability_timeout_and_malformed_output_fall_back() -> None:
    def unavailable(_request: Any) -> Any:
        raise ImportError("optional SDK is not installed")

    def slow(_request: Any) -> Any:
        time.sleep(0.05)
        return _provider_plan(
            "late", AdaptiveCandidateProviderKind.LEANSTRAL
        )

    result = AdaptivePlanner().plan(
        _goal(),
        _planning_context(),
        providers={
            AdaptiveCandidateProviderKind.LLM: unavailable,
            AdaptiveCandidateProviderKind.LEANSTRAL: slow,
            AdaptiveCandidateProviderKind.IPFS_DATASETS: (
                lambda _request: {"not": "a candidate"}
            ),
        },
        bounds=CandidateGenerationBounds(timeout_seconds=0.005),
    )

    assert result.selected_candidate_id == "baseline:ASI-G097"
    assert result.fallback_used
    assert [item.status for item in result.routing.outcomes] == [
        CandidateProviderStatus.SUCCEEDED,
        CandidateProviderStatus.FAILED,
        CandidateProviderStatus.TIMED_OUT,
        CandidateProviderStatus.MALFORMED,
    ]
    assert [item.reason_code for item in result.routing.outcomes[1:]] == [
        "provider_exception",
        "provider_timeout",
        "malformed_provider_result",
    ]


def test_adversarial_high_quality_candidate_cannot_compensate_for_authority() -> None:
    adversarial = replace(
        _provider_plan(
            "adversarial", AdaptiveCandidateProviderKind.LLM, cost=0.0001
        ),
        novelty=1.0,
        branch=replace(
            _provider_plan(
                "adversarial", AdaptiveCandidateProviderKind.LLM, cost=0.0001
            ).branch,
            expected_objective_delta=1.0,
            risk=0.0,
        ),
        estimated_tokens=0,
        estimated_runtime_seconds=0.0,
        estimated_resource_cost=0.0,
    )

    def gates(plan: Any, goal: Any, request: Any) -> Any:
        receipts = deterministic_hard_gate_receipts(plan, goal, request)
        if plan.candidate_id != "adversarial":
            return receipts
        return tuple(
            replace(
                item,
                passed=False,
                reason_codes=("authorization_denied",),
            )
            if item.constraint is HardPlanConstraint.AUTHORITY
            else item
            for item in receipts
        )

    result = AdaptivePlanner().plan(
        _goal(),
        _planning_context(),
        providers={
            AdaptiveCandidateProviderKind.LLM: (
                lambda _request: (adversarial,)
            )
        },
        hard_gate_evaluator=gates,
    )

    assert result.selected_candidate_id == "baseline:ASI-G097"
    rejected = result.selection.evaluation.rejected
    assert [item.candidate_id for item in rejected] == ["adversarial"]
    assert rejected[0].candidate.novelty == 1.0
    assert rejected[0].candidate.estimated_resource_cost == 0.0
    assert result.non_selection_reasons["adversarial"] == (
        "hard_gate_failed:conflict_scope_and_authority",
    )


def test_complete_planning_run_round_trips_persists_and_binds_context(
    tmp_path: Any,
) -> None:
    result = AdaptivePlanner().plan(_goal(), _planning_context())
    restored = AdaptivePlanningRunReceipt.from_dict(result.to_dict())

    assert restored == result
    store = AdaptivePlanningRunStore(tmp_path)
    path = store.persist(result)
    assert path.name == f"{result.run_id}.json"
    assert store.load(result.run_id) == result
    assert store.persist(result) == path

    tampered = copy.deepcopy(result.to_dict())
    tampered.pop("run_id")
    tampered["routing"].pop("routing_id")
    tampered["routing"]["request"]["context"]["title"] = "different goal context"
    with pytest.raises(
        AdaptivePlannerValidationError,
        match="context_id|identity",
    ):
        AdaptivePlanningRunReceipt.from_dict(tampered)

    tampered = copy.deepcopy(result.to_dict())
    tampered.pop("run_id")
    tampered["routing"].pop("routing_id")
    tampered["routing"]["candidates"][0]["estimated_tokens"] += 1
    with pytest.raises(
        AdaptivePlannerValidationError,
        match="routed candidate content",
    ):
        AdaptivePlanningRunReceipt.from_dict(tampered)


def test_provider_candidate_and_response_budgets_are_recorded_not_raised() -> None:
    oversized = tuple(
        _provider_plan(f"candidate-{index}", AdaptiveCandidateProviderKind.LLM)
        for index in range(3)
    )
    result = AdaptivePlanner().plan(
        _goal(),
        _planning_context(),
        providers={
            AdaptiveCandidateProviderKind.LLM: lambda _request: oversized,
        },
        bounds=CandidateGenerationBounds(max_candidates_per_provider=2),
    )

    assert result.selected_candidate_id == "baseline:ASI-G097"
    llm = result.routing.outcomes[1]
    assert llm.status is CandidateProviderStatus.BUDGET_REJECTED
    assert llm.reason_code == "provider_budget_exceeded"
