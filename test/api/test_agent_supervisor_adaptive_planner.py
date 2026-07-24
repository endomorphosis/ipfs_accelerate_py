from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adaptive_planner import (
    AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,
    AdaptivePlanCandidate,
    AdaptivePlanReceiptStore,
    AdaptivePlanSelectionReceipt,
    AdaptivePlanner,
    AdaptivePlannerValidationError,
    FrozenPlanningGoal,
    GateProducerKind,
    HardConstraintReceipt,
    HardPlanConstraint,
    select_adaptive_plan,
)
from ipfs_accelerate_py.agent_supervisor.plan_evaluator import (
    EvidenceAwarePlanCandidate,
    EvidenceAwarePlanPolicy,
    PlanBranch,
    PlanEvaluationDimension,
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
    receipts = tuple(
        HardConstraintReceipt(
            constraint=constraint,
            candidate_id=candidate_id,
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
        **bindings,
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
        AdaptivePlannerValidationError, match="was not rejected"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = tampered_payload()
    payload["authority_non_compensation_evidence"][
        "rejected_cost_millionths"
    ] = [1]
    with pytest.raises(
        AdaptivePlannerValidationError, match="rejected cost"
    ):
        AdaptivePlanSelectionReceipt.from_dict(payload)

    payload = tampered_payload()
    payload["authority_non_compensation_evidence"][
        "authority_receipt_ids"
    ] = ["receipt:forged"]
    with pytest.raises(
        AdaptivePlannerValidationError, match="authority receipt"
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
