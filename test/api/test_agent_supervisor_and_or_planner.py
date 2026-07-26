from __future__ import annotations

import copy
from dataclasses import replace
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.adaptive_planner import (
    AND_OR_SEARCH_REQUIREMENT_ID,
    AndOrNodeKind,
    AndOrPlanAlternative,
    AndOrPlannerBenchmark,
    AndOrProducerKind,
    AndOrSearchBounds,
    AndOrSearchReceipt,
    AdaptivePlannerValidationError,
    FrozenAndOrPlanningContext,
    compile_typed_goal_to_and_or_graph,
    evaluate_and_or_planner_promotion,
    search_typed_goal_plans,
)
from ipfs_accelerate_py.agent_supervisor.goal_quality import (
    AcceptanceCriterion,
    EvidenceAuthority,
    EvidenceProducer,
    FreshnessPolicy,
    FrozenRootIdentity,
    GoalScope,
    RefinementBudget,
    ResourceEnvelope,
    TypedGoal,
    UncertaintyDisposition,
    UncertaintyItem,
    UnsupportedSemantic,
    ValidationRule,
)
from ipfs_accelerate_py.agent_supervisor.plan_evaluator import (
    AndOrPlanBranch,
    PlanSearchHardConstraint,
    PlanSearchHardFailure,
    evaluate_and_or_plan_branches,
)


def _goal() -> TypedGoal:
    criteria = (
        AcceptanceCriterion(
            criterion_id="criterion:contract",
            statement="The typed plan graph has canonical AND/OR structure.",
            evidence_producer_ids=("producer:validation",),
            validation_rule_ids=("validation:pytest",),
            completion_signal="graph.valid",
        ),
        AcceptanceCriterion(
            criterion_id="criterion:pruning",
            statement="Every hard-invalid branch is pruned before scoring.",
            evidence_producer_ids=("producer:validation",),
            validation_rule_ids=("validation:pytest",),
            depends_on_criterion_ids=("criterion:contract",),
            completion_signal="evaluation.hard_violations == 0",
        ),
    )
    return TypedGoal(
        goal_id="ASI-G230",
        root=FrozenRootIdentity("ASI-G200", "objective:planning-v2"),
        outcome="Select one bounded, valid implementation plan.",
        scope=GoalScope(
            include=(
                "ipfs_accelerate_py/agent_supervisor/adaptive_planner.py",
                "ipfs_accelerate_py/agent_supervisor/plan_evaluator.py",
            ),
            exclude=("docs/architecture",),
            dependency_goal_ids=("ASI-G210",),
        ),
        assumptions=("The repository tree and policy are frozen.",),
        non_goals=("Grant completion authority.",),
        acceptance_criteria=criteria,
        evidence_producers=(
            EvidenceProducer(
                producer_id="producer:validation",
                kind="test_runner",
                output_schema="schema:pytest@1",
                authority=EvidenceAuthority.VALIDATION,
                capability_id="capability:pytest",
                independent=True,
            ),
        ),
        validation_rules=(
            ValidationRule(
                rule_id="validation:pytest",
                command=(
                    "python -m pytest "
                    "test/api/test_agent_supervisor_and_or_planner.py -q"
                ),
                producer_id="producer:validation",
                criterion_ids=(
                    "criterion:contract",
                    "criterion:pruning",
                ),
                hermetic=True,
            ),
        ),
        freshness=FreshnessPolicy(max_age_seconds=600),
        resources=ResourceEnvelope(
            max_wall_seconds=30,
            max_tokens=10_000,
            max_cost_microunits=100_000,
            max_artifacts=16,
            max_parallelism=4,
            max_scope_items=8,
        ),
        uncertainties=(
            UncertaintyItem(
                uncertainty_id="uncertainty:provider-quality",
                statement="An optional producer may improve the baseline.",
                disposition=UncertaintyDisposition.OPEN,
                impact="May alter the selected valid branch.",
                resolution="Compare typed branch features.",
            ),
        ),
        unsupported_semantics=(
            UnsupportedSemantic(
                semantic_id="semantic:provider-reasoning",
                statement="Provider reasoning text is not planning evidence.",
                fallback="Retain only typed features and reason codes.",
            ),
        ),
        refinement_budget=RefinementBudget(
            max_rounds=2,
            max_children=8,
            max_depth=4,
            max_debt_items=16,
            max_tokens=10_000,
        ),
    )


def _alternative(
    frozen: FrozenAndOrPlanningContext,
    alternative_id: str,
    producer_kind: AndOrProducerKind,
    *,
    obligation_id: str = "criterion:contract",
    **overrides: object,
) -> AndOrPlanAlternative:
    values: dict[str, object] = {
        "alternative_id": alternative_id,
        "obligation_id": obligation_id,
        "producer_kind": producer_kind,
        "goal_content_id": frozen.goal.content_id,
        "repository_tree_id": frozen.repository_tree_id,
        "context_id": frozen.context_id,
        "evidence_ids": (f"evidence:{alternative_id}",),
        "reduced_uncertainty_ids": ("uncertainty:provider-quality",),
        "changed_scopes": (),
        "authorized_scopes": (),
        "dependencies": ("ASI-G210",),
        "satisfied_dependencies": ("ASI-G210",),
        "critical_path_length": 0,
    }
    values.update(overrides)
    return AndOrPlanAlternative(**values)


def _frozen() -> FrozenAndOrPlanningContext:
    return FrozenAndOrPlanningContext(
        goal=_goal(),
        repository_tree_id="tree:asi-104",
        policy_revision="policy:asi-104",
        context={"task_id": "ASI-104", "tree": "tree:asi-104"},
    )


def test_compiler_builds_joint_and_alternative_or_nodes_with_baselines() -> None:
    frozen = _frozen()
    graph = compile_typed_goal_to_and_or_graph(
        frozen,
        (
            _alternative(frozen, "llm:contract", AndOrProducerKind.LLM),
            _alternative(
                frozen,
                "leanstral:pruning",
                AndOrProducerKind.LEANSTRAL,
                obligation_id="criterion:pruning",
            ),
            _alternative(
                frozen,
                "analysis:contract",
                AndOrProducerKind.ANALYSIS_PROVIDER,
            ),
        ),
    )

    by_id = {item.node_id: item for item in graph.nodes}
    root = by_id[graph.root_node_id]
    assert root.kind is AndOrNodeKind.AND
    assert len(root.child_ids) == len(frozen.goal.acceptance_criteria)
    assert all(by_id[item].kind is AndOrNodeKind.OR for item in root.child_ids)
    for or_id in root.child_ids:
        leaves = [by_id[item] for item in by_id[or_id].child_ids]
        assert any(
            item.producer_kind is AndOrProducerKind.DETERMINISTIC_BASELINE
            for item in leaves
        )
    assert {
        item.producer_kind
        for item in graph.nodes
        if item.kind is AndOrNodeKind.PRODUCER
    } == {
        AndOrProducerKind.DETERMINISTIC_BASELINE,
        AndOrProducerKind.LLM,
        AndOrProducerKind.LEANSTRAL,
        AndOrProducerKind.ANALYSIS_PROVIDER,
    }
    assert graph.max_depth == 3


def test_all_optional_providers_receive_the_same_frozen_context_object() -> None:
    observed: list[FrozenAndOrPlanningContext] = []

    def provider(kind: AndOrProducerKind):
        def generate(frozen: FrozenAndOrPlanningContext):
            observed.append(frozen)
            return (_alternative(frozen, f"{kind.value}:one", kind),)

        return generate

    receipt = search_typed_goal_plans(
        _goal(),
        repository_tree_id="tree:asi-104",
        policy_revision="policy:asi-104",
        context={"task_id": "ASI-104"},
        providers={
            kind: provider(kind)
            for kind in (
                AndOrProducerKind.LLM,
                AndOrProducerKind.LEANSTRAL,
                AndOrProducerKind.ANALYSIS_PROVIDER,
            )
        },
    )

    assert len(observed) == 3
    assert observed[0] is observed[1] is observed[2]
    with pytest.raises(TypeError):
        observed[0].context["task_id"] = "forged"  # type: ignore[index]
    assert receipt.selected is not None
    assert receipt.requirement_ids == (AND_OR_SEARCH_REQUIREMENT_ID,)
    assert not receipt.provider_failures


@pytest.mark.parametrize(
    ("constraint", "overrides"),
    (
        (PlanSearchHardConstraint.AUTHORITY, {"authority_granted": False}),
        (
            PlanSearchHardConstraint.SCOPE,
            {
                "changed_scopes": ("outside.py",),
                "authorized_scopes": (),
            },
        ),
        (
            PlanSearchHardConstraint.DEPENDENCY,
            {"satisfied_dependencies": ()},
        ),
        (
            PlanSearchHardConstraint.RESOURCE,
            {"resource_available": False},
        ),
        (PlanSearchHardConstraint.FRESHNESS, {"fresh": False}),
        (
            PlanSearchHardConstraint.VALIDATION,
            {"validation_feasible": False},
        ),
        (PlanSearchHardConstraint.PROOF, {"proof_feasible": False}),
    ),
)
def test_each_hard_violation_is_pruned_before_soft_scoring(
    constraint: PlanSearchHardConstraint,
    overrides: dict[str, object],
) -> None:
    frozen = _frozen()
    invalid = _alternative(
        frozen,
        f"invalid:{constraint.value}",
        AndOrProducerKind.LLM,
        **overrides,
    )
    receipt = search_typed_goal_plans(
        frozen.goal,
        repository_tree_id=frozen.repository_tree_id,
        policy_revision=frozen.policy_revision,
        context={"task_id": "ASI-104", "tree": "tree:asi-104"},
        alternatives=(invalid,),
    )

    matching = [
        item
        for item in receipt.evaluation.pruned
        if invalid.alternative_id in item.branch.alternative_ids
    ]
    assert matching
    assert all(item.score_millionths is None for item in matching)
    assert any(
        failure.constraint is constraint
        for item in matching
        for failure in item.branch.hard_failures
    )
    assert receipt.selected is not None
    assert not receipt.selected.hard_failures


def _evaluation_branch(branch_id: str, **overrides: object) -> AndOrPlanBranch:
    values: dict[str, object] = {
        "branch_id": branch_id,
        "goal_content_id": "goal:content",
        "repository_tree_id": "tree:one",
        "context_id": "context:one",
        "alternative_ids": (f"alternative:{branch_id}",),
        "producer_kinds": ("llm",),
        "required_obligation_ids": ("criterion:a",),
        "covered_obligation_ids": ("criterion:a",),
        "required_uncertainty_ids": ("uncertainty:a",),
        "reduced_uncertainty_ids": ("uncertainty:a",),
    }
    values.update(overrides)
    return AndOrPlanBranch(**values)


def test_soft_ranking_covers_all_six_dimensions_and_ties_are_stable() -> None:
    alpha = _evaluation_branch("alpha")
    zeta = _evaluation_branch("zeta")
    costly = _evaluation_branch(
        "costly",
        critical_path_length=5,
        conflict_risk_millionths=900_000,
        estimated_cost_microunits=900_000,
        historical_failure_millionths=800_000,
        reduced_uncertainty_ids=(),
    )

    forward = evaluate_and_or_plan_branches((zeta, costly, alpha))
    reverse = evaluate_and_or_plan_branches((alpha, costly, zeta))

    assert forward.to_dict() == reverse.to_dict()
    assert forward.selected is not None
    assert forward.selected.branch_id == "alpha"
    assert set(forward.selected.soft_scores) == {
        "evidence_coverage",
        "uncertainty_reduction",
        "critical_path",
        "conflict_risk",
        "cost",
        "historical_failure",
    }
    assert forward.ranked[-1].branch_id == "costly"


def test_search_enforces_depth_node_branch_token_and_time_bounds() -> None:
    with pytest.raises(AdaptivePlannerValidationError, match="max_depth"):
        AndOrSearchBounds(max_depth=2)

    frozen = _frozen()
    alternatives = tuple(
        _alternative(
            frozen,
            f"llm:{index}",
            AndOrProducerKind.LLM,
            estimated_tokens=10,
        )
        for index in range(6)
    )
    receipt = search_typed_goal_plans(
        frozen.goal,
        repository_tree_id=frozen.repository_tree_id,
        policy_revision=frozen.policy_revision,
        context={"task_id": "ASI-104", "tree": "tree:asi-104"},
        alternatives=alternatives,
        bounds=AndOrSearchBounds(
            max_depth=3,
            max_nodes=6,
            max_branches=2,
            max_alternatives_per_or=2,
            max_tokens=100,
            max_time_milliseconds=1_000,
        ),
    )

    assert receipt.visited_nodes <= receipt.bounds.max_nodes
    assert receipt.generated_branches <= receipt.bounds.max_branches
    assert receipt.consumed_tokens <= receipt.bounds.max_tokens
    assert receipt.elapsed_milliseconds <= receipt.bounds.max_time_milliseconds
    assert receipt.graph.truncated
    assert receipt.termination_reason in {
        "branch_budget_exhausted",
        "node_or_alternative_budget_exhausted",
    }


def test_slow_optional_provider_times_out_to_the_mandatory_baseline() -> None:
    def slow(_frozen: FrozenAndOrPlanningContext):
        time.sleep(0.05)
        return ()

    receipt = search_typed_goal_plans(
        _goal(),
        repository_tree_id="tree:asi-104",
        policy_revision="policy:asi-104",
        context={"task_id": "ASI-104"},
        providers={AndOrProducerKind.LLM: slow},
        bounds=AndOrSearchBounds(max_time_milliseconds=5),
    )

    assert receipt.selected is not None
    assert set(receipt.selected.producer_kinds) == {
        AndOrProducerKind.DETERMINISTIC_BASELINE.value
    }
    assert receipt.termination_reason == "time_budget_exhausted_baseline_fallback"
    assert receipt.provider_failures == (
        "llm:search_time_budget_exhausted",
    )


def test_search_receipt_round_trips_and_rejects_score_tampering() -> None:
    receipt = search_typed_goal_plans(
        _goal(),
        repository_tree_id="tree:asi-104",
        policy_revision="policy:asi-104",
        context={"task_id": "ASI-104"},
    )
    assert AndOrSearchReceipt.from_dict(receipt.to_dict()) == receipt

    tampered = copy.deepcopy(receipt.to_dict())
    tampered["evaluation"]["ranked"][0]["score_millionths"] = 1
    tampered["evaluation"]["selected"]["score_millionths"] = 1
    with pytest.raises(ValueError, match="recomputation"):
        AndOrSearchReceipt.from_dict(tampered)


def test_pruned_branch_can_never_win_even_with_perfect_soft_features() -> None:
    invalid = _evaluation_branch(
        "invalid-perfect",
        hard_failures=(
            PlanSearchHardFailure(
                PlanSearchHardConstraint.AUTHORITY,
                ("authority_denied",),
            ),
        ),
    )
    valid = _evaluation_branch(
        "valid-costly",
        critical_path_length=100,
        conflict_risk_millionths=999_999,
        estimated_cost_microunits=100_000_000,
        historical_failure_millionths=999_999,
        reduced_uncertainty_ids=(),
    )

    evaluation = evaluate_and_or_plan_branches((invalid, valid))

    assert evaluation.selected is not None
    assert evaluation.selected.branch_id == "valid-costly"
    assert evaluation.pruned[0].score_millionths is None


def test_v2_promotion_gate_accepts_either_metric_but_never_hard_violations() -> None:
    baseline = AndOrPlannerBenchmark(40, 100, 40, 100)
    valid_first = AndOrPlannerBenchmark(55, 100, 39, 100)
    fewer_invalid = AndOrPlannerBenchmark(40, 100, 30, 100)

    assert evaluate_and_or_planner_promotion(baseline, valid_first).passed
    assert evaluate_and_or_planner_promotion(baseline, fewer_invalid).passed
    blocked = evaluate_and_or_planner_promotion(
        baseline,
        replace(valid_first, hard_constraint_violations=1),
    )
    assert not blocked.passed
    assert "hard_constraint_violation" in blocked.reason_codes
