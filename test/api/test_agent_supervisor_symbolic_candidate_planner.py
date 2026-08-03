"""Contract tests for deterministic-first symbolic candidate portfolios."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import (
    AdaptivePlanner,
    FrozenPlanningGoal,
)
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority,
    FactTruth,
    ObservedFact,
    ProducerRule,
    TaskCandidate,
    TypedIntent,
    TypedPredicate,
    compile_obligation_graph,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import (
    EvidenceAwarePlanPolicy,
)
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    SYMBOLIC_CANDIDATE_PLANNER_INTERFACE,
    SymbolicCandidateBounds,
    SymbolicCandidatePlanner,
    SymbolicCandidatePlanningError,
    SymbolicCandidatePortfolio,
    SymbolicCandidateSource,
    SymbolicProviderStatus,
)


def _predicate(predicate_id: str) -> TypedPredicate:
    return TypedPredicate(
        predicate_id=predicate_id,
        predicate_type="behavior_state",
        subject_ref=predicate_id,
        provenance_refs=(f"src/{predicate_id}.py",),
        proof_requirement_refs=(f"proof:{predicate_id}",),
        validation_requirement_refs=(f"validation:{predicate_id}",),
    )


def _fixture():
    goal = _predicate("goal")
    prerequisite = _predicate("prerequisite")
    direct = ProducerRule(
        producer_id="producer:direct",
        effect_predicate_ids=(goal.predicate_id,),
        provenance_refs=("src/direct.py",),
        proof_requirement_refs=("proof:direct",),
    )
    layered = ProducerRule(
        producer_id="producer:layered",
        effect_predicate_ids=(goal.predicate_id,),
        required_predicate_ids=(prerequisite.predicate_id,),
        provenance_refs=("src/layered.py",),
        proof_requirement_refs=("proof:layered",),
    )
    direct_task = TaskCandidate(
        candidate_id="task:direct",
        closes_obligation_ids=(
            obligation_id_for_producer(direct.producer_id, goal.predicate_id),
        ),
        producer_id=direct.producer_id,
        provenance_refs=("src/direct.py",),
    )
    layered_task = TaskCandidate(
        candidate_id="task:layered",
        closes_obligation_ids=(
            obligation_id_for_producer(layered.producer_id, goal.predicate_id),
        ),
        producer_id=layered.producer_id,
        provenance_refs=("src/layered.py",),
    )
    graph = compile_obligation_graph(
        TypedIntent(
            intent_id="intent:test",
            desired_predicates=(goal,),
            source_refs=("intent-source:test",),
            current_root_id="tree:test",
        ),
        current_facts=(
            ObservedFact(
                fact_id="fact:prerequisite",
                predicate=prerequisite,
                truth=FactTruth.TRUE,
                authority=FactAuthority.CURRENT_ROOT_FACT,
                provenance_refs=("evidence:prerequisite",),
                current_root_id="tree:test",
            ),
        ),
        producers=(direct, layered),
        predicates=(prerequisite,),
        task_candidates=(direct_task, layered_task),
    )
    goal_binding = FrozenPlanningGoal(
        goal_id="goal:test",
        goal_content_id="goal-content:test",
        repository_tree_id="tree:test",
        policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=("acceptance:test",),
            evidence_terms=("evidence:test",),
            trusted_assumptions=("assumption:frozen",),
            supported_semantics=("semantics:typed",),
            allowed_scopes=("scope:src",),
            available_resource_classes=("cpu",),
            require_validation=True,
            require_proof=True,
        ),
    )
    context = {
        "repository_paths": ["src/direct.py", "src/layered.py"],
        "task_metadata": {
            "task:direct": {
                "predicted_files": ["src/direct.py"],
                "predicted_symbols": ["direct"],
                "scope_ids": ["scope:src"],
                "resource_classes": ["cpu"],
                "estimated_cost_millionths": 100_000,
            },
            "task:layered": {
                "predicted_files": ["src/layered.py"],
                "predicted_symbols": ["layered"],
                "scope_ids": ["scope:src"],
                "resource_classes": ["cpu"],
                "estimated_cost_millionths": 200_000,
            },
        },
    }
    return graph, goal_binding, context


def test_candidate_count_controls_stable_codebase_derived_or_portfolio() -> None:
    graph, goal, context = _fixture()
    planner = SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=2, max_model_candidates=0)
    )

    first = planner.plan(graph, goal, context, allow_model=False)
    second = planner.plan(graph, goal, context, allow_model=False)

    assert first.request.interface == SYMBOLIC_CANDIDATE_PLANNER_INTERFACE
    assert len(first.snapshots) == 2
    assert first.baseline.symbolic_candidate.source is (
        SymbolicCandidateSource.DETERMINISTIC_BASELINE
    )
    assert {
        item.symbolic_candidate.task_candidate_ids for item in first.snapshots
    } == {("task:direct",), ("task:layered",)}
    assert all(
        "backward_chaining" in item.symbolic_candidate.strategy_ids
        and "partial_order_scheduling" in item.symbolic_candidate.strategy_ids
        and "constraint_solving" in item.symbolic_candidate.strategy_ids
        and "expected_information_gain" in item.symbolic_candidate.strategy_ids
        for item in first.snapshots
    )
    assert first.to_json() == second.to_json()


def test_non_compensable_safety_failure_rejects_the_baseline() -> None:
    graph, goal, context = _fixture()
    context["unsafe_task_ids"] = ["task:direct"]

    portfolio = SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=2, max_model_candidates=0)
    ).plan(graph, goal, context, allow_model=False)

    assert portfolio.baseline.disposition == "rejected"
    assert any(
        "hard_gate_failed:conflict_scope_and_authority" in reason
        for reason in portfolio.baseline.reason_codes
    )
    assert portfolio.selected is not None
    assert portfolio.selected.symbolic_candidate.task_candidate_ids == (
        "task:layered",
    )


def test_model_proposal_uses_same_frozen_request_and_aggregate_count() -> None:
    graph, goal, context = _fixture()
    seen = []

    def provider(request):
        seen.append(request)
        return {
            "candidates": [
                {
                    "request_id": request.request_id,
                    "task_candidate_ids": ["task:layered"],
                }
            ],
            "input_tokens": 32,
            "output_tokens": 16,
        }

    portfolio = SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=2, max_model_candidates=1)
    ).plan(graph, goal, context, model_provider=provider)

    assert seen == [portfolio.request]
    assert len(portfolio.snapshots) == 2
    assert portfolio.provider_usage.status is SymbolicProviderStatus.SUCCEEDED
    assert portfolio.provider_usage.request_id == portfolio.request.request_id
    assert portfolio.provider_usage.usage_id.startswith("bagu")
    assert portfolio.snapshots[1].symbolic_candidate.source is (
        SymbolicCandidateSource.MODEL_PROPOSAL
    )


def test_portfolio_round_trip_recomputes_every_content_identity() -> None:
    graph, goal, context = _fixture()
    portfolio = SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=2, max_model_candidates=0)
    ).plan(graph, goal, context, allow_model=False)

    restored = SymbolicCandidatePortfolio.from_json(portfolio.to_json())
    assert restored == portfolio
    assert restored.portfolio_id == portfolio.portfolio_id

    tampered = copy.deepcopy(portfolio.to_dict())
    tampered["snapshots"][0]["symbolic_candidate"][
        "expected_information_gain_millionths"
    ] += 1
    with pytest.raises(ValueError, match="identity|projection|inconsistent"):
        SymbolicCandidatePortfolio.from_dict(tampered)


def test_bounds_are_fail_closed_and_count_one_keeps_baseline_only() -> None:
    graph, goal, context = _fixture()
    with pytest.raises(ValueError, match="candidate_count"):
        SymbolicCandidateBounds(candidate_count=0)
    with pytest.raises(ValueError, match="candidate_count"):
        SymbolicCandidateBounds(candidate_count=33)
    with pytest.raises(ValueError, match="candidate_count"):
        SymbolicCandidateBounds(candidate_count=True)

    portfolio = SymbolicCandidatePlanner(
        bounds=SymbolicCandidateBounds(candidate_count=1)
    ).plan(
        graph,
        goal,
        context,
        model_provider=lambda _request: pytest.fail("provider must not be called"),
    )
    assert len(portfolio.snapshots) == 1
    assert portfolio.baseline.symbolic_candidate.source is (
        SymbolicCandidateSource.DETERMINISTIC_BASELINE
    )


def test_adaptive_planner_bridge_uses_the_symbolic_portfolio() -> None:
    graph, goal, context = _fixture()

    portfolio = AdaptivePlanner(max_candidates=2).plan(
        goal,
        context,
        obligation_graph=graph,
        symbolic_bounds=SymbolicCandidateBounds(
            candidate_count=2,
            max_model_candidates=0,
        ),
        allow_model=False,
    )

    assert isinstance(portfolio, SymbolicCandidatePortfolio)
    assert len(portfolio.snapshots) == 2
    assert portfolio.baseline.symbolic_candidate.source is (
        SymbolicCandidateSource.DETERMINISTIC_BASELINE
    )


def test_blocked_or_pathless_graph_never_invents_repository_wide_work() -> None:
    graph, goal, context = _fixture()
    pathless = {
        "task_metadata": {},
        "protected_paths": ["src/direct.py", "src/layered.py"],
    }
    with pytest.raises(
        SymbolicCandidatePlanningError, match="codebase-derived"
    ):
        SymbolicCandidatePlanner().plan(
            graph, goal, pathless, allow_model=False
        )

    uncovered = compile_obligation_graph(
        TypedIntent(
            intent_id="intent:uncovered",
            desired_predicates=(_predicate("uncovered"),),
            source_refs=("intent-source:uncovered",),
            current_root_id="tree:test",
        )
    )
    with pytest.raises(
        SymbolicCandidatePlanningError, match="blocked obligation graph"
    ):
        SymbolicCandidatePlanner().plan(
            uncovered, goal, context, allow_model=False
        )
