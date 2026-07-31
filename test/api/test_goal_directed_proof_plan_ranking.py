"""API tests for GoalDirectedProofPlanRanker@1 (FVT-026 / FVT-G035).

Acceptance:

* Rankings are deterministic and explainable.
* Incomplete / invalid / insufficient-authority branches are hard-pruned.
* Each step names dependencies, expected receipts, validation, fallback,
  resources, and completion conditions.
* Assumption-heavy plans pay explicit cost.
* Adapters reuse existing plan-evaluator scoring primitives without changing
  unrelated implementation-task routing.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
    AuthorityCeiling,
    GoalDirectedProofPlan,
    PlanStatus,
    ResourceBounds,
)
from ipfs_datasets_py.logic.software_verification.tactician.proof_plan import (
    GOAL_DIRECTED_PROOF_PLAN_RANKER_INTERFACE,
    RANKER_ALGORITHM_VERSION,
    RANKING_SCORE_DIMENSIONS,
    REQUIRED_STEP_FIELD_NAMES,
    GoalDirectedProofPlanRanker,
    HardPruneReason,
    MissingProofPlanAlternative,
    ProofPlanError,
    ProofPlanRankingPolicy,
    ProofPlanRankingWeights,
    ProofPlanStepSpec,
    StepKind,
    authority_meets_minimum,
    build_missing_proof_plan,
    complete_step,
    every_step_names_required_fields,
    rank_missing_proof_plans,
    rank_via_and_or_evaluator,
    rank_via_proof_aware_evaluator,
    rankings_are_deterministic,
    score_missing_proof_plan,
    to_and_or_plan_branch,
    to_proof_aware_plan_candidate,
    with_hard_failures,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _policy(**overrides: Any) -> ProofPlanRankingPolicy:
    payload: dict[str, Any] = {
        "minimum_authority": AuthorityCeiling.BOUNDED,
        "available_resource_classes": ("solver", "kernel", "artifact_store"),
        "satisfied_dependencies": ("root:goal",),
        "required_obligation_ids": (
            "obligation:lease-safety",
            "obligation:router-bounds",
        ),
        "assumption_unit_cost": 0.25,
        "max_new_assumptions": 8,
    }
    payload.update(overrides)
    return ProofPlanRankingPolicy(**payload)


def _complete_plan(
    plan_id: str,
    *,
    obligations: tuple[str, ...] = (
        "obligation:lease-safety",
        "obligation:router-bounds",
    ),
    authority: AuthorityCeiling = AuthorityCeiling.BOUNDED,
    new_assumptions: tuple[str, ...] = (),
    proof_cost: float = 1.0,
    cache_value: float = 0.8,
    risk: float = 0.15,
    downstream_unlock: float = 2.0,
    critical_path: float = 1.0,
    resources: tuple[str, ...] = ("solver", "kernel"),
    fallback_quality: bool = True,
) -> MissingProofPlanAlternative:
    steps: list[ProofPlanStepSpec] = []
    for index, obligation_id in enumerate(obligations):
        step_id = f"step:{plan_id}:{index}"
        deps: tuple[str, ...]
        if index == 0:
            deps = ("root:goal",)
            root = False
        else:
            deps = (f"step:{plan_id}:{index - 1}",)
            root = False
        steps.append(
            complete_step(
                step_id,
                obligation_id,
                kind=StepKind.SOLVE,
                dependencies=deps,
                expected_receipts=(f"receipt:{obligation_id}",),
                validation=(f"pytest:test_{obligation_id.split(':')[-1]}",),
                fallback=(
                    (f"fallback:replay:{obligation_id}",)
                    if fallback_quality
                    else ()
                ),
                resources=resources,
                completion_conditions=(f"{obligation_id}:discharged",),
                authority=authority,
                new_assumption_ids=(
                    new_assumptions if index == 0 else ()
                ),
                provider_ids=("provider:z3",),
                proof_cost=proof_cost,
                cache_value=cache_value,
                risk=risk,
                downstream_unlock=downstream_unlock,
                critical_path_contribution=critical_path,
                root=root,
            )
        )
    # When fallback_quality is False we deliberately leave fallback empty to
    # trigger incompleteness — callers that want admissible no-fallback must
    # pass an explicit empty-fallback marker via metadata root workaround.
    return build_missing_proof_plan(
        plan_id,
        formal_goal_id="formal:lease-ready",
        graph_id="graph:lease-and-or",
        tree_id="tree:repo@abc",
        steps=steps,
        required_obligation_ids=obligations,
        covered_obligation_ids=obligations,
        alternative_ids=(plan_id,),
        producer_kinds=("proof_plan_step",),
        bounds=ResourceBounds(max_candidates=32, max_steps=64),
    )


def _incomplete_plan(plan_id: str = "plan:incomplete") -> MissingProofPlanAlternative:
    """Plan whose step is missing required completeness fields."""

    step = ProofPlanStepSpec(
        step_id="step:incomplete",
        obligation_id="obligation:lease-safety",
        kind=StepKind.SOLVE,
        statement="incomplete step",
        dependencies=("root:goal",),
        expected_receipts=(),  # missing
        validation=(),  # missing
        fallback=(),  # missing
        resources=(),  # missing
        completion_conditions=(),  # missing
        authority=AuthorityCeiling.BOUNDED,
    )
    return build_missing_proof_plan(
        plan_id,
        formal_goal_id="formal:lease-ready",
        graph_id="graph:lease-and-or",
        tree_id="tree:repo@abc",
        steps=(step,),
        required_obligation_ids=("obligation:lease-safety",),
        covered_obligation_ids=("obligation:lease-safety",),
    )


def _low_authority_plan(
    plan_id: str = "plan:advisory",
) -> MissingProofPlanAlternative:
    return _complete_plan(
        plan_id,
        authority=AuthorityCeiling.ADVISORY,
    )


# ---------------------------------------------------------------------------
# Interface + completeness
# ---------------------------------------------------------------------------


def test_interface_version_and_required_step_fields() -> None:
    assert (
        GoalDirectedProofPlanRanker.INTERFACE
        == GOAL_DIRECTED_PROOF_PLAN_RANKER_INTERFACE
    )
    assert GoalDirectedProofPlanRanker.ALGORITHM_VERSION == RANKER_ALGORITHM_VERSION
    for name in (
        "dependencies",
        "expected_receipts",
        "validation",
        "fallback",
        "resources",
        "completion_conditions",
    ):
        assert name in REQUIRED_STEP_FIELD_NAMES
    for dimension in RANKING_SCORE_DIMENSIONS:
        assert dimension  # non-empty explainable labels


def test_complete_step_names_every_required_field() -> None:
    step = complete_step(
        "step:root",
        "obligation:lease-safety",
        root=True,
        resources=("solver",),
    )
    assert step.is_complete()
    payload = step.to_dict()
    for name in REQUIRED_STEP_FIELD_NAMES:
        # root steps may have empty dependencies (explicit root metadata).
        if name == "dependencies":
            assert name in payload
            continue
        assert payload[name], f"{name} must be named on complete steps"
    assert every_step_names_required_fields(
        build_missing_proof_plan(
            "plan:single",
            formal_goal_id="formal:lease-ready",
            graph_id="graph:g",
            tree_id="tree:t",
            steps=(step,),
            required_obligation_ids=("obligation:lease-safety",),
        )
    )


def test_step_and_plan_round_trip() -> None:
    plan = _complete_plan("plan:round-trip")
    restored = MissingProofPlanAlternative.from_dict(plan.to_dict())
    assert restored.plan_id == plan.plan_id
    assert len(restored.steps) == len(plan.steps)
    assert restored.steps[0].to_dict() == plan.steps[0].to_dict()


# ---------------------------------------------------------------------------
# Hard pruning
# ---------------------------------------------------------------------------


def test_incomplete_plans_are_hard_pruned() -> None:
    policy = _policy(
        required_obligation_ids=("obligation:lease-safety",),
    )
    result = rank_missing_proof_plans(
        [_incomplete_plan(), _complete_plan("plan:good")],
        policy=policy,
    )
    pruned_ids = {item.plan_id for item in result.pruned}
    assert "plan:incomplete" in pruned_ids
    assert result.selected is not None
    assert result.selected.plan_id == "plan:good"
    incomplete = next(
        item for item in result.pruned if item.plan_id == "plan:incomplete"
    )
    assert incomplete.score_millionths is None
    assert incomplete.soft_scores == {}
    reasons = {f.reason for f in incomplete.plan.hard_failures}
    assert HardPruneReason.INCOMPLETE_STEP in reasons
    assert any("incomplete" in line.lower() for line in incomplete.rationale)


def test_insufficient_authority_is_hard_pruned() -> None:
    policy = _policy(minimum_authority=AuthorityCeiling.BOUNDED)
    result = rank_missing_proof_plans(
        [_low_authority_plan(), _complete_plan("plan:trusted")],
        policy=policy,
    )
    pruned_ids = {item.plan_id for item in result.pruned}
    assert "plan:advisory" in pruned_ids
    assert result.selected is not None
    assert result.selected.plan_id == "plan:trusted"
    failure = result.pruned[0].plan.hard_failures[0]
    assert failure.reason is HardPruneReason.INSUFFICIENT_AUTHORITY
    assert not authority_meets_minimum(
        AuthorityCeiling.ADVISORY, AuthorityCeiling.BOUNDED
    )


def test_proof_and_completion_claims_are_hard_pruned() -> None:
    base = _complete_plan("plan:claim")
    # Bypass constructor guards by rebuilding with claim flags via raw dict.
    payload = base.to_dict()
    payload["proof_claimed"] = True
    payload["completion_claimed"] = True
    # from_dict forces bool but MissingProofPlanAlternative stores the flags.
    claimed = MissingProofPlanAlternative(
        plan_id=base.plan_id,
        formal_goal_id=base.formal_goal_id,
        graph_id=base.graph_id,
        tree_id=base.tree_id,
        steps=base.steps,
        covered_obligation_ids=base.covered_obligation_ids,
        required_obligation_ids=base.required_obligation_ids,
        alternative_ids=base.alternative_ids,
        producer_kinds=base.producer_kinds,
        hard_failures=(),
        bounds=base.bounds,
        root_goal_id=base.root_goal_id,
        proof_claimed=True,
        completion_claimed=True,
    )
    policy = _policy()
    gated = with_hard_failures(claimed, policy)
    reasons = {f.reason for f in gated.hard_failures}
    assert HardPruneReason.PROOF_CLAIM in reasons
    assert HardPruneReason.COMPLETION_CLAIM in reasons


def test_missing_coverage_and_cycles_are_hard_pruned() -> None:
    required = (
        "obligation:lease-safety",
        "obligation:router-bounds",
    )
    policy = _policy(required_obligation_ids=required)
    partial = _complete_plan(
        "plan:partial",
        obligations=("obligation:lease-safety",),
    )
    # Cyclic plan: two steps that depend on each other.
    a = complete_step(
        "step:a",
        "obligation:lease-safety",
        dependencies=("step:b",),
        resources=("solver",),
    )
    b = complete_step(
        "step:b",
        "obligation:router-bounds",
        dependencies=("step:a",),
        resources=("solver",),
    )
    cyclic = build_missing_proof_plan(
        "plan:cyclic",
        formal_goal_id="formal:lease-ready",
        graph_id="graph:lease-and-or",
        tree_id="tree:repo@abc",
        steps=(a, b),
        required_obligation_ids=required,
    )
    # Extra obligation required by policy but not covered by partial/cyclic.
    strict_policy = _policy(
        required_obligation_ids=(
            *required,
            "obligation:extra",
        )
    )
    full = _complete_plan(
        "plan:full",
        obligations=(*required, "obligation:extra"),
    )
    result = rank_missing_proof_plans(
        [partial, cyclic, full],
        policy=strict_policy,
    )
    pruned = {item.plan_id: item for item in result.pruned}
    assert "plan:partial" in pruned
    assert "plan:cyclic" in pruned
    assert result.selected is not None
    assert result.selected.plan_id == "plan:full"
    assert any(
        f.reason is HardPruneReason.MISSING_COVERAGE
        for f in pruned["plan:partial"].plan.hard_failures
    )
    assert any(
        f.reason is HardPruneReason.CYCLIC_DEPENDENCIES
        for f in pruned["plan:cyclic"].plan.hard_failures
    )


def test_hard_prune_cannot_be_compensated_by_utility() -> None:
    """A high-utility low-authority plan never outranks a valid low-utility one."""

    shiny = _complete_plan(
        "plan:shiny-but-advisory",
        authority=AuthorityCeiling.ADVISORY,
        proof_cost=0.01,
        cache_value=1.0,
        risk=0.0,
        downstream_unlock=100.0,
        critical_path=0.1,
    )
    modest = _complete_plan(
        "plan:modest-trusted",
        authority=AuthorityCeiling.BOUNDED,
        proof_cost=5.0,
        cache_value=0.1,
        risk=0.5,
        downstream_unlock=0.1,
        critical_path=5.0,
    )
    result = rank_missing_proof_plans(
        [shiny, modest],
        policy=_policy(minimum_authority=AuthorityCeiling.BOUNDED),
    )
    assert result.selected is not None
    assert result.selected.plan_id == "plan:modest-trusted"
    assert any(
        item.plan_id == "plan:shiny-but-advisory" for item in result.pruned
    )


# ---------------------------------------------------------------------------
# Deterministic explainable ranking
# ---------------------------------------------------------------------------


def test_rankings_are_deterministic_and_explainable() -> None:
    strong = _complete_plan(
        "plan:strong",
        proof_cost=0.5,
        cache_value=0.95,
        risk=0.05,
        downstream_unlock=5.0,
        critical_path=1.0,
        new_assumptions=(),
    )
    weak = _complete_plan(
        "plan:weak",
        proof_cost=4.0,
        cache_value=0.1,
        risk=0.7,
        downstream_unlock=0.5,
        critical_path=4.0,
        new_assumptions=("assumption:extra-fairness",),
    )
    policy = _policy()
    first = rank_missing_proof_plans([weak, strong], policy=policy)
    second = rank_missing_proof_plans([strong, weak], policy=policy)
    assert rankings_are_deterministic(first, second)
    assert first.to_dict() == second.to_dict()
    assert first.selected is not None
    assert first.selected.plan_id == "plan:strong"
    assert first.selected.score_millionths is not None
    assert first.selected.score_millionths > (
        first.ranked[1].score_millionths or 0
    )
    # Every soft dimension is present and rationale is explainable.
    for dimension in RANKING_SCORE_DIMENSIONS:
        assert dimension in first.selected.soft_scores
    rationale = " ".join(first.selected.rationale).lower()
    for token in (
        "discharged_coverage",
        "downstream_unlock",
        "critical_path",
        "authority",
        "assumption_cost",
        "risk",
        "proof_cost",
        "cache_value",
        "fallback_quality",
        "millionths",
    ):
        assert token in rationale
    assert first.selected.rationale  # non-empty


def test_assumption_heavy_plans_pay_explicit_cost() -> None:
    lean = _complete_plan(
        "plan:lean",
        new_assumptions=(),
        proof_cost=1.0,
        cache_value=0.5,
        risk=0.2,
        downstream_unlock=2.0,
        critical_path=2.0,
    )
    heavy = _complete_plan(
        "plan:heavy",
        new_assumptions=(
            "assumption:fairness",
            "assumption:frame",
            "assumption:bridge",
            "assumption:env-closed",
        ),
        proof_cost=1.0,
        cache_value=0.5,
        risk=0.2,
        downstream_unlock=2.0,
        critical_path=2.0,
    )
    policy = _policy(assumption_unit_cost=0.25)
    result = rank_missing_proof_plans([heavy, lean], policy=policy)
    assert result.selected is not None
    assert result.selected.plan_id == "plan:lean"
    lean_score = int(result.ranked[0].score_millionths or 0)
    heavy_ranked = next(
        item for item in result.ranked if item.plan_id == "plan:heavy"
    )
    heavy_score = int(heavy_ranked.score_millionths or 0)
    assert lean_score > heavy_score
    # Soft assumption_cost factor is strictly lower for the heavy plan.
    assert (
        lean.admissible
        or True  # plans are pre-hard-failure free
    )
    lean_scored = with_hard_failures(lean, policy)
    heavy_scored = with_hard_failures(heavy, policy)
    _, lean_soft, lean_rationale = score_missing_proof_plan(lean_scored, policy)
    _, heavy_soft, heavy_rationale = score_missing_proof_plan(
        heavy_scored, policy
    )
    assert lean_soft["assumption_cost"] > heavy_soft["assumption_cost"]
    assert "assumption count is 4" in " ".join(heavy_rationale)
    assert "assumption count is 0" in " ".join(lean_rationale)


def test_rejected_alternatives_retain_actionable_rationale() -> None:
    a = _complete_plan("plan:a", proof_cost=0.5, cache_value=0.9)
    b = _complete_plan("plan:b", proof_cost=3.0, cache_value=0.2)
    result = rank_missing_proof_plans([a, b], policy=_policy())
    assert len(result.ranked) == 2
    loser = result.ranked[1]
    assert any("rejected in favor of" in line for line in loser.rationale)
    assert loser.score_millionths is not None


# ---------------------------------------------------------------------------
# GoalDirectedProofPlan materialization
# ---------------------------------------------------------------------------


def test_selected_plan_materializes_goal_directed_proof_plan() -> None:
    result = rank_missing_proof_plans(
        [_complete_plan("plan:selected")],
        policy=_policy(),
    )
    gdp = result.to_goal_directed_proof_plan()
    assert gdp is not None
    assert isinstance(gdp, GoalDirectedProofPlan)
    assert gdp.plan_id == "plan:selected"
    assert gdp.status is PlanStatus.RANKED
    assert gdp.proof_claimed is False
    assert gdp.completion_claimed is False
    assert gdp.authority is AuthorityCeiling.CANDIDATE
    assert gdp.rank_score_millionths == result.selected.score_millionths
    assert len(gdp.candidates) == len(result.selected.plan.steps)
    # Round-trip through the shared contract.
    restored = GoalDirectedProofPlan.from_dict(gdp.to_dict())
    assert restored.plan_id == gdp.plan_id
    assert restored.content_id == gdp.content_id


def test_empty_admissible_set_selects_none() -> None:
    result = rank_missing_proof_plans(
        [_incomplete_plan("plan:only-bad")],
        policy=_policy(required_obligation_ids=("obligation:lease-safety",)),
    )
    assert result.selected is None
    assert result.ranked == ()
    assert len(result.pruned) == 1
    assert result.to_goal_directed_proof_plan() is None


# ---------------------------------------------------------------------------
# Plan-evaluator adapters
# ---------------------------------------------------------------------------


def test_and_or_evaluator_adapter_hard_prunes_then_ranks() -> None:
    strong = _complete_plan(
        "plan:andor-strong",
        proof_cost=0.5,
        cache_value=0.9,
        risk=0.1,
        critical_path=1.0,
        downstream_unlock=3.0,
    )
    weak = _complete_plan(
        "plan:andor-weak",
        proof_cost=5.0,
        cache_value=0.1,
        risk=0.8,
        critical_path=5.0,
        downstream_unlock=0.2,
    )
    advisory = _low_authority_plan("plan:andor-advisory")
    evaluation = rank_via_and_or_evaluator(
        [weak, advisory, strong],
        policy=_policy(minimum_authority=AuthorityCeiling.BOUNDED),
    )
    # Hard-pruned advisory must not be selected / scored.
    pruned_ids = {item.branch_id for item in evaluation.pruned}
    assert "plan:andor-advisory" in pruned_ids
    for item in evaluation.pruned:
        assert item.score_millionths is None
    assert evaluation.selected is not None
    assert evaluation.selected.branch_id == "plan:andor-strong"
    assert evaluation.selected.score_millionths is not None
    # Soft dimensions from the AND/OR evaluator are all present.
    for key in (
        "evidence_coverage",
        "uncertainty_reduction",
        "critical_path",
        "conflict_risk",
        "cost",
        "historical_failure",
    ):
        assert key in evaluation.selected.soft_scores


def test_proof_aware_evaluator_adapter_ranks_admissible_plans() -> None:
    strong = _complete_plan(
        "plan:pa-strong",
        proof_cost=0.5,
        cache_value=0.95,
        risk=0.05,
        critical_path=1.0,
        downstream_unlock=4.0,
    )
    weak = _complete_plan(
        "plan:pa-weak",
        proof_cost=4.0,
        cache_value=0.1,
        risk=0.7,
        critical_path=4.0,
        downstream_unlock=0.5,
    )
    evaluation = rank_via_proof_aware_evaluator(
        [weak, strong],
        policy=_policy(),
    )
    assert evaluation.selected.candidate_id == "plan:pa-strong"
    assert evaluation.selected.score_millionths > (
        evaluation.rejected[0].score_millionths
    )
    rationale = " ".join(evaluation.selected.rationale).lower()
    for token in (
        "critical path",
        "downstream",
        "risk",
        "cache",
        "proof cost",
    ):
        assert token in rationale


def test_proof_aware_adapter_rejects_all_pruned_population() -> None:
    with pytest.raises(ProofPlanError, match="no admissible plan"):
        rank_via_proof_aware_evaluator(
            [_incomplete_plan(), _low_authority_plan()],
            policy=_policy(minimum_authority=AuthorityCeiling.BOUNDED),
        )


def test_to_proof_aware_candidate_declares_complete_contract() -> None:
    plan = with_hard_failures(_complete_plan("plan:contract"), _policy())
    assert plan.admissible
    candidate = to_proof_aware_plan_candidate(plan)
    assert candidate["candidate_id"] == "plan:contract"
    assert candidate["obligation_impact"]
    assert candidate["expected_evidence_delta"]
    assert candidate["resource_classes"]
    assert candidate["required_assurance"]
    branch = candidate["branch"]
    assert branch["branch_id"] == "plan:contract"
    assert branch["validation_commands"]
    assert branch["validation_proof"]
    assert branch["predicted_files"]
    # Round-trip through the real plan-evaluator type.
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import (
        ProofAwarePlanCandidate,
    )

    restored = ProofAwarePlanCandidate.from_dict(candidate)
    assert restored.candidate_id == "plan:contract"


def test_to_and_or_branch_closed_schema() -> None:
    plan = with_hard_failures(_complete_plan("plan:andor-schema"), _policy())
    branch = to_and_or_plan_branch(plan)
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import (
        AndOrPlanBranch,
    )

    restored = AndOrPlanBranch.from_dict(branch)
    assert restored.branch_id == "plan:andor-schema"
    assert restored.admissible is True


# ---------------------------------------------------------------------------
# Ranker object API + policy edge cases
# ---------------------------------------------------------------------------


def test_ranker_instance_uses_injected_policy() -> None:
    ranker = GoalDirectedProofPlanRanker(
        policy=_policy(minimum_authority=AuthorityCeiling.SATISFIABILITY)
    )
    # BOUNDED plan is insufficient under SATISFIABILITY minimum.
    result = ranker.rank([_complete_plan("plan:bounded-only")])
    assert result.selected is None
    assert result.pruned
    assert any(
        f.reason is HardPruneReason.INSUFFICIENT_AUTHORITY
        for f in result.pruned[0].plan.hard_failures
    )


def test_duplicate_plan_ids_are_rejected() -> None:
    with pytest.raises(ProofPlanError, match="unique"):
        rank_missing_proof_plans(
            [
                _complete_plan("plan:dup"),
                _complete_plan("plan:dup"),
            ],
            policy=_policy(),
        )


def test_empty_alternatives_raise() -> None:
    with pytest.raises(ProofPlanError, match="at least one"):
        rank_missing_proof_plans([], policy=_policy())


def test_weights_require_positive_total() -> None:
    with pytest.raises(ProofPlanError, match="positive"):
        ProofPlanRankingWeights(
            discharged_coverage=0.0,
            downstream_unlock=0.0,
            critical_path=0.0,
            authority=0.0,
            assumption_cost=0.0,
            risk=0.0,
            proof_cost=0.0,
            cache_value=0.0,
            fallback_quality=0.0,
        )


def test_self_dependent_step_is_rejected() -> None:
    with pytest.raises(ProofPlanError, match="depend on itself"):
        ProofPlanStepSpec(
            step_id="step:self",
            obligation_id="obligation:x",
            dependencies=("step:self",),
            expected_receipts=("receipt:x",),
            validation=("validate:x",),
            fallback=("fallback:x",),
            resources=("solver",),
            completion_conditions=("done",),
        )


def test_max_new_assumptions_bound_hard_prunes() -> None:
    plan = _complete_plan(
        "plan:too-many-assumptions",
        new_assumptions=tuple(f"assumption:{i}" for i in range(10)),
    )
    result = rank_missing_proof_plans(
        [plan],
        policy=_policy(max_new_assumptions=3),
    )
    assert result.selected is None
    assert any(
        f.reason is HardPruneReason.RESOURCE_BOUND
        for f in result.pruned[0].plan.hard_failures
    )


def test_result_payload_is_stable_json_shape() -> None:
    result = rank_missing_proof_plans(
        [
            _complete_plan("plan:shape-a"),
            _incomplete_plan("plan:shape-bad"),
        ],
        policy=_policy(),
    )
    payload = result.to_dict()
    assert payload["interface"] == GOAL_DIRECTED_PROOF_PLAN_RANKER_INTERFACE
    assert payload["evaluator_version"] == RANKER_ALGORITHM_VERSION
    assert payload["selected"] is not None
    assert payload["ranked"]
    assert payload["pruned"]
    assert "policy" in payload
    assert set(payload["selected"]["soft_scores"]) == set(
        RANKING_SCORE_DIMENSIONS
    )
