"""Bounded counterexample-guided tactic refinement (LPR-013)."""

from __future__ import annotations

import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    GoalDisposition,
    GoalFamily,
    HypothesisDisposition,
    LogicFacetKind,
    LogicFacetRef,
    LogicGap,
    GapDisposition,
    GapMissingClass,
    LogicHypothesis,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_prediction_cegis import (
    LOGIC_PREDICTION_CEGIS_INTERFACE,
    LogicPredictionCEGIS,
    LogicPredictionCegisAuthorityError,
    LogicPredictionCegisBoundsError,
    LogicPredictionCegisMonotonicityError,
    LogicRefinementBounds,
    LogicRefinementReceipt,
    LogicRefinementState,
    RefinementActionKind,
    RefinementDisposition,
    RefinementEvidence,
    RefinementStopReason,
    RoundDisposition,
    SubgoalRefinementProof,
    create_logic_prediction_cegis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:lpr-013",
        objective_id="objective:lpr-013",
        trace_id="trace:lpr-013",
        change_id="change:lpr-013",
        consumer_id="consumer:lpr-013",
        forest_id="forest:one",
        tree_id="tree:one",
        overlay_id="overlay:one",
        graph_id="graph:one",
        index_id="index:one",
        corpus_id="corpus:one",
        model_id="model:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
        environment_id="environment:one",
    )


def _facet(
    facet_id: str = "facet:type",
    kind: LogicFacetKind = LogicFacetKind.TYPE,
) -> LogicFacetRef:
    return LogicFacetRef(
        facet_id=facet_id,
        kind=kind,
        subject_symbol_id="symbol:process",
        contract_ref=f"contract:{facet_id}",
    )


def _goal(
    roots: ProgramLogicAuthorityRoots,
    goal_id: str = "goal:one",
    *,
    facets: tuple[LogicFacetRef, ...] | None = None,
) -> ProgramLogicGoal:
    return ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=GoalFamily.BEHAVIOR,
        disposition=GoalDisposition.OPEN,
        positive_statement_ref=f"stmt:{goal_id}",
        required_facets=facets if facets is not None else (_facet(), _facet("facet:effect", LogicFacetKind.EFFECT)),
        invalidation_refs=(roots.tree_id,),
    )


def _hypothesis(
    roots: ProgramLogicAuthorityRoots,
    hypothesis_id: str = "hyp:one",
    *,
    target_goal_id: str = "goal:one",
) -> LogicHypothesis:
    return LogicHypothesis(
        roots=roots,
        hypothesis_id=hypothesis_id,
        target_goal_id=target_goal_id,
        disposition=HypothesisDisposition.NOMINATED,
        claimed_consequence_ref=f"consequence:{hypothesis_id}",
        evidence_route_kinds=(SourceRouteKind.LOCAL_STATIC,),
        selected_premise_ids=("premise:a",),
        source_authority=SourceAuthorityClass.NOMINATING,
        proof_status=ProofStatus.UNPROVED,
        invalidation_refs=(roots.tree_id,),
    )


def _plan(
    roots: ProgramLogicAuthorityRoots,
    *,
    goal_ids: tuple[str, ...] = ("goal:one",),
    selected: tuple[str, ...] = ("premise:a", "premise:b"),
) -> TacticianSearchPlan:
    return TacticianSearchPlan(
        roots=roots,
        plan_id="plan:tactician-one",
        goal_ids=goal_ids,
        ordered_source_routes=(
            SourceRouteKind.LOCAL_STATIC,
            SourceRouteKind.REVIEWED_CONTRACT,
        ),
        selected_premise_ids=selected,
        invalidation_refs=(roots.tree_id,),
    )


def _validated_countermodel(
    roots: ProgramLogicAuthorityRoots,
    *,
    receipt_id: str = "cm:validated",
    via_negation: bool = False,
) -> CountermodelValidationReceipt:
    if via_negation:
        return CountermodelValidationReceipt(
            roots=roots,
            receipt_id=receipt_id,
            solver_countermodel_id="solver-cm:raw-1",
            translation_map_id="translation:one",
            originating_logic_ir_id="obligation:logic-ir",
            disposition=CountermodelDisposition.VALIDATED,
            proof_of_negation_id="kernel-proof:negation-1",
            invalidation_refs=(roots.tree_id,),
        )
    return CountermodelValidationReceipt(
        roots=roots,
        receipt_id=receipt_id,
        solver_countermodel_id="solver-cm:raw-1",
        translation_map_id="translation:one",
        originating_logic_ir_id="obligation:logic-ir",
        disposition=CountermodelDisposition.VALIDATED,
        raw_diagnostic_refs=("diag:solver-model",),
        replayed_rejection_evidence_refs=("replay:logic-ir",),
        replay_method="deterministic_logic_ir_replay",
        invalidation_refs=(roots.tree_id,),
    )


def _diagnostic_countermodel(
    roots: ProgramLogicAuthorityRoots,
    *,
    receipt_id: str = "cm:diag",
) -> CountermodelValidationReceipt:
    return CountermodelValidationReceipt(
        roots=roots,
        receipt_id=receipt_id,
        solver_countermodel_id="solver-cm:raw-2",
        translation_map_id="translation:one",
        originating_logic_ir_id="obligation:logic-ir",
        disposition=CountermodelDisposition.DIAGNOSTIC_ONLY,
        raw_diagnostic_refs=("diag:solver-model", "diag:assignment"),
        invalidation_refs=(roots.tree_id,),
    )


def _engine(**bound_overrides) -> LogicPredictionCEGIS:
    bounds = LogicRefinementBounds(**bound_overrides) if bound_overrides else LogicRefinementBounds()
    return LogicPredictionCEGIS(bounds=bounds)


# ---------------------------------------------------------------------------
# Bounds policy
# ---------------------------------------------------------------------------


def test_bounds_are_policy_fields_with_hard_ceilings() -> None:
    bounds = LogicRefinementBounds(
        max_rounds=4,
        max_goals=8,
        max_subgoals=16,
        max_premises=16,
        max_counterexamples=4,
        wall_time_ms=5_000,
        cpu_time_ms=2_000,
        memory_bytes=64 * 1024 * 1024,
        max_context_bytes=8_192,
    )
    assert bounds.to_dict()["max_rounds"] == 4
    tightened = bounds.tighten(LogicRefinementBounds(max_rounds=2, max_goals=100))
    assert tightened.max_rounds == 2
    assert tightened.max_goals == 8
    with pytest.raises(LogicPredictionCegisBoundsError, match="hard ceiling"):
        LogicRefinementBounds(max_rounds=10_000)


# ---------------------------------------------------------------------------
# Initial state / monotonic originals
# ---------------------------------------------------------------------------


def test_initial_state_preserves_goals_and_facets(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    goal = _goal(roots)
    hyp = _hypothesis(roots)
    plan = _plan(roots)
    state = engine.initial_state(
        roots=roots,
        goals=[goal],
        hypotheses=[hyp],
        plan=plan,
        authorized_premise_ids=("premise:a", "premise:b", "premise:c"),
    )
    assert state.original_goal_ids == ("goal:one",)
    assert "facet:type" in state.original_facet_ids
    assert "facet:effect" in state.original_facet_ids
    assert state.active_hypothesis_ids == ("hyp:one",)
    assert state.tactician_plan_id == "plan:tactician-one"
    assert state.state_id
    # Forged state_id rejected.
    with pytest.raises(LogicPredictionCegisAuthorityError, match="forged"):
        LogicRefinementState(
            roots=roots,
            original_goal_ids=("goal:one",),
            original_facet_ids=state.original_facet_ids,
            active_goal_ids=("goal:one",),
            active_hypothesis_ids=("hyp:one",),
            state_id="forged:identity",
        )


def test_cannot_drop_original_goal_from_state(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(LogicPredictionCegisMonotonicityError, match="delete or drop"):
        LogicRefinementState(
            roots=roots,
            original_goal_ids=("goal:one", "goal:two"),
            original_facet_ids=(),
            active_goal_ids=("goal:one",),  # dropped goal:two
            active_hypothesis_ids=(),
        )


# ---------------------------------------------------------------------------
# Raw diagnostic cannot reject; validated can
# ---------------------------------------------------------------------------


def test_raw_solver_countermodel_guides_diagnostic_only(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a", "premise:b"),
        selected_premise_ids=("premise:a",),
    )
    evidence = RefinementEvidence(
        raw_solver_countermodel_ids=("solver-cm:raw-unvalidated",),
        hypothesis_narrowings={
            # Even if a caller tries to bind rejections to raw ids, apply_round
            # only rejects via CountermodelValidationReceipt.
        },
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert "solver-cm:raw-unvalidated" in new_state.diagnostic_countermodel_ids
    assert new_state.active_hypothesis_ids == ("hyp:one",)
    assert not new_state.excluded_hypothesis_ids
    assert rnd.disposition is RoundDisposition.DIAGNOSTIC_ONLY
    assert any(
        a.kind is RefinementActionKind.DIAGNOSTIC_RETRIEVAL for a in rnd.actions
    )
    assert "countermodel_unvalidated" in rnd.reason_codes


def test_diagnostic_receipt_cannot_eliminate_hypothesis(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    diag = _diagnostic_countermodel(roots)
    evidence = RefinementEvidence(
        countermodel_receipts=(diag,),
        hypothesis_narrowings={
            diag.receipt_id: {"reject_hypothesis_ids": ["hyp:one"]},
        },
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert new_state.active_hypothesis_ids == ("hyp:one",)
    assert not new_state.excluded_hypothesis_ids
    assert diag.receipt_id in new_state.diagnostic_countermodel_ids
    assert diag.receipt_id not in new_state.validated_countermodel_ids
    assert "countermodel_unvalidated" in rnd.reason_codes
    # Explicit abstention on the refused rejection.
    abstentions = [
        a for a in rnd.actions if a.kind is RefinementActionKind.ABSTAIN
    ]
    assert abstentions
    assert abstentions[0].details.get("may_reject") is False


def test_validated_replay_receipt_may_reject_hypothesis(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots), _hypothesis(roots, "hyp:two")],
        authorized_premise_ids=("premise:a", "premise:b"),
        selected_premise_ids=("premise:a", "premise:b"),
    )
    validated = _validated_countermodel(roots)
    assert validated.may_reject_hypothesis is True
    evidence = RefinementEvidence(
        countermodel_receipts=(validated,),
        hypothesis_narrowings={
            validated.receipt_id: {
                "reject_hypothesis_ids": ["hyp:one"],
                "exclude_premise_ids": ["premise:b"],
            },
        },
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert "hyp:one" not in new_state.active_hypothesis_ids
    assert "hyp:one" in new_state.excluded_hypothesis_ids
    assert "hyp:two" in new_state.active_hypothesis_ids
    assert "premise:b" in new_state.excluded_premise_ids
    assert "premise:b" not in new_state.selected_premise_ids
    assert validated.receipt_id in new_state.validated_countermodel_ids
    assert rnd.disposition is RoundDisposition.APPLIED
    assert any(a.kind is RefinementActionKind.REJECT_HYPOTHESIS for a in rnd.actions)


def test_proof_of_negation_may_reject(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    validated = _validated_countermodel(roots, receipt_id="cm:neg", via_negation=True)
    evidence = RefinementEvidence(
        countermodel_receipts=(validated,),
        hypothesis_narrowings={
            validated.receipt_id: {"reject_hypothesis_ids": ["hyp:one"]},
        },
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert not new_state.active_hypothesis_ids
    assert "hyp:one" in new_state.excluded_hypothesis_ids
    reject = next(
        a for a in rnd.actions if a.kind is RefinementActionKind.REJECT_HYPOTHESIS
    )
    assert reject.details["proof_of_negation_id"] == "kernel-proof:negation-1"


# ---------------------------------------------------------------------------
# Residuals feed Tactician; goals preserved
# ---------------------------------------------------------------------------


def test_residuals_feed_tactician_and_preserve_goals(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    goal = _goal(roots)
    state = engine.initial_state(
        roots=roots,
        goals=[goal],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    gap = LogicGap(
        roots=roots,
        gap_id="gap:missing-premise",
        goal_id="goal:one",
        missing_class=GapMissingClass.PREMISE,
        disposition=GapDisposition.REQUIRED,
        observed_fact_ref="observed:none",
        required_fact_ref="required:premise-x",
        discrepancy_ref="disc:missing",
        invalidation_refs=(roots.tree_id,),
    )
    evidence = RefinementEvidence(
        residual_gaps=(gap,),
        query_hints=("route:local_static:premise-x",),
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert "gap:missing-premise" in new_state.residual_gap_ids
    assert new_state.original_goal_ids == state.original_goal_ids
    assert set(new_state.original_facet_ids) == set(state.original_facet_ids)
    assert "goal:one" in new_state.active_goal_ids
    assert rnd.residual_feedback is not None
    assert "gap:missing-premise" in rnd.residual_feedback["residual_gap_ids"]
    assert any(a.kind is RefinementActionKind.FEED_TACTICIAN for a in rnd.actions)

    feedback = engine.residual_feedback_for_tactician(new_state, stop_reason="residual")
    assert feedback.residual_gap_ids == new_state.residual_gap_ids
    assert feedback.to_dict()["schema"].endswith("tactician-residual-feedback@1")


# ---------------------------------------------------------------------------
# Subgoal refinement / no weaken / no model promotion / no unauthorized premise
# ---------------------------------------------------------------------------


def test_subgoal_conjunction_must_cover_original_facets(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    goal = _goal(roots)
    # Missing effect facet coverage.
    with pytest.raises(LogicPredictionCegisMonotonicityError, match="facet"):
        engine.prove_subgoal_refinement(
            parent_goal=goal,
            subgoals=[
                {
                    "subgoal_id": "subgoal:type-only",
                    "goal_id": "goal:one",
                    "covered_facet_ids": ["facet:type"],
                    "source_route": SourceRouteKind.LOCAL_STATIC.value,
                    "source_authority": SourceAuthorityClass.AUTHORITATIVE.value,
                }
            ],
        )

    proof = engine.prove_subgoal_refinement(
        parent_goal=goal,
        subgoals=[
            {
                "subgoal_id": "subgoal:type",
                "goal_id": "goal:one",
                "covered_facet_ids": ["facet:type"],
                "source_route": SourceRouteKind.LOCAL_STATIC.value,
                "source_authority": SourceAuthorityClass.AUTHORITATIVE.value,
            },
            {
                "subgoal_id": "subgoal:effect",
                "goal_id": "goal:one",
                "covered_facet_ids": ["facet:effect"],
                "source_route": SourceRouteKind.REVIEWED_CONTRACT.value,
                "source_authority": SourceAuthorityClass.AUTHORITATIVE.value,
            },
        ],
    )
    assert set(proof.covered_facet_ids) >= set(proof.required_facet_ids)
    assert proof.independent_source_authority is True


def test_model_decomposition_without_authority_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    goal = _goal(roots)
    with pytest.raises(LogicPredictionCegisAuthorityError, match="model"):
        engine.prove_subgoal_refinement(
            parent_goal=goal,
            subgoals=[
                {
                    "subgoal_id": "subgoal:llm-a",
                    "goal_id": "goal:one",
                    "covered_facet_ids": ["facet:type", "facet:effect"],
                    "source_route": SourceRouteKind.LLM.value,
                    "source_authority": SourceAuthorityClass.NOMINATING.value,
                }
            ],
            model_proposed=True,
        )


def test_authorized_decomposition_applied_without_deleting_goal(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    goal = _goal(roots)
    state = engine.initial_state(
        roots=roots,
        goals=[goal],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    proof = engine.prove_subgoal_refinement(
        parent_goal=goal,
        subgoals=[
            {
                "subgoal_id": "subgoal:type",
                "goal_id": "goal:one",
                "covered_facet_ids": ["facet:type"],
                "source_route": SourceRouteKind.LOCAL_STATIC.value,
                "source_authority": SourceAuthorityClass.AUTHORITATIVE.value,
            },
            {
                "subgoal_id": "subgoal:effect",
                "goal_id": "goal:one",
                "covered_facet_ids": ["facet:effect"],
                "source_route": SourceRouteKind.LOCAL_STATIC.value,
                "source_authority": SourceAuthorityClass.AUTHORITATIVE.value,
            },
        ],
    )
    evidence = RefinementEvidence(
        subgoal_decomposition=(
            {
                "subgoal_id": "subgoal:type",
                "goal_id": "goal:one",
            },
            {
                "subgoal_id": "subgoal:effect",
                "goal_id": "goal:one",
            },
        ),
        refinement_proof=proof,
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert "goal:one" in new_state.active_goal_ids
    assert "subgoal:type" in new_state.subgoal_ids
    assert "subgoal:effect" in new_state.subgoal_ids
    assert proof.proof_id in new_state.refinement_proof_ids
    assert any(a.kind is RefinementActionKind.DECOMPOSE_SUBGOALS for a in rnd.actions)


def test_unauthorized_premise_addition_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    evidence = RefinementEvidence(authorized_premises_to_add=("premise:forged",))
    with pytest.raises(LogicPredictionCegisAuthorityError, match="unauthorized premise"):
        engine.apply_round(state, evidence)


def test_cannot_readd_excluded_premise(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a", "premise:b"),
        selected_premise_ids=("premise:a", "premise:b"),
    )
    s1, _ = engine.apply_round(
        state, RefinementEvidence(premises_to_exclude=("premise:b",))
    )
    assert "premise:b" in s1.excluded_premise_ids
    with pytest.raises(LogicPredictionCegisMonotonicityError, match="excluded premise"):
        engine.apply_round(
            s1, RefinementEvidence(authorized_premises_to_add=("premise:b",))
        )


# ---------------------------------------------------------------------------
# Monotonic identity, cycles, repeated state
# ---------------------------------------------------------------------------


def test_state_identity_is_monotonic_across_rounds(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a", "premise:b"),
        selected_premise_ids=("premise:a",),
    )
    s1, _ = engine.apply_round(
        state, RefinementEvidence(authorized_premises_to_add=("premise:b",))
    )
    assert s1.state_id != state.state_id
    assert state.state_id in s1.lineage_state_ids
    assert s1.round_index == state.round_index + 1
    # Same content-addressed identity is stable.
    assert engine.state_identity(s1) == s1.state_id
    assert LogicRefinementState.from_dict(s1.to_dict()).state_id == s1.state_id


def test_repeated_state_and_cycles_terminate(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine(max_rounds=8, max_repeated_states=2)
    initial = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
        residual_gap_ids=("gap:open",),  # keep non-fixed-point
    )

    # Evidence that re-records the same residual (state content may stabilize).
    def stream(state: LogicRefinementState, round_index: int) -> RefinementEvidence:
        return RefinementEvidence(residual_gaps=("gap:open",), query_hints=("h",))

    receipt = engine.refine(initial, stream)
    assert receipt.disposition is RefinementDisposition.INCONCLUSIVE
    assert receipt.stop_reason in {
        RefinementStopReason.CYCLE_DETECTED,
        RefinementStopReason.REPEATED_STATE,
        RefinementStopReason.NO_PROGRESS,
    }
    # Residual gaps still present for Tactician.
    assert receipt.residual_gap_ids
    assert receipt.tactician_feedback is not None


# ---------------------------------------------------------------------------
# Bound exhaustion / cancellation → inconclusive with residuals
# ---------------------------------------------------------------------------


def test_max_rounds_returns_bound_exhausted_with_residuals(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine(max_rounds=2)
    initial = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots), _hypothesis(roots, "hyp:two")],
        authorized_premise_ids=("premise:a", "premise:extra"),
        selected_premise_ids=("premise:a",),
    )
    # Each round adds a distinct residual so state progresses until max rounds.
    evidence = [
        RefinementEvidence(residual_gaps=(f"gap:r{i}",))
        for i in range(4)
    ]
    receipt = engine.refine(initial, evidence)
    assert receipt.disposition is RefinementDisposition.BOUND_EXHAUSTED
    assert receipt.stop_reason is RefinementStopReason.MAX_ROUNDS
    assert receipt.residual_gap_ids
    assert receipt.tactician_feedback is not None
    assert receipt.final_state.original_goal_ids == ("goal:one",)


def test_max_counterexamples_enforced(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine(max_counterexamples=2)
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    evidence = RefinementEvidence(
        raw_solver_countermodel_ids=("raw:1", "raw:2", "raw:3"),
    )
    with pytest.raises(LogicPredictionCegisBoundsError, match="max_counterexamples"):
        engine.apply_round(state, evidence)


def test_cancellation_returns_cancelled_with_residual_gaps(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine(max_rounds=10)
    initial = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    flag = threading.Event()
    flag.set()
    receipt = engine.refine(
        initial,
        [RefinementEvidence(residual_gaps=("gap:x",))],
        cancelled=flag,
    )
    assert receipt.disposition is RefinementDisposition.CANCELLED
    assert receipt.stop_reason is RefinementStopReason.CANCELLED
    assert receipt.cancelled is True
    # Active hypothesis surfaced as residual work.
    assert receipt.residual_gap_ids
    assert receipt.tactician_feedback is not None


def test_engine_cancel_method_stops_refine(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine(max_rounds=10)
    initial = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    engine.cancel()
    receipt = engine.refine(
        initial,
        [RefinementEvidence(residual_gaps=("gap:x",))],
    )
    assert receipt.disposition is RefinementDisposition.CANCELLED
    engine.reset_cancellation()
    assert engine.cancelled is False


# ---------------------------------------------------------------------------
# Full refine success path
# ---------------------------------------------------------------------------


def test_refine_rejects_all_hypotheses_via_validated_countermodels(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine(max_rounds=4)
    initial = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots), _hypothesis(roots, "hyp:two")],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    cm1 = _validated_countermodel(roots, receipt_id="cm:v1")
    cm2 = _validated_countermodel(roots, receipt_id="cm:v2", via_negation=True)
    evidence = [
        RefinementEvidence(
            countermodel_receipts=(cm1,),
            hypothesis_narrowings={
                cm1.receipt_id: {"reject_hypothesis_ids": ["hyp:one"]},
            },
        ),
        RefinementEvidence(
            countermodel_receipts=(cm2,),
            hypothesis_narrowings={
                cm2.receipt_id: {"reject_hypothesis_ids": ["hyp:two"]},
            },
        ),
    ]
    receipt = engine.refine(initial, evidence)
    assert receipt.disposition in {
        RefinementDisposition.REFINED,
        RefinementDisposition.FIXED_POINT,
    }
    assert not receipt.final_state.active_hypothesis_ids
    assert set(receipt.final_state.excluded_hypothesis_ids) == {"hyp:one", "hyp:two"}
    assert set(receipt.original_goal_ids) == {"goal:one"}
    assert receipt.is_conclusive
    assert receipt.to_dict()["interface"] == LOGIC_PREDICTION_CEGIS_INTERFACE


# ---------------------------------------------------------------------------
# Deterministic replay is identity-equivalent
# ---------------------------------------------------------------------------


def test_deterministic_replay_is_identity_equivalent(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = create_logic_prediction_cegis(
        LogicRefinementBounds(max_rounds=4, max_counterexamples=8)
    )
    initial = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    cm = _validated_countermodel(roots)
    receipt = engine.refine(
        initial,
        [
            RefinementEvidence(
                countermodel_receipts=(cm,),
                residual_gaps=("gap:unsupported",),
                hypothesis_narrowings={
                    cm.receipt_id: {"reject_hypothesis_ids": ["hyp:one"]},
                },
            )
        ],
    )
    replayed = engine.replay(receipt)
    assert replayed.identity == receipt.identity
    assert replayed.to_dict() == receipt.to_dict()

    # Round-trip via dict.
    from_dict = LogicRefinementReceipt.from_dict(receipt.to_dict())
    replayed2 = engine.replay(from_dict.to_dict())
    assert replayed2.identity == receipt.identity
    assert replayed2.final_state.state_id == receipt.final_state.state_id
    assert replayed2.original_goal_ids == receipt.original_goal_ids
    assert replayed2.original_facet_ids == receipt.original_facet_ids


def test_round_and_state_dict_round_trip(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    _, rnd = engine.apply_round(
        state,
        RefinementEvidence(raw_solver_countermodel_ids=("raw:x",)),
    )
    assert LogicRefinementState.from_dict(state.to_dict()).state_id == state.state_id
    rebuilt_round = type(rnd).from_dict(rnd.to_dict())
    assert rebuilt_round.round_id == rnd.round_id
    assert rebuilt_round.disposition is rnd.disposition


# ---------------------------------------------------------------------------
# Cross-root validated receipt is non-authoritative
# ---------------------------------------------------------------------------


def test_cross_root_countermodel_is_diagnostic_only(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    other_roots = ProgramLogicAuthorityRoots(
        repository_id="repository:other",
        objective_id="objective:other",
        trace_id="trace:other",
        change_id="change:other",
        consumer_id="consumer:other",
        forest_id="forest:other",
        tree_id="tree:other",
        overlay_id="overlay:other",
        graph_id="graph:other",
        index_id="index:other",
        corpus_id="corpus:other",
        model_id="model:other",
        translator_id="translator:other",
        toolchain_id="toolchain:other",
        policy_id="policy:other",
        environment_id="environment:other",
    )
    foreign = _validated_countermodel(other_roots, receipt_id="cm:foreign")
    evidence = RefinementEvidence(
        countermodel_receipts=(foreign,),
        hypothesis_narrowings={
            foreign.receipt_id: {"reject_hypothesis_ids": ["hyp:one"]},
        },
    )
    new_state, rnd = engine.apply_round(state, evidence)
    assert "hyp:one" in new_state.active_hypothesis_ids
    assert foreign.receipt_id in new_state.diagnostic_countermodel_ids
    assert "countermodel_stale_roots" in rnd.reason_codes


# ---------------------------------------------------------------------------
# Decomposition without proof rejected
# ---------------------------------------------------------------------------


def test_decomposition_requires_refinement_proof(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = _engine()
    state = engine.initial_state(
        roots=roots,
        goals=[_goal(roots)],
        hypotheses=[_hypothesis(roots)],
        authorized_premise_ids=("premise:a",),
        selected_premise_ids=("premise:a",),
    )
    with pytest.raises(LogicPredictionCegisAuthorityError, match="SubgoalRefinementProof"):
        engine.apply_round(
            state,
            RefinementEvidence(
                subgoal_decomposition=(
                    {"subgoal_id": "subgoal:x", "goal_id": "goal:one"},
                )
            ),
        )


def test_subgoal_refinement_proof_from_dict_round_trip() -> None:
    proof = SubgoalRefinementProof(
        proof_id="proof:one",
        parent_goal_id="goal:one",
        subgoal_ids=("subgoal:a", "subgoal:b"),
        covered_facet_ids=("facet:type",),
        required_facet_ids=("facet:type",),
        independent_source_authority=True,
        model_proposed=False,
    )
    assert SubgoalRefinementProof.from_dict(proof.to_dict()).proof_id == proof.proof_id


def test_factory_create_logic_prediction_cegis() -> None:
    engine = create_logic_prediction_cegis({"max_rounds": 3, "max_goals": 5})
    assert engine.bounds.max_rounds == 3
    assert engine.bounds.max_goals == 5
    assert engine.producer_id.endswith("logic-prediction-cegis@1")
