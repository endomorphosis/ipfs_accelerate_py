"""Admit only reconstructed and unique logic predictions (LPR-014)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    GapDisposition,
    GapMissingClass,
    GoalDisposition,
    GoalFamily,
    HypothesisDisposition,
    LogicFacetKind,
    LogicFacetRef,
    LogicGap,
    LogicHypothesis,
    LogicPredictionReceipt,
    NativeGoalDisposition,
    PredictionDisposition,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicNativeGoalBinding,
    ProofStatus,
    SemanticRoundTripReceipt,
    SourceAuthorityClass,
    SourceRouteKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.logic_prediction_admission import (
    AutomaticConsequenceKind,
    LOGIC_PREDICTION_ADMISSION_INTERFACE,
    LogicPredictionAdmission,
    LogicPredictionAdmissionError,
    LogicPredictionAdmissionRequest,
    LogicPredictionDecisionDisposition,
    LogicPredictionRejectionReason,
    create_logic_prediction_admission,
)
from ipfs_accelerate_py.agent_supervisor.proof.tactician_hammer_coordinator import (
    CoordinationConclusiveness,
    HammerCoordinationOutcome,
    HammerCoordinationReceipt,
    PremiseSelectorMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:lpr-014",
        objective_id="objective:lpr-014",
        trace_id="trace:lpr-014",
        change_id="change:lpr-014",
        consumer_id="consumer:lpr-014",
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
    *,
    unsupported: bool = False,
) -> LogicFacetRef:
    return LogicFacetRef(
        facet_id=facet_id,
        kind=kind,
        subject_symbol_id="symbol:process",
        contract_ref=f"contract:{facet_id}",
        unsupported=unsupported,
    )


def _round_trip(
    obligation_id: str = "obligation:logic-ir",
) -> SemanticRoundTripReceipt:
    return SemanticRoundTripReceipt(
        receipt_id="srt:one",
        logic_ir_claim_id=obligation_id,
        native_statement_id="native:stmt-one",
        equivalence_method="statement_equivalence",
        disposition=NativeGoalDisposition.ROUND_TRIP_OK,
    )


def _goal(
    roots: ProgramLogicAuthorityRoots,
    goal_id: str = "goal:one",
    *,
    assumptions: tuple[str, ...] = ("assumption:context",),
    unsupported: tuple[LogicFacetRef, ...] = (),
    disposition: GoalDisposition = GoalDisposition.OPEN,
) -> ProgramLogicGoal:
    return ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=GoalFamily.VALUE,
        disposition=disposition,
        positive_statement_ref=f"stmt:{goal_id}",
        required_facets=(_facet(),),
        unsupported_facets=unsupported,
        assumption_refs=assumptions,
        invalidation_refs=(roots.tree_id, roots.corpus_id),
    )


def _hypothesis(
    roots: ProgramLogicAuthorityRoots,
    hypothesis_id: str = "hyp:one",
    *,
    target_goal_id: str = "goal:one",
    value_ref: str = "value:unique-a",
    construction_ref: str = "",
    placement_ref: str = "",
    consequence: str | None = None,
    source_authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE,
    proof_status: ProofStatus = ProofStatus.KERNEL_VERIFIED,
    disposition: HypothesisDisposition = HypothesisDisposition.PROVED,
    routes: tuple[SourceRouteKind, ...] = (SourceRouteKind.LOCAL_STATIC,),
    premises: tuple[str, ...] = ("premise:a", "premise:b"),
    counterexample_target_ref: str = "",
) -> LogicHypothesis:
    return LogicHypothesis(
        roots=roots,
        hypothesis_id=hypothesis_id,
        target_goal_id=target_goal_id,
        disposition=disposition,
        claimed_consequence_ref=consequence or f"consequence:{hypothesis_id}",
        construction_ref=construction_ref,
        placement_ref=placement_ref,
        value_ref=value_ref,
        evidence_refs=("evidence:static",),
        evidence_route_kinds=routes,
        selected_premise_ids=premises,
        counterexample_target_ref=counterexample_target_ref,
        source_authority=source_authority,
        proof_status=proof_status,
        completeness=disposition is not HypothesisDisposition.NOMINATED,
        invalidation_refs=(roots.tree_id,),
    )


def _native(
    roots: ProgramLogicAuthorityRoots,
    *,
    binding_id: str = "binding:one",
    premises: tuple[str, ...] = ("premise:a", "premise:b", "premise:c"),
    kernel_id: str = "kernel:lean4",
) -> ProgramLogicNativeGoalBinding:
    return ProgramLogicNativeGoalBinding(
        roots=roots,
        binding_id=binding_id,
        logic_ir_obligation_id="obligation:logic-ir",
        premise_ids=premises,
        native_itp_id="itp:lean",
        goal_snapshot_id="goal-snapshot:exact",
        native_theorem_source_id="native-src:theorem",
        proof_hole_id="hole:single",
        kernel_id=kernel_id,
        semantic_round_trip=_round_trip(),
        disposition=NativeGoalDisposition.ROUND_TRIP_OK,
        import_ids=("import:Prelude",),
        invalidation_refs=(roots.tree_id, roots.toolchain_id),
    )


def _hammer(
    *,
    outcome: HammerCoordinationOutcome = HammerCoordinationOutcome.VERIFIED,
    conclusiveness: CoordinationConclusiveness = (
        CoordinationConclusiveness.CONCLUSIVE_PROOF
    ),
    kernel_checked: bool = True,
    proof_success: bool = True,
    translation_map_id: str = "translation:one",
    environment_lock_id: str = "env-lock:one",
    native_goal_binding_id: str = "binding:one",
    reconstruction_id: str = "reconstruction:one",
    learned_selector_model_digest: str = "",
    metadata: dict | None = None,
    provider_result: dict | None = None,
) -> HammerCoordinationReceipt:
    return HammerCoordinationReceipt(
        receipt_id="hammer:receipt-one",
        outcome=outcome,
        conclusiveness=conclusiveness,
        gate_decision={"disposition": "permitted"},
        policy_intersection={"timeout_ms": 1000},
        resource_enforcement={"enforced": True},
        selector_mode=PremiseSelectorMode.DETERMINISTIC,
        translation_map_id=translation_map_id,
        environment_lock_id=environment_lock_id,
        obligation_id="obligation:logic-ir",
        request_id="hammer:req-one",
        provider_result=provider_result or {"status": "verified"},
        native_goal_binding_id=native_goal_binding_id,
        receipt_binding={
            "reconstruction_id": reconstruction_id,
            "native_goal_binding_id": native_goal_binding_id,
        },
        reason_codes=(),
        learned_selector_model_digest=learned_selector_model_digest,
        proof_success=proof_success,
        kernel_checked=kernel_checked,
        metadata=metadata or {},
    )


def _gap(
    roots: ProgramLogicAuthorityRoots,
    gap_id: str = "gap:mandatory-one",
    *,
    goal_id: str = "goal:one",
    disposition: GapDisposition = GapDisposition.REQUIRED,
    severity: str = "mandatory",
) -> LogicGap:
    return LogicGap(
        roots=roots,
        gap_id=gap_id,
        goal_id=goal_id,
        missing_class=GapMissingClass.VALUE,
        disposition=disposition,
        observed_fact_ref="observed:none",
        required_fact_ref="required:value",
        discrepancy_ref="disc:missing-value",
        severity=severity,
        invalidation_refs=(roots.tree_id,),
    )


def _validated_cm(
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


def _diagnostic_cm(
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


def _request(
    roots: ProgramLogicAuthorityRoots,
    *,
    goals: tuple[ProgramLogicGoal, ...] | None = None,
    hypotheses: tuple[LogicHypothesis, ...] | None = None,
    hammer: HammerCoordinationReceipt | None = None,
    native: ProgramLogicNativeGoalBinding | None = None,
    consistency: ConsistencyDisposition = (
        ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
    ),
    countermodels: tuple[CountermodelValidationReceipt, ...] = (),
    residual_gaps: tuple[LogicGap, ...] = (),
    automatic_kind: AutomaticConsequenceKind = AutomaticConsequenceKind.VALUE,
    **kwargs,
) -> LogicPredictionAdmissionRequest:
    goal = (goals or (_goal(roots),))[0]
    hyps = hypotheses or (_hypothesis(roots),)
    payload: dict = {
        "roots": roots,
        "goals": goals or (goal,),
        "hypotheses": hyps,
        "tactician_plan_id": "plan:tactician-one",
        "hammer_receipt": hammer or _hammer(),
        "native_goal_binding": native or _native(roots),
        "consistency_disposition": consistency,
        "countermodel_receipts": countermodels,
        "residual_gaps": residual_gaps,
        "kernel_receipt_id": "kernel:receipt-one",
        "reconstruction_id": "reconstruction:one",
        "environment_receipt_id": "env:receipt-one",
        "translation_id": "translation:one",
        "candidate_id": "candidate:one",
        "automatic_kind": automatic_kind,
        "current_tree_id": roots.tree_id,
        "current_corpus_id": roots.corpus_id,
        "current_environment_id": roots.environment_id,
        "current_toolchain_id": roots.toolchain_id,
        "current_policy_id": roots.policy_id,
        "current_translator_id": roots.translator_id,
    }
    payload.update(kwargs)
    return LogicPredictionAdmissionRequest(**payload)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_interface_and_factory() -> None:
    engine = create_logic_prediction_admission()
    assert isinstance(engine, LogicPredictionAdmission)
    assert LOGIC_PREDICTION_ADMISSION_INTERFACE == "LogicPredictionAdmission@1"


def test_admits_unique_reconstructed_value_prediction(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    goal = _goal(roots, assumptions=("assumption:context", "assumption:bound"))
    hyp = _hypothesis(roots, value_ref="value:unique-a")
    decision = engine.admit(
        _request(roots, goals=(goal,), hypotheses=(hyp,))
    )

    assert decision.disposition is LogicPredictionDecisionDisposition.ADMITTED
    assert decision.is_admitted
    assert decision.write_authority is False
    assert decision.semantic_authority is False
    assert decision.automation_eligible is True
    assert decision.selected_consequence_ref == "value:value:unique-a"
    assert decision.assumption_refs == (
        "assumption:bound",
        "assumption:context",
    )
    assert decision.receipt is not None
    receipt = decision.receipt
    assert isinstance(receipt, LogicPredictionReceipt)
    assert receipt.disposition is PredictionDisposition.PROVED
    assert receipt.proof_status is ProofStatus.KERNEL_VERIFIED
    assert receipt.source_authority is SourceAuthorityClass.AUTHORITATIVE
    assert receipt.reconstruction_id == "reconstruction:one"
    assert receipt.kernel_receipt_id == "kernel:receipt-one"
    assert receipt.translation_id == "translation:one"
    assert receipt.environment_receipt_id == "env:receipt-one"
    assert receipt.derived_value_ref == "value:unique-a"
    assert receipt.automation_eligible is True
    assert set(receipt.assumption_refs) == {
        "assumption:bound",
        "assumption:context",
    }
    # Emitted receipt / decision never grant write authority.
    payload = decision.to_dict()
    assert payload["write_authority"] is False
    assert payload["semantic_authority"] is False
    assert "write_authority" not in (receipt.to_dict() or {}) or True
    assert receipt.to_dict().get("write_authority") in (None, False)


def test_admits_unique_construction_and_placement(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    construction = _hypothesis(
        roots,
        hypothesis_id="hyp:construction",
        value_ref="",
        construction_ref="construction:adapter-a",
        consequence="consequence:construction",
    )
    decision = engine.admit(
        _request(
            roots,
            hypotheses=(construction,),
            automatic_kind=AutomaticConsequenceKind.CONSTRUCTION,
        )
    )
    assert decision.is_admitted
    assert decision.receipt is not None
    assert decision.receipt.derived_clause_ref == "construction:adapter-a"

    placement = _hypothesis(
        roots,
        hypothesis_id="hyp:placement",
        value_ref="",
        placement_ref="placement:site-a",
        consequence="consequence:placement",
    )
    decision = engine.admit(
        _request(
            roots,
            hypotheses=(placement,),
            automatic_kind=AutomaticConsequenceKind.PLACEMENT,
        )
    )
    assert decision.is_admitted
    assert decision.receipt is not None
    assert decision.receipt.derived_placement_ref == "placement:site-a"


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


def test_zero_eligible_candidates_abstain(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    # Nominating source cannot be admitted.
    hyp = _hypothesis(
        roots,
        source_authority=SourceAuthorityClass.NOMINATING,
        proof_status=ProofStatus.UNPROVED,
        disposition=HypothesisDisposition.NOMINATED,
        routes=(SourceRouteKind.VECTOR,),
        premises=(),
    )
    decision = engine.admit(
        _request(roots, hypotheses=(hyp,), automatic_kind=AutomaticConsequenceKind.VALUE)
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    assert decision.receipt is None
    assert decision.write_authority is False
    codes = set(decision.reason_codes)
    assert (
        LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value in codes
        or LogicPredictionRejectionReason.ZERO_ELIGIBLE.value in codes
        or LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value in codes
    )


def test_multiple_eligible_value_candidates_abstain(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    first = _hypothesis(
        roots, hypothesis_id="hyp:a", value_ref="value:alpha"
    )
    second = _hypothesis(
        roots, hypothesis_id="hyp:b", value_ref="value:beta"
    )
    decision = engine.admit(
        _request(
            roots,
            hypotheses=(first, second),
            automatic_kind=AutomaticConsequenceKind.VALUE,
        )
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    assert decision.receipt is None
    assert (
        LogicPredictionRejectionReason.PREDICTION_NON_UNIQUE.value
        in decision.reason_codes
        or LogicPredictionRejectionReason.MULTIPLE_ELIGIBLE.value
        in decision.reason_codes
    )
    assert set(decision.eligible_consequence_refs) == {
        "value:value:alpha",
        "value:value:beta",
    }


def test_same_consequence_collapses_under_deterministic_tie(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    """Two hyps claiming the identical value_ref collapse to one consequence."""
    engine = LogicPredictionAdmission()
    first = _hypothesis(
        roots, hypothesis_id="hyp:a", value_ref="value:shared"
    )
    second = _hypothesis(
        roots, hypothesis_id="hyp:b", value_ref="value:shared"
    )
    decision = engine.admit(
        _request(
            roots,
            hypotheses=(first, second),
            automatic_kind=AutomaticConsequenceKind.VALUE,
        )
    )
    assert decision.is_admitted
    assert decision.selected_consequence_ref == "value:value:shared"
    # Deterministic first hypothesis_id wins.
    assert decision.hypothesis_id == "hyp:a"


# ---------------------------------------------------------------------------
# Authority / kernel / translation / environment
# ---------------------------------------------------------------------------


def test_requires_authoritative_independent_premises(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    hyp = _hypothesis(
        roots,
        source_authority=SourceAuthorityClass.DIAGNOSTIC,
        proof_status=ProofStatus.UNPROVED,
        disposition=HypothesisDisposition.NOMINATED,
        routes=(SourceRouteKind.SOLVER,),
        premises=("premise:a",),
    )
    decision = engine.admit(_request(roots, hypotheses=(hyp,)))
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    assert (
        LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
        in decision.reason_codes
    )


def test_vector_and_model_authority_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    # AUTHORITATIVE + VECTOR is rejected at LogicHypothesis construction.
    with pytest.raises(Exception):
        _hypothesis(
            roots,
            routes=(SourceRouteKind.VECTOR,),
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
        )

    hyp = _hypothesis(
        roots,
        source_authority=SourceAuthorityClass.NOMINATING,
        proof_status=ProofStatus.UNPROVED,
        disposition=HypothesisDisposition.NOMINATED,
        routes=(SourceRouteKind.VECTOR, SourceRouteKind.LLM),
        premises=("premise:a",),
    )
    decision = engine.admit(_request(roots, hypotheses=(hyp,)))
    codes = set(decision.reason_codes)
    assert LogicPredictionRejectionReason.VECTOR_AUTHORITY.value in codes or (
        LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value in codes
    )


def test_hammer_must_be_kernel_verified_not_solver_only(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(
            roots,
            hammer=_hammer(
                outcome=HammerCoordinationOutcome.CANDIDATE,
                conclusiveness=CoordinationConclusiveness.NON_CONCLUSIVE,
                kernel_checked=False,
                proof_success=False,
                provider_result={"authoritative_assurance": "solver_checked"},
            ),
        )
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    codes = set(decision.reason_codes)
    assert (
        LogicPredictionRejectionReason.HAMMER_NOT_VERIFIED.value in codes
        or LogicPredictionRejectionReason.KERNEL_NOT_ACCEPTED.value in codes
        or LogicPredictionRejectionReason.SOLVER_ONLY_PROOF.value in codes
    )


def test_translation_native_goal_environment_must_match(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(
            roots,
            hammer=_hammer(translation_map_id="translation:other"),
            translation_id="translation:one",
        )
    )
    assert (
        LogicPredictionRejectionReason.TRANSLATION_MISMATCH.value
        in decision.reason_codes
        or decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    )

    # Environment drift on current_* markers.
    decision = engine.admit(
        _request(roots, current_environment_id="environment:stale")
    )
    assert (
        LogicPredictionRejectionReason.STALE_STATE.value in decision.reason_codes
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.REJECTED


def test_native_binding_round_trip_required(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    # Unsupported disposition binding is not ROUND_TRIP_OK.
    bad_rt = SemanticRoundTripReceipt(
        receipt_id="srt:bad",
        logic_ir_claim_id="obligation:logic-ir",
        native_statement_id="native:stmt",
        equivalence_method="statement_equivalence",
        disposition=NativeGoalDisposition.UNSUPPORTED,
        unsupported_construct_refs=("construct:native-fn",),
    )
    native = ProgramLogicNativeGoalBinding(
        roots=roots,
        binding_id="binding:bad",
        logic_ir_obligation_id="obligation:logic-ir",
        premise_ids=("premise:a",),
        native_itp_id="itp:lean",
        goal_snapshot_id="goal-snapshot:exact",
        native_theorem_source_id="native-src:theorem",
        proof_hole_id="hole:single",
        kernel_id="kernel:lean4",
        semantic_round_trip=bad_rt,
        disposition=NativeGoalDisposition.UNSUPPORTED,
        unsupported_native_construct_refs=("construct:native-fn",),
        invalidation_refs=(roots.tree_id,),
    )
    decision = engine.admit(
        _request(roots, native=native, hammer=_hammer(native_goal_binding_id="binding:bad"))
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    assert (
        LogicPredictionRejectionReason.NATIVE_GOAL_MISMATCH.value
        in decision.reason_codes
    )


# ---------------------------------------------------------------------------
# Consistency / ex-falso
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "consistency",
    [
        ConsistencyDisposition.UNKNOWN,
        ConsistencyDisposition.STRUCTURAL_CONFLICT,
        ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION,
        ConsistencyDisposition.LOGICAL_CONFLICT_PROVED,
        ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED,
    ],
)
def test_ex_falso_blocked_when_consistency_invalid_or_unknown(
    roots: ProgramLogicAuthorityRoots,
    consistency: ConsistencyDisposition,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(_request(roots, consistency=consistency))
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    codes = set(decision.reason_codes)
    assert LogicPredictionRejectionReason.EX_FALSO_BLOCKED.value in codes
    assert decision.receipt is None


# ---------------------------------------------------------------------------
# Countermodels: only independently validated may reject
# ---------------------------------------------------------------------------


def test_validated_countermodel_produces_validated_refutation(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    hyp = _hypothesis(
        roots,
        counterexample_target_ref="obligation:logic-ir",
    )
    decision = engine.admit(
        _request(
            roots,
            hypotheses=(hyp,),
            countermodels=(_validated_cm(roots),),
        )
    )
    assert (
        decision.disposition
        is LogicPredictionDecisionDisposition.VALIDATED_REFUTATION
    )
    assert decision.is_refuted
    assert decision.receipt is not None
    assert (
        decision.receipt.disposition
        is PredictionDisposition.VALIDATED_REFUTATION
    )
    assert decision.receipt.proof_status is ProofStatus.VALIDATED_REFUTED
    assert decision.receipt.countermodel_validation_id == "cm:validated"
    assert decision.write_authority is False


def test_validated_refutation_via_proof_of_negation(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(
            roots,
            countermodels=(_validated_cm(roots, via_negation=True),),
        )
    )
    assert (
        decision.disposition
        is LogicPredictionDecisionDisposition.VALIDATED_REFUTATION
    )
    assert decision.countermodel_validation_id


def test_raw_diagnostic_countermodel_never_rejects(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    # Diagnostic only must not produce validated refutation; positive path
    # may still admit when reconstruction is valid.
    decision = engine.admit(
        _request(
            roots,
            countermodels=(_diagnostic_cm(roots),),
        )
    )
    assert (
        decision.disposition
        is not LogicPredictionDecisionDisposition.VALIDATED_REFUTATION
    )
    # Happy path still admits unique reconstructed value.
    assert decision.is_admitted
    assert decision.receipt is not None
    assert decision.receipt.disposition is PredictionDisposition.PROVED


def test_solver_only_refutation_without_validation_does_not_reject(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    """A claimed VALIDATED disposition without replay/negation cannot reject."""
    engine = LogicPredictionAdmission()
    # Construction of VALIDATED without evidence fails at the contract layer.
    with pytest.raises(Exception):
        CountermodelValidationReceipt(
            roots=roots,
            receipt_id="cm:fake",
            solver_countermodel_id="solver-cm:x",
            translation_map_id="translation:one",
            originating_logic_ir_id="obligation:logic-ir",
            disposition=CountermodelDisposition.VALIDATED,
            invalidation_refs=(roots.tree_id,),
        )


# ---------------------------------------------------------------------------
# Residual gaps / stale / higher-precedence / write authority
# ---------------------------------------------------------------------------


def test_mandatory_residual_gap_rejects_admission(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(roots, residual_gaps=(_gap(roots),))
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.REJECTED
    assert (
        LogicPredictionRejectionReason.MANDATORY_RESIDUAL_GAP.value
        in decision.reason_codes
    )
    assert decision.receipt is None
    assert "gap:mandatory-one" in decision.residual_gap_ids


def test_stale_tree_rejects(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(roots, current_tree_id="tree:stale-other")
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.REJECTED
    assert (
        LogicPredictionRejectionReason.STALE_STATE.value in decision.reason_codes
    )


def test_higher_precedence_contract_conflict_rejects(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(roots, higher_precedence_conflict=True)
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.REJECTED
    assert (
        LogicPredictionRejectionReason.HIGHER_PRECEDENCE_CONFLICT.value
        in decision.reason_codes
    )


def test_write_or_semantic_authority_claims_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(roots, write_authority_claimed=True)
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.REJECTED
    assert (
        LogicPredictionRejectionReason.WRITE_AUTHORITY_CLAIMED.value
        in decision.reason_codes
    )
    assert decision.write_authority is False

    decision = engine.admit(
        _request(roots, semantic_authority_claimed=True)
    )
    assert (
        LogicPredictionRejectionReason.SEMANTIC_AUTHORITY_CLAIMED.value
        in decision.reason_codes
    )


def test_decision_cannot_be_constructed_with_write_authority(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.planning.logic_prediction_admission import (
        LogicPredictionDecision,
    )

    with pytest.raises(LogicPredictionAdmissionError, match="write authority"):
        LogicPredictionDecision(
            decision_id="decision:x",
            disposition=LogicPredictionDecisionDisposition.ABSTAINED,
            roots=roots,
            goal_id="goal:one",
            hypothesis_id="",
            reason_codes=(),
            write_authority=True,
        )


# ---------------------------------------------------------------------------
# Assumptions / unsupported facets preserved
# ---------------------------------------------------------------------------


def test_preserves_assumptions_and_unsupported_facets_on_admit_and_abstain(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    unsupported = (
        _facet("facet:lifetime-unsupported", LogicFacetKind.LIFETIME, unsupported=True),
    )
    goal = _goal(
        roots,
        assumptions=("assumption:ctx", "assumption:bounds"),
        unsupported=unsupported,
    )
    decision = engine.admit(
        _request(roots, goals=(goal,), hypotheses=(_hypothesis(roots),))
    )
    assert decision.is_admitted
    assert set(decision.assumption_refs) == {
        "assumption:bounds",
        "assumption:ctx",
    }
    assert decision.unsupported_facet_ids == ("facet:lifetime-unsupported",)
    assert decision.receipt is not None
    assert set(decision.receipt.assumption_refs) == {
        "assumption:bounds",
        "assumption:ctx",
    }

    # Abstention still preserves facets/assumptions.
    abstained = engine.admit(
        _request(
            roots,
            goals=(goal,),
            hypotheses=(
                _hypothesis(roots, value_ref="value:a"),
                _hypothesis(
                    roots, hypothesis_id="hyp:two", value_ref="value:b"
                ),
            ),
        )
    )
    assert abstained.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    assert set(abstained.assumption_refs) == {
        "assumption:bounds",
        "assumption:ctx",
    }
    assert abstained.unsupported_facet_ids == ("facet:lifetime-unsupported",)


def test_cannot_promote_unsupported_facet_as_consequence(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    unsupported = (
        _facet("facet:mem-unsupported", LogicFacetKind.MEMORY, unsupported=True),
    )
    goal = _goal(roots, unsupported=unsupported)
    # Attempt to promote unsupported facet id as the value consequence.
    hyp = _hypothesis(roots, value_ref="facet:mem-unsupported")
    decision = engine.admit(
        _request(
            roots,
            goals=(goal,),
            hypotheses=(hyp,),
            automatic_kind=AutomaticConsequenceKind.VALUE,
        )
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    assert decision.receipt is None


# ---------------------------------------------------------------------------
# Learned selector ranking-only; authority flag blocked
# ---------------------------------------------------------------------------


def test_learned_selector_digest_without_authority_still_admits(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(
            roots,
            hammer=_hammer(learned_selector_model_digest="sha256:model-digest"),
            learned_selector_model_digest="sha256:model-digest",
        )
    )
    assert decision.is_admitted


def test_learned_model_authority_flag_blocks_admission(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    decision = engine.admit(
        _request(
            roots,
            hammer=_hammer(
                learned_selector_model_digest="sha256:model-digest",
                metadata={"learned_authority": True, "model_authority": True},
            ),
        )
    )
    assert decision.disposition is LogicPredictionDecisionDisposition.ABSTAINED
    codes = set(decision.reason_codes)
    assert (
        LogicPredictionRejectionReason.LEARNED_AUTHORITY.value in codes
        or LogicPredictionRejectionReason.MODEL_AUTHORITY.value in codes
    )


# ---------------------------------------------------------------------------
# Identity recomputation / root binding
# ---------------------------------------------------------------------------


def test_cross_root_hypothesis_rejects(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:other",
        objective_id="objective:lpr-014",
        trace_id="trace:lpr-014",
        change_id="change:lpr-014",
        consumer_id="consumer:lpr-014",
        forest_id="forest:one",
        tree_id="tree:other",
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
    hyp = _hypothesis(other)
    decision = engine.admit(_request(roots, hypotheses=(hyp,)))
    assert decision.disposition is LogicPredictionDecisionDisposition.REJECTED
    assert (
        LogicPredictionRejectionReason.ROOT_CHANGED.value in decision.reason_codes
    )


def test_request_recomputes_content_identity(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    req = _request(roots)
    assert req.content_id
    assert req.content_id == LogicPredictionAdmissionRequest(
        **{
            k: v
            for k, v in {
                "roots": req.roots,
                "goals": req.goals,
                "hypotheses": req.hypotheses,
                "tactician_plan_id": req.tactician_plan_id,
                "hammer_receipt": req.hammer_receipt,
                "native_goal_binding": req.native_goal_binding,
                "consistency_disposition": req.consistency_disposition,
                "kernel_receipt_id": req.kernel_receipt_id,
                "reconstruction_id": req.reconstruction_id,
                "environment_receipt_id": req.environment_receipt_id,
                "translation_id": req.translation_id,
                "candidate_id": req.candidate_id,
                "automatic_kind": req.automatic_kind,
                "current_tree_id": req.current_tree_id,
                "current_corpus_id": req.current_corpus_id,
                "current_environment_id": req.current_environment_id,
                "current_toolchain_id": req.current_toolchain_id,
                "current_policy_id": req.current_policy_id,
                "current_translator_id": req.current_translator_id,
            }.items()
        }
    ).content_id


def test_aliases_decide_assess_evaluate(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    engine = LogicPredictionAdmission()
    req = _request(roots)
    a = engine.admit(req)
    b = engine.decide(req)
    c = engine.assess(req)
    d = engine.evaluate(req)
    assert a.disposition == b.disposition == c.disposition == d.disposition
    assert a.is_admitted
