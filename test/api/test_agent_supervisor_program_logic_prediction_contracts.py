"""Contract tests for bounded program-logic prediction records (LPR-001)."""

from __future__ import annotations

import math

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    ContextOverlayDisposition,
    CountermodelDisposition,
    CountermodelValidationReceipt,
    FixedPointAttachmentDisposition,
    ForgedProgramLogicIdentityError,
    GapDisposition,
    GapMissingClass,
    GoalDisposition,
    GoalFamily,
    HypothesisDisposition,
    LogicFacetKind,
    LogicFacetRef,
    LogicFixedPointEvidenceAttachment,
    LogicGap,
    LogicGuidedRepairPacket,
    LogicHypothesis,
    LogicPredictionReceipt,
    LogicSubgoal,
    NativeGoalDisposition,
    PredictionDisposition,
    ProgramLogicAuthorityError,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicNativeGoalBinding,
    ProgramLogicPredictionError,
    ProofStatus,
    SemanticRoundTripReceipt,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:one",
        objective_id="objective:one",
        trace_id="trace:one",
        change_id="change:one",
        consumer_id="consumer:one",
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
    contract_ref: str = "type:Context",
    unsupported: bool = False,
) -> LogicFacetRef:
    return LogicFacetRef(
        facet_id=facet_id,
        kind=kind,
        subject_symbol_id="symbol:process",
        contract_ref=contract_ref,
        unsupported=unsupported,
    )


def _subgoal(
    subgoal_id: str = "subgoal:one",
    *,
    goal_id: str = "goal:one",
    depends_on: tuple[str, ...] = (),
    parent_subgoal_id: str = "",
    source_route: SourceRouteKind = SourceRouteKind.LOCAL_STATIC,
    source_authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE,
) -> LogicSubgoal:
    return LogicSubgoal(
        subgoal_id=subgoal_id,
        goal_id=goal_id,
        disposition=SubgoalDisposition.PLANNED,
        claim_ref="claim:one",
        parent_subgoal_id=parent_subgoal_id,
        depends_on=depends_on,
        source_route=source_route,
        source_authority=source_authority,
        proof_status=ProofStatus.UNPROVED,
        score_millipercent=12_500,
    )


def _round_trip(
    *,
    obligation_id: str = "obligation:logic-ir",
    disposition: NativeGoalDisposition = NativeGoalDisposition.ROUND_TRIP_OK,
) -> SemanticRoundTripReceipt:
    return SemanticRoundTripReceipt(
        receipt_id="roundtrip:one",
        logic_ir_claim_id=obligation_id,
        native_statement_id="native-stmt:one",
        equivalence_method="statement_equivalence",
        disposition=disposition,
    )


def test_roots_bind_objective_trace_change_consumer_and_shared_identities(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    assert roots.objective_id == "objective:one"
    assert roots.trace_id == "trace:one"
    assert roots.change_id == "change:one"
    assert roots.consumer_id == "consumer:one"
    assert roots.forest_id == "forest:one"
    assert roots.tree_id == "tree:one"
    assert roots.overlay_id == "overlay:one"
    assert roots.graph_id == "graph:one"
    assert roots.index_id == "index:one"
    assert roots.corpus_id == "corpus:one"
    assert roots.model_id == "model:one"
    assert roots.translator_id == "translator:one"
    assert roots.toolchain_id == "toolchain:one"
    assert roots.policy_id == "policy:one"
    assert roots.environment_id == "environment:one"
    assert roots.content_id.startswith("b")
    assert ProgramLogicAuthorityRoots.from_dict(roots.to_record()) == roots


def test_forged_identity_rejected(roots: ProgramLogicAuthorityRoots) -> None:
    payload = roots.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedProgramLogicIdentityError):
        ProgramLogicAuthorityRoots.from_dict(payload)


def test_source_bodies_secrets_and_nonfinite_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(
        ProgramLogicPredictionError, match="unsupported fields|source bodies"
    ):
        ProgramLogicGoal.from_dict(
            {
                "schema": ProgramLogicGoal.SCHEMA,
                "contract_version": 1,
                "roots": roots.to_dict(),
                "goal_id": "goal:one",
                "family": GoalFamily.POSITIVE.value,
                "disposition": GoalDisposition.OPEN.value,
                "positive_statement_ref": "stmt:one",
                "invalidation_refs": ["tree:one"],
                "source_body": "def evil(): pass",
            }
        )

    poisoned_roots = dict(roots.to_dict())
    poisoned_roots["source_body"] = "def evil(): pass"
    with pytest.raises(ProgramLogicPredictionError, match="source bodies"):
        ProgramLogicGoal.from_dict(
            {
                "schema": ProgramLogicGoal.SCHEMA,
                "contract_version": 1,
                "roots": poisoned_roots,
                "goal_id": "goal:one",
                "family": GoalFamily.POSITIVE.value,
                "disposition": GoalDisposition.OPEN.value,
                "positive_statement_ref": "stmt:one",
                "invalidation_refs": ["tree:one"],
            }
        )

    with pytest.raises(
        ProgramLogicPredictionError, match="unsupported fields|secret material"
    ):
        ProgramLogicGoal.from_dict(
            {
                "schema": ProgramLogicGoal.SCHEMA,
                "contract_version": 1,
                "roots": roots.to_dict(),
                "goal_id": "goal:one",
                "family": GoalFamily.POSITIVE.value,
                "disposition": GoalDisposition.OPEN.value,
                "positive_statement_ref": "stmt:one",
                "invalidation_refs": ["tree:one"],
                "api_key": "sk-live-not-a-real-key",
            }
        )

    with pytest.raises(ProgramLogicPredictionError, match="secret material"):
        SemanticRoundTripReceipt(
            receipt_id="roundtrip:leaky",
            logic_ir_claim_id="obligation:logic-ir",
            native_statement_id="native-stmt:one",
            equivalence_method="password=hunter2",
            disposition=NativeGoalDisposition.BOUND,
        )

    with pytest.raises(ProgramLogicPredictionError):
        LogicSubgoal(
            subgoal_id="subgoal:x",
            goal_id="goal:one",
            disposition=SubgoalDisposition.PENDING,
            claim_ref="claim:one",
            score_millipercent=math.inf,  # type: ignore[arg-type]
        )


def test_program_logic_goal_round_trip_and_state_machine(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    goal = ProgramLogicGoal(
        roots=roots,
        goal_id="goal:one",
        family=GoalFamily.POSITIVE,
        disposition=GoalDisposition.OPEN,
        positive_statement_ref="stmt:positive",
        affected_symbol_ids=("symbol:process",),
        required_facets=(_facet(),),
        unsupported_facets=(
            _facet(
                "facet:memory-unmodeled",
                LogicFacetKind.MEMORY,
                contract_ref="memory:lifetime",
                unsupported=True,
            ),
        ),
        assumption_refs=("premise:reviewed",),
        assumption_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=ProofStatus.UNPROVED,
        invalidation_refs=("tree:one", "corpus:one"),
    )
    assert goal.content_id.startswith("b")
    assert ProgramLogicGoal.from_dict(goal.to_record()) == goal
    assert goal.required_facets[0].kind is LogicFacetKind.TYPE
    assert goal.unsupported_facets[0].kind is LogicFacetKind.MEMORY

    with pytest.raises(ProgramLogicAuthorityError, match="kernel verification"):
        ProgramLogicGoal(
            roots=roots,
            goal_id="goal:discharged",
            family=GoalFamily.POSITIVE,
            disposition=GoalDisposition.DISCHARGED,
            positive_statement_ref="stmt:positive",
            proof_status=ProofStatus.SOLVER_CHECKED,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicPredictionError, match="negative_target"):
        ProgramLogicGoal(
            roots=roots,
            goal_id="goal:neg",
            family=GoalFamily.NEGATIVE,
            disposition=GoalDisposition.OPEN,
            positive_statement_ref="stmt:positive",
            invalidation_refs=("tree:one",),
        )


def test_memory_resource_type_facets_remain_distinct() -> None:
    with pytest.raises(ProgramLogicAuthorityError, match="memory facets cannot bind resource"):
        LogicFacetRef(
            facet_id="facet:bad",
            kind=LogicFacetKind.MEMORY,
            subject_symbol_id="symbol:x",
            contract_ref="resource:max-bytes",
        )
    with pytest.raises(ProgramLogicAuthorityError, match="resource facets cannot bind memory"):
        LogicFacetRef(
            facet_id="facet:bad",
            kind=LogicFacetKind.RESOURCE,
            subject_symbol_id="symbol:x",
            contract_ref="memory:ownership",
        )
    with pytest.raises(ProgramLogicAuthorityError, match="type facets cannot bind"):
        LogicFacetRef(
            facet_id="facet:bad",
            kind=LogicFacetKind.TYPE,
            subject_symbol_id="symbol:x",
            contract_ref="memory:ownership",
        )
    ok = LogicFacetRef(
        facet_id="facet:resource",
        kind=LogicFacetKind.RESOURCE,
        subject_symbol_id="symbol:x",
        contract_ref="resource:max-bytes",
    )
    assert ok.kind is LogicFacetKind.RESOURCE


def test_logic_gap_forbids_semantic_authority(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = LogicGap(
        roots=roots,
        gap_id="gap:one",
        goal_id="goal:one",
        missing_class=GapMissingClass.VALUE,
        disposition=GapDisposition.REQUIRED,
        observed_fact_ref="fact:observed",
        required_fact_ref="fact:required",
        discrepancy_ref="disc:one",
        candidate_source_routes=(SourceRouteKind.DATAFLOW, SourceRouteKind.VECTOR),
        invalidation_refs=("tree:one",),
    )
    assert gap.semantic_authority is False
    assert LogicGap.from_dict(gap.to_record()) == gap

    with pytest.raises(ProgramLogicAuthorityError, match="semantic authority"):
        LogicGap(
            roots=roots,
            gap_id="gap:bad",
            goal_id="goal:one",
            missing_class=GapMissingClass.PREMISE,
            disposition=GapDisposition.REQUIRED,
            observed_fact_ref="fact:observed",
            required_fact_ref="fact:required",
            discrepancy_ref="disc:one",
            semantic_authority=True,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicPredictionError, match="unknown_frontier"):
        LogicGap(
            roots=roots,
            gap_id="gap:front",
            goal_id="goal:one",
            missing_class=GapMissingClass.FRONTIER,
            disposition=GapDisposition.FRONTIER,
            observed_fact_ref="fact:observed",
            required_fact_ref="fact:required",
            discrepancy_ref="disc:one",
            invalidation_refs=("tree:one",),
        )


def test_tactician_plan_rejects_cycles_and_semantic_authority(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    a = _subgoal("subgoal:a", depends_on=("subgoal:b",))
    b = _subgoal("subgoal:b", depends_on=("subgoal:a",))
    with pytest.raises(ProgramLogicPredictionError, match="cycle"):
        TacticianSearchPlan(
            roots=roots,
            plan_id="plan:one",
            goal_ids=("goal:one",),
            ordered_source_routes=(
                SourceRouteKind.LOCAL_STATIC,
                SourceRouteKind.VECTOR,
            ),
            subgoals=(a, b),
            invalidation_refs=("tree:one",),
        )

    leaf = _subgoal("subgoal:leaf")
    parent = _subgoal("subgoal:parent", depends_on=("subgoal:leaf",))
    plan = TacticianSearchPlan(
        roots=roots,
        plan_id="plan:one",
        goal_ids=("goal:one",),
        ordered_source_routes=(
            SourceRouteKind.REVIEWED_CONTRACT,
            SourceRouteKind.LOCAL_STATIC,
            SourceRouteKind.VECTOR,
        ),
        selected_premise_ids=("premise:a",),
        excluded_premise_ids=("premise:poison",),
        subgoals=(leaf, parent),
        planner_id="tactician:code",
        invalidation_refs=("tree:one", "corpus:one"),
    )
    assert plan.semantic_authority is False
    assert TacticianSearchPlan.from_dict(plan.to_record()) == plan

    with pytest.raises(ProgramLogicAuthorityError, match="semantic authority"):
        TacticianSearchPlan(
            roots=roots,
            plan_id="plan:bad",
            goal_ids=("goal:one",),
            ordered_source_routes=(SourceRouteKind.LOCAL_STATIC,),
            semantic_authority=True,
            invalidation_refs=("tree:one",),
        )


def test_subgoal_source_authority_separated_from_scores_and_nominations() -> None:
    scored = _subgoal(
        "subgoal:vec",
        source_route=SourceRouteKind.VECTOR,
        source_authority=SourceAuthorityClass.NOMINATING,
    )
    assert scored.score_millipercent == 12_500
    assert scored.source_authority is SourceAuthorityClass.NOMINATING
    assert scored.proof_status is ProofStatus.UNPROVED

    with pytest.raises(ProgramLogicAuthorityError, match="authoritative source class"):
        _subgoal(
            "subgoal:bad",
            source_route=SourceRouteKind.LLM,
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
        )

    with pytest.raises(ProgramLogicAuthorityError, match="verified or validated"):
        LogicSubgoal(
            subgoal_id="subgoal:solver",
            goal_id="goal:one",
            disposition=SubgoalDisposition.PLANNED,
            claim_ref="claim:one",
            source_route=SourceRouteKind.SOLVER,
            source_authority=SourceAuthorityClass.NOMINATING,
            proof_status=ProofStatus.KERNEL_VERIFIED,
        )


def test_hypothesis_rejects_nomination_semantic_and_solver_only_proved(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    nominated = LogicHypothesis(
        roots=roots,
        hypothesis_id="hyp:one",
        target_goal_id="goal:one",
        disposition=HypothesisDisposition.NOMINATED,
        claimed_consequence_ref="consequence:value",
        value_ref="value:ctx",
        evidence_refs=("evidence:vector",),
        evidence_route_kinds=(SourceRouteKind.VECTOR, SourceRouteKind.TACTICIAN),
        selected_premise_ids=("premise:a",),
        source_authority=SourceAuthorityClass.NOMINATING,
        proof_status=ProofStatus.UNPROVED,
        nomination_score_millipercent=88_000,
        invalidation_refs=("tree:one",),
    )
    assert nominated.semantic_authority is False
    assert nominated.nomination_score_millipercent == 88_000
    assert LogicHypothesis.from_dict(nominated.to_record()) == nominated

    with pytest.raises(ProgramLogicAuthorityError, match="semantic authority"):
        LogicHypothesis(
            roots=roots,
            hypothesis_id="hyp:bad",
            target_goal_id="goal:one",
            disposition=HypothesisDisposition.NOMINATED,
            claimed_consequence_ref="consequence:value",
            evidence_route_kinds=(SourceRouteKind.LLM,),
            semantic_authority=True,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicAuthorityError, match="authoritative source class"):
        LogicHypothesis(
            roots=roots,
            hypothesis_id="hyp:bad-auth",
            target_goal_id="goal:one",
            disposition=HypothesisDisposition.NOMINATED,
            claimed_consequence_ref="consequence:value",
            evidence_route_kinds=(SourceRouteKind.KNOWLEDGE_GRAPH,),
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicAuthorityError, match="kernel_verified|solver-only"):
        LogicHypothesis(
            roots=roots,
            hypothesis_id="hyp:solver",
            target_goal_id="goal:one",
            disposition=HypothesisDisposition.PROVED,
            claimed_consequence_ref="consequence:value",
            evidence_route_kinds=(SourceRouteKind.REVIEWED_CONTRACT,),
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
            proof_status=ProofStatus.SOLVER_CHECKED,
            invalidation_refs=("tree:one",),
        )


def test_prediction_receipt_requires_kernel_not_solver_only(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    proved = LogicPredictionReceipt(
        roots=roots,
        receipt_id="pred:one",
        goal_id="goal:one",
        hypothesis_id="hyp:one",
        tactician_plan_id="plan:one",
        corpus_id="corpus:one",
        disposition=PredictionDisposition.PROVED,
        hammer_request_id="hammer:req",
        translation_id="translation:one",
        candidate_id="candidate:one",
        reconstruction_id="reconstruction:one",
        kernel_receipt_id="kernel:receipt",
        environment_receipt_id="env:receipt",
        derived_clause_ref="clause:admitted",
        source_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=ProofStatus.KERNEL_VERIFIED,
        automation_eligible=True,
        invalidation_refs=("tree:one", "corpus:one"),
    )
    assert LogicPredictionReceipt.from_dict(proved.to_record()) == proved

    with pytest.raises(ProgramLogicAuthorityError, match="kernel and reconstruction"):
        LogicPredictionReceipt(
            roots=roots,
            receipt_id="pred:solver",
            goal_id="goal:one",
            hypothesis_id="hyp:one",
            tactician_plan_id="plan:one",
            corpus_id="corpus:one",
            disposition=PredictionDisposition.PROVED,
            candidate_id="candidate:solver",
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
            proof_status=ProofStatus.KERNEL_VERIFIED,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicAuthorityError, match="countermodel validation"):
        LogicPredictionReceipt(
            roots=roots,
            receipt_id="pred:raw-cm",
            goal_id="goal:one",
            hypothesis_id="hyp:one",
            tactician_plan_id="plan:one",
            corpus_id="corpus:one",
            disposition=PredictionDisposition.VALIDATED_REFUTATION,
            proof_status=ProofStatus.VALIDATED_REFUTED,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicAuthorityError, match="corpus_id must match"):
        LogicPredictionReceipt(
            roots=roots,
            receipt_id="pred:stale-corpus",
            goal_id="goal:one",
            hypothesis_id="hyp:one",
            tactician_plan_id="plan:one",
            corpus_id="corpus:other",
            disposition=PredictionDisposition.INCONCLUSIVE,
            invalidation_refs=("tree:one",),
        )


def test_native_goal_binding_carries_snapshot_kernel_and_round_trip(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    binding = ProgramLogicNativeGoalBinding(
        roots=roots,
        binding_id="binding:one",
        logic_ir_obligation_id="obligation:logic-ir",
        premise_ids=("premise:a", "premise:b"),
        native_itp_id="itp:lean",
        goal_snapshot_id="goal-snapshot:exact",
        native_theorem_source_id="native-src:theorem",
        proof_hole_id="hole:single",
        kernel_id="kernel:lean4",
        semantic_round_trip=_round_trip(),
        disposition=NativeGoalDisposition.ROUND_TRIP_OK,
        import_ids=("import:Prelude",),
        source_position_id="pos:line-10",
        invalidation_refs=("tree:one", "toolchain:one"),
    )
    assert binding.goal_snapshot_id == "goal-snapshot:exact"
    assert binding.kernel_id == "kernel:lean4"
    assert binding.semantic_round_trip.disposition is NativeGoalDisposition.ROUND_TRIP_OK
    assert binding.environment_id == roots.environment_id
    assert ProgramLogicNativeGoalBinding.from_dict(binding.to_record()) == binding

    with pytest.raises(ProgramLogicAuthorityError, match="LogicIR claim"):
        ProgramLogicNativeGoalBinding(
            roots=roots,
            binding_id="binding:bad",
            logic_ir_obligation_id="obligation:other",
            premise_ids=("premise:a",),
            native_itp_id="itp:lean",
            goal_snapshot_id="goal-snapshot:exact",
            native_theorem_source_id="native-src:theorem",
            proof_hole_id="hole:single",
            kernel_id="kernel:lean4",
            semantic_round_trip=_round_trip(obligation_id="obligation:logic-ir"),
            disposition=NativeGoalDisposition.ROUND_TRIP_OK,
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicPredictionError, match="inconsistent"):
        ProgramLogicNativeGoalBinding(
            roots=roots,
            binding_id="binding:inconsistent",
            logic_ir_obligation_id="obligation:logic-ir",
            premise_ids=("premise:a",),
            native_itp_id="itp:lean",
            goal_snapshot_id="goal-snapshot:exact",
            native_theorem_source_id="native-src:theorem",
            proof_hole_id="hole:single",
            kernel_id="kernel:lean4",
            semantic_round_trip=_round_trip(
                disposition=NativeGoalDisposition.INCONSISTENT
            ),
            disposition=NativeGoalDisposition.INCONSISTENT,
            invalidation_refs=("tree:one",),
        )


def test_countermodel_separates_raw_diagnostic_from_replayed_rejection(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    diagnostic = CountermodelValidationReceipt(
        roots=roots,
        receipt_id="cm:diag",
        solver_countermodel_id="solver-cm:raw",
        translation_map_id="translation:one",
        originating_logic_ir_id="obligation:logic-ir",
        disposition=CountermodelDisposition.DIAGNOSTIC_ONLY,
        raw_diagnostic_refs=("diag:solver-model", "diag:assignment"),
        invalidation_refs=("tree:one",),
    )
    assert diagnostic.may_reject_hypothesis is False
    assert not diagnostic.replayed_rejection_evidence_refs

    validated = CountermodelValidationReceipt(
        roots=roots,
        receipt_id="cm:valid",
        solver_countermodel_id="solver-cm:raw",
        translation_map_id="translation:one",
        originating_logic_ir_id="obligation:logic-ir",
        disposition=CountermodelDisposition.VALIDATED,
        raw_diagnostic_refs=("diag:solver-model",),
        replayed_rejection_evidence_refs=("replay:logic-ir",),
        replay_method="deterministic_model_check",
        invalidation_refs=("tree:one", "policy:one"),
    )
    assert validated.may_reject_hypothesis is True
    assert CountermodelValidationReceipt.from_dict(validated.to_record()) == validated

    with pytest.raises(ProgramLogicAuthorityError, match="rejection evidence"):
        CountermodelValidationReceipt(
            roots=roots,
            receipt_id="cm:bad-diag",
            solver_countermodel_id="solver-cm:raw",
            translation_map_id="translation:one",
            originating_logic_ir_id="obligation:logic-ir",
            disposition=CountermodelDisposition.DIAGNOSTIC_ONLY,
            raw_diagnostic_refs=("diag:solver-model",),
            replayed_rejection_evidence_refs=("replay:forged",),
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicPredictionError, match="disjoint"):
        CountermodelValidationReceipt(
            roots=roots,
            receipt_id="cm:overlap",
            solver_countermodel_id="solver-cm:raw",
            translation_map_id="translation:one",
            originating_logic_ir_id="obligation:logic-ir",
            disposition=CountermodelDisposition.VALIDATED,
            raw_diagnostic_refs=("shared:ref",),
            replayed_rejection_evidence_refs=("shared:ref",),
            replay_method="replay",
            invalidation_refs=("tree:one",),
        )


def test_logic_guided_repair_packet_is_context_overlay_not_write_authority(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    packet = LogicGuidedRepairPacket(
        roots=roots,
        packet_id="overlay:one",
        admitted_prediction_id="pred:one",
        rpr_packet_id="rpr-packet:change-prop",
        rpr_plan_id="rpr-plan:atomic",
        rpr_plan_step_id="rpr-step:3",
        writer_lease_id="lease:writer",
        disposition=ContextOverlayDisposition.MODEL_REQUIRED,
        context_capsule_id="capsule:context",
        permitted_read_paths=("pkg/caller.py", "pkg/process.py"),
        permitted_write_paths=("pkg/caller.py",),
        forbidden_semantic_change_refs=("delta:forbidden",),
        postcondition_refs=("post:types",),
        validation_refs=("validation:fixed-point",),
        model_id="model:router",
        invalidation_refs=("tree:one", "lease:writer"),
    )
    assert packet.write_authority is False
    assert packet.semantic_authority is False
    assert LogicGuidedRepairPacket.from_dict(packet.to_record()) == packet

    with pytest.raises(ProgramLogicAuthorityError, match="write authority"):
        LogicGuidedRepairPacket(
            roots=roots,
            packet_id="overlay:bad",
            admitted_prediction_id="pred:one",
            rpr_packet_id="rpr-packet:change-prop",
            rpr_plan_id="rpr-plan:atomic",
            rpr_plan_step_id="rpr-step:3",
            writer_lease_id="lease:writer",
            disposition=ContextOverlayDisposition.DETERMINISTIC,
            context_capsule_id="capsule:context",
            write_authority=True,
            invalidation_refs=("tree:one",),
        )


def test_write_scope_requires_rpr_plan_and_lease(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(ProgramLogicAuthorityError, match="write scope|RPR plan"):
        LogicGuidedRepairPacket(
            roots=roots,
            packet_id="overlay:no-lease",
            admitted_prediction_id="pred:one",
            rpr_packet_id="rpr-packet:change-prop",
            rpr_plan_id="rpr-plan:atomic",
            rpr_plan_step_id="rpr-step:3",
            writer_lease_id="",
            disposition=ContextOverlayDisposition.DETERMINISTIC,
            context_capsule_id="capsule:context",
            permitted_write_paths=("pkg/caller.py",),
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicAuthorityError, match="write scope|RPR plan"):
        LogicGuidedRepairPacket(
            roots=roots,
            packet_id="overlay:no-plan",
            admitted_prediction_id="pred:one",
            rpr_packet_id="rpr-packet:change-prop",
            rpr_plan_id="",
            rpr_plan_step_id="rpr-step:3",
            writer_lease_id="lease:writer",
            disposition=ContextOverlayDisposition.DETERMINISTIC,
            context_capsule_id="capsule:context",
            permitted_write_paths=("pkg/caller.py",),
            invalidation_refs=("tree:one",),
        )

    with pytest.raises(ProgramLogicAuthorityError, match="write paths"):
        LogicGuidedRepairPacket(
            roots=roots,
            packet_id="overlay:abstain",
            admitted_prediction_id="pred:one",
            rpr_packet_id="rpr-packet:change-prop",
            rpr_plan_id="rpr-plan:atomic",
            rpr_plan_step_id="rpr-step:3",
            writer_lease_id="lease:writer",
            disposition=ContextOverlayDisposition.ABSTAINED,
            context_capsule_id="capsule:context",
            permitted_write_paths=("pkg/caller.py",),
            invalidation_refs=("tree:one",),
        )


def test_fixed_point_attachment_extends_not_replaces_completion(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    attachment = LogicFixedPointEvidenceAttachment(
        roots=roots,
        attachment_id="attach:one",
        completion_receipt_id="completion:propagation",
        disposition=FixedPointAttachmentDisposition.ATTACHED,
        iteration_count=2,
        goal_root_ids=("goal:one",),
        corpus_root_ids=("corpus:one",),
        tactician_plan_ids=("plan:one",),
        hammer_receipt_ids=("hammer:one",),
        prediction_receipt_ids=("pred:one",),
        original_consumer_coverage_ids=("consumer:one",),
        second_order_consumer_coverage_ids=("consumer:second",),
        finalize_receipt_id="finalize:one",
        invalidation_refs=("tree:one",),
    )
    assert attachment.replaces_completion is False
    assert (
        LogicFixedPointEvidenceAttachment.from_dict(attachment.to_record())
        == attachment
    )

    with pytest.raises(ProgramLogicAuthorityError, match="extend rather than replace"):
        LogicFixedPointEvidenceAttachment(
            roots=roots,
            attachment_id="attach:bad",
            completion_receipt_id="completion:propagation",
            disposition=FixedPointAttachmentDisposition.ATTACHED,
            iteration_count=1,
            finalize_receipt_id="finalize:one",
            replaces_completion=True,
            invalidation_refs=("tree:one",),
        )

    residual = LogicFixedPointEvidenceAttachment(
        roots=roots,
        attachment_id="attach:residual",
        completion_receipt_id="completion:propagation",
        disposition=FixedPointAttachmentDisposition.RESIDUAL,
        iteration_count=3,
        residual_logic_gap_ids=("gap:open",),
        invalidation_refs=("tree:one",),
    )
    assert residual.disposition is FixedPointAttachmentDisposition.RESIDUAL

    rolled = LogicFixedPointEvidenceAttachment(
        roots=roots,
        attachment_id="attach:rollback",
        completion_receipt_id="completion:propagation",
        disposition=FixedPointAttachmentDisposition.ROLLED_BACK,
        iteration_count=1,
        compensating_rollback_receipt_id="rollback:one",
        invalidation_refs=("tree:one",),
    )
    assert rolled.compensating_rollback_receipt_id == "rollback:one"


@pytest.mark.parametrize("bad", ["../escape.py", "/absolute.py", "."])
def test_paths_must_be_repository_relative(
    roots: ProgramLogicAuthorityRoots, bad: str
) -> None:
    with pytest.raises(ProgramLogicAuthorityError):
        LogicGuidedRepairPacket(
            roots=roots,
            packet_id="overlay:path",
            admitted_prediction_id="pred:one",
            rpr_packet_id="rpr-packet:change-prop",
            rpr_plan_id="rpr-plan:atomic",
            rpr_plan_step_id="rpr-step:3",
            writer_lease_id="lease:writer",
            disposition=ContextOverlayDisposition.DETERMINISTIC,
            context_capsule_id="capsule:context",
            permitted_read_paths=(bad,),
            invalidation_refs=("tree:one",),
        )


def test_closed_dispositions_are_exhaustive_enums() -> None:
    """Smoke-check that each disposition family is a closed string enum."""
    families = (
        GoalDisposition,
        GapDisposition,
        SubgoalDisposition,
        HypothesisDisposition,
        PredictionDisposition,
        NativeGoalDisposition,
        CountermodelDisposition,
        ContextOverlayDisposition,
        FixedPointAttachmentDisposition,
        SourceAuthorityClass,
        ProofStatus,
        SourceRouteKind,
    )
    for family in families:
        values = {item.value for item in family}
        assert values
        assert all(isinstance(item, str) for item in values)
        # Reconstructing an unknown value fails closed.
        with pytest.raises(ValueError):
            family("not_a_real_disposition")
