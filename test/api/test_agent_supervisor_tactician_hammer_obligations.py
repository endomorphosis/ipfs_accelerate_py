"""Fail-closed coverage for Tactician → proof-obligation lowering (LPR-011)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    HypothesisDisposition,
    LogicFacetKind,
    LogicFacetRef,
    LogicHypothesis,
    LogicSubgoal,
    NativeGoalDisposition,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicNativeGoalBinding,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from ipfs_accelerate_py.agent_supervisor.proof.tactician_hammer_obligations import (
    AssumptionBinding,
    ChangedAssumptionsError,
    CrossRootPremiseError,
    ExistingObligationLink,
    InconsistentAssumptionError,
    LoweringDisposition,
    LoweringFacetKind,
    NativeITPKind,
    ObligationContext,
    OmittedFacetError,
    ProgramLogicNativeGoalCompiler,
    ResidualSemanticKind,
    SourceDriftError,
    TacticianHammerObligationCompiler,
    TacticianHammerObligationError,
    TranslatorCapabilityBinding,
    UnauthorizedAxiomError,
    WrongTheoremError,
    lower_tactician_plan,
)
from ipfs_accelerate_py.agent_supervisor.validation.tactician_plan_gate import (
    TacticianPlanGate,
    TacticianPlanGateDisposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:fixture",
        objective_id="objective:fixture",
        trace_id="trace:fixture",
        change_id="change:fixture",
        consumer_id="consumer:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        overlay_id="overlay:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        corpus_id="corpus:fixture",
        model_id="model:fixture",
        translator_id="translator:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        environment_id="environment:fixture",
    )


def _facet(
    facet_id: str = "facet:type-context",
    *,
    kind: LogicFacetKind = LogicFacetKind.TYPE,
    subject: str = "symbol:process",
    contract_ref: str = "contract:Context",
    unsupported: bool = False,
) -> LogicFacetRef:
    return LogicFacetRef(
        facet_id=facet_id,
        kind=kind,
        subject_symbol_id=subject,
        contract_ref=contract_ref,
        unsupported=unsupported,
    )


def _goal(
    roots: ProgramLogicAuthorityRoots,
    *,
    goal_id: str = "goal:repair-caller",
    disposition: GoalDisposition = GoalDisposition.PLANNED,
    family: GoalFamily = GoalFamily.POSITIVE,
    required_facets: tuple[LogicFacetRef, ...] | None = None,
    unsupported_facets: tuple[LogicFacetRef, ...] = (),
    affected_symbol_ids: tuple[str, ...] = ("symbol:process",),
) -> ProgramLogicGoal:
    if required_facets is None:
        required_facets = (
            _facet("facet:input-context", kind=LogicFacetKind.TYPE, contract_ref="input:Context"),
            _facet(
                "facet:information-value",
                kind=LogicFacetKind.INFORMATION,
                contract_ref="information:value",
            ),
            _facet(
                "facet:output-result",
                kind=LogicFacetKind.TYPE,
                contract_ref="output:Result",
            ),
            _facet(
                "facet:totality",
                kind=LogicFacetKind.TYPE,
                contract_ref="totality:total",
            ),
            _facet(
                "facet:error-notfound",
                kind=LogicFacetKind.ERROR,
                contract_ref="error:NotFound",
            ),
            _facet(
                "facet:effect-fs",
                kind=LogicFacetKind.EFFECT,
                contract_ref="effect:filesystem",
            ),
            _facet(
                "facet:auth-read",
                kind=LogicFacetKind.AUTHORIZATION,
                contract_ref="auth:vfs.read",
            ),
            _facet(
                "facet:resource-stack",
                kind=LogicFacetKind.RESOURCE,
                contract_ref="resource:stack",
            ),
            _facet(
                "facet:state-ready",
                kind=LogicFacetKind.STATE,
                contract_ref="state:ready",
            ),
            _facet(
                "facet:schema-context",
                kind=LogicFacetKind.SCHEMA,
                contract_ref="schema:Context",
            ),
            _facet(
                "facet:placement-callsite",
                kind=LogicFacetKind.PLACEMENT,
                contract_ref="placement:callsite",
            ),
            _facet(
                "facet:ownership-borrowed",
                kind=LogicFacetKind.MEMORY,
                contract_ref="ownership:borrowed",
            ),
        )
    return ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=family,
        disposition=disposition,
        positive_statement_ref=f"stmt:{goal_id}",
        negative_target_ref=f"neg:{goal_id}",
        counterexample_target_ref=f"cex:{goal_id}",
        affected_symbol_ids=affected_symbol_ids,
        source_refs=("source:reviewed-contract",),
        required_facets=required_facets,
        unsupported_facets=unsupported_facets,
        assumption_refs=("assumption:stable-api",),
        assumption_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=ProofStatus.UNPROVED,
        invalidation_refs=(roots.tree_id,),
    )


def _subgoal(
    *,
    subgoal_id: str = "subgoal:prove-input",
    goal_id: str = "goal:repair-caller",
    claim_ref: str = "facet:input-context",
    depends_on: tuple[str, ...] = (),
    source_route: SourceRouteKind = SourceRouteKind.DATAFLOW,
    source_authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE,
) -> LogicSubgoal:
    return LogicSubgoal(
        subgoal_id=subgoal_id,
        goal_id=goal_id,
        disposition=SubgoalDisposition.PLANNED,
        claim_ref=claim_ref,
        depends_on=depends_on,
        source_route=source_route,
        source_authority=source_authority,
        score_millipercent=25_000,
    )


def _all_subgoals() -> tuple[LogicSubgoal, ...]:
    """One subgoal per separately lowerable facet clause."""

    pairs = (
        ("subgoal:prove-input", "facet:input-context"),
        ("subgoal:prove-information", "facet:information-value"),
        ("subgoal:prove-output", "facet:output-result"),
        ("subgoal:prove-totality", "facet:totality"),
        ("subgoal:prove-error", "facet:error-notfound"),
        ("subgoal:prove-effect", "facet:effect-fs"),
        ("subgoal:prove-auth", "facet:auth-read"),
        ("subgoal:prove-resource", "facet:resource-stack"),
        ("subgoal:prove-state", "facet:state-ready"),
        ("subgoal:prove-schema", "facet:schema-context"),
        ("subgoal:prove-placement", "facet:placement-callsite"),
        ("subgoal:prove-ownership", "facet:ownership-borrowed"),
    )
    return tuple(
        _subgoal(subgoal_id=subgoal_id, claim_ref=claim_ref)
        for subgoal_id, claim_ref in pairs
    )


def _plan(
    roots: ProgramLogicAuthorityRoots,
    **extra: object,
) -> TacticianSearchPlan:
    subgoals = extra.pop("subgoals", _all_subgoals())
    values: dict[str, object] = {
        "roots": roots,
        "plan_id": "plan:fixture",
        "goal_ids": ("goal:repair-caller",),
        "ordered_source_routes": (
            SourceRouteKind.LOCAL_STATIC,
            SourceRouteKind.DATAFLOW,
            SourceRouteKind.GRAPH,
        ),
        "query_refs": ("query:fixture",),
        "selected_premise_ids": ("premise:type-context",),
        "excluded_premise_ids": (),
        "exclusion_rationale_refs": (),
        "subgoals": subgoals,
        "planned_logic_family_refs": ("family:fol",),
        "translation_refs": ("translation:fol-to-lean@1",),
        "planner_id": "planner:fixture",
        "model_id": "model:fixture",
        "config_id": "config:fixture",
        "stop_policy_ref": "stop:code-tactician.default@1",
        "escalation_policy_ref": "escalation:code-tactician.default@1",
        "abstention_policy_ref": "abstention:code-tactician.default@1",
        "resource_policy_ref": "resource:code-tactician.default@1",
        "invalidation_refs": (roots.tree_id,),
    }
    values.update(extra)
    return TacticianSearchPlan(**values)  # type: ignore[arg-type]


def _premise(
    roots: ProgramLogicAuthorityRoots,
    premise_id: str = "premise:type-context",
    *,
    source_class: PremiseSourceClass = PremiseSourceClass.REVIEWED_CONTRACT,
    statement_ref: str = "stmt:type-context",
    expectation_authority: bool = True,
) -> ProgramLogicPremise:
    return ProgramLogicPremise(
        roots=roots,
        premise_id=premise_id,
        source_class=source_class,
        statement_ref=statement_ref,
        statement_digest="sha256:" + ("11" * 32),
        lowering_ref=f"lower:{premise_id}",
        expectation_authority=expectation_authority,
        tree_identity=roots.tree_id,
        graph_identity=roots.graph_id,
    )


def _corpus(
    roots: ProgramLogicAuthorityRoots,
    premises: tuple[ProgramLogicPremise, ...] | None = None,
) -> ProgramLogicPremiseCorpus:
    if premises is None:
        premises = (_premise(roots),)
    return ProgramLogicPremiseCorpus(
        roots=roots,
        premises=premises,
        consistency_disposition=ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK,
    )


def _hypothesis(
    roots: ProgramLogicAuthorityRoots,
    *,
    hypothesis_id: str = "hypothesis:reuse-local-ctx",
    target_goal_id: str = "goal:repair-caller",
    disposition: HypothesisDisposition = HypothesisDisposition.PLAN_ADMITTED,
    claimed_consequence_ref: str = "consequence:input-context",
    construction_ref: str = "construction:local_ctx",
    placement_ref: str = "",
    value_ref: str = "value:local_ctx",
    selected_premise_ids: tuple[str, ...] = ("premise:type-context",),
    unsupported_flags: tuple[str, ...] = (),
) -> LogicHypothesis:
    return LogicHypothesis(
        roots=roots,
        hypothesis_id=hypothesis_id,
        target_goal_id=target_goal_id,
        disposition=disposition,
        claimed_consequence_ref=claimed_consequence_ref,
        construction_ref=construction_ref,
        placement_ref=placement_ref,
        value_ref=value_ref,
        evidence_refs=("evidence:fixture", "facet:input-context"),
        evidence_route_kinds=(
            SourceRouteKind.DATAFLOW,
            SourceRouteKind.LOCAL_STATIC,
        ),
        selected_premise_ids=selected_premise_ids,
        counterexample_target_ref="cex:goal:repair-caller",
        source_authority=SourceAuthorityClass.NOMINATING,
        proof_status=ProofStatus.UNPROVED,
        unsupported_flags=unsupported_flags,
        nomination_score_millipercent=12_500,
        invalidation_refs=(roots.tree_id,),
    )


def _capability(
    *,
    semantics: tuple[str, ...] = ("ir", "fol", "ownership", "memory"),
    itps: tuple[str, ...] = ("lean", "coq", "isabelle"),
    translator_id: str = "translator:fixture",
) -> TranslatorCapabilityBinding:
    return TranslatorCapabilityBinding(
        capability_id="datasets.logic_ir",
        capability_revision="logic:fixture-1",
        translator_id=translator_id,
        reconstruction_compatible=True,
        supported_semantics=semantics,
        supported_itps=itps,
    )


def _context(
    *,
    semantics: tuple[str, ...] = ("ir", "fol", "ownership", "memory"),
    native_itp: NativeITPKind = NativeITPKind.LEAN,
    links: tuple[ExistingObligationLink, ...] = (),
) -> ObligationContext:
    return ObligationContext(
        capability=_capability(semantics=semantics),
        assumptions=(
            AssumptionBinding(
                assumption_id="assumption:stable-api",
                kind="reviewed_assumption",
                evidence_ref="evidence:assumption-stable-api",
                authority=SourceAuthorityClass.AUTHORITATIVE,
            ),
        ),
        existing_obligation_links=links,
        native_itp=native_itp,
        translation_map_id="translation-map:fol-to-lean@1",
        hammer_premise_selection_id="hammer-premises:fixture",
    )


def _admitted_bundle(roots: ProgramLogicAuthorityRoots) -> dict[str, object]:
    plan = _plan(roots)
    goals = (_goal(roots),)
    hypotheses = (_hypothesis(roots),)
    corpus = _corpus(roots)
    receipt = TacticianPlanGate().require_valid(
        plan=plan,
        goals=goals,
        candidates=hypotheses,
        corpus=corpus,
        current_roots=roots,
    )
    assert receipt.disposition is TacticianPlanGateDisposition.ADMITTED
    return {
        "gate_receipt": receipt,
        "plan": plan,
        "goals": goals,
        "hypotheses": hypotheses,
        "corpus": corpus,
        "context": _context(),
        "current_roots": roots,
    }


# ---------------------------------------------------------------------------
# Happy path: separate facet lowering + full bindings
# ---------------------------------------------------------------------------


def test_lowers_input_information_output_totality_error_effect_auth_resource_state_schema_placement_ownership(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    bundle = _admitted_bundle(roots)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]

    kinds = {item.kind for item in compilation.obligations}
    expected = {
        LoweringFacetKind.INPUT,
        LoweringFacetKind.INFORMATION,
        LoweringFacetKind.OUTPUT,
        LoweringFacetKind.TOTALITY,
        LoweringFacetKind.ERROR,
        LoweringFacetKind.EFFECT,
        LoweringFacetKind.AUTH,
        LoweringFacetKind.RESOURCE,
        LoweringFacetKind.STATE,
        LoweringFacetKind.SCHEMA,
        LoweringFacetKind.PLACEMENT,
        LoweringFacetKind.OWNERSHIP,
    }
    assert expected <= kinds
    assert compilation.disposition in {
        LoweringDisposition.LOWERED,
        LoweringDisposition.PARTIAL,
    }

    for obligation in compilation.obligations:
        claim = obligation.claim
        assert claim.premise_ids
        assert claim.source_ids
        assert claim.assumption_ids == ("assumption:stable-api",)
        assert claim.repository_id == roots.repository_id
        assert claim.tree_id == roots.tree_id
        assert claim.corpus_id == roots.corpus_id
        assert claim.translator_id == roots.translator_id
        assert claim.toolchain_id == roots.toolchain_id
        assert claim.policy_id == roots.policy_id
        assert claim.environment_id == roots.environment_id
        assert claim.capability_id == "datasets.logic_ir"
        assert claim.capability_revision == "logic:fixture-1"
        assert obligation.code_obligation.metadata["claim_id"] == claim.claim_id
        assert obligation.code_obligation.task_id == "LPR-011"
        # No natural-language axioms in statements — opaque claim identity only.
        assert " " not in claim.predicate
        assert claim.predicate.startswith("lower:")


def test_each_obligation_has_native_binding_snapshot_source_hole_and_round_trip(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    bundle = _admitted_bundle(roots)
    compilation = lower_tactician_plan(**bundle)  # type: ignore[arg-type]

    assert len(compilation.native_bindings) == len(compilation.obligations)
    assert len(compilation.native_sources) == len(compilation.obligations)
    assert len(compilation.goal_snapshots) == len(compilation.obligations)

    native = ProgramLogicNativeGoalCompiler()
    for obligation, source, snapshot, binding in zip(
        compilation.obligations,
        compilation.native_sources,
        compilation.goal_snapshots,
        compilation.native_bindings,
        strict=True,
    ):
        assert source.claim_id == obligation.claim.claim_id
        assert source.proof_hole_marker == "sorry"
        assert source.source_text.count("sorry") == 1
        assert source.kernel_id == "kernel:lean4"
        assert source.toolchain_id == roots.toolchain_id
        assert snapshot.snapshot_id == binding.goal_snapshot_id
        assert binding.logic_ir_obligation_id == obligation.claim.claim_id
        assert binding.kernel_id == "kernel:lean4"
        assert binding.disposition is NativeGoalDisposition.ROUND_TRIP_OK
        assert (
            binding.semantic_round_trip.disposition
            is NativeGoalDisposition.ROUND_TRIP_OK
        )
        # Independent round-trip proves same LogicIR claim.
        recovered = native.round_trip_claim_id(source.source_text, itp=source.itp)
        assert recovered == obligation.claim.claim_id
        assert isinstance(binding, ProgramLogicNativeGoalBinding)


def test_deterministic_compilation_is_byte_stable(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    bundle = _admitted_bundle(roots)
    compiler = TacticianHammerObligationCompiler()
    first = compiler.compile(**bundle)  # type: ignore[arg-type]
    second = compiler.compile(**bundle)  # type: ignore[arg-type]
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.compilation_id == second.compilation_id
    assert first.obligation_ids == second.obligation_ids


def test_existing_contract_repair_and_change_propagation_links_are_carried(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    links = (
        ExistingObligationLink(
            interface="ContractRepairObligationCompilation",
            compilation_id="cr-compilation:one",
            obligation_ids=("cr-ob:1",),
            kind_refs=("caller_implies_receiver_precondition",),
        ),
        ExistingObligationLink(
            interface="ChangePropagationObligation",
            compilation_id="cp-compilation:one",
            obligation_ids=("cp-ob:1",),
            kind_refs=("information_sufficiency",),
        ),
    )
    bundle = _admitted_bundle(roots)
    bundle["context"] = _context(links=links)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]
    assert compilation.existing_obligation_links == tuple(
        sorted(links, key=lambda item: (item.interface, item.compilation_id))
    )
    for obligation in compilation.obligations:
        assert set(obligation.existing_obligation_refs) == {
            "cr-compilation:one",
            "cp-compilation:one",
        }


# ---------------------------------------------------------------------------
# Residuals for unsupported semantics
# ---------------------------------------------------------------------------


def test_lifetime_without_translator_support_is_typed_residual(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    lifetime = _facet(
        "facet:lifetime-borrow",
        kind=LogicFacetKind.LIFETIME,
        contract_ref="lifetime:borrow",
    )
    goal = _goal(
        roots,
        required_facets=(
            _facet("facet:input-context", kind=LogicFacetKind.TYPE, contract_ref="input:Context"),
            lifetime,
        ),
    )
    plan = _plan(
        roots,
        subgoals=(
            _subgoal(subgoal_id="subgoal:input", claim_ref="facet:input-context"),
            _subgoal(
                subgoal_id="subgoal:lifetime",
                claim_ref="facet:lifetime-borrow",
            ),
        ),
    )
    hypotheses = (_hypothesis(roots),)
    corpus = _corpus(roots)
    receipt = TacticianPlanGate().require_valid(
        plan=plan,
        goals=(goal,),
        candidates=hypotheses,
        corpus=corpus,
        current_roots=roots,
    )
    # Capability admits FOL/IR but not lifetime.
    context = _context(semantics=("ir", "fol"))
    compilation = TacticianHammerObligationCompiler().compile(
        receipt, plan, (goal,), hypotheses, corpus, context, current_roots=roots
    )
    residual_kinds = {item.kind for item in compilation.residuals}
    assert ResidualSemanticKind.LIFETIME in residual_kinds
    assert any(item.kind is LoweringFacetKind.INPUT for item in compilation.obligations)
    assert compilation.disposition is LoweringDisposition.PARTIAL


def test_higher_order_dependent_dynamic_native_concurrency_remain_residuals(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    goal = _goal(
        roots,
        required_facets=(
            _facet("facet:input-context", kind=LogicFacetKind.TYPE, contract_ref="input:Context"),
        ),
    )
    plan = _plan(
        roots,
        subgoals=(
            _subgoal(subgoal_id="subgoal:input", claim_ref="facet:input-context"),
            _subgoal(
                subgoal_id="subgoal:ho",
                claim_ref="claim:higher_order-callback",
            ),
            _subgoal(
                subgoal_id="subgoal:dep",
                claim_ref="claim:dependent-type",
            ),
            _subgoal(
                subgoal_id="subgoal:dyn",
                claim_ref="claim:dynamic-dispatch",
            ),
            _subgoal(
                subgoal_id="subgoal:native",
                claim_ref="claim:native-ffi",
            ),
            _subgoal(
                subgoal_id="subgoal:conc",
                claim_ref="claim:concurrency-lock",
            ),
        ),
    )
    hypotheses = (_hypothesis(roots),)
    corpus = _corpus(roots)
    receipt = TacticianPlanGate().require_valid(
        plan=plan,
        goals=(goal,),
        candidates=hypotheses,
        corpus=corpus,
        current_roots=roots,
    )
    compilation = TacticianHammerObligationCompiler().compile(
        receipt, plan, (goal,), hypotheses, corpus, _context(), current_roots=roots
    )
    residual_kinds = {item.kind for item in compilation.residuals}
    assert {
        ResidualSemanticKind.HIGHER_ORDER,
        ResidualSemanticKind.DEPENDENT,
        ResidualSemanticKind.DYNAMIC,
        ResidualSemanticKind.NATIVE,
        ResidualSemanticKind.CONCURRENCY,
    } <= residual_kinds


def test_explicit_unsupported_facet_is_residual_not_axiom(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    unsupported = _facet(
        "facet:reflection",
        kind=LogicFacetKind.STATE,
        contract_ref="state:reflection",
        unsupported=True,
    )
    goal = _goal(
        roots,
        required_facets=(
            _facet("facet:input-context", kind=LogicFacetKind.TYPE, contract_ref="input:Context"),
        ),
        unsupported_facets=(unsupported,),
    )
    plan = _plan(
        roots,
        subgoals=(
            _subgoal(subgoal_id="subgoal:input", claim_ref="facet:input-context"),
            _subgoal(
                subgoal_id="subgoal:reflection",
                claim_ref="facet:reflection",
                source_route=SourceRouteKind.LOCAL_STATIC,
            ),
        ),
    )
    hypotheses = (_hypothesis(roots),)
    corpus = _corpus(roots)
    receipt = TacticianPlanGate().require_valid(
        plan=plan,
        goals=(goal,),
        candidates=hypotheses,
        corpus=corpus,
        current_roots=roots,
    )
    compilation = TacticianHammerObligationCompiler().compile(
        receipt, plan, (goal,), hypotheses, corpus, _context(), current_roots=roots
    )
    assert any(
        item.kind is ResidualSemanticKind.UNSUPPORTED_FACET
        for item in compilation.residuals
    )
    assert all(
        item.semantic_authority is False for item in compilation.residuals
    )


# ---------------------------------------------------------------------------
# Fail-closed rejections
# ---------------------------------------------------------------------------


def test_rejects_natural_language_assumption_axioms(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(UnauthorizedAxiomError):
        AssumptionBinding(
            assumption_id="assumption:nl",
            kind="model_hypothesis_assumption",
            evidence_ref="llm:generated-text",
        )


def test_rejects_cross_root_premises(roots: ProgramLogicAuthorityRoots) -> None:
    bundle = _admitted_bundle(roots)
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:other",
        objective_id="objective:fixture",
        trace_id="trace:fixture",
        change_id="change:fixture",
        consumer_id="consumer:fixture",
        forest_id="forest:fixture",
        tree_id="tree:other",
        overlay_id="overlay:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        corpus_id="corpus:fixture",
        model_id="model:fixture",
        translator_id="translator:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        environment_id="environment:fixture",
    )
    with pytest.raises(CrossRootPremiseError):
        TacticianHammerObligationCompiler().compile(
            bundle["gate_receipt"],  # type: ignore[arg-type]
            bundle["plan"],  # type: ignore[arg-type]
            bundle["goals"],  # type: ignore[arg-type]
            bundle["hypotheses"],  # type: ignore[arg-type]
            bundle["corpus"],  # type: ignore[arg-type]
            bundle["context"],  # type: ignore[arg-type]
            current_roots=other,
        )


def test_rejects_gate_that_may_not_lower(roots: ProgramLogicAuthorityRoots) -> None:
    plan = _plan(roots, selected_premise_ids=("premise:missing",))
    goals = (_goal(roots),)
    hypotheses = (_hypothesis(roots),)
    corpus = _corpus(roots)
    receipt = TacticianPlanGate().evaluate(
        plan=plan,
        goals=goals,
        candidates=hypotheses,
        corpus=corpus,
        current_roots=roots,
    )
    assert receipt.disposition is not TacticianPlanGateDisposition.ADMITTED
    with pytest.raises(TacticianHammerObligationError, match="cannot lower"):
        TacticianHammerObligationCompiler().compile(
            receipt, plan, goals, hypotheses, corpus, _context(), current_roots=roots
        )


def test_rejects_omitted_required_facets_when_not_covered(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    # Admit a minimal plan, then present a drifted goal inventory that adds a
    # required facet.  A defective collector that drops that facet (neither
    # lowering nor residualizing it) must hard-fail with OmittedFacetError.
    goal = _goal(
        roots,
        required_facets=(
            _facet("facet:input-context", kind=LogicFacetKind.TYPE, contract_ref="input:Context"),
        ),
    )
    plan = _plan(
        roots,
        subgoals=(_subgoal(subgoal_id="subgoal:input", claim_ref="facet:input-context"),),
    )
    hypotheses = (_hypothesis(roots),)
    corpus = _corpus(roots)
    receipt = TacticianPlanGate().require_valid(
        plan=plan,
        goals=(goal,),
        candidates=hypotheses,
        corpus=corpus,
        current_roots=roots,
    )
    drifted = _goal(
        roots,
        required_facets=(
            _facet("facet:input-context", kind=LogicFacetKind.TYPE, contract_ref="input:Context"),
            _facet(
                "facet:auth-extra",
                kind=LogicFacetKind.AUTHORIZATION,
                contract_ref="auth:extra",
            ),
        ),
    )

    class DropAuthFacet(TacticianHammerObligationCompiler):
        def _collect_work_items(self, **kwargs):  # type: ignore[no-untyped-def]
            items = super()._collect_work_items(**kwargs)
            return [
                item
                for item in items
                if not (
                    item["facet"] is not None
                    and item["facet"].facet_id == "facet:auth-extra"
                )
            ]

    with pytest.raises(OmittedFacetError, match="facet:auth-extra"):
        DropAuthFacet().compile(
            receipt,
            plan,
            (drifted,),
            hypotheses,
            corpus,
            _context(),
            current_roots=roots,
        )


def test_rejects_wrong_theorem_source(roots: ProgramLogicAuthorityRoots) -> None:
    bundle = _admitted_bundle(roots)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]
    claim = compilation.obligations[0].claim
    native = ProgramLogicNativeGoalCompiler()
    with pytest.raises(WrongTheoremError):
        native.compile(
            claim,
            bundle["context"],  # type: ignore[arg-type]
            roots=roots,
            provided_source_text=(
                "-- import_id=import:Init\n"
                "-- assumption_id=assumption:stable-api\n"
                "-- theorem_id=thm:wrong\n"
                f"-- claim_id={claim.claim_id}\n"
                f"theorem thm:wrong : LogicIRClaim[{claim.claim_id}] := by\n"
                "  sorry\n"
            ),
            expected_theorem_id="thm:expected-other",
        )


def test_rejects_changed_assumptions_in_native_source(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    bundle = _admitted_bundle(roots)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]
    claim = compilation.obligations[0].claim
    native = ProgramLogicNativeGoalCompiler()
    with pytest.raises(ChangedAssumptionsError):
        native.compile(
            claim,
            bundle["context"],  # type: ignore[arg-type]
            roots=roots,
            expected_assumption_ids=("assumption:different",),
        )


def test_rejects_source_drift_when_claim_id_removed(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    bundle = _admitted_bundle(roots)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]
    claim = compilation.obligations[0].claim
    native = ProgramLogicNativeGoalCompiler()
    with pytest.raises((SourceDriftError, WrongTheoremError)):
        native.compile(
            claim,
            bundle["context"],  # type: ignore[arg-type]
            roots=roots,
            provided_source_text=(
                "-- import_id=import:Init\n"
                "-- assumption_id=assumption:stable-api\n"
                "-- theorem_id=thm:drifted\n"
                "-- claim_id=baguqeera_wrong_claim_identity\n"
                "theorem thm:drifted : LogicIRClaim[baguqeera_wrong_claim_identity] := by\n"
                "  sorry\n"
            ),
            expected_theorem_id="thm:drifted",
        )


def test_rejects_changed_imports(roots: ProgramLogicAuthorityRoots) -> None:
    bundle = _admitted_bundle(roots)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]
    claim = compilation.obligations[0].claim
    native = ProgramLogicNativeGoalCompiler()
    with pytest.raises(ChangedAssumptionsError):
        native.compile(
            claim,
            bundle["context"],  # type: ignore[arg-type]
            roots=roots,
            expected_import_ids=("import:Other",),
        )


def test_rejects_inconsistent_duplicate_assumption_evidence(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(InconsistentAssumptionError):
        ObligationContext(
            capability=_capability(),
            assumptions=(
                AssumptionBinding(
                    assumption_id="assumption:stable-api",
                    kind="reviewed_assumption",
                    evidence_ref="evidence:a",
                ),
                AssumptionBinding(
                    assumption_id="assumption:stable-api",
                    kind="reviewed_assumption",
                    evidence_ref="evidence:b",
                ),
            ),
        )


def test_rejects_plan_content_drift_from_gate_receipt(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    bundle = _admitted_bundle(roots)
    drifted_plan = _plan(roots, planner_id="planner:drifted")
    with pytest.raises(TacticianHammerObligationError, match="content identity drifted"):
        TacticianHammerObligationCompiler().compile(
            bundle["gate_receipt"],  # type: ignore[arg-type]
            drifted_plan,
            bundle["goals"],  # type: ignore[arg-type]
            bundle["hypotheses"],  # type: ignore[arg-type]
            bundle["corpus"],  # type: ignore[arg-type]
            bundle["context"],  # type: ignore[arg-type]
            current_roots=roots,
        )


# ---------------------------------------------------------------------------
# Native ITP variants (Lean / Coq / Isabelle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "itp,hole",
    [
        (NativeITPKind.LEAN, "sorry"),
        (NativeITPKind.COQ, "Admitted."),
        (NativeITPKind.ISABELLE, "sorry"),
    ],
)
def test_native_compiler_emits_single_goal_source_per_itp(
    roots: ProgramLogicAuthorityRoots,
    itp: NativeITPKind,
    hole: str,
) -> None:
    bundle = _admitted_bundle(roots)
    bundle["context"] = _context(native_itp=itp)
    compilation = TacticianHammerObligationCompiler().compile(**bundle)  # type: ignore[arg-type]
    for source in compilation.native_sources:
        assert source.itp is itp
        assert source.proof_hole_marker == hole
        assert source.source_text.count(hole) == 1
        assert source.kernel_id == {
            NativeITPKind.LEAN: "kernel:lean4",
            NativeITPKind.COQ: "kernel:coq",
            NativeITPKind.ISABELLE: "kernel:isabelle",
        }[itp]


def test_capability_must_admit_logic_ir_semantics(roots: ProgramLogicAuthorityRoots) -> None:
    with pytest.raises(TacticianHammerObligationError, match="LogicIR"):
        TranslatorCapabilityBinding(
            capability_id="datasets.logic_ir",
            capability_revision="logic:1",
            translator_id="translator:fixture",
            supported_semantics=("embedding_only",),
        )
