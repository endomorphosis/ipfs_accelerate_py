"""Conformance tests for program-logic goal compilation (LPR-006)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    GraphNodeRef,
    GraphProvenance,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    CallRequirementContract,
    EvidenceReference,
    MemorySafetyDisposition,
    MemorySafetyFacet,
    SourceSpan,
    TraceDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_goal_compiler import (
    PRODUCER_ID,
    CompilationDisposition,
    GoalDiagnosticKind,
    GoalObligationKind,
    GoalSourceBinding,
    GoalSourceKind,
    ProgramLogicGoalCompilation,
    ProgramLogicGoalCompiler,
    ProgramLogicGoalCompilerAuthorityError,
    ProgramLogicGoalCompilerError,
    ProseGoalNomination,
    all_obligation_kinds,
    all_source_kinds,
    compile_program_logic_goals,
    delta_obligation_kind,
    is_nominating_source,
    obligation_facet_kind,
    source_authority_for_kind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    LogicFacetKind,
    ProgramLogicAuthorityRoots,
    SourceAuthorityClass,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logic_roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:lpr-006",
        objective_id="objective:lpr-006",
        trace_id="trace:lpr-006",
        change_id="change:lpr-006",
        consumer_id="consumer:primary",
        forest_id="forest:lpr-006",
        tree_id="tree:candidate",
        overlay_id="overlay:lpr-006",
        graph_id="graph:lpr-006",
        index_id="index:lpr-006",
        corpus_id="corpus:lpr-006",
        model_id="model:lpr-006",
        translator_id="translator:lpr-006",
        toolchain_id="toolchain:lpr-006",
        policy_id="policy:lpr-006",
        environment_id="environment:lpr-006",
    )


@pytest.fixture
def repair_roots() -> AuthorityRoots:
    return AuthorityRoots(
        repository_id="repository:lpr-006",
        forest_id="forest:lpr-006",
        tree_id="tree:candidate",
        graph_id="graph:lpr-006",
        index_id="index:lpr-006",
        model_id="model:lpr-006",
        config_id="config:lpr-006",
        translator_id="translator:lpr-006",
        toolchain_id="toolchain:lpr-006",
        policy_id="policy:lpr-006",
    )


@pytest.fixture
def prop_roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:lpr-006",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:lpr-006",
        index_id="index:lpr-006",
        model_id="model:lpr-006",
        config_id="config:lpr-006",
        translator_id="translator:lpr-006",
        toolchain_id="toolchain:lpr-006",
        policy_id="policy:lpr-006",
    )


def _evidence(artifact_id: str = "evidence:one") -> EvidenceReference:
    return EvidenceReference(
        "resolver_receipt", artifact_id, "locator:call", "test.producer"
    )


def _span(path: str = "pkg/caller.py") -> SourceSpan:
    return SourceSpan(path, 10, 40, "blob:caller")


def _trace(
    roots: AuthorityRoots,
    *,
    disposition: TraceDisposition = TraceDisposition.RESOLVED_MISMATCH,
) -> BrokenContractTrace:
    evidence = _evidence()
    target = (
        SourceSpan("pkg/receiver.py", 4, 22, "blob:receiver")
        if disposition is TraceDisposition.RESOLVED_MISMATCH
        else None
    )
    return BrokenContractTrace(
        roots=roots,
        caller_span=_span(),
        caller_symbol_id="symbol:caller",
        receiver_reference="legacy.send",
        disposition=disposition,
        target_span=target,
        evidence_refs=(evidence,),
        graph_frontier_refs=("frontier:imports",)
        if disposition is TraceDisposition.DYNAMIC
        else (),
    )


def _call_requirement(roots: AuthorityRoots) -> CallRequirementContract:
    evidence = _evidence("evidence:requirement")
    return CallRequirementContract(
        roots=roots,
        trace_id="trace:call-1",
        caller_span=_span(),
        requirement_refs=(evidence,),
        evidence_refs=(evidence,),
        unsupported_clause_refs=("unsupported:reflection",),
    )


def _node(symbol_id: str = "symbol:consumer") -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{symbol_id}",
        kind="function",
        path="pkg/consumer.py",
        symbol_id=symbol_id,
        artifact_id="blob:consumer",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def _delta(
    roots: PropagationAuthorityRoots,
    *,
    clauses: tuple[ContractClauseDelta, ...] | None = None,
) -> ProgramContractDelta:
    if clauses is None:
        clauses = (
            ContractClauseDelta(
                clause_id="clause:param-add",
                kind=DeltaKind.PARAMETER_ADD,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:process",
                after_contract_ref="contract:after:process",
                reason="added required context parameter",
            ),
            ContractClauseDelta(
                clause_id="clause:result",
                kind=DeltaKind.RESULT_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:result",
                after_contract_ref="contract:after:result",
            ),
            ContractClauseDelta(
                clause_id="clause:nullability",
                kind=DeltaKind.NULLABILITY_CHANGE,
                disposition=DeltaDisposition.BEHAVIORAL,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:null",
                after_contract_ref="contract:after:null",
            ),
            ContractClauseDelta(
                clause_id="clause:error",
                kind=DeltaKind.ERROR_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:err",
                after_contract_ref="contract:after:err",
            ),
            ContractClauseDelta(
                clause_id="clause:effect",
                kind=DeltaKind.EFFECT_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:effect",
                after_contract_ref="contract:after:effect",
            ),
            ContractClauseDelta(
                clause_id="clause:auth",
                kind=DeltaKind.AUTHORIZATION_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:auth",
                after_contract_ref="contract:after:auth",
            ),
            ContractClauseDelta(
                clause_id="clause:resource",
                kind=DeltaKind.RESOURCE_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:res",
                after_contract_ref="contract:after:res",
            ),
            ContractClauseDelta(
                clause_id="clause:temporal",
                kind=DeltaKind.TEMPORAL_STATE_CHANGE,
                disposition=DeltaDisposition.BEHAVIORAL,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:temp",
                after_contract_ref="contract:after:temp",
            ),
            ContractClauseDelta(
                clause_id="clause:schema",
                kind=DeltaKind.SCHEMA_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:schema_consumers",
                before_contract_ref="contract:before:schema",
                after_contract_ref="contract:after:schema",
            ),
            ContractClauseDelta(
                clause_id="clause:ctor",
                kind=DeltaKind.CONSTRUCTOR_INTRO,
                disposition=DeltaDisposition.BEHAVIORAL,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                after_contract_ref="contract:after:ctor",
            ),
            ContractClauseDelta(
                clause_id="clause:ser",
                kind=DeltaKind.SERIALIZATION_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:schema_consumers",
                before_contract_ref="contract:before:ser",
                after_contract_ref="contract:after:ser",
            ),
            ContractClauseDelta(
                clause_id="clause:reg",
                kind=DeltaKind.SYMBOL_REGISTRATION,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:registration",
                after_contract_ref="contract:after:reg",
            ),
            ContractClauseDelta(
                clause_id="clause:place",
                kind=DeltaKind.SYMBOL_MOVE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:place",
                after_contract_ref="contract:after:place",
            ),
            ContractClauseDelta(
                clause_id="clause:memory",
                kind=DeltaKind.MEMORY_FACET_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:memory",
                before_contract_ref="contract:before:mem",
                after_contract_ref="contract:after:mem",
            ),
            ContractClauseDelta(
                clause_id="clause:compat",
                kind=DeltaKind.PARAMETER_RENAME,
                disposition=DeltaDisposition.COMPATIBLE,
                subject_symbol_id="symbol:process",
                consumer_domain="domain:python_callers",
                before_contract_ref="contract:before:rename",
                after_contract_ref="contract:after:rename",
            ),
        )
    return ProgramContractDelta(
        roots=roots,
        change_set_id="change-set:lpr-006",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before:process",
        after_contract_ref="contract:after:process",
        clauses=clauses,
        evidence_refs=("evidence:delta",),
    )


def _consumer(
    roots: PropagationAuthorityRoots,
    *,
    consumer_id: str = "consumer:primary",
    disposition: ConsumerDisposition = ConsumerDisposition.MIGRATE,
    clause_ids: tuple[str, ...] = ("clause:param-add", "clause:result"),
) -> ConsumerMigrationObligation:
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=f"obligation:{consumer_id}",
        consumer_id=consumer_id,
        delta_id="delta:lpr-006",
        disposition=disposition,
        clause_ids=clause_ids,
        node=_node(f"symbol:{consumer_id}"),
        proof_refs=("proof:plan",) if disposition is ConsumerDisposition.MIGRATE else (),
        invalidation_refs=("tree:candidate",),
    )


def _missing(roots: PropagationAuthorityRoots) -> MissingInputRequirement:
    return MissingInputRequirement(
        roots=roots,
        requirement_id="missing:context",
        obligation_id="obligation:consumer:primary",
        clause_id="clause:param-add",
        parameter_name="context",
        type_ref="type:SupportContext",
        nullability="non_null",
        information_content_ref="info:request-context",
        construction_precondition_refs=("pre:from_request",),
        result_postcondition_refs=("post:valid_context",),
        allowed_error_refs=("err:ContextError",),
        effect_refs=("effect:none",),
        capability_refs=("cap:context.read",),
        authorization_refs=("auth:session",),
        resource_refs=("res:memory_bound",),
        ownership_refs=("own:caller",),
        propagation_depth_bound=4,
        proof_refs=("proof:requirement",),
    )


def _behavior(
    roots: PropagationAuthorityRoots,
    *,
    hypothesis: bool = False,
) -> RequiredBehaviorContract:
    return RequiredBehaviorContract(
        roots=roots,
        behavior_id="behavior:SupportContext",
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence_precedence=(
            BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            if hypothesis
            else BehaviorEvidencePrecedence.REVIEWED_IDL
        ),
        field_refs=("field:trace_id",),
        constructor_refs=("ctor:SupportContext",),
        method_refs=("method:with_span",),
        invariant_refs=("inv:non_empty",),
        state_transition_refs=("tx:idle->active",),
        effect_refs=("effect:none",),
        capability_refs=("cap:context.read",),
        authorization_refs=("auth:session",),
        resource_refs=("res:1mb",),
        proof_refs=() if hypothesis else ("proof:behavior",),
        placement_decision_ref="placement:pkg.support",
        implementation_hypothesis=hypothesis,
    )


def _memory(
    roots: AuthorityRoots,
    *,
    disposition: MemorySafetyDisposition = MemorySafetyDisposition.SUPPORTED,
) -> MemorySafetyFacet:
    evidence = _evidence("evidence:memory")
    return MemorySafetyFacet(
        roots=roots,
        subject_span=SourceSpan("pkg/native.py", 1, 80, "blob:native"),
        language_runtime="rust",
        disposition=disposition,
        evidence_refs=(evidence,)
        if disposition
        in {
            MemorySafetyDisposition.SUPPORTED,
            MemorySafetyDisposition.EMPIRICAL,
            MemorySafetyDisposition.PROVED,
        }
        else (),
        proof_refs=(evidence,)
        if disposition is MemorySafetyDisposition.PROVED
        else (),
        unsupported_refs=("unsupported:ffi",)
        if disposition is MemorySafetyDisposition.UNSUPPORTED
        else (),
    )


# ---------------------------------------------------------------------------
# Vocabulary / mapping
# ---------------------------------------------------------------------------


def test_obligation_vocabulary_covers_plan_families() -> None:
    kinds = set(all_obligation_kinds())
    required = {
        GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
        GoalObligationKind.VALUE_SUFFICIENCY,
        GoalObligationKind.OUTPUT_REFINEMENT,
        GoalObligationKind.TOTALITY,
        GoalObligationKind.NULLABILITY,
        GoalObligationKind.RANGE,
        GoalObligationKind.ALLOWED_ERRORS,
        GoalObligationKind.PERMITTED_EFFECTS,
        GoalObligationKind.AUTHORIZATION,
        GoalObligationKind.RESOURCES,
        GoalObligationKind.TEMPORAL,
        GoalObligationKind.STATE,
        GoalObligationKind.CONCURRENCY,
        GoalObligationKind.SCHEMA,
        GoalObligationKind.CONSTRUCTOR,
        GoalObligationKind.SERIALIZATION,
        GoalObligationKind.REGISTRATION,
        GoalObligationKind.PLACEMENT,
        GoalObligationKind.OWNERSHIP,
        GoalObligationKind.LIFETIME,
        GoalObligationKind.INFORMATION_PROVENANCE,
        GoalObligationKind.MEMORY_SAFETY,
    }
    assert required.issubset(kinds)
    assert obligation_facet_kind(GoalObligationKind.MEMORY_SAFETY) is LogicFacetKind.MEMORY
    assert obligation_facet_kind(GoalObligationKind.RESOURCES) is LogicFacetKind.RESOURCE
    assert obligation_facet_kind(GoalObligationKind.OWNERSHIP) is LogicFacetKind.MEMORY
    assert obligation_facet_kind(GoalObligationKind.LIFETIME) is LogicFacetKind.LIFETIME
    assert delta_obligation_kind(DeltaKind.PARAMETER_ADD) is (
        GoalObligationKind.CALLER_INPUT_ACCEPTANCE
    )
    assert delta_obligation_kind(DeltaKind.RESULT_CHANGE) is (
        GoalObligationKind.OUTPUT_REFINEMENT
    )
    assert is_nominating_source(GoalSourceKind.TASK_PROSE)
    assert not is_nominating_source(GoalSourceKind.CONTRACT_DELTA)
    assert source_authority_for_kind(GoalSourceKind.TASK_PROSE) is (
        SourceAuthorityClass.NOMINATING
    )
    assert set(all_source_kinds()) >= {
        GoalSourceKind.BROKEN_TRACE,
        GoalSourceKind.CONTRACT_DELTA,
        GoalSourceKind.TASK_PROSE,
    }


# ---------------------------------------------------------------------------
# Core compilation paths
# ---------------------------------------------------------------------------


def test_compile_from_delta_and_consumer_covers_required_families(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        contract_deltas=(_delta(prop_roots),),
        consumer_obligations=(_consumer(prop_roots),),
        missing_inputs=(_missing(prop_roots),),
        behavior_contracts=(_behavior(prop_roots),),
    )

    assert compilation.producer_id == PRODUCER_ID
    assert compilation.disposition in {
        CompilationDisposition.PARTIAL,
        CompilationDisposition.COMPLETE,
    }
    assert compilation.goals
    assert compilation.source_bindings
    assert len(compilation.goals) == len(
        {item.goal_id for item in compilation.goals}
    )

    kinds = {
        binding.obligation_kind for binding in compilation.source_bindings
    }
    for required in (
        GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
        GoalObligationKind.OUTPUT_REFINEMENT,
        GoalObligationKind.NULLABILITY,
        GoalObligationKind.ALLOWED_ERRORS,
        GoalObligationKind.PERMITTED_EFFECTS,
        GoalObligationKind.AUTHORIZATION,
        GoalObligationKind.RESOURCES,
        GoalObligationKind.TEMPORAL,
        GoalObligationKind.SCHEMA,
        GoalObligationKind.CONSTRUCTOR,
        GoalObligationKind.SERIALIZATION,
        GoalObligationKind.REGISTRATION,
        GoalObligationKind.PLACEMENT,
        GoalObligationKind.VALUE_SUFFICIENCY,
        GoalObligationKind.INFORMATION_PROVENANCE,
        GoalObligationKind.TOTALITY,
        GoalObligationKind.RANGE,
        GoalObligationKind.OWNERSHIP,
        GoalObligationKind.CONSUMER_MIGRATION,
        GoalObligationKind.MEMORY_SAFETY,
    ):
        # memory comes from unsupported residual or behavior; value from missing
        if required is GoalObligationKind.MEMORY_SAFETY:
            # Present as unsupported residual from delta MEMORY_FACET_CHANGE
            assert required in kinds or any(
                "clause:memory" in ref for ref in compilation.unsupported_refs
            )
            continue
        assert required in kinds, f"missing obligation kind {required}"

    # One required goal per resolved consumer.
    consumer_goals = compilation.goals_for_consumer("consumer:primary")
    assert consumer_goals
    assert any(
        binding.obligation_kind is GoalObligationKind.CONSUMER_MIGRATION
        for binding in compilation.source_bindings
        if binding.consumer_id == "consumer:primary"
    )

    # Retain source precedence / actual / expected / counterexample / bounds.
    binding = next(
        item
        for item in compilation.source_bindings
        if item.obligation_kind is GoalObligationKind.CALLER_INPUT_ACCEPTANCE
    )
    assert binding.source_authority is SourceAuthorityClass.AUTHORITATIVE
    assert binding.actual_fact_ref
    assert binding.expected_fact_ref
    assert binding.counterexample_target_ref
    assert compilation.invalidation_refs
    assert logic_roots.tree_id in compilation.invalidation_refs

    # Round-trip
    assert ProgramLogicGoalCompilation.from_dict(compilation.to_record()) == compilation


def test_broken_trace_and_call_requirement_goals(
    logic_roots: ProgramLogicAuthorityRoots,
    repair_roots: AuthorityRoots,
) -> None:
    compilation = compile_program_logic_goals(
        logic_roots,
        broken_traces=(_trace(repair_roots),),
        call_requirements=(_call_requirement(repair_roots),),
    )
    assert any(
        item.family is GoalFamily.COUNTEREXAMPLE for item in compilation.goals
    )
    assert any(
        item.family in {GoalFamily.REFINEMENT, GoalFamily.POSITIVE}
        for item in compilation.goals
    )
    # Unsupported clause from call requirement remains explicit.
    assert "unsupported:reflection" in compilation.unsupported_refs
    assert any(
        item.kind is GoalDiagnosticKind.UNSUPPORTED_SEMANTIC
        for item in compilation.diagnostics
    )
    kinds = {b.obligation_kind for b in compilation.source_bindings}
    assert GoalObligationKind.CALLER_INPUT_ACCEPTANCE in kinds
    assert GoalObligationKind.OUTPUT_REFINEMENT in kinds
    assert GoalObligationKind.ALLOWED_ERRORS in kinds
    assert GoalObligationKind.PERMITTED_EFFECTS in kinds
    assert GoalObligationKind.AUTHORIZATION in kinds
    assert GoalObligationKind.RESOURCES in kinds


def test_dynamic_trace_remains_explicit_residual(
    logic_roots: ProgramLogicAuthorityRoots,
    repair_roots: AuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        broken_traces=(
            _trace(repair_roots, disposition=TraceDisposition.DYNAMIC),
        ),
    )
    assert compilation.disposition is CompilationDisposition.PARTIAL
    assert any(
        item.kind is GoalDiagnosticKind.DYNAMIC_FRONTIER
        for item in compilation.diagnostics
    )
    assert any(
        item.disposition is GoalDisposition.UNSUPPORTED for item in compilation.goals
    )
    assert compilation.residual_refs
    assert compilation.unsupported_refs


def test_memory_safety_ownership_lifetime_goals(
    logic_roots: ProgramLogicAuthorityRoots,
    repair_roots: AuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        memory_facets=(_memory(repair_roots),),
    )
    kinds = {b.obligation_kind for b in compilation.source_bindings}
    assert GoalObligationKind.MEMORY_SAFETY in kinds
    assert GoalObligationKind.OWNERSHIP in kinds
    assert GoalObligationKind.LIFETIME in kinds
    for goal in compilation.goals:
        for facet in goal.required_facets:
            # memory facets never bind resource contracts
            assert not facet.contract_ref.startswith("resource:")


def test_unsupported_memory_facet_stays_unsupported(
    logic_roots: ProgramLogicAuthorityRoots,
    repair_roots: AuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        memory_facets=(
            _memory(repair_roots, disposition=MemorySafetyDisposition.UNSUPPORTED),
        ),
    )
    assert any(
        item.kind is GoalDiagnosticKind.NATIVE_BOUNDARY
        for item in compilation.diagnostics
    )
    assert "unsupported:ffi" in compilation.unsupported_refs
    assert all(
        item.disposition is GoalDisposition.UNSUPPORTED for item in compilation.goals
    )


# ---------------------------------------------------------------------------
# Authority / conflict / prose rules
# ---------------------------------------------------------------------------


def test_prose_can_nominate_but_cannot_satisfy(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        prose_nominations=(
            ProseGoalNomination(
                obligation_kind=GoalObligationKind.VALUE_SUFFICIENCY,
                statement_ref="stmt:prose:context-must-exist",
            ),
        ),
    )
    assert compilation.goals
    binding = compilation.source_bindings[0]
    assert binding.nominating_only is True
    assert binding.source_authority is SourceAuthorityClass.NOMINATING
    assert binding.source_kind is GoalSourceKind.TASK_PROSE
    assert any(
        item.kind is GoalDiagnosticKind.PROSE_NOMINATION
        for item in compilation.diagnostics
    )
    # Prose goals remain open/unproved; they never claim discharge.
    for goal in compilation.goals:
        assert goal.disposition is not GoalDisposition.DISCHARGED
        assert goal.proof_status.value == "unproved"
        assert goal.assumption_authority in {
            SourceAuthorityClass.NOMINATING,
            SourceAuthorityClass.NONE,
        }


def test_implementation_hypothesis_behavior_is_nominating_only(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        behavior_contracts=(_behavior(prop_roots, hypothesis=True),),
    )
    assert any(
        item.kind is GoalDiagnosticKind.NON_AUTHORITATIVE_SOURCE
        for item in compilation.diagnostics
    )
    assert all(
        binding.nominating_only for binding in compilation.source_bindings
    )
    assert all(
        binding.source_authority is SourceAuthorityClass.NOMINATING
        for binding in compilation.source_bindings
    )


def test_conflicting_intent_creates_diagnostic(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    # Two breaking clauses of the same obligation kind with different expected facts
    # and identical facet identity via parameter_add twice would normally be unique
    # clause ids; force conflict via two deltas that map to the same obligation and
    # we synthesize facet collision by using the same clause consumer domain path
    # with different expected refs under VALUE_SUFFICIENCY via missing inputs is hard.
    # Instead: compile two call-site deltas that both map PARAMETER_DEFAULT with
    # different expected facts under the same consumer/obligation — facet ids differ
    # by clause. Force a direct conflict by injecting two drafts via two missing
    # inputs is also unique. Use two ProgramContractDelta PARAMETER_ADD clauses that
    # we manually give the same facet via a second delta with identical clause shape
    # is unique by clause_id.
    #
    # Practical approach: compile twice-sourced goals by providing two call
    # requirements is different facets. Use two Memory facets? Different.
    #
    # Direct path: two contract deltas with PARAMETER_ADD and we patch facet
    # collision by using the compiler's conflict key (consumer, obligation, facet).
    # Facet ids include clause_id so they won't collide.
    #
    # Create conflict via two MissingInputRequirement with same parameter and
    # different type_ref? Different requirement_ids produce different facet ids.
    #
    # We'll use two ProgramContractDelta RESULT_CHANGE clauses but force the
    # same facet_id by using the same clause_id across two deltas — second delta
    # with same clause_id but different expected fact.
    clause_a = ContractClauseDelta(
        clause_id="clause:shared-result",
        kind=DeltaKind.RESULT_CHANGE,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id="symbol:process",
        consumer_domain="domain:python_callers",
        before_contract_ref="contract:before:result-a",
        after_contract_ref="contract:after:result",
    )
    clause_b = ContractClauseDelta(
        clause_id="clause:shared-result",
        kind=DeltaKind.RESULT_CHANGE,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id="symbol:process",
        consumer_domain="domain:python_callers",
        before_contract_ref="contract:before:result-b",
        after_contract_ref="contract:after:result",
    )
    delta_a = ProgramContractDelta(
        roots=prop_roots,
        change_set_id="change-set:a",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before:a",
        after_contract_ref="contract:after:a",
        clauses=(clause_a,),
    )
    delta_b = ProgramContractDelta(
        roots=prop_roots,
        change_set_id="change-set:b",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before:b",
        after_contract_ref="contract:after:b",
        clauses=(clause_b,),
    )
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        contract_deltas=(delta_a, delta_b),
    )
    assert compilation.disposition is CompilationDisposition.CONFLICT
    assert any(
        item.kind is GoalDiagnosticKind.CONFLICTING_INTENT
        for item in compilation.diagnostics
    )
    assert any(item.family is GoalFamily.CONSISTENCY for item in compilation.goals)
    # Conflicting members are residual; consistency goal stays open.
    consistency = [g for g in compilation.goals if g.family is GoalFamily.CONSISTENCY]
    assert consistency
    assert all(g.disposition is GoalDisposition.OPEN for g in consistency)


def test_cross_root_artifact_rejected(
    logic_roots: ProgramLogicAuthorityRoots,
    repair_roots: AuthorityRoots,
) -> None:
    bad_roots = AuthorityRoots(
        repository_id="repository:other",
        forest_id="forest:lpr-006",
        tree_id="tree:candidate",
        graph_id="graph:lpr-006",
        index_id="index:lpr-006",
        model_id="model:lpr-006",
        config_id="config:lpr-006",
        translator_id="translator:lpr-006",
        toolchain_id="toolchain:lpr-006",
        policy_id="policy:lpr-006",
    )
    with pytest.raises(ProgramLogicGoalCompilerAuthorityError, match="repository_id"):
        ProgramLogicGoalCompiler(logic_roots).compile(
            broken_traces=(_trace(bad_roots),),
        )

    stale_tree = AuthorityRoots(
        repository_id="repository:lpr-006",
        forest_id="forest:lpr-006",
        tree_id="tree:stale",
        graph_id="graph:lpr-006",
        index_id="index:lpr-006",
        model_id="model:lpr-006",
        config_id="config:lpr-006",
        translator_id="translator:lpr-006",
        toolchain_id="toolchain:lpr-006",
        policy_id="policy:lpr-006",
    )
    with pytest.raises(ProgramLogicGoalCompilerAuthorityError, match="tree_id"):
        ProgramLogicGoalCompiler(logic_roots).compile(
            broken_traces=(_trace(stale_tree),),
        )


def test_frontier_consumer_is_residual_not_required_goal(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        consumer_obligations=(
            _consumer(
                prop_roots,
                consumer_id="consumer:frontier",
                disposition=ConsumerDisposition.FRONTIER,
            ),
        ),
    )
    assert any(
        item.kind is GoalDiagnosticKind.FRONTIER_CONSUMER
        for item in compilation.diagnostics
    )
    assert not compilation.goals_for_consumer("consumer:frontier")
    assert compilation.residual_refs


def test_one_required_goal_per_resolved_consumer_facet(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    consumers = (
        _consumer(prop_roots, consumer_id="consumer:a", clause_ids=("clause:a",)),
        _consumer(prop_roots, consumer_id="consumer:b", clause_ids=("clause:b",)),
    )
    # Override roots consumer_id is primary; consumer_id on obligations drives grouping.
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        consumer_obligations=consumers,
    )
    for consumer_id in ("consumer:a", "consumer:b"):
        goals = compilation.goals_for_consumer(consumer_id)
        assert goals, f"expected goals for {consumer_id}"
        migration = [
            b
            for b in compilation.source_bindings
            if b.consumer_id == consumer_id
            and b.obligation_kind is GoalObligationKind.CONSUMER_MIGRATION
        ]
        assert len(migration) == 1
        # Clause facets also produce goals
        clause_goals = [
            b
            for b in compilation.source_bindings
            if b.consumer_id == consumer_id
            and b.obligation_kind is GoalObligationKind.COMPATIBILITY
        ]
        assert clause_goals


def test_empty_input_abstains(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile()
    assert compilation.disposition is CompilationDisposition.ABSTAINED
    assert compilation.goals == ()
    assert compilation.source_bindings == ()


def test_goal_source_binding_rejects_authority_promotion() -> None:
    with pytest.raises(ProgramLogicGoalCompilerAuthorityError):
        GoalSourceBinding(
            binding_id="binding:bad",
            goal_id="goal:one",
            source_kind=GoalSourceKind.TASK_PROSE,
            source_ref="stmt:prose",
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
            obligation_kind=GoalObligationKind.VALUE_SUFFICIENCY,
            nominating_only=False,
        )
    with pytest.raises(ProgramLogicGoalCompilerAuthorityError):
        GoalSourceBinding(
            binding_id="binding:bad2",
            goal_id="goal:one",
            source_kind=GoalSourceKind.CONTRACT_DELTA,
            source_ref="delta:one",
            source_authority=SourceAuthorityClass.NOMINATING,
            obligation_kind=GoalObligationKind.SCHEMA,
            nominating_only=False,
        )


def test_deterministic_compilation(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    repair_roots: AuthorityRoots,
) -> None:
    kwargs = dict(
        contract_deltas=(_delta(prop_roots),),
        consumer_obligations=(_consumer(prop_roots),),
        missing_inputs=(_missing(prop_roots),),
        broken_traces=(_trace(repair_roots),),
        memory_facets=(_memory(repair_roots),),
        prose_nominations=(
            {
                "obligation_kind": GoalObligationKind.PLACEMENT,
                "statement_ref": "stmt:prose:place-here",
            },
        ),
    )
    first = ProgramLogicGoalCompiler(logic_roots).compile(**kwargs)
    second = ProgramLogicGoalCompiler(logic_roots).compile(**kwargs)
    assert first.content_id == second.content_id
    assert first.compilation_id == second.compilation_id
    assert [g.goal_id for g in first.goals] == [g.goal_id for g in second.goals]
    assert [b.binding_id for b in first.source_bindings] == [
        b.binding_id for b in second.source_bindings
    ]


def test_compilation_rejects_source_bodies(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    compilation = ProgramLogicGoalCompiler(logic_roots).compile(
        prose_nominations=(
            ProseGoalNomination(
                obligation_kind=GoalObligationKind.SCHEMA,
                statement_ref="stmt:prose:schema",
            ),
        ),
    )
    payload = compilation.to_record()
    payload["source_body"] = "def evil(): pass"
    with pytest.raises(ProgramLogicGoalCompilerError, match="unsupported fields|source bodies"):
        ProgramLogicGoalCompilation.from_dict(payload)


def test_module_entry_point(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    compilation = compile_program_logic_goals(
        logic_roots,
        contract_deltas=(_delta(prop_roots),),
        consumer_obligations=(_consumer(prop_roots),),
    )
    assert isinstance(compilation, ProgramLogicGoalCompilation)
    assert compilation.goals
    # Every goal retains roots and bound invalidators
    for goal in compilation.goals:
        assert goal.roots == logic_roots
        assert goal.invalidation_refs
        assert goal.logic_family_refs
        assert GoalFamily(goal.family)
