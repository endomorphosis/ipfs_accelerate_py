"""Fail-closed coverage for change-propagation LogicIR obligation lowering (RPR-035)."""

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
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
    ValueCandidate,
    ValueCandidateDisposition,
    ValueCandidateKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.value_provenance_graph import (
    PRODUCER_ID,
    ValueProvenanceGraph,
)
from ipfs_accelerate_py.agent_supervisor.integrations.change_propagation_capabilities import (
    ChangePropagationCapability,
    ChangePropagationCapabilityReport,
    ChangePropagationCapabilityStatus,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import Completeness, ProgramGraphRoots
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_obligations import (
    AssumptionBinding,
    BehaviorRefinementClaim,
    ChangePropagationObligationCompiler,
    ChangePropagationObligationError,
    IncompleteImpactClosureError,
    LogicCapabilityBinding,
    ObligationContext,
    ObligationKind,
    UnauthorizedAssumptionError,
    UnsupportedObligationError,
    UnsupportedSemantic,
    UnsupportedSemanticKind,
    ValueMappingClaim,
)


def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


def node(path: str = "pkg/caller.py", symbol: str = "symbol:caller") -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{symbol}",
        kind="function",
        path=path,
        symbol_id=symbol,
        artifact_id=f"blob:{symbol}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def clause() -> ContractClauseDelta:
    return ContractClauseDelta(
        clause_id="clause:param-add",
        kind=DeltaKind.PARAMETER_ADD,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id="symbol:process",
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        reason="third-argument-required",
    )


def delta(auth: PropagationAuthorityRoots | None = None) -> ProgramContractDelta:
    auth = auth or roots()
    return ProgramContractDelta(
        roots=auth,
        change_set_id="changeset:one",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause(),),
        evidence_refs=("evidence:delta",),
        proof_refs=("proof:delta",),
    )


def closure(
    auth: PropagationAuthorityRoots | None = None,
    *,
    completeness: ImpactCompleteness = ImpactCompleteness.COMPLETE,
    frontier_node_ids: tuple[str, ...] = (),
    frontier_edge_ids: tuple[str, ...] = (),
) -> ImpactClosureReceipt:
    auth = auth or roots()
    return ImpactClosureReceipt(
        roots=auth,
        delta_id="delta:one",
        completeness=completeness,
        consumers=(
            ImpactConsumer(
                consumer_id="consumer:one",
                node=node(),
                depth=1,
                mandatory=True,
                edge_refs=("edge:call",),
                path_condition_ref="path:always",
            ),
        ),
        frontier_node_ids=frontier_node_ids,
        frontier_edge_ids=frontier_edge_ids,
        evidence_refs=("evidence:closure",),
    )


def consumer(
    auth: PropagationAuthorityRoots | None = None,
    *,
    disposition: ConsumerDisposition = ConsumerDisposition.MIGRATE,
    missing_input_ids: tuple[str, ...] = ("missing:context",),
    behavior_contract_ids: tuple[str, ...] = (),
    proof_refs: tuple[str, ...] = ("proof:obligation",),
) -> ConsumerMigrationObligation:
    auth = auth or roots()
    return ConsumerMigrationObligation(
        roots=auth,
        obligation_id="obligation:consumer:one",
        consumer_id="consumer:one",
        delta_id="delta:one",
        disposition=disposition,
        clause_ids=("clause:param-add",),
        node=node(),
        proof_refs=proof_refs,
        missing_input_ids=missing_input_ids,
        behavior_contract_ids=behavior_contract_ids,
        invalidation_refs=("tree:candidate",),
    )


def missing_input(auth: PropagationAuthorityRoots | None = None) -> MissingInputRequirement:
    auth = auth or roots()
    return MissingInputRequirement(
        roots=auth,
        requirement_id="missing:context",
        obligation_id="obligation:consumer:one",
        clause_id="clause:param-add",
        parameter_name="context",
        type_ref="type:Context",
        nullability="nonnull",
        information_content_ref="info:request-context",
        construction_precondition_refs=("pre:context-ready",),
        result_postcondition_refs=("post:context-valid",),
        allowed_error_refs=("error:ContextMissing",),
        effect_refs=("effect:none",),
        capability_refs=("capability:request.read",),
        authorization_refs=("auth:caller",),
        resource_refs=("resource:stack",),
        ownership_refs=("ownership:borrowed",),
        propagation_depth_bound=2,
        proof_refs=("proof:missing",),
    )


def value_candidate(
    auth: PropagationAuthorityRoots | None = None,
    *,
    kind: ValueCandidateKind = ValueCandidateKind.PARAMETER,
    disposition: ValueCandidateDisposition = ValueCandidateDisposition.NOMINATED,
    semantic_authority: bool = False,
    proof_refs: tuple[str, ...] = (),
) -> ValueCandidate:
    auth = auth or roots()
    return ValueCandidate(
        roots=auth,
        candidate_id="candidate:ctx-param",
        requirement_id="missing:context",
        kind=kind,
        disposition=disposition,
        source_node=node(path="pkg/caller.py", symbol="symbol:ctx"),
        expression_ref="expr:ctx",
        type_ref="type:Context",
        semantic_authority=semantic_authority,
        proof_refs=proof_refs,
    )


def behavior(auth: PropagationAuthorityRoots | None = None) -> RequiredBehaviorContract:
    auth = auth or roots()
    return RequiredBehaviorContract(
        roots=auth,
        behavior_id="behavior:context-type",
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:Context",
        evidence_precedence=BehaviorEvidencePrecedence.REVIEWED_IDL,
        field_refs=("field:request_id",),
        constructor_refs=("ctor:Context",),
        method_refs=("method:validate",),
        invariant_refs=("inv:nonempty-id",),
        state_transition_refs=("state:init->ready",),
        effect_refs=("effect:none",),
        capability_refs=("capability:request.read",),
        authorization_refs=("auth:caller",),
        resource_refs=("resource:heap",),
        proof_refs=("proof:behavior",),
        placement_decision_ref="placement:pkg.types",
    )


def provenance(auth: PropagationAuthorityRoots | None = None) -> ValueProvenanceGraph:
    auth = auth or roots()
    return ValueProvenanceGraph(
        roots=ProgramGraphRoots(
            forest_id=auth.candidate_forest_id,
            tree_id=auth.candidate_tree_id,
            overlay_id=auth.candidate_overlay_id,
            config_id=auth.config_id,
            toolchain_id=auth.toolchain_id,
            extractor_id="extractor:value-provenance",
        ),
        producer_id=PRODUCER_ID,
        procedures=("proc:caller",),
        blocks=(),
        definitions=(),
        uses=(),
        def_use_chains=(),
        dominance_facts=(),
        path_conditions=(),
        type_refinements=(),
        information_provenances=(),
        interprocedural_threads=(),
        unknown_frontier=(),
        completeness=Completeness.PARTIAL,
    )


def capability() -> LogicCapabilityBinding:
    report = ChangePropagationCapabilityReport(
        capabilities=(
            ChangePropagationCapability(
                "datasets.logic_ir",
                ChangePropagationCapabilityStatus.AVAILABLE,
                module_paths=("/tmp/logic_ir.py",),
                interface_version="logic-ir@1",
                supported_semantics=("ir",),
                reconstruction_compatible=True,
                details={"capability_revision": "logic:one"},
            ),
        ),
        accelerator_module_paths=(),
        datasets_module_paths=("/tmp/logic_ir.py",),
        datasets_gitlink_revision="gitlink:one",
    )
    return LogicCapabilityBinding.from_report(report)


def context(
    *unsupported: UnsupportedSemantic,
    allow_partial_frontier: bool = False,
) -> ObligationContext:
    return ObligationContext(
        assumptions=(
            AssumptionBinding(
                assumption_id="assumption:reviewed-one",
                kind="reviewed_assumption",
                evidence_ref="evidence:assumption:one",
            ),
        ),
        capability=capability(),
        unsupported_semantics=unsupported,
        allow_partial_frontier=allow_partial_frontier,
    )


def test_migrate_lowers_separate_facet_obligations_with_full_root_binding() -> None:
    auth = roots()
    item = consumer(auth)
    result = ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(auth),
        item,
        context(),
        missing_inputs=(missing_input(auth),),
        value_candidates=(value_candidate(auth),),
        value_provenance=provenance(auth),
    )

    kinds = set(result.kinds)
    required = {
        ObligationKind.CLOSURE_COVERAGE,
        ObligationKind.CONSUMER_COMPATIBILITY,
        ObligationKind.SOURCE_SCOPE_PATH_AVAILABILITY,
        ObligationKind.TYPE_SCHEMA_RANGE_NULLABILITY,
        ObligationKind.INFORMATION_SUFFICIENCY,
        ObligationKind.CONVERSION_CONSTRUCTOR_TOTALITY,
        ObligationKind.ERROR_COMPATIBILITY,
        ObligationKind.EFFECT_COMPATIBILITY,
        ObligationKind.CAPABILITY_COMPATIBILITY,
        ObligationKind.AUTHORIZATION_COMPATIBILITY,
        ObligationKind.TRUST_COMPATIBILITY,
        ObligationKind.RESOURCE_COMPATIBILITY,
        ObligationKind.OWNERSHIP_LIFETIME,
        ObligationKind.MUTATION_CONCURRENCY,
        ObligationKind.DEPENDENCY_CYCLE_ABSENCE,
        ObligationKind.PARAMETER_THREADING,
    }
    assert required <= kinds
    assert len(result.obligations) <= 128
    assert result.value_mapping_claims
    mapping = result.value_mapping_claims[0]
    assert isinstance(mapping, ValueMappingClaim)
    assert mapping.parameter_name == "context"
    assert mapping.candidate_id == "candidate:ctx-param"

    for obligation in result.obligations:
        claim = obligation.claim
        assert claim.premise_ids
        assert claim.source_ids
        assert claim.assumption_ids == ("assumption:reviewed-one",)
        assert claim.repository_id == "repository:one"
        assert claim.tree_id == "tree:candidate"
        assert claim.graph_id  # provenance graph id or authority graph id
        assert claim.translator_id == "translator:one"
        assert claim.toolchain_id == "toolchain:one"
        assert claim.policy_id == "policy:one"
        assert claim.capability_id == "datasets.logic_ir"
        assert claim.capability_revision == "logic:one"
        assert claim.counterexample_targets
        assert obligation.code_obligation.metadata["claim_id"] == claim.content_id
        assert obligation.code_obligation.task_id == "RPR-035"
        assert obligation.code_obligation.repository_tree_id == claim.tree_id


def test_compilation_is_deterministic() -> None:
    auth = roots()
    compiler = ChangePropagationObligationCompiler()
    kwargs = dict(
        delta=delta(auth),
        closure=closure(auth),
        consumer_obligation=consumer(auth),
        context=context(),
        missing_inputs=(missing_input(auth),),
        value_candidates=(value_candidate(auth),),
        value_provenance=provenance(auth),
    )
    first = compiler.compile(**kwargs)
    second = compiler.compile(**kwargs)
    assert first.obligation_ids == second.obligation_ids
    assert first.to_dict() == second.to_dict()
    assert [item.claim_id for item in first.value_mapping_claims] == [
        item.claim_id for item in second.value_mapping_claims
    ]


def test_behavior_contracts_lower_invariants_state_serialization_and_placement() -> None:
    auth = roots()
    item = consumer(
        auth,
        missing_input_ids=(),
        behavior_contract_ids=("behavior:context-type",),
    )
    result = ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(auth),
        item,
        context(),
        behavior_contracts=(behavior(auth),),
        value_provenance=provenance(auth),
    )
    kinds = set(result.kinds)
    assert {
        ObligationKind.BEHAVIOR_INVARIANTS,
        ObligationKind.STATE_TRANSITIONS,
        ObligationKind.SERIALIZATION_MIGRATION,
        ObligationKind.PLACEMENT,
        ObligationKind.CLOSURE_COVERAGE,
        ObligationKind.CONSUMER_COMPATIBILITY,
    } <= kinds
    assert result.behavior_refinement_claims
    claim = result.behavior_refinement_claims[0]
    assert isinstance(claim, BehaviorRefinementClaim)
    assert claim.behavior_id == "behavior:context-type"
    assert claim.evidence_precedence == BehaviorEvidencePrecedence.REVIEWED_IDL.value
    assert "inv:nonempty-id" in claim.structural_clause_ids


def test_compatible_consumer_does_not_invent_mapping_or_placement() -> None:
    auth = roots()
    item = consumer(
        auth,
        disposition=ConsumerDisposition.COMPATIBLE,
        missing_input_ids=(),
        proof_refs=(),
    )
    result = ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(auth),
        item,
        context(),
    )
    assert set(result.kinds) == {
        ObligationKind.CLOSURE_COVERAGE,
        ObligationKind.CONSUMER_COMPATIBILITY,
    }
    assert result.value_mapping_claims == ()
    assert result.behavior_refinement_claims == ()


def test_partial_closure_fails_closed_without_explicit_frontier_policy() -> None:
    auth = roots()
    with pytest.raises(IncompleteImpactClosureError, match="partial impact"):
        ChangePropagationObligationCompiler().compile(
            delta(auth),
            closure(
                auth,
                completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
                frontier_node_ids=("node:dynamic",),
            ),
            consumer(auth, missing_input_ids=()),
            context(),
        )


def test_partial_closure_records_explicit_unsupported_frontier_when_allowed() -> None:
    auth = roots()
    result = ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(
            auth,
            completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
            frontier_node_ids=("node:dynamic",),
            frontier_edge_ids=("edge:reflect",),
        ),
        consumer(auth, missing_input_ids=()),
        context(
            UnsupportedSemantic(
                UnsupportedSemanticKind.DYNAMIC,
                "node:dynamic",
                "reason:dynamic-dispatch",
            ),
            allow_partial_frontier=True,
        ),
    )
    kinds = {item.kind for item in result.unsupported_semantics}
    assert UnsupportedSemanticKind.FRONTIER_NODE in kinds
    assert UnsupportedSemanticKind.FRONTIER_EDGE in kinds
    assert UnsupportedSemanticKind.DYNAMIC in kinds
    claim = result.obligations[0].claim
    assert any(item.startswith("frontier_node:") for item in claim.unsupported_semantic_ids)


def test_retrieved_and_model_statements_cannot_become_assumptions() -> None:
    with pytest.raises(UnauthorizedAssumptionError):
        AssumptionBinding(
            assumption_id="assumption:llm",
            kind="llm_assumption",
            evidence_ref="evidence:model",
        )
    with pytest.raises(UnauthorizedAssumptionError):
        AssumptionBinding(
            assumption_id="llm:forged",
            kind="reviewed_assumption",
            evidence_ref="evidence:ok",
        )
    with pytest.raises(UnauthorizedAssumptionError):
        AssumptionBinding(
            assumption_id="assumption:ok",
            kind="reviewed_assumption",
            evidence_ref="retrieved:vector-hit",
        )


def test_vector_nomination_cannot_claim_semantic_authority_or_axiom_status() -> None:
    auth = roots()
    with pytest.raises(Exception):
        # Contract layer already rejects vector + semantic_authority.
        value_candidate(
            auth,
            kind=ValueCandidateKind.VECTOR_NOMINATION,
            semantic_authority=True,
            disposition=ValueCandidateDisposition.NOMINATED,
        )


def test_vector_nomination_does_not_form_value_mapping_claim() -> None:
    auth = roots()
    result = ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(auth),
        consumer(auth),
        context(),
        missing_inputs=(missing_input(auth),),
        value_candidates=(
            value_candidate(
                auth,
                kind=ValueCandidateKind.VECTOR_NOMINATION,
                disposition=ValueCandidateDisposition.NOMINATED,
            ),
        ),
    )
    # Facet obligations still lower; mapping claim withheld without non-nomination source.
    assert ObligationKind.INFORMATION_SUFFICIENCY in set(result.kinds)
    assert result.value_mapping_claims == ()


def test_root_mismatch_fails_closed() -> None:
    auth = roots()
    other = PropagationAuthorityRoots(
        repository_id="repository:other",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate-other",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    with pytest.raises(ChangePropagationObligationError, match="roots"):
        ChangePropagationObligationCompiler().compile(
            delta(auth),
            closure(other),
            consumer(auth, missing_input_ids=()),
            context(),
        )


def test_missing_logic_ir_capability_fails_closed() -> None:
    with pytest.raises(UnsupportedObligationError, match="datasets.logic_ir"):
        LogicCapabilityBinding.from_report(
            ChangePropagationCapabilityReport(
                capabilities=(),
                accelerator_module_paths=(),
                datasets_module_paths=(),
                datasets_gitlink_revision="",
            )
        )


def test_unsupported_delta_clause_requires_explicit_unsupported_binding() -> None:
    auth = roots()
    unsupported_clause = ContractClauseDelta(
        clause_id="clause:native",
        kind=DeltaKind.PROTOCOL_CHANGE,
        disposition=DeltaDisposition.UNSUPPORTED,
        subject_symbol_id="symbol:process",
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        reason="native-boundary",
    )
    broken_delta = ProgramContractDelta(
        roots=auth,
        change_set_id="changeset:one",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause(), unsupported_clause),
        evidence_refs=("evidence:delta",),
    )
    with pytest.raises(UnsupportedObligationError, match="UnsupportedSemantic"):
        ChangePropagationObligationCompiler().compile(
            broken_delta,
            closure(auth),
            consumer(auth, missing_input_ids=()),
            context(),
        )

    result = ChangePropagationObligationCompiler().compile(
        broken_delta,
        closure(auth),
        consumer(auth, missing_input_ids=()),
        context(
            UnsupportedSemantic(
                UnsupportedSemanticKind.NATIVE,
                "clause:native",
                "reason:native-boundary",
            )
        ),
    )
    assert any(
        item.kind is UnsupportedSemanticKind.NATIVE for item in result.unsupported_semantics
    )


def test_frontier_consumer_cannot_compile_closed_obligations() -> None:
    auth = roots()
    with pytest.raises(ChangePropagationObligationError, match="frontier"):
        ChangePropagationObligationCompiler().compile(
            delta(auth),
            closure(auth),
            consumer(
                auth,
                disposition=ConsumerDisposition.FRONTIER,
                missing_input_ids=(),
                proof_refs=(),
            ),
            context(),
        )


def test_implementation_hypothesis_behavior_cannot_compile() -> None:
    auth = roots()
    with pytest.raises(Exception):
        RequiredBehaviorContract(
            roots=auth,
            behavior_id="behavior:guess",
            kind=BehaviorKind.CLASS,
            subject_symbol_id="symbol:Guess",
            evidence_precedence=BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
            field_refs=("field:x",),
            implementation_hypothesis=True,
            proof_refs=("proof:should-fail",),
        )


def test_migrate_requires_missing_input_records_when_listed() -> None:
    auth = roots()
    with pytest.raises(ChangePropagationObligationError, match="MissingInputRequirement"):
        ChangePropagationObligationCompiler().compile(
            delta(auth),
            closure(auth),
            consumer(auth),
            context(),
        )


def test_ir_claim_to_logic_ir_is_fully_bound() -> None:
    auth = roots()
    result = ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(auth),
        consumer(auth, missing_input_ids=()),
        context(),
        value_provenance=provenance(auth),
    )
    payload = result.obligations[0].claim.to_logic_ir()
    for key in (
        "predicate",
        "subject_id",
        "premise_ids",
        "source_ids",
        "assumption_ids",
        "repository_id",
        "tree_id",
        "graph_id",
        "translator_id",
        "toolchain_id",
        "policy_id",
        "capability_id",
        "capability_revision",
        "counterexample_targets",
    ):
        assert key in payload
        assert payload[key] not in (None, "", [])
