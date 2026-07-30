"""RPR-044: integrate transactional propagation and require fixed-point completion.

Covers the feature-gated pipeline order, analytical-first provider skip,
transaction + fixed-point gates, daemon/refinery/router cutovers, and
legacy @1/@2 repair compatibility.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_pipeline import (
    AnalysisPipeline,
    AnalysisPipelinePolicy,
)
from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AnalyticalTransform,
    AtomicPropagationPlan,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    TransformDisposition,
    TransformKind,
    TransactionState,
    obligation_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_pipeline import (
    CHANGE_PROPAGATION_PIPELINE_INTERFACE,
    CHANGE_PROPAGATION_PIPELINE_VERSION,
    PIPELINE_STAGE_ORDER,
    ChangePropagationPipeline,
    ChangePropagationPipelinePolicy,
    ChangePropagationPipelineRequest,
    PipelineDisposition,
    PipelineStage,
    run_change_propagation_pipeline,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
    ContractMismatchRefinery,
    ContractMismatchRefineryError,
    ContractMismatchRefineryPolicy,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_plan import (
    ChangePropagationPlanner,
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanResourceBounds,
    PlanValidationCommand,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_transaction import (
    ChangePropagationTransaction,
    StepApplyRequest,
    StepApplyResult,
    StepExecutionDisposition,
    TransactionLease,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
    materialize_change_propagation_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ProviderReason,
    RouteStatus,
    route_contract_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_validation import (
    CandidatePropagationEvidence,
    ChangePropagationValidationError,
    ChangePropagationValidator,
    ClosureRecomputeEvidence,
    ConsumerDischargeEvidence,
    DeltaReextractEvidence,
    FixedPointIterationEvidence,
    ProofReconstructionEvidence,
    PropagationIndexRebuildEvidence,
    ResolutionEvidence,
    SecondOrderImpactEvidence,
)
from ipfs_accelerate_py.agent_supervisor.validation.contract_repair_validation import (
    ImpactedTestEvidence,
    IntegrityEvidence,
    PolicyToolEvidence,
    build_passing_tool_evidence,
)
from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_validation import (
    DEFAULT_POLICY_REQUIRED_TOOLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-044",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-044",
        index_id="index:rpr-044",
        model_id="model:rpr-044",
        config_id="config:rpr-044",
        translator_id="translator:rpr-044",
        toolchain_id="toolchain:rpr-044",
        policy_id="policy:rpr-044",
    )


def _node(path: str = "pkg/caller.py", symbol: str = "symbol:caller") -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{symbol}",
        kind="function",
        path=path,
        symbol_id=symbol,
        artifact_id=f"blob:{symbol}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def _obligation(
    roots: PropagationAuthorityRoots,
    *,
    consumer_id: str = "consumer:one",
    path: str = "pkg/caller.py",
) -> ConsumerMigrationObligation:
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=f"obligation:{consumer_id}",
        consumer_id=consumer_id,
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=_node(path, f"symbol:{consumer_id}"),
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        behavior_contract_ids=(),
        invalidation_refs=("tree:candidate",),
    )


def _consumer(
    consumer_id: str = "consumer:one",
    path: str = "pkg/caller.py",
) -> ImpactConsumer:
    return ImpactConsumer(
        consumer_id=consumer_id,
        node=_node(path, f"symbol:{consumer_id}"),
        depth=1,
        mandatory=True,
        edge_refs=(f"edge:{consumer_id}",),
    )


def _closure(
    roots: PropagationAuthorityRoots,
    consumers: tuple[ImpactConsumer, ...],
) -> ImpactClosureReceipt:
    return ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=consumers,
        sccs=(),
        frontier_node_ids=(),
        frontier_edge_ids=(),
        validation_refs=("validation:impact",),
        resource_bound_refs=("bound:impact",),
        evidence_refs=("evidence:graph",),
    )


def _mapping(
    *,
    requirement_id: str = "missing:context",
    consumer_id: str = "consumer:one",
) -> ValueMappingProof:
    return ValueMappingProof(
        requirement_id=requirement_id,
        consumer_id=consumer_id,
        disposition=SynthesisDisposition.UNIQUE_PROVED,
        facet_results=(),
        proved_candidate_ids=("candidate:ctx",),
        refuted_candidate_ids=(),
        expression_ref="expr:ctx",
        type_ref="type:Context",
        repository_id="repository:rpr-044",
        tree_id="tree:candidate",
        toolchain_id="toolchain:rpr-044",
        policy_id="policy:rpr-044",
        reason_codes=("unique_source",),
    )


def _transform(
    roots: PropagationAuthorityRoots,
    *,
    transform_id: str = "transform:add-arg",
    obligation_ids: tuple[str, ...] = ("obligation:consumer:one",),
    path: str = "pkg/caller.py",
) -> AnalyticalTransform:
    return AnalyticalTransform(
        roots=roots,
        transform_id=transform_id,
        kind=TransformKind.ADD_ARGUMENT,
        disposition=TransformDisposition.ADMITTED,
        obligation_ids=obligation_ids,
        target_paths=(path,),
        expression_refs=("expr:ctx",),
        proof_refs=("proof:transform",),
    )


def _validation(*command_ids: str) -> tuple[PlanValidationCommand, ...]:
    if not command_ids:
        command_ids = ("validate:pytest",)
    return tuple(
        PlanValidationCommand(
            command_id=cid,
            argv=("python", "-m", "pytest", "-q", f"test_{cid.replace(':', '_')}.py"),
            required=True,
        )
        for cid in command_ids
    )


def _happy_bundle(roots: PropagationAuthorityRoots) -> PlanEvidenceBundle:
    consumer = _consumer()
    obligation = _obligation(roots)
    return PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (consumer,)),
        obligations=(obligation,),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        placement_decisions=(),
        read_spans=(
            PlanPathSpan(
                path="pkg/caller.py",
                start=0,
                end=40,
                artifact_id="blob:caller",
                before_hash="sha256:caller-before",
            ),
        ),
        write_spans=(
            PlanPathSpan(
                path="pkg/caller.py",
                start=10,
                end=30,
                artifact_id="blob:caller",
                before_hash="sha256:caller-before",
            ),
        ),
        validation_commands=_validation(),
        resource_bounds=PlanResourceBounds(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        candidate_set_id="candidate-set:one",
        expected_roots=roots,
    )


BEFORE_HASH = "sha256:caller-before"


def _passing_applicator(request: StepApplyRequest) -> StepApplyResult:
    return StepApplyResult(
        disposition=StepExecutionDisposition.PASSED,
        written_paths=request.step.write_paths,
        observed_before_hashes=tuple(
            PathBeforeHash(path=p, before_hash=BEFORE_HASH)
            for p in request.step.write_paths
        ),
    )


def _lease(paths: tuple[str, ...] = ("pkg/caller.py",)) -> TransactionLease:
    return TransactionLease(
        lease_id="lease:writer-1",
        fence_id="fence:1",
        holder_id="holder:txn",
        permitted_write_paths=paths,
        permitted_read_paths=paths,
        active=True,
    )


def _hashes(*paths: str) -> tuple[PathBeforeHash, ...]:
    return tuple(PathBeforeHash(path=path, before_hash=BEFORE_HASH) for path in paths)


def _index(tree: str = "tree:candidate", **changes: object) -> PropagationIndexRebuildEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "index_id": "index:rpr-044",
        "graph_id": "graph:rpr-044",
        "rebuilt_source_paths": ("pkg/caller.py",),
        "rebuilt_ast_paths": ("pkg/caller.py",),
        "rebuilt_vector_row_ids": ("vector:caller",),
        "rebuilt_graph_node_ids": ("node:symbol:consumer:one",),
        "tombstone_ids": (),
        "affected_paths": ("pkg/caller.py",),
        "clean_rebuild_equivalent": True,
    }
    values.update(changes)
    return PropagationIndexRebuildEvidence(**values)  # type: ignore[arg-type]


def _delta(
    tree: str = "tree:candidate",
    *,
    original_delta_id: str = "delta:one",
    reextracted_delta_id: str = "delta:one",
    **changes: object,
) -> DeltaReextractEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "original_delta_id": original_delta_id,
        "reextracted_delta_id": reextracted_delta_id,
        "breaking_delta_ids": (original_delta_id,),
        "unplanned_breaking_delta_ids": (),
        "extraction_receipt_id": "delta-receipt:ok",
        "matches_plan_delta": True,
    }
    values.update(changes)
    return DeltaReextractEvidence(**values)  # type: ignore[arg-type]


def _resolution(tree: str = "tree:candidate") -> ResolutionEvidence:
    return ResolutionEvidence(
        candidate_tree_id=tree,
        resolved_call_ids=("call:caller-process",),
        resolved_data_flow_ids=("data:ctx",),
        resolved_schema_ids=("schema:request",),
        resolved_wiring_ids=("wire:router",),
        unresolved_ids=(),
        resolution_receipt_id="resolution:ok",
        complete=True,
    )


def _closure_ev(
    tree: str = "tree:candidate",
    *,
    original_closure_id: str = "closure:one",
) -> ClosureRecomputeEvidence:
    return ClosureRecomputeEvidence(
        candidate_tree_id=tree,
        original_closure_id=original_closure_id,
        recomputed_closure_id="closure:recomputed",
        consumer_ids=("consumer:one",),
        mandatory_consumer_ids=("consumer:one",),
        frontier_node_ids=(),
        required_frontier_ids=(),
        uncovered_frontier_ids=(),
        complete=True,
        receipt_id="closure:ok",
    )


def _discharge(
    tree: str = "tree:candidate",
    *,
    original_obligation_ids: tuple[str, ...] = ("obligation:consumer:one",),
    discharged_obligation_ids: tuple[str, ...] = ("obligation:consumer:one",),
) -> ConsumerDischargeEvidence:
    return ConsumerDischargeEvidence(
        candidate_tree_id=tree,
        original_obligation_ids=original_obligation_ids,
        discharged_obligation_ids=discharged_obligation_ids,
        unresolved_mandatory_ids=(),
        omitted_dependent_ids=(),
        double_discharged_ids=(),
        receipt_id="discharge:ok",
    )


def _second_order(tree: str = "tree:candidate") -> SecondOrderImpactEvidence:
    return SecondOrderImpactEvidence(
        candidate_tree_id=tree,
        new_delta_ids=(),
        new_consumer_ids=(),
        residual_frontier_ids=(),
        requires_another_iteration=False,
        receipt_id="second:fixed",
    )


def _proofs(
    tree: str = "tree:candidate",
    *,
    original_proof_refs: tuple[str, ...] = ("proof:plan",),
    reconstructed_proof_refs: tuple[str, ...] = ("proof:plan", "proof:reconstructed"),
) -> ProofReconstructionEvidence:
    return ProofReconstructionEvidence(
        candidate_tree_id=tree,
        original_proof_refs=original_proof_refs,
        reconstructed_proof_refs=reconstructed_proof_refs,
        introduced_proof_refs=("proof:introduced",),
        failed_proof_refs=(),
        all_mandatory_reconstructed=True,
        receipt_id="proof:ok",
    )


def _tools(tree: str = "tree:candidate") -> PolicyToolEvidence:
    return PolicyToolEvidence(
        candidate_tree_id=tree,
        required_families=DEFAULT_POLICY_REQUIRED_TOOLS,
        results=build_passing_tool_evidence(tree, "policy:rpr-044").results,
        policy_id="policy:rpr-044",
    )


def _tests(tree: str = "tree:candidate") -> ImpactedTestEvidence:
    return ImpactedTestEvidence(
        candidate_tree_id=tree,
        focused_test_ids=("test:focused-caller",),
        impacted_test_ids=("test:impacted-process",),
        required_dependant_ids=("test:dependant-route",),
        executed_test_ids=(
            "test:focused-caller",
            "test:impacted-process",
            "test:dependant-route",
        ),
        passed_test_ids=(
            "test:focused-caller",
            "test:impacted-process",
            "test:dependant-route",
        ),
        failed_test_ids=(),
        omitted_dependant_ids=(),
        dependency_complete=True,
    )


def _integrity(tree: str = "tree:candidate") -> IntegrityEvidence:
    return IntegrityEvidence(
        candidate_tree_id=tree,
        contracts_deleted=(),
        contracts_weakened=(),
        tests_deleted=(),
        tests_weakened=(),
        checkers_deleted=(),
        checkers_weakened=(),
        findings_suppressed=(),
        original_finding_id="finding:propagation",
        original_finding_closed=True,
    )


def _candidate_evidence(
    plan: AtomicPropagationPlan | None = None,
) -> CandidatePropagationEvidence:
    closure_id = "closure:one"
    proof_refs: tuple[str, ...] = ("proof:plan",)
    obligation_ids: tuple[str, ...] = ("obligation:consumer:one",)
    delta_id = "delta:one"
    if plan is not None:
        closure_id = plan.impact_closure_id
        proof_refs = tuple(plan.proof_refs) or proof_refs
        obligation_ids = tuple(item.obligation_id for item in plan.obligations)
        delta_id = plan.delta_id
    iteration = FixedPointIterationEvidence(
        iteration=1,
        index_rebuild=_index(),
        delta_reextract=_delta(original_delta_id=delta_id, reextracted_delta_id=delta_id),
        resolution=_resolution(),
        closure_recompute=_closure_ev(original_closure_id=closure_id),
        consumer_discharge=_discharge(
            original_obligation_ids=obligation_ids,
            discharged_obligation_ids=obligation_ids,
        ),
        second_order=_second_order(),
        proof_reconstruction=_proofs(
            original_proof_refs=proof_refs,
            reconstructed_proof_refs=proof_refs + ("proof:reconstructed",),
        ),
        policy_tools=_tools(),
        impacted_tests=_tests(),
        integrity=_integrity(),
    )
    return CandidatePropagationEvidence(
        candidate_tree_id="tree:candidate",
        iterations=(iteration,),
    )


def _pipeline_request(
    roots: PropagationAuthorityRoots,
    *,
    execute_mutation: bool = False,
    **kwargs: Any,
) -> ChangePropagationPipelineRequest:
    evidence = _happy_bundle(roots)
    base = dict(
        roots=roots,
        evidence_bundle=evidence,
        impact_closure=evidence.impact_closure,
        obligations=evidence.obligations,
        analytical_transforms=evidence.analytical_transforms,
        value_mapping_proofs=evidence.value_mapping_proofs,
        execute_mutation=execute_mutation,
        task_write_paths=("pkg/caller.py",),
        writer_write_paths=("pkg/caller.py",),
    )
    base.update(kwargs)
    return ChangePropagationPipelineRequest(**base)


# ---------------------------------------------------------------------------
# Interface / feature gate
# ---------------------------------------------------------------------------


def test_pipeline_interface_and_stage_order() -> None:
    assert CHANGE_PROPAGATION_PIPELINE_INTERFACE == "ChangePropagationPipeline@1"
    assert CHANGE_PROPAGATION_PIPELINE_VERSION == 1
    assert ChangePropagationPipeline.INTERFACE == CHANGE_PROPAGATION_PIPELINE_INTERFACE
    assert PIPELINE_STAGE_ORDER[0] == "change_set"
    assert PIPELINE_STAGE_ORDER[-1] == "fixed_point_validation"
    assert "transaction" in PIPELINE_STAGE_ORDER
    assert "pre_provider_gate" in PIPELINE_STAGE_ORDER


def _analysis_pipeline(**policy_fields: Any) -> AnalysisPipeline:
    import tempfile
    from pathlib import Path

    from ipfs_accelerate_py.agent_supervisor.analysis.analysis_cache import (
        AnalysisCache,
    )

    root = Path(tempfile.mkdtemp(prefix="rpr044-cache-"))
    cache = AnalysisCache(root)

    def _analyzer(context):  # pragma: no cover - unused on propagation route
        raise AssertionError("legacy analyzer must not run on change-propagation route")

    return AnalysisPipeline(
        cache,
        _analyzer,
        policy=AnalysisPipelinePolicy(**policy_fields),
    )


def test_feature_gate_defaults_off(roots: PropagationAuthorityRoots) -> None:
    result = ChangePropagationPipeline().run(_pipeline_request(roots))
    assert result.enabled is False
    assert result.disposition == PipelineDisposition.DISABLED.value
    assert result.provider_invoked is False

    # Analysis pipeline flag also defaults off.
    analysis = _analysis_pipeline()
    gated = analysis.run_change_propagation(_pipeline_request(roots))
    assert gated.enabled is False
    assert gated.disposition == "disabled"


def test_analysis_pipeline_feature_flag_enables_route(
    roots: PropagationAuthorityRoots,
) -> None:
    analysis = _analysis_pipeline(enable_change_propagation=True)
    result = analysis.run_change_propagation(_pipeline_request(roots))
    assert result.enabled is True
    assert result.admitted is True
    assert result.packet is not None
    assert result.provider_invoked is False
    assert result.write_paths == ("pkg/caller.py",)


# ---------------------------------------------------------------------------
# Ordered pipeline stages / analytical success
# ---------------------------------------------------------------------------


def test_ordered_stages_admit_analytical_packet_without_provider(
    roots: PropagationAuthorityRoots,
) -> None:
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    result = ChangePropagationPipeline(policy=policy).run(_pipeline_request(roots))

    assert result.enabled
    assert result.disposition == PipelineDisposition.ADMITTED.value
    assert result.provider_invoked is False
    assert result.plan is not None
    assert result.plan.disposition is PlanDisposition.ADMITTED
    assert result.packet is not None
    assert result.analytical_step_ids
    assert result.model_required_step_ids == ()
    # Stages through pre_provider_gate (gate skipped but stage recorded).
    assert PipelineStage.CHANGE_SET.value in result.stages_completed
    assert PipelineStage.DELTA.value in result.stages_completed
    assert PipelineStage.GRAPH_INDEX.value in result.stages_completed
    assert PipelineStage.CLOSURE_FRONTIER.value in result.stages_completed
    assert PipelineStage.CONSUMER_INVENTORY.value in result.stages_completed
    assert PipelineStage.VALUE_BEHAVIOR_PROOF.value in result.stages_completed
    assert PipelineStage.PLAN_ADMISSION.value in result.stages_completed
    assert PipelineStage.PACKET_MATERIALIZE.value in result.stages_completed
    assert PipelineStage.PRE_PROVIDER_GATE.value in result.stages_completed
    # Mutation stages absent when not requested.
    assert PipelineStage.TRANSACTION.value not in result.stages_completed


def test_scope_mismatch_rejects(roots: PropagationAuthorityRoots) -> None:
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    result = ChangePropagationPipeline(policy=policy).run(
        _pipeline_request(
            roots,
            task_write_paths=("pkg/other.py",),
        )
    )
    assert result.disposition == PipelineDisposition.REJECTED.value
    assert "scope_mismatch" in result.reason_codes


def test_plan_abstention_preserves_no_packet(roots: PropagationAuthorityRoots) -> None:
    # Empty obligations with complete closure forces planner abstention.
    consumer = _consumer()
    closure = _closure(roots, (consumer,))
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=closure,
        obligations=(),  # omission → abstain
        value_mapping_proofs=(),
        analytical_transforms=(),
        write_spans=(
            PlanPathSpan(
                path="pkg/caller.py",
                start=0,
                end=10,
                artifact_id="blob:caller",
                before_hash="sha256:caller-before",
            ),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    result = ChangePropagationPipeline(policy=policy).run(
        ChangePropagationPipelineRequest(
            roots=roots,
            evidence_bundle=evidence,
            impact_closure=closure,
            obligations=(),
        )
    )
    assert result.disposition in {
        PipelineDisposition.ABSTAINED.value,
        PipelineDisposition.REJECTED.value,
    }
    assert result.packet is None


# ---------------------------------------------------------------------------
# Transaction + fixed-point completion
# ---------------------------------------------------------------------------


def test_mutation_requires_transaction_and_fixed_point(
    roots: PropagationAuthorityRoots,
) -> None:
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    plan = admission.plan
    result = ChangePropagationPipeline(policy=policy).run(
        _pipeline_request(
            roots,
            execute_mutation=True,
            transaction_lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
            step_applicator=_passing_applicator,
            candidate_evidence=_candidate_evidence(plan),
            evidence_bundle=evidence,
            impact_closure=evidence.impact_closure,
            obligations=evidence.obligations,
            analytical_transforms=evidence.analytical_transforms,
            value_mapping_proofs=evidence.value_mapping_proofs,
        )
    )
    assert result.complete, result.detail
    assert result.disposition == PipelineDisposition.COMPLETE.value
    assert result.transaction is not None
    assert result.transaction.state is TransactionState.COMMITTED
    assert isinstance(result.completion, PropagationCompletionReceipt)
    assert result.provider_invoked is False
    assert PipelineStage.TRANSACTION.value in result.stages_completed
    assert PipelineStage.FIXED_POINT_VALIDATION.value in result.stages_completed


def test_transaction_failure_rolls_back(roots: PropagationAuthorityRoots) -> None:
    def failing_applicator(request: StepApplyRequest) -> StepApplyResult:
        return StepApplyResult(
            disposition=StepExecutionDisposition.FAILED,
            written_paths=(),
        )

    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    restored: list[bool] = []

    def restore(checkpoint: Any) -> bool:
        restored.append(True)
        return True

    result = ChangePropagationPipeline(policy=policy).run(
        _pipeline_request(
            roots,
            execute_mutation=True,
            transaction_lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
            step_applicator=failing_applicator,
            restore_adapter=restore,
            candidate_evidence=_candidate_evidence(),
        )
    )
    assert result.disposition == PipelineDisposition.ROLLED_BACK.value
    assert result.rolled_back is True
    assert result.completion is None
    assert result.complete is False
    assert restored == [True]


def test_missing_candidate_evidence_incomplete(
    roots: PropagationAuthorityRoots,
) -> None:
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    result = ChangePropagationPipeline(policy=policy).run(
        _pipeline_request(
            roots,
            execute_mutation=True,
            transaction_lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
            step_applicator=_passing_applicator,
            # candidate_evidence omitted
        )
    )
    assert result.disposition == PipelineDisposition.INCOMPLETE.value
    assert result.transaction is not None
    assert result.transaction.state is TransactionState.COMMITTED
    assert result.completion is None


def test_module_entry_point_matches_class(roots: PropagationAuthorityRoots) -> None:
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    req = _pipeline_request(roots)
    a = run_change_propagation_pipeline(req, policy=policy)
    b = ChangePropagationPipeline(policy=policy).run(req)
    assert a.disposition == b.disposition
    assert a.plan_id == b.plan_id
    assert a.provider_invoked is False


# ---------------------------------------------------------------------------
# Provider router: analytical skip / legacy compatibility
# ---------------------------------------------------------------------------


def test_route_contract_packet_analytical_skips_provider(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    assert packet.model_required_step_ids == ()

    called = {"n": 0}

    def grok(*_a: Any, **_k: Any) -> dict[str, Any]:
        called["n"] += 1
        return {"proposal": {"patch": ""}}

    result = route_contract_packet(
        packet,
        current_snapshot_id="tree:candidate",
        grok_provider=grok,
    )
    assert called["n"] == 0
    assert result.status is RouteStatus.SUCCEEDED
    assert result.reason_code == ProviderReason.LOCAL_ONLY.value
    assert result.write_performed is False
    assert result.completion_authoritative is False


def test_route_contract_packet_legacy_still_works() -> None:
    """Legacy packets without propagation shape keep the classic path."""

    class _LegacyPacket:
        packet_id = "packet:legacy"
        implementable = True
        snapshot_id = "snap:1"

        def to_dict(self) -> dict[str, Any]:
            return {
                "packet_id": self.packet_id,
                "snapshot_id": self.snapshot_id,
                "schema": "ipfs_accelerate_py/agent-supervisor/code-edit-packet@1",
            }

    # Without providers, route should fail closed without raising TypeError
    # on the legacy ImplementationProviderRouter path.
    result = route_contract_packet(
        _LegacyPacket(),
        current_snapshot_id="snap:1",
        local_only=True,
    )
    assert result.status in {
        RouteStatus.SUCCEEDED,
        RouteStatus.FALLBACK,
        RouteStatus.DEFERRED,
        RouteStatus.REJECTED,
    }


# ---------------------------------------------------------------------------
# Refinery integration
# ---------------------------------------------------------------------------


def test_refinery_projects_change_propagation_packet(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    refinery = ContractMismatchRefinery(
        policy=ContractMismatchRefineryPolicy(
            accept_change_propagation_packets=True,
        )
    )
    projection = refinery.project_change_propagation(
        packet,
        current_roots=roots,
        current_tree_id=roots.candidate_tree_id,
        provider_outputs=packet.permitted_write_paths,
    )
    assert projection is not None
    reason = getattr(getattr(projection, "reason", None), "value", None)
    assert reason == "emitted", getattr(projection, "detail", reason)
    # Scopes equal admitted paths.
    projected = tuple(getattr(projection, "write_scope", ()) or ())
    assert tuple(sorted(projected)) == tuple(sorted(packet.permitted_write_paths))


def test_refinery_disabled_change_propagation(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    refinery = ContractMismatchRefinery(
        policy=ContractMismatchRefineryPolicy(
            accept_change_propagation_packets=False,
        )
    )
    with pytest.raises(ContractMismatchRefineryError, match="disabled"):
        refinery.project_change_propagation(packet, current_roots=roots)


# ---------------------------------------------------------------------------
# Daemon cutover
# ---------------------------------------------------------------------------


def test_daemon_execute_transactional_propagation(
    roots: PropagationAuthorityRoots,
) -> None:
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._record_event = MagicMock()  # type: ignore[attr-defined]

    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    result = daemon.execute_transactional_change_propagation(
        _pipeline_request(
            roots,
            execute_mutation=True,
            transaction_lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
            step_applicator=_passing_applicator,
            candidate_evidence=_candidate_evidence(admission.plan),
            evidence_bundle=evidence,
            impact_closure=evidence.impact_closure,
            obligations=evidence.obligations,
            analytical_transforms=evidence.analytical_transforms,
            value_mapping_proofs=evidence.value_mapping_proofs,
        ),
        enable=True,
    )
    assert result.complete, result.detail
    assert isinstance(result.completion, PropagationCompletionReceipt)
    daemon._record_event.assert_called()  # type: ignore[attr-defined]


def test_daemon_require_completion_and_bypass_guard(
    roots: PropagationAuthorityRoots,
) -> None:
    daemon = object.__new__(PortalImplementationDaemon)

    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    plan = admission.plan
    assert plan is not None
    lease = _lease()
    hashes = _hashes("pkg/caller.py")
    report = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan, lease=lease, path_before_hashes=hashes
    )
    assert report.committed

    completion = daemon.require_change_propagation_completion(
        plan,
        report.transaction,
        evidence=_candidate_evidence(plan),
        execution_report=report,
    )
    assert isinstance(completion, PropagationCompletionReceipt)

    # Bypass guards.
    with pytest.raises(RuntimeError, match="ChangePropagationTransaction"):
        daemon.assert_no_propagation_write_bypass(
            write_performed=True,
            transaction_committed=False,
            completion_present=False,
        )
    with pytest.raises(RuntimeError, match="PropagationCompletionReceipt"):
        daemon.assert_no_propagation_write_bypass(
            write_performed=True,
            transaction_committed=True,
            completion_present=False,
        )
    # Clean path.
    daemon.assert_no_propagation_write_bypass(
        write_performed=True,
        transaction_committed=True,
        completion_present=True,
    )


def test_daemon_require_completion_rejects_incomplete(
    roots: PropagationAuthorityRoots,
) -> None:
    daemon = object.__new__(PortalImplementationDaemon)
    obligations = (_obligation(roots),)
    from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
        PropagationPlanStep,
        PropagationTransaction,
    )

    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:incomplete",
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure_id="closure:one",
        disposition=PlanDisposition.ADMITTED,
        obligations=obligations,
        obligation_set_id=obligation_set_identity(obligations),
        steps=(
            PropagationPlanStep(
                step_id="step:migrate-one",
                kind=PlanStepKind.ANALYTICAL,
                obligation_ids=(obligations[0].obligation_id,),
                transform_id="transform:add-arg",
                write_paths=("pkg/caller.py",),
                read_paths=("pkg/caller.py",),
            ),
        ),
        permitted_read_paths=("pkg/caller.py",),
        permitted_write_paths=("pkg/caller.py",),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:plan",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    # Not committed → incomplete.
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:not-committed",
        plan_id=plan.plan_id,
        state=TransactionState.FAILED,
        checkpoint_id="checkpoint:x",
        completed_step_ids=(),
        diagnostic_refs=("diagnostic:failed",),
        lease_id="lease:x",
    )
    with pytest.raises(ChangePropagationValidationError):
        daemon.require_change_propagation_completion(
            plan, txn, evidence=_candidate_evidence(plan)
        )


# ---------------------------------------------------------------------------
# Cold import / legacy policy flags
# ---------------------------------------------------------------------------


def test_analysis_policy_preserves_legacy_defaults() -> None:
    policy = AnalysisPipelinePolicy()
    assert policy.enable_proof_gated_contract_repair is False
    assert policy.enable_change_propagation is False
    # Both can be enabled independently.
    both = AnalysisPipelinePolicy(
        enable_proof_gated_contract_repair=True,
        enable_change_propagation=True,
    )
    assert both.enable_proof_gated_contract_repair is True
    assert both.enable_change_propagation is True
    payload = both.to_dict()
    assert payload["enable_change_propagation"] is True


def test_change_propagation_module_imports_cold() -> None:
    """Importing the pipeline module must not force provider/daemon imports."""

    import importlib
    import sys

    # Drop heavy modules if already loaded in this process so we can observe
    # that the pipeline package itself does not pull them at import time.
    heavy = [
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.change_propagation_provider_router",
        "ipfs_accelerate_py.agent_supervisor.planning.change_propagation_transaction",
        "ipfs_accelerate_py.agent_supervisor.validation.change_propagation_validation",
    ]
    # Just ensure re-import of the pipeline succeeds without error; heavy
    # modules may already be in sys.modules from earlier tests.
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_pipeline"
    )
    assert mod.CHANGE_PROPAGATION_PIPELINE_VERSION == 1
    # Public surface present without executing providers.
    assert hasattr(mod, "ChangePropagationPipeline")
    assert hasattr(mod, "run_change_propagation_pipeline")
    # sys.modules may contain heavy modules from this suite; the pipeline
    # source itself uses local imports for those names.
    source = open(mod.__file__, encoding="utf-8").read()
    assert "from ..planning.change_propagation_transaction import" in source
    assert "from ..todo_daemon.change_propagation_provider_router import" in source
    # Top-level import block should not pull those modules.
    header = source.split("class ChangePropagationPipelineError")[0]
    assert "change_propagation_transaction" not in header
    assert "change_propagation_provider_router" not in header
    assert "change_propagation_validation" not in header
    del heavy, sys  # silence unused when import already warm
