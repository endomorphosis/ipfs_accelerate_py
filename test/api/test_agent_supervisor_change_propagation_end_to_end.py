"""RPR-047: end-to-end operations validation for change propagation.

Acceptance:

* Extend validation without removing legacy RPR-020/RPR-G100 checks;
* require RPR-G110/RPR-G220 and terminal RPR-047, correct dependency chain,
  change_propagation_policy gates and six new zero safety floors;
* verify protected paths/refill isolation and exact source bindings;
* seeded two-to-three argument case detects all callers, proves one source
  or threads it, applies an atomic analytical plan, rediffs to a fixed point
  and emits completion;
* negative wrong-value, unknown-frontier, partial-SCC and LLM-scope cases fail;
* stopped/running supervisor health remains correctly reported and a clean
  four-shard board can drain.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AnalyticalTransform,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    PlanDisposition,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    TransformDisposition,
    TransformKind,
    TransactionState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_pipeline import (
    ChangePropagationPipeline,
    ChangePropagationPipelinePolicy,
    ChangePropagationPipelineRequest,
    PipelineDisposition,
    PipelineStage,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_plan import (
    ChangePropagationPlanner,
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanResourceBounds,
    PlanValidationCommand,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_transaction import (
    StepApplyRequest,
    StepApplyResult,
    StepExecutionDisposition,
    TransactionLease,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)
from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_validation import (
    CandidatePropagationEvidence,
    ChangePropagationValidator,
    ClosureRecomputeEvidence,
    ConsumerDischargeEvidence,
    DEFAULT_POLICY_REQUIRED_TOOLS,
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPS_PATH = _REPO_ROOT / "scripts" / "validate_proof_gated_contract_repair.py"
_BEFORE_HASH = "sha256:caller-before-rpr047"


def _load_ops():
    name = "validate_proof_gated_contract_repair_rpr047_e2e"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _OPS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ops = _load_ops()


# ---------------------------------------------------------------------------
# Multi-caller two-to-three argument builders
# ---------------------------------------------------------------------------


CALLER_SPECS: tuple[tuple[str, str, str], ...] = (
    ("consumer:direct", "src/client.py", "direct"),
    ("consumer:aliased", "src/alias_api.py", "aliased"),
    ("consumer:wrapped", "src/wrapper.py", "wrapped"),
    ("consumer:method", "src/service.py", "method"),
)


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-047-e2e",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-047",
        index_id="index:rpr-047",
        model_id="model:rpr-047",
        config_id="config:rpr-047",
        translator_id="translator:rpr-047",
        toolchain_id="toolchain:rpr-047",
        policy_id="policy:rpr-047",
    )


def _node(path: str, symbol: str) -> GraphNodeRef:
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
    consumer_id: str,
    path: str,
) -> ConsumerMigrationObligation:
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=f"obligation:{consumer_id}",
        consumer_id=consumer_id,
        delta_id="delta:process-arity",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add-context",),
        node=_node(path, f"symbol:{consumer_id}"),
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        behavior_contract_ids=(),
        invalidation_refs=("tree:candidate",),
    )


def _mapping(*, consumer_id: str) -> ValueMappingProof:
    return ValueMappingProof(
        requirement_id="missing:context",
        consumer_id=consumer_id,
        disposition=SynthesisDisposition.UNIQUE_PROVED,
        facet_results=(),
        proved_candidate_ids=("candidate:request_context",),
        refuted_candidate_ids=(),
        expression_ref="expr:request_context",
        type_ref="type:Context",
        repository_id="repository:rpr-047-e2e",
        tree_id="tree:candidate",
        toolchain_id="toolchain:rpr-047",
        policy_id="policy:rpr-047",
        reason_codes=("unique_source",),
    )


def _transform(
    roots: PropagationAuthorityRoots,
    *,
    consumer_id: str,
    path: str,
) -> AnalyticalTransform:
    return AnalyticalTransform(
        roots=roots,
        transform_id=f"transform:add-arg:{consumer_id}",
        kind=TransformKind.ADD_ARGUMENT,
        disposition=TransformDisposition.ADMITTED,
        obligation_ids=(f"obligation:{consumer_id}",),
        target_paths=(path,),
        expression_refs=("expr:request_context",),
        proof_refs=("proof:transform",),
    )


def _two_to_three_bundle(roots: PropagationAuthorityRoots) -> PlanEvidenceBundle:
    consumers = tuple(
        ImpactConsumer(
            consumer_id=consumer_id,
            node=_node(path, f"symbol:{consumer_id}"),
            depth=1,
            mandatory=True,
            edge_refs=(f"edge:{kind}",),
        )
        for consumer_id, path, kind in CALLER_SPECS
    )
    obligations = tuple(
        _obligation(roots, consumer_id=consumer_id, path=path)
        for consumer_id, path, _kind in CALLER_SPECS
    )
    mappings = tuple(_mapping(consumer_id=consumer_id) for consumer_id, _, _ in CALLER_SPECS)
    transforms = tuple(
        _transform(roots, consumer_id=consumer_id, path=path)
        for consumer_id, path, _kind in CALLER_SPECS
    )
    write_spans = tuple(
        PlanPathSpan(
            path=path,
            start=0,
            end=40,
            artifact_id=f"blob:{consumer_id}",
            before_hash=_BEFORE_HASH,
        )
        for consumer_id, path, _kind in CALLER_SPECS
    )
    return PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:process-arity",
        delta_id="delta:process-arity",
        impact_closure=ImpactClosureReceipt(
            roots=roots,
            delta_id="delta:process-arity",
            completeness=ImpactCompleteness.COMPLETE,
            consumers=consumers,
            sccs=(),
            frontier_node_ids=(),
            frontier_edge_ids=(),
            validation_refs=("validation:impact",),
            resource_bound_refs=("bound:impact",),
            evidence_refs=("evidence:graph",),
        ),
        obligations=obligations,
        value_mapping_proofs=mappings,
        analytical_transforms=transforms,
        placement_decisions=(),
        read_spans=write_spans,
        write_spans=write_spans,
        validation_commands=(
            PlanValidationCommand(
                command_id="validate:pytest-arity",
                argv=("python", "-m", "pytest", "-q", "test_process_arity.py"),
                required=True,
            ),
        ),
        resource_bounds=PlanResourceBounds(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        candidate_set_id="candidate-set:process-arity",
        expected_roots=roots,
    )


def _lease(paths: tuple[str, ...]) -> TransactionLease:
    return TransactionLease(
        lease_id="lease:rpr-047",
        fence_id="fence:1",
        holder_id="holder:txn",
        permitted_write_paths=paths,
        permitted_read_paths=paths,
        active=True,
    )


def _hashes(*paths: str) -> tuple[PathBeforeHash, ...]:
    return tuple(PathBeforeHash(path=path, before_hash=_BEFORE_HASH) for path in paths)


def _passing_applicator(request: StepApplyRequest) -> StepApplyResult:
    return StepApplyResult(
        disposition=StepExecutionDisposition.PASSED,
        written_paths=request.step.write_paths,
        observed_before_hashes=tuple(
            PathBeforeHash(path=path, before_hash=_BEFORE_HASH)
            for path in request.step.write_paths
        ),
    )


def _candidate_evidence(
    plan,
    *,
    paths: tuple[str, ...],
    consumer_ids: tuple[str, ...],
    obligation_ids: tuple[str, ...],
) -> CandidatePropagationEvidence:
    node_ids = tuple(f"node:symbol:{cid}" for cid in consumer_ids)
    iteration = FixedPointIterationEvidence(
        iteration=1,
        index_rebuild=PropagationIndexRebuildEvidence(
            candidate_tree_id="tree:candidate",
            index_id="index:rpr-047",
            graph_id="graph:rpr-047",
            rebuilt_source_paths=paths,
            rebuilt_ast_paths=paths,
            rebuilt_vector_row_ids=tuple(f"vector:{cid}" for cid in consumer_ids),
            rebuilt_graph_node_ids=node_ids,
            tombstone_ids=(),
            affected_paths=paths,
            clean_rebuild_equivalent=True,
        ),
        delta_reextract=DeltaReextractEvidence(
            candidate_tree_id="tree:candidate",
            original_delta_id=plan.delta_id,
            reextracted_delta_id=plan.delta_id,
            breaking_delta_ids=(plan.delta_id,),
            unplanned_breaking_delta_ids=(),
            extraction_receipt_id="delta-receipt:ok",
            matches_plan_delta=True,
        ),
        resolution=ResolutionEvidence(
            candidate_tree_id="tree:candidate",
            resolved_call_ids=tuple(f"call:{cid}" for cid in consumer_ids),
            resolved_data_flow_ids=("data:context",),
            resolved_schema_ids=("schema:request",),
            resolved_wiring_ids=("wire:router",),
            unresolved_ids=(),
            resolution_receipt_id="resolution:ok",
            complete=True,
        ),
        closure_recompute=ClosureRecomputeEvidence(
            candidate_tree_id="tree:candidate",
            original_closure_id=plan.impact_closure_id,
            recomputed_closure_id="closure:recomputed",
            consumer_ids=consumer_ids,
            mandatory_consumer_ids=consumer_ids,
            frontier_node_ids=(),
            required_frontier_ids=(),
            uncovered_frontier_ids=(),
            complete=True,
            receipt_id="closure:ok",
        ),
        consumer_discharge=ConsumerDischargeEvidence(
            candidate_tree_id="tree:candidate",
            original_obligation_ids=obligation_ids,
            discharged_obligation_ids=obligation_ids,
            unresolved_mandatory_ids=(),
            omitted_dependent_ids=(),
            double_discharged_ids=(),
            receipt_id="discharge:ok",
        ),
        second_order=SecondOrderImpactEvidence(
            candidate_tree_id="tree:candidate",
            new_delta_ids=(),
            new_consumer_ids=(),
            residual_frontier_ids=(),
            requires_another_iteration=False,
            receipt_id="second:fixed",
        ),
        proof_reconstruction=ProofReconstructionEvidence(
            candidate_tree_id="tree:candidate",
            original_proof_refs=tuple(plan.proof_refs) or ("proof:plan",),
            reconstructed_proof_refs=tuple(plan.proof_refs) + ("proof:reconstructed",),
            introduced_proof_refs=("proof:introduced",),
            failed_proof_refs=(),
            all_mandatory_reconstructed=True,
            receipt_id="proof:ok",
        ),
        policy_tools=PolicyToolEvidence(
            candidate_tree_id="tree:candidate",
            required_families=DEFAULT_POLICY_REQUIRED_TOOLS,
            results=build_passing_tool_evidence(
                "tree:candidate", "policy:rpr-047"
            ).results,
            policy_id="policy:rpr-047",
        ),
        impacted_tests=ImpactedTestEvidence(
            candidate_tree_id="tree:candidate",
            focused_test_ids=("test:focused-arity",),
            impacted_test_ids=("test:impacted-process",),
            required_dependant_ids=("test:dependant-route",),
            executed_test_ids=(
                "test:focused-arity",
                "test:impacted-process",
                "test:dependant-route",
            ),
            passed_test_ids=(
                "test:focused-arity",
                "test:impacted-process",
                "test:dependant-route",
            ),
            failed_test_ids=(),
            omitted_dependant_ids=(),
            dependency_complete=True,
        ),
        integrity=IntegrityEvidence(
            candidate_tree_id="tree:candidate",
            contracts_deleted=(),
            contracts_weakened=(),
            tests_deleted=(),
            tests_weakened=(),
            checkers_deleted=(),
            checkers_weakened=(),
            findings_suppressed=(),
            original_finding_id="finding:arity",
            original_finding_closed=True,
        ),
    )
    return CandidatePropagationEvidence(
        candidate_tree_id="tree:candidate",
        iterations=(iteration,),
    )


# ---------------------------------------------------------------------------
# Operations surface / control-plane gates
# ---------------------------------------------------------------------------


def test_declared_outputs_exist() -> None:
    assert _OPS_PATH.is_file()
    assert Path(__file__).is_file()
    assert (
        _REPO_ROOT / "test/api/test_agent_supervisor_contract_repair_rollout.py"
    ).is_file()


def test_operations_symbols_and_extension_ids() -> None:
    assert hasattr(ops, "ProofGatedContractRepairOperations")
    assert hasattr(ops, "ChangePropagationEndToEnd")
    assert ops.TERMINAL_TASK_ID == "RPR-047"
    assert ops.EXTENSION_CONTROL_GOAL_ID == "RPR-G110"
    assert ops.EXTENSION_ROLLOUT_GOAL_ID == "RPR-G220"
    # Legacy constants remain for RPR-020 compatibility.
    assert ops.TASK_ID == "RPR-020"
    assert ops.GOAL_ID == "RPR-G100"
    assert len(ops.PROPAGATION_SAFETY_FLOOR_KEYS) == 6
    assert len(ops.SAFETY_FLOOR_KEYS) == 4


def test_legacy_and_extension_gates_pass() -> None:
    dag = ops.check_plan_objective_task_dag(_REPO_ROOT)
    assert dag.status is ops.CheckStatus.PASS, dag.detail
    assert "RPR-G100" in dag.evidence["goal_ids"]
    assert "RPR-G110" in dag.evidence["goal_ids"]
    assert "RPR-G220" in dag.evidence["goal_ids"]
    assert "RPR-020" in dag.evidence["task_ids"]
    assert "RPR-047" in dag.evidence["task_ids"]

    bindings = ops.check_exact_source_bindings(_REPO_ROOT)
    assert bindings.status is ops.CheckStatus.PASS, bindings.detail

    policy = ops.check_change_propagation_policy(_REPO_ROOT)
    assert policy.status is ops.CheckStatus.PASS, policy.detail
    for key in ops.PROPAGATION_SAFETY_FLOOR_KEYS:
        assert policy.evidence["propagation_safety_floors"][key] == 0

    protected = ops.check_protected_paths_and_refill_isolation(_REPO_ROOT)
    assert protected.status is ops.CheckStatus.PASS, protected.detail

    shards = ops.check_four_shard_board_drain(_REPO_ROOT)
    assert shards.status is ops.CheckStatus.PASS, shards.detail
    assert shards.evidence["ready_after_non_terminal_complete"] == ["RPR-047"]


# ---------------------------------------------------------------------------
# Seeded corpus end-to-end (fixture measurement path)
# ---------------------------------------------------------------------------


def test_seeded_corpus_positive_and_negatives() -> None:
    report = ops.ChangePropagationEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    assert report["interface"] == "ChangePropagationEndToEnd@1"
    assert report["task_id"] == "RPR-047"
    assert report["goal_id"] == "RPR-G220"
    assert report["valid"] is True, report

    positive = report["positive"]
    assert positive["ok"] is True
    assert positive["admitted"] is True
    assert positive["completion_success"] is True
    assert positive["analytical_path"] is True
    assert positive["unique_source_precise"] is True
    assert positive["consumer_precise"] is True
    assert positive["plan_complete"] is True
    assert positive["fixed_point_iterations"] >= 1
    assert positive["caller_count"] >= 4
    assert set(positive["caller_kinds"]) >= {
        "direct",
        "aliased",
        "wrapped",
        "method",
    }

    negatives = report["negatives"]
    for scenario in ops.ChangePropagationEndToEnd.NEGATIVE_SCENARIOS:
        item = negatives[scenario]
        assert item["present"] is True, scenario
        assert item["ok_fail_closed"] is True, (scenario, item)
        assert item["admitted"] is False
        assert item["completion_success"] is False


def test_negative_wrong_value_unknown_frontier_partial_scc_llm_scope() -> None:
    report = ops.ChangePropagationEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    wrong = report["negatives"]["same_typed_wrong_information"]
    assert wrong["outcome_kind"] == "wrong_value"

    frontier = report["negatives"]["reflection_plugin_registry_ffi_frontier"]
    assert frontier["outcome_kind"] == "open_frontier"

    partial = report["negatives"]["partial_transaction"]
    assert partial["scc_rollback"] is True
    assert partial["outcome_kind"] == "rollback_error"

    llm = report["negatives"]["llm_scope_escape"]
    assert llm["llm_scope_escape"] is False
    assert llm["admitted"] is False


# ---------------------------------------------------------------------------
# Full analytical pipeline: multi-caller two-to-three fixed-point completion
# ---------------------------------------------------------------------------


def test_two_to_three_detects_all_callers_and_completes(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _two_to_three_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted, getattr(admission, "detail", admission)
    plan = admission.plan
    assert plan is not None
    assert plan.disposition is PlanDisposition.ADMITTED

    # Every two-arg caller must receive its own migration obligation.
    obligation_ids = {item.obligation_id for item in plan.obligations}
    consumer_ids = tuple(cid for cid, _, _ in CALLER_SPECS)
    for consumer_id, path, _kind in CALLER_SPECS:
        assert f"obligation:{consumer_id}" in obligation_ids
    assert len(plan.obligations) == len(CALLER_SPECS)

    # Unique proved source is available for every consumer.
    assert all(
        mapping.disposition is SynthesisDisposition.UNIQUE_PROVED
        for mapping in evidence.value_mapping_proofs
    )

    paths = tuple(path for _, path, _ in CALLER_SPECS)
    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    result = ChangePropagationPipeline(policy=policy).run(
        ChangePropagationPipelineRequest(
            roots=roots,
            evidence_bundle=evidence,
            impact_closure=evidence.impact_closure,
            obligations=evidence.obligations,
            analytical_transforms=evidence.analytical_transforms,
            value_mapping_proofs=evidence.value_mapping_proofs,
            execute_mutation=True,
            task_write_paths=paths,
            writer_write_paths=paths,
            transaction_lease=_lease(paths),
            path_before_hashes=_hashes(*paths),
            step_applicator=_passing_applicator,
            candidate_evidence=_candidate_evidence(
                plan,
                paths=paths,
                consumer_ids=consumer_ids,
                obligation_ids=tuple(sorted(obligation_ids)),
            ),
        )
    )

    assert result.complete, result.detail
    assert result.disposition == PipelineDisposition.COMPLETE.value
    assert result.provider_invoked is False
    assert result.transaction is not None
    assert result.transaction.state is TransactionState.COMMITTED
    assert isinstance(result.completion, PropagationCompletionReceipt)
    assert PipelineStage.PLAN_ADMISSION.value in result.stages_completed
    assert PipelineStage.TRANSACTION.value in result.stages_completed
    assert PipelineStage.FIXED_POINT_VALIDATION.value in result.stages_completed
    assert result.analytical_step_ids
    assert result.model_required_step_ids == ()


def test_partial_scc_group_cannot_complete(roots: PropagationAuthorityRoots) -> None:
    """Partial consumer discharge (SCC-like residual) blocks fixed-point completion."""

    evidence = _two_to_three_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    plan = admission.plan
    assert plan is not None
    paths = tuple(path for _, path, _ in CALLER_SPECS)
    consumer_ids = tuple(cid for cid, _, _ in CALLER_SPECS)
    obligation_ids = tuple(item.obligation_id for item in plan.obligations)

    # Drop one mandatory obligation from discharge (partial SCC / partial plan).
    partial_discharged = obligation_ids[:-1]
    candidate = _candidate_evidence(
        plan,
        paths=paths,
        consumer_ids=consumer_ids,
        obligation_ids=obligation_ids,
    )
    # Rebuild with incomplete discharge.
    incomplete_iteration = FixedPointIterationEvidence(
        iteration=1,
        index_rebuild=candidate.iterations[0].index_rebuild,
        delta_reextract=candidate.iterations[0].delta_reextract,
        resolution=candidate.iterations[0].resolution,
        closure_recompute=candidate.iterations[0].closure_recompute,
        consumer_discharge=ConsumerDischargeEvidence(
            candidate_tree_id="tree:candidate",
            original_obligation_ids=obligation_ids,
            discharged_obligation_ids=partial_discharged,
            unresolved_mandatory_ids=(obligation_ids[-1],),
            omitted_dependent_ids=(),
            double_discharged_ids=(),
            receipt_id="discharge:partial",
        ),
        second_order=candidate.iterations[0].second_order,
        proof_reconstruction=candidate.iterations[0].proof_reconstruction,
        policy_tools=candidate.iterations[0].policy_tools,
        impacted_tests=candidate.iterations[0].impacted_tests,
        integrity=candidate.iterations[0].integrity,
    )
    bad_evidence = CandidatePropagationEvidence(
        candidate_tree_id="tree:candidate",
        iterations=(incomplete_iteration,),
    )

    policy = ChangePropagationPipelinePolicy(enable_change_propagation=True)
    result = ChangePropagationPipeline(policy=policy).run(
        ChangePropagationPipelineRequest(
            roots=roots,
            evidence_bundle=evidence,
            impact_closure=evidence.impact_closure,
            obligations=evidence.obligations,
            analytical_transforms=evidence.analytical_transforms,
            value_mapping_proofs=evidence.value_mapping_proofs,
            execute_mutation=True,
            task_write_paths=paths,
            writer_write_paths=paths,
            transaction_lease=_lease(paths),
            path_before_hashes=_hashes(*paths),
            step_applicator=_passing_applicator,
            candidate_evidence=bad_evidence,
        )
    )
    assert result.complete is False
    assert result.disposition != PipelineDisposition.COMPLETE.value
    # Partial discharge may emit a failed completion receipt but never a
    # successful fixed-point completion.
    if result.completion is not None:
        assert result.completion.disposition.value in {
            "failed",
            "incomplete",
            "abstained",
        }
        assert result.completion.unresolved_mandatory_ids
    assert "unresolved_mandatory" in result.reason_codes or (
        "consumer_not_discharged" in result.reason_codes
    )


def test_unknown_frontier_forces_plan_abstention(
    roots: PropagationAuthorityRoots,
) -> None:
    """Required unknown frontier cannot be silently closed into an admitted plan."""

    evidence = _two_to_three_bundle(roots)
    # Inject an open frontier on the impact closure.
    open_closure = ImpactClosureReceipt(
        roots=roots,
        delta_id=evidence.delta_id,
        completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
        consumers=evidence.impact_closure.consumers,
        sccs=(),
        frontier_node_ids=("node:dynamic-plugin",),
        frontier_edge_ids=("edge:reflection",),
        validation_refs=("validation:impact",),
        resource_bound_refs=("bound:impact",),
        evidence_refs=("evidence:graph",),
    )
    open_evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id=evidence.change_set_id,
        delta_id=evidence.delta_id,
        impact_closure=open_closure,
        obligations=evidence.obligations,
        value_mapping_proofs=evidence.value_mapping_proofs,
        analytical_transforms=evidence.analytical_transforms,
        placement_decisions=(),
        read_spans=evidence.read_spans,
        write_spans=evidence.write_spans,
        validation_commands=evidence.validation_commands,
        resource_bounds=evidence.resource_bounds,
        proof_refs=evidence.proof_refs,
        invalidation_refs=evidence.invalidation_refs,
        candidate_set_id=evidence.candidate_set_id,
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(open_evidence)
    assert admission.admitted is False
    assert admission.plan is None or admission.plan.disposition is not PlanDisposition.ADMITTED


def test_wrong_value_mapping_forces_abstention(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _two_to_three_bundle(roots)
    wrong_mappings = tuple(
        ValueMappingProof(
            requirement_id="missing:context",
            consumer_id=consumer_id,
            disposition=SynthesisDisposition.REFUTED,
            facet_results=(),
            proved_candidate_ids=(),
            refuted_candidate_ids=("candidate:same_typed_wrong",),
            expression_ref="expr:wrong",
            type_ref="type:Context",
            repository_id="repository:rpr-047-e2e",
            tree_id="tree:candidate",
            toolchain_id="toolchain:rpr-047",
            policy_id="policy:rpr-047",
            reason_codes=("wrong_value",),
        )
        for consumer_id, _, _ in CALLER_SPECS
    )
    wrong_evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id=evidence.change_set_id,
        delta_id=evidence.delta_id,
        impact_closure=evidence.impact_closure,
        obligations=evidence.obligations,
        value_mapping_proofs=wrong_mappings,
        analytical_transforms=evidence.analytical_transforms,
        placement_decisions=(),
        read_spans=evidence.read_spans,
        write_spans=evidence.write_spans,
        validation_commands=evidence.validation_commands,
        resource_bounds=evidence.resource_bounds,
        proof_refs=evidence.proof_refs,
        invalidation_refs=evidence.invalidation_refs,
        candidate_set_id=evidence.candidate_set_id,
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(wrong_evidence)
    assert admission.admitted is False


# ---------------------------------------------------------------------------
# Supervisor health + four-shard drain
# ---------------------------------------------------------------------------


def test_stopped_supervisor_health_is_healthy(tmp_path: Path) -> None:
    result = ops.check_supervisor_process_state(
        _REPO_ROOT, state_root=tmp_path / "missing-program"
    )
    assert result.status is ops.CheckStatus.PASS, result.detail
    assert result.evidence["master_status"] == "stopped"


def test_running_with_dead_pid_fails(tmp_path: Path) -> None:
    program = tmp_path / "program"
    lane0 = program / "state" / "lane-0"
    lane0.mkdir(parents=True)
    (lane0 / "rpr_lane_0_supervisor_status.json").write_text(
        json.dumps({"status": "running", "pid": 2**30, "supervisor_pid": 2**30}),
        encoding="utf-8",
    )
    (lane0 / "rpr_lane_0_task_state.json").write_text(
        json.dumps(
            {
                "status": "available",
                "active_task_id": "",
                "eligible_ready_count": 0,
                "blocked_count": 0,
            }
        ),
        encoding="utf-8",
    )
    bad = ops.check_supervisor_process_state(
        _REPO_ROOT, state_root=program, lane_count=1
    )
    assert bad.status is ops.CheckStatus.FAIL
    detail = bad.detail.casefold()
    assert "dead" in detail or "running" in detail


def test_running_alive_supervisor_is_reported(tmp_path: Path) -> None:
    program = tmp_path / "program"
    runtime = program / "runtime"
    runtime.mkdir(parents=True)
    # Use the current process pid so the liveness probe succeeds.
    import os

    pid = os.getpid()
    (runtime / "master.pid").write_text(str(pid), encoding="utf-8")
    for lane in range(4):
        lane_dir = program / "state" / f"lane-{lane}"
        lane_dir.mkdir(parents=True)
        (lane_dir / f"rpr_lane_{lane}_supervisor_status.json").write_text(
            json.dumps(
                {"status": "running", "pid": pid, "supervisor_pid": pid}
            ),
            encoding="utf-8",
        )
        (lane_dir / f"rpr_lane_{lane}_task_state.json").write_text(
            json.dumps(
                {
                    "status": "available",
                    "active_task_id": "",
                    "eligible_ready_count": 0,
                    "blocked_count": 0,
                }
            ),
            encoding="utf-8",
        )
    healthy = ops.check_supervisor_process_state(
        _REPO_ROOT, state_root=program, lane_count=4
    )
    assert healthy.status is ops.CheckStatus.PASS, healthy.detail
    assert healthy.evidence["master_status"] == "running"
    assert len(healthy.evidence["lanes"]) == 4


def test_clean_four_shard_board_drains() -> None:
    result = ops.check_four_shard_board_drain(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail
    assert result.evidence["max_lanes"] == 4
    assert result.evidence["strict_task_sharding"] is True
    assert len(set(result.evidence["entry_lanes"].values())) == 4
    assert result.evidence["ready_after_non_terminal_complete"] == ["RPR-047"]
    assert result.evidence["ready_after_full_complete"] == []


def test_fixed_point_validator_accepts_discharged_two_to_three(
    roots: PropagationAuthorityRoots,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
        CompletionDisposition,
        PropagationTransaction,
    )

    evidence = _two_to_three_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    plan = admission.plan
    assert plan is not None
    paths = tuple(path for _, path, _ in CALLER_SPECS)
    consumer_ids = tuple(cid for cid, _, _ in CALLER_SPECS)
    obligation_ids = tuple(item.obligation_id for item in plan.obligations)
    candidate = _candidate_evidence(
        plan,
        paths=paths,
        consumer_ids=consumer_ids,
        obligation_ids=obligation_ids,
    )
    transaction = PropagationTransaction(
        roots=roots,
        transaction_id="txn:rpr-047-fp",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id="checkpoint:rpr-047",
        completed_step_ids=tuple(step.step_id for step in plan.steps),
        lease_id="lease:rpr-047",
    )
    outcome = ChangePropagationValidator().validate(
        plan,
        transaction,
        evidence=candidate,
    )
    assert outcome.complete
    assert outcome.completion is not None
    assert isinstance(outcome.completion, PropagationCompletionReceipt)
    assert outcome.completion.disposition is CompletionDisposition.COMPLETE
    assert outcome.completion.completion_id
    assert outcome.completion.unresolved_mandatory_ids == ()
