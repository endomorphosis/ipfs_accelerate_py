"""LPR-018: joint program+logic post-edit fixed-point validation.

Covers:
* LogicRepairFixedPointValidator rebuild/replan/reprove/consumer stages
* LogicFixedPointEvidenceAttachment extends existing completion
* PropagationFinalizeReceipt only after residual-free success
* CompensatingRollbackReceipt after provisional-commit failure
* Partial SCC/packet completion never closes the task
* Pipeline and daemon completion wrappers
* Contract-repair route through atomic propagation fixed-point
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    CompletionDisposition,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    PropagationPlanStep,
    PropagationTransaction,
    TransactionState,
    obligation_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_pipeline import (
    ChangePropagationPipeline,
    ChangePropagationPipelinePolicy,
    ChangePropagationPipelineRequest,
    PipelineDisposition,
    daemon_require_completion,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    FixedPointAttachmentDisposition,
    LogicFixedPointEvidenceAttachment,
    ProgramLogicAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_transaction import (
    ChangePropagationTransaction,
    PropagationCheckpoint,
    TransactionFailureReason,
    create_propagation_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
)
from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_validation import (
    DEFAULT_POLICY_REQUIRED_TOOLS,
    CandidatePropagationEvidence,
    ChangePropagationValidator,
    ClosureRecomputeEvidence,
    ConsumerDischargeEvidence,
    DeltaReextractEvidence,
    FixedPointIterationEvidence,
    ProofReconstructionEvidence,
    PropagationIndexRebuildEvidence,
    ResolutionEvidence,
    SecondOrderImpactEvidence,
    validate_change_propagation_with_logic_fixed_point,
)
from ipfs_accelerate_py.agent_supervisor.validation.contract_repair_validation import (
    build_passing_tool_evidence,
    validate_contract_repair_with_logic_fixed_point,
)
from ipfs_accelerate_py.agent_supervisor.validation.contract_repair_validation import (
    ImpactedTestEvidence,
    IntegrityEvidence,
    PolicyToolEvidence,
)
from ipfs_accelerate_py.agent_supervisor.validation.logic_repair_fixed_point import (
    CandidateLogicRepairEvidence,
    CompensatingRollbackReceipt,
    LogicConsumerRevalidationEvidence,
    LogicFixedPointReason,
    LogicRebuildEvidence,
    LogicRepairFixedPointError,
    LogicRepairFixedPointValidator,
    LogicRepairIterationReceipt,
    LogicReplanEvidence,
    LogicReproveEvidence,
    PropagationFinalizeReceipt,
    daemon_require_logic_fixed_point,
    validate_logic_repair_fixed_point,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:lpr-018",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:lpr-018",
        index_id="index:lpr-018",
        model_id="model:lpr-018",
        config_id="config:lpr-018",
        translator_id="translator:lpr-018",
        toolchain_id="toolchain:lpr-018",
        policy_id="policy:lpr-018",
    )


@pytest.fixture
def logic_roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:lpr-018",
        objective_id="objective:lpr-018",
        trace_id="trace:lpr-018",
        change_id="change:lpr-018",
        consumer_id="consumer:one",
        forest_id="forest:candidate",
        tree_id="tree:candidate",
        overlay_id="overlay:candidate",
        graph_id="graph:lpr-018",
        index_id="index:lpr-018",
        corpus_id="corpus:lpr-018",
        model_id="model:lpr-018",
        translator_id="translator:lpr-018",
        toolchain_id="toolchain:lpr-018",
        policy_id="policy:lpr-018",
        environment_id="environment:lpr-018",
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


def _admitted_plan(roots: PropagationAuthorityRoots) -> AtomicPropagationPlan:
    obligation = _obligation(roots)
    step = PropagationPlanStep(
        step_id="step:migrate-one",
        kind=PlanStepKind.ANALYTICAL,
        obligation_ids=(obligation.obligation_id,),
        transform_id="transform:add-arg",
        write_paths=("pkg/caller.py",),
        read_paths=("pkg/caller.py",),
    )
    return AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:lpr-018",
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure_id="closure:one",
        disposition=PlanDisposition.ADMITTED,
        obligations=(obligation,),
        obligation_set_id=obligation_set_identity((obligation,)),
        steps=(step,),
        permitted_read_paths=("pkg/caller.py",),
        permitted_write_paths=("pkg/caller.py",),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:plan",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )


def _committed_txn(
    roots: PropagationAuthorityRoots,
    plan: AtomicPropagationPlan,
) -> PropagationTransaction:
    return PropagationTransaction(
        roots=roots,
        transaction_id="txn:lpr-018",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id="checkpoint:before",
        completed_step_ids=("step:migrate-one",),
        lease_id="lease:writer",
    )


def _checkpoint(plan: AtomicPropagationPlan) -> PropagationCheckpoint:
    return create_propagation_checkpoint(
        plan,
        path_before_hashes=(
            PathBeforeHash(path="pkg/caller.py", before_hash="sha256:before"),
        ),
        tree_snapshot_ref=plan.roots.candidate_tree_id,
    )


def _program_iteration(
    *,
    iteration: int = 1,
    residual_second_order: bool = False,
) -> FixedPointIterationEvidence:
    tree = "tree:candidate"
    second = (
        SecondOrderImpactEvidence(
            candidate_tree_id=tree,
            new_delta_ids=("delta:second-order",),
            new_consumer_ids=("consumer:second",),
            residual_frontier_ids=(),
            requires_another_iteration=True,
            receipt_id="second:residual",
        )
        if residual_second_order
        else SecondOrderImpactEvidence(
            candidate_tree_id=tree,
            new_delta_ids=(),
            new_consumer_ids=(),
            residual_frontier_ids=(),
            requires_another_iteration=False,
            receipt_id="second:fixed",
        )
    )
    return FixedPointIterationEvidence(
        iteration=iteration,
        index_rebuild=PropagationIndexRebuildEvidence(
            candidate_tree_id=tree,
            index_id="index:lpr-018",
            graph_id="graph:lpr-018",
            rebuilt_source_paths=("pkg/caller.py",),
            rebuilt_ast_paths=("pkg/caller.py",),
            rebuilt_vector_row_ids=("vector:caller",),
            rebuilt_graph_node_ids=("node:symbol:consumer:one",),
            tombstone_ids=(),
            affected_paths=("pkg/caller.py",),
            clean_rebuild_equivalent=True,
        ),
        delta_reextract=DeltaReextractEvidence(
            candidate_tree_id=tree,
            original_delta_id="delta:one",
            reextracted_delta_id="delta:one",
            breaking_delta_ids=("delta:one",),
            unplanned_breaking_delta_ids=(),
            extraction_receipt_id="delta-receipt:ok",
            matches_plan_delta=True,
        ),
        resolution=ResolutionEvidence(
            candidate_tree_id=tree,
            resolved_call_ids=("call:caller-process",),
            resolved_data_flow_ids=("data:ctx",),
            resolved_schema_ids=("schema:request",),
            resolved_wiring_ids=("wire:router",),
            unresolved_ids=(),
            resolution_receipt_id="resolution:ok",
            complete=True,
        ),
        closure_recompute=ClosureRecomputeEvidence(
            candidate_tree_id=tree,
            original_closure_id="closure:one",
            recomputed_closure_id="closure:recomputed",
            consumer_ids=("consumer:one",),
            mandatory_consumer_ids=("consumer:one",),
            frontier_node_ids=(),
            required_frontier_ids=(),
            uncovered_frontier_ids=(),
            complete=True,
            receipt_id="closure:ok",
        ),
        consumer_discharge=ConsumerDischargeEvidence(
            candidate_tree_id=tree,
            original_obligation_ids=("obligation:consumer:one",),
            discharged_obligation_ids=("obligation:consumer:one",),
            unresolved_mandatory_ids=(),
            omitted_dependent_ids=(),
            double_discharged_ids=(),
            receipt_id="discharge:ok",
        ),
        second_order=second,
        proof_reconstruction=ProofReconstructionEvidence(
            candidate_tree_id=tree,
            original_proof_refs=("proof:plan",),
            reconstructed_proof_refs=("proof:plan", "proof:reconstructed"),
            introduced_proof_refs=("proof:introduced",),
            failed_proof_refs=(),
            all_mandatory_reconstructed=True,
            receipt_id="proof:ok",
        ),
        policy_tools=PolicyToolEvidence(
            candidate_tree_id=tree,
            required_families=DEFAULT_POLICY_REQUIRED_TOOLS,
            results=build_passing_tool_evidence(tree, "policy:lpr-018").results,
            policy_id="policy:lpr-018",
        ),
        impacted_tests=ImpactedTestEvidence(
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
        ),
        integrity=IntegrityEvidence(
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
        ),
    )


def _program_evidence() -> CandidatePropagationEvidence:
    return CandidatePropagationEvidence(
        candidate_tree_id="tree:candidate",
        iterations=(_program_iteration(),),
    )


def _logic_iteration(
    *,
    iteration: int = 1,
    new_required_gaps: tuple[str, ...] = (),
    unresolved_mandatory: tuple[str, ...] = (),
    open_frontier: tuple[str, ...] = (),
    stale_predictions: tuple[str, ...] = (),
    unplanned_deltas: tuple[str, ...] = (),
    requires_another: bool = False,
    plan_current: bool = True,
    all_promoted_current: bool = True,
    newly_resolved: tuple[str, ...] = (),
    discharged_new: tuple[str, ...] = (),
) -> LogicRepairIterationReceipt:
    tree = "tree:candidate"
    return LogicRepairIterationReceipt(
        iteration=iteration,
        rebuild=LogicRebuildEvidence(
            candidate_tree_id=tree,
            repository_index_id="repo-index:rebuilt",
            ast_index_id="ast-index:rebuilt",
            vector_row_ids=("vector:caller",),
            kg_node_ids=("kg:caller",),
            call_graph_id="call-graph:rebuilt",
            dependency_graph_id="dep-graph:rebuilt",
            schema_graph_id="schema-graph:rebuilt",
            value_graph_id="value-graph:rebuilt",
            tombstone_ids=(),
            clean_rebuild_equivalent=True,
        ),
        replan=LogicReplanEvidence(
            candidate_tree_id=tree,
            corpus_root_id="corpus:lpr-018",
            goal_root_ids=("goal:caller-migrate",),
            gap_ids=(),
            required_gap_ids=(),
            new_required_gap_ids=new_required_gaps,
            tactician_plan_id="tactician:plan-1",
            plan_current=plan_current and not new_required_gaps,
        ),
        reprove=LogicReproveEvidence(
            candidate_tree_id=tree,
            hammer_receipt_ids=("hammer:receipt-1",),
            native_goal_binding_ids=("native:goal-1",),
            countermodel_receipt_ids=("countermodel:none",),
            prediction_receipt_ids=("prediction:admit-1",),
            stale_prediction_ids=stale_predictions,
            failed_reconstruction_ids=() if all_promoted_current else ("recon:fail",),
            all_promoted_clauses_current=all_promoted_current and not stale_predictions,
        ),
        consumer_revalidation=LogicConsumerRevalidationEvidence(
            candidate_tree_id=tree,
            original_consumer_ids=("consumer:one",),
            discharged_original_ids=("consumer:one",)
            if not unresolved_mandatory
            else (),
            newly_resolved_consumer_ids=newly_resolved,
            discharged_new_consumer_ids=discharged_new,
            unresolved_mandatory_ids=unresolved_mandatory,
            open_required_frontier_ids=open_frontier,
            second_order_consumer_ids=(),
            value_choice_ids=("value:default-arg",),
            behavior_choice_ids=("behavior:preserve",),
            placement_choice_ids=("placement:caller-site",),
            failed_value_behavior_placement_ids=(),
            policy_tool_receipt_ids=("tool:typecheck",),
            failed_policy_tool_ids=(),
        ),
        unplanned_breaking_delta_ids=unplanned_deltas,
        residual_logic_gap_ids=new_required_gaps,
        unsupported_logic_gap_ids=(),
        requires_another_iteration=requires_another,
    )


def _logic_evidence(
    logic_roots: ProgramLogicAuthorityRoots,
    *iterations: LogicRepairIterationReceipt,
) -> CandidateLogicRepairEvidence:
    if not iterations:
        iterations = (_logic_iteration(),)
    return CandidateLogicRepairEvidence(
        candidate_tree_id="tree:candidate",
        logic_roots=logic_roots,
        iterations=iterations,
        program_evidence=_program_evidence(),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_joint_fixed_point_success_attaches_logic_evidence(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    assert outcome.complete
    assert outcome.completion is not None
    assert outcome.completion.disposition is CompletionDisposition.COMPLETE
    assert isinstance(outcome.logic_attachment, LogicFixedPointEvidenceAttachment)
    assert (
        outcome.logic_attachment.disposition
        is FixedPointAttachmentDisposition.ATTACHED
    )
    assert outcome.logic_attachment.replaces_completion is False
    assert outcome.logic_attachment.completion_receipt_id == (
        outcome.completion.completion_id
    )
    assert isinstance(outcome.finalize, PropagationFinalizeReceipt)
    assert outcome.compensating_rollback is None
    assert not outcome.rolled_back
    assert outcome.logic_attachment.finalize_receipt_id == outcome.finalize.finalize_id
    assert "consumer:one" in outcome.logic_attachment.original_consumer_coverage_ids
    assert outcome.logic_attachment.goal_root_ids == ("goal:caller-migrate",)
    assert outcome.logic_attachment.hammer_receipt_ids == ("hammer:receipt-1",)


def test_program_only_path_without_logic_evidence(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=None,
    )
    assert not outcome.complete  # no logic attachment without evidence
    assert outcome.report.program_complete
    assert outcome.completion is not None
    assert outcome.completion.disposition is CompletionDisposition.COMPLETE
    assert outcome.logic_attachment is None
    assert outcome.finalize is None


def test_require_logic_evidence_fails_closed(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    checkpoint = _checkpoint(plan)
    # Align checkpoint id with transaction for rollback path.
    txn = PropagationTransaction(
        roots=roots,
        transaction_id=txn.transaction_id,
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id=checkpoint.checkpoint_id,
        completed_step_ids=txn.completed_step_ids,
        lease_id=txn.lease_id,
    )
    outcome = LogicRepairFixedPointValidator(
        require_logic_evidence=True
    ).validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=None,
        checkpoint=checkpoint,
        restore_adapter=lambda _cp: True,
    )
    assert not outcome.complete
    assert LogicFixedPointReason.MISSING_LOGIC_EVIDENCE.value in (
        outcome.report.reason_codes
    )
    assert outcome.rolled_back
    assert isinstance(outcome.compensating_rollback, CompensatingRollbackReceipt)


# ---------------------------------------------------------------------------
# Failure / residual cases
# ---------------------------------------------------------------------------


def test_new_required_logic_gap_triggers_compensating_rollback(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    checkpoint = _checkpoint(plan)
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:lpr-018-gap",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id=checkpoint.checkpoint_id,
        completed_step_ids=("step:migrate-one",),
        lease_id="lease:writer",
    )
    restored: list[str] = []

    def _restore(cp: PropagationCheckpoint) -> bool:
        restored.append(cp.checkpoint_id)
        return True

    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(
            logic_roots,
            _logic_iteration(new_required_gaps=("gap:required-new",)),
        ),
        checkpoint=checkpoint,
        restore_adapter=_restore,
    )
    assert not outcome.complete
    assert LogicFixedPointReason.NEW_REQUIRED_LOGIC_GAP.value in (
        outcome.report.reason_codes
    )
    assert outcome.rolled_back
    assert restored == [checkpoint.checkpoint_id]
    assert outcome.compensating_rollback is not None
    assert outcome.compensating_rollback.restored
    assert outcome.finalize is None


def test_unresolved_mandatory_consumer_fails(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(
            logic_roots,
            _logic_iteration(unresolved_mandatory=("consumer:one",)),
        ),
    )
    assert not outcome.complete
    assert LogicFixedPointReason.UNRESOLVED_MANDATORY_CONSUMER.value in (
        outcome.report.reason_codes
    )


def test_open_required_frontier_fails(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(
            logic_roots,
            _logic_iteration(open_frontier=("frontier:unknown-call",)),
        ),
    )
    assert not outcome.complete
    assert LogicFixedPointReason.UNCOVERED_FRONTIER.value in (
        outcome.report.reason_codes
    )


def test_stale_prediction_fails(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(
            logic_roots,
            _logic_iteration(stale_predictions=("prediction:stale",)),
        ),
    )
    assert not outcome.complete
    assert LogicFixedPointReason.PREDICTION_STALE.value in outcome.report.reason_codes


def test_unplanned_breaking_delta_fails(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(
            logic_roots,
            _logic_iteration(unplanned_deltas=("delta:unplanned",)),
        ),
    )
    assert not outcome.complete
    assert LogicFixedPointReason.UNPLANNED_BREAKING_DELTA.value in (
        outcome.report.reason_codes
    )


def test_newly_resolved_consumer_must_be_discharged(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(
            logic_roots,
            _logic_iteration(
                newly_resolved=("consumer:new",),
                discharged_new=(),
            ),
        ),
    )
    assert not outcome.complete
    assert LogicFixedPointReason.NEW_RESOLVED_CONSUMER_OPEN.value in (
        outcome.report.reason_codes
    )


def test_non_committed_transaction_rejected(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:pending",
        plan_id=plan.plan_id,
        state=TransactionState.PENDING,
        checkpoint_id="checkpoint:before",
        completed_step_ids=(),
        lease_id="lease:writer",
    )
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    assert not outcome.complete
    assert LogicFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value in (
        outcome.report.reason_codes
    )


def test_partial_packet_completion_forbidden(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    checkpoint = _checkpoint(plan)
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:partial",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id=checkpoint.checkpoint_id,
        completed_step_ids=(),  # incomplete — partial SCC/packet
        lease_id="lease:writer",
    )
    engine = ChangePropagationTransaction(restore_adapter=lambda _cp: True)
    with pytest.raises(Exception):
        engine.finalize_provisional(
            plan=plan,
            transaction=txn,
            checkpoint=checkpoint,
            completion_id="completion:x",
            fixed_point_receipt_id="fp:x",
        )


# ---------------------------------------------------------------------------
# Transaction finalize / compensating rollback
# ---------------------------------------------------------------------------


def test_finalize_provisional_requires_full_step_set(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    checkpoint = _checkpoint(plan)
    full = PropagationTransaction(
        roots=roots,
        transaction_id="txn:full",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id=checkpoint.checkpoint_id,
        completed_step_ids=("step:migrate-one",),
        lease_id="lease:writer",
    )
    engine = ChangePropagationTransaction(restore_adapter=lambda _cp: True)
    receipt = engine.finalize_provisional(
        plan=plan,
        transaction=full,
        checkpoint=checkpoint,
        completion_id="completion:ok",
        fixed_point_receipt_id="fp:ok",
        iteration_count=1,
    )
    assert receipt["partial_merge_allowed"] is False
    assert receipt["completion_id"] == "completion:ok"
    assert receipt["finalize_id"]


def test_compensating_rollback_restores_checkpoint(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    checkpoint = _checkpoint(plan)
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:rollback",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id=checkpoint.checkpoint_id,
        completed_step_ids=("step:migrate-one",),
        lease_id="lease:writer",
    )
    restored: list[str] = []
    engine = ChangePropagationTransaction(
        restore_adapter=lambda cp: restored.append(cp.checkpoint_id) or True
    )
    rb = engine.compensating_rollback(
        plan=plan,
        transaction=txn,
        checkpoint=checkpoint,
        reason_codes=(LogicFixedPointReason.BOUND_EXHAUSTED.value,),
    )
    assert rb.restored
    assert restored == [checkpoint.checkpoint_id]
    assert LogicFixedPointReason.BOUND_EXHAUSTED.value in rb.reason_codes


# ---------------------------------------------------------------------------
# Module entry points / wrappers
# ---------------------------------------------------------------------------


def test_validate_logic_repair_fixed_point_entry(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = validate_logic_repair_fixed_point(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    assert outcome.complete


def test_validate_change_propagation_with_logic_fixed_point(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = validate_change_propagation_with_logic_fixed_point(
        plan,
        txn,
        evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    assert outcome.complete
    assert outcome.logic_attachment is not None


def test_contract_repair_via_propagation_fixed_point(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = validate_contract_repair_with_logic_fixed_point(
        plan=plan,
        transaction=txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
        contract_repair_completion_id="contract-completion:1",
    )
    assert outcome.complete
    assert outcome.logic_attachment is not None
    assert (
        outcome.logic_attachment.completion_receipt_id == "contract-completion:1"
    )


def test_daemon_require_logic_fixed_point(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    completion = daemon_require_logic_fixed_point(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    assert isinstance(completion, PropagationCompletionReceipt)
    assert completion.disposition is CompletionDisposition.COMPLETE


def test_daemon_require_completion_with_logic_evidence(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    completion = daemon_require_completion(
        plan,
        txn,
        evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    assert completion.disposition is CompletionDisposition.COMPLETE


def test_require_complete_raises_on_failure(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    with pytest.raises(LogicRepairFixedPointError):
        LogicRepairFixedPointValidator().require_complete(
            plan,
            txn,
            program_evidence=_program_evidence(),
            logic_evidence=_logic_evidence(
                logic_roots,
                _logic_iteration(open_frontier=("frontier:x",)),
            ),
        )


# ---------------------------------------------------------------------------
# Multi-iteration residual then success
# ---------------------------------------------------------------------------


def test_second_iteration_reaches_fixed_point(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    # Iteration 1 discovers a newly resolved consumer; iteration 2 discharges it.
    iter1 = _logic_iteration(
        iteration=1,
        newly_resolved=("consumer:new",),
        discharged_new=("consumer:new",),
        requires_another=True,
    )
    # requires_another with residual: the receipt allows second-order consumers.
    # Force residual via residual_logic_gap that clears on iter 2.
    iter1 = LogicRepairIterationReceipt(
        iteration=1,
        rebuild=iter1.rebuild,
        replan=iter1.replan,
        reprove=iter1.reprove,
        consumer_revalidation=iter1.consumer_revalidation,
        unplanned_breaking_delta_ids=(),
        residual_logic_gap_ids=("gap:transient",),
        unsupported_logic_gap_ids=(),
        requires_another_iteration=True,
    )
    iter2 = _logic_iteration(iteration=2)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots, iter1, iter2),
    )
    assert outcome.complete
    assert outcome.report.iteration_count == 2
    assert outcome.logic_attachment is not None
    assert outcome.logic_attachment.iteration_count == 2


# ---------------------------------------------------------------------------
# Attachment contract invariants
# ---------------------------------------------------------------------------


def test_attachment_extends_not_replaces(
    roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = LogicRepairFixedPointValidator().validate(
        plan,
        txn,
        program_evidence=_program_evidence(),
        logic_evidence=_logic_evidence(logic_roots),
    )
    att = outcome.logic_attachment
    assert att is not None
    assert att.replaces_completion is False
    # Round-trip
    restored = LogicFixedPointEvidenceAttachment.from_dict(att.to_dict())
    assert restored.attachment_id == att.attachment_id
    assert restored.disposition is FixedPointAttachmentDisposition.ATTACHED


def test_program_validator_still_used_for_legacy_path(
    roots: PropagationAuthorityRoots,
) -> None:
    """Legacy ChangePropagationValidator path remains residual-free complete."""
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = ChangePropagationValidator().validate(
        plan, txn, evidence=_program_evidence()
    )
    assert outcome.complete
    assert outcome.completion is not None


def test_pipeline_policy_defaults_disable_logic_fp() -> None:
    policy = ChangePropagationPipelinePolicy()
    assert policy.enable_logic_fixed_point is False
    assert policy.require_logic_fixed_point_evidence is False
    assert "enable_logic_fixed_point" in policy.to_dict()
