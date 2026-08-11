"""RPR-043: fixed-point completion gate for change propagation.

Completion requires rebuilt indexes/graphs/tombstones, re-extracted delta,
re-resolved edges, recomputed closure/frontier, once-per-consumer discharge,
second-order discovery to a policy bound, reconstructed proofs, and
dependency-complete policy tools/tests.  Bound exhaustion and weakened checks fail.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    CompletionDisposition,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    FixedPointReceipt,
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
from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_validation import (
    CHANGE_PROPAGATION_VALIDATOR_INTERFACE,
    DEFAULT_FIXED_POINT_BOUND,
    DEFAULT_POLICY_REQUIRED_TOOLS,
    POLICY_TOOL_FAMILIES,
    PRODUCER_ID,
    CandidatePropagationEvidence,
    ChangePropagationValidationError,
    ChangePropagationValidator,
    ClosureRecomputeEvidence,
    ConsumerDischargeEvidence,
    DeltaReextractEvidence,
    FixedPointIterationEvidence,
    ProofReconstructionEvidence,
    PropagationIndexRebuildEvidence,
    PropagationValidationReason,
    ResolutionEvidence,
    SecondOrderImpactEvidence,
    validate_change_propagation,
)
from ipfs_accelerate_py.agent_supervisor.validation.contract_repair_validation import (
    ImpactedTestEvidence,
    IntegrityEvidence,
    PolicyToolEvidence,
    ToolGateResult,
    build_passing_tool_evidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-043v",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-043v",
        index_id="index:rpr-043v",
        model_id="model:rpr-043v",
        config_id="config:rpr-043v",
        translator_id="translator:rpr-043v",
        toolchain_id="toolchain:rpr-043v",
        policy_id="policy:rpr-043v",
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
        plan_id="plan:rpr-043v",
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
        transaction_id="txn:rpr-043v",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id="checkpoint:before",
        completed_step_ids=("step:migrate-one",),
        lease_id="lease:writer",
    )


def _index(tree: str = "tree:candidate", **changes: object) -> PropagationIndexRebuildEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "index_id": "index:rpr-043v",
        "graph_id": "graph:rpr-043v",
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


def _delta(tree: str = "tree:candidate", **changes: object) -> DeltaReextractEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "original_delta_id": "delta:one",
        "reextracted_delta_id": "delta:one",
        "breaking_delta_ids": ("delta:one",),
        "unplanned_breaking_delta_ids": (),
        "extraction_receipt_id": "delta-receipt:ok",
        "matches_plan_delta": True,
    }
    values.update(changes)
    return DeltaReextractEvidence(**values)  # type: ignore[arg-type]


def _resolution(tree: str = "tree:candidate", **changes: object) -> ResolutionEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "resolved_call_ids": ("call:caller-process",),
        "resolved_data_flow_ids": ("data:ctx",),
        "resolved_schema_ids": ("schema:request",),
        "resolved_wiring_ids": ("wire:router",),
        "unresolved_ids": (),
        "resolution_receipt_id": "resolution:ok",
        "complete": True,
    }
    values.update(changes)
    return ResolutionEvidence(**values)  # type: ignore[arg-type]


def _closure(tree: str = "tree:candidate", **changes: object) -> ClosureRecomputeEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "original_closure_id": "closure:one",
        "recomputed_closure_id": "closure:recomputed",
        "consumer_ids": ("consumer:one",),
        "mandatory_consumer_ids": ("consumer:one",),
        "frontier_node_ids": (),
        "required_frontier_ids": (),
        "uncovered_frontier_ids": (),
        "complete": True,
        "receipt_id": "closure:ok",
    }
    values.update(changes)
    return ClosureRecomputeEvidence(**values)  # type: ignore[arg-type]


def _discharge(
    tree: str = "tree:candidate",
    *,
    obligation_id: str = "obligation:consumer:one",
    **changes: object,
) -> ConsumerDischargeEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "original_obligation_ids": (obligation_id,),
        "discharged_obligation_ids": (obligation_id,),
        "unresolved_mandatory_ids": (),
        "omitted_dependent_ids": (),
        "double_discharged_ids": (),
        "receipt_id": "discharge:ok",
    }
    values.update(changes)
    return ConsumerDischargeEvidence(**values)  # type: ignore[arg-type]


def _second_order(
    tree: str = "tree:candidate",
    *,
    residual: bool = False,
    **changes: object,
) -> SecondOrderImpactEvidence:
    if residual:
        values: dict[str, object] = {
            "candidate_tree_id": tree,
            "new_delta_ids": ("delta:second-order",),
            "new_consumer_ids": ("consumer:second",),
            "residual_frontier_ids": (),
            "requires_another_iteration": True,
            "receipt_id": "second:residual",
        }
    else:
        values = {
            "candidate_tree_id": tree,
            "new_delta_ids": (),
            "new_consumer_ids": (),
            "residual_frontier_ids": (),
            "requires_another_iteration": False,
            "receipt_id": "second:fixed",
        }
    values.update(changes)
    return SecondOrderImpactEvidence(**values)  # type: ignore[arg-type]


def _proofs(tree: str = "tree:candidate", **changes: object) -> ProofReconstructionEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "original_proof_refs": ("proof:plan",),
        "reconstructed_proof_refs": ("proof:plan", "proof:reconstructed"),
        "introduced_proof_refs": ("proof:introduced",),
        "failed_proof_refs": (),
        "all_mandatory_reconstructed": True,
        "receipt_id": "proof:ok",
    }
    values.update(changes)
    return ProofReconstructionEvidence(**values)  # type: ignore[arg-type]


def _tools(tree: str = "tree:candidate", **changes: object) -> PolicyToolEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "required_families": DEFAULT_POLICY_REQUIRED_TOOLS,
        "results": build_passing_tool_evidence(tree, "policy:rpr-043v").results,
        "policy_id": "policy:rpr-043v",
    }
    values.update(changes)
    return PolicyToolEvidence(**values)  # type: ignore[arg-type]


def _tests(tree: str = "tree:candidate", **changes: object) -> ImpactedTestEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "focused_test_ids": ("test:focused-caller",),
        "impacted_test_ids": ("test:impacted-process",),
        "required_dependant_ids": ("test:dependant-route",),
        "executed_test_ids": (
            "test:focused-caller",
            "test:impacted-process",
            "test:dependant-route",
        ),
        "passed_test_ids": (
            "test:focused-caller",
            "test:impacted-process",
            "test:dependant-route",
        ),
        "failed_test_ids": (),
        "omitted_dependant_ids": (),
        "dependency_complete": True,
    }
    values.update(changes)
    return ImpactedTestEvidence(**values)  # type: ignore[arg-type]


def _integrity(tree: str = "tree:candidate", **changes: object) -> IntegrityEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": tree,
        "contracts_deleted": (),
        "contracts_weakened": (),
        "tests_deleted": (),
        "tests_weakened": (),
        "checkers_deleted": (),
        "checkers_weakened": (),
        "findings_suppressed": (),
        "original_finding_id": "finding:propagation",
        "original_finding_closed": True,
    }
    values.update(changes)
    return IntegrityEvidence(**values)  # type: ignore[arg-type]


def _iteration(
    *,
    iteration: int = 1,
    residual_second_order: bool = False,
    **overrides: object,
) -> FixedPointIterationEvidence:
    values: dict[str, object] = {
        "iteration": iteration,
        "index_rebuild": _index(),
        "delta_reextract": _delta(),
        "resolution": _resolution(),
        "closure_recompute": _closure(),
        "consumer_discharge": _discharge(),
        "second_order": _second_order(residual=residual_second_order),
        "proof_reconstruction": _proofs(),
        "policy_tools": _tools(),
        "impacted_tests": _tests(),
        "integrity": _integrity(),
    }
    values.update(overrides)
    return FixedPointIterationEvidence(**values)  # type: ignore[arg-type]


def _evidence(
    *iterations: FixedPointIterationEvidence,
) -> CandidatePropagationEvidence:
    if not iterations:
        iterations = (_iteration(),)
    return CandidatePropagationEvidence(
        candidate_tree_id="tree:candidate",
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Interface / happy path
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert CHANGE_PROPAGATION_VALIDATOR_INTERFACE == "ChangePropagationValidator@1"
    assert ChangePropagationValidator.INTERFACE == CHANGE_PROPAGATION_VALIDATOR_INTERFACE
    assert PRODUCER_ID == "change-propagation-validation@1"
    assert DEFAULT_FIXED_POINT_BOUND >= 1
    assert set(DEFAULT_POLICY_REQUIRED_TOOLS).issubset(POLICY_TOOL_FAMILIES)


def test_complete_fixed_point_returns_canonical_completion_receipt(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(_iteration())

    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)

    assert outcome.complete
    assert outcome.completion is not None
    assert isinstance(outcome.completion, PropagationCompletionReceipt)
    assert outcome.completion.disposition is CompletionDisposition.COMPLETE
    assert outcome.completion.SCHEMA.endswith("completion-receipt@1")
    assert outcome.completion.plan_id == plan.plan_id
    assert outcome.completion.transaction_id == txn.transaction_id
    assert outcome.completion.fixed_point_receipt is not None
    assert isinstance(outcome.completion.fixed_point_receipt, FixedPointReceipt)
    assert outcome.completion.fixed_point_receipt.is_fixed_point
    assert outcome.completion.unresolved_mandatory_ids == ()
    assert outcome.completion.omitted_dependent_ids == ()
    assert outcome.completion.uncovered_frontier_ids == ()
    assert outcome.completion.unplanned_breaking_delta_ids == ()
    assert "obligation:consumer:one" in outcome.completion.discharged_obligation_ids
    assert outcome.completion.proof_refs
    assert outcome.completion.validation_refs
    restored = PropagationCompletionReceipt.from_dict(outcome.completion.to_record())
    assert restored.disposition is CompletionDisposition.COMPLETE
    assert restored.fixed_point_receipt is not None


def test_module_entry_point_matches_class(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence()
    a = validate_change_propagation(plan, txn, evidence=evidence)
    b = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert a.complete and b.complete
    assert a.completion is not None and b.completion is not None
    assert a.completion.plan_id == b.completion.plan_id


def test_require_complete_raises_on_failure(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:rolled",
        plan_id=plan.plan_id,
        state=TransactionState.ROLLED_BACK,
        checkpoint_id="checkpoint:before",
        diagnostic_refs=("diagnostic:failed",),
        lease_id="lease:writer",
    )
    with pytest.raises(ChangePropagationValidationError, match="rejected"):
        ChangePropagationValidator().require_complete(
            plan, txn, evidence=_evidence()
        )


def test_uncommitted_transaction_cannot_complete(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:exec",
        plan_id=plan.plan_id,
        state=TransactionState.EXECUTING,
        checkpoint_id="checkpoint:before",
        lease_id="lease:writer",
        active_scc_group_id="group:one",
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=_evidence())
    assert not outcome.complete
    assert (
        PropagationValidationReason.TRANSACTION_NOT_COMMITTED.value
        in outcome.report.reason_codes
    )


# ---------------------------------------------------------------------------
# Stage failures
# ---------------------------------------------------------------------------


def test_index_rebuild_incomplete_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            index_rebuild=_index(clean_rebuild_equivalent=False),
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.INDEX_REBUILD_INCOMPLETE.value
        in outcome.report.reason_codes
    )


def test_unplanned_breaking_delta_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            delta_reextract=_delta(
                matches_plan_delta=False,
                unplanned_breaking_delta_ids=("delta:surprise",),
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.UNPLANNED_BREAKING_DELTA.value
        in outcome.report.reason_codes
    )
    assert outcome.completion is not None
    assert "delta:surprise" in outcome.completion.unplanned_breaking_delta_ids


def test_unresolved_mandatory_consumer_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            consumer_discharge=_discharge(
                discharged_obligation_ids=(),
                unresolved_mandatory_ids=("obligation:consumer:one",),
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.UNRESOLVED_MANDATORY.value
        in outcome.report.reason_codes
        or PropagationValidationReason.CONSUMER_NOT_DISCHARGED.value
        in outcome.report.reason_codes
    )


def test_omitted_dependent_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            consumer_discharge=_discharge(
                omitted_dependent_ids=("consumer:dependent",),
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert PropagationValidationReason.OMITTED_DEPENDENT.value in outcome.report.reason_codes


def test_uncovered_frontier_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            closure_recompute=_closure(
                complete=False,
                uncovered_frontier_ids=("frontier:plugin",),
                required_frontier_ids=("frontier:plugin",),
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.UNCOVERED_FRONTIER.value
        in outcome.report.reason_codes
        or PropagationValidationReason.CLOSURE_INCOMPLETE.value
        in outcome.report.reason_codes
    )


def test_skipped_required_tool_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    results = tuple(
        ToolGateResult(
            tool_id=f"tool:{family}",
            family=family,
            required=True,
            executed=False,
            passed=False,
            skipped=True,
            receipt_id=f"tool-receipt:{family}",
        )
        for family in DEFAULT_POLICY_REQUIRED_TOOLS
    )
    evidence = _evidence(
        _iteration(
            policy_tools=_tools(results=results),
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.SKIPPED_REQUIRED_TOOL.value
        in outcome.report.reason_codes
    )


def test_weakened_test_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            integrity=_integrity(tests_weakened=("test:focused-caller",)),
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert PropagationValidationReason.TEST_WEAKENED.value in outcome.report.reason_codes


def test_deleted_checker_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            integrity=_integrity(checkers_deleted=("checker:type",)),
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert PropagationValidationReason.CHECKER_DELETED.value in outcome.report.reason_codes


def test_impacted_test_omission_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            impacted_tests=_tests(
                executed_test_ids=("test:focused-caller",),
                passed_test_ids=("test:focused-caller",),
                omitted_dependant_ids=("test:dependant-route",),
                dependency_complete=False,
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.IMPACTED_TEST_OMITTED.value
        in outcome.report.reason_codes
    )


# ---------------------------------------------------------------------------
# Fixed-point iteration / bound exhaustion
# ---------------------------------------------------------------------------


def test_second_order_iteration_reaches_fixed_point(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    # Iteration 1 discovers residual; iteration 2 is clean.
    evidence = _evidence(
        _iteration(iteration=1, residual_second_order=True),
        _iteration(iteration=2, residual_second_order=False),
    )
    outcome = ChangePropagationValidator().validate(
        plan, txn, evidence=evidence, fixed_point_bound=4
    )
    assert outcome.complete
    assert outcome.completion is not None
    assert outcome.completion.fixed_point_receipt is not None
    assert outcome.completion.fixed_point_receipt.iteration_count >= 2
    assert outcome.report.iteration_count == 2


def test_bound_exhaustion_is_incomplete_not_success(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    # Only residual iterations within a tight bound.
    evidence = _evidence(
        _iteration(iteration=1, residual_second_order=True),
        _iteration(iteration=2, residual_second_order=True),
    )
    outcome = ChangePropagationValidator().validate(
        plan, txn, evidence=evidence, fixed_point_bound=2
    )
    assert not outcome.complete
    assert (
        PropagationValidationReason.BOUND_EXHAUSTED.value in outcome.report.reason_codes
        or PropagationValidationReason.FIXED_POINT_NOT_REACHED.value
        in outcome.report.reason_codes
        or PropagationValidationReason.SECOND_ORDER_RESIDUAL.value
        in outcome.report.reason_codes
    )
    assert outcome.completion is not None
    assert outcome.completion.disposition is not CompletionDisposition.COMPLETE
    # Bound exhaustion must not forge a residual-free fixed-point receipt.
    if outcome.completion.fixed_point_receipt is not None:
        assert not outcome.completion.fixed_point_receipt.is_fixed_point


def test_stale_candidate_tree_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = CandidatePropagationEvidence(
        candidate_tree_id="tree:stale",
        iterations=(_iteration(),),
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.STALE_CANDIDATE_TREE.value
        in outcome.report.reason_codes
    )


def test_double_discharge_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            consumer_discharge=_discharge(
                double_discharged_ids=("obligation:consumer:one",),
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.CONSUMER_DOUBLE_DISCHARGE.value
        in outcome.report.reason_codes
    )


def test_proof_reconstruction_failure(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            proof_reconstruction=_proofs(
                reconstructed_proof_refs=(),
                failed_proof_refs=("proof:plan",),
                all_mandatory_reconstructed=False,
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.PROOF_RECONSTRUCTION_FAILED.value
        in outcome.report.reason_codes
    )


def test_resolution_incomplete_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    evidence = _evidence(
        _iteration(
            resolution=_resolution(
                complete=False,
                unresolved_ids=("call:missing",),
            )
        )
    )
    outcome = ChangePropagationValidator().validate(plan, txn, evidence=evidence)
    assert not outcome.complete
    assert (
        PropagationValidationReason.RESOLUTION_INCOMPLETE.value
        in outcome.report.reason_codes
    )


def test_complete_report_denies_provider_authority(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    txn = _committed_txn(roots, plan)
    outcome = ChangePropagationValidator().validate(
        plan, txn, evidence=_evidence()
    )
    assert outcome.complete
    payload = outcome.report.to_dict()
    assert payload["provider_success_is_not_completion"] is True
    assert payload["interface"] == CHANGE_PROPAGATION_VALIDATOR_INTERFACE
