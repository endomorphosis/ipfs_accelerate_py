"""RPR-043: checkpointed SCC transactions for change propagation.

Execution creates a content-addressed checkpoint, verifies before-hashes and
leases, runs each SCC as one atomic group, and rolls back on failure, drift,
timeout, or scope escape.  Partial merge/completion is forbidden.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    PropagationPlanStep,
    PropagationSCCGroup,
    PropagationTransaction,
    TransactionState,
    obligation_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_transaction import (
    CHANGE_PROPAGATION_TRANSACTION_INTERFACE,
    PRODUCER_ID,
    ChangePropagationTransaction,
    ChangePropagationTransactionError,
    GroupExecutionDisposition,
    PropagationCheckpoint,
    PropagationRollbackReceipt,
    StepApplyRequest,
    StepApplyResult,
    StepExecutionDisposition,
    TransactionFailureReason,
    TransactionLease,
    create_propagation_checkpoint,
    execute_change_propagation_transaction,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-043",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-043",
        index_id="index:rpr-043",
        model_id="model:rpr-043",
        config_id="config:rpr-043",
        translator_id="translator:rpr-043",
        toolchain_id="toolchain:rpr-043",
        policy_id="policy:rpr-043",
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
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=_node(path, f"symbol:{consumer_id}"),
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        behavior_contract_ids=(),
        invalidation_refs=("tree:candidate",),
    )


def _admitted_plan(
    roots: PropagationAuthorityRoots,
    *,
    steps: tuple[PropagationPlanStep, ...] | None = None,
    scc_groups: tuple[PropagationSCCGroup, ...] = (),
    write_paths: tuple[str, ...] = ("pkg/caller.py",),
    consumers: tuple[str, ...] = ("consumer:one",),
) -> AtomicPropagationPlan:
    obligations = tuple(
        _obligation(
            roots,
            consumer_id=cid,
            path=write_paths[i] if i < len(write_paths) else write_paths[0],
        )
        for i, cid in enumerate(consumers)
    )
    if steps is None:
        steps = (
            PropagationPlanStep(
                step_id="step:migrate-one",
                kind=PlanStepKind.ANALYTICAL,
                obligation_ids=(obligations[0].obligation_id,),
                transform_id="transform:add-arg",
                write_paths=(write_paths[0],),
                read_paths=(write_paths[0],),
            ),
        )
    return AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:rpr-043",
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure_id="closure:one",
        disposition=PlanDisposition.ADMITTED,
        obligations=obligations,
        obligation_set_id=obligation_set_identity(obligations),
        steps=steps,
        scc_groups=scc_groups,
        permitted_read_paths=write_paths,
        permitted_write_paths=write_paths,
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:plan",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
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
    return tuple(
        PathBeforeHash(path=path, before_hash=f"sha256:{path.replace('/', '-')}")
        for path in paths
    )


def _passing_applicator(request: StepApplyRequest) -> StepApplyResult:
    return StepApplyResult(
        disposition=StepExecutionDisposition.PASSED,
        written_paths=request.step.write_paths,
        observed_before_hashes=tuple(
            PathBeforeHash(path=p, before_hash=f"sha256:{p.replace('/', '-')}")
            for p in request.step.write_paths
        ),
    )


# ---------------------------------------------------------------------------
# Interface / checkpoint
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert CHANGE_PROPAGATION_TRANSACTION_INTERFACE == "ChangePropagationTransaction@1"
    assert ChangePropagationTransaction.INTERFACE == CHANGE_PROPAGATION_TRANSACTION_INTERFACE
    assert PRODUCER_ID == "change-propagation-transaction@1"


def test_create_content_addressed_checkpoint_before_mutation(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    hashes = _hashes("pkg/caller.py")
    checkpoint = create_propagation_checkpoint(plan, path_before_hashes=hashes)

    assert isinstance(checkpoint, PropagationCheckpoint)
    assert checkpoint.plan_id == plan.plan_id
    assert checkpoint.plan_content_id == plan.content_id
    assert checkpoint.roots == roots
    assert checkpoint.strategy_ref == plan.checkpoint_strategy_ref
    assert checkpoint.path_before_hashes[0].path == "pkg/caller.py"
    assert checkpoint.path_before_hashes[0].before_hash.startswith("sha256:")
    # Deterministic content id.
    again = create_propagation_checkpoint(plan, path_before_hashes=hashes)
    assert again.checkpoint_id == checkpoint.checkpoint_id
    restored = PropagationCheckpoint.from_dict(checkpoint.to_record())
    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert restored.path_before_hashes == checkpoint.path_before_hashes


def test_checkpoint_rejects_non_admitted_plan(roots: PropagationAuthorityRoots) -> None:
    obligations = (_obligation(roots, consumer_id="consumer:one", path="pkg/caller.py"),)
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:abstain",
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure_id="closure:one",
        disposition=PlanDisposition.ABSTAINED,
        obligations=obligations,
        obligation_set_id=obligation_set_identity(obligations),
        steps=(),
        invalidation_refs=("tree:candidate",),
    )
    with pytest.raises(ChangePropagationTransactionError, match="admitted"):
        create_propagation_checkpoint(plan, path_before_hashes=_hashes("pkg/caller.py"))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_happy_path_returns_canonical_records(
    roots: PropagationAuthorityRoots,
) -> None:
    plan = _admitted_plan(roots)
    lease = _lease()
    hashes = _hashes("pkg/caller.py")
    txn = ChangePropagationTransaction(step_applicator=_passing_applicator)

    report = txn.execute(plan, lease=lease, path_before_hashes=hashes)

    assert report.committed is True
    assert report.partial_merge_allowed is False
    assert isinstance(report.plan, AtomicPropagationPlan)
    assert report.plan is plan or report.plan == plan
    assert isinstance(report.transaction, PropagationTransaction)
    assert report.transaction.state is TransactionState.COMMITTED
    assert report.transaction.plan_id == plan.plan_id
    assert report.transaction.lease_id == lease.lease_id
    assert report.transaction.completed_step_ids == ("step:migrate-one",)
    assert report.transaction.active_scc_group_id == ""
    assert report.rollback is None
    assert report.reason_codes == ()
    # Canonical round-trip.
    restored = PropagationTransaction.from_dict(report.transaction.to_record())
    assert restored == report.transaction
    restored_plan = AtomicPropagationPlan.from_dict(report.plan.to_record())
    assert restored_plan.plan_id == plan.plan_id


def test_module_entry_point_matches_class(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    lease = _lease()
    hashes = _hashes("pkg/caller.py")
    a = execute_change_propagation_transaction(
        plan,
        lease=lease,
        path_before_hashes=hashes,
        step_applicator=_passing_applicator,
    )
    b = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan, lease=lease, path_before_hashes=hashes
    )
    assert a.committed and b.committed
    assert a.transaction.state is TransactionState.COMMITTED
    assert a.plan.plan_id == b.plan.plan_id


def test_verify_before_hash_and_lease(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    lease = _lease()
    hashes = _hashes("pkg/caller.py")
    current = {"pkg/caller.py": "sha256:pkg-caller.py"}

    def probe(path: str) -> str:
        return current[path]

    report = ChangePropagationTransaction(
        step_applicator=_passing_applicator,
        hash_probe=probe,
    ).execute(plan, lease=lease, path_before_hashes=hashes)
    assert report.committed

    # Drift: live hash differs from checkpoint before mutation.
    current["pkg/caller.py"] = "sha256:drifted"
    drifted = ChangePropagationTransaction(
        step_applicator=_passing_applicator,
        hash_probe=probe,
    ).execute(plan, lease=lease, path_before_hashes=hashes)
    assert not drifted.committed
    assert TransactionFailureReason.BEFORE_HASH_MISMATCH.value in drifted.reason_codes
    assert drifted.transaction.state is TransactionState.ROLLED_BACK
    assert drifted.rollback is not None
    assert drifted.rollback.restored is True


def test_missing_before_hash_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    lease = _lease()
    report = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan, lease=lease, path_before_hashes=()
    )
    assert not report.committed
    assert TransactionFailureReason.BEFORE_HASH_MISSING.value in report.reason_codes


def test_inactive_lease_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    lease = TransactionLease(
        lease_id="lease:dead",
        fence_id="fence:1",
        holder_id="holder:txn",
        permitted_write_paths=("pkg/caller.py",),
        active=False,
    )
    report = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan, lease=lease, path_before_hashes=_hashes("pkg/caller.py")
    )
    assert not report.committed
    assert TransactionFailureReason.LEASE_INVALID.value in report.reason_codes
    assert report.transaction.state is TransactionState.FAILED


def test_lease_path_mismatch_fails(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots, write_paths=("pkg/caller.py", "pkg/other.py"))
    lease = _lease(paths=("pkg/caller.py",))  # missing other.py
    report = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan,
        lease=lease,
        path_before_hashes=_hashes("pkg/caller.py", "pkg/other.py"),
    )
    assert not report.committed
    assert TransactionFailureReason.LEASE_PATH_MISMATCH.value in report.reason_codes


# ---------------------------------------------------------------------------
# SCC atomic groups
# ---------------------------------------------------------------------------


def test_scc_executed_as_one_transaction_group(roots: PropagationAuthorityRoots) -> None:
    o1 = _obligation(roots, consumer_id="consumer:a", path="pkg/a.py")
    o2 = _obligation(roots, consumer_id="consumer:b", path="pkg/b.py")
    steps = (
        PropagationPlanStep(
            step_id="step:a",
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=(o1.obligation_id,),
            transform_id="transform:a",
            write_paths=("pkg/a.py",),
            scc_group_id="group:scc-cycle",
        ),
        PropagationPlanStep(
            step_id="step:b",
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=(o2.obligation_id,),
            transform_id="transform:b",
            write_paths=("pkg/b.py",),
            dependency_step_ids=("step:a",),
            scc_group_id="group:scc-cycle",
        ),
    )
    scc = PropagationSCCGroup(
        group_id="group:scc-cycle",
        scc_id="scc:cycle",
        step_ids=("step:a", "step:b"),
        consumer_ids=("consumer:a", "consumer:b"),
    )
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:scc",
        change_set_id="changeset:scc",
        delta_id="delta:one",
        impact_closure_id="closure:scc",
        disposition=PlanDisposition.ADMITTED,
        obligations=(o1, o2),
        obligation_set_id=obligation_set_identity((o1, o2)),
        steps=steps,
        scc_groups=(scc,),
        permitted_read_paths=("pkg/a.py", "pkg/b.py"),
        permitted_write_paths=("pkg/a.py", "pkg/b.py"),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:plan",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    applied: list[str] = []

    def applicator(request: StepApplyRequest) -> StepApplyResult:
        applied.append(request.step.step_id)
        return _passing_applicator(request)

    report = ChangePropagationTransaction(step_applicator=applicator).execute(
        plan,
        lease=_lease(paths=("pkg/a.py", "pkg/b.py")),
        path_before_hashes=_hashes("pkg/a.py", "pkg/b.py"),
    )
    assert report.committed
    assert len(report.group_receipts) == 1
    group = report.group_receipts[0]
    assert group.group_id == "group:scc-cycle"
    assert group.scc_id == "scc:cycle"
    assert set(group.step_ids) == {"step:a", "step:b"}
    assert group.disposition is GroupExecutionDisposition.PASSED
    assert set(applied) == {"step:a", "step:b"}
    assert set(report.transaction.completed_step_ids) == {"step:a", "step:b"}


def test_partial_scc_group_rolls_back_and_retains_diagnostics(
    roots: PropagationAuthorityRoots,
) -> None:
    o1 = _obligation(roots, consumer_id="consumer:a", path="pkg/a.py")
    o2 = _obligation(roots, consumer_id="consumer:b", path="pkg/b.py")
    steps = (
        PropagationPlanStep(
            step_id="step:a",
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=(o1.obligation_id,),
            transform_id="transform:a",
            write_paths=("pkg/a.py",),
            scc_group_id="group:scc-cycle",
        ),
        PropagationPlanStep(
            step_id="step:b",
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=(o2.obligation_id,),
            transform_id="transform:b",
            write_paths=("pkg/b.py",),
            scc_group_id="group:scc-cycle",
        ),
    )
    scc = PropagationSCCGroup(
        group_id="group:scc-cycle",
        scc_id="scc:cycle",
        step_ids=("step:a", "step:b"),
        consumer_ids=("consumer:a", "consumer:b"),
    )
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:partial",
        change_set_id="changeset:partial",
        delta_id="delta:one",
        impact_closure_id="closure:partial",
        disposition=PlanDisposition.ADMITTED,
        obligations=(o1, o2),
        obligation_set_id=obligation_set_identity((o1, o2)),
        steps=steps,
        scc_groups=(scc,),
        permitted_read_paths=("pkg/a.py", "pkg/b.py"),
        permitted_write_paths=("pkg/a.py", "pkg/b.py"),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:plan",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    restored: list[str] = []

    def applicator(request: StepApplyRequest) -> StepApplyResult:
        if request.step.step_id == "step:b":
            return StepApplyResult(
                disposition=StepExecutionDisposition.FAILED,
                reason_codes=(TransactionFailureReason.STEP_FAILURE.value,),
                diagnostic_refs=("diagnostic:step-b-failed",),
            )
        return _passing_applicator(request)

    def restore(checkpoint: PropagationCheckpoint) -> bool:
        restored.append(checkpoint.checkpoint_id)
        return True

    report = ChangePropagationTransaction(
        step_applicator=applicator,
        restore_adapter=restore,
    ).execute(
        plan,
        lease=_lease(paths=("pkg/a.py", "pkg/b.py")),
        path_before_hashes=_hashes("pkg/a.py", "pkg/b.py"),
    )

    assert not report.committed
    assert report.transaction.state is TransactionState.ROLLED_BACK
    assert report.rollback is not None
    assert isinstance(report.rollback, PropagationRollbackReceipt)
    assert report.rollback.restored is True
    assert "diagnostic:step-b-failed" in report.group_receipts[0].diagnostic_refs or any(
        "step-b" in ref or "step_failure" in report.reason_codes
        for ref in report.rollback.diagnostic_refs
    )
    assert report.transaction.diagnostic_refs
    assert restored  # checkpoint restored
    # Partial completion: completed_step_ids must not claim the failed SCC members as committed work.
    assert "step:b" not in report.transaction.completed_step_ids
    assert report.partial_merge_allowed is False


# ---------------------------------------------------------------------------
# Failure modes: scope escape, timeout, restore failure
# ---------------------------------------------------------------------------


def test_scope_escape_rolls_back(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)

    def applicator(request: StepApplyRequest) -> StepApplyResult:
        return StepApplyResult(
            disposition=StepExecutionDisposition.PASSED,
            written_paths=("pkg/escaped.py",),  # outside authority
        )

    report = ChangePropagationTransaction(step_applicator=applicator).execute(
        plan,
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
    )
    assert not report.committed
    assert TransactionFailureReason.SCOPE_ESCAPE.value in report.reason_codes
    assert report.transaction.state is TransactionState.ROLLED_BACK
    assert report.rollback is not None


def test_timeout_rolls_back(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    report = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan,
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        observe_timeout=True,
    )
    assert not report.committed
    assert TransactionFailureReason.TIMEOUT.value in report.reason_codes
    assert report.rollback is not None
    assert report.rollback.restored is True


def test_step_timeout_disposition_rolls_back(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)

    def applicator(request: StepApplyRequest) -> StepApplyResult:
        return StepApplyResult(
            disposition=StepExecutionDisposition.TIMED_OUT,
            reason_codes=(TransactionFailureReason.TIMEOUT.value,),
        )

    report = ChangePropagationTransaction(step_applicator=applicator).execute(
        plan,
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
    )
    assert not report.committed
    assert TransactionFailureReason.TIMEOUT.value in report.reason_codes


def test_restore_failure_raises(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)

    def applicator(request: StepApplyRequest) -> StepApplyResult:
        return StepApplyResult(
            disposition=StepExecutionDisposition.FAILED,
            reason_codes=(TransactionFailureReason.STEP_FAILURE.value,),
        )

    def bad_restore(checkpoint: PropagationCheckpoint) -> bool:
        return False

    with pytest.raises(ChangePropagationTransactionError, match="restore failed"):
        ChangePropagationTransaction(
            step_applicator=applicator,
            restore_adapter=bad_restore,
        ).execute(
            plan,
            lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
        )


def test_require_committed_raises_on_failure(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    lease = TransactionLease(
        lease_id="lease:dead",
        fence_id="fence:1",
        holder_id="holder:txn",
        permitted_write_paths=("pkg/caller.py",),
        active=False,
    )
    with pytest.raises(ChangePropagationTransactionError, match="rejected"):
        ChangePropagationTransaction(step_applicator=_passing_applicator).require_committed(
            plan,
            lease=lease,
            path_before_hashes=_hashes("pkg/caller.py"),
        )


def test_partial_merge_flag_cannot_be_true(roots: PropagationAuthorityRoots) -> None:
    plan = _admitted_plan(roots)
    lease = _lease()
    report = ChangePropagationTransaction(step_applicator=_passing_applicator).execute(
        plan, lease=lease, path_before_hashes=_hashes("pkg/caller.py")
    )
    assert report.partial_merge_allowed is False
    payload = report.to_dict()
    assert payload["partial_merge_allowed"] is False
    assert payload["provider_success_is_not_merge"] is True


def test_dependency_order_across_singleton_groups(
    roots: PropagationAuthorityRoots,
) -> None:
    o1 = _obligation(roots, consumer_id="consumer:a", path="pkg/a.py")
    o2 = _obligation(roots, consumer_id="consumer:b", path="pkg/b.py")
    steps = (
        PropagationPlanStep(
            step_id="step:first",
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=(o1.obligation_id,),
            transform_id="transform:a",
            write_paths=("pkg/a.py",),
        ),
        PropagationPlanStep(
            step_id="step:second",
            kind=PlanStepKind.ANALYTICAL,
            obligation_ids=(o2.obligation_id,),
            transform_id="transform:b",
            write_paths=("pkg/b.py",),
            dependency_step_ids=("step:first",),
        ),
    )
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:deps",
        change_set_id="changeset:deps",
        delta_id="delta:one",
        impact_closure_id="closure:deps",
        disposition=PlanDisposition.ADMITTED,
        obligations=(o1, o2),
        obligation_set_id=obligation_set_identity((o1, o2)),
        steps=steps,
        scc_groups=(),
        permitted_read_paths=("pkg/a.py", "pkg/b.py"),
        permitted_write_paths=("pkg/a.py", "pkg/b.py"),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:plan",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    order: list[str] = []

    def applicator(request: StepApplyRequest) -> StepApplyResult:
        order.append(request.step.step_id)
        return _passing_applicator(request)

    report = ChangePropagationTransaction(step_applicator=applicator).execute(
        plan,
        lease=_lease(paths=("pkg/a.py", "pkg/b.py")),
        path_before_hashes=_hashes("pkg/a.py", "pkg/b.py"),
    )
    assert report.committed
    assert order == ["step:first", "step:second"]
    assert len(report.group_receipts) == 2
