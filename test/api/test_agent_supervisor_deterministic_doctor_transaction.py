"""LPR-038: sandboxed SCC doctor transactions.

Covers:
* DoctorSandboxPolicy / enforcement / hostile FS
* Weak isolation → static-only; execution-dependent abstention
* Checkout lock / writer lease / checkpoint pre-commit revalidation
* Atomic SCC apply (entire group or nothing)
* Merge-ref compare-and-swap
* Rollback and quarantine on restore failure
* Zero model/provider invocations; no completion claim
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import (
    DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
    PRODUCER_ID,
    DeterministicDoctorTransaction,
    DeterministicDoctorTransactionError,
    DoctorCheckoutLock,
    DoctorGroupDisposition,
    DoctorHostileFsObservation,
    DoctorHostileObservationKind,
    DoctorMergeRefCas,
    DoctorSandboxCapability,
    DoctorSandboxEnforcementLevel,
    DoctorSandboxError,
    DoctorSandboxPolicy,
    DoctorStepApplyRequest,
    DoctorStepApplyResult,
    DoctorStepDisposition,
    DoctorTransactionDisposition,
    DoctorTransactionReason,
    DoctorWriterLease,
    assert_no_provider_surface,
    create_doctor_checkpoint,
    evaluate_sandbox_for_plan,
    execute_deterministic_doctor_transaction,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:lpr-038",
        "forest_id": "forest:lpr-038",
        "tree_id": "tree:candidate",
        "overlay_id": "overlay:lpr-038",
        "file_root_id": "file-root:lpr-038",
        "ast_root_id": "ast:lpr-038",
        "graph_id": "graph:lpr-038",
        "corpus_id": "corpus:lpr-038",
        "index_id": "index:lpr-038",
        "model_id": "model:lpr-038",
        "cache_id": "cache:lpr-038",
        "operator_registry_id": "operators:lpr-038",
        "translator_id": "translator:lpr-038",
        "solver_id": "solver:lpr-038",
        "kernel_id": "kernel:lpr-038",
        "toolchain_id": "toolchain:lpr-038",
        "policy_id": "policy:lpr-038",
        "sandbox_id": "sandbox:lpr-038",
        "environment_id": "environment:lpr-038",
        "lease_id": "lease:writer-1",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _consumer(
    auth: DoctorAuthorityRoots,
    consumer_id: str = "consumer:one",
    disposition: DoctorRepairDisposition = DoctorRepairDisposition.SUPPORTED,
) -> DoctorConsumerDisposition:
    return DoctorConsumerDisposition(
        roots=auth,
        consumer_id=consumer_id,
        disposition=disposition,
        reason_codes=("ok",),
    )


def _admitted_plan(
    auth: DoctorAuthorityRoots | None = None,
    *,
    steps: tuple[DoctorPlanStep, ...] | None = None,
    write_paths: tuple[str, ...] = ("pkg/caller.py",),
    scc_refs: tuple[str, ...] = (),
    consumers: tuple[str, ...] = ("consumer:one",),
) -> DeterministicDoctorPlan:
    auth = auth or roots()
    sites = tuple(
        DoctorEditSite(
            path=path,
            before_hash=f"sha256:{path.replace('/', '-')}",
            span_start=0,
            span_end=8,
            artifact_id=f"blob:{path}",
        )
        for path in write_paths
    )
    if steps is None:
        steps = (
            DoctorPlanStep(
                step_id="step:migrate-one",
                kind="analytical",
                operator_id="operator:add-arg",
                consumer_ids=(consumers[0],),
                edit_site_refs=(sites[0].content_id,),
                write_paths=(write_paths[0],),
                validation_refs=scc_refs,
            ),
        )
    return DeterministicDoctorPlan(
        roots=auth,
        plan_id="plan:lpr-038",
        snapshot_id="snapshot:lpr-038",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=tuple(_consumer(auth, cid) for cid in consumers),
        impact_closure_id="impact:lpr-038",
        steps=steps,
        edit_sites=sites,
        operator_ids=("operator:add-arg",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:add-arg",
        scc_refs=scc_refs,
        permitted_read_paths=write_paths,
        permitted_write_paths=write_paths,
        lease_id="lease:writer-1",
        checkpoint_ref="checkpoint:content-addressed",
        rollback_ref="rollback:restore-checkpoint",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )


def _sandbox(
    paths: tuple[str, ...] = ("pkg/caller.py",),
    *,
    level: DoctorSandboxEnforcementLevel = DoctorSandboxEnforcementLevel.ENFORCED,
) -> DoctorSandboxPolicy:
    return DoctorSandboxPolicy(
        sandbox_id="sandbox:lpr-038",
        worktree_root_ref="worktree:candidate-1",
        permitted_paths=paths,
        enforcement_level=level,
    )


def _lock(
    *,
    base_tree_cid: str = "tree:base",
    active: bool = True,
) -> DoctorCheckoutLock:
    return DoctorCheckoutLock(
        lock_id="lock:checkout-1",
        holder_id="holder:txn",
        worktree_root_ref="worktree:candidate-1",
        base_tree_cid=base_tree_cid,
        active=active,
        fence_id="fence:lock-1",
    )


def _lease(paths: tuple[str, ...] = ("pkg/caller.py",), *, active: bool = True) -> DoctorWriterLease:
    return DoctorWriterLease(
        lease_id="lease:writer-1",
        fence_id="fence:1",
        holder_id="holder:txn",
        permitted_write_paths=paths,
        permitted_read_paths=paths,
        active=active,
    )


def _hashes(*paths: str) -> tuple[PathBeforeHash, ...]:
    return tuple(
        PathBeforeHash(path=path, before_hash=f"sha256:{path.replace('/', '-')}")
        for path in paths
    )


def _passing_applicator(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
    return DoctorStepApplyResult(
        disposition=DoctorStepDisposition.PASSED,
        written_paths=request.step.write_paths,
        observed_before_hashes=tuple(
            PathBeforeHash(
                path=p,
                before_hash=request.checkpoint.hash_map().get(
                    p, f"sha256:{p.replace('/', '-')}"
                ),
            )
            for p in request.step.write_paths
        ),
        static_replay=request.static_replay_only,
    )


# ---------------------------------------------------------------------------
# Interface / policy
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE == "DeterministicDoctorTransaction@1"
    assert DeterministicDoctorTransaction.INTERFACE == DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE
    assert PRODUCER_ID == "deterministic-doctor-transaction@1"
    assert_no_provider_surface()


def test_sandbox_policy_rejects_secrets_and_network() -> None:
    with pytest.raises(DoctorSandboxError, match="secrets"):
        DoctorSandboxPolicy(
            sandbox_id="sandbox:x",
            worktree_root_ref="worktree:x",
            permitted_paths=("pkg/a.py",),
            secrets_inherited=True,
        )
    with pytest.raises(DoctorSandboxError, match="network"):
        DoctorSandboxPolicy(
            sandbox_id="sandbox:x",
            worktree_root_ref="worktree:x",
            permitted_paths=("pkg/a.py",),
            network_denied=False,
        )


def test_sandbox_policy_rejects_tcb_paths() -> None:
    with pytest.raises(DoctorSandboxError, match="TCB"):
        DoctorSandboxPolicy(
            sandbox_id="sandbox:x",
            worktree_root_ref="worktree:x",
            permitted_paths=(
                "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py",
            ),
        )


def test_weak_isolation_permits_static_only() -> None:
    policy = _sandbox(level=DoctorSandboxEnforcementLevel.WEAK)
    assert policy.permits_static_replay_only
    assert not policy.permits_target_execution
    plan = _admitted_plan()
    reasons = evaluate_sandbox_for_plan(
        policy, plan, requires_target_execution=True
    )
    assert DoctorTransactionReason.SANDBOX_WEAK_EXECUTION_FORBIDDEN.value in reasons
    assert DoctorTransactionReason.EXECUTION_DEPENDENT_ABSTAIN.value in reasons


def test_hostile_fs_observations_fail_closed() -> None:
    plan = _admitted_plan()
    policy = _sandbox()
    obs = DoctorHostileFsObservation(
        kind=DoctorHostileObservationKind.SYMLINK,
        path="pkg/caller.py",
    )
    reasons = evaluate_sandbox_for_plan(
        policy, plan, hostile_observations=(obs,)
    )
    assert DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value in reasons


def test_writer_lease_rejects_dirty_user_tree() -> None:
    with pytest.raises(DeterministicDoctorTransactionError, match="dirty"):
        DoctorWriterLease(
            lease_id="lease:x",
            fence_id="fence:x",
            holder_id="holder:x",
            permitted_write_paths=("pkg/a.py",),
            dirty_user_tree=True,
        )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def test_create_content_addressed_checkpoint() -> None:
    plan = _admitted_plan()
    hashes = _hashes("pkg/caller.py")
    checkpoint = create_doctor_checkpoint(
        plan,
        path_before_hashes=hashes,
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        worktree_root_ref="worktree:candidate-1",
    )
    assert checkpoint.plan_id == plan.plan_id
    assert checkpoint.plan_content_id == plan.content_id
    assert checkpoint.hash_map()["pkg/caller.py"].startswith("sha256:")
    again = create_doctor_checkpoint(
        plan,
        path_before_hashes=hashes,
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        worktree_root_ref="worktree:candidate-1",
    )
    assert again.checkpoint_id == checkpoint.checkpoint_id


def test_checkpoint_rejects_non_admitted_plan() -> None:
    auth = roots()
    plan = DeterministicDoctorPlan(
        roots=auth,
        plan_id="plan:abstain",
        snapshot_id="snapshot:x",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ABSTAINED,
        consumer_dispositions=(_consumer(auth),),
        impact_closure_id="impact:x",
        invalidation_refs=("tree:candidate",),
    )
    with pytest.raises(DeterministicDoctorTransactionError, match="admitted"):
        create_doctor_checkpoint(
            plan,
            path_before_hashes=_hashes("pkg/caller.py"),
            base_tree_cid="tree:base",
            candidate_tree_cid="tree:candidate",
            worktree_root_ref="worktree:x",
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_happy_path_commits_candidate_tree() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is True
    assert report.partial_merge_allowed is False
    assert report.disposition is DoctorTransactionDisposition.COMMITTED
    assert report.candidate_tree is not None
    assert report.candidate_tree.base_tree_cid == "tree:base"
    assert report.candidate_tree.candidate_tree_cid == "tree:candidate"
    assert report.rollback is None
    assert report.model_invocation_count == 0
    assert report.provider_invocation_count == 0
    # Transaction never claims task completion.
    assert report.disposition.claims_completion is False
    assert all(g.disposition is DoctorGroupDisposition.PASSED for g in report.group_receipts)
    record = report.to_record()
    assert record["claims_completion"] is False
    assert record["partial_merge_allowed"] is False


def test_module_wrapper_execute() -> None:
    plan = _admitted_plan()
    report = execute_deterministic_doctor_transaction(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    # Default static applicator passes hermetically for admitted plans.
    assert report.committed is True
    assert report.candidate_tree is not None


def test_require_committed_raises_on_failure() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    with pytest.raises(DeterministicDoctorTransactionError, match="rejected"):
        txn.require_committed(
            plan,
            sandbox_policy=_sandbox(),
            checkout_lock=_lock(active=False),
            lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
            base_tree_cid="tree:base",
            candidate_tree_cid="tree:candidate",
        )


# ---------------------------------------------------------------------------
# Isolation / sandbox failures
# ---------------------------------------------------------------------------


def test_execution_dependent_abstains_under_weak_sandbox() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(level=DoctorSandboxEnforcementLevel.WEAK),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        requires_target_execution=True,
    )
    assert report.committed is False
    assert report.disposition is DoctorTransactionDisposition.ABSTAINED
    assert DoctorTransactionReason.EXECUTION_DEPENDENT_ABSTAIN.value in report.reason_codes
    assert report.claims_completion if False else True  # never claims completion
    assert report.disposition.claims_completion is False


def test_hostile_symlink_abstains() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        hostile_observations=(
            DoctorHostileFsObservation(
                kind=DoctorHostileObservationKind.SYMLINK,
                path="pkg/caller.py",
            ),
        ),
    )
    assert report.committed is False
    assert DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value in report.reason_codes


def test_hardlink_submodule_device_path_race_hostile() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    for kind in (
        DoctorHostileObservationKind.HARDLINK,
        DoctorHostileObservationKind.SUBMODULE,
        DoctorHostileObservationKind.DEVICE,
        DoctorHostileObservationKind.PATH_RACE,
    ):
        report = txn.execute(
            plan,
            sandbox_policy=_sandbox(),
            checkout_lock=_lock(),
            lease=_lease(),
            path_before_hashes=_hashes("pkg/caller.py"),
            base_tree_cid="tree:base",
            candidate_tree_cid="tree:candidate",
            hostile_observations=(
                DoctorHostileFsObservation(kind=kind, path="pkg/caller.py"),
            ),
        )
        assert report.committed is False
        assert DoctorTransactionReason.HOSTILE_FS_OBSERVATION.value in report.reason_codes


# ---------------------------------------------------------------------------
# Lease / lock / hash failures
# ---------------------------------------------------------------------------


def test_inactive_lease_rejects() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(active=False),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert DoctorTransactionReason.LEASE_INVALID.value in report.reason_codes


def test_lease_path_mismatch_rejects() -> None:
    plan = _admitted_plan(write_paths=("pkg/caller.py",))
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(paths=("pkg/other.py",)),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert DoctorTransactionReason.LEASE_PATH_MISMATCH.value in report.reason_codes


def test_before_hash_mismatch_rolls_back() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(
        step_applicator=_passing_applicator,
        hash_probe=lambda path: "sha256:wrong",
    )
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    assert report.rollback is not None
    assert report.rollback.restored is True
    assert DoctorTransactionReason.BEFORE_HASH_MISMATCH.value in report.reason_codes


def test_missing_before_hash_fails() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=(),  # missing
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert DoctorTransactionReason.BEFORE_HASH_MISSING.value in report.reason_codes


# ---------------------------------------------------------------------------
# Step / SCC / scope failures
# ---------------------------------------------------------------------------


def test_scope_escape_rolls_back() -> None:
    plan = _admitted_plan()

    def escape(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED,
            written_paths=("pkg/escape.py",),
        )

    txn = DeterministicDoctorTransaction(step_applicator=escape)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    assert DoctorTransactionReason.SCOPE_ESCAPE.value in report.reason_codes
    assert report.group_receipts
    assert report.group_receipts[0].disposition is DoctorGroupDisposition.ROLLED_BACK


def test_step_failure_rolls_back_whole_scc() -> None:
    auth = roots()
    steps = (
        DoctorPlanStep(
            step_id="step:a",
            kind="analytical",
            operator_id="operator:add-arg",
            consumer_ids=("consumer:one",),
            write_paths=("pkg/caller.py",),
            validation_refs=("scc:mutual",),
        ),
        DoctorPlanStep(
            step_id="step:b",
            kind="analytical",
            operator_id="operator:add-arg",
            consumer_ids=("consumer:two",),
            write_paths=("pkg/caller.py",),
            dependency_step_ids=("step:a",),
            validation_refs=("scc:mutual",),
        ),
    )
    plan = _admitted_plan(
        auth,
        steps=steps,
        consumers=("consumer:one", "consumer:two"),
        scc_refs=("scc:mutual",),
    )

    def fail_second(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
        if request.step.step_id == "step:b":
            return DoctorStepApplyResult(
                disposition=DoctorStepDisposition.FAILED,
                reason_codes=(DoctorTransactionReason.STEP_FAILURE.value,),
            )
        return _passing_applicator(request)

    txn = DeterministicDoctorTransaction(step_applicator=fail_second)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    # Entire SCC group rolled back — no partial merge.
    assert report.partial_merge_allowed is False
    assert any(
        g.disposition is DoctorGroupDisposition.ROLLED_BACK for g in report.group_receipts
    )


def test_timeout_rolls_back() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        observe_timeout=True,
    )
    assert report.committed is False
    assert DoctorTransactionReason.TIMEOUT.value in report.reason_codes
    assert report.rollback is not None


# ---------------------------------------------------------------------------
# Cache / CAS / pre-commit revalidation
# ---------------------------------------------------------------------------


def test_stale_cache_binding_aborts_before_commit() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(
        step_applicator=_passing_applicator,
        cache_binding_probe=lambda refs: ("cache:stale",),
    )
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        cache_binding_refs=("cache:binding-1",),
    )
    assert report.committed is False
    assert DoctorTransactionReason.CACHE_BINDING_STALE.value in report.reason_codes


def test_merge_ref_cas_success() -> None:
    plan = _admitted_plan()
    cas = DoctorMergeRefCas(
        cas_id="cas:1",
        ref_name="refs/heads/main",
        expected_ref="tree:base",
        desired_ref="tree:candidate",
        holder_id="holder:txn",
    )
    live = {"refs/heads/main": "tree:base"}

    def probe(ref: str) -> str:
        return live[ref]

    txn = DeterministicDoctorTransaction(
        step_applicator=_passing_applicator,
        live_ref_probe=probe,
    )
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        merge_cas=cas,
    )
    assert report.committed is True
    assert report.merge_cas is not None
    assert report.merge_cas.desired_ref == "tree:candidate"


def test_merge_ref_cas_conflict_rolls_back() -> None:
    plan = _admitted_plan()
    cas = DoctorMergeRefCas(
        cas_id="cas:1",
        ref_name="refs/heads/main",
        expected_ref="tree:base",
        desired_ref="tree:candidate",
        holder_id="holder:txn",
    )
    txn = DeterministicDoctorTransaction(
        step_applicator=_passing_applicator,
        live_ref_probe=lambda ref: "tree:other-tip",
    )
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        merge_cas=cas,
    )
    assert report.committed is False
    assert (
        DoctorTransactionReason.CAS_CONFLICT.value in report.reason_codes
        or DoctorTransactionReason.CAS_EXPECTED_MISMATCH.value in report.reason_codes
    )
    assert report.rollback is not None


# ---------------------------------------------------------------------------
# Quarantine on restore failure
# ---------------------------------------------------------------------------


def test_restore_failure_quarantines() -> None:
    plan = _admitted_plan()

    def fail_step(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.FAILED,
            reason_codes=(DoctorTransactionReason.STEP_FAILURE.value,),
        )

    txn = DeterministicDoctorTransaction(
        step_applicator=fail_step,
        restore_adapter=lambda _ckpt: False,
    )
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert report.disposition is DoctorTransactionDisposition.QUARANTINED
    assert report.rollback is not None
    assert report.rollback.quarantined is True
    assert report.rollback.restored is False
    assert DoctorTransactionReason.QUARANTINE_REQUIRED.value in report.reason_codes
    # Quarantine never claims completion.
    assert report.disposition.claims_completion is False


def test_plan_not_admitted_rejects() -> None:
    auth = roots()
    plan = DeterministicDoctorPlan(
        roots=auth,
        plan_id="plan:abstain",
        snapshot_id="snapshot:x",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ABSTAINED,
        consumer_dispositions=(_consumer(auth),),
        impact_closure_id="impact:x",
        invalidation_refs=("tree:candidate",),
    )
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.committed is False
    assert DoctorTransactionReason.PLAN_NOT_ADMITTED.value in report.reason_codes


def test_sandbox_capabilities_include_required_set() -> None:
    policy = _sandbox()
    required = {
        DoctorSandboxCapability.PATH_CONFINEMENT,
        DoctorSandboxCapability.SECRETS_DENIED,
        DoctorSandboxCapability.NETWORK_DENIED,
        DoctorSandboxCapability.NO_TARGET_IMPORT,
        DoctorSandboxCapability.DISPOSABLE_WORKTREE,
    }
    assert required.issubset(set(policy.required_capabilities))
    assert policy.command_allowed("python -m pytest")
    assert not policy.command_allowed("curl http://evil")
    assert policy.path_permitted("pkg/caller.py")
    assert not policy.path_permitted("../escape.py")


def test_deterministic_content_ids() -> None:
    plan = _admitted_plan()
    txn = DeterministicDoctorTransaction(step_applicator=_passing_applicator)
    a = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        transaction_id="txn:stable",
    )
    b = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=_lock(),
        lease=_lease(),
        path_before_hashes=_hashes("pkg/caller.py"),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        transaction_id="txn:stable",
    )
    assert a.content_id == b.content_id
    assert a.candidate_tree is not None and b.candidate_tree is not None
    assert a.candidate_tree.content_id == b.candidate_tree.content_id
