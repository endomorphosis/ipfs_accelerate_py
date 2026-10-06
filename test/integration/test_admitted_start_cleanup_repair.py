"""Signed native cleanup of interrupted START, without a provider invocation."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    MutationRecoveryAction,
    MutationTransactionPhase,
    StaleTreeError,
    TransactionConflictError,
)
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleSagaPhase,
    ProcessIdentityMismatch,
)
from test.integration.test_admitted_benchmark_runtime import admitted  # noqa: F401


def _interrupt_after_durable_launch(runtime, monkeypatch):
    snapshot = runtime.process.snapshot
    injected = []

    def fail_one_postlaunch_snapshot(profile):
        witnesses = getattr(runtime.process, "_launch_witnesses", {})
        if witnesses and not injected:
            checkpoint = runtime.orchestrator.store.latest()[profile.target_id]
            assert checkpoint.phase is LifecycleSagaPhase.VERIFYING_HEALTH
            assert checkpoint.new_tree is not None
            assert len(checkpoint.new_tree.roots) == 1
            root = checkpoint.new_tree.roots[0]
            assert witnesses[root.pid].identity_id == root.identity_id
            assert runtime.process.identity_alive(root)
            assert any(child.pid == root.pid and child.poll() is None
                       for child in runtime._children)
            injected.append(root)
            raise ProcessIdentityMismatch("injected post-launch descendant uncertainty")
        return snapshot(profile)

    monkeypatch.setattr(runtime.process, "snapshot", fail_one_postlaunch_snapshot)
    failed = runtime.start()
    assert not failed.succeeded
    assert failed.error.details["exception_type"] == "ProcessIdentityMismatch"
    assert len(injected) == 1
    request = runtime._requests[failed.request_id]
    transaction = runtime.service._state_store.get_mutation(request)
    assert transaction.phase is MutationTransactionPhase.REPAIR_REQUIRED
    assert transaction.applied_effect_ids == ()
    assert runtime.process.identity_alive(injected[0])
    assert not (runtime.state / "start-cleanup-repair-receipt.json").exists()
    return failed, request, transaction, injected[0]


def _verify_repair_stop_and_close(runtime, owner, prepared, original_task,
                                  failed, request, root, monkeypatch,
                                  *, expected_repair_calls=1, repair_receipt_expected=True):
    original_start = (runtime.state / "start-receipt.json").read_bytes()
    launch_count = len(runtime._children)
    observed_sagas = []
    recover = runtime.service.recover_mutation

    def capture_repair(*args, **kwargs):
        transaction = recover(*args, **kwargs)
        state = runtime.orchestrator.store.latest()[runtime.profile.target_id]
        assert state.phase is LifecycleSagaPhase.FAILED
        assert state.failure_code == "interrupted_start_cleanup_fenced"
        assert transaction.phase is MutationTransactionPhase.REPAIRED
        assert transaction.result is None or not transaction.result.succeeded
        observed_sagas.append(state)
        return transaction

    monkeypatch.setattr(runtime.service, "recover_mutation", capture_repair)
    stopped = runtime.stop()
    assert stopped.succeeded, stopped.error
    assert len(observed_sagas) == expected_repair_calls
    assert len(runtime._children) == launch_count
    assert not runtime.process.identity_alive(root)
    assert not runtime.process.snapshot(runtime.profile).members
    assert all(child.poll() is not None for child in runtime._children)
    transaction = runtime.service._state_store.get_mutation(request)
    assert transaction.phase is MutationTransactionPhase.REPAIRED
    assert (runtime.state / "start-receipt.json").read_bytes() == original_start
    assert json.loads(original_start)["status"] == failed.status.value
    proof = json.loads((runtime.state / "start-cleanup-process-proof-receipt.json").read_text())
    assert proof["schema"] == "interrupted-start-cleanup-repair@1"
    assert proof["phase"] == LifecycleSagaPhase.FAILED.value
    assert proof["process_tree_absent"] is True
    assert proof["start_succeeded"] is False
    assert proof["completion_authority"] is False
    repair_path = runtime.state / "start-cleanup-repair-receipt.json"
    if repair_receipt_expected:
        assert json.loads(repair_path.read_text())["phase"] == MutationTransactionPhase.REPAIRED.value
    else:
        # The control journal is authoritative when its committed response
        # was lost before the convenience receipt could be written.
        assert not repair_path.exists()
    observation = driver._start_cleanup_observation(runtime, failed.to_dict())
    assert observation["status"] == ("available" if repair_receipt_expected else "partial")
    assert observation["proof_observation"] == "observed"
    assert observation["lifecycle_phase"] == "failed"
    assert observation["marker_bound_process_tree_absent"] is True
    assert observation["start_succeeded"] is False
    assert observation["control_observation"] == ("observed" if repair_receipt_expected else "missing")
    assert observation["control_phase"] == ("repaired" if repair_receipt_expected else None)
    assert observation["absence_scope"] == "recorded_marker_bound_tree"
    assert observation["completion_authority"] is False
    assert observation["retry_authority"] is False
    assert observation["execution_authority"] is False
    assert all(key not in observation for key in (
        "request_id", "transaction_id", "transition_id", "message", "body"))
    assert owner.source.get_task(prepared["task_cid"]) == original_task
    assert runtime.manifest["provider_dispatch_allowed"] is False

    # The real fixture closes the owner/coordinator after this test. Verify
    # actual close succeeds, without calling close twice or replacing it.
    close = runtime.close

    def verified_close():
        close()
        assert all(child.poll() is not None for child in runtime._children)
        assert not runtime.process.snapshot(runtime.profile).members
        assert runtime._bootstrap_stop.is_set()
        assert not runtime._bootstrap_thread.is_alive()
        assert runtime._listener.fileno() == -1

    monkeypatch.setattr(runtime, "close", verified_close)


def test_signed_native_failed_start_repairs_custody_then_stops(admitted, monkeypatch):
    runtime, owner, prepared = admitted
    original_task = owner.source.get_task(prepared["task_cid"])
    failed, request, _transaction, root = _interrupt_after_durable_launch(runtime, monkeypatch)
    _verify_repair_stop_and_close(runtime, owner, prepared, original_task,
                                  failed, request, root, monkeypatch)


@pytest.mark.parametrize("loss_mode", ["before-control-cas", "after-control-cas"])
def test_signed_native_cleanup_replays_after_lost_control_response(admitted, monkeypatch, loss_mode):
    runtime, owner, prepared = admitted
    original_task = owner.source.get_task(prepared["task_cid"])
    failed, request, transaction, root = _interrupt_after_durable_launch(runtime, monkeypatch)
    original_start = (runtime.state / "start-receipt.json").read_bytes()
    store = runtime.service._state_store
    compare_and_swap = store.compare_and_swap_mutation
    lost = []

    def lose_one_repair_response(*args, **kwargs):
        if kwargs.get("phase") is MutationTransactionPhase.REPAIRED and not lost:
            checkpoint = runtime.orchestrator.store.latest()[runtime.profile.target_id]
            assert checkpoint.phase is LifecycleSagaPhase.FAILED
            assert not runtime.process.identity_alive(root)
            assert not runtime.process.snapshot(runtime.profile).members
            lost.append(True)
            if loss_mode == "after-control-cas":
                compare_and_swap(*args, **kwargs)
            raise OSError("injected lost control repair response")
        return compare_and_swap(*args, **kwargs)

    monkeypatch.setattr(store, "compare_and_swap_mutation", lose_one_repair_response)
    with pytest.raises(OSError, match="injected lost control repair response"):
        runtime.stop()
    assert lost == [True]
    assert (runtime.state / "start-receipt.json").read_bytes() == original_start
    assert not (runtime.state / "start-cleanup-repair-receipt.json").exists()
    current = runtime.service._state_store.get_mutation(request)
    if loss_mode == "before-control-cas":
        assert current == transaction
    else:
        assert current.phase is MutationTransactionPhase.REPAIRED
    _verify_repair_stop_and_close(runtime, owner, prepared, original_task,
                                  failed, request, root, monkeypatch,
                                  expected_repair_calls=int(loss_mode == "before-control-cas"),
                                  repair_receipt_expected=loss_mode == "before-control-cas")


@pytest.mark.parametrize("denial", ["revoked-profile", "signed-profile-drift", "stale-revision"])
def test_signed_native_cleanup_revalidates_authority_and_identity(admitted, monkeypatch, denial):
    runtime, owner, prepared = admitted
    original_task = owner.source.get_task(prepared["task_cid"])
    failed, request, transaction, root = _interrupt_after_durable_launch(runtime, monkeypatch)
    original_profile = runtime.profile
    marker = runtime.profile_dir / profile_authority.REVOKE_MARKER
    expected_revision = transaction.revision
    if denial == "revoked-profile":
        marker.write_text("revoked for cleanup authorization qualification\n")
        marker.chmod(0o600)
        error = StaleTreeError
    elif denial == "signed-profile-drift":
        runtime.profile = replace(original_profile, profile_id="", environment=(
            *original_profile.environment, ("UNSIGNED_CLEANUP_MARKER", "changed")))
        error = StaleTreeError
    else:
        expected_revision -= 1
        error = TransactionConflictError
    try:
        with pytest.raises(error):
            runtime.service.recover_mutation(request, expected_revision=expected_revision,
                                             action=MutationRecoveryAction.REPAIR)
        assert runtime.process.identity_alive(root)
        assert runtime.service._state_store.get_mutation(request) == transaction
        assert not (runtime.state / "start-cleanup-process-proof-receipt.json").exists()
        assert owner.source.get_task(prepared["task_cid"]) == original_task
    finally:
        runtime.profile = original_profile
        if denial == "revoked-profile":
            marker.unlink()
    _verify_repair_stop_and_close(runtime, owner, prepared, original_task,
                                  failed, request, root, monkeypatch)
