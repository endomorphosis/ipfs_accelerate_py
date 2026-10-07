"""Process-only lifecycle transitions cannot consume container custody.

The process adapter is an authored double; the lifecycle journal and private
cleanup namespace are real. These are refusal tests, not Docker cleanup proofs.
"""
from pathlib import Path
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleSagaPhase, ProcessTreeNotFenced,
)
from test.api.test_agent_supervisor_lifecycle_orchestrator import (
    Clock, FakeProcessAdapter, _orchestrator, _profile, _request,
)


def _custody(profile, kind="binding"):
    directory = Path(profile.run_root) / "provider-cleanup-bindings"
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if kind == "symlink":
        target = directory.parent / "foreign-cleanup"
        target.mkdir(mode=0o700)
        directory.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        directory.write_text("not a directory\n")
    else:
        directory.mkdir(mode=0o700)
        suffix = {"binding": ".json", "retired": ".authority",
                  "completion": ".completion", "malformed": ".unknown"}[kind]
        record = directory / ("a" * 64 + suffix)
        record.write_text('{"fixture":"unconsumed custody"}\n')
        record.chmod(0o600)
    return directory


@pytest.mark.parametrize("operation", [Operation.START, Operation.STOP, Operation.RESTART])
@pytest.mark.parametrize("kind", ["binding", "retired", "completion", "malformed", "symlink", "file"])
def test_existing_cleanup_refuses_process_effects_and_latches_custody(tmp_path, operation, kind):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    if operation is Operation.RESTART:
        adapter.seed_tree()
    owner = _orchestrator(profile, adapter, clock)
    _custody(profile, kind)
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        owner.execute(_request(profile, operation=operation))
    state = owner.store.latest()[profile.target_id]
    assert state.phase is LifecycleSagaPhase.PARTIAL_FAILURE
    assert state.failure_code == "provider_cleanup_owner_required"
    assert state.receipt is None and not state.old_tree_fenced
    assert adapter.launches == adapter.terminations == 0


def test_deleted_custody_cannot_clear_latched_refusal_on_restart(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import durable_cleanup_observer

    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    request = _request(profile)
    owner = _orchestrator(profile, adapter, clock)
    directory = _custody(profile)
    with pytest.raises(ProcessTreeNotFenced):
        owner.execute(request)
    previous = owner.store.latest()[profile.target_id]
    for entry in directory.iterdir():
        entry.unlink()
    directory.rmdir()
    monkeypatch.setattr(durable_cleanup_observer, "has_cleanup_custody",
                        lambda *_: pytest.fail("latched refusal must precede new observation"))
    resumed = _orchestrator(profile, adapter, clock)
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        resumed.execute(request)
    assert resumed.store.latest()[profile.target_id] == previous
    assert adapter.launches == adapter.terminations == 0


def test_cached_stop_receipt_cannot_bypass_current_cleanup_custody(tmp_path):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    owner = _orchestrator(profile, adapter, clock)
    request = _request(profile, operation=Operation.STOP)
    receipt = owner.execute(request)
    previous = owner.store.latest()[profile.target_id]
    _custody(profile, "retired")
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        owner.execute(request)
    assert owner.store.latest()[profile.target_id] == previous
    assert previous.receipt == receipt
    assert adapter.launches == adapter.terminations == 0


def test_cleanup_appearing_during_empty_process_observation_prevents_fence(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    owner = _orchestrator(profile, adapter, clock)
    old = adapter.snapshot(profile)
    snapshot = adapter.snapshot

    def publish_then_snapshot(value):
        _custody(profile)
        return snapshot(value)

    monkeypatch.setattr(adapter, "snapshot", publish_then_snapshot)
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        owner._prove_absent(profile, old)
    assert adapter.launches == adapter.terminations == 0


def test_cleanup_appearing_after_signal_prevents_old_fenced_and_restart(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    owner = _orchestrator(profile, adapter, clock)
    terminate = adapter.terminate

    def stopped(*args, **kwargs):
        terminate(*args, **kwargs)
        _custody(profile)

    monkeypatch.setattr(adapter, "terminate", stopped)
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        owner.execute(_request(profile))
    state = owner.store.latest()[profile.target_id]
    assert not state.old_tree_fenced and state.receipt is None
    assert state.failure_code == "provider_cleanup_owner_required"
    assert adapter.terminations == 1 and adapter.launches == 0


@pytest.mark.parametrize("healthy", [False, True])
def test_cleanup_created_by_child_retains_custody_during_start_and_repair(tmp_path, monkeypatch, healthy):
    from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
        MutationRecoveryAction, MutationTransactionPhase, MutationTransactionState,
    )

    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.healthy_value = healthy
    owner = _orchestrator(profile, adapter, clock)
    request = _request(profile, operation=Operation.START, deadline_ms=50)
    launch = adapter.launch

    def child_publishes(*args, **kwargs):
        identity = launch(*args, **kwargs)
        _custody(profile)
        return identity

    monkeypatch.setattr(adapter, "launch", child_publishes)
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        owner.execute(request)
    previous = owner.store.latest()[profile.target_id]
    assert previous.new_tree is not None and previous.receipt is None
    assert previous.failure_code == "provider_cleanup_owner_required"
    assert adapter.live and adapter.launches == 1 and adapter.terminations == 0
    transaction = replace(
        MutationTransactionState.prepare(request, now_ms=clock.now_ms()),
        phase=MutationTransactionPhase.REPAIR_REQUIRED,
        recovery_action=MutationRecoveryAction.REPAIR,
        failure_code="provider_cleanup_owner_required",
    )
    with pytest.raises(ProcessTreeNotFenced, match="durable provider cleanup"):
        owner.repair_start_cleanup(request, transaction, timeout_ms=50)
    assert owner.store.latest()[profile.target_id] == previous
    assert adapter.live and adapter.launches == 1 and adapter.terminations == 0
