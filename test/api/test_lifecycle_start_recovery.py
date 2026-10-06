"""Owned-process exit races and exact interrupted START custody."""
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LinuxProcessAdapter, LifecycleAction, LifecycleSagaPhase, ProcessIdentityMismatch, ProcessTreeNotFenced,
    ProcessTreeSnapshot, CONFIGURATION_ROOT_ENV,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    MutationTransactionState, MutationTransactionPhase, MutationRecoveryAction,
    TransactionConflictError, StaleLeaseError,
)
from test.api.test_agent_supervisor_lifecycle_orchestrator import (
    Clock, FakeProcessAdapter, _orchestrator, _profile, _request,
)


@pytest.mark.parametrize("empty_zombie_environment", [False, True])
def test_snapshot_omits_child_that_exits_between_stat_and_environment(
    tmp_path, monkeypatch, empty_zombie_environment,
):
    profile = _profile(tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-c", "import os; os.write(1,b'R'); os.read(0,1); os._exit(0)"],
        env=profile.launch_environment(9), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        start_new_session=True,
    )
    adapter = LinuxProcessAdapter()
    environ = adapter._environ
    reads = 0

    def exit_at_identity_read(pid):
        nonlocal reads
        assert pid == child.pid
        reads += 1
        if reads == 2:
            child.stdin.write(b"X")
            child.stdin.flush()
            until = time.monotonic() + 3
            while time.monotonic() < until:
                # Do not reap: /proc/environ for this real zombie is empty.
                if Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].startswith("Z "):
                    break
                time.sleep(.005)
            else:
                pytest.fail("owned child did not exit at the fixture barrier")
            # This host denies zombie environ access. Exercise that real
            # branch and the empty-environ read branch separately, retaining
            # the actual child exit and independent /proc zombie observation.
            if empty_zombie_environment:
                return {}
        return environ(pid)

    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter((Path(f"/proc/{child.pid}"),))
                        if str(path) == "/proc" else original_iterdir(path))
    monkeypatch.setattr(adapter, "_environ", exit_at_identity_read)
    try:
        assert child.stdout.read(1) == b"R"
        assert adapter.snapshot(profile).members == ()
        assert reads == 2
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
        child.stdin.close()
        child.stdout.close()


def test_postlaunch_snapshot_failure_retains_exact_root_for_recovery(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    owner = _orchestrator(profile, adapter, clock)
    snapshot = adapter.snapshot
    failed = False

    def fail_after_launch(selected):
        nonlocal failed
        if adapter.launches and not failed:
            failed = True
            raise ProcessIdentityMismatch("injected descendant identity uncertainty")
        return snapshot(selected)

    monkeypatch.setattr(adapter, "snapshot", fail_after_launch)
    request = _request(profile, operation=Operation.START)
    with pytest.raises(ProcessIdentityMismatch):
        owner.start(request)
    checkpoint = owner.store.latest()[profile.target_id]
    assert checkpoint.phase is LifecycleSagaPhase.VERIFYING_HEALTH
    assert checkpoint.new_tree.roots[0].identity_id == snapshot(profile).roots[0].identity_id
    assert adapter.launches == 1
    receipt = owner.start(request)
    assert receipt.succeeded
    assert adapter.launches == 1


def _interrupted(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    owner = _orchestrator(profile, adapter, clock)
    snapshot = adapter.snapshot

    def fail(selected):
        if adapter.launches:
            raise ProcessIdentityMismatch("injected descendant identity uncertainty")
        return snapshot(selected)

    request = _request(profile, operation=Operation.START)
    with monkeypatch.context() as patch:
        patch.setattr(adapter, "snapshot", fail)
        with pytest.raises(ProcessIdentityMismatch):
            owner.start(request)
    transaction = replace(MutationTransactionState.prepare(request, now_ms=clock.now_ms()),
        phase=MutationTransactionPhase.REPAIR_REQUIRED, revision=2,
        recovery_action=MutationRecoveryAction.REPAIR, failure_code="conflict")
    return profile, clock, adapter, owner, request, transaction


def test_cleanup_repair_fences_only_existing_tree_then_normal_stop(tmp_path, monkeypatch):
    profile, clock, adapter, owner, request, transaction = _interrupted(tmp_path, monkeypatch)
    proof = owner.repair_start_cleanup(request, transaction, timeout_ms=100)
    assert proof["process_tree_absent"] is True
    assert proof["start_succeeded"] is False
    state = owner.store.latest()[profile.target_id]
    assert state.phase is LifecycleSagaPhase.FAILED
    assert state.receipt is None
    assert state.failure_code == "interrupted_start_cleanup_fenced"
    # Recovery can be repeated after process fencing but before control CAS.
    assert owner.repair_start_cleanup(request, transaction, timeout_ms=100) == proof
    assert adapter.terminations == 1
    stopped = owner.stop(_request(profile, operation=Operation.STOP, key="stop:cleanup"))
    assert stopped.succeeded
    assert stopped.old_tree_fenced
    assert adapter.launches == 1
    assert adapter.terminations == 1


@pytest.mark.parametrize("mutation", ["transaction", "intent", "birth", "fence", "unknown", "invisible"])
def test_cleanup_refuses_changed_or_unknown_custody_before_signaling(tmp_path, monkeypatch, mutation):
    profile, clock, adapter, owner, request, transaction = _interrupted(tmp_path, monkeypatch)
    if mutation == "transaction":
        transaction = replace(transaction, lease_id="lease:other", transaction_id="")
    elif mutation == "intent":
        changed = _request(profile, operation=Operation.START, key="start:foreign")
        transaction = replace(MutationTransactionState.prepare(changed, now_ms=clock.now_ms()),
            phase=MutationTransactionPhase.REPAIR_REQUIRED, revision=2,
            recovery_action=MutationRecoveryAction.REPAIR, failure_code="conflict")
        request = changed
    elif mutation in {"birth", "fence"}:
        root = adapter.snapshot(profile).roots[0]
        altered = replace(root, identity_id="", **({"start_time_ticks": root.start_time_ticks + 1}
            if mutation == "birth" else {"fencing_epoch": root.fencing_epoch + 1}))
        adapter.live.pop(root.identity_id)
        adapter.live[altered.identity_id] = altered
    elif mutation == "unknown":
        def unknown(_profile):
            raise ProcessIdentityMismatch("unreadable or foreign selected identity")
        monkeypatch.setattr(adapter, "snapshot", unknown)
    else:
        monkeypatch.setattr(adapter, "snapshot", lambda selected: ProcessTreeSnapshot(
            profile_id=selected.profile_id, run_id=selected.run_id, members=(), captured_at_ms=clock.now_ms()))
    with pytest.raises((TransactionConflictError, ProcessIdentityMismatch)):
        owner.repair_start_cleanup(request, transaction, timeout_ms=100)
    assert adapter.terminations == 0
    assert adapter.launches == 1
    assert owner.store.latest()[profile.target_id].phase is LifecycleSagaPhase.VERIFYING_HEALTH


def test_cleanup_refuses_expired_original_permit(tmp_path, monkeypatch):
    profile, clock, adapter, owner, request, transaction = _interrupted(tmp_path, monkeypatch)
    clock.value_ms = request.authorization.expires_at_ms
    with pytest.raises(StaleLeaseError, match="expired"):
        owner.repair_start_cleanup(request, transaction, timeout_ms=100)
    assert adapter.terminations == 0


def test_cleanup_retains_custody_if_a_descendant_survives(tmp_path, monkeypatch):
    profile, clock, adapter, owner, request, transaction = _interrupted(tmp_path, monkeypatch)
    adapter.leave_descendant = True
    with pytest.raises(ProcessTreeNotFenced, match="absence"):
        owner.repair_start_cleanup(request, transaction, timeout_ms=100)
    assert owner.store.latest()[profile.target_id].phase is LifecycleSagaPhase.VERIFYING_HEALTH
    assert adapter.live


def test_cleanup_never_adopts_tree_when_launch_checkpoint_is_missing(tmp_path):
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    owner = _orchestrator(profile, adapter, clock)
    request = _request(profile, operation=Operation.START)
    state = owner._reserve(owner._intent(request, profile, LifecycleAction.START))
    owner._advance(state, LifecycleSagaPhase.STARTING_NEW)
    adapter.seed_tree()
    transaction = replace(MutationTransactionState.prepare(request, now_ms=clock.now_ms()),
        phase=MutationTransactionPhase.REPAIR_REQUIRED, revision=2,
        recovery_action=MutationRecoveryAction.REPAIR, failure_code="conflict")
    with pytest.raises(TransactionConflictError, match="custody"):
        owner.repair_start_cleanup(request, transaction, timeout_ms=100)
    assert adapter.terminations == 0


@pytest.mark.parametrize("boundary", ["persistent_marker", "birth_after_environ", "birth_after_argv"])
def test_identity_still_rejects_persistent_markers_and_pid_reuse(tmp_path, monkeypatch, boundary):
    profile = _profile(tmp_path)
    adapter = LinuxProcessAdapter()
    environment = profile.launch_environment(9)
    before = (1, 123, 123, 90)
    after = (1, 123, 123, 91)
    observations = iter([before, after] if boundary == "birth_after_environ" else
                        [before, before, after] if boundary == "birth_after_argv" else [before, before])
    if boundary == "persistent_marker":
        environment[CONFIGURATION_ROOT_ENV] = "configuration:foreign"
    monkeypatch.setattr(adapter, "_stat", lambda _pid: next(observations))
    monkeypatch.setattr(adapter, "_environ", lambda _pid: environment)
    monkeypatch.setattr(adapter, "_argv", lambda _pid: profile.argv)
    monkeypatch.setattr("os.readlink", lambda _path: str(tmp_path))
    with pytest.raises(ProcessIdentityMismatch):
        adapter._identity(123, profile)
