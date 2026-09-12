"""Native lifecycle recovery keeps its first-pass generation and evidence."""

import json
import subprocess
import sys

import pytest
from test.api.test_agent_supervisor_implementation_protected_paths import (
    ProcessBirthIdentity,
    _persist_active_attempt_state,
    _persist_stale_implementation_lock,
    _protected_git_worktree_daemon,
    _seed_active_lifecycle,
    _task,
)

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnershipError,
)

ADOPT = """import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore,WorktreeLifecycleError
v=json.loads(sys.argv[1]);s=WorktreeLifecycleStore(Path(v['repo_root']))
if len(sys.argv)>2: print('ready',flush=True)
try:
 r=s.adopt_dead_owner(v['workspace_path'],expected_record_id=v['record_id'],expected_fence=v['fence'],expected_lease_id=v['lease_id'],expected_task_id=v['task_id'],expected_canonical_task_cid=v['canonical_task_cid'],expected_attempt=v['attempt'],expected_branch=v['branch'],expected_merge_target=v['merge_target'],expected_repo_root=v['repo_root'],expected_state_dir=v['state_dir'],lane_id='actual-successor')
 print(json.dumps({'record':r.to_dict()}),flush=True)
except WorktreeLifecycleError as exc:
 print(json.dumps({'refused':str(exc)}),flush=True)
"""


def seeded(tmp_path):
    daemon, _repo, workspace, _ = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    record = _seed_active_lifecycle(
        daemon,
        task,
        workspace,
        ProcessBirthIdentity(pid=2**30 - 7, start_time_ticks=1, boot_id="dead-owner"),
    )
    daemon._require_implementation_protected_snapshot(
        task=task, attempt=1, workspace_path=workspace
    )
    _persist_active_attempt_state(daemon, task=task, workspace=workspace)
    lock = _persist_stale_implementation_lock(daemon, task)
    paths = [
        daemon._implementation_protected_active_snapshot_path(),
        daemon.state_path,
        lock,
    ]
    return daemon, workspace, record, {path: path.read_bytes() for path in paths}


def preserve(before):
    assert all(path.read_bytes() == data for path, data in before.items())


def adopt(record):
    child = subprocess.run(
        [sys.executable, "-c", ADOPT, json.dumps(record.to_dict())],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(child.stdout)["record"]


def after_preflight(daemon, monkeypatch, action):
    original = daemon._reconcile_quiesced_worktree_lifecycle
    ran = False

    def wrapped(*args, **kwargs):
        nonlocal ran
        result = original(*args, **kwargs)
        if not kwargs["terminalize"] and not ran:
            ran = True
            action()
        return result

    monkeypatch.setattr(daemon, "_reconcile_quiesced_worktree_lifecycle", wrapped)


def test_native_successor_between_passes_is_never_adopted(tmp_path, monkeypatch):
    daemon, workspace, original, before = seeded(tmp_path)
    successor = {}
    after_preflight(daemon, monkeypatch, lambda: successor.update(adopt(original)))
    result = daemon.reconcile_quiesced_active_attempt()
    assert result["blocked"] and not result["reconciled"]
    assert successor["fence"] > original.fence
    assert successor["owner"]["pid"] != original.owner.pid
    assert daemon.worktree_lifecycle.load_workspace(workspace).to_dict() == successor
    preserve(before)


@pytest.mark.parametrize("drift", ["owner", "phase", "lease", "delete", "index"])
def test_phase_drift_preserves_original_evidence(tmp_path, monkeypatch, drift):
    daemon, workspace, original, before = seeded(tmp_path)
    record_path = daemon.worktree_lifecycle.workspace_path_for(workspace)
    index_path = daemon.worktree_lifecycle.task_index_path_for(
        canonical_task_cid=original.canonical_task_cid,
        task_id=original.task_id,
        attempt=original.attempt,
    )
    changed = {}

    def change():
        if drift == "delete":
            record_path.unlink()
            index_path.unlink()
        else:
            target = index_path if drift == "index" else record_path
            payload = json.loads(target.read_bytes())
            if drift == "owner":
                payload["owner"] = ProcessBirthIdentity(
                    pid=2**30 - 9, start_time_ticks=2, boot_id="replacement"
                ).to_dict()
            elif drift == "index":
                payload["task_id"] = "different-task"
            elif drift == "phase":
                payload["state"] = "quarantined"
            else:
                payload["lease_id"] = "replacement-lease"
            target.write_text(json.dumps(payload))
            changed[target] = target.read_bytes()

    after_preflight(daemon, monkeypatch, change)
    result = daemon.reconcile_quiesced_active_attempt()
    assert result["blocked"] and not result["reconciled"]
    preserve(before)
    preserve(changed)
    if drift == "delete":
        assert not record_path.exists() and not index_path.exists()


def test_native_adoption_is_excluded_through_protected_effect(tmp_path, monkeypatch):
    daemon, workspace, original, _ = seeded(tmp_path)
    child = None
    protected = daemon._reconcile_implementation_protected_path_fence

    def overlap():
        nonlocal child
        child = subprocess.Popen(
            [sys.executable, "-c", ADOPT, json.dumps(original.to_dict()), "overlap"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=0.25)
        assert daemon.worktree_lifecycle.load_workspace(workspace) == original
        return protected()

    monkeypatch.setattr(
        daemon, "_reconcile_implementation_protected_path_fence", overlap
    )
    try:
        result = daemon.reconcile_quiesced_active_attempt()
        assert result["reconciled"] and not result["blocked"]
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        assert "refused" in json.loads(output)
        terminal = daemon.worktree_lifecycle.load_workspace(workspace)
        assert terminal.is_terminal and terminal.fence == original.fence + 1
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.parametrize("exception_type", [RuntimeError, OSError, OwnershipError])
def test_protected_callback_error_is_preserved_without_cleanup(
    tmp_path, monkeypatch, exception_type
):
    daemon, workspace, original, before = seeded(tmp_path)
    failure = exception_type("original protected recovery error")

    def fail():
        raise failure

    monkeypatch.setattr(daemon, "_reconcile_implementation_protected_path_fence", fail)
    with pytest.raises(exception_type) as caught:
        daemon.reconcile_quiesced_active_attempt()
    assert caught.value is failure
    assert daemon.worktree_lifecycle.load_workspace(workspace) == original
    preserve(before)


def test_protected_refusal_keeps_original_lifecycle(tmp_path, monkeypatch):
    daemon, workspace, original, before = seeded(tmp_path)
    refusal = {"blocked": True, "reason": "original protected refusal"}
    monkeypatch.setattr(
        daemon, "_reconcile_implementation_protected_path_fence", lambda: refusal
    )
    result = daemon.reconcile_quiesced_active_attempt()
    assert result["blocked"] and result["protected_path_reconciliation"] == refusal
    assert daemon.worktree_lifecycle.load_workspace(workspace) == original
    preserve(before)


def test_controlled_restart_cas_refuses_a_replaced_captured_record(tmp_path):
    daemon, workspace, original, before = seeded(tmp_path)
    successor = adopt(original)
    assert (
        daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
            workspace, expected_state_dir=original.state_dir, expected_record=original
        )
        is None
    )
    assert daemon.worktree_lifecycle.load_workspace(workspace).to_dict() == successor
    preserve(before)


@pytest.mark.parametrize("already_terminal", [False, True])
def test_success_retains_native_terminal_record_and_clears_exact_evidence(
    tmp_path, already_terminal
):
    daemon, workspace, original, _before = seeded(tmp_path)
    if already_terminal:
        original = daemon.worktree_lifecycle.mark_terminal(
            workspace,
            lease_id=original.lease_id,
            expected_fence=original.fence,
            reason="interrupted_terminal_cleanup",
        )
    result = daemon.reconcile_quiesced_active_attempt()
    assert result["reconciled"] and not result["blocked"]
    terminal = daemon.worktree_lifecycle.load_workspace(workspace)
    assert terminal.is_terminal
    assert terminal.record_id == original.record_id
    assert terminal.fence == original.fence + (0 if already_terminal else 1)
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    assert not daemon._implementation_lock_path().exists()
    state = json.loads(daemon.state_path.read_bytes())
    assert not state["implementation_in_progress"] and not state["active_task_id"]


@pytest.mark.parametrize("appears_after_preflight", [False, True])
def test_legacy_absence_is_pinned_during_effect(
    tmp_path, monkeypatch, appears_after_preflight
):
    daemon, workspace, original, before = seeded(tmp_path)
    record_path = daemon.worktree_lifecycle.workspace_path_for(workspace)
    index_path = daemon.worktree_lifecycle.task_index_path_for(
        canonical_task_cid=original.canonical_task_cid,
        task_id=original.task_id,
        attempt=original.attempt,
    )
    record_path.unlink()
    index_path.unlink()
    if appears_after_preflight:

        def claim():
            _seed_active_lifecycle(
                daemon, _task(outputs=["src/example.py"]), workspace, original.owner
            )

        after_preflight(daemon, monkeypatch, claim)
    result = daemon.reconcile_quiesced_active_attempt()
    assert result["blocked"] is appears_after_preflight
    assert result["reconciled"] is not appears_after_preflight
    if appears_after_preflight:
        preserve(before)
        assert not daemon.worktree_lifecycle.load_workspace(workspace).is_terminal
    else:
        assert not record_path.exists()


def test_legacy_index_lease_remains_a_routing_hint(tmp_path):
    daemon, workspace, original, _ = seeded(tmp_path)
    index_path = daemon.worktree_lifecycle.task_index_path_for(
        canonical_task_cid=original.canonical_task_cid,
        task_id=original.task_id,
        attempt=original.attempt,
    )
    payload = json.loads(index_path.read_bytes())
    payload["lease_id"] = "older-heartbeat-routing-hint"
    index_path.write_text(json.dumps(payload))
    result = daemon.reconcile_quiesced_active_attempt()
    assert result["reconciled"] and not result["blocked"]
    terminal = daemon.worktree_lifecycle.load_workspace(workspace)
    assert terminal.is_terminal and terminal.fence == original.fence + 1

LEGACY_WRITE = """import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore,WorktreeLifecycleError
v=json.loads(sys.argv[1]);mode=sys.argv[2];s=WorktreeLifecycleStore(Path(v['repo_root']))
print('ready',flush=True)
try:
 if mode=='transition': r=s.mark_settling(v['workspace_path'],lease_id=v['lease_id'],expected_fence=v['fence'])
 elif mode=='renew': r=s.renew_lease(v['workspace_path'],lease_id=v['lease_id'],expected_fence=v['fence'])
 elif mode=='rebind': r=s.rebind_workspace(v['workspace_path'],v['workspace_path']+'-moved',lease_id=v['lease_id'],expected_fence=v['fence'])
 elif mode=='reclaim': r=s.reclaim_stale(v['workspace_path'],now=v['expires_at']+1)
 elif mode=='restart': r=s.reclaim_dead_owner_for_controlled_restart(v['workspace_path'],expected_state_dir=v['state_dir'])
 elif mode=='delete': r=s.compare_and_delete(v['workspace_path'],expected_fence=v['fence'],lease_id=v['lease_id'])
 else: raise AssertionError(mode)
 print(json.dumps({'result':r.to_dict() if hasattr(r,'to_dict') else r}),flush=True)
except WorktreeLifecycleError as exc:
 print(json.dumps({'refused':str(exc)}),flush=True)
"""


@pytest.mark.parametrize('mode', ['transition', 'renew', 'rebind', 'reclaim', 'restart', 'delete'])
def test_every_legacy_writer_takes_index_before_workspace(tmp_path, mode):
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import serialized_lock_update

    daemon, workspace, original, before = seeded(tmp_path)
    store = daemon.worktree_lifecycle
    index = store.task_index_path_for(canonical_task_cid=original.canonical_task_cid,
                                     task_id=original.task_id, attempt=original.attempt)
    record_path = store.workspace_path_for(workspace)
    child = None
    try:
        with serialized_lock_update(index):
            child = subprocess.Popen([sys.executable, '-c', LEGACY_WRITE,
                                      json.dumps(original.to_dict()), mode],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert child.stdout.readline().strip() == 'ready'
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=0.25)
            # The waiting child must not hold the workspace while waiting for
            # its index; native begin/adopt acquire in the opposite order.
            with serialized_lock_update(record_path, timeout_seconds=0.25):
                assert store.load_workspace(workspace) == original
                preserve(before)
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        result = json.loads(output)
        assert 'result' in result and 'refused' not in result
        if mode == 'delete':
            assert result['result'] is True
            assert store.load_workspace(workspace) is None
        else:
            assert result['result']['fence'] == original.fence + 1
            if mode == 'rebind':
                assert store.load_workspace(workspace) is None
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.parametrize('mode', ['transition', 'renew', 'rebind', 'reclaim', 'restart', 'delete'])
def test_legacy_writer_refuses_changed_index_without_repair(tmp_path, mode):
    daemon, workspace, original, before = seeded(tmp_path)
    store = daemon.worktree_lifecycle
    index = store.task_index_path_for(canonical_task_cid=original.canonical_task_cid,
                                     task_id=original.task_id, attempt=original.attempt)
    data = json.loads(index.read_bytes())
    data['task_id'] = 'different-task'
    index.write_text(json.dumps(data))
    before[index] = index.read_bytes()
    before[store.workspace_path_for(workspace)] = store.workspace_path_for(workspace).read_bytes()
    child = subprocess.run([sys.executable, '-c', LEGACY_WRITE,
                            json.dumps(original.to_dict()), mode],
                           capture_output=True, text=True, timeout=20, check=True)
    assert 'refused' in json.loads(child.stdout.splitlines()[-1])
    preserve(before)
    assert store.load_workspace(workspace) == original


def test_native_heartbeat_keeps_legacy_routing_index_compatible(tmp_path):
    daemon, workspace, original, _ = seeded(tmp_path)
    store = daemon.worktree_lifecycle
    index = store.task_index_path_for(canonical_task_cid=original.canonical_task_cid,
                                     task_id=original.task_id, attempt=original.attempt)
    original_index = index.read_bytes()
    renewed = store.renew_lease(workspace, lease_id=original.lease_id,
                               expected_fence=original.fence)
    assert renewed.fence == original.fence + 1
    assert index.read_bytes() == original_index
    result = daemon.reconcile_quiesced_active_attempt()
    assert result['reconciled'] and not result['blocked']
    assert store.load_workspace(workspace).fence == renewed.fence + 1
