"""Real registry and process checks for captured dispatch/cleanup custody."""
from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnershipError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState
from test.api.test_agent_supervisor_runtime_authority_vectors import (
    _daemon, _diff_task, _git, _repository,
)


def _owned_workspace(tmp_path):
    repo = tmp_path / "repo"
    baseline = _repository(repo)
    daemon = _daemon(repo)
    workspace = tmp_path / "workspace"
    branch = "implementation/captured"
    _git(repo, "worktree", "add", "-b", branch, str(workspace), "HEAD")
    record = daemon.worktree_lifecycle.begin_preparing(
        task_id="CAPTURED", canonical_task_cid="cid:captured", attempt=1,
        lane_id="lane-a", workspace_path=workspace, branch=branch,
        merge_target="main",
    )
    daemon._active_worktree_lifecycle = record
    return daemon, workspace, record, baseline


def _replace_same_token(store, record):
    terminal = store.mark_terminal(
        record.workspace_path, lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    assert store.compare_and_delete(
        record.workspace_path, lease_id=terminal.lease_id,
        expected_fence=terminal.fence,
    )
    new = store.begin_preparing(
        task_id="REPLACEMENT", canonical_task_cid="cid:replacement", attempt=1,
        lane_id="lane-b", workspace_path=record.workspace_path,
        branch="implementation/replacement", merge_target="main",
        lease_id=record.lease_id,
    )
    assert new.fence == record.fence
    return new


def test_commit_revalidates_after_decision_callback_preparation(tmp_path, monkeypatch):
    daemon, workspace, record, _ = _owned_workspace(tmp_path)
    replacements = []
    effects = []

    def decision(_kind, _payload, effect):
        replacements.append(_replace_same_token(daemon.worktree_lifecycle, record))
        return effect()

    monkeypatch.setattr(daemon, "_decision_runtime_mutation", decision)
    monkeypatch.setattr(daemon, "_commit_worktree_changes_unchecked", lambda *_a, **_k: effects.append(True))
    with pytest.raises(OwnershipError, match="captured record"):
        daemon._commit_worktree_changes(workspace, _diff_task("src/alpha.py"), 1)
    assert effects == []
    assert daemon.worktree_lifecycle.load_workspace(workspace) == replacements[0]
    assert daemon._active_worktree_lifecycle == record


def test_cleanup_cannot_adopt_reused_lease_and_fence(tmp_path, monkeypatch):
    daemon, workspace, record, _ = _owned_workspace(tmp_path)
    replacement = _replace_same_token(daemon.worktree_lifecycle, record)
    effects = []
    monkeypatch.setattr(daemon, "_cleanup_merged_worktree_unchecked", lambda *_a, **_k: effects.append(True))
    result = daemon._cleanup_merged_worktree(workspace, record.branch, reusable=False)
    assert result["cleaned"] is False
    assert result["lifecycle_finalize"]["finalized"] is False
    assert effects == []
    assert workspace.is_dir()
    assert daemon.worktree_lifecycle.load_workspace(workspace) == replacement
    assert daemon._active_worktree_lifecycle == record


def test_failed_physical_cleanup_preserves_custody_for_normal_retry(tmp_path, monkeypatch):
    daemon, workspace, record, _ = _owned_workspace(tmp_path)
    store = daemon.worktree_lifecycle
    paths = [store.workspace_path_for(workspace), store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid, task_id=record.task_id, attempt=record.attempt,
    )]
    before = [p.read_bytes() for p in paths]
    cleanup = daemon._cleanup_worktree_submodules
    monkeypatch.setattr(daemon, "_cleanup_worktree_submodules", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("disposal incomplete")))
    failed = daemon._cleanup_merged_worktree(workspace, record.branch, reusable=False)
    assert failed["cleaned"] is False
    assert failed["lifecycle_finalize"]["finalized"] is False
    assert [p.read_bytes() for p in paths] == before
    assert daemon._active_worktree_lifecycle == record
    monkeypatch.setattr(daemon, "_cleanup_worktree_submodules", cleanup)
    succeeded = daemon._cleanup_merged_worktree(workspace, record.branch, reusable=False)
    assert succeeded["cleaned"] is True
    assert succeeded["lifecycle_finalize"]["finalized"] is True
    assert not workspace.exists()
    assert daemon._active_worktree_lifecycle is None
    assert all(not p.exists() for p in paths)


@pytest.mark.parametrize("boundary", ["commit", "validation", "cleanup"])
def test_captured_effects_require_exact_task_index(tmp_path, monkeypatch, boundary):
    daemon, workspace, record, baseline = _owned_workspace(tmp_path)
    store = daemon.worktree_lifecycle
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid, task_id=record.task_id, attempt=record.attempt,
    )
    index = json.loads(index_path.read_text())
    index["record_id"] = "cid:replacement"
    index_path.write_text(json.dumps(index))
    before = index_path.read_bytes()
    effects = []
    monkeypatch.setattr(daemon, "_commit_worktree_changes_unchecked", lambda *_a, **_k: effects.append(True))
    monkeypatch.setattr(daemon.validation_scheduler, "run", lambda *_a, **_k: effects.append(True))
    monkeypatch.setattr(daemon, "_cleanup_merged_worktree_unchecked", lambda *_a, **_k: effects.append(True))
    task = _diff_task("src/alpha.py")
    if boundary == "cleanup":
        assert daemon._cleanup_merged_worktree(workspace, record.branch)["cleaned"] is False
    else:
        with pytest.raises(OwnershipError, match="task index"):
            if boundary == "commit":
                daemon._commit_worktree_changes(workspace, task, 1)
            else:
                daemon._run_validation_commands(
                    workspace, task, tmp_path / "validation.log", state=PortalTaskState(),
                    lifecycle_record=record, baseline_ref=baseline,
                )
    assert effects == []
    assert index_path.read_bytes() == before
    assert store.load_workspace(workspace) == record


@pytest.mark.skipif(sys.platform != "linux", reason="actual Linux flock waiter observation")
@pytest.mark.parametrize("boundary", ["commit", "validation", "cleanup"])
def test_independent_process_cannot_replace_during_captured_effect(tmp_path, monkeypatch, boundary):
    daemon, workspace, record, baseline = _owned_workspace(tmp_path)
    store = daemon.worktree_lifecycle
    source = str(Path(__file__).resolve().parents[2])
    child_code = '''import json,sys
sys.path.insert(0,sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore, DuplicateAttemptError
store=WorktreeLifecycleStore(sys.argv[2],store_dir=sys.argv[3])
print("attempting",flush=True)
try:
 r=store.begin_preparing(task_id="NEXT",canonical_task_cid="cid:next",attempt=1,lane_id="lane-b",workspace_path=sys.argv[4],branch="implementation/next",merge_target="main")
except DuplicateAttemptError:
 print("owned",flush=True)
else:
 print(json.dumps(r.to_dict()),flush=True)
'''
    children = []

    def contend(*_args, **_kwargs):
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", child_code, source, str(daemon.repo_root), str(store.store_dir), str(workspace)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        children.append(child)
        assert select.select([child.stdout], [], [], 5)[0]
        assert child.stdout.readline().strip() == "attempting"
        deadline = time.monotonic() + 3
        waiter = ""
        while time.monotonic() < deadline and child.poll() is None:
            waiter = Path(f"/proc/{child.pid}/wchan").read_text().strip()
            if "lock" in waiter:
                break
            time.sleep(.01)
        assert "lock" in waiter
        assert child.poll() is None
        if boundary == "commit":
            return {"committed": False}
        if boundary == "validation":
            return {"attempted": True, "passed": True, "returncode": 0, "results": []}
        return []

    try:
        if boundary == "commit":
            monkeypatch.setattr(daemon, "_commit_worktree_changes_unchecked", contend)
            daemon._commit_worktree_changes(workspace, _diff_task("src/alpha.py"), 1)
        elif boundary == "validation":
            monkeypatch.setattr(daemon.validation_scheduler, "run", contend)
            daemon._run_validation_commands(
                workspace, _diff_task("src/alpha.py"), tmp_path / "validation.log",
                lifecycle_record=record, baseline_ref=baseline,
            )
        else:
            monkeypatch.setattr(daemon, "_cleanup_worktree_submodules", contend)
            result = daemon._cleanup_merged_worktree(workspace, record.branch, reusable=False)
            assert result["cleaned"] is True
        assert len(children) == 1
        stdout, stderr = children[0].communicate(timeout=5)
        assert children[0].returncode == 0, stderr
        if boundary == "cleanup":
            replacement = json.loads(stdout)
            assert replacement["record_id"] != record.record_id
            assert store.load_workspace(workspace).record_id == replacement["record_id"]
        else:
            assert stdout.strip() == "owned"
            assert store.load_workspace(workspace) == record
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize("legacy_projection", [False, True])
def test_current_renewed_record_dispatches_with_legacy_index_projection(
    tmp_path, monkeypatch, legacy_projection,
):
    daemon, workspace, record, _ = _owned_workspace(tmp_path)
    store = daemon.worktree_lifecycle
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid, task_id=record.task_id, attempt=record.attempt,
    )
    legacy_index = index_path.read_bytes()
    current = store.renew_lease(
        workspace, lease_id=record.lease_id, expected_fence=record.fence,
    )
    if legacy_projection:
        # Native predecessor releases wrote only the renewed workspace record.
        index_path.write_bytes(legacy_index)
    daemon._active_worktree_lifecycle = current
    calls = []
    monkeypatch.setattr(daemon, "_commit_worktree_changes_unchecked", lambda *_a, **_k: calls.append(True))
    daemon._commit_worktree_changes(workspace, _diff_task("src/alpha.py"), 1)
    assert calls == [True]
    assert store.load_workspace(workspace) == current
    with pytest.raises(OwnershipError):
        store.run_exact_owner_effect(record, effect=lambda: pytest.fail("stale token dispatched"))


@pytest.mark.parametrize("operation", ["transition", "delete", "rebind"])
def test_old_workspace_writer_cannot_change_new_task_index(tmp_path, operation):
    daemon, workspace, record, _ = _owned_workspace(tmp_path)
    store = daemon.worktree_lifecycle
    old = store.mark_terminal(workspace, lease_id=record.lease_id, expected_fence=record.fence)
    current = store.begin_preparing(
        task_id=record.task_id, canonical_task_cid=record.canonical_task_cid,
        attempt=record.attempt, lane_id="lane-b", workspace_path=tmp_path / "replacement",
        branch="implementation/replacement", merge_target="main",
    )
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid, task_id=record.task_id, attempt=record.attempt,
    )
    before = index_path.read_bytes()
    with pytest.raises(OwnershipError, match="task index"):
        if operation == "transition":
            store.mark_terminal(workspace, lease_id=old.lease_id, expected_fence=old.fence)
        elif operation == "delete":
            store.compare_and_delete(workspace, lease_id=old.lease_id, expected_fence=old.fence)
        else:
            store.rebind_workspace(workspace, tmp_path / "third", lease_id=old.lease_id, expected_fence=old.fence)
    assert index_path.read_bytes() == before
    assert store.load_workspace(current.workspace_path) == current
    assert store.load_workspace(workspace) == old


@pytest.mark.parametrize("reusable", [False, True])
def test_actual_pool_cleanup_closes_captured_record_before_next_lease(tmp_path, reusable):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import WorktreePool

    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    pool = WorktreePool(
        repo_root=repo, worktree_root=daemon.worktree_root,
        reuse_authorizer=daemon._authorize_pooled_worktree_reuse,
    )
    daemon.worktree_pool = pool
    workspace = daemon.worktree_root / "captured"
    store = daemon.worktree_lifecycle
    record = store.begin_preparing(
        task_id="POOL-A", canonical_task_cid="cid:pool-a", attempt=1,
        lane_id="lane-a", workspace_path=workspace,
        branch="implementation/pool-a", merge_target="main",
    )
    daemon._active_worktree_lifecycle = record
    lease = pool.acquire(cache_key="capture", branch_name=record.branch, worktree_path=workspace)
    assert lease.path.resolve() == workspace.resolve()
    daemon._worktree_pool_leases[workspace.resolve()] = lease
    result = daemon._cleanup_merged_worktree(workspace, record.branch, reusable=reusable)
    assert result["cleaned"] is True
    assert result["pool_release"]["released"] is True
    assert result["lifecycle_finalize"]["finalized"] is True
    assert store.load_workspace(workspace) is None
    assert daemon._active_worktree_lifecycle is None
    assert workspace.exists() is reusable
    successor = store.begin_preparing(
        task_id="POOL-B", canonical_task_cid="cid:pool-b", attempt=1,
        lane_id="lane-b", workspace_path=workspace,
        branch="implementation/pool-b", merge_target="main",
    )
    daemon._active_worktree_lifecycle = successor
    second = pool.acquire(cache_key="capture", branch_name=successor.branch, worktree_path=workspace)
    assert second.reused is reusable
    daemon._worktree_pool_leases[workspace.resolve()] = second
    assert daemon._cleanup_merged_worktree(workspace, successor.branch, reusable=False)["cleaned"] is True
