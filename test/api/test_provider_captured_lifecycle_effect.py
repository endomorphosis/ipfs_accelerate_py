"""Provider effects retain exact custody after the native sealed handoff check."""
from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from test.api.test_portal_cleanup_uncertainty import _daemon, _install_fixture_gate


def test_sealed_provider_refuses_takeover_after_native_last_check(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, ephemeral=True)
    _install_fixture_gate(daemon, monkeypatch)
    check = daemon._assert_residual_provider_lifecycle_current
    checks = []
    replacements = []
    providers = []

    def checked_then_replaced(**kwargs):
        record = check(**kwargs)
        checks.append(record)
        if len(checks) == 2:
            terminal = daemon.worktree_lifecycle.mark_terminal(
                record.workspace_path, lease_id=record.lease_id,
                expected_fence=record.fence,
            )
            assert daemon.worktree_lifecycle.compare_and_delete(
                record.workspace_path, lease_id=terminal.lease_id,
                expected_fence=terminal.fence,
            )
            replacement = daemon.worktree_lifecycle.begin_preparing(
                task_id="REPLACEMENT", canonical_task_cid="cid:replacement",
                attempt=1, lane_id="peer", workspace_path=record.workspace_path,
                branch="implementation/replacement", merge_target=record.merge_target,
                lease_id=record.lease_id,
            )
            replacement = daemon.worktree_lifecycle.mark_active(
                replacement.workspace_path, lease_id=replacement.lease_id,
                expected_fence=replacement.fence,
            )
            assert replacement.fence == record.fence
            replacements.append(replacement)
        return record

    monkeypatch.setattr(daemon, "_assert_residual_provider_lifecycle_current", checked_then_replaced)
    monkeypatch.setattr(module, "run_process_group_stream", lambda *a, **k: providers.append(True))
    result = daemon.run_once()["implementation_result"]
    assert len(checks) == 2 and len(replacements) == 1
    assert providers == []
    assert result["provider_dispatched"] is False
    assert result["exception_result"]["exception_type"] == "OwnershipError"
    assert daemon.worktree_lifecycle.load_workspace(replacements[0].workspace_path) == replacements[0]


@pytest.mark.skipif(sys.platform != "linux", reason="kernel flock waiter observation")
def test_sealed_provider_retains_guard_until_runner_returns(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, ephemeral=True)
    _install_fixture_gate(daemon, monkeypatch)
    children = []
    source = str(Path(__file__).resolve().parents[2])
    child_code = '''import json,sys
sys.path.insert(0,sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore, WorkspaceLifecycleRecord, WorktreeLifecycleError
store=WorktreeLifecycleStore(sys.argv[2],store_dir=sys.argv[3])
record=WorkspaceLifecycleRecord.from_dict(json.loads(sys.argv[4]))
print("attempting",flush=True)
try:
 store.run_exact_owner_effect(record,effect=lambda: print("acquired",flush=True))
except WorktreeLifecycleError:
 print("changed",flush=True)
'''

    def runner(*args, **kwargs):
        record = daemon._active_worktree_lifecycle
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", child_code, source, str(daemon.repo_root),
             str(daemon.worktree_lifecycle.store_dir), json.dumps(record.to_dict())],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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
        assert "lock" in waiter and child.poll() is None
        raise subprocess.TimeoutExpired("fixture-provider", 1)

    monkeypatch.setattr(module, "run_process_group_stream", runner)
    try:
        daemon.run_once()
        assert len(children) == 1
        out, err = children[0].communicate(timeout=5)
        assert children[0].returncode == 0, err
        assert out.strip() in {"acquired", "changed"}
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


def test_no_change_cannot_complete_after_failed_exact_finalization(tmp_path, monkeypatch):
    from test.api.test_agent_supervisor_runtime_authority_vectors import (
        _daemon as make_daemon, _diff_task, _repository,
    )
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = make_daemon(repo)
    _install_fixture_gate(daemon, monkeypatch)
    task = _diff_task("src/alpha.py")
    task.metadata["no_change_completion"] = "allowed"
    cleanup = daemon._cleanup_merged_worktree
    advanced = []
    monkeypatch.setattr(daemon, "_task_uses_typed_local_execution", lambda task: False)
    monkeypatch.setattr(daemon, "_build_implementation_command", lambda *a, **k: [sys.executable, "-c", "pass"])

    def advance_then_cleanup(*args, **kwargs):
        if not advanced:
            captured = daemon._active_worktree_lifecycle
            advanced.append(daemon.worktree_lifecycle.renew_lease(
                captured.workspace_path, lease_id=captured.lease_id,
                expected_fence=captured.fence,
            ))
        return cleanup(*args, **kwargs)

    monkeypatch.setattr(daemon, "_cleanup_merged_worktree", advance_then_cleanup)
    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task, state=module.PortalTaskState(), attempt=1,
        started_at="2026-09-12T00:00:00+00:00", log_path=tmp_path / "nochange.log",
        prompt="fixture no change",
    )
    assert advanced
    assert result["commit_result"]["no_change_guard"]["allowed"] is True
    assert result["cleanup_result"]["lifecycle_finalize"]["finalized"] is False
    assert result["returncode"] == 1
    assert result["validation_result"]["reason"] == "no_change_lifecycle_finalization_failed"
    assert result["board_completion"]["complete"] is False
    assert "todo_update_result" not in result
