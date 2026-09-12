"""Retained candidates keep their exact index fence through all Git effects."""
from dataclasses import replace
import json
import select
import subprocess
import sys

import pytest

from test_agent_supervisor_implementation_protected_paths import (
    PortalImplementationDaemon,
    _protected_git_worktree_daemon,
    _task,
    _git,
    test_ephemeral_verification_lock_deferral_does_not_consume_attempt as _full_recovery,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnershipError,
    WorktreeLifecycleStore,
)


def _terminal(tmp_path):
    daemon, repo, workspace, _ = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    record = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id, canonical_task_cid=daemon._canonical_ref(task),
        attempt=3, lane_id="retained", workspace_path=workspace,
        branch=_git(workspace, "branch", "--show-current"), merge_target="main",
        state_dir=str(daemon.state_path.parent),
    )
    record = daemon.worktree_lifecycle.mark_terminal(
        workspace, lease_id=record.lease_id, expected_fence=record.fence,
        reason="verification_deferred_checkout_lease_unavailable",
    )
    return daemon, repo, workspace, task, record


def _records(store, record):
    return {
        "record": store.workspace_path_for(record.workspace_path).read_bytes(),
        "index": store.task_index_path_for(
            canonical_task_cid=record.canonical_task_cid,
            task_id=record.task_id, attempt=record.attempt,
        ).read_bytes(),
    }


def test_native_terminal_prepare_error_retains_exact_custody(tmp_path):
    daemon, _, _, _, record = _terminal(tmp_path)
    store = daemon.worktree_lifecycle
    before = _records(store, record)
    error = RuntimeError("original preservation failure")

    def fail():
        raise error

    with pytest.raises(RuntimeError) as raised:
        store.recover_exact_terminal_after_effect(
            record, reason="authorized", prepare=fail,
            cleanup=lambda *_: pytest.fail("cleanup ran after failed preservation"),
        )
    assert raised.value is error
    assert _records(store, record) == before


def test_native_terminal_cleanup_error_retains_authorized_fence(tmp_path):
    daemon, _, _, _, record = _terminal(tmp_path)
    store = daemon.worktree_lifecycle
    error = RuntimeError("original cleanup failure")
    observed = []

    def fail(prepared, authorized):
        assert prepared == "rescue-proved"
        assert authorized.fence == record.fence + 1
        assert store.load_workspace(record.workspace_path) == authorized
        observed.append(authorized)
        raise error

    with pytest.raises(RuntimeError) as raised:
        store.recover_exact_terminal_after_effect(
            record, reason="authorized", prepare=lambda: "rescue-proved", cleanup=fail,
        )
    assert raised.value is error
    assert store.load_workspace(record.workspace_path) == observed[0]
    assert store.load_task_attempt(
        canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id, attempt=record.attempt,
    ) == observed[0]


@pytest.mark.parametrize("drift", ["record", "index", "nonterminal"])
def test_native_terminal_capture_refuses_drift_before_prepare(tmp_path, drift):
    daemon, _, _, _, record = _terminal(tmp_path)
    store = daemon.worktree_lifecycle
    captured = record
    if drift == "record":
        store.mark_terminal(
            record.workspace_path, lease_id=record.lease_id,
            expected_fence=record.fence, reason="changed",
        )
    elif drift == "index":
        store.begin_preparing(
            task_id=record.task_id, canonical_task_cid=record.canonical_task_cid,
            attempt=record.attempt, lane_id="successor", workspace_path=tmp_path / "new",
            branch="new", merge_target="main",
        )
    else:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            WorkspaceLifecycleState,
        )
        captured = replace(record, state=WorkspaceLifecycleState.PREPARING)
    before = _records(store, record)
    with pytest.raises(OwnershipError):
        store.recover_exact_terminal_after_effect(
            captured, reason="authorized",
            prepare=lambda: pytest.fail("unowned prepare ran"),
            cleanup=lambda *_: pytest.fail("unowned cleanup ran"),
        )
    assert _records(store, record) == before


def test_native_retained_fingerprint_rechecked_under_effect_guard(tmp_path):
    daemon, repo, workspace, task, record = _terminal(tmp_path)
    baseline = _git(workspace, "rev-parse", "HEAD")
    output = workspace / "src" / "example.py"
    output.parent.mkdir(exist_ok=True)
    output.write_text("retained = True\n")
    fingerprint = daemon._retained_workspace_content_fingerprint(
        workspace, baseline_ref=baseline,
    )
    output.write_text("changed_after_inspection = True\n")
    before = (_git(repo, "show-ref"), _git(workspace, "rev-parse", "HEAD"),
              _records(daemon.worktree_lifecycle, record), output.read_bytes())
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        daemon._preserve_interrupted_worktree(
            workspace, record.branch, task, record.attempt,
            evidence={}, rescue_suffix="verification-deferred",
            event_type="verification_deferred_retained_candidate_preserved",
            evidence_field="retained_candidate_recovery", baseline_ref=baseline,
            retained_recovery_lifecycle=record,
            retained_recovery_fingerprint=fingerprint,
        )
    assert (_git(repo, "show-ref"), _git(workspace, "rev-parse", "HEAD"),
            _records(daemon.worktree_lifecycle, record), output.read_bytes()) == before


SUCCESSOR = """import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
v=json.loads(sys.argv[1]);store=WorktreeLifecycleStore(Path(v.pop('repo')))
print('ready',flush=True)
r=store.begin_preparing(**v)
print(json.dumps(r.to_dict()),flush=True)
"""


def test_actual_successor_excluded_through_native_commit_rescue_and_cleanup(tmp_path, monkeypatch):
    original_commit = PortalImplementationDaemon._commit_worktree_changes
    original_cleanup = PortalImplementationDaemon._cleanup_merged_worktree_unchecked
    children = []
    observed = []

    def committed(self, workspace, task, attempt, **kwargs):
        record = self.worktree_lifecycle.load_workspace(workspace)
        assert record.terminal_reason == "verification_deferred_checkout_lease_unavailable"
        child = subprocess.Popen(
            [sys.executable, "-P", "-c", SUCCESSOR, json.dumps({
                "repo": str(self.repo_root), "task_id": record.task_id,
                "canonical_task_cid": record.canonical_task_cid,
                "attempt": record.attempt, "lane_id": "actual-successor",
                "workspace_path": str(tmp_path / "successor-workspace"),
                "branch": "actual-successor", "merge_target": "main",
            })], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        children.append(child)
        assert select.select([child.stdout], [], [], 15)[0]
        assert child.stdout.readline().strip() == "ready"
        assert not select.select([child.stdout], [], [], 0.2)[0]
        assert child.poll() is None
        before = _records(self.worktree_lifecycle, record)
        result = original_commit(self, workspace, task, attempt, **kwargs)
        assert child.poll() is None
        assert _records(self.worktree_lifecycle, record) == before
        observed.append("commit")
        return result

    def cleaned(self, workspace, branch, **kwargs):
        assert children and children[0].poll() is None
        record = self.worktree_lifecycle.load_workspace(workspace)
        assert record.terminal_reason == "verification_deferred_candidate_recovery_authorized"
        assert self._git_ref_exists(self._interrupted_worktree_rescue_branch_name(
            branch, "verification-deferred",
        ))
        result = original_cleanup(self, workspace, branch, **kwargs)
        assert children[0].poll() is None
        observed.append("cleanup")
        return result

    monkeypatch.setattr(PortalImplementationDaemon, "_commit_worktree_changes", committed)
    monkeypatch.setattr(PortalImplementationDaemon, "_cleanup_merged_worktree_unchecked", cleaned)
    try:
        _full_recovery(tmp_path, monkeypatch, acquire_successor=False)
        assert observed == ["commit", "cleanup"] and len(children) == 1
        output, error = children[0].communicate(timeout=15)
        assert children[0].returncode == 0, error
        successor = json.loads(output)
        store = WorktreeLifecycleStore(successor["repo_root"])
        assert store.load_task_attempt(
            canonical_task_cid=successor["canonical_task_cid"],
            task_id=successor["task_id"], attempt=successor["attempt"],
        ).to_dict() == successor
    finally:
        # Only this test's exact disposable child; never a supervisor actor.
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=15)
