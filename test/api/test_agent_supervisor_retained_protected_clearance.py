"""Exercise real crash reconciliation after a native clearance receipt was saved."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)


PROTECTED = "docs/protected-policy.json"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _incident(tmp_path: Path, *, retain_approval: bool = True):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Operator Fixture")
    _git(repo, "config", "user.email", "operator@example.invalid")
    policy = repo / PROTECTED
    policy.parent.mkdir()
    policy.write_text('{"version": 1}\n')
    _git(repo, "add", PROTECTED)
    _git(repo, "commit", "-m", "initial policy")
    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    workspace = worktree_root / "lane"
    _git(repo, "worktree", "add", "-b", "lane", str(workspace), "HEAD")
    state = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=state / "task-state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=(PROTECTED,),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    task = PortalTask(
        task_id="REPLAY-001", title="Retained clearance", status="ready",
        completion="manual", priority="P1", track="quality", outputs=[],
        metadata={},
    )
    before = daemon._require_implementation_protected_snapshot(
        task=task, attempt=1, workspace_path=workspace
    )
    policy.write_text('{"version": 2}\n')
    violation = daemon._implementation_protected_path_violation(
        task=task, attempt=1, workspace_path=workspace, before=before
    )
    assert violation["reason"] == "implementation_protected_path_mutated"
    _git(repo, "add", PROTECTED)
    _git(repo, "commit", "-m", "reviewed operator policy update")
    commit = _git(repo, "rev-parse", "HEAD")
    incident_path = daemon._implementation_protected_incident_path()
    active_path = daemon._implementation_protected_active_snapshot_path()
    incident_bytes, active_bytes = incident_path.read_bytes(), active_path.read_bytes()
    receipt = None
    if retain_approval:
        cleared = daemon.clear_implementation_protected_path_incident(
            approved_commits=[commit], operator_note="Reviewed exact operator change"
        )
        assert cleared["cleared"] is True
        receipt = Path(cleared["receipt_path"])
        # Production services create controller receipts with UMask=0077.
        receipt.chmod(0o600)
        # Model a crash after the durable approval receipt, before fence removal.
        incident_path.write_bytes(incident_bytes)
        active_path.write_bytes(active_bytes)
    return daemon, repo, workspace, policy, receipt


def _assert_fenced(daemon: PortalImplementationDaemon) -> None:
    result = daemon._reconcile_implementation_protected_path_fence()
    assert result["blocked"] is True
    assert result["reason"] == "implementation_protected_path_incident_latched"
    assert daemon._implementation_protected_incident_path().exists()
    assert daemon._implementation_protected_active_snapshot_path().exists()


def test_real_reconcile_replays_exact_retained_operator_clearance(tmp_path: Path):
    daemon, repo, workspace, policy, retained_receipt = _incident(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    workspace_before = (workspace / PROTECTED).read_bytes()

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["cleared"] is True
    assert result["auto"] is True
    assert result["blocked"] is False
    assert result["reason"] == "retained_operator_clearance_replayed"
    assert result["retained_receipt_path"] == str(retained_receipt)
    assert not daemon._implementation_protected_incident_path().exists()
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    assert _git(repo, "rev-parse", "HEAD") == head
    assert (workspace / PROTECTED).read_bytes() == workspace_before
    assert policy.read_text() == '{"version": 2}\n'


def test_real_reconcile_does_not_trust_clean_unapproved_operator_head(tmp_path: Path):
    daemon, *_ = _incident(tmp_path, retain_approval=False)
    _assert_fenced(daemon)


@pytest.mark.parametrize("staged", [False, True])
def test_real_reconcile_retained_approval_denies_dirty_shared_paths(tmp_path: Path, staged: bool):
    daemon, repo, _, policy, _ = _incident(tmp_path)
    policy.write_text('{"unreviewed": true}\n')
    if staged:
        _git(repo, "add", PROTECTED)
    _assert_fenced(daemon)


def test_real_reconcile_retained_approval_never_admits_workspace_mutation(tmp_path: Path):
    daemon, _, workspace, _, _ = _incident(tmp_path)
    (workspace / PROTECTED).write_text('{"provider_mutation": true}\n')
    _assert_fenced(daemon)


def test_real_reconcile_retained_approval_denies_live_worker(tmp_path: Path, monkeypatch):
    daemon, _, workspace, _, _ = _incident(tmp_path)
    monkeypatch.setattr(
        daemon, "_list_process_commands",
        lambda: [f"python -m ipfs_accelerate_py.agent_supervisor.grok_cli_runner --workspace {workspace}"],
    )
    _assert_fenced(daemon)


@pytest.mark.parametrize("mutation", ["incident", "receipt", "workspace", "new_head", "writable_receipt", "foreign_lock"])
def test_real_reconcile_does_not_reuse_foreign_or_changed_approval(tmp_path: Path, mutation: str):
    daemon, repo, _, _, receipt_path = _incident(tmp_path)
    assert receipt_path is not None
    if mutation == "incident":
        incident_path = daemon._implementation_protected_incident_path()
        incident = json.loads(incident_path.read_text())
        incident["latched_at"] = "2099-01-01T00:00:00Z"
        incident_path.write_text(json.dumps(incident))
    elif mutation == "receipt":
        receipt = json.loads(receipt_path.read_text())
        receipt["approved_commits"] = []
        receipt_path.write_text(json.dumps(receipt))
    elif mutation == "workspace":
        active_path = daemon._implementation_protected_active_snapshot_path()
        active = json.loads(active_path.read_text())
        active["workspace_path"] = str(tmp_path / "foreign")
        active_path.write_text(json.dumps(active))
    elif mutation == "writable_receipt":
        receipt_path.chmod(0o666)
    elif mutation == "foreign_lock":
        daemon._implementation_lock_path().write_text(json.dumps({
            "kind": "implementation", "state_dir": str(tmp_path / "foreign"),
            "repo_root": str(repo), "pid": 999999999,
        }))
    else:
        (repo / "unrelated.txt").write_text("different operator state\n")
        _git(repo, "add", "unrelated.txt")
        _git(repo, "commit", "-m", "later operator commit")
    _assert_fenced(daemon)
