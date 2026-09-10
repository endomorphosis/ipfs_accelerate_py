"""Ancestry and dead-owner fences must not erase unknown callback evidence."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import ProcessBirthIdentity
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon
from test.api.test_agent_supervisor_reconciliation_auto_unblock import (
    _database_program,
    _git,
    _init_repo,
    _seed_completed_database_portal_attempt,
    _supervisor,
)


@pytest.fixture
def completed_workspace(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("baseline\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "baseline")
    supervisor = _supervisor(repo, worktree_root=repo / "worktrees", database_program=_database_program())
    paths, _, identity, task = _seed_completed_database_portal_attempt(
        supervisor, task_alias="PORTAL-060", task_cid="task:retention",
    )
    branch = f"rescue/worktree/implementation-portal-060-{identity.semantic_fingerprint[:12]}-attempt-1-123-abcdef012345"
    workspace = repo / "worktrees" / "old-attempt"
    _git(repo, "worktree", "add", "-b", branch, str(workspace), "main")
    reads = []
    current = [task]

    @contextmanager
    def source(endpoint, **kwargs):
        assert endpoint == supervisor.config.database_program.quack_endpoint
        assert kwargs["owner_scope"].startswith("worktree-reconciliation:")
        reads.append(endpoint)
        record = current[0]
        if isinstance(record, Exception):
            raise record
        yield SimpleNamespace(get_task=lambda alias: record if alias == "PORTAL-060" else None)

    monkeypatch.setattr(supervisor, "_canonical_database_task_source", source)
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [])
    return SimpleNamespace(repo=repo, supervisor=supervisor, paths=paths, task=task,
                           branch=branch, workspace=workspace, reads=reads, current=current)


def test_projection_peer_keeps_dead_owner_terminal_worktree(completed_workspace):
    case = completed_workspace
    state = case.repo / "peer-state"
    peer = TodoImplementationDaemon(
        todo_path=case.paths.task_projection, state_path=state / "state.json",
        strategy_path=state / "strategy.json", events_path=state / "events.jsonl",
        repo_root=case.repo, worktree_root=case.workspace.parent,
        isolate_merge_queue_to_task_projection=True, merged_worktree_cleanup_max=5,
    )
    record = peer.worktree_lifecycle.begin_preparing(
        task_id="PORTAL-060", canonical_task_cid="task:retention", attempt=1,
        lane_id="old-lane", state_dir=str(case.repo / "old-state"),
        workspace_path=case.workspace, branch=case.branch, merge_target="main",
        owner=ProcessBirthIdentity(pid=2**30, start_time_ticks=1, boot_id="dead"),
    )
    terminal = peer.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
        case.workspace, expected_state_dir=record.state_dir,
    )
    assert terminal.is_terminal
    assert terminal.terminal_reason == "controlled_restart_dead_owner"
    # Reproduce the exact false authority combination: clean merged branch and
    # terminal lifecycle, while the projection worker cannot read peer state.
    assert peer._authorize_worktree_cleanup(case.workspace, case.branch)["allowed"]
    before = _git(case.repo, "rev-parse", case.branch)
    assert _git(case.workspace, "status", "--porcelain") == ""
    result = peer._cleanup_already_merged_worktrees()
    assert result["removed_count"] == 0
    assert result["reason"] == "task_projection_has_no_peer_cleanup_authority"
    assert case.workspace.is_dir()
    assert _git(case.repo, "rev-parse", case.branch) == before
    assert peer.worktree_lifecycle.load_workspace(case.workspace) == terminal


@pytest.mark.parametrize("failure", ["quarantined", "missing", "receipt_missing", "owner_unavailable"])
def test_supervisor_preserves_unsettled_or_unverifiable_workspace(completed_workspace, failure):
    case = completed_workspace
    if failure == "missing":
        case.current[0] = None
    elif failure == "owner_unavailable":
        case.current[0] = ConnectionError("owner unavailable")
    else:
        case.current[0] = SimpleNamespace(**{
            **vars(case.task),
            "status": "quarantined" if failure == "quarantined" else "completed",
            "body": {"failure": {"kind": "provider_callback_outcome_unknown"}},
        })
    before = _git(case.repo, "rev-parse", case.branch)
    result = case.supervisor.cleanup_backlogged_worktrees()
    assert result["removed_count"] == 0
    assert result["skipped"][0]["reason"] == "canonical_completion_required_for_cleanup"
    assert case.workspace.is_dir()
    assert _git(case.repo, "rev-parse", case.branch) == before
    assert len(case.reads) == 1


def test_verified_canonical_rescue_cleanup_retains_exact_branch(completed_workspace):
    case = completed_workspace
    before = _git(case.repo, "rev-parse", case.branch)
    result = case.supervisor.cleanup_backlogged_worktrees()
    assert result["removed_count"] == 1
    assert len(case.reads) == 2
    assert not case.workspace.exists()
    assert _git(case.repo, "rev-parse", case.branch) == before
    removed = result["removed"][0]
    assert removed["completion_proof"]["verified"] is True
    assert removed["branch_preserved"] is True
    assert removed["head"] == before


def test_completion_loss_at_mutation_boundary_keeps_workspace(completed_workspace, monkeypatch):
    case = completed_workspace
    original = case.supervisor._revalidate_worktree_mutation_preimage

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        case.current[0] = ConnectionError("owner changed before removal")
        return result

    monkeypatch.setattr(case.supervisor, "_revalidate_worktree_mutation_preimage", change)
    result = case.supervisor.cleanup_backlogged_worktrees()
    assert result["removed_count"] == 0
    assert len(case.reads) == 2
    assert case.workspace.is_dir()
    assert result["skipped"][0]["reason"] == "canonical_completion_changed_before_cleanup"


def test_forged_projection_cannot_supply_cleanup_authority(completed_workspace):
    case = completed_workspace
    binding = case.paths.task_projection.parent / "database-attempt-binding.json"
    binding.write_text("{}")
    result = case.supervisor.cleanup_backlogged_worktrees()
    assert result["removed_count"] == 0
    assert case.workspace.is_dir()
    assert result["skipped"][0]["completion_proof"]["verified"] is False
