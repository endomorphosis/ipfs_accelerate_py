"""Older runtimes must retain callbacks when canonical cleanup proof is absent."""
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import ProcessBirthIdentity
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import DatabaseProgramConfig
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon
from test.api.test_agent_supervisor_reconciliation_auto_unblock import _git, _init_repo, _supervisor


@pytest.fixture
def workspace(tmp_path):
    repo = _init_repo(tmp_path / 'repo')
    (repo / 'README.md').write_text('baseline\n')
    _git(repo, 'add', 'README.md')
    _git(repo, 'commit', '-m', 'baseline')
    branch = 'implementation-portal-060-unknown-callback'
    path = repo / 'worktrees' / 'unknown-callback'
    _git(repo, 'worktree', 'add', '-b', branch, str(path), 'main')
    return repo, path, branch


def test_projection_retains_terminal_dead_owner_peer(workspace):
    repo, path, branch = workspace
    state = repo / 'state'
    todo = repo / 'todo.md'
    todo.write_text('# Todos\n')
    daemon = TodoImplementationDaemon(
        todo_path=todo, state_path=state / 'state.json',
        strategy_path=state / 'strategy.json', events_path=state / 'events.jsonl',
        repo_root=repo, worktree_root=path.parent,
        merged_worktree_cleanup_max=5,
    )
    record = daemon.worktree_lifecycle.begin_preparing(
        task_id='PORTAL-060', canonical_task_cid='task:unknown', attempt=1,
        lane_id='old', state_dir=str(repo / 'old-state'), workspace_path=path,
        branch=branch, merge_target='main',
        owner=ProcessBirthIdentity(pid=2**30, start_time_ticks=1, boot_id='dead'),
    )
    terminal = daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
        path, expected_state_dir=record.state_dir,
    )
    assert terminal.is_terminal
    assert daemon._authorize_worktree_cleanup(path, branch)['allowed']
    head = _git(repo, 'rev-parse', branch)
    result = daemon._cleanup_already_merged_worktrees()
    assert result['reason'] == 'canonical_peer_cleanup_api_unavailable'
    assert result['removed_count'] == 0
    assert path.is_dir()
    assert _git(repo, 'rev-parse', branch) == head
    assert daemon.worktree_lifecycle.load_workspace(path) == terminal


@pytest.mark.parametrize('missing_source', [False, True])
def test_database_background_cleanup_requires_native_proof_api(workspace, monkeypatch, missing_source):
    repo, path, branch = workspace
    supervisor = _supervisor(repo, worktree_root=path.parent)
    supervisor.config = replace(supervisor.config, database_program=DatabaseProgramConfig(authority_mode="quack", task_source_kind="duckdb", endpoint_secret_handle="env://QUACK_TOKEN", quack_endpoint="quack:127.0.0.1:45123", store_id="control.duckdb", store_generation="generation-1", schema_revision="schema-v1"))
    if missing_source:
        import shutil
        shutil.rmtree(path)
    head = _git(repo, 'rev-parse', branch)
    def forbidden(*args, **kwargs):
        pytest.fail('cleanup without canonical proof attempted Git mutation')
    # This native revision has no migration-ref cleaner. Guard every Git
    # mutation, including prune, at the subprocess boundary it actually uses.
    monkeypatch.setattr('ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.subprocess.run', forbidden)
    result = supervisor._cleanup_backlogged_worktrees_locked()
    assert result['reason'] == 'canonical_completion_cleanup_api_unavailable'
    assert result['removed_count'] == 0
    monkeypatch.undo()
    assert _git(repo, 'rev-parse', branch) == head
    assert path.exists() is not missing_source
