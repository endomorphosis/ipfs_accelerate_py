"""Real protected completion commits must retain their Native producer identity."""
from dataclasses import asdict
import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import checkout_mutation_lock_path


def git(root, *args, env=None):
    result = subprocess.run(['git', *args], cwd=root, env=env, text=True,
        capture_output=True, check=False)
    assert result.returncode == 0, (args, result.stderr)
    return result.stdout.strip()


def seed(root):
    root.mkdir()
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    board = root / 'tasks.todo.md'
    board.write_text('# Tasks\n\n## EX-001 Actual completion\n\n- Status: todo\n'
        '- Completion: validation\n- Outputs: effect.py\n- Validation: python -c "pass"\n'
        '- Acceptance: Persist the actual completed task.\n')
    (root / 'effect.py').write_text('EFFECT = "owned synthetic result"\n')
    git(root, 'add', 'tasks.todo.md', 'effect.py')
    git(root, 'commit', '-m', 'initial synthetic task')
    return board


def daemon_for(tmp_path, root, board):
    daemon = native.PortalImplementationDaemon(todo_path=board,
        state_path=tmp_path / 'state/task-state.json', strategy_path=tmp_path / 'state/strategy.json',
        events_path=tmp_path / 'state/events.jsonl', repo_root=root, task_header_prefix='EX-',
        implement=False, implementation_command='must-not-run-provider', worktree_pool_enabled=False,
        merge_target_branch='main', implementation_protected_paths=(board.relative_to(root).as_posix(),))
    task = native.parse_task_file(board, task_header_prefix='EX-')[0]
    identity = daemon._identity_for_task(task)
    state = native.PortalTaskState(implementation_attempts={'EX-001': 3},
        implementation_attempts_by_cid={identity.canonical_task_cid: 3},
        task_identities={'EX-001': identity.to_dict()})
    state.save(daemon.state_path)
    return daemon, task, state


def poison_identity(monkeypatch, *, empty=False):
    values = {'GIT_AUTHOR_NAME': '' if empty else 'Inherited CI Actor',
        'GIT_AUTHOR_EMAIL': '' if empty else 'github-actions[bot]@users.noreply.github.com',
        'GIT_COMMITTER_NAME': '' if empty else 'Inherited CI Committer',
        'GIT_COMMITTER_EMAIL': '' if empty else 'ci@example.invalid'}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


@pytest.mark.parametrize('empty', [False, True])
def test_real_protected_completion_overrides_inherited_git_identity(tmp_path, monkeypatch, empty):
    root = tmp_path / 'repo'; board = seed(root)
    daemon, task, state = daemon_for(tmp_path, root, board)
    before = asdict(state)
    inherited = poison_identity(monkeypatch, empty=empty)
    unrelated = root / 'unrelated.txt'; unrelated.write_text('retain staged work\n')
    git(root, 'add', 'unrelated.txt')
    result = daemon._mark_tasks_completed_in_todo(['EX-001'], primary_task_id='EX-001',
        completion_reason='validated', expected_task_cids={'EX-001': task.canonical_task_cid},
        expected_task_statuses={'EX-001': 'todo'})
    assert result['durable'] is True, result
    proof = result['protected_board_postcondition']['release_proof']
    assert proof['trusted'] is True and proof['clean'] is True, proof
    assert git(root, 'log', '-1', '--format=%an%n%ae%n%cn%n%ce').splitlines() == [
        'Implementation Daemon', 'implementation-daemon@example.invalid',
        'Implementation Daemon', 'implementation-daemon@example.invalid']
    assert native.parse_task_file(board, task_header_prefix='EX-')[0].status == 'completed'
    assert git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD') == 'tasks.todo.md'
    assert git(root, 'diff', '--cached', '--name-only') == 'unrelated.txt'
    assert not checkout_mutation_lock_path(root).exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert {key: os.environ[key] for key in inherited} == inherited


def test_generated_child_board_and_parent_gitlink_share_owned_identity(tmp_path, monkeypatch):
    child_source = tmp_path / 'child-source'; seed(child_source)
    root = tmp_path / 'repo'; root.mkdir()
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'Fixture'); git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(child_source), 'deps/child')
    git(root, 'commit', '-m', 'initial synthetic parent')
    board = root / 'deps/child/tasks.todo.md'
    daemon, task, state = daemon_for(tmp_path, root, board)
    before = asdict(state); poison_identity(monkeypatch)
    result = daemon._mark_tasks_completed_in_todo(['EX-001'], primary_task_id='EX-001',
        completion_reason='validated', expected_task_cids={'EX-001': task.canonical_task_cid})
    assert result['durable'] is True, result
    histories = result['protected_board_postcondition']['release_proof']['histories']
    assert {row['repository'] for row in histories} == {'.', 'deps/child'}
    for repo in (root, board.parent):
        assert git(repo, 'log', '-1', '--format=%ae%n%ce').splitlines() == [
            'implementation-daemon@example.invalid', 'implementation-daemon@example.invalid']
        assert git(repo, 'status', '--porcelain') == ''
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert not checkout_mutation_lock_path(root).exists()


def test_owned_commit_does_not_approve_foreign_protected_history(tmp_path, monkeypatch):
    root = tmp_path / 'repo'; board = seed(root)
    daemon, task, _ = daemon_for(tmp_path, root, board)
    inherited = poison_identity(monkeypatch)
    def foreign_then_owned():
        board.write_text(board.read_text() + '\nForeign protected edit.\n')
        git(root, 'add', 'tasks.todo.md')
        git(root, 'commit', '-m', 'EX-OTHER: mark todo completed')
        return daemon._mark_tasks_completed_in_todo(['EX-001'], primary_task_id='EX-001',
            completion_reason='validated', expected_task_cids={'EX-001': task.canonical_task_cid})
    result = daemon._run_checkout_mutation_transaction(task_id='EX-001',
        operation='mark_tasks_completed', callback=foreign_then_owned,
        failure_fields={'updated': False}, extra={})
    assert result['durable'] is False, result
    assert result['protected_board_postcondition']['release_proof']['reason'] == 'protected_release_history_untrusted'
    assert result['checkout_mutation_lease_retained'] is True
    assert checkout_mutation_lock_path(root).exists()
    assert git(root, 'log', '-2', '--format=%ae').splitlines() == [
        'implementation-daemon@example.invalid', inherited['GIT_AUTHOR_EMAIL']]
