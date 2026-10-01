"""Isolated owner publication must work without ambient Git identity."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime
from ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution import GIT_OWNER_ENV
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon, PortalTask


IDENTITY_KEYS = {'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL'}


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def test_native_merge_uses_signed_owner_identity_without_global_config(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith('GIT_') or name == 'EMAIL':
            monkeypatch.delenv(name)
    configured = admitted_benchmark_runtime._candidate_git_environment()
    # First reproduce the old exact launch configuration: no ambient or local
    # author is available, while baseline/candidate commits set per-call -c.
    for name, value in configured.items():
        if name not in IDENTITY_KEYS:
            monkeypatch.setenv(name, value)
    root = tmp_path / 'repository'
    root.mkdir()
    _git(root, 'init', '-q', '--initial-branch=main')
    _git(root, 'config', 'user.useConfigOnly', 'true')
    source = root / 'answer.py'
    source.write_text('def answer():\n    return 1\n')
    _git(root, 'add', 'answer.py')
    commit_prefix = ('-c', 'user.name=Implementation Daemon', '-c', 'user.email=implementation-daemon@example.invalid')
    _git(root, *commit_prefix, 'commit', '-qm', 'fixture baseline')
    baseline = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'checkout', '-qb', 'implementation/identity-probe')
    source.write_text('def answer():\n    return 2\n')
    _git(root, *commit_prefix, 'commit', '-qam', 'fixture candidate')
    candidate = _git(root, 'rev-parse', 'HEAD')
    candidate_tree = _git(root, 'rev-parse', 'HEAD^{tree}')
    _git(root, 'checkout', '-q', 'main')
    state = tmp_path / 'state'
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / 'tasks.md', state_path=state / 'tasks.json',
        strategy_path=state / 'strategy.json', events_path=state / 'events.jsonl',
        repo_root=root, merge_target_branch='main', worktree_pool_enabled=False,
        llm_merge_resolver_command='',
    )
    task = PortalTask(task_id='IDENTITY-PROBE', title='Merge public answer fixture',
                      status='todo', completion='manual', priority='P1', track='ops')
    try:
        def merge():
            return daemon._merge_branch_to_main(
                'implementation/identity-probe', task, 1, baseline_ref=baseline,
                expected_candidate_commit=candidate, expected_candidate_tree=candidate_tree,
            )

        missing = merge()
        assert missing['returncode'] == 128 and not missing['merged'], missing
        assert 'identity unknown' in missing['stderr'].lower(), missing
        assert _git(root, 'rev-parse', 'HEAD') == baseline
        assert source.read_text() == 'def answer():\n    return 1\n'
        for name in IDENTITY_KEYS:
            monkeypatch.setenv(name, configured[name])
        published = merge()
        assert published['merged'] and published['returncode'] == 0, published
        head = _git(root, 'rev-parse', 'HEAD')
        assert _git(root, 'rev-list', '--parents', '-n', '1', head).split() == [head, baseline, candidate]
        assert _git(root, 'show', '-s', '--format=%an <%ae>|%cn <%ce>', head) == (
            'Isolated Supervisor <supervisor@example.invalid>|Isolated Supervisor <supervisor@example.invalid>'
        )
        assert source.read_text() == 'def answer():\n    return 2\n'
        assert _git(root, 'status', '--porcelain') == ''
        assert subprocess.run(['git', '-C', str(root), 'config', '--local', '--get', 'user.name'], capture_output=True).returncode == 1
        assert subprocess.run(['git', '-C', str(root), 'config', '--local', '--get', 'user.email'], capture_output=True).returncode == 1
        # Candidate validation defaults are unchanged and carry no owner
        # identity; only the admitted daemon's signed environment gains it.
        assert not IDENTITY_KEYS.intersection(GIT_OWNER_ENV)
        for name, value in GIT_OWNER_ENV.items():
            if name != 'GIT_CONFIG_COUNT':
                assert configured[name] == value
        base_count = int(GIT_OWNER_ENV['GIT_CONFIG_COUNT'])
        configured_count = int(configured['GIT_CONFIG_COUNT'])
        assert configured_count in {base_count, base_count + 1}
        if configured_count > base_count:
            assert configured[f'GIT_CONFIG_KEY_{base_count}'] == 'safe.directory'
            assert configured[f'GIT_CONFIG_VALUE_{base_count}'] == str(
                Path(admitted_benchmark_runtime.__file__).absolute().parents[3]
            )
    finally:
        daemon.close_event_runtime()
