"""A kernel stall is not authority to clear a live resource lease."""
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as mod


@pytest.mark.parametrize('alias', ['PCTDD-005', 'PCTDD-008'])
@pytest.mark.parametrize('state', ['D', 'T', 't'])
@pytest.mark.parametrize('claimed_path', ['ipfs_datasets_py', 'ipfs_datasets_py/logic'])
def test_live_stuck_owner_retains_exact_and_overlapping_claims(tmp_path, monkeypatch, alias, state, claimed_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setattr(mod, 'process_command_line', lambda _: 'python -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon')
    monkeypatch.setattr(mod, 'process_kernel_state', lambda _: state, raising=False)
    monkeypatch.setattr(mod, 'process_has_live_grok_descendant', lambda _: False, raising=False)
    daemon = mod.PortalImplementationDaemon(
        todo_path=repo / 'todo.md', state_path=repo / 'lane/state.json',
        strategy_path=repo / 'lane/strategy.json', events_path=repo / 'lane/events.jsonl',
        repo_root=repo, task_header_prefix='## PCTDD-', implement=True,
        worktree_submodule_paths=('ipfs_datasets_py',),
    )
    task = mod.PortalTask(task_id=alias, title='Resource claimant', status='todo',
        completion='auto', priority='P0', track='implementation',
        outputs=['ipfs_datasets_py/logic/identity.py'])
    path = daemon._implementation_resource_claim_path(claimed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'kind': mod.IMPLEMENTATION_RESOURCE_CLAIM_LOCK_KIND, 'lease_id': 'existing-lease',
        'pid': os.getpid(), 'owner_script': 'implementation_daemon.py',
        'repo_root': str(repo.resolve()), 'state_dir': str((repo / 'peer').resolve()),
        'task_id': 'PCTDD-006', 'resource_kind': 'submodule', 'resource_path': claimed_path,
    }
    original = json.dumps(metadata).encode()
    path.write_bytes(original)
    assert daemon._implementation_resource_claim_owner_is_active(metadata)
    claims, unavailable, reason, _ = daemon._acquire_implementation_resource_claims(
        task, attempt=1, started_at='2026-09-10T04:00:00+00:00')
    assert claims == []
    assert unavailable
    assert reason in {'lock_exists', 'overlapping_claim_exists'}
    assert path.read_bytes() == original
