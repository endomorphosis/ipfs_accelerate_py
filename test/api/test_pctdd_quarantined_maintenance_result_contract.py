"""A retained workspace must not turn native prelaunch observation into TypeError."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine as q
from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    reconciliation_guardrail_records,
    resolved_reconciliation_guardrail_keys,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
)
from test.api.test_agent_supervisor_reconciliation_auto_unblock import _supervisor
from test.api.test_workspace_root_quarantine import seed


def test_native_frozen_cleanup_remains_a_deferred_unobserved_scan(tmp_path):
    repo, root, _, lease, lifecycle, record = seed(tmp_path)
    (lease.path / 'README').write_bytes(b'unknown callback output stays retained\n')
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    # This deliberately lacks every unguarded cleanup dependency: the native
    # decorator must still prevent the entire Git/scanning/mutation body.
    owner = SimpleNamespace(config=SimpleNamespace(repo_root=repo))
    result = PortalImplementationSupervisor._cleanup_backlogged_worktrees_locked(owner)

    # Production used this exact consumer after guarded cleanup returned. On
    # 7fec51e1 the decorator's skipped=True raises the retained bool TypeError.
    assert reconciliation_guardrail_records(cleanup_result=result) == []
    assert resolved_reconciliation_guardrail_keys(cleanup_result=result) == set()
    assert result['attempted'] is False
    assert result['maintenance_deferred'] is True
    assert result['skipped'] == []  # no workspace census was performed
    assert result['reason'] == 'retained_workspace_scope'
    assert q.verify(repo, root) == frozen
    assert lifecycle.load_workspace(lease.path) == record
    assert (lease.path / 'README').read_bytes() == b'unknown callback output stays retained\n'


def test_native_prelaunch_guardrail_consumers_preserve_quarantine_and_strategy(tmp_path):
    repo, root, _, lease, lifecycle, record = seed(tmp_path)
    supervisor = _supervisor(repo, worktree_root=root)
    supervisor.config.reconciliation_guardrail_enabled = True
    supervisor.config.strategy_path.parent.mkdir(exist_ok=True)
    supervisor.config.strategy_path.write_text(json.dumps({
        'blocked_tasks': ['PCTDD-005'],
        'reconciliation_guardrail_seen_fingerprints': ['retained-unknown'],
    }))
    strategy = supervisor.config.strategy_path.read_bytes()
    todo = supervisor.config.todo_path.read_bytes()
    frozen = q.freeze(repo, root, expected=q.census(repo, root))

    reconciliation = supervisor.reconcile_backlogged_worktrees()
    cleanup = supervisor._cleanup_backlogged_worktrees_locked()
    assert supervisor.record_reconciliation_guardrails(reconciliation, cleanup) == []
    assert supervisor.release_completed_guardrail_blocks(
        reconciliation_result=reconciliation, cleanup_result=cleanup, replay_result={},
    ) == []
    assert supervisor.config.strategy_path.read_bytes() == strategy
    assert supervisor.config.todo_path.read_bytes() == todo
    assert q.verify(repo, root) == frozen
    assert lifecycle.load_workspace(lease.path) == record


def test_malformed_external_cleanup_boolean_is_not_silently_coerced():
    with pytest.raises(TypeError, match="'bool' object is not iterable"):
        reconciliation_guardrail_records(cleanup_result={
            'attempted': True, 'skipped': True,
        })
