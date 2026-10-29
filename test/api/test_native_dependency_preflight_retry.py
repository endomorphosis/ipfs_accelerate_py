"""Real Native preflight/custody/merge; the provider is an authored local shim."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import bind_task_state_control_plane
from ipfs_accelerate_py.agent_supervisor.todo_daemon import dependency_preflight_retry as retry
from ipfs_accelerate_py.agent_supervisor.validation import project_dependency_preflight as probe


def git(repo, *args):
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_HOME', str(tmp_path / 'agent-home'))
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'orchestration'))
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'Native fixture')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    (repo / 'pyproject.toml').write_text('[project]\nname="native-dependency-fixture"\nversion="0.0.0"\ndependencies=["pytest>=7"]\n')
    (repo / 'test_feature.py').write_text('from pathlib import Path\ndef test_candidate():\n    assert Path("feature.py").read_text() == "VALUE = 2\\n"\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'Authored fixture baseline')
    now = [int(time.time() * 1000)]
    owner = DatabaseImplementationDaemon(database_path=tmp_path / 'control.duckdb',
        coordination_path=tmp_path / 'coordination.duckdb', execution_path=tmp_path / 'execution.duckdb',
        state_dir=tmp_path / 'native', state_prefix='fixture', owner_session_id='fixture-dependency',
        authority_mode='embedded_exclusive', require_real_execution=True,
        lease_ms=7200000, clock_ms=lambda: now[0], max_task_attempts=3).open()
    owner.materialize_population({'repository_tree_id': git(repo, 'rev-parse', 'HEAD'),
        'objectives': [{'objective_id': 'fixture', 'goal_cid': 'fixture-goal', 'title': 'Fixture'}],
        'tasks': [{'task_cid': 'sha256:' + hashlib.sha256(b'dependency-fixture').hexdigest(),
            'task_id': 'DEP-001', 'goal_cid': 'fixture-goal', 'status': 'ready', 'ordinal': 1,
            'title': 'Bounded dependency retry', 'outputs': [{'path': 'feature.py'}],
            'validation': ['python3 -m pytest -q test_feature.py'], 'completion': 'auto',
            'acceptance': 'Real pytest checks the exact source value.'}]})
    portals = []
    def factory(paths, alias):
        bind_task_state_control_plane(paths.state, str(paths.root / 'portal-state.duckdb'))
        portal = PortalImplementationDaemon(todo_path=paths.task_projection, state_path=paths.state,
            strategy_path=paths.strategy, events_path=paths.events, repo_root=repo,
            task_header_prefix='## DEP-', board_namespace='fixture-dependency',
            worktree_root=tmp_path / 'worktrees', merge_target_branch='main',
            worktree_submodule_paths=[], implement=True, use_ephemeral_worktree=True, implementation_timeout=3600,
            implementation_command='python3 -c \'from pathlib import Path; Path("feature.py").write_text("VALUE = 2\\n")\'')
        portals.append(portal)
        return portal
    bridge = DatabasePortalExecutionBridge(task_source=owner.task_source,
        attempt_root=tmp_path / 'native' / 'attempts', portal_factory=factory,
        repo_root=repo, board_namespace='fixture-dependency', merge_target_branch='main',
        prior_attempt_authority=owner.authorize_superseded_portal_attempt_binding)
    owner.bind_execution_callbacks(provider_fn=bridge.run_provider,
        effect_fn=bridge.apply_effect, validation_fn=bridge.validate_effect)
    (tmp_path / 'native').mkdir(exist_ok=True)
    attempt = owner.claim_next()
    paths, binding = bridge._ensure_attempt_projection(attempt, owner.task_source.get(attempt.task_cid))
    try:
        yield owner, bridge, attempt, paths, now, repo
    finally:
        for portal in portals:
            portal.close_event_runtime()
        owner.close()


@pytest.mark.parametrize('legacy_unknown', [False, True])
def test_actual_probe_timeout_waits_then_completes_the_same_attempt(case, monkeypatch, legacy_unknown):
    owner, bridge, attempt, paths, now, repo = case
    # Exercise the actual child timeout, without replacing Native or its probe.
    with monkeypatch.context() as patch:
        patch.setattr(probe, 'DEPENDENCY_PROBE_TIMEOUT_SECONDS', 0.000001)
        if legacy_unknown:
            # Simulate the old owner exiting before persisting the known wait.
            # Native still produces the real refusal, state and cleanup proof.
            patch.setattr(retry, 'settle', lambda *_args, **_kwargs: False)
        first = owner.run_once()
    if legacy_unknown:
        assert owner.provider_invocation_recorded(attempt.attempt_id,
            idempotency_key='provider:' + attempt.attempt_id)['callback_state'] == 'started_outcome_unknown'
        before = paths.events.read_bytes()
        owner.run_once()
        assert paths.events.read_bytes() == before
    pending = owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id)
    assert pending['schema'] == retry.SCHEMA, json.dumps(first)
    assert pending['callback_state'] == 'not_dispatched'
    assert pending['retry_count'] == 1
    events = paths.events.read_bytes()
    assert b'implementation_started' not in events
    waiting = owner.run_once()
    assert waiting['implementation_result']['reason'] == retry.REASON
    assert paths.events.read_bytes() == events
    now[0] = pending['native']['retry_not_before_ms'] + 1
    result = owner.run_once()
    assert owner.get_attempt(attempt.attempt_id).status == 'succeeded', json.dumps(result)
    assert owner.get_attempt(attempt.attempt_id).attempt_number == 1
    assert owner.task_source.get_task(attempt.task_alias).status == 'completed'
    assert (repo / 'feature.py').read_text() == 'VALUE = 2\n'
    assert sum(json.loads(s).get('type') == 'implementation_started' for s in paths.events.read_text().splitlines()) == 1


def test_expired_original_deadline_does_not_dispatch_or_reset_wait(case, monkeypatch):
    owner, bridge, attempt, paths, now, repo = case
    with monkeypatch.context() as patch:
        patch.setattr(probe, 'DEPENDENCY_PROBE_TIMEOUT_SECONDS', 0.000001)
        owner.run_once()
    pending = owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id)
    assert pending['schema'] == retry.SCHEMA
    before = paths.events.read_bytes()
    now[0] = pending['native']['retry_deadline_ms']
    result = owner.run_once()
    assert result['implementation_result']['reason'] == 'validation_dependency_repair_required'
    assert paths.events.read_bytes() == before
    assert owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id) == pending
    assert not (repo / 'feature.py').exists()


def test_changed_native_event_chain_cannot_settle_an_unknown_callback(case, monkeypatch):
    owner, bridge, attempt, paths, now, repo = case
    with monkeypatch.context() as patch:
        patch.setattr(probe, 'DEPENDENCY_PROBE_TIMEOUT_SECONDS', 0.000001)
        patch.setattr(retry, 'settle', lambda *_args, **_kwargs: False)
        owner.run_once()
    original = owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id)
    events = [json.loads(s) for s in paths.events.read_text().splitlines()]
    next(e for e in events if e['type'] == 'implementation_finished')['backoff_seconds'] = 1
    paths.events.write_text(''.join(json.dumps(e) + '\n' for e in events))
    result = owner.run_once()
    assert result['implementation_result']['reason'] == 'provider_callback_outcome_unknown'
    assert owner.provider_invocation_recorded(attempt.attempt_id, idempotency_key='provider:' + attempt.attempt_id) == original
    assert not (repo / 'feature.py').exists()
