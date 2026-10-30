"""Actual Native rejection may admit one bounded fresh repair claim, never UNKNOWN."""
from copy import deepcopy
import shlex
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_validation_failure_retry import (
    _proof, bind_feedback, rearm, render_feedback,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_text
from test.api.test_native_provider_failure_settlement import _repository
from test.api.test_agent_supervisor_database_implementation_daemon import _population
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import bind_task_state_control_plane


def _chain(tmp_path):
    repo = tmp_path / 'repository'
    branch = _repository(repo)
    (repo / 'test_result.py').write_text('import result\ndef test_result():\n    assert result.VALUE == 2\n')
    subprocess.run(['git','add','test_result.py'], cwd=repo, check=True)
    subprocess.run(['git','commit','-qm','Declare actual semantic validation'], cwd=repo, check=True)
    state = tmp_path / 'state'; state.mkdir()
    owner = DatabaseImplementationDaemon(database_path=tmp_path / 'control.duckdb',
        execution_path=tmp_path / 'execution.duckdb', coordination_path=tmp_path / 'coordination.duckdb',
        state_dir=state, state_prefix='retry', owner_session_id='fixture-native-validation',
        authority_mode='embedded_exclusive', require_real_execution=True,
        max_task_attempts=3, lease_ms=7200000).open()
    population = _population(1)
    # Keep the actual installed governance plugins importable in this tiny
    # authored repository; sealed Python deliberately ignores editable .pth.
    criterion = 'PYTHONPATH=' + shlex.quote(str(Path(__file__).resolve().parents[2])) + ' python3 -m pytest -q test_result.py'
    population['tasks'][0].update(outputs=[{'path':'result.py'}],
        validation=[criterion], completion='auto',
        acceptance='The actual Python result is two.')
    owner.materialize_population(population)
    portals = []
    def factory(paths, alias):
        bind_task_state_control_plane(paths.state, str(paths.root / 'portal-state.duckdb'))
        portal = PortalImplementationDaemon(todo_path=paths.task_projection, state_path=paths.state,
            strategy_path=paths.strategy, events_path=paths.events, repo_root=repo,
            task_header_prefix='## DQP-', board_namespace='fixture', implement=True,
            implementation_command=shlex.join([sys.executable, str(tmp_path / 'router_timeout.py')]),
            implementation_timeout=60, use_ephemeral_worktree=True,
            worktree_root=tmp_path / 'worktrees', worktree_submodule_paths=[],
            implementation_log_dir=paths.implementation_logs, merge_target_branch=branch,
            max_task_attempts=3, implementation_protected_paths=(), implementation_declared_outputs_only=True)
        portals.append(portal)
        return portal
    bridge = DatabasePortalExecutionBridge(task_source=owner.task_source,
        attempt_root=tmp_path / 'attempts', portal_factory=factory, repo_root=repo,
        board_namespace='fixture', task_header_prefix='## DQP-', merge_target_branch=branch, max_task_attempts=3)
    owner.bind_execution_callbacks(provider_fn=bridge.run_provider,
        effect_fn=bridge.apply_effect, validation_fn=bridge.validate_effect)
    return owner, bridge, portals



def test_real_rejected_candidate_retries_with_feedback_and_preserves_paid_history(tmp_path, monkeypatch):
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'orchestration'))
    owner, bridge, portals = _chain(tmp_path)
    script = tmp_path / 'router_timeout.py'
    script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 1\\n')\n")
    owner.max_task_attempts = 3
    bridge.max_task_attempts = 3
    bridge.provider_handoff_binder = lambda **kwargs: bind_feedback(owner=owner, **kwargs)
    try:
        first = owner.run_once()
        old = owner.get_attempt(first['attempt_id'])
        assert old.status == 'failed'
        callback = owner.provider_invocation_recorded(old.attempt_id, idempotency_key='provider:' + old.attempt_id)
        assert callback['native_exit']['schema'] == 'database-native-provider-exit@2'
        old_phases = deepcopy(owner.phase_history(old.attempt_id))
        task = owner.task_source.get_task(old.task_cid)
        assert rearm(owner, task) is None  # Other campaigns retain their policy.
        assert owner.task_source.get_task(old.task_cid).status == 'blocked'
        owner.native_validation_failure_retry_enabled = True
        proof = _proof(owner, old)
        assert proof is not None
        # Changed evidence and an exhausted claim budget cannot grant a retry.
        events = bridge._paths(old).events
        original_events = events.read_bytes()
        events.write_bytes(original_events + b'{}\n')
        assert _proof(owner, old) is None
        events.write_bytes(original_events)
        owner.max_task_attempts = old.attempt_number
        assert rearm(owner, task) is None
        owner.max_task_attempts = 3
        script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 2\\n')\n")
        second = owner.run_once()
        current = owner.get_attempt(second['attempt_id'])
        assert current.attempt_number == old.attempt_number + 1
        assert current.status == 'succeeded', second
        assert owner.phase_history(old.attempt_id) == old_phases
        assert owner.provider_invocation_recorded(old.attempt_id, idempotency_key='provider:' + old.attempt_id) == callback
        assert owner.task_source.get_task(old.task_cid).status == 'completed'
        assert (tmp_path / 'repository/result.py').read_text() == 'VALUE = 2\n'
        value = portals[-1]._native_validation_retry_feedback
        assert value['predecessor_attempt_id'] == old.attempt_id
        assert value['proof'] == proof and value['completion_authority'] is False
        task = parse_task_text(bridge._paths(current).task_projection.read_text(),
            path=bridge._paths(current).task_projection, task_header_prefix='## DQP-')[0]
        text = render_feedback(portals[-1], task)
        assert proof['retained_candidate'] in text and 'diagnostic only' in text
        assert 'python3 -m pytest -q test_result.py' in text
        assert not owner.reconcile_recoverable_portal_failure_rearms()
    finally:
        for portal in portals:
            portal.close_event_runtime()
        owner.close()


def test_unknown_callback_cannot_enter_known_validation_retry(tmp_path, monkeypatch):
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'orchestration'))
    owner, bridge, portals = _chain(tmp_path)
    script = tmp_path / 'router_timeout.py'
    script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 1\\n')\n")
    owner.max_task_attempts = bridge.max_task_attempts = 3
    owner.native_validation_failure_retry_enabled = True
    monkeypatch.setattr(bridge, '_native_provider_exit_failure', lambda **kwargs: None)
    try:
        first = owner.run_once()
        old = owner.get_attempt(first['attempt_id'])
        before = owner.provider_invocation_recorded(old.attempt_id, idempotency_key='provider:' + old.attempt_id)
        assert old.status == 'running' and before['callback_state'] == 'started_outcome_unknown'
        assert rearm(owner, owner.task_source.get_task(old.task_cid)) is None
        assert _proof(owner, old) is None
        assert owner.get_attempt(old.attempt_id).status == 'running'
        assert owner.provider_invocation_recorded(old.attempt_id, idempotency_key='provider:' + old.attempt_id) == before
    finally:
        for portal in portals:
            portal.close_event_runtime()
        owner.close()
