"""Native coordination resumes an actual lost rearm response without provider replay."""
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_validation_failure_retry import bind_feedback, rearm
from test.api.test_native_validation_failure_retry import _chain


def test_retrying_control_crash_completes_coordination_then_new_native_claim(tmp_path, monkeypatch):
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'orchestration'))
    owner, bridge, portals = _chain(tmp_path)
    script = tmp_path / 'router_timeout.py'
    script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 1\\n')\n")
    bridge.provider_handoff_binder = lambda **kwargs: bind_feedback(owner=owner, **kwargs)
    try:
        first = owner.run_once()
        old = owner.get_attempt(first['attempt_id'])
        assert old.status == 'failed'
        callback = deepcopy(owner.provider_invocation_recorded(old.attempt_id, idempotency_key='provider:' + old.attempt_id))
        phases = deepcopy(owner.phase_history(old.attempt_id))
        task = owner.task_source.get_task(old.task_cid)
        owner.native_validation_failure_retry_enabled = True
        original = owner.coordinator.rearm_failed_task
        def crashed_after_control(**kwargs):
            assert owner.task_source.get_task(old.task_cid).status == 'retrying'
            raise RuntimeError('authored crash after canonical retry CAS')
        monkeypatch.setattr(owner.coordinator, 'rearm_failed_task', crashed_after_control)
        with pytest.raises(RuntimeError, match='after canonical retry CAS'):
            rearm(owner, task)
        assert owner.task_source.get_task(old.task_cid).revision == task.revision + 1
        assert owner.coordinator.get_task_attempt(old.attempt_id).status.value == 'failed'
        monkeypatch.setattr(owner.coordinator, 'rearm_failed_task', original)
        script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 2\\n')\n")
        result = owner.run_once()
        current = owner.get_attempt(result['attempt_id'])
        assert current.status == 'succeeded', result
        assert current.attempt_number == old.attempt_number + 1
        assert owner.phase_history(old.attempt_id) == phases
        assert owner.provider_invocation_recorded(old.attempt_id, idempotency_key='provider:' + old.attempt_id) == callback
        assert portals[-1]._native_validation_retry_feedback['predecessor_attempt_id'] == old.attempt_id
        assert owner.task_source.get_task(old.task_cid).status == 'completed'
    finally:
        for portal in portals:
            portal.close_event_runtime()
        owner.close()
