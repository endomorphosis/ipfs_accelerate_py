import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_shutdown import DatabaseDaemonShutdown
from test.api.test_retained_attempt_fairness import setup, snapshot
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon

SOURCE = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('mode', ['query', 'close'])
def test_actual_cli_signal_during_native_query_or_close_never_unwinds_native_call(tmp_path, mode):
    output = tmp_path / 'result.json'
    child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).parent / 'fixtures/database_shutdown_process_fixture.py'),
                              str(SOURCE), str(output), mode], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    result = json.loads(output.read_text())
    assert result['module'] == str(SOURCE / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py')
    assert result['native_query_completed'] is True
    assert result['closed'] is True
    assert result['passes'] == 1
    assert result['exception_type'] == 'SystemExit' and result['exit_code'] == 143
    assert not Path('/proc', str(child.pid)).exists()


def test_signal_handler_is_first_request_memory_only_until_safe_checkpoint():
    value = DatabaseDaemonShutdown()
    value.request(signal.SIGTERM)
    value.request(signal.SIGINT)
    assert value.signum == signal.SIGTERM
    with pytest.raises(SystemExit) as error:
        value.checkpoint()
    assert error.value.code == 143


@pytest.mark.parametrize('when', ['before_pass', 'after_ready_read'])
def test_shutdown_prevents_new_claim_and_preserves_unknown_for_next_start(tmp_path, monkeypatch, when):
    daemon, attempt, now, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, attempt, artifact)
    stop = DatabaseDaemonShutdown()
    daemon._cooperative_shutdown = stop
    provider, validate = daemon._provider_fn, daemon._validation_fn
    if when == 'before_pass':
        stop.request(signal.SIGTERM)
    else:
        original = daemon.task_source.ready_tasks
        def ready(*args, **kwargs):
            result = original(*args, **kwargs)
            stop.request(signal.SIGTERM)
            return result
        monkeypatch.setattr(daemon.task_source, 'ready_tasks', ready)
    try:
        with pytest.raises(SystemExit):
            daemon.run_once()
        assert calls == []
        assert snapshot(daemon, attempt, artifact) == before
        assert daemon.task_source.get('task:cid:002').status == 'ready'
    finally:
        monkeypatch.undo()
        daemon.close()
    replacement = _open_daemon(tmp_path, session='session:sawm-fairness', provider_fn=provider,
                               lease_ms=5000, clock_ms=lambda: now['ms'], task_shard_count=4,
                               task_shard_index=3, strict_task_sharding=True, task_prefix='SAWM')
    replacement.require_real_execution = True
    replacement._validation_fn = validate
    try:
        result = replacement.run_once()
        assert result['active_task_id'] == 'SAWM-023'
        assert result['implementation_result']['status'] == 'succeeded'
        assert calls == ['task:cid:002']
        assert snapshot(replacement, attempt, artifact) == before
        assert result['retained_attempts'][0]['attempt_consumed'] == 'unknown'
    finally:
        replacement.close()


def test_shutdown_after_native_renewal_denies_provider_callback_without_settling_old_attempt(tmp_path, monkeypatch):
    daemon, old, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, old, artifact)
    attempt = daemon.commit_phase(daemon.claim_next(exclude_task_cids=(old.task_cid,)), 'context')
    stop = DatabaseDaemonShutdown()
    daemon._cooperative_shutdown = stop
    original = daemon._renew_attempt_lease
    def renew(*args, **kwargs):
        result = original(*args, **kwargs)
        stop.request(signal.SIGTERM)
        return result
    monkeypatch.setattr(daemon, '_renew_attempt_lease', renew)
    try:
        with pytest.raises(SystemExit):
            daemon.run_provider(attempt)
        assert calls == []
        assert daemon.provider_invocation_recorded(attempt.attempt_id,
                    idempotency_key='provider:' + attempt.attempt_id) is None
        assert snapshot(daemon, old, artifact) == before
    finally:
        daemon.close()


def test_already_started_callback_records_its_result_before_stopping_new_effect(tmp_path):
    daemon, old, _, calls, artifact = setup(tmp_path)
    before = snapshot(daemon, old, artifact)
    attempt = daemon.commit_phase(daemon.claim_next(exclude_task_cids=(old.task_cid,)), 'context')
    stop = DatabaseDaemonShutdown()
    daemon._cooperative_shutdown = stop
    original = daemon._provider_fn
    def provider(current):
        result = original(current)
        stop.request(signal.SIGTERM)
        return result
    try:
        current, result, duplicated = daemon.run_provider(attempt, provider_fn=provider)
        assert not duplicated and calls == ['task:cid:002']
        assert daemon.provider_invocation_recorded(attempt.attempt_id,
                    idempotency_key='provider:' + attempt.attempt_id) == result
        with pytest.raises(SystemExit):
            daemon.run_effect(current, result)
        assert snapshot(daemon, old, artifact) == before
    finally:
        daemon.close()
