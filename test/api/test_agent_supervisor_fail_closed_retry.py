"""Sealed bootstrap rejection must survive both supervisor retry layers."""
import json
import sys
from unittest.mock import Mock

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor, TodoSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop, SupervisorLoopConfig, SupervisorLoopResult,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    RestartPolicy, child_exit_should_restart,
)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('limit', [0, 1, 3])
def test_real_fail_closed_child_runs_once(tmp_path, limit):
    marker = tmp_path / 'launches'
    command = (sys.executable, '-c',
               "from pathlib import Path; import sys; "
               "p=Path(sys.argv[1]); p.open('a').write('launch\\n'); sys.exit(78)",
               str(marker))
    spec = ManagedDaemonSpec(
        name='sealed-test', schema='test', repo_root=tmp_path,
        daemon_dir=tmp_path, runner=command,
        status_path=tmp_path / 'child.json',
        supervisor_status_path=tmp_path / 'supervisor.json',
        supervisor_pid_path=tmp_path / 'supervisor.pid',
        child_pid_path=tmp_path / 'child.pid',
        supervisor_out_path=tmp_path / 'supervisor.out',
        ensure_status_path=tmp_path / 'ensure.json',
        ensure_check_path=tmp_path / 'check.json',
    )
    loop = SupervisorLoop(SupervisorLoopConfig(
        spec=spec, command=command, log_prefix='sealed', max_restarts=limit,
        heartbeat_seconds=.01, poll_seconds=.01,
        watchdog_startup_grace_seconds=60,
        restart_policy=RestartPolicy(restart_backoff_seconds=0, fast_restart_backoff_seconds=0),
    ))
    result = loop.run()
    assert marker.read_text() == 'launch\n'
    assert result.status == 'typed_child_blocker'
    assert result.last_exit_code == 78
    assert result.last_recycle_reason == 'typed_fail_closed_exit'
    status = json.loads(spec.supervisor_status_path.read_text())
    assert status['status'] == 'typed_child_blocker'
    assert status['last_exit_code'] == 78
    assert not spec.child_pid_path.exists()


@pytest.mark.parametrize('status,exit_code', [
    ('typed_child_blocker', 78), ('child_exited', 78),
    ('max_restarts_reached', 78), ('typed_child_blocker', None),
])
def test_outer_loop_does_not_maintain_or_retry_rejected_child(tmp_path, status, exit_code):
    supervisor = TodoImplementationSupervisor(TodoSupervisorConfig(
        repo_root=tmp_path, todo_path=tmp_path / 'todo.md',
        state_path=tmp_path / 'state.json', strategy_path=tmp_path / 'strategy.json',
        events_path=tmp_path / 'events.jsonl', state_dir=tmp_path,
    ))
    supervisor.ensure_event_log_file = Mock()
    supervisor.ensure_managed_daemon_pid_file = Mock(return_value={})
    supervisor.repair_main_checkout_merge_state = Mock()
    supervisor._uses_supervisor_state_owner_bootstrap = Mock(return_value=False)
    supervisor.build_supervisor_loop_config = Mock(return_value=object())
    supervisor.run_once = Mock(return_value={})
    supervisor._supervisor_loop_recovery_delay_seconds = Mock(return_value=0)
    supervisor._record_event = Mock()
    loop = Mock()
    loop.run.side_effect = [SupervisorLoopResult(
        status=status, restart_count=1, last_exit_code=exit_code,
        last_recycle_reason='typed_fail_closed_exit',
    ), AssertionError('outer loop retried rejected child')]
    supervisor.shared_supervisor_loop_class = Mock(return_value=loop)
    assert supervisor._run_forever_loop() == 78
    loop.run.assert_called_once()
    supervisor.run_once.assert_called_once_with(include_refill=False)
    assert not any('recovery' in call.args[0]
                   for call in supervisor._record_event.call_args_list)


def test_fail_closed_exit_is_not_a_retryable_crash():
    assert not child_exit_should_restart(exit_code=78, restart_count=0, restart_limit=3)
    assert child_exit_should_restart(exit_code=1, restart_count=0, restart_limit=3)
