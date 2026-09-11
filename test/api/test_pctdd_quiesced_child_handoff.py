"""A native maintenance fence must hand a stopped child back to its loop."""

from dataclasses import replace
import json
import os
import signal

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor, TodoSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop, SupervisorLoopDecision,
)


def _configured_supervisor(tmp_path):
    script = tmp_path / 'idle-daemon.py'
    script.write_text('import time\ntime.sleep(60)\n')
    state = tmp_path / 'state'
    supervisor = PortalImplementationSupervisor(TodoSupervisorConfig(
        repo_root=tmp_path, todo_path=tmp_path / 'todo.md',
        state_dir=state, state_path=state / 'task-state.json',
        strategy_path=state / 'strategy.json', events_path=state / 'events.jsonl',
        daemon_script_path=script,
    ))
    config = supervisor.build_supervisor_loop_config()
    config = replace(config, watchdog_startup_grace_seconds=0, max_restarts=1,
                     child_env={**config.child_env,
                        runtime.SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(supervisor._managed_daemon_identity_path()),
                        runtime.SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(supervisor._managed_daemon_owner_scope())})
    return supervisor, config


def _close_owned(child):
    if runtime.owner_liveness(child.identity_process_birth) is runtime.OwnerLiveness.ALIVE:
        os.kill(child.pid, signal.SIGTERM)
    try:
        os.waitpid(child.pid, 0)
    except ChildProcessError:
        pass


@pytest.mark.parametrize('action', ['stop', 'recycle'])
def test_loop_recognizes_actual_native_quiescence_after_markers_removed(tmp_path, action):
    supervisor, config = _configured_supervisor(tmp_path)
    observed = []
    children = []
    def maintenance(_loop, child, _status):
        children.append(child)
        result = supervisor._quiesce_supervised_child_for_control_gate(child, reason='supervisor_watchdog_maintenance')
        observed.append(result)
        assert result['quiesced'] is True
        assert result['markers_removed'] is True
        assert not child.identity_path.exists()
        assert not child.child_pid_path.exists()
        return SupervisorLoopDecision(action=action, reason='supervisor_maintenance_completed_after_quiescence')
    loop = SupervisorLoop(config, watchdog_hook=maintenance, sleep=lambda _seconds: None)
    try:
        result = loop.run()
        assert observed and observed[0]['supervised_child_alive'] is False
        assert result.status == ('stopped' if action == 'stop' else 'max_restarts_reached')
        assert result.last_recycle_reason == 'supervisor_maintenance_completed_after_quiescence'
        assert result.restart_count == (0 if action == 'stop' else 1)
    finally:
        for child in children:
            _close_owned(child)


def _child(tmp_path):
    supervisor, config = _configured_supervisor(tmp_path)
    loop = SupervisorLoop(config)
    return runtime.launch_supervised_child(loop._child_spec('test-child'))


@pytest.mark.parametrize('marker', ['missing', 'malformed', 'replacement'])
def test_dead_frozen_birth_does_not_authorize_marker_cleanup_or_signal(tmp_path, monkeypatch, marker):
    child = _child(tmp_path)
    try:
        assert runtime.terminate_supervised_child(child) is True
        _close_owned(child)
        if marker != 'missing':
            child.child_pid_path.write_text('123\n')
            child.identity_path.write_text('unavailable\n' if marker == 'malformed' else json.dumps({'replacement': True}))
        before = {p: p.read_bytes() for p in (child.child_pid_path, child.identity_path) if p.exists()}
        monkeypatch.setattr(runtime, 'terminate_pid_tree', lambda *_a, **_k: pytest.fail('stale handle signalled a process'))
        assert runtime.supervised_child_is_proven_dead(child) is (marker == 'missing')
        assert runtime.terminate_supervised_child(child) is False
        assert runtime.clear_child_pid_file(child) is False
        assert {p: p.read_bytes() for p in before} == before
    finally:
        _close_owned(child)


def test_absent_markers_do_not_prove_live_or_unknown_birth_dead(tmp_path, monkeypatch):
    child = _child(tmp_path)
    try:
        child.identity_path.unlink()
        child.child_pid_path.unlink()
        assert runtime.supervised_child_is_proven_dead(child) is False
        monkeypatch.setattr(runtime, 'owner_liveness', lambda _birth: runtime.OwnerLiveness.UNKNOWN)
        assert runtime.supervised_child_is_proven_dead(child) is False
    finally:
        # Use the owned direct child, independent of the deliberately unknown observer.
        os.kill(child.pid, signal.SIGTERM)
        try: os.waitpid(child.pid, 0)
        except ChildProcessError: pass


@pytest.mark.parametrize('change', ['missing_birth', 'wrong_pid', 'zero_birth', 'missing_boot', 'missing_record', 'wrong_group'])
def test_incomplete_frozen_handle_cannot_replace_missing_identity(tmp_path, monkeypatch, change):
    child = _child(tmp_path)
    try:
        assert runtime.terminate_supervised_child(child) is True
        _close_owned(child)
        changes = {
            'missing_birth': {'identity_process_birth': None},
            'wrong_pid': {'identity_process_birth': replace(child.identity_process_birth, pid=child.pid + 1)},
            'zero_birth': {'identity_process_birth': replace(child.identity_process_birth, start_time_ticks=0)},
            'missing_boot': {'identity_process_birth': replace(child.identity_process_birth, boot_id='')},
            'missing_record': {'identity_record_id': ''},
            'wrong_group': {'owned_process_group_id': child.pid + 1},
        }
        monkeypatch.setattr(runtime, 'owner_liveness', lambda _birth: runtime.OwnerLiveness.DEAD)
        assert runtime.supervised_child_is_proven_dead(replace(child, **changes[change])) is False
    finally:
        _close_owned(child)


def test_valid_replacement_generation_is_preserved_and_native_adoption_refuses_it(tmp_path):
    supervisor, config = _configured_supervisor(tmp_path)
    spec = SupervisorLoop(config)._child_spec('original')
    original = runtime.launch_supervised_child(spec)
    replacement = None
    try:
        assert runtime.terminate_supervised_child(original) is True
        _close_owned(original)
        # A distinct real child now owns the same marker paths. Its command
        # differs, so original desired launch must refuse this replacement.
        replacement = runtime.launch_supervised_child(replace(spec, command=(*spec.command, '--foreign-owner')))
        before = {p: p.read_bytes() for p in (replacement.child_pid_path, replacement.identity_path)}
        assert runtime.supervised_child_is_proven_dead(original) is False
        assert runtime.terminate_supervised_child(original) is False
        assert runtime.clear_child_pid_file(original) is False
        with pytest.raises(RuntimeError):
            runtime.adopt_or_launch_supervised_child(spec, launch_lock_path=config.spec.supervisor_lock_path)
        assert runtime.owner_liveness(replacement.identity_process_birth) is runtime.OwnerLiveness.ALIVE
        assert {p: p.read_bytes() for p in before} == before
    finally:
        _close_owned(original)
        if replacement is not None:
            _close_owned(replacement)


@pytest.mark.parametrize('group_observation', ['present', 'unavailable'])
def test_dead_original_birth_still_requires_absent_owned_group(tmp_path, monkeypatch, group_observation):
    child = _child(tmp_path)
    try:
        assert runtime.terminate_supervised_child(child) is True
        _close_owned(child)
        calls = []
        def observe_group(pgid, sig):
            calls.append((pgid, sig))
            if group_observation == 'unavailable':
                raise PermissionError('group census unavailable')
        monkeypatch.setattr(runtime.os, 'killpg', observe_group)
        assert runtime.supervised_child_is_proven_dead(child) is False
        assert calls == [(child.pid, 0)]
    finally:
        _close_owned(child)
