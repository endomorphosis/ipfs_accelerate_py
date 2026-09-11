"""Original process custody survives native removal of its marker files."""
from dataclasses import replace
import json
import os
import signal
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime


def _child(tmp_path):
    return runtime.launch_supervised_child(runtime.SupervisedChildSpec(
        repo_root=tmp_path, command=(sys.executable, "-c", "import time; time.sleep(60)"),
        log_path=tmp_path / 'child.log', child_pid_path=tmp_path / 'child.pid',
        env={runtime.SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(tmp_path / 'child.identity.json'),
             runtime.SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps({'repo_root': str(tmp_path)})},
    ))


def _close_owned(child):
    if runtime.owner_liveness(child.identity_process_birth) is runtime.OwnerLiveness.ALIVE:
        os.kill(child.pid, signal.SIGTERM)
    try:
        os.waitpid(child.pid, 0)
    except ChildProcessError:
        pass


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


@pytest.mark.parametrize('marker_name', ['identity_path', 'child_pid_path'])
@pytest.mark.parametrize('error', [PermissionError, OSError])
def test_unavailable_marker_is_not_completed_cleanup(tmp_path, monkeypatch, marker_name, error):
    child = _child(tmp_path)
    try:
        assert runtime.terminate_supervised_child(child) is True
        _close_owned(child)
        original = runtime.os.lstat
        def unavailable(path, *args, **kwargs):
            if path == getattr(child, marker_name):
                raise error('marker observation unavailable')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(runtime.os, 'lstat', unavailable)
        assert runtime.supervised_child_is_proven_dead(child) is False
    finally:
        _close_owned(child)
