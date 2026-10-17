"""Status publication must not rearm tasks or provisionally complete goals."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server as module


@pytest.fixture(autouse=True)
def forbid_processes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('status publisher source fixtures cannot launch subprocesses')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def publisher(tmp_path, monkeypatch, *, lifecycle, credential_error=None,
              status_error=None, publish_error=None):
    owner = object.__new__(module.QuackStateServer)
    owner._lifecycle = lifecycle
    owner.config = SimpleNamespace(state_dir=tmp_path)
    calls = []
    public = {'schema': 'authored-status-fixture', 'lifecycle': lifecycle.value,
              'identity': {'store_id': 'authored-fixture-only'}}
    def handoff():
        calls.append('credential_handoff')
        if credential_error:
            raise credential_error
    def forbidden_business_write():
        calls.append('FORBIDDEN_BUSINESS_WRITE')
        raise AssertionError('status publication must never mutate task or goal state')
    def status():
        calls.append('status_projection')
        if status_error:
            raise status_error
        return public
    def publish(path, value, *, mode):
        calls.append(('publish', Path(path), value, mode))
        if publish_error:
            raise publish_error
    owner._ensure_client_token_handoff = handoff
    owner._unstall_false_terminal_blocked = forbidden_business_write
    owner._provisionally_complete_terminal_goals = forbidden_business_write
    owner.status = status
    owner._log = lambda message: calls.append(('log', message))
    monkeypatch.setattr(module, '_atomic_write_json', publish)
    return owner, calls, public


def test_ready_status_keeps_handoff_projection_and_private_atomic_publication(tmp_path, monkeypatch):
    owner, calls, public = publisher(tmp_path, monkeypatch, lifecycle=module.ServerLifecycle.READY)
    owner._write_status()
    assert calls == ['credential_handoff', 'status_projection',
                     ('publish', tmp_path / module.STATUS_FILENAME, public, 0o600)]


@pytest.mark.parametrize('lifecycle', [module.ServerLifecycle.CREATED,
    module.ServerLifecycle.STARTING, module.ServerLifecycle.STOPPING,
    module.ServerLifecycle.STOPPED, module.ServerLifecycle.FAILED])
def test_nonready_status_does_not_republish_credentials_or_maintain_business_state(tmp_path, monkeypatch, lifecycle):
    owner, calls, public = publisher(tmp_path, monkeypatch, lifecycle=lifecycle)
    owner._write_status()
    assert calls == ['status_projection',
                     ('publish', tmp_path / module.STATUS_FILENAME, public, 0o600)]


@pytest.mark.parametrize('stage,error_type', [
    ('credential', OSError),
    ('status', ValueError),
    ('publication', PermissionError),
])
def test_existing_error_boundary_logs_only_error_type_without_business_writes(tmp_path, monkeypatch, stage, error_type):
    error = error_type('fixture-private-message-must-not-be-logged')
    args = {'credential_error': None, 'status_error': None, 'publish_error': None}
    args[{'credential':'credential_error','status':'status_error','publication':'publish_error'}[stage]] = error
    owner, calls, public = publisher(tmp_path, monkeypatch, lifecycle=module.ServerLifecycle.READY, **args)
    owner._write_status()
    assert calls[-1] == ('log', 'status write warning: ' + error_type.__name__)
    assert 'FORBIDDEN_BUSINESS_WRITE' not in calls
    assert 'fixture-private-message-must-not-be-logged' not in repr(calls)
    if stage == 'credential':
        assert calls[:-1] == ['credential_handoff']
    elif stage == 'status':
        assert calls[:-1] == ['credential_handoff', 'status_projection']
    else:
        assert calls[:-1] == ['credential_handoff', 'status_projection',
                              ('publish', tmp_path / module.STATUS_FILENAME, public, 0o600)]
