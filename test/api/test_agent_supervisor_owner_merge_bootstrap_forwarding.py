"""The native supervisor forwards paired scope only on its inherited descriptor."""
from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as supervisor
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon


@pytest.mark.parametrize('config_cid,plan_cid,descriptor', [
    ('config:current', 'plan:current', 99),
    ('config:current', '', 99),
    ('', 'plan:current', 99),
    ('config:current', 'plan:current', -1),
])
def test_paired_scope_round_trips_to_daemon_or_refuses(tmp_path, config_cid, plan_cid, descriptor):
    todo = tmp_path / 'board.md'
    todo.write_text('# Tasks\n')
    argv = ['--implement', '--todo-path', str(todo), '--state-dir', str(tmp_path / 'state'),
        '--worktree-root', str(tmp_path / 'worktrees'),
        '--task-source-kind', 'duckdb', '--authority-mode', 'quack',
        '--endpoint-secret-handle', 'env://QUACK_TOKEN',
        '--state-store-generation', 'generation-1', '--state-schema-revision', 'schema-1',
        '--event-store-path', 'state/events', '--runtime-registry-path', 'state/registry',
        '--export-profile', 'operator-export',
        '--quack-endpoint', 'quack:127.0.0.1:45123', '--state-store-id', 'control.duckdb',
        '--owner-merge-config-cid', config_cid, '--owner-merge-plan-cid', plan_cid,
        '--state-owner-bootstrap-fd', str(descriptor), '--state-owner-bootstrap-store-id', 'control.duckdb']
    args = supervisor.parse_args(argv)
    config = supervisor.supervisor_config_from_args(args, repo_root=tmp_path)
    instance = supervisor.PortalImplementationSupervisor(config)
    if not config_cid or not plan_cid or descriptor < 3:
        with pytest.raises(RuntimeError, match='paired owner bootstrap'):
            instance._build_daemon_command()
        return
    command = instance._build_daemon_command()
    entry = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'
    child = daemon.parse_args(command[command.index(entry) + 1:])
    assert child.owner_merge_config_cid == config_cid
    assert child.owner_merge_plan_cid == plan_cid
    assert child.state_owner_bootstrap_fd == descriptor
    assert command.count('--owner-merge-config-cid') == 1
    assert command.count('--owner-merge-plan-cid') == 1


@pytest.mark.parametrize('failure', ['construct', 'bind', 'run'])
def test_daemon_closes_paired_channels_on_every_post_attach_exit(tmp_path, monkeypatch, failure):
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        owner_merge_bootstrap as bootstrap, quack_state_client, typed_database_task_source,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_TOKEN_ENV, TYPED_STATE_OWNER_SOCKET_ENV,
    )
    endpoint = 'quack:127.0.0.1:45123'
    closed, requested, bound = [], [], []
    class Credentials:
        client_id = 'database-implementation-daemon:native-lane-0'
        process_birth_id = 'birth:current'
        server_id = 'server:task'
        execution_route_policy = None
        def install_environment(self):
            os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = 'private-task-token'
            os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = '/tmp/private-task-socket'
    credentials = Credentials()
    credentials.endpoint = endpoint
    credentials.store_id = 'control.duckdb'
    runtime = object()
    attachment = SimpleNamespace(runtime=runtime, close=lambda: closed.append('paired'))
    def attach(**kwargs):
        assert kwargs['lane_id'] == '0'
        assert kwargs['admitted_config_cid'] == 'config:current'
        assert kwargs['admitted_plan_cid'] == 'plan:current'
        assert kwargs['attempt_root'] == tmp_path / 'state/native_database_portal_attempts'
        return attachment
    bundle = SimpleNamespace(task=credentials, attach_merge_runtime=attach)
    def request(fd, **kwargs):
        requested.append((fd, kwargs))
        return bundle
    class Client:
        def __init__(self, **kwargs):
            pass
        def attach(self, *args, **kwargs):
            pass
        def close(self):
            closed.append('client')
    class Source:
        def __init__(self, client, **kwargs):
            self.client = client
        def close(self):
            closed.append('task')
            self.client.close()
    class Daemon:
        def __init__(self, **kwargs):
            self.source = kwargs['task_source']
            if failure == 'construct':
                raise RuntimeError('construct failed')
        def close_event_runtime(self):
            closed.append('daemon')
            self.source.close()
        def run_once(self, **kwargs):
            raise RuntimeError('run failed')
        def run_pass(self, **kwargs):
            raise RuntimeError('run failed')
    def bind(*args, **kwargs):
        bound.append(kwargs)
        assert kwargs['owner_merge_runtime'] is runtime
        assert kwargs['admitted_owner_merge_config_cid'] == 'config:current'
        if failure == 'bind':
            raise RuntimeError('bind failed')
    monkeypatch.setattr(bootstrap, 'request_owner_merge_bootstrap', request)
    monkeypatch.setattr(quack_state_client, 'QuackStateClient', Client)
    monkeypatch.setattr(typed_database_task_source, 'TypedDatabaseTaskSource', Source)
    monkeypatch.setattr(daemon, 'DatabaseImplementationDaemon', Daemon)
    monkeypatch.setattr(daemon, 'bind_database_portal_execution_from_args', bind)
    monkeypatch.setattr(daemon, 'database_program_from_daemon_namespace', lambda args: SimpleNamespace(
        authority_mode='quack', task_source_kind='duckdb', store_id='control.duckdb',
        quack_endpoint=endpoint, schema_revision='schema:test'))
    monkeypatch.setattr(daemon, 'resolve_database_implementation_paths', lambda *args, **kwargs: {
        'database_path': tmp_path / 'task.duckdb', 'coordination_path': tmp_path / 'coord.duckdb',
        'execution_path': tmp_path / 'execution.duckdb'})
    with pytest.raises(RuntimeError, match=failure + ' failed'):
        daemon.main(['--once', '--todo-path', str(tmp_path / 'board.md'),
            '--state-dir', str(tmp_path / 'state'), '--state-prefix', 'native',
            '--task-source-kind', 'duckdb', '--authority-mode', 'quack',
            '--quack-endpoint', endpoint, '--state-store-id', 'control.duckdb',
            '--owner-session-id', 'native-lane-0', '--state-owner-bootstrap-fd', '99',
            '--state-owner-bootstrap-store-id', 'control.duckdb',
            '--owner-merge-config-cid', 'config:current', '--owner-merge-plan-cid', 'plan:current'])
    assert len(requested) == 1
    assert closed.count('paired') == 1
    assert closed.count('task') == 1
    assert TYPED_STATE_OWNER_TOKEN_ENV not in os.environ
    assert TYPED_STATE_OWNER_SOCKET_ENV not in os.environ
