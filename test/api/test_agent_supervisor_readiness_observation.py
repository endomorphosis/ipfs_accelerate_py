from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.readiness_observation import observe_readiness


@pytest.mark.parametrize('include_blocked', [False, True])
def test_real_task_dependencies_share_one_client(tmp_path, monkeypatch, include_blocked):
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource

    path = tmp_path / 'readiness.duckdb'
    with DatabaseTaskSource(path) as source:
        source.intent.upsert_goal(goal_cid='goal:test', goal_alias='TEST', title='Test')
        for index in range(12):
            source.intent.upsert_task(task_cid=f'task:{index}', task_alias=f'TEST-{index}',
                                      goal_cid='goal:test', status='ready')
    opened = []
    original = intent_repository.open_duckdb_connection
    def counted(*args, **kwargs):
        conn = original(*args, **kwargs)
        opened.append(conn)
        return conn
    monkeypatch.setattr(intent_repository, 'open_duckdb_connection', counted)
    with DatabaseTaskSource(path, install_schema=False) as source:
        result = observe_readiness(source, in_scope=lambda task: True,
                                   blocked_recoverable=lambda task, source: False,
                                   include_blocked=include_blocked)
    assert len(result.ready) == 12
    assert result.active == ()
    assert len(opened) == 1


@pytest.mark.parametrize('failure', ['oom', 'truncated'])
def test_required_read_failure_closes_session_without_observation(failure):
    exits = []
    @contextmanager
    def session():
        try:
            yield
        finally:
            exits.append(True)
    def ready_tasks(**kwargs):
        if failure == 'oom':
            raise MemoryError('owner read failed')
        return SimpleNamespace(tasks=(), next_cursor='more', revision=1)
    source = SimpleNamespace(intent=SimpleNamespace(read_session=session),
                             ready_tasks=ready_tasks,
                             list_tasks=lambda **kw: SimpleNamespace(tasks=(), next_cursor='', revision=1))
    with pytest.raises((MemoryError, RuntimeError)):
        observe_readiness(source, in_scope=lambda task: True,
                          blocked_recoverable=lambda task, source: False, include_blocked=True)
    assert exits == [True]
