"""Portal startup reads existing DuckDB catalogs without schema mutation.

Real DuckDB files and READ_ONLY attachments exercise SQL/schema behavior. The
Quack transport opener is replaced only by that real local attachment fixture;
no network, provider, DuckLake extension or external database is consulted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import portal_task_state_control_plane as portal
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

URI = 'quack:127.0.0.1:41307'


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('portal read-only catalog tests must not launch an external process')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(portal, '_BOUND_STATE_TARGETS', {})
    for name in (portal.ENV_QUACK, portal.ENV_DUCKDB, portal.ENV_DUCKLAKE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(portal, '_project_ducklake', lambda *_args, **_kwargs: {
        'status': 'authored_fixture_only', 'authoritative': False, 'completion_authority': False,
    })


def _config(tmp_path):
    return {'threads': 1, 'memory_limit': '64MB', 'temp_directory': str(tmp_path / 'spill'),
            'autoinstall_known_extensions': 'false', 'autoload_known_extensions': 'false'}


def _database(tmp_path, *, schema='valid', portal_body=None, handoff_body=None):
    path = tmp_path / 'owner.duckdb'
    connection = duckdb.connect(str(path), config=_config(tmp_path))
    try:
        if schema != 'absent':
            sql = portal.DDL
            if schema == 'wrong_type':
                sql = sql.replace('revision BIGINT', 'revision VARCHAR')
            if schema == 'missing_column':
                sql = sql.replace('    body_json VARCHAR NOT NULL,\n', '')
            connection.execute(sql)
            if schema != 'missing_handoff':
                connection.execute(portal.HANDOFF_DDL)
            if schema == 'extra_column':
                connection.execute(f'ALTER TABLE {portal.TABLE} ADD COLUMN future_metadata VARCHAR')
            if portal_body is not None:
                connection.execute(f'INSERT INTO {portal.TABLE} (lane_key, body_json, revision, completion_authority, updated_at) VALUES (?, ?, 1, FALSE, ?)',
                                   ['lane', json.dumps(portal_body), 'fixture-time'])
            if handoff_body is not None:
                connection.execute(f'INSERT INTO {portal.HANDOFF_TABLE} VALUES (?, ?, 1, FALSE, ?)',
                                   ['handoff', json.dumps(handoff_body), 'fixture-time'])
        connection.execute('CREATE TABLE unrelated_owner_state (value BIGINT)')
        connection.execute('INSERT INTO unrelated_owner_state VALUES (17)')
    finally:
        connection.close()
    return path


class RecordedReadOnlyConnection:
    def __init__(self, path, tmp_path):
        self.raw = duckdb.connect(':memory:', config=_config(tmp_path))
        escaped = str(path).replace("'", "''")
        self.raw.execute(f"ATTACH '{escaped}' AS control_plane (READ_ONLY)")
        self.raw.execute('USE control_plane')
        self.statements = []
        self.closed = False

    def execute(self, sql, parameters=None):
        self.statements.append(str(sql))
        assert str(sql).lstrip().upper().startswith(('SELECT ', 'DESCRIBE ')), sql
        return self.raw.execute(sql) if parameters is None else self.raw.execute(sql, parameters)

    def close(self):
        self.closed = True
        self.raw.close()


def _readonly_opener(monkeypatch, path, tmp_path):
    opened = []
    def open_existing(target, *, prefer_quack, **kwargs):
        assert target == URI and prefer_quack is True
        connection = RecordedReadOnlyConnection(path, tmp_path)
        opened.append(connection)
        return connection
    monkeypatch.setattr(duckdb_state, 'open_duckdb_connection', open_existing)
    return opened


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.stat().st_mtime_ns


def test_actual_readonly_attachment_allows_portal_load_without_ddl(tmp_path, monkeypatch):
    path = _database(tmp_path, portal_body={'active_task_id': 'fixture-task', 'completion_authority': True})
    before = _fingerprint(path)
    opened = _readonly_opener(monkeypatch, path, tmp_path)
    monkeypatch.setenv(portal.ENV_QUACK, URI)
    loaded = portal.load_portal_task_state('lane', state_path=tmp_path / 'lane.json')
    assert loaded['active_task_id'] == 'fixture-task'
    assert loaded['completion_authority'] is False
    assert opened and all(connection.closed for connection in opened)
    assert all(sql.lstrip().upper().startswith(('SELECT ', 'DESCRIBE '))
               for connection in opened for sql in connection.statements)
    assert _fingerprint(path) == before


def test_portal_task_state_load_startup_path_uses_existing_readonly_row(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState
    path = _database(tmp_path, portal_body={'active_task_id': 'fixture-startup', 'active_phase': 'observe'})
    before = _fingerprint(path)
    opened = _readonly_opener(monkeypatch, path, tmp_path)
    state_path = tmp_path / 'lane.json'
    portal.bind_task_state_control_plane(state_path, URI)
    state = PortalTaskState.load(state_path)
    assert state.active_task_id == 'fixture-startup'
    assert state.active_phase == 'observe'
    assert not state_path.exists()
    assert all(connection.closed for connection in opened)
    assert _fingerprint(path) == before


@pytest.mark.parametrize('schema', ('absent', 'missing_handoff', 'missing_column', 'wrong_type'))
def test_readonly_missing_or_malformed_schema_refuses_and_closes(tmp_path, monkeypatch, schema):
    path = _database(tmp_path, schema=schema)
    before = _fingerprint(path)
    opened = _readonly_opener(monkeypatch, path, tmp_path)
    with pytest.raises(portal.PortalTaskStateControlPlaneError, match='schema (unavailable|differs)'):
        portal._connect(URI)
    assert len(opened) == 1 and opened[0].closed
    assert _fingerprint(path) == before


def test_reader_accepts_compatible_extra_owner_columns(tmp_path, monkeypatch):
    path = _database(tmp_path, schema='extra_column')
    before = _fingerprint(path)
    opened = _readonly_opener(monkeypatch, path, tmp_path)
    connection = portal._connect(URI)
    connection.close()
    assert opened[0].closed and _fingerprint(path) == before


def test_readonly_handoff_load_does_not_initialize_catalog(tmp_path, monkeypatch):
    body = {'merged': False, 'writes_repository': False, 'completion_authority': True}
    path = _database(tmp_path, handoff_body=body)
    before = _fingerprint(path)
    opened = _readonly_opener(monkeypatch, path, tmp_path)
    loaded = portal.load_merge_handoff('handoff', target=URI)
    assert loaded['merged'] is loaded['writes_repository'] is loaded['completion_authority'] is False
    assert opened[0].closed and _fingerprint(path) == before


def test_duckdb_fixture_really_enforces_readonly(tmp_path):
    path = _database(tmp_path)
    before = _fingerprint(path)
    connection = RecordedReadOnlyConnection(path, tmp_path)
    try:
        with pytest.raises(Exception, match='(?i)read.?only'):
            connection.raw.execute(f"INSERT INTO {portal.TABLE} VALUES ('forbidden', '{{}}', 1, FALSE, 'now')")
    finally:
        connection.close()
    assert _fingerprint(path) == before


def test_local_writer_fresh_catalog_initializes_both_relations(tmp_path):
    path = tmp_path / 'writer' / 'lane.duckdb'
    connection = portal._connect(str(path))
    try:
        tables = {row[0] for row in connection.execute('SHOW TABLES').fetchall()}
        assert {portal.TABLE, portal.HANDOFF_TABLE} <= tables
        connection.execute(f'INSERT INTO {portal.TABLE} VALUES (?, ?, 1, FALSE, ?)',
                           ['lane', '{"active_task_id":"writer-fixture"}', 'fixture-time'])
    finally:
        connection.close()
    assert path.is_file()
    assert portal.load_portal_task_state('lane', state_path=_bound_state(path, tmp_path), create=False)['active_task_id'] == 'writer-fixture'


def _bound_state(path, tmp_path):
    state_path = tmp_path / 'lane.json'
    portal.bind_task_state_control_plane(state_path, str(path))
    return state_path


def test_create_false_missing_local_catalog_does_not_create_directory_or_file(tmp_path, monkeypatch):
    path = tmp_path / 'missing-parent' / 'lane.duckdb'
    monkeypatch.setenv(portal.ENV_DUCKDB, str(path))
    assert portal.load_portal_task_state('lane', create=False) == {}
    assert not path.parent.exists()
    with pytest.raises(portal.PortalTaskStateControlPlaneError, match='existing catalog'):
        portal._connect(str(path), initialize=False)
    assert not path.parent.exists()
    assert portal.load_merge_handoff('handoff', target=str(path)) == {}
    assert not path.parent.exists()


def test_existing_local_reader_missing_schema_refuses_without_initializing(tmp_path, monkeypatch):
    path = _database(tmp_path, schema='absent')
    before = _fingerprint(path)
    monkeypatch.setenv(portal.ENV_DUCKDB, str(path))
    with pytest.raises(portal.PortalTaskStateControlPlaneError, match='schema unavailable'):
        portal.load_portal_task_state('lane', create=False)
    assert _fingerprint(path) == before


def test_normal_board_writer_and_existing_board_reader_share_original_schema(tmp_path, monkeypatch):
    path = tmp_path / 'board-control-plane' / 'control.duckdb'
    connection = portal._connect(str(path))
    try:
        connection.execute('INSERT INTO artefact_generic VALUES (?, ?, ?, ?, ?)',
                           ['portal-task-state', 'portal_task_state', 'lane', '{"active_task_id":"board-fixture"}', 1])
    finally:
        connection.close()
    before = _fingerprint(path)
    monkeypatch.setenv(portal.ENV_DUCKDB, str(path))
    loaded = portal.load_portal_task_state('lane', create=False)
    assert loaded['active_task_id'] == 'board-fixture' and loaded['completion_authority'] is False
    assert _fingerprint(path) == before


def test_exclusive_control_database_name_guard_still_refuses_before_open(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('exclusive-name guard must precede connection opening')
    monkeypatch.setattr(duckdb_state, 'open_duckdb_connection', forbidden)
    with pytest.raises(portal.PortalTaskStateControlPlaneError, match='exclusive extra-gate'):
        portal._connect(str(tmp_path / 'control.duckdb'))
    assert not (tmp_path / 'control.duckdb').exists()
