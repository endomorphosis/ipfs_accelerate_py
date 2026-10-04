"""Live native owner publication preserves independently borrowed readers.

The current transport opens individual connections and serves the owner's
live database. It has no attachment pool or replica refresh to simulate.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
from test.common.native_quack_fixtures import _seed
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities


@pytest.fixture
def attachments():
    import duckdb

    state.reset_quack_transport_cache()
    wrappers = []

    def add(port):
        uri = f"quack:127.0.0.1:{port}"
        connection = state.DuckDBConnection.wrap(duckdb.connect(":memory:"))
        connection.execute("CREATE TABLE retained AS SELECT 7 AS value")
        wrappers.append(connection)
        return uri, connection

    yield add
    state.reset_quack_transport_cache()
    for connection in wrappers:
        connection.close()


def test_endpoint_publication_hook_preserves_both_borrowed_connections(attachments):
    uri_a, a = attachments(41351)
    uri_b, b = attachments(41352)
    b.execute("BEGIN TRANSACTION")
    state.reset_quack_transport_cache(uri_a)
    assert a.execute("SELECT value FROM retained").fetchone()[0] == 7
    assert b.execute("SELECT value FROM retained").fetchone()[0] == 7
    b.rollback()


def test_invalid_endpoint_does_not_reset_other_connections(attachments):
    uri, borrowed = attachments(41353)
    state.reset_quack_transport_cache("not-an-admitted-quack-endpoint")
    assert borrowed.execute("SELECT value FROM retained").fetchone()[0] == 7


def test_store_publication_hook_does_not_own_individual_readers(attachments):
    uri_a, a = attachments(41354)
    uri_b, b = attachments(41355)
    uri_unknown, unknown = attachments(41356)
    b.execute("BEGIN TRANSACTION")
    state.reset_quack_transport_cache(store_id="store:a")
    for connection in (a, b, unknown):
        assert connection.execute("SELECT value FROM retained").fetchone()[0] == 7
    b.rollback()


def test_conflicting_cache_selectors_deny_before_effects(attachments):
    uri, borrowed = attachments(41357)
    with pytest.raises(state.DuckDBConnectionPolicyError):
        state.reset_quack_transport_cache(uri, store_id="store:a")
    assert borrowed.execute("SELECT value FROM retained").fetchone()[0] == 7


def test_native_command_does_not_close_other_store_borrowed_attachment(tmp_path, monkeypatch):
    servers = []
    read_started, write_done = threading.Event(), threading.Event()
    reads = []
    reader = None
    state.reset_quack_transport_cache()
    try:
        for name in ('a', 'b'):
            root = tmp_path / name
            root.mkdir()
            database = root / 'control/control.duckdb'
            _seed(database)
            server = build_server(database_path=database, typed_command_socket_path=root/'owner.sock',
                state_dir=root/'control/quack-owner', store_id=str(database),
                repository_id='repository:'+name,
                capability_probe=lambda **kwargs: probe_quack_capabilities())
            server.start()
            servers.append(server)
        a,b = servers
        for name in ('IPFS_ACCELERATE_AGENT_STATE_STORE_ID', 'IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION',
                     'IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT', str(tmp_path))
        # Keep an actual read transaction on B across A's typed publication.
        # Tokens belong only to these disposable test owners.
        monkeypatch.setenv('IPFS_ACCELERATE_AGENT_STATE_STORE_ID', b.identity.store_id)
        def read_b():
            borrowed_b = None
            try:
                borrowed_b = state.open_quack_transport_connection(b.identity.listen_uri,
                    token=b._vault.resolve(b.identity.secret_handle))
                borrowed_b.execute('BEGIN TRANSACTION')
                assert borrowed_b.execute('SELECT revision FROM tasks').fetchone()[0] == 1
                read_started.set()
                assert write_done.wait(10)
                reads.append(borrowed_b.execute('SELECT revision FROM tasks').fetchone()[0])
                borrowed_b.rollback()
            except BaseException as error:
                reads.append(error)
            finally:
                if borrowed_b is not None:
                    borrowed_b.close()
        reader = threading.Thread(target=read_b)
        reader.start()
        assert read_started.wait(5)
        for name, value in a.start_supervisor_grant_broker().items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
        monkeypatch.setenv('IPFS_ACCELERATE_AGENT_STATE_STORE_ID', a.identity.store_id)
        result = state.submit_quack_owner_command('compare_and_set_status',
            {'task_cid_or_alias':'task:test','expected_revision':1,'status':'in_progress'}, timeout_seconds=5)
        assert result['changed'] is True
        write_done.set()
        reader.join(10)
        assert not reader.is_alive()
        assert reads == [1], reads
    finally:
        write_done.set()
        if reader is not None:
            reader.join(10)
        state.reset_quack_transport_cache()
        for server in reversed(servers):
            server.stop()


def _admit_test_broker(server, monkeypatch):
    for name, value in server.start_supervisor_grant_broker().items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)


def test_native_broker_serializes_commands_through_publication_hook(tmp_path, monkeypatch):
    """The live owner commits atomically; its client lock spans publication.

    Borrowed readers can see a committed revision immediately. A second typed
    writer must still wait until the first client's publication hook returns.
    """
    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    server = build_server(database_path=database, state_dir=tmp_path / "control/quack-owner",
        typed_command_socket_path=tmp_path / "owner.sock", store_id=str(database),
        repository_id="repository:test", capability_probe=lambda **_kwargs: probe_quack_capabilities())
    identity = server.start()
    _admit_test_broker(server, monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(tmp_path))
    publication_entered, publication_release = threading.Event(), threading.Event()
    second_started, second_done = threading.Event(), threading.Event()
    results, publications = [], []
    reset = state.reset_quack_transport_cache

    def paused_publication(*args, **kwargs):
        publications.append(kwargs)
        if len(publications) == 1:
            publication_entered.set()
            assert publication_release.wait(10)
        return reset(*args, **kwargs)

    def write(expected_revision, status):
        if expected_revision == 2:
            second_started.set()
        try:
            result = state.submit_quack_owner_command("compare_and_set_status", {
                "task_cid_or_alias": "task:test", "expected_revision": expected_revision,
                "status": status,
            }, timeout_seconds=10)
            results.append((expected_revision, result))
        except BaseException as error:
            results.append((expected_revision, error))
        finally:
            if expected_revision == 2:
                second_done.set()

    first = threading.Thread(target=write, args=(1, "in_progress"))
    second = threading.Thread(target=write, args=(2, "blocked"))
    borrowed = None
    try:
        borrowed = state.open_quack_transport_connection(identity.listen_uri,
            token=server._vault.resolve(identity.secret_handle))
        assert dict(borrowed.execute("SELECT revision, status FROM tasks").fetchone()) == {"revision": 1, "status": "ready"}
        monkeypatch.setattr(state, "reset_quack_transport_cache", paused_publication)
        first.start()
        assert publication_entered.wait(5), results
        assert dict(borrowed.execute("SELECT revision, status FROM tasks").fetchone()) == {"revision": 2, "status": "in_progress"}
        second.start()
        assert second_started.wait(5)
        assert not second_done.wait(.05), results
        assert results == []
        publication_release.set()
        first.join(10)
        second.join(10)
        assert not first.is_alive() and not second.is_alive()
        assert len(results) == 2, results
        assert all(isinstance(result, dict) and result["changed"] is True for _, result in results), results
        assert dict(borrowed.execute("SELECT revision, status FROM tasks").fetchone()) == {"revision": 3, "status": "blocked"}
        assert publications == [{"store_id": str(database)}] * 2
    finally:
        publication_release.set()
        for worker in (first, second):
            if worker.ident is not None:
                worker.join(10)
        monkeypatch.setattr(state, "reset_quack_transport_cache", reset)
        if borrowed is not None:
            borrowed.close()
        server.stop()


def test_native_broker_command_lock_contention_has_no_grant_or_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner

    database = tmp_path / "control.duckdb"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET", str(tmp_path / "broker.sock"))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD", "123456")
    effects = []

    def forbidden(**_kwargs):
        effects.append("grant")
        raise AssertionError("contended store requested a grant")

    monkeypatch.setattr(typed_state_owner, "request_database_task_command_credential", forbidden)
    lock_path = state.quack_owner_mutation_write_lock_path(str(database))
    assert lock_path is not None
    outcomes = []

    def contend():
        try:
            state.submit_quack_owner_command(
                "compare_and_set_status",
                {"task_cid_or_alias": "task:test", "expected_revision": 1,
                 "status": "in_progress"}, timeout_seconds=0.05,
            )
        except BaseException as error:
            outcomes.append(error)

    with state.exclusive_file_lock(lock_path, timeout_seconds=1):
        worker = threading.Thread(target=contend)
        worker.start()
        worker.join(2)
        assert not worker.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], TimeoutError), outcomes
    assert effects == []
    assert not database.exists()
