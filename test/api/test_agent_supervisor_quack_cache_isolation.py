"""A replica refresh must preserve other databases' borrowed attachments."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _admitted_observation,
    _isolation_receipt,
    _isolation_server_kwargs,
    _seed,
    build_server,
    probe_quack_capabilities,
)


@pytest.fixture
def attachments():
    import duckdb

    state.reset_quack_transport_cache()
    wrappers = []

    def add(port, store_id=None):
        uri = f"quack:127.0.0.1:{port}"
        connection = state.DuckDBConnection.wrap(duckdb.connect(":memory:"))
        connection.execute("CREATE TABLE retained AS SELECT 7 AS value")
        connection._pooled = True
        connection._quack_uri = uri
        if store_id is not None:
            connection._quack_mutation_binding = {"store_id": store_id}
        with state._QUACK_ATTACH_LOCK:
            state._QUACK_TRANSPORT_CACHE[uri] = connection
        wrappers.append(connection)
        return uri, connection

    yield add
    state.reset_quack_transport_cache()
    for connection in wrappers:
        connection._discard_pooled_connection()


def test_endpoint_eviction_preserves_both_borrowed_connections(attachments):
    uri_a, a = attachments(41351, "store:a")
    uri_b, b = attachments(41352, "store:b")
    b.execute("BEGIN TRANSACTION")
    state.reset_quack_transport_cache(uri_a)
    assert a.execute("SELECT value FROM retained").fetchone()[0] == 7
    assert b.execute("SELECT value FROM retained").fetchone()[0] == 7
    b.rollback()
    assert uri_a not in state._QUACK_TRANSPORT_CACHE
    assert state._QUACK_TRANSPORT_CACHE[uri_b] is b


def test_invalid_endpoint_does_not_reset_other_connections(attachments):
    uri, borrowed = attachments(41353, "store:a")
    state.reset_quack_transport_cache("not-an-admitted-quack-endpoint")
    assert state._QUACK_TRANSPORT_CACHE[uri] is borrowed
    assert borrowed.execute("SELECT value FROM retained").fetchone()[0] == 7


def test_store_eviction_preserves_unrelated_and_unbound_readers(attachments):
    uri_a, a = attachments(41354, "store:a")
    uri_b, b = attachments(41355, "store:b")
    uri_unknown, unknown = attachments(41356)
    b.execute("BEGIN TRANSACTION")
    state.reset_quack_transport_cache(store_id="store:a")
    assert uri_a not in state._QUACK_TRANSPORT_CACHE
    with pytest.raises(state.DuckDBConnectionPolicyError):
        a.execute("SELECT value FROM retained")
    for uri, connection in ((uri_b, b), (uri_unknown, unknown)):
        assert state._QUACK_TRANSPORT_CACHE[uri] is connection
        assert connection.execute("SELECT value FROM retained").fetchone()[0] == 7
    b.rollback()


def test_conflicting_cache_selectors_deny_before_effects(attachments):
    uri, borrowed = attachments(41357, "store:a")
    with pytest.raises(state.DuckDBConnectionPolicyError):
        state.reset_quack_transport_cache(uri, store_id="store:a")
    assert state._QUACK_TRANSPORT_CACHE[uri] is borrowed
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
            receipt_path, receipt = _isolation_receipt(root)
            server = build_server(database_path=database, typed_command_socket_path=root/'owner.sock',
                state_dir=receipt_path.parent, **_isolation_server_kwargs(receipt), store_id=str(database),
                repository_id='repository:'+name, isolation_receipt_path=receipt_path,
                isolation_observer=_admitted_observation,
                capability_probe=lambda **kwargs: probe_quack_capabilities())
            server.start()
            servers.append(server)
        a,b = servers
        for name in ('IPFS_ACCELERATE_AGENT_STATE_STORE_ID', 'IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION',
                     'IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT', str(tmp_path))
        # This exact retained lock and connection are obtained while B is the
        # admitted store. Tokens belong only to these disposable test owners.
        monkeypatch.setenv('IPFS_ACCELERATE_AGENT_STATE_STORE_ID', b.identity.store_id)
        repository_b = open_intent_repository(b.identity.listen_uri, install_schema=False)
        borrowed_b = state.open_quack_transport_connection(b.identity.listen_uri,
            token=b._vault.resolve(b.identity.secret_handle))
        assert borrowed_b.execute('SELECT revision FROM tasks').fetchone()[0] == 1
        def read_b():
            try:
                with repository_b._connection() as native_b_reader:
                    assert native_b_reader is borrowed_b
                    read_started.set()
                    assert write_done.wait(10)
                    reads.append(native_b_reader.execute('SELECT revision FROM tasks').fetchone()[0])
            except BaseException as error:
                reads.append(error)
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


def test_native_broker_command_holds_read_lock_through_refresh_and_cache_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real replica reader cannot cross either native publication boundary."""
    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    receipt_path, receipt = _isolation_receipt(tmp_path)
    server = build_server(
        database_path=database, state_dir=receipt_path.parent,
        typed_command_socket_path=tmp_path / "owner.sock",
        **_isolation_server_kwargs(receipt), store_id=str(database),
        repository_id="repository:test", isolation_receipt_path=receipt_path,
        isolation_observer=_admitted_observation,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
    )
    identity = server.start()
    _admit_test_broker(server, monkeypatch)
    for key, value in {
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": identity.secret_handle,
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": str(database),
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": str(identity.generation),
        "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": str(identity.schema_revision),
        "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    source = DatabaseTaskSource(identity.listen_uri, install_schema=False)
    refresh_entered, refresh_release = threading.Event(), threading.Event()
    eviction_entered, eviction_release = threading.Event(), threading.Event()
    read_started, read_done = threading.Event(), threading.Event()
    writes, reads = [], []
    refreshes = []
    refresh = server._refresh_read_replica  # noqa: SLF001
    evict = state.reset_quack_transport_cache

    def paused_refresh():
        refreshes.append("refresh")
        refresh_entered.set()
        assert refresh_release.wait(10)
        return refresh()

    def paused_eviction(*args, **kwargs):
        eviction_entered.set()
        assert eviction_release.wait(10)
        return evict(*args, **kwargs)

    def write():
        try:
            writes.append(source.compare_and_set_status(
                "task:test", expected_revision=1, status="in_progress",
            ))
        except BaseException as error:
            writes.append(error)

    def read():
        read_started.set()
        try:
            reads.append(source.get_task("task:test"))
        except BaseException as error:
            reads.append(error)
        finally:
            read_done.set()

    writer, reader = threading.Thread(target=write), threading.Thread(target=read)
    try:
        # Prime a real attachment to the replica that the writer will replace.
        assert source.get_task("task:test").revision == 1
        monkeypatch.setattr(server, "_refresh_read_replica", paused_refresh)
        monkeypatch.setattr(state, "reset_quack_transport_cache", paused_eviction)
        writer.start()
        assert refresh_entered.wait(5)
        reader.start()
        assert read_started.wait(5)
        assert not read_done.wait(0.05), reads
        refresh_release.set()
        assert eviction_entered.wait(5)
        assert not read_done.wait(0.05), reads
        eviction_release.set()
        writer.join(10)
        reader.join(10)
        assert not writer.is_alive() and not reader.is_alive()
        assert len(writes) == len(reads) == 1
        assert not isinstance(writes[0], BaseException), writes
        assert not isinstance(reads[0], BaseException), reads
        assert writes[0].revision == reads[0].revision == 2
        assert reads[0].status == "in_progress"
        assert refreshes == ["refresh"]
        assert server.status()["read_replica"]["live"] is True
    finally:
        refresh_release.set()
        eviction_release.set()
        if writer.ident is not None:
            writer.join(10)
        if reader.ident is not None:
            reader.join(10)
        monkeypatch.setattr(state, "reset_quack_transport_cache", evict)
        source.close()
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
