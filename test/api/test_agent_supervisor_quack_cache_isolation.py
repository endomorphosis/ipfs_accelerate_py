"""Native command publication must preserve other stores' active readers."""

from __future__ import annotations

import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _admit_native_command_route,
    _admitted_observation,
    _isolation_receipt,
    _isolation_server_kwargs,
    _seed,
    build_server,
    probe_quack_capabilities,
)


@pytest.mark.parametrize("command_route", ["broker", "legacy"])
def test_native_command_preserves_other_store_borrowed_attachment(
    tmp_path, monkeypatch, command_route,
):
    servers = []
    read_started, write_done = threading.Event(), threading.Event()
    legacy_stopping, legacy_errors = threading.Event(), []
    reads = []
    reader = legacy_worker = repository_b = None
    state.reset_quack_transport_cache()
    try:
        for name in ("a", "b"):
            root = tmp_path / name
            root.mkdir()
            database = root / "control/control.duckdb"
            _seed(database)
            receipt_path, receipt = _isolation_receipt(root)
            server = build_server(
                database_path=database,
                typed_command_socket_path=root / "owner.sock",
                state_dir=receipt_path.parent,
                **_isolation_server_kwargs(receipt),
                store_id=str(database),
                repository_id="repository:" + name,
                isolation_receipt_path=receipt_path,
                isolation_observer=_admitted_observation,
                capability_probe=lambda **_kwargs: probe_quack_capabilities(),
            )
            server.start()
            servers.append(server)
        a, b = servers
        for name in (
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
            "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION",
            "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(tmp_path))
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", a.identity.store_id)
        old_a = state.open_quack_transport_connection(
            a.identity.listen_uri, token=a._vault.resolve(a.identity.secret_handle),
        )
        assert old_a.execute("SELECT revision FROM tasks").fetchone()[0] == 1
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", b.identity.store_id)
        repository_b = open_intent_repository(b.identity.listen_uri, install_schema=False)
        borrowed_b = state.open_quack_transport_connection(
            b.identity.listen_uri, token=b._vault.resolve(b.identity.secret_handle),
        )
        assert borrowed_b.execute("SELECT revision FROM tasks").fetchone()[0] == 1

        def read_b():
            try:
                # Retain the real B store lock and actual cached attachment
                # across A's successful command and read-replica replacement.
                with repository_b._connection() as native_b_reader:
                    assert native_b_reader is borrowed_b
                    read_started.set()
                    assert write_done.wait(10)
                    reads.append(
                        native_b_reader.execute("SELECT revision FROM tasks").fetchone()[0]
                    )
            except BaseException as error:
                reads.append(error)

        reader = threading.Thread(target=read_b)
        reader.start()
        assert read_started.wait(5)
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", a.identity.store_id)
        legacy_worker = _admit_native_command_route(
            a, monkeypatch, command_route, legacy_stopping, legacy_errors,
        )
        result = state.submit_quack_owner_command(
            "compare_and_set_status",
            {"task_cid_or_alias": "task:test", "expected_revision": 1,
             "status": "in_progress"},
            timeout_seconds=5,
        )
        assert result["changed"] is True
        # Only A's old snapshot is retired. A's next attachment observes the
        # committed revision, while the held B reader remains usable.
        assert old_a._closed is True
        assert state._QUACK_TRANSPORT_CACHE[b.identity.listen_uri] is borrowed_b
        fresh_a = state.open_quack_transport_connection(
            a.identity.listen_uri, token=a._vault.resolve(a.identity.secret_handle),
        )
        assert fresh_a is not old_a
        assert fresh_a.execute("SELECT revision FROM tasks").fetchone()[0] == 2
        write_done.set()
        reader.join(10)
        assert not reader.is_alive()
        assert reads == [1], reads
        assert not legacy_errors, legacy_errors
    finally:
        write_done.set()
        if reader is not None:
            reader.join(10)
        legacy_stopping.set()
        if legacy_worker is not None:
            legacy_worker.join(10)
        if repository_b is not None:
            repository_b.close()
        state.reset_quack_transport_cache()
        for server in reversed(servers):
            server.stop()


class _CachedSession:
    def __init__(self, binding):
        self._quack_mutation_binding = binding
        self._pooled = True
        self.closed = False

    def close(self):
        self.closed = True


def test_store_eviction_preserves_foreign_unknown_and_path_alias_bindings(monkeypatch):
    bindings = [
        {"store_id": "state/a.duckdb"},
        {"store_id": "state/a.duckdb"},
        {"store_id": "state/b.duckdb"},
        {"store_id": "state/./a.duckdb"},
        {}, None, "state/a.duckdb",
    ]
    sessions = [_CachedSession(binding) for binding in bindings]
    cache = {f"quack:127.0.0.1:{32001 + index}": session
             for index, session in enumerate(sessions)}
    monkeypatch.setattr(state, "_QUACK_TRANSPORT_CACHE", cache)
    state.reset_quack_transport_cache(store_id="state/a.duckdb")
    assert [session.closed for session in sessions] == [True, True] + [False] * 5
    assert list(cache.values()) == sessions[2:]


def test_cache_selectors_cannot_be_combined_or_invalid_uri_become_global(monkeypatch):
    session = _CachedSession({"store_id": "state/a.duckdb"})
    cache = {"quack:127.0.0.1:32001": session}
    monkeypatch.setattr(state, "_QUACK_TRANSPORT_CACHE", cache)
    with pytest.raises(state.DuckDBConnectionPolicyError, match="either an endpoint or a store"):
        state.reset_quack_transport_cache(
            "quack:127.0.0.1:32001", store_id="state/a.duckdb",
        )
    state.reset_quack_transport_cache("not-a-quack-uri")
    assert list(cache.values()) == [session]
    assert not session.closed
    state.reset_quack_transport_cache("quack:127.0.0.1:32001")
    assert not cache
    assert not session.closed  # Endpoint-only eviction cannot close a borrower.


def test_legacy_store_lock_contention_publishes_no_request(tmp_path, monkeypatch):
    database = tmp_path / "control.duckdb"
    for name in (
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(tmp_path))
    effects, outcomes = [], []

    def forbidden(*_args, **_kwargs):
        effects.append("token_resolution")
        raise AssertionError("contended command must not acquire credentials or publish")

    monkeypatch.setattr(state, "resolve_quack_attach_token", forbidden)
    lock_path = state.quack_owner_mutation_write_lock_path(str(database))
    assert lock_path is not None

    def contend():
        try:
            state.submit_quack_owner_command(
                "compare_and_set_status",
                {"task_cid_or_alias": "task:test", "expected_revision": 1,
                 "status": "in_progress"},
                timeout_seconds=0.05,
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
    assert not state.quack_owner_command_dir().exists()
    assert not database.exists()
