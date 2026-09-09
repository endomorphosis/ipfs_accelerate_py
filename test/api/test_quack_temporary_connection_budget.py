"""Temporary Quack clients cap native workers at creation, before any SQL."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server as server
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state

URI = "quack:127.0.0.1:45123"
BINDING = {
    "server_id": "server:test",
    "store_id": "store:test",
    "database_uuid": "database:test",
    "schema_revision": 4,
    "generation": 1,
    "process_birth_id": "birth:test",
    "listen_uri": URI,
    "extension_fingerprint": "extension:test",
    "schema_fingerprint": "schema:test",
}


class Client:
    def __init__(self, *, fail_at=None, error=None, missing_identity=False):
        self.fail_at = fail_at
        self.error = error if error is not None else RuntimeError("injected client failure")
        self.missing_identity = missing_identity
        self.statements = []
        self.close_count = 0

    def execute(self, sql, parameters=None):
        self.statements.append(sql)
        if self.fail_at and self.fail_at in sql:
            raise self.error
        return self

    def fetchall(self):
        return [(1,)]

    def fetchone(self):
        sql = self.statements[-1]
        if ".state_servers" in sql:
            if self.missing_identity:
                return None
            return tuple(BINDING[key] for key in (
                "server_id", "store_id", "database_uuid", "schema_revision",
                "generation", "process_birth_id", "listen_uri", "extension_fingerprint",
            ))
        if ".control_plane_metadata" in sql:
            return (BINDING["schema_fingerprint"],)
        if ".store_generations" in sql:
            return (4, BINDING["database_uuid"], BINDING["process_birth_id"])
        return (1,)

    def close(self):
        self.close_count += 1


@pytest.fixture
def connect_clients(monkeypatch):
    for name in (
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION",
        "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)

    def install(*clients, connect_error=None):
        calls = []

        def connect(database, **kwargs):
            # A late SET threads cannot satisfy this assertion: the cap must
            # already be present when the native database is constructed.
            assert database == ":memory:"
            assert kwargs == {"config": {"threads": "1"}}
            calls.append((database, kwargs))
            if connect_error is not None:
                raise connect_error
            client = clients[len(calls) - 1]
            assert client.statements == []
            return client

        monkeypatch.setitem(sys.modules, "duckdb", SimpleNamespace(connect=connect))
        return calls

    return install


def probe_live():
    transport = server.InProcessQuackTransport()
    transport._started = True
    transport._listen_uri = URI
    transport._server_identity = dict(BINDING)
    identity = SimpleNamespace(matches=lambda **fields: all(
        BINDING[key] == value for key, value in fields.items()
    ))
    return transport.live_query(None, identity=identity, token="test-token")


def test_attach_caps_workers_before_load_and_keeps_successful_client_open(connect_clients):
    client = Client()
    calls = connect_clients(client)
    attached, binding = state._attach_quack_once(URI, "test-token")
    assert attached is client and binding == BINDING
    assert len(calls) == 1 and client.statements[0] == "LOAD quack"
    assert client.close_count == 0
    attached.close()


@pytest.mark.parametrize("fail_at", ["LOAD quack", "ATTACH ", ".store_generations"])
def test_attach_failure_closes_the_bounded_client(connect_clients, fail_at):
    client = Client(fail_at=fail_at)
    connect_clients(client)
    with pytest.raises(RuntimeError, match="injected client failure"):
        state._attach_quack_once(URI, "test-token")
    assert client.close_count == 1


def test_attach_identity_rejection_closes_the_bounded_client(connect_clients):
    client = Client(missing_identity=True)
    connect_clients(client)
    with pytest.raises(state.DuckDBConnectionPolicyError, match="complete live server binding"):
        state._attach_quack_once(URI, "test-token")
    assert client.close_count == 1


def test_attach_connect_failure_propagates_without_a_client(connect_clients):
    failure = RuntimeError("connect failed")
    calls = connect_clients(connect_error=failure)
    with pytest.raises(RuntimeError, match="connect failed") as caught:
        state._attach_quack_once(URI, "test-token")
    assert caught.value is failure and len(calls) == 1


def test_live_probe_caps_workers_before_load_and_closes_client(connect_clients):
    client = Client()
    calls = connect_clients(client)
    assert probe_live()["live"] is True
    assert len(calls) == 1 and client.statements[0] == "LOAD quack"
    assert client.close_count == 1


@pytest.mark.parametrize("fail_at", ["LOAD quack", "quack_query("])
def test_live_probe_failure_closes_the_bounded_client(connect_clients, fail_at):
    client = Client(fail_at=fail_at)
    connect_clients(client)
    with pytest.raises(server.QuackStateServerReadyError) as caught:
        probe_live()
    assert caught.value.__cause__ is client.error
    assert client.close_count == 1


def test_live_probe_connect_failure_preserves_fail_closed_result(connect_clients):
    failure = RuntimeError("connect failed")
    calls = connect_clients(connect_error=failure)
    with pytest.raises(server.QuackStateServerReadyError) as caught:
        probe_live()
    assert caught.value.__cause__ is failure and len(calls) == 1


def test_each_live_probe_startup_retry_has_a_new_bounded_closed_client(
    connect_clients, monkeypatch,
):
    class InvalidInputException(Exception):
        pass

    first = Client(
        fail_at="quack_query(", error=InvalidInputException("Invalid connection id"),
    )
    second = Client()
    calls = connect_clients(first, second)
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
    assert probe_live()["live"] is True
    assert len(calls) == 2
    assert first.close_count == second.close_count == 1
