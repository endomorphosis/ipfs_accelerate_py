"""Current Quack table-function transport compatibility and optional live smoke."""
from __future__ import annotations

import os
import secrets
import socket
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport,
    _allocate_loopback_port,
)


def _identity():
    return SimpleNamespace(server_id="transport-test", store_id="temporary-control",
                           database_uuid="temporary-uuid", schema_revision=1,
                           schema_fingerprint="test-schema", generation=1,
                           process_birth_id="test-birth")


class _Connection:
    def __init__(self):
        self.calls = []

    def execute(self, sql, parameters=None):
        self.calls.append((sql, parameters))
        return self

    def fetchall(self):
        return []


def test_native_transport_uses_parameter_bound_table_serve_and_targeted_stop():
    connection = _Connection()
    transport = InProcessQuackTransport()
    token = secrets.token_urlsafe(32)
    observed = transport.start(connection, host="127.0.0.1", port=23456,
                               token=token, identity=_identity())
    serve_sql, parameters = connection.calls[1]
    assert serve_sql.startswith("SELECT * FROM quack_serve(")
    assert parameters == ["quack:127.0.0.1:23456", token, True]
    assert token not in serve_sql
    assert token not in repr(observed)
    transport.stop(connection)
    assert connection.calls[-1] == ("SELECT * FROM quack_stop(?)", ["quack:127.0.0.1:23456"])
    count = len(connection.calls)
    transport.stop(connection)
    assert len(connection.calls) == count


def test_native_transport_keeps_tls_for_non_loopback_host():
    connection = _Connection()
    transport = InProcessQuackTransport()
    transport.start(connection, host="192.0.2.1", port=23456,
                    token=secrets.token_urlsafe(32), identity=_identity())
    assert connection.calls[1][1][-1] is False
    transport.stop(connection)


@pytest.mark.skipif(os.environ.get("IPFS_ACCELERATE_RUN_LIVE_QUACK") != "1",
                    reason="explicit opt-in for temporary authenticated loopback listener")
def test_live_installed_quack_serves_authenticated_temp_database_and_stops(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    owner = duckdb.connect(str(tmp_path / "transport-smoke.duckdb"),
                           config={"autoinstall_known_extensions": False,
                                   "autoload_known_extensions": True})
    client = unauthorized = None
    transport = InProcessQuackTransport()
    port = _allocate_loopback_port()
    token = secrets.token_urlsafe(32)
    uri = f"quack:127.0.0.1:{port}"
    try:
        owner.execute("CREATE TABLE tasks(task_id VARCHAR)")
        owner.execute("INSERT INTO tasks VALUES ('only-temporary-test')")
        transport.start(owner, host="127.0.0.1", port=port, token=token, identity=_identity())
        unauthorized = duckdb.connect(":memory:")
        unauthorized.execute("LOAD quack")
        with pytest.raises(Exception):
            unauthorized.execute(f"ATTACH '{uri}' AS unauthenticated")
        unauthorized.close()
        unauthorized = None
        client = duckdb.connect(":memory:")
        client.execute("LOAD quack")
        try:
            # ATTACH does not accept prepared parameters. The generated token
            # is URL-safe and the query/exception is never reported verbatim.
            client.execute(f"ATTACH '{uri}' AS accepted (READ_WRITE, TOKEN '{token}')")
        except Exception as exc:
            pytest.fail(f"authenticated temporary attach failed: {type(exc).__name__}", pytrace=False)
        assert client.execute("SELECT count(*) FROM accepted.tasks").fetchone()[0] == 1
        client.close()
        client = None
        transport.stop(owner)
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        if client is not None:
            client.close()
        if unauthorized is not None:
            unauthorized.close()
        try:
            transport.stop(owner)
        finally:
            owner.close()
