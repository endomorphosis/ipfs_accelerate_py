"""Older native attachment and readiness clients keep bounded failure paths."""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport,
    QuackStateServerReadyError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DEFAULT_MEMORY_LIMIT
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientTransportError,
    QuackStateClient,
)


@pytest.mark.parametrize("kind", ["attachment", "readiness"])
@pytest.mark.parametrize("fail_call", [1, 2])
def test_disposable_clients_bound_before_load_and_close_on_failure(monkeypatch, kind, fail_call):
    import duckdb

    clients = []
    connections = []

    class Client:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def execute(self, *args, **kwargs):
            self.calls += 1
            if self.calls == fail_call:
                raise RuntimeError("simulated transport failure")
            return self

        def close(self):
            self.closed = True

    def connect(*args, **kwargs):
        connections.append((args, kwargs))
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setattr(duckdb, "connect", connect)
    for _ in range(2):
        if kind == "attachment":
            client = object.__new__(QuackStateClient)
            with pytest.raises(QuackClientTransportError):
                client._open_quack_connection(SimpleNamespace(quack_uri="quack:127.0.0.1:1234"))
        else:
            transport = InProcessQuackTransport()
            transport._started = True
            transport._listen_uri = "quack:127.0.0.1:1234"
            with pytest.raises(QuackStateServerReadyError):
                transport.live_query(None, identity=None, token="test-only")
    assert len(connections) == 2
    assert all(kwargs["config"] == {"threads": 1, "memory_limit": DEFAULT_MEMORY_LIMIT}
               for _args, kwargs in connections)
    assert all(client.closed for client in clients)
