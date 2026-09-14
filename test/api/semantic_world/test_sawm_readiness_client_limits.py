"""Every native readiness client is bounded before extension loading."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def operator():
    path = Path(__file__).resolve().parents[3] / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
    spec = importlib.util.spec_from_file_location("sawm_readiness_limits_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("failure", ["none", "query_once", "load"])
def test_native_readiness_bounds_and_closes_every_client(operator, monkeypatch, failure):
    clients = []

    class Client:
        closed = False

        def execute(self, sql, params):
            assert params == ["quack://test", "SELECT count(*) FROM tasks", "test-token"]
            assert "token := ?" in sql
            if failure == "query_once" and len(clients) == 1:
                raise RuntimeError("transient test query failure")
            return SimpleNamespace(fetchall=lambda: [(1,)])

        def close(self):
            self.closed = True

    def connect(path, *, config):
        assert path == ":memory:"
        assert config["threads"] == "1"
        assert config["memory_limit"] == "256MB"
        assert config["lock_configuration"] == "true"
        assert config["allow_unsigned_extensions"] == "false"
        assert config["autoinstall_known_extensions"] == "false"
        assert config["autoload_known_extensions"] == "false"
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setitem(sys.modules, "duckdb", SimpleNamespace(connect=connect))
    monkeypatch.setattr(operator.time, "sleep", lambda seconds: None)
    transport = operator._SawmQuackTransport({})
    transport._serve_uri = "quack://test"
    transport._owner_token = "test-token"
    monkeypatch.setattr(transport, "_ensure_extension_projection", lambda: SimpleNamespace(extension_directory=Path("/test/extensions")))

    def load(client):
        assert clients and clients[-1] is client
        assert not client.closed
        if failure == "load":
            raise RuntimeError("test extension rejection")

    monkeypatch.setattr(transport, "_load_reviewed_extensions", load)
    if failure == "load":
        with pytest.raises(RuntimeError, match="test extension rejection"):
            transport._probe()
    else:
        transport._probe()
    assert len(clients) == (2 if failure == "query_once" else 1)
    assert all(client.closed for client in clients)
