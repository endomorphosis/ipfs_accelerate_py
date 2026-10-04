#!/usr/bin/env python3
"""Base-MCP conformance tests for the IPFS Accelerate Flask dashboard.

Ensures a stock MCP client can complete the handshake against the dashboard
JSON-RPC endpoint (mounted at both ``/jsonrpc`` and ``/mcp``): initialize ->
notifications/initialized -> tools/list -> ping, with MCP 2024-11-05 shapes.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("flask")
pytest.importorskip("flask_cors")

try:
    from ipfs_accelerate_py.mcp_dashboard import MCPDashboard
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"MCPDashboard unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    dashboard = MCPDashboard(port=3003, enable_autoscaler=False)
    return dashboard.app.test_client()


def _rpc(client, body, path="/mcp"):
    return client.post(path, json=body)


def test_initialize_on_mcp_and_jsonrpc(client):
    for path in ("/mcp", "/jsonrpc"):
        resp = _rpc(client, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, path)
        assert resp.status_code == 200, path
        result = resp.get_json()["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "mcp++"
        assert result["capabilities"]["tools"]["listChanged"] is True


def test_initialized_notification(client):
    resp = _rpc(client, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert resp.status_code == 202


def test_ping(client):
    resp = _rpc(client, {"jsonrpc": "2.0", "id": 5, "method": "ping"})
    assert resp.status_code == 200
    assert resp.get_json() == {"jsonrpc": "2.0", "id": 5, "result": {}}


def test_tools_list_shape(client):
    resp = _rpc(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert resp.status_code == 200
    tools = resp.get_json()["result"]["tools"]
    assert isinstance(tools, list) and tools
    for tool in tools:
        assert set(("name", "description", "inputSchema")).issubset(tool.keys())


def test_status_health_endpoint(client):
    assert client.get("/api/mcp/status").status_code == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
