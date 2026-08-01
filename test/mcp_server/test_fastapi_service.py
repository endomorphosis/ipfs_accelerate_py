from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ipfs_accelerate_py.mcp_server import fastapi_service
from ipfs_accelerate_py.mcp_server.fastapi_config import UnifiedFastAPIConfig


class _FakeManager:
    async def dispatch(
        self,
        category: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "category": category,
            "tool": tool_name,
            "parameters": parameters,
        }


class _FakeServer:
    def __init__(self) -> None:
        inner = FastAPI()

        @inner.get("/mounted-only")
        async def mounted_only() -> dict[str, bool]:
            return {"mounted": True}

        async def list_categories() -> dict[str, list[str]]:
            return {"categories": ["demo"]}

        async def get_schema(category: str, tool_name: str) -> dict[str, str]:
            return {"category": category, "tool": tool_name}

        async def dispatch(
            category: str,
            tool_name: str,
            parameters: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "category": category,
                "tool": tool_name,
                "parameters": parameters,
            }

        async def echo(value: Any) -> dict[str, Any]:
            return {"echo": value}

        self.app = inner
        self.tools = {
            "echo": {
                "function": echo,
                "description": "Echo one value.",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {}},
                    "required": ["value"],
                },
            },
            "tools_dispatch": {
                "function": dispatch,
                "description": "Dispatch a hierarchical tool.",
                "input_schema": {"type": "object"},
            },
            "tools_get_schema": {
                "function": get_schema,
                "description": "Read a hierarchical tool schema.",
                "input_schema": {"type": "object"},
            },
            "tools_list_categories": {
                "function": list_categories,
                "description": "List hierarchical categories.",
                "input_schema": {"type": "object"},
            },
        }
        self.mcp = SimpleNamespace(tools=self.tools)
        self._unified_tool_manager = _FakeManager()
        self._unified_supported_profiles = (
            "mcp++/idl",
            "mcp++/risk-scheduling",
        )


def _client(monkeypatch: Any) -> TestClient:
    server = _FakeServer()
    monkeypatch.setattr(
        fastapi_service,
        "create_server",
        lambda **_kwargs: server,
    )
    app = fastapi_service.create_fastapi_app(
        UnifiedFastAPIConfig(
            name="test-accelerate-mcp",
            mount_path="/mcp",
        )
    )
    return TestClient(app)


def _text_content(response: dict[str, Any]) -> Any:
    result = response["result"]
    assert result["isError"] is False
    return json.loads(result["content"][0]["text"])


def test_swissknife_initialize_and_tool_discovery_routes(monkeypatch: Any) -> None:
    with _client(monkeypatch) as client:
        health = client.get("/mcp/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        initialized = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            },
        )
        assert initialized.status_code == 200
        init_result = initialized.json()["result"]
        assert init_result["protocolVersion"] == "2024-11-05"
        assert init_result["capabilities"]["tools"] == {"listChanged": False}
        assert init_result["capabilities"]["experimental"] == {
            "mcp++/risk-scheduling": True
        }

        listed = client.get("/mcp/tools/list")
        assert listed.status_code == 200
        assert {item["name"] for item in listed.json()["result"]["tools"]} == {
            "echo",
            "tools_dispatch",
            "tools_get_schema",
            "tools_list_categories",
        }

        posted = client.post(
            "/mcp/tools/list",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        assert posted.status_code == 200
        assert posted.json()["id"] == 2

        empty_post = client.post("/mcp/tools/list")
        assert empty_post.status_code == 200
        assert empty_post.json()["result"]["tools"]

        rooted = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        )
        assert rooted.status_code == 200
        assert rooted.json()["id"] == 3


def test_swissknife_tool_calls_accept_wire_and_hierarchical_aliases(
    monkeypatch: Any,
) -> None:
    with _client(monkeypatch) as client:
        echoed = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {"value": {"hello": "world"}},
                },
            },
        )
        assert echoed.status_code == 200
        assert _text_content(echoed.json()) == {
            "echo": {"hello": "world"}
        }

        schema = client.post(
            "/mcp/tools/call",
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "tools_get_schema",
                    "arguments": {"category": "demo", "tool": "echo"},
                },
            },
        )
        assert schema.status_code == 200
        assert _text_content(schema.json()) == {
            "category": "demo",
            "tool": "echo",
        }

        dispatched = client.post(
            "/mcp/tools/call",
            json={
                "name": "tools_dispatch",
                "args": {
                    "category": "demo",
                    "tool": "echo",
                    "params": {"value": 7},
                },
            },
        )
        assert dispatched.status_code == 200
        assert _text_content(dispatched.json()) == {
            "category": "demo",
            "parameters": {"value": 7},
            "tool": "echo",
        }

        dotted = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "demo.echo",
                    "arguments": {"value": 8},
                },
            },
        )
        assert dotted.status_code == 200
        assert _text_content(dotted.json()) == {
            "category": "demo",
            "parameters": {"value": 8},
            "tool": "echo",
        }


def test_profile_g_routes_are_explicit_and_do_not_shadow_mount(
    monkeypatch: Any,
) -> None:
    from ipfs_accelerate_py.mcp_server.mcplusplus import profile_g_transport

    class _Dispatcher:
        def dispatch(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            return {"method": method, "params": params}

    monkeypatch.setattr(
        profile_g_transport,
        "get_profile_g_dispatcher",
        lambda: _Dispatcher(),
    )

    with _client(monkeypatch) as client:
        profile = client.get("/mcp/risk/profile?limit=2")
        assert profile.status_code == 200
        assert profile.json() == {
            "method": "mcp++/risk/profile",
            "params": {"limit": 2},
        }

        goal = client.get("/mcp/goals/baguqgoal?at_ms=10")
        assert goal.status_code == 200
        assert goal.json() == {
            "method": "mcp++/goals/get",
            "params": {"at_ms": 10, "goal_cid": "baguqgoal"},
        }

        mounted = client.get("/mcp/mounted-only")
        assert mounted.status_code == 200
        assert mounted.json() == {"mounted": True}
