"""MCP, MCP++, and package exports for the llm_allocation schema."""

from __future__ import annotations

import asyncio
import inspect
import json

from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import HierarchicalToolManager
from ipfs_accelerate_py.mcp_server.server import StandaloneMCP, register_tools
from ipfs_accelerate_py.mcp_server.tools.llm_allocation_tools import (
    register_native_llm_allocation_tools,
)
from ipfs_accelerate_py.mcplusplus_module.tools import register_llm_allocation_tools


EXPECTED_TOOLS = {
    "llm_cli_tools_status",
    "llm_choose_cli_route",
    "llm_migrate_cli_session",
    "llm_resolve_session",
    "llm_get_allocation_session",
    "llm_register_api_key",
    "llm_list_api_key_slots",
    "llm_bind_session_api_key",
    "llm_ensure_cli_tool",
    "llm_router_generate_text",
    "generate_text",
}

PACKAGE_EXPORTS = (
    "generate_text",
    "get_allocation_session",
    "choose_cli_session_route",
    "migrate_cli_session_route",
    "cli_tools_status",
    "render_cli_tools_status",
    "register_api_key",
    "list_api_key_slots",
    "bind_session_api_key",
    "select_api_key",
    "resolve_session",
    "choose_cli_route",
    "migrate_cli_session",
    "get_allocation_store",
    "resume_kwargs_for_session",
    "fingerprint_api_key",
    "ensure_cli_tool",
)


def test_register_tools_exposes_llm_allocation_schema() -> None:
    mcp = StandaloneMCP(name="test-llm-allocation")
    register_tools(mcp, include_p2p_taskqueue_tools=False)
    names = set(mcp.tools)
    missing = EXPECTED_TOOLS - names
    assert not missing, missing


def test_package_exports_llm_allocation_apis() -> None:
    import ipfs_accelerate_py as pkg

    for name in PACKAGE_EXPORTS:
        assert hasattr(pkg, name), name
        assert callable(getattr(pkg, name)), name
        assert name in pkg.__all__, name
    assert pkg.llm_allocation_available is True


def test_generate_text_mcp_schema_includes_allocation_fields() -> None:
    mcp = StandaloneMCP(name="test-llm-allocation-schema")
    register_tools(mcp, include_p2p_taskqueue_tools=False)
    schema = mcp.tools["generate_text"]["input_schema"]
    props = schema.get("properties") or {}
    assert "allocation_session_id" in props
    assert "allocation_path" in props
    assert "provider" in props
    router_schema = mcp.tools["llm_router_generate_text"]["input_schema"]
    router_props = router_schema.get("properties") or {}
    assert "allocation_session_id" in router_props
    assert "allocation_path" in router_props
    sig = inspect.signature(
        __import__(
            "ipfs_accelerate_py.mcp_server.tools.shared_tools.native_shared_tools",
            fromlist=["generate_text"],
        ).generate_text
    )
    assert "allocation_session_id" in sig.parameters
    assert "allocation_path" in sig.parameters
    llm_generate = __import__(
        "ipfs_accelerate_py.mcp_server.tools.ai_router_tools.text_embedding",
        fromlist=["llm_generate"],
    ).llm_generate
    llm_sig = inspect.signature(llm_generate)
    assert "allocation_session_id" in llm_sig.parameters
    assert "allocation_path" in llm_sig.parameters


def test_mcplusplus_register_tools_exposes_allocation_schema() -> None:
    mcp = StandaloneMCP(name="test-mcplusplus-allocation")
    register_llm_allocation_tools(mcp)
    missing = EXPECTED_TOOLS - {"generate_text"} - set(mcp.tools)
    assert not missing, missing
    schema = mcp.tools["llm_migrate_cli_session"]["input_schema"]
    props = schema.get("properties") or {}
    assert "history" in props
    assert "session_id" in props
    assert "to_provider" in props


def test_hierarchical_manager_exposes_allocation_category() -> None:
    manager = HierarchicalToolManager()
    register_native_llm_allocation_tools(manager)
    names = {item["name"] for item in manager.list_tools("llm_allocation_tools")}
    missing = EXPECTED_TOOLS - {"generate_text"} - names
    assert not missing, missing
    schema = manager.get_tool_schema("llm_allocation_tools", "llm_bind_session_api_key")
    assert set(schema["input_schema"]["required"]) == {
        "session_id",
        "backend",
        "slot_id",
    }


def test_mcp_register_api_key_strips_secrets(monkeypatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.llm_allocation.register_api_key",
        lambda backend, api_key, slot_id="", label="": {
            "backend": backend,
            "slot_id": slot_id or "primary",
            "key_fingerprint": "abcd1234",
            "secret_name": "api_key/openai/primary",
            "api_key": api_key,
        },
    )
    mcp = StandaloneMCP(name="test-allocation-secrets")
    register_native_llm_allocation_tools(mcp)
    result = asyncio.run(
        mcp.tools["llm_register_api_key"]["function"](
            backend="openai",
            api_key="sk-live-secret-value",
            slot_id="primary",
            label="burst",
        )
    )
    dumped = json.dumps(result)
    assert result["success"] is True
    assert "sk-live-secret-value" not in dumped
    assert "api_key" not in dumped
    assert "secret_name" not in dumped
    assert result["slot_id"] == "primary"


def test_mcplusplus_allocation_tool_strips_secrets(monkeypatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.llm_allocation.choose_cli_route",
        lambda session_id=None: {
            "provider": "muse_code",
            "allocation_session_id": session_id or "new-1",
            "api_key": "should-not-leak",
        },
    )
    mcp = StandaloneMCP(name="test-mcplusplus-alloc-rpc")
    register_llm_allocation_tools(mcp)
    result = asyncio.run(
        mcp.tools["llm_choose_cli_route"]["function"](session_id="sess-rpc")
    )
    dumped = json.dumps(result)
    assert result["success"] is True
    assert result["provider"] == "muse_code"
    assert "should-not-leak" not in dumped
    assert "api_key" not in dumped
