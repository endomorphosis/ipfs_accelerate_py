"""Tool surface parity tests between MCP and MCP++.

Goal: MCP++ should be a superset of MCP for all shared tool names.

These tests avoid making network calls; they validate registration parity by
comparing tool sets and input schemas derived from signatures.
"""

from __future__ import annotations

import inspect
import asyncio
from typing import cast
from urllib.parse import urlparse


def _run_async(coro):
    """Run a coroutine in sync tests."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # If a loop is already running, the caller should convert the test to async.
    raise RuntimeError("Cannot run coroutine: event loop already running")


def _resource_key(uri: str) -> str:
    """Normalize FastMCP URL-like URIs to StandaloneMCP-style keys."""

    parsed = urlparse(uri)
    if parsed.scheme:
        combined = (parsed.netloc + parsed.path).lstrip("/")
        combined = combined.split("/{", 1)[0]
        return combined or uri
    return uri.lstrip("/")


def _schema_surface(schema: object) -> dict[str, object] | None:
    """Extract the stable subset of JSON schema used for parity checks.

    FastMCP and StandaloneMCP do not generate identical *deep* JSON schemas.
    For parity we validate the user-facing parameter surface: top-level
    parameter names and which are required.
    """

    if not isinstance(schema, dict):
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    required = schema.get("required")
    required_list: list[str] = []
    if isinstance(required, list):
        required_list = sorted(str(x) for x in required)

    return {
        "type": schema.get("type"),
        "properties": sorted(str(k) for k in properties.keys()),
        "required": required_list,
    }


def _tools_map(server: object) -> dict:
    tools = getattr(server, "tools", None)
    if isinstance(tools, dict):
        return tools

    if hasattr(server, "list_tools"):
        tool_list = _run_async(server.list_tools())
        out: dict[str, dict] = {}
        for t in tool_list:
            d = t.model_dump() if hasattr(t, "model_dump") else (t if isinstance(t, dict) else {})
            name = d.get("name") or getattr(t, "name", None)
            if not name:
                continue
            out[name] = {
                "input_schema": d.get("parameters") or d.get("input_schema"),
                "function": d.get("fn") or getattr(t, "fn", None),
                "description": d.get("description") or getattr(t, "description", None) or "",
            }
        return out

    return {}


def _resource_names(server: object) -> set[str]:
    resources = getattr(server, "resources", None)
    if isinstance(resources, dict):
        return set(resources.keys())

    names: set[str] = set()

    # FastMCP separates concrete resources from parameterized templates.
    if hasattr(server, "list_resources"):
        rs = _run_async(server.list_resources())
        for r in rs:
            uri = str(getattr(r, "uri", ""))
            if uri:
                names.add(_resource_key(uri))

    if hasattr(server, "list_resource_templates"):
        ts = _run_async(server.list_resource_templates())
        for t in ts:
            uri_template = getattr(t, "uri_template", None)
            if not uri_template and hasattr(t, "model_dump"):
                d = t.model_dump()
                uri_template = d.get("uri_template")
            if uri_template:
                names.add(_resource_key(str(uri_template)))

    return names


def _prompt_names(server: object) -> set[str]:
    prompts = getattr(server, "prompts", None)
    if isinstance(prompts, dict):
        return set(prompts.keys())

    if hasattr(server, "list_prompts"):
        ps = _run_async(server.list_prompts())
        names: set[str] = set()
        for p in ps:
            name = getattr(p, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    return set()


def _signature_of(fn):
    unwrapped = inspect.unwrap(fn)
    return inspect.signature(unwrapped)


def _make_primary_mcp():
    from ipfs_accelerate_py.mcp.server import StandaloneMCP
    from ipfs_accelerate_py.mcp.tools import register_all_tools
    from ipfs_accelerate_py.mcp.resources import register_all_resources

    mcp = StandaloneMCP("mcp")
    register_all_tools(mcp)
    register_all_resources(mcp)
    try:
        mcp.register_prompt(
            name="ipfs_help",
            template="help",
            description="help",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    except Exception:
        pass
    return mcp


def _make_mcpplus_mcp():
    from ipfs_accelerate_py.mcplusplus_module.trio.server import TrioMCPServer

    server = TrioMCPServer(name="mcp++")
    server.setup()
    return server.mcp


def test_mcpplus_is_superset_of_mcp_tools_and_resources() -> None:
    mcp = _make_primary_mcp()
    mcpplus = _make_mcpplus_mcp()

    mcp_tools_dict = _tools_map(mcp)
    mcpplus_tools_dict = _tools_map(mcpplus)

    mcp_tools = set(mcp_tools_dict.keys())
    mcpplus_tools = set(mcpplus_tools_dict.keys())

    assert mcp_tools <= mcpplus_tools

    # For shared tool names, validate schema parity.
    shared = sorted(mcp_tools & mcpplus_tools)
    for name in shared:
        schema_a = (mcp_tools_dict[name] or {}).get("input_schema")
        schema_b = (mcpplus_tools_dict[name] or {}).get("input_schema")
        # Some registries may omit schemas (or provide empty placeholders).
        # When that happens, rely on signature parity below.
        if (
            isinstance(schema_a, dict)
            and isinstance(schema_b, dict)
            and bool(schema_a.get("properties"))
            and bool(schema_b.get("properties"))
        ):
            surf_a = _schema_surface(schema_a)
            surf_b = _schema_surface(schema_b)
            if surf_a is not None and surf_b is not None:
                assert surf_a == surf_b, f"Schema surface mismatch for tool: {name}"

        fn_a = (mcp_tools_dict[name] or {}).get("function")
        fn_b = (mcpplus_tools_dict[name] or {}).get("function")
        assert callable(fn_a)
        assert callable(fn_b)
        assert inspect.iscoroutinefunction(fn_a) == inspect.iscoroutinefunction(fn_b)
        assert _signature_of(fn_a) == _signature_of(fn_b)

    # Resources/prompts should also be present.
    assert set((mcp.resources or {}).keys()) <= _resource_names(mcpplus)
    assert set((mcp.prompts or {}).keys()) <= _prompt_names(mcpplus)


def test_p2p_taskqueue_tool_schemas_match_between_mcp_and_mcpplus() -> None:
    from ipfs_accelerate_py.mcp.server import StandaloneMCP

    mcp = StandaloneMCP("mcp-p2p")
    mcpplus = StandaloneMCP("mcpplus-p2p")

    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import register_tools as register_mcp
    from ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools import (
        register_p2p_taskqueue_tools as register_mcpplus,
    )

    register_mcp(mcp)
    register_mcpplus(mcpplus)

    mcp_names = set(mcp.tools.keys())
    mcpplus_names = set(mcpplus.tools.keys())

    assert mcp_names == mcpplus_names

    for name in sorted(mcp_names):
        assert mcp.tools[name]["input_schema"] == mcpplus.tools[name]["input_schema"]
        assert bool(mcp.tools[name]["description"].strip())
        assert bool(mcpplus.tools[name]["description"].strip())

        fn_a = mcp.tools[name]["function"]
        fn_b = mcpplus.tools[name]["function"]
        assert inspect.iscoroutinefunction(fn_a) == inspect.iscoroutinefunction(fn_b)
        assert _signature_of(fn_a) == _signature_of(fn_b)


def test_run_in_trio_from_asyncio_context_smoke() -> None:
    import threading

    import anyio
    import sniffio

    from ipfs_accelerate_py.mcplusplus_module.trio.bridge import run_in_trio

    async def trio_nursery_smoke(x: int) -> tuple[int, int, str]:
        import trio

        async with trio.open_nursery() as nursery:
            nursery.start_soon(trio.sleep, 0)
        return x * 2, threading.get_ident(), sniffio.current_async_library()

    async def main() -> tuple[int, int, int, str]:
        outer_tid = threading.get_ident()
        trio_result = await run_in_trio(trio_nursery_smoke, 21)
        result, inner_tid, lib = cast(tuple[int, int, str], trio_result)
        return result, outer_tid, inner_tid, lib

    result, outer_tid, inner_tid, lib = anyio.run(main, backend="asyncio")
    assert result == 42
    assert lib == "trio"
    assert inner_tid != outer_tid


def test_run_in_trio_inside_trio_context_smoke() -> None:
    import threading

    import anyio
    import pytest
    import sniffio

    pytest.importorskip("trio")

    from ipfs_accelerate_py.mcplusplus_module.trio.bridge import run_in_trio

    async def check_thread_and_lib() -> tuple[int, str]:
        return threading.get_ident(), sniffio.current_async_library()

    async def main() -> tuple[int, int, str]:
        outer_tid = threading.get_ident()
        trio_result = await run_in_trio(check_thread_and_lib)
        inner_tid, lib = cast(tuple[int, str], trio_result)
        return outer_tid, inner_tid, lib

    outer_tid, inner_tid, lib = anyio.run(main, backend="trio")
    assert lib == "trio"
    assert inner_tid == outer_tid
