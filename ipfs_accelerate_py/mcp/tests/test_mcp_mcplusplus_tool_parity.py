"""Tool surface parity tests between MCP and MCP++.

Goal: MCP++ should be a superset of MCP for all shared tool names.

These tests avoid making network calls; they validate registration parity by
comparing tool sets and input schemas derived from signatures.
"""

from __future__ import annotations

import inspect


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

    mcp_tools = set((mcp.tools or {}).keys())
    mcpplus_tools = set((mcpplus.tools or {}).keys())

    assert mcp_tools <= mcpplus_tools

    # For shared tool names, validate schema parity.
    shared = sorted(mcp_tools & mcpplus_tools)
    for name in shared:
        assert (mcp.tools[name] or {}).get("input_schema") == (mcpplus.tools[name] or {}).get("input_schema")

        fn_a = (mcp.tools[name] or {}).get("function")
        fn_b = (mcpplus.tools[name] or {}).get("function")
        assert callable(fn_a)
        assert callable(fn_b)
        assert inspect.iscoroutinefunction(fn_a) == inspect.iscoroutinefunction(fn_b)
        assert _signature_of(fn_a) == _signature_of(fn_b)

    # Resources/prompts should also be present.
    assert set((mcp.resources or {}).keys()) <= set((mcpplus.resources or {}).keys())
    assert set((mcp.prompts or {}).keys()) <= set((mcpplus.prompts or {}).keys())


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
        result, inner_tid, lib = await run_in_trio(trio_nursery_smoke, 21)
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
        inner_tid, lib = await run_in_trio(check_thread_and_lib)
        return outer_tid, inner_tid, lib

    outer_tid, inner_tid, lib = anyio.run(main, backend="trio")
    assert lib == "trio"
    assert inner_tid == outer_tid
