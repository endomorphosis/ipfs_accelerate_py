#!/usr/bin/env python3
"""Transport parity tests for unified MCP runtime router."""

import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.exceptions import RuntimeExecutionError
from ipfs_accelerate_py.mcp_server.exceptions import RuntimeRoutingError
from ipfs_accelerate_py.mcp_server.runtime_router import RuntimeRouter


class TestMCPServerTransportParity(unittest.TestCase):
    """Validate fastapi/trio/auto runtime routing behavior."""

    def test_fastapi_runtime_executes_tool(self) -> None:
        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi")

            async def tool(value: str):
                return {"value": value}

            result = await router.route_tool_call("demo.tool", tool, value="ok")
            self.assertEqual(result, {"value": "ok"})

        anyio.run(_run)

    def test_trio_runtime_uses_trio_executor_path(self) -> None:
        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi")

            async def tool(value: str):
                return {"value": value}

            router.register_tool_runtime("demo.tool", "trio")

            with patch.object(router, "_execute_trio", AsyncMock(return_value={"value": "trio-ok"})) as mock_trio:
                result = await router.route_tool_call("demo.tool", tool, value="ok")

            self.assertEqual(result, {"value": "trio-ok"})
            self.assertEqual(mock_trio.await_count, 1)

        anyio.run(_run)

    def test_auto_runtime_falls_back_to_default_runtime(self) -> None:
        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi")

            async def tool(value: str):
                return {"value": value}

            router.register_tool_runtime("demo.tool", "auto")

            with patch.object(router, "_execute_fastapi", AsyncMock(return_value={"value": "auto-ok"})) as mock_fastapi:
                result = await router.route_tool_call("demo.tool", tool, value="ok")

            self.assertEqual(result, {"value": "auto-ok"})
            self.assertEqual(mock_fastapi.await_count, 1)

        anyio.run(_run)

    def test_runtime_router_timeout_path_updates_metrics(self) -> None:
        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi")

            async def slow_tool() -> dict:
                await anyio.sleep(0.05)
                return {"ok": True}

            # Attach timeout metadata through function attribute for this test.
            setattr(slow_tool, "__mcp_timeout_seconds__", 0.001)

            with self.assertRaises(RuntimeExecutionError):
                await router.route_tool_call("demo.slow_tool", slow_tool)

            metrics = router.get_metrics()
            self.assertGreaterEqual(metrics["fastapi"]["timeout_count"], 1)
            self.assertGreaterEqual(metrics["fastapi"]["error_count"], 1)

        anyio.run(_run)

    def test_trio_runtime_falls_back_when_bridge_missing_by_default(self) -> None:
        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi")

            async def tool(value: str):
                return {"value": value}

            router.register_tool_runtime("demo.tool", "trio")

            with patch(
                "ipfs_accelerate_py.mcp_server.runtime_router.RuntimeRouter._execute_trio",
                side_effect=ImportError("missing trio bridge"),
            ):
                # Ensure fallback behavior is preserved when strict mode is off.
                result = await router._execute_fastapi(tool, value="ok")

            self.assertEqual(result, {"value": "ok"})

        anyio.run(_run)

    def test_trio_runtime_can_fail_closed_when_bridge_required(self) -> None:
        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi", trio_bridge_required=True)

            async def tool(value: str):
                return {"value": value}

            router.register_tool_runtime("demo.tool", "trio")

            # Simulate bridge import failure at execution point.
            with patch.dict("sys.modules", {"ipfs_accelerate_py.mcplusplus_module.trio.bridge": None}):
                with self.assertRaises(RuntimeRoutingError):
                    await router._execute_trio(tool, value="ok")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
