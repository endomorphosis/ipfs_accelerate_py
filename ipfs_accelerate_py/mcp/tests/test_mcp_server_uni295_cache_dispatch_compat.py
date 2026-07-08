#!/usr/bin/env python3
"""UNI-295 cache-tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class TestMCPServerUNI295CacheDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_cache_dispatch_infers_error_status_from_contradictory_delegate_payloads(
        self, mock_wrapper
    ) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(
                self,
                name,
                function,
                description,
                input_schema,
                execution_context=None,
                tags=None,
            ):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class _ContradictoryManager:
            def get(self, key: str, namespace: str = "default") -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": f"get failed for {namespace}:{key}",
                }

            def get_stats(self, namespace: str | None = None) -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": f"stats failed for {namespace or 'all'}",
                }

            def optimize(self, **_: object) -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": "optimize failed",
                }

            def get_cached_embeddings(self, text: str, model: str = "default") -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": f"embedding lookup failed for {model}:{text}",
                }

        mock_wrapper.return_value = DummyServer()

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch(
                "ipfs_accelerate_py.mcp_server.tools.cache_tools.native_cache_tools._get_cache_manager",
                return_value=_ContradictoryManager(),
            ):
                server = create_mcp_server(name="cache-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                cached = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "cache_tools",
                        "cache_get",
                        {"key": "alpha", "namespace": "default"},
                    )
                )
                self.assertEqual(cached.get("status"), "error")
                self.assertEqual(cached.get("success"), False)
                self.assertEqual(cached.get("error"), "get failed for default:alpha")

                managed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "cache_tools",
                        "manage_cache",
                        {"action": "stats"},
                    )
                )
                self.assertEqual(managed.get("status"), "error")
                self.assertEqual(managed.get("success"), False)
                self.assertEqual(managed.get("error"), "stats failed for all")

                optimized = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "cache_tools",
                        "optimize_cache",
                        {"cache_type": "embeddings", "strategy": "lru"},
                    )
                )
                self.assertEqual(optimized.get("status"), "error")
                self.assertEqual(optimized.get("success"), False)
                self.assertEqual(optimized.get("error"), "optimize failed")

                embedded = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "cache_tools",
                        "get_cached_embeddings",
                        {"text": "hello", "model": "demo"},
                    )
                )
                self.assertEqual(embedded.get("status"), "error")
                self.assertEqual(embedded.get("success"), False)
                self.assertEqual(
                    embedded.get("error"),
                    "embedding lookup failed for demo:hello",
                )

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
