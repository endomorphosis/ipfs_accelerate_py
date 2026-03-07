#!/usr/bin/env python3
"""UNI-184 embedding dispatch compatibility tests for focused parity coverage."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.embedding_tools import native_embedding_tools


class TestMCPServerUNI184EmbeddingDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_embedding_dispatch_preserves_success_defaults_for_chunk_and_endpoint_tools(self, mock_wrapper) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.dict(
                native_embedding_tools._API,
                {
                    "chunk_text": _minimal,
                    "manage_endpoints": _minimal,
                    "shard_embeddings": _minimal,
                },
                clear=False,
            ):
                server = create_mcp_server(name="embedding-dispatch-compat")

                dispatch = server.tools["tools_dispatch"]["function"]

                chunked = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "embedding_tools",
                        "chunk_text_for_embeddings",
                        {"text": "hello world"},
                    )
                )
                self.assertEqual(chunked.get("status"), "success")
                self.assertEqual(chunked.get("original_length"), len("hello world"))
                self.assertEqual(chunked.get("chunk_count"), 0)

                listed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "embedding_tools",
                        "manage_embedding_endpoints",
                        {"action": "list", "model": "all-MiniLM"},
                    )
                )
                self.assertEqual(listed.get("status"), "success")
                self.assertEqual(listed.get("action"), "list")
                self.assertEqual(listed.get("model"), "all-MiniLM")
                self.assertEqual(listed.get("endpoints"), [])

                sharded = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "embedding_tools",
                        "shard_embeddings",
                        {"embeddings": [[0.1], [0.2]], "shard_count": 2},
                    )
                )
                self.assertEqual(sharded.get("status"), "success")
                self.assertEqual(sharded.get("shard_count"), 2)
                self.assertEqual(sharded.get("total_embeddings"), 2)
                self.assertEqual(sharded.get("shards"), [])

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()