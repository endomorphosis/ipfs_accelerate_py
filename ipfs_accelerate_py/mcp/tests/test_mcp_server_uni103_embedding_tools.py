#!/usr/bin/env python3
"""UNI-103 embedding_tools endpoint/management parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.embedding_tools.native_embedding_tools import (
    chunk_text_for_embeddings,
    manage_embedding_endpoints,
    register_native_embedding_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI103EmbeddingTools(unittest.TestCase):
    def test_register_includes_endpoint_and_chunk_tools(self) -> None:
        manager = _DummyManager()
        register_native_embedding_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("chunk_text_for_embeddings", names)
        self.assertIn("manage_embedding_endpoints", names)

    def test_chunk_text_rejects_empty_input(self) -> None:
        async def _run() -> None:
            result = await chunk_text_for_embeddings(text="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_endpoints_rejects_unknown_action(self) -> None:
        async def _run() -> None:
            result = await manage_embedding_endpoints(action="remove", model="all-MiniLM")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("Unknown action", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_endpoints_requires_endpoint_for_add(self) -> None:
        async def _run() -> None:
            result = await manage_embedding_endpoints(action="add", model="all-MiniLM", endpoint="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("endpoint must be provided", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_endpoints_list_success_shape(self) -> None:
        async def _run() -> None:
            result = await manage_embedding_endpoints(action="list", model="all-MiniLM")
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "list")
            self.assertIsInstance(result.get("endpoints"), list)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
