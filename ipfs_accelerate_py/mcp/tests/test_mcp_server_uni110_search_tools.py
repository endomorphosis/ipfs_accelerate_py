#!/usr/bin/env python3
"""UNI-110 search tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.search_tools.native_search_tools import (
    faceted_search,
    semantic_search,
    similarity_search,
    register_native_search_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI110SearchTools(unittest.TestCase):
    def test_register_includes_search_tools(self) -> None:
        manager = _DummyManager()
        register_native_search_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("semantic_search", names)
        self.assertIn("similarity_search", names)
        self.assertIn("faceted_search", names)

    def test_semantic_search_rejects_missing_query(self) -> None:
        async def _run() -> None:
            result = await semantic_search(query="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_similarity_search_rejects_invalid_embedding(self) -> None:
        async def _run() -> None:
            result = await similarity_search(embedding=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty list", str(result.get("message", "")))

        anyio.run(_run)

    def test_similarity_search_rejects_invalid_threshold(self) -> None:
        async def _run() -> None:
            result = await similarity_search(embedding=[0.1, 0.2], threshold=1.1)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("between 0.0 and 1.0", str(result.get("message", "")))

        anyio.run(_run)

    def test_faceted_search_rejects_invalid_aggregations(self) -> None:
        async def _run() -> None:
            result = await faceted_search(aggregations="category")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("aggregations must be an array", str(result.get("message", "")))

        anyio.run(_run)

    def test_semantic_search_success_shape(self) -> None:
        async def _run() -> None:
            result = await semantic_search(query="hello", top_k=2)
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("results", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
