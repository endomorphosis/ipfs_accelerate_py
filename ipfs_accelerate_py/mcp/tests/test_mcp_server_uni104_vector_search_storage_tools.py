#!/usr/bin/env python3
"""UNI-104 vector/search/storage integration parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.vector_tools.native_vector_tools import (
    orchestrate_vector_search_storage,
    register_native_vector_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI104VectorSearchStorageTools(unittest.TestCase):
    def test_register_includes_integration_tool(self) -> None:
        manager = _DummyManager()
        register_native_vector_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("orchestrate_vector_search_storage", names)

    def test_orchestration_rejects_missing_vectors(self) -> None:
        async def _run() -> None:
            result = await orchestrate_vector_search_storage(vectors=[], query_vector=[0.1, 0.2])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("vectors must be a non-empty", str(result.get("error", "")))

        anyio.run(_run)

    def test_orchestration_success_contract_without_persistence(self) -> None:
        async def _run() -> None:
            result = await orchestrate_vector_search_storage(
                vectors=[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
                query_vector=[0.1, 0.2],
                top_k=3,
                persist_audit=False,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertIn("index_id", result)
            self.assertEqual((result.get("search") or {}).get("result_count"), 0)
            similarity = result.get("search_tools_similarity") or {}
            self.assertGreaterEqual(int(similarity.get("total_found", -1)), 0)
            self.assertIsInstance(similarity.get("results"), list)
            self.assertFalse((result.get("storage") or {}).get("stored"))

        anyio.run(_run)

    def test_orchestration_persists_storage_audit(self) -> None:
        async def _run() -> None:
            result = await orchestrate_vector_search_storage(
                vectors=[[0.1, 0.2], [0.2, 0.3]],
                query_vector=[0.2, 0.3],
                persist_audit=True,
                audit_collection="uni104-audit",
            )
            self.assertEqual(result.get("status"), "success")
            storage = result.get("storage") or {}
            self.assertTrue(storage.get("stored"))
            self.assertEqual(storage.get("collection"), "uni104-audit")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
