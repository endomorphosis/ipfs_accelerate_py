#!/usr/bin/env python3
"""UNI-128 vector store tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools import (
    enhanced_vector_index,
    enhanced_vector_search,
    enhanced_vector_storage,
    register_native_vector_store_tools,
    vector_index,
    vector_metadata,
    vector_retrieval,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI128VectorStoreTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_vector_store_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        index_schema = by_name["vector_index"]["input_schema"]
        self.assertIn("create", index_schema["properties"]["action"].get("enum", []))

        retrieval_schema = by_name["vector_retrieval"]["input_schema"]
        self.assertEqual(retrieval_schema["properties"]["limit"].get("minimum"), 1)

        metadata_schema = by_name["vector_metadata"]["input_schema"]
        self.assertIn("update", metadata_schema["properties"]["action"].get("enum", []))

        enhanced_index_schema = by_name["enhanced_vector_index"]["input_schema"]
        self.assertIn("list", enhanced_index_schema["properties"]["action"].get("enum", []))

        enhanced_search_schema = by_name["enhanced_vector_search"]["input_schema"]
        self.assertEqual(enhanced_search_schema["properties"]["query_vector"].get("minItems"), 1)

        enhanced_storage_schema = by_name["enhanced_vector_storage"]["input_schema"]
        self.assertIn("get_metadata", enhanced_storage_schema["properties"]["action"].get("enum", []))

    def test_vector_index_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await vector_index(action="truncate", index_name="idx")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("action must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_vector_retrieval_rejects_invalid_limit(self) -> None:
        async def _run() -> None:
            result = await vector_retrieval(limit=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("limit must be an integer >= 1", str(result.get("message", "")))

        anyio.run(_run)

    def test_vector_metadata_requires_metadata_for_update(self) -> None:
        async def _run() -> None:
            result = await vector_metadata(action="update", collection="docs", ids=["a"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("metadata is required for update action", str(result.get("message", "")))

        anyio.run(_run)

    def test_vector_retrieval_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await vector_retrieval(collection="docs", limit=5)
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("collection"), "docs")
            self.assertEqual(result.get("limit"), 5)

        anyio.run(_run)

    def test_vector_index_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools._API"
            ) as mock_api:
                async def _impl(**kwargs):
                    _ = kwargs
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await vector_index(action="create", index_name="idx")

                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("action"), "create")
                self.assertEqual(result.get("index_name"), "idx")
                self.assertEqual(result.get("result"), {})
                self.assertEqual(result.get("success"), True)

        anyio.run(_run)

    def test_vector_retrieval_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools._API"
            ) as mock_api:
                async def _impl(**kwargs):
                    _ = kwargs
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await vector_retrieval(collection="docs", ids=["v1"], limit=2)

                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("collection"), "docs")
                self.assertEqual(result.get("ids"), ["v1"])
                self.assertEqual(result.get("limit"), 2)
                self.assertEqual(result.get("results"), [])
                self.assertEqual(result.get("total_found"), 0)

        anyio.run(_run)

    def test_vector_metadata_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools._API"
            ) as mock_api:
                async def _impl(**kwargs):
                    _ = kwargs
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await vector_metadata(action="get", collection="docs", ids=["v1"])

                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("action"), "get")
                self.assertEqual(result.get("collection"), "docs")
                self.assertEqual(result.get("ids"), ["v1"])
                self.assertEqual(result.get("metadata"), {})

        anyio.run(_run)

    def test_enhanced_vector_search_rejects_empty_query_vector(self) -> None:
        async def _run() -> None:
            result = await enhanced_vector_search(collection="docs", query_vector=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty list of numbers", str(result.get("message", "")))

        anyio.run(_run)

    def test_enhanced_vector_storage_rejects_invalid_vector_ids(self) -> None:
        async def _run() -> None:
            result = await enhanced_vector_storage(action="delete", vector_ids=[""]) 
            self.assertEqual(result.get("status"), "error")
            self.assertIn("list of non-empty strings", str(result.get("message", "")))

        anyio.run(_run)

    def test_enhanced_vector_success_shapes(self) -> None:
        async def _run() -> None:
            index_result = await enhanced_vector_index(action="list")
            self.assertEqual(index_result.get("status"), "success")
            self.assertIn("result", index_result)

            search_result = await enhanced_vector_search(
                collection="docs",
                query_vector=[0.1, 0.2, 0.3],
                top_k=3,
            )
            self.assertEqual(search_result.get("status"), "success")
            self.assertIn("results", search_result)

            storage_result = await enhanced_vector_storage(action="list", collection="docs")
            self.assertEqual(storage_result.get("status"), "success")
            self.assertIn("vectors", storage_result)

        anyio.run(_run)

    def test_vector_store_wrappers_infer_error_status_from_contradictory_delegate_payload(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                __import__(
                    "ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools",
                    fromlist=["_API"],
                )._API,
                {
                    "vector_index": _contradictory_failure,
                    "vector_retrieval": _contradictory_failure,
                    "vector_metadata": _contradictory_failure,
                    "enhanced_vector_index": _contradictory_failure,
                    "enhanced_vector_search": _contradictory_failure,
                    "enhanced_vector_storage": _contradictory_failure,
                },
                clear=False,
            ):
                indexed = await vector_index(action="create", index_name="idx")
                retrieved = await vector_retrieval(collection="docs")
                metadata = await vector_metadata(action="get", collection="docs")
                enhanced_indexed = await enhanced_vector_index(action="list")
                searched = await enhanced_vector_search(collection="docs", query_vector=[0.1, 0.2])
                stored = await enhanced_vector_storage(action="list", collection="docs")

            self.assertEqual(indexed.get("status"), "error")
            self.assertEqual(indexed.get("error"), "delegate failed")

            self.assertEqual(retrieved.get("status"), "error")
            self.assertEqual(retrieved.get("error"), "delegate failed")

            self.assertEqual(metadata.get("status"), "error")
            self.assertEqual(metadata.get("error"), "delegate failed")

            self.assertEqual(enhanced_indexed.get("status"), "error")
            self.assertEqual(enhanced_indexed.get("error"), "delegate failed")

            self.assertEqual(searched.get("status"), "error")
            self.assertEqual(searched.get("error"), "delegate failed")

            self.assertEqual(stored.get("status"), "error")
            self.assertEqual(stored.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
