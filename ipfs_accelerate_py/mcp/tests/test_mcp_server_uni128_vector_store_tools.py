#!/usr/bin/env python3
"""UNI-128 vector store tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools import (
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


if __name__ == "__main__":
    unittest.main()
