#!/usr/bin/env python3
"""UNI-155 deterministic parity tests for native vector_tools."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.vector_tools import native_vector_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI155VectorTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_vector_tools.register_native_vector_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        create_props = schemas["create_vector_index"]["properties"]
        self.assertEqual(create_props["vectors"].get("minItems"), 1)
        self.assertEqual(create_props["dimension"].get("minimum"), 1)

        search_props = schemas["search_vector_index"]["properties"]
        self.assertEqual(search_props["index_id"].get("minLength"), 1)
        self.assertEqual(search_props["query_vector"].get("minItems"), 1)

        list_indexes_props = schemas["list_vector_indexes"]["properties"]
        self.assertEqual(list_indexes_props["backend"].get("default"), "all")

        manage_props = schemas["manage_vector_store"]["properties"]
        self.assertIn("create", manage_props["operation"].get("enum", []))
        self.assertEqual(manage_props["top_k"].get("minimum"), 1)

        create_store_props = schemas["create_store"]["properties"]
        self.assertEqual(create_store_props["name"].get("minLength"), 1)

        list_stores_props = schemas["list_stores"]["properties"]
        self.assertEqual(list_stores_props["include_details"].get("default"), False)

        store_info_props = schemas["get_vector_store_info"]["properties"]
        self.assertEqual(store_info_props["store_name"].get("minLength"), 1)

        orchestrate_props = schemas["orchestrate_vector_search_storage"]["properties"]
        self.assertEqual(orchestrate_props["audit_collection"].get("minLength"), 1)

    def test_create_vector_index_rejects_shape_and_dimension_mismatch(self) -> None:
        async def _run() -> None:
            invalid_vectors = await native_vector_tools.create_vector_index(vectors=[[1.0], []])
            self.assertEqual(invalid_vectors.get("status"), "error")
            self.assertIn("non-empty numeric vectors", str(invalid_vectors.get("error", "")))

            mismatched_dimension = await native_vector_tools.create_vector_index(
                vectors=[[1.0, 2.0], [3.0, 4.0]],
                dimension=3,
            )
            self.assertEqual(mismatched_dimension.get("status"), "error")
            self.assertIn("dimension must match vector length", str(mismatched_dimension.get("error", "")))

        anyio.run(_run)

    def test_search_vector_index_validates_query_and_top_k(self) -> None:
        async def _run() -> None:
            missing_query = await native_vector_tools.search_vector_index(index_id="idx")
            self.assertEqual(missing_query.get("status").lower(), "error")
            self.assertIn("query_vector", str(missing_query.get("error", "")))

            invalid_top_k = await native_vector_tools.search_vector_index(
                index_id="idx",
                query_vector=[0.1, 0.2],
                top_k="bad",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_top_k.get("status").lower(), "error")
            self.assertIn("top_k must be a positive integer", str(invalid_top_k.get("error", "")))

        anyio.run(_run)

    def test_delegate_exceptions_return_error_envelopes(self) -> None:
        async def _boom_create(**_: object) -> dict:
            raise RuntimeError("create boom")

        async def _boom_search(**_: object) -> dict:
            raise RuntimeError("search boom")

        async def _run() -> None:
            with patch.dict(
                native_vector_tools._API,
                {
                    "create_vector_index": _boom_create,
                    "search_vector_index": _boom_search,
                },
                clear=False,
            ):
                create_result = await native_vector_tools.create_vector_index(vectors=[[1.0, 2.0]])
                self.assertEqual(create_result.get("status"), "error")
                self.assertIn("create_vector_index failed", str(create_result.get("error", "")))

                search_result = await native_vector_tools.search_vector_index(
                    index_id="idx",
                    query_vector=[0.1, 0.2],
                )
                self.assertEqual(search_result.get("status"), "error")
                self.assertIn("search_vector_index failed", str(search_result.get("error", "")))

        anyio.run(_run)

    def test_vector_store_management_aliases_validate_and_normalize(self) -> None:
        async def _run() -> None:
            invalid_backend = await native_vector_tools.list_vector_indexes(backend="sqlite")
            self.assertEqual(invalid_backend.get("status"), "error")
            self.assertIn("backend must be one of", str(invalid_backend.get("error", "")))

            invalid_operation = await native_vector_tools.manage_vector_store(operation="rename")
            self.assertEqual(invalid_operation.get("status"), "error")
            self.assertIn("operation must be one of", str(invalid_operation.get("error", "")))

            invalid_index_ids = await native_vector_tools.manage_vector_store(
                operation="index",
                collection_name="legal",
                vectors=[[1.0, 2.0], [3.0, 4.0]],
                ids=["doc-1"],
            )
            self.assertEqual(invalid_index_ids.get("status"), "error")
            self.assertIn("same length as vectors", str(invalid_index_ids.get("error", "")))

            invalid_load_flag = await native_vector_tools.load_store(name="legal", create_if_missing="yes")  # type: ignore[arg-type]
            self.assertEqual(invalid_load_flag.get("status"), "error")
            self.assertIn("create_if_missing must be a boolean", str(invalid_load_flag.get("error", "")))

            created = await native_vector_tools.create_store(name="legal", backend="faiss")
            self.assertEqual(created.get("status"), "success")
            self.assertEqual(created.get("store_name"), "legal")

            indexed = await native_vector_tools.manage_vector_store(
                operation="index",
                store_type="faiss",
                collection_name="legal",
                vectors=[[1.0, 2.0], [3.0, 4.0]],
                ids=["doc-1", "doc-2"],
                metadata=[{"topic": "alpha"}, {"topic": "beta"}],
            )
            self.assertEqual(indexed.get("status"), "success")
            self.assertEqual(indexed.get("indexed_count"), 2)

            queried = await native_vector_tools.manage_vector_store(
                operation="query",
                store_type="faiss",
                collection_name="legal",
                query_vector=[1.0, 2.0],
                top_k=1,
            )
            self.assertEqual(queried.get("status"), "success")
            self.assertEqual(queried.get("results_count"), 1)

            listed = await native_vector_tools.list_stores(backend="all", include_details=True)
            self.assertEqual(listed.get("status"), "success")
            self.assertTrue(any(store.get("store_name") == "legal" for store in listed.get("stores", [])))

            info = await native_vector_tools.get_vector_store_info(store_name="legal", backend="faiss")
            self.assertEqual(info.get("status"), "success")
            self.assertEqual(info.get("vector_count"), 2)

            saved = await native_vector_tools.save_store(store_name="legal", backend="faiss")
            self.assertEqual(saved.get("status"), "success")
            self.assertEqual(saved.get("saved"), True)

            optimized = await native_vector_tools.optimize_vector_store(store_type="faiss", collection_name="legal")
            self.assertEqual(optimized.get("status"), "success")

            deleted = await native_vector_tools.delete_vector_index(index_name="legal", backend="faiss")
            self.assertIn(deleted.get("status"), ["success", "error"])
            self.assertEqual(deleted.get("backend"), "faiss")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
