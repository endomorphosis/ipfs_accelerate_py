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


if __name__ == "__main__":
    unittest.main()
