#!/usr/bin/env python3
"""UNI-154 deterministic parity tests for native search tools."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.search_tools import native_search_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI154SearchTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_search_tools.register_native_search_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        semantic_props = schemas["semantic_search"]["properties"]
        self.assertEqual(semantic_props["query"].get("minLength"), 1)
        self.assertEqual(semantic_props["top_k"].get("minimum"), 1)

        similarity_props = schemas["similarity_search"]["properties"]
        self.assertEqual(similarity_props["embedding"].get("minItems"), 1)
        self.assertEqual(similarity_props["threshold"].get("maximum"), 1.0)

        faceted_props = schemas["faceted_search"]["properties"]
        self.assertEqual(faceted_props["top_k"].get("default"), 20)

    def test_semantic_validation_rejects_non_numeric_top_k_and_filters_shape(self) -> None:
        async def _run() -> None:
            non_numeric_top_k = await native_search_tools.semantic_search(
                query="hello",
                top_k="abc",  # type: ignore[arg-type]
            )
            self.assertEqual(non_numeric_top_k.get("status"), "error")
            self.assertIn("positive integer", str(non_numeric_top_k.get("message", "")))

            invalid_filters = await native_search_tools.semantic_search(
                query="hello",
                filters=["tag"],  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_filters.get("status"), "error")
            self.assertIn("filters must be an object", str(invalid_filters.get("message", "")))

        anyio.run(_run)

    def test_api_exceptions_are_wrapped_in_error_envelopes(self) -> None:
        async def _boom_semantic(**_: object) -> dict:
            raise RuntimeError("semantic exploded")

        async def _boom_similarity(**_: object) -> dict:
            raise RuntimeError("similarity exploded")

        async def _boom_faceted(**_: object) -> dict:
            raise RuntimeError("faceted exploded")

        async def _run() -> None:
            with patch.dict(
                native_search_tools._API,
                {
                    "semantic": _boom_semantic,
                    "similarity": _boom_similarity,
                    "faceted": _boom_faceted,
                },
                clear=False,
            ):
                semantic_result = await native_search_tools.semantic_search(query="hello")
                self.assertEqual(semantic_result.get("status"), "error")
                self.assertIn("semantic search failed", str(semantic_result.get("message", "")))

                similarity_result = await native_search_tools.similarity_search(embedding=[0.1, 0.2])
                self.assertEqual(similarity_result.get("status"), "error")
                self.assertIn("similarity search failed", str(similarity_result.get("message", "")))

                faceted_result = await native_search_tools.faceted_search(query="hello")
                self.assertEqual(faceted_result.get("status"), "error")
                self.assertIn("faceted search failed", str(faceted_result.get("message", "")))

        anyio.run(_run)

    def test_similarity_success_includes_default_dimension_envelope(self) -> None:
        async def _ok_similarity(**_: object) -> dict:
            return {"results": []}

        async def _run() -> None:
            with patch.dict(native_search_tools._API, {"similarity": _ok_similarity}, clear=False):
                result = await native_search_tools.similarity_search(embedding=[0.1, 0.2, 0.3])
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("embedding_dimension"), 3)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
