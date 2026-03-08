#!/usr/bin/env python3
"""UNI-165 deterministic faceted-search parity tests for native search tools."""

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


class TestMCPServerUNI165SearchTools(unittest.TestCase):
    def test_registration_schema_tightens_facets_and_aggregations(self) -> None:
        manager = _DummyManager()
        native_search_tools.register_native_search_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        faceted_props = schemas["faceted_search"]["properties"]
        facets_schema = faceted_props["facets"]
        facets_items = facets_schema["additionalProperties"]["items"]
        self.assertEqual(facets_items.get("minLength"), 1)

        aggregations_schema = faceted_props["aggregations"]
        self.assertEqual((aggregations_schema.get("items") or {}).get("minLength"), 1)

    def test_faceted_search_rejects_invalid_facets_value_shape(self) -> None:
        async def _run() -> None:
            result = await native_search_tools.faceted_search(
                query="hello",
                facets={"category": "news"},  # type: ignore[dict-item]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "each facets value must be an array of non-empty strings",
                str(result.get("message", "")),
            )

        anyio.run(_run)

    def test_faceted_search_rejects_invalid_aggregation_items(self) -> None:
        async def _run() -> None:
            result = await native_search_tools.faceted_search(
                query="hello",
                aggregations=["category", "   "],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "aggregations must contain only non-empty strings",
                str(result.get("message", "")),
            )

        anyio.run(_run)

    def test_faceted_search_success_with_normalized_filters(self) -> None:
        async def _run() -> None:
            result = await native_search_tools.faceted_search(
                query="hello",
                facets={" category ": [" news ", "legal"]},
                aggregations=[" category "],
                top_k=2,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertIn("results", result)

        anyio.run(_run)

    def test_semantic_search_success_applies_deterministic_result_defaults(self) -> None:
        async def _semantic_minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch.dict(native_search_tools._API, {"semantic": _semantic_minimal}, clear=False):
                result = await native_search_tools.semantic_search(
                    query="hello",
                    model="demo-model",
                    top_k=3,
                    collection="demo-collection",
                )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("query"), "hello")
            self.assertEqual(result.get("model"), "demo-model")
            self.assertEqual(result.get("top_k"), 3)
            self.assertEqual(result.get("collection"), "demo-collection")
            self.assertEqual(result.get("results"), [])
            self.assertEqual(result.get("total_found"), 0)

        anyio.run(_run)

    def test_similarity_search_success_applies_deterministic_result_defaults(self) -> None:
        async def _similarity_minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch.dict(native_search_tools._API, {"similarity": _similarity_minimal}, clear=False):
                result = await native_search_tools.similarity_search(
                    embedding=[0.1, 0.2, 0.3],
                    top_k=4,
                    threshold=0.2,
                    collection="demo-collection",
                )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("embedding_dimension"), 3)
            self.assertEqual(result.get("top_k"), 4)
            self.assertEqual(result.get("threshold"), 0.2)
            self.assertEqual(result.get("collection"), "demo-collection")
            self.assertEqual(result.get("results"), [])
            self.assertEqual(result.get("total_found"), 0)

        anyio.run(_run)

    def test_faceted_search_success_applies_deterministic_result_defaults(self) -> None:
        async def _faceted_minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch.dict(native_search_tools._API, {"faceted": _faceted_minimal}, clear=False):
                result = await native_search_tools.faceted_search(
                    query="hello",
                    facets={"category": ["news"]},
                    aggregations=["category"],
                    top_k=4,
                    collection="demo-collection",
                )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("query"), "hello")
            self.assertEqual(result.get("facets"), {"category": ["news"]})
            self.assertEqual(result.get("aggregations"), ["category"])
            self.assertEqual(result.get("top_k"), 4)
            self.assertEqual(result.get("collection"), "demo-collection")
            self.assertEqual(result.get("results"), [])
            self.assertEqual(result.get("facet_counts"), {})
            self.assertEqual(result.get("total_found"), 0)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
