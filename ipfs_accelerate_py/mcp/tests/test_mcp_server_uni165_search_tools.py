#!/usr/bin/env python3
"""UNI-165 deterministic faceted-search parity tests for native search tools."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
