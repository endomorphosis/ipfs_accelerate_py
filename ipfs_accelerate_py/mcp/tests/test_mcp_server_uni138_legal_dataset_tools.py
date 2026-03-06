#!/usr/bin/env python3
"""UNI-138 legal_dataset_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.legal_dataset_tools.native_legal_dataset_tools import (
    expand_legal_query,
    get_legal_relationships,
    get_legal_synonyms,
    list_state_jurisdictions,
    register_native_legal_dataset_tools,
    scrape_state_laws,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI138LegalDatasetTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_legal_dataset_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        scrape_schema = by_name["scrape_state_laws"]["input_schema"]
        props = scrape_schema["properties"]

        self.assertEqual(props["output_format"]["enum"], ["json", "csv", "parquet"])
        self.assertEqual(props["rate_limit_delay"]["minimum"], 0)
        self.assertEqual(props["min_full_text_chars"]["minimum"], 1)

        expand_schema = by_name["expand_legal_query"]["input_schema"]
        expand_props = expand_schema["properties"]
        self.assertEqual(expand_props["strategy"]["enum"], ["conservative", "balanced", "aggressive"])
        self.assertEqual(expand_props["max_expansions"]["maximum"], 50)

        relationship_schema = by_name["get_legal_relationships"]["input_schema"]
        relationship_props = relationship_schema["properties"]
        self.assertIn("hierarchical", relationship_props["relationship_type"]["enum"])

    def test_scrape_state_laws_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_state_laws(states=[""], output_format="json")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("states", str(result.get("error", "")))

            result = await scrape_state_laws(output_format="xml")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("output_format", str(result.get("error", "")))

            result = await scrape_state_laws(output_format="json", min_full_text_chars=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("min_full_text_chars", str(result.get("error", "")))

        anyio.run(_run)

    def test_scrape_state_laws_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await scrape_state_laws(
                states=["ca"],
                output_format="json",
                include_metadata=True,
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("output_format"), "json")
            self.assertEqual(result.get("states"), ["CA"])

        anyio.run(_run)

    def test_list_state_jurisdictions_success_shape(self) -> None:
        async def _run() -> None:
            result = await list_state_jurisdictions()
            self.assertIn(result.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_expand_legal_query_validation(self) -> None:
        async def _run() -> None:
            result = await expand_legal_query(query="", strategy="balanced")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query", str(result.get("error", "")))

            result = await expand_legal_query(query="epa water rules", strategy="wide")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("strategy", str(result.get("error", "")))

            result = await expand_legal_query(query="epa water rules", max_expansions=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_expansions", str(result.get("error", "")))

            result = await expand_legal_query(query="epa water rules", domains=[""])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("domains", str(result.get("error", "")))

        anyio.run(_run)

    def test_expand_legal_query_success_shape(self) -> None:
        async def _run() -> None:
            result = await expand_legal_query(
                query="epa water rules",
                strategy="balanced",
                max_expansions=3,
                domains=["environmental"],
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("original_query"), "epa water rules")
            self.assertEqual(result.get("strategy_used"), "balanced")
            self.assertIn("total_expansions", result)

        anyio.run(_run)

    def test_get_legal_synonyms_validation_and_shape(self) -> None:
        async def _run() -> None:
            invalid = await get_legal_synonyms(term="   ")
            self.assertEqual(invalid.get("status"), "error")
            self.assertIn("term", str(invalid.get("error", "")))

            result = await get_legal_synonyms(term="regulation")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("term"), "regulation")

        anyio.run(_run)

    def test_get_legal_relationships_validation_and_shape(self) -> None:
        async def _run() -> None:
            invalid = await get_legal_relationships(relationship_type="graph")
            self.assertEqual(invalid.get("status"), "error")
            self.assertIn("relationship_type", str(invalid.get("error", "")))

            result = await get_legal_relationships(term="regulation", relationship_type="hierarchical")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("term"), "regulation")
            self.assertEqual(result.get("relationship_type"), "hierarchical")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
