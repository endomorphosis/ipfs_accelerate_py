#!/usr/bin/env python3
"""UNI-139 investigation_tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.investigation_tools.native_investigation_tools import (
    analyze_entities,
    analyze_deontological_conflicts,
    analyze_entity_timeline,
    detect_patterns,
    explore_entity,
    extract_geographic_entities,
    ingest_document_collection,
    ingest_news_article,
    ingest_news_feed,
    ingest_website,
    map_spatiotemporal_events,
    map_relationships,
    query_deontic_conflicts,
    query_deontic_statements,
    query_geographic_context,
    register_native_investigation_tools,
    track_provenance,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI139InvestigationTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_investigation_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        self.assertIn("explore_entity", by_name)
        self.assertIn("ingest_news_article", by_name)
        self.assertIn("query_geographic_context", by_name)

        analyze_schema = by_name["analyze_entities"]["input_schema"]
        self.assertEqual(analyze_schema["properties"]["analysis_type"]["minLength"], 1)
        self.assertEqual(analyze_schema["properties"]["confidence_threshold"]["minimum"], 0)
        self.assertEqual(analyze_schema["properties"]["confidence_threshold"]["maximum"], 1)

        map_schema = by_name["map_relationships"]["input_schema"]
        self.assertEqual(map_schema["properties"]["min_strength"]["minimum"], 0)
        self.assertEqual(map_schema["properties"]["min_strength"]["maximum"], 1)
        self.assertEqual(map_schema["properties"]["max_depth"]["minimum"], 1)

        timeline_schema = by_name["analyze_entity_timeline"]["input_schema"]
        self.assertEqual(
            timeline_schema["properties"]["time_granularity"]["enum"],
            ["hour", "day", "week", "month"],
        )

        deontic_schema = by_name["query_deontic_statements"]["input_schema"]
        self.assertIn("obligation", deontic_schema["properties"]["modality"]["enum"])

        geo_schema = by_name["query_geographic_context"]["input_schema"]
        self.assertEqual(geo_schema["properties"]["radius_km"]["exclusiveMinimum"], 0)

    def test_analyze_entities_validation(self) -> None:
        async def _run() -> None:
            result = await analyze_entities(corpus_data="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("corpus_data", str(result.get("error", "")))

            result = await analyze_entities(corpus_data="{}", confidence_threshold=1.5)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("confidence_threshold", str(result.get("error", "")))

        anyio.run(_run)

    def test_map_relationships_validation(self) -> None:
        async def _run() -> None:
            result = await map_relationships(corpus_data="{}", max_depth=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_depth", str(result.get("error", "")))

            result = await map_relationships(corpus_data="{}", min_strength=-0.1)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("min_strength", str(result.get("error", "")))

        anyio.run(_run)

    def test_expanded_validation_contracts(self) -> None:
        async def _run() -> None:
            result = await explore_entity(entity_id="", corpus_data="{}")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("entity_id", str(result.get("error", "")))

            result = await analyze_entity_timeline(corpus_data="{}", entity_id="e1", time_granularity="year")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("time_granularity", str(result.get("error", "")))

            result = await detect_patterns(corpus_data="{}", pattern_types=["made_up"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("pattern_types", str(result.get("error", "")))

            result = await track_provenance(corpus_data="{}", entity_id="e1", trace_depth=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("trace_depth", str(result.get("error", "")))

            result = await ingest_news_article(url=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("url", str(result.get("error", "")))

            result = await ingest_news_feed(feed_url="https://example.com/feed", max_articles=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_articles", str(result.get("error", "")))

            result = await ingest_website(base_url="https://example.com", max_pages=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_pages", str(result.get("error", "")))

            result = await ingest_document_collection(document_paths=123)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("document_paths", str(result.get("error", "")))

            result = await analyze_deontological_conflicts(corpus_data="{}", severity_threshold="urgent")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("severity_threshold", str(result.get("error", "")))

            result = await query_deontic_statements(corpus_data="{}", modality="should")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("modality", str(result.get("error", "")))

            result = await query_deontic_conflicts(corpus_data="{}", severity="critical")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("severity", str(result.get("error", "")))

            result = await extract_geographic_entities(corpus_data="{}", confidence_threshold=2.0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("confidence_threshold", str(result.get("error", "")))

            result = await map_spatiotemporal_events(corpus_data="{}", clustering_distance=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("clustering_distance", str(result.get("error", "")))

            result = await query_geographic_context(query="", corpus_data="{}")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            analyze_result = await analyze_entities(
                corpus_data='{"documents": []}',
                analysis_type="comprehensive",
            )
            self.assertIn(analyze_result.get("status"), ["success", "error"])
            self.assertEqual(analyze_result.get("analysis_type"), "comprehensive")

            map_result = await map_relationships(
                corpus_data='{"documents": []}',
                max_depth=2,
            )
            self.assertIn(map_result.get("status"), ["success", "error"])
            self.assertEqual(map_result.get("max_depth"), 2)

            explore_result = await explore_entity(entity_id="entity-1", corpus_data='{"documents": []}')
            self.assertIn(explore_result.get("status"), ["success", "error"])
            self.assertEqual(explore_result.get("entity_id"), "entity-1")

            timeline_result = await analyze_entity_timeline(
                corpus_data='{"documents": []}',
                entity_id="entity-1",
                time_granularity="day",
            )
            self.assertIn(timeline_result.get("status"), ["success", "error"])
            self.assertEqual(timeline_result.get("entity_id"), "entity-1")

            deontic_result = await query_deontic_statements(
                corpus_data='{"documents": []}',
                modality="obligation",
            )
            self.assertIn(deontic_result.get("status"), ["success", "error"])

            geo_result = await query_geographic_context(
                query="incident near river",
                corpus_data='{"documents": []}',
                radius_km=25,
            )
            self.assertIn(geo_result.get("status"), ["success", "error"])
            self.assertEqual(geo_result.get("query"), "incident near river")

        anyio.run(_run)

    def test_analyze_entities_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.investigation_tools.native_investigation_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await analyze_entities(corpus_data='{"documents": []}')

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("analysis_type"), "comprehensive")
            self.assertEqual(result.get("entities"), [])
            self.assertEqual(result.get("relationships"), [])
            self.assertEqual(result.get("clusters"), [])
            self.assertEqual(result.get("statistics"), {})

        anyio.run(_run)

    def test_map_relationships_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.investigation_tools.native_investigation_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await map_relationships(corpus_data='{"documents": []}', max_depth=2)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("max_depth"), 2)
            self.assertEqual(result.get("entities"), [])
            self.assertEqual(result.get("relationships"), [])
            self.assertEqual(result.get("graph_metrics"), {})

        anyio.run(_run)

    def test_query_geographic_context_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.investigation_tools.native_investigation_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"error": "geo backend unavailable"}

                result = await query_geographic_context(
                    query="incident near river",
                    corpus_data='{"documents": []}',
                    radius_km=25,
                )

            self.assertEqual(result.get("status"), "error")
            self.assertIn("geo backend unavailable", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
