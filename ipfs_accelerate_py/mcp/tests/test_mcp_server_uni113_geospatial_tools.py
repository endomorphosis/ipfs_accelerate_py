#!/usr/bin/env python3
"""UNI-113 geospatial tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.geospatial_tools.native_geospatial_tools import (
    extract_geographic_entities,
    map_spatiotemporal_events,
    query_geographic_context,
    register_native_geospatial_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI113GeospatialTools(unittest.TestCase):
    def test_register_includes_geospatial_tools(self) -> None:
        manager = _DummyManager()
        register_native_geospatial_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("extract_geographic_entities", names)
        self.assertIn("map_spatiotemporal_events", names)
        self.assertIn("query_geographic_context", names)

    def test_extract_geographic_entities_rejects_invalid_threshold(self) -> None:
        async def _run() -> None:
            result = await extract_geographic_entities(corpus_data="sample", confidence_threshold=2.0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("between 0 and 1", str(result.get("message", "")))

        anyio.run(_run)

    def test_map_spatiotemporal_events_rejects_invalid_resolution(self) -> None:
        async def _run() -> None:
            result = await map_spatiotemporal_events(corpus_data="sample", temporal_resolution="quarter")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_query_geographic_context_rejects_invalid_radius(self) -> None:
        async def _run() -> None:
            result = await query_geographic_context(query="city", corpus_data="sample", radius_km=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be positive", str(result.get("message", "")))

        anyio.run(_run)

    def test_query_geographic_context_rejects_empty_query(self) -> None:
        async def _run() -> None:
            result = await query_geographic_context(query="   ", corpus_data="sample")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_map_spatiotemporal_events_success_shape(self) -> None:
        async def _run() -> None:
            result = await map_spatiotemporal_events(corpus_data="sample", temporal_resolution="day")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("clusters", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
