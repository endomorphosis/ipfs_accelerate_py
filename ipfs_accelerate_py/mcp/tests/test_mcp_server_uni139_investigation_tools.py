#!/usr/bin/env python3
"""UNI-139 investigation_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.investigation_tools.native_investigation_tools import (
    analyze_entities,
    map_relationships,
    register_native_investigation_tools,
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

        analyze_schema = by_name["analyze_entities"]["input_schema"]
        self.assertEqual(analyze_schema["properties"]["analysis_type"]["minLength"], 1)
        self.assertEqual(analyze_schema["properties"]["confidence_threshold"]["minimum"], 0)
        self.assertEqual(analyze_schema["properties"]["confidence_threshold"]["maximum"], 1)

        map_schema = by_name["map_relationships"]["input_schema"]
        self.assertEqual(map_schema["properties"]["min_strength"]["minimum"], 0)
        self.assertEqual(map_schema["properties"]["min_strength"]["maximum"], 1)
        self.assertEqual(map_schema["properties"]["max_depth"]["minimum"], 1)

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

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
