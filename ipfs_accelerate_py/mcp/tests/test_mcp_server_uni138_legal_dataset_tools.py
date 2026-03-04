#!/usr/bin/env python3
"""UNI-138 legal_dataset_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.legal_dataset_tools.native_legal_dataset_tools import (
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


if __name__ == "__main__":
    unittest.main()
