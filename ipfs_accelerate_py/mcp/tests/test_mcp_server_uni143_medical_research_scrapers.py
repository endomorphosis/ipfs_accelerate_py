#!/usr/bin/env python3
"""UNI-143 medical_research_scrapers parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.medical_research_scrapers.native_medical_research_scrapers import (
    register_native_medical_research_scrapers,
    scrape_clinical_trials,
    scrape_pubmed_medical_research,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI143MedicalResearchScrapers(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_medical_research_scrapers(manager)
        by_name = {call["name"]: call for call in manager.calls}

        pubmed_schema = by_name["scrape_pubmed_medical_research"]["input_schema"]
        self.assertEqual(pubmed_schema["properties"]["query"]["minLength"], 1)
        self.assertEqual(pubmed_schema["properties"]["max_results"]["minimum"], 1)

        trials_schema = by_name["scrape_clinical_trials"]["input_schema"]
        self.assertEqual(trials_schema["properties"]["query"]["minLength"], 1)
        self.assertEqual(trials_schema["properties"]["max_results"]["minimum"], 1)

    def test_scrape_pubmed_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_pubmed_medical_research(query="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query", str(result.get("error", "")))

            result = await scrape_pubmed_medical_research(query="diabetes", max_results=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_results", str(result.get("error", "")))

            result = await scrape_pubmed_medical_research(
                query="diabetes", email="   "
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("email", str(result.get("error", "")))

        anyio.run(_run)

    def test_scrape_clinical_trials_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_clinical_trials(query="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query", str(result.get("error", "")))

            result = await scrape_clinical_trials(query="diabetes", max_results=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_results", str(result.get("error", "")))

            result = await scrape_clinical_trials(query="diabetes", condition="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("condition", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            pubmed_result = await scrape_pubmed_medical_research(
                query="diabetes",
                max_results=1,
            )
            self.assertIn(pubmed_result.get("status"), ["success", "error"])
            self.assertEqual(pubmed_result.get("query"), "diabetes")
            self.assertEqual(pubmed_result.get("max_results"), 1)

            trials_result = await scrape_clinical_trials(
                query="diabetes",
                max_results=1,
            )
            self.assertIn(trials_result.get("status"), ["success", "error"])
            self.assertEqual(trials_result.get("query"), "diabetes")
            self.assertEqual(trials_result.get("max_results"), 1)

        anyio.run(_run)

    def test_scrape_pubmed_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.medical_research_scrapers.native_medical_research_scrapers._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await scrape_pubmed_medical_research(query="diabetes", max_results=3)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("articles"), [])
            self.assertEqual(result.get("total_count"), 0)

        anyio.run(_run)

    def test_scrape_clinical_trials_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.medical_research_scrapers.native_medical_research_scrapers._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"error": "clinical trials unavailable"}

                mock_api.__getitem__.return_value = _impl
                result = await scrape_clinical_trials(query="diabetes", max_results=3)

            self.assertEqual(result.get("status"), "error")
            self.assertIn("clinical trials unavailable", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
