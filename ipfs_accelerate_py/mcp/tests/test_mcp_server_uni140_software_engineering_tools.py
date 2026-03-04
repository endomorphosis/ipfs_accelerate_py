#!/usr/bin/env python3
"""UNI-140 software_engineering_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.software_engineering_tools.native_software_engineering_tools import (
    register_native_software_engineering_tools,
    scrape_repository,
    search_repositories,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI140SoftwareEngineeringTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_software_engineering_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        scrape_schema = by_name["scrape_repository"]["input_schema"]
        self.assertEqual(scrape_schema["properties"]["max_items"]["minimum"], 1)

        search_schema = by_name["search_repositories"]["input_schema"]
        self.assertEqual(search_schema["properties"]["max_results"]["minimum"], 1)

    def test_scrape_repository_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_repository(repository_url="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("repository_url", str(result.get("error", "")))

            result = await scrape_repository(repository_url="https://example.com/repo")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("github.com", str(result.get("error", "")))

            result = await scrape_repository(
                repository_url="https://github.com/example/repo",
                max_items=0,
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_items", str(result.get("error", "")))

        anyio.run(_run)

    def test_search_repositories_validation(self) -> None:
        async def _run() -> None:
            result = await search_repositories(query="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query", str(result.get("error", "")))

            result = await search_repositories(query="smoke", max_results=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_results", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            scrape_result = await scrape_repository(
                repository_url="https://github.com/example/repo",
                max_items=1,
            )
            self.assertIn(scrape_result.get("status"), ["success", "error"])
            self.assertEqual(scrape_result.get("repository_url"), "https://github.com/example/repo")

            search_result = await search_repositories(query="mcp", max_results=1)
            self.assertIn(search_result.get("status"), ["success", "error"])
            self.assertEqual(search_result.get("query"), "mcp")
            self.assertEqual(search_result.get("max_results"), 1)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
