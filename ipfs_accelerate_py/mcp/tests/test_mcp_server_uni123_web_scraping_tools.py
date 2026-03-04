#!/usr/bin/env python3
"""UNI-123 web scraping tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.web_scraping_tools.native_web_scraping_tools import (
    register_native_web_scraping_tools,
    scrape_multiple_urls_tool,
    scrape_url_tool,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI123WebScrapingTools(unittest.TestCase):
    def test_register_includes_web_scraping_tools(self) -> None:
        manager = _DummyManager()
        register_native_web_scraping_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("scrape_url_tool", names)
        self.assertIn("scrape_multiple_urls_tool", names)
        self.assertIn("check_scraper_methods_tool", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_web_scraping_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        scrape_schema = by_name["scrape_url_tool"]["input_schema"]
        self.assertIn("playwright", scrape_schema["properties"]["method"].get("enum", []))

        multi_schema = by_name["scrape_multiple_urls_tool"]["input_schema"]
        self.assertEqual(multi_schema["properties"]["max_concurrent"].get("minimum"), 1)

    def test_scrape_url_rejects_empty_url(self) -> None:
        async def _run() -> None:
            result = await scrape_url_tool(url="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("url is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_scrape_url_rejects_invalid_method(self) -> None:
        async def _run() -> None:
            result = await scrape_url_tool(url="https://example.com", method="selenium")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("method must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_scrape_multiple_rejects_bad_urls_shape(self) -> None:
        async def _run() -> None:
            result = await scrape_multiple_urls_tool(urls=["https://example.com", "   "])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("urls cannot contain empty strings", str(result.get("message", "")))

        anyio.run(_run)

    def test_scrape_multiple_rejects_bad_max_concurrent(self) -> None:
        async def _run() -> None:
            result = await scrape_multiple_urls_tool(urls=["https://example.com"], max_concurrent=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_concurrent must be an integer >= 1", str(result.get("message", "")))

        anyio.run(_run)

    def test_scrape_url_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await scrape_url_tool(url="https://example.com", method="requests_only")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("url"), "https://example.com")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
