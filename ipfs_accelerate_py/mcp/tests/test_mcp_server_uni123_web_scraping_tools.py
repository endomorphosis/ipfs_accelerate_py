#!/usr/bin/env python3
"""UNI-123 web scraping tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.web_scraping_tools import native_web_scraping_tools
from ipfs_accelerate_py.mcp_server.tools.web_scraping_tools.native_web_scraping_tools import (
    check_scraper_methods_tool,
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

    def test_scrape_url_success_defaults_with_minimal_payload(self) -> None:
        async def _minimal_scrape_url(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.web_scraping_tools.native_web_scraping_tools._API",
                {
                    "scrape_url_tool": _minimal_scrape_url,
                    "scrape_multiple_urls_tool": None,
                    "check_scraper_methods_tool": None,
                },
            ):
                result = await scrape_url_tool(url="https://example.com")
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("url"), "https://example.com")
            self.assertEqual(result.get("content"), "")
            self.assertEqual(result.get("title"), "")
            self.assertEqual(result.get("links"), [])
            self.assertEqual(result.get("method_used"), "fallback")

        anyio.run(_run)

    def test_scrape_multiple_and_methods_defaults_with_minimal_payloads(self) -> None:
        async def _minimal_scrape_multiple(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_methods() -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.web_scraping_tools.native_web_scraping_tools._API",
                {
                    "scrape_url_tool": None,
                    "scrape_multiple_urls_tool": _minimal_scrape_multiple,
                    "check_scraper_methods_tool": _minimal_methods,
                },
            ):
                multi_result = await scrape_multiple_urls_tool(urls=["https://example.com", "https://example.org"])
                methods_result = await check_scraper_methods_tool()

            self.assertEqual(multi_result.get("status"), "success")
            self.assertEqual(multi_result.get("total_urls"), 2)
            self.assertEqual(multi_result.get("results"), [])
            self.assertEqual(multi_result.get("successful_count"), 0)
            self.assertEqual(multi_result.get("failed_count"), 2)

            self.assertEqual(methods_result.get("status"), "success")
            self.assertEqual(methods_result.get("available_methods"), {})
            self.assertEqual(methods_result.get("unavailable_methods"), [])
            self.assertEqual(methods_result.get("recommended_installs"), [])
            self.assertEqual(methods_result.get("all_methods"), [])
            self.assertEqual(methods_result.get("fallback_sequence"), [])

        anyio.run(_run)

    def test_failed_delegate_payloads_infer_error_status(self) -> None:
        async def _failed(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_web_scraping_tools._API,
                {
                    "scrape_url_tool": _failed,
                    "scrape_multiple_urls_tool": _failed,
                    "check_scraper_methods_tool": _failed,
                },
                clear=False,
            ):
                scraped = await scrape_url_tool(url="https://example.com")
                self.assertEqual(scraped.get("status"), "error")

                scraped_many = await scrape_multiple_urls_tool(urls=["https://example.com"])
                self.assertEqual(scraped_many.get("status"), "error")

                methods = await check_scraper_methods_tool()
                self.assertEqual(methods.get("status"), "error")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
