#!/usr/bin/env python3
"""UNI-106 web-archive tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.web_archive_tools.native_web_archive_tools import (
    archive_to_wayback,
    extract_dataset_from_cdxj,
    extract_links_from_warc,
    extract_metadata_from_warc,
    extract_text_from_warc,
    get_common_crawl_content,
    get_wayback_content,
    index_warc,
    list_common_crawl_indexes,
    register_native_web_archive_tools,
    search_archive_is,
    search_wayback_machine,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI106WebArchiveTools(unittest.TestCase):
    def test_register_includes_common_crawl_helpers(self) -> None:
        manager = _DummyManager()
        register_native_web_archive_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("archive_to_wayback", names)
        self.assertIn("extract_text_from_warc", names)
        self.assertIn("extract_dataset_from_cdxj", names)
        self.assertIn("extract_links_from_warc", names)
        self.assertIn("extract_metadata_from_warc", names)
        self.assertIn("index_warc", names)
        self.assertIn("get_common_crawl_content", names)
        self.assertIn("list_common_crawl_indexes", names)
        self.assertIn("search_wayback_machine", names)
        self.assertIn("search_archive_is", names)
        self.assertIn("get_wayback_content", names)

    def test_get_common_crawl_content_requires_url(self) -> None:
        async def _run() -> None:
            result = await get_common_crawl_content(url=" ", timestamp="20240101000000")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("'url' is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_get_common_crawl_content_requires_timestamp(self) -> None:
        async def _run() -> None:
            result = await get_common_crawl_content(url="https://example.com", timestamp="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("'timestamp' is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_common_crawl_helpers_fallback_shape(self) -> None:
        async def _run() -> None:
            content_result = await get_common_crawl_content(
                url="https://example.com",
                timestamp="20240101000000",
            )
            self.assertIn(content_result.get("status"), ["success", "error"])
            if content_result.get("status") == "error":
                self.assertIn("error", content_result)

            indexes_result = await list_common_crawl_indexes()
            self.assertIn(indexes_result.get("status"), ["success", "error"])
            if indexes_result.get("status") == "success":
                self.assertIn("indexes", indexes_result)

        anyio.run(_run)

    def test_search_wayback_machine_validation(self) -> None:
        async def _run() -> None:
            missing_url = await search_wayback_machine(url=" ")
            self.assertEqual(missing_url.get("status"), "error")
            self.assertIn("'url' is required", str(missing_url.get("error", "")))

            invalid_limit = await search_wayback_machine(url="https://example.com", limit=0)
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("'limit' must be greater than 0", str(invalid_limit.get("error", "")))

            invalid_format = await search_wayback_machine(
                url="https://example.com",
                output_format="xml",
            )
            self.assertEqual(invalid_format.get("status"), "error")
            self.assertIn("'output_format' must be one of", str(invalid_format.get("error", "")))

        anyio.run(_run)

    def test_search_wayback_machine_fallback_shape(self) -> None:
        async def _run() -> None:
            result = await search_wayback_machine(url="https://example.com", limit=5)
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("results", result)
                self.assertIn("count", result)

        anyio.run(_run)

    def test_search_archive_is_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_domain = await search_archive_is(domain="")
            self.assertEqual(missing_domain.get("status"), "error")
            self.assertIn("'domain' is required", str(missing_domain.get("error", "")))

            invalid_limit = await search_archive_is(domain="example.com", limit=0)
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("'limit' must be greater than 0", str(invalid_limit.get("error", "")))

            result = await search_archive_is(domain="example.com", limit=5)
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("results", result)
                self.assertIn("count", result)

        anyio.run(_run)

    def test_get_wayback_content_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_url = await get_wayback_content(url="")
            self.assertEqual(missing_url.get("status"), "error")
            self.assertIn("'url' is required", str(missing_url.get("error", "")))

            result = await get_wayback_content(url="https://example.com")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "error":
                self.assertIn("error", result)

        anyio.run(_run)

    def test_archive_to_wayback_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_url = await archive_to_wayback(url="  ")
            self.assertEqual(missing_url.get("status"), "error")
            self.assertIn("'url' is required", str(missing_url.get("error", "")))

            result = await archive_to_wayback(url="https://example.com")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "error":
                self.assertIn("error", result)

        anyio.run(_run)

    def test_warc_extractors_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_text_path = await extract_text_from_warc(warc_path="")
            self.assertEqual(missing_text_path.get("status"), "error")
            self.assertIn("'warc_path' is required", str(missing_text_path.get("error", "")))

            missing_links_path = await extract_links_from_warc(warc_path="")
            self.assertEqual(missing_links_path.get("status"), "error")
            self.assertIn("'warc_path' is required", str(missing_links_path.get("error", "")))

            missing_meta_path = await extract_metadata_from_warc(warc_path=" ")
            self.assertEqual(missing_meta_path.get("status"), "error")
            self.assertIn("'warc_path' is required", str(missing_meta_path.get("error", "")))

            missing_index_path = await index_warc(warc_path="")
            self.assertEqual(missing_index_path.get("status"), "error")
            self.assertIn("'warc_path' is required", str(missing_index_path.get("error", "")))

            missing_cdxj_path = await extract_dataset_from_cdxj(cdxj_path="")
            self.assertEqual(missing_cdxj_path.get("status"), "error")
            self.assertIn("'cdxj_path' is required", str(missing_cdxj_path.get("error", "")))

            invalid_output_format = await extract_dataset_from_cdxj(
                cdxj_path="/tmp/mock.cdxj",
                output_format="csv",
            )
            self.assertEqual(invalid_output_format.get("status"), "error")
            self.assertIn("'output_format' must be one of", str(invalid_output_format.get("error", "")))

            text_result = await extract_text_from_warc(warc_path="/tmp/mock.warc")
            self.assertIn(text_result.get("status"), ["success", "error"])
            if text_result.get("status") == "error":
                self.assertIn("error", text_result)

            links_result = await extract_links_from_warc(warc_path="/tmp/mock.warc")
            self.assertIn(links_result.get("status"), ["success", "error"])
            if links_result.get("status") == "error":
                self.assertIn("error", links_result)

            meta_result = await extract_metadata_from_warc(warc_path="/tmp/mock.warc")
            self.assertIn(meta_result.get("status"), ["success", "error"])
            if meta_result.get("status") == "error":
                self.assertIn("error", meta_result)

            index_result = await index_warc(warc_path="/tmp/mock.warc", output_path="/tmp/mock.cdxj")
            self.assertIn(index_result.get("status"), ["success", "error"])
            if index_result.get("status") == "error":
                self.assertIn("error", index_result)

            cdxj_result = await extract_dataset_from_cdxj(
                cdxj_path="/tmp/mock.cdxj",
                output_format="dict",
            )
            self.assertIn(cdxj_result.get("status"), ["success", "error"])
            if cdxj_result.get("status") == "error":
                self.assertIn("error", cdxj_result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
