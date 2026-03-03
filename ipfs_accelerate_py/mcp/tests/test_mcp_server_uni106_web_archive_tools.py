#!/usr/bin/env python3
"""UNI-106 web-archive tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.web_archive_tools.native_web_archive_tools import (
    archive_to_archive_is,
    archive_to_wayback,
    batch_archive_to_archive_is,
    batch_scrape_with_autoscraper,
    check_archive_status,
    create_autoscraper_model,
    extract_dataset_from_cdxj,
    extract_links_from_warc,
    extract_metadata_from_warc,
    extract_text_from_warc,
    fetch_warc_record_advanced,
    get_archive_is_content,
    get_common_crawl_collection_info_advanced,
    get_common_crawl_content,
    get_ipwb_content,
    get_wayback_content,
    index_warc,
    index_warc_to_ipwb,
    list_common_crawl_indexes,
    list_common_crawl_collections_advanced,
    list_autoscraper_models,
    optimize_autoscraper_model,
    register_native_web_archive_tools,
    scrape_with_autoscraper,
    search_archive_is,
    search_common_crawl_advanced,
    search_ipwb_archive,
    search_wayback_machine,
    start_ipwb_replay,
    verify_ipwb_archive,
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
        self.assertIn("archive_to_archive_is", names)
        self.assertIn("batch_archive_to_archive_is", names)
        self.assertIn("extract_text_from_warc", names)
        self.assertIn("extract_dataset_from_cdxj", names)
        self.assertIn("extract_links_from_warc", names)
        self.assertIn("extract_metadata_from_warc", names)
        self.assertIn("fetch_warc_record_advanced", names)
        self.assertIn("index_warc", names)
        self.assertIn("index_warc_to_ipwb", names)
        self.assertIn("start_ipwb_replay", names)
        self.assertIn("search_ipwb_archive", names)
        self.assertIn("get_ipwb_content", names)
        self.assertIn("verify_ipwb_archive", names)
        self.assertIn("get_common_crawl_content", names)
        self.assertIn("list_common_crawl_indexes", names)
        self.assertIn("list_common_crawl_collections_advanced", names)
        self.assertIn("get_common_crawl_collection_info_advanced", names)
        self.assertIn("list_autoscraper_models", names)
        self.assertIn("create_autoscraper_model", names)
        self.assertIn("scrape_with_autoscraper", names)
        self.assertIn("optimize_autoscraper_model", names)
        self.assertIn("batch_scrape_with_autoscraper", names)
        self.assertIn("search_wayback_machine", names)
        self.assertIn("search_archive_is", names)
        self.assertIn("search_common_crawl_advanced", names)
        self.assertIn("get_archive_is_content", names)
        self.assertIn("check_archive_status", names)
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

            advanced_missing_domain = await search_common_crawl_advanced(domain="")
            self.assertEqual(advanced_missing_domain.get("status"), "error")
            self.assertIn("'domain' is required", str(advanced_missing_domain.get("error", "")))

            advanced_invalid_max = await search_common_crawl_advanced(
                domain="example.com",
                max_matches=0,
            )
            self.assertEqual(advanced_invalid_max.get("status"), "error")
            self.assertIn("'max_matches' must be greater than 0", str(advanced_invalid_max.get("error", "")))

            advanced_search_result = await search_common_crawl_advanced(domain="example.com", max_matches=5)
            self.assertIn(advanced_search_result.get("status"), ["success", "error"])
            if advanced_search_result.get("status") == "success":
                self.assertIn("results", advanced_search_result)

            missing_warc_filename = await fetch_warc_record_advanced(
                warc_filename="",
                warc_offset=0,
                warc_length=100,
            )
            self.assertEqual(missing_warc_filename.get("status"), "error")
            self.assertIn("'warc_filename' is required", str(missing_warc_filename.get("error", "")))

            invalid_warc_offset = await fetch_warc_record_advanced(
                warc_filename="sample.warc.gz",
                warc_offset=-1,
                warc_length=100,
            )
            self.assertEqual(invalid_warc_offset.get("status"), "error")
            self.assertIn("'warc_offset' must be >= 0", str(invalid_warc_offset.get("error", "")))

            invalid_warc_length = await fetch_warc_record_advanced(
                warc_filename="sample.warc.gz",
                warc_offset=0,
                warc_length=0,
            )
            self.assertEqual(invalid_warc_length.get("status"), "error")
            self.assertIn("'warc_length' must be greater than 0", str(invalid_warc_length.get("error", "")))

            warc_fetch_result = await fetch_warc_record_advanced(
                warc_filename="sample.warc.gz",
                warc_offset=0,
                warc_length=100,
            )
            self.assertIn(warc_fetch_result.get("status"), ["success", "error"])
            if warc_fetch_result.get("status") == "error":
                self.assertIn("error", warc_fetch_result)

            collections_result = await list_common_crawl_collections_advanced()
            self.assertIn(collections_result.get("status"), ["success", "error"])
            if collections_result.get("status") == "success":
                self.assertIn("collections", collections_result)

            missing_collection = await get_common_crawl_collection_info_advanced(collection="")
            self.assertEqual(missing_collection.get("status"), "error")
            self.assertIn("'collection' is required", str(missing_collection.get("error", "")))

            collection_info_result = await get_common_crawl_collection_info_advanced(
                collection="CC-MAIN-2024-10"
            )
            self.assertIn(collection_info_result.get("status"), ["success", "error"])
            if collection_info_result.get("status") == "error":
                self.assertIn("error", collection_info_result)

        anyio.run(_run)

    def test_autoscraper_helpers_validation_and_shape(self) -> None:
        async def _run() -> None:
            models_result = await list_autoscraper_models()
            self.assertIn(models_result.get("status"), ["success", "error"])
            if models_result.get("status") == "success":
                self.assertIn("models", models_result)

            missing_model_path = await scrape_with_autoscraper(model_path="", target_urls=["https://example.com"])
            self.assertEqual(missing_model_path.get("status"), "error")
            self.assertIn("'model_path' is required", str(missing_model_path.get("error", "")))

            missing_target_urls = await scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                target_urls=[],
            )
            self.assertEqual(missing_target_urls.get("status"), "error")
            self.assertIn("'target_urls' must be a non-empty list", str(missing_target_urls.get("error", "")))

            empty_target_urls = await scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                target_urls=[" "],
            )
            self.assertEqual(empty_target_urls.get("status"), "error")
            self.assertIn(
                "'target_urls' must contain at least one non-empty URL",
                str(empty_target_urls.get("error", "")),
            )

            scrape_result = await scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                target_urls=["https://example.com"],
                grouped=True,
            )
            self.assertIn(scrape_result.get("status"), ["success", "error"])
            if scrape_result.get("status") == "error":
                self.assertIn("error", scrape_result)

            missing_sample_url = await create_autoscraper_model(
                sample_url="",
                wanted_data=["item"],
                model_name="model1",
            )
            self.assertEqual(missing_sample_url.get("status"), "error")
            self.assertIn("'sample_url' is required", str(missing_sample_url.get("error", "")))

            missing_model_name = await create_autoscraper_model(
                sample_url="https://example.com",
                wanted_data=["item"],
                model_name=" ",
            )
            self.assertEqual(missing_model_name.get("status"), "error")
            self.assertIn("'model_name' is required", str(missing_model_name.get("error", "")))

            missing_wanted_data = await create_autoscraper_model(
                sample_url="https://example.com",
                wanted_data=[],
                model_name="model1",
            )
            self.assertEqual(missing_wanted_data.get("status"), "error")
            self.assertIn("'wanted_data' must be a non-empty list", str(missing_wanted_data.get("error", "")))

            create_result = await create_autoscraper_model(
                sample_url="https://example.com",
                wanted_data=["item"],
                model_name="model1",
            )
            self.assertIn(create_result.get("status"), ["success", "error"])
            if create_result.get("status") == "error":
                self.assertIn("error", create_result)

            missing_opt_path = await optimize_autoscraper_model(
                model_path="",
                new_sample_urls=["https://example.com"],
            )
            self.assertEqual(missing_opt_path.get("status"), "error")
            self.assertIn("'model_path' is required", str(missing_opt_path.get("error", "")))

            missing_opt_urls = await optimize_autoscraper_model(
                model_path="/tmp/mock.pkl",
                new_sample_urls=[],
            )
            self.assertEqual(missing_opt_urls.get("status"), "error")
            self.assertIn("'new_sample_urls' must be a non-empty list", str(missing_opt_urls.get("error", "")))

            empty_opt_urls = await optimize_autoscraper_model(
                model_path="/tmp/mock.pkl",
                new_sample_urls=[" "],
            )
            self.assertEqual(empty_opt_urls.get("status"), "error")
            self.assertIn(
                "'new_sample_urls' must contain at least one non-empty URL",
                str(empty_opt_urls.get("error", "")),
            )

            optimize_result = await optimize_autoscraper_model(
                model_path="/tmp/mock.pkl",
                new_sample_urls=["https://example.com"],
            )
            self.assertIn(optimize_result.get("status"), ["success", "error"])
            if optimize_result.get("status") == "error":
                self.assertIn("error", optimize_result)

            missing_batch_path = await batch_scrape_with_autoscraper(
                model_path="",
                urls_file="/tmp/urls.txt",
            )
            self.assertEqual(missing_batch_path.get("status"), "error")
            self.assertIn("'model_path' is required", str(missing_batch_path.get("error", "")))

            missing_urls_file = await batch_scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                urls_file="",
            )
            self.assertEqual(missing_urls_file.get("status"), "error")
            self.assertIn("'urls_file' is required", str(missing_urls_file.get("error", "")))

            invalid_output = await batch_scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                urls_file="/tmp/urls.txt",
                output_format="xml",
            )
            self.assertEqual(invalid_output.get("status"), "error")
            self.assertIn("'output_format' must be one of", str(invalid_output.get("error", "")))

            invalid_batch_size = await batch_scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                urls_file="/tmp/urls.txt",
                batch_size=0,
            )
            self.assertEqual(invalid_batch_size.get("status"), "error")
            self.assertIn("'batch_size' must be greater than 0", str(invalid_batch_size.get("error", "")))

            invalid_delay = await batch_scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                urls_file="/tmp/urls.txt",
                delay_seconds=-1,
            )
            self.assertEqual(invalid_delay.get("status"), "error")
            self.assertIn("'delay_seconds' must be >= 0", str(invalid_delay.get("error", "")))

            batch_result = await batch_scrape_with_autoscraper(
                model_path="/tmp/mock.pkl",
                urls_file="/tmp/urls.txt",
                output_format="json",
                batch_size=10,
                delay_seconds=0,
            )
            self.assertIn(batch_result.get("status"), ["success", "error"])
            if batch_result.get("status") == "error":
                self.assertIn("error", batch_result)

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

    def test_archive_is_content_and_status_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_archive_url = await get_archive_is_content(archive_url="")
            self.assertEqual(missing_archive_url.get("status"), "error")
            self.assertIn("'archive_url' is required", str(missing_archive_url.get("error", "")))

            missing_submission_id = await check_archive_status(submission_id=" ")
            self.assertEqual(missing_submission_id.get("status"), "error")
            self.assertIn("'submission_id' is required", str(missing_submission_id.get("error", "")))

            content_result = await get_archive_is_content(archive_url="https://archive.is/abc123")
            self.assertIn(content_result.get("status"), ["success", "error"])
            if content_result.get("status") == "error":
                self.assertIn("error", content_result)

            status_result = await check_archive_status(submission_id="abc123")
            self.assertIn(status_result.get("status"), ["success", "pending", "error"])
            if status_result.get("status") == "error":
                self.assertIn("error", status_result)

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

    def test_archive_to_archive_is_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_url = await archive_to_archive_is(url="")
            self.assertEqual(missing_url.get("status"), "error")
            self.assertIn("'url' is required", str(missing_url.get("error", "")))

            invalid_timeout = await archive_to_archive_is(url="https://example.com", timeout=0)
            self.assertEqual(invalid_timeout.get("status"), "error")
            self.assertIn("'timeout' must be greater than 0", str(invalid_timeout.get("error", "")))

            result = await archive_to_archive_is(url="https://example.com", wait_for_completion=False)
            self.assertIn(result.get("status"), ["success", "pending", "error", "timeout"])
            if result.get("status") == "error":
                self.assertIn("error", result)

        anyio.run(_run)

    def test_batch_archive_to_archive_is_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_urls = await batch_archive_to_archive_is(urls=[])
            self.assertEqual(missing_urls.get("status"), "error")
            self.assertIn("'urls' must be a non-empty list", str(missing_urls.get("error", "")))

            empty_urls = await batch_archive_to_archive_is(urls=[" "])
            self.assertEqual(empty_urls.get("status"), "error")
            self.assertIn("'urls' must contain at least one non-empty URL", str(empty_urls.get("error", "")))

            invalid_delay = await batch_archive_to_archive_is(
                urls=["https://example.com"],
                delay_seconds=-1,
            )
            self.assertEqual(invalid_delay.get("status"), "error")
            self.assertIn("'delay_seconds' must be >= 0", str(invalid_delay.get("error", "")))

            invalid_max_concurrent = await batch_archive_to_archive_is(
                urls=["https://example.com"],
                max_concurrent=0,
            )
            self.assertEqual(invalid_max_concurrent.get("status"), "error")
            self.assertIn(
                "'max_concurrent' must be greater than 0",
                str(invalid_max_concurrent.get("error", "")),
            )

            result = await batch_archive_to_archive_is(
                urls=["https://example.com", "https://example.org"],
                delay_seconds=0,
                max_concurrent=1,
            )
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("results", result)
                self.assertIn("success_count", result)

        anyio.run(_run)

    def test_ipwb_wrappers_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_warc_path = await index_warc_to_ipwb(warc_path="")
            self.assertEqual(missing_warc_path.get("status"), "error")
            self.assertIn("'warc_path' is required", str(missing_warc_path.get("error", "")))

            invalid_compression = await index_warc_to_ipwb(
                warc_path="/tmp/mock.warc",
                compression="zip",
            )
            self.assertEqual(invalid_compression.get("status"), "error")
            self.assertIn("'compression' must be one of", str(invalid_compression.get("error", "")))

            missing_cdxj_replay = await start_ipwb_replay(cdxj_path="", port=5000)
            self.assertEqual(missing_cdxj_replay.get("status"), "error")
            self.assertIn("'cdxj_path' is required", str(missing_cdxj_replay.get("error", "")))

            invalid_port = await start_ipwb_replay(cdxj_path="/tmp/mock.cdxj", port=0)
            self.assertEqual(invalid_port.get("status"), "error")
            self.assertIn("'port' must be greater than 0", str(invalid_port.get("error", "")))

            missing_cdxj_search = await search_ipwb_archive(cdxj_path="", url_pattern="example")
            self.assertEqual(missing_cdxj_search.get("status"), "error")
            self.assertIn("'cdxj_path' is required", str(missing_cdxj_search.get("error", "")))

            missing_pattern = await search_ipwb_archive(cdxj_path="/tmp/mock.cdxj", url_pattern="")
            self.assertEqual(missing_pattern.get("status"), "error")
            self.assertIn("'url_pattern' is required", str(missing_pattern.get("error", "")))

            invalid_limit = await search_ipwb_archive(
                cdxj_path="/tmp/mock.cdxj",
                url_pattern="example",
                limit=0,
            )
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("'limit' must be greater than 0", str(invalid_limit.get("error", "")))

            missing_hash = await get_ipwb_content(ipfs_hash=" ")
            self.assertEqual(missing_hash.get("status"), "error")
            self.assertIn("'ipfs_hash' is required", str(missing_hash.get("error", "")))

            missing_cdxj_verify = await verify_ipwb_archive(cdxj_path="")
            self.assertEqual(missing_cdxj_verify.get("status"), "error")
            self.assertIn("'cdxj_path' is required", str(missing_cdxj_verify.get("error", "")))

            invalid_sample_size = await verify_ipwb_archive(cdxj_path="/tmp/mock.cdxj", sample_size=0)
            self.assertEqual(invalid_sample_size.get("status"), "error")
            self.assertIn("'sample_size' must be greater than 0", str(invalid_sample_size.get("error", "")))

            index_result = await index_warc_to_ipwb(warc_path="/tmp/mock.warc")
            self.assertIn(index_result.get("status"), ["success", "error"])
            if index_result.get("status") == "error":
                self.assertIn("error", index_result)

            replay_result = await start_ipwb_replay(cdxj_path="/tmp/mock.cdxj", port=5000)
            self.assertIn(replay_result.get("status"), ["success", "error"])
            if replay_result.get("status") == "error":
                self.assertIn("error", replay_result)

            search_result = await search_ipwb_archive(
                cdxj_path="/tmp/mock.cdxj",
                url_pattern="example",
                limit=5,
            )
            self.assertIn(search_result.get("status"), ["success", "error"])
            if search_result.get("status") == "success":
                self.assertIn("results", search_result)

            content_result = await get_ipwb_content(ipfs_hash="QmMockHash123")
            self.assertIn(content_result.get("status"), ["success", "error"])
            if content_result.get("status") == "error":
                self.assertIn("error", content_result)

            verify_result = await verify_ipwb_archive(cdxj_path="/tmp/mock.cdxj", sample_size=5)
            self.assertIn(verify_result.get("status"), ["success", "error"])
            if verify_result.get("status") == "error":
                self.assertIn("error", verify_result)

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
