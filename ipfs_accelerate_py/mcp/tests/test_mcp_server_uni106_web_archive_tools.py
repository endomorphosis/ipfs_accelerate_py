#!/usr/bin/env python3
"""UNI-106 web-archive tools parity tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

import ipfs_accelerate_py.mcp_server.tools.web_archive_tools.native_web_archive_tools as native_web_archive_tools

from ipfs_accelerate_py.mcp_server.tools.web_archive_tools.native_web_archive_tools import (
    archive_to_archive_is,
    archive_to_wayback,
    batch_archive_to_archive_is,
    batch_search_brave,
    batch_search_google,
    batch_scrape_with_autoscraper,
    check_archive_status,
    clear_brave_cache,
    create_warc,
    create_autoscraper_model,
    extract_dataset_from_cdxj,
    extract_links_from_warc,
    extract_metadata_from_warc,
    extract_text_from_warc,
    fetch_warc_record_advanced,
    get_huggingface_model_info,
    get_archive_is_content,
    get_brave_cache_stats,
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
    search_github_code,
    search_github_issues,
    search_github_repositories,
    search_github_users,
    batch_search_github,
    search_huggingface_datasets,
    search_huggingface_models,
    search_huggingface_spaces,
    batch_search_huggingface,
    search_openverse_audio,
    search_openverse_images,
    batch_search_openverse,
    search_serpstack,
    search_serpstack_images,
    batch_search_serpstack,
    scrape_with_autoscraper,
    search_common_crawl,
    search_archive_is,
    search_brave,
    search_brave_images,
    search_brave_news,
    search_common_crawl_advanced,
    search_google,
    search_google_images,
    search_ipwb_archive,
    search_wayback_machine,
    start_ipwb_replay,
    unified_agentic_discover_and_fetch,
    unified_fetch,
    unified_health,
    unified_search,
    unified_search_and_fetch,
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
        self.assertIn("search_brave", names)
        self.assertIn("search_brave_news", names)
        self.assertIn("search_brave_images", names)
        self.assertIn("batch_search_brave", names)
        self.assertIn("get_brave_cache_stats", names)
        self.assertIn("clear_brave_cache", names)
        self.assertIn("search_google", names)
        self.assertIn("search_google_images", names)
        self.assertIn("batch_search_google", names)
        self.assertIn("search_github_repositories", names)
        self.assertIn("search_github_code", names)
        self.assertIn("search_github_users", names)
        self.assertIn("search_github_issues", names)
        self.assertIn("batch_search_github", names)
        self.assertIn("search_huggingface_models", names)
        self.assertIn("search_huggingface_datasets", names)
        self.assertIn("search_huggingface_spaces", names)
        self.assertIn("get_huggingface_model_info", names)
        self.assertIn("batch_search_huggingface", names)
        self.assertIn("search_openverse_images", names)
        self.assertIn("search_openverse_audio", names)
        self.assertIn("batch_search_openverse", names)
        self.assertIn("search_serpstack", names)
        self.assertIn("search_serpstack_images", names)
        self.assertIn("batch_search_serpstack", names)
        self.assertIn("unified_search", names)
        self.assertIn("unified_fetch", names)
        self.assertIn("unified_search_and_fetch", names)
        self.assertIn("unified_health", names)
        self.assertIn("unified_agentic_discover_and_fetch", names)
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
        self.assertIn("search_common_crawl", names)
        self.assertIn("get_archive_is_content", names)
        self.assertIn("check_archive_status", names)
        self.assertIn("get_wayback_content", names)

    def test_register_schema_contracts_for_provider_wrappers(self) -> None:
        manager = _DummyManager()
        register_native_web_archive_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        search_common_crawl_schema = by_name["search_common_crawl"]["input_schema"]
        self.assertEqual(search_common_crawl_schema.get("required"), ["domain"])
        self.assertEqual(
            search_common_crawl_schema["properties"]["output_format"].get("enum"),
            ["json", "cdx"],
        )

        search_github_repositories_schema = by_name["search_github_repositories"]["input_schema"]
        self.assertEqual(search_github_repositories_schema.get("required"), ["query"])
        self.assertEqual(
            search_github_repositories_schema["properties"]["order"].get("enum"),
            ["asc", "desc"],
        )

        batch_search_huggingface_schema = by_name["batch_search_huggingface"]["input_schema"]
        self.assertEqual(batch_search_huggingface_schema.get("required"), ["queries"])
        self.assertEqual(
            batch_search_huggingface_schema["properties"]["search_type"].get("enum"),
            ["models", "datasets", "spaces"],
        )

        unified_search_schema = by_name["unified_search"]["input_schema"]
        self.assertEqual(unified_search_schema.get("required"), ["query"])
        self.assertEqual(
            unified_search_schema["properties"]["mode"].get("enum"),
            ["max_throughput", "balanced", "max_quality", "low_cost"],
        )

        unified_fetch_schema = by_name["unified_fetch"]["input_schema"]
        self.assertEqual(unified_fetch_schema.get("required"), ["url"])
        self.assertEqual(
            unified_fetch_schema["properties"]["mode"].get("default"),
            "balanced",
        )

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
            missing_domain = await search_common_crawl(domain="")
            self.assertEqual(missing_domain.get("status"), "error")
            self.assertIn("'domain' is required", str(missing_domain.get("error", "")))

            invalid_limit = await search_common_crawl(domain="example.com", limit=0)
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("'limit' must be greater than 0", str(invalid_limit.get("error", "")))

            invalid_output_format = await search_common_crawl(
                domain="example.com",
                output_format="xml",
            )
            self.assertEqual(invalid_output_format.get("status"), "error")
            self.assertIn("'output_format' must be one of", str(invalid_output_format.get("error", "")))

            search_result = await search_common_crawl(domain="example.com", limit=5)
            self.assertIn(search_result.get("status"), ["success", "error"])
            if search_result.get("status") == "success":
                self.assertIn("results", search_result)

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

    def test_representative_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _run() -> None:
            async def _contradictory_failure(**_: object) -> dict:
                return {"status": "success", "success": False, "error": "delegate failure"}

            with patch.dict(
                native_web_archive_tools._API,
                {
                    "create_warc": _contradictory_failure,
                    "archive_to_wayback": _contradictory_failure,
                    "search_common_crawl": _contradictory_failure,
                    "get_common_crawl_content": _contradictory_failure,
                    "search_github_repositories": _contradictory_failure,
                    "unified_search": _contradictory_failure,
                    "unified_fetch": _contradictory_failure,
                },
                clear=False,
            ):
                created = await create_warc(url="https://example.com")
                self.assertEqual(created.get("status"), "error")
                self.assertEqual(created.get("success"), False)
                self.assertEqual(created.get("error"), "delegate failure")

                archived = await archive_to_wayback(url="https://example.com")
                self.assertEqual(archived.get("status"), "error")
                self.assertEqual(archived.get("success"), False)
                self.assertEqual(archived.get("error"), "delegate failure")

                searched = await search_common_crawl(domain="example.com")
                self.assertEqual(searched.get("status"), "error")
                self.assertEqual(searched.get("success"), False)
                self.assertEqual(searched.get("error"), "delegate failure")

                content = await get_common_crawl_content(
                    url="https://example.com",
                    timestamp="20240101000000",
                )
                self.assertEqual(content.get("status"), "error")
                self.assertEqual(content.get("success"), False)
                self.assertEqual(content.get("error"), "delegate failure")

                repositories = await search_github_repositories(query="ipfs")
                self.assertEqual(repositories.get("status"), "error")
                self.assertEqual(repositories.get("success"), False)
                self.assertEqual(repositories.get("error"), "delegate failure")

                unified_search_result = await unified_search(query="ipfs")
                self.assertEqual(unified_search_result.get("status"), "error")
                self.assertEqual(unified_search_result.get("success"), False)
                self.assertEqual(unified_search_result.get("error"), "delegate failure")

                unified_fetch_result = await unified_fetch(url="https://example.com")
                self.assertEqual(unified_fetch_result.get("status"), "error")
                self.assertEqual(unified_fetch_result.get("success"), False)
                self.assertEqual(unified_fetch_result.get("error"), "delegate failure")

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

    def test_brave_google_provider_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_brave_query = await search_brave(query="")
            self.assertEqual(missing_brave_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_brave_query.get("error", "")))

            invalid_brave_count = await search_brave(query="cats", count=0)
            self.assertEqual(invalid_brave_count.get("status"), "error")
            self.assertIn("'count' must be greater than 0", str(invalid_brave_count.get("error", "")))

            invalid_brave_offset = await search_brave(query="cats", offset=-1)
            self.assertEqual(invalid_brave_offset.get("status"), "error")
            self.assertIn("'offset' must be >= 0", str(invalid_brave_offset.get("error", "")))

            invalid_brave_safesearch = await search_brave(query="cats", safesearch="bad")
            self.assertEqual(invalid_brave_safesearch.get("status"), "error")
            self.assertIn("'safesearch' must be one of", str(invalid_brave_safesearch.get("error", "")))

            invalid_brave_freshness = await search_brave(query="cats", freshness="pz")
            self.assertEqual(invalid_brave_freshness.get("status"), "error")
            self.assertIn("'freshness' must be one of", str(invalid_brave_freshness.get("error", "")))

            brave_result = await search_brave(query="cats", count=3)
            self.assertIn(brave_result.get("status"), ["success", "error"])

            brave_news_result = await search_brave_news(query="cats")
            self.assertIn(brave_news_result.get("status"), ["success", "error"])

            brave_images_result = await search_brave_images(query="cats")
            self.assertIn(brave_images_result.get("status"), ["success", "error"])

            missing_brave_batch_queries = await batch_search_brave(queries=[])
            self.assertEqual(missing_brave_batch_queries.get("status"), "error")
            self.assertIn("'queries' must be a non-empty list", str(missing_brave_batch_queries.get("error", "")))

            invalid_brave_batch_delay = await batch_search_brave(queries=["cats"], delay_seconds=-1)
            self.assertEqual(invalid_brave_batch_delay.get("status"), "error")
            self.assertIn("'delay_seconds' must be >= 0", str(invalid_brave_batch_delay.get("error", "")))

            brave_batch_result = await batch_search_brave(queries=["cats", "dogs"], count=2, delay_seconds=0)
            self.assertIn(brave_batch_result.get("status"), ["success", "error"])

            brave_cache_stats = await get_brave_cache_stats()
            self.assertIn(brave_cache_stats.get("status"), ["success", "error", "unavailable"])

            brave_cache_clear = await clear_brave_cache()
            self.assertIn(brave_cache_clear.get("status"), ["success", "error", "unavailable"])

            missing_google_query = await search_google(query="")
            self.assertEqual(missing_google_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_google_query.get("error", "")))

            invalid_google_num = await search_google(query="cats", num=0)
            self.assertEqual(invalid_google_num.get("status"), "error")
            self.assertIn("'num' must be greater than 0", str(invalid_google_num.get("error", "")))

            invalid_google_start = await search_google(query="cats", start=0)
            self.assertEqual(invalid_google_start.get("status"), "error")
            self.assertIn("'start' must be greater than 0", str(invalid_google_start.get("error", "")))

            invalid_google_safe = await search_google(query="cats", safe="bad")
            self.assertEqual(invalid_google_safe.get("status"), "error")
            self.assertIn("'safe' must be one of", str(invalid_google_safe.get("error", "")))

            invalid_google_type = await search_google(query="cats", search_type="video")
            self.assertEqual(invalid_google_type.get("status"), "error")
            self.assertIn("'search_type' must be 'image' or null", str(invalid_google_type.get("error", "")))

            google_result = await search_google(query="cats", num=3)
            self.assertIn(google_result.get("status"), ["success", "error"])

            google_images_result = await search_google_images(query="cats", num=3)
            self.assertIn(google_images_result.get("status"), ["success", "error"])

            missing_google_batch_queries = await batch_search_google(queries=[])
            self.assertEqual(missing_google_batch_queries.get("status"), "error")
            self.assertIn("'queries' must be a non-empty list", str(missing_google_batch_queries.get("error", "")))

            invalid_google_batch_delay = await batch_search_google(queries=["cats"], delay_seconds=-1)
            self.assertEqual(invalid_google_batch_delay.get("status"), "error")
            self.assertIn("'delay_seconds' must be >= 0", str(invalid_google_batch_delay.get("error", "")))

            google_batch_result = await batch_search_google(queries=["cats", "dogs"], num=2, delay_seconds=0)
            self.assertIn(google_batch_result.get("status"), ["success", "error"])

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

    def test_provider_suite_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_github_query = await search_github_repositories(query="")
            self.assertEqual(missing_github_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_github_query.get("error", "")))

            invalid_github_sort = await search_github_repositories(query="ipfs", sort="bad")
            self.assertEqual(invalid_github_sort.get("status"), "error")
            self.assertIn("'sort' must be one of", str(invalid_github_sort.get("error", "")))

            github_repo_result = await search_github_repositories(query="ipfs", per_page=2)
            self.assertIn(github_repo_result.get("status"), ["success", "error"])

            invalid_github_code_sort = await search_github_code(query="ipfs", sort="stars")
            self.assertEqual(invalid_github_code_sort.get("status"), "error")
            self.assertIn("'sort' must be 'indexed' or null", str(invalid_github_code_sort.get("error", "")))

            github_code_result = await search_github_code(query="def add", per_page=2)
            self.assertIn(github_code_result.get("status"), ["success", "error"])

            github_users_result = await search_github_users(query="barberb", per_page=2)
            self.assertIn(github_users_result.get("status"), ["success", "error"])

            github_issues_result = await search_github_issues(query="repo:octocat/Hello-World is:issue", per_page=2)
            self.assertIn(github_issues_result.get("status"), ["success", "error"])

            invalid_github_batch_type = await batch_search_github(queries=["ipfs"], search_type="bad")
            self.assertEqual(invalid_github_batch_type.get("status"), "error")
            self.assertIn("'search_type' must be one of", str(invalid_github_batch_type.get("error", "")))

            github_batch_result = await batch_search_github(queries=["ipfs", "mcp"], per_page=2, delay_seconds=0)
            self.assertIn(github_batch_result.get("status"), ["success", "error"])

            invalid_hf_sort = await search_huggingface_models(sort="bad")
            self.assertEqual(invalid_hf_sort.get("status"), "error")
            self.assertIn("'sort' must be one of", str(invalid_hf_sort.get("error", "")))

            hf_models_result = await search_huggingface_models(query="bert", limit=2)
            self.assertIn(hf_models_result.get("status"), ["success", "error"])

            hf_datasets_result = await search_huggingface_datasets(query="squad", limit=2)
            self.assertIn(hf_datasets_result.get("status"), ["success", "error"])

            hf_spaces_result = await search_huggingface_spaces(query="chat", limit=2)
            self.assertIn(hf_spaces_result.get("status"), ["success", "error"])

            missing_hf_model_id = await get_huggingface_model_info(model_id="")
            self.assertEqual(missing_hf_model_id.get("status"), "error")
            self.assertIn("'model_id' is required", str(missing_hf_model_id.get("error", "")))

            hf_model_info = await get_huggingface_model_info(model_id="bert-base-uncased")
            self.assertIn(hf_model_info.get("status"), ["success", "error"])

            invalid_hf_batch_type = await batch_search_huggingface(queries=["bert"], search_type="bad")
            self.assertEqual(invalid_hf_batch_type.get("status"), "error")
            self.assertIn("'search_type' must be one of", str(invalid_hf_batch_type.get("error", "")))

            hf_batch_result = await batch_search_huggingface(queries=["bert", "gpt"], limit=2, delay_seconds=0)
            self.assertIn(hf_batch_result.get("status"), ["success", "error"])

            missing_openverse_query = await search_openverse_images(query="")
            self.assertEqual(missing_openverse_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_openverse_query.get("error", "")))

            openverse_images_result = await search_openverse_images(query="cat", page_size=2)
            self.assertIn(openverse_images_result.get("status"), ["success", "error"])

            openverse_audio_result = await search_openverse_audio(query="music", page_size=2)
            self.assertIn(openverse_audio_result.get("status"), ["success", "error"])

            invalid_openverse_batch_type = await batch_search_openverse(queries=["cat"], search_type="video")
            self.assertEqual(invalid_openverse_batch_type.get("status"), "error")
            self.assertIn("'search_type' must be one of", str(invalid_openverse_batch_type.get("error", "")))

            openverse_batch_result = await batch_search_openverse(queries=["cat", "dog"], page_size=2, delay_seconds=0)
            self.assertIn(openverse_batch_result.get("status"), ["success", "error"])

            missing_serp_query = await search_serpstack(query="")
            self.assertEqual(missing_serp_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_serp_query.get("error", "")))

            invalid_serp_engine = await search_serpstack(query="ipfs", engine="duckduckgo")
            self.assertEqual(invalid_serp_engine.get("status"), "error")
            self.assertIn("'engine' must be one of", str(invalid_serp_engine.get("error", "")))

            serp_result = await search_serpstack(query="ipfs", num=2)
            self.assertIn(serp_result.get("status"), ["success", "error"])

            serp_images_result = await search_serpstack_images(query="ipfs", num=2)
            self.assertIn(serp_images_result.get("status"), ["success", "error"])

            invalid_serp_batch_delay = await batch_search_serpstack(queries=["ipfs"], delay_seconds=-1)
            self.assertEqual(invalid_serp_batch_delay.get("status"), "error")
            self.assertIn("'delay_seconds' must be >= 0", str(invalid_serp_batch_delay.get("error", "")))

            serp_batch_result = await batch_search_serpstack(queries=["ipfs", "mcp"], num=2, delay_seconds=0)
            self.assertIn(serp_batch_result.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_unified_api_helpers_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_unified_query = await unified_search(query="")
            self.assertEqual(missing_unified_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_unified_query.get("error", "")))

            invalid_unified_mode = await unified_search(query="ipfs", mode="bad")
            self.assertEqual(invalid_unified_mode.get("status"), "error")
            self.assertIn("'mode' must be one of", str(invalid_unified_mode.get("error", "")))

            invalid_unified_offset = await unified_search(query="ipfs", offset=-1)
            self.assertEqual(invalid_unified_offset.get("status"), "error")
            self.assertIn("'offset' must be >= 0", str(invalid_unified_offset.get("error", "")))

            unified_search_result = await unified_search(query="ipfs", max_results=3)
            self.assertIn(unified_search_result.get("status"), ["success", "error"])

            missing_unified_fetch_url = await unified_fetch(url="")
            self.assertEqual(missing_unified_fetch_url.get("status"), "error")
            self.assertIn("'url' is required", str(missing_unified_fetch_url.get("error", "")))

            invalid_unified_fetch_mode = await unified_fetch(url="https://example.com", mode="bad")
            self.assertEqual(invalid_unified_fetch_mode.get("status"), "error")
            self.assertIn("'mode' must be one of", str(invalid_unified_fetch_mode.get("error", "")))

            unified_fetch_result = await unified_fetch(url="https://example.com", mode="balanced")
            self.assertIn(unified_fetch_result.get("status"), ["success", "error"])

            missing_unified_saf_query = await unified_search_and_fetch(query="")
            self.assertEqual(missing_unified_saf_query.get("status"), "error")
            self.assertIn("'query' is required", str(missing_unified_saf_query.get("error", "")))

            invalid_unified_max_docs = await unified_search_and_fetch(query="ipfs", max_documents=0)
            self.assertEqual(invalid_unified_max_docs.get("status"), "error")
            self.assertIn("'max_documents' must be greater than 0", str(invalid_unified_max_docs.get("error", "")))

            unified_saf_result = await unified_search_and_fetch(query="ipfs", max_results=3, max_documents=2)
            self.assertIn(unified_saf_result.get("status"), ["success", "error"])

            unified_health_result = await unified_health()
            self.assertIn(unified_health_result.get("status"), ["success", "error"])

            missing_seed_urls = await unified_agentic_discover_and_fetch(seed_urls=[], target_terms=["ipfs"])
            self.assertEqual(missing_seed_urls.get("status"), "error")
            self.assertIn("'seed_urls' must be a non-empty list", str(missing_seed_urls.get("error", "")))

            missing_target_terms = await unified_agentic_discover_and_fetch(
                seed_urls=["https://example.com"],
                target_terms=[],
            )
            self.assertEqual(missing_target_terms.get("status"), "error")
            self.assertIn("'target_terms' must be a non-empty list", str(missing_target_terms.get("error", "")))

            invalid_agentic_mode = await unified_agentic_discover_and_fetch(
                seed_urls=["https://example.com"],
                target_terms=["ipfs"],
                mode="bad",
            )
            self.assertEqual(invalid_agentic_mode.get("status"), "error")
            self.assertIn("'mode' must be one of", str(invalid_agentic_mode.get("error", "")))

            unified_agentic_result = await unified_agentic_discover_and_fetch(
                seed_urls=["https://example.com"],
                target_terms=["ipfs"],
                max_hops=1,
                max_pages=2,
                mode="balanced",
            )
            self.assertIn(unified_agentic_result.get("status"), ["success", "error"])

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
