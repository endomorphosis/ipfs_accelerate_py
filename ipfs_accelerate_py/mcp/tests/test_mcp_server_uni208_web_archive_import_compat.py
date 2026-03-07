#!/usr/bin/env python3
"""UNI-208 web archive import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.web_archive_tools import (
    archive_to_archive_is,
    archive_to_wayback,
    batch_archive_to_archive_is,
    batch_scrape_with_autoscraper,
    batch_search_brave,
    batch_search_github,
    batch_search_google,
    batch_search_huggingface,
    batch_search_openverse,
    batch_search_serpstack,
    check_archive_status,
    clear_brave_cache,
    create_autoscraper_model,
    create_warc,
    extract_dataset_from_cdxj,
    extract_links_from_warc,
    extract_metadata_from_warc,
    extract_text_from_warc,
    fetch_warc_record_advanced,
    get_archive_is_content,
    get_brave_cache_stats,
    get_common_crawl_collection_info_advanced,
    get_common_crawl_content,
    get_huggingface_model_info,
    get_ipwb_content,
    get_wayback_content,
    index_warc,
    index_warc_to_ipwb,
    list_autoscraper_models,
    list_common_crawl_collections_advanced,
    list_common_crawl_indexes,
    optimize_autoscraper_model,
    scrape_with_autoscraper,
    search_archive_is,
    search_brave,
    search_brave_images,
    search_brave_news,
    search_common_crawl,
    search_common_crawl_advanced,
    search_github_code,
    search_github_issues,
    search_github_repositories,
    search_github_users,
    search_google,
    search_google_images,
    search_huggingface_datasets,
    search_huggingface_models,
    search_huggingface_spaces,
    search_ipwb_archive,
    search_openverse_audio,
    search_openverse_images,
    search_serpstack,
    search_serpstack_images,
    search_wayback_machine,
    start_ipwb_replay,
    unified_agentic_discover_and_fetch,
    unified_fetch,
    unified_health,
    unified_search,
    unified_search_and_fetch,
    verify_ipwb_archive,
)
from ipfs_accelerate_py.mcp_server.tools.web_archive_tools import native_web_archive_tools


def test_web_archive_package_exports_supported_native_functions() -> None:
    assert create_warc is native_web_archive_tools.create_warc
    assert extract_dataset_from_cdxj is native_web_archive_tools.extract_dataset_from_cdxj
    assert extract_links_from_warc is native_web_archive_tools.extract_links_from_warc
    assert extract_metadata_from_warc is native_web_archive_tools.extract_metadata_from_warc
    assert extract_text_from_warc is native_web_archive_tools.extract_text_from_warc
    assert index_warc is native_web_archive_tools.index_warc
    assert search_common_crawl is native_web_archive_tools.search_common_crawl
    assert get_common_crawl_content is native_web_archive_tools.get_common_crawl_content
    assert list_common_crawl_indexes is native_web_archive_tools.list_common_crawl_indexes
    assert search_common_crawl_advanced is native_web_archive_tools.search_common_crawl_advanced
    assert fetch_warc_record_advanced is native_web_archive_tools.fetch_warc_record_advanced
    assert list_common_crawl_collections_advanced is native_web_archive_tools.list_common_crawl_collections_advanced
    assert get_common_crawl_collection_info_advanced is native_web_archive_tools.get_common_crawl_collection_info_advanced
    assert search_wayback_machine is native_web_archive_tools.search_wayback_machine
    assert get_wayback_content is native_web_archive_tools.get_wayback_content
    assert archive_to_wayback is native_web_archive_tools.archive_to_wayback
    assert index_warc_to_ipwb is native_web_archive_tools.index_warc_to_ipwb
    assert start_ipwb_replay is native_web_archive_tools.start_ipwb_replay
    assert search_ipwb_archive is native_web_archive_tools.search_ipwb_archive
    assert get_ipwb_content is native_web_archive_tools.get_ipwb_content
    assert verify_ipwb_archive is native_web_archive_tools.verify_ipwb_archive
    assert create_autoscraper_model is native_web_archive_tools.create_autoscraper_model
    assert scrape_with_autoscraper is native_web_archive_tools.scrape_with_autoscraper
    assert optimize_autoscraper_model is native_web_archive_tools.optimize_autoscraper_model
    assert batch_scrape_with_autoscraper is native_web_archive_tools.batch_scrape_with_autoscraper
    assert list_autoscraper_models is native_web_archive_tools.list_autoscraper_models
    assert archive_to_archive_is is native_web_archive_tools.archive_to_archive_is
    assert search_archive_is is native_web_archive_tools.search_archive_is
    assert get_archive_is_content is native_web_archive_tools.get_archive_is_content
    assert check_archive_status is native_web_archive_tools.check_archive_status
    assert batch_archive_to_archive_is is native_web_archive_tools.batch_archive_to_archive_is
    assert search_brave is native_web_archive_tools.search_brave
    assert search_brave_news is native_web_archive_tools.search_brave_news
    assert search_brave_images is native_web_archive_tools.search_brave_images
    assert batch_search_brave is native_web_archive_tools.batch_search_brave
    assert get_brave_cache_stats is native_web_archive_tools.get_brave_cache_stats
    assert clear_brave_cache is native_web_archive_tools.clear_brave_cache
    assert search_google is native_web_archive_tools.search_google
    assert search_google_images is native_web_archive_tools.search_google_images
    assert batch_search_google is native_web_archive_tools.batch_search_google
    assert search_github_repositories is native_web_archive_tools.search_github_repositories
    assert search_github_code is native_web_archive_tools.search_github_code
    assert search_github_users is native_web_archive_tools.search_github_users
    assert search_github_issues is native_web_archive_tools.search_github_issues
    assert batch_search_github is native_web_archive_tools.batch_search_github
    assert search_huggingface_models is native_web_archive_tools.search_huggingface_models
    assert search_huggingface_datasets is native_web_archive_tools.search_huggingface_datasets
    assert search_huggingface_spaces is native_web_archive_tools.search_huggingface_spaces
    assert get_huggingface_model_info is native_web_archive_tools.get_huggingface_model_info
    assert batch_search_huggingface is native_web_archive_tools.batch_search_huggingface
    assert search_openverse_images is native_web_archive_tools.search_openverse_images
    assert search_openverse_audio is native_web_archive_tools.search_openverse_audio
    assert batch_search_openverse is native_web_archive_tools.batch_search_openverse
    assert search_serpstack is native_web_archive_tools.search_serpstack
    assert search_serpstack_images is native_web_archive_tools.search_serpstack_images
    assert batch_search_serpstack is native_web_archive_tools.batch_search_serpstack
    assert unified_search is native_web_archive_tools.unified_search
    assert unified_fetch is native_web_archive_tools.unified_fetch
    assert unified_search_and_fetch is native_web_archive_tools.unified_search_and_fetch
    assert unified_health is native_web_archive_tools.unified_health
    assert unified_agentic_discover_and_fetch is native_web_archive_tools.unified_agentic_discover_and_fetch
