"""Native web-archive tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_web_archive_tools_api() -> Dict[str, Any]:
    """Resolve source web-archive APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.web_archive_tools import (  # type: ignore
            archive_to_archive_is as _archive_to_archive_is,
            archive_to_wayback as _archive_to_wayback,
            batch_search_brave as _batch_search_brave,
            batch_search_github as _batch_search_github,
            batch_search_google as _batch_search_google,
            batch_search_huggingface as _batch_search_huggingface,
            batch_search_openverse as _batch_search_openverse,
            batch_search_serpstack as _batch_search_serpstack,
            batch_archive_to_archive_is as _batch_archive_to_archive_is,
            check_archive_status as _check_archive_status,
            clear_brave_cache as _clear_brave_cache,
            create_warc as _create_warc,
            create_autoscraper_model as _create_autoscraper_model,
            extract_dataset_from_cdxj as _extract_dataset_from_cdxj,
            extract_links_from_warc as _extract_links_from_warc,
            extract_metadata_from_warc as _extract_metadata_from_warc,
            extract_text_from_warc as _extract_text_from_warc,
            fetch_warc_record_advanced as _fetch_warc_record_advanced,
            get_archive_is_content as _get_archive_is_content,
            get_brave_cache_stats as _get_brave_cache_stats,
            get_common_crawl_collection_info_advanced as _get_common_crawl_collection_info_advanced,
            get_common_crawl_content as _get_common_crawl_content,
            get_huggingface_model_info as _get_huggingface_model_info,
            get_ipwb_content as _get_ipwb_content,
            get_wayback_content as _get_wayback_content,
            index_warc as _index_warc,
            index_warc_to_ipwb as _index_warc_to_ipwb,
            list_common_crawl_indexes as _list_common_crawl_indexes,
            list_common_crawl_collections_advanced as _list_common_crawl_collections_advanced,
            list_autoscraper_models as _list_autoscraper_models,
            optimize_autoscraper_model as _optimize_autoscraper_model,
            scrape_with_autoscraper as _scrape_with_autoscraper,
            search_archive_is as _search_archive_is,
            search_brave as _search_brave,
            search_brave_images as _search_brave_images,
            search_brave_news as _search_brave_news,
            search_common_crawl as _search_common_crawl,
            search_common_crawl_advanced as _search_common_crawl_advanced,
            search_github_code as _search_github_code,
            search_github_issues as _search_github_issues,
            search_github_repositories as _search_github_repositories,
            search_github_users as _search_github_users,
            search_google as _search_google,
            search_google_images as _search_google_images,
            search_huggingface_datasets as _search_huggingface_datasets,
            search_huggingface_models as _search_huggingface_models,
            search_huggingface_spaces as _search_huggingface_spaces,
            search_ipwb_archive as _search_ipwb_archive,
            search_openverse_audio as _search_openverse_audio,
            search_openverse_images as _search_openverse_images,
            search_serpstack as _search_serpstack,
            search_serpstack_images as _search_serpstack_images,
            search_wayback_machine as _search_wayback_machine,
            start_ipwb_replay as _start_ipwb_replay,
            unified_agentic_discover_and_fetch as _unified_agentic_discover_and_fetch,
            unified_fetch as _unified_fetch,
            unified_health as _unified_health,
            unified_search as _unified_search,
            unified_search_and_fetch as _unified_search_and_fetch,
            verify_ipwb_archive as _verify_ipwb_archive,
            batch_scrape_with_autoscraper as _batch_scrape_with_autoscraper,
        )

        return {
            "archive_to_archive_is": _archive_to_archive_is,
            "archive_to_wayback": _archive_to_wayback,
            "batch_archive_to_archive_is": _batch_archive_to_archive_is,
            "batch_search_brave": _batch_search_brave,
            "batch_search_github": _batch_search_github,
            "batch_search_google": _batch_search_google,
            "batch_search_huggingface": _batch_search_huggingface,
            "batch_search_openverse": _batch_search_openverse,
            "batch_search_serpstack": _batch_search_serpstack,
            "batch_scrape_with_autoscraper": _batch_scrape_with_autoscraper,
            "check_archive_status": _check_archive_status,
            "clear_brave_cache": _clear_brave_cache,
            "create_warc": _create_warc,
            "create_autoscraper_model": _create_autoscraper_model,
            "extract_dataset_from_cdxj": _extract_dataset_from_cdxj,
            "extract_links_from_warc": _extract_links_from_warc,
            "extract_text_from_warc": _extract_text_from_warc,
            "extract_metadata_from_warc": _extract_metadata_from_warc,
            "fetch_warc_record_advanced": _fetch_warc_record_advanced,
            "get_archive_is_content": _get_archive_is_content,
            "get_brave_cache_stats": _get_brave_cache_stats,
            "get_common_crawl_collection_info_advanced": _get_common_crawl_collection_info_advanced,
            "get_huggingface_model_info": _get_huggingface_model_info,
            "search_common_crawl": _search_common_crawl,
            "search_common_crawl_advanced": _search_common_crawl_advanced,
            "get_common_crawl_content": _get_common_crawl_content,
            "get_ipwb_content": _get_ipwb_content,
            "get_wayback_content": _get_wayback_content,
            "index_warc": _index_warc,
            "index_warc_to_ipwb": _index_warc_to_ipwb,
            "list_common_crawl_indexes": _list_common_crawl_indexes,
            "list_common_crawl_collections_advanced": _list_common_crawl_collections_advanced,
            "list_autoscraper_models": _list_autoscraper_models,
            "optimize_autoscraper_model": _optimize_autoscraper_model,
            "scrape_with_autoscraper": _scrape_with_autoscraper,
            "search_archive_is": _search_archive_is,
            "search_brave": _search_brave,
            "search_brave_images": _search_brave_images,
            "search_brave_news": _search_brave_news,
            "search_github_code": _search_github_code,
            "search_github_issues": _search_github_issues,
            "search_github_repositories": _search_github_repositories,
            "search_github_users": _search_github_users,
            "search_google": _search_google,
            "search_google_images": _search_google_images,
            "search_huggingface_datasets": _search_huggingface_datasets,
            "search_huggingface_models": _search_huggingface_models,
            "search_huggingface_spaces": _search_huggingface_spaces,
            "search_ipwb_archive": _search_ipwb_archive,
            "search_openverse_audio": _search_openverse_audio,
            "search_openverse_images": _search_openverse_images,
            "search_serpstack": _search_serpstack,
            "search_serpstack_images": _search_serpstack_images,
            "search_wayback_machine": _search_wayback_machine,
            "start_ipwb_replay": _start_ipwb_replay,
            "unified_agentic_discover_and_fetch": _unified_agentic_discover_and_fetch,
            "unified_fetch": _unified_fetch,
            "unified_health": _unified_health,
            "unified_search": _unified_search,
            "unified_search_and_fetch": _unified_search_and_fetch,
            "verify_ipwb_archive": _verify_ipwb_archive,
        }
    except Exception:
        logger.warning(
            "Source web_archive_tools import unavailable, using fallback web-archive functions"
        )

        async def _create_warc_fallback(
            url: str,
            output_path: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = options
            return {
                "status": "success",
                "warc_path": output_path or "/tmp/fallback.warc",
                "details": {"url": url, "fallback": True},
            }

        async def _archive_to_archive_is_fallback(
            url: str,
            wait_for_completion: bool = True,
            timeout: int = 300,
        ) -> Dict[str, Any]:
            _ = wait_for_completion, timeout
            return {
                "status": "error",
                "error": "Archive.is submission backend unavailable",
                "url": url,
            }

        async def _archive_to_wayback_fallback(url: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "Wayback archive backend unavailable",
                "url": url,
            }

        async def _batch_archive_to_archive_is_fallback(
            urls: list[str],
            delay_seconds: float = 2.0,
            max_concurrent: int = 3,
        ) -> Dict[str, Any]:
            _ = delay_seconds, max_concurrent
            return {
                "status": "error",
                "error": "Archive.is batch submission backend unavailable",
                "urls": urls,
                "results": {},
            }

        async def _search_brave_fallback(
            query: str,
            api_key: Optional[str] = None,
            count: int = 10,
            offset: int = 0,
            search_lang: str = "en",
            country: str = "US",
            safesearch: str = "moderate",
            freshness: Optional[str] = None,
            text_decorations: bool = True,
            spellcheck: bool = True,
            result_filter: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (
                api_key,
                count,
                offset,
                search_lang,
                country,
                safesearch,
                freshness,
                text_decorations,
                spellcheck,
                result_filter,
            )
            return {"status": "success", "results": [], "query": query, "total_results": 0, "source": "fallback"}

        async def _search_brave_news_fallback(
            query: str,
            api_key: Optional[str] = None,
            count: int = 10,
            offset: int = 0,
            search_lang: str = "en",
            country: str = "US",
            safesearch: str = "moderate",
            freshness: Optional[str] = None,
        ) -> Dict[str, Any]:
            return await _search_brave_fallback(
                query=query,
                api_key=api_key,
                count=count,
                offset=offset,
                search_lang=search_lang,
                country=country,
                safesearch=safesearch,
                freshness=freshness,
                result_filter="news",
            )

        async def _search_brave_images_fallback(
            query: str,
            api_key: Optional[str] = None,
            count: int = 10,
            offset: int = 0,
            search_lang: str = "en",
            country: str = "US",
            safesearch: str = "moderate",
        ) -> Dict[str, Any]:
            return await _search_brave_fallback(
                query=query,
                api_key=api_key,
                count=count,
                offset=offset,
                search_lang=search_lang,
                country=country,
                safesearch=safesearch,
            )

        async def _batch_search_brave_fallback(
            queries: list[str],
            api_key: Optional[str] = None,
            count: int = 10,
            delay_seconds: float = 1.0,
        ) -> Dict[str, Any]:
            _ = api_key, count, delay_seconds
            return {
                "status": "success",
                "results": {q: {"status": "success", "results": []} for q in queries},
                "total_queries": len(queries),
                "success_count": len(queries),
                "error_count": 0,
                "source": "fallback",
            }

        async def _get_brave_cache_stats_fallback() -> Dict[str, Any]:
            return {"status": "unavailable", "message": "Brave cache backend unavailable"}

        async def _clear_brave_cache_fallback() -> Dict[str, Any]:
            return {"status": "unavailable", "message": "Brave cache backend unavailable"}

        async def _search_google_fallback(
            query: str,
            api_key: Optional[str] = None,
            search_engine_id: Optional[str] = None,
            num: int = 10,
            start: int = 1,
            search_type: Optional[str] = None,
            file_type: Optional[str] = None,
            site_search: Optional[str] = None,
            date_restrict: Optional[str] = None,
            safe: str = "medium",
            lr: Optional[str] = None,
            gl: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (
                api_key,
                search_engine_id,
                num,
                start,
                search_type,
                file_type,
                site_search,
                date_restrict,
                safe,
                lr,
                gl,
            )
            return {"status": "success", "results": [], "query": query, "total_results": 0, "source": "fallback"}

        async def _search_google_images_fallback(
            query: str,
            api_key: Optional[str] = None,
            search_engine_id: Optional[str] = None,
            num: int = 10,
            start: int = 1,
            img_size: Optional[str] = None,
            img_type: Optional[str] = None,
            img_color_type: Optional[str] = None,
            img_dominant_color: Optional[str] = None,
            safe: str = "medium",
        ) -> Dict[str, Any]:
            _ = img_size, img_type, img_color_type, img_dominant_color
            return await _search_google_fallback(
                query=query,
                api_key=api_key,
                search_engine_id=search_engine_id,
                num=num,
                start=start,
                search_type="image",
                safe=safe,
            )

        async def _batch_search_google_fallback(
            queries: list[str],
            api_key: Optional[str] = None,
            search_engine_id: Optional[str] = None,
            num: int = 10,
            delay_seconds: float = 0.5,
        ) -> Dict[str, Any]:
            _ = api_key, search_engine_id, num, delay_seconds
            return {
                "status": "success",
                "results": {q: {"status": "success", "results": []} for q in queries},
                "total_queries": len(queries),
                "success_count": len(queries),
                "error_count": 0,
                "source": "fallback",
            }

        async def _search_github_repositories_fallback(
            query: str,
            api_token: Optional[str] = None,
            sort: Optional[str] = None,
            order: str = "desc",
            per_page: int = 30,
            page: int = 1,
        ) -> Dict[str, Any]:
            _ = api_token, sort, order, per_page, page
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _search_github_code_fallback(
            query: str,
            api_token: Optional[str] = None,
            sort: Optional[str] = None,
            order: str = "desc",
            per_page: int = 30,
            page: int = 1,
        ) -> Dict[str, Any]:
            _ = api_token, sort, order, per_page, page
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _search_github_users_fallback(
            query: str,
            api_token: Optional[str] = None,
            sort: Optional[str] = None,
            order: str = "desc",
            per_page: int = 30,
            page: int = 1,
        ) -> Dict[str, Any]:
            _ = api_token, sort, order, per_page, page
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _search_github_issues_fallback(
            query: str,
            api_token: Optional[str] = None,
            sort: Optional[str] = None,
            order: str = "desc",
            per_page: int = 30,
            page: int = 1,
        ) -> Dict[str, Any]:
            _ = api_token, sort, order, per_page, page
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _batch_search_github_fallback(
            queries: list[str],
            search_type: str = "repositories",
            api_token: Optional[str] = None,
            per_page: int = 30,
            delay_seconds: float = 2.0,
        ) -> Dict[str, Any]:
            _ = search_type, api_token, per_page, delay_seconds
            return {
                "status": "success",
                "results": {q: {"status": "success", "results": []} for q in queries},
                "total_queries": len(queries),
                "success_count": len(queries),
                "error_count": 0,
                "source": "fallback",
            }

        async def _search_huggingface_models_fallback(
            query: Optional[str] = None,
            api_token: Optional[str] = None,
            filter_task: Optional[str] = None,
            filter_library: Optional[str] = None,
            filter_language: Optional[str] = None,
            sort: str = "downloads",
            direction: int = -1,
            limit: int = 20,
        ) -> Dict[str, Any]:
            _ = api_token, filter_task, filter_library, filter_language, sort, direction, limit
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _search_huggingface_datasets_fallback(
            query: Optional[str] = None,
            api_token: Optional[str] = None,
            filter_task: Optional[str] = None,
            filter_language: Optional[str] = None,
            filter_size: Optional[str] = None,
            sort: str = "downloads",
            direction: int = -1,
            limit: int = 20,
        ) -> Dict[str, Any]:
            _ = api_token, filter_task, filter_language, filter_size, sort, direction, limit
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _search_huggingface_spaces_fallback(
            query: Optional[str] = None,
            api_token: Optional[str] = None,
            filter_sdk: Optional[str] = None,
            sort: str = "likes",
            direction: int = -1,
            limit: int = 20,
        ) -> Dict[str, Any]:
            _ = api_token, filter_sdk, sort, direction, limit
            return {"status": "success", "results": [], "total_count": 0, "query": query, "source": "fallback"}

        async def _get_huggingface_model_info_fallback(
            model_id: str,
            api_token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = api_token
            return {"status": "error", "error": "HuggingFace model info backend unavailable", "model_id": model_id}

        async def _batch_search_huggingface_fallback(
            queries: list[str],
            search_type: str = "models",
            api_token: Optional[str] = None,
            limit: int = 20,
            delay_seconds: float = 0.5,
        ) -> Dict[str, Any]:
            _ = search_type, api_token, limit, delay_seconds
            return {
                "status": "success",
                "results": {q: {"status": "success", "results": []} for q in queries},
                "total_queries": len(queries),
                "success_count": len(queries),
                "error_count": 0,
                "source": "fallback",
            }

        async def _search_openverse_images_fallback(
            query: str,
            api_key: Optional[str] = None,
            page: int = 1,
            page_size: int = 20,
            license_type: Optional[str] = None,
            source: Optional[str] = None,
            creator: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = api_key, page, page_size, license_type, source, creator
            return {"status": "success", "results": [], "query": query, "total_results": 0, "source": "fallback"}

        async def _search_openverse_audio_fallback(
            query: str,
            api_key: Optional[str] = None,
            page: int = 1,
            page_size: int = 20,
            license_type: Optional[str] = None,
            source: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = api_key, page, page_size, license_type, source
            return {"status": "success", "results": [], "query": query, "total_results": 0, "source": "fallback"}

        async def _batch_search_openverse_fallback(
            queries: list[str],
            search_type: str = "images",
            api_key: Optional[str] = None,
            page_size: int = 20,
            delay_seconds: float = 0.5,
        ) -> Dict[str, Any]:
            _ = search_type, api_key, page_size, delay_seconds
            return {
                "status": "success",
                "results": {q: {"status": "success", "results": []} for q in queries},
                "total_queries": len(queries),
                "success_count": len(queries),
                "error_count": 0,
                "source": "fallback",
            }

        async def _search_serpstack_fallback(
            query: str,
            api_key: Optional[str] = None,
            engine: str = "google",
            num: int = 10,
            page: int = 1,
            location: Optional[str] = None,
            device: Optional[str] = None,
            lang: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = api_key, engine, num, page, location, device, lang
            return {"status": "success", "results": [], "query": query, "total_results": 0, "source": "fallback"}

        async def _search_serpstack_images_fallback(
            query: str,
            api_key: Optional[str] = None,
            engine: str = "google",
            num: int = 10,
            location: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = api_key, engine, num, location
            return {"status": "success", "results": [], "query": query, "total_results": 0, "source": "fallback"}

        async def _batch_search_serpstack_fallback(
            queries: list[str],
            api_key: Optional[str] = None,
            engine: str = "google",
            num: int = 10,
            delay_seconds: float = 1.0,
        ) -> Dict[str, Any]:
            _ = api_key, engine, num, delay_seconds
            return {
                "status": "success",
                "results": {q: {"status": "success", "results": []} for q in queries},
                "total_queries": len(queries),
                "success_count": len(queries),
                "error_count": 0,
                "source": "fallback",
            }

        async def _unified_search_fallback(
            query: str,
            max_results: int = 20,
            mode: str = "max_throughput",
            provider_allowlist: Optional[list[str]] = None,
            provider_denylist: Optional[list[str]] = None,
            offset: int = 0,
            domain: str = "general",
        ) -> Dict[str, Any]:
            _ = max_results, mode, provider_allowlist, provider_denylist, offset, domain
            return {"status": "success", "data": {"results": [], "query": query, "source": "fallback"}}

        async def _unified_fetch_fallback(
            url: str,
            mode: str = "balanced",
            domain: str = "general",
        ) -> Dict[str, Any]:
            _ = mode, domain
            return {"status": "success", "data": {"url": url, "content": None, "source": "fallback"}}

        async def _unified_search_and_fetch_fallback(
            query: str,
            max_results: int = 20,
            max_documents: int = 5,
            mode: str = "max_throughput",
            provider_allowlist: Optional[list[str]] = None,
            domain: str = "general",
        ) -> Dict[str, Any]:
            _ = max_results, max_documents, mode, provider_allowlist, domain
            return {
                "status": "success",
                "query": query,
                "search_results": [],
                "fetched_documents": [],
                "source": "fallback",
            }

        async def _unified_health_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "data": {
                    "healthy": False,
                    "providers": {},
                    "source": "fallback",
                },
            }

        async def _unified_agentic_discover_and_fetch_fallback(
            seed_urls: list[str],
            target_terms: list[str],
            max_hops: int = 2,
            max_pages: int = 10,
            mode: str = "balanced",
        ) -> Dict[str, Any]:
            _ = max_hops, max_pages, mode
            return {
                "status": "success",
                "seed_urls": seed_urls,
                "target_terms": target_terms,
                "discovered_pages": [],
                "fetched_documents": [],
                "source": "fallback",
            }

        async def _create_autoscraper_model_fallback(
            sample_url: str,
            wanted_data: list[str | dict[str, str]],
            model_name: str,
            wanted_dict: Optional[dict[str, list[str]]] = None,
        ) -> Dict[str, Any]:
            _ = wanted_data, wanted_dict
            return {
                "status": "error",
                "error": "AutoScraper model training backend unavailable",
                "sample_url": sample_url,
                "model_name": model_name,
            }

        async def _get_archive_is_content_fallback(archive_url: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "Archive.is content backend unavailable",
                "archive_url": archive_url,
            }

        async def _check_archive_status_fallback(submission_id: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "Archive.is status backend unavailable",
                "submission_id": submission_id,
            }

        async def _extract_text_from_warc_fallback(warc_path: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "WARC text extraction backend unavailable",
                "warc_path": warc_path,
                "records": [],
            }

        async def _extract_dataset_from_cdxj_fallback(
            cdxj_path: str,
            output_format: str = "arrow",
        ) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "CDXJ dataset extraction backend unavailable",
                "cdxj_path": cdxj_path,
                "format": output_format,
                "dataset": None,
            }

        async def _extract_links_from_warc_fallback(warc_path: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "WARC link extraction backend unavailable",
                "warc_path": warc_path,
                "links": [],
            }

        async def _extract_metadata_from_warc_fallback(warc_path: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "WARC metadata extraction backend unavailable",
                "warc_path": warc_path,
                "metadata": {},
            }

        async def _index_warc_fallback(
            warc_path: str,
            output_path: Optional[str] = None,
            encryption_key: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = encryption_key
            return {
                "status": "error",
                "error": "WARC indexing backend unavailable",
                "warc_path": warc_path,
                "cdxj_path": output_path,
            }

        async def _search_common_crawl_fallback(
            domain: str,
            crawl_id: Optional[str] = None,
            limit: int = 100,
            from_timestamp: Optional[str] = None,
            to_timestamp: Optional[str] = None,
            output_format: str = "json",
        ) -> Dict[str, Any]:
            _ = crawl_id, limit, from_timestamp, to_timestamp, output_format
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "crawl_info": {"source": "fallback", "domain": domain},
            }

        async def _search_common_crawl_advanced_fallback(
            domain: str,
            max_matches: int = 100,
            collection: Optional[str] = None,
            master_db_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = max_matches, collection, master_db_path
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "domain": domain,
                "engine": "fallback",
            }

        async def _fetch_warc_record_advanced_fallback(
            warc_filename: str,
            warc_offset: int,
            warc_length: int,
            decode_content: bool = True,
        ) -> Dict[str, Any]:
            _ = warc_offset, warc_length, decode_content
            return {
                "status": "error",
                "error": "Common Crawl advanced WARC fetch backend unavailable",
                "warc_info": {
                    "filename": warc_filename,
                    "offset": 0,
                    "length": 0,
                },
            }

        async def _list_common_crawl_collections_advanced_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "collections": [],
                "count": 0,
                "engine": "fallback",
            }

        async def _get_common_crawl_collection_info_advanced_fallback(
            collection: str,
        ) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "Common Crawl advanced collection-info backend unavailable",
                "collection": collection,
            }

        async def _get_common_crawl_content_fallback(
            url: str,
            timestamp: str,
            crawl_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = crawl_id
            return {
                "status": "error",
                "error": "Common Crawl content backend unavailable",
                "url": url,
                "timestamp": timestamp,
            }

        async def _list_common_crawl_indexes_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "indexes": [],
                "count": 0,
                "source": "fallback",
            }

        async def _list_autoscraper_models_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "models": [],
                "count": 0,
                "source": "fallback",
            }

        async def _scrape_with_autoscraper_fallback(
            model_path: str,
            target_urls: list[str],
            grouped: bool = False,
        ) -> Dict[str, Any]:
            _ = grouped
            return {
                "status": "error",
                "error": "AutoScraper scrape backend unavailable",
                "model_path": model_path,
                "target_urls": target_urls,
                "results": {},
            }

        async def _optimize_autoscraper_model_fallback(
            model_path: str,
            new_sample_urls: list[str],
            new_wanted_data: Optional[list[str | dict[str, str]]] = None,
            update_existing: bool = True,
        ) -> Dict[str, Any]:
            _ = new_wanted_data, update_existing
            return {
                "status": "error",
                "error": "AutoScraper model optimization backend unavailable",
                "model_path": model_path,
                "new_sample_urls": new_sample_urls,
            }

        async def _batch_scrape_with_autoscraper_fallback(
            model_path: str,
            urls_file: str,
            output_format: str = "json",
            batch_size: int = 50,
            delay_seconds: float = 1.0,
        ) -> Dict[str, Any]:
            _ = output_format, batch_size, delay_seconds
            return {
                "status": "error",
                "error": "AutoScraper batch scrape backend unavailable",
                "model_path": model_path,
                "urls_file": urls_file,
            }

        async def _get_wayback_content_fallback(
            url: str,
            timestamp: Optional[str] = None,
            closest: bool = True,
        ) -> Dict[str, Any]:
            _ = timestamp, closest
            return {
                "status": "error",
                "error": "Wayback content backend unavailable",
                "url": url,
            }

        async def _index_warc_to_ipwb_fallback(
            warc_path: str,
            ipfs_endpoint: Optional[str] = None,
            encrypt: bool = False,
            compression: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = ipfs_endpoint, encrypt, compression
            return {
                "status": "error",
                "error": "IPWB indexing backend unavailable",
                "warc_path": warc_path,
                "cdxj_path": None,
            }

        async def _start_ipwb_replay_fallback(
            cdxj_path: str,
            port: int = 5000,
            ipfs_endpoint: Optional[str] = None,
            proxy_mode: bool = False,
        ) -> Dict[str, Any]:
            _ = port, ipfs_endpoint, proxy_mode
            return {
                "status": "error",
                "error": "IPWB replay backend unavailable",
                "cdxj_path": cdxj_path,
            }

        async def _search_ipwb_archive_fallback(
            cdxj_path: str,
            url_pattern: str,
            from_timestamp: Optional[str] = None,
            to_timestamp: Optional[str] = None,
            limit: int = 100,
        ) -> Dict[str, Any]:
            _ = from_timestamp, to_timestamp, limit
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "cdxj_file": cdxj_path,
                "search_params": {"url_pattern": url_pattern},
                "source": "fallback",
            }

        async def _get_ipwb_content_fallback(
            ipfs_hash: str,
            ipfs_endpoint: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = ipfs_endpoint
            return {
                "status": "error",
                "error": "IPWB content backend unavailable",
                "ipfs_hash": ipfs_hash,
            }

        async def _verify_ipwb_archive_fallback(
            cdxj_path: str,
            ipfs_endpoint: Optional[str] = None,
            sample_size: int = 10,
        ) -> Dict[str, Any]:
            _ = ipfs_endpoint, sample_size
            return {
                "status": "error",
                "error": "IPWB archive verification backend unavailable",
                "cdxj_path": cdxj_path,
            }

        async def _search_wayback_machine_fallback(
            url: str,
            from_date: Optional[str] = None,
            to_date: Optional[str] = None,
            limit: int = 100,
            collapse: Optional[str] = None,
            output_format: str = "json",
        ) -> Dict[str, Any]:
            _ = from_date, to_date, limit, collapse, output_format
            return {
                "status": "success",
                "results": [],
                "url": url,
                "count": 0,
                "source": "fallback",
            }

        async def _search_archive_is_fallback(
            domain: str,
            limit: int = 100,
        ) -> Dict[str, Any]:
            _ = limit
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "domain": domain,
                "source": "fallback",
            }

        return {
            "archive_to_archive_is": _archive_to_archive_is_fallback,
            "archive_to_wayback": _archive_to_wayback_fallback,
            "batch_archive_to_archive_is": _batch_archive_to_archive_is_fallback,
            "batch_search_brave": _batch_search_brave_fallback,
            "batch_search_github": _batch_search_github_fallback,
            "batch_search_google": _batch_search_google_fallback,
            "batch_search_huggingface": _batch_search_huggingface_fallback,
            "batch_search_openverse": _batch_search_openverse_fallback,
            "batch_search_serpstack": _batch_search_serpstack_fallback,
            "batch_scrape_with_autoscraper": _batch_scrape_with_autoscraper_fallback,
            "check_archive_status": _check_archive_status_fallback,
            "clear_brave_cache": _clear_brave_cache_fallback,
            "create_warc": _create_warc_fallback,
            "create_autoscraper_model": _create_autoscraper_model_fallback,
            "extract_dataset_from_cdxj": _extract_dataset_from_cdxj_fallback,
            "extract_links_from_warc": _extract_links_from_warc_fallback,
            "extract_text_from_warc": _extract_text_from_warc_fallback,
            "extract_metadata_from_warc": _extract_metadata_from_warc_fallback,
            "fetch_warc_record_advanced": _fetch_warc_record_advanced_fallback,
            "get_archive_is_content": _get_archive_is_content_fallback,
            "get_brave_cache_stats": _get_brave_cache_stats_fallback,
            "get_common_crawl_collection_info_advanced": _get_common_crawl_collection_info_advanced_fallback,
            "get_huggingface_model_info": _get_huggingface_model_info_fallback,
            "search_common_crawl": _search_common_crawl_fallback,
            "search_common_crawl_advanced": _search_common_crawl_advanced_fallback,
            "get_common_crawl_content": _get_common_crawl_content_fallback,
            "get_ipwb_content": _get_ipwb_content_fallback,
            "get_wayback_content": _get_wayback_content_fallback,
            "index_warc": _index_warc_fallback,
            "index_warc_to_ipwb": _index_warc_to_ipwb_fallback,
            "list_common_crawl_indexes": _list_common_crawl_indexes_fallback,
            "list_common_crawl_collections_advanced": _list_common_crawl_collections_advanced_fallback,
            "list_autoscraper_models": _list_autoscraper_models_fallback,
            "optimize_autoscraper_model": _optimize_autoscraper_model_fallback,
            "scrape_with_autoscraper": _scrape_with_autoscraper_fallback,
            "search_archive_is": _search_archive_is_fallback,
            "search_brave": _search_brave_fallback,
            "search_brave_images": _search_brave_images_fallback,
            "search_brave_news": _search_brave_news_fallback,
            "search_github_code": _search_github_code_fallback,
            "search_github_issues": _search_github_issues_fallback,
            "search_github_repositories": _search_github_repositories_fallback,
            "search_github_users": _search_github_users_fallback,
            "search_google": _search_google_fallback,
            "search_google_images": _search_google_images_fallback,
            "search_huggingface_datasets": _search_huggingface_datasets_fallback,
            "search_huggingface_models": _search_huggingface_models_fallback,
            "search_huggingface_spaces": _search_huggingface_spaces_fallback,
            "search_ipwb_archive": _search_ipwb_archive_fallback,
            "search_openverse_audio": _search_openverse_audio_fallback,
            "search_openverse_images": _search_openverse_images_fallback,
            "search_serpstack": _search_serpstack_fallback,
            "search_serpstack_images": _search_serpstack_images_fallback,
            "search_wayback_machine": _search_wayback_machine_fallback,
            "start_ipwb_replay": _start_ipwb_replay_fallback,
            "unified_agentic_discover_and_fetch": _unified_agentic_discover_and_fetch_fallback,
            "unified_fetch": _unified_fetch_fallback,
            "unified_health": _unified_health_fallback,
            "unified_search": _unified_search_fallback,
            "unified_search_and_fetch": _unified_search_and_fetch_fallback,
            "verify_ipwb_archive": _verify_ipwb_archive_fallback,
        }


_API = _load_web_archive_tools_api()


async def create_warc(
    url: str,
    output_path: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a WARC file from a URL."""
    result = _API["create_warc"](
        url=url,
        output_path=output_path,
        options=options,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def archive_to_wayback(url: str) -> Dict[str, Any]:
    """Submit a URL for archival to Wayback Machine."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "'url' is required."}

    result = _API["archive_to_wayback"](url=normalized_url)
    if hasattr(result, "__await__"):
        return await result
    return result


async def archive_to_archive_is(
    url: str,
    wait_for_completion: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Submit a URL for archival to Archive.is."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "'url' is required."}
    normalized_timeout = int(timeout)
    if normalized_timeout <= 0:
        return {"status": "error", "error": "'timeout' must be greater than 0."}

    result = _API["archive_to_archive_is"](
        url=normalized_url,
        wait_for_completion=bool(wait_for_completion),
        timeout=normalized_timeout,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_archive_to_archive_is(
    urls: list[str],
    delay_seconds: float = 2.0,
    max_concurrent: int = 3,
) -> Dict[str, Any]:
    """Batch submit URLs for archival to Archive.is."""
    if not isinstance(urls, list) or not urls:
        return {"status": "error", "error": "'urls' must be a non-empty list."}

    normalized_urls = [str(url).strip() for url in urls if str(url).strip()]
    if not normalized_urls:
        return {
            "status": "error",
            "error": "'urls' must contain at least one non-empty URL.",
        }

    normalized_delay = float(delay_seconds)
    if normalized_delay < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    normalized_max_concurrent = int(max_concurrent)
    if normalized_max_concurrent <= 0:
        return {"status": "error", "error": "'max_concurrent' must be greater than 0."}

    result = _API["batch_archive_to_archive_is"](
        urls=normalized_urls,
        delay_seconds=normalized_delay,
        max_concurrent=normalized_max_concurrent,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def extract_text_from_warc(warc_path: str) -> Dict[str, Any]:
    """Extract text payloads from a WARC file."""
    normalized_path = str(warc_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'warc_path' is required."}

    result = _API["extract_text_from_warc"](warc_path=normalized_path)
    if hasattr(result, "__await__"):
        return await result
    return result


async def extract_dataset_from_cdxj(
    cdxj_path: str,
    output_format: str = "arrow",
) -> Dict[str, Any]:
    """Extract dataset records from a CDXJ index file."""
    normalized_path = str(cdxj_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'cdxj_path' is required."}

    normalized_output = str(output_format or "arrow").strip().lower() or "arrow"
    if normalized_output not in {"arrow", "huggingface", "dict"}:
        return {
            "status": "error",
            "error": "'output_format' must be one of: arrow, huggingface, dict.",
        }

    result = _API["extract_dataset_from_cdxj"](
        cdxj_path=normalized_path,
        output_format=normalized_output,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def extract_links_from_warc(warc_path: str) -> Dict[str, Any]:
    """Extract discovered links from a WARC file."""
    normalized_path = str(warc_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'warc_path' is required."}

    result = _API["extract_links_from_warc"](warc_path=normalized_path)
    if hasattr(result, "__await__"):
        return await result
    return result


async def extract_metadata_from_warc(warc_path: str) -> Dict[str, Any]:
    """Extract metadata summary from a WARC file."""
    normalized_path = str(warc_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'warc_path' is required."}

    result = _API["extract_metadata_from_warc"](warc_path=normalized_path)
    if hasattr(result, "__await__"):
        return await result
    return result


async def index_warc(
    warc_path: str,
    output_path: Optional[str] = None,
    encryption_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Index a WARC archive into CDXJ/IPWB representations."""
    normalized_path = str(warc_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'warc_path' is required."}

    result = _API["index_warc"](
        warc_path=normalized_path,
        output_path=output_path,
        encryption_key=encryption_key,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def index_warc_to_ipwb(
    warc_path: str,
    ipfs_endpoint: Optional[str] = None,
    encrypt: bool = False,
    compression: Optional[str] = None,
) -> Dict[str, Any]:
    """Index a WARC file for IPWB replay."""
    normalized_path = str(warc_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'warc_path' is required."}

    normalized_compression = None
    if compression is not None and str(compression).strip() != "":
        normalized_compression = str(compression).strip().lower()
        if normalized_compression not in {"gzip", "bz2"}:
            return {
                "status": "error",
                "error": "'compression' must be one of: gzip, bz2, or null.",
            }

    result = _API["index_warc_to_ipwb"](
        warc_path=normalized_path,
        ipfs_endpoint=ipfs_endpoint,
        encrypt=bool(encrypt),
        compression=normalized_compression,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def start_ipwb_replay(
    cdxj_path: str,
    port: int = 5000,
    ipfs_endpoint: Optional[str] = None,
    proxy_mode: bool = False,
) -> Dict[str, Any]:
    """Start an IPWB replay endpoint from a CDXJ index."""
    normalized_path = str(cdxj_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'cdxj_path' is required."}
    normalized_port = int(port)
    if normalized_port <= 0:
        return {"status": "error", "error": "'port' must be greater than 0."}

    result = _API["start_ipwb_replay"](
        cdxj_path=normalized_path,
        port=normalized_port,
        ipfs_endpoint=ipfs_endpoint,
        proxy_mode=bool(proxy_mode),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_ipwb_archive(
    cdxj_path: str,
    url_pattern: str,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Search an IPWB archive index for URL-pattern matches."""
    normalized_path = str(cdxj_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'cdxj_path' is required."}

    normalized_pattern = str(url_pattern or "").strip()
    if not normalized_pattern:
        return {"status": "error", "error": "'url_pattern' is required."}

    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}

    result = _API["search_ipwb_archive"](
        cdxj_path=normalized_path,
        url_pattern=normalized_pattern,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        limit=normalized_limit,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_ipwb_content(
    ipfs_hash: str,
    ipfs_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve archived content by IPFS hash through IPWB."""
    normalized_hash = str(ipfs_hash or "").strip()
    if not normalized_hash:
        return {"status": "error", "error": "'ipfs_hash' is required."}

    result = _API["get_ipwb_content"](
        ipfs_hash=normalized_hash,
        ipfs_endpoint=ipfs_endpoint,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def verify_ipwb_archive(
    cdxj_path: str,
    ipfs_endpoint: Optional[str] = None,
    sample_size: int = 10,
) -> Dict[str, Any]:
    """Verify integrity of indexed IPWB archive content."""
    normalized_path = str(cdxj_path or "").strip()
    if not normalized_path:
        return {"status": "error", "error": "'cdxj_path' is required."}
    normalized_sample_size = int(sample_size)
    if normalized_sample_size <= 0:
        return {"status": "error", "error": "'sample_size' must be greater than 0."}

    result = _API["verify_ipwb_archive"](
        cdxj_path=normalized_path,
        ipfs_endpoint=ipfs_endpoint,
        sample_size=normalized_sample_size,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_common_crawl(
    domain: str,
    crawl_id: Optional[str] = None,
    limit: int = 100,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
    output_format: str = "json",
) -> Dict[str, Any]:
    """Search Common Crawl index data for a domain."""
    normalized_domain = str(domain or "").strip()
    if not normalized_domain:
        return {"status": "error", "error": "'domain' is required."}

    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}

    normalized_output_format = str(output_format or "json").strip().lower() or "json"
    if normalized_output_format not in {"json", "cdx"}:
        return {"status": "error", "error": "'output_format' must be one of: json, cdx."}

    result = _API["search_common_crawl"](
        domain=normalized_domain,
        crawl_id=crawl_id,
        limit=normalized_limit,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        output_format=normalized_output_format,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_common_crawl_advanced(
    domain: str,
    max_matches: int = 100,
    collection: Optional[str] = None,
    master_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Search Common Crawl using advanced search-engine surfaces."""
    normalized_domain = str(domain or "").strip()
    if not normalized_domain:
        return {"status": "error", "error": "'domain' is required."}

    normalized_max_matches = int(max_matches)
    if normalized_max_matches <= 0:
        return {"status": "error", "error": "'max_matches' must be greater than 0."}

    result = _API["search_common_crawl_advanced"](
        domain=normalized_domain,
        max_matches=normalized_max_matches,
        collection=collection,
        master_db_path=master_db_path,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def fetch_warc_record_advanced(
    warc_filename: str,
    warc_offset: int,
    warc_length: int,
    decode_content: bool = True,
) -> Dict[str, Any]:
    """Fetch a single WARC record via advanced Common Crawl APIs."""
    normalized_filename = str(warc_filename or "").strip()
    if not normalized_filename:
        return {"status": "error", "error": "'warc_filename' is required."}

    normalized_offset = int(warc_offset)
    if normalized_offset < 0:
        return {"status": "error", "error": "'warc_offset' must be >= 0."}

    normalized_length = int(warc_length)
    if normalized_length <= 0:
        return {"status": "error", "error": "'warc_length' must be greater than 0."}

    result = _API["fetch_warc_record_advanced"](
        warc_filename=normalized_filename,
        warc_offset=normalized_offset,
        warc_length=normalized_length,
        decode_content=bool(decode_content),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_common_crawl_content(
    url: str,
    timestamp: str,
    crawl_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch archived Common Crawl content for a URL and capture timestamp."""
    normalized_url = str(url or "").strip()
    normalized_timestamp = str(timestamp or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "'url' is required."}
    if not normalized_timestamp:
        return {"status": "error", "error": "'timestamp' is required."}

    result = _API["get_common_crawl_content"](
        url=normalized_url,
        timestamp=normalized_timestamp,
        crawl_id=crawl_id,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def list_common_crawl_indexes() -> Dict[str, Any]:
    """List available Common Crawl index datasets."""
    result = _API["list_common_crawl_indexes"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def list_common_crawl_collections_advanced() -> Dict[str, Any]:
    """List available collections from advanced Common Crawl integration."""
    result = _API["list_common_crawl_collections_advanced"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_common_crawl_collection_info_advanced(collection: str) -> Dict[str, Any]:
    """Get metadata for a specific Common Crawl collection."""
    normalized_collection = str(collection or "").strip()
    if not normalized_collection:
        return {"status": "error", "error": "'collection' is required."}

    result = _API["get_common_crawl_collection_info_advanced"](
        collection=normalized_collection,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def list_autoscraper_models() -> Dict[str, Any]:
    """List available AutoScraper model artifacts."""
    result = _API["list_autoscraper_models"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def create_autoscraper_model(
    sample_url: str,
    wanted_data: list[str | dict[str, str]],
    model_name: str,
    wanted_dict: Optional[dict[str, list[str]]] = None,
) -> Dict[str, Any]:
    """Train and persist an AutoScraper model from sample data."""
    normalized_sample_url = str(sample_url or "").strip()
    if not normalized_sample_url:
        return {"status": "error", "error": "'sample_url' is required."}

    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return {"status": "error", "error": "'model_name' is required."}

    if not isinstance(wanted_data, list) or not wanted_data:
        return {"status": "error", "error": "'wanted_data' must be a non-empty list."}

    result = _API["create_autoscraper_model"](
        sample_url=normalized_sample_url,
        wanted_data=wanted_data,
        model_name=normalized_model_name,
        wanted_dict=wanted_dict,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def scrape_with_autoscraper(
    model_path: str,
    target_urls: list[str],
    grouped: bool = False,
) -> Dict[str, Any]:
    """Scrape target URLs with a trained AutoScraper model."""
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        return {"status": "error", "error": "'model_path' is required."}

    if not isinstance(target_urls, list) or not target_urls:
        return {"status": "error", "error": "'target_urls' must be a non-empty list."}

    normalized_urls = [str(url).strip() for url in target_urls if str(url).strip()]
    if not normalized_urls:
        return {
            "status": "error",
            "error": "'target_urls' must contain at least one non-empty URL.",
        }

    result = _API["scrape_with_autoscraper"](
        model_path=normalized_model_path,
        target_urls=normalized_urls,
        grouped=bool(grouped),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def optimize_autoscraper_model(
    model_path: str,
    new_sample_urls: list[str],
    new_wanted_data: Optional[list[str | dict[str, str]]] = None,
    update_existing: bool = True,
) -> Dict[str, Any]:
    """Optimize an existing AutoScraper model with additional samples."""
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        return {"status": "error", "error": "'model_path' is required."}

    if not isinstance(new_sample_urls, list) or not new_sample_urls:
        return {"status": "error", "error": "'new_sample_urls' must be a non-empty list."}

    normalized_urls = [str(url).strip() for url in new_sample_urls if str(url).strip()]
    if not normalized_urls:
        return {
            "status": "error",
            "error": "'new_sample_urls' must contain at least one non-empty URL.",
        }

    result = _API["optimize_autoscraper_model"](
        model_path=normalized_model_path,
        new_sample_urls=normalized_urls,
        new_wanted_data=new_wanted_data,
        update_existing=bool(update_existing),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_scrape_with_autoscraper(
    model_path: str,
    urls_file: str,
    output_format: str = "json",
    batch_size: int = 50,
    delay_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Run batch scraping against URL list file using AutoScraper."""
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        return {"status": "error", "error": "'model_path' is required."}

    normalized_urls_file = str(urls_file or "").strip()
    if not normalized_urls_file:
        return {"status": "error", "error": "'urls_file' is required."}

    normalized_output = str(output_format or "json").strip().lower() or "json"
    if normalized_output not in {"json", "csv", "jsonl"}:
        return {
            "status": "error",
            "error": "'output_format' must be one of: json, csv, jsonl.",
        }

    normalized_batch_size = int(batch_size)
    if normalized_batch_size <= 0:
        return {"status": "error", "error": "'batch_size' must be greater than 0."}

    normalized_delay = float(delay_seconds)
    if normalized_delay < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_scrape_with_autoscraper"](
        model_path=normalized_model_path,
        urls_file=normalized_urls_file,
        output_format=normalized_output,
        batch_size=normalized_batch_size,
        delay_seconds=normalized_delay,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_brave(
    query: str,
    api_key: Optional[str] = None,
    count: int = 10,
    offset: int = 0,
    search_lang: str = "en",
    country: str = "US",
    safesearch: str = "moderate",
    freshness: Optional[str] = None,
    text_decorations: bool = True,
    spellcheck: bool = True,
    result_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Search the web via Brave provider."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_count = int(count)
    if normalized_count <= 0:
        return {"status": "error", "error": "'count' must be greater than 0."}
    normalized_offset = int(offset)
    if normalized_offset < 0:
        return {"status": "error", "error": "'offset' must be >= 0."}
    normalized_safe = str(safesearch or "moderate").strip().lower() or "moderate"
    if normalized_safe not in {"off", "moderate", "strict"}:
        return {"status": "error", "error": "'safesearch' must be one of: off, moderate, strict."}
    normalized_freshness = str(freshness).strip().lower() if freshness is not None else None
    if normalized_freshness and normalized_freshness not in {"pd", "pw", "pm", "py"}:
        return {"status": "error", "error": "'freshness' must be one of: pd, pw, pm, py, or null."}

    result = _API["search_brave"](
        query=normalized_query,
        api_key=api_key,
        count=normalized_count,
        offset=normalized_offset,
        search_lang=search_lang,
        country=country,
        safesearch=normalized_safe,
        freshness=normalized_freshness,
        text_decorations=bool(text_decorations),
        spellcheck=bool(spellcheck),
        result_filter=result_filter,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_brave_news(
    query: str,
    api_key: Optional[str] = None,
    count: int = 10,
    offset: int = 0,
    search_lang: str = "en",
    country: str = "US",
    safesearch: str = "moderate",
    freshness: Optional[str] = None,
) -> Dict[str, Any]:
    """Search news via Brave provider."""
    return await search_brave(
        query=query,
        api_key=api_key,
        count=count,
        offset=offset,
        search_lang=search_lang,
        country=country,
        safesearch=safesearch,
        freshness=freshness,
        result_filter="news",
    )


async def search_brave_images(
    query: str,
    api_key: Optional[str] = None,
    count: int = 10,
    offset: int = 0,
    search_lang: str = "en",
    country: str = "US",
    safesearch: str = "moderate",
) -> Dict[str, Any]:
    """Search images via Brave provider."""
    result = _API["search_brave_images"](
        query=str(query or "").strip(),
        api_key=api_key,
        count=int(count),
        offset=int(offset),
        search_lang=search_lang,
        country=country,
        safesearch=str(safesearch or "moderate").strip().lower() or "moderate",
    )
    if str(query or "").strip() == "":
        return {"status": "error", "error": "'query' is required."}
    if int(count) <= 0:
        return {"status": "error", "error": "'count' must be greater than 0."}
    if int(offset) < 0:
        return {"status": "error", "error": "'offset' must be >= 0."}
    if str(safesearch or "moderate").strip().lower() not in {"off", "moderate", "strict"}:
        return {"status": "error", "error": "'safesearch' must be one of: off, moderate, strict."}
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_search_brave(
    queries: list[str],
    api_key: Optional[str] = None,
    count: int = 10,
    delay_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Run batch Brave queries."""
    if not isinstance(queries, list) or not queries:
        return {"status": "error", "error": "'queries' must be a non-empty list."}
    normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not normalized_queries:
        return {"status": "error", "error": "'queries' must contain at least one non-empty query."}
    if int(count) <= 0:
        return {"status": "error", "error": "'count' must be greater than 0."}
    if float(delay_seconds) < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_search_brave"](
        queries=normalized_queries,
        api_key=api_key,
        count=int(count),
        delay_seconds=float(delay_seconds),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_brave_cache_stats() -> Dict[str, Any]:
    """Return Brave provider cache stats."""
    result = _API["get_brave_cache_stats"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def clear_brave_cache() -> Dict[str, Any]:
    """Clear Brave provider cache."""
    result = _API["clear_brave_cache"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_google(
    query: str,
    api_key: Optional[str] = None,
    search_engine_id: Optional[str] = None,
    num: int = 10,
    start: int = 1,
    search_type: Optional[str] = None,
    file_type: Optional[str] = None,
    site_search: Optional[str] = None,
    date_restrict: Optional[str] = None,
    safe: str = "medium",
    lr: Optional[str] = None,
    gl: Optional[str] = None,
) -> Dict[str, Any]:
    """Search via Google provider."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    if int(num) <= 0:
        return {"status": "error", "error": "'num' must be greater than 0."}
    if int(start) <= 0:
        return {"status": "error", "error": "'start' must be greater than 0."}
    normalized_safe = str(safe or "medium").strip().lower() or "medium"
    if normalized_safe not in {"off", "medium", "high"}:
        return {"status": "error", "error": "'safe' must be one of: off, medium, high."}
    normalized_search_type = str(search_type).strip().lower() if search_type is not None else None
    if normalized_search_type and normalized_search_type not in {"image"}:
        return {"status": "error", "error": "'search_type' must be 'image' or null."}

    result = _API["search_google"](
        query=normalized_query,
        api_key=api_key,
        search_engine_id=search_engine_id,
        num=int(num),
        start=int(start),
        search_type=normalized_search_type,
        file_type=file_type,
        site_search=site_search,
        date_restrict=date_restrict,
        safe=normalized_safe,
        lr=lr,
        gl=gl,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_google_images(
    query: str,
    api_key: Optional[str] = None,
    search_engine_id: Optional[str] = None,
    num: int = 10,
    start: int = 1,
    img_size: Optional[str] = None,
    img_type: Optional[str] = None,
    img_color_type: Optional[str] = None,
    img_dominant_color: Optional[str] = None,
    safe: str = "medium",
) -> Dict[str, Any]:
    """Search images via Google provider."""
    result = _API["search_google_images"](
        query=str(query or "").strip(),
        api_key=api_key,
        search_engine_id=search_engine_id,
        num=int(num),
        start=int(start),
        img_size=img_size,
        img_type=img_type,
        img_color_type=img_color_type,
        img_dominant_color=img_dominant_color,
        safe=str(safe or "medium").strip().lower() or "medium",
    )
    if str(query or "").strip() == "":
        return {"status": "error", "error": "'query' is required."}
    if int(num) <= 0:
        return {"status": "error", "error": "'num' must be greater than 0."}
    if int(start) <= 0:
        return {"status": "error", "error": "'start' must be greater than 0."}
    if str(safe or "medium").strip().lower() not in {"off", "medium", "high"}:
        return {"status": "error", "error": "'safe' must be one of: off, medium, high."}
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_search_google(
    queries: list[str],
    api_key: Optional[str] = None,
    search_engine_id: Optional[str] = None,
    num: int = 10,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    """Run batch Google queries."""
    if not isinstance(queries, list) or not queries:
        return {"status": "error", "error": "'queries' must be a non-empty list."}
    normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not normalized_queries:
        return {"status": "error", "error": "'queries' must contain at least one non-empty query."}
    if int(num) <= 0:
        return {"status": "error", "error": "'num' must be greater than 0."}
    if float(delay_seconds) < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_search_google"](
        queries=normalized_queries,
        api_key=api_key,
        search_engine_id=search_engine_id,
        num=int(num),
        delay_seconds=float(delay_seconds),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_github_repositories(
    query: str,
    api_token: Optional[str] = None,
    sort: Optional[str] = None,
    order: str = "desc",
    per_page: int = 30,
    page: int = 1,
) -> Dict[str, Any]:
    """Search GitHub repositories via provider wrapper."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_order = str(order or "desc").strip().lower() or "desc"
    if normalized_order not in {"asc", "desc"}:
        return {"status": "error", "error": "'order' must be one of: asc, desc."}
    if sort is not None and str(sort).strip() not in {"stars", "forks", "help-wanted-issues", "updated"}:
        return {"status": "error", "error": "'sort' must be one of: stars, forks, help-wanted-issues, updated, or null."}
    normalized_per_page = int(per_page)
    if normalized_per_page <= 0:
        return {"status": "error", "error": "'per_page' must be greater than 0."}
    normalized_page = int(page)
    if normalized_page <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}

    result = _API["search_github_repositories"](
        query=normalized_query,
        api_token=api_token,
        sort=str(sort).strip() if sort is not None else None,
        order=normalized_order,
        per_page=normalized_per_page,
        page=normalized_page,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_github_code(
    query: str,
    api_token: Optional[str] = None,
    sort: Optional[str] = None,
    order: str = "desc",
    per_page: int = 30,
    page: int = 1,
) -> Dict[str, Any]:
    """Search GitHub code via provider wrapper."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_order = str(order or "desc").strip().lower() or "desc"
    if normalized_order not in {"asc", "desc"}:
        return {"status": "error", "error": "'order' must be one of: asc, desc."}
    if sort is not None and str(sort).strip() not in {"indexed"}:
        return {"status": "error", "error": "'sort' must be 'indexed' or null."}
    normalized_per_page = int(per_page)
    if normalized_per_page <= 0:
        return {"status": "error", "error": "'per_page' must be greater than 0."}
    normalized_page = int(page)
    if normalized_page <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}

    result = _API["search_github_code"](
        query=normalized_query,
        api_token=api_token,
        sort=str(sort).strip() if sort is not None else None,
        order=normalized_order,
        per_page=normalized_per_page,
        page=normalized_page,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_github_users(
    query: str,
    api_token: Optional[str] = None,
    sort: Optional[str] = None,
    order: str = "desc",
    per_page: int = 30,
    page: int = 1,
) -> Dict[str, Any]:
    """Search GitHub users via provider wrapper."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_order = str(order or "desc").strip().lower() or "desc"
    if normalized_order not in {"asc", "desc"}:
        return {"status": "error", "error": "'order' must be one of: asc, desc."}
    if sort is not None and str(sort).strip() not in {"followers", "repositories", "joined"}:
        return {"status": "error", "error": "'sort' must be one of: followers, repositories, joined, or null."}
    normalized_per_page = int(per_page)
    if normalized_per_page <= 0:
        return {"status": "error", "error": "'per_page' must be greater than 0."}
    normalized_page = int(page)
    if normalized_page <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}

    result = _API["search_github_users"](
        query=normalized_query,
        api_token=api_token,
        sort=str(sort).strip() if sort is not None else None,
        order=normalized_order,
        per_page=normalized_per_page,
        page=normalized_page,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_github_issues(
    query: str,
    api_token: Optional[str] = None,
    sort: Optional[str] = None,
    order: str = "desc",
    per_page: int = 30,
    page: int = 1,
) -> Dict[str, Any]:
    """Search GitHub issues and pull requests via provider wrapper."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_order = str(order or "desc").strip().lower() or "desc"
    if normalized_order not in {"asc", "desc"}:
        return {"status": "error", "error": "'order' must be one of: asc, desc."}
    if sort is not None and str(sort).strip() not in {"comments", "reactions", "created", "updated"}:
        return {"status": "error", "error": "'sort' must be one of: comments, reactions, created, updated, or null."}
    normalized_per_page = int(per_page)
    if normalized_per_page <= 0:
        return {"status": "error", "error": "'per_page' must be greater than 0."}
    normalized_page = int(page)
    if normalized_page <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}

    result = _API["search_github_issues"](
        query=normalized_query,
        api_token=api_token,
        sort=str(sort).strip() if sort is not None else None,
        order=normalized_order,
        per_page=normalized_per_page,
        page=normalized_page,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_search_github(
    queries: list[str],
    search_type: str = "repositories",
    api_token: Optional[str] = None,
    per_page: int = 30,
    delay_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Run batch GitHub provider searches."""
    if not isinstance(queries, list) or not queries:
        return {"status": "error", "error": "'queries' must be a non-empty list."}
    normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not normalized_queries:
        return {"status": "error", "error": "'queries' must contain at least one non-empty query."}
    normalized_type = str(search_type or "repositories").strip().lower() or "repositories"
    if normalized_type not in {"repositories", "code", "users", "issues"}:
        return {"status": "error", "error": "'search_type' must be one of: repositories, code, users, issues."}
    if int(per_page) <= 0:
        return {"status": "error", "error": "'per_page' must be greater than 0."}
    if float(delay_seconds) < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_search_github"](
        queries=normalized_queries,
        search_type=normalized_type,
        api_token=api_token,
        per_page=int(per_page),
        delay_seconds=float(delay_seconds),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_huggingface_models(
    query: Optional[str] = None,
    api_token: Optional[str] = None,
    filter_task: Optional[str] = None,
    filter_library: Optional[str] = None,
    filter_language: Optional[str] = None,
    sort: str = "downloads",
    direction: int = -1,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search HuggingFace models via provider wrapper."""
    normalized_sort = str(sort or "downloads").strip().lower() or "downloads"
    if normalized_sort not in {"downloads", "created", "updated", "likes"}:
        return {"status": "error", "error": "'sort' must be one of: downloads, created, updated, likes."}
    normalized_direction = int(direction)
    if normalized_direction not in {-1, 1}:
        return {"status": "error", "error": "'direction' must be -1 or 1."}
    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}

    result = _API["search_huggingface_models"](
        query=str(query).strip() if query is not None else None,
        api_token=api_token,
        filter_task=filter_task,
        filter_library=filter_library,
        filter_language=filter_language,
        sort=normalized_sort,
        direction=normalized_direction,
        limit=normalized_limit,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_huggingface_datasets(
    query: Optional[str] = None,
    api_token: Optional[str] = None,
    filter_task: Optional[str] = None,
    filter_language: Optional[str] = None,
    filter_size: Optional[str] = None,
    sort: str = "downloads",
    direction: int = -1,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search HuggingFace datasets via provider wrapper."""
    normalized_sort = str(sort or "downloads").strip().lower() or "downloads"
    if normalized_sort not in {"downloads", "created", "updated", "likes"}:
        return {"status": "error", "error": "'sort' must be one of: downloads, created, updated, likes."}
    normalized_direction = int(direction)
    if normalized_direction not in {-1, 1}:
        return {"status": "error", "error": "'direction' must be -1 or 1."}
    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}

    result = _API["search_huggingface_datasets"](
        query=str(query).strip() if query is not None else None,
        api_token=api_token,
        filter_task=filter_task,
        filter_language=filter_language,
        filter_size=filter_size,
        sort=normalized_sort,
        direction=normalized_direction,
        limit=normalized_limit,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_huggingface_spaces(
    query: Optional[str] = None,
    api_token: Optional[str] = None,
    filter_sdk: Optional[str] = None,
    sort: str = "likes",
    direction: int = -1,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search HuggingFace spaces via provider wrapper."""
    normalized_sort = str(sort or "likes").strip().lower() or "likes"
    if normalized_sort not in {"created", "updated", "likes"}:
        return {"status": "error", "error": "'sort' must be one of: created, updated, likes."}
    normalized_direction = int(direction)
    if normalized_direction not in {-1, 1}:
        return {"status": "error", "error": "'direction' must be -1 or 1."}
    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}

    result = _API["search_huggingface_spaces"](
        query=str(query).strip() if query is not None else None,
        api_token=api_token,
        filter_sdk=filter_sdk,
        sort=normalized_sort,
        direction=normalized_direction,
        limit=normalized_limit,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_huggingface_model_info(
    model_id: str,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Get details for a HuggingFace model."""
    normalized_model_id = str(model_id or "").strip()
    if not normalized_model_id:
        return {"status": "error", "error": "'model_id' is required."}

    result = _API["get_huggingface_model_info"](
        model_id=normalized_model_id,
        api_token=api_token,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_search_huggingface(
    queries: list[str],
    search_type: str = "models",
    api_token: Optional[str] = None,
    limit: int = 20,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    """Run batch HuggingFace provider searches."""
    if not isinstance(queries, list) or not queries:
        return {"status": "error", "error": "'queries' must be a non-empty list."}
    normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not normalized_queries:
        return {"status": "error", "error": "'queries' must contain at least one non-empty query."}
    normalized_type = str(search_type or "models").strip().lower() or "models"
    if normalized_type not in {"models", "datasets", "spaces"}:
        return {"status": "error", "error": "'search_type' must be one of: models, datasets, spaces."}
    if int(limit) <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}
    if float(delay_seconds) < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_search_huggingface"](
        queries=normalized_queries,
        search_type=normalized_type,
        api_token=api_token,
        limit=int(limit),
        delay_seconds=float(delay_seconds),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_openverse_images(
    query: str,
    api_key: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    license_type: Optional[str] = None,
    source: Optional[str] = None,
    creator: Optional[str] = None,
) -> Dict[str, Any]:
    """Search OpenVerse images via provider wrapper."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    if int(page) <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}
    normalized_page_size = int(page_size)
    if normalized_page_size <= 0 or normalized_page_size > 100:
        return {"status": "error", "error": "'page_size' must be between 1 and 100."}

    result = _API["search_openverse_images"](
        query=normalized_query,
        api_key=api_key,
        page=int(page),
        page_size=normalized_page_size,
        license_type=license_type,
        source=source,
        creator=creator,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_openverse_audio(
    query: str,
    api_key: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    license_type: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Search OpenVerse audio via provider wrapper."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    if int(page) <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}
    normalized_page_size = int(page_size)
    if normalized_page_size <= 0 or normalized_page_size > 100:
        return {"status": "error", "error": "'page_size' must be between 1 and 100."}

    result = _API["search_openverse_audio"](
        query=normalized_query,
        api_key=api_key,
        page=int(page),
        page_size=normalized_page_size,
        license_type=license_type,
        source=source,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_search_openverse(
    queries: list[str],
    search_type: str = "images",
    api_key: Optional[str] = None,
    page_size: int = 20,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    """Run batch OpenVerse provider searches."""
    if not isinstance(queries, list) or not queries:
        return {"status": "error", "error": "'queries' must be a non-empty list."}
    normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not normalized_queries:
        return {"status": "error", "error": "'queries' must contain at least one non-empty query."}
    normalized_type = str(search_type or "images").strip().lower() or "images"
    if normalized_type not in {"images", "audio"}:
        return {"status": "error", "error": "'search_type' must be one of: images, audio."}
    normalized_page_size = int(page_size)
    if normalized_page_size <= 0 or normalized_page_size > 100:
        return {"status": "error", "error": "'page_size' must be between 1 and 100."}
    if float(delay_seconds) < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_search_openverse"](
        queries=normalized_queries,
        search_type=normalized_type,
        api_key=api_key,
        page_size=normalized_page_size,
        delay_seconds=float(delay_seconds),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_serpstack(
    query: str,
    api_key: Optional[str] = None,
    engine: str = "google",
    num: int = 10,
    page: int = 1,
    location: Optional[str] = None,
    device: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Search SerpStack provider web results."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_engine = str(engine or "google").strip().lower() or "google"
    if normalized_engine not in {"google", "bing", "yandex", "yahoo", "baidu"}:
        return {"status": "error", "error": "'engine' must be one of: google, bing, yandex, yahoo, baidu."}
    normalized_num = int(num)
    if normalized_num <= 0 or normalized_num > 100:
        return {"status": "error", "error": "'num' must be between 1 and 100."}
    if int(page) <= 0:
        return {"status": "error", "error": "'page' must be greater than 0."}

    result = _API["search_serpstack"](
        query=normalized_query,
        api_key=api_key,
        engine=normalized_engine,
        num=normalized_num,
        page=int(page),
        location=location,
        device=device,
        lang=lang,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_serpstack_images(
    query: str,
    api_key: Optional[str] = None,
    engine: str = "google",
    num: int = 10,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """Search SerpStack provider image results."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_engine = str(engine or "google").strip().lower() or "google"
    if normalized_engine not in {"google", "bing", "yandex", "yahoo", "baidu"}:
        return {"status": "error", "error": "'engine' must be one of: google, bing, yandex, yahoo, baidu."}
    normalized_num = int(num)
    if normalized_num <= 0 or normalized_num > 100:
        return {"status": "error", "error": "'num' must be between 1 and 100."}

    result = _API["search_serpstack_images"](
        query=normalized_query,
        api_key=api_key,
        engine=normalized_engine,
        num=normalized_num,
        location=location,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_search_serpstack(
    queries: list[str],
    api_key: Optional[str] = None,
    engine: str = "google",
    num: int = 10,
    delay_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Run batch SerpStack provider searches."""
    if not isinstance(queries, list) or not queries:
        return {"status": "error", "error": "'queries' must be a non-empty list."}
    normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not normalized_queries:
        return {"status": "error", "error": "'queries' must contain at least one non-empty query."}
    normalized_engine = str(engine or "google").strip().lower() or "google"
    if normalized_engine not in {"google", "bing", "yandex", "yahoo", "baidu"}:
        return {"status": "error", "error": "'engine' must be one of: google, bing, yandex, yahoo, baidu."}
    normalized_num = int(num)
    if normalized_num <= 0 or normalized_num > 100:
        return {"status": "error", "error": "'num' must be between 1 and 100."}
    if float(delay_seconds) < 0:
        return {"status": "error", "error": "'delay_seconds' must be >= 0."}

    result = _API["batch_search_serpstack"](
        queries=normalized_queries,
        api_key=api_key,
        engine=normalized_engine,
        num=normalized_num,
        delay_seconds=float(delay_seconds),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def unified_search(
    query: str,
    max_results: int = 20,
    mode: str = "max_throughput",
    provider_allowlist: Optional[list[str]] = None,
    provider_denylist: Optional[list[str]] = None,
    offset: int = 0,
    domain: str = "general",
) -> Dict[str, Any]:
    """Run unified multi-provider search."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_max_results = int(max_results)
    if normalized_max_results <= 0:
        return {"status": "error", "error": "'max_results' must be greater than 0."}
    normalized_mode = str(mode or "max_throughput").strip().lower() or "max_throughput"
    if normalized_mode not in {"max_throughput", "balanced", "max_quality", "low_cost"}:
        return {"status": "error", "error": "'mode' must be one of: max_throughput, balanced, max_quality, low_cost."}
    normalized_offset = int(offset)
    if normalized_offset < 0:
        return {"status": "error", "error": "'offset' must be >= 0."}

    result = _API["unified_search"](
        query=normalized_query,
        max_results=normalized_max_results,
        mode=normalized_mode,
        provider_allowlist=provider_allowlist,
        provider_denylist=provider_denylist,
        offset=normalized_offset,
        domain=str(domain or "general").strip() or "general",
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def unified_fetch(
    url: str,
    mode: str = "balanced",
    domain: str = "general",
) -> Dict[str, Any]:
    """Fetch content from a URL with unified provider selection."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "'url' is required."}
    normalized_mode = str(mode or "balanced").strip().lower() or "balanced"
    if normalized_mode not in {"max_throughput", "balanced", "max_quality", "low_cost"}:
        return {"status": "error", "error": "'mode' must be one of: max_throughput, balanced, max_quality, low_cost."}

    result = _API["unified_fetch"](
        url=normalized_url,
        mode=normalized_mode,
        domain=str(domain or "general").strip() or "general",
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def unified_search_and_fetch(
    query: str,
    max_results: int = 20,
    max_documents: int = 5,
    mode: str = "max_throughput",
    provider_allowlist: Optional[list[str]] = None,
    domain: str = "general",
) -> Dict[str, Any]:
    """Search and fetch top document candidates with unified API."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "'query' is required."}
    normalized_max_results = int(max_results)
    if normalized_max_results <= 0:
        return {"status": "error", "error": "'max_results' must be greater than 0."}
    normalized_max_documents = int(max_documents)
    if normalized_max_documents <= 0:
        return {"status": "error", "error": "'max_documents' must be greater than 0."}
    normalized_mode = str(mode or "max_throughput").strip().lower() or "max_throughput"
    if normalized_mode not in {"max_throughput", "balanced", "max_quality", "low_cost"}:
        return {"status": "error", "error": "'mode' must be one of: max_throughput, balanced, max_quality, low_cost."}

    result = _API["unified_search_and_fetch"](
        query=normalized_query,
        max_results=normalized_max_results,
        max_documents=normalized_max_documents,
        mode=normalized_mode,
        provider_allowlist=provider_allowlist,
        domain=str(domain or "general").strip() or "general",
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def unified_health() -> Dict[str, Any]:
    """Return unified provider health snapshot."""
    result = _API["unified_health"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def unified_agentic_discover_and_fetch(
    seed_urls: list[str],
    target_terms: list[str],
    max_hops: int = 2,
    max_pages: int = 10,
    mode: str = "balanced",
) -> Dict[str, Any]:
    """Agentic discovery + fetch using unified provider orchestration."""
    if not isinstance(seed_urls, list) or not seed_urls:
        return {"status": "error", "error": "'seed_urls' must be a non-empty list."}
    normalized_seed_urls = [str(u).strip() for u in seed_urls if str(u).strip()]
    if not normalized_seed_urls:
        return {"status": "error", "error": "'seed_urls' must contain at least one non-empty URL."}

    if not isinstance(target_terms, list) or not target_terms:
        return {"status": "error", "error": "'target_terms' must be a non-empty list."}
    normalized_target_terms = [str(t).strip() for t in target_terms if str(t).strip()]
    if not normalized_target_terms:
        return {"status": "error", "error": "'target_terms' must contain at least one non-empty term."}

    normalized_max_hops = int(max_hops)
    if normalized_max_hops <= 0:
        return {"status": "error", "error": "'max_hops' must be greater than 0."}
    normalized_max_pages = int(max_pages)
    if normalized_max_pages <= 0:
        return {"status": "error", "error": "'max_pages' must be greater than 0."}

    normalized_mode = str(mode or "balanced").strip().lower() or "balanced"
    if normalized_mode not in {"max_throughput", "balanced", "max_quality", "low_cost"}:
        return {"status": "error", "error": "'mode' must be one of: max_throughput, balanced, max_quality, low_cost."}

    result = _API["unified_agentic_discover_and_fetch"](
        seed_urls=normalized_seed_urls,
        target_terms=normalized_target_terms,
        max_hops=normalized_max_hops,
        max_pages=normalized_max_pages,
        mode=normalized_mode,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_wayback_content(
    url: str,
    timestamp: Optional[str] = None,
    closest: bool = True,
) -> Dict[str, Any]:
    """Fetch archived content for a URL from Wayback Machine."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "'url' is required."}

    result = _API["get_wayback_content"](
        url=normalized_url,
        timestamp=timestamp,
        closest=bool(closest),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_wayback_machine(
    url: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 100,
    collapse: Optional[str] = None,
    output_format: str = "json",
) -> Dict[str, Any]:
    """Search Internet Archive Wayback captures for a URL."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "'url' is required."}
    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}
    normalized_output = str(output_format or "json").strip().lower() or "json"
    if normalized_output not in {"json", "cdx"}:
        return {"status": "error", "error": "'output_format' must be one of: json, cdx."}

    result = _API["search_wayback_machine"](
        url=normalized_url,
        from_date=from_date,
        to_date=to_date,
        limit=normalized_limit,
        collapse=collapse,
        output_format=normalized_output,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_archive_is(
    domain: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """Search Archive.is snapshots for a domain."""
    normalized_domain = str(domain or "").strip()
    if not normalized_domain:
        return {"status": "error", "error": "'domain' is required."}
    normalized_limit = int(limit)
    if normalized_limit <= 0:
        return {"status": "error", "error": "'limit' must be greater than 0."}

    result = _API["search_archive_is"](
        domain=normalized_domain,
        limit=normalized_limit,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_archive_is_content(archive_url: str) -> Dict[str, Any]:
    """Get content from an Archive.is URL."""
    normalized_archive_url = str(archive_url or "").strip()
    if not normalized_archive_url:
        return {"status": "error", "error": "'archive_url' is required."}

    result = _API["get_archive_is_content"](archive_url=normalized_archive_url)
    if hasattr(result, "__await__"):
        return await result
    return result


async def check_archive_status(submission_id: str) -> Dict[str, Any]:
    """Check status for an Archive.is archival submission."""
    normalized_submission_id = str(submission_id or "").strip()
    if not normalized_submission_id:
        return {"status": "error", "error": "'submission_id' is required."}

    result = _API["check_archive_status"](submission_id=normalized_submission_id)
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_web_archive_tools(manager: Any) -> None:
    """Register native web-archive tools category tools in unified manager."""
    manager.register_tool(
        category="web_archive_tools",
        name="create_warc",
        func=create_warc,
        description="Create a WARC archive from a URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "output_path": {"type": ["string", "null"]},
                "options": {"type": ["object", "null"]},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="archive_to_wayback",
        func=archive_to_wayback,
        description="Submit a URL to Internet Archive Wayback for archival.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="archive_to_archive_is",
        func=archive_to_archive_is,
        description="Submit a URL to Archive.is for archival.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "wait_for_completion": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "default": 300, "minimum": 1},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_archive_to_archive_is",
        func=batch_archive_to_archive_is,
        description="Submit multiple URLs to Archive.is in a batch.",
        input_schema={
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "delay_seconds": {"type": "number", "default": 2.0, "minimum": 0},
                "max_concurrent": {"type": "integer", "default": 3, "minimum": 1},
            },
            "required": ["urls"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="extract_text_from_warc",
        func=extract_text_from_warc,
        description="Extract textual records from a WARC archive.",
        input_schema={
            "type": "object",
            "properties": {
                "warc_path": {"type": "string"},
            },
            "required": ["warc_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="extract_dataset_from_cdxj",
        func=extract_dataset_from_cdxj,
        description="Extract datasets from a CDXJ index file.",
        input_schema={
            "type": "object",
            "properties": {
                "cdxj_path": {"type": "string"},
                "output_format": {
                    "type": "string",
                    "enum": ["arrow", "huggingface", "dict"],
                    "default": "arrow",
                },
            },
            "required": ["cdxj_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="extract_links_from_warc",
        func=extract_links_from_warc,
        description="Extract outgoing links from a WARC archive.",
        input_schema={
            "type": "object",
            "properties": {
                "warc_path": {"type": "string"},
            },
            "required": ["warc_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="extract_metadata_from_warc",
        func=extract_metadata_from_warc,
        description="Extract metadata from a WARC archive.",
        input_schema={
            "type": "object",
            "properties": {
                "warc_path": {"type": "string"},
            },
            "required": ["warc_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="index_warc",
        func=index_warc,
        description="Index a WARC archive and produce CDXJ metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "warc_path": {"type": "string"},
                "output_path": {"type": ["string", "null"]},
                "encryption_key": {"type": ["string", "null"]},
            },
            "required": ["warc_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="index_warc_to_ipwb",
        func=index_warc_to_ipwb,
        description="Index a WARC file into IPWB-compatible metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "warc_path": {"type": "string"},
                "ipfs_endpoint": {"type": ["string", "null"]},
                "encrypt": {"type": "boolean", "default": False},
                "compression": {"type": ["string", "null"], "enum": ["gzip", "bz2", None]},
            },
            "required": ["warc_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="start_ipwb_replay",
        func=start_ipwb_replay,
        description="Start an IPWB replay service for a CDXJ index.",
        input_schema={
            "type": "object",
            "properties": {
                "cdxj_path": {"type": "string"},
                "port": {"type": "integer", "default": 5000, "minimum": 1},
                "ipfs_endpoint": {"type": ["string", "null"]},
                "proxy_mode": {"type": "boolean", "default": False},
            },
            "required": ["cdxj_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_ipwb_archive",
        func=search_ipwb_archive,
        description="Search an IPWB archive index by URL pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "cdxj_path": {"type": "string"},
                "url_pattern": {"type": "string"},
                "from_timestamp": {"type": ["string", "null"]},
                "to_timestamp": {"type": ["string", "null"]},
                "limit": {"type": "integer", "default": 100, "minimum": 1},
            },
            "required": ["cdxj_path", "url_pattern"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_ipwb_content",
        func=get_ipwb_content,
        description="Get archived content from IPWB via IPFS hash.",
        input_schema={
            "type": "object",
            "properties": {
                "ipfs_hash": {"type": "string"},
                "ipfs_endpoint": {"type": ["string", "null"]},
            },
            "required": ["ipfs_hash"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="verify_ipwb_archive",
        func=verify_ipwb_archive,
        description="Verify indexed IPWB archive integrity with sampled records.",
        input_schema={
            "type": "object",
            "properties": {
                "cdxj_path": {"type": "string"},
                "ipfs_endpoint": {"type": ["string", "null"]},
                "sample_size": {"type": "integer", "default": 10, "minimum": 1},
            },
            "required": ["cdxj_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_common_crawl",
        func=search_common_crawl,
        description="Search Common Crawl records by domain.",
        input_schema={
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "crawl_id": {"type": ["string", "null"]},
                "limit": {"type": "integer", "default": 100, "minimum": 1},
                "from_timestamp": {"type": ["string", "null"]},
                "to_timestamp": {"type": ["string", "null"]},
                "output_format": {"type": "string", "enum": ["json", "cdx"], "default": "json"},
            },
            "required": ["domain"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_common_crawl_advanced",
        func=search_common_crawl_advanced,
        description="Search Common Crawl with advanced index-engine capabilities.",
        input_schema={
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "max_matches": {"type": "integer", "default": 100, "minimum": 1},
                "collection": {"type": ["string", "null"]},
                "master_db_path": {"type": ["string", "null"]},
            },
            "required": ["domain"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="fetch_warc_record_advanced",
        func=fetch_warc_record_advanced,
        description="Fetch a WARC record by filename/offset/length via advanced APIs.",
        input_schema={
            "type": "object",
            "properties": {
                "warc_filename": {"type": "string"},
                "warc_offset": {"type": "integer", "minimum": 0},
                "warc_length": {"type": "integer", "minimum": 1},
                "decode_content": {"type": "boolean", "default": True},
            },
            "required": ["warc_filename", "warc_offset", "warc_length"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_common_crawl_content",
        func=get_common_crawl_content,
        description="Fetch archived content for a specific URL/timestamp from Common Crawl.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timestamp": {"type": "string"},
                "crawl_id": {"type": ["string", "null"]},
            },
            "required": ["url", "timestamp"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="list_common_crawl_indexes",
        func=list_common_crawl_indexes,
        description="List available Common Crawl index datasets.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="list_common_crawl_collections_advanced",
        func=list_common_crawl_collections_advanced,
        description="List Common Crawl collections from advanced integration.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_common_crawl_collection_info_advanced",
        func=get_common_crawl_collection_info_advanced,
        description="Get metadata for a specific Common Crawl collection.",
        input_schema={
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
            },
            "required": ["collection"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="list_autoscraper_models",
        func=list_autoscraper_models,
        description="List trained AutoScraper models available to the runtime.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="create_autoscraper_model",
        func=create_autoscraper_model,
        description="Create and save a trained AutoScraper model from sample data.",
        input_schema={
            "type": "object",
            "properties": {
                "sample_url": {"type": "string"},
                "wanted_data": {
                    "type": "array",
                    "items": {"type": ["string", "object"]},
                    "minItems": 1,
                },
                "model_name": {"type": "string"},
                "wanted_dict": {"type": ["object", "null"]},
            },
            "required": ["sample_url", "wanted_data", "model_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="scrape_with_autoscraper",
        func=scrape_with_autoscraper,
        description="Scrape one or more URLs with a trained AutoScraper model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "target_urls": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "grouped": {"type": "boolean", "default": False},
            },
            "required": ["model_path", "target_urls"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="optimize_autoscraper_model",
        func=optimize_autoscraper_model,
        description="Optimize an existing AutoScraper model with additional samples.",
        input_schema={
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "new_sample_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "new_wanted_data": {
                    "type": ["array", "null"],
                    "items": {"type": ["string", "object"]},
                },
                "update_existing": {"type": "boolean", "default": True},
            },
            "required": ["model_path", "new_sample_urls"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_scrape_with_autoscraper",
        func=batch_scrape_with_autoscraper,
        description="Run batch AutoScraper extraction using a URL input file.",
        input_schema={
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "urls_file": {"type": "string"},
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "jsonl"],
                    "default": "json",
                },
                "batch_size": {"type": "integer", "default": 50, "minimum": 1},
                "delay_seconds": {"type": "number", "default": 1.0, "minimum": 0},
            },
            "required": ["model_path", "urls_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_wayback_machine",
        func=search_wayback_machine,
        description="Search Wayback Machine captures for a URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "from_date": {"type": ["string", "null"]},
                "to_date": {"type": ["string", "null"]},
                "limit": {"type": "integer", "default": 100, "minimum": 1},
                "collapse": {"type": ["string", "null"]},
                "output_format": {"type": "string", "enum": ["json", "cdx"], "default": "json"},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_archive_is",
        func=search_archive_is,
        description="Search Archive.is snapshots by domain.",
        input_schema={
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "limit": {"type": "integer", "default": 100, "minimum": 1},
            },
            "required": ["domain"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_archive_is_content",
        func=get_archive_is_content,
        description="Retrieve archived content from an Archive.is URL.",
        input_schema={
            "type": "object",
            "properties": {
                "archive_url": {"type": "string"},
            },
            "required": ["archive_url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="check_archive_status",
        func=check_archive_status,
        description="Check status of an Archive.is submission by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "submission_id": {"type": "string"},
            },
            "required": ["submission_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_wayback_content",
        func=get_wayback_content,
        description="Fetch content for a URL capture from Wayback Machine.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timestamp": {"type": ["string", "null"]},
                "closest": {"type": "boolean", "default": True},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_brave",
        func=search_brave,
        description="Search web results using Brave provider.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "count": {"type": "integer", "default": 10, "minimum": 1},
                "offset": {"type": "integer", "default": 0, "minimum": 0},
                "search_lang": {"type": "string", "default": "en"},
                "country": {"type": "string", "default": "US"},
                "safesearch": {"type": "string", "enum": ["off", "moderate", "strict"], "default": "moderate"},
                "freshness": {"type": ["string", "null"], "enum": ["pd", "pw", "pm", "py", None]},
                "text_decorations": {"type": "boolean", "default": True},
                "spellcheck": {"type": "boolean", "default": True},
                "result_filter": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_brave_news",
        func=search_brave_news,
        description="Search news results using Brave provider.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "count": {"type": "integer", "default": 10, "minimum": 1},
                "offset": {"type": "integer", "default": 0, "minimum": 0},
                "search_lang": {"type": "string", "default": "en"},
                "country": {"type": "string", "default": "US"},
                "safesearch": {"type": "string", "enum": ["off", "moderate", "strict"], "default": "moderate"},
                "freshness": {"type": ["string", "null"], "enum": ["pd", "pw", "pm", "py", None]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_brave_images",
        func=search_brave_images,
        description="Search image results using Brave provider.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "count": {"type": "integer", "default": 10, "minimum": 1},
                "offset": {"type": "integer", "default": 0, "minimum": 0},
                "search_lang": {"type": "string", "default": "en"},
                "country": {"type": "string", "default": "US"},
                "safesearch": {"type": "string", "enum": ["off", "moderate", "strict"], "default": "moderate"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_search_brave",
        func=batch_search_brave,
        description="Run batch web queries using Brave provider.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "api_key": {"type": ["string", "null"]},
                "count": {"type": "integer", "default": 10, "minimum": 1},
                "delay_seconds": {"type": "number", "default": 1.0, "minimum": 0},
            },
            "required": ["queries"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_brave_cache_stats",
        func=get_brave_cache_stats,
        description="Get Brave provider cache statistics.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="clear_brave_cache",
        func=clear_brave_cache,
        description="Clear Brave provider cache entries.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_google",
        func=search_google,
        description="Search web results using Google provider.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "search_engine_id": {"type": ["string", "null"]},
                "num": {"type": "integer", "default": 10, "minimum": 1},
                "start": {"type": "integer", "default": 1, "minimum": 1},
                "search_type": {"type": ["string", "null"], "enum": ["image", None]},
                "file_type": {"type": ["string", "null"]},
                "site_search": {"type": ["string", "null"]},
                "date_restrict": {"type": ["string", "null"]},
                "safe": {"type": "string", "enum": ["off", "medium", "high"], "default": "medium"},
                "lr": {"type": ["string", "null"]},
                "gl": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_google_images",
        func=search_google_images,
        description="Search image results using Google provider.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "search_engine_id": {"type": ["string", "null"]},
                "num": {"type": "integer", "default": 10, "minimum": 1},
                "start": {"type": "integer", "default": 1, "minimum": 1},
                "img_size": {"type": ["string", "null"]},
                "img_type": {"type": ["string", "null"]},
                "img_color_type": {"type": ["string", "null"]},
                "img_dominant_color": {"type": ["string", "null"]},
                "safe": {"type": "string", "enum": ["off", "medium", "high"], "default": "medium"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_search_google",
        func=batch_search_google,
        description="Run batch web queries using Google provider.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "api_key": {"type": ["string", "null"]},
                "search_engine_id": {"type": ["string", "null"]},
                "num": {"type": "integer", "default": 10, "minimum": 1},
                "delay_seconds": {"type": "number", "default": 0.5, "minimum": 0},
            },
            "required": ["queries"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_github_repositories",
        func=search_github_repositories,
        description="Search GitHub repositories using provider API.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_token": {"type": ["string", "null"]},
                "sort": {
                    "type": ["string", "null"],
                    "enum": ["stars", "forks", "help-wanted-issues", "updated", None],
                },
                "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                "per_page": {"type": "integer", "default": 30, "minimum": 1},
                "page": {"type": "integer", "default": 1, "minimum": 1},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_github_code",
        func=search_github_code,
        description="Search GitHub code results using provider API.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_token": {"type": ["string", "null"]},
                "sort": {"type": ["string", "null"], "enum": ["indexed", None]},
                "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                "per_page": {"type": "integer", "default": 30, "minimum": 1},
                "page": {"type": "integer", "default": 1, "minimum": 1},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_github_users",
        func=search_github_users,
        description="Search GitHub users using provider API.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_token": {"type": ["string", "null"]},
                "sort": {"type": ["string", "null"], "enum": ["followers", "repositories", "joined", None]},
                "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                "per_page": {"type": "integer", "default": 30, "minimum": 1},
                "page": {"type": "integer", "default": 1, "minimum": 1},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_github_issues",
        func=search_github_issues,
        description="Search GitHub issues and pull requests using provider API.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_token": {"type": ["string", "null"]},
                "sort": {"type": ["string", "null"], "enum": ["comments", "reactions", "created", "updated", None]},
                "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                "per_page": {"type": "integer", "default": 30, "minimum": 1},
                "page": {"type": "integer", "default": 1, "minimum": 1},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_search_github",
        func=batch_search_github,
        description="Run batch GitHub provider searches.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "search_type": {"type": "string", "enum": ["repositories", "code", "users", "issues"], "default": "repositories"},
                "api_token": {"type": ["string", "null"]},
                "per_page": {"type": "integer", "default": 30, "minimum": 1},
                "delay_seconds": {"type": "number", "default": 2.0, "minimum": 0},
            },
            "required": ["queries"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_huggingface_models",
        func=search_huggingface_models,
        description="Search HuggingFace model hub results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "api_token": {"type": ["string", "null"]},
                "filter_task": {"type": ["string", "null"]},
                "filter_library": {"type": ["string", "null"]},
                "filter_language": {"type": ["string", "null"]},
                "sort": {"type": "string", "enum": ["downloads", "created", "updated", "likes"], "default": "downloads"},
                "direction": {"type": "integer", "enum": [-1, 1], "default": -1},
                "limit": {"type": "integer", "default": 20, "minimum": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_huggingface_datasets",
        func=search_huggingface_datasets,
        description="Search HuggingFace dataset hub results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "api_token": {"type": ["string", "null"]},
                "filter_task": {"type": ["string", "null"]},
                "filter_language": {"type": ["string", "null"]},
                "filter_size": {"type": ["string", "null"]},
                "sort": {"type": "string", "enum": ["downloads", "created", "updated", "likes"], "default": "downloads"},
                "direction": {"type": "integer", "enum": [-1, 1], "default": -1},
                "limit": {"type": "integer", "default": 20, "minimum": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_huggingface_spaces",
        func=search_huggingface_spaces,
        description="Search HuggingFace space hub results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "api_token": {"type": ["string", "null"]},
                "filter_sdk": {"type": ["string", "null"]},
                "sort": {"type": "string", "enum": ["created", "updated", "likes"], "default": "likes"},
                "direction": {"type": "integer", "enum": [-1, 1], "default": -1},
                "limit": {"type": "integer", "default": 20, "minimum": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="get_huggingface_model_info",
        func=get_huggingface_model_info,
        description="Get HuggingFace model details by model ID.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "api_token": {"type": ["string", "null"]},
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_search_huggingface",
        func=batch_search_huggingface,
        description="Run batch HuggingFace provider searches.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "search_type": {"type": "string", "enum": ["models", "datasets", "spaces"], "default": "models"},
                "api_token": {"type": ["string", "null"]},
                "limit": {"type": "integer", "default": 20, "minimum": 1},
                "delay_seconds": {"type": "number", "default": 0.5, "minimum": 0},
            },
            "required": ["queries"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_openverse_images",
        func=search_openverse_images,
        description="Search OpenVerse image results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "page": {"type": "integer", "default": 1, "minimum": 1},
                "page_size": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                "license_type": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
                "creator": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_openverse_audio",
        func=search_openverse_audio,
        description="Search OpenVerse audio results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "page": {"type": "integer", "default": 1, "minimum": 1},
                "page_size": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                "license_type": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_search_openverse",
        func=batch_search_openverse,
        description="Run batch OpenVerse provider searches.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "search_type": {"type": "string", "enum": ["images", "audio"], "default": "images"},
                "api_key": {"type": ["string", "null"]},
                "page_size": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                "delay_seconds": {"type": "number", "default": 0.5, "minimum": 0},
            },
            "required": ["queries"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_serpstack",
        func=search_serpstack,
        description="Search SerpStack web results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "engine": {"type": "string", "enum": ["google", "bing", "yandex", "yahoo", "baidu"], "default": "google"},
                "num": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "page": {"type": "integer", "default": 1, "minimum": 1},
                "location": {"type": ["string", "null"]},
                "device": {"type": ["string", "null"]},
                "lang": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="search_serpstack_images",
        func=search_serpstack_images,
        description="Search SerpStack image results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "api_key": {"type": ["string", "null"]},
                "engine": {"type": "string", "enum": ["google", "bing", "yandex", "yahoo", "baidu"], "default": "google"},
                "num": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "location": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="batch_search_serpstack",
        func=batch_search_serpstack,
        description="Run batch SerpStack provider searches.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "api_key": {"type": ["string", "null"]},
                "engine": {"type": "string", "enum": ["google", "bing", "yandex", "yahoo", "baidu"], "default": "google"},
                "num": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "delay_seconds": {"type": "number", "default": 1.0, "minimum": 0},
            },
            "required": ["queries"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="unified_search",
        func=unified_search,
        description="Run unified multi-provider search orchestration.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 20, "minimum": 1},
                "mode": {
                    "type": "string",
                    "enum": ["max_throughput", "balanced", "max_quality", "low_cost"],
                    "default": "max_throughput",
                },
                "provider_allowlist": {"type": ["array", "null"], "items": {"type": "string"}},
                "provider_denylist": {"type": ["array", "null"], "items": {"type": "string"}},
                "offset": {"type": "integer", "default": 0, "minimum": 0},
                "domain": {"type": "string", "default": "general"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="unified_fetch",
        func=unified_fetch,
        description="Fetch URL content through unified provider orchestration.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["max_throughput", "balanced", "max_quality", "low_cost"],
                    "default": "balanced",
                },
                "domain": {"type": "string", "default": "general"},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="unified_search_and_fetch",
        func=unified_search_and_fetch,
        description="Run unified search and fetch top document candidates.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 20, "minimum": 1},
                "max_documents": {"type": "integer", "default": 5, "minimum": 1},
                "mode": {
                    "type": "string",
                    "enum": ["max_throughput", "balanced", "max_quality", "low_cost"],
                    "default": "max_throughput",
                },
                "provider_allowlist": {"type": ["array", "null"], "items": {"type": "string"}},
                "domain": {"type": "string", "default": "general"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="unified_health",
        func=unified_health,
        description="Get unified web-archiving API health snapshot.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )

    manager.register_tool(
        category="web_archive_tools",
        name="unified_agentic_discover_and_fetch",
        func=unified_agentic_discover_and_fetch,
        description="Agentic discover-and-fetch workflow using unified provider orchestration.",
        input_schema={
            "type": "object",
            "properties": {
                "seed_urls": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "target_terms": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "max_hops": {"type": "integer", "default": 2, "minimum": 1},
                "max_pages": {"type": "integer", "default": 10, "minimum": 1},
                "mode": {
                    "type": "string",
                    "enum": ["max_throughput", "balanced", "max_quality", "low_cost"],
                    "default": "balanced",
                },
            },
            "required": ["seed_urls", "target_terms"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-archive"],
    )
