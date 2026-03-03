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
            batch_archive_to_archive_is as _batch_archive_to_archive_is,
            check_archive_status as _check_archive_status,
            create_warc as _create_warc,
            extract_dataset_from_cdxj as _extract_dataset_from_cdxj,
            extract_links_from_warc as _extract_links_from_warc,
            extract_metadata_from_warc as _extract_metadata_from_warc,
            extract_text_from_warc as _extract_text_from_warc,
            get_archive_is_content as _get_archive_is_content,
            get_common_crawl_content as _get_common_crawl_content,
            get_ipwb_content as _get_ipwb_content,
            get_wayback_content as _get_wayback_content,
            index_warc as _index_warc,
            index_warc_to_ipwb as _index_warc_to_ipwb,
            list_common_crawl_indexes as _list_common_crawl_indexes,
            search_archive_is as _search_archive_is,
            search_common_crawl as _search_common_crawl,
            search_ipwb_archive as _search_ipwb_archive,
            search_wayback_machine as _search_wayback_machine,
            start_ipwb_replay as _start_ipwb_replay,
            verify_ipwb_archive as _verify_ipwb_archive,
        )

        return {
            "archive_to_archive_is": _archive_to_archive_is,
            "archive_to_wayback": _archive_to_wayback,
            "batch_archive_to_archive_is": _batch_archive_to_archive_is,
            "check_archive_status": _check_archive_status,
            "create_warc": _create_warc,
            "extract_dataset_from_cdxj": _extract_dataset_from_cdxj,
            "extract_links_from_warc": _extract_links_from_warc,
            "extract_text_from_warc": _extract_text_from_warc,
            "extract_metadata_from_warc": _extract_metadata_from_warc,
            "get_archive_is_content": _get_archive_is_content,
            "search_common_crawl": _search_common_crawl,
            "get_common_crawl_content": _get_common_crawl_content,
            "get_ipwb_content": _get_ipwb_content,
            "get_wayback_content": _get_wayback_content,
            "index_warc": _index_warc,
            "index_warc_to_ipwb": _index_warc_to_ipwb,
            "list_common_crawl_indexes": _list_common_crawl_indexes,
            "search_archive_is": _search_archive_is,
            "search_ipwb_archive": _search_ipwb_archive,
            "search_wayback_machine": _search_wayback_machine,
            "start_ipwb_replay": _start_ipwb_replay,
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
            "check_archive_status": _check_archive_status_fallback,
            "create_warc": _create_warc_fallback,
            "extract_dataset_from_cdxj": _extract_dataset_from_cdxj_fallback,
            "extract_links_from_warc": _extract_links_from_warc_fallback,
            "extract_text_from_warc": _extract_text_from_warc_fallback,
            "extract_metadata_from_warc": _extract_metadata_from_warc_fallback,
            "get_archive_is_content": _get_archive_is_content_fallback,
            "search_common_crawl": _search_common_crawl_fallback,
            "get_common_crawl_content": _get_common_crawl_content_fallback,
            "get_ipwb_content": _get_ipwb_content_fallback,
            "get_wayback_content": _get_wayback_content_fallback,
            "index_warc": _index_warc_fallback,
            "index_warc_to_ipwb": _index_warc_to_ipwb_fallback,
            "list_common_crawl_indexes": _list_common_crawl_indexes_fallback,
            "search_archive_is": _search_archive_is_fallback,
            "search_ipwb_archive": _search_ipwb_archive_fallback,
            "search_wayback_machine": _search_wayback_machine_fallback,
            "start_ipwb_replay": _start_ipwb_replay_fallback,
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
    result = _API["search_common_crawl"](
        domain=domain,
        crawl_id=crawl_id,
        limit=limit,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        output_format=output_format,
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
                "limit": {"type": "integer"},
                "from_timestamp": {"type": ["string", "null"]},
                "to_timestamp": {"type": ["string", "null"]},
                "output_format": {"type": "string"},
            },
            "required": ["domain"],
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
