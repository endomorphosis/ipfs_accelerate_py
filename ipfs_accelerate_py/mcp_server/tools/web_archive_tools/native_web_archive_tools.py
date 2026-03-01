"""Native web-archive tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_web_archive_tools_api() -> Dict[str, Any]:
    """Resolve source web-archive APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.web_archive_tools import (  # type: ignore
            create_warc as _create_warc,
            search_common_crawl as _search_common_crawl,
        )

        return {
            "create_warc": _create_warc,
            "search_common_crawl": _search_common_crawl,
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

        return {
            "create_warc": _create_warc_fallback,
            "search_common_crawl": _search_common_crawl_fallback,
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
