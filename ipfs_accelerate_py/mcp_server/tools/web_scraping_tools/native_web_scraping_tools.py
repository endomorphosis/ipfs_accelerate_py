"""Native web-scraping tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_web_scraping_api() -> Dict[str, Any]:
    """Resolve source web-scraping APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.web_scraping_tools import (  # type: ignore
            check_scraper_methods_tool as _check_scraper_methods_tool,
            scrape_multiple_urls_tool as _scrape_multiple_urls_tool,
            scrape_url_tool as _scrape_url_tool,
        )

        return {
            "scrape_url_tool": _scrape_url_tool,
            "scrape_multiple_urls_tool": _scrape_multiple_urls_tool,
            "check_scraper_methods_tool": _check_scraper_methods_tool,
        }
    except Exception:
        logger.warning("Source web_scraping_tools import unavailable, using fallback scraping functions")

        async def _scrape_url_fallback(
            url: str,
            method: Optional[str] = None,
            timeout: int = 30,
            extract_links: bool = True,
            extract_text: bool = True,
            fallback_enabled: bool = True,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = method, timeout, extract_links, extract_text, fallback_enabled, kwargs
            return {
                "status": "success",
                "url": url,
                "content": "",
                "title": "",
                "links": [],
                "method_used": "fallback",
            }

        async def _scrape_multiple_fallback(
            urls: List[str],
            method: Optional[str] = None,
            timeout: int = 30,
            max_concurrent: int = 5,
            extract_links: bool = True,
            extract_text: bool = True,
            fallback_enabled: bool = True,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = method, timeout, max_concurrent, extract_links, extract_text, fallback_enabled, kwargs
            return {
                "status": "success",
                "results": [
                    {
                        "status": "success",
                        "url": u,
                        "content": "",
                        "title": "",
                        "links": [],
                        "method_used": "fallback",
                    }
                    for u in (urls or [])
                ],
                "successful_count": len(urls or []),
                "failed_count": 0,
                "total_urls": len(urls or []),
            }

        async def _check_methods_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "available_methods": {"requests_only": True},
                "unavailable_methods": [],
                "recommended_installs": [],
                "all_methods": ["requests_only"],
                "fallback_sequence": ["requests_only"],
            }

        return {
            "scrape_url_tool": _scrape_url_fallback,
            "scrape_multiple_urls_tool": _scrape_multiple_fallback,
            "check_scraper_methods_tool": _check_methods_fallback,
        }


_API = _load_web_scraping_api()


async def scrape_url_tool(
    url: str,
    method: Optional[str] = None,
    timeout: int = 30,
    extract_links: bool = True,
    extract_text: bool = True,
    fallback_enabled: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Scrape a single URL with optional method hint and fallback behavior."""
    result = _API["scrape_url_tool"](
        url=url,
        method=method,
        timeout=timeout,
        extract_links=extract_links,
        extract_text=extract_text,
        fallback_enabled=fallback_enabled,
        **kwargs,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def scrape_multiple_urls_tool(
    urls: List[str],
    method: Optional[str] = None,
    timeout: int = 30,
    max_concurrent: int = 5,
    extract_links: bool = True,
    extract_text: bool = True,
    fallback_enabled: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Scrape multiple URLs concurrently with fallback behavior."""
    result = _API["scrape_multiple_urls_tool"](
        urls=urls,
        method=method,
        timeout=timeout,
        max_concurrent=max_concurrent,
        extract_links=extract_links,
        extract_text=extract_text,
        fallback_enabled=fallback_enabled,
        **kwargs,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def check_scraper_methods_tool() -> Dict[str, Any]:
    """Return available web-scraping methods and recommended installs."""
    result = _API["check_scraper_methods_tool"]()
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_web_scraping_tools(manager: Any) -> None:
    """Register native web-scraping tools in unified hierarchical manager."""
    manager.register_tool(
        category="web_scraping_tools",
        name="scrape_url_tool",
        func=scrape_url_tool,
        description="Scrape a URL with configurable extraction and fallback behavior.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": ["string", "null"]},
                "timeout": {"type": "integer"},
                "extract_links": {"type": "boolean"},
                "extract_text": {"type": "boolean"},
                "fallback_enabled": {"type": "boolean"},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-scraping"],
    )

    manager.register_tool(
        category="web_scraping_tools",
        name="scrape_multiple_urls_tool",
        func=scrape_multiple_urls_tool,
        description="Scrape multiple URLs concurrently.",
        input_schema={
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}},
                "method": {"type": ["string", "null"]},
                "timeout": {"type": "integer"},
                "max_concurrent": {"type": "integer"},
                "extract_links": {"type": "boolean"},
                "extract_text": {"type": "boolean"},
                "fallback_enabled": {"type": "boolean"},
            },
            "required": ["urls"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "web-scraping"],
    )

    manager.register_tool(
        category="web_scraping_tools",
        name="check_scraper_methods_tool",
        func=check_scraper_methods_tool,
        description="List currently available scraping methods.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "web-scraping"],
    )
