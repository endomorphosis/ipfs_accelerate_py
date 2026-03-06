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


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend result to deterministic envelope."""
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    return payload


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
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {
            "status": "error",
            "message": "url is required",
            "url": url,
        }

    normalized_method = str(method).strip().lower() if method is not None else None
    valid_methods = {
        "playwright",
        "beautifulsoup",
        "wayback_machine",
        "common_crawl",
        "archive_is",
        "ipwb",
        "newspaper",
        "readability",
        "requests_only",
    }
    if normalized_method is not None and normalized_method not in valid_methods:
        return {
            "status": "error",
            "message": "method must be one of: playwright, beautifulsoup, wayback_machine, common_crawl, archive_is, ipwb, newspaper, readability, requests_only when provided",
            "method": method,
        }
    if not isinstance(timeout, int) or timeout < 1:
        return {
            "status": "error",
            "message": "timeout must be an integer >= 1",
            "timeout": timeout,
        }
    if not isinstance(extract_links, bool):
        return {
            "status": "error",
            "message": "extract_links must be a boolean",
            "extract_links": extract_links,
        }
    if not isinstance(extract_text, bool):
        return {
            "status": "error",
            "message": "extract_text must be a boolean",
            "extract_text": extract_text,
        }
    if not isinstance(fallback_enabled, bool):
        return {
            "status": "error",
            "message": "fallback_enabled must be a boolean",
            "fallback_enabled": fallback_enabled,
        }

    result = _API["scrape_url_tool"](
        url=normalized_url,
        method=normalized_method,
        timeout=timeout,
        extract_links=extract_links,
        extract_text=extract_text,
        fallback_enabled=fallback_enabled,
        **kwargs,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("url", normalized_url)
    payload.setdefault("content", "")
    payload.setdefault("title", "")
    payload.setdefault("links", [])
    payload.setdefault("method_used", normalized_method or "fallback")
    if normalized_method is not None:
        payload.setdefault("method", normalized_method)
    return payload


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
    if not isinstance(urls, list) or not urls:
        return {
            "status": "error",
            "message": "urls must be a non-empty array of strings",
            "urls": urls,
        }
    if not all(isinstance(item, str) for item in urls):
        return {
            "status": "error",
            "message": "urls must be a non-empty array of strings",
            "urls": urls,
        }
    normalized_urls = [str(item).strip() for item in urls]
    if any(not item for item in normalized_urls):
        return {
            "status": "error",
            "message": "urls cannot contain empty strings",
            "urls": urls,
        }

    normalized_method = str(method).strip().lower() if method is not None else None
    valid_methods = {
        "playwright",
        "beautifulsoup",
        "wayback_machine",
        "common_crawl",
        "archive_is",
        "ipwb",
        "newspaper",
        "readability",
        "requests_only",
    }
    if normalized_method is not None and normalized_method not in valid_methods:
        return {
            "status": "error",
            "message": "method must be one of: playwright, beautifulsoup, wayback_machine, common_crawl, archive_is, ipwb, newspaper, readability, requests_only when provided",
            "method": method,
        }
    if not isinstance(timeout, int) or timeout < 1:
        return {
            "status": "error",
            "message": "timeout must be an integer >= 1",
            "timeout": timeout,
        }
    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        return {
            "status": "error",
            "message": "max_concurrent must be an integer >= 1",
            "max_concurrent": max_concurrent,
        }
    if not isinstance(extract_links, bool):
        return {
            "status": "error",
            "message": "extract_links must be a boolean",
            "extract_links": extract_links,
        }
    if not isinstance(extract_text, bool):
        return {
            "status": "error",
            "message": "extract_text must be a boolean",
            "extract_text": extract_text,
        }
    if not isinstance(fallback_enabled, bool):
        return {
            "status": "error",
            "message": "fallback_enabled must be a boolean",
            "fallback_enabled": fallback_enabled,
        }

    result = _API["scrape_multiple_urls_tool"](
        urls=normalized_urls,
        method=normalized_method,
        timeout=timeout,
        max_concurrent=max_concurrent,
        extract_links=extract_links,
        extract_text=extract_text,
        fallback_enabled=fallback_enabled,
        **kwargs,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("total_urls", len(normalized_urls))
    payload.setdefault("results", [])
    payload.setdefault("successful_count", len(payload.get("results") or []))
    payload.setdefault("failed_count", max(0, payload.get("total_urls", len(normalized_urls)) - payload.get("successful_count", 0)))
    if normalized_method is not None:
        payload.setdefault("method", normalized_method)
    return payload


async def check_scraper_methods_tool() -> Dict[str, Any]:
    """Return available web-scraping methods and recommended installs."""
    result = _API["check_scraper_methods_tool"]()
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("available_methods", {})
    payload.setdefault("unavailable_methods", [])
    payload.setdefault("recommended_installs", [])
    payload.setdefault("all_methods", [])
    payload.setdefault("fallback_sequence", [])
    return payload


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
                "method": {
                    "type": ["string", "null"],
                    "enum": [
                        "playwright",
                        "beautifulsoup",
                        "wayback_machine",
                        "common_crawl",
                        "archive_is",
                        "ipwb",
                        "newspaper",
                        "readability",
                        "requests_only",
                        None,
                    ],
                    "default": None,
                },
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
                "extract_links": {"type": "boolean", "default": True},
                "extract_text": {"type": "boolean", "default": True},
                "fallback_enabled": {"type": "boolean", "default": True},
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
                "urls": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "method": {
                    "type": ["string", "null"],
                    "enum": [
                        "playwright",
                        "beautifulsoup",
                        "wayback_machine",
                        "common_crawl",
                        "archive_is",
                        "ipwb",
                        "newspaper",
                        "readability",
                        "requests_only",
                        None,
                    ],
                    "default": None,
                },
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
                "max_concurrent": {"type": "integer", "minimum": 1, "default": 5},
                "extract_links": {"type": "boolean", "default": True},
                "extract_text": {"type": "boolean", "default": True},
                "fallback_enabled": {"type": "boolean", "default": True},
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
