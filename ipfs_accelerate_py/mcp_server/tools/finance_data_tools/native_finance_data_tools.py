"""Native finance-data-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _load_finance_data_tools_api() -> Dict[str, Any]:
    """Resolve source finance-data-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.finance_data_tools.news_scrapers import (  # type: ignore
            scrape_financial_news as _scrape_financial_news,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.finance_data_tools.stock_scrapers import (  # type: ignore
            scrape_stock_data as _scrape_stock_data,
        )

        return {
            "scrape_stock_data": _scrape_stock_data,
            "scrape_financial_news": _scrape_financial_news,
        }
    except Exception:
        logger.warning(
            "Source finance_data_tools import unavailable, using fallback finance-data functions"
        )

        async def _scrape_stock_fallback(
            symbols: List[str],
            days: int = 5,
            include_volume: bool = True,
        ) -> Dict[str, Any]:
            _ = include_volume
            return {
                "status": "success",
                "data": [],
                "metadata": {"symbols": symbols, "days": days, "fallback": True},
            }

        async def _scrape_news_fallback(
            topics: List[str],
            max_articles: int = 3,
            include_content: bool = True,
        ) -> Dict[str, Any]:
            _ = include_content
            return {
                "status": "success",
                "data": [],
                "metadata": {"topics": topics, "max_articles": max_articles, "fallback": True},
            }

        return {
            "scrape_stock_data": _scrape_stock_fallback,
            "scrape_financial_news": _scrape_news_fallback,
        }


_API = _load_finance_data_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        if "status" not in envelope:
            if envelope.get("error") or envelope.get("success") is False:
                envelope["status"] = "error"
            else:
                envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def scrape_stock_data(
    symbols: List[str],
    days: int = 5,
    include_volume: bool = True,
) -> Dict[str, Any]:
    """Scrape stock market data for a list of symbols."""
    if not isinstance(symbols, list) or not symbols or not all(
        isinstance(symbol, str) and symbol.strip() for symbol in symbols
    ):
        return _error_result("symbols must be a non-empty list of non-empty strings", symbols=symbols)
    if not isinstance(days, int) or days < 1:
        return _error_result("days must be an integer >= 1", days=days)
    if not isinstance(include_volume, bool):
        return _error_result("include_volume must be a boolean", include_volume=include_volume)

    clean_symbols = [symbol.strip().upper() for symbol in symbols]
    try:
        result = _API["scrape_stock_data"](
            symbols=clean_symbols,
            days=days,
            include_volume=include_volume,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("symbols", clean_symbols)
        envelope.setdefault("days", days)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("data", [])
            envelope.setdefault(
                "metadata",
                {"symbols": clean_symbols, "days": days, "include_volume": include_volume},
            )
        return envelope
    except Exception as exc:
        return _error_result(str(exc), symbols=clean_symbols, days=days)


async def scrape_financial_news(
    topics: List[str],
    max_articles: int = 3,
    include_content: bool = True,
) -> Dict[str, Any]:
    """Scrape financial news articles for one or more topics."""
    if not isinstance(topics, list) or not topics or not all(
        isinstance(topic, str) and topic.strip() for topic in topics
    ):
        return _error_result("topics must be a non-empty list of non-empty strings", topics=topics)
    if not isinstance(max_articles, int) or max_articles < 1:
        return _error_result("max_articles must be an integer >= 1", max_articles=max_articles)
    if not isinstance(include_content, bool):
        return _error_result("include_content must be a boolean", include_content=include_content)

    clean_topics = [topic.strip() for topic in topics]
    try:
        result = _API["scrape_financial_news"](
            topics=clean_topics,
            max_articles=max_articles,
            include_content=include_content,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("topics", clean_topics)
        envelope.setdefault("max_articles", max_articles)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("data", [])
            envelope.setdefault(
                "metadata",
                {
                    "topics": clean_topics,
                    "max_articles": max_articles,
                    "include_content": include_content,
                },
            )
        return envelope
    except Exception as exc:
        return _error_result(str(exc), topics=clean_topics, max_articles=max_articles)


def register_native_finance_data_tools(manager: Any) -> None:
    """Register native finance-data-tools category tools in unified manager."""
    manager.register_tool(
        category="finance_data_tools",
        name="scrape_stock_data",
        func=scrape_stock_data,
        description="Fetch historical stock market data for ticker symbols.",
        input_schema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "days": {"type": "integer", "minimum": 1, "default": 5},
                "include_volume": {"type": "boolean", "default": True},
            },
            "required": ["symbols"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "finance-data-tools"],
    )

    manager.register_tool(
        category="finance_data_tools",
        name="scrape_financial_news",
        func=scrape_financial_news,
        description="Fetch financial news for given topics.",
        input_schema={
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "max_articles": {"type": "integer", "minimum": 1, "default": 3},
                "include_content": {"type": "boolean", "default": True},
            },
            "required": ["topics"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "finance-data-tools"],
    )
