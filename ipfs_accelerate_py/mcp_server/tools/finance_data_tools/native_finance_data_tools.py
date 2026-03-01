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


async def scrape_stock_data(
    symbols: List[str],
    days: int = 5,
    include_volume: bool = True,
) -> Dict[str, Any]:
    """Scrape stock market data for a list of symbols."""
    result = _API["scrape_stock_data"](
        symbols=symbols,
        days=days,
        include_volume=include_volume,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def scrape_financial_news(
    topics: List[str],
    max_articles: int = 3,
    include_content: bool = True,
) -> Dict[str, Any]:
    """Scrape financial news articles for one or more topics."""
    result = _API["scrape_financial_news"](
        topics=topics,
        max_articles=max_articles,
        include_content=include_content,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "symbols": {"type": "array", "items": {"type": "string"}},
                "days": {"type": "integer"},
                "include_volume": {"type": "boolean"},
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
                "topics": {"type": "array", "items": {"type": "string"}},
                "max_articles": {"type": "integer"},
                "include_content": {"type": "boolean"},
            },
            "required": ["topics"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "finance-data-tools"],
    )
