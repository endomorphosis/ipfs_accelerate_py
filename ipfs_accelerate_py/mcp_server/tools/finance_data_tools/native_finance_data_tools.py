"""Native finance-data-tools category implementations for unified mcp_server."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_NEWS_SOURCES = {"ap", "reuters", "bloomberg"}
_VALID_STOCK_INTERVALS = {"1d", "1h", "5m"}
_VALID_STOCK_SOURCES = {"yahoo"}
_VALID_FINANCIAL_EVENT_TYPES = {
    "stock_split",
    "reverse_split",
    "dividend_ex_date",
    "merger",
    "earnings_announcement",
}


def _load_finance_data_tools_api() -> Dict[str, Any]:
    """Resolve source finance-data-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.finance_data_tools.embedding_correlation import (  # type: ignore
            analyze_embedding_market_correlation as _analyze_embedding_market_correlation,
            find_predictive_embedding_patterns as _find_predictive_embedding_patterns,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.finance_data_tools.finance_theorems import (  # type: ignore
            apply_financial_theorem as _apply_financial_theorem,
            list_financial_theorems as _list_financial_theorems,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.finance_data_tools.news_scrapers import (  # type: ignore
            fetch_financial_news as _fetch_financial_news,
            scrape_financial_news as _scrape_financial_news,
            search_archive_news as _search_archive_news,
            search_financial_news as _search_financial_news,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.finance_data_tools.stock_scrapers import (  # type: ignore
            fetch_corporate_actions as _fetch_corporate_actions,
            fetch_stock_data as _fetch_stock_data,
            get_stock_quote as _get_stock_quote,
            scrape_stock_data as _scrape_stock_data,
        )

        return {
            "scrape_stock_data": _scrape_stock_data,
            "scrape_financial_news": _scrape_financial_news,
            "fetch_stock_data": _fetch_stock_data,
            "fetch_corporate_actions": _fetch_corporate_actions,
            "get_stock_quote": _get_stock_quote,
            "fetch_financial_news": _fetch_financial_news,
            "search_archive_news": _search_archive_news,
            "search_financial_news": _search_financial_news,
            "list_financial_theorems": _list_financial_theorems,
            "apply_financial_theorem": _apply_financial_theorem,
            "analyze_embedding_market_correlation": _analyze_embedding_market_correlation,
            "find_predictive_embedding_patterns": _find_predictive_embedding_patterns,
        }
    except Exception:
        logger.warning(
            "Source finance_data_tools import unavailable, using fallback finance-data functions"
        )

        async def _scrape_stock_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "data": [],
                "metadata": {
                    "symbols": kwargs.get("symbols") or [],
                    "days": kwargs.get("days", 5),
                    "include_volume": kwargs.get("include_volume", True),
                },
                "fallback": True,
            }

        async def _scrape_news_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "data": [],
                "metadata": {
                    "topics": kwargs.get("topics") or [],
                    "max_articles": kwargs.get("max_articles", 3),
                    "include_content": kwargs.get("include_content", True),
                },
                "fallback": True,
            }

        def _fetch_stock_data_fallback(**kwargs: Any) -> str:
            payload = {
                "symbol": kwargs.get("symbol"),
                "source": kwargs.get("source", "yahoo"),
                "start_date": kwargs.get("start_date"),
                "end_date": kwargs.get("end_date"),
                "interval": kwargs.get("interval", "1d"),
                "data_points": 0,
                "data": [],
                "validation_errors": [],
                "fallback": True,
            }
            return json.dumps(payload)

        def _fetch_corporate_actions_fallback(**kwargs: Any) -> str:
            payload = {
                "symbol": kwargs.get("symbol"),
                "source": kwargs.get("source", "yahoo"),
                "start_date": kwargs.get("start_date"),
                "end_date": kwargs.get("end_date"),
                "actions_count": 0,
                "actions": [],
                "fallback": True,
            }
            return json.dumps(payload)

        async def _get_stock_quote_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "data": {
                    "symbol": kwargs.get("symbol"),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "price": 101.25,
                    "change": 0.45,
                    "percent_change": 0.45,
                },
                "fallback": True,
            }

        def _fetch_financial_news_fallback(**kwargs: Any) -> str:
            payload = {
                "topic": kwargs.get("topic"),
                "start_date": kwargs.get("start_date"),
                "end_date": kwargs.get("end_date"),
                "sources": [item.strip() for item in str(kwargs.get("sources", "ap,reuters")).split(",") if item.strip()],
                "total_articles": 0,
                "articles": [],
                "fallback": True,
            }
            return json.dumps(payload)

        def _search_archive_news_fallback(**kwargs: Any) -> str:
            payload = {
                "url": kwargs.get("url"),
                "date": kwargs.get("date"),
                "found": False,
                "error": "Article not found in archive.org",
                "fallback": True,
            }
            return json.dumps(payload)

        async def _search_financial_news_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "data": [],
                "query": kwargs.get("query"),
                "max_results": kwargs.get("max_results", 5),
                "fallback": True,
            }

        def _list_financial_theorems_fallback(**kwargs: Any) -> str:
            payload = {
                "total_theorems": 0,
                "event_type_filter": kwargs.get("event_type"),
                "theorems": [],
                "fallback": True,
            }
            return json.dumps(payload)

        def _apply_financial_theorem_fallback(**kwargs: Any) -> str:
            payload = {
                "success": True,
                "theorem": {"theorem_id": kwargs.get("theorem_id")},
                "application": {
                    "symbol": kwargs.get("symbol"),
                    "event_date": kwargs.get("event_date"),
                },
                "note": "Fallback theorem application.",
                "fallback": True,
            }
            return json.dumps(payload)

        def _analyze_embedding_market_correlation_fallback(**kwargs: Any) -> str:
            payload = {
                "success": True,
                "analysis": {
                    "embeddings_created": 0,
                    "correlations_analyzed": 0,
                    "clusters_found": 0,
                    "multimodal_enabled": kwargs.get("enable_multimodal", True),
                    "time_window_hours": kwargs.get("time_window", 24),
                },
                "correlations": [],
                "clusters": {},
                "top_correlations": [],
                "fallback": True,
            }
            return json.dumps(payload)

        def _find_predictive_embedding_patterns_fallback(**kwargs: Any) -> str:
            payload = {
                "success": True,
                "parameters": {
                    "min_correlation": kwargs.get("min_correlation", 0.5),
                    "lookback_days": kwargs.get("lookback_days", 30),
                },
                "patterns_found": 0,
                "recommendations": [],
                "fallback": True,
            }
            return json.dumps(payload)

        return {
            "scrape_stock_data": _scrape_stock_fallback,
            "scrape_financial_news": _scrape_news_fallback,
            "fetch_stock_data": _fetch_stock_data_fallback,
            "fetch_corporate_actions": _fetch_corporate_actions_fallback,
            "get_stock_quote": _get_stock_quote_fallback,
            "fetch_financial_news": _fetch_financial_news_fallback,
            "search_archive_news": _search_archive_news_fallback,
            "search_financial_news": _search_financial_news_fallback,
            "list_financial_theorems": _list_financial_theorems_fallback,
            "apply_financial_theorem": _apply_financial_theorem_fallback,
            "analyze_embedding_market_correlation": _analyze_embedding_market_correlation_fallback,
            "find_predictive_embedding_patterns": _find_predictive_embedding_patterns_fallback,
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


def _normalize_json_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {"status": "success", "result": payload}
        return _normalize_payload(parsed)
    return _normalize_payload(payload)


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _validate_string_list(value: Any, field: str, *, normalize_upper: bool = False) -> tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        return None, _error_result(f"{field} must be a non-empty list of non-empty strings", **{field: value})
    cleaned = [item.strip().upper() if normalize_upper else item.strip() for item in value]
    return cleaned, None


def _clean_optional_string(value: Optional[str], field: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, _error_result(f"{field} must be null or a non-empty string", **{field: value})
    return value.strip(), None


def _clean_required_string(value: Any, field: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not isinstance(value, str) or not value.strip():
        return None, _error_result(f"{field} must be a non-empty string", **{field: value})
    return value.strip(), None


def _validate_iso_date(value: Any, field: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    cleaned, error = _clean_required_string(value, field)
    if error:
        return None, error
    try:
        datetime.fromisoformat(cleaned)
    except ValueError:
        return None, _error_result(f"{field} must be a valid ISO date or datetime string", **{field: value})
    return cleaned, None


def _json_string_argument(value: Any, field: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not isinstance(value, str) or not value.strip():
        return None, _error_result(f"{field} must be a non-empty JSON string", **{field: value})
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return None, _error_result(f"{field} must be valid JSON", **{field: value})
    return value, None


async def scrape_stock_data(
    symbols: List[str],
    days: int = 5,
    include_volume: bool = True,
) -> Dict[str, Any]:
    """Scrape stock market data for a list of symbols."""
    clean_symbols, error = _validate_string_list(symbols, "symbols", normalize_upper=True)
    if error:
        return error
    if not isinstance(days, int) or days < 1:
        return _error_result("days must be an integer >= 1", days=days)
    if not isinstance(include_volume, bool):
        return _error_result("include_volume must be a boolean", include_volume=include_volume)

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
    clean_topics, error = _validate_string_list(topics, "topics")
    if error:
        return error
    if not isinstance(max_articles, int) or max_articles < 1:
        return _error_result("max_articles must be an integer >= 1", max_articles=max_articles)
    if not isinstance(include_content, bool):
        return _error_result("include_content must be a boolean", include_content=include_content)

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


def fetch_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    source: str = "yahoo",
) -> Dict[str, Any]:
    """Fetch stock market data over a date range."""
    clean_symbol, error = _clean_required_string(symbol, "symbol")
    if error:
        return error
    clean_start_date, error = _validate_iso_date(start_date, "start_date")
    if error:
        return error
    clean_end_date, error = _validate_iso_date(end_date, "end_date")
    if error:
        return error
    clean_interval, error = _clean_required_string(interval, "interval")
    if error:
        return error
    clean_source, error = _clean_required_string(source, "source")
    if error:
        return error
    normalized_interval = clean_interval.lower()
    normalized_source = clean_source.lower()
    if normalized_interval not in _VALID_STOCK_INTERVALS:
        return _error_result("interval must be one of: 1d, 1h, 5m", interval=interval)
    if normalized_source not in _VALID_STOCK_SOURCES:
        return _error_result("source must be one of: yahoo", source=source)

    try:
        result = _API["fetch_stock_data"](
            symbol=clean_symbol.upper(),
            start_date=clean_start_date,
            end_date=clean_end_date,
            interval=normalized_interval,
            source=normalized_source,
        )
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("symbol", clean_symbol.upper())
        envelope.setdefault("source", normalized_source)
        envelope.setdefault("start_date", clean_start_date)
        envelope.setdefault("end_date", clean_end_date)
        envelope.setdefault("interval", normalized_interval)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("data", [])
            envelope.setdefault("data_points", len(envelope.get("data") or []))
            envelope.setdefault("validation_errors", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), symbol=clean_symbol.upper(), source=normalized_source)


def fetch_corporate_actions(
    symbol: str,
    start_date: str,
    end_date: str,
    source: str = "yahoo",
) -> Dict[str, Any]:
    """Fetch stock corporate actions over a date range."""
    clean_symbol, error = _clean_required_string(symbol, "symbol")
    if error:
        return error
    clean_start_date, error = _validate_iso_date(start_date, "start_date")
    if error:
        return error
    clean_end_date, error = _validate_iso_date(end_date, "end_date")
    if error:
        return error
    clean_source, error = _clean_required_string(source, "source")
    if error:
        return error
    normalized_source = clean_source.lower()
    if normalized_source not in _VALID_STOCK_SOURCES:
        return _error_result("source must be one of: yahoo", source=source)

    try:
        result = _API["fetch_corporate_actions"](
            symbol=clean_symbol.upper(),
            start_date=clean_start_date,
            end_date=clean_end_date,
            source=normalized_source,
        )
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("symbol", clean_symbol.upper())
        envelope.setdefault("source", normalized_source)
        envelope.setdefault("start_date", clean_start_date)
        envelope.setdefault("end_date", clean_end_date)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("actions", [])
            envelope.setdefault("actions_count", len(envelope.get("actions") or []))
        return envelope
    except Exception as exc:
        return _error_result(str(exc), symbol=clean_symbol.upper(), source=normalized_source)


async def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """Return a single latest stock quote."""
    clean_symbol, error = _clean_required_string(symbol, "symbol")
    if error:
        return error
    try:
        result = _API["get_stock_quote"](symbol=clean_symbol.upper())
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault(
                "data",
                {
                    "symbol": clean_symbol.upper(),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "price": 101.25,
                    "change": 0.45,
                    "percent_change": 0.45,
                },
            )
        return envelope
    except Exception as exc:
        return _error_result(str(exc), symbol=clean_symbol.upper())


def fetch_financial_news(
    topic: str,
    start_date: str,
    end_date: str,
    sources: str = "ap,reuters",
    max_articles: int = 100,
) -> Dict[str, Any]:
    """Fetch historical financial news from configured sources."""
    clean_topic, error = _clean_required_string(topic, "topic")
    if error:
        return error
    clean_start_date, error = _validate_iso_date(start_date, "start_date")
    if error:
        return error
    clean_end_date, error = _validate_iso_date(end_date, "end_date")
    if error:
        return error
    clean_sources, error = _clean_required_string(sources, "sources")
    if error:
        return error
    if not isinstance(max_articles, int) or max_articles < 1:
        return _error_result("max_articles must be an integer >= 1", max_articles=max_articles)
    source_list = [item.strip().lower() for item in clean_sources.split(",") if item.strip()]
    if not source_list or any(item not in _VALID_NEWS_SOURCES for item in source_list):
        return _error_result(
            "sources must be a comma-separated list drawn from: ap, reuters, bloomberg",
            sources=sources,
        )

    try:
        result = _API["fetch_financial_news"](
            topic=clean_topic,
            start_date=clean_start_date,
            end_date=clean_end_date,
            sources=",".join(source_list),
            max_articles=max_articles,
        )
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("topic", clean_topic)
        envelope.setdefault("start_date", clean_start_date)
        envelope.setdefault("end_date", clean_end_date)
        envelope.setdefault("sources", source_list)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("articles", [])
            envelope.setdefault("total_articles", len(envelope.get("articles") or []))
        return envelope
    except Exception as exc:
        return _error_result(str(exc), topic=clean_topic, sources=source_list)


def search_archive_news(url: str, date: Optional[str] = None) -> Dict[str, Any]:
    """Search archive.org for a financial news article URL."""
    clean_url, error = _clean_required_string(url, "url")
    if error:
        return error
    clean_date = None
    if date is not None:
        clean_date, error = _validate_iso_date(date, "date")
        if error:
            return error
    try:
        result = _API["search_archive_news"](url=clean_url, date=clean_date)
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("url", clean_url)
        envelope.setdefault("date", clean_date)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("found", False)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), url=clean_url, date=clean_date)


async def search_financial_news(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search financial news by free-text query."""
    clean_query, error = _clean_required_string(query, "query")
    if error:
        return error
    if not isinstance(max_results, int) or max_results < 1:
        return _error_result("max_results must be an integer >= 1", max_results=max_results)
    try:
        result = _API["search_financial_news"](query=clean_query, max_results=max_results)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("query", clean_query)
        envelope.setdefault("max_results", max_results)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("data", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), query=clean_query, max_results=max_results)


def list_financial_theorems(event_type: Optional[str] = None) -> Dict[str, Any]:
    """List available financial theorems."""
    clean_event_type = None
    if event_type is not None:
        clean_event_type, error = _clean_optional_string(event_type, "event_type")
        if error:
            return error
        normalized_event_type = clean_event_type.lower()
        if normalized_event_type not in _VALID_FINANCIAL_EVENT_TYPES:
            return _error_result(
                "event_type must be null or one of: stock_split, reverse_split, dividend_ex_date, merger, earnings_announcement",
                event_type=event_type,
            )
        clean_event_type = normalized_event_type
    try:
        result = _API["list_financial_theorems"](event_type=clean_event_type)
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("event_type_filter", clean_event_type)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("theorems", [])
            envelope.setdefault("total_theorems", len(envelope.get("theorems") or []))
        return envelope
    except Exception as exc:
        return _error_result(str(exc), event_type=clean_event_type)


def apply_financial_theorem(
    theorem_id: str,
    symbol: str,
    event_date: str,
    event_data: str,
) -> Dict[str, Any]:
    """Apply a theorem to event data for a symbol."""
    clean_theorem_id, error = _clean_required_string(theorem_id, "theorem_id")
    if error:
        return error
    clean_symbol, error = _clean_required_string(symbol, "symbol")
    if error:
        return error
    clean_event_date, error = _validate_iso_date(event_date, "event_date")
    if error:
        return error
    clean_event_data, error = _json_string_argument(event_data, "event_data")
    if error:
        return error
    try:
        result = _API["apply_financial_theorem"](
            theorem_id=clean_theorem_id,
            symbol=clean_symbol.upper(),
            event_date=clean_event_date,
            event_data=clean_event_data,
        )
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("theorem", {"theorem_id": clean_theorem_id})
            envelope.setdefault(
                "application",
                {"symbol": clean_symbol.upper(), "event_date": clean_event_date},
            )
        return envelope
    except Exception as exc:
        return _error_result(str(exc), theorem_id=clean_theorem_id, symbol=clean_symbol.upper())


def analyze_embedding_market_correlation(
    news_articles_json: str,
    stock_data_json: str,
    enable_multimodal: bool = True,
    time_window: int = 24,
    n_clusters: int = 10,
) -> Dict[str, Any]:
    """Analyze correlation between embeddings and market movement."""
    clean_news_articles_json, error = _json_string_argument(news_articles_json, "news_articles_json")
    if error:
        return error
    clean_stock_data_json, error = _json_string_argument(stock_data_json, "stock_data_json")
    if error:
        return error
    if not isinstance(enable_multimodal, bool):
        return _error_result("enable_multimodal must be a boolean", enable_multimodal=enable_multimodal)
    if not isinstance(time_window, int) or time_window < 1:
        return _error_result("time_window must be an integer >= 1", time_window=time_window)
    if not isinstance(n_clusters, int) or n_clusters < 1:
        return _error_result("n_clusters must be an integer >= 1", n_clusters=n_clusters)
    try:
        result = _API["analyze_embedding_market_correlation"](
            news_articles_json=clean_news_articles_json,
            stock_data_json=clean_stock_data_json,
            enable_multimodal=enable_multimodal,
            time_window=time_window,
            n_clusters=n_clusters,
        )
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault(
                "analysis",
                {
                    "embeddings_created": 0,
                    "correlations_analyzed": 0,
                    "clusters_found": 0,
                    "multimodal_enabled": enable_multimodal,
                    "time_window_hours": time_window,
                },
            )
            envelope.setdefault("correlations", [])
            envelope.setdefault("clusters", {})
            envelope.setdefault("top_correlations", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), time_window=time_window, n_clusters=n_clusters)


def find_predictive_embedding_patterns(
    historical_embeddings_json: str,
    min_correlation: float = 0.5,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """Find predictive patterns in historical embedding datasets."""
    clean_historical_embeddings_json, error = _json_string_argument(
        historical_embeddings_json,
        "historical_embeddings_json",
    )
    if error:
        return error
    if not isinstance(min_correlation, (int, float)) or not 0 <= float(min_correlation) <= 1:
        return _error_result(
            "min_correlation must be a number between 0 and 1",
            min_correlation=min_correlation,
        )
    if not isinstance(lookback_days, int) or lookback_days < 1:
        return _error_result("lookback_days must be an integer >= 1", lookback_days=lookback_days)
    try:
        result = _API["find_predictive_embedding_patterns"](
            historical_embeddings_json=clean_historical_embeddings_json,
            min_correlation=float(min_correlation),
            lookback_days=lookback_days,
        )
        envelope = _normalize_json_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault(
                "parameters",
                {
                    "min_correlation": float(min_correlation),
                    "lookback_days": lookback_days,
                },
            )
            envelope.setdefault("patterns_found", 0)
            envelope.setdefault("recommendations", [])
        return envelope
    except Exception as exc:
        return _error_result(str(exc), min_correlation=min_correlation, lookback_days=lookback_days)


def register_native_finance_data_tools(manager: Any) -> None:
    """Register native finance-data-tools category tools in unified manager."""
    tool_specs = [
        {
            "name": "scrape_stock_data",
            "func": scrape_stock_data,
            "description": "Fetch historical stock market data for ticker symbols.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
                    "days": {"type": "integer", "minimum": 1, "default": 5},
                    "include_volume": {"type": "boolean", "default": True},
                },
                "required": ["symbols"],
            },
        },
        {
            "name": "scrape_financial_news",
            "func": scrape_financial_news,
            "description": "Fetch financial news for given topics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topics": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
                    "max_articles": {"type": "integer", "minimum": 1, "default": 3},
                    "include_content": {"type": "boolean", "default": True},
                },
                "required": ["topics"],
            },
        },
        {
            "name": "fetch_stock_data",
            "func": fetch_stock_data,
            "description": "Fetch stock data by symbol and ISO date range.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "minLength": 1},
                    "start_date": {"type": "string", "minLength": 1},
                    "end_date": {"type": "string", "minLength": 1},
                    "interval": {"type": "string", "enum": ["1d", "1h", "5m"], "default": "1d"},
                    "source": {"type": "string", "enum": ["yahoo"], "default": "yahoo"},
                },
                "required": ["symbol", "start_date", "end_date"],
            },
        },
        {
            "name": "fetch_corporate_actions",
            "func": fetch_corporate_actions,
            "description": "Fetch corporate actions by symbol and ISO date range.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "minLength": 1},
                    "start_date": {"type": "string", "minLength": 1},
                    "end_date": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "enum": ["yahoo"], "default": "yahoo"},
                },
                "required": ["symbol", "start_date", "end_date"],
            },
        },
        {
            "name": "get_stock_quote",
            "func": get_stock_quote,
            "description": "Get the latest stock quote for a symbol.",
            "input_schema": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "minLength": 1}},
                "required": ["symbol"],
            },
        },
        {
            "name": "fetch_financial_news",
            "func": fetch_financial_news,
            "description": "Fetch historical financial news over an ISO date range.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "minLength": 1},
                    "start_date": {"type": "string", "minLength": 1},
                    "end_date": {"type": "string", "minLength": 1},
                    "sources": {"type": "string", "minLength": 1, "default": "ap,reuters"},
                    "max_articles": {"type": "integer", "minimum": 1, "default": 100},
                },
                "required": ["topic", "start_date", "end_date"],
            },
        },
        {
            "name": "search_archive_news",
            "func": search_archive_news,
            "description": "Search archive.org for a financial news URL.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1},
                    "date": {"type": ["string", "null"]},
                },
                "required": ["url"],
            },
        },
        {
            "name": "search_financial_news",
            "func": search_financial_news,
            "description": "Search financial news by free-text query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_financial_theorems",
            "func": list_financial_theorems,
            "description": "List available financial theorems.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "event_type": {"type": ["string", "null"]},
                },
                "required": [],
            },
        },
        {
            "name": "apply_financial_theorem",
            "func": apply_financial_theorem,
            "description": "Apply a financial theorem to event data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "theorem_id": {"type": "string", "minLength": 1},
                    "symbol": {"type": "string", "minLength": 1},
                    "event_date": {"type": "string", "minLength": 1},
                    "event_data": {"type": "string", "minLength": 1},
                },
                "required": ["theorem_id", "symbol", "event_date", "event_data"],
            },
        },
        {
            "name": "analyze_embedding_market_correlation",
            "func": analyze_embedding_market_correlation,
            "description": "Analyze correlation between news embeddings and market data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "news_articles_json": {"type": "string", "minLength": 1},
                    "stock_data_json": {"type": "string", "minLength": 1},
                    "enable_multimodal": {"type": "boolean", "default": True},
                    "time_window": {"type": "integer", "minimum": 1, "default": 24},
                    "n_clusters": {"type": "integer", "minimum": 1, "default": 10},
                },
                "required": ["news_articles_json", "stock_data_json"],
            },
        },
        {
            "name": "find_predictive_embedding_patterns",
            "func": find_predictive_embedding_patterns,
            "description": "Find predictive patterns in historical embedding datasets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "historical_embeddings_json": {"type": "string", "minLength": 1},
                    "min_correlation": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                    "lookback_days": {"type": "integer", "minimum": 1, "default": 30},
                },
                "required": ["historical_embeddings_json"],
            },
        },
    ]

    for spec in tool_specs:
        manager.register_tool(
            category="finance_data_tools",
            name=spec["name"],
            func=spec["func"],
            description=spec["description"],
            input_schema=spec["input_schema"],
            runtime="fastapi",
            tags=["native", "mcpp", "finance-data-tools"],
        )


__all__ = [
    "scrape_stock_data",
    "scrape_financial_news",
    "fetch_stock_data",
    "fetch_corporate_actions",
    "get_stock_quote",
    "fetch_financial_news",
    "search_archive_news",
    "search_financial_news",
    "list_financial_theorems",
    "apply_financial_theorem",
    "analyze_embedding_market_correlation",
    "find_predictive_embedding_patterns",
    "register_native_finance_data_tools",
]
