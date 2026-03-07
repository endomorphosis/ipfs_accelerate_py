#!/usr/bin/env python3
"""UNI-137 finance_data_tools parity hardening tests."""

import json
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools import (
    analyze_embedding_market_correlation,
    apply_financial_theorem,
    fetch_corporate_actions,
    fetch_financial_news,
    fetch_stock_data,
    find_predictive_embedding_patterns,
    get_stock_quote,
    list_financial_theorems,
    register_native_finance_data_tools,
    scrape_financial_news,
    scrape_stock_data,
    search_archive_news,
    search_financial_news,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestUni137FinanceDataTools(unittest.TestCase):
    """Validate finance_data_tools wrapper contracts and normalized envelopes."""

    def test_get_tools_reports_expanded_schemas(self) -> None:
        manager = _DummyManager()
        register_native_finance_data_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        self.assertIn("fetch_stock_data", by_name)
        self.assertIn("get_stock_quote", by_name)
        self.assertIn("list_financial_theorems", by_name)
        self.assertIn("find_predictive_embedding_patterns", by_name)

        stock_schema = by_name["scrape_stock_data"]["input_schema"]
        self.assertEqual(stock_schema["properties"]["symbols"]["minItems"], 1)
        self.assertEqual(stock_schema["properties"]["symbols"]["items"]["minLength"], 1)
        self.assertEqual(stock_schema["properties"]["days"]["minimum"], 1)

        fetch_stock_schema = by_name["fetch_stock_data"]["input_schema"]
        self.assertEqual(fetch_stock_schema["properties"]["interval"]["enum"], ["1d", "1h", "5m"])
        self.assertEqual(fetch_stock_schema["properties"]["source"]["enum"], ["yahoo"])

        theorem_schema = by_name["apply_financial_theorem"]["input_schema"]
        self.assertEqual(
            theorem_schema["required"],
            ["theorem_id", "symbol", "event_date", "event_data"],
        )

        embedding_schema = by_name["find_predictive_embedding_patterns"]["input_schema"]
        self.assertEqual(embedding_schema["properties"]["min_correlation"]["maximum"], 1)
        self.assertEqual(embedding_schema["properties"]["lookback_days"]["minimum"], 1)

    def test_scrape_stock_data_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_stock_data(symbols=[], days=5, include_volume=True)
            self.assertEqual(result["status"], "error")
            self.assertIn("symbols", result["error"])

            result = await scrape_stock_data(symbols=["AAPL"], days=0, include_volume=True)
            self.assertEqual(result["status"], "error")
            self.assertIn("days", result["error"])

            result = await scrape_stock_data(symbols=["AAPL"], days=5, include_volume="yes")
            self.assertEqual(result["status"], "error")
            self.assertIn("include_volume", result["error"])

        anyio.run(_run)

    def test_scrape_financial_news_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_financial_news(topics=[], max_articles=3, include_content=True)
            self.assertEqual(result["status"], "error")
            self.assertIn("topics", result["error"])

            result = await scrape_financial_news(
                topics=["markets"], max_articles=0, include_content=True
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("max_articles", result["error"])

            result = await scrape_financial_news(
                topics=["markets"], max_articles=3, include_content="yes"
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("include_content", result["error"])

        anyio.run(_run)

    def test_stock_and_news_sync_wrapper_validation(self) -> None:
        result = fetch_stock_data("AAPL", "2026-02-01", "2026-02-28", interval="15m")
        self.assertEqual(result["status"], "error")
        self.assertIn("interval", result["error"])

        result = fetch_corporate_actions("", "2026-02-01", "2026-02-28")
        self.assertEqual(result["status"], "error")
        self.assertIn("symbol", result["error"])

        result = fetch_financial_news(
            "inflation",
            "2026-02-01",
            "2026-02-28",
            sources="nyt",
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("sources", result["error"])

        result = search_archive_news("https://example.com/article", date="not-a-date")
        self.assertEqual(result["status"], "error")
        self.assertIn("date", result["error"])

    def test_quote_search_theorem_and_embedding_validation(self) -> None:
        async def _run() -> None:
            result = await get_stock_quote("")
            self.assertEqual(result["status"], "error")
            self.assertIn("symbol", result["error"])

            result = await search_financial_news("earnings", max_results=0)
            self.assertEqual(result["status"], "error")
            self.assertIn("max_results", result["error"])

        anyio.run(_run)

        result = list_financial_theorems(event_type="unknown")
        self.assertEqual(result["status"], "error")
        self.assertIn("event_type", result["error"])

        result = apply_financial_theorem("split-theorem", "AAPL", "2026-02-01", "not-json")
        self.assertEqual(result["status"], "error")
        self.assertIn("event_data", result["error"])

        result = analyze_embedding_market_correlation("{}", "not-json")
        self.assertEqual(result["status"], "error")
        self.assertIn("stock_data_json", result["error"])

        result = find_predictive_embedding_patterns("{}", min_correlation=1.5)
        self.assertEqual(result["status"], "error")
        self.assertIn("min_correlation", result["error"])

    def test_success_envelope_shapes_for_sync_wrappers(self) -> None:
        stock = fetch_stock_data("aapl", "2026-02-01", "2026-02-28")
        self.assertIn(stock.get("status"), ["success", "error"])
        self.assertEqual(stock.get("symbol"), "AAPL")
        self.assertEqual(stock.get("interval"), "1d")

        corp = fetch_corporate_actions("aapl", "2026-02-01", "2026-02-28")
        self.assertIn(corp.get("status"), ["success", "error"])
        self.assertEqual(corp.get("symbol"), "AAPL")

        news = fetch_financial_news("inflation", "2026-02-01", "2026-02-28", sources="ap,reuters")
        self.assertIn(news.get("status"), ["success", "error"])
        self.assertEqual(news.get("topic"), "inflation")
        self.assertEqual(news.get("sources"), ["ap", "reuters"])

        archive = search_archive_news("https://example.com/article", date="2026-02-01")
        self.assertIn(archive.get("status"), ["success", "error"])
        self.assertEqual(archive.get("url"), "https://example.com/article")

    def test_success_envelope_shapes_for_async_wrappers(self) -> None:
        async def _run() -> None:
            stock = await scrape_stock_data(symbols=["aapl"], days=1, include_volume=False)
            self.assertIn(stock.get("status"), ["success", "error"])
            self.assertEqual(stock.get("symbols"), ["AAPL"])
            self.assertEqual(stock.get("days"), 1)

            news = await scrape_financial_news(
                topics=["inflation"],
                max_articles=1,
                include_content=False,
            )
            self.assertIn(news.get("status"), ["success", "error"])
            self.assertEqual(news.get("topics"), ["inflation"])
            self.assertEqual(news.get("max_articles"), 1)

            quote = await get_stock_quote("aapl")
            self.assertIn(quote.get("status"), ["success", "error"])
            self.assertEqual((quote.get("data") or {}).get("symbol"), "AAPL")

            search = await search_financial_news("inflation", max_results=2)
            self.assertIn(search.get("status"), ["success", "error"])
            self.assertEqual(search.get("query"), "inflation")
            self.assertEqual(search.get("max_results"), 2)

        anyio.run(_run)

    def test_success_envelope_shapes_for_theorem_and_embedding_wrappers(self) -> None:
        theorems = list_financial_theorems(event_type="stock_split")
        self.assertIn(theorems.get("status"), ["success", "error"])
        self.assertEqual(theorems.get("event_type_filter"), "stock_split")

        applied = apply_financial_theorem(
            "split-theorem",
            "aapl",
            "2026-02-01",
            json.dumps({"ratio": "2:1"}),
        )
        self.assertIn(applied.get("status"), ["success", "error"])
        self.assertEqual((applied.get("theorem") or {}).get("theorem_id"), "split-theorem")

        correlation = analyze_embedding_market_correlation("[]", "[]", time_window=12, n_clusters=3)
        self.assertIn(correlation.get("status"), ["success", "error"])
        self.assertEqual((correlation.get("analysis") or {}).get("time_window_hours"), 12)

        patterns = find_predictive_embedding_patterns("[]", min_correlation=0.7, lookback_days=10)
        self.assertIn(patterns.get("status"), ["success", "error"])
        self.assertEqual((patterns.get("parameters") or {}).get("lookback_days"), 10)

    def test_minimal_success_defaults_for_sync_wrappers(self) -> None:
        with patch(
            "ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools._API"
        ) as mock_api:
            mock_api.__getitem__.return_value = lambda **_: '{"status": "success"}'

            stock = fetch_stock_data("aapl", "2026-02-01", "2026-02-28")
            corp = fetch_corporate_actions("aapl", "2026-02-01", "2026-02-28")
            news = fetch_financial_news("inflation", "2026-02-01", "2026-02-28")
            archive = search_archive_news("https://example.com/article")
            theorems = list_financial_theorems()
            applied = apply_financial_theorem(
                "split-theorem",
                "aapl",
                "2026-02-01",
                json.dumps({"ratio": "2:1"}),
            )
            correlation = analyze_embedding_market_correlation("[]", "[]")
            patterns = find_predictive_embedding_patterns("[]")

        self.assertEqual(stock.get("data"), [])
        self.assertEqual(stock.get("data_points"), 0)
        self.assertEqual(corp.get("actions"), [])
        self.assertEqual(news.get("articles"), [])
        self.assertEqual(news.get("total_articles"), 0)
        self.assertEqual(archive.get("found"), False)
        self.assertEqual(theorems.get("theorems"), [])
        self.assertEqual(theorems.get("total_theorems"), 0)
        self.assertEqual((applied.get("theorem") or {}).get("theorem_id"), "split-theorem")
        self.assertEqual((correlation.get("analysis") or {}).get("time_window_hours"), 24)
        self.assertEqual(patterns.get("patterns_found"), 0)

    def test_async_minimal_success_defaults_and_error_inference(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools._API"
            ) as mock_api:
                async def _success(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _success

                stock = await scrape_stock_data(symbols=["aapl"], days=2, include_volume=False)
                news = await scrape_financial_news(
                    topics=["inflation"],
                    max_articles=2,
                    include_content=False,
                )
                quote = await get_stock_quote("aapl")
                search = await search_financial_news("inflation", max_results=4)

            self.assertEqual(stock.get("data"), [])
            self.assertEqual(
                stock.get("metadata"),
                {"symbols": ["AAPL"], "days": 2, "include_volume": False},
            )
            self.assertEqual(news.get("data"), [])
            self.assertEqual(
                news.get("metadata"),
                {"topics": ["inflation"], "max_articles": 2, "include_content": False},
            )
            self.assertEqual((quote.get("data") or {}).get("symbol"), "AAPL")
            self.assertEqual(search.get("data"), [])

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools._API"
            ) as mock_error_api:
                async def _error(**_: object) -> dict:
                    return {"error": "backend unavailable"}

                mock_error_api.__getitem__.return_value = _error
                error_result = await search_financial_news("inflation")

            self.assertEqual(error_result.get("status"), "error")
            self.assertIn("backend unavailable", str(error_result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
