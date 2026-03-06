#!/usr/bin/env python3
"""UNI-137 finance_data_tools parity hardening tests."""

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools import (
    register_native_finance_data_tools,
    scrape_financial_news,
    scrape_stock_data,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestUni137FinanceDataTools(unittest.TestCase):
    """Validate finance_data_tools wrapper contracts and normalized envelopes."""

    def test_get_tools_reports_tightened_schemas(self) -> None:
        manager = _DummyManager()
        register_native_finance_data_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        stock_schema = by_name["scrape_stock_data"]["input_schema"]
        self.assertEqual(stock_schema["properties"]["symbols"]["minItems"], 1)
        self.assertEqual(stock_schema["properties"]["symbols"]["items"]["minLength"], 1)
        self.assertEqual(stock_schema["properties"]["days"]["minimum"], 1)

        news_schema = by_name["scrape_financial_news"]["input_schema"]
        self.assertEqual(news_schema["properties"]["topics"]["minItems"], 1)
        self.assertEqual(news_schema["properties"]["topics"]["items"]["minLength"], 1)
        self.assertEqual(news_schema["properties"]["max_articles"]["minimum"], 1)

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

    def test_scrape_stock_data_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await scrape_stock_data(symbols=["aapl"], days=1, include_volume=False)
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("symbols"), ["AAPL"])
            self.assertEqual(result.get("days"), 1)

        anyio.run(_run)

    def test_scrape_financial_news_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await scrape_financial_news(
                topics=["inflation"],
                max_articles=1,
                include_content=False,
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("topics"), ["inflation"])
            self.assertEqual(result.get("max_articles"), 1)

        anyio.run(_run)

    def test_scrape_stock_data_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl

                result = await scrape_stock_data(symbols=["aapl"], days=2, include_volume=False)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("symbols"), ["AAPL"])
            self.assertEqual(result.get("days"), 2)
            self.assertEqual(result.get("data"), [])
            self.assertEqual(
                result.get("metadata"),
                {"symbols": ["AAPL"], "days": 2, "include_volume": False},
            )

        anyio.run(_run)

    def test_scrape_financial_news_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl

                result = await scrape_financial_news(
                    topics=["inflation"],
                    max_articles=2,
                    include_content=False,
                )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("topics"), ["inflation"])
            self.assertEqual(result.get("max_articles"), 2)
            self.assertEqual(result.get("data"), [])
            self.assertEqual(
                result.get("metadata"),
                {"topics": ["inflation"], "max_articles": 2, "include_content": False},
            )

        anyio.run(_run)

    def test_scrape_financial_news_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.finance_data_tools.native_finance_data_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"error": "backend unavailable"}

                mock_api.__getitem__.return_value = _impl

                result = await scrape_financial_news(topics=["inflation"])

            self.assertEqual(result.get("status"), "error")
            self.assertIn("backend unavailable", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
