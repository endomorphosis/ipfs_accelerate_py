#!/usr/bin/env python3
"""UNI-199 finance import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.finance_data_tools import (
    analyze_embedding_market_correlation,
    apply_financial_theorem,
    fetch_corporate_actions,
    fetch_financial_news,
    fetch_stock_data,
    find_predictive_embedding_patterns,
    get_stock_quote,
    list_financial_theorems,
    scrape_financial_news,
    scrape_stock_data,
    search_archive_news,
    search_financial_news,
)
from ipfs_accelerate_py.mcp_server.tools.finance_data_tools import native_finance_data_tools


def test_finance_package_exports_supported_native_functions() -> None:
    assert scrape_stock_data is native_finance_data_tools.scrape_stock_data
    assert scrape_financial_news is native_finance_data_tools.scrape_financial_news
    assert fetch_stock_data is native_finance_data_tools.fetch_stock_data
    assert fetch_corporate_actions is native_finance_data_tools.fetch_corporate_actions
    assert get_stock_quote is native_finance_data_tools.get_stock_quote
    assert fetch_financial_news is native_finance_data_tools.fetch_financial_news
    assert search_archive_news is native_finance_data_tools.search_archive_news
    assert search_financial_news is native_finance_data_tools.search_financial_news
    assert list_financial_theorems is native_finance_data_tools.list_financial_theorems
    assert apply_financial_theorem is native_finance_data_tools.apply_financial_theorem
    assert analyze_embedding_market_correlation is native_finance_data_tools.analyze_embedding_market_correlation
    assert find_predictive_embedding_patterns is native_finance_data_tools.find_predictive_embedding_patterns
