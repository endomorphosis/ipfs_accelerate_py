#!/usr/bin/env python3
"""UNI-229 web scraping import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.web_scraping_tools import (
    check_scraper_methods_tool,
    scrape_multiple_urls_tool,
    scrape_url_tool,
)
from ipfs_accelerate_py.mcp_server.tools.web_scraping_tools import native_web_scraping_tools


def test_web_scraping_package_exports_supported_native_functions() -> None:
    assert scrape_url_tool is native_web_scraping_tools.scrape_url_tool
    assert scrape_multiple_urls_tool is native_web_scraping_tools.scrape_multiple_urls_tool
    assert check_scraper_methods_tool is native_web_scraping_tools.check_scraper_methods_tool