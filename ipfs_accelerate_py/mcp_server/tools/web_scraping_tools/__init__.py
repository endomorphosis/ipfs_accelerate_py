"""Web scraping tools category for unified mcp_server."""

from .native_web_scraping_tools import (
	check_scraper_methods_tool,
	register_native_web_scraping_tools,
	scrape_multiple_urls_tool,
	scrape_url_tool,
)

__all__ = [
	"scrape_url_tool",
	"scrape_multiple_urls_tool",
	"check_scraper_methods_tool",
	"register_native_web_scraping_tools",
]
