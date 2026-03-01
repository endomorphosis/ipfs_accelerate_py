"""Native software-engineering-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_software_engineering_tools_api() -> Dict[str, Any]:
    """Resolve source software-engineering-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.software_engineering_tools.github_repository_scraper import (  # type: ignore
            scrape_repository as _scrape_repository,
            search_repositories as _search_repositories,
        )

        return {
            "scrape_repository": _scrape_repository,
            "search_repositories": _search_repositories,
        }
    except Exception:
        logger.warning(
            "Source software_engineering_tools import unavailable, using fallback software-engineering functions"
        )

        async def _scrape_repository_fallback(
            repository_url: str,
            include_prs: bool = True,
            include_issues: bool = True,
            include_workflows: bool = True,
            include_commits: bool = True,
            max_items: int = 100,
            github_token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (
                include_prs,
                include_issues,
                include_workflows,
                include_commits,
                max_items,
                github_token,
            )
            return {
                "status": "success",
                "repository_url": repository_url,
                "data": {},
                "fallback": True,
            }

        async def _search_repositories_fallback(
            query: str,
            max_results: int = 3,
            github_token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = github_token
            return {
                "status": "success",
                "query": query,
                "results": [],
                "count": 0,
                "max_results": max_results,
                "fallback": True,
            }

        return {
            "scrape_repository": _scrape_repository_fallback,
            "search_repositories": _search_repositories_fallback,
        }


_API = _load_software_engineering_tools_api()


async def scrape_repository(
    repository_url: str,
    include_prs: bool = True,
    include_issues: bool = True,
    include_workflows: bool = True,
    include_commits: bool = True,
    max_items: int = 100,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Scrape repository metadata and software engineering signals."""
    result = _API["scrape_repository"](
        repository_url=repository_url,
        include_prs=include_prs,
        include_issues=include_issues,
        include_workflows=include_workflows,
        include_commits=include_commits,
        max_items=max_items,
        github_token=github_token,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_repositories(
    query: str,
    max_results: int = 3,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Search GitHub repositories for software engineering analysis workflows."""
    result = _API["search_repositories"](
        query=query,
        max_results=max_results,
        github_token=github_token,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_software_engineering_tools(manager: Any) -> None:
    """Register native software-engineering-tools category tools in unified manager."""
    manager.register_tool(
        category="software_engineering_tools",
        name="scrape_repository",
        func=scrape_repository,
        description="Scrape GitHub repository metadata and engineering activity.",
        input_schema={
            "type": "object",
            "properties": {
                "repository_url": {"type": "string"},
                "include_prs": {"type": "boolean"},
                "include_issues": {"type": "boolean"},
                "include_workflows": {"type": "boolean"},
                "include_commits": {"type": "boolean"},
                "max_items": {"type": "integer"},
                "github_token": {"type": ["string", "null"]},
            },
            "required": ["repository_url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "software-engineering-tools"],
    )

    manager.register_tool(
        category="software_engineering_tools",
        name="search_repositories",
        func=search_repositories,
        description="Search GitHub repositories by query for engineering workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "github_token": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "software-engineering-tools"],
    )
