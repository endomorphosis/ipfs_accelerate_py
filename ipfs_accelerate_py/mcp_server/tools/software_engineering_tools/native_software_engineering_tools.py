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
    if not isinstance(repository_url, str) or not repository_url.strip():
        return _error_result(
            "repository_url must be a non-empty string",
            repository_url=repository_url,
        )
    clean_repository_url = repository_url.strip()
    if not (
        clean_repository_url.startswith("https://github.com/")
        or clean_repository_url.startswith("http://github.com/")
    ):
        return _error_result(
            "repository_url must start with https://github.com/ or http://github.com/",
            repository_url=clean_repository_url,
        )
    if not isinstance(include_prs, bool):
        return _error_result("include_prs must be a boolean", include_prs=include_prs)
    if not isinstance(include_issues, bool):
        return _error_result("include_issues must be a boolean", include_issues=include_issues)
    if not isinstance(include_workflows, bool):
        return _error_result("include_workflows must be a boolean", include_workflows=include_workflows)
    if not isinstance(include_commits, bool):
        return _error_result("include_commits must be a boolean", include_commits=include_commits)
    if not isinstance(max_items, int) or max_items < 1:
        return _error_result("max_items must be an integer >= 1", max_items=max_items)
    if github_token is not None and (not isinstance(github_token, str) or not github_token.strip()):
        return _error_result("github_token must be null or a non-empty string", github_token=github_token)

    clean_github_token = github_token.strip() if isinstance(github_token, str) else None
    try:
        result = _API["scrape_repository"](
            repository_url=clean_repository_url,
            include_prs=include_prs,
            include_issues=include_issues,
            include_workflows=include_workflows,
            include_commits=include_commits,
            max_items=max_items,
            github_token=clean_github_token,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("repository_url", clean_repository_url)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("data", {})
        return envelope
    except Exception as exc:
        return _error_result(str(exc), repository_url=clean_repository_url)


async def search_repositories(
    query: str,
    max_results: int = 3,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Search GitHub repositories for software engineering analysis workflows."""
    if not isinstance(query, str) or not query.strip():
        return _error_result("query must be a non-empty string", query=query)
    if not isinstance(max_results, int) or max_results < 1:
        return _error_result("max_results must be an integer >= 1", max_results=max_results)
    if github_token is not None and (not isinstance(github_token, str) or not github_token.strip()):
        return _error_result("github_token must be null or a non-empty string", github_token=github_token)

    clean_query = query.strip()
    clean_github_token = github_token.strip() if isinstance(github_token, str) else None

    try:
        result = _API["search_repositories"](
            query=clean_query,
            max_results=max_results,
            github_token=clean_github_token,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("query", clean_query)
        envelope.setdefault("max_results", max_results)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("results", [])
            envelope.setdefault("count", len(envelope.get("results") or []))
        return envelope
    except Exception as exc:
        return _error_result(str(exc), query=clean_query, max_results=max_results)


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
                "include_prs": {"type": "boolean", "default": True},
                "include_issues": {"type": "boolean", "default": True},
                "include_workflows": {"type": "boolean", "default": True},
                "include_commits": {"type": "boolean", "default": True},
                "max_items": {"type": "integer", "minimum": 1, "default": 100},
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
                "max_results": {"type": "integer", "minimum": 1, "default": 3},
                "github_token": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "software-engineering-tools"],
    )
