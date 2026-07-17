"""Native github-tools category implementations for unified mcp_server.

Exposes GitHub CLI operations (workflow management, runner management, cache
operations) from the legacy ``ipfs_accelerate_py.mcp.tools.github_tools``
module through the unified MCP++ tool dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_github_tools_api() -> Dict[str, Any]:
    """Resolve source github-tools APIs with compatibility fallback."""
    try:
        import ipfs_accelerate_py.mcp.tools.github_tools as _gh_mod  # type: ignore

        return {"_module": _gh_mod}
    except Exception:
        logger.warning("Source github_tools import unavailable, using fallback stubs")
        return {}


_API = _load_github_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _call_gh(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Delegate a call to the legacy github_tools module function."""
    mod = _API.get("_module")
    if mod is None:
        return {"runners": [], "workflows": [], "cache": {}, "auth": {}, "runs": [], "labels": []}
    fn = getattr(mod, func_name, None)
    if fn is None:
        return {}
    return fn(**kwargs)


async def github_list_runners(
    repo: Optional[str] = None,
    org: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List GitHub Actions runners for a repository or organisation."""
    try:
        result = _call_gh("gh_list_runners", repo=repo, org=org, status=status)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def github_create_workflow_queues(
    workflow_names: Optional[List[str]] = None,
    repo: Optional[str] = None,
) -> Dict[str, Any]:
    """Create P2P workflow queues for GitHub Actions workflows."""
    try:
        result = _call_gh(
            "gh_create_workflow_queues",
            workflow_names=workflow_names,
            repo=repo,
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def github_get_cache_stats(repo: Optional[str] = None) -> Dict[str, Any]:
    """Get GitHub Actions cache statistics for a repository."""
    try:
        result = _call_gh("gh_get_cache_stats", repo=repo)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def github_get_auth_status() -> Dict[str, Any]:
    """Get the current GitHub CLI authentication status."""
    try:
        result = _call_gh("gh_get_auth_status")
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def github_list_workflow_runs(
    repo: Optional[str] = None,
    workflow: Optional[str] = None,
    branch: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """List recent GitHub Actions workflow runs."""
    try:
        result = _call_gh(
            "gh_list_workflow_runs",
            repo=repo,
            workflow=workflow,
            branch=branch,
            limit=limit,
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def github_get_runner_labels(
    repo: Optional[str] = None,
    org: Optional[str] = None,
) -> Dict[str, Any]:
    """Get available labels for GitHub Actions runners."""
    try:
        result = _call_gh("gh_get_runner_labels", repo=repo, org=org)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


def register_native_github_tools(manager: Any) -> None:
    """Register native github-tools category tools in unified manager."""
    manager.register_tool(
        category="github_tools",
        name="github_list_runners",
        func=github_list_runners,
        description="List GitHub Actions runners for a repository or organisation.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format."},
                "org": {"type": "string", "description": "Organisation name."},
                "status": {
                    "type": "string",
                    "description": "Filter by runner status (active, idle, offline).",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "github-tools"],
    )
    manager.register_tool(
        category="github_tools",
        name="github_create_workflow_queues",
        func=github_create_workflow_queues,
        description="Create P2P workflow queues for GitHub Actions workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of workflow file names.",
                },
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format."},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "github-tools"],
    )
    manager.register_tool(
        category="github_tools",
        name="github_get_cache_stats",
        func=github_get_cache_stats,
        description="Get GitHub Actions cache statistics for a repository.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format."}
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "github-tools"],
    )
    manager.register_tool(
        category="github_tools",
        name="github_get_auth_status",
        func=github_get_auth_status,
        description="Get the current GitHub CLI authentication status.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "github-tools"],
    )
    manager.register_tool(
        category="github_tools",
        name="github_list_workflow_runs",
        func=github_list_workflow_runs,
        description="List recent GitHub Actions workflow runs.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format."},
                "workflow": {"type": "string", "description": "Workflow file name or ID."},
                "branch": {"type": "string", "description": "Branch name filter."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of runs to return.",
                    "default": 20,
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "github-tools"],
    )
    manager.register_tool(
        category="github_tools",
        name="github_get_runner_labels",
        func=github_get_runner_labels,
        description="Get available labels for GitHub Actions runners.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format."},
                "org": {"type": "string", "description": "Organisation name."},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "github-tools"],
    )
