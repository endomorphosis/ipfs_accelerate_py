"""GitHub-tools category for unified mcp_server."""

from .native_github_tools import (
    github_create_workflow_queues,
    github_get_auth_status,
    github_get_cache_stats,
    github_get_runner_labels,
    github_list_runners,
    github_list_workflow_runs,
    register_native_github_tools,
)

__all__ = [
    "github_list_runners",
    "github_create_workflow_queues",
    "github_get_cache_stats",
    "github_get_auth_status",
    "github_list_workflow_runs",
    "github_get_runner_labels",
    "register_native_github_tools",
    "register_github_tools",
]

# Canonical alias expected by test_github_copilot_integration and other callers.
register_github_tools = register_native_github_tools
# Alias matching legacy ``ipfs_accelerate_py.mcp.tools.github_tools.register_tools`` name.
register_tools = register_native_github_tools
