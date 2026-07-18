"""Canonical model tools wrapper for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.tools.model_tools_wrapper``.
Re-exports from the legacy module while it is present; provides stub
implementations when it is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ipfs_accelerate_py.mcp.tools.model_tools_wrapper import (  # type: ignore[import]
        search_models_tool,
        recommend_models_tool,
        get_model_details_tool,
        get_model_stats_tool,
    )
except Exception:
    def search_models_tool(  # type: ignore[misc]
        query: str,
        task_filter: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Stub: returns empty list when legacy module is not available."""
        return []

    def recommend_models_tool(  # type: ignore[misc]
        task_type: str,
        hardware: str = "cpu",
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Stub: returns empty list when legacy module is not available."""
        return []

    def get_model_details_tool(model_id: str) -> Dict[str, Any]:  # type: ignore[misc]
        """Stub: returns empty dict when legacy module is not available."""
        return {}

    def get_model_stats_tool() -> Dict[str, Any]:  # type: ignore[misc]
        """Stub: returns empty dict when legacy module is not available."""
        return {}


__all__ = [
    "search_models_tool",
    "recommend_models_tool",
    "get_model_details_tool",
    "get_model_stats_tool",
]
