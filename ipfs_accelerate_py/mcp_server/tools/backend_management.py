"""Canonical backend_management shim for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.tools.backend_management``.
Re-exports from the legacy module while it is present; provides stubs otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ipfs_accelerate_py.mcp.tools.backend_management import (  # type: ignore[import]
        list_inference_backends,
        get_backend_status,
        select_backend_for_inference,
    )
except Exception:
    def list_inference_backends(  # type: ignore[misc]
        filter_available: bool = False,
        **kw: Any,
    ) -> List[Dict[str, Any]]:
        """Stub: returns empty list when legacy module is unavailable."""
        return []

    def get_backend_status() -> Dict[str, Any]:  # type: ignore[misc]
        """Stub: returns empty dict when legacy module is unavailable."""
        return {}

    def select_backend_for_inference(  # type: ignore[misc]
        task_type: str = "",
        **kw: Any,
    ) -> Optional[str]:
        """Stub: returns None when legacy module is unavailable."""
        return None


__all__ = [
    "list_inference_backends",
    "get_backend_status",
    "select_backend_for_inference",
]
