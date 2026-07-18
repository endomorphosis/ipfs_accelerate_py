"""Canonical dashboard-data helpers for ipfs_accelerate_py.mcp_server.

Drop-in replacement for the legacy ``ipfs_accelerate_py.mcp.tools.dashboard_data``
module.  Where a canonical implementation exists in mcp_server/tools it is
preferred; otherwise the legacy implementation is re-exported while the legacy
package is still present.
"""

from __future__ import annotations

from typing import Any, Dict


def get_cache_stats() -> Dict[str, Any]:
    """Return cache statistics.

    Tries the canonical cache_tools implementation first, then falls back to
    the legacy dashboard_data implementation, and finally returns an empty
    dict so callers never raise an ImportError.
    """
    try:
        from ipfs_accelerate_py.mcp_server.tools.cache_tools.native_cache_tools import (  # noqa: PLC0415
            get_cache_stats as _get_cache_stats,
        )
        import asyncio  # noqa: PLC0415
        result = _get_cache_stats(None)  # type: ignore[call-arg]
        if asyncio.iscoroutine(result):
            return asyncio.get_event_loop().run_until_complete(result)
        return result  # type: ignore[return-value]
    except Exception:
        pass

    try:
        from ipfs_accelerate_py.mcp.tools.dashboard_data import (  # type: ignore[import]  # noqa: PLC0415
            get_cache_stats as _legacy_get_cache_stats,
        )
        return _legacy_get_cache_stats()
    except Exception:
        return {}


def get_peer_status() -> Dict[str, Any]:
    """Return peer status information.

    Falls back to the legacy dashboard_data implementation, then an empty dict.
    """
    try:
        from ipfs_accelerate_py.mcp.tools.dashboard_data import (  # type: ignore[import]  # noqa: PLC0415
            get_peer_status as _legacy_get_peer_status,
        )
        return _legacy_get_peer_status()
    except Exception:
        return {}


def get_user_info() -> Dict[str, Any]:
    """Return current user information.

    Tries the canonical auth_tools implementation first, then falls back to
    the legacy dashboard_data implementation, and finally returns an empty dict.
    """
    try:
        from ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools import (  # noqa: PLC0415
            get_user_info as _get_user_info,
        )
        import asyncio  # noqa: PLC0415
        result = _get_user_info(None)  # type: ignore[call-arg]
        if asyncio.iscoroutine(result):
            return asyncio.get_event_loop().run_until_complete(result)
        return result  # type: ignore[return-value]
    except Exception:
        pass

    try:
        from ipfs_accelerate_py.mcp.tools.dashboard_data import (  # type: ignore[import]  # noqa: PLC0415
            get_user_info as _legacy_get_user_info,
        )
        return _legacy_get_user_info()
    except Exception:
        return {}


def get_system_metrics(start_time: float | None = None) -> Dict[str, Any]:
    """Return system metrics.

    Falls back to the legacy dashboard_data implementation, then an empty dict.
    """
    try:
        from ipfs_accelerate_py.mcp.tools.dashboard_data import (  # type: ignore[import]  # noqa: PLC0415
            get_system_metrics as _legacy_get_system_metrics,
        )
        return _legacy_get_system_metrics(start_time=start_time)
    except Exception:
        return {}


__all__ = ["get_cache_stats", "get_peer_status", "get_user_info", "get_system_metrics"]
