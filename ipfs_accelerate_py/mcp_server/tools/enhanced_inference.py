"""Canonical enhanced_inference shim for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.tools.enhanced_inference``.
Re-exports from the legacy module while it is present.
"""

from __future__ import annotations

from typing import Any, Dict

# Defaults used when legacy is unavailable.
CLI_PROVIDERS: Dict[str, Any] = {}
QUEUE_MONITOR: Dict[str, Any] = {}
HAVE_CLI_ADAPTERS: bool = False
CLI_ADAPTER_REGISTRY: Dict[str, Any] = {}


def register_tools(mcp: Any) -> None:
    """Stub: no-op when legacy module is unavailable."""


def get_queue_status() -> Dict[str, Any]:
    """Stub: empty dict when legacy module is unavailable."""
    return {}


def get_queue_history(limit: int = 50) -> Dict[str, Any]:
    """Stub: empty dict when legacy module is unavailable."""
    return {}


try:
    from ipfs_accelerate_py.mcp.tools.enhanced_inference import (  # type: ignore[import]
        CLI_PROVIDERS as CLI_PROVIDERS,  # noqa: F811
        QUEUE_MONITOR as QUEUE_MONITOR,  # noqa: F811
        register_tools as register_tools,  # noqa: F811
    )

    try:
        from ipfs_accelerate_py.mcp.tools.enhanced_inference import (  # type: ignore[import]
            HAVE_CLI_ADAPTERS as HAVE_CLI_ADAPTERS,  # noqa: F811
            CLI_ADAPTER_REGISTRY as CLI_ADAPTER_REGISTRY,  # noqa: F811
        )
    except ImportError:
        pass

    try:
        from ipfs_accelerate_py.mcp.tools.enhanced_inference import (  # type: ignore[import]
            get_queue_status as get_queue_status,  # noqa: F811
            get_queue_history as get_queue_history,  # noqa: F811
        )
    except ImportError:
        pass
except Exception:
    pass


__all__ = [
    "CLI_PROVIDERS",
    "QUEUE_MONITOR",
    "HAVE_CLI_ADAPTERS",
    "CLI_ADAPTER_REGISTRY",
    "register_tools",
    "get_queue_status",
    "get_queue_history",
]
