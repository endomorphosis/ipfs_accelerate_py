"""Canonical cli_endpoint_adapters shim for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters``.
Re-exports from the legacy module while it is present; provides stub classes
when it is not.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore[import]
        ClaudeCodeAdapter,
        OpenAICodexAdapter,
        GeminiCLIAdapter,
        register_cli_endpoint,
        list_cli_endpoints,
        execute_cli_inference,
    )

    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore[import]
            CLI_ADAPTER_REGISTRY,
        )
    except ImportError:
        CLI_ADAPTER_REGISTRY: Dict[str, Any] = {}

except Exception:
    CLI_ADAPTER_REGISTRY = {}

    class ClaudeCodeAdapter:  # type: ignore[no-redef]
        """Stub when legacy module unavailable."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

    class OpenAICodexAdapter:  # type: ignore[no-redef]
        """Stub when legacy module unavailable."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

    class GeminiCLIAdapter:  # type: ignore[no-redef]
        """Stub when legacy module unavailable."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

    def register_cli_endpoint(adapter: Any) -> Dict[str, Any]:  # type: ignore[misc]
        return {}

    def list_cli_endpoints() -> List[Dict[str, Any]]:  # type: ignore[misc]
        return []

    def execute_cli_inference(  # type: ignore[misc]
        endpoint_id: str,
        prompt: str,
        **kw: Any,
    ) -> Dict[str, Any]:
        return {}


__all__ = [
    "ClaudeCodeAdapter",
    "OpenAICodexAdapter",
    "GeminiCLIAdapter",
    "register_cli_endpoint",
    "list_cli_endpoints",
    "execute_cli_inference",
    "CLI_ADAPTER_REGISTRY",
]
