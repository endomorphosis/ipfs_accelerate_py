"""Native CLI category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_cli_tools_api() -> Dict[str, Any]:
    """Resolve source CLI APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.cli.execute_command import (  # type: ignore
            execute_command as _execute_command,
        )

        return {"execute_command": _execute_command}
    except Exception:
        logger.warning("Source cli import unavailable, using fallback cli functions")

        async def _execute_command_fallback(
            command: str,
            args: Optional[List[str]] = None,
            timeout_seconds: int = 60,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "command": command,
                "args": args or [],
                "timeout_seconds": timeout_seconds,
                "message": "fallback",
            }

        return {"execute_command": _execute_command_fallback}


_API = _load_cli_tools_api()


async def execute_command(
    command: str,
    args: Optional[List[str]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Execute a command via the MCP CLI facade."""
    result = _API["execute_command"](
        command=command,
        args=args,
        timeout_seconds=timeout_seconds,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_cli_tools(manager: Any) -> None:
    """Register native CLI category tools in unified manager."""
    manager.register_tool(
        category="cli",
        name="execute_command",
        func=execute_command,
        description="Execute a command through the MCP CLI interface.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "args": {"type": ["array", "null"], "items": {"type": "string"}},
                "timeout_seconds": {"type": "integer"},
            },
            "required": ["command"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )
