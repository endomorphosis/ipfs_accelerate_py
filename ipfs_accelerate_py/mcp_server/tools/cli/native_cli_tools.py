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


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    return {"result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def execute_command(
    command: str,
    args: Optional[List[str]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Execute a command via the MCP CLI facade."""
    if not isinstance(command, str) or not command.strip():
        return _error_result("command must be a non-empty string", command=command)
    if args is not None and (
        not isinstance(args, list)
        or not all(isinstance(arg, str) and arg.strip() for arg in args)
    ):
        return _error_result("args must be null or a list of non-empty strings", args=args)
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        return _error_result(
            "timeout_seconds must be an integer >= 1",
            timeout_seconds=timeout_seconds,
        )

    clean_command = command.strip()
    clean_args = [arg.strip() for arg in args] if args is not None else []

    try:
        result = _API["execute_command"](
            command=clean_command,
            args=clean_args,
            timeout_seconds=timeout_seconds,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("command", clean_command)
        envelope.setdefault("args", clean_args)
        envelope.setdefault("timeout_seconds", timeout_seconds)
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            command=clean_command,
            args=clean_args,
            timeout_seconds=timeout_seconds,
        )


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
                "command": {"type": "string", "minLength": 1},
                "args": {"type": ["array", "null"], "items": {"type": "string"}},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 60},
            },
            "required": ["command"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli"],
    )
