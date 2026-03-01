"""Native function-tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_function_api() -> Dict[str, Any]:
    """Resolve source function APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.functions.execute_python_snippet import (  # type: ignore
            execute_python_snippet as _execute_python_snippet,
        )

        return {
            "execute_python_snippet": _execute_python_snippet,
        }
    except Exception:
        logger.warning("Source function tools import unavailable, using fallback function implementations")

        async def _execute_fallback(
            code: str,
            timeout_seconds: int = 30,
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = timeout_seconds, context
            return {
                "status": "success",
                "message": (
                    "Code snippet received (length: "
                    + str(len(code))
                    + " chars) but not executed for security reasons."
                ),
                "execution_time_ms": 0,
            }

        return {
            "execute_python_snippet": _execute_fallback,
        }


_API = _load_function_api()


async def execute_python_snippet(
    code: str,
    timeout_seconds: int = 30,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a Python snippet using source function tools with safe fallback."""
    result = _API["execute_python_snippet"](
        code=code,
        timeout_seconds=timeout_seconds,
        context=context,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_function_tools(manager: Any) -> None:
    """Register native function tools in unified hierarchical manager."""
    manager.register_tool(
        category="functions",
        name="execute_python_snippet",
        func=execute_python_snippet,
        description="Receive and safely handle Python snippet execution requests.",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
                "context": {"type": ["object", "null"]},
            },
            "required": ["code"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "function"],
    )
