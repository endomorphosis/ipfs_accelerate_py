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


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelope."""
    if isinstance(result, dict):
        payload = dict(result)
        failed = bool(payload.get("error")) or payload.get("success") is False
        if failed:
            payload["status"] = "error"
        else:
            payload.setdefault("status", "success")
        return payload
    return {"status": "success", "result": result}


async def execute_python_snippet(
    code: str,
    timeout_seconds: int = 30,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a Python snippet using source function tools with safe fallback."""
    normalized_code = str(code or "")
    if not normalized_code.strip():
        return {
            "status": "error",
            "message": "code must be a non-empty string",
            "code": code,
        }
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        return {
            "status": "error",
            "message": "timeout_seconds must be an integer >= 1",
            "timeout_seconds": timeout_seconds,
        }
    if context is not None and not isinstance(context, dict):
        return {
            "status": "error",
            "message": "context must be an object when provided",
            "context": context,
        }

    try:
        result = _API["execute_python_snippet"](
            code=normalized_code,
            timeout_seconds=timeout_seconds,
            context=context,
        )
        if hasattr(result, "__await__"):
            payload = _normalize_payload(await result)
        else:
            payload = _normalize_payload(result)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "timeout_seconds": timeout_seconds,
        }

    payload.setdefault("timeout_seconds", timeout_seconds)
    if payload.get("status") == "success":
        payload.setdefault("message", "Execution request processed")
        payload.setdefault("execution_time_ms", 0)
    return payload


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
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 30},
                "context": {"type": ["object", "null"]},
            },
            "required": ["code"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "function"],
    )
