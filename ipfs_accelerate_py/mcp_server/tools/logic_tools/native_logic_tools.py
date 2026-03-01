"""Native logic-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_logic_tools_api() -> Dict[str, Any]:
    """Resolve source logic-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.logic_capabilities_tool import (  # type: ignore
            logic_capabilities as _logic_capabilities,
            logic_health as _logic_health,
        )

        return {
            "logic_capabilities": _logic_capabilities,
            "logic_health": _logic_health,
        }
    except Exception:
        logger.warning("Source logic_tools import unavailable, using fallback logic functions")

        async def _logic_capabilities_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "logics": {},
                "conversions": [],
                "fallback": True,
            }

        async def _logic_health_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "healthy": 0,
                "total": 0,
                "fallback": True,
            }

        return {
            "logic_capabilities": _logic_capabilities_fallback,
            "logic_health": _logic_health_fallback,
        }


_API = _load_logic_tools_api()


async def logic_capabilities() -> Dict[str, Any]:
    """Return discovered logic-module capabilities for the unified runtime."""
    result = _API["logic_capabilities"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def logic_health() -> Dict[str, Any]:
    """Return logic-module health status for the unified runtime."""
    result = _API["logic_health"]()
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_logic_tools(manager: Any) -> None:
    """Register native logic-tools category tools in unified manager."""
    manager.register_tool(
        category="logic_tools",
        name="logic_capabilities",
        func=logic_capabilities,
        description="List capabilities for available logic modules.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="logic_health",
        func=logic_health,
        description="Get health status for available logic modules.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )
