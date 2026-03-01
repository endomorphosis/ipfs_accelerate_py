"""Native mcplusplus category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _load_mcplusplus_api() -> Dict[str, Any]:
    """Resolve source mcplusplus APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.mcplusplus import (  # type: ignore
            PeerEngine,
            TaskQueueEngine,
            WorkflowEngine,
        )

        return {
            "TaskQueueEngine": TaskQueueEngine,
            "PeerEngine": PeerEngine,
            "WorkflowEngine": WorkflowEngine,
        }
    except Exception:
        logger.warning("Source mcplusplus import unavailable, using fallback mcplusplus functions")
        return {}


_API = _load_mcplusplus_api()


async def mcplusplus_engine_status() -> Dict[str, Any]:
    """Return availability and instantiation status for MCP++ engine shims."""
    if not _API:
        return {
            "status": "success",
            "engines": {},
            "available": False,
            "fallback": True,
        }

    engines: Dict[str, Dict[str, Any]] = {}
    for name in ["TaskQueueEngine", "PeerEngine", "WorkflowEngine"]:
        cls = _API.get(name)
        if cls is None:
            engines[name] = {"available": False}
            continue
        try:
            _ = cls()
            engines[name] = {"available": True, "instantiated": True}
        except Exception as exc:
            engines[name] = {"available": True, "instantiated": False, "error": str(exc)}

    return {
        "status": "success",
        "available": True,
        "engines": engines,
    }


async def mcplusplus_list_engines() -> Dict[str, List[str]]:
    """List MCP++ engine classes exposed through source compatibility shims."""
    if not _API:
        return {"status": "success", "engines": [], "fallback": True}

    return {
        "status": "success",
        "engines": sorted(list(_API.keys())),
    }


def register_native_mcplusplus_tools(manager: Any) -> None:
    """Register native mcplusplus category tools in unified manager."""
    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_engine_status",
        func=mcplusplus_engine_status,
        description="Get availability status for MCP++ engine shim classes.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_list_engines",
        func=mcplusplus_list_engines,
        description="List MCP++ engine class names exported by source shim module.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )
