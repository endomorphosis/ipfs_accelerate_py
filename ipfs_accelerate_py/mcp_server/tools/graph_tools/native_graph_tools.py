"""Native graph-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_graph_tools_api() -> Dict[str, Any]:
    """Resolve source graph-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.graph_tools import (  # type: ignore
            graph_add_entity as _graph_add_entity,
            graph_create as _graph_create,
        )

        return {
            "graph_create": _graph_create,
            "graph_add_entity": _graph_add_entity,
        }
    except Exception:
        logger.warning("Source graph_tools import unavailable, using fallback graph-tools functions")

        async def _create_fallback(driver_url: Optional[str] = None) -> Dict[str, Any]:
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "driver_url": driver_url or "ipfs://localhost:5001",
            }

        async def _add_entity_fallback(
            entity_id: str,
            entity_type: str,
            properties: Optional[Dict[str, Any]] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = properties, driver_url
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "entity_id": entity_id,
                "entity_type": entity_type,
            }

        return {
            "graph_create": _create_fallback,
            "graph_add_entity": _add_entity_fallback,
        }


_API = _load_graph_tools_api()


async def graph_create(driver_url: Optional[str] = None) -> Dict[str, Any]:
    """Initialize or connect to a knowledge graph backend."""
    result = _API["graph_create"](driver_url=driver_url)
    if hasattr(result, "__await__"):
        return await result
    return result


async def graph_add_entity(
    entity_id: str,
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add an entity node to a knowledge graph."""
    result = _API["graph_add_entity"](
        entity_id=entity_id,
        entity_type=entity_type,
        properties=properties,
        driver_url=driver_url,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_graph_tools(manager: Any) -> None:
    """Register native graph-tools category tools in unified manager."""
    manager.register_tool(
        category="graph_tools",
        name="graph_create",
        func=graph_create,
        description="Initialize a graph backend for graph operations.",
        input_schema={
            "type": "object",
            "properties": {
                "driver_url": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_add_entity",
        func=graph_add_entity,
        description="Add a typed entity to the graph.",
        input_schema={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "entity_type": {"type": "string"},
                "properties": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["entity_id", "entity_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )
