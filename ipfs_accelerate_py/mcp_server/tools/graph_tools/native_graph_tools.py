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
            graph_add_relationship as _graph_add_relationship,
            graph_create as _graph_create,
            graph_query_cypher as _graph_query_cypher,
        )

        return {
            "graph_create": _graph_create,
            "graph_add_entity": _graph_add_entity,
            "graph_add_relationship": _graph_add_relationship,
            "graph_query_cypher": _graph_query_cypher,
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

        async def _add_relationship_fallback(
            source_id: str,
            target_id: str,
            relationship_type: str,
            properties: Optional[Dict[str, Any]] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = properties, driver_url
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
            }

        async def _query_cypher_fallback(
            query: str,
            parameters: Optional[Dict[str, Any]] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = parameters, driver_url
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "query": query,
                "results": [],
            }

        return {
            "graph_create": _create_fallback,
            "graph_add_entity": _add_entity_fallback,
            "graph_add_relationship": _add_relationship_fallback,
            "graph_query_cypher": _query_cypher_fallback,
        }


_API = _load_graph_tools_api()


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message, "message": message}
    payload.update(extra)
    return payload


async def _await_maybe(result: Any) -> Dict[str, Any]:
    """Await coroutine-like API results while supporting direct return values."""
    if hasattr(result, "__await__"):
        return await result
    return result


async def graph_create(driver_url: Optional[str] = None) -> Dict[str, Any]:
    """Initialize or connect to a knowledge graph backend."""
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(_API["graph_create"](driver_url=normalized_driver_url))
    except Exception as exc:
        return _error_result(f"graph_create failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_add_entity(
    entity_id: str,
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add an entity node to a knowledge graph."""
    normalized_entity_id = str(entity_id or "").strip()
    normalized_entity_type = str(entity_type or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()

    if not normalized_entity_id or not normalized_entity_type:
        return _error_result(
            "entity_id and entity_type must be provided",
            entity_id=entity_id,
            entity_type=entity_type,
        )
    if properties is not None and not isinstance(properties, dict):
        return _error_result("properties must be an object when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["graph_add_entity"](
                entity_id=normalized_entity_id,
                entity_type=normalized_entity_type,
                properties=properties,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_add_entity failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_add_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a typed relationship between two graph entities."""
    normalized_source = str(source_id or "").strip()
    normalized_target = str(target_id or "").strip()
    normalized_type = str(relationship_type or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()

    if not normalized_source or not normalized_target or not normalized_type:
        return _error_result(
            "source_id, target_id, and relationship_type must be provided",
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
            },
        )
    if properties is not None and not isinstance(properties, dict):
        return _error_result("properties must be an object when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["graph_add_relationship"](
                source_id=normalized_source,
                target_id=normalized_target,
                relationship_type=normalized_type,
                properties=properties,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_add_relationship failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_query_cypher(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a Cypher query against the configured graph backend."""
    normalized_query = str(query or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_query:
        return _error_result("query must be provided", query=query)
    if parameters is not None and not isinstance(parameters, dict):
        return _error_result("parameters must be an object when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["graph_query_cypher"](
                query=normalized_query,
                parameters=parameters,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_query_cypher failed: {exc}", query=normalized_query)

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


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
                "entity_id": {"type": "string", "minLength": 1},
                "entity_type": {"type": "string", "minLength": 1},
                "properties": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["entity_id", "entity_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_add_relationship",
        func=graph_add_relationship,
        description="Add a typed relationship between two entities in the graph.",
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1},
                "target_id": {"type": "string", "minLength": 1},
                "relationship_type": {"type": "string", "minLength": 1},
                "properties": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["source_id", "target_id", "relationship_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_query_cypher",
        func=graph_query_cypher,
        description="Execute a Cypher query against the graph backend.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "parameters": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )
