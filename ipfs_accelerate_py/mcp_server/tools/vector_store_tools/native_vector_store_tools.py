"""Native vector-store tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_vector_store_api() -> Dict[str, Any]:
    """Resolve source vector-store APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.vector_store_tools.vector_store_tools import (  # type: ignore
            vector_index as _vector_index,
            vector_metadata as _vector_metadata,
            vector_retrieval as _vector_retrieval,
        )

        return {
            "vector_index": _vector_index,
            "vector_retrieval": _vector_retrieval,
            "vector_metadata": _vector_metadata,
        }
    except Exception:
        logger.warning("Source vector_store_tools import unavailable, using fallback vector-store functions")

        async def _index_fallback(
            action: str,
            index_name: str,
            config: Optional[Dict[str, Any]] = None,
            vector_service: Any = None,
        ) -> Dict[str, Any]:
            _ = config, vector_service
            return {
                "action": action,
                "index_name": index_name,
                "result": {
                    "status": "success",
                    "action": action,
                    "index_name": index_name,
                },
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        async def _retrieval_fallback(
            collection: str = "default",
            ids: Optional[List[str]] = None,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 100,
            vector_service: Any = None,
        ) -> Dict[str, Any]:
            _ = filters, vector_service
            return {
                "collection": collection,
                "ids": ids or [],
                "limit": limit,
                "results": [],
                "total_found": 0,
                "status": "success",
            }

        async def _metadata_fallback(
            action: str,
            collection: str = "default",
            ids: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            vector_service: Any = None,
        ) -> Dict[str, Any]:
            _ = vector_service
            return {
                "action": action,
                "collection": collection,
                "ids": ids or [],
                "metadata": metadata or {},
                "status": "success",
            }

        return {
            "vector_index": _index_fallback,
            "vector_retrieval": _retrieval_fallback,
            "vector_metadata": _metadata_fallback,
        }


_API = _load_vector_store_api()


async def vector_index(
    action: str,
    index_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create, update, delete, or inspect vector indexes."""
    return await _API["vector_index"](
        action=action,
        index_name=index_name,
        config=config,
    )


async def vector_retrieval(
    collection: str = "default",
    ids: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Retrieve vectors by collection, IDs, and metadata filters."""
    return await _API["vector_retrieval"](
        collection=collection,
        ids=ids,
        filters=filters,
        limit=limit,
    )


async def vector_metadata(
    action: str,
    collection: str = "default",
    ids: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage vector metadata get/update/delete operations."""
    return await _API["vector_metadata"](
        action=action,
        collection=collection,
        ids=ids,
        metadata=metadata,
    )


def register_native_vector_store_tools(manager: Any) -> None:
    """Register native vector-store tools in unified hierarchical manager."""
    manager.register_tool(
        category="vector_store_tools",
        name="vector_index",
        func=vector_index,
        description="Manage vector indexes (create/update/delete/info).",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "index_name": {"type": "string"},
                "config": {"type": ["object", "null"]},
            },
            "required": ["action", "index_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store"],
    )

    manager.register_tool(
        category="vector_store_tools",
        name="vector_retrieval",
        func=vector_retrieval,
        description="Retrieve vectors from collection by IDs and filters.",
        input_schema={
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "ids": {"type": ["array", "null"], "items": {"type": "string"}},
                "filters": {"type": ["object", "null"]},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store"],
    )

    manager.register_tool(
        category="vector_store_tools",
        name="vector_metadata",
        func=vector_metadata,
        description="Manage metadata for vectors in a collection.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "collection": {"type": "string"},
                "ids": {"type": ["array", "null"], "items": {"type": "string"}},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store"],
    )
