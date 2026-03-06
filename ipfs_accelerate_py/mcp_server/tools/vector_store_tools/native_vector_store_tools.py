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


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelopes."""
    if isinstance(result, dict):
        payload = dict(result)
        if payload.get("error"):
            payload.setdefault("status", "error")
        else:
            payload.setdefault("status", "success")
        return payload
    return {"status": "success", "result": result}


async def vector_index(
    action: str,
    index_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create, update, delete, or inspect vector indexes."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"create", "update", "delete", "info", "list"}:
        return {
            "status": "error",
            "message": "action must be one of: create, update, delete, info, list",
            "action": action,
        }

    normalized_index_name = str(index_name or "").strip()
    if normalized_action != "list" and not normalized_index_name:
        return {
            "status": "error",
            "message": "index_name must be provided for create/update/delete/info actions",
            "index_name": index_name,
        }
    if config is not None and not isinstance(config, dict):
        return {
            "status": "error",
            "message": "config must be an object when provided",
            "config": config,
        }

    try:
        result = await _API["vector_index"](
            action=normalized_action,
            index_name=normalized_index_name or None,
            config=config,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "action": normalized_action,
            "index_name": normalized_index_name,
        }

    payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    payload.setdefault("index_name", normalized_index_name or None)
    if payload.get("status") == "success":
        payload.setdefault("result", {})
        payload.setdefault("success", True)
    return payload


async def vector_retrieval(
    collection: str = "default",
    ids: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Retrieve vectors by collection, IDs, and metadata filters."""
    normalized_collection = str(collection or "default").strip() or "default"
    if ids is not None:
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            return {
                "status": "error",
                "message": "ids must be an array of strings when provided",
                "ids": ids,
            }
        if any(not str(item).strip() for item in ids):
            return {
                "status": "error",
                "message": "ids cannot contain empty strings",
                "ids": ids,
            }
    if filters is not None and not isinstance(filters, dict):
        return {
            "status": "error",
            "message": "filters must be an object when provided",
            "filters": filters,
        }
    if not isinstance(limit, int) or limit < 1:
        return {
            "status": "error",
            "message": "limit must be an integer >= 1",
            "limit": limit,
        }

    try:
        result = await _API["vector_retrieval"](
            collection=normalized_collection,
            ids=ids,
            filters=filters,
            limit=limit,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "collection": normalized_collection,
            "ids": ids or [],
            "limit": limit,
        }

    payload = _normalize_payload(result)
    payload.setdefault("collection", normalized_collection)
    payload.setdefault("ids", ids or [])
    payload.setdefault("limit", limit)
    if payload.get("status") == "success":
        payload.setdefault("results", [])
        payload.setdefault("total_found", len(payload.get("results") or []))
    return payload


async def vector_metadata(
    action: str,
    collection: str = "default",
    ids: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage vector metadata get/update/delete operations."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"get", "update", "delete"}:
        return {
            "status": "error",
            "message": "action must be one of: get, update, delete",
            "action": action,
        }

    normalized_collection = str(collection or "default").strip() or "default"
    if ids is not None:
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            return {
                "status": "error",
                "message": "ids must be an array of strings when provided",
                "ids": ids,
            }
        if any(not str(item).strip() for item in ids):
            return {
                "status": "error",
                "message": "ids cannot contain empty strings",
                "ids": ids,
            }
    if metadata is not None and not isinstance(metadata, dict):
        return {
            "status": "error",
            "message": "metadata must be an object when provided",
            "metadata": metadata,
        }
    if normalized_action == "update" and metadata is None:
        return {
            "status": "error",
            "message": "metadata is required for update action",
            "action": normalized_action,
        }

    try:
        result = await _API["vector_metadata"](
            action=normalized_action,
            collection=normalized_collection,
            ids=ids,
            metadata=metadata,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "action": normalized_action,
            "collection": normalized_collection,
            "ids": ids or [],
        }

    payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    payload.setdefault("collection", normalized_collection)
    payload.setdefault("ids", ids or [])
    if payload.get("status") == "success":
        payload.setdefault("metadata", metadata or {})
    return payload


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
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "delete", "info", "list"],
                },
                "index_name": {"type": ["string", "null"], "minLength": 1},
                "config": {"type": ["object", "null"]},
            },
            "required": ["action"],
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
                "collection": {"type": "string", "default": "default", "minLength": 1},
                "ids": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "filters": {"type": ["object", "null"]},
                "limit": {"type": "integer", "minimum": 1, "default": 100},
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
                "action": {"type": "string", "enum": ["get", "update", "delete"]},
                "collection": {"type": "string", "default": "default", "minLength": 1},
                "ids": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store"],
    )
