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

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.vector_store_tools.enhanced_vector_store_tools import (  # type: ignore
                enhanced_vector_index as _enhanced_vector_index,
                enhanced_vector_search as _enhanced_vector_search,
                enhanced_vector_storage as _enhanced_vector_storage,
            )
        except Exception:
            _enhanced_vector_index = None
            _enhanced_vector_search = None
            _enhanced_vector_storage = None

        return {
            "vector_index": _vector_index,
            "vector_retrieval": _vector_retrieval,
            "vector_metadata": _vector_metadata,
            "enhanced_vector_index": _enhanced_vector_index,
            "enhanced_vector_search": _enhanced_vector_search,
            "enhanced_vector_storage": _enhanced_vector_storage,
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

        async def _enhanced_index_fallback(
            action: str,
            index_name: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            return {
                "action": action,
                "index_name": index_name,
                "result": {
                    "status": "success" if action != "list" else "success",
                    "index_name": index_name,
                    "config": config or {},
                    "indexes": [] if action == "list" else None,
                    "count": 0 if action == "list" else None,
                },
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        async def _enhanced_search_fallback(
            collection: str,
            query_vector: List[float],
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            score_threshold: Optional[float] = None,
            include_metadata: bool = True,
            include_vectors: bool = False,
            rerank: bool = False,
        ) -> Dict[str, Any]:
            _ = filters, score_threshold, include_metadata, include_vectors, rerank
            return {
                "collection": collection,
                "query_dimension": len(query_vector),
                "results": [],
                "total_results": 0,
                "top_k_requested": top_k,
                "status": "success",
            }

        async def _enhanced_storage_fallback(
            action: str,
            collection: Optional[str] = None,
            vectors: Optional[List[Any]] = None,
            vector_ids: Optional[List[str]] = None,
            vector_id: Optional[str] = None,
            metadata_updates: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            return {
                "action": action,
                "collection": collection or "default",
                "vector_id": vector_id,
                "result": {
                    "status": "success",
                    "vectors": [],
                    "count": len(vectors or []),
                    "ids": vector_ids or [],
                    "metadata": metadata_updates or {},
                },
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        return {
            "vector_index": _index_fallback,
            "vector_retrieval": _retrieval_fallback,
            "vector_metadata": _metadata_fallback,
            "enhanced_vector_index": _enhanced_index_fallback,
            "enhanced_vector_search": _enhanced_search_fallback,
            "enhanced_vector_storage": _enhanced_storage_fallback,
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


async def enhanced_vector_index(
    action: str,
    index_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Expose enhanced vector-index lifecycle operations from the source surface."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"create", "update", "delete", "info", "list"}:
        return {"status": "error", "message": "action must be one of: create, update, delete, info, list", "action": action}
    normalized_index_name = str(index_name or "").strip() or None
    if normalized_action != "list" and not normalized_index_name:
        return {"status": "error", "message": "index_name must be provided for create/update/delete/info actions", "index_name": index_name}
    if config is not None and not isinstance(config, dict):
        return {"status": "error", "message": "config must be an object when provided", "config": config}

    result = await _API["enhanced_vector_index"](
        action=normalized_action,
        index_name=normalized_index_name,
        config=config,
    )
    payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    payload.setdefault("index_name", normalized_index_name)
    payload.setdefault("result", {})
    return payload


async def enhanced_vector_search(
    collection: str,
    query_vector: List[float],
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None,
    include_metadata: bool = True,
    include_vectors: bool = False,
    rerank: bool = False,
) -> Dict[str, Any]:
    """Expose enhanced vector search with filtering and reranking controls."""
    normalized_collection = str(collection or "").strip()
    if not normalized_collection:
        return {"status": "error", "message": "collection must be a non-empty string", "collection": collection}
    if not isinstance(query_vector, list) or not query_vector or not all(isinstance(item, (int, float)) for item in query_vector):
        return {"status": "error", "message": "query_vector must be a non-empty list of numbers", "query_vector": query_vector}
    if not isinstance(top_k, int) or top_k < 1:
        return {"status": "error", "message": "top_k must be an integer >= 1", "top_k": top_k}
    if filters is not None and not isinstance(filters, dict):
        return {"status": "error", "message": "filters must be an object when provided", "filters": filters}
    if score_threshold is not None and not isinstance(score_threshold, (int, float)):
        return {"status": "error", "message": "score_threshold must be a number when provided", "score_threshold": score_threshold}
    for name, value in {
        "include_metadata": include_metadata,
        "include_vectors": include_vectors,
        "rerank": rerank,
    }.items():
        if not isinstance(value, bool):
            return {"status": "error", "message": f"{name} must be a boolean", name: value}

    result = await _API["enhanced_vector_search"](
        collection=normalized_collection,
        query_vector=[float(item) for item in query_vector],
        top_k=top_k,
        filters=filters,
        score_threshold=float(score_threshold) if isinstance(score_threshold, (int, float)) else None,
        include_metadata=include_metadata,
        include_vectors=include_vectors,
        rerank=rerank,
    )
    payload = _normalize_payload(result)
    payload.setdefault("collection", normalized_collection)
    payload.setdefault("query_dimension", len(query_vector))
    payload.setdefault("results", [])
    payload.setdefault("total_results", len(payload.get("results") or []))
    payload.setdefault("top_k_requested", top_k)
    return payload


async def enhanced_vector_storage(
    action: str,
    collection: Optional[str] = None,
    vectors: Optional[List[Any]] = None,
    vector_ids: Optional[List[str]] = None,
    vector_id: Optional[str] = None,
    metadata_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Expose enhanced vector storage lifecycle operations from the source surface."""
    normalized_action = str(action or "").strip().lower()
    valid_actions = {"add", "batch_add", "update", "delete", "get", "list", "get_metadata"}
    if normalized_action not in valid_actions:
        return {"status": "error", "message": "action must be one of: add, batch_add, update, delete, get, get_metadata, list", "action": action}
    normalized_collection = str(collection or "default").strip() or "default"
    if vectors is not None and not isinstance(vectors, list):
        return {"status": "error", "message": "vectors must be an array when provided", "vectors": vectors}
    if vector_ids is not None:
        if not isinstance(vector_ids, list) or not all(isinstance(item, str) and item.strip() for item in vector_ids):
            return {"status": "error", "message": "vector_ids must be a list of non-empty strings when provided", "vector_ids": vector_ids}
    if vector_id is not None and (not isinstance(vector_id, str) or not vector_id.strip()):
        return {"status": "error", "message": "vector_id must be a non-empty string when provided", "vector_id": vector_id}
    if metadata_updates is not None and not isinstance(metadata_updates, dict):
        return {"status": "error", "message": "metadata_updates must be an object when provided", "metadata_updates": metadata_updates}

    result = await _API["enhanced_vector_storage"](
        action=normalized_action,
        collection=normalized_collection,
        vectors=vectors,
        vector_ids=[item.strip() for item in (vector_ids or [])],
        vector_id=vector_id.strip() if isinstance(vector_id, str) else vector_id,
        metadata_updates=metadata_updates,
    )
    payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    payload.setdefault("collection", normalized_collection)
    if normalized_action in {"get", "list"}:
        payload.setdefault("vectors", [])
    if vector_id is not None and isinstance(vector_id, str) and vector_id.strip():
        payload.setdefault("vector_id", vector_id.strip())
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

    manager.register_tool(
        category="vector_store_tools",
        name="enhanced_vector_index",
        func=enhanced_vector_index,
        description="Manage enhanced vector index lifecycle operations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "update", "delete", "info", "list"]},
                "index_name": {"type": ["string", "null"], "minLength": 1},
                "config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store", "enhanced"],
    )

    manager.register_tool(
        category="vector_store_tools",
        name="enhanced_vector_search",
        func=enhanced_vector_search,
        description="Perform enhanced vector similarity search with richer filtering controls.",
        input_schema={
            "type": "object",
            "properties": {
                "collection": {"type": "string", "minLength": 1},
                "query_vector": {"type": "array", "minItems": 1, "items": {"type": "number"}},
                "top_k": {"type": "integer", "minimum": 1, "default": 10},
                "filters": {"type": ["object", "null"]},
                "score_threshold": {"type": ["number", "null"]},
                "include_metadata": {"type": "boolean", "default": True},
                "include_vectors": {"type": "boolean", "default": False},
                "rerank": {"type": "boolean", "default": False},
            },
            "required": ["collection", "query_vector"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store", "enhanced"],
    )

    manager.register_tool(
        category="vector_store_tools",
        name="enhanced_vector_storage",
        func=enhanced_vector_storage,
        description="Manage enhanced vector storage lifecycle operations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "batch_add", "update", "delete", "get", "list", "get_metadata"]},
                "collection": {"type": ["string", "null"], "minLength": 1, "default": "default"},
                "vectors": {"type": ["array", "null"]},
                "vector_ids": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "vector_id": {"type": ["string", "null"], "minLength": 1},
                "metadata_updates": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-store", "enhanced"],
    )
