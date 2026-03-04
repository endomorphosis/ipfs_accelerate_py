"""Native vector-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_vector_tools_api() -> Dict[str, Any]:
    """Resolve source vector-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.vector_tools import (  # type: ignore
            create_vector_index as _create_vector_index,
            search_vector_index as _search_vector_index,
        )

        return {
            "create_vector_index": _create_vector_index,
            "search_vector_index": _search_vector_index,
        }
    except Exception:
        logger.warning("Source vector_tools import unavailable, using fallback vector-tools functions")

        async def _create_fallback(
            vectors: List[List[float]],
            dimension: Optional[int] = None,
            metric: str = "cosine",
            metadata: Optional[List[Dict[str, Any]]] = None,
            index_id: Optional[str] = None,
            index_name: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = metadata, index_name
            inferred_dim = dimension
            if inferred_dim is None and vectors and vectors[0]:
                inferred_dim = len(vectors[0])
            return {
                "status": "success",
                "index_id": index_id or "fallback-index",
                "num_vectors": len(vectors),
                "dimension": inferred_dim or 0,
                "metric": metric,
                "vector_ids": list(range(len(vectors))),
            }

        async def _search_fallback(
            index_id: str,
            query_vector: Optional[List[float]] = None,
            top_k: int = 5,
            include_metadata: bool = True,
            include_distances: bool = True,
            filter_metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = include_metadata, include_distances, filter_metadata
            if query_vector is None:
                return {
                    "status": "error",
                    "error": "query_vector must be provided",
                    "index_id": index_id,
                }
            return {
                "status": "success",
                "index_id": index_id,
                "top_k": top_k,
                "num_results": 0,
                "results": [],
            }

        return {
            "create_vector_index": _create_fallback,
            "search_vector_index": _search_fallback,
        }


_API = _load_vector_tools_api()


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def _is_numeric_vector(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (int, float)) for item in value)


async def create_vector_index(
    vectors: List[List[float]],
    dimension: Optional[int] = None,
    metric: str = "cosine",
    metadata: Optional[List[Dict[str, Any]]] = None,
    index_id: Optional[str] = None,
    index_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a vector index for similarity search."""
    if not isinstance(vectors, list) or not vectors:
        return _error_result("vectors must be a non-empty array of numeric vectors", vectors=vectors)
    if not all(_is_numeric_vector(vector) for vector in vectors):
        return _error_result("vectors must contain only non-empty numeric vectors")

    inferred_dimension = len(vectors[0])
    if not all(len(vector) == inferred_dimension for vector in vectors):
        return _error_result("all vectors must have identical dimensions")

    normalized_dimension: Optional[int]
    if dimension is None:
        normalized_dimension = inferred_dimension
    else:
        try:
            normalized_dimension = int(dimension)
        except (TypeError, ValueError):
            return _error_result("dimension must be a positive integer when provided", dimension=dimension)
        if normalized_dimension <= 0:
            return _error_result("dimension must be a positive integer when provided", dimension=dimension)
        if normalized_dimension != inferred_dimension:
            return _error_result(
                "dimension must match vector length",
                dimension=normalized_dimension,
                inferred_dimension=inferred_dimension,
            )

    normalized_metric = str(metric or "").strip().lower()
    if not normalized_metric:
        return _error_result("metric must be a non-empty string", metric=metric)

    if metadata is not None and not isinstance(metadata, list):
        return _error_result("metadata must be an array when provided", metadata=metadata)

    try:
        result = _API["create_vector_index"](
            vectors=vectors,
            dimension=normalized_dimension,
            metric=normalized_metric,
            metadata=metadata,
            index_id=index_id,
            index_name=index_name,
        )
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"create_vector_index failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "success")
    return normalized


async def search_vector_index(
    index_id: str,
    query_vector: Optional[List[float]] = None,
    top_k: int = 5,
    include_metadata: bool = True,
    include_distances: bool = True,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search an existing vector index using similarity."""
    normalized_index_id = str(index_id or "").strip()
    if not normalized_index_id:
        return _error_result("index_id is required", index_id=index_id)

    if not _is_numeric_vector(query_vector):
        return _error_result("query_vector must be a non-empty list of numbers", query_vector=query_vector)

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be a positive integer", top_k=top_k)
    if normalized_top_k <= 0:
        return _error_result("top_k must be a positive integer", top_k=top_k)

    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean", include_metadata=include_metadata)
    if not isinstance(include_distances, bool):
        return _error_result("include_distances must be a boolean", include_distances=include_distances)
    if filter_metadata is not None and not isinstance(filter_metadata, dict):
        return _error_result("filter_metadata must be an object when provided", filter_metadata=filter_metadata)

    try:
        result = _API["search_vector_index"](
            index_id=normalized_index_id,
            query_vector=query_vector,
            top_k=normalized_top_k,
            include_metadata=include_metadata,
            include_distances=include_distances,
            filter_metadata=filter_metadata,
        )
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"search_vector_index failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "success")
    normalized.setdefault("index_id", normalized_index_id)
    return normalized


async def orchestrate_vector_search_storage(
    vectors: List[List[float]],
    query_vector: List[float],
    *,
    index_id: Optional[str] = None,
    metric: str = "cosine",
    top_k: int = 5,
    persist_audit: bool = False,
    audit_collection: str = "vector-search-audit",
) -> Dict[str, Any]:
    """Run a representative vector/search/storage integration flow.

    Flow:
    1) create vector index
    2) search that index
    3) optionally persist an audit record via storage tools
    """
    if not isinstance(vectors, list) or not vectors:
        return _error_result("vectors must be a non-empty list")
    if not _is_numeric_vector(query_vector):
        return _error_result("query_vector must be a non-empty list of numbers")
    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be a positive integer", top_k=top_k)
    if normalized_top_k <= 0:
        return _error_result("top_k must be a positive integer", top_k=top_k)
    if not isinstance(persist_audit, bool):
        return _error_result("persist_audit must be a boolean", persist_audit=persist_audit)

    created = await create_vector_index(
        vectors=vectors,
        dimension=len(vectors[0]) if vectors and vectors[0] else None,
        metric=metric,
        index_id=index_id,
    )
    if created.get("status") == "error":
        return created
    resolved_index_id = str(created.get("index_id") or index_id or "")

    searched = await search_vector_index(
        index_id=resolved_index_id,
        query_vector=query_vector,
        top_k=normalized_top_k,
        include_metadata=True,
        include_distances=True,
    )
    if searched.get("status") == "error":
        return searched

    from ipfs_accelerate_py.mcp_server.tools.search_tools.native_search_tools import similarity_search

    try:
        search_tools_result = await similarity_search(
            embedding=list(query_vector),
            top_k=normalized_top_k,
            threshold=0.0,
            collection=resolved_index_id or "default",
        )
    except Exception as exc:
        return _error_result(f"similarity_search integration failed: {exc}")

    search_results = searched.get("results") if isinstance(searched, dict) else []
    result_count = len(search_results or [])

    storage_receipt: Dict[str, Any] = {
        "stored": False,
        "collection": audit_collection,
    }
    if persist_audit:
        from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import store_data

        audit_payload = {
            "index_id": resolved_index_id,
            "metric": metric,
            "top_k": normalized_top_k,
            "result_count": result_count,
        }
        try:
            persisted = await store_data(
                data=audit_payload,
                storage_type="memory",
                compression="none",
                collection=str(audit_collection or "vector-search-audit"),
                metadata={"source": "orchestrate_vector_search_storage"},
                tags=["vector", "search", "storage", "integration"],
            )
        except Exception as exc:
            return _error_result(f"storage audit persistence failed: {exc}")
        storage_receipt = {
            "stored": bool(persisted.get("stored")),
            "collection": str(persisted.get("collection") or audit_collection),
            "item_id": persisted.get("item_id"),
        }

    return {
        "status": "success",
        "index_id": resolved_index_id,
        "metric": metric,
        "top_k": max(1, int(top_k)),
        "search": {
            "result_count": result_count,
            "results": search_results or [],
        },
        "search_tools_similarity": {
            "total_found": int(search_tools_result.get("total_found") or 0),
            "results": list(search_tools_result.get("results") or []),
            "collection": str(search_tools_result.get("collection") or (resolved_index_id or "default")),
        },
        "storage": storage_receipt,
    }


def register_native_vector_tools(manager: Any) -> None:
    """Register native vector-tools category tools in unified manager."""
    manager.register_tool(
        category="vector_tools",
        name="create_vector_index",
        func=create_vector_index,
        description="Create a vector index from raw vectors.",
        input_schema={
            "type": "object",
            "properties": {
                "vectors": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "number"},
                    },
                },
                "dimension": {"type": ["integer", "null"], "minimum": 1},
                "metric": {"type": "string", "minLength": 1, "default": "cosine"},
                "metadata": {"type": ["array", "null"], "items": {"type": "object"}},
                "index_id": {"type": ["string", "null"], "minLength": 1},
                "index_name": {"type": ["string", "null"], "minLength": 1},
            },
            "required": ["vectors"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="search_vector_index",
        func=search_vector_index,
        description="Search a vector index with a query vector.",
        input_schema={
            "type": "object",
            "properties": {
                "index_id": {"type": "string", "minLength": 1},
                "query_vector": {"type": "array", "minItems": 1, "items": {"type": "number"}},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "include_metadata": {"type": "boolean", "default": True},
                "include_distances": {"type": "boolean", "default": True},
                "filter_metadata": {"type": ["object", "null"]},
            },
            "required": ["index_id", "query_vector"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="orchestrate_vector_search_storage",
        func=orchestrate_vector_search_storage,
        description="Run representative vector index/search flow with optional storage audit persistence.",
        input_schema={
            "type": "object",
            "properties": {
                "vectors": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "number"},
                    },
                },
                "query_vector": {"type": "array", "minItems": 1, "items": {"type": "number"}},
                "index_id": {"type": ["string", "null"], "minLength": 1},
                "metric": {"type": "string", "minLength": 1, "default": "cosine"},
                "top_k": {"type": "integer", "minimum": 1},
                "persist_audit": {"type": "boolean", "default": False},
                "audit_collection": {"type": "string", "minLength": 1, "default": "vector-search-audit"},
            },
            "required": ["vectors", "query_vector"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools", "integration"],
    )
