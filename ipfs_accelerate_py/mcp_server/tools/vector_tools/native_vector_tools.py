"""Native vector-tools category implementations for unified mcp_server."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_VECTOR_BACKENDS = {"all", "faiss", "qdrant", "elasticsearch"}
_VALID_STORE_ACTIONS = {"create", "index", "query", "delete"}
_VALID_OPTIMIZATION_TYPES = {"index", "memory", "disk"}


def _load_vector_tools_api() -> Dict[str, Any]:
    """Resolve source vector-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.vector_tools import (  # type: ignore
            create_vector_index as _create_vector_index,
            search_vector_index as _search_vector_index,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.vector_tools.vector_store_management import (  # type: ignore
            delete_vector_index as _delete_vector_index,
            list_vector_indexes as _list_vector_indexes,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.vector_tools.vector_stores import (  # type: ignore
            manage_vector_store as _manage_vector_store,
            optimize_vector_store as _optimize_vector_store,
        )

        return {
            "create_vector_index": _create_vector_index,
            "search_vector_index": _search_vector_index,
            "list_vector_indexes": _list_vector_indexes,
            "delete_vector_index": _delete_vector_index,
            "manage_vector_store": _manage_vector_store,
            "optimize_vector_store": _optimize_vector_store,
        }
    except Exception:
        logger.warning("Source vector_tools import unavailable, using fallback vector-tools functions")

        fallback_stores: Dict[str, Dict[str, Any]] = {}

        def _store_key(backend: str, name: str) -> str:
            return f"{backend}:{name}"

        def _ensure_store(
            backend: str,
            name: str,
            persist_path: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            key = _store_key(backend, name)
            store = fallback_stores.get(key)
            now = datetime.now().isoformat()
            if store is None:
                store = {
                    "store_name": name,
                    "backend": backend,
                    "persist_path": persist_path,
                    "metadata": dict(metadata or {}),
                    "vectors": [],
                    "created_at": now,
                    "updated_at": now,
                }
                fallback_stores[key] = store
            else:
                if persist_path:
                    store["persist_path"] = persist_path
                if metadata:
                    merged = dict(store.get("metadata") or {})
                    merged.update(metadata)
                    store["metadata"] = merged
                store["updated_at"] = now
            return store

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

        async def _list_indexes_fallback(backend: str = "all") -> Dict[str, Any]:
            if backend == "all":
                indexes = {
                    name: [store["store_name"] for store in fallback_stores.values() if store["backend"] == name]
                    for name in sorted(_VALID_VECTOR_BACKENDS - {"all"})
                }
            else:
                indexes = {
                    backend: [
                        store["store_name"]
                        for store in fallback_stores.values()
                        if store["backend"] == backend
                    ]
                }
            return {
                "status": "success",
                "backend": backend,
                "indexes": indexes,
                "count": sum(len(items) for items in indexes.values()),
            }

        async def _delete_index_fallback(
            index_name: str,
            backend: str = "faiss",
            config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = config
            deleted = fallback_stores.pop(_store_key(backend, index_name), None) is not None
            return {
                "status": "success" if deleted else "error",
                "index_name": index_name,
                "backend": backend,
                "deleted": deleted,
                "message": (
                    f"Deleted {backend} vector index '{index_name}'"
                    if deleted
                    else f"Vector index '{index_name}' not found"
                ),
            }

        async def _manage_store_fallback(
            operation: str,
            store_type: str = "faiss",
            collection_name: str = "default",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            normalized_operation = str(operation or "").strip().lower()
            store = _ensure_store(
                store_type,
                collection_name,
                persist_path=kwargs.get("persist_path"),
                metadata=kwargs.get("store_metadata") if isinstance(kwargs.get("store_metadata"), dict) else None,
            )
            if normalized_operation == "create":
                return {
                    "status": "success",
                    "operation": "create",
                    "store_type": store_type,
                    "collection_name": collection_name,
                    "message": f"Created {store_type} vector store for collection '{collection_name}'",
                }
            if normalized_operation == "index":
                vectors = list(kwargs.get("vectors") or [])
                ids = list(kwargs.get("ids") or [])
                metadata = list(kwargs.get("metadata") or [])
                for idx, vector in enumerate(vectors):
                    vector_id = ids[idx] if idx < len(ids) else f"{collection_name}-{len(store['vectors']) + idx}"
                    vector_metadata = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
                    store["vectors"].append({"id": vector_id, "vector": list(vector), "metadata": vector_metadata})
                store["updated_at"] = datetime.now().isoformat()
                return {
                    "status": "success",
                    "operation": "index",
                    "store_type": store_type,
                    "collection_name": collection_name,
                    "indexed_count": len(vectors),
                    "message": f"Indexed {len(vectors)} vectors in {store_type}",
                }
            if normalized_operation == "query":
                query_vector = kwargs.get("query_vector") or []
                top_k = int(kwargs.get("top_k", 5) or 5)
                results = []
                for stored in store["vectors"][:max(top_k, 0)]:
                    results.append(
                        {
                            "id": stored.get("id"),
                            "score": 1.0,
                            "metadata": stored.get("metadata", {}),
                        }
                    )
                return {
                    "status": "success",
                    "operation": "query",
                    "store_type": store_type,
                    "collection_name": collection_name,
                    "results": results,
                    "results_count": len(results),
                    "query_dimension": len(query_vector),
                    "message": f"Query executed on {store_type} store",
                }
            if normalized_operation == "delete":
                ids = list(kwargs.get("ids") or [])
                if ids:
                    before = len(store["vectors"])
                    store["vectors"] = [item for item in store["vectors"] if item.get("id") not in set(ids)]
                    deleted_count = before - len(store["vectors"])
                    store["updated_at"] = datetime.now().isoformat()
                    return {
                        "status": "success",
                        "operation": "delete",
                        "store_type": store_type,
                        "collection_name": collection_name,
                        "deleted_count": deleted_count,
                        "message": f"Deleted {deleted_count} vectors from {store_type}",
                    }
                fallback_stores.pop(_store_key(store_type, collection_name), None)
                return {
                    "status": "success",
                    "operation": "delete",
                    "store_type": store_type,
                    "collection_name": collection_name,
                    "message": f"Cleared {store_type} vector store collection '{collection_name}'",
                }
            return {
                "status": "error",
                "message": f"Unknown operation: {normalized_operation}. Valid operations: create, index, query, delete",
            }

        async def _optimize_store_fallback(
            store_type: str = "faiss",
            collection_name: str = "default",
            optimization_type: str = "index",
        ) -> Dict[str, Any]:
            _ensure_store(store_type, collection_name)
            return {
                "status": "success",
                "store_type": store_type,
                "collection_name": collection_name,
                "optimization_type": optimization_type,
                "message": f"Optimized {store_type} store ({optimization_type})",
            }

        return {
            "create_vector_index": _create_fallback,
            "search_vector_index": _search_fallback,
            "list_vector_indexes": _list_indexes_fallback,
            "delete_vector_index": _delete_index_fallback,
            "manage_vector_store": _manage_store_fallback,
            "optimize_vector_store": _optimize_store_fallback,
            "store_registry": fallback_stores,
        }


_API = _load_vector_tools_api()


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def _is_numeric_vector(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (int, float)) for item in value)


def _normalize_backend(value: Any, *, allow_all: bool = False) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    allowed = _VALID_VECTOR_BACKENDS if allow_all else (_VALID_VECTOR_BACKENDS - {"all"})
    if normalized not in allowed:
        return None
    return normalized


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
    resolved_index_id = str(created.get("index_id") or index_id or "").strip() or "vector-index"

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


async def list_vector_indexes(backend: str = "all") -> Dict[str, Any]:
    """List available vector indexes grouped by backend."""
    normalized_backend = _normalize_backend(backend, allow_all=True)
    if normalized_backend is None:
        return _error_result("backend must be one of: all, faiss, qdrant, elasticsearch", backend=backend)

    try:
        result = _API["list_vector_indexes"](backend=normalized_backend)
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"list_vector_indexes failed: {exc}", backend=normalized_backend)

    normalized = dict(payload or {})
    normalized.setdefault("status", "success")
    normalized.setdefault("backend", normalized_backend)
    normalized.setdefault("indexes", {})
    return normalized


async def delete_vector_index(
    index_name: str,
    backend: str = "faiss",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Delete a named vector index from the selected backend."""
    normalized_index_name = str(index_name or "").strip()
    if not normalized_index_name:
        return _error_result("index_name must be a non-empty string", index_name=index_name)

    normalized_backend = _normalize_backend(backend)
    if normalized_backend is None:
        return _error_result("backend must be one of: faiss, qdrant, elasticsearch", backend=backend)
    if config is not None and not isinstance(config, dict):
        return _error_result("config must be an object when provided", config=config)

    try:
        result = _API["delete_vector_index"](
            index_name=normalized_index_name,
            backend=normalized_backend,
            config=config,
        )
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(
            f"delete_vector_index failed: {exc}",
            index_name=normalized_index_name,
            backend=normalized_backend,
        )

    normalized = dict(payload or {})
    normalized.setdefault("status", "success")
    normalized.setdefault("index_name", normalized_index_name)
    normalized.setdefault("backend", normalized_backend)
    return normalized


async def manage_vector_store(
    operation: str,
    store_type: str = "faiss",
    collection_name: str = "default",
    vectors: Optional[List[List[float]]] = None,
    ids: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    query_vector: Optional[List[float]] = None,
    top_k: int = 5,
    persist_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create, index, query, or delete a vector store collection."""
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation not in _VALID_STORE_ACTIONS:
        return _error_result(
            "operation must be one of: create, index, query, delete",
            operation=operation,
        )

    normalized_backend = _normalize_backend(store_type)
    if normalized_backend is None:
        return _error_result(
            "store_type must be one of: faiss, qdrant, elasticsearch",
            store_type=store_type,
        )

    normalized_collection = str(collection_name or "").strip()
    if not normalized_collection:
        return _error_result("collection_name must be a non-empty string", collection_name=collection_name)

    if normalized_operation == "index":
        if not isinstance(vectors, list) or not vectors or not all(_is_numeric_vector(vector) for vector in vectors):
            return _error_result("vectors must be a non-empty array of numeric vectors", vectors=vectors)
        if ids is not None:
            if not isinstance(ids, list) or len(ids) != len(vectors):
                return _error_result("ids must be an array with the same length as vectors", ids=ids)
            if any(not isinstance(item, str) or not item.strip() for item in ids):
                return _error_result("ids must contain only non-empty strings", ids=ids)
        if metadata is not None:
            if not isinstance(metadata, list) or len(metadata) != len(vectors):
                return _error_result("metadata must be an array with the same length as vectors", metadata=metadata)
            if any(not isinstance(item, dict) for item in metadata):
                return _error_result("metadata entries must be objects", metadata=metadata)
    if normalized_operation == "query":
        if not _is_numeric_vector(query_vector):
            return _error_result("query_vector must be a non-empty list of numbers", query_vector=query_vector)
        try:
            normalized_top_k = int(top_k)
        except (TypeError, ValueError):
            return _error_result("top_k must be a positive integer", top_k=top_k)
        if normalized_top_k <= 0:
            return _error_result("top_k must be a positive integer", top_k=top_k)
    else:
        normalized_top_k = int(top_k) if isinstance(top_k, int) else 5

    if persist_path is not None and (not isinstance(persist_path, str) or not persist_path.strip()):
        return _error_result("persist_path must be a non-empty string when provided", persist_path=persist_path)

    try:
        result = _API["manage_vector_store"](
            operation=normalized_operation,
            store_type=normalized_backend,
            collection_name=normalized_collection,
            vectors=vectors,
            ids=[item.strip() for item in (ids or [])] or None,
            metadata=metadata,
            query_vector=query_vector,
            top_k=normalized_top_k,
            persist_path=persist_path.strip() if isinstance(persist_path, str) else None,
        )
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(
            f"manage_vector_store failed: {exc}",
            operation=normalized_operation,
            store_type=normalized_backend,
            collection_name=normalized_collection,
        )

    normalized = dict(payload or {})
    normalized.setdefault("status", "success")
    normalized.setdefault("operation", normalized_operation)
    normalized.setdefault("store_type", normalized_backend)
    normalized.setdefault("collection_name", normalized_collection)
    return normalized


async def optimize_vector_store(
    store_type: str = "faiss",
    collection_name: str = "default",
    optimization_type: str = "index",
) -> Dict[str, Any]:
    """Optimize an existing vector store collection."""
    normalized_backend = _normalize_backend(store_type)
    if normalized_backend is None:
        return _error_result("store_type must be one of: faiss, qdrant, elasticsearch", store_type=store_type)
    normalized_collection = str(collection_name or "").strip()
    if not normalized_collection:
        return _error_result("collection_name must be a non-empty string", collection_name=collection_name)
    normalized_optimization = str(optimization_type or "").strip().lower()
    if normalized_optimization not in _VALID_OPTIMIZATION_TYPES:
        return _error_result(
            "optimization_type must be one of: index, memory, disk",
            optimization_type=optimization_type,
        )

    try:
        result = _API["optimize_vector_store"](
            store_type=normalized_backend,
            collection_name=normalized_collection,
            optimization_type=normalized_optimization,
        )
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(
            f"optimize_vector_store failed: {exc}",
            store_type=normalized_backend,
            collection_name=normalized_collection,
        )

    normalized = dict(payload or {})
    normalized.setdefault("status", "success")
    normalized.setdefault("store_type", normalized_backend)
    normalized.setdefault("collection_name", normalized_collection)
    normalized.setdefault("optimization_type", normalized_optimization)
    return normalized


async def create_store(
    name: str,
    backend: str = "faiss",
    persist_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Documented alias to create a named vector store."""
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result("metadata must be an object when provided", metadata=metadata)
    result = await manage_vector_store(
        operation="create",
        store_type=backend,
        collection_name=name,
        persist_path=persist_path,
    )
    if result.get("status") == "error":
        return result
    return {
        "status": "success",
        "store_name": str(name or "").strip(),
        "backend": result.get("store_type", backend),
        "persist_path": persist_path,
        "metadata": dict(metadata or {}),
        "created": True,
    }


async def load_store(
    name: str,
    backend: str = "faiss",
    persist_path: Optional[str] = None,
    create_if_missing: bool = False,
) -> Dict[str, Any]:
    """Documented alias to load a named vector store."""
    if not isinstance(create_if_missing, bool):
        return _error_result("create_if_missing must be a boolean", create_if_missing=create_if_missing)
    listed = await list_stores(backend=backend, include_details=True)
    if listed.get("status") == "error":
        return listed
    stores = listed.get("stores") or []
    match = next((store for store in stores if store.get("store_name") == str(name or "").strip()), None)
    if match is None and create_if_missing:
        created = await create_store(name=name, backend=backend, persist_path=persist_path)
        if created.get("status") == "error":
            return created
        match = {
            "store_name": created.get("store_name"),
            "backend": created.get("backend"),
            "persist_path": created.get("persist_path"),
            "metadata": created.get("metadata", {}),
            "vector_count": 0,
        }
    if match is None:
        return _error_result("store not found", store_name=name, backend=backend)
    return {
        "status": "success",
        "store_name": match.get("store_name"),
        "backend": match.get("backend"),
        "persist_path": match.get("persist_path", persist_path),
        "metadata": match.get("metadata", {}),
        "loaded": True,
    }


async def save_store(
    store_name: str,
    backend: str = "faiss",
    destination_path: Optional[str] = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Documented alias to persist vector store metadata."""
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean", include_metadata=include_metadata)
    info = await get_vector_store_info(store_name=store_name, backend=backend)
    if info.get("status") == "error":
        return info
    return {
        "status": "success",
        "store_name": info.get("store_name"),
        "backend": info.get("backend"),
        "destination_path": destination_path or info.get("persist_path"),
        "saved": True,
        "metadata": info.get("metadata", {}) if include_metadata else {},
    }


async def list_stores(
    backend: str = "all",
    include_details: bool = False,
) -> Dict[str, Any]:
    """Documented alias to list named vector stores."""
    if not isinstance(include_details, bool):
        return _error_result("include_details must be a boolean", include_details=include_details)
    indexes = await list_vector_indexes(backend=backend)
    if indexes.get("status") == "error":
        return indexes
    raw_indexes = indexes.get("indexes") if isinstance(indexes.get("indexes"), dict) else {}
    registry = _API.get("store_registry") if isinstance(_API.get("store_registry"), dict) else {}
    stores: List[Dict[str, Any]] = []
    for backend_name, names in raw_indexes.items():
        if not isinstance(names, list):
            continue
        for name in names:
            registry_entry = registry.get(f"{backend_name}:{name}", {}) if isinstance(registry, dict) else {}
            entry = {
                "store_name": name,
                "backend": backend_name,
            }
            if include_details:
                entry.update(
                    {
                        "vector_count": len(registry_entry.get("vectors", [])) if isinstance(registry_entry, dict) else 0,
                        "persist_path": registry_entry.get("persist_path") if isinstance(registry_entry, dict) else None,
                        "metadata": registry_entry.get("metadata", {}) if isinstance(registry_entry, dict) else {},
                    }
                )
            stores.append(entry)
    return {
        "status": "success",
        "backend": indexes.get("backend", backend),
        "stores": stores,
        "count": len(stores),
    }


async def get_vector_store_info(
    store_name: str,
    backend: str = "faiss",
) -> Dict[str, Any]:
    """Documented alias to fetch basic vector store metadata."""
    normalized_name = str(store_name or "").strip()
    if not normalized_name:
        return _error_result("store_name must be a non-empty string", store_name=store_name)
    listed = await list_stores(backend=backend, include_details=True)
    if listed.get("status") == "error":
        return listed
    for store in listed.get("stores") or []:
        if store.get("store_name") == normalized_name:
            return {
                "status": "success",
                "store_name": normalized_name,
                "backend": store.get("backend", backend),
                "vector_count": int(store.get("vector_count") or 0),
                "persist_path": store.get("persist_path"),
                "metadata": store.get("metadata", {}),
            }
    return _error_result("store not found", store_name=normalized_name, backend=backend)


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

    manager.register_tool(
        category="vector_tools",
        name="list_vector_indexes",
        func=list_vector_indexes,
        description="List vector indexes grouped by backend.",
        input_schema={
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS),
                    "default": "all",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="delete_vector_index",
        func=delete_vector_index,
        description="Delete a named vector index from a backend.",
        input_schema={
            "type": "object",
            "properties": {
                "index_name": {"type": "string", "minLength": 1},
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
                "config": {"type": ["object", "null"]},
            },
            "required": ["index_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="manage_vector_store",
        func=manage_vector_store,
        description="Create, index, query, or delete vectors in a named store.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": sorted(_VALID_STORE_ACTIONS)},
                "store_type": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
                "collection_name": {"type": "string", "minLength": 1, "default": "default"},
                "vectors": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "number"},
                    },
                },
                "ids": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                },
                "metadata": {"type": ["array", "null"], "items": {"type": "object"}},
                "query_vector": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "persist_path": {"type": ["string", "null"], "minLength": 1},
            },
            "required": ["operation"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="optimize_vector_store",
        func=optimize_vector_store,
        description="Optimize vector store performance for a named collection.",
        input_schema={
            "type": "object",
            "properties": {
                "store_type": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
                "collection_name": {"type": "string", "minLength": 1, "default": "default"},
                "optimization_type": {
                    "type": "string",
                    "enum": sorted(_VALID_OPTIMIZATION_TYPES),
                    "default": "index",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="create_store",
        func=create_store,
        description="Create a named vector store.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
                "persist_path": {"type": ["string", "null"], "minLength": 1},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="load_store",
        func=load_store,
        description="Load a named vector store, optionally creating it if missing.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
                "persist_path": {"type": ["string", "null"], "minLength": 1},
                "create_if_missing": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="save_store",
        func=save_store,
        description="Persist named vector store metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "store_name": {"type": "string", "minLength": 1},
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
                "destination_path": {"type": ["string", "null"], "minLength": 1},
                "include_metadata": {"type": "boolean", "default": True},
            },
            "required": ["store_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="list_stores",
        func=list_stores,
        description="List named vector stores.",
        input_schema={
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS),
                    "default": "all",
                },
                "include_details": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )

    manager.register_tool(
        category="vector_tools",
        name="get_vector_store_info",
        func=get_vector_store_info,
        description="Get basic metadata for a named vector store.",
        input_schema={
            "type": "object",
            "properties": {
                "store_name": {"type": "string", "minLength": 1},
                "backend": {
                    "type": "string",
                    "enum": sorted(_VALID_VECTOR_BACKENDS - {"all"}),
                    "default": "faiss",
                },
            },
            "required": ["store_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )
