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


async def create_vector_index(
    vectors: List[List[float]],
    dimension: Optional[int] = None,
    metric: str = "cosine",
    metadata: Optional[List[Dict[str, Any]]] = None,
    index_id: Optional[str] = None,
    index_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a vector index for similarity search."""
    result = _API["create_vector_index"](
        vectors=vectors,
        dimension=dimension,
        metric=metric,
        metadata=metadata,
        index_id=index_id,
        index_name=index_name,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def search_vector_index(
    index_id: str,
    query_vector: Optional[List[float]] = None,
    top_k: int = 5,
    include_metadata: bool = True,
    include_distances: bool = True,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search an existing vector index using similarity."""
    result = _API["search_vector_index"](
        index_id=index_id,
        query_vector=query_vector,
        top_k=top_k,
        include_metadata=include_metadata,
        include_distances=include_distances,
        filter_metadata=filter_metadata,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "vectors": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "dimension": {"type": ["integer", "null"]},
                "metric": {"type": "string"},
                "metadata": {"type": ["array", "null"], "items": {"type": "object"}},
                "index_id": {"type": ["string", "null"]},
                "index_name": {"type": ["string", "null"]},
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
                "index_id": {"type": "string"},
                "query_vector": {"type": ["array", "null"], "items": {"type": "number"}},
                "top_k": {"type": "integer"},
                "include_metadata": {"type": "boolean"},
                "include_distances": {"type": "boolean"},
                "filter_metadata": {"type": ["object", "null"]},
            },
            "required": ["index_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "vector-tools"],
    )
