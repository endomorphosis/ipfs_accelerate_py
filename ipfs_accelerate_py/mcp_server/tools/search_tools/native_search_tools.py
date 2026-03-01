"""Native search tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_search_api() -> Dict[str, Any]:
    """Resolve source search API functions with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.search.search_tools_api import (  # type: ignore
            faceted_search_from_parameters,
            semantic_search_from_parameters,
            similarity_search_from_parameters,
        )

        return {
            "semantic": semantic_search_from_parameters,
            "similarity": similarity_search_from_parameters,
            "faceted": faceted_search_from_parameters,
        }
    except Exception:
        async def _semantic_fallback(*, vector_service: Any, query: str, model: str = "sentence-transformers/all-MiniLM-L6-v2", top_k: int = 5, collection: str = "default", filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            _ = vector_service, filters
            results = [
                {
                    "id": f"doc_{i}",
                    "text": f"Mock result {i} for query: {query}",
                    "score": 0.9 - (i * 0.1),
                    "metadata": {"model": model, "collection": collection},
                }
                for i in range(min(top_k, 3))
            ]
            return {
                "query": query,
                "model": model,
                "top_k": top_k,
                "collection": collection,
                "results": results,
                "total_found": len(results),
            }

        async def _similarity_fallback(*, vector_service: Any, embedding: List[float], top_k: int = 10, threshold: float = 0.5, collection: str = "default") -> Dict[str, Any]:
            _ = vector_service
            results = [
                {
                    "id": f"embedding_{i}",
                    "similarity": 0.95 - (i * 0.05),
                    "metadata": {"collection": collection, "dimension": len(embedding)},
                }
                for i in range(min(top_k, 5))
                if 0.95 - (i * 0.05) >= threshold
            ]
            return {
                "embedding_dimension": len(embedding),
                "top_k": top_k,
                "threshold": threshold,
                "collection": collection,
                "results": results,
                "total_found": len(results),
            }

        async def _faceted_fallback(*, vector_service: Any, query: str = "", facets: Optional[Dict[str, List[str]]] = None, aggregations: Optional[List[str]] = None, top_k: int = 20, collection: str = "default") -> Dict[str, Any]:
            _ = vector_service
            facets = facets or {}
            aggregations = aggregations or []
            results = [
                {
                    "id": f"doc_{i}",
                    "text": f"Document {i} matching facets: {facets}",
                    "score": 0.8 - (i * 0.1),
                    "metadata": {
                        "category": f"category_{i % 3}",
                        "tags": [f"tag_{j}" for j in range(i % 2 + 1)],
                        "date": f"2024-01-{i + 1:02d}",
                    },
                }
                for i in range(min(top_k, 10))
            ]
            facet_counts = {
                "category": {"category_0": 4, "category_1": 3, "category_2": 3},
                "tags": {"tag_0": 8, "tag_1": 5},
            }
            return {
                "query": query,
                "facets": facets,
                "aggregations": aggregations,
                "top_k": top_k,
                "collection": collection,
                "results": results,
                "facet_counts": facet_counts,
                "total_found": len(results),
            }

        logger.warning("Source search_tools_api import unavailable, using fallback search functions")
        return {
            "semantic": _semantic_fallback,
            "similarity": _similarity_fallback,
            "faceted": _faceted_fallback,
        }


_API = _load_search_api()


async def semantic_search(
    query: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    collection: str = "default",
    filters: Optional[Dict[str, Any]] = None,
    vector_service: Any = None,
) -> Dict[str, Any]:
    """Perform semantic search on vector embeddings."""
    return await _API["semantic"](
        vector_service=vector_service,
        query=query,
        model=model,
        top_k=top_k,
        collection=collection,
        filters=filters or {},
    )


async def similarity_search(
    embedding: List[float],
    top_k: int = 10,
    threshold: float = 0.5,
    collection: str = "default",
    vector_service: Any = None,
) -> Dict[str, Any]:
    """Find similar embeddings based on a reference vector."""
    return await _API["similarity"](
        vector_service=vector_service,
        embedding=embedding,
        top_k=top_k,
        threshold=threshold,
        collection=collection,
    )


async def faceted_search(
    query: str = "",
    facets: Optional[Dict[str, List[str]]] = None,
    aggregations: Optional[List[str]] = None,
    top_k: int = 20,
    collection: str = "default",
    vector_service: Any = None,
) -> Dict[str, Any]:
    """Perform faceted search with metadata filtering."""
    return await _API["faceted"](
        vector_service=vector_service,
        query=query,
        facets=facets or {},
        aggregations=aggregations or [],
        top_k=top_k,
        collection=collection,
    )


def register_native_search_tools(manager: Any) -> None:
    """Register native search tools in unified hierarchical manager."""
    manager.register_tool(
        category="search_tools",
        name="semantic_search",
        func=semantic_search,
        description="Perform semantic search on vector embeddings.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "model": {"type": "string"},
                "top_k": {"type": "integer"},
                "collection": {"type": "string"},
                "filters": {"type": ["object", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "search"],
    )

    manager.register_tool(
        category="search_tools",
        name="similarity_search",
        func=similarity_search,
        description="Find similar embeddings from a reference vector.",
        input_schema={
            "type": "object",
            "properties": {
                "embedding": {"type": "array", "items": {"type": "number"}},
                "top_k": {"type": "integer"},
                "threshold": {"type": "number"},
                "collection": {"type": "string"},
            },
            "required": ["embedding"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "search"],
    )

    manager.register_tool(
        category="search_tools",
        name="faceted_search",
        func=faceted_search,
        description="Perform faceted search with metadata filtering.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "facets": {"type": ["object", "null"]},
                "aggregations": {"type": ["array", "null"], "items": {"type": "string"}},
                "top_k": {"type": "integer"},
                "collection": {"type": "string"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "search"],
    )
