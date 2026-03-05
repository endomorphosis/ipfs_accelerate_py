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


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "message": message}
    payload.update(extra)
    return payload


async def semantic_search(
    query: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    collection: str = "default",
    filters: Optional[Dict[str, Any]] = None,
    vector_service: Any = None,
) -> Dict[str, Any]:
    """Perform semantic search on vector embeddings."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query is required", query=query)

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be a positive integer", top_k=top_k)
    if normalized_top_k <= 0:
        return _error_result("top_k must be a positive integer", top_k=top_k)

    if filters is not None and not isinstance(filters, dict):
        return _error_result("filters must be an object when provided", filters=filters)

    try:
        result = await _API["semantic"](
            vector_service=vector_service,
            query=normalized_query,
            model=str(model or "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=normalized_top_k,
            collection=str(collection or "default"),
            filters=filters or {},
        )
    except Exception as exc:
        return _error_result("semantic search failed", error=str(exc))
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("query", normalized_query)
    return payload


async def similarity_search(
    embedding: List[float],
    top_k: int = 10,
    threshold: float = 0.5,
    collection: str = "default",
    vector_service: Any = None,
) -> Dict[str, Any]:
    """Find similar embeddings based on a reference vector."""
    if not isinstance(embedding, list) or not embedding:
        return _error_result("embedding must be a non-empty list of numbers", embedding=embedding)
    if not all(isinstance(x, (int, float)) for x in embedding):
        return _error_result("embedding must contain only numbers", embedding=embedding)

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be a positive integer", top_k=top_k)
    if normalized_top_k <= 0:
        return _error_result("top_k must be a positive integer", top_k=top_k)

    try:
        normalized_threshold = float(threshold)
    except (TypeError, ValueError):
        return _error_result("threshold must be between 0.0 and 1.0", threshold=threshold)
    if normalized_threshold < 0.0 or normalized_threshold > 1.0:
        return _error_result("threshold must be between 0.0 and 1.0", threshold=threshold)

    try:
        result = await _API["similarity"](
            vector_service=vector_service,
            embedding=embedding,
            top_k=normalized_top_k,
            threshold=normalized_threshold,
            collection=str(collection or "default"),
        )
    except Exception as exc:
        return _error_result("similarity search failed", error=str(exc))
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("embedding_dimension", len(embedding))
    return payload


async def faceted_search(
    query: str = "",
    facets: Optional[Dict[str, List[str]]] = None,
    aggregations: Optional[List[str]] = None,
    top_k: int = 20,
    collection: str = "default",
    vector_service: Any = None,
) -> Dict[str, Any]:
    """Perform faceted search with metadata filtering."""
    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be a positive integer", top_k=top_k)
    if normalized_top_k <= 0:
        return _error_result("top_k must be a positive integer", top_k=top_k)

    normalized_facets: Dict[str, List[str]] = {}
    if facets is not None:
        if not isinstance(facets, dict):
            return _error_result("facets must be an object when provided", facets=facets)
        for facet_name, facet_values in facets.items():
            if not isinstance(facet_name, str) or not facet_name.strip():
                return _error_result(
                    "facets keys must be non-empty strings",
                    facets=facets,
                )
            if not isinstance(facet_values, list) or not all(
                isinstance(value, str) and value.strip() for value in facet_values
            ):
                return _error_result(
                    "each facets value must be an array of non-empty strings",
                    facets=facets,
                )
            normalized_facets[facet_name.strip()] = [value.strip() for value in facet_values]

    normalized_aggregations: List[str] = []
    if aggregations is not None:
        if not isinstance(aggregations, list):
            return _error_result("aggregations must be an array when provided", aggregations=aggregations)
        if not all(isinstance(item, str) and item.strip() for item in aggregations):
            return _error_result(
                "aggregations must contain only non-empty strings",
                aggregations=aggregations,
            )
        normalized_aggregations = [item.strip() for item in aggregations]

    try:
        result = await _API["faceted"](
            vector_service=vector_service,
            query=str(query or ""),
            facets=normalized_facets,
            aggregations=normalized_aggregations,
            top_k=normalized_top_k,
            collection=str(collection or "default"),
        )
    except Exception as exc:
        return _error_result("faceted search failed", error=str(exc))
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


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
                "query": {"type": "string", "minLength": 1},
                "model": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "collection": {"type": "string", "minLength": 1, "default": "default"},
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
                "embedding": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                "top_k": {"type": "integer", "minimum": 1, "default": 10},
                "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                "collection": {"type": "string", "minLength": 1, "default": "default"},
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
                "query": {"type": "string", "default": ""},
                "facets": {
                    "type": ["object", "null"],
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "aggregations": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                },
                "top_k": {"type": "integer", "minimum": 1, "default": 20},
                "collection": {"type": "string", "minLength": 1, "default": "default"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "search"],
    )
