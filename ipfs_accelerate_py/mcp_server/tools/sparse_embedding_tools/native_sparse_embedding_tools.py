"""Native sparse-embedding tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_sparse_api() -> Dict[str, Any]:
    """Resolve source sparse-embedding APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.sparse_embedding_tools.sparse_embedding_tools import (  # type: ignore
            generate_sparse_embedding as _generate_sparse_embedding,
            index_sparse_collection as _index_sparse_collection,
            manage_sparse_models as _manage_sparse_models,
            sparse_search as _sparse_search,
        )

        return {
            "generate_sparse_embedding": _generate_sparse_embedding,
            "index_sparse_collection": _index_sparse_collection,
            "sparse_search": _sparse_search,
            "manage_sparse_models": _manage_sparse_models,
        }
    except Exception:
        logger.warning(
            "Source sparse_embedding_tools import unavailable, using fallback sparse-embedding functions"
        )

        async def _generate_fallback(
            text: str,
            model: str = "splade",
            top_k: int = 100,
            normalize: bool = True,
            return_dense: bool = False,
        ) -> Dict[str, Any]:
            _ = normalize, return_dense
            return {
                "text": text,
                "model": model,
                "sparse_embedding": {
                    "indices": [0, 1],
                    "values": [1.0, 0.5],
                    "dimension": top_k,
                    "sparsity": 1.0,
                    "num_nonzero": 2,
                },
                "generated_at": "fallback",
            }

        async def _index_fallback(
            collection_name: str,
            dataset: str,
            split: str = "train",
            column: str = "text",
            models: Optional[List[str]] = None,
            batch_size: int = 100,
            index_config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = split, column, models, batch_size, index_config
            return {
                "collection_name": collection_name,
                "dataset": dataset,
                "total_documents": 0,
                "results": {},
            }

        async def _search_fallback(
            query: str,
            collection_name: str,
            model: str = "splade",
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            search_config: Optional[Dict[str, Any]] = None,
            explain_scores: bool = False,
        ) -> Dict[str, Any]:
            _ = model, top_k, filters, search_config, explain_scores
            return {
                "query": query,
                "results": [],
                "metadata": {"collection": collection_name},
                "total_found": 0,
                "has_more": False,
            }

        async def _manage_fallback(
            action: str,
            model_name: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = config
            return {
                "status": "success",
                "action": action,
                "model_name": model_name,
            }

        return {
            "generate_sparse_embedding": _generate_fallback,
            "index_sparse_collection": _index_fallback,
            "sparse_search": _search_fallback,
            "manage_sparse_models": _manage_fallback,
        }


_API = _load_sparse_api()


async def generate_sparse_embedding(
    text: str,
    model: str = "splade",
    top_k: int = 100,
    normalize: bool = True,
    return_dense: bool = False,
) -> Dict[str, Any]:
    """Generate sparse embeddings from text."""
    result = _API["generate_sparse_embedding"](
        text=text,
        model=model,
        top_k=top_k,
        normalize=normalize,
        return_dense=return_dense,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def index_sparse_collection(
    collection_name: str,
    dataset: str,
    split: str = "train",
    column: str = "text",
    models: Optional[List[str]] = None,
    batch_size: int = 100,
    index_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Index sparse embeddings for a dataset collection."""
    result = _API["index_sparse_collection"](
        collection_name=collection_name,
        dataset=dataset,
        split=split,
        column=column,
        models=models,
        batch_size=batch_size,
        index_config=index_config,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def sparse_search(
    query: str,
    collection_name: str,
    model: str = "splade",
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    search_config: Optional[Dict[str, Any]] = None,
    explain_scores: bool = False,
) -> Dict[str, Any]:
    """Run sparse vector search over an indexed sparse collection."""
    result = _API["sparse_search"](
        query=query,
        collection_name=collection_name,
        model=model,
        top_k=top_k,
        filters=filters,
        search_config=search_config,
        explain_scores=explain_scores,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def manage_sparse_models(
    action: str,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage sparse embedding model configuration and lifecycle actions."""
    result = _API["manage_sparse_models"](
        action=action,
        model_name=model_name,
        config=config,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_sparse_embedding_tools(manager: Any) -> None:
    """Register native sparse-embedding tools in unified hierarchical manager."""
    manager.register_tool(
        category="sparse_embedding_tools",
        name="generate_sparse_embedding",
        func=generate_sparse_embedding,
        description="Generate sparse embedding vectors from input text.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "model": {"type": "string"},
                "top_k": {"type": "integer"},
                "normalize": {"type": "boolean"},
                "return_dense": {"type": "boolean"},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "sparse-embedding"],
    )

    manager.register_tool(
        category="sparse_embedding_tools",
        name="index_sparse_collection",
        func=index_sparse_collection,
        description="Index sparse embeddings for a dataset collection.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string"},
                "dataset": {"type": "string"},
                "split": {"type": "string"},
                "column": {"type": "string"},
                "models": {"type": ["array", "null"], "items": {"type": "string"}},
                "batch_size": {"type": "integer"},
                "index_config": {"type": ["object", "null"]},
            },
            "required": ["collection_name", "dataset"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "sparse-embedding"],
    )

    manager.register_tool(
        category="sparse_embedding_tools",
        name="sparse_search",
        func=sparse_search,
        description="Search documents using sparse-vector retrieval.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "collection_name": {"type": "string"},
                "model": {"type": "string"},
                "top_k": {"type": "integer"},
                "filters": {"type": ["object", "null"]},
                "search_config": {"type": ["object", "null"]},
                "explain_scores": {"type": "boolean"},
            },
            "required": ["query", "collection_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "sparse-embedding"],
    )

    manager.register_tool(
        category="sparse_embedding_tools",
        name="manage_sparse_models",
        func=manage_sparse_models,
        description="Manage sparse model catalog and configuration actions.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "model_name": {"type": ["string", "null"]},
                "config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "sparse-embedding"],
    )
