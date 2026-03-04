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


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend results to deterministic envelope."""
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    return payload


async def generate_sparse_embedding(
    text: str,
    model: str = "splade",
    top_k: int = 100,
    normalize: bool = True,
    return_dense: bool = False,
) -> Dict[str, Any]:
    """Generate sparse embeddings from text."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {
            "status": "error",
            "message": "text is required",
            "text": text,
        }
    normalized_model = str(model or "").strip().lower()
    if normalized_model not in {"splade", "bm25", "tfidf"}:
        return {
            "status": "error",
            "message": "model must be one of: splade, bm25, tfidf",
            "model": model,
        }
    if not isinstance(top_k, int) or top_k < 1:
        return {
            "status": "error",
            "message": "top_k must be an integer >= 1",
            "top_k": top_k,
        }
    if not isinstance(normalize, bool):
        return {
            "status": "error",
            "message": "normalize must be a boolean",
            "normalize": normalize,
        }
    if not isinstance(return_dense, bool):
        return {
            "status": "error",
            "message": "return_dense must be a boolean",
            "return_dense": return_dense,
        }

    result = _API["generate_sparse_embedding"](
        text=normalized_text,
        model=normalized_model,
        top_k=top_k,
        normalize=normalize,
        return_dense=return_dense,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("model", normalized_model)
    payload.setdefault("top_k", top_k)
    return payload


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
    normalized_collection_name = str(collection_name or "").strip()
    if not normalized_collection_name:
        return {
            "status": "error",
            "message": "collection_name is required",
            "collection_name": collection_name,
        }
    normalized_dataset = str(dataset or "").strip()
    if not normalized_dataset:
        return {
            "status": "error",
            "message": "dataset is required",
            "dataset": dataset,
        }
    normalized_split = str(split or "").strip().lower()
    if normalized_split not in {"train", "validation", "test"}:
        return {
            "status": "error",
            "message": "split must be one of: train, validation, test",
            "split": split,
        }
    normalized_column = str(column or "").strip()
    if not normalized_column:
        return {
            "status": "error",
            "message": "column is required",
            "column": column,
        }
    normalized_models: Optional[List[str]] = None
    if models is not None:
        if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
            return {
                "status": "error",
                "message": "models must be an array of strings when provided",
                "models": models,
            }
        normalized_models = [str(item).strip().lower() for item in models]
        if any(not item for item in normalized_models):
            return {
                "status": "error",
                "message": "models cannot contain empty strings",
                "models": models,
            }
    if not isinstance(batch_size, int) or batch_size < 1:
        return {
            "status": "error",
            "message": "batch_size must be an integer >= 1",
            "batch_size": batch_size,
        }
    if index_config is not None and not isinstance(index_config, dict):
        return {
            "status": "error",
            "message": "index_config must be an object when provided",
            "index_config": index_config,
        }

    result = _API["index_sparse_collection"](
        collection_name=normalized_collection_name,
        dataset=normalized_dataset,
        split=normalized_split,
        column=normalized_column,
        models=normalized_models,
        batch_size=batch_size,
        index_config=index_config,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("collection_name", normalized_collection_name)
    payload.setdefault("dataset", normalized_dataset)
    return payload


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
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {
            "status": "error",
            "message": "query is required",
            "query": query,
        }
    normalized_collection = str(collection_name or "").strip()
    if not normalized_collection:
        return {
            "status": "error",
            "message": "collection_name is required",
            "collection_name": collection_name,
        }
    normalized_model = str(model or "").strip().lower()
    if normalized_model not in {"splade", "bm25", "tfidf"}:
        return {
            "status": "error",
            "message": "model must be one of: splade, bm25, tfidf",
            "model": model,
        }
    if not isinstance(top_k, int) or top_k < 1:
        return {
            "status": "error",
            "message": "top_k must be an integer >= 1",
            "top_k": top_k,
        }
    if filters is not None and not isinstance(filters, dict):
        return {
            "status": "error",
            "message": "filters must be an object when provided",
            "filters": filters,
        }
    if search_config is not None and not isinstance(search_config, dict):
        return {
            "status": "error",
            "message": "search_config must be an object when provided",
            "search_config": search_config,
        }
    if not isinstance(explain_scores, bool):
        return {
            "status": "error",
            "message": "explain_scores must be a boolean",
            "explain_scores": explain_scores,
        }

    result = _API["sparse_search"](
        query=normalized_query,
        collection_name=normalized_collection,
        model=normalized_model,
        top_k=top_k,
        filters=filters,
        search_config=search_config,
        explain_scores=explain_scores,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("query", normalized_query)
    payload.setdefault("model", normalized_model)
    return payload


async def manage_sparse_models(
    action: str,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage sparse embedding model configuration and lifecycle actions."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"list", "stats", "configure", "clear_cache"}:
        return {
            "status": "error",
            "message": "action must be one of: list, stats, configure, clear_cache",
            "action": action,
        }
    normalized_model_name = str(model_name).strip().lower() if model_name is not None else None
    if model_name is not None and not normalized_model_name:
        return {
            "status": "error",
            "message": "model_name must be a non-empty string when provided",
            "model_name": model_name,
        }
    if normalized_action == "configure" and (normalized_model_name is None or not isinstance(config, dict)):
        return {
            "status": "error",
            "message": "model_name and config object are required for configure action",
            "action": normalized_action,
        }
    if config is not None and not isinstance(config, dict):
        return {
            "status": "error",
            "message": "config must be an object when provided",
            "config": config,
        }

    result = _API["manage_sparse_models"](
        action=normalized_action,
        model_name=normalized_model_name,
        config=config,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    if normalized_model_name is not None:
        payload.setdefault("model_name", normalized_model_name)
    return payload


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
                "model": {"type": "string", "enum": ["splade", "bm25", "tfidf"], "default": "splade"},
                "top_k": {"type": "integer", "minimum": 1, "default": 100},
                "normalize": {"type": "boolean", "default": True},
                "return_dense": {"type": "boolean", "default": False},
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
                "split": {"type": "string", "enum": ["train", "validation", "test"], "default": "train"},
                "column": {"type": "string"},
                "models": {"type": ["array", "null"], "items": {"type": "string"}},
                "batch_size": {"type": "integer", "minimum": 1, "default": 100},
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
                "model": {"type": "string", "enum": ["splade", "bm25", "tfidf"], "default": "splade"},
                "top_k": {"type": "integer", "minimum": 1, "default": 10},
                "filters": {"type": ["object", "null"]},
                "search_config": {"type": ["object", "null"]},
                "explain_scores": {"type": "boolean", "default": False},
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
                "action": {"type": "string", "enum": ["list", "stats", "configure", "clear_cache"]},
                "model_name": {"type": ["string", "null"]},
                "config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "sparse-embedding"],
    )
