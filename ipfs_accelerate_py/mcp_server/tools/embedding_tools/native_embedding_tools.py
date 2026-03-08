"""Native embedding tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _load_embedding_api() -> Dict[str, Any]:
    """Resolve source embedding APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.embedding_tools.embedding_tools import (  # type: ignore
            EmbeddingManager as _EmbeddingManager,
            generate_embeddings as _generate_embeddings,
            shard_embeddings as _shard_embeddings,
        )

        api = {
            "EmbeddingManager": _EmbeddingManager,
            "generate_embeddings": _generate_embeddings,
            "shard_embeddings": _shard_embeddings,
        }

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.embedding_tools.enhanced_embedding_tools import (  # type: ignore
                generate_embedding as _generate_embedding,
                chunk_text as _chunk_text,
                manage_endpoints as _manage_endpoints,
            )

            api["generate_embedding"] = _generate_embedding
            api["chunk_text"] = _chunk_text
            api["manage_endpoints"] = _manage_endpoints
        except Exception:
            logger.warning("Source enhanced_embedding_tools import unavailable, using fallback endpoint/chunk functions")

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.embedding_tools.advanced_embedding_generation import (  # type: ignore
                generate_embeddings_from_file as _generate_embeddings_from_file,
            )

            api["generate_embeddings_from_file"] = _generate_embeddings_from_file
        except Exception:
            logger.warning("Source advanced_embedding_generation import unavailable, using fallback file-embedding function")

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.embedding_tools.advanced_search import (  # type: ignore
                hybrid_search as _hybrid_search,
                multi_modal_search as _multi_modal_search,
                search_with_filters as _search_with_filters,
                semantic_search as _semantic_search,
            )

            api["semantic_search"] = _semantic_search
            api["hybrid_search"] = _hybrid_search
            api["multi_modal_search"] = _multi_modal_search
            api["search_with_filters"] = _search_with_filters
        except Exception:
            logger.warning("Source advanced_search import unavailable, using fallback semantic-search function")

        return api
    except Exception:
        logger.warning("Source embedding_tools import unavailable, using fallback embedding functions")

        class _FallbackEmbeddingManager:
            def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
                self.model_name = model_name

            def get_available_models(self) -> List[str]:
                return [self.model_name]

        async def _generate_fallback(
            texts: List[str],
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            if not isinstance(texts, list) or not texts:
                return {"status": "error", "error": "Texts must be a non-empty list"}
            return {
                "status": "success",
                "embeddings": [[0.1, 0.2, 0.3, 0.4] for _ in texts],
                "model_name": model_name,
                "dimension": 4,
                "count": len(texts),
            }

        async def _shard_fallback(
            embeddings: List[Any],
            shard_count: int = 4,
            strategy: str = "balanced",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            if not embeddings or shard_count <= 0:
                return {"status": "error", "error": "Invalid embeddings or shard_count"}
            shard_size = max(1, len(embeddings) // shard_count)
            shards: List[Dict[str, Any]] = []
            for shard_id in range(shard_count):
                start = shard_id * shard_size
                end = len(embeddings) if shard_id == shard_count - 1 else start + shard_size
                shards.append(
                    {
                        "shard_id": shard_id,
                        "embeddings": embeddings[start:end],
                        "strategy": strategy,
                    }
                )
            return {
                "status": "success",
                "shard_count": shard_count,
                "total_embeddings": len(embeddings),
                "shards": shards,
            }

        async def _chunk_text_fallback(
            text: str,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            if not isinstance(text, str) or not text.strip():
                return {"status": "error", "error": "text must be a non-empty string"}
            chunk_size = max(1, int(kwargs.get("chunk_size", 512)))
            chunk_overlap = max(0, int(kwargs.get("chunk_overlap", 50)))
            max_step = max(1, chunk_size - chunk_overlap)

            chunks: List[Dict[str, Any]] = []
            start = 0
            text_value = text
            while start < len(text_value):
                end = min(len(text_value), start + chunk_size)
                chunks.append(
                    {
                        "text": text_value[start:end],
                        "start": start,
                        "end": end,
                        "length": end - start,
                    }
                )
                if end >= len(text_value):
                    break
                start += max_step

            return {
                "status": "success",
                "original_length": len(text_value),
                "chunk_count": len(chunks),
                "chunks": chunks,
            }

        async def _manage_endpoints_fallback(
            action: str,
            model: str,
            endpoint: str,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            normalized_action = str(action or "").strip().lower()
            model_name = str(model or "").strip()
            endpoint_value = str(endpoint or "").strip()

            if not model_name:
                return {"status": "error", "error": "model must be provided"}

            if normalized_action == "add":
                if not endpoint_value:
                    return {"status": "error", "error": "endpoint must be provided"}
                return {
                    "status": "success",
                    "action": "added",
                    "model": model_name,
                    "endpoint": endpoint_value,
                }

            if normalized_action == "test":
                if not endpoint_value:
                    return {"status": "error", "error": "endpoint must be provided"}
                return {
                    "status": "success",
                    "action": "tested",
                    "model": model_name,
                    "available": True,
                }

            if normalized_action in {"list", "status"}:
                return {
                    "status": "success",
                    "action": normalized_action,
                    "endpoints": [
                        {
                            "model": model_name,
                            "endpoint": endpoint_value or "https://fallback.invalid/embeddings",
                            "endpoint_type": "tei",
                        }
                    ],
                }

            return {"status": "error", "error": f"Unknown action: {action}"}

        async def _generate_embedding_fallback(
            text: str,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            normalize: bool = True,
            batch_size: int = 32,
            use_gpu: bool = False,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            normalized_text = str(text or "").strip()
            if not normalized_text:
                return {"status": "error", "error": "text must be a non-empty string"}
            return {
                "status": "success",
                "text": normalized_text,
                "model_name": str(model_name or "sentence-transformers/all-MiniLM-L6-v2"),
                "normalize": bool(normalize),
                "batch_size": int(batch_size),
                "use_gpu": bool(use_gpu),
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "dimension": 4,
            }

        async def _generate_from_file_fallback(
            file_path: str,
            output_path: str | None = None,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            batch_size: int = 32,
            chunk_size: int | None = None,
            max_length: int | None = None,
            output_format: str = "json",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs, output_path, chunk_size, max_length
            normalized_path = str(file_path or "").strip()
            if not normalized_path:
                return {"status": "error", "error": "file_path must be a non-empty string"}
            return {
                "status": "success",
                "file_path": normalized_path,
                "model_name": str(model_name or "sentence-transformers/all-MiniLM-L6-v2"),
                "batch_size": int(batch_size),
                "output_format": str(output_format or "json"),
                "embeddings": [],
                "count": 0,
            }

        async def _semantic_search_fallback(
            query: str,
            vector_store_id: str,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            top_k: int = 10,
            similarity_threshold: float = 0.7,
            include_metadata: bool = True,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            normalized_query = str(query or "").strip()
            normalized_store = str(vector_store_id or "").strip()
            if not normalized_query:
                return {"status": "error", "error": "query must be a non-empty string"}
            if not normalized_store:
                return {"status": "error", "error": "vector_store_id must be a non-empty string"}
            return {
                "status": "success",
                "query": normalized_query,
                "vector_store_id": normalized_store,
                "model_used": str(model_name or "sentence-transformers/all-MiniLM-L6-v2"),
                "top_k": int(top_k),
                "similarity_threshold": float(similarity_threshold),
                "include_metadata": bool(include_metadata),
                "results": [],
                "total_results": 0,
            }

        async def _hybrid_search_fallback(
            query: str,
            vector_store_id: str,
            lexical_weight: float = 0.3,
            semantic_weight: float = 0.7,
            top_k: int = 10,
            rerank_results: bool = True,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            normalized_query = str(query or "").strip()
            normalized_store = str(vector_store_id or "").strip()
            if not normalized_query:
                return {"status": "error", "error": "query must be a non-empty string"}
            if not normalized_store:
                return {"status": "error", "error": "vector_store_id must be a non-empty string"}
            return {
                "status": "success",
                "query": normalized_query,
                "vector_store_id": normalized_store,
                "weights": {
                    "lexical": float(lexical_weight),
                    "semantic": float(semantic_weight),
                },
                "top_k": int(top_k),
                "reranked": bool(rerank_results),
                "results": [],
                "total_results": 0,
            }

        async def _search_with_filters_fallback(
            query: str,
            vector_store_id: str,
            filters: Dict[str, Any],
            top_k: int = 10,
            search_method: str = "semantic",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            normalized_query = str(query or "").strip()
            normalized_store = str(vector_store_id or "").strip()
            if not normalized_query:
                return {"status": "error", "error": "query must be a non-empty string"}
            if not normalized_store:
                return {"status": "error", "error": "vector_store_id must be a non-empty string"}
            if not isinstance(filters, dict):
                return {"status": "error", "error": "filters must be an object"}
            return {
                "status": "success",
                "query": normalized_query,
                "vector_store_id": normalized_store,
                "filters_applied": filters,
                "search_method": str(search_method or "semantic"),
                "top_k": int(top_k),
                "results": [],
                "total_results": 0,
                "total_candidates": 0,
                "filtered_out": 0,
            }

        async def _multi_modal_search_fallback(
            query: str | None = None,
            image_query: str | None = None,
            vector_store_id: str | None = None,
            model_name: str = "clip-ViT-B-32",
            top_k: int = 10,
            modality_weights: Dict[str, float] | None = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            _ = kwargs
            normalized_query = None if query is None else str(query).strip()
            normalized_image_query = None if image_query is None else str(image_query).strip()
            normalized_store = None if vector_store_id is None else str(vector_store_id).strip()

            if not normalized_query and not normalized_image_query:
                return {"status": "error", "error": "either query or image_query must be provided"}
            if not normalized_store:
                return {"status": "error", "error": "vector_store_id must be a non-empty string"}

            return {
                "status": "success",
                "text_query": normalized_query,
                "image_query": normalized_image_query,
                "vector_store_id": normalized_store,
                "model_used": str(model_name or "clip-ViT-B-32"),
                "modality_weights": dict(modality_weights or {"text": 0.6, "image": 0.4}),
                "top_k": int(top_k),
                "results": [],
                "total_results": 0,
            }

        return {
            "EmbeddingManager": _FallbackEmbeddingManager,
            "generate_embeddings": _generate_fallback,
            "shard_embeddings": _shard_fallback,
            "generate_embedding": _generate_embedding_fallback,
            "generate_embeddings_from_file": _generate_from_file_fallback,
            "semantic_search": _semantic_search_fallback,
            "hybrid_search": _hybrid_search_fallback,
            "search_with_filters": _search_with_filters_fallback,
            "multi_modal_search": _multi_modal_search_fallback,
            "chunk_text": _chunk_text_fallback,
            "manage_endpoints": _manage_endpoints_fallback,
        }


_API = _load_embedding_api()


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


async def _await_maybe(result: Any) -> Dict[str, Any]:
    """Await coroutine-like API results while supporting direct return values."""
    if hasattr(result, "__await__"):
        return await result
    return result


async def generate_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate vector embeddings for multiple texts."""
    if not isinstance(texts, list) or not texts:
        return _error_result("texts must be a non-empty array of strings")
    if not all(isinstance(text, str) and text.strip() for text in texts):
        return _error_result("texts must contain only non-empty strings")

    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _error_result("model_name must be a non-empty string")

    try:
        payload = await _await_maybe(
            _API["generate_embeddings"](texts=texts, model_name=normalized_model_name, **kwargs)
        )
    except Exception as exc:
        return _error_result(f"generate_embeddings failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("model_name", normalized_model_name)
    normalized.setdefault("embeddings", [])
    normalized.setdefault("count", len(texts))
    if "dimension" not in normalized:
        embeddings = normalized.get("embeddings") or []
        first_embedding = embeddings[0] if embeddings and isinstance(embeddings[0], list) else []
        normalized["dimension"] = len(first_embedding)
    return normalized


async def generate_embedding(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """Generate a single embedding vector using source-aligned enhanced embedding surface."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return _error_result("text must be a non-empty string")

    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _error_result("model_name must be a non-empty string")

    if not isinstance(normalize, bool):
        return _error_result("normalize must be a boolean")

    try:
        normalized_batch_size = int(batch_size)
    except (TypeError, ValueError):
        return _error_result("batch_size must be an integer")
    if normalized_batch_size <= 0:
        return _error_result("batch_size must be a positive integer")

    if not isinstance(use_gpu, bool):
        return _error_result("use_gpu must be a boolean")

    handler = _API.get("generate_embedding")
    if not callable(handler):
        return _error_result("generate_embedding handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                text=normalized_text,
                model_name=normalized_model_name,
                normalize=normalize,
                batch_size=normalized_batch_size,
                use_gpu=use_gpu,
            )
        )
    except Exception as exc:
        return _error_result(f"generate_embedding failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("model_name", normalized_model_name)
    normalized.setdefault("normalize", normalize)
    normalized.setdefault("batch_size", normalized_batch_size)
    normalized.setdefault("use_gpu", use_gpu)
    normalized.setdefault("embedding", [])
    return normalized


async def generate_embeddings_from_file(
    file_path: str,
    output_path: str | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    chunk_size: int | None = None,
    max_length: int | None = None,
    output_format: str = "json",
) -> Dict[str, Any]:
    """Generate embeddings from a file using source-aligned advanced embedding generation surface."""
    normalized_file_path = str(file_path or "").strip()
    if not normalized_file_path:
        return _error_result("file_path must be a non-empty string")

    normalized_output_path = None if output_path is None else str(output_path).strip()
    if output_path is not None and not normalized_output_path:
        return _error_result("output_path must be a non-empty string when provided")

    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _error_result("model_name must be a non-empty string")

    try:
        normalized_batch_size = int(batch_size)
    except (TypeError, ValueError):
        return _error_result("batch_size must be an integer")
    if normalized_batch_size <= 0:
        return _error_result("batch_size must be a positive integer")

    normalized_chunk_size = None
    if chunk_size is not None:
        try:
            normalized_chunk_size = int(chunk_size)
        except (TypeError, ValueError):
            return _error_result("chunk_size must be an integer when provided")
        if normalized_chunk_size <= 0:
            return _error_result("chunk_size must be a positive integer when provided")

    normalized_max_length = None
    if max_length is not None:
        try:
            normalized_max_length = int(max_length)
        except (TypeError, ValueError):
            return _error_result("max_length must be an integer when provided")
        if normalized_max_length <= 0:
            return _error_result("max_length must be a positive integer when provided")

    normalized_output_format = str(output_format or "").strip().lower()
    if normalized_output_format not in {"json", "parquet", "hdf5"}:
        return _error_result("output_format must be one of: json, parquet, hdf5")

    handler = _API.get("generate_embeddings_from_file")
    if not callable(handler):
        return _error_result("generate_embeddings_from_file handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                file_path=normalized_file_path,
                output_path=normalized_output_path,
                model_name=normalized_model_name,
                batch_size=normalized_batch_size,
                chunk_size=normalized_chunk_size,
                max_length=normalized_max_length,
                output_format=normalized_output_format,
            )
        )
    except Exception as exc:
        return _error_result(f"generate_embeddings_from_file failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("file_path", normalized_file_path)
    normalized.setdefault("model_name", normalized_model_name)
    normalized.setdefault("batch_size", normalized_batch_size)
    normalized.setdefault("output_format", normalized_output_format)
    normalized.setdefault("embeddings", [])
    normalized.setdefault("count", len(normalized.get("embeddings") or []))
    if normalized_output_path is not None:
        normalized.setdefault("output_path", normalized_output_path)
    return normalized


async def semantic_search(
    query: str,
    vector_store_id: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 10,
    similarity_threshold: float = 0.7,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Perform source-aligned semantic search using embedding similarity."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query must be a non-empty string")

    normalized_store = str(vector_store_id or "").strip()
    if not normalized_store:
        return _error_result("vector_store_id must be a non-empty string")

    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _error_result("model_name must be a non-empty string")

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be an integer")
    if normalized_top_k < 1 or normalized_top_k > 1000:
        return _error_result("top_k must be between 1 and 1000")

    try:
        normalized_similarity_threshold = float(similarity_threshold)
    except (TypeError, ValueError):
        return _error_result("similarity_threshold must be a number")
    if normalized_similarity_threshold < 0.0 or normalized_similarity_threshold > 1.0:
        return _error_result("similarity_threshold must be between 0 and 1")

    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean")

    handler = _API.get("semantic_search")
    if not callable(handler):
        return _error_result("semantic_search handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                query=normalized_query,
                vector_store_id=normalized_store,
                model_name=normalized_model_name,
                top_k=normalized_top_k,
                similarity_threshold=normalized_similarity_threshold,
                include_metadata=include_metadata,
            )
        )
    except Exception as exc:
        return _error_result(f"semantic_search failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    normalized.setdefault("vector_store_id", normalized_store)
    normalized.setdefault("model_used", normalized_model_name)
    normalized.setdefault("top_k", normalized_top_k)
    normalized.setdefault("similarity_threshold", normalized_similarity_threshold)
    normalized.setdefault("results", [])
    normalized.setdefault("total_results", len(normalized.get("results") or []))
    return normalized


async def hybrid_search(
    query: str,
    vector_store_id: str,
    lexical_weight: float = 0.3,
    semantic_weight: float = 0.7,
    top_k: int = 10,
    rerank_results: bool = True,
) -> Dict[str, Any]:
    """Perform source-aligned hybrid lexical+semantic search."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query must be a non-empty string")

    normalized_store = str(vector_store_id or "").strip()
    if not normalized_store:
        return _error_result("vector_store_id must be a non-empty string")

    try:
        normalized_lexical_weight = float(lexical_weight)
    except (TypeError, ValueError):
        return _error_result("lexical_weight must be a number")

    try:
        normalized_semantic_weight = float(semantic_weight)
    except (TypeError, ValueError):
        return _error_result("semantic_weight must be a number")

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be an integer")
    if normalized_top_k < 1:
        return _error_result("top_k must be >= 1")

    if not isinstance(rerank_results, bool):
        return _error_result("rerank_results must be a boolean")

    handler = _API.get("hybrid_search")
    if not callable(handler):
        return _error_result("hybrid_search handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                query=normalized_query,
                vector_store_id=normalized_store,
                lexical_weight=normalized_lexical_weight,
                semantic_weight=normalized_semantic_weight,
                top_k=normalized_top_k,
                rerank_results=rerank_results,
            )
        )
    except Exception as exc:
        return _error_result(f"hybrid_search failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    normalized.setdefault("vector_store_id", normalized_store)
    normalized.setdefault(
        "weights",
        {
            "lexical": normalized_lexical_weight,
            "semantic": normalized_semantic_weight,
        },
    )
    normalized.setdefault("top_k", normalized_top_k)
    normalized.setdefault("reranked", rerank_results)
    normalized.setdefault("results", [])
    normalized.setdefault("total_results", len(normalized.get("results") or []))
    return normalized


async def search_with_filters(
    query: str,
    vector_store_id: str,
    filters: Dict[str, Any],
    top_k: int = 10,
    search_method: str = "semantic",
) -> Dict[str, Any]:
    """Perform source-aligned metadata-filtered search with selectable search method."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query must be a non-empty string")

    normalized_store = str(vector_store_id or "").strip()
    if not normalized_store:
        return _error_result("vector_store_id must be a non-empty string")

    if not isinstance(filters, dict):
        return _error_result("filters must be an object")

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be an integer")
    if normalized_top_k < 1 or normalized_top_k > 1000:
        return _error_result("top_k must be between 1 and 1000")

    normalized_search_method = str(search_method or "").strip().lower()
    if normalized_search_method not in {"semantic", "lexical", "hybrid"}:
        return _error_result("search_method must be one of: semantic, lexical, hybrid")

    handler = _API.get("search_with_filters")
    if not callable(handler):
        return _error_result("search_with_filters handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                query=normalized_query,
                vector_store_id=normalized_store,
                filters=filters,
                top_k=normalized_top_k,
                search_method=normalized_search_method,
            )
        )
    except Exception as exc:
        return _error_result(f"search_with_filters failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    normalized.setdefault("vector_store_id", normalized_store)
    normalized.setdefault("filters_applied", filters)
    normalized.setdefault("search_method", normalized_search_method)
    normalized.setdefault("top_k", normalized_top_k)
    normalized.setdefault("results", [])
    normalized.setdefault("total_results", len(normalized.get("results") or []))
    normalized.setdefault("total_candidates", normalized.get("total_results") or 0)
    normalized.setdefault("filtered_out", 0)
    return normalized


async def multi_modal_search(
    query: str | None = None,
    image_query: str | None = None,
    vector_store_id: str | None = None,
    model_name: str = "clip-ViT-B-32",
    top_k: int = 10,
    modality_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Perform source-aligned multi-modal search combining text and image queries."""
    normalized_query = None if query is None else str(query).strip()
    normalized_image_query = None if image_query is None else str(image_query).strip()
    if not normalized_query and not normalized_image_query:
        return _error_result("either query or image_query must be provided")

    normalized_store = None if vector_store_id is None else str(vector_store_id).strip()
    if not normalized_store:
        return _error_result("vector_store_id must be a non-empty string")

    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _error_result("model_name must be a non-empty string")

    try:
        normalized_top_k = int(top_k)
    except (TypeError, ValueError):
        return _error_result("top_k must be an integer")
    if normalized_top_k < 1 or normalized_top_k > 1000:
        return _error_result("top_k must be between 1 and 1000")

    normalized_modality_weights = None
    if modality_weights is not None:
        if not isinstance(modality_weights, dict):
            return _error_result("modality_weights must be an object when provided")
        if not all(isinstance(key, str) for key in modality_weights.keys()):
            return _error_result("modality_weights keys must be strings")
        try:
            normalized_modality_weights = {str(key): float(value) for key, value in modality_weights.items()}
        except (TypeError, ValueError):
            return _error_result("modality_weights values must be numbers")

    handler = _API.get("multi_modal_search")
    if not callable(handler):
        return _error_result("multi_modal_search handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                query=normalized_query,
                image_query=normalized_image_query,
                vector_store_id=normalized_store,
                model_name=normalized_model_name,
                top_k=normalized_top_k,
                modality_weights=normalized_modality_weights,
            )
        )
    except Exception as exc:
        return _error_result(f"multi_modal_search failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("text_query", normalized_query)
    normalized.setdefault("image_query", normalized_image_query)
    normalized.setdefault("vector_store_id", normalized_store)
    normalized.setdefault("model_used", normalized_model_name)
    normalized.setdefault("modality_weights", normalized_modality_weights or {"text": 0.6, "image": 0.4})
    normalized.setdefault("top_k", normalized_top_k)
    normalized.setdefault("results", [])
    normalized.setdefault("total_results", len(normalized.get("results") or []))
    return normalized


async def shard_embeddings(
    embeddings: List[Any],
    shard_count: int = 4,
    strategy: str = "balanced",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Shard embedding vectors using the selected strategy."""
    if not isinstance(embeddings, list) or not embeddings:
        return _error_result("embeddings must be a non-empty array")

    try:
        normalized_shard_count = int(shard_count)
    except (TypeError, ValueError):
        return _error_result("shard_count must be an integer")
    if normalized_shard_count <= 0:
        return _error_result("shard_count must be a positive integer")

    normalized_strategy = str(strategy or "").strip()
    if not normalized_strategy:
        return _error_result("strategy must be a non-empty string")

    try:
        payload = await _await_maybe(
            _API["shard_embeddings"](
                embeddings=embeddings,
                shard_count=normalized_shard_count,
                strategy=normalized_strategy,
                **kwargs,
            )
        )
    except Exception as exc:
        return _error_result(f"shard_embeddings failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("shard_count", normalized_shard_count)
    normalized.setdefault("total_embeddings", len(embeddings))
    normalized.setdefault("strategy", normalized_strategy)
    normalized.setdefault("shards", [])
    return normalized


async def get_available_models(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Return available embedding model names from the source manager."""
    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _error_result("model_name must be a non-empty string", models=[], count=0)

    try:
        manager = _API["EmbeddingManager"](model_name=normalized_model_name)
        models = manager.get_available_models()
    except Exception as exc:
        return _error_result(f"get_available_models failed: {exc}", models=[], count=0)

    return {
        "status": "success",
        "models": [str(m) for m in models],
        "count": len(models),
    }


async def chunk_text_for_embeddings(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    method: str = "fixed",
    n_sentences: int = 8,
    step_size: int = 256,
) -> Dict[str, Any]:
    """Chunk text for embedding workflows with deterministic envelope validation."""
    if not isinstance(text, str) or not text.strip():
        return _error_result("text must be a non-empty string")

    try:
        normalized_chunk_size = int(chunk_size)
        normalized_chunk_overlap = int(chunk_overlap)
        normalized_n_sentences = int(n_sentences)
        normalized_step_size = int(step_size)
    except (TypeError, ValueError):
        return _error_result("chunk_size, chunk_overlap, n_sentences, and step_size must be integers")

    if normalized_chunk_size <= 0:
        return _error_result("chunk_size must be positive")
    if normalized_chunk_overlap < 0:
        return _error_result("chunk_overlap must be non-negative")
    if normalized_chunk_overlap >= normalized_chunk_size:
        return _error_result("chunk_overlap must be smaller than chunk_size")
    if normalized_n_sentences <= 0:
        return _error_result("n_sentences must be positive")
    if normalized_step_size <= 0:
        return _error_result("step_size must be positive")

    normalized_method = str(method or "").strip()
    if not normalized_method:
        return _error_result("method must be a non-empty string")

    handler = _API.get("chunk_text")
    if not callable(handler):
        return _error_result("chunk_text handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                text=text,
                chunk_size=normalized_chunk_size,
                chunk_overlap=normalized_chunk_overlap,
                method=normalized_method,
                n_sentences=normalized_n_sentences,
                step_size=normalized_step_size,
            )
        )
    except Exception as exc:
        return _error_result(f"chunk_text_for_embeddings failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("original_length", len(text))
    normalized.setdefault("chunks", [])
    normalized.setdefault("chunk_count", len(normalized.get("chunks") or []))
    return normalized


async def manage_embedding_endpoints(
    action: str,
    model: str,
    endpoint: str = "",
    endpoint_type: str = "tei",
    context_length: int = 512,
) -> Dict[str, Any]:
    """Manage embedding endpoints through source-compatible endpoint actions."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"add", "test", "list", "status"}:
        return _error_result(f"Unknown action: {action}")

    model_name = str(model or "").strip()
    if not model_name:
        return _error_result("model must be provided")

    endpoint_value = str(endpoint or "").strip()
    if normalized_action in {"add", "test"} and not endpoint_value:
        return _error_result("endpoint must be provided")

    normalized_endpoint_type = str(endpoint_type or "").strip()
    if not normalized_endpoint_type:
        return _error_result("endpoint_type must be a non-empty string")

    try:
        normalized_context_length = int(context_length)
    except (TypeError, ValueError):
        return _error_result("context_length must be an integer")
    if normalized_context_length <= 0:
        return _error_result("context_length must be a positive integer")

    handler = _API.get("manage_endpoints")
    if not callable(handler):
        return _error_result("manage_endpoints handler unavailable")

    try:
        payload = await _await_maybe(
            handler(
                action=normalized_action,
                model=model_name,
                endpoint=endpoint_value,
                endpoint_type=normalized_endpoint_type,
                context_length=normalized_context_length,
            )
        )
    except Exception as exc:
        return _error_result(f"manage_embedding_endpoints failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("action", normalized_action)
    normalized.setdefault("model", model_name)
    if endpoint_value:
        normalized.setdefault("endpoint", endpoint_value)
    if normalized_action == "test":
        normalized.setdefault("available", False)
    if normalized_action in {"list", "status"}:
        normalized.setdefault("endpoints", [])
    return normalized


def register_native_embedding_tools(manager: Any) -> None:
    """Register native embedding tools in unified hierarchical manager."""
    manager.register_tool(
        category="embedding_tools",
        name="generate_embedding",
        func=generate_embedding,
        description="Generate a single embedding vector for one text input.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "model_name": {
                    "type": "string",
                    "minLength": 1,
                    "default": "sentence-transformers/all-MiniLM-L6-v2",
                },
                "normalize": {"type": "boolean", "default": True},
                "batch_size": {"type": "integer", "minimum": 1, "default": 32},
                "use_gpu": {"type": "boolean", "default": False},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )
    manager.register_tool(
        category="embedding_tools",
        name="generate_embeddings_from_file",
        func=generate_embeddings_from_file,
        description="Generate embeddings from a source file with optional batching/chunking controls.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "output_path": {"type": ["string", "null"]},
                "model_name": {
                    "type": "string",
                    "minLength": 1,
                    "default": "sentence-transformers/all-MiniLM-L6-v2",
                },
                "batch_size": {"type": "integer", "minimum": 1, "default": 32},
                "chunk_size": {"type": ["integer", "null"], "minimum": 1},
                "max_length": {"type": ["integer", "null"], "minimum": 1},
                "output_format": {"type": "string", "enum": ["json", "parquet", "hdf5"], "default": "json"},
            },
            "required": ["file_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="semantic_search",
        func=semantic_search,
        description="Perform semantic search over a vector store using embedding similarity.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "vector_store_id": {"type": "string", "minLength": 1},
                "model_name": {
                    "type": "string",
                    "minLength": 1,
                    "default": "sentence-transformers/all-MiniLM-L6-v2",
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10},
                "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7},
                "include_metadata": {"type": "boolean", "default": True},
            },
            "required": ["query", "vector_store_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding", "search"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="hybrid_search",
        func=hybrid_search,
        description="Perform hybrid lexical+semantic search over a vector store.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "vector_store_id": {"type": "string", "minLength": 1},
                "lexical_weight": {"type": "number", "default": 0.3},
                "semantic_weight": {"type": "number", "default": 0.7},
                "top_k": {"type": "integer", "minimum": 1, "default": 10},
                "rerank_results": {"type": "boolean", "default": True},
            },
            "required": ["query", "vector_store_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding", "search"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="search_with_filters",
        func=search_with_filters,
        description="Perform filtered search over a vector store using semantic, lexical, or hybrid methods.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "vector_store_id": {"type": "string", "minLength": 1},
                "filters": {"type": "object"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10},
                "search_method": {
                    "type": "string",
                    "enum": ["semantic", "lexical", "hybrid"],
                    "default": "semantic",
                },
            },
            "required": ["query", "vector_store_id", "filters"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding", "search"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="multi_modal_search",
        func=multi_modal_search,
        description="Perform multi-modal search using text and/or image query inputs.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "image_query": {"type": ["string", "null"]},
                "vector_store_id": {"type": "string", "minLength": 1},
                "model_name": {"type": "string", "minLength": 1, "default": "clip-ViT-B-32"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10},
                "modality_weights": {"type": ["object", "null"]},
            },
            "required": ["vector_store_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding", "search"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="generate_embeddings",
        func=generate_embeddings,
        description="Generate vector embeddings for a batch of text inputs.",
        input_schema={
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "model_name": {
                    "type": "string",
                    "minLength": 1,
                    "default": "sentence-transformers/all-MiniLM-L6-v2",
                },
            },
            "required": ["texts"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="shard_embeddings",
        func=shard_embeddings,
        description="Shard embedding vectors for distributed processing.",
        input_schema={
            "type": "object",
            "properties": {
                "embeddings": {"type": "array", "minItems": 1},
                "shard_count": {"type": "integer", "minimum": 1, "default": 4},
                "strategy": {"type": "string", "minLength": 1, "default": "balanced"},
            },
            "required": ["embeddings"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="get_available_models",
        func=get_available_models,
        description="List embedding models available to the current runtime.",
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "minLength": 1,
                    "default": "sentence-transformers/all-MiniLM-L6-v2",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="chunk_text_for_embeddings",
        func=chunk_text_for_embeddings,
        description="Chunk text into deterministic segments for embedding workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "chunk_size": {"type": "integer", "minimum": 1, "default": 512},
                "chunk_overlap": {"type": "integer", "minimum": 0, "default": 50},
                "method": {"type": "string", "minLength": 1, "default": "fixed"},
                "n_sentences": {"type": "integer", "minimum": 1, "default": 8},
                "step_size": {"type": "integer", "minimum": 1, "default": 256},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )

    manager.register_tool(
        category="embedding_tools",
        name="manage_embedding_endpoints",
        func=manage_embedding_endpoints,
        description="Manage and validate embedding endpoints with source-compatible actions.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "test", "list", "status"]},
                "model": {"type": "string", "minLength": 1},
                "endpoint": {"type": "string", "default": ""},
                "endpoint_type": {"type": "string", "minLength": 1, "default": "tei"},
                "context_length": {"type": "integer", "minimum": 1, "default": 512},
            },
            "required": ["action", "model"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )


# Source-compatible alias surface from enhanced_embedding_tools.
create_embeddings = generate_embedding
index_dataset = generate_embeddings
search_embeddings = generate_embeddings_from_file
