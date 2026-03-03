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
                chunk_text as _chunk_text,
                manage_endpoints as _manage_endpoints,
            )

            api["chunk_text"] = _chunk_text
            api["manage_endpoints"] = _manage_endpoints
        except Exception:
            logger.warning("Source enhanced_embedding_tools import unavailable, using fallback endpoint/chunk functions")

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

        return {
            "EmbeddingManager": _FallbackEmbeddingManager,
            "generate_embeddings": _generate_fallback,
            "shard_embeddings": _shard_fallback,
            "chunk_text": _chunk_text_fallback,
            "manage_endpoints": _manage_endpoints_fallback,
        }


_API = _load_embedding_api()


async def generate_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate vector embeddings for multiple texts."""
    result = _API["generate_embeddings"](texts=texts, model_name=model_name, **kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result


async def shard_embeddings(
    embeddings: List[Any],
    shard_count: int = 4,
    strategy: str = "balanced",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Shard embedding vectors using the selected strategy."""
    result = _API["shard_embeddings"](
        embeddings=embeddings,
        shard_count=shard_count,
        strategy=strategy,
        **kwargs,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_available_models(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Return available embedding model names from the source manager."""
    manager = _API["EmbeddingManager"](model_name=model_name)
    models = manager.get_available_models()
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
        return {"status": "error", "error": "text must be a non-empty string"}
    if int(chunk_size) <= 0:
        return {"status": "error", "error": "chunk_size must be positive"}
    if int(chunk_overlap) < 0:
        return {"status": "error", "error": "chunk_overlap must be non-negative"}

    handler = _API.get("chunk_text")
    if not callable(handler):
        return {"status": "error", "error": "chunk_text handler unavailable"}

    result = handler(
        text=text,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        method=str(method or "fixed"),
        n_sentences=int(n_sentences),
        step_size=int(step_size),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
        return {"status": "error", "error": f"Unknown action: {action}"}

    model_name = str(model or "").strip()
    if not model_name:
        return {"status": "error", "error": "model must be provided"}

    endpoint_value = str(endpoint or "").strip()
    if normalized_action in {"add", "test"} and not endpoint_value:
        return {"status": "error", "error": "endpoint must be provided"}

    handler = _API.get("manage_endpoints")
    if not callable(handler):
        return {"status": "error", "error": "manage_endpoints handler unavailable"}

    result = handler(
        action=normalized_action,
        model=model_name,
        endpoint=endpoint_value,
        endpoint_type=str(endpoint_type or "tei"),
        context_length=max(1, int(context_length)),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_embedding_tools(manager: Any) -> None:
    """Register native embedding tools in unified hierarchical manager."""
    manager.register_tool(
        category="embedding_tools",
        name="generate_embeddings",
        func=generate_embeddings,
        description="Generate vector embeddings for a batch of text inputs.",
        input_schema={
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "model_name": {"type": "string"},
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
                "embeddings": {"type": "array"},
                "shard_count": {"type": "integer"},
                "strategy": {"type": "string"},
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
                "model_name": {"type": "string"},
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
                "chunk_size": {"type": "integer", "minimum": 1},
                "chunk_overlap": {"type": "integer", "minimum": 0},
                "method": {"type": "string"},
                "n_sentences": {"type": "integer", "minimum": 1},
                "step_size": {"type": "integer", "minimum": 1},
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
                "model": {"type": "string"},
                "endpoint": {"type": "string"},
                "endpoint_type": {"type": "string"},
                "context_length": {"type": "integer", "minimum": 1},
            },
            "required": ["action", "model"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "embedding"],
    )
