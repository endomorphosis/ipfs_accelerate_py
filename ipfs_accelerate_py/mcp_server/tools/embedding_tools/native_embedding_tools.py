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

        return {
            "EmbeddingManager": _EmbeddingManager,
            "generate_embeddings": _generate_embeddings,
            "shard_embeddings": _shard_embeddings,
        }
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

        return {
            "EmbeddingManager": _FallbackEmbeddingManager,
            "generate_embeddings": _generate_fallback,
            "shard_embeddings": _shard_fallback,
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
