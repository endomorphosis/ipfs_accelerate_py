"""Thin adapter layer over the embeddings router.

This keeps legacy imports working for modules that expect a `utils.embedding_adapter`
surface while the canonical implementation lives in `embeddings_router`.
"""

from __future__ import annotations

from typing import List, Optional

from ..embeddings_router import embed_text as _router_embed_text
from ..embeddings_router import embed_texts as _router_embed_texts


def embed_text(
    text: str,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: object,
) -> List[float]:
    """Return one embedding vector using the canonical embeddings router."""
    return _router_embed_text(
        text,
        model_name=model_name,
        device=device,
        provider=provider,
        **kwargs,
    )


def embed_texts(
    texts: list[str],
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: object,
) -> list[list[float]]:
    """Return embedding vectors using the canonical embeddings router."""
    return _router_embed_texts(
        texts,
        model_name=model_name,
        device=device,
        provider=provider,
        **kwargs,
    )
