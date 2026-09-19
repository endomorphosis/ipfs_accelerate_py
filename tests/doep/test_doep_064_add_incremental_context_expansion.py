"""DOEP-064 incremental context expansion."""

from __future__ import annotations

import pytest

from ipfs_datasets_py.proof_context.context_pack import (
    ContextPack,
    ContextSpan,
    ContextPackError,
    expand_context_pack,
)


def test_expansion_requires_unresolved_question() -> None:
    pack = ContextPack(spans=[], unresolved=[], budget=16)
    with pytest.raises(ContextPackError, match="unresolved"):
        expand_context_pack(pack, question="missing", candidates=[])


def test_required_span_is_never_dropped() -> None:
    pack = ContextPack(
        spans=[ContextSpan("core", "policy", True, 4, "policy")],
        unresolved=["what next?"],
        budget=16,
    )
    expanded = expand_context_pack(
        pack,
        question="what next?",
        candidates=[{"span_id": "proof", "kind": "proof", "required": True, "tokens": 4, "content": "qed"}],
    )
    assert [span.span_id for span in expanded.spans] == ["core", "proof"]
    assert expanded.unresolved == []


def test_embeddings_cannot_suppress_required_expansion() -> None:
    pack = ContextPack(spans=[], unresolved=["need proof"], budget=16)
    with pytest.raises(ContextPackError, match="embeddings"):
        expand_context_pack(
            pack,
            question="need proof",
            candidates=[
                {
                    "span_id": "ann",
                    "kind": "ann",
                    "required": True,
                    "tokens": 2,
                    "content": "similar neighbour",
                }
            ],
        )
