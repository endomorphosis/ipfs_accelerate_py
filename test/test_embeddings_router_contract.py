from __future__ import annotations

import math
from typing import Iterable, Optional

import pytest

from ipfs_accelerate_py.embeddings_router import (
    EmbeddingsRouterError,
    clear_embeddings_router_caches,
    embed_texts,
    embed_texts_batched,
    get_embedding_progress,
    get_last_embedding_trace,
)
from ipfs_accelerate_py.router_deps import RouterDeps


class _OrderedProvider:
    router_provider_name = "ordered_fixture"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> list[list[float]]:
        _ = (model_name, device, kwargs)
        items = list(texts)
        self.calls.append(items)
        return [[float(item), float(item) + 0.5] for item in items]


@pytest.fixture(autouse=True)
def _isolated_router_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    clear_embeddings_router_caches()


def test_batched_embeddings_preserve_order_and_report_progress() -> None:
    provider = _OrderedProvider()
    progress_events: list[dict[str, object]] = []

    vectors = embed_texts_batched(
        ["0", "1", "2", "3", "4"],
        batch_size=2,
        max_workers=2,
        provider_instance=provider,
        progress_callback=progress_events.append,
    )

    assert vectors == [
        [0.0, 0.5],
        [1.0, 1.5],
        [2.0, 2.5],
        [3.0, 3.5],
        [4.0, 4.5],
    ]
    assert len(provider.calls) == 3
    assert progress_events[-1]["stage"] == "done"
    assert get_embedding_progress() == progress_events[-1]
    trace = get_last_embedding_trace()
    assert trace["status"] == "ok"
    assert trace["provider_used"] == "ordered_fixture"
    assert trace["input_count"] == 5
    assert trace["output_count"] == 5
    assert trace["dimension"] == 2
    assert trace["batch_count"] == 3


@pytest.mark.parametrize(
    "vectors",
    (
        [[1.0, 2.0]],
        [[1.0, 2.0], [3.0]],
        [[1.0, 2.0], [math.nan, 4.0]],
    ),
)
def test_provider_output_must_match_count_dimension_and_finiteness(
    vectors: list[list[float]],
) -> None:
    class _InvalidProvider:
        router_provider_name = "invalid_fixture"

        def embed_texts(self, texts: Iterable[str], **kwargs: object) -> list[list[float]]:
            _ = (texts, kwargs)
            return vectors

    with pytest.raises(EmbeddingsRouterError):
        embed_texts(
            ["first", "second"],
            provider_instance=_InvalidProvider(),
        )

    assert get_last_embedding_trace()["status"] == "error"


def test_response_cache_reuses_vectors_and_reports_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    provider = _OrderedProvider()
    deps = RouterDeps()

    first = embed_texts(
        ["7", "8"],
        provider_instance=provider,
        deps=deps,
        model_name="fixture-model",
    )
    second = embed_texts(
        ["7", "8"],
        provider_instance=provider,
        deps=deps,
        model_name="fixture-model",
    )

    assert second == first
    assert len(provider.calls) == 1
    trace = get_last_embedding_trace()
    assert trace["cache_hits"] == 2
    assert trace["cache_misses"] == 0


def test_local_huggingface_failure_is_not_retried_as_its_own_fallback() -> None:
    class _BrokenLocalProvider:
        router_provider_name = "huggingface"

        def __init__(self) -> None:
            self.calls = 0

        def embed_texts(self, texts: Iterable[str], **kwargs: object) -> list[list[float]]:
            _ = (texts, kwargs)
            self.calls += 1
            raise RuntimeError("fixture failure")

    provider = _BrokenLocalProvider()

    with pytest.raises(RuntimeError, match="fixture failure"):
        embed_texts(["one"], provider_instance=provider)

    assert provider.calls == 1
    assert get_last_embedding_trace()["status"] == "error"
