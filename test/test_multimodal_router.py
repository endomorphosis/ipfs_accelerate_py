"""Offline regression tests for the multimodal router invocation surface."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Union

import pytest

from ipfs_accelerate_py import multimodal_router
from ipfs_accelerate_py.router_deps import RouterDeps


@pytest.fixture(autouse=True)
def _isolated_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multimodal_router, "_PROVIDER_REGISTRY", {})
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    multimodal_router.clear_multimodal_router_caches()


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, bytes]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "image": image,
                "model_name": model_name,
                "device": device,
                "kwargs": kwargs,
            }
        )
        return f"{model_name}:{prompt}"


def test_registered_provider_generation_preserves_arguments() -> None:
    instance = _RecordingProvider()
    multimodal_router.register_multimodal_provider(
        "fixture",
        lambda: instance,
    )

    result = multimodal_router.generate_multimodal(
        "describe",
        image=b"image-bytes",
        model_name="fixture/vision",
        device="cpu",
        provider="fixture",
        max_tokens=32,
    )

    assert result == "fixture/vision:describe"
    assert instance.calls == [
        {
            "prompt": "describe",
            "image": b"image-bytes",
            "model_name": "fixture/vision",
            "device": "cpu",
            "kwargs": {"max_tokens": 32},
        }
    ]


def test_provider_instance_bypasses_discovery() -> None:
    instance = _RecordingProvider()

    result = multimodal_router.generate_multimodal(
        "caption",
        image="https://example.invalid/image.png",
        provider_instance=instance,
    )

    assert result == "None:caption"
    assert instance.calls[0]["image"] == "https://example.invalid/image.png"


def test_response_cache_includes_image_and_reuses_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    instance = _RecordingProvider()
    deps = RouterDeps()

    first = multimodal_router.generate_multimodal(
        "describe",
        image=b"first",
        provider_instance=instance,
        model_name="fixture/model",
        deps=deps,
    )
    second = multimodal_router.generate_multimodal(
        "describe",
        image=b"first",
        provider_instance=instance,
        model_name="fixture/model",
        deps=deps,
    )
    multimodal_router.generate_multimodal(
        "describe",
        image=b"second",
        provider_instance=instance,
        model_name="fixture/model",
        deps=deps,
    )

    assert second == first
    assert len(instance.calls) == 2


def test_unknown_explicit_provider_raises_without_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown multimodal provider"):
        multimodal_router.get_multimodal_provider(
            "does-not-exist",
            use_cache=False,
        )


def test_image_encoding_supports_bytes_urls_and_local_files(
    tmp_path: Path,
) -> None:
    raw = b"\x89PNG\r\nfixture"
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(raw)

    encoded_bytes, bytes_kind = multimodal_router._encode_image_for_api(raw)
    encoded_path, path_kind = multimodal_router._encode_image_for_api(
        str(image_path)
    )
    encoded_url, url_kind = multimodal_router._encode_image_for_api(
        "https://example.invalid/image.png"
    )

    expected_payload = base64.b64encode(raw).decode("ascii")
    assert encoded_bytes == f"data:image/jpeg;base64,{expected_payload}"
    assert bytes_kind == "base64"
    assert encoded_path == f"data:image/png;base64,{expected_payload}"
    assert path_kind == "base64"
    assert encoded_url == "https://example.invalid/image.png"
    assert url_kind == "url"


def test_message_builder_and_text_fallback_keep_image_context(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"png")
    provider = _RecordingProvider()

    messages = multimodal_router.build_multimodal_messages(
        prompt="What is shown?",
        image_paths=(image_path,),
        image_urls=("https://example.invalid/remote.png",),
        system_prompt="Be concise.",
        additional_text_blocks=("Focus on color.",),
    )
    result = multimodal_router.generate_multimodal_text(
        "ignored because explicit messages are supplied",
        messages=messages,
        provider_instance=provider,
    )

    assert messages[0] == {"role": "system", "content": "Be concise."}
    assert [part["type"] for part in messages[1]["content"]] == [
        "text",
        "text",
        "image_url",
        "image_url",
    ]
    assert result.startswith("None:system: Be concise.")
    flattened = str(provider.calls[0]["prompt"])
    assert "[image attachment included]" in flattened
    assert "[image: https://example.invalid/remote.png]" in flattened


def test_router_facade_merges_configuration_and_call_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(prompt: str, **kwargs: object) -> str:
        captured.update({"prompt": prompt, **kwargs})
        return "ok"

    monkeypatch.setattr(
        multimodal_router,
        "generate_multimodal_text",
        fake_generate,
    )
    router = multimodal_router.MultimodalRouter(
        provider="fixture",
        model_name="configured-model",
        temperature=0.2,
        max_tokens=10,
    )

    assert (
        router.generate(
            "hello",
            model_name="request-model",
            temperature=0.7,
        )
        == "ok"
    )
    assert captured["provider"] == "fixture"
    assert captured["model_name"] == "request-model"
    assert captured["temperature"] == 0.7
    assert captured["max_tokens"] == 10
