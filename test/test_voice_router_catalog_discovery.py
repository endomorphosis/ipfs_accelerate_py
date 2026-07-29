"""Offline contract tests for voice-router catalog discovery."""

from __future__ import annotations

import builtins
import json
import urllib.request

import pytest

from ipfs_accelerate_py.model_catalog import (
    CatalogSnapshot,
    ModelDescriptor,
    Operation,
    ProviderDescriptor,
)
from ipfs_accelerate_py import voice_router
from ipfs_accelerate_py.voice_router import (
    VoiceProviderCapabilities,
    get_catalog_snapshot,
    get_provider_descriptor,
    get_voice_provider,
    list_models,
    list_providers,
    register_voice_provider,
    resolve_model,
)


@pytest.fixture(autouse=True)
def clean_dynamic_registry():
    registry = dict(voice_router._PROVIDER_REGISTRY)
    revisions = dict(voice_router._PROVIDER_REGISTRY_REVISIONS)
    try:
        yield
    finally:
        voice_router._PROVIDER_REGISTRY.clear()
        voice_router._PROVIDER_REGISTRY.update(registry)
        voice_router._PROVIDER_REGISTRY_REVISIONS.clear()
        voice_router._PROVIDER_REGISTRY_REVISIONS.update(revisions)
        voice_router.clear_voice_router_caches()


def operations(descriptor):
    return {
        operation
        for capability in descriptor.capabilities
        for operation in capability.operations
    }


def test_builtins_publish_typed_deterministic_operation_descriptors(monkeypatch):
    for variable in (
        "OPENAI_API_KEY",
        "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    first = list_providers()
    second = list_providers()

    assert first == second
    assert tuple(item.name for item in first) == tuple(
        sorted(item.name for item in first)
    )
    assert all(isinstance(item, ProviderDescriptor) for item in first)
    by_name = {item.name: item for item in first}
    assert Operation.AUDIO_SYNTHESIZE in operations(by_name["elevenlabs"])
    assert Operation.AUDIO_TRANSCRIBE not in operations(by_name["elevenlabs"])
    assert Operation.AUDIO_TRANSCRIBE in operations(by_name["assemblyai"])
    assert Operation.AUDIO_SYNTHESIZE not in operations(by_name["assemblyai"])
    assert {
        Operation.AUDIO_TRANSCRIBE,
        Operation.AUDIO_SYNTHESIZE,
    }.issubset(operations(by_name["openai"]))

    openai_labels = dict(by_name["openai"].labels)
    assert openai_labels["audio.languages"] == "multilingual"
    assert openai_labels["audio.default_voice"] == "alloy"
    assert openai_labels["streaming"] == "false"
    assert openai_labels["batching"] == "false"
    assert openai_labels["locality"] == "remote"
    assert openai_labels["device"] == "remote"
    assert by_name["openai"].state.authorized is False
    assert by_name["openai"].state.reachable is None
    assert by_name["openai"].state.healthy is None

    transcription = next(
        capability
        for capability in by_name["abby_whisper"].capabilities
        if Operation.AUDIO_TRANSCRIBE in capability.operations
    )
    assert "audio/wav" in transcription.media_types
    assert transcription.max_input_bytes is None

    publicus = by_name["abby_indextts"]
    publicus_labels = dict(publicus.labels)
    assert publicus.display_name == "Publicus IndexTTS (Abby)"
    assert publicus.state.configured is True
    assert Operation.BATCH in operations(publicus)
    assert publicus_labels["backend"] == "publicus_gradio"
    assert publicus_labels["batching"] == "true"
    assert publicus_labels["gradio.single_api"] == "/gen_single"
    assert publicus_labels["gradio.single_fn_index"] == "6"
    assert publicus_labels["gradio.batch_api"] == "/gen_batch"
    assert publicus_labels["gradio.batch_fn_index"] == "7"
    assert publicus_labels["gradio.input_count"] == "25"


def test_listing_is_side_effect_free(monkeypatch):
    constructed = []

    class Provider:
        def synthesize(self, text, **kwargs):
            return b"audio"

        def transcribe(self, audio, **kwargs):
            return "text"

    def factory():
        constructed.append(True)
        raise AssertionError("discovery constructed a dynamic provider")

    register_voice_provider(
        "catalog-side-effect-test",
        factory,
        capabilities=VoiceProviderCapabilities(
            transcription=True,
            synthesis=False,
            audio_formats=("wav",),
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("discovery performed I/O or provider resolution")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(voice_router, "_get_openai_provider", forbidden)
    monkeypatch.setattr(voice_router, "_get_elevenlabs_provider", forbidden)
    monkeypatch.setattr(voice_router, "_get_assemblyai_provider", forbidden)
    monkeypatch.setattr(voice_router, "_get_huggingface_provider", forbidden)
    monkeypatch.setattr(voice_router, "_get_backend_manager_provider", forbidden)
    monkeypatch.setattr(voice_router, "_builtin_provider_by_name", forbidden)

    descriptor = get_provider_descriptor("catalog-side-effect-test")
    models = list_models("catalog-side-effect-test")
    snapshot = get_catalog_snapshot()

    assert descriptor.name == "catalog-side-effect-test"
    assert Operation.AUDIO_TRANSCRIBE in operations(descriptor)
    assert Operation.AUDIO_SYNTHESIZE not in operations(descriptor)
    assert models and all(isinstance(item, ModelDescriptor) for item in models)
    assert isinstance(snapshot, CatalogSnapshot)
    assert constructed == []


def test_aliases_and_dynamic_registration_are_deterministic():
    canonical = get_provider_descriptor("huggingface")
    assert get_provider_descriptor("HF") == canonical
    assert get_provider_descriptor("local_hf") == canonical
    assert canonical.aliases == tuple(sorted(canonical.aliases))
    assert {"hf", "local_hf"}.issubset(set(canonical.aliases))

    publicus = get_provider_descriptor("publicus")
    assert publicus == get_provider_descriptor("publicus_indextts")
    assert publicus == get_provider_descriptor("indextts")
    assert {
        "publicus",
        "publicus_indextts",
        "publicus_tts",
        "indextts",
    }.issubset(set(publicus.aliases))

    class Dynamic:
        def synthesize(self, text, **kwargs):
            return b"dynamic"

        def transcribe(self, audio, **kwargs):
            raise NotImplementedError

    register_voice_provider(
        "z-dynamic-voice",
        lambda: Dynamic(),
        capabilities=VoiceProviderCapabilities(
            transcription=False,
            synthesis=True,
            streaming=True,
            audio_formats=("MP3", "wav"),
        ),
    )
    first = list_providers()
    second = list_providers()
    dynamic = get_provider_descriptor("Z-DYNAMIC-VOICE")

    assert first == second
    assert dynamic in first
    assert Operation.AUDIO_SYNTHESIZE in operations(dynamic)
    assert Operation.AUDIO_TRANSCRIBE not in operations(dynamic)
    assert Operation.STREAM in operations(dynamic)
    assert dict(dynamic.labels)["readiness"] == "registered-unverified"
    assert [item.name for item in first] == sorted(item.name for item in first)


def test_models_report_operation_specific_voice_metadata(monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_PY_OPENAI_TTS_MODEL", "TTS-1-HD")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_OPENAI_STT_MODEL", "Whisper-1")

    synthesis = list_models("openai", operation="tts")
    transcription = list_models("openai", operation="speech_to_text")

    assert [item.name for item in synthesis] == ["tts-1-hd"]
    assert [item.name for item in transcription] == ["whisper-1"]
    assert all(Operation.AUDIO_SYNTHESIZE in operations(item) for item in synthesis)
    assert all(Operation.AUDIO_TRANSCRIBE in operations(item) for item in transcription)
    assert "audio/mpeg" in synthesis[0].capabilities[0].media_types
    assert "audio/webm" in transcription[0].capabilities[0].media_types
    assert dict(synthesis[0].labels)["audio.default_voice"] == "alloy"
    assert dict(transcription[0].labels)["audio.languages"] == "multilingual"


def test_explicit_resolution_agrees_with_invocation_provider_and_overrides():
    instance = object()
    calls = []

    def factory():
        calls.append("factory")
        return instance

    register_voice_provider(
        "explicit-catalog-provider",
        factory,
        capabilities=VoiceProviderCapabilities(
            transcription=False,
            synthesis=True,
            audio_formats=("wav",),
        ),
    )

    resolved = resolve_model(
        "vendor/Voice Model v2",
        provider="EXPLICIT-CATALOG-PROVIDER",
        operation="synthesis",
        media_type="audio/wav",
    )

    assert calls == []
    assert resolved.name == "vendor/voice-model-v2"
    assert resolved.provider_id == get_provider_descriptor(
        "explicit-catalog-provider"
    ).provider_id
    assert Operation.AUDIO_SYNTHESIZE in operations(resolved)
    assert dict(resolved.labels)["explicit_override"] == "true"
    assert get_voice_provider(
        "explicit-catalog-provider", use_cache=False
    ) is instance
    assert calls == ["factory"]

    with pytest.raises(ValueError, match="No compatible"):
        resolve_model(
            "vendor/voice-model-v2",
            provider="explicit-catalog-provider",
            operation="transcription",
        )


def test_filters_keep_unknown_distinct_from_false(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY", raising=False)

    assert {item.name for item in list_providers(operation="stt")} >= {
        "abby_whisper",
        "assemblyai",
        "huggingface",
        "openai",
    }
    assert "openai" not in {
        item.name for item in list_providers(operation="stt", authorized=True)
    }
    assert "huggingface" in {
        item.name for item in list_providers(operation="stt", authorized=True)
    }
    assert list_providers(operation="tts", streaming=True) == ()
    assert tuple(
        item.name for item in list_providers(operation="tts", batching=True)
    ) == ("abby_indextts",)
    assert "huggingface" in {
        item.name
        for item in list_providers(
            operation="tts", locality="local", device="cuda"
        )
    }


def test_catalog_snapshot_is_complete_stable_and_json_safe():
    first = get_catalog_snapshot()
    second = voice_router.catalog_snapshot()

    assert first == second
    assert first.revision == second.revision
    assert len(first.providers) == len(list_providers())
    assert len(first.models) == len(first.bindings)
    assert all(binding.router == "voice_router" for binding in first.bindings)
    assert {model.model_id for model in first.models} == {
        binding.model_id for binding in first.bindings
    }
    encoded = json.dumps(first.to_dict(), sort_keys=True)
    assert "api_key" not in encoded.casefold()
    assert "secret" not in encoded.casefold()
