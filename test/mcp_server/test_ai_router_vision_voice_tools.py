"""Contract tests for bounded MCP multimodal and voice router invocation."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pytest

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py.mcp_server.tools.ai_router_tools import vision_voice
from ipfs_accelerate_py.model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog
from ipfs_accelerate_py.model_catalog.sources.static import (
    CatalogSourceResult,
    SourceMetadata,
)
from ipfs_accelerate_py.model_manager import ModelManager


def _run(awaitable: Any) -> Dict[str, Any]:
    return asyncio.run(awaitable)


def _records(
    provider_name: str,
    router: str,
    operation: Operation,
    *,
    priority: int = 0,
    model_name: str = "fixture-model",
    input_modalities: Optional[Tuple[Modality, ...]] = None,
    output_modalities: Optional[Tuple[Modality, ...]] = None,
    media_types: Tuple[str, ...] = (),
    max_input_bytes: Optional[int] = 4096,
    max_output_bytes: Optional[int] = 4096,
) -> Tuple[Any, ...]:
    defaults = {
        Operation.VISION_GENERATE: ((Modality.TEXT, Modality.IMAGE), (Modality.TEXT,)),
        Operation.AUDIO_TRANSCRIBE: ((Modality.AUDIO,), (Modality.TEXT,)),
        Operation.AUDIO_SYNTHESIZE: ((Modality.TEXT,), (Modality.AUDIO,)),
    }
    default_inputs, default_outputs = defaults[operation]
    capability = CapabilityDescriptor(
        operations=(operation,),
        input_modalities=input_modalities or default_inputs,
        output_modalities=output_modalities or default_outputs,
        media_types=media_types,
        max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
    )
    provenance = (Provenance(source="fixture.router"),)
    state = OperationalState(
        known=True,
        configured=True,
        authorized=True,
        reachable=True,
        healthy=True,
        routable=True,
    )
    provider = ProviderDescriptor(
        name=provider_name,
        capabilities=(capability,),
        state=state,
        provenance=provenance,
        labels=(("invocation_provider", provider_name),),
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name=model_name,
        capabilities=(capability,),
        state=state,
        provenance=provenance,
        labels=(("invocation_model", model_name),),
    )
    binding = RouterBinding(
        router=router,
        provider_id=provider.provider_id,
        model_id=model.model_id,
        operations=(operation,),
        priority=priority,
        state=state,
        provenance=provenance,
        labels=(
            ("invocation_model", model_name),
            ("invocation_provider", provider_name),
        ),
    )
    return provider, model, binding


def _snapshot(*groups: Iterable[Any]) -> CatalogSnapshot:
    records = tuple(item for group in groups for item in group)
    return CatalogSnapshot(
        providers=tuple(
            item for item in records if isinstance(item, ProviderDescriptor)
        ),
        models=tuple(item for item in records if isinstance(item, ModelDescriptor)),
        bindings=tuple(item for item in records if isinstance(item, RouterBinding)),
    )


class MemorySource:
    source = "fixture.router"
    precedence = 30
    side_effecting = False

    def __init__(self, snapshot: CatalogSnapshot) -> None:
        self.current = snapshot

    def load(self) -> CatalogSourceResult:
        return CatalogSourceResult(
            snapshot=self.current,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=self.current.revision,
            ),
        )


@pytest.fixture
def install_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in (
        "HAVE_STORAGE_WRAPPER",
        "HAVE_IPFS_KIT_STORAGE",
        "HAVE_DATASETS_INTEGRATION",
        "HAVE_GRAPHRAG",
    ):
        monkeypatch.setattr(model_manager_module, name, False)
    managers = []

    def install(snapshot: CatalogSnapshot) -> ModelManager:
        source = MemorySource(snapshot)
        manager = ModelManager(
            storage_path=str(tmp_path / ("router-%d.json" % len(managers))),
            use_database=False,
            enable_ipfs=False,
            catalog=AIServiceCatalog({source.source: source}),
            project_legacy_models=False,
        )
        managers.append(manager)
        monkeypatch.setattr(
            model_manager_module,
            "get_default_model_manager",
            lambda: manager,
        )
        return manager

    yield install

    vision_voice.configure_media_loader(None)
    for manager in managers:
        manager.close()


class ToolRecorder:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[definition["name"]] = definition


def _image(data: bytes = b"fixture-image") -> Dict[str, Any]:
    return {
        "source": "inline",
        "mime_type": "image/png",
        "data_base64": base64.b64encode(data).decode("ascii"),
        "byte_length": len(data),
        "width": 32,
        "height": 24,
    }


def _audio(data: bytes = b"fixture-audio") -> Dict[str, Any]:
    return {
        "source": "inline",
        "mime_type": "audio/wav",
        "data_base64": base64.b64encode(data).decode("ascii"),
        "byte_length": len(data),
        "duration_seconds": 1.25,
        "sample_rate_hz": 16_000,
    }


def _wav(sample_rate: int = 24_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * 24)
    return output.getvalue()


def test_registration_is_cold_and_schemas_discriminate_media_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_manager_module,
        "get_default_model_manager",
        lambda: pytest.fail("registration must not resolve the catalog"),
    )
    registry = ToolRecorder()

    vision_voice.register_native_ai_router_tools(registry)

    assert {
        "multimodal_generate",
        "voice_transcribe",
        "voice_synthesize",
    } == set(registry.tools)
    schema = registry.tools["multimodal_generate"]["input_schema"]
    media = schema["properties"]["media"]
    variants = media["items"]["oneOf"]
    assert media["maxItems"] == vision_voice.MAX_MEDIA_ITEMS
    assert [item["properties"]["source"]["const"] for item in variants] == [
        "inline",
        "uri",
        "artifact",
    ]
    assert "data_base64" in variants[0]["properties"]
    assert "uri" in variants[1]["properties"]
    assert "artifact_ref" in variants[2]["properties"]
    assert schema["properties"]["allow_remote_media"]["default"] is False
    assert schema["properties"]["timeout"]["maximum"] == 120.0


def test_multimodal_routes_through_canonical_router_with_mcp_parity(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = install_manager(
        _snapshot(
            _records(
                "vision-provider",
                "multimodal_router",
                Operation.VISION_GENERATE,
                model_name="vision-model",
                media_types=("image/png",),
            )
        )
    )
    calls = []

    def generate(prompt: str, **kwargs: Any) -> str:
        calls.append((prompt, kwargs))
        return "visible answer"

    monkeypatch.setattr(
        vision_voice.multimodal_router, "generate_multimodal", generate
    )
    request = {
        "prompt": "private question",
        "media": [_image()],
        "provider": "vision-provider",
    }
    direct = _run(vision_voice.multimodal_generate(**request))
    registry = ToolRecorder()
    vision_voice.register_native_ai_router_tools(registry)
    through_mcp = _run(registry.tools["multimodal_generate"]["func"](**request))

    assert direct == through_mcp
    assert direct["status"] == "success"
    assert direct["text"] == "visible answer"
    assert direct["catalog_revision"] == manager.catalog_revision
    assert direct["selected_binding"]["router"] == "multimodal_router"
    assert direct["receipt"]["operation"] == "vision.generate"
    assert direct["receipt"]["input"]["media"][0] == {
        "source": "inline",
        "mime_type": "image/png",
        "bytes": len(b"fixture-image"),
        "width": 32,
        "height": 24,
    }
    receipt_json = json.dumps(direct["receipt"])
    assert "fixture-image" not in receipt_json
    assert "data_base64" not in receipt_json
    assert calls[0][0] == "private question"
    assert calls[0][1]["image"] == b"fixture-image"
    assert calls[0][1]["provider"] == "vision-provider"
    assert calls[0][1]["model_name"] == "vision-model"


def test_voice_transcribe_and_synthesize_dispatch_only_through_voice_router(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = install_manager(
        _snapshot(
            _records(
                "stt-provider",
                "voice_router",
                Operation.AUDIO_TRANSCRIBE,
                model_name="stt-model",
                media_types=("audio/wav",),
            ),
            _records(
                "tts-provider",
                "voice_router",
                Operation.AUDIO_SYNTHESIZE,
                model_name="tts-model",
                media_types=("audio/wav",),
            ),
        )
    )
    transcribe_calls = []
    synthesize_calls = []
    output_audio = _wav()

    def transcribe(audio: bytes, **kwargs: Any) -> str:
        transcribe_calls.append((audio, kwargs))
        return "bounded transcript"

    def synthesize(text: str, **kwargs: Any) -> bytes:
        synthesize_calls.append((text, kwargs))
        return output_audio

    monkeypatch.setattr(vision_voice.voice_router, "speech_to_text", transcribe)
    monkeypatch.setattr(vision_voice.voice_router, "text_to_speech", synthesize)

    transcript = _run(
        vision_voice.voice_transcribe(
            _audio(),
            provider="stt-provider",
            model="stt-model",
            language="en",
        )
    )
    speech = _run(
        vision_voice.voice_synthesize(
            "private speech text",
            provider="tts-provider",
            model="tts-model",
        )
    )

    assert transcript["status"] == speech["status"] == "success"
    assert transcript["catalog_revision"] == manager.catalog_revision
    assert transcript["text"] == "bounded transcript"
    assert transcribe_calls[0][0] == b"fixture-audio"
    assert transcribe_calls[0][1]["provider"] == "stt-provider"
    assert synthesize_calls[0][0] == "private speech text"
    assert synthesize_calls[0][1]["output_format"] == "wav"
    assert base64.b64decode(speech["audio"]["data_base64"]) == output_audio
    assert speech["audio"]["sha256"]
    assert "data_base64" not in json.dumps(speech["receipt"])
    assert "private speech text" not in json.dumps(speech["receipt"])
    assert speech["selected_binding"]["router"] == "voice_router"


def test_wrong_modality_unsupported_mime_and_item_count_fail_before_dispatch(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "wrong-modality",
                "multimodal_router",
                Operation.VISION_GENERATE,
                input_modalities=(Modality.TEXT,),
                output_modalities=(Modality.TEXT,),
                media_types=("image/png",),
            )
        )
    )
    calls = 0

    def forbidden(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return "unexpected"

    monkeypatch.setattr(
        vision_voice.multimodal_router, "generate_multimodal", forbidden
    )
    wrong = _run(
        vision_voice.multimodal_generate(
            "question", [_image()], provider="wrong-modality"
        )
    )
    bad_mime = dict(_image())
    bad_mime["mime_type"] = "image/svg+xml"
    unsupported = _run(
        vision_voice.multimodal_generate("question", [bad_mime])
    )
    too_many = _run(
        vision_voice.multimodal_generate("question", [_image(), _image()])
    )

    assert wrong["error"]["code"] == "selection_denied"
    assert wrong["catalog_revision"]
    assert unsupported["error"]["code"] == "unsupported_mime"
    assert too_many["error"]["code"] == "item_count_exceeded"
    assert calls == 0


def test_media_byte_duration_sample_rate_and_dimension_limits_fail_closed(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "vision-provider",
                "multimodal_router",
                Operation.VISION_GENERATE,
                media_types=("image/png",),
            ),
            _records(
                "voice-provider",
                "voice_router",
                Operation.AUDIO_TRANSCRIBE,
                media_types=("audio/wav",),
            ),
        )
    )
    monkeypatch.setattr(
        vision_voice.multimodal_router,
        "generate_multimodal",
        lambda *args, **kwargs: "unexpected",
    )
    monkeypatch.setattr(
        vision_voice.voice_router,
        "speech_to_text",
        lambda *args, **kwargs: "unexpected",
    )

    oversized = _image(b"x")
    oversized["byte_length"] = vision_voice.MAX_MEDIA_BYTES + 1
    huge_dimensions = _image()
    huge_dimensions["width"] = vision_voice.MAX_IMAGE_WIDTH
    huge_dimensions["height"] = vision_voice.MAX_IMAGE_HEIGHT
    long_audio = _audio()
    long_audio["duration_seconds"] = (
        vision_voice.MAX_MEDIA_DURATION_SECONDS + 1
    )
    fast_audio = _audio()
    fast_audio["sample_rate_hz"] = vision_voice.MAX_SAMPLE_RATE_HZ + 1

    byte_result = _run(
        vision_voice.multimodal_generate("q", [oversized])
    )
    dimension_result = _run(
        vision_voice.multimodal_generate("q", [huge_dimensions])
    )
    duration_result = _run(vision_voice.voice_transcribe(long_audio))
    sample_rate_result = _run(vision_voice.voice_transcribe(fast_audio))

    assert byte_result["error"]["code"] == "invalid_request"
    assert dimension_result["error"]["code"] == "dimension_limit_exceeded"
    assert duration_result["error"]["code"] == "invalid_request"
    assert sample_rate_result["error"]["code"] == "invalid_request"


def test_uri_is_ssrf_filtered_remote_disabled_and_loader_delegated(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "vision-provider",
                "multimodal_router",
                Operation.VISION_GENERATE,
                media_types=("image/png",),
            )
        )
    )
    router_calls = []
    loader_calls = []

    class Loader:
        def load(
            self,
            descriptor: Dict[str, Any],
            *,
            max_bytes: int,
            timeout: float,
        ) -> bytes:
            loader_calls.append((descriptor, max_bytes, timeout))
            return b"loaded"

    vision_voice.configure_media_loader(Loader())
    monkeypatch.setattr(
        vision_voice.multimodal_router,
        "generate_multimodal",
        lambda *args, **kwargs: (router_calls.append((args, kwargs)) or "ok"),
    )
    uri = {
        "source": "uri",
        "uri": "https://media.example.test/image.png",
        "mime_type": "image/png",
        "byte_length": len(b"loaded"),
        "width": 10,
        "height": 10,
    }
    unsafe = dict(uri, uri="https://127.0.0.1/private")

    rejected = _run(
        vision_voice.multimodal_generate(
            "q", [unsafe], allow_remote_media=True
        )
    )
    disabled = _run(vision_voice.multimodal_generate("q", [uri]))
    loaded = _run(
        vision_voice.multimodal_generate(
            "q", [uri], allow_remote_media=True
        )
    )

    assert rejected["error"]["code"] == "unsafe_media_uri"
    assert disabled["error"]["code"] == "remote_media_disabled"
    assert loaded["status"] == "success"
    assert len(loader_calls) == 1
    assert loader_calls[0][0]["uri"] == uri["uri"]
    assert loader_calls[0][1] == len(b"loaded")
    assert router_calls[0][1]["image"] == b"loaded"
    assert uri["uri"] not in json.dumps(loaded["receipt"])


def test_artifact_materialization_uses_same_bounded_media_layer(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "voice-provider",
                "voice_router",
                Operation.AUDIO_TRANSCRIBE,
                media_types=("audio/wav",),
            )
        )
    )
    loader_descriptors = []

    class Loader:
        async def load(
            self,
            descriptor: Dict[str, Any],
            *,
            max_bytes: int,
            timeout: float,
        ) -> Dict[str, Any]:
            loader_descriptors.append(descriptor)
            return {
                "data": b"artifact-audio",
                "mime_type": "audio/wav",
                "duration_seconds": 1.0,
                "sample_rate_hz": 16_000,
            }

    vision_voice.configure_media_loader(Loader())
    monkeypatch.setattr(
        vision_voice.voice_router,
        "speech_to_text",
        lambda audio, **kwargs: "artifact transcript",
    )
    descriptor = {
        "source": "artifact",
        "artifact_ref": "artifact:private-reference",
        "mime_type": "audio/wav",
        "byte_length": len(b"artifact-audio"),
        "duration_seconds": 1.0,
        "sample_rate_hz": 16_000,
    }

    result = _run(vision_voice.voice_transcribe(descriptor))

    assert result["status"] == "success"
    assert loader_descriptors[0]["artifact_ref"] == "artifact:private-reference"
    assert "private-reference" not in json.dumps(result["receipt"])


def test_output_and_streaming_limits_are_enforced(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "vision-provider",
                "multimodal_router",
                Operation.VISION_GENERATE,
                media_types=("image/png",),
                max_output_bytes=1024,
            ),
            _records(
                "voice-provider",
                "voice_router",
                Operation.AUDIO_SYNTHESIZE,
                media_types=("audio/wav",),
                max_output_bytes=4096,
            ),
        )
    )
    monkeypatch.setattr(
        vision_voice.multimodal_router,
        "generate_multimodal",
        lambda *args, **kwargs: "too large",
    )
    monkeypatch.setattr(
        vision_voice.voice_router,
        "text_to_speech",
        lambda *args, **kwargs: _wav(),
    )

    output = _run(
        vision_voice.multimodal_generate(
            "q",
            [_image()],
            provider="vision-provider",
            max_output_bytes=2,
        )
    )
    streaming = _run(
        vision_voice.voice_synthesize(
            "hello", provider="voice-provider", stream=True
        )
    )
    sample_rate = _run(
        vision_voice.voice_synthesize(
            "hello",
            provider="voice-provider",
            sample_rate_hz=vision_voice.MAX_SAMPLE_RATE_HZ + 1,
        )
    )

    assert output["error"]["code"] == "output_limit_exceeded"
    assert streaming["error"]["code"] == "streaming_unsupported"
    assert sample_rate["error"]["code"] == "invalid_request"


def test_timeout_cancellation_and_provider_errors_are_safe(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "vision-provider",
                "multimodal_router",
                Operation.VISION_GENERATE,
                media_types=("image/png",),
            )
        )
    )
    monkeypatch.setattr(
        vision_voice.multimodal_router,
        "generate_multimodal",
        lambda *args, **kwargs: (time.sleep(0.05) or "late"),
    )
    timed_out = _run(
        vision_voice.multimodal_generate(
            "q", [_image()], provider="vision-provider", timeout=0.005
        )
    )
    assert timed_out["error"]["code"] == "timeout"

    def raises_secret(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("Bearer provider-private-secret")

    monkeypatch.setattr(
        vision_voice.multimodal_router, "generate_multimodal", raises_secret
    )
    failed = _run(
        vision_voice.multimodal_generate(
            "q", [_image()], provider="vision-provider"
        )
    )
    assert failed["error"]["code"] == "router_error"
    assert failed["error"]["cause"] == "RuntimeError"
    assert "provider-private-secret" not in json.dumps(failed)

    started = threading.Event()
    release = threading.Event()

    def blocking(*args: Any, **kwargs: Any) -> str:
        started.set()
        release.wait(1)
        return "released"

    monkeypatch.setattr(
        vision_voice.multimodal_router, "generate_multimodal", blocking
    )

    async def cancel_call() -> None:
        task = asyncio.create_task(
            vision_voice.multimodal_generate(
                "q", [_image()], provider="vision-provider", timeout=1
            )
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    asyncio.run(cancel_call())
