"""Cross-surface conformance and opt-in live smokes for the AI service catalog.

The default suite is deliberately offline.  It uses immutable fixture records
and injected fake providers; the five live checks run only when their modality
is named in ``IPFS_ACCELERATE_PY_AI_CATALOG_LIVE``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import textwrap
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import pytest

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py import (
    embeddings_router,
    llm_router,
    multimodal_router,
    voice_router,
)
from ipfs_accelerate_py.api_backends.api_models_registry import (
    APIModelsRegistry,
    api_models,
)
from ipfs_accelerate_py.api_integrations.model_registry import (
    APIModelRegistry,
    LEGACY_REGISTRY_DEPRECATION,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    build_ai_catalog_v1_descriptor,
    validate_ai_catalog_payload,
)
from ipfs_accelerate_py.mcp_server.tools.ai_router_tools import (
    text_embedding,
    vision_voice,
)
from ipfs_accelerate_py.mcp_server.tools.model_tools import native_model_tools
from ipfs_accelerate_py.mcplusplus_module.service_registry import ServiceRecord
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
from ipfs_accelerate_py.model_catalog.sources.routers import RouterCatalogSource
from ipfs_accelerate_py.model_manager import ModelManager


CATALOG_MCP_NAMES = frozenset(
    {
        "model_catalog_list_services",
        "model_catalog_list_models",
        "model_catalog_get",
        "model_catalog_resolve",
        "model_catalog_health",
        "model_catalog_refresh",
    }
)
INVOKE_MCP_NAMES = frozenset(
    {
        "llm_generate",
        "embeddings_generate",
        "multimodal_generate",
        "voice_transcribe",
        "voice_synthesize",
    }
)
MCP_COMPATIBILITY_NAMES = frozenset(
    {"generate_text", "generate_embeddings", "generate_embedding"}
)
LEGACY_MODEL_MCP_NAMES = frozenset(
    {
        "model_search",
        "model_recommend",
        "model_get_details",
        "model_get_stats",
        "model_list_served",
        "model_get_served",
    }
)

_ROUTERS = (
    ("llm", llm_router, "llm_router"),
    ("embeddings", embeddings_router, "embeddings_router"),
    ("multimodal", multimodal_router, "multimodal_router"),
    ("voice", voice_router, "voice_router"),
)


def _run(awaitable: Any) -> Dict[str, Any]:
    return asyncio.run(awaitable)


def _text_snapshot() -> CatalogSnapshot:
    capability = CapabilityDescriptor(
        operations=(Operation.TEXT_GENERATE, Operation.TEXT_CHAT),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
        max_context_tokens=4096,
    )
    provenance = (Provenance(source="conformance.fixture"),)
    state = OperationalState(
        known=True,
        configured=True,
        authorized=True,
        reachable=True,
        healthy=True,
        routable=True,
    )
    provider = ProviderDescriptor(
        name="openai",
        aliases=("openai_api",),
        capabilities=(capability,),
        state=state,
        provenance=provenance,
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="gpt-4",
        aliases=("gpt4",),
        capabilities=(capability,),
        state=state,
        provenance=provenance,
        labels=(("invocation_model", "gpt-4"),),
    )
    binding = RouterBinding(
        router="llm_router",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        operations=(Operation.TEXT_GENERATE, Operation.TEXT_CHAT),
        state=state,
        provenance=provenance,
        labels=(("invocation_model", "gpt-4"),),
    )
    return CatalogSnapshot(
        providers=(provider,),
        models=(model,),
        bindings=(binding,),
    )


class _FixtureRouter:
    __name__ = "llm_router"

    def __init__(self, snapshot: CatalogSnapshot) -> None:
        self._snapshot = snapshot

    def get_catalog_snapshot(self) -> CatalogSnapshot:
        return self._snapshot


class _ToolRecorder:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[str(definition["name"])] = definition


def _invocation_model(model: ModelDescriptor, binding: RouterBinding) -> str:
    labels = dict(model.labels)
    labels.update(dict(binding.labels))
    return str(
        labels.get(
            "invocation_model",
            labels.get("router_model_name", model.name),
        )
    )


@dataclass(frozen=True)
class _BindingConformanceRow:
    surface: str
    binding_id: str
    provider_id: str
    model_id: str
    operation: Operation


def _binding_rows() -> Tuple[_BindingConformanceRow, ...]:
    """Generate the drift matrix from current router-owned snapshots."""

    rows = []
    for surface, router, expected_router in _ROUTERS:
        snapshot = router.get_catalog_snapshot()
        providers = {record.provider_id: record for record in snapshot.providers}
        models = {record.model_id: record for record in snapshot.models}
        assert all(binding.router == expected_router for binding in snapshot.bindings)
        for binding in snapshot.bindings:
            provider = providers[binding.provider_id]
            model = models[binding.model_id]
            assert (
                RouterBinding.from_dict(binding.to_dict()).binding_id
                == binding.binding_id
            )
            for operation in binding.operations:
                resolved = router.resolve_model(
                    _invocation_model(model, binding),
                    provider=provider.name,
                    operation=operation,
                )
                assert resolved.provider_id == binding.provider_id
                assert resolved.model_id == binding.model_id
                rows.append(
                    _BindingConformanceRow(
                        surface=surface,
                        binding_id=binding.binding_id,
                        provider_id=resolved.provider_id,
                        model_id=resolved.model_id,
                        operation=operation,
                    )
                )
    return tuple(rows)


def test_every_declared_router_binding_resolves_to_its_canonical_identity() -> None:
    rows = _binding_rows()

    assert {row.surface for row in rows} == {
        "llm",
        "embeddings",
        "multimodal",
        "voice",
    }
    assert len({(row.binding_id, row.operation) for row in rows}) == len(rows)
    assert all(row.provider_id.startswith("provider_") for row in rows)
    assert all(row.model_id.startswith("model_") for row in rows)


def test_router_manager_legacy_mcp_and_mcplusplus_identity_revision_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _text_snapshot()
    router = _FixtureRouter(snapshot)
    source = RouterCatalogSource(router, source="routers/llm_router")
    catalog = AIServiceCatalog({source.source: source})

    for name in (
        "HAVE_STORAGE_WRAPPER",
        "HAVE_IPFS_KIT_STORAGE",
        "HAVE_DATASETS_INTEGRATION",
        "HAVE_GRAPHRAG",
    ):
        monkeypatch.setattr(model_manager_module, name, False)
    manager = ModelManager(
        storage_path=str(tmp_path / "catalog.json"),
        use_database=False,
        enable_ipfs=False,
        catalog=catalog,
        project_legacy_models=False,
    )
    monkeypatch.setattr(
        model_manager_module,
        "get_default_model_manager",
        lambda: manager,
    )
    try:
        legacy = APIModelRegistry(catalog=catalog)
        backend_projection = api_models(registry=legacy)
        canonical_model = snapshot.models[0]

        mcp_models = _run(native_model_tools.model_catalog_list_models(limit=10))
        mcp_services = _run(
            native_model_tools.model_catalog_list_services(limit=10)
        )
        advertisement = ServiceRecord(
            service_name="ipfs-accelerate-mcp",
            peer_id="conformance-peer",
            issuer="conformance-peer",
            multiaddrs=["/memory/conformance-peer"],
            catalog_cid=manager.catalog_revision,
            catalog_revision=manager.catalog_revision,
            operation_summary=["text.chat", "text.generate"],
            interface_cids=["cidv1-ai-catalog"],
            issued_at=100.0,
            expires_at=400.0,
        )
        advertisement.sign()

        assert snapshot.revision == catalog.revision == manager.catalog_revision
        assert legacy.catalog_revision == backend_projection.catalog_revision
        assert legacy.catalog_revision == manager.catalog_revision
        assert legacy.resolve_model_id("openai/gpt-4") == canonical_model.model_id
        assert manager.list_catalog_models().items[0].model_id == canonical_model.model_id
        assert mcp_models["items"][0]["model_id"] == canonical_model.model_id
        assert (
            mcp_services["items"][0]["provider_id"]
            == snapshot.providers[0].provider_id
        )
        assert mcp_models["catalog_revision"] == manager.catalog_revision
        assert mcp_services["catalog_revision"] == manager.catalog_revision
        assert (
            validate_ai_catalog_payload(
                "model_catalog_list_models",
                mcp_models,
                direction="output",
            )["catalog_revision"]
            == manager.catalog_revision
        )
        assert (
            validate_ai_catalog_payload(
                "model_catalog_list_services",
                mcp_services,
                direction="output",
            )["items"][0]["provider_id"]
            == snapshot.providers[0].provider_id
        )
        assert advertisement.catalog_cid == manager.catalog_revision
        assert advertisement.catalog_revision == manager.catalog_revision
        assert advertisement.verify_signature()
    finally:
        manager.close()


def test_public_python_mcp_and_mcplusplus_compatibility_fixture() -> None:
    python_names: Mapping[Any, Tuple[str, ...]] = {
        llm_router: (
            "list_providers",
            "get_provider_descriptor",
            "list_models",
            "resolve_model",
            "get_catalog_snapshot",
            "catalog_snapshot",
            "generate_text",
            "register_llm_provider",
        ),
        embeddings_router: (
            "list_providers",
            "get_provider_descriptor",
            "list_models",
            "resolve_model",
            "get_catalog_snapshot",
            "catalog_snapshot",
            "embed_text",
            "embed_texts",
            "register_embeddings_provider",
        ),
        multimodal_router: (
            "list_providers",
            "get_provider_descriptor",
            "list_models",
            "resolve_model",
            "get_catalog_snapshot",
            "catalog_snapshot",
            "generate_multimodal",
            "register_multimodal_provider",
        ),
        voice_router: (
            "list_providers",
            "get_provider_descriptor",
            "list_models",
            "resolve_model",
            "get_catalog_snapshot",
            "catalog_snapshot",
            "speech_to_text",
            "text_to_speech",
            "get_tts_provider",
            "register_tts_provider",
        ),
    }
    for module, names in python_names.items():
        assert all(callable(getattr(module, name, None)) for name in names)
    assert voice_router.TTSProvider is voice_router.VoiceProvider
    assert voice_router.get_tts_provider is voice_router.get_voice_provider
    assert voice_router.register_tts_provider is voice_router.register_voice_provider
    assert (
        voice_router.clear_tts_router_caches
        is voice_router.clear_voice_router_caches
    )
    assert (
        multimodal_router.generate_text
        is multimodal_router.generate_multimodal_text
    )
    assert ModelManager.refresh_catalog is ModelManager.refresh
    assert APIModelsRegistry is api_models

    recorder = _ToolRecorder()
    native_model_tools.register_native_model_tools(recorder)
    text_embedding.register_native_ai_router_tools(recorder)
    vision_voice.register_native_ai_router_tools(recorder)
    assert (
        CATALOG_MCP_NAMES
        | INVOKE_MCP_NAMES
        | MCP_COMPATIBILITY_NAMES
        | LEGACY_MODEL_MCP_NAMES
    ) <= set(recorder.tools)

    mcpp_operations = {
        method["operation"]
        for method in build_ai_catalog_v1_descriptor()["methods"]
    }
    assert mcpp_operations == CATALOG_MCP_NAMES | INVOKE_MCP_NAMES
    assert LEGACY_REGISTRY_DEPRECATION == {
        "deprecated": True,
        "replacement": "ModelManager model catalog",
        "removal_scheduled": False,
        "reversible": True,
    }


def test_cold_import_listing_and_tool_registration_are_side_effect_free() -> None:
    code = textwrap.dedent(
        r"""
        import json
        import os
        import sys
        import ctypes.util
        import subprocess as subprocess_module

        forbidden_events = []
        credential_paths = (
            "/.aws/credentials",
            "/.config/gcloud/",
            "/.huggingface/token",
            "/.netrc",
            "/auth.json",
            "/token.json",
        )
        model_suffixes = (".bin", ".gguf", ".pt", ".pth", ".safetensors")

        def audit(event, args):
            if event in {
                "subprocess.Popen",
                "os.system",
                "socket.connect",
                "socket.getaddrinfo",
            }:
                forbidden_events.append([event, repr(args)])
                raise RuntimeError("forbidden side effect: " + event)
            if event == "open" and args:
                path = str(args[0]).replace("\\", "/").casefold()
                if any(item in path for item in credential_paths):
                    forbidden_events.append("credential.read")
                    raise RuntimeError("forbidden credential read")
                if path.endswith(model_suffixes):
                    forbidden_events.append("model.load")
                    raise RuntimeError("forbidden model load")

        # Optional dependencies ask ctypes to locate libc and pthread while
        # importing.  On Linux, ctypes may implement those lookups by spawning
        # ldconfig.  Pin the well-known sonames so the audit below measures
        # catalog behavior rather than platform-library probes.
        original_find_library = ctypes.util.find_library
        def safe_find_library(name):
            pinned = {"c": "libc.so.6", "pthread": "libpthread.so.0"}
            return pinned.get(name) or original_find_library(name)
        ctypes.util.find_library = safe_find_library
        # NumPy's test helper performs an eager `lscpu` SVE probe when it is
        # imported through sentence-transformers.  Return the equivalent
        # negative probe result without starting a process; any other process
        # attempt still reaches the audit hook and fails the test.
        original_run = subprocess_module.run
        subprocess_module.run = lambda command, *args, **kwargs: (
            subprocess_module.CompletedProcess(command, 0, "", "")
            if command == "lscpu"
            else original_run(command, *args, **kwargs)
        )
        sys.addaudithook(audit)

        from ipfs_accelerate_py import (
            embeddings_router,
            llm_router,
            multimodal_router,
            voice_router,
        )
        from ipfs_accelerate_py.api_backends.api_models_registry import api_models
        from ipfs_accelerate_py.api_integrations.model_registry import APIModelRegistry
        from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
            build_ai_catalog_v1_descriptor,
        )
        from ipfs_accelerate_py.mcp_server.tools.ai_router_tools import (
            text_embedding,
            vision_voice,
        )
        from ipfs_accelerate_py.mcp_server.tools.model_tools import native_model_tools
        from ipfs_accelerate_py.mcplusplus_module.service_registry import ServiceRegistry
        from ipfs_accelerate_py.model_manager import ModelManager

        class Recorder:
            def __init__(self):
                self.names = []
            def register_tool(self, **definition):
                self.names.append(definition["name"])

        snapshots = [
            router.get_catalog_snapshot()
            for router in (
                llm_router,
                embeddings_router,
                multimodal_router,
                voice_router,
            )
        ]
        legacy = APIModelRegistry()
        backend_projection = api_models(registry=legacy)
        assert legacy.list_models()
        assert backend_projection.model_lists
        assert callable(ModelManager.list_services)
        assert build_ai_catalog_v1_descriptor()["methods"]
        assert ServiceRegistry is not None
        recorder = Recorder()
        native_model_tools.register_native_model_tools(recorder)
        text_embedding.register_native_ai_router_tools(recorder)
        vision_voice.register_native_ai_router_tools(recorder)
        print(json.dumps({
            "events": forbidden_events,
            "revisions": [item.revision for item in snapshots],
            "tools": sorted(recorder.names),
        }, sort_keys=True))
        """
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(
            marker in key.casefold()
            for marker in ("token", "secret", "password", "api_key", "apikey")
        )
    }
    environment["IPFS_ACCELERATE_PY_ROUTER_CACHE"] = "0"
    environment["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "0"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["events"] == [], payload
    assert len(payload["revisions"]) == 4
    assert CATALOG_MCP_NAMES | INVOKE_MCP_NAMES <= set(payload["tools"])


def test_default_invocation_conformance_uses_only_injected_fake_providers() -> None:
    calls = []

    class FakeText:
        def generate(self, prompt: str, **kwargs: Any) -> str:
            calls.append(("text", prompt, kwargs.get("model_name")))
            return "fake-text"

    class FakeEmbeddings:
        router_provider_name = "fake-embeddings"

        def embed_texts(self, texts: Iterable[str], **kwargs: Any) -> list[list[float]]:
            values = list(texts)
            calls.append(("embeddings", tuple(values), kwargs.get("model_name")))
            return [[float(index), 1.0] for index, _ in enumerate(values)]

    class FakeMultimodal:
        def generate(self, prompt: str, **kwargs: Any) -> str:
            calls.append(("multimodal", prompt, kwargs.get("model_name")))
            return "fake-vision"

    class FakeVoice:
        def transcribe(self, audio: Any, **kwargs: Any) -> str:
            calls.append(("transcription", len(audio), kwargs.get("model_name")))
            return "fake-transcript"

        def synthesize(self, text: str, **kwargs: Any) -> bytes:
            calls.append(("synthesis", text, kwargs.get("model_name")))
            return b"fake-audio"

    assert (
        llm_router.generate_text(
            "offline",
            model_name="fixture-text",
            provider_instance=FakeText(),
        )
        == "fake-text"
    )
    assert embeddings_router.embed_texts(
        ["offline"],
        model_name="fixture-embedding",
        provider_instance=FakeEmbeddings(),
    ) == [[0.0, 1.0]]
    assert (
        multimodal_router.generate_multimodal(
            "offline",
            image=b"image",
            model_name="fixture-vision",
            provider_instance=FakeMultimodal(),
        )
        == "fake-vision"
    )
    voice = FakeVoice()
    assert (
        voice_router.speech_to_text(
            b"audio",
            model_name="fixture-stt",
            provider_instance=voice,
        )
        == "fake-transcript"
    )
    assert (
        voice_router.text_to_speech(
            "offline",
            model_name="fixture-tts",
            provider_instance=voice,
        )
        == b"fake-audio"
    )
    assert [call[0] for call in calls] == [
        "text",
        "embeddings",
        "multimodal",
        "transcription",
        "synthesis",
    ]


def _live_modalities() -> frozenset[str]:
    raw = os.getenv("IPFS_ACCELERATE_PY_AI_CATALOG_LIVE", "")
    selected = {item.strip().casefold() for item in raw.split(",") if item.strip()}
    if selected & {"1", "all", "true", "yes"}:
        return frozenset(
            {"text", "embeddings", "multimodal", "transcription", "synthesis"}
        )
    return frozenset(selected)


def _live_value(modality: str, suffix: str) -> str | None:
    value = os.getenv(
        "IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_%s_%s"
        % (modality.upper(), suffix.upper())
    )
    return value.strip() if value and value.strip() else None


def _silent_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * 1600)
    return output.getvalue()


@pytest.mark.parametrize(
    "modality",
    ("text", "embeddings", "multimodal", "transcription", "synthesis"),
)
def test_opt_in_live_provider_smoke(modality: str) -> None:
    """Exercise only explicitly selected modalities; unavailable ones may be omitted."""

    if modality not in _live_modalities():
        pytest.skip(
            "select this live smoke with "
            "IPFS_ACCELERATE_PY_AI_CATALOG_LIVE=%s" % modality
        )
    provider = _live_value(modality, "provider")
    model = _live_value(modality, "model")

    if modality == "text":
        result = llm_router.generate_text(
            "Reply with catalog-ok.",
            provider=provider,
            model_name=model,
            max_tokens=16,
            timeout=30,
        )
        assert isinstance(result, str) and result.strip()
    elif modality == "embeddings":
        result = embeddings_router.embed_texts(
            ["catalog smoke"],
            provider=provider,
            model_name=model,
            timeout=30,
        )
        assert len(result) == 1 and len(result[0]) > 0
        assert all(isinstance(value, (int, float)) for value in result[0])
    elif modality == "multimodal":
        result = multimodal_router.generate_multimodal(
            "Describe this tiny test image.",
            image=(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde"
            ),
            provider=provider,
            model_name=model,
            max_tokens=32,
            timeout=30,
        )
        assert isinstance(result, str) and result.strip()
    elif modality == "transcription":
        result = voice_router.speech_to_text(
            _silent_wav(),
            provider=provider,
            model_name=model,
            language="en",
            timeout=30,
        )
        assert isinstance(result, str)
    else:
        result = voice_router.text_to_speech(
            "Catalog smoke.",
            provider=provider,
            model_name=model,
            output_format="wav",
            timeout=30,
        )
        assert isinstance(result, bytes) and result
