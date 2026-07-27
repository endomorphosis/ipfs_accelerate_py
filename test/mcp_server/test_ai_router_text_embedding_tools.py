"""Contract tests for bounded MCP text and embedding router invocation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pytest

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py.mcp_server.tools.ai_router_tools import text_embedding
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
    dimensions: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_input_bytes: int = 4096,
    max_output_bytes: int = 4096,
    labels: Tuple[Tuple[str, str], ...] = (),
) -> Tuple[Any, ...]:
    output_modality = (
        Modality.EMBEDDING
        if operation is Operation.EMBEDDING_GENERATE
        else Modality.TEXT
    )
    operations = (
        (operation, Operation.BATCH)
        if max_batch_size is not None
        else (operation,)
    )
    capability = CapabilityDescriptor(
        operations=operations,
        input_modalities=(Modality.TEXT,),
        output_modalities=(output_modality,),
        max_batch_size=max_batch_size,
        max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
        embedding_dimensions=dimensions,
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
        labels=labels,
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
        operations=operations,
        priority=priority,
        state=state,
        provenance=provenance,
        labels=(("invocation_model", model_name),),
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
        self.load_calls = 0

    def load(self) -> CatalogSourceResult:
        self.load_calls += 1
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

    for manager in managers:
        manager.close()


class ToolRecorder:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[definition["name"]] = definition


def test_registration_is_cold_bounded_and_includes_compatibility_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_manager_module,
        "get_default_model_manager",
        lambda: pytest.fail("registration must not resolve the catalog"),
    )
    registry = ToolRecorder()

    text_embedding.register_native_ai_router_tools(registry)

    assert {
        "llm_generate",
        "embeddings_generate",
        "generate_text",
        "generate_embeddings",
        "generate_embedding",
    } <= set(registry.tools)
    text_schema = registry.tools["llm_generate"]["input_schema"]
    embedding_schema = registry.tools["embeddings_generate"]["input_schema"]
    assert text_schema["additionalProperties"] is False
    assert text_schema["properties"]["timeout"]["maximum"] == 120.0
    assert (
        embedding_schema["properties"]["texts"]["maxItems"]
        == text_embedding.MAX_INPUT_ITEMS
    )
    assert (
        embedding_schema["properties"]["dimensions"]["maximum"]
        == text_embedding.MAX_EMBEDDING_DIMENSIONS
    )


def test_text_routes_through_llm_router_with_revision_receipt_and_mcp_parity(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = install_manager(
        _snapshot(
            _records(
                "text-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
                model_name="chat-model",
            )
        )
    )
    calls = []
    trace: Dict[str, Any] = {}

    def fake_generate(prompt: str, **kwargs: Any) -> str:
        calls.append((prompt, dict(kwargs)))
        trace.update(
            effective_provider_name="text-provider",
            effective_model_name="chat-model",
        )
        return "bounded response"

    monkeypatch.setattr(text_embedding.llm_router, "generate_text", fake_generate)
    monkeypatch.setattr(
        text_embedding.llm_router,
        "get_last_generation_trace",
        lambda: dict(trace),
    )
    direct = _run(
        text_embedding.llm_generate(
            "private prompt",
            service="text-provider",
            model="chat-model",
        )
    )
    registry = ToolRecorder()
    text_embedding.register_native_ai_router_tools(registry)
    through_mcp = _run(
        registry.tools["llm_generate"]["func"](
            prompt="private prompt",
            service="text-provider",
            model="chat-model",
        )
    )

    assert direct == through_mcp
    assert direct["status"] == "success"
    assert direct["text"] == "bounded response"
    assert direct["catalog_revision"] == manager.catalog_revision
    assert direct["selected_binding"]["router"] == "llm_router"
    assert direct["receipt"]["catalog_revision"] == manager.catalog_revision
    assert direct["receipt"]["input"] == {"count": 1, "text_bytes": 14}
    assert "private prompt" not in json.dumps(
        {key: value for key, value in direct.items() if key != "text"}
    )
    assert calls[0][0] == "private prompt"
    assert calls[0][1]["provider"] == "text-provider"
    assert calls[0][1]["model_name"] == "chat-model"
    assert calls[0][1]["allow_local_fallback"] is False
    assert calls[0][1]["disable_model_retry"] is True


def test_embeddings_route_through_canonical_router_and_validate_dimensions(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = install_manager(
        _snapshot(
            _records(
                "vector-provider",
                "embeddings_router",
                Operation.EMBEDDING_GENERATE,
                model_name="vector-model",
                dimensions=3,
                max_batch_size=4,
            )
        )
    )
    calls = []
    trace: Dict[str, Any] = {}

    def fake_embed(texts: Any, **kwargs: Any) -> Any:
        calls.append((list(texts), dict(kwargs)))
        trace.update(
            provider_used="vector-provider",
            model_name="vector-model",
            fallback_used=False,
        )
        return [[1, 2.5, 3] for _ in texts]

    monkeypatch.setattr(text_embedding.embeddings_router, "embed_texts", fake_embed)
    monkeypatch.setattr(
        text_embedding.embeddings_router,
        "get_last_embedding_trace",
        lambda: dict(trace),
    )

    result = _run(
        text_embedding.embeddings_generate(
            ["alpha", "beta"],
            provider="vector-provider",
            dimensions=3,
        )
    )

    assert result["status"] == "success"
    assert result["embeddings"] == [[1.0, 2.5, 3.0], [1.0, 2.5, 3.0]]
    assert result["dimensions"] == 3
    assert result["catalog_revision"] == manager.catalog_revision
    assert result["selected_binding"]["router"] == "embeddings_router"
    assert calls == [
        (
            ["alpha", "beta"],
            {
                "model_name": "vector-model",
                "provider": "vector-provider",
                "device": None,
                "dimensions": 3,
            },
        )
    ]


def test_constraint_mismatch_and_policy_denial_do_not_invoke_router(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "allowed-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
                labels=(("policy.tenant", "allowed"),),
            ),
            _records(
                "other-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
            ),
        )
    )
    calls = 0

    def forbidden(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return "unexpected"

    monkeypatch.setattr(text_embedding.llm_router, "generate_text", forbidden)

    mismatch = _run(
        text_embedding.llm_generate(
            "hello",
            service="allowed-provider",
            provider="other-provider",
        )
    )
    denied = _run(
        text_embedding.llm_generate(
            "hello",
            provider="allowed-provider",
            policy={"tenant": "private-policy-value"},
        )
    )

    assert mismatch["error"]["code"] == "selection_denied"
    assert denied["error"]["code"] == "selection_denied"
    assert calls == 0
    assert "private-policy-value" not in json.dumps(denied)
    assert mismatch["catalog_revision"]
    assert denied["catalog_revision"]


def test_fallback_is_confined_to_candidates_from_captured_revision(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = install_manager(
        _snapshot(
            _records(
                "first-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
                priority=10,
                model_name="first-model",
            ),
            _records(
                "second-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
                priority=0,
                model_name="second-model",
            ),
        )
    )
    trace = {
        "effective_provider_name": "second-provider",
        "effective_model_name": "second-model",
    }
    monkeypatch.setattr(
        text_embedding.llm_router,
        "generate_text",
        lambda *args, **kwargs: "fallback output",
    )
    monkeypatch.setattr(
        text_embedding.llm_router,
        "get_last_generation_trace",
        lambda: dict(trace),
    )

    blocked = _run(text_embedding.llm_generate("hello"))
    allowed = _run(
        text_embedding.llm_generate("hello", allow_fallback=True)
    )

    assert blocked["error"]["code"] == "fallback_boundary_exceeded"
    assert allowed["status"] == "success"
    assert allowed["receipt"]["fallback"]["used"] is True
    assert allowed["receipt"]["catalog_revision"] == manager.catalog_revision
    assert (
        allowed["selected_binding"]["binding_id"]
        == allowed["receipt"]["fallback"]["boundary_binding_ids"][1]
    )

    trace.update(
        effective_provider_name="not-in-catalog",
        effective_model_name="unknown",
    )
    escaped = _run(
        text_embedding.llm_generate("hello", allow_fallback=True)
    )
    assert escaped["error"]["code"] == "fallback_boundary_exceeded"


def test_batch_text_dimension_output_and_streaming_bounds_fail_closed(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "vector-provider",
                "embeddings_router",
                Operation.EMBEDDING_GENERATE,
                dimensions=3,
                max_batch_size=2,
            ),
            _records(
                "text-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
            ),
        )
    )
    embedding_calls = 0
    text_calls = 0

    def fake_embed(texts: Any, **kwargs: Any) -> Any:
        nonlocal embedding_calls
        embedding_calls += 1
        return [[1.0, 2.0] for _ in texts]

    def fake_text(prompt: str, **kwargs: Any) -> str:
        nonlocal text_calls
        text_calls += 1
        return "too large"

    monkeypatch.setattr(text_embedding.embeddings_router, "embed_texts", fake_embed)
    monkeypatch.setattr(text_embedding.llm_router, "generate_text", fake_text)

    global_batch = _run(
        text_embedding.embeddings_generate(
            ["x"] * (text_embedding.MAX_INPUT_ITEMS + 1)
        )
    )
    catalog_batch = _run(
        text_embedding.embeddings_generate(
            ["a", "b", "c"],
            provider="vector-provider",
        )
    )
    dimensions = _run(
        text_embedding.embeddings_generate(
            ["a"],
            provider="vector-provider",
            dimensions=3,
        )
    )
    output = _run(
        text_embedding.llm_generate(
            "hello",
            provider="text-provider",
            max_output_bytes=2,
        )
    )
    streaming = _run(text_embedding.llm_generate("hello", stream=True))

    assert global_batch["error"]["code"] == "input_limit_exceeded"
    assert catalog_batch["error"]["code"] == "input_limit_exceeded"
    assert dimensions["error"]["code"] == "dimension_mismatch"
    assert output["error"]["code"] == "output_limit_exceeded"
    assert streaming["error"]["code"] == "streaming_unsupported"
    assert embedding_calls == 1
    assert text_calls == 1


def test_timeout_cancellation_and_provider_errors_are_safe(
    install_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_manager(
        _snapshot(
            _records(
                "text-provider",
                "llm_router",
                Operation.TEXT_GENERATE,
            )
        )
    )

    monkeypatch.setattr(
        text_embedding.llm_router,
        "generate_text",
        lambda *args, **kwargs: (time.sleep(0.05) or "late"),
    )
    timed_out = _run(
        text_embedding.llm_generate(
            "hello",
            provider="text-provider",
            timeout=0.005,
        )
    )
    assert timed_out["error"]["code"] == "timeout"

    def raises_secret(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("Bearer provider-private-secret")

    monkeypatch.setattr(text_embedding.llm_router, "generate_text", raises_secret)
    failed = _run(
        text_embedding.llm_generate("hello", provider="text-provider")
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

    monkeypatch.setattr(text_embedding.llm_router, "generate_text", blocking)

    async def cancel_call() -> None:
        task = asyncio.create_task(
            text_embedding.llm_generate(
                "hello",
                provider="text-provider",
                timeout=1,
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


def test_compatibility_aliases_delegate_to_canonical_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_calls = []
    embedding_calls = []

    async def fake_llm(**kwargs: Any) -> Dict[str, Any]:
        text_calls.append(kwargs)
        return {"status": "success", "sentinel": "text"}

    async def fake_embeddings(**kwargs: Any) -> Dict[str, Any]:
        embedding_calls.append(kwargs)
        return {"status": "success", "sentinel": "embedding"}

    monkeypatch.setattr(text_embedding, "llm_generate", fake_llm)
    monkeypatch.setattr(text_embedding, "embeddings_generate", fake_embeddings)

    text_result = _run(
        text_embedding.generate_text("hello", model="auto", timeout=4)
    )
    batch_result = _run(
        text_embedding.generate_embeddings(
            ["one", "two"],
            model_name="vector-model",
        )
    )
    single_result = _run(
        text_embedding.generate_embedding(
            "one",
            model_name="vector-model",
        )
    )

    assert text_result["sentinel"] == "text"
    assert text_calls == [{"prompt": "hello", "model": None, "timeout": 4}]
    assert batch_result["sentinel"] == single_result["sentinel"] == "embedding"
    assert embedding_calls == [
        {"texts": ["one", "two"], "model": "vector-model"},
        {"texts": ["one"], "model": "vector-model"},
    ]
