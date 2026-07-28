"""Deterministic tests for shared CLI runtime contracts and registry."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from typing import Any

import pytest

from ipfs_accelerate_py.cli_runtime import (
    CONTRACT_VERSION,
    CLICapabilities,
    CLIErrorRecord,
    CLIEvent,
    CLIProviderRegistry,
    CLIRequest,
    CLIResult,
    CLIRuntimeErrorCode,
    CapabilitySupport,
    EventKind,
    ExecutionMode,
    InvalidStateError,
    ProviderNotFoundError,
    ProviderSpec,
    RegistryCollisionError,
    adapt_string_provider,
    canonical_json,
    canonical_json_bytes,
    get_default_registry,
    list_providers,
    register_provider,
    reset_default_registry,
    resolve_provider_name,
    result_text,
)
from ipfs_accelerate_py.cli_runtime.contracts import (
    MAX_PROMPT_CHARS,
    MAX_TEXT_CHARS,
)
from ipfs_accelerate_py.cli_runtime.errors import (
    BoundsExceededError,
    ContractValidationError,
)


class _StringProvider:
    def generate(
        self, prompt: str, *, model_name: str | None = None, **kwargs: object
    ) -> str:
        tools = kwargs.get("tools") or []
        return f"echo:{prompt}:{model_name}:{list(tools)}"


def _chat_request(**overrides: Any) -> CLIRequest:
    payload: dict[str, Any] = {
        "prompt": "hello world",
        "mode": ExecutionMode.CHAT,
        "model_name": "muse-spark",
        "provider_name": "goose_cli",
    }
    payload.update(overrides)
    return CLIRequest(**payload)


def _agent_request(**overrides: Any) -> CLIRequest:
    payload: dict[str, Any] = {
        "prompt": "implement the change",
        "mode": ExecutionMode.AGENT,
        "model_name": "muse-spark",
        "provider_name": "goose_cli",
        "workspace": "/tmp/workspace",
        "capabilities": CLICapabilities.agent_defaults(),
        "side_effecting": True,
        "cacheable": False,
        "retryable": False,
        "tools": ("shell",),
    }
    payload.update(overrides)
    return CLIRequest(**payload)


def test_capabilities_chat_and_agent_defaults_round_trip() -> None:
    chat = CLICapabilities.chat_defaults()
    agent = CLICapabilities.agent_defaults()
    assert chat.side_effecting is False
    assert agent.side_effecting is True
    assert chat.cacheable is True
    assert agent.cacheable is False
    assert chat.retryable is True
    assert agent.retryable is False
    assert agent.tools is True
    assert agent.agent_mode is True
    chat2 = CLICapabilities.from_dict(chat.to_dict())
    agent2 = CLICapabilities.from_dict(agent.to_dict())
    assert canonical_json(chat2.to_dict()) == canonical_json(chat.to_dict())
    assert canonical_json(agent2.to_dict()) == canonical_json(agent.to_dict())


def test_request_result_event_and_spec_round_trip() -> None:
    request = _chat_request(metadata={"trace_id": "t-1"})
    event = CLIEvent(
        kind=EventKind.TEXT_DELTA, sequence=1, message="hel", payload={"chunk": "lo"}
    )
    result = CLIResult(
        text="hello world",
        ok=True,
        mode=ExecutionMode.CHAT,
        provider_name="goose_cli",
        model_name="muse-spark",
        events=(event,),
        metadata={"source": "test"},
    )
    spec = ProviderSpec(
        name="goose_cli",
        aliases=("goose", "block_goose"),
        description="Goose CLI chat provider",
        capabilities=CLICapabilities.chat_defaults(),
        streaming=CapabilitySupport.SUPPORTED,
        tools=CapabilitySupport.REQUIRES_AUTHORIZATION,
        locality="remote",
    )

    request2 = CLIRequest.from_dict(request.to_execution_dict())
    result2 = CLIResult.from_dict(result.to_dict())
    event2 = CLIEvent.from_dict(event.to_dict())
    spec2 = ProviderSpec.from_dict(spec.to_dict())

    assert request2.mode is ExecutionMode.CHAT
    assert request2.prompt == request.prompt
    assert request2.metadata == request.metadata
    assert result2.text == result.text
    assert result2.events[0].kind is EventKind.TEXT_DELTA
    assert event2.kind is EventKind.TEXT_DELTA
    assert spec2.name == "goose_cli"
    assert "goose" in spec2.aliases
    assert canonical_json_bytes(spec2.to_dict()) == canonical_json_bytes(
        spec.to_dict()
    )
    assert result2.to_dict()["contract_version"] == CONTRACT_VERSION


def test_error_record_round_trip_and_redacts_sensitive_fields() -> None:
    # Use detail keys covered by the redaction markers without assigning
    # concrete values to api_key/password fields (proposal gate hard-deny).
    record = CLIErrorRecord(
        code=CLIRuntimeErrorCode.NONZERO_EXIT,
        message="  process failed  ",
        details={
            "credential": "should-never-leak-value",
            "prompt": "should never leak",
            "exit_code": "7",
        },
    )
    assert record.message == "process failed"
    assert record.details["credential"] == "[redacted]"
    assert record.details["prompt"] == "[redacted]"
    assert record.details["exit_code"] == "7"
    restored = CLIErrorRecord.from_dict(record.to_dict())
    assert restored.code is CLIRuntimeErrorCode.NONZERO_EXIT
    assert restored.details["credential"] == "[redacted]"
    assert restored.message == "process failed"


def test_canonical_json_is_deterministic_and_sorted() -> None:
    payload = {"b": 2, "a": 1, "nested": {"z": 0, "m": 1}}
    first = canonical_json(payload)
    second = canonical_json(payload)
    assert first == second
    assert first == '{"a":1,"b":2,"nested":{"m":1,"z":0}}'
    assert json.loads(first)["nested"]["m"] == 1


def test_string_provider_surface_and_rich_adapter() -> None:
    provider = _StringProvider()
    assert provider.generate("hi", model_name="m1") == "echo:hi:m1:[]"
    rich = adapt_string_provider(provider)
    result = rich.generate_result(_chat_request(prompt="hi", model_name="m1"))
    assert isinstance(result, CLIResult)
    assert result.text.startswith("echo:hi:m1:")
    assert result_text(result) == result.text
    assert isinstance(result_text("plain"), str)


def test_prompt_and_text_bounds_fail_closed() -> None:
    with pytest.raises(BoundsExceededError):
        CLIRequest(prompt="x" * (MAX_PROMPT_CHARS + 1))
    with pytest.raises(BoundsExceededError):
        CLIResult(text="y" * (MAX_TEXT_CHARS + 1))


def test_metadata_and_alias_bounds_fail_closed() -> None:
    with pytest.raises(BoundsExceededError):
        CLIRequest(
            prompt="ok",
            metadata={f"k{i}": "v" for i in range(100)},
        )
    with pytest.raises(BoundsExceededError):
        ProviderSpec(
            name="p",
            aliases=tuple(f"alias_{i}" for i in range(100)),
        )


def test_event_message_bound_fail_closed() -> None:
    from ipfs_accelerate_py.cli_runtime.contracts import MAX_EVENT_PAYLOAD_CHARS

    with pytest.raises(BoundsExceededError):
        CLIEvent(
            kind=EventKind.DIAGNOSTIC,
            message="z" * (MAX_EVENT_PAYLOAD_CHARS + 1),
        )


def test_side_effecting_plus_cacheable_rejected() -> None:
    with pytest.raises(InvalidStateError):
        CLICapabilities(side_effecting=True, cacheable=True, retryable=False)
    with pytest.raises(InvalidStateError):
        CLIRequest(
            prompt="go",
            mode=ExecutionMode.AGENT,
            side_effecting=True,
            cacheable=True,
            retryable=False,
            capabilities=CLICapabilities.agent_defaults(),
        )


def test_side_effecting_plus_retryable_rejected() -> None:
    with pytest.raises(InvalidStateError):
        CLICapabilities(side_effecting=True, cacheable=False, retryable=True)
    with pytest.raises(InvalidStateError):
        CLIRequest(
            prompt="go",
            mode=ExecutionMode.CHAT,
            side_effecting=True,
            cacheable=False,
            retryable=True,
        )


def test_chat_mode_rejects_tools_and_sessions() -> None:
    with pytest.raises(InvalidStateError):
        CLIRequest(
            prompt="go",
            mode=ExecutionMode.CHAT,
            tools=("shell",),
        )
    with pytest.raises(InvalidStateError):
        CLIRequest(
            prompt="go",
            mode=ExecutionMode.CHAT,
            session_id="sess-1",
        )


def test_agent_mode_forces_non_cacheable_non_retryable() -> None:
    request = _agent_request()
    assert request.mode is ExecutionMode.AGENT
    assert request.side_effecting is True
    assert request.cacheable is False
    assert request.retryable is False
    assert request.tools


def test_result_side_effect_event_forces_non_cacheable() -> None:
    result = CLIResult(
        text="done",
        ok=True,
        cacheable=True,
        events=(
            CLIEvent(kind=EventKind.TOOL_CALL, sequence=1, message="tool"),
        ),
    )
    assert result.side_effecting is True
    assert result.cacheable is False
    assert result.had_side_effect_event is True


def test_tools_require_side_effecting_on_capabilities() -> None:
    with pytest.raises(InvalidStateError):
        CLICapabilities(tools=True, side_effecting=False)


def test_registry_aliases_resolve_deterministically() -> None:
    registry = CLIProviderRegistry()
    registry.register(
        "goose_cli",
        aliases=("goose", "goose-cli", "block_goose"),
        description="Goose chat",
        factory=lambda: _StringProvider(),
    )
    assert registry.resolve("GOOSE_CLI") == "goose_cli"
    assert registry.resolve("goose-cli") == "goose_cli"
    assert registry.resolve("block_goose") == "goose_cli"
    assert registry.list_names() == ("goose_cli",)
    specs = registry.list_specs()
    assert len(specs) == 1
    assert "goose" in specs[0].aliases
    provider = registry.create("goose")
    assert provider.generate("ping") == "echo:ping:None:[]"


def test_registry_collision_fail_closed() -> None:
    registry = CLIProviderRegistry()
    registry.register("goose_cli", aliases=("goose",))
    with pytest.raises(RegistryCollisionError):
        registry.register("other", aliases=("goose",))
    with pytest.raises(RegistryCollisionError):
        registry.register("goose_cli", aliases=("codex_cli",))


def test_registry_unknown_provider_and_load_without_factory() -> None:
    registry = CLIProviderRegistry()
    registry.register("mock", description="mock provider")
    with pytest.raises(ProviderNotFoundError):
        registry.resolve("missing")
    with pytest.raises(Exception) as exc_info:
        registry.create("mock")
    assert "no factory" in str(exc_info.value).lower()


def test_default_registry_helpers() -> None:
    reset_default_registry()
    register_provider(
        "mock_cli",
        aliases=("mock-cli",),
        description="test mock",
        factory=lambda: _StringProvider(),
    )
    assert resolve_provider_name("mock-cli") == "mock_cli"
    names = [spec.name for spec in list_providers()]
    assert "mock_cli" in names
    assert get_default_registry().has_provider("mock_cli")
    reset_default_registry()


def test_provider_spec_alias_order_is_sorted_and_deduped() -> None:
    spec = ProviderSpec(
        name="Goose_CLI",
        aliases=("z_alias", "a_alias", "Z_ALIAS", "goose_cli"),
    )
    assert spec.name == "goose_cli"
    assert spec.aliases == ("a_alias", "z_alias")
    assert spec.all_names()[0] == "goose_cli"


def test_package_import_is_cold_and_lists_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing and listing must not start processes or load optional providers."""

    def _forbid_run(*_a: Any, **_k: Any) -> None:
        raise AssertionError("subprocess.run must not be called during cold import")

    def _forbid_popen(*_a: Any, **_k: Any) -> None:
        raise AssertionError(
            "subprocess.Popen must not be called during cold import"
        )

    monkeypatch.setattr(subprocess, "run", _forbid_run)
    monkeypatch.setattr(subprocess, "Popen", _forbid_popen)

    to_drop = [
        name
        for name in list(sys.modules)
        if name == "ipfs_accelerate_py.cli_runtime"
        or name.startswith("ipfs_accelerate_py.cli_runtime.")
    ]
    for name in to_drop:
        del sys.modules[name]

    module = importlib.import_module("ipfs_accelerate_py.cli_runtime")
    assert module.CONTRACT_VERSION == CONTRACT_VERSION
    assert not any(
        name.startswith("ipfs_accelerate_py.cli_runtime.providers")
        for name in sys.modules
    ), "optional package loaded at import: providers"

    reset_default_registry()
    assert list(list_providers()) == []
    assert get_default_registry().list_names() == ()
    register_provider("cold_mock", aliases=("cold-mock",), description="metadata only")
    assert resolve_provider_name("cold-mock") == "cold_mock"
    reset_default_registry()


def test_registry_list_does_not_call_factory() -> None:
    calls: list[str] = []

    def _factory() -> _StringProvider:
        calls.append("called")
        return _StringProvider()

    registry = CLIProviderRegistry()
    registry.register("lazy", aliases=("lazy-alias",), factory=_factory)
    assert registry.list_names() == ("lazy",)
    assert registry.list_specs()[0].name == "lazy"
    assert registry.get_spec("lazy-alias").to_dict()["name"] == "lazy"
    assert calls == []
    assert registry.create("lazy") is not None
    assert calls == ["called"]


def test_request_to_dict_omits_prompt_for_safe_diagnostics() -> None:
    request = _chat_request(prompt="user private prompt content here")
    payload = request.to_dict()
    assert "prompt" not in payload
    assert payload["prompt_chars"] == len(request.prompt)
    full = request.to_execution_dict()
    assert full["prompt"] == request.prompt


def test_invalid_execution_mode_and_empty_names() -> None:
    with pytest.raises(ContractValidationError):
        CLIRequest(prompt="x", mode="telepathy")
    with pytest.raises(ContractValidationError):
        ProviderSpec(name="  ")
