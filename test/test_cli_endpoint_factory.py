"""Tests for the canonical CLI endpoint factory and registry (GOOSE-006)."""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from ipfs_accelerate_py.cli_runtime.contracts import MAX_PROMPT_CHARS, MAX_TEXT_CHARS
from ipfs_accelerate_py.cli_runtime.endpoints import (
    CLIEndpointFactory,
    CLIEndpointRegistry,
    EndpointHealth,
    EndpointStats,
    UnsupportedEndpointToolError,
    bound_prompt,
    bound_result_text,
    create_cli_endpoint,
    error_envelope,
    execute_cli_inference,
    get_cli_endpoint,
    get_default_endpoint_factory,
    get_default_endpoint_registry,
    list_cli_endpoint_tools,
    list_cli_endpoints,
    register_cli_endpoint,
    reset_default_endpoint_registry,
    sanitize_error_payload,
)
from ipfs_accelerate_py.cli_runtime.errors import (
    BoundsExceededError,
    RegistryCollisionError,
)
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
    CLIEndpointAdapter,
    ClaudeCodeAdapter,
    GeminiCLIAdapter,
    OpenAICodexAdapter,
    VSCodeCLIAdapter,
    register_cli_endpoint as adapters_register,
    list_cli_endpoints as adapters_list,
    execute_cli_inference as adapters_execute,
    get_cli_endpoint as adapters_get,
    create_cli_endpoint as adapters_create,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_default_endpoint_registry()
    yield
    reset_default_endpoint_registry()


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Concrete stand-in that never touches a real CLI binary."""

    def __init__(
        self,
        endpoint_id: str,
        cli_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        available: bool = True,
        returncode: int = 0,
        result_text: str = "ok",
        hang: bool = False,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.cli_path = cli_path or "/usr/bin/fake-cli"
        self.config = dict(config or {})
        self._available = available
        self._returncode = returncode
        self._result_text = result_text
        self._hang = hang
        self._stats_lock = threading.Lock()
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
        }
        self.execute_calls = 0

    def is_available(self) -> bool:
        return self._available

    def validate_config(self) -> Dict[str, Any]:
        return {"valid": True, "issues": [], "config": self.config}

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self.stats)
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
            "cli_path": self.cli_path,
            "available": self._available,
            "stats": stats,
        }

    def execute(
        self,
        prompt: str,
        task_type: str = "text_generation",
        timeout: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.execute_calls += 1
        if self._hang:
            time.sleep(0.05)
        with self._stats_lock:
            self.stats["requests"] += 1
        if self._returncode != 0:
            with self._stats_lock:
                self.stats["failures"] += 1
            return {
                "error": f"CLI exited with status {self._returncode}",
                "status": "error",
                "success": False,
                "returncode": self._returncode,
                "error_code": "nonzero_exit",
                "endpoint_id": self.endpoint_id,
            }
        with self._stats_lock:
            self.stats["successes"] += 1
        return {
            "result": self._result_text,
            "status": "success",
            "success": True,
            "returncode": 0,
            "endpoint_id": self.endpoint_id,
        }


def _register_fake(
    endpoint_id: str = "fake_1",
    **kwargs: Any,
) -> _FakeAdapter:
    adapter = _FakeAdapter(endpoint_id, **kwargs)
    result = register_cli_endpoint(adapter, tool="custom", replace=True)
    assert result.get("status") == "success", result
    return adapter


# ---------------------------------------------------------------------------
# Factory: never abstract, lazy, typed unsupported
# ---------------------------------------------------------------------------


def test_factory_never_instantiates_abstract_base() -> None:
    factory = get_default_endpoint_factory()
    for tool in factory.list_tool_names():
        adapter = factory.create(tool, f"ep_{tool}")
        assert not inspect.isabstract(type(adapter))
        assert type(adapter).__name__ != "CLIEndpointAdapter"
        assert isinstance(adapter, CLIEndpointAdapter)


def test_create_cli_endpoint_concrete_types() -> None:
    assert isinstance(create_cli_endpoint("claude", "c1"), ClaudeCodeAdapter)
    assert isinstance(create_cli_endpoint("openai_cli", "o1"), OpenAICodexAdapter)
    assert isinstance(create_cli_endpoint("gemini", "g1"), GeminiCLIAdapter)
    assert isinstance(create_cli_endpoint("copilot", "v1"), VSCodeCLIAdapter)


def test_unsupported_tool_returns_typed_error() -> None:
    result = register_cli_endpoint(tool="definitely_not_a_tool")
    assert result["status"] == "error"
    assert result.get("registered") is False
    assert result.get("error_code") == "provider_not_found"
    assert "definitely_not_a_tool" in result.get("error", "")
    assert result.get("success") is False


def test_unsupported_tool_raises_on_resolve() -> None:
    factory = CLIEndpointFactory()
    with pytest.raises(UnsupportedEndpointToolError) as excinfo:
        factory.resolve_tool("nope_tool_xyz")
    assert excinfo.value.code.value == "provider_not_found"


def test_list_tools_is_lazy_and_does_not_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    probes: List[str] = []

    def _boom(*_a: Any, **_k: Any) -> Any:
        probes.append("probed")
        raise AssertionError("must not probe")

    monkeypatch.setattr("shutil.which", _boom)
    monkeypatch.setattr("os.path.isfile", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("probe")))

    tools = list_cli_endpoint_tools()
    assert probes == []
    names = {t["name"] for t in tools}
    assert "claude" in names
    assert "openai" in names
    assert "gemini" in names
    assert "vscode" in names


def test_registration_is_lazy_does_not_probe_every_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    availability_checks: List[str] = []
    original_available = ClaudeCodeAdapter.is_available

    def tracking_available(self: Any) -> bool:
        availability_checks.append(self.endpoint_id)
        return False

    monkeypatch.setattr(ClaudeCodeAdapter, "is_available", tracking_available)
    # register without probe
    result = register_cli_endpoint(
        tool="claude", endpoint_id="lazy_claude", probe=False
    )
    assert result["status"] == "success"
    assert result.get("available") is None  # not probed at register
    # Construction may call is_available once in adapter __init__; that is fine.
    # Listing must not re-probe every provider.
    availability_checks.clear()
    listed = list_cli_endpoints(probe=False)
    assert any(e["endpoint_id"] == "lazy_claude" for e in listed)
    assert availability_checks == []


# ---------------------------------------------------------------------------
# Registry collision
# ---------------------------------------------------------------------------


def test_registry_collision_fail_closed() -> None:
    _register_fake("collide_1")
    adapter2 = _FakeAdapter("collide_1")
    with pytest.raises(RegistryCollisionError):
        get_default_endpoint_registry().register_adapter(
            adapter2, tool="custom", replace=False
        )
    # Public API returns typed error envelope instead of raising.
    result = register_cli_endpoint(adapter2, tool="custom", replace=False)
    assert result["status"] == "error"
    assert result.get("error_code") == "registry_collision"
    assert result.get("registered") is False


def test_registry_collision_replace() -> None:
    a1 = _register_fake("replace_me", result_text="first")
    a2 = _FakeAdapter("replace_me", result_text="second")
    result = register_cli_endpoint(a2, tool="custom", replace=True)
    assert result["status"] == "success"
    got = get_cli_endpoint("replace_me")
    assert got is a2
    assert got is not a1


def test_tool_type_alias_collision() -> None:
    factory = CLIEndpointFactory()
    with pytest.raises(RegistryCollisionError):
        factory.register_tool_type(
            {
                "name": "other_tool",
                "aliases": ("claude",),  # collides with default claude
                "adapter_class_name": "ClaudeCodeAdapter",
            }
        )


# ---------------------------------------------------------------------------
# Unavailable provider
# ---------------------------------------------------------------------------


def test_unavailable_provider_execute() -> None:
    _register_fake("down_1", available=False)
    result = execute_cli_inference("down_1", "hello world")
    assert result["status"] == "error"
    assert result.get("success") is False
    assert "not available" in result.get("error", "").lower()
    assert "hello world" not in str(result)
    assert "prompt" not in result


def test_missing_endpoint_execute() -> None:
    result = execute_cli_inference("no_such_endpoint", "secret prompt text")
    assert result["status"] == "error"
    assert result.get("error_code") == "provider_not_found"
    assert "secret prompt text" not in str(result)


# ---------------------------------------------------------------------------
# Nonzero exit
# ---------------------------------------------------------------------------


def test_nonzero_exit_is_failure() -> None:
    _register_fake("nz_1", returncode=7, available=True)
    result = execute_cli_inference("nz_1", "do the thing")
    assert result["status"] == "error"
    assert result.get("success") is False
    assert result.get("returncode") == 7
    assert result.get("error_code") == "nonzero_exit"
    assert "do the thing" not in str(result)


def test_concrete_adapter_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ClaudeCodeAdapter("nz_claude", cli_path="/bin/true", config={})
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    class _Proc:
        returncode = 3
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _Proc(),
    )
    result = adapter.execute("private prompt content")
    assert result["status"] == "error"
    assert result["returncode"] == 3
    assert result.get("error_code") == "nonzero_exit"
    assert "private prompt content" not in str(result)
    assert adapter.stats["failures"] >= 1
    assert adapter.stats["successes"] == 0


# ---------------------------------------------------------------------------
# Bounded result / request
# ---------------------------------------------------------------------------


def test_prompt_bound_enforced() -> None:
    _register_fake("bound_1")
    huge = "x" * (MAX_PROMPT_CHARS + 10)
    result = execute_cli_inference("bound_1", huge)
    assert result["status"] == "error"
    assert result.get("error_code") == "bounds_exceeded"
    assert huge not in str(result)


def test_bound_prompt_helper() -> None:
    with pytest.raises(BoundsExceededError):
        bound_prompt("y" * (MAX_PROMPT_CHARS + 1))
    assert bound_prompt("ok") == "ok"


def test_bound_result_clips() -> None:
    text = "z" * (MAX_TEXT_CHARS + 50)
    clipped = bound_result_text(text)
    assert len(clipped) <= MAX_TEXT_CHARS
    assert clipped.endswith("...")


def test_success_result_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = "R" * (MAX_TEXT_CHARS + 100)
    adapter = _FakeAdapter("bound_ok", result_text=huge, available=True)
    register_cli_endpoint(adapter, tool="custom", replace=True)
    result = execute_cli_inference("bound_ok", "hi")
    assert result["status"] == "success"
    assert len(result["result"]) <= MAX_TEXT_CHARS


# ---------------------------------------------------------------------------
# Concurrent stats
# ---------------------------------------------------------------------------


def test_endpoint_stats_concurrency_safe() -> None:
    stats = EndpointStats()

    def worker() -> None:
        for _ in range(200):
            stats.record_success(0.001)
            stats.record_failure(0.001)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = stats.snapshot()
    assert snap["requests"] == 8 * 200 * 2
    assert snap["successes"] == 8 * 200
    assert snap["failures"] == 8 * 200


def test_adapter_stats_concurrency_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ClaudeCodeAdapter("stats_c", cli_path="/bin/true", config={})
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    class _Proc:
        returncode = 0
        stdout = "hello"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    monkeypatch.setattr(
        adapter,
        "_format_prompt",
        lambda prompt, task_type, **kw: [adapter.cli_path, "x"],
    )
    monkeypatch.setattr(
        adapter,
        "_parse_response",
        lambda stdout, stderr: {"result": stdout},
    )

    errors: List[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                adapter.execute("p")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    snap = adapter.get_stats()["stats"]
    assert snap["requests"] == 6 * 50
    assert snap["successes"] == 6 * 50
    assert snap["failures"] == 0


def test_registry_stats_under_concurrent_execute() -> None:
    adapter = _FakeAdapter("conc_1", available=True)
    register_cli_endpoint(adapter, tool="custom", replace=True)
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(40):
                execute_cli_inference("conc_1", "msg")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    record = get_default_endpoint_registry().get_record("conc_1")
    assert record is not None
    snap = record.stats.snapshot()
    assert snap["requests"] == 5 * 40
    assert snap["successes"] == 5 * 40


# ---------------------------------------------------------------------------
# Errors never echo prompts
# ---------------------------------------------------------------------------


def test_error_envelope_strips_prompt() -> None:
    secret = "SUPER_SECRET_USER_PROMPT_42"
    env = error_envelope(
        "failed",
        prompt=secret,
        details={"prompt": secret, "other": "ok"},
    )
    assert "prompt" not in env
    assert secret not in str(env)
    assert env["details"]["prompt"] == "[redacted]"
    assert env["details"]["other"] == "ok"


def test_sanitize_error_payload_redacts_embedded_prompt() -> None:
    secret = "unique-prompt-body-xyz"
    cleaned = sanitize_error_payload(
        {"message": f"failed on {secret}", "prompt": secret},
        prompt=secret,
    )
    assert cleaned["prompt"] == "[redacted]"
    assert secret not in cleaned["message"]


# ---------------------------------------------------------------------------
# Existing endpoint types continue to register and execute
# ---------------------------------------------------------------------------


def test_existing_endpoint_types_register_and_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for tool, endpoint_id in (
        ("claude", "t_claude"),
        ("openai", "t_openai"),
        ("gemini", "t_gemini"),
        ("vscode", "t_vscode"),
    ):
        result = register_cli_endpoint(tool=tool, endpoint_id=endpoint_id)
        assert result["status"] == "success", result
        adapter = get_cli_endpoint(endpoint_id)
        assert adapter is not None
        assert isinstance(adapter, CLIEndpointAdapter)

        monkeypatch.setattr(adapter, "is_available", lambda: True)

        class _Proc:
            returncode = 0
            stdout = f"response-from-{endpoint_id}"
            stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
        # Avoid depending on real CLI argv shape for formatting.
        monkeypatch.setattr(
            adapter,
            "_format_prompt",
            lambda prompt, task_type, **kw: [adapter.cli_path or "cli", "run"],
        )
        out = execute_cli_inference(endpoint_id, "ping")
        assert out.get("status") == "success", out
        assert out.get("success") is True
        assert "ping" not in str(out.get("error", ""))


# ---------------------------------------------------------------------------
# Deprecated imports remain compatibility shims
# ---------------------------------------------------------------------------


def test_deprecated_imports_are_shims() -> None:
    reset_default_endpoint_registry()
    adapter = _FakeAdapter("shim_1", available=True)
    result = adapters_register(adapter, tool="custom", replace=True)
    assert result["status"] == "success"
    assert adapters_get("shim_1") is adapter
    listed = adapters_list()
    assert any(e.get("endpoint_id") == "shim_1" for e in listed)
    out = adapters_execute("shim_1", "hello")
    assert out.get("status") == "success"
    # create_cli_endpoint re-exported
    created = adapters_create("claude", "shim_claude")
    assert isinstance(created, ClaudeCodeAdapter)


def test_cli_adapter_registry_view_shared() -> None:
    from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
        CLI_ADAPTER_REGISTRY,
    )
    from ipfs_accelerate_py.cli_runtime.endpoints import CLI_ADAPTER_REGISTRY as CANON

    adapter = _FakeAdapter("view_1")
    register_cli_endpoint(adapter, tool="custom", replace=True)
    assert "view_1" in CLI_ADAPTER_REGISTRY
    assert "view_1" in CANON
    assert CLI_ADAPTER_REGISTRY.get("view_1") is adapter


# ---------------------------------------------------------------------------
# Lifecycle dispatch
# ---------------------------------------------------------------------------


def test_lifecycle_liveness_readiness_cancel() -> None:
    adapter = _register_fake("life_1", available=True)
    reg = get_default_endpoint_registry()
    live = reg.liveness("life_1")
    assert live["live"] is True
    ready = reg.readiness("life_1")
    assert ready["ready"] is True
    cancel = reg.cancel("life_1")
    assert cancel.get("cancelled") is False
    described = reg.describe("life_1", probe=False)
    assert described["status"] == "success"
    assert described["endpoint"]["endpoint_id"] == "life_1"
    assert adapter.execute_calls == 0


def test_stream_emits_completed_without_prompt() -> None:
    _register_fake("stream_1", result_text="streamed", available=True)
    events = list(
        get_default_endpoint_registry().stream("stream_1", "secret-stream-prompt")
    )
    kinds = [e["event"] for e in events]
    assert "started" in kinds
    assert "completed" in kinds
    blob = str(events)
    assert "secret-stream-prompt" not in blob
