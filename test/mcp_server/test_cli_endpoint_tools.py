"""MCP native tool parity tests for the CLI endpoint factory (GOOSE-006)."""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any, Dict, List, Optional

import pytest

from ipfs_accelerate_py.cli_runtime.contracts import MAX_PROMPT_CHARS
from ipfs_accelerate_py.cli_runtime.endpoints import (
    reset_default_endpoint_registry,
    register_cli_endpoint as py_register,
    execute_cli_inference as py_execute,
    list_cli_endpoints as py_list,
    get_cli_endpoint as py_get,
)
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
    CLIEndpointAdapter,
    ClaudeCodeAdapter,
)
from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_tools import (
    cli_endpoint_execute,
    cli_endpoint_get,
    cli_endpoint_list,
    cli_endpoint_register,
    register_native_cli_endpoint_tools,
)
from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_tools.native_cli_endpoint_tools import (
    cli_endpoint_register as native_register,
    cli_endpoint_execute as native_execute,
    cli_endpoint_list as native_list,
    cli_endpoint_get as native_get,
)


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_default_endpoint_registry()
    yield
    reset_default_endpoint_registry()


class _FakeAdapter:
    def __init__(
        self,
        endpoint_id: str,
        *,
        available: bool = True,
        returncode: int = 0,
        result_text: str = "mcp-ok",
    ) -> None:
        self.endpoint_id = endpoint_id
        self.cli_path = "/usr/bin/fake-cli"
        self.config: Dict[str, Any] = {"tool": "custom"}
        self._available = available
        self._returncode = returncode
        self._result_text = result_text
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
        }

    def is_available(self) -> bool:
        return self._available

    def validate_config(self) -> Dict[str, Any]:
        return {"valid": True, "issues": [], "config": self.config}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
            "cli_path": self.cli_path,
            "available": self._available,
            "stats": dict(self.stats),
        }

    def execute(
        self,
        prompt: str,
        task_type: str = "text_generation",
        timeout: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.stats["requests"] += 1
        if self._returncode != 0:
            self.stats["failures"] += 1
            return {
                "error": f"CLI exited with status {self._returncode}",
                "status": "error",
                "success": False,
                "returncode": self._returncode,
                "error_code": "nonzero_exit",
                "endpoint_id": self.endpoint_id,
            }
        self.stats["successes"] += 1
        return {
            "result": self._result_text,
            "status": "success",
            "success": True,
            "returncode": 0,
            "endpoint_id": self.endpoint_id,
        }


# ---------------------------------------------------------------------------
# Native registration never instantiates abstract base
# ---------------------------------------------------------------------------


def test_native_register_uses_concrete_factory_not_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abstract_calls: List[str] = []

    original_init = CLIEndpointAdapter.__init__

    def tracking_init(self: Any, *a: Any, **k: Any) -> None:
        # Only fire for direct abstract construction attempts.
        if type(self) is CLIEndpointAdapter:
            abstract_calls.append("abstract")
        return original_init(self, *a, **k)

    monkeypatch.setattr(CLIEndpointAdapter, "__init__", tracking_init)

    result = _run(
        native_register(tool="claude", endpoint_id="mcp_claude_1", config={})
    )
    assert result.get("status") == "success", result
    assert result.get("registered") is not False
    assert abstract_calls == []
    adapter = py_get("mcp_claude_1")
    assert adapter is not None
    assert isinstance(adapter, ClaudeCodeAdapter)
    assert not inspect.isabstract(type(adapter))


def test_native_unsupported_tool_typed_error_not_swallowed() -> None:
    result = _run(native_register(tool="not_a_real_cli_tool_zzz"))
    assert result["status"] == "error"
    assert result.get("success") is False
    assert result.get("registered") is False
    assert result.get("error")
    assert result.get("error_code") in {
        "provider_not_found",
        "internal",
    }
    # Must not look like a silent registered:false with no error.
    assert "error" in result
    assert result.get("tool") == "not_a_real_cli_tool_zzz"


# ---------------------------------------------------------------------------
# Python / MCP parity
# ---------------------------------------------------------------------------


def test_python_mcp_register_list_parity() -> None:
    py_result = py_register(tool="claude", endpoint_id="parity_claude")
    assert py_result["status"] == "success"

    mcp_list = _run(native_list())
    py_list_result = py_list()

    assert mcp_list["status"] == "success"
    mcp_ids = {e.get("endpoint_id") for e in mcp_list.get("endpoints", [])}
    py_ids = {e.get("endpoint_id") for e in py_list_result}
    assert "parity_claude" in mcp_ids
    assert "parity_claude" in py_ids

    mcp_get = _run(native_get("parity_claude"))
    assert mcp_get["status"] == "success"
    assert mcp_get["endpoint"]["endpoint_id"] == "parity_claude"


def test_python_mcp_execute_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter("parity_exec", available=True, result_text="same")
    py_register(adapter, tool="custom", replace=True)

    py_out = py_execute("parity_exec", "hello parity")
    mcp_out = _run(native_execute("parity_exec", "hello parity"))

    assert py_out.get("status") == "success"
    assert mcp_out.get("status") == "success"
    assert py_out.get("result") == mcp_out.get("result") == "same"
    # Neither path echoes the prompt.
    assert "hello parity" not in str(py_out.get("error", ""))
    assert "prompt" not in mcp_out
    assert "hello parity" not in str(mcp_out.get("error", ""))


def test_python_mcp_nonzero_exit_parity() -> None:
    adapter = _FakeAdapter("parity_nz", available=True, returncode=9)
    py_register(adapter, tool="custom", replace=True)

    py_out = py_execute("parity_nz", "secret-prompt-body")
    mcp_out = _run(native_execute("parity_nz", "secret-prompt-body"))

    assert py_out["status"] == "error"
    assert mcp_out["status"] == "error"
    assert py_out.get("returncode") == mcp_out.get("returncode") == 9
    assert "secret-prompt-body" not in str(py_out)
    assert "secret-prompt-body" not in str(mcp_out)
    assert "prompt" not in mcp_out


def test_python_mcp_unavailable_parity() -> None:
    adapter = _FakeAdapter("parity_down", available=False)
    py_register(adapter, tool="custom", replace=True)

    py_out = py_execute("parity_down", "hidden-prompt")
    mcp_out = _run(native_execute("parity_down", "hidden-prompt"))

    assert py_out["status"] == "error"
    assert mcp_out["status"] == "error"
    assert "hidden-prompt" not in str(py_out)
    assert "hidden-prompt" not in str(mcp_out)


def test_python_mcp_bounds_parity() -> None:
    adapter = _FakeAdapter("parity_bound", available=True)
    py_register(adapter, tool="custom", replace=True)
    huge = "q" * (MAX_PROMPT_CHARS + 5)

    py_out = py_execute("parity_bound", huge)
    mcp_out = _run(native_execute("parity_bound", huge))

    assert py_out["status"] == "error"
    assert mcp_out["status"] == "error"
    assert huge not in str(py_out)
    assert huge not in str(mcp_out)


# ---------------------------------------------------------------------------
# Concurrent stats via MCP execute
# ---------------------------------------------------------------------------


def test_mcp_concurrent_stats() -> None:
    adapter = _FakeAdapter("mcp_conc", available=True)
    py_register(adapter, tool="custom", replace=True)
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(25):
                _run(native_execute("mcp_conc", "x"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # Adapter-level counters should reflect all executes.
    assert adapter.stats["requests"] == 4 * 25
    assert adapter.stats["successes"] == 4 * 25


# ---------------------------------------------------------------------------
# Tool registration surface
# ---------------------------------------------------------------------------


class _StubManager:
    def __init__(self) -> None:
        self.tools: List[Dict[str, Any]] = []

    def register_tool(self, **kwargs: Any) -> None:
        self.tools.append(kwargs)


def test_register_native_tools_surface() -> None:
    manager = _StubManager()
    register_native_cli_endpoint_tools(manager)
    names = {t["name"] for t in manager.tools}
    assert names == {
        "cli_endpoint_list",
        "cli_endpoint_get",
        "cli_endpoint_execute",
        "cli_endpoint_register",
    }
    for tool in manager.tools:
        assert tool["category"] == "cli_endpoint_tools"
        assert "native" in tool["tags"]


def test_native_execute_error_never_includes_prompt() -> None:
    result = _run(
        native_execute("missing_endpoint_xyz", "THIS_IS_THE_PROMPT_VALUE")
    )
    assert result["status"] == "error"
    assert "prompt" not in result
    assert "THIS_IS_THE_PROMPT_VALUE" not in str(result)


def test_public_exports_match_package() -> None:
    # Package re-exports should be the same callables used in tests.
    assert cli_endpoint_register is native_register
    assert cli_endpoint_execute is native_execute
    assert cli_endpoint_list is native_list
    assert cli_endpoint_get is native_get


# ---------------------------------------------------------------------------
# Goose MCP schema bounds and authority rejection (GOOSE-007)
# ---------------------------------------------------------------------------


def test_native_register_goose_tool() -> None:
    result = _run(
        native_register(tool="goose", endpoint_id="mcp_goose_1", config={})
    )
    assert result.get("status") == "success", result
    assert result.get("registered") is not False
    adapter = py_get("mcp_goose_1")
    assert adapter is not None
    from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import GooseCLIAdapter

    assert isinstance(adapter, GooseCLIAdapter)


def test_native_register_goose_cli_alias() -> None:
    result = _run(
        native_register(tool="goose_cli", endpoint_id="mcp_goose_cli", config={})
    )
    assert result.get("status") == "success", result


def test_mcp_execute_rejects_unknown_authority_options() -> None:
    adapter = _FakeAdapter("auth_rej", available=True)
    py_register(adapter, tool="custom", replace=True)

    result = _run(
        native_execute(
            "auth_rej",
            "hello",
            extra_options={
                "super_secret_allow_side_effects_override": True,
            },
        )
    )
    assert result["status"] == "error"
    assert result.get("error_code") == "policy_denied"
    assert "unknown authority" in (result.get("error") or "").lower()
    assert "prompt" not in result
    assert "hello" not in str(result.get("error", ""))


def test_mcp_execute_schema_exposes_goose_authority_fields() -> None:
    manager = _StubManager()
    register_native_cli_endpoint_tools(manager)
    execute_tool = next(t for t in manager.tools if t["name"] == "cli_endpoint_execute")
    props = execute_tool["input_schema"]["properties"]
    for field in (
        "execution_mode",
        "allow_side_effects",
        "enable_agent",
        "cwd",
        "path_root",
        "approval_mode",
        "builtins",
        "extensions",
        "max_turns",
        "timeout_seconds",
        "max_output_bytes",
        "goose_provider",
        "session_id",
        "extra_options",
    ):
        assert field in props, f"missing MCP schema field {field}"
    assert execute_tool["input_schema"].get("additionalProperties") is False

    register_tool = next(
        t for t in manager.tools if t["name"] == "cli_endpoint_register"
    )
    tool_desc = register_tool["input_schema"]["properties"]["tool"]["description"]
    assert "goose" in tool_desc.lower()


def test_mcp_goose_agent_policy_denied_without_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP path surfaces policy denial when agent lacks package enable."""
    # Register a goose endpoint that will not run a real binary for policy checks.
    result = _run(
        native_register(
            tool="goose",
            endpoint_id="mcp_agent_deny",
            config={},
        )
    )
    assert result.get("status") == "success", result

    out = _run(
        native_execute(
            "mcp_agent_deny",
            "SECRET_AGENT_PROMPT",
            execution_mode="agent",
            allow_side_effects=True,
            cwd="/tmp/work",
            path_root="/tmp",
            approval_mode="approve",
            max_turns=3,
            timeout_seconds=10,
            max_output_bytes=1024,
        )
    )
    assert out["status"] == "error"
    assert out.get("error_code") in {"policy_denied", "provider_load_failed"}
    assert "SECRET_AGENT_PROMPT" not in str(out)
    assert "prompt" not in out
