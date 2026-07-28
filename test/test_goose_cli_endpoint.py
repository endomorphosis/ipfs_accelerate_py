"""Offline tests for Goose one-shot CLI endpoint and MCP operations (GOOSE-007).

Uses fake goose executables only — no live network or real Goose binary.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from ipfs_accelerate_py.cli_runtime.endpoints import (
    EndpointHealth,
    create_cli_endpoint,
    execute_cli_inference,
    get_cli_endpoint,
    get_default_endpoint_registry,
    list_cli_endpoint_tools,
    list_cli_endpoints,
    register_cli_endpoint,
    reset_default_endpoint_registry,
)
from ipfs_accelerate_py.cli_runtime.providers.goose import (
    PINNED_GOOSE_VERSION,
    capabilities_for_version,
)
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import GooseCLIAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_default_endpoint_registry()
    yield
    reset_default_endpoint_registry()


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    return tmp_path / "bin"


def _write_fake_goose(
    directory: Path,
    *,
    script: str,
    name: str = "goose",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _json_success_script(
    text: str = "hello from goose endpoint",
    *,
    include_tool: bool = False,
    exit_code: int = 0,
    version: str = PINNED_GOOSE_VERSION,
) -> str:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    if include_tool:
        content.append(
            {
                "type": "tool_use",
                "id": "t1",
                "name": "developer__shell",
                "input": {"command": "echo hi"},
            }
        )
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "assistant", "content": content},
        ],
        "metadata": {
            "total_tokens": 12,
            "status": "completed",
        },
    }
    body = json.dumps(payload)
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import json, os, sys

        argv = sys.argv[1:]
        if "--version" in argv or "-V" in argv:
            print("goose {version}")
            sys.exit(0)

        if not argv or argv[0] != "run":
            print("expected run", file=sys.stderr)
            sys.exit(2)

        if os.environ.get("GOOSE_FAKE_ARGV_PATH"):
            with open(os.environ["GOOSE_FAKE_ARGV_PATH"], "w", encoding="utf-8") as fh:
                json.dump({{
                    "argv": argv,
                    "env_mode": os.environ.get("GOOSE_MODE"),
                    "env_provider": os.environ.get("GOOSE_PROVIDER"),
                    "env_model": os.environ.get("GOOSE_MODEL"),
                    "env_path_root": os.environ.get("GOOSE_PATH_ROOT"),
                    "cwd": os.getcwd(),
                }}, fh)

        mode = os.environ.get("GOOSE_MODE", "")
        if mode == "chat":
            for flag in ("--no-session", "--no-profile", "--output-format",
                         "--max-turns", "--max-tool-repetitions"):
                if flag not in argv:
                    print(f"missing {{flag}}", file=sys.stderr)
                    sys.exit(3)
            if "--with-builtin" in argv or "--with-extension" in argv:
                print("chat must not enable extensions", file=sys.stderr)
                sys.exit(3)

        if "--instructions" not in argv and "-i" not in argv:
            print("missing instructions", file=sys.stderr)
            sys.exit(3)
        _ = sys.stdin.read()
        print(json.dumps(json.loads({body!r})))
        sys.exit({exit_code})
        """
    )


def _version_only_script(version: str) -> str:
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        argv = sys.argv[1:]
        if "--version" in argv or "-V" in argv:
            print("goose {version}")
            sys.exit(0)
        print("no run support", file=sys.stderr)
        sys.exit(2)
        """
    )


def _stub_manifest() -> Dict[str, Any]:
    return {
        "pinned_version": PINNED_GOOSE_VERSION,
        "assets": [
            {
                "os": "linux",
                "arch": "x86_64",
                "libc": "gnu",
                "variant": "standard",
                "asset_name": "goose.tar.bz2",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    }


def _register_goose(
    exe: Path,
    endpoint_id: str = "goose_ep",
    *,
    config: Optional[Dict[str, Any]] = None,
    tool: str = "goose",
) -> Dict[str, Any]:
    cfg = dict(config or {})
    return register_cli_endpoint(
        tool=tool,
        endpoint_id=endpoint_id,
        cli_path=str(exe),
        config=cfg,
        replace=True,
        probe=False,
    )


# ---------------------------------------------------------------------------
# Registration / aliases
# ---------------------------------------------------------------------------


def test_goose_and_goose_cli_aliases_register() -> None:
    tools = list_cli_endpoint_tools()
    names = {t["name"] for t in tools}
    assert "goose" in names
    # Alias resolution
    adapter = create_cli_endpoint("goose_cli", "g_alias")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter2 = create_cli_endpoint("goose", "g_main")
    assert isinstance(adapter2, GooseCLIAdapter)
    adapter3 = create_cli_endpoint("block_goose", "g_block")
    assert isinstance(adapter3, GooseCLIAdapter)


def test_register_goose_lazy_no_model_request(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registration and list must not invoke goose run / model path."""
    run_calls: List[Any] = []

    def tracking_run(*a: Any, **k: Any) -> Any:
        run_calls.append((a, k))
        raise AssertionError("must not run processes on register/list")

    monkeypatch.setattr("subprocess.run", tracking_run)
    result = register_cli_endpoint(
        tool="goose",
        endpoint_id="lazy_goose",
        cli_path="/nonexistent/goose-binary",
        config={},
        probe=False,
    )
    assert result["status"] == "success"
    assert result.get("available") is None
    listed = list_cli_endpoints(probe=False)
    assert any(e["endpoint_id"] == "lazy_goose" for e in listed)
    assert run_calls == []


def test_list_and_liveness_no_model_request(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script())
    _register_goose(exe, "live_goose")

    execute_calls: List[str] = []
    original = GooseCLIAdapter.execute

    def tracking_execute(self: Any, *a: Any, **k: Any) -> Any:
        execute_calls.append("execute")
        return original(self, *a, **k)

    monkeypatch.setattr(GooseCLIAdapter, "execute", tracking_execute)

    listed = list_cli_endpoints(probe=False)
    assert any(e["endpoint_id"] == "live_goose" for e in listed)

    registry = get_default_endpoint_registry()
    live = registry.liveness("live_goose")
    assert live["live"] is True
    assert live["success"] is True
    assert execute_calls == []


# ---------------------------------------------------------------------------
# Health states
# ---------------------------------------------------------------------------


def test_health_missing_when_not_installed(tmp_path: Path) -> None:
    adapter = GooseCLIAdapter(
        "missing_g",
        cli_path=str(tmp_path / "no-such-goose"),
        config={},
    )
    # Force provider not to find anything useful.
    health = adapter.assess_health()
    assert health["installed"] is False
    assert health["health"] == EndpointHealth.MISSING.value
    assert health["ready"] is False


def test_health_installed_without_auth(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(
        fake_bin, script=_version_only_script(PINNED_GOOSE_VERSION)
    )
    # Clear auth env markers.
    for name in (
        "GOOSE_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "OLLAMA_HOST",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    adapter = GooseCLIAdapter("inst_g", cli_path=str(exe), config={})
    adapter._get_provider().discover_kwargs = {"manifest": _stub_manifest()}
    health = adapter.assess_health()
    assert health["installed"] is True
    assert health["unsupported_version"] is False
    assert health["health"] == EndpointHealth.INSTALLED.value
    assert health["ready"] is False


def test_health_ready_with_auth(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(
        fake_bin, script=_version_only_script(PINNED_GOOSE_VERSION)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    adapter = GooseCLIAdapter("ready_g", cli_path=str(exe), config={})
    adapter._get_provider().discover_kwargs = {"manifest": _stub_manifest()}
    health = adapter.assess_health()
    assert health["installed"] is True
    assert health["configured"] is True
    assert health["ready"] is True
    assert health["health"] == EndpointHealth.READY.value
    assert health.get("goose_version")


def test_health_configured_with_endpoint_config_only(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(
        fake_bin, script=_version_only_script(PINNED_GOOSE_VERSION)
    )
    for name in (
        "GOOSE_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    adapter = GooseCLIAdapter(
        "cfg_g",
        cli_path=str(exe),
        config={"goose_provider": "openai", "model": "gpt-test"},
    )
    adapter._get_provider().discover_kwargs = {"manifest": _stub_manifest()}
    health = adapter.assess_health()
    assert health["installed"] is True
    assert health["configured"] is True
    assert health["health"] in {
        EndpointHealth.CONFIGURED.value,
        EndpointHealth.READY.value,
    }


def test_health_unsupported_version(fake_bin: Path) -> None:
    exe = _write_fake_goose(fake_bin, script=_version_only_script("1.0.0"))
    adapter = GooseCLIAdapter("old_g", cli_path=str(exe), config={})
    adapter._get_provider().discover_kwargs = {"manifest": _stub_manifest()}
    # Force version interpretation.
    adapter._get_provider().version = "1.0.0"
    adapter._get_provider().capabilities = capabilities_for_version("1.0.0")
    health = adapter.assess_health()
    # assess_health re-discovers version from binary output "1.0.0"
    assert health["installed"] is True
    assert health["unsupported_version"] is True
    assert health["health"] == EndpointHealth.UNSUPPORTED_VERSION.value
    assert health["ready"] is False


def test_readiness_uses_assess_health(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(
        fake_bin, script=_version_only_script(PINNED_GOOSE_VERSION)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _register_goose(exe, "ready_probe")
    adapter = get_cli_endpoint("ready_probe")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().discover_kwargs = {"manifest": _stub_manifest()}

    registry = get_default_endpoint_registry()
    result = registry.readiness("ready_probe")
    assert result["health"] in {
        EndpointHealth.READY.value,
        EndpointHealth.CONFIGURED.value,
        EndpointHealth.INSTALLED.value,
    }
    assert "ready" in result


# ---------------------------------------------------------------------------
# Chat execute (safe profile)
# ---------------------------------------------------------------------------


def test_default_execute_uses_safe_chat_profile(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_path = fake_bin / "argv.json"
    script = _json_success_script("chat-ok")
    exe = _write_fake_goose(fake_bin, script=script)
    monkeypatch.setenv("GOOSE_FAKE_ARGV_PATH", str(argv_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _register_goose(
        exe,
        "chat_g",
        config={"model": "gpt-test", "goose_provider": "openai"},
    )
    adapter = get_cli_endpoint("chat_g")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )
    adapter._get_provider().executable = str(exe)

    secret = "SECRET_PROMPT_MUST_NOT_LEAK"
    out = execute_cli_inference("chat_g", secret, timeout=15)
    assert out.get("status") == "success", out
    assert out.get("success") is True
    assert out.get("provider") == "goose_cli"
    assert out.get("execution_mode") == "chat"
    assert out.get("text") == "chat-ok" or out.get("result") == "chat-ok"
    assert out.get("side_effects_started") is False
    assert out.get("tool_call_count") == 0
    assert "elapsed_time" in out
    assert secret not in str(out)
    assert "prompt" not in out
    assert "sk-test" not in str(out)

    assert argv_path.is_file()
    recorded = json.loads(argv_path.read_text(encoding="utf-8"))
    assert recorded["env_mode"] == "chat"
    assert "--no-session" in recorded["argv"]
    assert "--no-profile" in recorded["argv"]
    assert "--with-builtin" not in recorded["argv"]
    assert "--with-extension" not in recorded["argv"]


def test_chat_envelope_fields(fake_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("fields"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _register_goose(
        exe,
        "fields_g",
        config={"model": "m1", "goose_provider": "openai"},
    )
    adapter = get_cli_endpoint("fields_g")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )
    adapter._get_provider().executable = str(exe)

    out = execute_cli_inference("fields_g", "hello", timeout=15)
    for key in (
        "provider",
        "execution_mode",
        "text",
        "goose_version",
        "underlying_provider",
        "model",
        "session",
        "tool_call_count",
        "side_effects_started",
        "elapsed_time",
    ):
        assert key in out, f"missing field {key} in {out}"
    assert out["provider"] == "goose_cli"
    assert out["execution_mode"] == "chat"
    assert out["error"] is None or out.get("status") == "success"


# ---------------------------------------------------------------------------
# Agent policy gates
# ---------------------------------------------------------------------------


def test_agent_requires_execution_mode_and_package_enable(
    fake_bin: Path,
) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("x"))
    _register_goose(exe, "agent_gate")
    adapter = get_cli_endpoint("agent_gate")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )
    adapter._get_provider().executable = str(exe)

    # Missing package enable
    out = adapter.execute(
        "do stuff",
        execution_mode="agent",
        allow_side_effects=True,
        cwd="/tmp/work",
        path_root="/tmp",
        approval_mode="approve",
        builtins=[],
        extensions=[],
        max_turns=5,
        timeout_seconds=30,
        max_output_bytes=1024,
    )
    assert out["status"] == "error"
    assert out["error_code"] == "policy_denied"
    assert "package enable" in (out.get("error") or "").lower() or "enable_agent" in (
        out.get("error") or ""
    )


def test_agent_requires_allow_side_effects(fake_bin: Path) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("x"))
    _register_goose(exe, "agent_side", config={"enable_agent": True})
    adapter = get_cli_endpoint("agent_side")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )

    out = adapter.execute(
        "do stuff",
        execution_mode="agent",
        enable_agent=True,
        allow_side_effects=False,
        cwd="/tmp/work",
        path_root="/tmp",
    )
    assert out["status"] == "error"
    assert out["error_code"] == "policy_denied"
    assert "allow_side_effects" in (out.get("error") or "")


def test_agent_requires_absolute_cwd_root(fake_bin: Path) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("x"))
    _register_goose(exe, "agent_cwd", config={"enable_agent": True})
    adapter = get_cli_endpoint("agent_cwd")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )

    out = adapter.execute(
        "do stuff",
        execution_mode="agent",
        enable_agent=True,
        allow_side_effects=True,
        cwd="relative/path",
        path_root="/tmp",
        approval_mode="approve",
    )
    assert out["status"] == "error"
    assert out["error_code"] in {"policy_denied", "invalid_contract"}


def test_agent_execute_success(
    fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    root = tmp_path
    argv_path = fake_bin / "agent_argv.json"
    exe = _write_fake_goose(
        fake_bin, script=_json_success_script("agent-done", include_tool=True)
    )
    monkeypatch.setenv("GOOSE_FAKE_ARGV_PATH", str(argv_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _register_goose(
        exe,
        "agent_ok",
        config={"enable_agent": True, "model": "m1", "goose_provider": "openai"},
    )
    adapter = get_cli_endpoint("agent_ok")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )
    adapter._get_provider().executable = str(exe)

    secret = "AGENT_SECRET_PROMPT"
    out = adapter.execute(
        secret,
        execution_mode="agent",
        enable_agent=True,
        allow_side_effects=True,
        cwd=str(work),
        path_root=str(root),
        approval_mode="approve",
        builtins=["developer"],
        extensions=[],
        max_turns=8,
        max_tool_repetitions=3,
        timeout_seconds=60,
        max_output_bytes=65536,
        session_id="sess-1",
        allowed_cwd_roots=[str(root)],
    )
    assert out.get("status") == "success", out
    assert out["execution_mode"] == "agent"
    assert out["provider"] == "goose_cli"
    assert out["text"] == "agent-done" or out["result"] == "agent-done"
    assert out["side_effects_started"] is True
    assert out["tool_call_count"] >= 1
    assert out.get("session") == "sess-1"
    assert secret not in str(out)
    assert "prompt" not in out

    recorded = json.loads(argv_path.read_text(encoding="utf-8"))
    assert recorded["env_mode"] == "approve"
    assert recorded["env_path_root"] == str(root.resolve())
    assert "--with-builtin" in recorded["argv"]


def test_agent_rejects_unknown_authority_option(fake_bin: Path) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("x"))
    _register_goose(exe, "agent_unk", config={"enable_agent": True})
    adapter = get_cli_endpoint("agent_unk")
    assert isinstance(adapter, GooseCLIAdapter)

    out = adapter.execute(
        "x",
        execution_mode="agent",
        enable_agent=True,
        allow_side_effects=True,
        cwd="/tmp/w",
        path_root="/tmp",
        super_secret_allow_side_effects_override=True,  # unknown authority
    )
    assert out["status"] == "error"
    assert out["error_code"] == "policy_denied"
    assert "unknown authority" in (out.get("error") or "").lower()


def test_errors_never_echo_prompt_or_credentials(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(
        fake_bin,
        script=textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys
            if "--version" in sys.argv:
                print("goose {PINNED_GOOSE_VERSION}")
                sys.exit(0)
            print("auth failed: invalid api key sk-LEAKED", file=sys.stderr)
            sys.exit(1)
            """
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-REAL-CREDENTIAL-VALUE")
    _register_goose(exe, "err_g")
    adapter = get_cli_endpoint("err_g")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )
    adapter._get_provider().executable = str(exe)

    secret = "USER_PROMPT_CONFIDENTIAL_12345"
    out = execute_cli_inference("err_g", secret, timeout=10)
    assert out.get("status") == "error"
    blob = str(out)
    assert secret not in blob
    assert "sk-REAL-CREDENTIAL-VALUE" not in blob
    assert "prompt" not in out


# ---------------------------------------------------------------------------
# Stream / cancel lifecycle
# ---------------------------------------------------------------------------


def test_stream_and_cancel_lifecycle(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("streamed"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _register_goose(exe, "stream_g", config={"model": "m"})
    adapter = get_cli_endpoint("stream_g")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(
        PINNED_GOOSE_VERSION
    )
    adapter._get_provider().executable = str(exe)

    registry = get_default_endpoint_registry()
    events = list(registry.stream("stream_g", "hi stream"))
    assert events[0]["event"] == "started"
    assert events[-1]["event"] in {"completed", "failed"}
    # No prompt echo in events
    assert all("hi stream" not in str(e.get("error", "")) for e in events)

    cancel = registry.cancel("stream_g")
    assert cancel["status"] == "success"
    assert cancel.get("cancelled") is False  # nothing in-flight after stream done


def test_unsupported_version_on_chat_execute(fake_bin: Path) -> None:
    exe = _write_fake_goose(fake_bin, script=_version_only_script("1.0.0"))
    _register_goose(exe, "old_exec")
    adapter = get_cli_endpoint("old_exec")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = "1.0.0"
    adapter._get_provider().capabilities = capabilities_for_version("1.0.0")
    adapter._get_provider().executable = str(exe)

    out = adapter.execute("hello")
    assert out["status"] == "error"
    assert out.get("error_code") in {
        "unsupported_capability",
        "policy_denied",
        "internal",
    }
    # typed fields still present
    assert out.get("provider") == "goose_cli"
    assert out.get("execution_mode") == "chat"
    assert "side_effects_started" in out


def test_goose_tool_alias_goose_cli_registration(fake_bin: Path) -> None:
    exe = _write_fake_goose(fake_bin, script=_json_success_script("via-alias"))
    result = _register_goose(exe, "alias_ep", tool="goose_cli")
    assert result["status"] == "success"
    assert result.get("tool") in {"goose", "goose_cli"}
    adapter = get_cli_endpoint("alias_ep")
    assert isinstance(adapter, GooseCLIAdapter)


# ---------------------------------------------------------------------------
# Persistent Goose ACP session lifecycle (GOOSE-008)
# ---------------------------------------------------------------------------


def _write_fake_acp_server(directory: Path, name: str = "goose") -> Path:
    """Minimal NDJSON ACP fake for endpoint integration tests."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, sys, uuid

            sessions = {{}}

            def send(obj):
                sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\\n")
                sys.stdout.flush()

            def respond(msg_id, result):
                send({{"jsonrpc": "2.0", "id": msg_id, "result": result}})

            while True:
                line = sys.stdin.readline()
                if line == "":
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                method = msg.get("method")
                msg_id = msg.get("id")
                params = msg.get("params") or {{}}
                if method == "initialize":
                    respond(msg_id, {{
                        "protocolVersion": 1,
                        "agentCapabilities": {{"loadSession": True}},
                        "agentInfo": {{"name": "fake-acp", "version": "test"}},
                        "authMethods": [],
                    }})
                elif method == "session/new":
                    sid = "ep-" + uuid.uuid4().hex[:8]
                    sessions[sid] = True
                    respond(msg_id, {{"sessionId": sid}})
                elif method == "session/load":
                    sid = params.get("sessionId")
                    sessions[sid] = True
                    respond(msg_id, {{"sessionId": sid}})
                elif method == "session/close":
                    sessions.pop(params.get("sessionId"), None)
                    respond(msg_id, {{}})
                elif method == "session/cancel":
                    pass
                elif method == "session/prompt":
                    sid = params.get("sessionId")
                    send({{
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {{
                            "sessionId": sid,
                            "update": {{
                                "sessionUpdate": "agent_message_chunk",
                                "content": {{"type": "text", "text": "acp-ok"}},
                            }},
                        }},
                    }})
                    respond(msg_id, {{"stopReason": "end_turn"}})
                elif msg_id is not None:
                    send({{
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {{"code": -32601, "message": "unknown"}},
                    }})
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_acp_endpoint_session_lifecycle(fake_bin: Path, tmp_path: Path) -> None:
    """Endpoint-local ACP: start, session_new, prompt, stream, cancel, stop."""
    from ipfs_accelerate_py.cli_runtime.endpoints import EndpointLifecycleOp

    exe = _write_fake_acp_server(fake_bin)
    state_root = tmp_path / "acp-state"
    state_root.mkdir()
    _register_goose(
        exe,
        "acp_ep",
        config={
            "model": "m",
            "acp_state_root": str(state_root),
            "cli_path": str(exe),
        },
    )
    registry = get_default_endpoint_registry()

    # One-shot path still works independently (may fail without fake run handler,
    # so we only assert ACP path here).
    start = registry.acp_start(
        "acp_ep",
        executable=str(exe),
        state_root=str(state_root),
        restart_on_unexpected_exit=False,
    )
    assert start.get("success") is True, start
    assert start.get("auto_replay_agent_work") is False
    assert start["acp"]["state_root"] == str(state_root.resolve())

    desc = registry.acp_describe("acp_ep")
    assert desc["success"] is True
    assert desc["acp"]["ready"] is True

    sess = registry.session_new("acp_ep", cwd=str(state_root))
    assert sess.get("success") is True, sess
    sid = sess["session_id"]

    secret = "ENDPOINT_ACP_SECRET_PROMPT"
    prompt_out = registry.session_prompt("acp_ep", sid, secret)
    assert prompt_out.get("success") is True, prompt_out
    assert "acp-ok" in (prompt_out.get("text") or "")
    assert secret not in str(prompt_out)
    assert prompt_out.get("cacheable") is False
    assert prompt_out.get("retryable") is False

    events = list(registry.stream("acp_ep", "stream-secret", session_id=sid))
    assert events[0]["event"] == "started"
    assert any(e.get("event") == "completed" for e in events)
    assert all("stream-secret" not in str(e) for e in events)

    cancel = registry.cancel("acp_ep", session_id=sid)
    assert cancel["status"] == "success"

    # Dispatch surface
    dispatched = registry.dispatch(
        EndpointLifecycleOp.SESSION_CLOSE,
        endpoint_id="acp_ep",
        session_id=sid,
    )
    assert dispatched.get("closed") is True or dispatched.get("success") is True

    # Explicit restart never claims auto-replay
    restart = registry.acp_restart("acp_ep")
    assert restart.get("success") is True, restart
    assert restart.get("auto_replay_agent_work") is False

    stop = registry.acp_stop("acp_ep")
    assert stop.get("stopped") is True

    # Endpoint describe includes acp metadata when running
    start2 = registry.acp_start(
        "acp_ep", executable=str(exe), state_root=str(state_root)
    )
    assert start2["success"]
    listed = registry.list_endpoints(probe=False)
    acp_rows = [r for r in listed if r["endpoint_id"] == "acp_ep"]
    assert acp_rows
    assert acp_rows[0].get("acp") is not None
    registry.acp_stop("acp_ep")


def test_acp_requires_explicit_executable_and_state_root(
    fake_bin: Path,
) -> None:
    # Register without cli_path / state_root.
    result = register_cli_endpoint(
        tool="goose",
        endpoint_id="acp_missing",
        config={"model": "m"},
        replace=True,
    )
    assert result["status"] == "success"
    registry = get_default_endpoint_registry()
    out = registry.acp_start("acp_missing")
    assert out.get("success") is False
    assert out.get("status") == "error"
    # Must not start serve
    assert "serve" not in str(out).lower() or "not supported" in str(out).lower()


def test_acp_not_enabled_for_non_goose_tools(fake_bin: Path, tmp_path: Path) -> None:
    # Use factory to register a non-goose tool if available; otherwise skip.
    registry = get_default_endpoint_registry()
    try:
        result = registry.register_tool(
            "claude",
            endpoint_id="claude_no_acp",
            cli_path="/usr/bin/true",
            config={},
            replace=True,
        )
    except Exception:
        pytest.skip("claude adapter not constructible offline")
        return
    if result.get("status") != "success":
        pytest.skip("claude registration failed offline")
    out = registry.acp_start(
        "claude_no_acp",
        executable="/usr/bin/true",
        state_root=str(tmp_path / "s"),
    )
    assert out.get("success") is False
    assert out.get("error_code") in {
        "unsupported_capability",
        "policy_denied",
        "invalid_state",
    }


def test_unregister_stops_acp_client(fake_bin: Path, tmp_path: Path) -> None:
    exe = _write_fake_acp_server(fake_bin)
    state_root = tmp_path / "u-state"
    state_root.mkdir()
    _register_goose(
        exe,
        "acp_unreg",
        config={"acp_state_root": str(state_root), "cli_path": str(exe)},
    )
    registry = get_default_endpoint_registry()
    start = registry.acp_start(
        "acp_unreg", executable=str(exe), state_root=str(state_root)
    )
    assert start["success"]
    record = registry.get_record("acp_unreg")
    assert record is not None
    assert record.acp_client is not None
    assert registry.unregister("acp_unreg") is True
    assert registry.get_record("acp_unreg") is None


# ---------------------------------------------------------------------------
# GOOSE-011 security matrix anchors (endpoint surface)
# ---------------------------------------------------------------------------


def test_matrix_chat_safe_and_agent_requires_authorization(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat keeps safety flags; unauthorized agent is policy-denied."""
    argv_path = fake_bin / "mx_argv.json"
    exe = _write_fake_goose(fake_bin, script=_json_success_script("mx-chat"))
    monkeypatch.setenv("GOOSE_FAKE_ARGV_PATH", str(argv_path))
    monkeypatch.setenv("OPENAI_API_KEY", "matrix-cred-not-for-leak")
    _register_goose(exe, "mx_chat_ep", config={"model": "m"})
    adapter = get_cli_endpoint("mx_chat_ep")
    assert isinstance(adapter, GooseCLIAdapter)
    adapter._get_provider().version = PINNED_GOOSE_VERSION
    adapter._get_provider().capabilities = capabilities_for_version(PINNED_GOOSE_VERSION)
    adapter._get_provider().executable = str(exe)
    secret = "ENDPOINT_MATRIX_PROMPT_SECRET"
    out = execute_cli_inference("mx_chat_ep", secret, timeout=30)
    assert out.get("status") == "success", out
    assert secret not in str(out)
    assert "matrix-cred-not-for-leak" not in str(out)
    if argv_path.exists():
        recorded = json.loads(argv_path.read_text(encoding="utf-8"))
        assert "--no-profile" in recorded["argv"]
        assert recorded["env_mode"] == "chat"

    _register_goose(exe, "mx_agent_ep", config={"enable_agent": False})
    adapter2 = get_cli_endpoint("mx_agent_ep")
    assert isinstance(adapter2, GooseCLIAdapter)
    denied = adapter2.execute(
        "agent",
        execution_mode="agent",
        allow_side_effects=True,
        cwd="/tmp/w",
        path_root="/tmp",
    )
    assert denied.get("status") == "error"
    assert denied.get("error_code") == "policy_denied"



# ---------------------------------------------------------------------------
# GOOSE-012 operator documentation contracts (endpoint surface)
# ---------------------------------------------------------------------------


def test_goose_operator_docs_cover_endpoint_lifecycle() -> None:
    """Docs describe readiness vs liveness, agent endpoint auth, cancel/recovery."""
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "LLM_ROUTER.md").read_text(encoding="utf-8")
    for marker in (
        "readiness",
        "liveness",
        "enable_agent",
        "allow_side_effects",
        "path_root",
        "GOOSE_PATH_ROOT",
        "cancel",
        "side_effects_started",
        "execute_cli_inference",
        "register_cli_endpoint",
    ):
        assert marker in doc, f"missing endpoint guidance marker: {marker}"
    # Chat remains default; agent is separate
    assert "chat" in doc.lower()
    assert "agent" in doc.lower()
    # Managed install location for operators wiring endpoints
    assert "ipfs_accelerate_py/goose" in doc or "managed" in doc.lower()


def test_goose_operator_docs_quickstart_points_at_endpoint_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    quick = (root / "docs" / "guides" / "QUICKSTART.md").read_text(encoding="utf-8")
    assert "goose_cli" in quick
    assert "LLM_ROUTER" in quick or "llm router" in quick.lower()
    index = (root / "docs" / "INDEX.md").read_text(encoding="utf-8")
    assert "Goose" in index or "goose" in index
