"""Tests for Muse Code CLI integration facade and command construction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from ipfs_accelerate_py.cli_integrations.muse_code_cli_integration import (
    MuseCodeCLIIntegration,
    reset_muse_code_cli_integration,
)
from ipfs_accelerate_py.cli_runtime.providers.muse import (
    build_muse_exec_argv,
    classify_muse_failure,
    parse_muse_jsonl,
)


def test_login_present_does_not_inject_stored_key(tmp_path: Path, monkeypatch) -> None:
    from ipfs_accelerate_py.cli_runtime.providers.muse import build_muse_process_env

    home = tmp_path / "home"
    auth = home / ".config" / "muse"
    auth.mkdir(parents=True)
    (auth / "auth.json").write_text('{"login":true}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(
        "ipfs_accelerate_py.common.meta_model_api.resolve_meta_model_api_key",
        lambda *a, **k: "stored-secret-key",
    )
    overlay = build_muse_process_env(base_env=os.environ)
    assert overlay.get("META_API_KEY") != "stored-secret-key"
    assert "META_API_KEY" not in overlay


def test_build_muse_exec_argv_uses_exec_and_prompt_file() -> None:
    argv = build_muse_exec_argv(
        executable="muse",
        prompt_file="/tmp/prompt.txt",
        model_name="muse-spark-1.2",
        max_model_steps=8,
        json_events=True,
        disable_approval=True,
    )
    assert argv[0] == "muse"
    assert "exec" in argv
    assert "--prompt-file" in argv
    assert argv[argv.index("--prompt-file") + 1] == "/tmp/prompt.txt"
    assert "--json" in argv
    assert "--disable-approval" in argv
    assert "--yolo" not in argv
    assert "--model" in argv
    assert "muse-spark-1.2" in argv
    # Prompt text must not appear as a free argv blob.
    assert not any("write a function" in part for part in argv)


def test_build_muse_exec_argv_yolo_omits_disable_approval() -> None:
    argv = build_muse_exec_argv(
        executable="/usr/bin/muse",
        prompt_file="/tmp/p.txt",
        yolo=True,
        disable_approval=True,
        json_events=False,
    )
    assert "--yolo" in argv
    assert "--disable-approval" not in argv
    assert "--json" not in argv


def test_parse_muse_jsonl_last_assistant_text() -> None:
    stdout = "\n".join(
        [
            '{"type":"started"}',
            '{"type":"text","role":"assistant","text":"hello"}',
            '{"type":"tool_call","name":"shell"}',
            '{"type":"text","role":"assistant","text":"done"}',
            '{"type":"completed","status":"ok"}',
        ]
    )
    parsed = parse_muse_jsonl(stdout)
    assert parsed.text == "done"
    assert parsed.tool_call_count == 1
    assert parsed.raw_format == "jsonl"
    assert parsed.side_effects_started is True


def test_parse_muse_jsonl_falls_back_to_plain_text() -> None:
    parsed = parse_muse_jsonl("plain assistant output\n")
    assert parsed.text == "plain assistant output"
    assert parsed.raw_format == "text"


def test_parse_muse_jsonl_captures_session_id() -> None:
    stdout = "\n".join(
        [
            '{"type":"session_start","session_id":"11111111-2222-3333-4444-555555555555"}',
            '{"type":"text","role":"assistant","text":"continuing"}',
            '{"type":"completed","session":{"id":"11111111-2222-3333-4444-555555555555"}}',
        ]
    )
    parsed = parse_muse_jsonl(stdout)
    assert parsed.metadata.get("session_id") == "11111111-2222-3333-4444-555555555555"
    assert parsed.text == "continuing"


def test_parse_muse_jsonl_envelope_skips_user_prompt() -> None:
    stdout = "\n".join(
        [
            '{"payload_type":"turn.input.user","stream":{"kind":"session","id":"sess-live"},"payload":{"text":"Reply with exactly the two words: hello world"}}',
            '{"payload_type":"turn.output.assistant","payload":{"text":"hello world","role":"assistant"}}',
        ]
    )
    parsed = parse_muse_jsonl(stdout)
    assert parsed.text == "hello world"
    assert parsed.metadata.get("session_id") == "sess-live"


def test_parse_muse_jsonl_collects_run_metadata_without_prompt() -> None:
    stdout = "\n".join(
        [
            '{"payload_type":"run.model.configured","stream":{"kind":"session","id":"sess-1"},"payload":{"model_id":"muse-spark-1.3-contributor","display_label":"muse-spark-1.3-contributor","profile_id":"tbh","provider_id":"meta","command_id":"cmd-1","run_stream":{"id":"run-1","kind":"run"}}}',
            '{"payload_type":"turn.input.user","payload":{"prompt":"SECRET_PROMPT_SHOULD_NOT_LEAK","command_id":"cmd-1"}}',
            '{"payload_type":"task.lifecycle.status","payload":{"command_id":"cmd-1","event":{"details":{"phase":"stream_succeeded","facets":[{"kind":"external_attempt","attempt":1,"max_attempts":10,"operation":"model.response","error_kind":"stream_succeeded","system":"meta"},{"kind":"producer","detail":{"kind":"provider","model":"muse-spark-1.3-contributor","provider":"meta","request_id":"req-1","response_id":"resp-1","stream":{"bytes_read":172793,"time_to_first_event_ms":45,"wire_events_seen":11,"last_wire_event_type":"response.completed"}}}]}}}}',
            '{"payload_type":"run.output.delta","payload":{"text":"hello"}}',
            '{"payload_type":"run.output.delta","payload":{"text":" world"}}',
            '{"payload_type":"run.terminal.completed","payload":{"terminal":"completed","text":"hello world","run_stream":{"id":"run-1"}}}',
        ]
    )
    parsed = parse_muse_jsonl(stdout)
    assert parsed.text == "hello world"
    assert parsed.metadata.get("session_id") == "sess-1"
    assert parsed.metadata.get("run_id") == "run-1"
    assert parsed.metadata.get("model_id") == "muse-spark-1.3-contributor"
    assert parsed.metadata.get("provider_id") == "meta"
    assert parsed.metadata.get("request_id") == "req-1"
    assert parsed.metadata.get("response_id") == "resp-1"
    assert parsed.metadata.get("time_to_first_event_ms") == "45"
    assert parsed.metadata.get("stream_bytes_read") == "172793"
    assert "SECRET_PROMPT" not in json.dumps(parsed.metadata)
    assert "prompt" not in parsed.metadata


def test_parse_muse_jsonl_echo_provider_output_delta() -> None:
    stdout = "\n".join(
        [
            '{"payload_type":"turn.input.user","payload":{"prompt":"hello world"}}',
            '{"payload_type":"run.output.delta","payload":{"text":"echo: hello world\\n"}}',
            '{"payload_type":"run.terminal.completed","payload":{"text":"echo: hello world\\n"}}',
        ]
    )
    parsed = parse_muse_jsonl(stdout)
    assert "hello world" in parsed.text.lower()


def test_classify_muse_failure_exit_codes() -> None:
    assert classify_muse_failure(exit_code=2, stdout="", stderr="bad flag") == classify_muse_failure(
        exit_code=2, stdout="", stderr="usage"
    )
    from ipfs_accelerate_py.cli_runtime.providers.muse import MuseErrorKind

    assert classify_muse_failure(exit_code=2, stdout="", stderr="x") is MuseErrorKind.USAGE_ERROR
    assert classify_muse_failure(exit_code=130, stdout="", stderr="") is MuseErrorKind.CANCELLATION
    assert (
        classify_muse_failure(exit_code=1, stdout="", stderr="not logged in")
        is MuseErrorKind.AUTHENTICATION
    )
    assert (
        classify_muse_failure(
            exit_code=1,
            stdout="",
            stderr="Meta AI HTTP 429: rate limit exceeded; model is currently overloaded",
        )
        is MuseErrorKind.QUOTA_RATE_LIMIT
    )
    assert (
        classify_muse_failure(
            exit_code=1,
            stdout="",
            stderr="Error: insufficient_quota — payment required for Meta AI",
        )
        is MuseErrorKind.BILLING
    )


def test_constructing_muse_facade_starts_no_subprocess() -> None:
    reset_muse_code_cli_integration()
    with patch("subprocess.run", side_effect=AssertionError("init probe")):
        integ = MuseCodeCLIIntegration(muse_path="/nonexistent/muse-binary-xyz")
        assert integ.get_tool_name() == "Muse Code CLI"
        assert integ.is_available(probe=False) is False


def test_muse_chat_delegates_to_canonical_adapter() -> None:
    mock_adapter = MagicMock()
    mock_adapter.generate.return_value = "hello from muse"

    integ = MuseCodeCLIIntegration(adapter=mock_adapter)
    result = integ.chat("hi", model="muse-spark-1.2", timeout=12.0)

    mock_adapter.generate.assert_called_once()
    args, kwargs = mock_adapter.generate.call_args
    assert args[0] == "hi"
    assert kwargs.get("model_name") == "muse-spark-1.2"
    assert kwargs.get("agent") is False
    assert kwargs.get("timeout") == 12.0
    assert kwargs.get("disable_approval") is True
    assert result["success"] is True
    assert result["text"] == "hello from muse"
    assert result["provider"] == "muse_code"
    assert result["side_effecting"] is True
    assert result["command_contract"] == "muse exec"


def test_muse_agent_delegates_with_side_effect_flags() -> None:
    mock_adapter = MagicMock()
    mock_adapter.generate.return_value = "agent done"

    integ = MuseCodeCLIIntegration(adapter=mock_adapter)
    result = integ.agent("do work", workspace="/tmp/ws", max_model_steps=5, yolo=False)

    mock_adapter.generate.assert_called_once()
    _, kwargs = mock_adapter.generate.call_args
    assert kwargs.get("agent") is True
    assert kwargs.get("workspace") == "/tmp/ws"
    assert kwargs.get("allow_side_effects") is True
    assert kwargs.get("yolo") is False
    assert result["side_effecting"] is True
    assert result["mode"] == "agent"


def test_muse_get_adapter_uses_create_muse_provider() -> None:
    sentinel = MagicMock(name="MuseCLIProvider")
    with patch(
        "ipfs_accelerate_py.cli_runtime.providers.muse.create_muse_provider",
        return_value=sentinel,
    ) as create:
        integ = MuseCodeCLIIntegration(muse_path="/usr/bin/muse")
        adapter = integ.get_adapter()
        assert adapter is sentinel
        create.assert_called_once()
        assert integ.get_adapter() is sentinel
        assert create.call_count == 1


def test_muse_provider_generate_uses_exec_and_parses_jsonl(tmp_path: Path) -> None:
    from ipfs_accelerate_py.cli_runtime.process_runner import ProcessRunResult
    from ipfs_accelerate_py.cli_runtime.providers.muse import MuseCLIProvider

    captured: dict[str, object] = {}

    class _FakeRunner:
        def run(self, spec, **_kwargs):
            captured["argv"] = list(spec.argv)
            captured["cwd"] = spec.cwd
            return ProcessRunResult(
                exit_code=0,
                stdout='{"type":"text","role":"assistant","text":"refactored"}\n',
                stderr="",
                elapsed_seconds=0.01,
                ok=True,
            )

    binary = tmp_path / "muse"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    provider = MuseCLIProvider(
        executable=str(binary),
        runner=_FakeRunner(),
        default_model="muse-spark-1.2",
    )
    text = provider.generate("refactor auth", workspace=str(tmp_path), max_model_steps=4)
    assert text == "refactored"
    argv = captured["argv"]
    assert argv[0] == str(binary)
    assert "exec" in argv
    assert "--disable-approval" in argv
    assert "--prompt-file" in argv
    assert "--max-model-steps" in argv
    assert "4" in argv
    assert captured["cwd"] == str(tmp_path)


def test_discover_prefers_configured_path(tmp_path: Path, monkeypatch) -> None:
    from ipfs_accelerate_py.cli_integrations.muse_code_cli_integration import (
        _discover_muse_executable,
    )

    binary = tmp_path / "muse"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("IPFS_ACCELERATE_MUSE_PATH", str(binary))
    assert _discover_muse_executable() == str(binary)
