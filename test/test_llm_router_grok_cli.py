"""Tests for the first-class Grok CLI LLM router provider."""

from __future__ import annotations

import json
import subprocess

import pytest
from ipfs_accelerate_py import llm_router


class _Provider:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str, **kwargs: object) -> str:
        return f"{self.name}:{prompt}"


def _successful_grok_result(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    payload = {
        "text": "grok-cli-ok",
        "stopReason": "EndTurn",
        "sessionId": "session-1",
        "requestId": "request-1",
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")


def test_grok_cli_discovery_survives_systemd_minimal_path(
    monkeypatch,
    tmp_path,
) -> None:
    fake_home = tmp_path / "home"
    fake_grok = fake_home / ".local" / "bin" / "grok"
    fake_grok.parent.mkdir(parents=True)
    fake_grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_grok.chmod(0o700)
    auth_path = fake_home / ".grok" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv(
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    )
    for name in (
        "ipfs_accelerate_py_GROK_CLI_CMD",
        "IPFS_ACCELERATE_PY_GROK_CLI_CMD",
        "IPFS_DATASETS_PY_GROK_CLI_CMD",
        "IPFS_ACCELERATE_AGENT_GROK_BIN",
        "GROK_CLI_CMD",
        "GROK_BIN",
        "GROK_HOME",
        "XAI_API_KEY",
        "ipfs_accelerate_py_XAI_API_KEY",
        "IPFS_ACCELERATE_PY_XAI_API_KEY",
        "IPFS_DATASETS_PY_XAI_API_KEY",
        "GROK_AUTH_PROVIDER_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)

    assert llm_router.find_grok_cli() == str(fake_grok)
    assert llm_router._grok_cli_command() == str(fake_grok)
    assert llm_router._grok_cli_auth_available() is True
    assert llm_router._get_grok_cli_provider() is not None


def test_grok_cli_provider_uses_bounded_headless_json_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        captured["prompt"] = open(cmd[cmd.index("--prompt-file") + 1], encoding="utf-8").read()
        return _successful_grok_result(list(cmd))

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(llm_router.subprocess, "run", fake_run)
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")
    monkeypatch.setenv("ipfs_accelerate_py_XAI_API_KEY", "alternate-test-key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    result = provider.generate(
        'Reply to "quoted text".',
        model_name="grok-4.5",
        timeout=12,
    )

    assert result == "grok-cli-ok"
    assert captured["prompt"] == 'Reply to "quoted text".'
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "grok"
    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-plan" in cmd
    assert "--no-subagents" in cmd
    assert "--disable-web-search" in cmd
    assert "--no-memory" in cmd
    assert "--verbatim" in cmd
    assert 'Reply to "quoted text".' not in cmd
    assert captured["env"]["XAI_API_KEY"] == "alternate-test-key"


def test_grok_cli_agent_command_is_noninteractive(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("implement the task", encoding="utf-8")

    cmd = llm_router.build_grok_cli_command(
        mode="agent",
        workspace=tmp_path,
        model_name="grok-4.5",
        max_turns=100_000,
        grok_bin="grok",
        prompt_file=prompt_path,
    )

    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert cmd[cmd.index("--max-turns") + 1] == "100000"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    assert cmd[cmd.index("--output-format") + 1] == "plain"
    assert cmd[cmd.index("--cwd") + 1] == str(tmp_path.resolve())
    assert cmd[cmd.index("--prompt-file") + 1] == str(prompt_path)
    assert "--always-approve" in cmd
    assert "--tools" not in cmd


def test_grok_cli_provider_reports_missing_auth(monkeypatch) -> None:
    error = {
        "type": "error",
        "message": "Not signed in. To authenticate, run grok login --device-code.",
    }

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout=json.dumps(error),
            stderr="Error: Not signed in",
        ),
    )
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    with pytest.raises(llm_router.LLMRouterError, match="grok login --device-code"):
        provider.generate("hello", disable_model_retry=True)


def test_grok_cli_provider_rejects_empty_success_payload(monkeypatch) -> None:
    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"text": "", "stopReason": "EndTurn"}),
            stderr="",
        ),
    )
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    with pytest.raises(llm_router.LLMRouterError, match="returned no response text"):
        provider.generate("hello")


def test_grok_cli_command_template_preserves_prompt_as_one_argument(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout="template-ok", stderr="")

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(llm_router.subprocess, "run", fake_run)
    monkeypatch.setenv(
        "ipfs_accelerate_py_GROK_CLI_CMD",
        "grok-wrapper --model {model} --prompt {prompt}",
    )

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    result = provider.generate('Keep "this prompt" together.', model_name="grok-4.5")

    assert result == "template-ok"
    assert captured["cmd"] == [
        "grok-wrapper",
        "--model",
        "grok-4.5",
        "--prompt",
        'Keep "this prompt" together.',
    ]
    assert captured["input"] is None


def test_grok_cli_trace_records_usage_without_prompt(monkeypatch, tmp_path) -> None:
    secret_prompt = "private prompt that must not be traced"

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda cmd, **kwargs: _successful_grok_result(list(cmd)),
    )
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")

    trace_path = tmp_path / "grok.jsonl"
    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    assert provider.generate(secret_prompt, trace_jsonl_path=str(trace_path)) == "grok-cli-ok"

    trace_text = trace_path.read_text(encoding="utf-8")
    record = json.loads(trace_text)
    assert record["provider"] == "grok_cli"
    assert record["sessionId"] == "session-1"
    assert record["usage"]["total_tokens"] == 5
    assert secret_prompt not in trace_text


def test_grok_alias_prefers_cli_and_explicit_api_alias_uses_rest(monkeypatch) -> None:
    cli = _Provider("cli")
    api = _Provider("api")
    monkeypatch.setattr(llm_router, "_get_grok_cli_provider", lambda: cli)
    monkeypatch.setattr(llm_router, "_get_xai_provider", lambda: api)

    assert llm_router._builtin_provider_by_name("grok") is cli
    assert llm_router._builtin_provider_by_name("grok_cli") is cli
    assert llm_router._builtin_provider_by_name("xai_cli") is cli
    assert llm_router._builtin_provider_by_name("xai") is api
    assert llm_router._builtin_provider_by_name("grok_api") is api


def test_grok_alias_falls_back_to_rest_when_cli_is_missing(monkeypatch) -> None:
    api = _Provider("api")
    monkeypatch.setattr(llm_router, "_get_grok_cli_provider", lambda: None)
    monkeypatch.setattr(llm_router, "_get_xai_provider", lambda: api)

    assert llm_router._builtin_provider_by_name("grok") is api


def test_grok_cli_auto_discovery_requires_auth(monkeypatch) -> None:
    calls: list[str] = []
    grok = _Provider("grok")

    def fake_builtin(name: str):
        calls.append(name)
        return grok if name == "grok_cli" else None

    monkeypatch.delenv("ipfs_accelerate_py_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm_router, "_get_accelerate_provider", lambda _deps: None)
    monkeypatch.setattr(llm_router, "_get_local_hf_provider", lambda **kwargs: None)
    monkeypatch.setattr(llm_router, "_builtin_provider_by_name", fake_builtin)
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: False)

    with pytest.raises(RuntimeError, match="No LLM provider available"):
        llm_router._resolve_provider_uncached(None, deps=llm_router.get_default_router_deps())
    assert "grok_cli" not in calls

    calls.clear()
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: True)
    assert llm_router._resolve_provider_uncached(
        None,
        deps=llm_router.get_default_router_deps(),
    ) is grok
    assert "grok_cli" in calls
