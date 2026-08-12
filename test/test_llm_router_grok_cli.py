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
        model_name="grok-4.6",
        timeout=12,
    )

    assert result == "grok-cli-ok"
    assert captured["prompt"] == 'Reply to "quoted text".'
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "grok"
    assert cmd[cmd.index("--model") + 1] == "grok-4.6"
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
        model_name="grok-4.6",
        max_turns=100_000,
        grok_bin="grok",
        prompt_file=prompt_path,
    )

    assert cmd[cmd.index("--model") + 1] == "grok-4.6"
    assert cmd[cmd.index("--max-turns") + 1] == "100000"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    assert cmd[cmd.index("--output-format") + 1] == "plain"
    assert cmd[cmd.index("--cwd") + 1] == str(tmp_path.resolve())
    assert cmd[cmd.index("--prompt-file") + 1] == str(prompt_path)
    assert "--always-approve" in cmd
    assert "--tools" not in cmd


def test_grok_cli_agent_command_supports_fail_closed_sandbox_and_denies(
    tmp_path,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("implement the task", encoding="utf-8")

    cmd = llm_router.build_grok_cli_command(
        mode="agent",
        workspace=tmp_path,
        model_name="grok-4.6",
        grok_bin="grok",
        prompt_file=prompt_path,
        sandbox_profile="provider-isolated",
        deny_rules=("Bash(codex *)", "Bash(copilot *)"),
    )

    assert cmd[cmd.index("--sandbox") + 1] == "provider-isolated"
    assert cmd.count("--deny") == 2
    assert "Bash(codex *)" in cmd
    assert "Bash(copilot *)" in cmd


def test_grok_cli_isolated_env_withholds_alternate_provider_authority() -> None:
    source = {
        "PATH": "/usr/local/bin:/usr/bin",
        "HOME": "/home/runner",
        "XAI_API_KEY": "grok-authority",
        "CODEX_HOME": "/private/codex",
        "OPENAI_API_KEY": "openai-authority",
        "COPILOT_GITHUB_TOKEN": "copilot-authority",
        "GH_TOKEN": "github-authority",
        "GITHUB_TOKEN": "github-authority-2",
        "ipfs_accelerate_py_CODEX_MODEL": "gpt-5.6-sol",
        "IPFS_ACCELERATE_PY_OPENAI_API_KEY": "package-openai-authority",
        "ipfs_accelerate_py_OPENAI_BASE_URL": "https://peer.invalid/v1",
        "IPFS_DATASETS_PY_OPENAI_MODEL": "peer-model",
        "AZURE_OPENAI_API_KEY": "azure-openai-authority",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-sol",
        "IPFS_ACCELERATE_AGENT_COPILOT_MODEL": "gpt-5.6-sol",
        "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI": "1",
        "DOCKER_HOST": "unix:///run/docker.sock",
        "DOCKER_AUTH_CONFIG": "container-registry-authority",
        "CONTAINER_HOST": "unix:///run/podman/podman.sock",
        "KUBECONFIG": "/private/kube-config",
        "GOOSE_PROVIDER": "openai",
        "META_AI_API_KEY": "meta-authority",
        "MODEL_API_KEY": "meta-authority-2",
        "ANTHROPIC_API_KEY": "anthropic-authority",
        "IPFS_ACCELERATE_PY_GEMINI_CLI_CMD": "gemini",
        "IPFS_DATASETS_PY_MISTRAL_API_KEY": "mistral-authority",
        "OPENROUTER_API_KEY": "openrouter-authority",
        "HF_TOKEN": "hugging-face-authority",
        "IPFS_ACCELERATE_PY_HF_API_TOKEN": "hf-package-authority",
        "ipfs_accelerate_py_HF_API_TOKEN": "hf-package-authority-2",
        "IPFS_DATASETS_PY_HF_API_TOKEN": "hf-package-authority-3",
        "HUGGINGFACEHUB_API_TOKEN": "hf-hub-authority",
        "GOOGLE_API_KEY": "gemini-authority",
        "DATABRICKS_TOKEN": "goose-backend-authority",
        "GROQ_API_KEY": "goose-backend-authority-2",
        "LLVM_API_KEY": "llvm-authority",
        "OVMS_API_KEY": "ovms-authority",
        "VLLM_API_KEY": "vllm-authority",
        "S3_ACCESS_KEY": "s3-access-authority",
        "S3_SECRET_KEY": "s3-secret-authority",
        "GROK_AUTH_PROVIDER_COMMAND": "/workspace/steal-auth",
        "GROK_CODE_BACKEND_URL": "https://redirect.invalid",
        "GROK_MODELS_LIST_URL": "https://redirect.invalid/models",
        "XAI_API_BASE_URL": "https://redirect.invalid/v1",
        "CLI_CHAT_PROXY_BASE_URL": "https://redirect.invalid/proxy",
        "GROK_MANAGED_CONFIG_URL": "https://redirect.invalid/config",
        "GROK_WORKSPACE_BUNDLED_SKILLS_DIR": "/workspace/injected-skills",
        "GROK_SANDBOX_AUTO_ALLOW_BASH": "1",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "SSH_AUTH_SOCK": "/run/user/1000/ssh-agent",
        "UNRELATED_SETTING": "retained",
    }

    isolated = llm_router.build_grok_cli_env(
        base_env=source,
        isolate_alternate_providers=True,
    )

    assert isolated["XAI_API_KEY"] == "grok-authority"
    assert "UNRELATED_SETTING" not in isolated
    assert isolated["PATH"] == "/usr/bin:/bin"
    for forbidden in (
        "CODEX_HOME",
        "OPENAI_API_KEY",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "ipfs_accelerate_py_CODEX_MODEL",
        "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
        "ipfs_accelerate_py_OPENAI_BASE_URL",
        "IPFS_DATASETS_PY_OPENAI_MODEL",
        "AZURE_OPENAI_API_KEY",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL",
        "IPFS_ACCELERATE_AGENT_COPILOT_MODEL",
        "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI",
        "DOCKER_HOST",
        "DOCKER_AUTH_CONFIG",
        "CONTAINER_HOST",
        "KUBECONFIG",
        "GOOSE_PROVIDER",
        "META_AI_API_KEY",
        "MODEL_API_KEY",
        "ANTHROPIC_API_KEY",
        "IPFS_ACCELERATE_PY_GEMINI_CLI_CMD",
        "IPFS_DATASETS_PY_MISTRAL_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "IPFS_ACCELERATE_PY_HF_API_TOKEN",
        "ipfs_accelerate_py_HF_API_TOKEN",
        "IPFS_DATASETS_PY_HF_API_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "GOOGLE_API_KEY",
        "DATABRICKS_TOKEN",
        "GROQ_API_KEY",
        "LLVM_API_KEY",
        "OVMS_API_KEY",
        "VLLM_API_KEY",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "GROK_AUTH_PROVIDER_COMMAND",
        "GROK_CODE_BACKEND_URL",
        "GROK_MODELS_LIST_URL",
        "XAI_API_BASE_URL",
        "CLI_CHAT_PROXY_BASE_URL",
        "GROK_MANAGED_CONFIG_URL",
        "GROK_WORKSPACE_BUNDLED_SKILLS_DIR",
        "GROK_SANDBOX_AUTO_ALLOW_BASH",
        "DBUS_SESSION_BUS_ADDRESS",
        "SSH_AUTH_SOCK",
    ):
        assert forbidden not in isolated
    assert isolated["GROK_CODEX_SKILLS_ENABLED"] == "0"
    assert isolated["GROK_CODEX_SESSIONS_ENABLED"] == "0"
    # The caller/parent mapping remains able to authorize a later quota-gated
    # fallback; sanitization applies only to Grok's child environment.
    assert source["OPENAI_API_KEY"] == "openai-authority"
    assert source["CODEX_HOME"] == "/private/codex"


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
    result = provider.generate('Keep "this prompt" together.', model_name="grok-4.6")

    assert result == "template-ok"
    assert captured["cmd"] == [
        "grok-wrapper",
        "--model",
        "grok-4.6",
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
