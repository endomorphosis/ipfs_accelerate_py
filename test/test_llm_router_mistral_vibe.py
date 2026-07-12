from __future__ import annotations

import subprocess

import pytest

import ipfs_accelerate_py.llm_router as llm_router
from ipfs_accelerate_py.utils.mistral_vibe import MistralVibeInstallResult


_VIBE_ENV_NAMES = (
    "IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD",
    "IPFS_ACCELERATE_PY_MISTRAL_VIBE_CLI_CMD",
    "ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD",
    "IPFS_ACCELERATE_MISTRAL_VIBE_MODEL",
    "IPFS_ACCELERATE_PY_MISTRAL_VIBE_MODEL",
    "ipfs_accelerate_py_MISTRAL_VIBE_MODEL",
)


def _clear_vibe_env(monkeypatch):
    for name in _VIBE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_explicit_mistral_vibe_provider_auto_installs(monkeypatch):
    _clear_vibe_env(monkeypatch)
    install_calls = []
    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: False)

    def fake_install(**kwargs):
        install_calls.append(kwargs)
        return MistralVibeInstallResult(
            available=True,
            installed=True,
            executable="/home/test/.local/bin/vibe",
            method="uv_tool",
        )

    monkeypatch.setattr(llm_router, "ensure_mistral_vibe", fake_install)

    provider = llm_router._builtin_provider_by_name("mistral_vibe", auto_install=True)

    assert provider is not None
    assert install_calls == [{"auto_install": True}]


def test_automatic_provider_discovery_does_not_install_vibe(monkeypatch):
    _clear_vibe_env(monkeypatch)
    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: False)
    monkeypatch.setattr(
        llm_router,
        "ensure_mistral_vibe",
        lambda **_kwargs: pytest.fail("generic discovery must not install optional CLIs"),
    )

    assert llm_router._builtin_provider_by_name("mistral_vibe") is None


def test_mistral_vibe_provider_forwards_model_agent_and_prompt(monkeypatch):
    _clear_vibe_env(monkeypatch)
    captured = {}
    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)

    def fake_run(command, prompt, **kwargs):
        captured.update(command=command, prompt=prompt, kwargs=kwargs)
        return '{"classification":"compiler_rule_gap"}'

    monkeypatch.setattr(llm_router, "_run_cli_command", fake_run)
    provider = llm_router._get_mistral_vibe_provider()

    assert provider is not None
    response = provider.generate(
        "audit this",
        model_name="Leanstral",
        mistral_vibe_agent="lean",
        timeout=12,
    )

    assert response == '{"classification":"compiler_rule_gap"}'
    assert "--agent {agent}" in captured["command"]
    assert captured["prompt"] == "audit this"
    assert captured["kwargs"]["template_vars"] == {
        "agent": "lean",
        "model": "Leanstral",
    }
    assert captured["kwargs"]["extra_env"]["VIBE_ACTIVE_MODEL"] == "Leanstral"
    assert captured["kwargs"]["timeout_seconds"] == 12


def test_mistral_vibe_provider_rejects_unsafe_agent_name(monkeypatch):
    _clear_vibe_env(monkeypatch)
    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    provider = llm_router._get_mistral_vibe_provider()

    assert provider is not None
    with pytest.raises(ValueError, match="mistral_vibe_agent"):
        provider.generate("audit", mistral_vibe_agent="lean; rm -rf")


def test_cli_template_values_remain_single_arguments(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ROUTER_OK", stderr="")

    monkeypatch.setattr(llm_router.subprocess, "run", fake_run)

    result = llm_router._run_cli_command(
        "vibe --prompt {prompt} --agent {agent} --output text",
        "Return exactly ROUTER_OK and do not use tools.",
        template_vars={"agent": "lean agent"},
    )

    assert result == "ROUTER_OK"
    assert captured["command"] == [
        "vibe",
        "--prompt",
        "Return exactly ROUTER_OK and do not use tools.",
        "--agent",
        "lean agent",
        "--output",
        "text",
    ]
    assert captured["kwargs"]["input"] is None
