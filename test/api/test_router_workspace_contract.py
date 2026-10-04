"""The coding adapter preserves native context and names its writable checkout."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("purpose", ["planning", "coding"])
def test_actual_router_input_and_native_context_have_distinct_verified_hashes(tmp_path, monkeypatch, purpose):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
                        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    observed = []

    def generate(prompt, **kwargs):
        observed.append(prompt)
        set_last_cli_observation("codex_cli", {"exit_code": 0})
        return "no external model call"

    monkeypatch.setattr(llm_router, "generate_text", generate)
    native = '  Native context\n{"query":"preserve  exact whitespace ☃"}\n/app/declared.py\n'
    _, receipt = runner.run(prompt=native, provider="codex_cli", model="pinned", timeout=1,
                            max_output_tokens=128, purpose=purpose)
    assert len(observed) == 1
    actual = observed[0]
    assert receipt["purpose"] == purpose
    assert receipt["native_prompt_sha256"] == receipt["prompt_sha256"] == hashlib.sha256(native.encode()).hexdigest()
    assert receipt["native_prompt_bytes"] == receipt["prompt_bytes"] == len(native.encode())
    assert receipt["model_prompt_sha256"] == hashlib.sha256(actual.encode()).hexdigest()
    assert receipt["model_prompt_bytes"] == len(actual.encode())
    assert receipt["completion_authority"] is False
    if purpose == "planning":
        assert actual == native
        assert receipt["workspace_advisory_bytes"] == 0
        assert receipt["workspace_advisory_sha256"] is None
    else:
        prefix = actual[:-len(native)]
        assert actual.endswith(native)
        assert str(tmp_path) in prefix
        assert "/app/<relative-path>" in prefix
        assert "Do not write to canonical /app" in prefix
        assert "keep the task's requested logical path names" in prefix
        assert "Do not stage or commit Git changes" in prefix
        assert receipt["workspace_advisory_bytes"] == len(prefix.encode())
        assert receipt["workspace_advisory_sha256"] == hashlib.sha256(prefix.encode()).hexdigest()
        assert receipt["model_prompt_sha256"] != receipt["native_prompt_sha256"]


def test_unknown_purpose_and_overlarge_coding_prompt_refuse_before_provider(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(llm_router, "generate_text", lambda *args, **kwargs: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="purpose"):
        runner.run(prompt="task", provider="codex_cli", model="pinned", timeout=1,
                   max_output_tokens=128, purpose="validation")
    with pytest.raises(ValueError, match="workspace contract exceeds"):
        runner.run(prompt="x" * 256_000, provider="codex_cli", model="pinned", timeout=1,
                   max_output_tokens=128)


def test_workspace_path_is_quoted_and_advisory_is_bounded():
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt

    native = "native prompt"
    result, advisory = render_model_prompt(prompt=native, purpose="coding", workspace=Path('/allocated/odd"\npath'))
    assert '"/allocated/odd\\"\\npath"' in advisory
    assert result == advisory + native
    with pytest.raises(ValueError, match="advisory exceeds"):
        render_model_prompt(prompt=native, purpose="coding", workspace=Path("/" + "x" * 9000))
