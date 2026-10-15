"""Shared-router decision adapters preserve binding and failed-attempt usage."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def decision_worktree(tmp_path, monkeypatch):
    repository, workspace = tmp_path / "source", tmp_path / "work"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run([
        "git", "-C", str(repository), "-c", "user.name=Adapter fixture",
        "-c", "user.email=adapter@example.invalid", "commit", "-q", "--allow-empty",
        "-m", "Fixture baseline",
    ], check=True)
    subprocess.run([
        "git", "-C", str(repository), "worktree", "add", "-q", "-b", "attempt",
        str(workspace),
    ], check=True)
    monkeypatch.chdir(workspace)
    return repository, workspace


def _adapter_and_call(decision_worktree, provider):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
        PromptTokenCounter, render_model_prompt,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import SemanticDecisionCall
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_provider import SharedRouterDecisionProvider

    repository, workspace = decision_worktree
    counter = PromptTokenCounter(
        provider=provider, model="pinned", tokenizer_id="fixture",
        tokenizer_revision="fixture@1", special_token_policy="literal",
        count_tokens=lambda text: len(text), count_kind="estimate",
        compatibility_basis="authored fixture only",
    )
    prompt = "Return one checked semantic decision."
    rendered, _ = render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
    call = SemanticDecisionCall(
        call_id="a" * 32, session_id="b" * 32, generation=0,
        binding_sha256="c" * 64, provider=provider, model="pinned", prompt=prompt,
        timeout=30, max_output_tokens=512,
        model_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
    )
    adapter = SharedRouterDecisionProvider(
        provider=provider, model="pinned", repository=repository, workspace=workspace,
        prompt_token_counter=counter,
    )
    return adapter, call, rendered


def _grok_route(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner

    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
                        SimpleNamespace(provider="grok_cli", model_name="pinned"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(llm_router, "_grok_cli_command", lambda: "/fixture/bin/grok")
    monkeypatch.setattr(grok_cli_runner, "_resolve_trusted_grok_bin", lambda **kwargs: "/fixture/bin/grok")
    return llm_router


@pytest.mark.parametrize("provider", ["codex_cli", "grok_cli"])
def test_decision_adapter_keeps_complete_wire_and_exact_route(decision_worktree, monkeypatch, provider):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    adapter, call, rendered = _adapter_and_call(decision_worktree, provider)
    invocations = []
    if provider == "codex_cli":
        def execute(**kwargs):
            invocations.append(kwargs)
            native = {
                "provider": provider, "model": "pinned", "model_prompt_sha256": call.model_prompt_sha256,
                "usage": {"prompt_tokens": 17, "completion_tokens": 3, "cached_tokens": 5},
                "native_rollout_usage": {"usage": {"input_tokens": 17, "output_tokens": 3}},
            }
            kwargs["receipt_observer"](native)
            return "decision", native
        monkeypatch.setattr(runner, "run", execute)
    else:
        llm_router = _grok_route(monkeypatch)
        def execute(prompt, **kwargs):
            invocations.append({"prompt": prompt, **kwargs})
            set_last_cli_observation(provider, {
                "exit_code": 0, "prompt_tokens": 17, "completion_tokens": 3, "cached_tokens": 5,
            })
            return "decision"
        monkeypatch.setattr(llm_router, "generate_text", execute)
    reply = adapter(call)
    assert reply.text == "decision"
    assert reply.receipt["provider"] == provider and reply.receipt["model"] == "pinned"
    assert reply.receipt["model_prompt_sha256"] == call.model_prompt_sha256
    assert reply.receipt["usage"] == {
        "prompt_tokens": 17, "completion_tokens": 3, "cached_tokens": 5,
        **({"exit_code": 0} if provider == "grok_cli" else {}),
    }
    assert reply.receipt["usage_status"] == "reported"
    assert reply.receipt["completion_authority"] is False
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation["timeout"] == 30 and invocation["max_output_tokens" if provider == "codex_cli" else "max_tokens"] == 512
    if provider == "codex_cli":
        assert invocation["prompt"] == call.prompt and invocation["semantic_repository"] is None
        assert invocation["prompt_token_counter"] is adapter.prompt_token_counter
    else:
        assert invocation["prompt"] == rendered
        assert invocation["allow_local_fallback"] is False and invocation["allow_cross_provider_fallback"] is False
        assert invocation["grok_tools"] == "" and invocation["grok_max_turns"] == 1
        assert invocation["grok_cli_cmd"] == ["/fixture/bin/grok"]
        assert invocation["chat_session_id"] == call.call_id
        assert "resume_session_id" not in invocation and "continue_session" not in invocation


@pytest.mark.parametrize("provider,reported", [("codex_cli", True), ("grok_cli", True), ("grok_cli", False)])
def test_decision_adapter_retains_failed_usage_without_inventing_zero(decision_worktree, monkeypatch, provider, reported):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import SemanticDecisionProviderFailure
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    adapter, call, _ = _adapter_and_call(decision_worktree, provider)
    if provider == "codex_cli":
        def execute(**kwargs):
            kwargs["receipt_observer"]({
                "provider": provider, "model": "pinned", "model_prompt_sha256": call.model_prompt_sha256,
                "status": "failed", "usage": {"prompt_tokens": 9, "timed_out": True},
            })
            raise subprocess.TimeoutExpired(["fixture"], 30)
        monkeypatch.setattr(runner, "run", execute)
    else:
        llm_router = _grok_route(monkeypatch)
        # A previous successful invocation must not contaminate this timeout.
        set_last_cli_observation(provider, {"exit_code": 0, "prompt_tokens": 99, "completion_tokens": 99})
        def execute(prompt, **kwargs):
            if reported:
                set_last_cli_observation(provider, {"prompt_tokens": 9, "timed_out": True})
            raise subprocess.TimeoutExpired(["fixture"], 30)
        monkeypatch.setattr(llm_router, "generate_text", execute)
    with pytest.raises(SemanticDecisionProviderFailure) as captured:
        adapter(call)
    receipt = captured.value.receipt
    assert receipt["status"] == "failed" and receipt["provider_called"] is True
    assert receipt["provider"] == provider and receipt["model_prompt_sha256"] == call.model_prompt_sha256
    assert receipt["usage_status"] == ("partial" if reported else "unknown")
    assert receipt["usage"] == ({"prompt_tokens": 9, "timed_out": True} if reported else {})
    assert "completion_tokens" not in receipt["usage"]


@pytest.mark.parametrize("alteration", ["model", "wire", "cwd", "authority"])
def test_decision_adapter_rejects_changed_attempt_before_provider(decision_worktree, monkeypatch, alteration):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    adapter, call, _ = _adapter_and_call(decision_worktree, "codex_cli")
    invocations = []
    monkeypatch.setattr(runner, "run", lambda **kwargs: invocations.append(kwargs))
    if alteration == "model":
        call = replace(call, model="foreign")
    elif alteration == "wire":
        call = replace(call, model_prompt_sha256="d" * 64)
    elif alteration == "cwd":
        monkeypatch.chdir(decision_worktree[0])
    else:
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "fixture-only")
    with pytest.raises(ValueError):
        adapter(call)
    assert not invocations


@pytest.mark.parametrize("command", ["custom-wrapper", "grok --tools read_file"])
def test_grok_adapter_refuses_commands_that_can_bypass_tool_free_profile(decision_worktree, monkeypatch, command):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import SemanticDecisionProviderFailure

    adapter, call, _ = _adapter_and_call(decision_worktree, "grok_cli")
    llm_router = _grok_route(monkeypatch)
    monkeypatch.setattr(llm_router, "_grok_cli_command", lambda: command)
    invocations = []
    monkeypatch.setattr(llm_router, "generate_text", lambda *args, **kwargs: invocations.append(kwargs))
    with pytest.raises(SemanticDecisionProviderFailure) as captured:
        adapter(call)
    assert captured.value.receipt["provider_called"] is False
    assert captured.value.receipt["usage"] is None and captured.value.receipt["usage_status"] == "unknown"
    assert not invocations


@pytest.mark.parametrize("refusal", ["untrusted_binary", "project_extension"])
def test_grok_adapter_reuses_existing_executable_and_extension_guards(decision_worktree, monkeypatch, refusal):
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import SemanticDecisionProviderFailure

    adapter, call, _ = _adapter_and_call(decision_worktree, "grok_cli")
    llm_router = _grok_route(monkeypatch)
    invocations = []
    monkeypatch.setattr(llm_router, "generate_text", lambda *args, **kwargs: invocations.append(kwargs))
    if refusal == "untrusted_binary":
        monkeypatch.setattr(grok_cli_runner, "_resolve_trusted_grok_bin", lambda **kwargs: "")
    else:
        extension = decision_worktree[1] / ".mcp.json"
        extension.write_text("{}")
    with pytest.raises(SemanticDecisionProviderFailure) as error:
        adapter(call)
    assert error.value.receipt["provider_called"] is False
    assert not invocations


@pytest.mark.parametrize("outcome", ["nonzero", "empty", "unknown_exit"])
def test_grok_adapter_refuses_unobserved_or_empty_success(decision_worktree, monkeypatch, outcome):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import SemanticDecisionProviderFailure
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    adapter, call, _ = _adapter_and_call(decision_worktree, "grok_cli")
    llm_router = _grok_route(monkeypatch)
    def execute(prompt, **kwargs):
        set_last_cli_observation("grok_cli", {
            "prompt_tokens": 8, "completion_tokens": 2,
            **({} if outcome == "unknown_exit" else {"exit_code": 1 if outcome == "nonzero" else 0}),
        })
        return "" if outcome == "empty" else "partial output"
    monkeypatch.setattr(llm_router, "generate_text", execute)
    with pytest.raises(SemanticDecisionProviderFailure) as captured:
        adapter(call)
    assert captured.value.receipt["usage"]["prompt_tokens"] == 8
    assert captured.value.receipt["status"] == "failed"


@pytest.mark.parametrize("automatic", [True, False])
def test_shared_router_convenience_binds_adapter_and_counter(decision_worktree, monkeypatch, automatic):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_dialogue as dialogue
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_provider import (
        SharedRouterDecisionProvider, run_shared_router_decision_dialogue,
    )

    original, _, _ = _adapter_and_call(decision_worktree, "grok_cli")
    counter = original.prompt_token_counter
    resolutions, invocations = [], []
    provenance = {"selection": "authored automatic fixture", "remote_compatibility_verified": False}
    def resolve(**kwargs):
        resolutions.append(kwargs)
        return counter, provenance
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", resolve)
    policy = SemanticDecisionPolicy((), ())
    cancellation = lambda: False
    def execute(**kwargs):
        invocations.append(kwargs)
        assert isinstance(kwargs["invoke"], SharedRouterDecisionProvider)
        assert kwargs["invoke"].provider == "grok_cli" and kwargs["invoke"].model == "pinned"
        assert kwargs["invoke"].prompt_token_counter is counter and kwargs["count_tokens"] is counter
        assert kwargs["invoke"].repository == decision_worktree[0]
        assert kwargs["invoke"].workspace == decision_worktree[1]
        assert kwargs["policy"] is policy and kwargs["cancelled"] is cancellation
        assert (kwargs["max_calls"], kwargs["timeout"], kwargs["max_output_tokens"], kwargs["max_input_tokens"]) == (2, 45, 1024, 100_000)
        assert (kwargs["max_expansion_bytes"], kwargs["max_expansion_tokens"]) == (8192, 2048)
        return dialogue.SemanticDecisionResult("done", json.dumps({"provider": "grok_cli"}))
    monkeypatch.setattr(dialogue, "run_semantic_decision_dialogue", execute)
    result = run_shared_router_decision_dialogue(
        native_prompt="fixture task", repository=decision_worktree[0], workspace=decision_worktree[1],
        policy=policy, provider="grok_cli", model="pinned",
        prompt_token_counter=None if automatic else counter,
        max_calls=2, timeout=45, max_output_tokens=1024, max_input_tokens=100_000,
        max_expansion_bytes=8192, max_expansion_tokens=2048, cancelled=cancellation,
    )
    assert result.text == "done" and len(invocations) == 1
    assert resolutions == ([{"provider": "grok_cli", "model": "pinned"}] if automatic else [])
    assert result.receipt.get("prompt_tokenizer_source") == (provenance if automatic else None)
