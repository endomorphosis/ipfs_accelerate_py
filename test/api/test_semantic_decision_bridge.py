"""Native producer -> pinned provider adapter -> selective delivery, offline."""

import json
import os
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import (
    FINAL_SCHEMA, REQUEST_SCHEMA, ROUND_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_provider import run_shared_router_decision_dialogue

pytest_plugins = ("test.api.test_semantic_router_integration",)


@pytest.mark.parametrize("provider_name", ["grok_cli", "codex_cli"])
def test_native_packet_reaches_pinned_router_and_selective_continuation(allocated, monkeypatch, provider_name):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    repository, workspace, prompt, source = allocated
    parent = json.loads((repository / ".runtime/semantic/worker-context.json").read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=ids[:1])
    calls = []
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: [provider_name])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
                        SimpleNamespace(provider=provider_name, model_name="pinned", catalog_revision="fixture"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(llm_router, "_grok_cli_command", lambda: "/fixture/grok")
    monkeypatch.setattr(grok_cli_runner, "_resolve_trusted_grok_bin", lambda **kwargs: "/fixture/grok")

    def generate(model_prompt, **kwargs):
        calls.append((model_prompt, kwargs))
        protocol = next(json.loads(line) for line in model_prompt.splitlines()
                        if line.startswith('{"binding_sha256":') and ROUND_SCHEMA in line)
        assert kwargs["provider"] == provider_name and kwargs["model_name"] == "pinned"
        assert kwargs["allow_cross_provider_fallback"] is False
        set_last_cli_observation(provider_name, {
            "exit_code": 0, "model_id": "pinned", "prompt_tokens": 20 + len(calls),
            "completion_tokens": 5, "cached_tokens": 10,
        })
        response = {key: protocol[key] for key in ("session_id", "generation", "binding_sha256")}
        if len(calls) == 1:
            response.update(schema=REQUEST_SCHEMA, capsule_ids=[ids[0]])
        else:
            assert "Supervisor verified capsule deliveries:" in model_prompt
            assert protocol["generation"] == 1
            response.update(schema=FINAL_SCHEMA, text="Candidate prepared; native validation remains required.")
            if provider_name == "codex_cli":
                (workspace / "mod.py").write_text(source + "\n# candidate-only private edit\n")
        return json.dumps(response)

    monkeypatch.setattr(llm_router, "generate_text", generate)
    result = run_shared_router_decision_dialogue(
        native_prompt=prompt, repository=repository, workspace=workspace, policy=policy,
        provider=provider_name, model="pinned", timeout=90,
        prompt_token_counter=PromptTokenCounter(
            provider=provider_name, model="pinned", tokenizer_id="fixture", tokenizer_revision="1",
            special_token_policy="authored fixture only", count_kind="estimate",
            compatibility_basis="authored_fixture_only",
            count_tokens=lambda text: 100 if ROUND_SCHEMA in text else 1000,
        ),
    )
    assert len(calls) == len(result.receipt["attempts"]) == 2
    assert len(result.receipt["expansions"]) == 1
    assert result.receipt["expansions"][0]["capsule_ids"] == [ids[0]]
    assert (repository / "mod.py").read_text() == source
    assert result.receipt["completion_authority"] is False
    assert result.receipt["whole_session_token_savings"] is None
    assert result.receipt["provider_usage_complete"] is False
    assert [row["provider_receipt"]["usage"]["prompt_tokens"] for row in result.receipt["attempts"]] == [21, 22]
    if provider_name == "grok_cli":
        assert all(options["grok_tools"] == "read_file"
                   and options["grok_disallowed_tools"] == "read_file,search_tool,use_tool"
                   and options["grok_permission_mode"] == "dontAsk"
                   and options["grok_max_turns"] == 1 for _, options in calls)
        assert all(row["provider_receipt"]["effective_toolset_verified"] is False
                   for row in result.receipt["attempts"])
        assert calls[0][1]["chat_session_id"] != calls[1][1]["chat_session_id"]
    else:
        assert (workspace / "mod.py").read_text() != source
        assert all(row["provider_receipt"]["native_invocation"]["schema"] == "router-implementation-invocation@1"
                   for row in result.receipt["attempts"])


@pytest.mark.parametrize("provider_fails", [False, True])
def test_receipt_observer_cannot_erase_failure_or_skip_environment_restoration(
    allocated, provider, monkeypatch, provider_fails,
):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    state, _ = provider
    monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "workspace-write")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "owner-original")
    observed = []

    def observe(receipt):
        observed.append(receipt)
        raise ValueError("authored observer failure")

    def fail():
        raise TimeoutError("authored provider failure")

    if provider_fails:
        state["after_generate"] = fail
        with pytest.raises(TimeoutError, match="authored provider failure"):
            runner.run(prompt="native fixture", provider="codex_cli", model="pinned", timeout=1,
                       max_output_tokens=128, receipt_observer=observe)
    else:
        _, receipt = runner.run(prompt="native fixture", provider="codex_cli", model="pinned", timeout=1,
                                max_output_tokens=128, receipt_observer=observe)
        assert receipt["receipt_observer_error_type"] == "ValueError"
    assert len(observed) == 1 and observed[0]["usage"]["prompt_tokens"] == 123
    assert os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] == "workspace-write"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "owner-original"
