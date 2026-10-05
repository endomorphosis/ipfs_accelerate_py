"""Grok shares router dispatch while retaining its distinct native accounting."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation, grok_usage_receipt


def envelope(**changes):
    return {"text": "private-response", "stopReason": "end_turn", "num_turns": 2,
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 20,
                      "cache_creation_input_tokens": 0, "output_tokens": 3, "total_tokens": 33}, **changes}


def test_final_native_categories_preserve_cache_semantics_and_unknown_completeness():
    value = grok_usage_receipt(native_grok_observation(envelope(), exit_code=0))
    assert value["usage"] == {"input_tokens": 10, "cached_input_tokens": 20,
                              "cache_write_input_tokens": 0, "output_tokens": 3, "total_tokens": 33}
    assert value["cache_included_in_input"] is False
    assert value["totals_scope"] == "native_final_envelope"
    assert value["task_complete_observed"] is True
    assert value["usage_complete_observed"] is None
    assert value["billing_total_verified"] is False
    assert "private-response" not in json.dumps(value)


@pytest.mark.parametrize("value", [None, {}, {"modelUsage": {"grok": {"input_tokens": 900}}},
    {"text": json.dumps({"usage": {"input_tokens": 900}})}, {"usage": []},
    {"usage": {"input_tokens": True}}, {"usage": {"input_tokens": -1}},
    {"usage": {"input_tokens": 1.0}}, {"usage": {"input_tokens": "1"}},
    {"usage": {"input_tokens": 3, "inputTokens": 4}},
    {"usage": {"native_grok_usage_observed": True, "native_grok_input_tokens": 99}}])
def test_non_native_and_invalid_usage_remains_unknown(value):
    assert grok_usage_receipt(native_grok_observation(value, exit_code=0)) is None


@pytest.mark.parametrize("status,timeout", [(1, False), (None, True)])
def test_partial_native_failure_keeps_observed_categories(status, timeout):
    observed = native_grok_observation(envelope(usage={"input_tokens": 10}), exit_code=status, timed_out=timeout)
    value = grok_usage_receipt(observed)
    assert value["usage"] == {"input_tokens": 10}
    assert value["task_complete_observed"] is False


@pytest.mark.parametrize("stop_reason", [[], {}, True, 1, None])
def test_invalid_native_stop_reason_cannot_assert_completion(stop_reason):
    value = grok_usage_receipt(native_grok_observation(envelope(stopReason=stop_reason), exit_code=0))
    assert value["usage"]["total_tokens"] == 33
    assert value["task_complete_observed"] is False


@pytest.mark.parametrize("exit_code", [0, 1])
def test_shared_grok_adapter_records_native_success_and_failure_usage(monkeypatch, exit_code):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    payload = envelope(**({"type": "error", "message": "not signed in private-marker"} if exit_code else {}))
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        assert Path(command[command.index("--prompt-file") + 1]).read_text() == "private-prompt"
        return SimpleNamespace(returncode=exit_code, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    provider = llm_router._get_grok_cli_provider()
    if exit_code:
        with pytest.raises(llm_router.LLMRouterError):
            provider.generate("private-prompt", model_name="grok-4.7", grok_tools="", grok_max_turns=1)
    else:
        assert provider.generate("private-prompt", model_name="grok-4.7", grok_tools="", grok_max_turns=1) == "private-response"
    observed = get_last_cli_observation("grok_cli")
    assert observed["exit_code"] == exit_code
    assert grok_usage_receipt(observed)["usage"]["total_tokens"] == 33
    assert grok_usage_receipt(observed)["task_complete_observed"] is (exit_code == 0)
    assert not Path(commands[0][commands[0].index("--prompt-file") + 1]).exists()


def test_shared_grok_timeout_removes_prompt_and_keeps_partial_native_usage(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    prompts = []
    def execute(command, **kwargs):
        prompts.append(Path(command[command.index("--prompt-file") + 1]))
        raise subprocess.TimeoutExpired(command, 1, output=json.dumps(envelope()).encode())
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    with pytest.raises(subprocess.TimeoutExpired):
        llm_router._get_grok_cli_provider().generate("private", model_name="grok-4.7", timeout=1)
    observed = get_last_cli_observation("grok_cli")
    assert observed["timed_out"] is True
    assert grok_usage_receipt(observed)["task_complete_observed"] is False
    assert "exit_code" not in observed
    assert not prompts[0].exists()


@pytest.fixture
def routed_workspace(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **_:
        SimpleNamespace(provider="grok_cli", model_name="grok-4.7", catalog_revision="fixture"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *_, **__: object())
    return llm_router


@pytest.mark.parametrize("purpose", ["planning", "coding"])
def test_runner_uses_explicit_grok_tools_and_separate_accounting(routed_workspace, monkeypatch, purpose):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", lambda **_: {"fixture": True})
    def generate(prompt, **kwargs):
        assert kwargs["provider"] == "grok_cli"
        assert kwargs["allow_local_fallback"] is kwargs["allow_cross_provider_fallback"] is False
        assert kwargs["grok_max_turns"] == (1 if purpose == "planning" else 128)
        assert ("run_terminal_cmd" in kwargs["grok_tools"]) is (purpose == "coding")
        assert kwargs["grok_permission_mode"] == ("dontAsk" if purpose == "planning" else "bypassPermissions")
        set_last_cli_observation("grok_cli", {"exit_code": 0, **native_grok_observation(envelope(), exit_code=0)})
        return "done"
    monkeypatch.setattr(routed_workspace, "generate_text", generate)
    _, receipt = runner.run(prompt="instruction", provider="grok_cli", model="grok-4.7",
        timeout=1, max_output_tokens=10, purpose=purpose,
        **({"container_boundary": Path("/fixture/boundary"), "container_boundary_sha256": "a"*64} if purpose == "coding" else {}))
    assert receipt["native_rollout_usage"]["schema"] == "native-grok-final-usage@1"
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 33
    assert receipt["completion_authority"] is False


def test_grok_coding_requires_container_boundary_before_provider_call(routed_workspace, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    monkeypatch.setattr(routed_workspace, "generate_text", lambda *_a, **_kw: pytest.fail("provider reached"))
    with pytest.raises(ValueError, match="verified isolated container"):
        runner.run(prompt="instruction", provider="grok_cli", model="grok-4.7", timeout=1, max_output_tokens=10)


def test_runner_grok_failure_receipt_keeps_partial_usage_without_text(routed_workspace, monkeypatch, capsys):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    def fail(*_a, **_kw):
        set_last_cli_observation("grok_cli", {"exit_code": 1,
            **native_grok_observation(envelope(), exit_code=1)})
        raise routed_workspace.LLMRouterError("not authenticated private-auth-body")
    monkeypatch.setattr(routed_workspace, "generate_text", fail)
    with pytest.raises(routed_workspace.LLMRouterError):
        runner.run(prompt="private-instruction", provider="grok_cli", model="grok-4.7",
            timeout=1, max_output_tokens=10, purpose="planning")
    raw = capsys.readouterr().out
    receipt = json.loads(raw)
    assert receipt["status"] == "failed"
    assert receipt["provider_failure"]["reason_code"] == "authentication"
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 33
    assert "private-auth-body" not in raw and "private-instruction" not in raw
