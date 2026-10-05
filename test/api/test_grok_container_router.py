"""Grok shares router dispatch while retaining its distinct native accounting."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.cli_runtime.grok_native_usage import (
    native_grok_observation, grok_outcome_receipt, grok_usage_receipt,
)


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
        assert kwargs["grok_max_turns"] == (2 if purpose == "planning" else 128)
        assert kwargs["timeout"] == 180
        assert kwargs["grok_disallowed_tools"] == "search_tool,use_tool"
        assert ("run_terminal_cmd" in kwargs["grok_tools"]) is (purpose == "coding")
        assert kwargs["grok_permission_mode"] == ("dontAsk" if purpose == "planning" else "bypassPermissions")
        set_last_cli_observation("grok_cli", {"exit_code": 0, **native_grok_observation(envelope(), exit_code=0)})
        return "done"
    monkeypatch.setattr(routed_workspace, "generate_text", generate)
    _, receipt = runner.run(prompt="instruction", provider="grok_cli", model="grok-4.7",
        timeout=180, max_output_tokens=10, purpose=purpose,
        **({"container_boundary": Path("/fixture/boundary"), "container_boundary_sha256": "a"*64} if purpose == "coding" else {}))
    assert receipt["native_rollout_usage"]["schema"] == "native-grok-final-usage@1"
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 33
    assert receipt["completion_authority"] is False
    assert receipt["provider_invocation_policy"] == {
        "max_turns": 2 if purpose == "planning" else 128,
        "tools_profile": "none" if purpose == "planning" else "isolated_coding",
        "permission_mode": "dontAsk" if purpose == "planning" else "bypassPermissions",
        "disallowed_tools": ["search_tool", "use_tool"],
    }
    assert receipt["timeout_seconds"] == 180


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


@pytest.mark.parametrize("payload,exit_code,timed_out,reason,stop", [
    (None, None, True, "timeout", "unknown"),
    ({"stopReason": "end_turn"}, 0, False, "end_turn", "end_turn"),
    ({"stopReason": "end_turn"}, 1, False, "process_error", "end_turn"),
    ({"type": "ERROR", "stopReason": "end_turn"}, 0, False, "execution_error", "end_turn"),
    ({"stopReason": "max_tokens"}, 0, False, "max_tokens", "max_tokens"),
    ({"stop_reason": "max_turn_requests"}, 1, False, "max_turns", "max_turn_requests"),
    ({"subtype": "error_max_turns"}, 1, False, "max_turns", "unknown"),
    ({"type": "max_turns_reached"}, 1, False, "max_turns", "unknown"),
    ({"subtype": "error_max_structured_output_retries"}, 1, False, "structured_output_retries", "unknown"),
    ({"subtype": "error_during_execution"}, 1, False, "execution_error", "unknown"),
    ({"stopReason": "refusal"}, 0, False, "refusal", "refusal"),
    ({"stopReason": "cancelled"}, 1, False, "cancelled", "cancelled"),
    ({"stopReason": "tool_use"}, 0, False, "other_stop", "tool_use"),
    ({"stopReason": "max_turn_requests"}, None, True, "timeout", "max_turn_requests"),
])
def test_closed_native_outcome_survives_absent_usage(payload, exit_code, timed_out, reason, stop):
    observed = native_grok_observation(payload, exit_code=exit_code, timed_out=timed_out)
    assert grok_usage_receipt(observed) is None
    receipt = grok_outcome_receipt(observed)
    assert receipt["reason_code"] == reason
    assert receipt["stop_reason"] == stop
    assert receipt["completion_authority"] is False
    assert receipt["raw_provider_data_exported"] is False


@pytest.mark.parametrize("message", ["max turns reached", "max turns reached (limit: 1)",
                                   "Reached the maximum number of turns"])
def test_native_plain_json_max_turn_guard_has_closed_message_classification(message):
    value = grok_outcome_receipt(native_grok_observation(
        {"type": "error", "message": message}, exit_code=1))
    assert value["reason_code"] == "max_turns"
    assert value["classification_source"] == "message_marker"
    assert "message" not in value


@pytest.mark.parametrize("payload", [
    {"text": "max turns reached (limit: 1)"},
    {"message": "max turns reached"},
    {"type": "error", "message": "private-prefix max turns reached"},
    {"type": "error", "message": "max turns reached\nprivate-tail"},
    {"type": "error", "message": "max turns reached (limit: 1234567)"},
    {"type": "error", "message": {"subtype": "error_max_turns"}},
    {"usage": {"native_grok_reason_code": "max_turns"}},
    {"text": {"stopReason": "max_turn_requests"}},
    {"stopReason": "end_turn", "stop_reason": "max_turn_requests"},
    {"stopReason": ["max_turn_requests"], "subtype": {"private": "error_max_turns"}, "type": []},
    {"stopReason": True, "subtype": 1, "type": {}},
])
def test_untrusted_or_malformed_outcome_fields_cannot_claim_a_turn_guard(payload):
    value = grok_outcome_receipt(native_grok_observation(payload, exit_code=1))
    assert value["reason_code"] != "max_turns"
    assert "private" not in json.dumps(value)


def test_closed_outcome_projection_rejects_spoofed_observation_fields():
    value = grok_outcome_receipt({
        "native_grok_reason_code": "private arbitrary body", "native_grok_stop_reason": [],
        "native_grok_error_subtype": True, "native_grok_process_outcome": {},
        "native_grok_classification_source": "private arbitrary body",
        "native_grok_error_observed": "true", "native_grok_envelope_observed": 1,
        "stop_reason": "max_turn_requests"})
    assert value["reason_code"] == value["stop_reason"] == value["error_subtype"] == "unknown"
    assert value["error_observed"] is value["envelope_observed"] is False
    assert "private" not in json.dumps(value)


def test_shared_adapter_retains_usage_free_native_error_outcome(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    payload = {"type": "error", "message": "max turns reached (limit: 1)",
               "text": "private native body", "usage": {"native_grok_stop_reason": "end_turn"}}
    monkeypatch.setattr(llm_router.subprocess, "run", lambda *_a, **_kw:
        SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr=""))
    with pytest.raises(llm_router.LLMRouterError):
        llm_router._get_grok_cli_provider().generate("private instruction", model_name="grok-4.7")
    observed = get_last_cli_observation("grok_cli")
    assert grok_usage_receipt(observed) is None
    assert grok_outcome_receipt(observed)["reason_code"] == "max_turns"
    assert grok_outcome_receipt(observed)["stop_reason"] == "unknown"


@pytest.mark.parametrize("second", ["private malformed JSON", json.dumps({
    "type": "error", "message": "private error", "stopReason": [],
    "native_grok_reason_code": "end_turn", "native_grok_usage_observed": True})])
def test_shared_adapter_does_not_reuse_previous_success_outcome(monkeypatch, second):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    results = iter([SimpleNamespace(returncode=0, stdout=json.dumps(envelope()), stderr=""),
                    SimpleNamespace(returncode=1, stdout=second, stderr="")])
    monkeypatch.setattr(llm_router.subprocess, "run", lambda *_a, **_kw: next(results))
    provider = llm_router._get_grok_cli_provider()
    assert provider.generate("first", model_name="grok-4.7") == "private-response"
    with pytest.raises(llm_router.LLMRouterError):
        provider.generate("second", model_name="grok-4.7")
    observed = get_last_cli_observation("grok_cli")
    assert grok_usage_receipt(observed) is None
    assert grok_outcome_receipt(observed)["reason_code"] in {"process_error", "execution_error"}
    assert grok_outcome_receipt(observed)["stop_reason"] == "unknown"
    assert "private" not in json.dumps(grok_outcome_receipt(observed))


@pytest.mark.parametrize("timeout", [False, True])
def test_runner_retains_usage_free_native_outcome_without_reusing_stale_state(
        routed_workspace, monkeypatch, capsys, timeout):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    set_last_cli_observation("grok_cli", native_grok_observation(envelope(), exit_code=0))
    def fail(*_a, **_kw):
        assert get_last_cli_observation("grok_cli") == {}
        payload = None if timeout else {"type": "error", "stopReason": "max_turn_requests"}
        set_last_cli_observation("grok_cli", native_grok_observation(
            payload, exit_code=None if timeout else 1, timed_out=timeout))
        if timeout:
            raise subprocess.TimeoutExpired(["grok", "private-command"], 1)
        raise routed_workspace.LLMRouterError("private native error")
    monkeypatch.setattr(routed_workspace, "generate_text", fail)
    with pytest.raises((routed_workspace.LLMRouterError, subprocess.TimeoutExpired)):
        runner.run(prompt="private instruction", provider="grok_cli", model="grok-4.7",
                   timeout=1, max_output_tokens=10, purpose="planning")
    raw = capsys.readouterr().out
    receipt = json.loads(raw)
    assert receipt["native_rollout_usage"] is None
    assert receipt["native_provider_outcome"]["reason_code"] == ("timeout" if timeout else "max_turns")
    assert receipt["usage"].get("stop_reason") == (None if timeout else "max_turn_requests")
    assert "private" not in raw


@pytest.mark.parametrize("option,expected", [({}, None), ({"grok_disallowed_tools": None}, None),
    ({"grok_disallowed_tools": ""}, None),
    ({"grok_disallowed_tools": "search_tool,use_tool"}, "search_tool,use_tool"),
    ({"grok_disallowed_tools": "search_tool, use_tool,search_tool"}, "search_tool,use_tool"),
    ({"grok_disallowed_tools": "Agent(explore)"}, "Agent(explore)")])
def test_shared_grok_disallowed_tools_is_opt_in_and_uses_literal_argv(monkeypatch, option, expected):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs.get("shell", False) is False
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope()), stderr="")
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    llm_router._get_grok_cli_provider().generate("instruction", model_name="grok-4.7", **option)
    command = commands[0]
    assert ("--disallowed-tools" in command) is (expected is not None)
    if expected is not None:
        assert command[command.index("--disallowed-tools") + 1] == expected


@pytest.mark.parametrize("value", [True, 1, [], {}, "x" * 4097, "search_tool,,use_tool",
    "--no-plan", "search_tool\nuse_tool", "$(private)", "Agent(", ",".join(["x"] * 33)])
def test_invalid_grok_disallowed_tools_fails_before_native_call(monkeypatch, value):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    monkeypatch.setattr(llm_router.subprocess, "run", lambda *_a, **_kw: pytest.fail("native call reached"))
    with pytest.raises(ValueError, match="grok_disallowed_tools"):
        llm_router._get_grok_cli_provider().generate("instruction", model_name="grok-4.7",
                                                    grok_disallowed_tools=value)


@pytest.mark.parametrize("suffix,accepted", [
    (["--disallowed-tools", "search_tool,use_tool"], True),
    (["--disallowed-tools=search_tool,use_tool"], True),
    (["--disallowed-tools", "read_file"], False),
    (["--disallowed-tools=read_file"], False),
    (["--disallowed-tools", "search_tool,use_tool", "--disallowed-tools", "read_file"], False),
    (["--disallowed-tools"], False),
])
def test_explicit_grok_tool_denial_cannot_be_weakened_by_custom_command(monkeypatch, suffix, accepted):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope()), stderr="")
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    provider = llm_router._get_grok_cli_provider()
    options = {"grok_cli_cmd": ["/fixture/grok", *suffix],
               "grok_disallowed_tools": "search_tool,use_tool"}
    if accepted:
        assert provider.generate("instruction", model_name="grok-4.7", **options) == "private-response"
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="conflicts"):
            provider.generate("instruction", model_name="grok-4.7", **options)
        assert calls == []


@pytest.mark.parametrize("command", ["/fixture/custom-wrapper", ["/fixture/grok", "--"],
                                    "/fixture/grok --"])
def test_grok_tool_denial_rejects_unsupported_wrapper_or_option_terminator(monkeypatch, command):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    monkeypatch.setattr(llm_router, "_run_cli_command", lambda *_a, **_kw: pytest.fail("wrapper reached"))
    monkeypatch.setattr(llm_router.subprocess, "run", lambda *_a, **_kw: pytest.fail("native call reached"))
    with pytest.raises(ValueError, match="grok_disallowed_tools"):
        llm_router._get_grok_cli_provider().generate("instruction", model_name="grok-4.7",
            grok_cli_cmd=command, grok_disallowed_tools="search_tool,use_tool")


@pytest.mark.parametrize("options", [{}, {"grok_disallowed_tools": None}, {"grok_disallowed_tools": ""}])
def test_nonstructured_grok_wrapper_preserves_default_behavior(monkeypatch, options):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    calls = []
    def execute(*args, **kwargs):
        calls.append(args)
        return "done"
    monkeypatch.setattr(llm_router, "_run_cli_command", execute)
    assert llm_router._get_grok_cli_provider().generate("instruction", model_name="grok-4.7",
        grok_cli_cmd="/fixture/custom-wrapper", **options) == "done"
    assert len(calls) == 1
