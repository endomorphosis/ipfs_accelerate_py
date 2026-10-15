"""Authored native-rollout observations survive the real router receipt path.

Only the external provider and its discovery/allocation are replaced. Native
recovery, telemetry, prompt rendering, failure handling and benchmark accounting
run normally against private pytest-owned files; no model or credentials are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.native_usage_accounting import native_usage_contract
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.codex_interaction_telemetry import (
    MAX_TOKEN_OBSERVATIONS,
    MAX_TOOL_PAIRS,
)


THREAD = "authored-telemetry-thread-123456"
PRIVATE_COMMAND = "PRIVATE_COMMAND_cat_private_source_秘密"
PRIVATE_OUTPUT = "PRIVATE_OUTPUT_credentials_and_source_秘密"
PRIVATE_ID = "PRIVATE_CALL_IDENTIFIER_123456"
PRIVATE_TOOL = "PRIVATE_ARBITRARY_TOOL_NAME"
FIRST_USAGE = {
    "input_tokens": 1000, "cached_input_tokens": 800,
    "output_tokens": 20, "reasoning_output_tokens": 10, "total_tokens": 1020,
}
FINAL_USAGE = {
    "input_tokens": 1300, "cached_input_tokens": 1024,
    "output_tokens": 40, "reasoning_output_tokens": 20, "total_tokens": 1340,
}


def _event(kind, **payload):
    return {"type": "event_msg", "payload": {"type": kind, **payload}}


def _checkpoint(usage):
    return _event("token_count", info={"total_token_usage": dict(usage)})


def _tool_records():
    return [
        {"timestamp": "2026-10-11T00:00:01Z", "type": "response_item", "payload": {
            "type": "function_call", "name": "functions.exec_command", "call_id": PRIVATE_ID,
            "arguments": json.dumps({"cmd": PRIVATE_COMMAND}, ensure_ascii=False)}},
        {"timestamp": "2026-10-11T00:00:03Z", "type": "response_item", "payload": {
            "type": "function_call_output", "call_id": PRIVATE_ID,
            "output": {"exit_code": 0, "stdout": PRIVATE_OUTPUT}}},
        {"timestamp": "2026-10-11T00:00:04Z", "type": "response_item", "payload": {
            "type": "custom_tool_call", "name": PRIVATE_TOOL, "call_id": PRIVATE_ID + "-2",
            "input": PRIVATE_COMMAND}},
        {"timestamp": "2026-10-11T00:00:05Z", "type": "response_item", "payload": {
            "type": "custom_tool_call_output", "call_id": PRIVATE_ID + "-2",
            "output": PRIVATE_OUTPUT}},
    ]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    workspace = tmp_path / "allocated"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    native_home = tmp_path / "private-native-storage"
    native_home.mkdir(mode=0o700)
    folder = native_home / "sessions/2026/10/11"
    folder.mkdir(parents=True)
    rollout = folder / ("rollout-" + THREAD + ".jsonl")
    monkeypatch.chdir(workspace)
    # Scoped to this test and restored by pytest; never access the user's home.
    monkeypatch.setenv("CODEX_HOME", str(native_home))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **_:
        SimpleNamespace(provider="codex_cli", model_name="authored-model",
                        reasoning_effort="high", catalog_revision="authored-test"))
    instance = object()
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *_, **__: instance)
    calls = []
    prompt = "Repair the authored fixture; preserve literal $(text) and Unicode λ."
    output = "Authored provider output is unchanged.\n"

    def write(records, *, cwd=None, malformed=False):
        metadata = {"type": "session_meta", "payload": {
            "id": THREAD, "cwd": str(cwd if cwd is not None else workspace)}}
        rollout.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                     for row in [metadata, *records]) + "\n", encoding="utf-8")
        if malformed:
            with rollout.open("a", encoding="utf-8") as stream:
                stream.write("{incomplete native record")
        rollout.chmod(0o600)

    def dispatch(*, exit_code=0, timeout=False):
        def generate(provider_prompt, **kwargs):
            calls.append((provider_prompt, kwargs))
            observation = {"thread_id": THREAD, "prompt_tokens": 15, "completion_tokens": 4}
            observation.update({"timed_out": True} if timeout else {"exit_code": exit_code})
            set_last_cli_observation("codex_cli", observation)
            if timeout:
                raise TimeoutError("authored timeout " + PRIVATE_OUTPUT)
            return output

        monkeypatch.setattr(llm_router, "generate_text", generate)
        return runner.run(prompt=prompt, provider="codex_cli", model="authored-model",
                          timeout=30, max_output_tokens=512, reasoning_effort="high")

    return SimpleNamespace(workspace=workspace, rollout=rollout, write=write,
                           dispatch=dispatch, calls=calls, instance=instance,
                           prompt=prompt, output=output)


def _printed_receipt(capsys):
    captured = capsys.readouterr()
    receipts = []
    for line in captured.out.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("schema") == "router-implementation-invocation@1":
            receipts.append(value)
    assert len(receipts) == 1
    # Tool text and arbitrary native IDs cannot escape through stdout or stderr.
    for private in (PRIVATE_COMMAND, PRIVATE_OUTPUT, PRIVATE_ID, PRIVATE_TOOL):
        assert private not in captured.out + captured.err
        assert private not in json.dumps(receipts[0], ensure_ascii=False)
    return receipts[0], captured.out


def test_complete_native_telemetry_reaches_accounting_without_changing_dispatch(pipeline, capsys):
    rows = [_event("task_started"), _checkpoint(FIRST_USAGE), *_tool_records(),
            _checkpoint(FINAL_USAGE), _event("task_complete")]
    pipeline.write(rows)
    output, returned = pipeline.dispatch()
    emitted, _ = _printed_receipt(capsys)
    assert output == pipeline.output and returned == emitted
    assert len(pipeline.calls) == 1
    prompt, options = pipeline.calls[0]
    expected_prompt, _ = runner.render_model_prompt(
        prompt=pipeline.prompt, purpose="coding", workspace=pipeline.workspace)
    assert prompt == expected_prompt and prompt.endswith(pipeline.prompt)
    assert options["provider_instance"] is pipeline.instance
    assert options["allocation_session_id"] == emitted["invocation_id"]
    assert set(options) == {
        "provider", "model_name", "provider_instance", "deps", "allow_local_fallback",
        "allow_cross_provider_fallback", "timeout", "max_tokens", "max_new_tokens",
        "reasoning_effort", "side_effecting", "task_kind", "allocation_path", "allocation_session_id",
    }
    assert {key: value for key, value in options.items()
            if key not in {"provider_instance", "deps", "allocation_session_id"}} == {
        "provider": "codex_cli", "model_name": "authored-model", "allow_local_fallback": False,
        "allow_cross_provider_fallback": False, "timeout": 30, "max_tokens": 512,
        "max_new_tokens": 512, "reasoning_effort": "high", "side_effecting": True,
        "task_kind": "coding", "allocation_path": "cli",
    }
    assert emitted["model_prompt_bytes"] == len(prompt.encode("utf-8"))
    assert emitted["model_prompt_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert emitted["usage"]["prompt_tokens"] == 15
    native = emitted["native_rollout_usage"]
    assert native["usage"] == FINAL_USAGE
    assert native["rollout_sha256"] == hashlib.sha256(pipeline.rollout.read_bytes()).hexdigest()
    assert native_usage_contract(emitted, native) == (True, "native_session_cumulative", True)
    assert native["usage"]["total_tokens"] == native["usage"]["input_tokens"] + native["usage"]["output_tokens"]
    telemetry = native["interaction_telemetry"]
    assert telemetry["schema"] == "native-codex-interaction-telemetry@1"
    assert telemetry["tool_observations_complete"] is True
    assert telemetry["native_aggregate_usage_modified"] is False
    assert telemetry["raw_content_exported"] is False
    assert telemetry["provider_request_count_known"] is False
    assert telemetry["causal_tool_token_attribution"] is False
    first, second = telemetry["token_observations"]
    assert first["adjacent_counter_delta"] is None and first["interval_status"] == "unknown_baseline"
    assert second["adjacent_counter_delta"] == {
        "input_tokens": 300, "cached_input_tokens": 224, "output_tokens": 20,
        "reasoning_output_tokens": 10, "total_tokens": 320,
    }
    assert second["observed_tool_call_events_since_previous_checkpoint"] == 2
    tool, custom = telemetry["tool_pairs"]
    expected_output = json.dumps(_tool_records()[1]["payload"]["output"], ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert tool["observed_output_bytes"] == len(expected_output)
    assert tool["recorded_observation_elapsed_seconds"] == 2.0
    assert custom["tool_name"] == "other"
    assert custom["input_bytes"] == len(PRIVATE_COMMAND.encode("utf-8"))
    assert custom["observed_output_bytes"] == len(PRIVATE_OUTPUT.encode("utf-8"))


@pytest.mark.parametrize("scenario", ["incomplete", "malformed", "nonzero_exit", "timeout", "missing_usage"])
def test_failure_receipts_preserve_native_totals_and_do_not_invent_completeness(pipeline, capsys, scenario):
    rows = [_event("task_started"), *_tool_records()]
    if scenario != "missing_usage":
        rows.extend([_checkpoint(FIRST_USAGE), _checkpoint(FINAL_USAGE)])
    if scenario in {"malformed", "nonzero_exit", "missing_usage"}:
        rows.append(_event("task_complete"))
    pipeline.write(rows, malformed=scenario == "malformed")
    if scenario == "nonzero_exit":
        with pytest.raises(RuntimeError, match="observed successful exit"):
            pipeline.dispatch(exit_code=1)
    elif scenario == "timeout":
        with pytest.raises(TimeoutError):
            pipeline.dispatch(timeout=True)
    else:
        output, _ = pipeline.dispatch()
        assert output == pipeline.output
    emitted, _ = _printed_receipt(capsys)
    native = emitted["native_rollout_usage"]
    assert native["usage"] == ({} if scenario == "missing_usage" else FINAL_USAGE)
    assert native["usage_available"] is (scenario != "missing_usage")
    assert native["billing_total_verified"] is False and native["dollar_cost"] is None
    # CLI outcome and native session completeness are separate observations.
    complete = scenario in {"nonzero_exit", "missing_usage"}
    assert native_usage_contract(emitted, native) == (complete, "native_session_cumulative", True)
    if scenario in {"timeout", "nonzero_exit"}:
        assert emitted["status"] == "failed"
        assert emitted["error_type"] == ("TimeoutError" if scenario == "timeout" else "RuntimeError")
    telemetry = native["interaction_telemetry"]
    assert telemetry["task_complete_observed"] is complete
    assert telemetry["native_aggregate_usage_modified"] is False
    if scenario == "malformed":
        assert telemetry["malformed_json_records"] == 1
        assert all(row["adjacent_counter_delta"] is None for row in telemetry["token_observations"])
    if scenario == "missing_usage":
        assert telemetry["token_observations"] == []
        assert native["observed_token_count_records"] == 0


@pytest.mark.parametrize("binding", ["missing", "foreign_workspace", "ambiguous"])
def test_missing_or_unbound_native_rollout_stays_unknown(pipeline, capsys, binding):
    if binding != "missing":
        pipeline.write([_checkpoint(FINAL_USAGE), _event("task_complete")],
                       cwd=pipeline.workspace.parent if binding == "foreign_workspace" else None)
    if binding == "ambiguous":
        pipeline.rollout.with_name("duplicate-" + THREAD + ".jsonl").write_bytes(pipeline.rollout.read_bytes())
    output, _ = pipeline.dispatch()
    emitted, _ = _printed_receipt(capsys)
    assert output == pipeline.output
    assert emitted["native_rollout_usage"] is None
    assert native_usage_contract(emitted, emitted["native_rollout_usage"]) == (None, None, None)
    assert emitted["usage"]["prompt_tokens"] == 15


def test_bounded_telemetry_remains_loggable_and_keeps_final_authoritative_usage(pipeline, capsys):
    rows = [_event("task_started")]
    for ordinal in range(MAX_TOOL_PAIRS + 3):
        call_id = PRIVATE_ID + str(ordinal)
        rows.extend([
            {"type": "response_item", "payload": {"type": "function_call", "name": PRIVATE_TOOL,
                "call_id": call_id, "arguments": PRIVATE_COMMAND}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id,
                "output": PRIVATE_OUTPUT * 128}},
        ])
    for ordinal in range(MAX_TOKEN_OBSERVATIONS + 3):
        rows.append(_checkpoint({"input_tokens": ordinal * 1000, "cached_input_tokens": ordinal * 500,
                                 "output_tokens": ordinal * 10, "total_tokens": ordinal * 1010}))
    rows.append(_event("task_complete"))
    pipeline.write(rows)
    _, returned = pipeline.dispatch()
    emitted, log = _printed_receipt(capsys)
    assert emitted == returned
    # The supervisor admits worker logs up to 8 MB. Keep this projected receipt
    # below 1 MB even after both telemetry populations reach their retention caps.
    assert len(log.encode("utf-8")) < 1_000_000
    native = emitted["native_rollout_usage"]
    final_ordinal = MAX_TOKEN_OBSERVATIONS + 2
    assert native["usage"]["total_tokens"] == final_ordinal * 1010
    assert native_usage_contract(emitted, native) == (True, "native_session_cumulative", True)
    telemetry = native["interaction_telemetry"]
    assert len(telemetry["tool_pairs"]) == MAX_TOOL_PAIRS
    assert len(telemetry["token_observations"]) == MAX_TOKEN_OBSERVATIONS
    assert telemetry["omitted_tool_events"] == 6 and telemetry["omitted_token_observations"] == 3
    assert telemetry["tool_observations_complete"] is False
    assert telemetry["token_observations_complete"] is False
