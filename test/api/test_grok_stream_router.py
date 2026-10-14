"""Real authored CLI processes through the shared router; no model calls."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    set_last_cli_observation("grok_cli", {})
    return llm_router._get_grok_cli_provider()


def command(tmp_path, frames, *, suffix=""):
    script = tmp_path / "authored_cli.py"
    script.write_text("import json, sys, time\nfrom pathlib import Path\n"
        "assert sys.argv[sys.argv.index('--output-format')+1] == 'streaming-json'\n"
        "assert Path(sys.argv[sys.argv.index('--prompt-file')+1]).read_text() == 'private prompt'\n"
        f"Path({str(tmp_path / 'prompt_path')!r}).write_text(sys.argv[sys.argv.index('--prompt-file')+1])\n"
        + "\n".join(f"print({json.dumps(frame)!r}, flush=True)" for frame in frames) + "\n" + suffix)
    return [sys.executable, str(script)]


def end(**fields):
    return {"type": "end", "stopReason": "end_turn", "num_turns": 2,
            "usage": {"input_tokens": 12, "output_tokens": 4}, **fields}


def test_final_response_segment_and_only_terminal_usage_reach_router(provider, tmp_path):
    frames = [{"type": "text", "data": "private interim thought"},
        {"type": "usage", "stopReason": "tool_use", "usage": {"input_tokens": 10000}},
        {"type": "tool_call", "toolName": "read_file", "status": "in_progress",
         "toolCallId": "private identifier", "args": {"path": "private path"}},
        {"type": "text", "data": "Final "}, {"type": "text", "data": "answer"},
        {"type": "usage", "stopReason": "end_turn"}, end()]
    snapshots = []
    argv = command(tmp_path, frames)
    assert provider.generate("private prompt", timeout=3, grok_cli_cmd=argv,
        grok_stream_progress=True, grok_progress_callback=snapshots.append) == "Final answer"
    observation = get_last_cli_observation("grok_cli")
    assert observation["native_grok_input_tokens"] == 12
    assert observation["native_grok_reason_code"] == "end_turn"
    progress = observation["native_grok_progress"]
    assert progress["response_usable"] is True
    assert progress["process_exit_observed"] is True
    assert progress["stdout_eof"] is progress["stderr_eof"] is True
    usage_events = progress["usage_observations"]
    assert usage_events["observed_events"] == 2
    assert usage_events["rows"][0]["usage"]["input_tokens"] == 10000
    assert usage_events["rows"][1]["usage"]["input_tokens"] is None
    assert usage_events["summation_permitted"] is False
    assert snapshots
    encoded = json.dumps([progress, *snapshots])
    for secret in ("private", "Final answer", "path", "args", "toolCallId"):
        assert secret not in encoded
    assert not Path((tmp_path / "prompt_path").read_text()).exists()


def test_end_event_then_sleep_stays_timeout_with_observation_and_prompt_cleanup(provider, tmp_path):
    argv = command(tmp_path, [{"type": "text", "data": "private response"}, end()], suffix="time.sleep(30)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        provider.generate("private prompt", timeout=0.3, grok_cli_cmd=argv, grok_stream_progress=True)
    observation = get_last_cli_observation("grok_cli")
    assert observation["native_grok_reason_code"] == "timeout"
    assert observation["native_grok_task_complete"] is False
    assert observation["native_grok_input_tokens"] == 12
    progress = observation["native_grok_progress"]
    assert progress["timed_out"] is True
    assert progress["response_usable"] is False
    assert not Path((tmp_path / "prompt_path").read_text()).exists()


@pytest.mark.parametrize("closing", [{"type": "end"}, end(stopReason="max_turn_requests")])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_max_turns_then_end_retains_failure_and_only_explicit_final_usage(provider, tmp_path, closing, exit_code):
    from ipfs_accelerate_py.cli_runtime.grok_native_usage import grok_usage_receipt, grok_outcome_receipt
    frames = [*({"type": "usage", "stopReason": "tool_use", "usage": {"input_tokens": 9000}}
                for _ in range(8)), {"type": "max_turns_reached"}, closing]
    argv = command(tmp_path, frames, suffix=f"sys.exit({exit_code})\n")
    with pytest.raises(llm_router.LLMRouterError):
        provider.generate("private prompt", timeout=3, grok_cli_cmd=argv, grok_stream_progress=True)
    observation = get_last_cli_observation("grok_cli")
    outcome = grok_outcome_receipt(observation)
    assert outcome["reason_code"] == "max_turns"
    assert outcome["envelope_observed"] is True
    assert observation["native_grok_task_complete"] is False
    usage = grok_usage_receipt(observation)
    if "usage" in closing:
        assert usage["usage"] == {"input_tokens": 12, "output_tokens": 4}
        assert usage["usage_complete_observed"] is None
        assert usage["billing_total_verified"] is False
    else:
        assert usage is None
    progress = observation["native_grok_progress"]
    assert progress["stream_tainted"] is False
    assert progress["stream_complete"] is True
    assert progress["response_usable"] is False
    assert progress["events_after_final"] == 1
    assert progress["usage_observations"]["retained_events"] == 8
    assert progress["usage_observations"]["summation_permitted"] is False
    assert not Path((tmp_path / "prompt_path").read_text()).exists()


def test_error_with_end_marker_preserves_failure_without_native_usage(provider, tmp_path):
    argv = command(tmp_path, [{"type": "error", "message": "private failure"}, {"type": "end"}],
                   suffix="sys.exit(1)\n")
    with pytest.raises(llm_router.LLMRouterError):
        provider.generate("private prompt", timeout=3, grok_cli_cmd=argv, grok_stream_progress=True)
    observation = get_last_cli_observation("grok_cli")
    assert observation["native_grok_reason_code"] == "execution_error"
    assert "native_grok_usage_observed" not in observation
    assert observation["native_grok_progress"]["stream_tainted"] is False
    assert observation["native_grok_progress"]["response_usable"] is False


def test_rejected_end_shape_survives_router_metadata_without_raw_values(provider, tmp_path):
    argv = command(tmp_path, [{"type": "max_turns_reached"},
        end(stopReason="max_turn_requests", sessionId=None, requestId="PRIVATE_IDENTIFIER")],
        suffix="sys.exit(1)\n")
    with pytest.raises(llm_router.LLMRouterError):
        provider.generate("private prompt", timeout=3, grok_cli_cmd=argv, grok_stream_progress=True)
    observation = get_last_cli_observation("grok_cli")
    progress = observation["native_grok_progress"]
    assert progress["schema"] == "native-grok-stream-progress@3"
    assert progress["response_usable"] is False
    rejection = progress["terminal_rejection"]
    assert rejection["reason"] == "invalid_end_field_shape"
    assert rejection["rejected_fields"] == ["sessionId"]
    assert rejection["field_shapes"]["sessionId"] == "null"
    assert rejection["field_shapes"]["requestId"] == "string"
    assert "PRIVATE_IDENTIFIER" not in json.dumps(progress)
    assert "native_grok_usage_observed" not in observation


@pytest.mark.parametrize("frames", [
    [{"type": "text", "data": "partial only"}],
    [{"type": "text", "data": "answer"}, end(), end()],
    [{"type": "text", "data": "answer"}, end(), {"type": "text", "data": "after end"}],
    [{"type": "text", "data": "answer"}, {"type": "usage", "stopReason": "end_turn"}],
])
def test_zero_exit_cannot_return_partial_or_ambiguous_response(provider, tmp_path, frames):
    with pytest.raises(llm_router.LLMRouterError, match="incomplete or invalid"):
        provider.generate("private prompt", timeout=3, grok_cli_cmd=command(tmp_path, frames),
            grok_stream_progress=True)
    assert get_last_cli_observation("grok_cli")["native_grok_progress"]["response_usable"] is False


def test_callback_error_does_not_trigger_reinvocation_or_change_result(provider, tmp_path):
    def broken(_snapshot):
        raise RuntimeError("private observer failure")
    assert provider.generate("private prompt", timeout=3,
        grok_cli_cmd=command(tmp_path, [{"type": "text", "data": "done"}, end()]),
        grok_stream_progress=True, grok_progress_callback=broken) == "done"
    assert get_last_cli_observation("grok_cli")["native_grok_progress"]["callback_errors"] > 0


@pytest.mark.parametrize("options", [
    {"grok_stream_progress": 1},
    {"grok_progress_callback": lambda _: None},
    {"grok_stream_progress": True, "grok_progress_callback": "not callable"},
    {"grok_stream_progress": True, "response_format": {"type": "json_object"}},
    {"grok_stream_progress": True, "grok_cli_cmd": ["grok", "--output-format=json"]},
    {"grok_stream_progress": True, "grok_cli_cmd": ["grok", "--output-format", "streaming-json", "--output-format", "streaming-json"]},
    {"grok_stream_progress": True, "grok_cli_cmd": ["grok", "--"]},
    {"grok_stream_progress": True, "grok_cli_cmd": "custom-wrapper --prompt {prompt}"},
])
def test_invalid_stream_selection_fails_before_any_process(provider, monkeypatch, options):
    monkeypatch.setattr(llm_router.subprocess, "Popen", lambda *a, **k: pytest.fail("process started"))
    with pytest.raises(ValueError):
        provider.generate("private prompt", **options)


def test_legacy_json_transport_unchanged(provider, monkeypatch):
    def execute(argv, **kwargs):
        assert argv[argv.index("--output-format")+1] == "json"
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0, stdout=json.dumps({"text": "legacy"}), stderr="")
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    assert provider.generate("private prompt") == "legacy"
    assert "native_grok_progress" not in get_last_cli_observation("grok_cli")


def test_unvalidated_nested_progress_is_not_retained():
    set_last_cli_observation("grok_cli", {"native_grok_progress": {"text": "private payload"}})
    assert "native_grok_progress" not in get_last_cli_observation("grok_cli")


def test_partial_usage_stream_survives_timeout_without_inventing_final_accounting(provider, tmp_path):
    frames = [{"type": "usage", "messageId": "private id", "signature": "private signature",
               "stopReason": "tool_use", "usage": {"input_tokens": 12, "cache_read_input_tokens": 4000}}]
    argv = command(tmp_path, frames, suffix="time.sleep(30)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        provider.generate("private prompt", timeout=0.3, grok_cli_cmd=argv, grok_stream_progress=True)
    observation = get_last_cli_observation("grok_cli")
    assert "native_grok_input_tokens" not in observation
    assert "native_grok_usage_observed" not in observation
    progress = observation["native_grok_progress"]
    assert progress["timed_out"] is True
    assert progress["response_usable"] is False
    assert progress["usage_observations"]["rows"][0]["usage"]["input_tokens"] == 12
    assert "private" not in json.dumps(progress)
