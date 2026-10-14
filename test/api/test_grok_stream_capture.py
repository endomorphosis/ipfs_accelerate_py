"""Authored protocol/process tests; never invokes Grok or a model provider."""
import json
import os
import signal
import subprocess
import sys

import pytest

from ipfs_accelerate_py.cli_runtime.grok_stream_capture import (
    GrokStreamParser, capture_grok_stream, validate_grok_stream_progress,
)


def encoded(*events):
    return b"".join(json.dumps(event, ensure_ascii=True).encode() + b"\n" for event in events)


def end(**kwargs):
    return {"type": "end", "stopReason": "end_turn", "usage": {"input_tokens": 7}, **kwargs}


def close(parser, code=0):
    parser.eof("stdout", elapsed_seconds=1)
    parser.eof("stderr", elapsed_seconds=1)
    parser.process_exit(code, elapsed_seconds=1)
    return parser.snapshot(elapsed_seconds=1)


def test_initial_factory_is_closed_valid_and_copies_nested_fields():
    parser = GrokStreamParser()
    snapshot = parser.snapshot()
    assert validate_grok_stream_progress(snapshot) == snapshot
    snapshot["event_counts"]["text"] = 19
    assert parser.snapshot()["event_counts"]["text"] == 0
    assert len(json.dumps(snapshot)) < 4096


def test_documented_frames_bytewise_and_text_usage_boundaries():
    parser = GrokStreamParser()
    data = encoded(
        {"type": "thought", "data": "private thought"},
        {"type": "text", "data": "private intermediate"},
        {"type": "usage", "stopReason": "tool_use", "usage": {"input_tokens": 999999}},
        {"type": "tool_call", "toolName": "read_file", "toolCallId": "private id", "status": "in_progress", "rawInput": {"path": "private path"}},
        {"type": "tool_call_update", "status": "completed", "rawOutput": "private output"},
        {"type": "text", "data": "old segment"}, {"type": "usage", "stopReason": "pause_turn"},
        {"type": "text", "data": "final π"}, {"type": "usage", "stopReason": "end_turn"}, end())
    for byte in data:
        parser.feed(bytes([byte]), elapsed_seconds=0.5)
    snapshot = close(parser)
    assert parser.response_text == "final π"
    assert parser.final_payload["usage"]["input_tokens"] == 7
    assert snapshot["response_usable"] is True
    assert snapshot["tool_name_counts"]["read_file"] == 1
    assert snapshot["tool_status_counts"]["completed"] == 1
    assert snapshot["discarded_text_bytes"] == len(b"private intermediateold segment")
    assert snapshot["frames"] == 10
    for secret in ("private", "final", "rawInput", "toolCallId"):
        # 'final' is a key component, never the model's text.
        if secret != "final":
            assert secret not in json.dumps(snapshot)


@pytest.mark.parametrize("line", [b'not-json\n', b'[]\n', b'null\n', b'123\n',
    b'{"type":"text","type":"end"}\n', b'{"type":"text","data":NaN}\n',
    b'{"type":"text","data":Infinity}\n', b'{"type":"end","usage":{"x":1e999}}\n', b'{"type":"text","data":"\xff"}\n',
    b'{"type":"text","data":3}\n', b'{"type":"text","data":"\\ud800"}\n',
    b'{"type":"unknown-secret-event"}\n', b'{"type":[]}\n', b'{}\n'])
def test_malformed_unknown_or_non_object_frames_cannot_return_success(line):
    parser = GrokStreamParser()
    parser.feed(line + encoded({"type": "text", "data": "partial"}, end()))
    assert close(parser)["response_usable"] is False
    assert parser.final_payload is None
    assert "secret" not in json.dumps(parser.snapshot())


@pytest.mark.parametrize("tail", [encoded(end()), encoded({"type": "error", "message": "secret"}),
    encoded({"type": "text", "data": "secret"}), b'bad\n', b'[]\n'])
def test_any_event_after_terminal_invalidates_terminal_payload(tail):
    parser = GrokStreamParser()
    parser.feed(encoded(end()) + tail)
    assert close(parser)["stream_tainted"] is True
    assert parser.final_payload is None
    assert parser.snapshot()["events_after_final"] == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_max_turns_marker_then_matching_end_retains_final_usage_not_event_sum(chunked):
    # Authored from the pinned CLI's documented protocol; the live failure
    # retained event counts, not the raw closing envelope or its token totals.
    from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation, grok_usage_receipt
    parser = GrokStreamParser()
    terminal = end(stopReason="max_turn_requests", sessionId="private-session", num_turns=8,
                   usage={"input_tokens": 13, "output_tokens": 5}, usage_is_incomplete=True)
    frames = encoded(*({"type": "usage", "stopReason": "tool_use", "usage": {"input_tokens": 9000}}
                       for _ in range(8)), {"type": "max_turns_reached"}, terminal)
    for chunk in ([bytes([byte]) for byte in frames] if chunked else [frames]):
        parser.feed(chunk, elapsed_seconds=0.5)
    state = close(parser, code=1)
    assert parser.final_payload == terminal
    assert state["stream_tainted"] is False
    assert state["stream_complete"] is True
    assert state["response_usable"] is False
    assert state["final_reason_code"] == "max_turns"
    assert state["event_counts"]["max_turns_reached"] == state["event_counts"]["end"] == 1
    assert state["events_after_final"] == 1
    observed = native_grok_observation(parser.final_payload, exit_code=1)
    assert observed["native_grok_reason_code"] == "max_turns"
    assert grok_usage_receipt(observed)["usage"] == {"input_tokens": 13, "output_tokens": 5}
    assert grok_usage_receipt(observed)["usage_complete_observed"] is False
    assert state["usage_observations"]["retained_events"] == 8
    assert state["usage_observations"]["counter_semantics"] == "unverified"
    assert state["usage_observations"]["summation_permitted"] is False
    assert "private-session" not in json.dumps(state)


@pytest.mark.parametrize("terminal,code,usable,reason", [
    (end(), 0, True, "end_turn"),
    ({"type": "max_turns_reached"}, 1, False, "max_turns"),
    ({"type": "max_turns_reached", "usage": {"input_tokens": 7}}, 1, False, "max_turns"),
    ({"type": "error", "message": "private failure"}, 1, False, "execution_error"),
])
def test_one_bare_end_marker_preserves_terminal_outcome_and_usage(terminal, code, usable, reason):
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "text", "data": "answer"}, terminal, {"type": "end"}))
    state = close(parser, code)
    assert parser.final_payload == terminal
    assert state["stream_tainted"] is False
    assert state["stream_complete"] is True
    assert state["response_usable"] is usable
    assert state["final_reason_code"] == reason


@pytest.mark.parametrize("tail", [
    [end()], [end(stopReason="cancelled")],
    [end(stopReason="max_turn_requests", text="private response")],
    [end(stopReason="max_turn_requests", stop_reason="end_turn")],
    [end(stopReason="max_turn_requests", usage="invalid")],
    [end(stopReason="max_turn_requests", usage_is_incomplete="false")],
    [{"type": "text", "data": "private response"}],
    [{"type": "usage", "usage": {"input_tokens": 19}}],
    [{"type": "error", "message": "private failure"}],
    [{"type": "end"}, {"type": "end"}],
    [end(stopReason="max_turn_requests"), {"type": "end"}],
    [{"type": "end"}, {"type": "tool_call", "toolName": "read_file"}],
])
def test_contradictory_or_substantive_records_after_failure_still_taint_stream(tail):
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "max_turns_reached"}, *tail))
    state = close(parser, code=1)
    assert state["stream_tainted"] is True
    assert state["response_usable"] is False
    assert parser.final_payload is None


@pytest.mark.parametrize("field,previous,current", [
    ("usage", {"input_tokens": 9}, {"input_tokens": 7}),
    ("usage", {"input_tokens": True}, {"input_tokens": 1}),
    ("usage_is_incomplete", True, False),
    ("sessionId", "private-one", "private-two"),
    ("requestId", "private-one", "private-two"),
    ("num_turns", 8, 9),
    ("num_turns", True, 1),
    ("modelUsage", {"model": {"inputTokens": 7}}, {"model": {"inputTokens": 8}}),
    ("modelUsage", {"model": {"inputTokens": False}}, {"model": {"inputTokens": 0}}),
    ("cost_is_partial", False, True),
    ("total_cost_usd", 0, 0.0),
    ("total_cost_usd_ticks", 1, 2),
])
def test_conflicting_terminal_identity_or_spend_cannot_be_replaced_by_end(field, previous, current):
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "max_turns_reached", field: previous},
                        end(stopReason="max_turn_requests", **{field: current})))
    assert close(parser, code=1)["stream_tainted"] is True
    assert parser.final_payload is None


@pytest.mark.parametrize("field,value", [
    ("usage", {"input_tokens": 9}), ("usage_is_incomplete", True),
    ("sessionId", "private-session"), ("requestId", "private-request"),
    ("num_turns", 8), ("modelUsage", {"model": {"inputTokens": 7}}),
    ("cost_is_partial", False), ("total_cost_usd", 0), ("total_cost_usd_ticks", 1),
])
def test_end_cannot_erase_existing_terminal_identity_or_spend(field, value):
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "max_turns_reached", field: value},
                        {"type": "end", "stopReason": "max_turn_requests"}))
    assert close(parser, code=1)["stream_tainted"] is True
    assert parser.final_payload is None


def test_matching_frozen_metadata_is_retained_without_merging_counters():
    parser = GrokStreamParser()
    fields = dict(sessionId="private-session", requestId="private-request", num_turns=8,
                  usage={"output_tokens": 5, "input_tokens": 13}, usage_is_incomplete=True,
                  modelUsage={"model": {"inputTokens": 13, "outputTokens": 5}}, cost_is_partial=True)
    terminal = {"type": "max_turns_reached", **fields}
    closing = {"type": "end", "stopReason": "max_turn_requests", **fields,
               "usage": {"input_tokens": 13, "output_tokens": 5}}
    parser.feed(encoded(terminal, closing))
    assert close(parser, code=1)["stream_tainted"] is False
    assert parser.final_payload == closing


def test_last_line_without_newline_and_blank_lines():
    parser = GrokStreamParser()
    parser.feed(b"\n \r\n" + encoded({"type": "text", "data": "done"}, end()).rstrip(b"\n"))
    assert close(parser)["response_usable"] is True
    assert parser.snapshot()["frames"] == 2


def test_missing_terminal_is_incomplete_even_after_usage_end_turn_and_exit():
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "text", "data": "partial"}, {"type": "usage", "stopReason": "end_turn"}))
    assert close(parser)["response_usable"] is False
    assert parser.final_payload is None


def test_frame_overflow_discards_until_newline_and_keeps_memory_bounded():
    parser = GrokStreamParser(max_frame_bytes=80)
    for _ in range(50):
        parser.feed(b"x" * 50)
        assert len(parser._pending) <= 80
    parser.feed(b"\n" + encoded(end()))
    result = close(parser)
    assert result["oversized_frames"] == 1
    assert result["stream_tainted"] is True
    assert parser.final_payload is None


def test_text_overflow_is_not_a_successful_partial_or_later_recovered_response():
    parser = GrokStreamParser(max_text_bytes=4)
    parser.feed(encoded({"type": "text", "data": "123"}, {"type": "text", "data": "45"},
        {"type": "usage", "stopReason": "tool_use"}, {"type": "text", "data": "ok"}, end()))
    result = close(parser)
    assert result["text_truncated"] is True
    assert result["response_usable"] is False
    assert result["text_bytes"] == result["discarded_text_bytes"] == 7
    assert parser.response_text == ""


def test_closed_tool_buckets_do_not_retain_arbitrary_tool_names():
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "tool_call", "toolName": "private-command", "status": "private-status"},
        {"type": "auto_compact_start", "data": "private"}, {"type": "plan", "data": "private"}, end()))
    state = close(parser)
    assert state["tool_name_counts"]["unknown"] == 1
    assert state["tool_status_counts"]["unknown"] == 1
    assert state["event_counts"]["auto_compact"] == 1
    assert "private" not in json.dumps(state)


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(raw_text="private"), lambda s: s.update(completion_authority=True),
    lambda s: s.update(stdout_bytes=True), lambda s: s.update(stderr_bytes=-1),
    lambda s: s.update(duration_seconds=float("nan")), lambda s: s.update(duration_seconds=float("inf")),
    lambda s: s.update(duration_seconds=True), lambda s: s.update(response_usable=True),
    lambda s: s.update(stream_complete=True), lambda s: s.update(returncode=False),
    lambda s: s.update(timeout_phase="private"), lambda s: s.update(final_event_type="private"),
    lambda s: s["tool_name_counts"].update(private=1), lambda s: s["event_counts"].update(text=True),
    lambda s: s.update(timed_out=True), lambda s: s.update(stdout_eof=1),
])
def test_validator_rejects_open_or_inconsistent_snapshot(mutation):
    snapshot = GrokStreamParser().snapshot()
    mutation(snapshot)
    with pytest.raises(ValueError):
        validate_grok_stream_progress(snapshot)


def authored(tmp_path, source, *, timeout=3, callback=None):
    script = tmp_path / "authored.py"
    script.write_text(source)
    return capture_grok_stream([sys.executable, str(script)], env=os.environ.copy(),
        timeout=timeout, cwd=tmp_path, progress_callback=callback)


def test_real_process_concurrently_drains_stderr_and_stdout_without_raw_export(tmp_path):
    source = "import os\n" + "\n".join([
        "os.write(2, b'private-stderr' * 100000)",
        f"os.write(1, {encoded({'type': 'text', 'data': 'done'}, end())!r})",
    ])
    snapshots = []
    result = authored(tmp_path, source, callback=snapshots.append)
    assert result.response_text == "done"
    assert json.loads(result.stdout)["text"] == "done"
    assert result.stderr == ""
    assert result.progress["stderr_bytes"] == len(b"private-stderr") * 100000
    assert result.progress["stdout_eof"] is result.progress["stderr_eof"] is True
    assert result.progress["process_exit_observed"] is True
    assert result.progress["response_usable"] is True
    assert len(snapshots) == 2
    assert "private" not in json.dumps(result.progress)
    assert "done" not in repr(result)


def test_callbacks_keep_count_only_summary_and_final_result_retains_usage_history(tmp_path):
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import LEGACY_SCHEMA, SCHEMA
    from ipfs_accelerate_py.cli_runtime.grok_stream_usage import MAX_COUNT, MAX_USAGE_EVENTS, USAGE_FIELDS
    frame = encoded({"type": "usage", "stopReason": "tool_use", "messageId": "private-id",
                     "usage": dict.fromkeys(USAGE_FIELDS, MAX_COUNT)})
    events = frame * (MAX_USAGE_EVENTS + 1) + encoded({"type": "text", "data": "done"}, end())
    snapshots = []
    result = authored(tmp_path, f"import os\nos.write(1, {events!r})\n", callback=snapshots.append)
    assert result.progress["schema"] == SCHEMA
    assert result.progress["usage_observations"]["retained_events"] == MAX_USAGE_EVENTS
    assert result.progress["usage_observations"]["omitted_events"] == 1
    assert snapshots[-1]["event_counts"]["usage"] == MAX_USAGE_EVENTS + 1
    assert all(snapshot["schema"] == LEGACY_SCHEMA and "usage_observations" not in snapshot
               for snapshot in snapshots)
    assert all(len(json.dumps(snapshot)) < 4096 for snapshot in snapshots)
    assert result.response_text == "done"


def test_end_event_then_running_process_times_out_and_direct_child_is_reaped(tmp_path):
    result = authored(tmp_path, "import os,time\nfrom pathlib import Path\n"
        "Path('pid').write_text(str(os.getpid()))\n"
        f"os.write(1, {encoded({'type': 'text', 'data': 'private'}, end())!r})\n"
        "time.sleep(30)\n", timeout=0.3)
    assert result.timed_out is True
    assert result.progress["timeout_phase"] == "process_wait"
    assert result.progress["final_envelope_observed"] is True
    assert result.progress["process_exit_observed"] is True
    assert result.progress["response_usable"] is False
    assert result.final_payload["usage"]["input_tokens"] == 7
    assert result.stdout == result.stderr == result.response_text == ""
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="authored inherited pipe requires POSIX fork")
def test_child_retaining_pipe_after_parent_exit_is_bounded_pipe_drain_timeout(tmp_path):
    pidfile = tmp_path / "owned-descendant.pid"
    try:
        result = authored(tmp_path, "import os,time\nfrom pathlib import Path\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            "    time.sleep(30)\n"
            "    os._exit(0)\n"
            f"Path({str(pidfile)!r}).write_text(str(pid))\n"
            f"os.write(1, {encoded({'type': 'text', 'data': 'done'}, end())!r})\n"
            "os._exit(0)\n", timeout=0.3)
        assert result.timed_out is True
        assert result.returncode == 0
        assert result.progress["timeout_phase"] == "pipe_drain"
        assert result.progress["process_exit_observed"] is True
        assert not result.progress["stdout_eof"]
        assert not result.progress["stderr_eof"]
        assert result.progress["response_usable"] is False
        assert result.progress["duration_seconds"] < 2
        # The capture owns the direct parent only. This test owns and cleans its child.
        os.kill(int(pidfile.read_text()), 0)
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_closed_pipes_with_live_process_still_time_out_waiting_for_exit(tmp_path):
    result = authored(tmp_path, f"import os,time\nos.write(1, {encoded(end())!r})\n"
        "os.close(1)\nos.close(2)\ntime.sleep(30)\n", timeout=0.3)
    assert result.progress["stdout_eof"] is result.progress["stderr_eof"] is True
    assert result.progress["timeout_phase"] == "process_wait"
    assert result.progress["response_usable"] is False


@pytest.mark.parametrize("failure", [RuntimeError])
def test_callback_failure_is_diagnostic_and_does_not_mask_result(tmp_path, failure):
    def callback(_snapshot):
        raise failure("private error")
    result = authored(tmp_path, f"import os\nos.write(1, {encoded({'type': 'text', 'data': 'ok'}, end())!r})\n", callback=callback)
    assert result.response_text == "ok"
    assert result.progress["callback_errors"] == 2
    assert result.returncode == 0


def test_nonzero_process_exit_retains_clean_terminal_for_existing_accounting(tmp_path):
    result = authored(tmp_path, f"import os\nos.write(1, {encoded({'type': 'error', 'message': 'private failure'})!r})\nos._exit(9)\n")
    assert result.returncode == 9
    assert result.progress["stream_complete"] is True
    assert result.progress["response_usable"] is False
    assert json.loads(result.stdout)["type"] == "error"
    assert result.response_text == ""
    assert "private" not in repr(result)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "1"])
def test_invalid_timeout_rejected_before_process(monkeypatch, timeout):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("process started"))
    with pytest.raises(ValueError):
        capture_grok_stream(["authored"], env={}, timeout=timeout)


@pytest.mark.parametrize("changes", [{"stopReason": "max_tokens"}, {"stopReason": "cancelled"},
    {"stopReason": "unknown"}, {"stop_reason": "max_tokens"}, {"is_error": True},
    {"subtype": "error_during_execution"}])
def test_terminal_end_requires_native_success_reason(changes):
    parser = GrokStreamParser()
    parser.feed(encoded({"type": "text", "data": "partial"}, end(**changes)))
    result = close(parser)
    assert result["stream_complete"] is True
    assert result["response_usable"] is False
    assert result["final_reason_code"] != "end_turn"


def test_terminal_end_without_response_text_is_not_usable():
    parser = GrokStreamParser()
    parser.feed(encoded(end()))
    assert close(parser)["response_usable"] is False


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
def test_callback_cancellation_propagates_after_owned_process_cleanup(tmp_path, monkeypatch, failure):
    actual_popen = subprocess.Popen
    processes = []
    def tracked(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", tracked)
    def callback(_snapshot):
        raise failure("authored cancellation")
    with pytest.raises(failure):
        authored(tmp_path, "import time\ntime.sleep(30)\n", callback=callback)
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed is processes[0].stderr.closed is True
