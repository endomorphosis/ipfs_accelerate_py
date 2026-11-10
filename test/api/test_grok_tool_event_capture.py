"""Authored stream/process integration; no provider or retained transcript reads."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py.cli_runtime import grok_stream_capture as capture
from test.api.test_grok_stream_capture import close, encoded, end


def tool(**changes):
    return {"type": "tool_call", "toolCallId": "authored-private-call", "toolName": "read_file",
        "status": "in_progress", "rawInput": {"path": "authored-private-path"}, **changes}


def completed(**changes):
    return {"type": "tool_call_update", "toolCallId": "authored-private-call", "status": "completed",
        "rawOutput": {"content": "authored-private-output"}, **changes}


def finish_frames(*events):
    return encoded(*events, {"type": "text", "data": "done"}, end())


@pytest.mark.parametrize("frames", [
    finish_frames(tool(), completed()),
    finish_frames(tool(toolCallId=None, toolName="authored-private-unknown")),
    finish_frames(tool(), completed(rawOutput=None), {"type": "usage", "usage": {"input_tokens": 17}}),
    encoded(end(), completed()),
    b'{"type":"tool_call","rawInput":NaN}\n' + finish_frames(tool()),
])
def test_explicit_tool_census_preserves_legacy_stream_and_accounting(frames):
    old = capture.GrokStreamParser()
    new = capture.GrokStreamParser(tool_event_census=True)
    for parser in (old, new):
        parser.feed(frames, elapsed_seconds=.5)
    baseline, observed = close(old), close(new)
    assert baseline["schema"] == capture.SCHEMA == "native-grok-stream-progress@3"
    assert "tool_event_observations" not in baseline
    assert observed["schema"] == capture.TOOL_EVENTS_SCHEMA == "native-grok-stream-progress@4"
    assert "tool_event_observations" in observed
    restored = {key: value for key, value in observed.items() if key != "tool_event_observations"}
    restored["schema"] = capture.SCHEMA
    assert restored == baseline
    assert new.response_text == old.response_text
    assert new.final_payload == old.final_payload
    assert capture.grok_stream_progress_summary(observed) == capture.grok_stream_progress_summary(baseline)
    assert "authored-private" not in json.dumps(observed)


def test_prior_schemas_remain_exact_and_reject_new_nested_fields():
    parser = capture.GrokStreamParser()
    parser.feed(finish_frames(tool(), completed()), elapsed_seconds=.5)
    state = close(parser)
    for schema in (capture.SCHEMA, capture.USAGE_SCHEMA, capture.LEGACY_SCHEMA):
        historical = deepcopy(state)
        historical["schema"] = schema
        if schema != capture.SCHEMA:
            historical.pop("terminal_rejection")
        if schema == capture.LEGACY_SCHEMA:
            historical.pop("usage_observations")
        assert capture.validate_grok_stream_progress(historical) == historical
        historical["tool_event_observations"] = {}
        with pytest.raises(ValueError):
            capture.validate_grok_stream_progress(historical)


@pytest.mark.parametrize("bytewise", [False, True])
def test_tool_frames_count_exact_utf8_lines_excluding_lf_and_keep_payloads_canonical(bytewise):
    first = tool(rawInput={"path": "authored-private-café/世界"})
    second = completed(rawOutput={"text": "authored-private-世界", "line": "a\nb"})
    first_line = b"  " + json.dumps(first, ensure_ascii=False, indent=None).encode() + b"\r"
    second_line = json.dumps(second, ensure_ascii=False, separators=(",", ":")).encode()
    parser = capture.GrokStreamParser(tool_event_census=True)
    raw = first_line + b"\n" + second_line + b"\n" + finish_frames()
    for chunk in ([bytes([byte]) for byte in raw] if bytewise else [raw]):
        parser.feed(chunk, elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    assert [row["frame_bytes"] for row in observed["rows"]] == [len(first_line), len(second_line)]
    assert observed["retained_frame_bytes"] == len(first_line) + len(second_line)
    for row, field, value in ((observed["rows"][0], "input", first["rawInput"]),
            (observed["rows"][1], "output", second["rawOutput"])):
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        assert row[field]["canonical_json_bytes"] == len(canonical)
    assert [row["frame_ordinal"] for row in observed["rows"]] == [1, 2]
    assert state["stdout_bytes"] == len(raw)
    assert observed["observation_complete"] is True
    observed["rows"][0]["input"]["canonical_json_bytes"] = 0
    assert parser.snapshot()["tool_event_observations"]["rows"][0]["input"]["canonical_json_bytes"] > 0


def test_unterminated_final_tool_line_is_observed_without_inventing_stream_completion():
    parser = capture.GrokStreamParser(tool_event_census=True)
    line = encoded(completed()).removesuffix(b"\n")
    parser.feed(line, elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    assert len(observed["rows"]) == 1
    assert observed["rows"][0]["frame_bytes"] == len(line)
    assert observed["observation_complete"] is False
    assert state["final_envelope_observed"] is False and state["response_usable"] is False


@pytest.mark.parametrize("case,value,expected", [
    ("missing", None, "missing"), ("present", None, "null"),
    ("present", "", "empty"), ("present", {}, "empty"), ("present", [], "empty"),
    ("present", {"content": "authored-private-output"}, "valid"),
    ("present", "\ud800", "invalid"),
])
def test_absent_empty_null_and_invalid_payloads_are_distinct(case, value, expected):
    event = completed(rawOutput=value)
    if case == "missing":
        event.pop("rawOutput")
    parser = capture.GrokStreamParser(tool_event_census=True)
    parser.feed(finish_frames(event), elapsed_seconds=.5)
    observed = close(parser)["tool_event_observations"]
    assert observed["rows"][0]["output"]["state"] == expected
    assert "authored-private" not in json.dumps(observed)
    if expected == "invalid":
        assert observed["observation_complete"] is False


@pytest.mark.parametrize("line,options", [
    (b'not-json\n', {}), (b'[]\n', {}),
    (b'{"type":"authored-private-event"}\n', {}),
    (b'x' * 300 + b'\n', {"max_frame_bytes": 256}),
    (encoded({"type": "text", "data": "long text"}), {"max_text_bytes": 4}),
    (encoded(tool(toolCallId=None)), {}),
    (encoded(tool(toolName="authored-private-tool")), {}),
    (encoded(tool(authored_private_field="secret")), {}),
])
def test_partial_or_unknown_observations_cannot_claim_complete_coverage(line, options):
    parser = capture.GrokStreamParser(tool_event_census=True, **options)
    parser.feed(line + finish_frames(), elapsed_seconds=.5)
    observed = close(parser)["tool_event_observations"]
    assert observed["observation_complete"] is False
    assert any(observed["incompleteness"].values())
    assert "authored-private" not in json.dumps(observed)


def test_content_and_raw_output_are_separate_overlapping_payload_lanes():
    content = [{"type": "text", "text": "authored-private-overlapping-result"}]
    output = {"text": "authored-private-overlapping-result"}
    parser = capture.GrokStreamParser(tool_event_census=True)
    parser.feed(finish_frames(tool(), completed(content=content, rawOutput=output, locations=[])),
        elapsed_seconds=.5)
    observed = close(parser)["tool_event_observations"]
    row = observed["rows"][1]
    for name, value in (("content", content), ("output", output)):
        count = len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        assert row[name]["state"] == "valid" and row[name]["canonical_json_bytes"] == count
        assert observed["payloads"][name]["canonical_json_bytes"] == count
    assert observed["observation_complete"] is True
    assert observed["tool_result_coverage_complete"] is False
    assert observed["provider_request_bytes"] is observed["history_resend_bytes"] is None
    assert observed["request_history_complete"] is False
    assert "authored-private" not in json.dumps(observed)


@pytest.mark.parametrize("case,value,expected", [
    ("missing", None, "missing"), ("present", None, "null"), ("present", "", "empty"),
    ("present", 1, "invalid"), ("present", [], "invalid"), ("present", "x" * 1025, "oversized"),
])
def test_missing_or_invalid_call_identity_never_invents_attribution(case, value, expected):
    event = tool(toolCallId=value)
    if case == "missing":
        event.pop("toolCallId")
    parser = capture.GrokStreamParser(tool_event_census=True)
    parser.feed(finish_frames(event), elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    assert observed["rows"][0]["call_id_status"] == expected
    assert observed["rows"][0]["call_id_sha256"] is None
    assert observed["rows"][0]["call_relation"] == "unattributed"
    assert observed["observation_complete"] is False
    assert state["response_usable"] is True


def test_payload_structure_bound_keeps_valid_stream_response_and_marks_partial_coverage():
    from ipfs_accelerate_py.cli_runtime.grok_tool_events import MAX_PAYLOAD_DEPTH
    nested = "authored-private-deep-payload"
    for _ in range(MAX_PAYLOAD_DEPTH + 2):
        nested = [nested]
    parser = capture.GrokStreamParser(tool_event_census=True)
    parser.feed(finish_frames(tool(rawInput=nested)), elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    assert observed["rows"][0]["input"]["state"] == "oversized"
    assert observed["rows"][0]["input"]["canonical_json_bytes"] is None
    assert observed["observation_complete"] is False
    assert state["stream_tainted"] is False and state["response_usable"] is True


def test_post_terminal_tools_are_measured_and_remain_incomplete():
    parser = capture.GrokStreamParser(tool_event_census=True)
    parser.feed(encoded(end()), elapsed_seconds=.25)
    parser.feed(encoded(tool(), completed()), elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    assert [row["after_final"] for row in observed["rows"]] == [True, True]
    assert state["events_after_final"] == 2 and state["stream_tainted"] is True
    assert observed["observation_complete"] is False
    assert parser.final_payload is None


def test_tool_row_and_identity_overflow_is_explicit_and_does_not_change_response():
    from ipfs_accelerate_py.cli_runtime.grok_tool_events import MAX_TOOL_EVENTS
    parser = capture.GrokStreamParser(tool_event_census=True)
    count = MAX_TOOL_EVENTS + 3
    parser.feed(finish_frames(*(tool(toolCallId=f"authored-private-{index}") for index in range(count))),
        elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    assert observed["observed_events"] == count
    assert observed["retained_events"] == len(observed["rows"]) == MAX_TOOL_EVENTS
    assert observed["omitted_events"] == 3 and observed["observation_complete"] is False
    assert state["response_usable"] is True and parser.response_text == "done"
    assert "authored-private" not in json.dumps(observed)


@pytest.mark.parametrize("change", ["extra", "parent_count", "row_bytes", "after_final", "taint"])
def test_tool_observations_must_join_parent_counters_times_and_coverage(change):
    parser = capture.GrokStreamParser(tool_event_census=True)
    parser.feed(finish_frames(tool(), completed()), elapsed_seconds=.5)
    state = close(parser)
    observed = state["tool_event_observations"]
    if change == "extra":
        observed["private"] = "authored-private-output"
    elif change == "parent_count":
        state["event_counts"]["tool_call"] += 1
    elif change == "row_bytes":
        observed["rows"][0]["frame_bytes"] += 1
    elif change == "after_final":
        observed["rows"][0]["after_final"] = True
    else:
        state.update(stream_tainted=True, stream_complete=False, response_usable=False)
    with pytest.raises(ValueError):
        capture.validate_grok_stream_progress(state)


@pytest.mark.parametrize("value", [None, 0, 1, "true", {}])
def test_tool_census_requires_explicit_boolean_before_starting_process(monkeypatch, value):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("process started"))
    with pytest.raises(ValueError):
        capture.GrokStreamParser(tool_event_census=value)
    with pytest.raises(ValueError):
        capture.capture_grok_stream(["authored"], env={}, timeout=1, tool_event_census=value)


def authored(tmp_path, source, *, timeout=3, callback=None):
    path = tmp_path / "authored-tool-process.py"
    path.write_text(source)
    return capture.capture_grok_stream([sys.executable, str(path)], env=os.environ.copy(),
        timeout=timeout, cwd=tmp_path, progress_callback=callback, tool_event_census=True)


def test_real_process_repeated_large_tool_outputs_keep_periodic_callbacks_count_only(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "CALLBACK_INTERVAL_SECONDS", .01)
    secret = "authored-private-output-世界" * 1800
    updates = [tool(), completed(rawOutput={"content": secret}), completed(rawOutput={"content": secret}),
        tool(toolCallId="authored-private-second"),
        completed(toolCallId="authored-private-second", rawOutput={"content": secret})]
    source = "import os,time\n" + "\n".join(
        f"os.write(1, {encoded(event)!r})\ntime.sleep(.04)" for event in updates)
    source += f"\nos.write(1, {finish_frames()!r})\n"
    callbacks = []
    result = authored(tmp_path, source, callback=callbacks.append)
    assert result.returncode == 0 and result.response_text == "done" and not result.timed_out
    assert result.progress["schema"] == capture.TOOL_EVENTS_SCHEMA
    assert result.progress["event_counts"]["tool_call"] == 2
    assert result.progress["event_counts"]["tool_call_update"] == 3
    observed = result.progress["tool_event_observations"]
    assert observed["payloads"]["output"]["repeated_events"] == 2
    assert observed["payloads"]["output"]["same_call_repeated_events"] == 1
    assert observed["payloads"]["output"]["cross_call_repeated_events"] == 1
    assert result.progress["response_usable"] is True
    assert result.final_payload["usage"] == {"input_tokens": 7}
    assert len(callbacks) >= 3
    assert all(snapshot["schema"] == capture.LEGACY_SCHEMA and "tool_event_observations" not in snapshot
        and "usage_observations" not in snapshot and "terminal_rejection" not in snapshot
        and len(json.dumps(snapshot)) < 4096 for snapshot in callbacks)
    assert "authored-private" not in json.dumps(result.progress)
    assert "authored-private" not in json.dumps(callbacks)
    assert "authored-private" not in repr(result)
    assert result.stderr == ""


def test_tool_observation_survives_timeout_without_changing_owned_child_cleanup(tmp_path):
    result = authored(tmp_path, "import os,time\nfrom pathlib import Path\n"
        "Path('authored-pid').write_text(str(os.getpid()))\n"
        f"os.write(1, {finish_frames(tool(), completed())!r})\n"
        "time.sleep(30)\n", timeout=.3)
    assert result.timed_out is True
    assert result.progress["timeout_phase"] == "process_wait"
    assert result.progress["process_exit_observed"] is True
    assert result.progress["response_usable"] is False
    assert result.progress["schema"] == capture.TOOL_EVENTS_SCHEMA
    assert result.progress["event_counts"]["tool_call_update"] == 1
    assert result.progress["tool_event_observations"]["observation_complete"] is False
    assert result.stdout == result.stderr == result.response_text == ""
    assert result.final_payload["usage"] == {"input_tokens": 7}
    assert "authored-private" not in json.dumps(result.progress)
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "authored-pid").read_text()), 0)
