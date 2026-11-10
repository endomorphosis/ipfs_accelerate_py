"""Authored coverage-schema transport tests; no provider or retained streams."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py.cli_runtime import grok_stream_capture as capture
from test.api.test_grok_stream_capture import close, encoded, end
from test.api.test_grok_tool_event_capture import completed, finish_frames, tool


V1 = "native-grok-tool-events@1"
V2 = "native-grok-tool-events@2"


def parser(**kwargs):
    return capture.GrokStreamParser(tool_event_census=True, tool_event_schema=V2, **kwargs)


@pytest.mark.parametrize("frames", [
    finish_frames(tool(), completed()),
    finish_frames(tool(toolName="run_terminal_command"), completed()),
    finish_frames(tool(toolName="authored-private-unknown", kind="execute"),
        completed(status=None)),
    finish_frames(tool(), completed(rawOutput={"rows": list(range(4100))})),
    finish_frames(tool(), completed(rawOutput=None),
        {"type": "usage", "usage": {"input_tokens": 999999}}),
    encoded(end(), completed()),
    b'{"type":"tool_call","rawInput":NaN}\n' + finish_frames(tool()),
])
def test_coverage_schema_preserves_parent_counters_response_and_final_usage(frames):
    old = capture.GrokStreamParser()
    new = parser()
    for selected in (old, new):
        selected.feed(frames, elapsed_seconds=.5)
    baseline, observed = close(old), close(new)
    assert baseline["schema"] == capture.SCHEMA == "native-grok-stream-progress@3"
    assert observed["schema"] == capture.TOOL_COVERAGE_SCHEMA == "native-grok-stream-progress@5"
    assert observed["tool_event_observations"]["schema"] == V2
    restored = {key: value for key, value in observed.items() if key != "tool_event_observations"}
    restored["schema"] = capture.SCHEMA
    assert restored == baseline
    assert new.final_payload == old.final_payload
    assert new.response_text == old.response_text
    assert capture.grok_stream_progress_summary(observed) == capture.grok_stream_progress_summary(baseline)
    assert capture.validate_grok_stream_progress(observed) == observed
    assert "authored-private" not in json.dumps(observed)


@pytest.mark.parametrize("kwargs,expected,nested", [
    ({}, capture.SCHEMA, None),
    ({"tool_event_schema": V1}, capture.SCHEMA, None),
    ({"tool_event_census": False, "tool_event_schema": V1}, capture.SCHEMA, None),
    ({"tool_event_census": True}, capture.TOOL_EVENTS_SCHEMA, V1),
    ({"tool_event_census": True, "tool_event_schema": V1}, capture.TOOL_EVENTS_SCHEMA, V1),
    ({"tool_event_census": True, "tool_event_schema": V2}, capture.TOOL_COVERAGE_SCHEMA, V2),
])
def test_schema_selection_is_explicit_and_defaults_remain_unchanged(kwargs, expected, nested):
    selected = capture.GrokStreamParser(**kwargs)
    state = selected.snapshot()
    assert state["schema"] == expected
    assert capture.validate_grok_stream_progress(state) == state
    assert state.get("tool_event_observations", {}).get("schema") == nested
    selected.feed(finish_frames(tool(), completed()), elapsed_seconds=.5)
    state = close(selected)
    assert state["schema"] == expected
    assert capture.validate_grok_stream_progress(state) == state
    if nested == V1:
        default = capture.GrokStreamParser(tool_event_census=True)
        default.feed(finish_frames(tool(), completed()), elapsed_seconds=.5)
        assert state == close(default)


@pytest.mark.parametrize("value", [None, 0, 1, True, False, "", "native-grok-tool-events@3",
    "native-grok-stream-progress@5", V2 + " ", {}, [], [V2]])
@pytest.mark.parametrize("census", [False, True])
def test_invalid_schema_rejected_before_any_process_start(monkeypatch, value, census):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("process started"))
    with pytest.raises(ValueError, match="schema"):
        capture.GrokStreamParser(tool_event_census=census, tool_event_schema=value)
    with pytest.raises(ValueError, match="schema"):
        capture.capture_grok_stream(["authored"], env={}, timeout=1,
            tool_event_census=census, tool_event_schema=value)


def test_string_subclasses_do_not_select_a_schema(monkeypatch):
    class Subclass(str):
        pass
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("process started"))
    with pytest.raises(ValueError, match="schema"):
        capture.capture_grok_stream(["authored"], env={}, timeout=1,
            tool_event_census=True, tool_event_schema=Subclass(V2))


@pytest.mark.parametrize("census", [False, None, 0, 1, "true", {}])
def test_coverage_requires_explicit_boolean_census_before_process_start(monkeypatch, census):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("process started"))
    with pytest.raises(ValueError):
        capture.GrokStreamParser(tool_event_census=census, tool_event_schema=V2)
    with pytest.raises(ValueError):
        capture.capture_grok_stream(["authored"], env={}, timeout=1,
            tool_event_census=census, tool_event_schema=V2)


@pytest.mark.parametrize("selected_schema,wrong_parent", [
    (V1, capture.TOOL_COVERAGE_SCHEMA), (V2, capture.TOOL_EVENTS_SCHEMA),
])
def test_nested_schema_must_match_parent_version(selected_schema, wrong_parent):
    selected = capture.GrokStreamParser(tool_event_census=True, tool_event_schema=selected_schema)
    selected.feed(finish_frames(tool(), completed()), elapsed_seconds=.5)
    state = close(selected)
    state["schema"] = wrong_parent
    with pytest.raises(ValueError, match="schemas differ"):
        capture.validate_grok_stream_progress(state)


@pytest.mark.parametrize("historical", [capture.LEGACY_SCHEMA, capture.USAGE_SCHEMA, capture.SCHEMA])
def test_historical_progress_versions_reject_coverage_rows(historical):
    selected = parser()
    selected.feed(finish_frames(tool()), elapsed_seconds=.5)
    state = close(selected)
    nested = state.pop("tool_event_observations")
    state["schema"] = historical
    if historical != capture.SCHEMA:
        state.pop("terminal_rejection")
    if historical == capture.LEGACY_SCHEMA:
        state.pop("usage_observations")
    assert capture.validate_grok_stream_progress(state) == state
    state["tool_event_observations"] = nested
    with pytest.raises(ValueError, match="fields"):
        capture.validate_grok_stream_progress(state)


@pytest.mark.parametrize("change", ["extra", "parent_count", "row_bytes", "after_final", "taint"])
def test_coverage_rows_still_join_parent_counters_and_stream_state(change):
    selected = parser()
    selected.feed(finish_frames(tool(), completed()), elapsed_seconds=.5)
    state = close(selected)
    rows = state["tool_event_observations"]
    if change == "extra":
        rows["authored-private-extra"] = "authored-private-value"
    elif change == "parent_count":
        state["event_counts"]["tool_call"] += 1
    elif change == "row_bytes":
        rows["rows"][0]["frame_bytes"] += 1
    elif change == "after_final":
        rows["rows"][0]["after_final"] = True
    else:
        state.update(stream_tainted=True, stream_complete=False, response_usable=False)
    with pytest.raises(ValueError):
        capture.validate_grok_stream_progress(state)


@pytest.mark.parametrize("tail", [
    encoded({"type": "max_turns_reached"}, end(stopReason="cancelled")),
    encoded({"type": "max_turns_reached"}, end(stopReason="max_turn_requests")),
    encoded({"type": "max_turns_reached"}, end(stopReason="cancelled", extra="authored-private")),
    encoded(end(), {"type": "end"}),
    encoded(end(), end()),
    encoded(end(), completed()),
])
def test_coverage_never_changes_terminal_trailer_or_accounting_policy(tail):
    baseline, observed = capture.GrokStreamParser(), parser()
    frames = encoded(tool(), completed(), {"type": "usage", "usage": {"input_tokens": 999999}}) + tail
    for selected in (baseline, observed):
        selected.feed(frames, elapsed_seconds=.5)
    old, new = close(baseline), close(observed)
    assert observed.final_payload == baseline.final_payload
    assert new["terminal_rejection"] == old["terminal_rejection"]
    assert new["stream_complete"] == old["stream_complete"]
    assert capture.native_grok_stream_observation(observed.final_payload, progress=new, exit_code=0) == (
        capture.native_grok_stream_observation(baseline.final_payload, progress=old, exit_code=0))
    assert "authored-private" not in json.dumps(new)


@pytest.mark.parametrize("bytewise", [False, True])
def test_coverage_preserves_exact_frame_bytes_and_nested_copy_isolation(bytewise):
    line = b"  " + json.dumps(tool(rawInput={"path": "authored-private-世界"}), ensure_ascii=False).encode() + b"\r"
    raw = line + b"\n" + finish_frames(completed())
    selected = parser()
    for chunk in ([bytes([value]) for value in raw] if bytewise else [raw]):
        selected.feed(chunk, elapsed_seconds=.5)
    state = close(selected)
    assert state["stdout_bytes"] == len(raw)
    assert state["tool_event_observations"]["rows"][0]["frame_bytes"] == len(line)
    snapshot = deepcopy(state)
    state["tool_event_observations"]["rows"][0]["input"]["canonical_json_bytes"] = 0
    assert selected.snapshot() == snapshot


def test_documented_alias_and_omitted_update_status_preserve_legacy_parent_buckets():
    update = completed()
    update.pop("status")
    selected = parser()
    selected.feed(finish_frames(tool(toolName="run_terminal_command"), update), elapsed_seconds=.5)
    state = close(selected)
    observed = state["tool_event_observations"]
    call, update_row = observed["rows"]
    assert state["tool_name_counts"]["unknown"] == 1
    assert state["tool_name_counts"]["run_terminal_cmd"] == 0
    assert state["tool_status_counts"]["unknown"] == 1
    assert call["tool"] == "unknown"
    assert call["resolved_tool"] == "run_terminal_cmd"
    assert call["tool_resolution_basis"] == "documented_alias"
    assert update_row["status_present"] is False and update_row["status_shape"] == "missing"
    assert observed["observation_complete"] is True
    assert state["response_usable"] is True


@pytest.mark.parametrize("status", [None, [], "authored-private-status"])
def test_explicit_invalid_update_status_stays_distinct_from_omission(status):
    selected = parser()
    selected.feed(finish_frames(tool(), completed(status=status)), elapsed_seconds=.5)
    state = close(selected)
    observed = state["tool_event_observations"]
    assert observed["rows"][1]["status_present"] is True
    assert observed["observation_complete"] is False
    assert state["response_usable"] is True
    assert "authored-private" not in json.dumps(state)


def test_dense_payload_measures_under_v2_without_reclassifying_v1():
    update = completed(rawOutput={"rows": list(range(4100))})
    assert len(encoded(update)) < capture.MAX_FRAME_BYTES
    old, new = capture.GrokStreamParser(tool_event_census=True), parser()
    for selected in (old, new):
        selected.feed(finish_frames(tool(), update), elapsed_seconds=.5)
    baseline, state = close(old), close(new)
    old_payload = baseline["tool_event_observations"]["rows"][1]["output"]
    observed = state["tool_event_observations"]
    payload = observed["rows"][1]["output"]
    assert baseline["oversized_frames"] == state["oversized_frames"] == 0
    assert old_payload["state"] == "oversized" and old_payload["canonical_json_bytes"] is None
    assert baseline["tool_event_observations"]["observation_complete"] is False
    assert payload["state"] == "valid" and payload["bound_reason"] is None
    canonical = json.dumps(update["rawOutput"], sort_keys=True, separators=(",", ":")).encode()
    assert payload["canonical_json_bytes"] == len(canonical)
    assert observed["observation_complete"] is True
    assert baseline["response_usable"] is state["response_usable"] is True
    assert new.final_payload == old.final_payload == end()


@pytest.mark.parametrize("reason", ["depth", "integer_bits"])
def test_other_payload_bounds_remain_explicit_and_keep_final_response(reason):
    payload = 1 << 4096
    if reason == "depth":
        payload = "authored-private-output"
        for _ in range(34):
            payload = [payload]
    update = completed(rawOutput=payload)
    assert len(encoded(update)) < capture.MAX_FRAME_BYTES
    selected = parser()
    selected.feed(finish_frames(tool(), update), elapsed_seconds=.5)
    state = close(selected)
    observed = state["tool_event_observations"]
    assert state["oversized_frames"] == 0
    assert observed["rows"][1]["output"]["state"] == "oversized"
    assert observed["rows"][1]["output"]["bound_reason"] == reason
    assert observed["rows"][1]["output"]["canonical_json_bytes"] is None
    assert observed["observation_complete"] is False
    assert state["response_usable"] is True
    assert selected.final_payload["usage"] == {"input_tokens": 7}


def test_frame_byte_cap_still_precedes_payload_observation():
    update = completed(rawOutput="x" * capture.MAX_FRAME_BYTES)
    assert len(encoded(update)) > capture.MAX_FRAME_BYTES
    selected = parser()
    selected.feed(finish_frames(tool(), update), elapsed_seconds=.5)
    state = close(selected)
    observed = state["tool_event_observations"]
    assert state["oversized_frames"] == 1
    assert state["event_counts"]["tool_call_update"] == 0
    assert len(observed["rows"]) == 1
    assert observed["rows"][0]["output"]["state"] == "missing"
    assert observed["observation_complete"] is False
    assert state["response_usable"] is False


def authored(tmp_path, source, *, timeout=3, callback=None, schema=V2):
    path = tmp_path / "authored-coverage-process.py"
    path.write_text(source)
    return capture.capture_grok_stream([sys.executable, str(path)], env=os.environ.copy(),
        timeout=timeout, cwd=tmp_path, progress_callback=callback,
        tool_event_census=True, tool_event_schema=schema)


@pytest.mark.parametrize("schema,progress_schema", [(V1, capture.TOOL_EVENTS_SCHEMA),
    (V2, capture.TOOL_COVERAGE_SCHEMA)])
def test_process_callbacks_stay_count_only_and_payloads_stay_private(tmp_path, monkeypatch,
        schema, progress_schema):
    monkeypatch.setattr(capture, "CALLBACK_INTERVAL_SECONDS", .01)
    secret = "authored-private-output-世界" * 1800
    events = [tool(), completed(rawOutput={"content": secret}), completed(rawOutput={"content": secret})]
    source = "import os,time\n" + "\n".join(
        f"os.write(1, {encoded(event)!r})\ntime.sleep(.04)" for event in events)
    source += f"\nos.write(1, {finish_frames()!r})\n"
    callbacks = []
    result = authored(tmp_path, source, callback=callbacks.append, schema=schema)
    assert result.returncode == 0 and result.response_text == "done" and not result.timed_out
    assert result.progress["schema"] == progress_schema
    assert result.progress["tool_event_observations"]["schema"] == schema
    assert result.progress["tool_event_observations"]["payloads"]["output"]["repeated_events"] == 1
    assert result.final_payload["usage"] == {"input_tokens": 7}
    assert len(callbacks) >= 3
    assert all(snapshot["schema"] == capture.LEGACY_SCHEMA
        and "tool_event_observations" not in snapshot and "usage_observations" not in snapshot
        and "terminal_rejection" not in snapshot and len(json.dumps(snapshot)) < 4096 for snapshot in callbacks)
    assert "authored-private" not in json.dumps(result.progress)
    assert "authored-private" not in json.dumps(callbacks)
    assert "authored-private" not in repr(result)
    assert result.stderr == ""


def test_coverage_timeout_keeps_final_usage_and_reaps_owned_process(tmp_path):
    result = authored(tmp_path, "import os,time\nfrom pathlib import Path\n"
        "Path('authored-pid').write_text(str(os.getpid()))\n"
        f"os.write(1, {finish_frames(tool(), completed())!r})\n"
        "time.sleep(30)\n", timeout=.3)
    assert result.timed_out is True and result.progress["process_exit_observed"] is True
    assert result.progress["timeout_phase"] == "process_wait"
    assert result.progress["schema"] == capture.TOOL_COVERAGE_SCHEMA
    assert result.progress["tool_event_observations"]["observation_complete"] is False
    assert result.progress["response_usable"] is False
    assert result.stdout == result.stderr == result.response_text == ""
    assert result.final_payload["usage"] == {"input_tokens": 7}
    assert "authored-private" not in json.dumps(result.progress)
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "authored-pid").read_text()), 0)
