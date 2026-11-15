"""Authored tool observations distinguish event repetition from request history."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from ipfs_accelerate_py.cli_runtime import grok_tool_events as census


def _call(identity="PRIVATE-CALL-A", **fields):
    return {"type": "tool_call", "toolCallId": identity, "toolName": "read_file", "status": "completed", **fields}


def _update(identity="PRIVATE-CALL-A", **fields):
    return {"type": "tool_call_update", "toolCallId": identity, "status": "completed", **fields}


def _state(events, frames):
    event_counts = dict.fromkeys(("tool_call", "tool_call_update"), 0)
    names = dict.fromkeys(census.TOOLS, 0); statuses = dict.fromkeys(census.STATUSES, 0)
    for event in events:
        event_counts[event["type"]] += 1
        name, status = event.get("toolName"), event.get("status")
        if event["type"] == "tool_call": names[name if type(name) is str and name in names else "unknown"] += 1
        statuses[status if type(status) is str and status in statuses else "unknown"] += 1
    return dict(frames=len(events)+1, stdout_bytes=sum(len(frame)+1 for frame in frames)+32,
        malformed_frames=0, oversized_frames=0, non_object_frames=0, unknown_events=0, events_after_final=0,
        stream_complete=True, stream_tainted=False, text_truncated=False, timed_out=False,
        duration_seconds=len(events)+2.0, final_event_seconds=len(events)+1.0,
        event_counts=event_counts, tool_name_counts=names, tool_status_counts=statuses)


def _observe(events, *, overrides=None, frame_bytes=None):
    observer = census.GrokToolEventObserver()
    # Exact authored frames, including surrounding whitespace/CR, excluding LF.
    frames = [b"  "+json.dumps(event, ensure_ascii=True).encode()+b"\r" for event in events] if frame_bytes is None else [b"x"*frame_bytes for _ in events]
    for ordinal, (event, frame) in enumerate(zip(events, frames), 1):
        observer.observe(event, frame_bytes=len(frame), frame_ordinal=ordinal,
            elapsed_seconds=float(ordinal), after_final=False)
    state = _state(events, frames)
    if overrides: state.update(overrides)
    return observer, state, frames


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_exact_frames_and_separate_canonical_payload_lanes_do_not_claim_requests():
    body = {"PRIVATE-PATH": 'café 世界 \\ "\n'}
    events = [_call(rawInput=body, title="PRIVATE-TITLE", kind="read", locations=[{"path": "PRIVATE-PATH"}]),
        _update(rawOutput=body, content=body)]
    observer, state, frames = _observe(events)
    result = observer.snapshot(stream_state=state)
    assert result["retained_frame_bytes"] == sum(map(len, frames))
    assert result["rows"][0]["frame_bytes"] == len(frames[0])
    for lane in census.PAYLOAD_LANES:
        assert result["payloads"][lane]["canonical_json_bytes"] == len(_canonical(body))
        assert result["payloads"][lane]["repeated_events"] == 0
    assert result["rows"][0]["input"]["sha256"] == hashlib.sha256(_canonical(body)).hexdigest()
    assert result["rows"][1]["output"]["sha256"] == result["rows"][1]["content"]["sha256"]
    assert result["payload_lanes_may_overlap"] is True and result["unique_output_bytes_measured"] is False
    assert result["observation_complete"] is True
    assert result["call_identities"] == 1
    assert result["rows"][1]["call_relation"] == "lifecycle_update"
    assert result["provider_request_bytes"] is result["history_resend_bytes"] is result["provider_tokens"] is None
    for key in ("request_history_complete", "tool_result_coverage_complete", "event_origin_authenticated",
        "proof_authority", "execution_authority", "completion_authority", "omission_authority", "token_savings_claimed"):
        assert result[key] is False
    serialized = json.dumps(result)
    assert "PRIVATE" not in serialized and "café" not in serialized
    assert census.validate_grok_tool_events(result, stream_state=state) == result


def test_repetition_is_separated_by_call_identity_with_explicit_overlap():
    body = {"text": "PRIVATE repeated output 世界"}
    events = [_call("A", rawOutput=body), _update("A", rawOutput=body),
        _call("B", rawOutput=body), _update("A", rawOutput=body)]
    observer, state, _ = _observe(events); result = observer.snapshot(stream_state=state)
    assert [row["output"]["repetition"] for row in result["rows"]] == [
        "first_observed", "same_call", "cross_call", "same_and_cross_call"]
    stats = result["payloads"]["output"]; size = len(_canonical(body))
    assert stats["repeated_events"] == 3 and stats["repeated_bytes"] == 3*size
    assert stats["same_call_repeated_events"] == stats["cross_call_repeated_events"] == 2
    assert stats["same_call_repeated_bytes"] == stats["cross_call_repeated_bytes"] == 2*size
    assert result["repetition_categories_overlap"] is True
    assert result["call_relation_counts"]["new_call"] == 2
    assert result["call_relation_counts"]["lifecycle_update"] == 2


def test_duplicate_call_event_or_orphan_update_is_not_a_new_call():
    observer, state, _ = _observe([_call("A"), _call("A"), _update("B"), _update("B")])
    result = observer.snapshot(stream_state=state)
    assert [row["call_relation"] for row in result["rows"]] == [
        "new_call", "repeated_call", "orphan_update", "lifecycle_update"]
    assert result["call_identities"] == 2


@pytest.mark.parametrize("payload,status,shape", [(None, "null", "null"), ("", "empty", "string"),
    ({}, "empty", "object"), ([], "empty", "array"), (False, "valid", "boolean"),
    (0, "valid", "integer"), (0.0, "valid", "number"), ("\ud800", "invalid", "invalid")])
def test_payload_null_empty_valid_and_invalid_are_distinct(payload, status, shape):
    observer, state, _ = _observe([_call(rawOutput=payload)])
    result = observer.snapshot(stream_state=state); row = result["rows"][0]
    assert row["output"]["state"] == status and row["output"]["shape"] == shape
    assert row["input"]["state"] == row["content"]["state"] == "missing"
    if status == "invalid":
        assert row["output"]["canonical_json_bytes"] is row["output"]["sha256"] is None
        assert result["incompleteness"]["invalid_payload"] is True
    else:
        assert row["output"]["canonical_json_bytes"] == len(_canonical(payload))
        assert result["observation_complete"] is True


@pytest.mark.parametrize("identity,status", [(None, "null"), ("", "empty"), (False, "invalid"),
    ("\ud800", "invalid"), ("é"*1024, "oversized")])
def test_invalid_missing_or_bounded_identity_never_fabricates_a_call(identity, status):
    events = [_call(identity, rawOutput="same"), _update(identity, rawOutput="same")]
    observer, state, _ = _observe(events); result = observer.snapshot(stream_state=state)
    assert all(row["call_id_status"] == status and row["call_id_sha256"] is None for row in result["rows"])
    assert result["rows"][1]["output"]["repetition"] == "unattributed"
    assert result["payloads"]["output"]["same_call_repeated_events"] == 0
    assert result["payloads"]["output"]["cross_call_repeated_events"] == 0
    assert result["call_identities"] == 0 and result["observation_complete"] is False


def test_missing_identity_unknown_tool_status_and_fields_are_closed():
    event = _call(toolName={"PRIVATE-NAME": True}, status="PRIVATE-STATUS", **{"PRIVATE-FIELD": "PRIVATE-CONTENT"})
    del event["toolCallId"]
    observer, state, _ = _observe([event]); result = observer.snapshot(stream_state=state)
    row = result["rows"][0]
    assert row["tool"] == row["status"] == "unknown"
    assert row["call_id_status"] == "missing" and row["unknown_field_count"] == 1
    assert result["incompleteness"]["unknown_event_shape"] is True
    assert result["incompleteness"]["unattributed_call"] is True
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("shape", ["unknown", "null", "list", "known", "omitted"])
def test_explicit_unknown_update_tool_name_differs_from_an_omitted_optional_name(shape):
    event = _update()
    if shape != "omitted":
        event["toolName"] = {"unknown": "PRIVATE-TOOL", "null": None, "list": [], "known": "read_file"}[shape]
    observer, state, _ = _observe([event])
    result = observer.snapshot(stream_state=state)
    assert result["rows"][0]["tool_name_present"] is (shape != "omitted")
    assert result["incompleteness"]["unknown_event_shape"] is (shape in {"unknown", "null", "list"})
    assert result["observation_complete"] is (shape in {"known", "omitted"})
    assert "PRIVATE" not in json.dumps(result)
    wrong = deepcopy(result); wrong["rows"][0]["tool_name_present"] = 1
    with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)


@pytest.mark.parametrize("bound", ["bytes", "nodes", "depth", "integer"])
def test_payload_bounds_mark_observed_coverage_without_numeric_overflow(bound):
    payload = "é"*census.MAX_PAYLOAD_BYTES
    if bound == "nodes": payload = [None]*census.MAX_PAYLOAD_NODES
    elif bound == "depth":
        payload = 1
        for _ in range(census.MAX_PAYLOAD_DEPTH+1): payload = [payload]
    elif bound == "integer": payload = 1 << 100000
    # Direct malformed/beyond-bound observer inputs, not a claim that the
    # upstream frame parser admits a frame exceeding its own byte limit.
    observer, state, _ = _observe([_call(rawOutput=payload)], frame_bytes=200)
    result = observer.snapshot(stream_state=state)
    assert result["rows"][0]["output"]["state"] == "oversized"
    assert result["incompleteness"]["payload_bound_exceeded"] is True
    assert result["payloads"]["output"]["canonical_json_bytes"] == 0
    assert result["observation_complete"] is False


@pytest.mark.parametrize("payload", [float("inf"), float("nan"), {"x": {1, 2}}])
def test_direct_invalid_payload_shapes_are_closed_and_do_not_interrupt_observation(payload):
    observer, state, _ = _observe([_call(rawOutput=payload)], frame_bytes=200)
    result = observer.snapshot(stream_state=state)
    assert result["rows"][0]["output"]["state"] == "invalid"
    assert result["incompleteness"]["invalid_payload"] is True


def test_first_64_rows_bound_identity_memory_and_observed_only_aggregates():
    events = [_call(str(n), rawOutput="repeated") for n in range(census.MAX_TOOL_EVENTS+3)]
    observer, state, frames = _observe(events); result = observer.snapshot(stream_state=state)
    assert result["observed_events"] == 67 and result["retained_events"] == 64 and result["omitted_events"] == 3
    assert len(observer.rows) == result["call_identities"] == 64
    assert result["retained_frame_bytes"] == sum(map(len, frames[:64]))
    assert result["payloads"]["output"]["cross_call_repeated_events"] == 63
    assert result["incompleteness"]["row_limit_exceeded"] is True and result["observation_complete"] is False


def test_saturated_parent_counters_remain_bounded_and_explicitly_incomplete():
    events = [_call(str(n)) for n in range(64)]
    observer, state, _ = _observe(events)
    observer.observed_events = census.MAX_COUNT-1
    observer.observe(_call(), frame_bytes=100, frame_ordinal=census.MAX_COUNT, elapsed_seconds=65.0, after_final=False)
    state.update(frames=census.MAX_COUNT, stdout_bytes=census.MAX_COUNT, stream_complete=False)
    state["event_counts"]["tool_call"] = census.MAX_COUNT
    state["tool_name_counts"]["read_file"] = census.MAX_COUNT
    state["tool_status_counts"]["completed"] = census.MAX_COUNT
    result = observer.snapshot(stream_state=state)
    assert result["observed_events"] == census.MAX_COUNT
    assert result["omitted_events"] == census.MAX_COUNT-64
    assert result["incompleteness"]["counter_overflow"] is True


@pytest.mark.parametrize("flag", ["malformed_frames", "oversized_frames", "non_object_frames", "unknown_events",
    "stream_tainted", "text_truncated", "timed_out"])
def test_parser_incompleteness_is_joined_without_error_callbacks(flag):
    boolean = flag in {"stream_tainted", "text_truncated", "timed_out"}
    observer, state, _ = _observe([_call()], overrides={flag: True if boolean else 1, "stream_complete": False})
    result = observer.snapshot(stream_state=state)
    assert result["incompleteness"][flag] is True and result["observation_complete"] is False
    wrong = deepcopy(result); wrong["incompleteness"][flag] = False
    with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)


def test_post_final_tool_rows_have_joined_time_and_frame_boundary():
    events = [_call("A"), _update("A")]
    observer, state, _ = _observe(events)
    observer.rows[1].update(frame_ordinal=3, elapsed_seconds=3.0, after_final=True)
    state.update(final_event_seconds=2.0, events_after_final=1, stream_complete=False, stream_tainted=True)
    result = observer.snapshot(stream_state=state)
    assert result["incompleteness"]["post_final_events"] is True
    for change in ({"frame_ordinal": 2}, {"elapsed_seconds": 1.0}, {"after_final": False}):
        wrong = deepcopy(result); wrong["rows"][1].update(change)
        with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)


def test_closed_validator_recomputes_repetition_counts_types_and_parent_joins():
    observer, state, _ = _observe([_call("A", rawOutput="same"), _update("A", rawOutput="same")])
    result = observer.snapshot(stream_state=state)
    def bad_parent():
        changed = deepcopy(state); changed["tool_status_counts"]["completed"] += 1
        return changed
    with pytest.raises(ValueError): census.validate_grok_tool_events(result, stream_state=bad_parent())
    def payload_bytes(value): value["rows"][0]["output"].update(canonical_json_bytes=True)
    def relation(value): value["rows"][1].update(call_relation="new_call")
    def repetition(value): value["rows"][1]["output"].update(repetition="cross_call")
    def aggregate(value): value["payloads"]["output"]["repeated_bytes"] += 1
    def extra(value): value["rows"][0]["output"]["PRIVATE"] = "SECRET"
    def large_time(value): value["rows"][0]["elapsed_seconds"] = 10**1000
    for mutate in (payload_bytes, relation, repetition, aggregate, extra, large_time):
        wrong = deepcopy(result); mutate(wrong)
        with pytest.raises(ValueError) as error: census.validate_grok_tool_events(wrong, stream_state=state)
        assert "PRIVATE" not in str(error.value) and "SECRET" not in str(error.value)
    for key, value in [("request_history_complete", True), ("proof_authority", True),
        ("retained_events", True), ("provider_request_bytes", 0), ("unknown", "PRIVATE")]:
        with pytest.raises(ValueError): census.validate_grok_tool_events({**result, key: value}, stream_state=state)


def test_empty_snapshot_and_returned_nested_rows_are_independent():
    observer, state, _ = _observe([])
    result = observer.snapshot(stream_state=state)
    assert result["rows"] == [] and result["retained_frame_bytes"] == 0
    observer, state, _ = _observe([_call(rawOutput={"value": "PRIVATE"})])
    result = observer.snapshot(stream_state=state)
    result["rows"][0]["output"]["sha256"] = "0"*64
    result["payloads"]["output"]["canonical_json_bytes"] = 0
    assert observer.snapshot(stream_state=state) != result


def test_observer_snapshot_validation_perform_no_io(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("tool census attempted IO")
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    observer, state, _ = _observe([_call(rawInput={"path": "PRIVATE"})])
    result = observer.snapshot(stream_state=state)
    assert census.validate_grok_tool_events(result, stream_state=state) == result
