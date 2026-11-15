"""Versioned tool coverage improves diagnosis without rewriting legacy events."""
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_grok_tool_events import _call, _update, _state
from ipfs_accelerate_py.cli_runtime import grok_tool_events as census


def _observe(events, *, schema=census.COVERAGE_SCHEMA, direct=False, trailer=False):
    observer = census.GrokToolEventObserver() if schema is None else census.GrokToolEventObserver(schema=schema)
    frames = [b"x"*200 for _ in events] if direct else [json.dumps(event, ensure_ascii=True).encode() for event in events]
    for ordinal, (event, frame) in enumerate(zip(events, frames), 1):
        assert len(frame) <= census.MAX_FRAME_BYTES
        observer.observe(event, frame_bytes=len(frame), frame_ordinal=ordinal,
            elapsed_seconds=float(ordinal), after_final=False)
    state = _state(events, frames)
    if trailer:
        # An authored parent state after its accepted terminal trailer. The
        # parser integration suite verifies actual trailer admission separately.
        state.update(frames=state["frames"]+1, stdout_bytes=state["stdout_bytes"]+32, events_after_final=1)
    return observer, state, observer.snapshot(stream_state=state)


def test_default_and_explicit_v1_keep_original_closed_projection():
    events = [_call(rawInput={"path": "PRIVATE"}), _update(rawOutput={"result": 1}, content=[])]
    _, state, implicit = _observe(events, schema=None)
    _, _, explicit = _observe(events, schema=census.SCHEMA)
    assert implicit == explicit and set(implicit) == census._FIELDS
    assert explicit["payload_node_limit"] == 4096
    assert "classification_counts" not in explicit
    assert all(set(row) == census._ROW_FIELDS for row in explicit["rows"])
    assert all(set(row[lane]) == census._PAYLOAD_FIELDS for row in explicit["rows"] for lane in census.PAYLOAD_LANES)
    assert census.validate_grok_tool_events(explicit, stream_state=state) == explicit
    _, _, modern = _observe(events)
    stripped = {key: modern[key] for key in census._FIELDS}
    stripped.update(schema=census.SCHEMA, payload_node_limit=4096)
    stripped["rows"] = [{key: value for key, value in row.items() if key in census._ROW_FIELDS} for row in modern["rows"]]
    for row in stripped["rows"]:
        for lane in census.PAYLOAD_LANES:
            row[lane] = {key: value for key, value in row[lane].items() if key in census._PAYLOAD_FIELDS}
    assert stripped == explicit


@pytest.mark.parametrize("schema", [None, "", "native-grok-tool-events@3", True, {}, []])
def test_unsupported_constructor_schema_is_rejected(schema):
    with pytest.raises(ValueError, match="unsupported Grok tool census schema"):
        census.GrokToolEventObserver(schema=schema)


@pytest.mark.parametrize("name,resolved,basis", [
    ("read_file", "read_file", "exact"), ("run_terminal_cmd", "run_terminal_cmd", "exact"),
    ("run_terminal_command", "run_terminal_cmd", "documented_alias"),
    ("RunTerminalCommand", "unknown", "unknown"), ("terminal", "unknown", "unknown"),
    ("PRIVATE-TOOL", "unknown", "unknown")])
def test_only_pinned_alias_resolves_without_changing_legacy_tool_buckets(name, resolved, basis):
    events = [_call(toolName=name)]
    _, state, value = _observe(events)
    row = value["rows"][0]
    assert row["resolved_tool"] == resolved and row["tool_resolution_basis"] == basis
    assert row["tool_name_shape"] == "string" and row["tool_name_status"] == "valid"
    assert row["tool_name_sha256"] == hashlib.sha256(name.encode()).hexdigest()
    assert row["tool"] == (resolved if basis == "exact" else "unknown")
    assert state["tool_name_counts"][row["tool"]] == 1
    assert value["classification_counts"]["resolved_tools"][resolved] == 1
    assert value["classification_counts"]["resolution_bases"][basis] == 1
    assert value["observation_complete"] is (resolved != "unknown")
    assert "PRIVATE" not in json.dumps(value)
    _, _, old = _observe(events, schema=census.SCHEMA)
    assert old["rows"][0]["tool"] == row["tool"]
    if basis == "documented_alias": assert old["observation_complete"] is False


@pytest.mark.parametrize("name,shape,status", [(None, "null", "null"), ("", "string", "empty"),
    (True, "boolean", "invalid"), ([], "array", "invalid"), ({}, "object", "invalid"),
    ("\ud800", "string", "invalid"), ("é"*1024, "string", "oversized")])
def test_noncanonical_tool_names_export_only_shape_status_and_bounded_digest(name, shape, status):
    _, _, value = _observe([_update(toolName=name)])
    row = value["rows"][0]
    assert row["tool_name_present"] is True
    assert (row["tool_name_shape"], row["tool_name_status"], row["tool_name_sha256"]) == (shape, status, None)
    assert row["resolved_tool"] == row["tool_resolution_basis"] == "unknown"
    assert value["classification_counts"]["tool_name_shapes"][shape] == 1
    assert value["classification_counts"]["tool_name_statuses"][status] == 1
    assert value["observation_complete"] is False


@pytest.mark.parametrize("status,shape", [(None, "null"), ([], "array"), ({}, "object"),
    (False, "boolean"), (12, "integer"), ("PRIVATE-STATUS", "string")])
def test_present_unknown_update_status_stays_incomplete(status, shape):
    _, _, value = _observe([_update(status=status)])
    row = value["rows"][0]
    assert row["status_present"] is True and row["status_shape"] == shape and row["status"] == "unknown"
    assert value["incompleteness"]["unknown_event_shape"] is True
    assert value["classification_counts"]["status_presence"] == dict(present=1, missing=0)
    assert "PRIVATE" not in json.dumps(value)


def test_omitted_update_status_and_name_are_distinct_from_missing_initial_call_fields():
    event = _update(rawOutput={"result": 1}); del event["status"]
    _, state, value = _observe([_call(toolName="run_terminal_command"), event])
    row = value["rows"][1]
    assert row["status_present"] is row["tool_name_present"] is False
    assert row["status_shape"] == row["tool_name_shape"] == row["tool_name_status"] == "missing"
    assert row["tool_name_sha256"] is None
    assert row["status"] == row["tool"] == row["resolved_tool"] == "unknown"
    assert value["observation_complete"] is True
    assert state["tool_status_counts"]["unknown"] == 1
    assert value["classification_counts"]["status_presence"] == dict(present=1, missing=1)
    _, _, old = _observe([event], schema=census.SCHEMA)
    assert old["incompleteness"]["unknown_event_shape"] is True
    initial = _call(); del initial["status"]
    _, _, missing_call = _observe([initial])
    assert missing_call["incompleteness"]["unknown_event_shape"] is True


def test_byte_coupled_v2_node_limit_measures_dense_real_json_without_rewriting_v1():
    payload = [0]*5000
    event = _call(rawOutput=payload)
    assert len(json.dumps(event).encode()) < census.MAX_FRAME_BYTES
    _, _, old = _observe([event], schema=census.SCHEMA)
    _, _, current = _observe([event])
    assert old["rows"][0]["output"]["state"] == "oversized"
    assert "bound_reason" not in old["rows"][0]["output"]
    assert old["payload_node_limit"] == 4096
    assert current["payload_node_limit"] == current["payload_byte_limit"]//2 == 131072
    measured = current["rows"][0]["output"]
    serialized = json.dumps(payload, separators=(",", ":")).encode()
    assert measured["state"] == "valid" and measured["bound_reason"] is None
    assert measured["canonical_json_bytes"] == len(serialized)
    assert measured["sha256"] == hashlib.sha256(serialized).hexdigest()
    assert current["observation_complete"] is True


@pytest.mark.parametrize("reason", ["bytes", "nodes", "depth", "integer_bits"])
def test_direct_beyond_bound_values_keep_precise_closed_reasons(reason):
    payload = "é"*census.MAX_PAYLOAD_BYTES
    if reason == "nodes": payload = [0]*census.COVERAGE_MAX_PAYLOAD_NODES
    elif reason == "integer_bits": payload = 1 << census.MAX_INTEGER_BITS
    elif reason == "depth":
        payload = 0
        for _ in range(census.MAX_PAYLOAD_DEPTH+1): payload = [payload]
    # The raw observer contract is tested directly; this does not assert that
    # an upstream parser admits JSON frames larger than its own frame limit.
    _, state, value = _observe([_call(rawInput=payload, rawOutput=payload, content=payload)], direct=True)
    for lane in census.PAYLOAD_LANES:
        measured = value["rows"][0][lane]
        assert measured["state"] == "oversized" and measured["bound_reason"] == reason
        assert measured["sha256"] is measured["canonical_json_bytes"] is None
        assert value["classification_counts"]["payload_bounds"][lane][reason] == 1
    assert value["incompleteness"]["payload_bound_exceeded"] is True
    wrong = deepcopy(value); wrong["rows"][0]["output"]["bound_reason"] = None
    with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)


def test_missing_null_empty_and_invalid_payloads_do_not_claim_a_bound_reason():
    events = [_call(rawInput=None, rawOutput={}, content=""), _update(rawOutput="\ud800")]
    _, _, value = _observe(events)
    assert value["rows"][0]["input"]["state"] == "null"
    assert value["rows"][0]["output"]["state"] == value["rows"][0]["content"]["state"] == "empty"
    assert value["rows"][1]["input"]["state"] == "missing"
    assert value["rows"][1]["output"]["state"] == "invalid"
    assert all(row[lane]["bound_reason"] is None for row in value["rows"] for lane in census.PAYLOAD_LANES)


def test_accepted_terminal_trailer_does_not_revoke_v2_tool_coverage():
    events = [_call(rawInput={"x": 1}), _update(rawOutput={"x": 2})]
    _, state, value = _observe(events, trailer=True)
    assert state["events_after_final"] == 1 and state["stream_complete"] is True
    assert value["incompleteness"]["post_final_events"] is False and value["observation_complete"] is True
    _, _, old = _observe(events, trailer=True, schema=census.SCHEMA)
    assert old["incompleteness"]["post_final_events"] is True and old["observation_complete"] is False
    tainted = deepcopy(state); tainted.update(stream_tainted=True, stream_complete=False)
    observer, _, _ = _observe(events, trailer=True)
    assert observer.snapshot(stream_state=tainted)["observation_complete"] is False


def test_post_final_tool_events_and_omitted_row_uncertainty_remain_incomplete():
    observer, state, _ = _observe([_call(), _update()])
    observer.rows[1].update(frame_ordinal=3, elapsed_seconds=3.0, after_final=True)
    state.update(final_event_seconds=2.0, events_after_final=1, stream_complete=False, stream_tainted=True)
    value = observer.snapshot(stream_state=state)
    assert value["incompleteness"]["post_final_events"] is True
    _, _, overflow = _observe([_call(str(n)) for n in range(65)], trailer=True)
    assert overflow["omitted_events"] == 1
    assert overflow["incompleteness"]["post_final_events"] is True
    assert overflow["incompleteness"]["row_limit_exceeded"] is True


def test_validator_binds_name_digest_alias_basis_presence_and_strict_classification_counts():
    _, state, value = _observe([_call(toolName="run_terminal_command", rawOutput=[1, 2])])
    changes = [dict(tool_name_sha256="0"*64), dict(resolved_tool="read_file"),
        dict(tool_resolution_basis="exact"), dict(tool="run_terminal_cmd"), dict(status_present=False),
        dict(tool_name_present=False), dict(status_shape="missing"), dict(tool_name_shape="object")]
    for change in changes:
        wrong = deepcopy(value); wrong["rows"][0].update(change)
        with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)
    wrong = deepcopy(value)
    wrong["classification_counts"]["resolved_tools"]["run_terminal_cmd"] = True
    with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)
    wrong = deepcopy(value)
    wrong["classification_counts"]["payload_bounds"]["output"]["nodes"] = 1
    with pytest.raises(ValueError): census.validate_grok_tool_events(wrong, stream_state=state)
    wrong = deepcopy(value); wrong["classification_counts"]["PRIVATE"] = "VALUE"
    with pytest.raises(ValueError) as error: census.validate_grok_tool_events(wrong, stream_state=state)
    assert "PRIVATE" not in str(error.value)


def test_versions_are_closed_and_cannot_be_relabelled_without_their_contract():
    _, state, old = _observe([_call()], schema=census.SCHEMA)
    _, _, current = _observe([_call()])
    for value, schema in ((old, census.COVERAGE_SCHEMA), (current, census.SCHEMA), (current, "unsupported@3")):
        with pytest.raises(ValueError): census.validate_grok_tool_events({**value, "schema": schema}, stream_state=state)
    for field, bad in (("payload_node_limit", 4096), ("provider_request_bytes", 0),
            ("request_history_complete", True), ("event_origin_authenticated", True)):
        with pytest.raises(ValueError): census.validate_grok_tool_events({**current, field: bad}, stream_state=state)


def test_v2_repetition_and_content_lanes_keep_original_observed_only_semantics():
    body = {"PRIVATE": "世界"}
    events = [_call("A", rawOutput=body, content=body), _update("A", rawOutput=body, content=body),
        _call("B", rawOutput=body, content=body)]
    observer, state, value = _observe(events)
    for lane in ("output", "content"):
        assert value["payloads"][lane]["same_call_repeated_events"] == 1
        assert value["payloads"][lane]["cross_call_repeated_events"] == 1
        assert value["payloads"][lane]["repeated_events"] == 2
    assert value["payload_lanes_may_overlap"] is True and value["unique_output_bytes_measured"] is False
    assert value["provider_request_bytes"] is value["history_resend_bytes"] is None
    assert value["request_history_complete"] is value["tool_result_coverage_complete"] is False
    assert "PRIVATE" not in json.dumps(value) and "世界" not in json.dumps(value)
    copied = census.validate_grok_tool_events(value, stream_state=state)
    copied["classification_counts"]["resolved_tools"]["unknown"] = 999
    assert observer.snapshot(stream_state=state) == value
