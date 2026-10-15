"""Authored protocol diagnostics; no raw native transcript or model calls."""
import copy
import json

import pytest

from ipfs_accelerate_py.cli_runtime.grok_stream_capture import (
    GrokStreamParser, LEGACY_SCHEMA, USAGE_SCHEMA, SCHEMA,
    grok_stream_progress_summary, validate_grok_stream_progress,
)


def end(**changes):
    return {"type": "end", "stopReason": "max_turn_requests", **changes}


def parse(*events):
    parser = GrokStreamParser()
    for i, event in enumerate(events):
        frame = event if type(event) is bytes else json.dumps(event).encode() + b"\n"
        parser.feed(frame, elapsed_seconds=i)
    parser.eof("stdout", elapsed_seconds=len(events))
    parser.eof("stderr", elapsed_seconds=len(events))
    parser.process_exit(1, elapsed_seconds=len(events))
    return parser, parser.snapshot()


@pytest.mark.parametrize("events,reason,fields", [
    ([{"type": "max_turns_reached"}, end(PRIVATE_KEY="PRIVATE_VALUE")], "unknown_end_fields", []),
    ([{"type": "max_turns_reached"}, end(message="PRIVATE_VALUE")], "unknown_end_fields", []),
    ([{"type": "max_turns_reached"}, end(stopReason="end_turn")], "end_reason_mismatch", []),
    ([{"type": "max_turns_reached"}, end(stopReason="PRIVATE_VALUE")], "end_reason_mismatch", []),
    ([{"type": "max_turns_reached"}, end(sessionId=None)], "invalid_end_field_shape", ["sessionId"]),
    ([{"type": "max_turns_reached", "usage": {"input_tokens": 1}}, end()], "missing_frozen_fields", ["usage"]),
    ([{"type": "max_turns_reached", "usage": {"input_tokens": 1}}, end(usage={"input_tokens": True})], "conflicting_frozen_fields", ["usage"]),
    ([{"type": "max_turns_reached"}, {"type": "text", "data": "PRIVATE_VALUE"}], "not_end_event", []),
    ([{"type": "max_turns_reached"}, {"type": "auto_compact_PRIVATE", "data": "PRIVATE_VALUE"}], "not_end_event", []),
    ([{"type": "end", "stopReason": "end_turn"}, end()], "second_final_envelope", []),
    ([{"type": "end"}, {"type": "end"}], "unknown_prior_reason", []),
    ([{"type": "max_turns_reached"}, b"bad\n", end()], "prior_payload_unavailable", []),
    ([{"type": "max_turns_reached"}, {"type": "end"}, {"type": "end"}], "duplicate_end_trailer", []),
])
def test_terminal_rejection_identifies_closed_branch_without_provider_values(events, reason, fields):
    parser, state = parse(*events)
    rejection = state["terminal_rejection"]
    assert state["schema"] == SCHEMA
    assert state["stream_tainted"] is True
    assert state["response_usable"] is False
    assert parser.final_payload is None
    assert rejection["reason"] == reason
    assert rejection["rejected_fields"] == fields
    assert rejection["raw_provider_data_exported"] is False
    assert rejection["completion_authority"] is False
    assert "PRIVATE" not in json.dumps(state)


def test_known_unexpected_field_names_and_unknown_count_reveal_shape_only():
    _, state = parse({"type": "max_turns_reached", "sessionId": "PRIVATE_FIRST"},
                     end(message="PRIVATE_MESSAGE", PRIVATE_KEY="PRIVATE_VALUE", sessionId=None))
    row = state["terminal_rejection"]
    assert row["reason"] == "unknown_end_fields"
    assert row["unexpected_fields"] == ["message"]
    assert row["unknown_field_count"] == 1
    assert row["field_shapes"]["sessionId"] == "null"
    assert row["prior_field_shapes"]["sessionId"] == "string"
    assert row["field_shapes"]["usage"] == "missing"
    assert row["incoming_stop_reason"] == "max_turn_requests"
    assert row["incoming_reason_code"] == row["prior_reason_code"] == "max_turns"
    assert "PRIVATE" not in json.dumps(row)
    assert len(json.dumps(row)) < 2048


def test_only_first_rejection_is_retained_and_snapshot_is_independent():
    parser, state = parse({"type": "max_turns_reached"}, end(sessionId=None), end(message="PRIVATE"))
    row = state["terminal_rejection"]
    assert row["frame_ordinal"] == 2
    assert row["reason"] == "invalid_end_field_shape"
    row["field_shapes"]["sessionId"] = "string"
    row["rejected_fields"].clear()
    assert parser.snapshot()["terminal_rejection"]["field_shapes"]["sessionId"] == "null"
    assert parser.snapshot()["terminal_rejection"]["rejected_fields"] == ["sessionId"]


def test_matching_end_still_accepted_without_diagnostic_or_counter_summation():
    parser, state = parse({"type": "usage", "usage": {"input_tokens": 12345}},
                         {"type": "max_turns_reached"}, end(usage={"input_tokens": 7}))
    assert state["terminal_rejection"] is None
    assert state["stream_tainted"] is False
    assert state["response_usable"] is False
    assert parser.final_payload["usage"] == {"input_tokens": 7}
    assert state["usage_observations"]["summation_permitted"] is False


@pytest.mark.parametrize("fields,reason", [
    ({"PRIVATE_KEY": "PRIVATE_VALUE"}, "unknown_end_fields"),
    ({"text": "PRIVATE_VALUE"}, "unknown_end_fields"),
    ({"stop_reason": "end_turn"}, "unknown_end_fields"),
    ({"usage": None}, "invalid_end_field_shape"),
    ({"modelUsage": []}, "invalid_end_field_shape"),
    ({"sessionId": None}, "invalid_end_field_shape"),
    ({"requestId": False}, "invalid_end_field_shape"),
    ({"num_turns": True}, "invalid_end_field_shape"),
    ({"total_cost_usd_ticks": -1}, "invalid_end_field_shape"),
    ({"total_cost_usd": True}, "invalid_end_field_shape"),
    ({"usage_is_incomplete": "false"}, "invalid_end_field_shape"),
    ({"cost_is_partial": 0}, "invalid_end_field_shape"),
])
def test_turn_guard_cancelled_compatibility_keeps_all_trailer_shape_checks(fields, reason):
    parser, state = parse({"type": "max_turns_reached"}, end(stopReason="cancelled", **fields))
    assert state["stream_tainted"] is True
    assert state["response_usable"] is False
    assert state["terminal_rejection"]["reason"] == reason
    assert parser.final_payload is None
    assert "PRIVATE" not in json.dumps(state)


@pytest.mark.parametrize("first,reason", [
    ({"type": "error", "subtype": "error_max_turns"}, "end_reason_mismatch"),
    ({"type": "error", "stopReason": "max_turn_requests"}, "end_reason_mismatch"),
    ({"type": "end", "stopReason": "max_turn_requests"}, "second_final_envelope"),
])
def test_cancelled_trailer_alias_is_limited_to_exact_native_turn_guard(first, reason):
    parser, state = parse(first, end(stopReason="cancelled", usage={"input_tokens": 7}))
    assert state["terminal_rejection"]["reason"] == reason
    assert parser.final_payload is None
    assert state["response_usable"] is False


@pytest.mark.parametrize("tail", [
    end(stopReason="cancelled", usage={"input_tokens": 7}), {"type": "end"},
    {"type": "usage", "usage": {"input_tokens": 7}}, b"malformed\n",
])
def test_cancelled_trailer_cannot_be_duplicated_or_followed_by_more_accounting(tail):
    parser, state = parse({"type": "max_turns_reached"},
                         end(stopReason="cancelled", usage={"input_tokens": 7}), tail)
    assert state["stream_tainted"] is True
    assert state["response_usable"] is False
    assert parser.final_payload is None


@pytest.mark.parametrize("schema", [LEGACY_SCHEMA, USAGE_SCHEMA])
def test_historical_progress_replays_exactly_without_inventing_diagnostics(schema):
    _, current = parse({"type": "max_turns_reached"}, end(sessionId=None))
    historical = copy.deepcopy(current)
    historical["schema"] = schema
    del historical["terminal_rejection"]
    if schema == LEGACY_SCHEMA:
        del historical["usage_observations"]
    original = json.dumps(historical, sort_keys=True)
    restored = validate_grok_stream_progress(historical)
    assert json.dumps(restored, sort_keys=True) == original
    assert "terminal_rejection" not in restored


def test_callbacks_keep_historical_count_only_projection():
    _, state = parse({"type": "max_turns_reached"}, end(sessionId=None))
    callback = grok_stream_progress_summary(state)
    assert callback["schema"] == LEGACY_SCHEMA
    assert "terminal_rejection" not in callback
    assert "usage_observations" not in callback


@pytest.mark.parametrize("change", [
    lambda s: s["terminal_rejection"].update(raw_end="PRIVATE"),
    lambda s: s["terminal_rejection"].update(reason="PRIVATE"),
    lambda s: s["terminal_rejection"].update(incoming_stop_reason="PRIVATE"),
    lambda s: s["terminal_rejection"].update(completion_authority=True),
    lambda s: s["terminal_rejection"].update(unknown_field_count=True),
    lambda s: s["terminal_rejection"].update(frame_ordinal=999),
    lambda s: s["terminal_rejection"].update(frame_ordinal=1),
    lambda s: s["terminal_rejection"].update(elapsed_seconds=999),
    lambda s: s["terminal_rejection"].update(prior_terminal_type="error"),
    lambda s: s["terminal_rejection"].update(prior_reason_code="end_turn"),
    lambda s: s["terminal_rejection"].update(event_type="text"),
    lambda s: s["terminal_rejection"]["field_shapes"].update(PRIVATE="string"),
    lambda s: s["terminal_rejection"]["field_shapes"].update(message="PRIVATE"),
    lambda s: s["terminal_rejection"].update(unexpected_fields=["PRIVATE"]),
    lambda s: s["terminal_rejection"].update(rejected_fields=["PRIVATE"]),
    lambda s: s.update(events_after_final=0),
])
def test_diagnostic_validator_refuses_open_or_unjoined_observations(change):
    _, state = parse({"type": "max_turns_reached"}, end(sessionId=None))
    change(state)
    with pytest.raises(ValueError):
        validate_grok_stream_progress(state)
