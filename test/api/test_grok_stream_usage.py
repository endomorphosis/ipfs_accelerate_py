"""Authored Grok streaming-json producer shapes; no live model calls.

The local shipped headless reference (1.0.52, section streaming-json) documents
the usage/messageId/stopReason/signature frame used here. These are authored
fixtures, not captured transcripts from the previous 1.0.46 benchmark.
"""
import copy
import json

import pytest

from ipfs_accelerate_py.cli_runtime.grok_stream_capture import (
    GrokStreamParser, LEGACY_SCHEMA, validate_grok_stream_progress,
)
from ipfs_accelerate_py.cli_runtime.grok_stream_usage import MAX_COUNT, MAX_USAGE_EVENTS, USAGE_FIELDS
from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation


def encode(*events):
    return b"".join(json.dumps(event).encode() + b"\n" for event in events)


def usage(**overrides):
    return {"type": "usage", "messageId": "private-message-id", "stopReason": "tool_use",
            "usage": {"input_tokens": 812, "output_tokens": 45, "cache_read_input_tokens": 4000,
                      "cache_creation_input_tokens": 0, "reasoning_tokens": 3},
            "signature": "private-signature", **overrides}


def observed(*events):
    parser = GrokStreamParser()
    parser.feed(encode(*events), elapsed_seconds=0.5)
    return parser, parser.snapshot()


def test_documented_usage_shape_retains_counters_without_ids_or_invented_totals():
    parser, state = observed({"type": "thought", "data": "private thought"}, usage(),
        usage(stopReason="end_turn", usage={"input_tokens": 12, "cache_read_input_tokens": 8000}))
    evidence = state["usage_observations"]
    assert evidence["observed_events"] == evidence["retained_events"] == 2
    assert evidence["omitted_events"] == 0
    assert evidence["counter_semantics"] == "unverified"
    assert evidence["summation_permitted"] is evidence["ordinals_are_turn_ids"] is False
    first, second = evidence["rows"]
    assert (first["event_ordinal"], first["frame_ordinal"], first["elapsed_seconds"]) == (1, 2, 0.5)
    assert first["usage"]["input_tokens"] == 812
    assert first["usage"]["cache_read_input_tokens"] == 4000
    assert first["usage"]["reasoning_tokens"] == 3
    assert first["usage"]["total_tokens"] is None
    assert second["usage"]["input_tokens"] == 12  # Neither delta nor cumulative is assumed.
    assert second["usage"]["output_tokens"] is None
    assert "private" not in json.dumps(state)
    assert "messageId" not in json.dumps(state)
    assert "signature" not in json.dumps(state)
    state["usage_observations"]["rows"][0]["usage"]["input_tokens"] = 999
    state["usage_observations"]["rows"].clear()
    assert parser.snapshot()["usage_observations"]["rows"][0]["usage"]["input_tokens"] == 812


def test_bytewise_feed_and_terminal_usage_stay_separate():
    parser = GrokStreamParser()
    events = encode(usage(), usage(usage={"input_tokens": 999999}),
        {"type": "text", "data": "done"},
        {"type": "end", "stopReason": "end_turn", "usage": {"input_tokens": 7, "total_tokens": 9}})
    for item in events:
        parser.feed(bytes([item]), elapsed_seconds=0.5)
    parser.eof("stdout", elapsed_seconds=1)
    parser.eof("stderr", elapsed_seconds=1)
    parser.process_exit(0, elapsed_seconds=1)
    state = parser.snapshot()
    assert state["response_usable"] is True
    assert state["usage_observations"]["observed_events"] == 2
    final = native_grok_observation(parser.final_payload, exit_code=0)
    assert final["native_grok_input_tokens"] == 7
    assert final["native_grok_total_tokens"] == 9
    assert "native_grok_usage_complete" not in final


@pytest.mark.parametrize("raw,status,container,invalid,present", [
    (None, "missing", "missing", 0, 0), ({}, "missing", "mapping", 0, 0),
    ({"private_key": 1}, "missing", "mapping", 0, 0),
    ([], "invalid", "invalid", 0, 0), (True, "invalid", "invalid", 0, 0),
    ({"input_tokens": True}, "invalid", "mapping", 1, 0),
    ({"input_tokens": -1}, "invalid", "mapping", 1, 0),
    ({"input_tokens": MAX_COUNT + 1}, "invalid", "mapping", 1, 0),
    ({"input_tokens": "12"}, "invalid", "mapping", 1, 0),
    ({"input_tokens": 1.0}, "invalid", "mapping", 1, 0),
    ({"input_tokens": 0, "output_tokens": -1}, "partial", "mapping", 1, 1),
    ({"input_tokens": MAX_COUNT, "output_tokens": 0}, "valid", "mapping", 0, 2),
])
def test_bad_or_missing_counters_remain_unknown_without_changing_stream_authority(raw, status, container, invalid, present):
    _, state = observed(usage(usage=raw, stopReason="private reason"))
    row, = state["usage_observations"]["rows"]
    assert row["counter_status"] == status
    assert row["usage_container"] == container
    assert row["invalid_counter_fields"] == invalid
    assert sum(value is not None for value in row["usage"].values()) == present
    assert row["stop_reason"] == "unknown"
    assert state["stream_tainted"] is False  # Diagnostics cannot alter provider outcomes.
    assert state["response_usable"] is False
    assert "private" not in json.dumps(state)


def test_more_than_limit_is_bounded_and_explicit_even_when_later_usage_missing():
    parser = GrokStreamParser()
    largest = usage(usage=dict.fromkeys(USAGE_FIELDS, MAX_COUNT))
    for _ in range(MAX_USAGE_EVENTS + 37):
        parser.feed(encode(largest), elapsed_seconds=MAX_COUNT)
    state = parser.snapshot()
    evidence = state["usage_observations"]
    assert evidence["observed_events"] == 165
    assert evidence["retained_events"] == len(evidence["rows"]) == MAX_USAGE_EVENTS
    assert evidence["omitted_events"] == 37
    assert evidence["rows"][-1]["event_ordinal"] == 128
    assert len(json.dumps(state).encode()) < 128 * 1024
    assert len(parser._pending) == 0


@pytest.mark.parametrize("line", [b'{"type":"usage","usage":{"input_tokens":NaN}}\n',
    b'{"type":"usage","usage":{"input_tokens":1,"input_tokens":2}}\n',
    b'{"type":"usage","usage":{"input_tokens":1}',
    b'{"type":"usage","usage":{"input_tokens":' + b'9' * 1000 + b'}}\n'])
def test_malformed_partial_and_oversized_frames_do_not_become_usage_rows(line):
    parser = GrokStreamParser(max_frame_bytes=256)
    parser.feed(encode(usage(usage={"input_tokens": 5})))
    parser.feed(line, elapsed_seconds=0.5)
    parser.eof("stdout", elapsed_seconds=1)
    state = parser.snapshot()
    assert state["stream_tainted"] is True
    assert state["usage_observations"]["observed_events"] == 1
    assert state["usage_observations"]["rows"][0]["usage"]["input_tokens"] == 5
    assert state["response_usable"] is False


def test_missing_end_and_post_end_usage_are_observations_not_completions():
    parser, state = observed(usage())
    assert state["usage_observations"]["rows"][0]["after_final"] is False
    assert state["response_usable"] is False
    parser.feed(encode({"type": "end", "stopReason": "end_turn"}, usage()), elapsed_seconds=1)
    state = parser.snapshot()
    assert state["usage_observations"]["rows"][-1]["after_final"] is True
    assert state["stream_tainted"] is True
    assert parser.final_payload is None


def test_historical_progress_stays_historical_and_has_no_invented_usage_rows():
    _, state = observed(usage())
    state["schema"] = LEGACY_SCHEMA
    del state["usage_observations"]
    assert validate_grok_stream_progress(state) == state
    assert "usage_observations" not in validate_grok_stream_progress(state)


@pytest.mark.parametrize("mutate", [
    lambda s: s["usage_observations"].update(raw="private"),
    lambda s: s["usage_observations"].update(summation_permitted=True),
    lambda s: s["usage_observations"].update(retained_events=0),
    lambda s: s["usage_observations"].update(observed_events=2),
    lambda s: s["usage_observations"].update(omitted_events=1),
    lambda s: s["usage_observations"]["rows"].append(copy.deepcopy(s["usage_observations"]["rows"][0])),
    lambda s: s["usage_observations"]["rows"][0].update(event_ordinal=True),
    lambda s: s["usage_observations"]["rows"][0].update(frame_ordinal=2),
    lambda s: s["usage_observations"]["rows"][0].update(elapsed_seconds=float("nan")),
    lambda s: s["usage_observations"]["rows"][0].update(elapsed_seconds=2),
    lambda s: s["usage_observations"]["rows"][0].update(after_final=1),
    lambda s: s["usage_observations"]["rows"][0].update(after_final=True),
    lambda s: s["usage_observations"]["rows"][0].update(stop_reason="private"),
    lambda s: s["usage_observations"]["rows"][0].update(counter_status="missing"),
    lambda s: s["usage_observations"]["rows"][0].update(usage_container="missing"),
    lambda s: s["usage_observations"]["rows"][0].update(invalid_counter_fields=6),
    lambda s: s["usage_observations"]["rows"][0]["usage"].update(input_tokens=True),
    lambda s: s["usage_observations"]["rows"][0]["usage"].update(input_tokens=-1),
    lambda s: s["usage_observations"]["rows"][0]["usage"].update(input_tokens=MAX_COUNT + 1),
    lambda s: s["usage_observations"]["rows"][0]["usage"].update(private=123),
])
def test_usage_validator_rejects_open_invalid_or_inconsistent_observations(mutate):
    _, state = observed(usage())
    mutate(state)
    with pytest.raises(ValueError):
        validate_grok_stream_progress(state)


@pytest.mark.parametrize("mutate", [
    lambda s: s["usage_observations"]["rows"][1].update(elapsed_seconds=0.25),
    lambda s: s["usage_observations"]["rows"][2].update(after_final=False),
    lambda s: s.update(events_after_final=1),
])
def test_post_final_rows_must_join_terminal_time_count_and_order(mutate):
    parser = GrokStreamParser()
    parser.feed(encode(usage()), elapsed_seconds=0.1)
    parser.feed(encode({"type": "end", "stopReason": "end_turn"}), elapsed_seconds=0.5)
    parser.feed(encode(usage(), usage()), elapsed_seconds=1)
    state = parser.snapshot()
    mutate(state)
    with pytest.raises(ValueError):
        validate_grok_stream_progress(state)
