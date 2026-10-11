"""Authored rollout records; no native prompts or benchmark task data."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import codex_interaction_telemetry as telemetry


def event(kind, **payload):
    return {"type": "event_msg", "payload": {"type": kind, **payload}}


def checkpoint(input_tokens=10, output_tokens=2, **values):
    counters = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens, **values}
    return event("token_count", info={"total_token_usage": counters})


def call(identity="call-a", *, custom=False, value="echo café", stamp=None, name="exec_command"):
    record = {"type": "response_item", "payload": {
        "type": "custom_tool_call" if custom else "function_call", "call_id": identity,
        "name": name, "input" if custom else "arguments": value}}
    if stamp is not None:
        record["timestamp"] = stamp
    return record


def output(identity="call-a", *, custom=False, value="café", stamp=None):
    record = {"type": "response_item", "payload": {
        "type": "custom_tool_call_output" if custom else "function_call_output",
        "call_id": identity, "output": value}}
    if stamp is not None:
        record["timestamp"] = stamp
    return record


def build(records, **kwargs):
    return telemetry.build_interaction_telemetry(records, completed=True, malformed=0, **kwargs)


def test_utf8_bytes_and_recorded_observation_elapsed_are_not_execution_latency():
    result = build([call(stamp="2026-10-11T12:00:00+00:00"),
                    output(stamp="2026-10-11T12:00:02Z")])
    row = result["tool_pairs"][0]
    assert row["input_bytes"] == len("echo café".encode())
    assert row["observed_output_bytes"] == len("café".encode())
    assert row["input_byte_scope"] == row["output_byte_scope"] == "decoded_utf8_string"
    assert row["recorded_observation_elapsed_seconds"] == 2
    assert row["structured_output_status"] is None
    assert result["tool_observations_complete"] is True
    assert "not tool execution latency" in result["elapsed_scope"]


def test_canonical_structured_json_bytes_and_explicit_exit_status():
    value = {"output": "é", "exit_code": 7}
    result = build([call(value={"cmd": "true"}), output(value=value)])
    row = result["tool_pairs"][0]
    assert row["observed_output_bytes"] == len(json.dumps(value, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode())
    assert row["output_byte_scope"] == "canonical_json_utf8"
    assert row["structured_exit_code"] == 7
    assert row["structured_output_status"] == "observed_nonzero_exit"


@pytest.mark.parametrize("value,code,status", [
    ("Process exited with code 0; error: failure", None, None),
    ('{"exit_code": 0}', None, None),
    ({"exit_code": False}, None, None),
    ({"exit_code": 0, "error": True}, None, "conflicting_structured_status"),
    ({"error": "secret diagnostic"}, None, None),
    ({"error": True}, None, "observed_error_flag"),
    ({"isError": True, "is_error": False}, None, "conflicting_structured_status"),
])
def test_only_well_defined_structured_status(value, code, status):
    row = build([call(), output(value=value)])["tool_pairs"][0]
    assert row["structured_exit_code"] == code
    assert row["structured_output_status"] == status


def test_multiple_output_chunks_are_observations_not_duplicate_executions():
    result = build([call(custom=True, stamp="2026-10-11T12:00:00Z"),
                    output(custom=True, value="a", stamp="2026-10-11T12:00:01Z"),
                    output(custom=True, value="a", stamp="2026-10-11T12:00:02Z")])
    row = result["tool_pairs"][0]
    assert row["call_event_count"] == 1
    assert row["output_event_count"] == 2
    assert row["observed_output_bytes"] == 2
    assert row["pairing_status"] == "multiple_output_chunks_observed"
    assert row["recorded_observation_elapsed_seconds"] is None
    assert result["tool_observations_complete"] is False


@pytest.mark.parametrize("records,status", [
    ([call(), call(), output()], "ambiguous_call_identity"),
    ([call(), output(custom=True)], "ambiguous_call_identity"),
    ([output(), call()], "output_precedes_call"),
    ([output()], "output_without_observed_call"),
    ([call()], "call_without_observed_output"),
])
def test_ambiguous_and_partial_pairs_never_claim_complete(records, status):
    result = build(records)
    assert result["tool_pairs"][0]["pairing_status"] == status
    assert result["tool_pairs"][0]["recorded_observation_elapsed_seconds"] is None
    assert result["tool_observations_complete"] is False


@pytest.mark.parametrize("start,stop", [
    ("2026-10-11T12:00:00", "2026-10-11T12:00:01Z"),
    ("2026-10-11T12:00:02Z", "2026-10-11T12:00:01Z"),
    ("2026-10-11T12:00:00Z", "2026-10-13T12:00:00Z"),
    ("unknown", "2026-10-11T12:00:00Z"),
    (None, None),
])
def test_invalid_timestamp_pairs_have_unknown_elapsed(start, stop):
    assert build([call(stamp=start), output(stamp=stop)])["tool_pairs"][0][
        "recorded_observation_elapsed_seconds"] is None


def test_clock_regression_elsewhere_withholds_elapsed():
    result = build([{"timestamp": "2026-10-11T13:00:00Z", "type": "other"},
                    call(stamp="2026-10-11T12:00:00Z"),
                    output(stamp="2026-10-11T12:00:02Z")])
    assert result["timestamp_order_monotonic"] is False
    assert result["tool_pairs"][0]["recorded_observation_elapsed_seconds"] is None


def test_first_snapshot_unknown_then_adjacent_and_repeated_snapshots():
    result = build([event("task_started"), checkpoint(cached_input_tokens=5), call(), output(),
                    checkpoint(20, 5, cached_input_tokens=10),
                    checkpoint(20, 5, cached_input_tokens=10)])
    first, second, third = result["token_observations"]
    assert first["adjacent_counter_delta"] is None
    assert first["explicit_boundary_event_ordinal"] == 1
    assert second["adjacent_counter_delta"] == {
        "input_tokens": 10, "output_tokens": 3, "total_tokens": 13, "cached_input_tokens": 5}
    assert second["observed_tool_call_events_since_previous_checkpoint"] == 1
    assert second["observed_tool_output_events_since_previous_checkpoint"] == 1
    assert third["interval_status"] == "repeated_snapshot"
    assert third["adjacent_counter_delta"]["total_tokens"] == 0
    assert result["repeated_cumulative_snapshots"] == 1
    assert result["provider_request_count_known"] is False
    assert result["tool_token_attribution_available"] is False


def test_session_meta_does_not_assert_zero_origin_or_join_session_boundaries():
    meta = {"type": "session_meta", "payload": {"id": "secret-thread-id"}}
    result = build([meta, call(), checkpoint(), meta, output(), checkpoint(20, 5)])
    assert all(row["adjacent_counter_delta"] is None for row in result["token_observations"])
    assert len(result["tool_pairs"]) == 2
    assert result["tool_observations_complete"] is False


def test_zero_origin_requires_explicit_assertion_and_no_malformed_records():
    assert build([checkpoint()], session_start_known=True)["token_observations"][0][
        "adjacent_counter_delta"]["total_tokens"] == 12
    result = telemetry.build_interaction_telemetry([checkpoint()], completed=True,
        malformed=1, session_start_known=True)
    assert result["token_observations"][0]["adjacent_counter_delta"] is None


@pytest.mark.parametrize("bad", [
    {"input_tokens": True}, {"output_tokens": -1}, {"total_tokens": 13},
    {"cached_input_tokens": 11}, {"reasoning_output_tokens": 3},
    {"cache_write_input_tokens": False}, {"input_tokens": 1 << 64},
])
def test_invalid_snapshot_breaks_counter_continuity(bad):
    invalid = checkpoint()
    invalid["payload"]["info"]["total_token_usage"].update(bad)
    result = build([checkpoint(), invalid, checkpoint(20, 5), checkpoint(30, 8)])
    rows = result["token_observations"]
    assert rows[1]["cumulative_counters"] is None
    assert rows[1]["adjacent_counter_delta"] is None
    assert rows[2]["adjacent_counter_delta"] is None
    assert rows[3]["adjacent_counter_delta"]["total_tokens"] == 13
    assert result["token_observations_complete"] is False


@pytest.mark.parametrize("invalid", [[], {"total_token_usage": None}, {"total_token_usage": []},
    {"total_token_usage": {"input_tokens": 1}}])
def test_missing_or_malformed_snapshot_is_not_zero(invalid):
    result = build([event("token_count", info=invalid)])
    assert result["token_observations"][0]["cumulative_counters"] is None
    assert result["invalid_token_snapshots"] == 1


@pytest.mark.parametrize("info", [None, {}, {"rate_limits": {"remaining": 100}}])
def test_quota_updates_preserve_adjacent_usage_interval_and_tool_counts(info):
    result = build([checkpoint(), call(), event("token_count", info=info),
                    output(), checkpoint(20, 5)])
    assert result["non_usage_token_events"] == 1
    assert result["observed_token_checkpoints"] == 2
    assert result["invalid_token_snapshots"] == 0
    row = result["token_observations"][-1]
    assert row["adjacent_counter_delta"]["total_tokens"] == 13
    assert row["observed_tool_call_events_since_previous_checkpoint"] == 1
    assert row["observed_tool_output_events_since_previous_checkpoint"] == 1
    assert result["token_observations_complete"] is True


def test_decrease_invalidates_interval_and_next_baseline():
    result = build([checkpoint(20, 5), checkpoint(10, 2), checkpoint(30, 8), checkpoint(40, 10)])
    rows = result["token_observations"]
    assert rows[1]["interval_status"] == "counter_decrease"
    assert rows[2]["adjacent_counter_delta"] is None
    assert rows[3]["adjacent_counter_delta"]["total_tokens"] == 12
    assert result["token_observations_complete"] is False


def test_optional_counters_missing_on_one_endpoint_stay_absent():
    result = build([checkpoint(), checkpoint(20, 5, cached_input_tokens=10)])
    assert "cached_input_tokens" not in result["token_observations"][1]["adjacent_counter_delta"]


def test_invalid_subset_delta_is_unknown_and_breaks_continuity():
    result = build([checkpoint(cached_input_tokens=0),
                    checkpoint(11, 3, cached_input_tokens=10), checkpoint(20, 5)])
    assert result["token_observations"][1]["interval_status"] == "invalid_interval"
    assert result["token_observations"][2]["adjacent_counter_delta"] is None
    assert result["token_observations_complete"] is False


def test_unknown_malformed_record_positions_null_all_deltas():
    result = telemetry.build_interaction_telemetry([checkpoint(), checkpoint(20, 5)],
                                                  completed=True, malformed=1)
    assert all(row["adjacent_counter_delta"] is None for row in result["token_observations"])
    assert result["token_observations_complete"] is False


@pytest.mark.parametrize("bad", [None, [], "secret", {}, {"type": []}, {"type": True},
    {"type": "event_msg", "payload": []},
    {"type": "event_msg", "payload": {"type": []}},
    {"type": "response_item", "payload": {"type": {}}}])
def test_malformed_shapes_do_not_raise(bad):
    result = build([checkpoint(), bad, checkpoint(20, 5)])
    assert result["invalid_record_shapes"] == 1
    assert result["token_observations"][-1]["adjacent_counter_delta"] is None
    assert result["token_observations_complete"] is False


def test_arbitrary_content_never_leaks_from_names_identifiers_payloads_or_errors():
    secret = "PRIVATE-MARKER-/home/user/token"
    result = build([{"type": "session_meta", "payload": {"id": secret, "cwd": secret}},
                    call(secret, name=secret, value={"cmd": secret}),
                    output(secret, value={"error": secret, "stdout": secret, "exit_code": 1}),
                    {"type": secret, "payload": {"name": secret}},
                    checkpoint()])
    assert secret not in json.dumps(result)
    assert result["tool_pairs"][0]["tool_name"] == "other"
    assert result["raw_content_exported"] is False
    assert result["native_aggregate_usage_modified"] is False


def test_tool_and_checkpoint_limits_retain_counts_without_unbounded_rows():
    records = []
    for index in range(140):
        records.extend([call(f"call-{index}"), output(f"call-{index}")])
    records.extend(checkpoint(index, index) for index in range(270))
    result = build(iter(records))
    assert len(result["tool_pairs"]) == 128
    assert result["omitted_tool_events"] == 24
    assert result["observed_tool_call_events"] == 140
    assert result["observed_tool_output_events"] == 140
    assert len(result["token_observations"]) == 256
    assert result["omitted_token_observations"] == 14
    assert result["tool_observations_complete"] is False
    assert result["token_observations_complete"] is False
    assert len(json.dumps(result).encode()) < 250_000


def test_record_limit_stops_unbounded_iterable(monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RECORDS", 3)
    def records():
        while True:
            yield checkpoint()
    result = build(records())
    assert result["observed_records"] == 3
    assert result["record_limit_exceeded"] is True
    assert result["token_observations_complete"] is False


@pytest.mark.parametrize("value", [None, True, float("nan"), {"x": float("inf")},
    {1: "not JSON object key"}, "\ud800"])
def test_unknown_payload_size_is_not_zero(value):
    result = build([call(value=value), output(value=value)])
    assert result["tool_pairs"][0]["input_bytes"] is None
    assert result["tool_pairs"][0]["observed_output_bytes"] is None
    assert result["tool_observations_complete"] is False


def test_payload_bounds_cycles_and_unknown_ids(monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_PAYLOAD_BYTES", 8)
    cycle = []
    cycle.append(cycle)
    result = build([call(value="a" * 9), output(value=cycle), call(identity=None)])
    assert result["tool_pairs"][0]["input_bytes"] is None
    assert result["tool_pairs"][0]["observed_output_bytes"] is None
    assert result["invalid_tool_identity_events"] == 1


def test_false_completion_and_invalid_malformed_count_never_claim_complete():
    result = telemetry.build_interaction_telemetry([call(), output(), checkpoint()],
        completed=False, malformed=False)
    assert result["malformed_json_records"] is None
    assert result["tool_observations_complete"] is False
    assert result["token_observations_complete"] is False


def test_complete_empty_census_never_establishes_usage_completeness():
    result = build([])
    assert result["observed_token_checkpoints"] == 0
    assert result["token_observations_complete"] is True
    assert result["usage_completeness_established"] is False
    assert result["valid_timestamp_events"] == 0
    assert "observation census only" in result["completeness_scope"]
