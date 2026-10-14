"""Content-free fingerprints over existing native pairing observations."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import codex_interaction_telemetry as interaction
from ipfs_accelerate_py.agent_supervisor.runtime import codex_usage_receipt as usage
from ipfs_accelerate_py.agent_supervisor.runtime import native_context_delivery as ledger


def item(kind, **fields):
    return {"type": "response_item", "payload": {"type": kind, **fields}}


def call(identity="private-call-id", *, name="exec_command", value='{"cmd":"PRIVATE /secret/path"}', custom=False):
    return item("custom_tool_call" if custom else "function_call", call_id=identity, name=name,
                **{"input" if custom else "arguments": value})


def output(identity="private-call-id", *, value="PRIVATE tool output", custom=False):
    return item("custom_tool_call_output" if custom else "function_call_output", call_id=identity, output=value)


def token(value=100):
    return {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {
        "input_tokens": value, "cached_input_tokens": value // 2, "output_tokens": 10, "total_tokens": value + 10}}}}


def build(records, *, completed=True, malformed=0):
    telemetry = interaction.build_interaction_telemetry(records, completed=completed, malformed=malformed)
    return ledger.build_native_context_delivery(records, telemetry=telemetry), telemetry


def reseal(value):
    value["ledger_sha256"] = ledger._sha(ledger._wire({k: v for k, v in value.items() if k != "ledger_sha256"}))
    return value


def test_exact_string_fingerprints_do_not_decode_json_or_export_private_content():
    arguments = '{"cmd": "é /secret/path"}'
    text = '{ "output": "é PRIVATE" }\n'
    records = [call(value=arguments), output(value=text)]
    result, telemetry = build(records)
    before = deepcopy(result)
    one, two = result["records"]
    assert one["payload_sha256"] == hashlib.sha256(arguments.encode()).hexdigest()
    assert two["payload_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert two["payload_bytes"] == len(text.encode()) > len(text)
    assert two["payload_byte_scope"] == "decoded_utf8_string"
    assert two["call_id_sha256"] == hashlib.sha256(b"private-call-id").hexdigest()
    assert one["tool_name"] == two["tool_name"] == "exec_command"
    assert [r["pairing_status"] for r in result["records"]] == [telemetry["tool_pairs"][0]["pairing_status"]] * 2
    assert result["observation_census_complete"] is result["fingerprints_complete"] is True
    assert result["interaction_telemetry_sha256"] == ledger._sha(ledger._wire(telemetry))
    assert all(result[k] is False for k in ledger._FALSE_FIELDS)
    encoded = ledger._wire(result)
    for private in (b"PRIVATE", b"/secret/path", b"private-call-id", b'"arguments"', b'"output"'):
        assert private not in encoded
    detached = ledger.validate_native_context_delivery(result)
    detached["records"][0]["payload_bytes"] += 1
    assert result == before


@pytest.mark.parametrize("value", [{"z": ["é", None], "a": 2}, ["PRIVATE", {"x": False}]])
def test_structured_output_is_canonical_json_not_original_wire_text(value):
    result, _ = build([call(), output(value=value)])
    observed = result["records"][1]
    assert observed["payload_byte_scope"] == "canonical_json_utf8"
    assert observed["payload_bytes"] == len(ledger._wire(value))
    assert observed["payload_sha256"] == ledger._sha(ledger._wire(value))
    assert result["payload_measurements"]["canonical_json_utf8"]["output_events"] == 1
    assert "neither transport frames nor provider" in result["byte_scope"]


@pytest.mark.parametrize("records,status", [
    ([call(), output(), output(value="second private chunk")], "multiple_output_chunks_observed"),
    ([call(), call(), output()], "ambiguous_call_identity"),
    ([output(), call()], "output_precedes_call"),
    ([output()], "output_without_observed_call"),
    ([call()], "call_without_observed_output"),
    ([call(), output(custom=True)], "ambiguous_call_identity"),
])
def test_pairing_and_every_chunk_remain_existing_observations(records, status):
    result, telemetry = build(records)
    assert result["retained_tool_events"] == len(records)
    assert [r["record_ordinal"] for r in result["records"]] == list(range(1, len(records) + 1))
    assert all(r["pairing_status"] == status for r in result["records"])
    assert result["tool_observations_complete"] is telemetry["tool_observations_complete"] is False
    assert result["measured_payload_bytes"] == sum(r["payload_bytes"] for r in result["records"])


def test_custom_tool_and_unknown_name_are_hashed_not_exposed():
    result, _ = build([call(name="private.tool./secret/path", custom=True), output(custom=True)])
    for row in result["records"]:
        assert row["tool_name"] == "other" and row["tool_kind"] == "custom"
        assert row["tool_name_sha256"] == hashlib.sha256(b"private.tool./secret/path").hexdigest()
    assert b"private.tool" not in ledger._wire(result)


def test_same_call_id_in_separate_existing_session_scopes_is_not_merged():
    records = [call(), output(), {"type": "session_meta", "payload": {}}, call(), output(value="different")]
    result, telemetry = build(records)
    assert len(telemetry["tool_pairs"]) == 2
    assert [r["interaction_observation_ordinal"] for r in result["records"]] == [1, 1, 2, 2]
    assert len({r["call_id_sha256"] for r in result["records"]}) == 1


def test_token_checkpoint_positions_are_retained_ordinals_not_request_attribution():
    records = [token(), call(), output(), token(130), call("other"), output("other")]
    result, _ = build(records)
    assert [r["preceding_token_observation_ordinal"] for r in result["records"]] == [1, 1, 2, 2]
    assert result["provider_request_count_inferred"] is result["causal_tool_token_attribution"] is False
    assert "preceding retained" in result["token_position_scope"].lower()
    result, _ = build([call(), output(), token()])
    assert all(r["preceding_token_observation_ordinal"] is None for r in result["records"])


@pytest.mark.parametrize("identity", [None, "", 123, "x" * 257, "\ud800"])
def test_invalid_or_unhashable_identity_is_explicit(identity):
    result, telemetry = build([call(identity), output(identity)])
    # Hashability is an additional observation, not a rewrite of the existing
    # owner's pairing decision (whose ID grammar can admit escaped surrogates).
    assert result["tool_observations_complete"] is telemetry["tool_observations_complete"]
    assert all(r["call_id_sha256"] is r["interaction_observation_ordinal"] is r["pairing_status"] is None
               for r in result["records"])
    assert all(r["call_id_unknown_reason"] for r in result["records"])


@pytest.mark.parametrize("value", [None, True, 2, float("nan"), "\ud800", {"value": float("inf")}, {1: "x"}])
def test_unknown_payloads_never_fabricate_a_hash_or_zero_bytes(value):
    result, _ = build([call(), output(value=value)])
    row = result["records"][1]
    assert row["payload_sha256"] is row["payload_bytes"] is None
    assert row["payload_unknown_reason"] is not None
    assert result["payloads_without_fingerprint"] == 1 and result["fingerprints_complete"] is False
    assert result["payload_measurements"]["unavailable"]["bytes"] is None


def test_nested_cycle_and_excessive_depth_are_bounded_unknowns():
    cycle = []; cycle.append(cycle)
    result, _ = build([call(), output(value=cycle)])
    assert result["records"][1]["payload_unknown_reason"] == "structure_limit"
    value = "PRIVATE"
    for _ in range(35):
        value = [value]
    result, _ = build([call(), output(value=value)])
    assert result["records"][1]["payload_unknown_reason"] == "structure_limit"


@pytest.mark.parametrize("limit", ["events", "bytes"])
def test_bounded_prefix_preserves_observed_denominator_and_explicit_omissions(monkeypatch, limit):
    if limit == "events":
        monkeypatch.setattr(ledger, "MAX_LEDGER_EVENTS", 3)
    else:
        monkeypatch.setattr(ledger, "MAX_LEDGER_BYTES", 900)
    records = [row for i in range(12) for row in (call(str(i)), output(str(i)))]
    result, _ = build(records)
    assert result["observed_tool_events"] == 24
    assert result["retained_tool_events"] < 24 and result["ledger_truncated"] is True
    assert result["omitted_tool_events"] == 24 - len(result["records"])
    assert result["observation_census_complete"] is result["fingerprints_complete"] is False
    assert result["retained_rows_canonical_bytes"] == len(ledger._wire(result["records"]))


def test_aggregate_payload_budget_stops_measurement_before_further_payload_work(monkeypatch):
    monkeypatch.setattr(ledger, "MAX_MEASURED_PAYLOAD_BYTES", 5)
    records = [call(value="abc"), output(value="def"), call("second", value="ghi"), output("second", value="jkl")]
    telemetry = interaction.build_interaction_telemetry(records, completed=True, malformed=0)
    original = interaction._payload_size
    queried = []
    def measured(value):
        if type(value) is str:
            queried.append(value)
        return original(value)
    monkeypatch.setattr(interaction, "_payload_size", measured)
    result = ledger.build_native_context_delivery(records, telemetry=telemetry)
    assert queried == ["abc", "def"]
    assert result["measured_payload_bytes"] == 3
    assert result["payloads_without_fingerprint"] == 3
    assert all(r["payload_unknown_reason"] == "aggregate_payload_limit" for r in result["records"][1:])


@pytest.mark.parametrize("completed,malformed", [(False, 0), (True, 1), (True, 100001)])
def test_incomplete_rollout_and_malformed_gaps_remain_explicit(completed, malformed):
    result, telemetry = build([call(), output()], completed=completed, malformed=malformed)
    assert result["observation_census_complete"] is result["fingerprints_complete"] is False
    assert result["tool_observations_complete"] is telemetry["tool_observations_complete"] is False
    assert result["malformed_json_records"] == malformed


def test_record_bound_counts_only_same_existing_prefix(monkeypatch):
    monkeypatch.setattr(interaction, "MAX_RECORDS", 2)
    result, telemetry = build([call(), output(), call("extra"), output("extra")])
    assert result["observed_records"] == 2 and result["observed_tool_events"] == 2
    assert result["record_limit_exceeded"] is telemetry["record_limit_exceeded"] is True
    assert result["observation_census_complete"] is False


@pytest.mark.parametrize("record", [[], {"type": "response_item", "payload": []},
    {"type": "response_item", "payload": {"type": ["function_call"]}}])
def test_invalid_native_record_shapes_never_crash_or_become_tool_events(record):
    result, _ = build([call(), record, output()])
    assert result["observed_tool_events"] == 2
    assert result["observation_census_complete"] is False


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(observed_tool_events=0),
    lambda x: x.update(observed_tool_call_events=True),
    lambda x: x.update(owner_response_match_verified=True),
    lambda x: x.update(raw_content_exported=True),
    lambda x: x.update(raw_output="PRIVATE"),
    lambda x: x["records"][0].update(record_ordinal=True),
    lambda x: x["records"][0].update(event_kind=["function_call"]),
    lambda x: x["records"][0].update(payload_unknown_reason=[]),
    lambda x: x["records"][0].update(payload_bytes=True),
    lambda x: x["records"][0].update(tool_name="private.tool"),
    lambda x: x["records"][0].update(tool_name_sha256="0" * 64),
    lambda x: x["records"][0].update(pairing_status="executed"),
    lambda x: x["records"].reverse(),
    lambda x: x["payload_measurements"]["decoded_utf8_string"].update(output_events=0),
])
def test_resealed_inconsistent_ledger_and_authority_forgery_are_rejected(mutate):
    result, _ = build([call(), output()])
    mutate(result)
    with pytest.raises(ValueError):
        ledger.validate_native_context_delivery(reseal(result))


def test_resealed_scope_counter_swaps_cannot_mislabel_string_bytes_as_structured():
    result, _ = build([call(), output()])
    result["payload_measurements"]["canonical_json_utf8"] = result["payload_measurements"]["decoded_utf8_string"]
    result["payload_measurements"]["decoded_utf8_string"] = {"call_events": 0, "output_events": 0, "bytes": 0}
    with pytest.raises(ValueError, match="scope counters"):
        ledger.validate_native_context_delivery(reseal(result))


@pytest.mark.parametrize("mutate", [lambda x: x.update(schema="unknown"), lambda x: x.update(observed_records=0),
    lambda x: x.update(observed_tool_output_events=0), lambda x: x["tool_pairs"][0].update(first_record_ordinal=999),
    lambda x: x["tool_pairs"][0].update(pairing_status="executed")])
def test_overlay_requires_existing_same_record_census_and_pair_positions(mutate):
    records = [call(), output()]
    telemetry = interaction.build_interaction_telemetry(records, completed=True, malformed=0)
    mutate(telemetry)
    with pytest.raises(ValueError):
        ledger.build_native_context_delivery(records, telemetry=telemetry)


def test_native_usage_recovery_builds_telemetry_once_preserves_totals_and_hashes(tmp_path, monkeypatch):
    home = tmp_path / "home"; workspace = tmp_path / "work"; workspace.mkdir()
    folder = home / "sessions/2026/10/14"; folder.mkdir(parents=True)
    thread = "native-context-thread-12345"
    records = [{"type": "session_meta", "payload": {"id": thread, "cwd": str(workspace)}},
        call(), output(value="Unicode é PRIVATE"), token(), {"type": "event_msg", "payload": {"type": "task_complete"}}]
    path = folder / ("rollout-" + thread + ".jsonl")
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in records); path.write_bytes(raw)
    original = usage.build_interaction_telemetry; invocations = []
    def once(*args, **kwargs):
        invocations.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(usage, "build_interaction_telemetry", once)
    recovered = usage.recover_codex_usage(home=home, thread_id=thread, workspace=workspace)
    assert invocations == [1]
    assert recovered["usage"] == token()["payload"]["info"]["total_token_usage"]
    assert recovered["task_complete_observed"] is True
    assert recovered["rollout_sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert recovered["native_context_delivery"]["interaction_telemetry_sha256"] == ledger._sha(ledger._wire(recovered["interaction_telemetry"]))
    assert recovered["native_context_delivery"]["records"][1]["payload_sha256"] == hashlib.sha256("Unicode é PRIVATE".encode()).hexdigest()
    assert b"PRIVATE" not in ledger._wire(recovered["native_context_delivery"])


def test_empty_completed_observation_does_not_claim_delivery_or_source_authority():
    result, _ = build([])
    assert result["records"] == [] and result["observed_tool_events"] == 0
    assert result["observation_census_complete"] is True
    assert all(result[k] is False for k in ledger._FALSE_FIELDS)


def test_maximum_existing_pair_and_checkpoint_populations_are_supported():
    records = [row for i in range(interaction.MAX_TOOL_PAIRS) for row in (call(str(i), value="x"), output(str(i), value="y"))]
    records += [token(100 + i) for i in range(interaction.MAX_TOKEN_OBSERVATIONS)]
    result, telemetry = build(records)
    assert len(telemetry["tool_pairs"]) == interaction.MAX_TOOL_PAIRS
    assert len(telemetry["token_observations"]) == interaction.MAX_TOKEN_OBSERVATIONS
    assert result["schema"] == ledger.SCHEMA
    assert result["observed_tool_events"] == interaction.MAX_TOOL_PAIRS * 2


def test_optional_observer_failure_preserves_original_usage_and_completion(tmp_path, monkeypatch):
    home = tmp_path / "home"; workspace = tmp_path / "work"; workspace.mkdir()
    folder = home / "sessions/2026/10/14"; folder.mkdir(parents=True)
    thread = "observer-failure-thread-12345"
    records = [{"type": "session_meta", "payload": {"id": thread, "cwd": str(workspace)}},
        call(), output(), token(), {"type": "event_msg", "payload": {"type": "task_complete"}}]
    path = folder / ("rollout-" + thread + ".jsonl")
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in records); path.write_bytes(raw)
    baseline = usage.recover_codex_usage(home=home, thread_id=thread, workspace=workspace)
    def failed(*args, **kwargs):
        raise ValueError("PRIVATE observer error /secret/path")
    monkeypatch.setattr(usage, "build_native_context_delivery", failed)
    observed = usage.recover_codex_usage(home=home, thread_id=thread, workspace=workspace)
    assert {k: v for k, v in observed.items() if k != "native_context_delivery"} == {
        k: v for k, v in baseline.items() if k != "native_context_delivery"}
    fallback = ledger.validate_native_context_delivery(observed["native_context_delivery"])
    assert fallback["schema"] == ledger.UNAVAILABLE_SCHEMA and fallback["status"] == "unavailable"
    assert fallback["interaction_telemetry_sha256"] == ledger._sha(ledger._wire(observed["interaction_telemetry"]))
    assert b"PRIVATE" not in ledger._wire(fallback) and b"/secret/path" not in ledger._wire(fallback)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("mutate", [lambda x: x.update(reason="PRIVATE"), lambda x: x.update(provider_ingestion_verified=True),
    lambda x: x.update(observed_tool_events=0), lambda x: x.update(interaction_telemetry_sha256=True)])
def test_unavailable_variant_has_no_fabricated_counts_or_authority(mutate):
    unavailable = ledger.unavailable_native_context_delivery()
    mutate(unavailable)
    with pytest.raises(ValueError):
        ledger.validate_native_context_delivery(unavailable)
