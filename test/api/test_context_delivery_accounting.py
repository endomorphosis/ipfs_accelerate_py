"""Native session accounting and bounded context evidence, without model calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as accounting


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _receipt(invocation="first", *, provider="codex_cli", thread="native-session-123456789",
             inputs=100, outputs=20, cached=60, records=1, complete=True, artifact="snapshot-1"):
    native = dict(schema="native-codex-cumulative-usage@1", thread_id=thread,
        rollout_sha256=_sha(artifact), observed_token_count_records=records,
        usage={"input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs,
               "cached_input_tokens": cached, "reasoning_output_tokens": outputs // 2},
        usage_available=True, task_complete_observed=complete, cache_included_in_input=True,
        billing_total_verified=False)
    return dict(schema="router-implementation-invocation@1", invocation_id=invocation,
        provider=provider, model="gpt-6.1-sol", purpose="coding", status="provider_returned",
        workspace="/private/allocated/worktree", model_prompt_sha256=_sha("prompt"),
        model_prompt_bytes=6, native_rollout_usage=native)


def _grok(invocation="grok-first", *, complete=True):
    row = _receipt(invocation, provider="grok_cli")
    row["model"] = "grok-4.7"
    row["native_rollout_usage"] = dict(schema="native-grok-final-usage@1",
        usage={"input_tokens": 10, "cached_input_tokens": 20, "cache_write_input_tokens": 5,
               "output_tokens": 7, "reasoning_output_tokens": 3, "total_tokens": 42},
        usage_available=True, usage_observation_sha256=_sha("grok-envelope"),
        task_complete_observed=True, usage_complete_observed=complete,
        totals_scope="native_final_envelope", cache_included_in_input=False,
        observed_final_usage_records=1, billing_total_verified=False, raw_provider_data_exported=False)
    return row


def _summary(*receipts, marker=False):
    return accounting.summarize_invocations([accounting.observe_invocation(r) for r in receipts],
                                           unreceipted_provider_attempt=marker)


def _reseal(row):
    body = {k: v for k, v in row.items() if k != "observation_sha256"}
    row["observation_sha256"] = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ledger_fixture():
    from ipfs_accelerate_py.agent_supervisor.runtime.codex_interaction_telemetry import build_interaction_telemetry
    from ipfs_accelerate_py.agent_supervisor.runtime.native_context_delivery import build_native_context_delivery
    records = [
        {"type": "session_meta", "payload": {"id": "native-session-123456789"}},
        {"type": "response_item", "payload": {"type": "function_call", "call_id": "private-call-id",
         "name": "exec_command", "arguments": '{"cmd":"read /private/source.py"}'}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "private-call-id",
         "output": "secret source Ω"}},
    ]
    telemetry = build_interaction_telemetry(records, completed=True, malformed=0)
    return telemetry, build_native_context_delivery(records, telemetry=telemetry)


def test_observation_retains_bound_prompt_and_status_without_private_content():
    row = _receipt()
    row.update(prompt="secret prompt", error="secret failure", arbitrary={"raw_source": "secret source"})
    observed = accounting.observe_invocation(row)
    wire = json.dumps(observed)
    assert all(text not in wire for text in ("secret prompt", "secret failure", "secret source", "/private/"))
    assert observed["status"] == "provider_returned"
    assert observed["model_prompt"] == {"sha256": _sha("prompt"), "bytes": 6,
                                        "count_scope": "rendered_user_prompt_only"}
    assert observed["invocation_id_sha256"] == _sha("first")
    assert observed["native_usage"]["complete"] is True
    assert observed["native_context_delivery"]["status"] == "unavailable"
    assert observed["raw_content_exported"] is False
    assert observed["provider_ingestion_verified"] is False


def test_projection_pin_ignores_unknown_raw_fields_and_input_is_detached():
    receipt = _receipt()
    original = accounting.observe_invocation(receipt)
    receipt["unknown"] = object()
    assert accounting.observe_invocation(receipt) == original
    receipt["native_rollout_usage"]["usage"]["input_tokens"] = 999
    assert original["native_usage"]["counts"]["input_tokens"] == 100


def test_codex_cache_and_reasoning_are_subsets_and_fresh_split_is_unknown():
    result = _summary(_receipt())
    counts = result["complete_native_counts"]
    assert result["complete_native_total_tokens"] == 120
    assert counts["input_tokens_including_cache"] == 100
    assert counts["cached_input_tokens"] == 60
    assert counts["output_tokens"] == 20
    assert counts["reasoning_output_tokens"] == 10
    assert counts["fresh_input_tokens"] is None
    assert counts["cache_write_input_tokens"] is None
    assert result["delivery_bytes_added_to_native_tokens"] is False


def test_optional_codex_cache_write_does_not_invent_disjointness():
    receipt = _receipt(cached=80)
    receipt["native_rollout_usage"]["usage"]["cache_write_input_tokens"] = 70
    result = _summary(receipt)
    assert result["complete_native_total_tokens"] == 120
    assert result["complete_native_counts"]["fresh_input_tokens"] is None
    assert result["complete_native_counts"]["cache_write_input_tokens"] == 70


def test_grok_disjoint_input_categories_are_counted_once():
    result = _summary(_grok())
    assert result["complete_native_total_tokens"] == 42
    assert result["complete_native_counts"]["input_tokens_including_cache"] == 35
    assert result["complete_native_counts"]["fresh_input_tokens"] == 10
    assert result["native_session_count"] == 0
    assert result["invocation_envelope_count"] == 1
    assert result["campaign_call_population_complete"] is None
    assert result["billing_total_verified"] is False


@pytest.mark.parametrize("marker", [None, True])
def test_missing_attempt_information_withholds_complete_totals(marker):
    result = _summary(_receipt(), marker=marker)
    assert result["reported_native_token_subtotal"] == 120
    assert result["complete_native_total_tokens"] is None
    assert result["complete_supplied_native_usage"] is False


@pytest.mark.parametrize("field,value", [
    ("thread_id", None), ("rollout_sha256", None),
    ("observed_token_count_records", None), ("observed_token_count_records", 0),
])
def test_unbound_codex_snapshot_does_not_inflate_known_session_subtotal(field, value):
    valid = _receipt("identified")
    unbound = _receipt("unbound", thread="other-native-session-123456")
    unbound["native_rollout_usage"][field] = value
    result = _summary(valid, unbound)
    assert result["reported_native_token_subtotal"] == 120
    assert result["complete_native_total_tokens"] is None
    # The literal unbound observation is retained for diagnosis; its counters
    # cannot be attributed as an additional disjoint session charge.
    assert result["observations"][1]["native_usage"]["counts"]["total_tokens"] == 120


@pytest.mark.parametrize("field,value", [
    ("usage_observation_sha256", None),
    ("observed_final_usage_records", None), ("observed_final_usage_records", 0),
])
def test_unbound_grok_envelope_does_not_inflate_known_subtotal(field, value):
    valid = _grok("identified")
    unbound = _grok("unbound")
    unbound["native_rollout_usage"][field] = value
    result = _summary(valid, unbound)
    assert result["reported_native_token_subtotal"] == 42
    assert result["complete_native_total_tokens"] is None
    assert result["observations"][1]["native_usage"]["counts"]["total_tokens"] == 42


def test_resealed_grok_projection_cannot_invent_a_second_final_envelope_record():
    observed = accounting.observe_invocation(_grok())
    observed["native_usage"]["observed_records"] = 2
    _reseal(observed)
    with pytest.raises(ValueError, match="invalid_observed_records"):
        accounting.summarize_invocations([observed], unreceipted_provider_attempt=False)


def test_resealed_unsupported_native_schema_cannot_claim_complete_usage():
    observed = accounting.observe_invocation(_grok())
    native = observed["native_usage"]
    native.update(schema=None, scope=None, identity_kind=None)
    native["normalized_counts"] = accounting._normalized(None, native["counts"])
    _reseal(observed)
    with pytest.raises(ValueError, match="invalid_complete_native_projection"):
        accounting.summarize_invocations([observed], unreceipted_provider_attempt=False)


def test_duplicate_observation_keeps_physical_population_without_extra_charge():
    result = _summary(_receipt(), _receipt())
    assert result["supplied_observation_count"] == 2
    assert len(result["observations"]) == 2
    assert result["duplicate_observation_count"] == 1
    assert result["supplied_invocation_count"] == 1
    assert result["complete_native_total_tokens"] == 120


def test_increasing_cumulative_session_selects_latest_instead_of_summing():
    early = _receipt("early", complete=False)
    later = _receipt("later", inputs=150, outputs=30, cached=90, records=2, artifact="snapshot-2")
    result = _summary(early, later)
    assert result["complete_native_total_tokens"] == 180
    assert result["native_session_count"] == 1
    assert result["sessions"][0]["observation_indices"] == [0, 1]
    assert result["sessions"][0]["selected_observation_index"] == 1
    assert result["cumulative_snapshots_are_additive"] is False


def test_equal_snapshot_across_different_invocations_is_counted_once():
    result = _summary(_receipt("first"), _receipt("second"))
    assert result["supplied_invocation_count"] == 2
    assert result["native_session_count"] == 1
    assert result["complete_native_total_tokens"] == 120


def test_fresh_failed_retry_is_retained_and_prevents_complete_population_total():
    failed = _receipt("retry", thread="different-session-123456789", complete=False, inputs=70, outputs=10)
    failed["status"] = "failed"
    failed["error_type"] = "TimeoutError"
    result = _summary(_receipt(), failed)
    assert result["failed_invocation_count"] == 1
    assert result["reported_native_token_subtotal"] == 200
    assert result["complete_native_total_tokens"] is None
    assert result["observations"][1]["status"] == "failed"


def test_postprocessing_failure_does_not_erase_complete_native_usage():
    failed = _receipt()
    failed["status"] = "failed"
    failed["error_type"] = "ValueError"
    result = _summary(failed)
    assert result["failed_invocation_count"] == 1
    assert result["complete_native_total_tokens"] == 120


@pytest.mark.parametrize("change,reason", [
    ("counter", "native_snapshot_counter_regression"),
    ("records", "native_snapshot_record_regression"),
    ("equal", "conflicting_equal_record_snapshots"),
    ("artifact", "native_artifact_counter_conflict"),
    ("model", "native_session_model_conflict"),
    ("workspace", "native_session_workspace_conflict"),
])
def test_conflicting_cumulative_snapshots_are_quarantined(change, reason):
    early = _receipt("early", inputs=100, outputs=20, records=2, artifact="early")
    later = _receipt("later", inputs=150, outputs=30, records=3, artifact="later")
    if change == "counter":
        later["native_rollout_usage"]["usage"].update(input_tokens=90, total_tokens=120)
    elif change == "records":
        later["native_rollout_usage"]["observed_token_count_records"] = 1
    elif change == "equal":
        later["native_rollout_usage"]["observed_token_count_records"] = 2
    elif change == "artifact":
        later["native_rollout_usage"]["rollout_sha256"] = _sha("early")
    elif change == "model":
        later["model"] = "gpt-other"
    elif change == "workspace":
        later["workspace"] = "/another/worktree"
    result = _summary(early, later)
    assert result["sessions"][0]["status"] == "conflict"
    assert reason in result["sessions"][0]["reasons"]
    assert result["sessions"][0]["selected_observation_index"] is None
    assert result["complete_native_total_tokens"] is None
    assert result["reported_native_token_subtotal"] is None


def test_conflicting_reuse_of_invocation_id_cannot_create_two_charges():
    changed = _receipt(thread="another-native-session-123456789")
    result = _summary(_receipt(), changed)
    assert result["conflicting_invocation_count"] == 1
    assert all(s["status"] == "conflict" for s in result["sessions"])
    assert result["reported_native_token_subtotal"] is None


def test_grok_equal_counts_do_not_deduplicate_distinct_envelopes_by_text():
    result = _summary(_grok("one"), _grok("two"))
    assert result["complete_native_total_tokens"] == 84
    assert result["invocation_envelope_count"] == 2


@pytest.mark.parametrize("schema", ["grok", "codex"])
def test_missing_native_completeness_remains_unknown(schema):
    receipt = _grok() if schema == "grok" else _receipt()
    native = receipt["native_rollout_usage"]
    native.pop("usage_complete_observed" if schema == "grok" else "task_complete_observed")
    result = _summary(receipt)
    assert result["complete_native_total_tokens"] is None
    assert result["reported_native_token_subtotal"] == (42 if schema == "grok" else 120)
    assert result["observations"][0]["native_usage"]["complete"] is None


def test_partial_native_categories_keep_nulls_without_zero_imputation():
    receipt = _grok()
    del receipt["native_rollout_usage"]["usage"]["cache_write_input_tokens"]
    observed = accounting.observe_invocation(receipt)
    assert observed["native_usage"]["counts"]["cache_write_input_tokens"] is None
    assert observed["native_usage"]["normalized_counts"]["input_tokens_including_cache"] is None
    result = accounting.summarize_invocations([observed], False)
    assert result["complete_native_total_tokens"] is None
    assert result["reported_native_token_subtotal"] is None


@pytest.mark.parametrize("change", ["cached", "reasoning", "total", "provider", "cache_semantics", "latest"])
def test_invalid_native_evidence_cannot_emit_normalized_negative_or_complete_totals(change):
    receipt = _receipt()
    native = receipt["native_rollout_usage"]
    if change == "cached":
        native["usage"]["cached_input_tokens"] = 101
    elif change == "reasoning":
        native["usage"]["reasoning_output_tokens"] = 21
    elif change == "total":
        native["usage"]["total_tokens"] = 121
    elif change == "provider":
        receipt["provider"] = "grok_cli"
    elif change == "cache_semantics":
        native["cache_included_in_input"] = False
    else:
        native["latest_usage_snapshot_valid"] = False
    observed = accounting.observe_invocation(receipt)
    assert observed["native_usage"]["status"] == "invalid"
    assert all(v is None or v >= 0 for v in observed["native_usage"]["normalized_counts"].values())
    result = accounting.summarize_invocations([observed], False)
    assert result["complete_native_total_tokens"] is None
    assert result["reported_native_token_subtotal"] is None


def test_generic_router_counts_never_substitute_for_missing_native_usage():
    receipt = _receipt()
    receipt.pop("native_rollout_usage")
    receipt["usage"] = {"total_tokens": 999}
    result = _summary(receipt)
    assert result["reported_native_token_subtotal"] is None
    assert result["complete_native_total_tokens"] is None


def test_unknown_native_schema_is_not_an_arbitrary_projection():
    receipt = _receipt()
    receipt["native_rollout_usage"] = {"schema": "unknown-native@1", "usage": {"total_tokens": 999},
                                      "source": "private-source"}
    observed = accounting.observe_invocation(receipt)
    assert observed["native_usage"]["status"] == "unsupported"
    assert "private-source" not in json.dumps(observed)
    assert _summary(receipt)["complete_native_total_tokens"] is None


@pytest.mark.parametrize("field,value", [
    ("input_tokens", True), ("input_tokens", -1), ("total_tokens", 1 << 63),
    ("output_tokens", float("nan")), ("input_tokens", "100"),
])
def test_native_counter_types_and_bounds_refuse_before_publication(field, value):
    receipt = _receipt()
    receipt["native_rollout_usage"]["usage"][field] = value
    with pytest.raises(accounting.ContextDeliveryAccountingError):
        accounting.observe_invocation(receipt)


@pytest.mark.parametrize("field,value", [
    ("invocation_id", " first"), ("model_prompt_sha256", "ABC"), ("model_prompt_bytes", True),
    ("model_prompt_bytes", 256001), ("workspace", []), ("provider", None),
])
def test_declared_bindings_are_exact_and_strict(field, value):
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(accounting.ContextDeliveryAccountingError):
        accounting.observe_invocation(receipt)


def test_unbound_prompt_part_is_refused():
    receipt = _receipt()
    receipt.pop("model_prompt_sha256")
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="prompt_binding"):
        accounting.observe_invocation(receipt)


def test_empty_population_is_unavailable_not_zero_usage():
    result = accounting.summarize_invocations([], False)
    assert result["complete_native_total_tokens"] is None
    assert result["reported_native_token_subtotal"] is None
    assert result["supplied_observation_count"] == 0


@pytest.mark.parametrize("marker", [0, 1, "false", {}])
def test_missing_attempt_marker_requires_boolean_or_unknown(marker):
    with pytest.raises(accounting.ContextDeliveryAccountingError):
        accounting.summarize_invocations([], marker)


def test_resealed_arbitrary_projection_and_authority_inflation_are_refused():
    observed = accounting.observe_invocation(_receipt())
    observed["native_usage"]["normalized_counts"]["total_tokens"] = 999
    _reseal(observed)
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="normalized"):
        accounting.summarize_invocations([observed], False)
    observed = accounting.observe_invocation(_receipt())
    observed["provider_ingestion_verified"] = True
    _reseal(observed)
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="authority"):
        accounting.summarize_invocations([observed], False)


def test_summary_returns_detached_observations():
    observed = accounting.observe_invocation(_receipt())
    result = accounting.summarize_invocations([observed], False)
    observed["native_usage"]["counts"]["input_tokens"] = 999
    assert result["observations"][0]["native_usage"]["counts"]["input_tokens"] == 100


def test_population_and_pre_snapshot_input_bounds(monkeypatch):
    observed = accounting.observe_invocation(_receipt())
    monkeypatch.setattr(accounting, "MAX_INVOCATIONS", 1)
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="population_bound"):
        accounting.summarize_invocations([observed, observed], False)
    monkeypatch.setattr(accounting, "MAX_AGGREGATE_OBSERVATION_BYTES", 10)
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="aggregate_observation_bound"):
        accounting.summarize_invocations([observed], False)


def test_supported_nested_receipt_bound_precedes_copy(monkeypatch):
    receipt = _receipt()
    receipt["native_rollout_usage"]["unknown"] = "x" * 1000
    monkeypatch.setattr(accounting, "MAX_INPUT_BYTES", 100)
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="byte_bound"):
        accounting.observe_invocation(receipt)


def test_aggregate_counter_overflow_refuses_instead_of_wrapping():
    a = _receipt("one", thread="first-native-session-123456789", inputs=accounting.MAX_COUNTER - 1,
                 outputs=1, cached=0)
    b = _receipt("two", thread="second-native-session-123456789")
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="aggregate_counter_bound"):
        _summary(a, b)


def test_real_native_ledger_projects_only_hash_count_coverage_without_event_copies():
    telemetry, ledger = _ledger_fixture()
    receipt = _receipt()
    receipt["native_rollout_usage"].update(interaction_telemetry=telemetry, native_context_delivery=ledger)
    observed = accounting.observe_invocation(receipt)
    projection = observed["native_context_delivery"]
    assert projection["status"] == "observed"
    assert projection["counts"]["observed_tool_events"] == 2
    assert projection["telemetry_binding_verified"] is True
    assert projection["coverage"]["fingerprints_complete"] is True
    assert all(text not in json.dumps(observed) for text in ("secret source", "private-call-id", "source.py"))
    assert "records" not in projection
    assert "ledger" not in projection
    assert accounting.summarize_invocations([observed], False)["complete_native_total_tokens"] == 120


def test_dual_ledger_aliases_require_exact_equality():
    telemetry, ledger = _ledger_fixture()
    receipt = _receipt()
    receipt["native_context_delivery"] = deepcopy(ledger)
    receipt["native_rollout_usage"].update(interaction_telemetry=telemetry, native_context_delivery=deepcopy(ledger))
    assert accounting.observe_invocation(receipt)["native_context_delivery"]["status"] == "observed"
    receipt["native_context_delivery"]["records"].pop()
    with pytest.raises(accounting.ContextDeliveryAccountingError, match="alias_conflict"):
        accounting.observe_invocation(receipt)


def test_native_ledger_tamper_is_refused_even_with_valid_native_token_counts():
    _, ledger = _ledger_fixture()
    receipt = _receipt()
    receipt["native_context_delivery"] = ledger
    ledger["records"][0]["payload_bytes"] += 1
    with pytest.raises(ValueError):
        accounting.observe_invocation(receipt)


def test_unavailable_optional_ledger_preserves_native_usage():
    from ipfs_accelerate_py.agent_supervisor.runtime.native_context_delivery import unavailable_native_context_delivery
    receipt = _receipt()
    receipt["native_context_delivery"] = unavailable_native_context_delivery(None)
    observed = accounting.observe_invocation(receipt)
    assert observed["native_context_delivery"]["status"] == "unavailable"
    assert observed["native_context_delivery"]["counts"] is None
    assert accounting.summarize_invocations([observed], False)["complete_native_total_tokens"] == 120
