"""Authored receipt populations; no providers, files, databases or tokenizers."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_supervisor_usage_accounting as usage


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _codex(identity="first", *, thread="native-session-123456789", inputs=100, output=20,
           cached=60, records=1, complete=True, artifact="snapshot-1"):
    return dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="codex_cli", model="gpt-6.1-sol", purpose="coding", status="provider_returned",
        workspace="/private/worktree", model_prompt_sha256=_sha("private prompt"), model_prompt_bytes=14,
        native_rollout_usage=dict(schema="native-codex-cumulative-usage@1", thread_id=thread,
            rollout_sha256=_sha(artifact), observed_token_count_records=records,
            usage=dict(input_tokens=inputs, output_tokens=output, total_tokens=inputs + output,
                       cached_input_tokens=cached, reasoning_output_tokens=output // 2),
            usage_available=True, task_complete_observed=complete, cache_included_in_input=True,
            billing_total_verified=False))


def _grok(identity="grok-first", *, complete=True):
    return dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="grok_cli", model="grok-4.7", purpose="coding", status="provider_returned",
        native_rollout_usage=dict(schema="native-grok-final-usage@1",
            usage=dict(input_tokens=10, cached_input_tokens=20, cache_write_input_tokens=5,
                       output_tokens=7, reasoning_output_tokens=3, total_tokens=42),
            usage_available=True, usage_observation_sha256=_sha("grok-envelope-" + identity),
            task_complete_observed=True, usage_complete_observed=complete,
            totals_scope="native_final_envelope", cache_included_in_input=False,
            observed_final_usage_records=1, billing_total_verified=False))


def _report(*receipts, **extra):
    return dict(provider_invocations=list(receipts), **extra)


def test_increasing_session_selects_latest_once_not_sum():
    early = _codex("early", complete=False)
    late = _codex("late", inputs=150, output=30, cached=90, records=2, artifact="snapshot-2")
    result = usage.summarize_supervisor_usage(_report(early, late, late))
    assert result["total_tokens"] == 180
    assert result["input_tokens"] == 150
    assert result["cached_input_tokens"] == 90
    assert result["provider_calls"] == 2
    assert result["supplied_receipt_count"] == 3
    assert result["duplicate_receipt_count"] == 1
    assert result["native_usage_accounting"]["native_session_count"] == 1
    assert result["harbor_usage_exportable"] is True
    assert result["observed_complete_sessions"] is False
    assert result["billing_total_verified"] is False


def test_equal_snapshot_different_invocation_keeps_census_but_charges_once():
    result = usage.summarize_supervisor_usage(_report(_codex("first"), _codex("second")))
    assert result["provider_calls"] == 2
    assert result["total_tokens"] == 120
    assert result["known_token_subtotals"]["total_tokens"] == 120


def test_independent_sessions_add_once_each():
    result = usage.summarize_supervisor_usage(_report(_codex(), _codex("other", thread="different-session-12345")))
    assert result["total_tokens"] == 240
    assert result["native_usage_accounting"]["native_session_count"] == 2


def test_grok_raw_and_normalized_cache_categories_are_distinct():
    result = usage.summarize_supervisor_usage(_report(_grok()))
    assert result["input_tokens"] == 10
    assert result["cached_input_tokens"] == 20
    assert result["cache_write_input_tokens"] == 5
    assert result["total_tokens"] == 42
    assert result["normalized_tokens"]["input_tokens_including_cache"] == 35
    assert result["cache_included_in_input"] is False
    assert result["harbor_usage_exportable"] is True


def test_mixed_providers_have_valid_total_but_no_ambiguous_raw_input_export():
    result = usage.summarize_supervisor_usage(_report(_codex(), _grok()))
    assert result["total_tokens"] == 162
    assert result["input_tokens"] is None
    assert result["known_token_subtotals"]["input_tokens"] is None
    assert result["normalized_tokens"]["input_tokens_including_cache"] == 135
    assert result["harbor_usage_exportable"] is False
    assert result["token_count_scope"] is None
    assert result["cache_included_in_input"] is None


@pytest.mark.parametrize("marker, expected", [(False, False), (True, True), (None, None),
    ({"phase": "coding"}, True), (["coding"], True)])
def test_missing_attempt_marker_preserves_three_valued_coverage(marker, expected):
    result = usage.summarize_supervisor_usage(_report(_codex(), unreceipted_provider_attempt=marker))
    assert result["unreceipted_provider_attempt"] is expected
    assert result["total_tokens"] == (120 if expected is False else None)
    assert result["known_token_subtotals"]["total_tokens"] == 120
    assert result["coverage"]["campaign_call_population_complete"] is None


def test_legacy_absent_marker_is_supplied_scope_only():
    result = usage.summarize_supervisor_usage(_report(_codex()))
    assert result["total_tokens"] == 120
    assert result["all_invocations_receipted"] is True
    assert result["coverage"]["campaign_call_population_complete"] is None


@pytest.mark.parametrize("marker", [0, 1, "false", "true", {}, []])
def test_malformed_missing_attempt_marker_cannot_be_coerced_to_false(marker):
    with pytest.raises(ValueError, match="invalid_unreceipted"):
        usage.summarize_supervisor_usage(_report(_codex(), unreceipted_provider_attempt=marker))


@pytest.mark.parametrize("key", ["thread_id", "rollout_sha256", "observed_token_count_records"])
def test_missing_session_custody_cannot_double_count_identified_snapshot(key):
    unidentified = _codex("unknown")
    del unidentified["native_rollout_usage"][key]
    result = usage.summarize_supervisor_usage(_report(_codex(), unidentified))
    assert result["total_tokens"] is None
    assert result["known_token_subtotals"]["total_tokens"] == 120
    assert result["known_normalized_counter_subtotals"]["total_tokens"] == 120
    assert result["provider_calls"] == 2
    assert len(result["unsupported_invocations"]) == 1
    assert result["unsupported_invocations"][0]["unqualified_native_counters"]["total_tokens"] == 120
    assert result["harbor_usage_exportable"] is False


@pytest.mark.parametrize("key", ["provider", "model"])
def test_missing_router_identity_is_retained_without_fabricating_defaults(key):
    incomplete = _codex("incomplete")
    del incomplete[key]
    result = usage.summarize_supervisor_usage(_report(_codex(), incomplete))
    assert result["total_tokens"] is None
    assert result["known_token_subtotals"]["total_tokens"] == 120
    assert result["provider_calls"] == 2
    assert result["unsupported_invocations"][0]["reason"] == "missing_or_invalid_" + key


def test_sparse_legacy_counts_are_retained_individually_never_summed():
    sparse = dict(invocation_id="legacy", native_rollout_usage=dict(
        usage=dict(input_tokens=100, cached_input_tokens=80, output_tokens=20, total_tokens=120)))
    result = usage.summarize_supervisor_usage(_report(sparse, {**sparse, "invocation_id": "legacy-two"}))
    assert result["known_token_subtotals"]["total_tokens"] is None
    assert result["total_tokens"] is None
    assert result["unsupported_invocations"][0]["unqualified_native_counters"]["total_tokens"] == 120


@pytest.mark.parametrize("value", [True, -1, 1.5, "100", None])
def test_malformed_native_counts_never_qualify(value):
    row = _codex()
    row["native_rollout_usage"]["usage"]["input_tokens"] = value
    result = usage.summarize_supervisor_usage(_report(row))
    assert result["total_tokens"] is None
    assert result["harbor_usage_exportable"] is False
    assert result["known_token_subtotals"]["total_tokens"] is None


def test_incoherent_native_counters_are_not_known_costs():
    row = _codex()
    row["native_rollout_usage"]["usage"]["total_tokens"] = 999
    result = usage.summarize_supervisor_usage(_report(row))
    assert result["native_usage_complete_observed"] is False
    assert result["known_token_subtotals"]["total_tokens"] is None


def test_failed_incomplete_session_preserves_observed_native_cost():
    row = _codex(complete=False)
    row["status"] = "failed"
    result = usage.summarize_supervisor_usage(_report(row))
    assert result["native_usage_complete_observed"] is False
    assert result["total_tokens"] is None
    assert result["known_token_subtotals"]["total_tokens"] == 120
    assert result["native_usage_accounting"]["failed_invocation_count"] == 1


def test_closed_session_without_counters_preserves_endpoint_event_only():
    row = _codex()
    row["native_rollout_usage"]["usage"] = {}
    row["native_rollout_usage"]["observed_token_count_records"] = 0
    result = usage.summarize_supervisor_usage(_report(row))
    assert result["observed_complete_sessions"] is True
    assert result["native_usage_complete_observed"] is None
    assert result["total_tokens"] is None


@pytest.mark.parametrize("change", ["model", "workspace", "artifact", "regression"])
def test_same_session_conflicts_withhold_selected_cost(change):
    first, second = _codex("first"), _codex("second")
    if change == "model":
        second["model"] = "other-model"
    elif change == "workspace":
        second["workspace"] = "/different/private/worktree"
    elif change == "artifact":
        second["native_rollout_usage"]["rollout_sha256"] = _sha("different")
    else:
        second["native_rollout_usage"]["usage"].update(input_tokens=90, total_tokens=110)
    result = usage.summarize_supervisor_usage(_report(first, second))
    assert result["total_tokens"] is None
    assert result["known_token_subtotals"]["total_tokens"] is None
    assert result["native_usage_complete_observed"] is False


def test_same_invocation_conflict_in_unknown_private_field_is_rejected():
    first, second = _codex(), _codex()
    first["unknown_private_payload"] = "first private source"
    second["unknown_private_payload"] = "second private source"
    with pytest.raises(ValueError, match="conflicting_provider_invocation"):
        usage.summarize_supervisor_usage(_report(first, second))


def test_boolean_and_integer_unknown_fields_do_not_compare_as_equal():
    first, second = _codex(), _codex()
    first["unknown"] = True
    second["unknown"] = 1
    with pytest.raises(ValueError, match="conflicting_provider_invocation"):
        usage.summarize_supervisor_usage(_report(first, second))


def test_summary_aggregation_reconciles_cross_trial_session():
    early = usage.summarize_supervisor_usage(_report(_codex("early", complete=False)))
    later = usage.summarize_supervisor_usage(_report(_codex("late", inputs=150, output=30,
        cached=90, records=2, artifact="snapshot-2")))
    combined = usage.summarize_supervisor_usage_summaries([early, later])
    assert combined["total_tokens"] == 180
    assert combined["provider_calls"] == 2
    assert combined["native_usage_accounting"]["native_session_count"] == 1


def test_report_aggregation_and_summary_aggregation_are_identical():
    reports = [_report(_codex()), _report(_grok())]
    assert usage.summarize_supervisor_usage_reports(reports) == usage.summarize_supervisor_usage_summaries(
        [usage.summarize_supervisor_usage(report) for report in reports])


def test_cross_summary_identical_invocation_is_deduplicated():
    summary = usage.summarize_supervisor_usage(_report(_codex()))
    result = usage.summarize_supervisor_usage_summaries([summary, deepcopy(summary)])
    assert result["total_tokens"] == 120
    assert result["provider_calls"] == 1
    assert result["duplicate_receipt_count"] == 1


def test_cross_summary_conflict_cannot_be_hidden_by_projected_receipt_hash():
    first, second = _codex(), _codex()
    second["unknown_private_field"] = "different"
    summaries = [usage.summarize_supervisor_usage(_report(row)) for row in (first, second)]
    with pytest.raises(ValueError, match="conflicting_provider_invocation"):
        usage.summarize_supervisor_usage_summaries(summaries)


def test_aggregate_unknown_and_unreceipted_markers_are_preserved():
    good = usage.summarize_supervisor_usage(_report(_codex()))
    unknown = usage.summarize_supervisor_usage(_report(unreceipted_provider_attempt=None))
    missing = usage.summarize_supervisor_usage(_report(unreceipted_provider_attempt={"phase": "failed"}))
    assert usage.summarize_supervisor_usage_summaries([good, unknown])["unreceipted_provider_attempt"] is None
    assert usage.summarize_supervisor_usage_summaries([good, unknown, missing])["unreceipted_provider_attempt"] is True
    assert usage.summarize_supervisor_usage_summaries([good, unknown])["total_tokens"] is None


def test_empty_trial_report_does_not_disappear_from_aggregate_coverage():
    good = usage.summarize_supervisor_usage(_report(_codex()))
    empty = usage.summarize_supervisor_usage({})
    result = usage.summarize_supervisor_usage_summaries([good, empty])
    assert result["unreceipted_provider_attempt"] is None
    assert result["total_tokens"] is None
    assert result["known_token_subtotals"]["total_tokens"] == 120
    assert result["harbor_usage_exportable"] is False


def test_summary_mutation_and_resealed_forged_totals_are_refused():
    summary = usage.summarize_supervisor_usage(_report(_codex()))
    summary["total_tokens"] = 1
    with pytest.raises(ValueError, match="pin_mismatch"):
        usage.summarize_supervisor_usage_summaries([summary])
    summary["summary_sha256"] = _sha(json.dumps({key: value for key, value in summary.items()
        if key != "summary_sha256"}, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    with pytest.raises(ValueError, match="regeneration_mismatch"):
        usage.summarize_supervisor_usage_summaries([summary])


def test_private_content_is_not_exported_and_output_is_detached():
    row = _codex()
    row["prompt"] = "secret-private-prompt"
    row["native_rollout_usage"]["arbitrary"] = "secret-private-native-source"
    result = usage.summarize_supervisor_usage(_report(row))
    wire = json.dumps(result)
    assert all(text not in wire for text in ("secret-private", "/private/", "private prompt"))
    row["native_rollout_usage"]["usage"]["total_tokens"] = 999
    assert result["total_tokens"] == 120
    assert result["raw_content_exported"] is False
    assert result["coverage"]["auxiliary_calls_complete"] is None


def test_aggregate_unqualified_counters_are_detached_from_input_summaries():
    row = _codex()
    del row["model"]
    summary = usage.summarize_supervisor_usage(_report(row))
    aggregate = usage.summarize_supervisor_usage_summaries([summary])
    summary["receipt_inventory"][0]["unqualified_native_counters"]["total_tokens"] = 999
    assert aggregate["receipt_inventory"][0]["unqualified_native_counters"]["total_tokens"] == 120
    assert aggregate["unsupported_invocations"][0]["unqualified_native_counters"]["total_tokens"] == 120


@pytest.mark.parametrize("report", [None, [], {"provider_invocations": None},
    {"provider_invocations": [{}]}, {"provider_invocations": [{"invocation_id": 1}]}])
def test_invalid_populations_are_rejected(report):
    with pytest.raises(ValueError):
        usage.summarize_supervisor_usage(report)


def test_bounds_are_enforced_before_serialization_and_large_population():
    with pytest.raises(ValueError, match="population_bound"):
        usage.summarize_supervisor_usage(_report(*[_codex()] * (usage.MAX_ROWS + 1)))
    row = _codex()
    row["unknown"] = "x" * (usage.MAX_ROW_BYTES + 1)
    with pytest.raises(ValueError, match="byte_bound"):
        usage.summarize_supervisor_usage(_report(row))
    row["unknown"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        usage.summarize_supervisor_usage(_report(row))


def test_empty_population_and_complete_zero_usage_are_distinct():
    empty = usage.summarize_supervisor_usage({})
    zero = usage.summarize_supervisor_usage(_report(_codex(inputs=0, output=0, cached=0)))
    assert empty["total_tokens"] is None
    assert zero["total_tokens"] == 0
    assert zero["harbor_usage_exportable"] is True
