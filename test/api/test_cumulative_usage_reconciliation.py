"""Accounting regressions using the retained router/native receipt schemas.

These authored receipts qualify reconciliation only, not provider usage,
billing, tokenizer compatibility, task completion, or model performance.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
    COMPLETION_BASIS,
    MAX_COUNTER,
    MAX_RECEIPTS,
    reconcile_router_cumulative_usage,
)

THREAD = "native-thread-123456789"
OTHER_THREAD = "native-thread-987654321"


def receipt(*, invocation="allocation-1", thread=THREAD, count=1,
            inputs=100, outputs=20, complete=True, legacy=False):
    """Use exact keys emitted by router_implementation_runner/recover_codex_usage."""
    native = {
        "schema": "native-codex-cumulative-usage@1", "thread_id": thread,
        "rollout_sha256": hashlib.sha256(f"{thread}:{count}:{inputs}:{outputs}".encode()).hexdigest(),
        "usage": {"input_tokens": inputs, "cached_input_tokens": inputs // 2,
                  "output_tokens": outputs, "reasoning_output_tokens": outputs // 2,
                  "total_tokens": inputs + outputs},
        "usage_available": True, "observed_token_count_records": count,
        "task_complete_observed": complete, "billing_total_verified": False,
        "cache_included_in_input": True, "dollar_cost": None,
    }
    if not legacy:
        native.update(completion_basis=COMPLETION_BASIS,
                      lifecycle_malformed_records=0, usage_lifecycle_complete=complete)
    return {
        "schema": "router-implementation-invocation@1", "invocation_id": invocation,
        "provider": "codex_cli", "model": "gpt-5.6-sol", "workspace": "/isolated/checkout",
        "status": "provider_returned", "phase": "coding", "router_calls": 1,
        "usage": {"thread_id": thread, "model_id": "gpt-5.6-sol"},
        "native_rollout_usage": native, "completion_authority": False,
    }


def one(row):
    result = reconcile_router_cumulative_usage([row])
    assert result["native_sessions_observed"] == 1
    return result, result["sessions"][0]


def test_complete_scope_is_input_plus_output_with_subsets_once():
    row = receipt()
    row["tool_tokens"] = 9000
    row["prompt_selection"] = {"selected_content_tokens": 8000}
    result, session = one(row)
    assert result["provider_total_tokens"] == 120
    assert result["known_provider_token_subtotal"] == 120
    assert result["known_counter_subtotals"]["cached_input_tokens"] == 50
    assert result["known_counter_subtotals"]["reasoning_output_tokens"] == 10
    assert session["status"] == "complete"
    assert result["supplied_receipt_usage_complete"] is True
    assert result["campaign_usage_complete"] is None
    for field in ("billing_total_verified", "execution_authority", "completion_authority",
                  "correctness_established", "tool_tokens_added"):
        assert result[field] is False


def test_exact_repeated_allocation_receipt_does_not_charge_again_or_mutate_input():
    row = receipt()
    original = deepcopy(row)
    result = reconcile_router_cumulative_usage([row, deepcopy(row)])
    assert result["provider_total_tokens"] == 120
    assert result["repeated_receipt_observations"] == 1
    assert result["unique_invocation_ids"] == result["native_sessions_observed"] == 1
    assert row == original


def test_distinct_allocations_retain_one_native_session_latest_cumulative_only():
    early = receipt(complete=False)
    later = receipt(invocation="allocation-2", count=3, inputs=160, outputs=40)
    result = reconcile_router_cumulative_usage([early, later])
    assert result["unique_invocation_ids"] == 2
    assert result["native_sessions_observed"] == 1
    assert result["provider_total_tokens"] == 200
    assert result["sessions"][0]["selected_observation_ordinal"] == 1


def test_fresh_retry_and_expansion_sessions_are_charged_even_when_failed():
    planning = receipt(invocation="planning")
    failed = receipt(invocation="retry", thread=OTHER_THREAD, inputs=70, outputs=10, complete=False)
    failed.update(status="failed", phase="expansion", error_type="TimeoutExpired")
    result = reconcile_router_cumulative_usage([planning, failed])
    assert result["native_sessions_observed"] == 2
    assert result["known_provider_token_subtotal"] == 200
    assert result["provider_total_tokens"] is None
    assert result["sessions"][1]["strict_usage_complete"] is False


def test_legacy_any_earlier_completion_flag_is_not_final_usage_completeness():
    result, session = one(receipt(legacy=True))
    assert session["producer_declared_complete"] is True
    assert session["strict_usage_complete"] is None
    assert session["status"] == "partial"
    assert session["known_provider_token_subtotal"] == 120
    assert "legacy_final_usage_completeness_unknown" in session["reasons"]
    assert result["provider_total_tokens"] is None


def test_latest_partial_snapshot_does_not_union_earlier_missing_categories():
    early = receipt(complete=False)
    late = receipt(invocation="allocation-2", count=2, inputs=130, outputs=25)
    late["native_rollout_usage"]["usage"].pop("output_tokens")
    result = reconcile_router_cumulative_usage([early, late])
    session = result["sessions"][0]
    assert session["known_counter_subtotals"]["input_tokens"] == 130
    assert session["known_counter_subtotals"]["output_tokens"] is None
    assert session["known_provider_token_subtotal"] == 130
    assert session["strict_usage_complete"] is None
    assert result["provider_total_tokens"] is None


@pytest.mark.parametrize("field", ["model", "workspace"])
def test_same_native_thread_changed_binding_is_one_quarantined_session(field):
    early, late = receipt(), receipt(invocation="allocation-2", count=2)
    late[field] = "other-model" if field == "model" else "/other/checkout"
    if field == "model":
        late["usage"]["model_id"] = late[field]
    result = reconcile_router_cumulative_usage([early, late])
    assert result["native_sessions_observed"] == 1
    assert result["sessions"][0]["status"] == "conflict"
    assert "native_session_binding_changed" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


@pytest.mark.parametrize("mutation", ["native_thread", "outer_session", "model", "provider"])
def test_outer_native_identity_conflicts_are_not_usage(mutation):
    row = receipt()
    if mutation == "native_thread":
        row["native_rollout_usage"]["thread_id"] = OTHER_THREAD
    elif mutation == "outer_session":
        row["usage"]["session_id"] = OTHER_THREAD
    elif mutation == "model":
        row["usage"]["model_id"] = "different-model"
    else:
        row["provider"] = "grok_cli"
    result, session = one(row)
    assert session["status"] == "conflict"
    assert result["known_provider_token_subtotal"] is None


def test_documented_outer_session_fallback_binds_native_thread():
    row = receipt()
    row["usage"]["session_id"] = row["usage"].pop("thread_id")
    assert one(row)[0]["provider_total_tokens"] == 120


@pytest.mark.parametrize("value", [True, -1, 1.5, "100", MAX_COUNTER + 1])
def test_invalid_counters_quarantine_instead_of_coercion(value):
    row = receipt()
    row["native_rollout_usage"]["usage"]["input_tokens"] = value
    result, session = one(row)
    assert "invalid_native_counter" in session["reasons"]
    assert result["known_provider_token_subtotal"] is None


@pytest.mark.parametrize("field,value,reason", [
    ("cached_input_tokens", 101, "subset_exceeds_provider_counter"),
    ("reasoning_output_tokens", 21, "subset_exceeds_provider_counter"),
    ("total_tokens", 999, "reported_provider_total_conflict"),
])
def test_subset_and_reported_total_conflicts(field, value, reason):
    row = receipt()
    row["native_rollout_usage"]["usage"][field] = value
    result, session = one(row)
    assert reason in session["reasons"]
    assert result["known_provider_token_subtotal"] is None


@pytest.mark.parametrize("field,value", [("usage_available", False),
                                         ("observed_token_count_records", True),
                                         ("rollout_sha256", "not-a-hash"),
                                         ("cache_included_in_input", False)])
def test_inconsistent_native_accounting_metadata_is_quarantined(field, value):
    row = receipt()
    row["native_rollout_usage"][field] = value
    result, session = one(row)
    assert session["status"] == "conflict"
    assert result["known_provider_token_subtotal"] is None


@pytest.mark.parametrize("field,value", [("lifecycle_malformed_records", 1),
                                         ("lifecycle_malformed_records", False),
                                         ("task_complete_observed", False),
                                         ("usage_lifecycle_complete", "true")])
def test_new_completion_metadata_must_be_consistent(field, value):
    row = receipt()
    row["native_rollout_usage"][field] = value
    assert one(row)[1]["status"] == "conflict"


def test_unsupported_completion_basis_preserves_only_observed_subtotal():
    row = receipt()
    row["native_rollout_usage"]["completion_basis"] = "unknown-basis@1"
    result, session = one(row)
    assert session["strict_usage_complete"] is None
    assert "unsupported_completion_basis" in session["reasons"]
    assert result["known_provider_token_subtotal"] == 120
    assert result["provider_total_tokens"] is None


def test_no_native_usage_remains_unknown_without_fabricated_zero():
    row = receipt()
    row["native_rollout_usage"] = None
    result = reconcile_router_cumulative_usage([row])
    assert result["known_provider_token_subtotal"] is None
    assert all(value is None for value in result["known_counter_subtotals"].values())
    assert result["issues"][0]["reason"] == "native_usage_unavailable"


def test_native_receipt_without_any_token_record_is_unavailable():
    row = receipt(complete=False)
    native = row["native_rollout_usage"]
    native.update(usage={}, usage_available=False, observed_token_count_records=0)
    result, session = one(row)
    assert session["status"] == "unavailable"
    assert session["known_provider_token_subtotal"] is None
    assert result["provider_total_tokens"] is None


def test_unsupported_grok_cache_convention_is_not_normalized_as_codex():
    row = receipt()
    row.update(provider="grok_cli")
    row["native_rollout_usage"] = {
        "schema": "native-grok-final-usage@1", "usage": {
            "input_tokens": 9396, "cached_input_tokens": 2048,
            "output_tokens": 3894, "reasoning_output_tokens": 3866, "total_tokens": 15338},
        "cache_included_in_input": False, "task_complete_observed": True,
    }
    result = reconcile_router_cumulative_usage([row])
    assert result["issues"][0]["reason"] == "unsupported_native_usage_schema"
    assert result["known_provider_token_subtotal"] is None


def test_same_count_different_values_conflicts_instead_of_selecting_max():
    early, late = receipt(), receipt(invocation="allocation-2", inputs=130)
    result = reconcile_router_cumulative_usage([early, late])
    assert "same_record_count_counter_conflict" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_count_regression_in_explicit_caller_order_is_not_invented_chronology():
    early = receipt(count=5)
    late = receipt(invocation="allocation-2", count=4, inputs=150)
    result = reconcile_router_cumulative_usage([early, late])
    assert "cumulative_record_count_regression" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_increasing_count_with_counter_regression_is_quarantined():
    early = receipt(count=2)
    late = receipt(invocation="allocation-2", count=3, inputs=80)
    result = reconcile_router_cumulative_usage([early, late])
    assert "cumulative_counter_regression" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_regression_detected_across_missing_intermediate_category_without_union():
    early = receipt(count=1)
    middle = receipt(invocation="allocation-2", count=2, inputs=120)
    middle["native_rollout_usage"]["usage"].pop("input_tokens")
    late = receipt(invocation="allocation-3", count=3, inputs=90)
    result = reconcile_router_cumulative_usage([early, middle, late])
    assert "cumulative_counter_regression" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_conflicting_allocation_alias_quarantines_both_native_threads():
    result = reconcile_router_cumulative_usage([receipt(), receipt(thread=OTHER_THREAD)])
    assert result["native_sessions_observed"] == 2
    assert all("invocation_identity_alias_conflict" in row["reasons"] for row in result["sessions"])
    assert result["known_provider_token_subtotal"] is None


@pytest.mark.parametrize("field", ["native_prompt_sha256", "model_prompt_sha256"])
def test_same_allocation_changed_source_input_digest_is_not_exact_replay(field):
    early = receipt()
    early[field] = "a" * 64
    late = deepcopy(early)
    late[field] = "b" * 64
    result = reconcile_router_cumulative_usage([early, late])
    assert "invocation_identity_alias_conflict" in result["sessions"][0]["reasons"]
    assert result["repeated_receipt_observations"] == 0
    assert result["known_provider_token_subtotal"] is None


@pytest.mark.parametrize("field,value", [("status", "failed"), ("phase", "planning")])
def test_same_allocation_changed_lifecycle_binding_is_not_exact_replay(field, value):
    early = receipt()
    late = deepcopy(early)
    late[field] = value
    result = reconcile_router_cumulative_usage([early, late])
    assert "invocation_identity_alias_conflict" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_same_rollout_bytes_cannot_assert_changed_final_completion():
    early = receipt()
    late = receipt(invocation="allocation-2", complete=False)
    assert late["native_rollout_usage"]["rollout_sha256"] == early["native_rollout_usage"]["rollout_sha256"]
    result = reconcile_router_cumulative_usage([early, late])
    assert "rollout_digest_alias_conflict" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_malformed_source_digest_is_not_a_provider_binding():
    row = receipt()
    row["model_prompt_sha256"] = "arbitrary-private-value"
    result, session = one(row)
    assert "invalid_prompt_binding_digest" in session["reasons"]
    assert "arbitrary-private-value" not in json.dumps(result)


def test_same_rollout_digest_conflicting_cardinality_quarantines_all_evidence():
    early = receipt()
    late = receipt(invocation="allocation-2", count=2, inputs=150)
    late["native_rollout_usage"]["rollout_sha256"] = early["native_rollout_usage"]["rollout_sha256"]
    result = reconcile_router_cumulative_usage([early, late])
    assert "rollout_digest_alias_conflict" in result["sessions"][0]["reasons"]
    assert result["known_provider_token_subtotal"] is None


def test_conflicted_session_excluded_while_other_genuine_session_remains_charged():
    bad = receipt(invocation="bad")
    bad["native_rollout_usage"]["usage"]["total_tokens"] = 1
    result = reconcile_router_cumulative_usage([bad, receipt(invocation="good", thread=OTHER_THREAD)])
    assert result["known_provider_token_subtotal"] == 120
    assert result["provider_total_tokens"] is None


def test_missing_receipt_prevents_complete_supplied_population():
    unavailable = receipt(invocation="missing")
    unavailable["native_rollout_usage"] = None
    result = reconcile_router_cumulative_usage([receipt(), unavailable])
    assert result["known_provider_token_subtotal"] == 120
    assert result["provider_total_tokens"] is None
    assert result["supplied_receipt_usage_complete"] is False


def test_workspace_paths_and_arbitrary_source_prompt_or_credentials_are_not_exported():
    row = receipt()
    row.update(prompt="DO-NOT-EXPORT-PROMPT", credentials="DO-NOT-EXPORT-CREDENTIAL")
    row["usage"]["raw_tool_text"] = "DO-NOT-EXPORT-TOOL"
    result, session = one(row)
    encoded = json.dumps(result)
    assert "DO-NOT-EXPORT" not in encoded
    assert row["workspace"] not in encoded
    assert session["workspace_binding_sha256"] == hashlib.sha256(row["workspace"].encode()).hexdigest()


def test_empty_population_keeps_totals_unknown():
    result = reconcile_router_cumulative_usage([])
    assert result["known_provider_token_subtotal"] is None
    assert result["provider_total_tokens"] is None
    assert result["campaign_usage_complete"] is None


def test_population_bound_and_explicit_order_required():
    with pytest.raises(ValueError, match="bounded ordered"):
        reconcile_router_cumulative_usage(iter([receipt()]))
    with pytest.raises(ValueError, match="bounded ordered"):
        reconcile_router_cumulative_usage([None] * (MAX_RECEIPTS + 1))


def test_snapshot_evidence_is_bounded_without_changing_latest_charge():
    rows = [receipt(invocation=f"allocation-{index}", count=index, inputs=100 + index)
            for index in range(1, 21)]
    result = reconcile_router_cumulative_usage(rows)
    session = result["sessions"][0]
    assert len(session["snapshot_evidence"]) == 16
    assert session["snapshot_evidence_omitted_count"] == 4
    assert result["provider_total_tokens"] == 140


def test_retained_headless_pilot_categories_have_lower_bound_391436():
    """Exact sanitized categories from the retained two-session pilot, not a new run."""
    planning = receipt(invocation="planning", count=1, inputs=19748, outputs=1131, legacy=True)
    coding = receipt(invocation="coding", thread=OTHER_THREAD, count=10,
                     inputs=360004, outputs=10553, legacy=True)
    result = reconcile_router_cumulative_usage([planning, coding])
    assert result["known_counter_subtotals"]["input_tokens"] == 379752
    assert result["known_counter_subtotals"]["output_tokens"] == 11684
    assert result["known_provider_token_subtotal"] == 391436
    assert result["provider_total_tokens"] is None
    assert result["supplied_receipt_usage_complete"] is False
