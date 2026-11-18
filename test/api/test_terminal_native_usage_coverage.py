"""Authored native receipts qualify adapter accounting, never provider savings."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import native_usage_accounting as scopes
from benchmarks.agent_supervisor.container_coding import terminal_decision_usage_projection as projection
from ipfs_accelerate_py.agent_supervisor.runtime import cumulative_usage_reconciliation as cumulative


def _codex(identity="first", *, thread="authored-native-session-123456789", records=1,
           inputs=100, outputs=20, complete=True, legacy=False):
    native = dict(schema=cumulative.NATIVE_SCHEMA, thread_id=thread,
        rollout_sha256=hashlib.sha256(f"{thread}:{records}:{inputs}:{outputs}".encode()).hexdigest(),
        observed_token_count_records=records, usage_available=True, cache_included_in_input=True,
        task_complete_observed=complete, usage=dict(input_tokens=inputs, cached_input_tokens=inputs // 2,
            output_tokens=outputs, reasoning_output_tokens=outputs // 2, total_tokens=inputs + outputs))
    if not legacy:
        native.update(completion_basis=cumulative.COMPLETION_BASIS, lifecycle_malformed_records=0,
                      usage_lifecycle_complete=complete)
    return dict(schema=cumulative.INVOCATION_SCHEMA, invocation_id=identity, provider="codex_cli",
        model="authored-model", workspace="/authored/worktree", phase="coding",
        status="provider_returned" if complete else "failed", native_rollout_usage=native,
        usage=dict(thread_id=thread, model_id="authored-model"))


def _grok(*, complete=True):
    return dict(schema=cumulative.INVOCATION_SCHEMA, invocation_id="grok-envelope", provider="grok_cli",
        model="authored-grok-model", workspace="/authored/worktree", phase="coding", status="provider_returned",
        native_rollout_usage=dict(schema="native-grok-final-usage@1", task_complete_observed=True,
            usage_complete_observed=complete, usage_available=True, totals_scope="native_final_envelope",
            cache_included_in_input=False, usage=dict(input_tokens=10, cached_input_tokens=30,
                cache_write_input_tokens=0, output_tokens=2, total_tokens=42)))


def _collection(*, complete=True):
    counts = dict.fromkeys(scopes._COLLECTION_COUNTERS, 0)
    counts.update(entries_examined=1, log_files_seen=1, log_files_read=1,
                  bytes_read=100, lines_examined=1, provider_observations=1)
    if not complete:
        counts.update(entries_examined=2, log_files_seen=2, oversized_logs=1)
    return dict(schema="terminal-provider-receipt-collection@1", status="observed" if complete else "incomplete",
        complete=complete, counts=counts, reason_codes=[] if complete else ["oversized_log"],
        scope="retained_state_logs_only", attempt_population_complete=None, provider_call_count=None,
        campaign_call_population_complete=None, observation_only=True,
        **dict.fromkeys(scopes._COLLECTION_FALSE_FIELDS, False))


def test_ordinary_codex_adapter_selects_latest_session_once_and_ignores_stored_total():
    early = _codex(complete=False)
    latest = _codex("latest", records=3, inputs=160, outputs=40)
    report = dict(provider_invocations=[early, latest, deepcopy(latest)],
                  provider_usage_accounting={"report_provider_total_tokens": 1})
    measured = adapter.measured_usage(report)
    assert measured["input_tokens"] == 160 and measured["cached_input_tokens"] == 80
    assert measured["output_tokens"] == 40 and measured["total_tokens"] == 200
    assert measured["provider_calls"] == 2 and measured["observed_complete_sessions"] is True
    assert measured["known_provider_token_subtotal"] == 200
    assert measured["reconciliation"]["native_sessions_observed"] == 1
    assert measured["reconciliation"]["repeated_receipt_observations"] == 1
    assert measured["harbor_usage_exportable"] is True and measured["billing_total_verified"] is False
    assert measured["campaign_usage_complete"] is None


def test_ordinary_codex_failed_fresh_retry_keeps_subtotal_and_withholds_harbor_total():
    retry = _codex("retry", thread="authored-fresh-retry-123456789", inputs=70, outputs=10, complete=False)
    measured = adapter.measured_usage({"provider_invocations": [_codex(), retry], "task_completed": True})
    assert measured["known_provider_token_subtotal"] == 200
    assert measured["reconciliation"]["native_sessions_observed"] == 2
    assert measured["provider_calls"] == 2
    assert measured["total_tokens"] is None and measured["observed_complete_sessions"] is False
    assert measured["harbor_usage_exportable"] is False


@pytest.mark.parametrize("marker", [{}, None, False])
def test_ordinary_codex_missing_attempt_key_presence_never_becomes_complete(marker):
    measured = adapter.measured_usage(dict(provider_invocations=[_codex()], unreceipted_provider_attempt=marker))
    assert measured["known_provider_token_subtotal"] == 120
    assert measured["reconciliation"]["provider_total_tokens"] == 120
    assert measured["total_tokens"] is None and measured["all_invocations_receipted"] is False
    assert measured["provider_calls"] is None and measured["observed_provider_invocations"] == 1
    assert measured["harbor_usage_exportable"] is False


@pytest.mark.parametrize("false_marker_present", [False, True])
def test_ordinary_codex_collection_gap_blocks_adapter_and_final_projection(false_marker_present):
    report = dict(provider_invocations=[_codex()], provider_receipt_collection=_collection(complete=False))
    if false_marker_present:
        report["unreceipted_provider_attempt"] = False
    original = deepcopy(report)
    measured = adapter.measured_usage(report)
    projection.finalize_usage_accounting(report)
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == original
    assert measured["known_provider_token_subtotal"] == 120
    assert measured["total_tokens"] is None and measured["provider_calls"] is None
    assert measured["harbor_usage_exportable"] is False
    coverage = measured["semantic_decision_coverage"]
    assert coverage["provider_receipt_collection_valid"] is True
    assert coverage["provider_receipt_collection_complete_reported"] is False
    assert coverage["authoritative_usage_coverage_complete"] is False
    assert "incomplete_provider_receipt_collection" in {issue["reason"] for issue in coverage["issues"]}
    assert report["provider_usage_accounting"]["report_provider_total_tokens"] is None
    assert report["provider_usage_accounting"]["reconciliation"]["known_provider_token_subtotal"] == 120


def test_invalid_collection_authority_cannot_qualify_ordinary_codex_usage_or_export_text():
    collection = _collection()
    collection.update(provider_dispatch_count_verified=True, transcript="private-authored-marker")
    measured = adapter.measured_usage(dict(provider_invocations=[_codex()], provider_receipt_collection=collection))
    assert measured["total_tokens"] is None and measured["known_provider_token_subtotal"] == 120
    assert measured["provider_calls"] is None
    coverage = measured["semantic_decision_coverage"]
    assert coverage["provider_receipt_collection_valid"] is False
    assert coverage["provider_receipt_collection"] is None
    assert "invalid_provider_receipt_collection" in {issue["reason"] for issue in coverage["issues"]}
    assert "private-authored-marker" not in json.dumps(measured)


def test_plain_codex_historical_completion_flag_is_not_strict_final_usage():
    measured = adapter.measured_usage({"provider_invocations": [_codex(legacy=True)]})
    assert measured["known_provider_token_subtotal"] == 120
    assert measured["total_tokens"] is None and measured["observed_complete_sessions"] is False
    assert measured["harbor_usage_exportable"] is False


def test_codex_provider_with_foreign_native_schema_cannot_fall_back_to_raw_sum():
    receipt = _grok()
    receipt["provider"] = "codex_cli"
    measured = adapter.measured_usage({"provider_invocations": [receipt]})
    assert measured["total_tokens"] is None and measured["known_provider_token_subtotal"] is None
    assert measured["harbor_usage_exportable"] is False
    assert "unsupported_native_usage_schema" in {issue["reason"] for issue in measured["reconciliation"]["issues"]}


@pytest.mark.parametrize("provider", [None, "grok_cli"])
def test_native_codex_schema_with_missing_or_foreign_provider_stays_on_strict_route(provider):
    receipt = _codex()
    if provider is None:
        receipt.pop("provider")
    else:
        receipt["provider"] = provider
    measured = adapter.measured_usage({"provider_invocations": [receipt]})
    assert measured["total_tokens"] is None and measured["harbor_usage_exportable"] is False
    assert "reconciliation" in measured


@pytest.mark.parametrize("complete", [True, False, None])
def test_ordinary_grok_native_final_envelope_policy_keeps_its_categories_and_completeness(complete):
    measured = adapter.measured_usage({"provider_invocations": [_grok(complete=complete)]})
    assert measured["input_tokens"] == 10 and measured["cached_input_tokens"] == 30
    assert measured["output_tokens"] == 2 and measured["total_tokens"] == 42
    assert measured["cache_write_input_tokens"] == 0 and measured["cache_included_in_input"] is False
    assert measured["token_count_scope"] == "native_final_envelope"
    assert measured["harbor_usage_exportable"] is (complete is True)
    assert measured["billing_total_verified"] is False
    assert "reconciliation" not in measured


def test_collection_gap_can_only_downgrade_qualified_ordinary_grok_scope():
    measured = adapter.measured_usage(dict(provider_invocations=[_grok()], provider_receipt_collection=_collection(complete=False)))
    assert measured["provider_calls"] is None and measured["total_tokens"] is None
    assert measured["token_count_scope"] == "native_final_envelope"
    assert measured["harbor_usage_exportable"] is False


@pytest.mark.parametrize("marker", [{}, None, False])
def test_ordinary_grok_missing_attempt_presence_withholds_complete_counts(marker):
    measured = adapter.measured_usage(dict(provider_invocations=[_grok()], unreceipted_provider_attempt=marker))
    assert measured["total_tokens"] is None and measured["provider_calls"] is None
    assert measured["observed_provider_invocations"] == 1
    assert measured["all_invocations_receipted"] is False and measured["harbor_usage_exportable"] is False

