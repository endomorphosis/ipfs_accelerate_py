"""Authored native JSONL crosses the real log collector and comparison reader.

The sparse oversized file represents unavailable local evidence. It does not
establish that a provider dispatched another call or authenticate native facts.
"""
from copy import deepcopy
import json

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as terminal
from benchmarks.agent_supervisor.container_coding.test_benchmark_native_session_accounting import campaign


def _collect_logs(campaign, *, oversized=False, marker=True):
    report = campaign["report"]
    calls = deepcopy(report["provider_invocations"])
    report["provider_invocations"] = []
    report.update(coding_dispatch_possible=True, implementation_route="model_router")
    if not marker:
        report.pop("unreceipted_provider_attempt", None)
    state = campaign["root"] / "receipt-state"
    state.mkdir()
    (state / "retained.log").write_text("".join(json.dumps(call) + "\n" for call in calls))
    if oversized:
        with (state / "unavailable.log").open("wb") as stream:
            stream.truncate(8_000_001)
    observation = terminal._collect_provider_usage_logs(report, state=state)
    assert observation == report["provider_receipt_collection"]
    campaign["full"].write_text(json.dumps(campaign["value"], sort_keys=True))
    result = comparison.collect(baseline=campaign["baseline"], supervisors=[campaign["full"]])
    row = next(value for value in result["rows"] if value["arm"] == "full")
    return report, observation, result, row


def test_incomplete_receipt_scan_withholds_usage_after_retaining_native_calls(campaign, record_property):
    report, scan, result, row = _collect_logs(campaign, oversized=True)
    assert scan["schema"] == "terminal-provider-receipt-collection@1"
    assert scan["status"] == "incomplete" and scan["complete"] is False
    counts = scan["counts"]
    assert counts["log_files_seen"] == 2 and counts["log_files_read"] == 1
    assert counts["oversized_logs"] == 1 and counts["provider_observations"] == 3
    assert isinstance(report["unreceipted_provider_attempt"], dict)
    assert len(report["provider_invocations"]) == row["observed_provider_invocations"] == 3
    assert row["provider_calls"] is None
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["reported_cost_usd"] is None and row["native_usage_complete_observed"] is False
    assert row["known_token_subtotals"]["total_tokens"] == 247
    native = row["native_session_accounting"]
    assert native["native_session_count"] == 2 and native["failed_invocation_count"] == 1
    assert native["known_native_counter_subtotals"]["total_tokens"] == 247
    coverage = row["usage_coverage_observation"]
    assert coverage["complete_usage_blocked"] is True
    assert coverage["provider_receipt_collection"]["complete"] is False
    assert coverage["campaign_usage_complete"] is None
    read = row["context_delivery_observation"]["repository_source_mcp"]["read_scope_counts"]
    assert read["exposed_content_utf8_bytes"] == 36
    assert result["arm_totals"]["full"]["tokens"]["total_tokens"] is None
    assert result["arm_totals"]["full"]["observed_efficiency"]["total_tokens_per_official_pass"] is None
    assert row["official_reward"] == 1 and row["whole_interaction_qualified"] is False
    assert scan["attempt_population_complete"] is scan["provider_call_count"] is None
    assert scan["campaign_call_population_complete"] is None
    for flag in ("provider_usage_authenticity_verified", "provider_dispatch_count_verified",
                 "token_savings_measured", "completion_authority", "execution_authority"):
        assert scan[flag] is False
    observation = dict(schema="terminal-usage-coverage-observation@1",
        measurement_scope="authored_native_jsonl_and_local_receipt_logs",
        scan_log_files_seen=2, scan_log_files_read=1, scan_oversized_logs=1,
        retained_invocations=3, failed_retained_invocations=1, known_native_total_tokens=247,
        source_read_exposed_utf8_bytes=36, reported_complete_total_tokens=None,
        expected_provider_attempt_count=None, campaign_usage_complete=None,
        possible_unreceipted_attempt_reported=True, complete_provider_calls_withheld=True,
        receipt_collection_complete=False, provider_receipts_authenticity_verified=False,
        live_provider_execution=False, token_savings_measured=False, billing_total_verified=False,
        causal_claim=False, whole_interaction_qualified=False, local_prompt_bytes_added_to_native_usage=False)
    record_property("terminal_usage_coverage_observation", json.dumps(observation, sort_keys=True))


def test_clean_collection_preserves_counts_without_claiming_campaign_coverage(campaign):
    report, scan, result, row = _collect_logs(campaign)
    assert scan["status"] == "observed" and scan["complete"] is True
    assert report["unreceipted_provider_attempt"] is False
    assert row["tokens"]["total_tokens"] == 247 and row["provider_calls"] == 3
    assert row["usage_coverage_observation"]["complete_usage_blocked"] is False
    assert row["usage_coverage_observation"]["campaign_usage_complete"] is None
    assert row["whole_interaction_qualified"] is False
    assert result["causal_claim"] is False


def test_clean_collection_does_not_repair_unknown_missing_attempt_marker(campaign):
    report, scan, _, row = _collect_logs(campaign, marker=False)
    assert scan["complete"] is True and "unreceipted_provider_attempt" not in report
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 247
    assert row["native_session_accounting"]["unreceipted_provider_attempt"] is None
    assert row["usage_coverage_observation"]["campaign_usage_complete"] is None
