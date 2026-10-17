"""Actual retained-summary collectors with authored history, no provider calls."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_comparison import markdown
from benchmarks.agent_supervisor.container_coding.test_benchmark_session_accounting import _comparison, _codex
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_observation import (
    evidence, _authored, _collect)
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_correlation_population import _authored_v2, _lifecycle
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_worker_forwarding_capture import authored_captured_worker_evidence


def _report(evidence, *, partial=False):
    first = authored_captured_worker_evidence(_authored_v2(evidence["first"]))
    first["row"].update(model="gpt-6.1-sol", native_rollout_usage=_codex()["native_rollout_usage"])
    evidence["write"](first)
    _lifecycle(evidence, first)
    evidence["report"]["provider_invocations"] = [first["row"]]
    if partial:
        second = authored_captured_worker_evidence(_authored_v2(_authored("3" * 32, "4" * 32)))
        second["row"].update(model="gpt-6.1-sol", native_rollout_usage=_codex("second",
            thread="other-native-session-123456789", artifact="other-native-snapshot")["native_rollout_usage"])
        second["row"].pop("repository_source_worker_forwarding")
        evidence["write"](second)
        evidence["report"]["provider_invocations"].append(second["row"])
    report = deepcopy(evidence["report"])
    report["repository_source_owner_observation"] = _collect(evidence)
    assert report["repository_source_owner_observation"]["status"] == "observed"
    return report


def _compare(tmp_path, report):
    # The original owner collector treats an absent attempt marker as absent;
    # native usage separately keeps whole-population completeness unknown.
    return _comparison(tmp_path, report["provider_invocations"], marker=None, report_extra=report)


def _usage(row):
    return {key: row[key] for key in ("tokens", "known_token_subtotals", "reported_cost_usd",
        "known_reported_cost_subtotal_usd", "native_usage_complete_observed")}


def test_actual_owner_collector_is_projected_through_public_comparison_and_markdown(evidence, tmp_path, request):
    report = _report(evidence)
    result, row = _compare(tmp_path, report)
    owner = row["repository_source_owner_observation"]
    assert owner["status"] == "observed" and owner["session_count"] == 1
    assert owner["prepared_mcp_counts"]["successful_reads"] == 1
    assert owner["operation_correlation"]["counts"]["completed_socket_send_successful_reads"] == 1
    worker = owner["worker_forwarding_observation"]
    assert worker["status"] == "observed" and worker["known_session_count"] == 1
    assert worker["counts"]["reported_completed_stdout_successful_reads"] == 1
    assert worker["worker_reported_only"] is True
    assert worker["independent_worker_receipt_verified"] is worker["provider_ingestion_verified"] is False
    assert owner["lifecycle_observation"]["own_service_reservation_release_reported"] is True
    assert owner["source_audits_replayed_here"] is owner["token_savings_measured"] is False
    assert owner["tool_adoption_established"] is owner["worker_receipts_independently_verified"] is False
    assert row["tokens"]["total_tokens"] is None and row["known_token_subtotals"]["total_tokens"] == 120
    without = deepcopy(report)
    without.pop("repository_source_owner_observation")
    _, original = _compare(tmp_path, without)
    assert _usage(row) == _usage(original)
    text = markdown(result)
    assert "Reported pipe-written reads" in text and "worker-reported pipe writes" in text
    assert "provider ingestion, token savings or tool adoption" in text
    assert "Repository observations are retained summaries" in text
    encoded = json.dumps(owner)
    assert "/authored" not in encoded and "request_sha256" not in encoded and "stdout_sha256" not in encoded
    assert len(result["repository_owner_report_projector_source_sha256"]) == 64
    request.node.user_properties.append(("benchmark_owner_reporting", json.dumps(dict(
        authored_history=True, owner=owner, native_usage=_usage(row)), sort_keys=True)))


def test_partial_worker_population_preserves_unknown_totals_and_known_coverage(evidence, tmp_path):
    result, row = _compare(tmp_path, _report(evidence, partial=True))
    owner = row["repository_source_owner_observation"]
    assert owner["status"] == "observed" and owner["prepared_mcp_counts"]["successful_reads"] == 2
    worker = owner["worker_forwarding_observation"]
    assert worker["status"] == "unavailable" and worker["counts"] is None
    assert worker["known_session_count"] == worker["unknown_session_count"] == 1
    assert "1 / 1" in markdown(result)
    assert row["known_token_subtotals"]["total_tokens"] == 240


@pytest.mark.parametrize("change", ["counts", "descriptor", "authority", "duplicate_planning", "policy"])
def test_invalid_owner_summary_does_not_invalidate_native_usage(evidence, tmp_path, change):
    report = _report(evidence)
    _, original = _compare(tmp_path, report)
    owner = report["repository_source_owner_observation"]
    if change == "counts":
        owner["prepared_mcp_counts"]["successful_reads"] += 1
    elif change == "descriptor":
        owner["sessions"][0]["descriptor_sha256"] = "0" * 64
    elif change == "authority":
        owner["provider_ingestion_verified"] = True
    elif change == "duplicate_planning":
        # Exercise the pure projector directly: comparison already rejects a
        # conflicting duplicate native invocation independently of context.
        from benchmarks.agent_supervisor.container_coding.benchmark_repository_owner_observation import project
        report["provider_invocations"].append(dict(invocation_id=report["provider_invocations"][0]["invocation_id"], phase="planning"))
        assert project(report)["status"] == "unavailable"
        return
    else:
        report["repository_source_policy"] = "repository-source-raw@1"
    _, row = _compare(tmp_path, report)
    projected = row["repository_source_owner_observation"]
    assert projected["status"] == "unavailable" and projected["prepared_mcp_counts"] is None
    assert _usage(row) == _usage(original)


@pytest.mark.parametrize("change", ["sum", "hash", "worker_authority", "owner_correlation",
    "owner_payload", "owner_categories", "count_relationship", "private_field"])
def test_optional_worker_summary_failure_preserves_prepared_counts(evidence, tmp_path, change):
    report = _report(evidence)
    _, original = _compare(tmp_path, report)
    owner = report["repository_source_owner_observation"]
    worker = owner["worker_forwarding_observation"]
    if change == "sum":
        worker["counts"]["reported_received_operations"] += 1
    elif change == "hash":
        worker["sessions"][0]["audit_sha256"] = "0" * 64
    elif change == "worker_authority":
        worker["independent_worker_receipt_verified"] = True
    elif change == "owner_correlation":
        owner["sessions"][0]["operation_correlation"].update(status="unavailable", counts=None)
    elif change in {"owner_payload", "owner_categories"}:
        counts = owner["sessions"][0]["operation_correlation"]["counts"]
        if change == "owner_payload":
            counts["completed_socket_send_native_payload_bytes"] = 121
        else:
            counts.update(completed_socket_send_successful_tool_results=0,
                          completed_socket_send_failed_tool_results=1)
        owner["operation_correlation"]["counts"] = deepcopy(counts)
    elif change == "count_relationship":
        worker["sessions"][0]["counts"]["reported_received_tool_results"] = 0
        worker["counts"]["reported_received_tool_results"] = 0
    else:
        worker["private"] = "/never-exported"
    result, row = _compare(tmp_path, report)
    projected = row["repository_source_owner_observation"]
    assert projected["status"] == "observed" and projected["prepared_mcp_counts"]["successful_reads"] == 1
    if change != "private_field":
        assert projected["worker_forwarding_observation"]["status"] == "unavailable"
        assert projected["worker_forwarding_observation"]["counts"] is None
    assert _usage(row) == _usage(original)
    assert "/never-exported" not in json.dumps(result)


def test_missing_optional_history_remains_unknown(evidence, tmp_path):
    report = _report(evidence)
    owner = report["repository_source_owner_observation"]
    for key in ("operation_correlation", "worker_forwarding_observation", "lifecycle_observation"):
        owner.pop(key)
    _, row = _compare(tmp_path, report)
    projection = row["repository_source_owner_observation"]
    assert projection["status"] == "observed" and projection["prepared_mcp_counts"]["successful_reads"] == 1
    assert projection["worker_forwarding_observation"]["counts"] is None
    assert projection["operation_correlation"]["counts"] is None
    report.pop("repository_source_owner_observation")
    _, absent = _compare(tmp_path, report)
    assert absent["repository_source_owner_observation"]["status"] == "unavailable"
    assert absent["repository_source_owner_observation"]["prepared_mcp_counts"] is None
    assert _usage(row) == _usage(absent)


def _failed(*, reason="rate_limit", hint=30):
    invocation = _codex(complete=False)
    invocation.update(status="failed", failure_phase="provider_invocation", error_type="ProviderError",
        provider_failure=dict(phase="provider_invocation", reason_code=reason))
    if hint is not None:
        invocation["provider_failure"]["retry_after_seconds"] = hint
    return invocation


def test_rate_limit_hint_and_failed_usage_survive_comparison(tmp_path):
    invocation = _failed()
    result, row = _comparison(tmp_path, [invocation])
    projection = row["provider_stop_observation"]
    assert projection["status"] == "observed"
    assert projection["stop"]["state"] == "rate_limited"
    assert projection["stop"]["reason_code"] == "rate_limit" and projection["stop"]["retry_after_seconds"] == 30
    assert projection["retry_authority"] is projection["automatic_replay_allowed"] is False
    assert row["tokens"]["total_tokens"] is None and row["known_token_subtotals"]["total_tokens"] == 120
    text = markdown(result)
    assert "rate_limited" in text and "rate_limit" in text and "Retry-after seconds" in text
    assert "does not authorize retries" in text


@pytest.mark.parametrize("change", ["missing_hint", "invalid_hint", "unbounded", "private_diagnostic", "billing"])
def test_provider_stop_retains_unknown_or_invalid_cooldown_without_raw_text(tmp_path, change):
    invocation = _failed()
    diagnostic = invocation["provider_failure"]
    if change == "missing_hint": diagnostic.pop("retry_after_seconds")
    elif change == "invalid_hint": diagnostic["retry_after_seconds"] = True
    elif change == "unbounded":
        diagnostic.pop("retry_after_seconds")
        diagnostic["retry_after_unbounded"] = True
    elif change == "private_diagnostic": diagnostic["message"] = "/never-exported"
    else:
        diagnostic.pop("retry_after_seconds")
        diagnostic["reason_code"] = "billing"
    result, row = _comparison(tmp_path, [invocation])
    projection = row["provider_stop_observation"]
    assert projection["status"] == "observed" and projection["stop"]["retry_after_seconds"] is None
    assert projection["stop"]["state"] == ("rate_limited" if change == "missing_hint" else "provider_blocked")
    assert row["known_token_subtotals"]["total_tokens"] == 120 and row["tokens"]["total_tokens"] is None
    assert "/never-exported" not in json.dumps(result)


def test_copied_provider_stop_without_failed_receipt_cannot_authorize_or_claim_cooldown(tmp_path):
    _, row = _comparison(tmp_path, [_codex()], report_extra={
        "provider_stop": {"state": "rate_limited", "retry_after_seconds": 30, "private": "/never-exported"}})
    projected = row["provider_stop_observation"]
    assert projected["status"] == "unavailable" and projected["stop"] is None
    assert row["tokens"]["total_tokens"] == 120
    assert "/never-exported" not in json.dumps(row)


def test_distinct_failed_invocation_stops_are_not_erased_by_strongest_trial_stop(tmp_path):
    limited, blocked = _failed(), _failed(reason="billing", hint=None)
    blocked["invocation_id"] = "second"
    result, row = _comparison(tmp_path, [limited, blocked])
    assert row["provider_stop_observation"]["stop"]["reason_code"] == "billing"
    stops = [call["provider_stop_observation"]["stop"] for call in row["invocations"]]
    assert [stop["reason_code"] for stop in stops] == ["rate_limit", "billing"]
    assert [stop["retry_after_seconds"] for stop in stops] == [30, None]
    assert "rate_limit" in markdown(result) and "billing" in markdown(result)
