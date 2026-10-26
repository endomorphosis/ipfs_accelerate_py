"""Authored receipt comparisons; no provider, container or native replay runs.

These fixtures exercise the public retained-receipt collector. Their counters
are test data, not measured benchmark outcomes or provider billing evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_comparison import (
    BASELINE,
    SUPERVISOR,
    collect,
)


_MISSING = object()


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _codex(invocation="first", *, thread="native-session-123456789", inputs=100,
           cached=60, outputs=20, records=1, artifact="snapshot-1", complete=True):
    return {
        "schema": "router-implementation-invocation@1",
        "invocation_id": invocation,
        "provider": "codex_cli",
        "model": "gpt-6.1-sol",
        "purpose": "coding",
        "phase": "coding",
        "status": "provider_returned",
        "workspace": "/private/authored-workspace",
        "model_prompt_sha256": _sha("prompt"),
        "model_prompt_bytes": 6,
        "native_rollout_usage": {
            "schema": "native-codex-cumulative-usage@1",
            "thread_id": thread,
            "rollout_sha256": _sha(artifact),
            "observed_token_count_records": records,
            "usage_available": True,
            "task_complete_observed": complete,
            "cache_included_in_input": True,
            "billing_total_verified": False,
            "usage": {
                "input_tokens": inputs,
                "cached_input_tokens": cached,
                "output_tokens": outputs,
                "total_tokens": inputs + outputs,
            },
        },
    }


def _grok(invocation="grok-first", *, complete=True):
    row = _codex(invocation)
    row.update(provider="grok_cli", model="grok-4.7")
    row["native_rollout_usage"] = {
        "schema": "native-grok-final-usage@1",
        "usage_available": True,
        "task_complete_observed": True,
        "usage_complete_observed": complete,
        "usage_observation_sha256": _sha("envelope-" + invocation),
        "observed_final_usage_records": 1,
        "totals_scope": "native_final_envelope",
        "cache_included_in_input": False,
        "billing_total_verified": False,
        "usage": {
            "input_tokens": 10,
            "cached_input_tokens": 30,
            "cache_write_input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 42,
        },
    }
    return row


def _comparison(tmp_path, invocations, *, marker=False, report_extra=None):
    metadata = {
        "task": "authored-accounting-case",
        "model": "pinned-comparison-profile",
        "reasoning_effort": "high",
        "cli_version": "authored-cli",
        "original_task_inputs_unchanged": True,
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({**metadata, "schema": BASELINE,
                                    "trial_count": 0, "trials": []}))
    report = {"provider_invocations": deepcopy(invocations)}
    if marker is not _MISSING:
        report["unreceipted_provider_attempt"] = marker
    report.update(report_extra or {})
    supervisor = tmp_path / "supervisor.json"
    supervisor.write_text(json.dumps({
        **metadata, "schema": SUPERVISOR, "arm": "full", "trial_count": 1,
        "trials": [{"trial": "authored-trial", "reward": {"reward": 1},
                    "supervisor": report}],
    }))
    result = collect(baseline=baseline, supervisors=[supervisor])
    assert len(result["rows"]) == 1
    return result, result["rows"][0]


def _summary(row):
    reconciliation = row["native_usage_reconciliation"]
    assert reconciliation["status"] == "observed"
    assert reconciliation["summary"] is not None
    return reconciliation["summary"]


def test_latest_cumulative_session_snapshot_replaces_earlier_charge(tmp_path):
    early = _codex("early", complete=False)
    later = _codex("later", inputs=150, cached=90, outputs=30, records=2,
                   artifact="snapshot-2")
    result, row = _comparison(tmp_path, [early, later])
    summary = _summary(row)
    assert row["tokens"] == dict(input_tokens=150, cached_input_tokens=90,
                                  output_tokens=30, total_tokens=180)
    assert row["known_token_subtotals"]["total_tokens"] == 180
    assert [call["known_token_subtotals"]["total_tokens"] for call in row["invocations"]] == [120, 180]
    assert summary["native_session_count"] == 1
    assert summary["sessions"][0]["selected_observation_index"] == 1
    assert summary["cumulative_snapshots_are_additive"] is False
    assert result["arm_totals"]["full"]["observed_efficiency"]["total_tokens_per_official_pass"] == 180


def test_equal_snapshot_with_different_invocation_ids_is_charged_once(tmp_path):
    _, row = _comparison(tmp_path, [_codex("first"), _codex("second")])
    assert len(row["invocations"]) == 2
    assert row["tokens"]["total_tokens"] == 120
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert _summary(row)["native_session_count"] == 1


def test_equal_counts_from_distinct_native_sessions_remain_separate(tmp_path):
    _, row = _comparison(tmp_path, [_codex("first"),
        _codex("second", thread="other-session-123456789", artifact="other-snapshot")])
    assert row["tokens"]["total_tokens"] == 240
    assert _summary(row)["native_session_count"] == 2


def test_session_shared_by_trials_has_no_aggregate_usage_or_per_pass_attribution(tmp_path):
    first = _codex("first")
    first["native_rollout_usage"]["dollar_cost"] = 0.5
    _comparison(tmp_path, [first])
    path = tmp_path / "supervisor.json"
    receipt = json.loads(path.read_text())
    later = _codex("later", inputs=150, cached=90, outputs=30, records=2,
                   artifact="snapshot-2")
    later["native_rollout_usage"]["dollar_cost"] = 0.75
    receipt["trials"].append({
        "trial": "second-authored-trial", "reward": {"reward": 1},
        "supervisor": {"provider_invocations": [later], "unreceipted_provider_attempt": False},
    })
    receipt["trial_count"] = 2
    path.write_text(json.dumps(receipt))
    result = collect(baseline=tmp_path / "baseline.json", supervisors=[path])
    assert [row["tokens"]["total_tokens"] for row in result["rows"]] == [120, 180]
    assert [row["reported_cost_usd"] for row in result["rows"]] == [0.5, 0.75]
    totals = result["arm_totals"]["full"]
    assert totals["native_session_overlap_detected"] is True
    assert all(value is None for value in totals["tokens"].values())
    assert all(value is None for value in totals["known_token_subtotals"].values())
    assert totals["reported_cost_usd"] is None
    assert totals["known_reported_cost_subtotal_usd"] is None
    efficiency = totals["observed_efficiency"]
    assert efficiency["total_tokens_per_official_pass"] is None
    assert efficiency["reported_cost_usd_per_official_pass"] is None
    assert "native_sessions_shared_across_trials" in efficiency["token_ratio_unavailable_reasons"]


def _assert_shared_session_aggregate_is_unavailable(totals):
    assert totals["native_session_overlap_detected"] is True
    assert all(value is None for value in totals["tokens"].values())
    assert all(value is None for value in totals["known_token_subtotals"].values())
    assert totals["reported_cost_usd"] is None
    assert totals["known_reported_cost_subtotal_usd"] is None
    efficiency = totals["observed_efficiency"]
    assert efficiency["total_tokens_per_official_pass"] is None
    assert efficiency["reported_cost_usd_per_official_pass"] is None
    assert "native_sessions_shared_across_trials" in efficiency["token_ratio_unavailable_reasons"]
    assert "native_sessions_shared_across_trials" in efficiency["cost_ratio_unavailable_reasons"]


@pytest.mark.parametrize("reverse_receipts", [False, True])
def test_shared_session_across_arms_invalidates_each_single_trial_aggregate(tmp_path, reverse_receipts):
    first = _codex("full-invocation")
    first["native_rollout_usage"]["dollar_cost"] = 0.5
    _comparison(tmp_path, [first])
    full_path = tmp_path / "supervisor.json"
    no_index = json.loads(full_path.read_text())
    no_index["arm"] = "no-index"
    # Deliberately keep the same trial name: retained trial identity also
    # includes its arm, so these are distinct outcomes sharing one session.
    later = _codex("no-index-invocation", inputs=150, cached=90, outputs=30,
                   records=2, artifact="snapshot-2")
    later["native_rollout_usage"]["dollar_cost"] = 0.75
    no_index["trials"][0]["supervisor"]["provider_invocations"] = [later]
    no_index_path = tmp_path / "no-index.json"
    no_index_path.write_text(json.dumps(no_index))
    paths = [full_path, no_index_path]
    if reverse_receipts:
        paths.reverse()
    result = collect(baseline=tmp_path / "baseline.json", supervisors=paths)
    by_arm = {row["arm"]: row for row in result["rows"]}
    assert by_arm["full"]["tokens"]["total_tokens"] == 120
    assert by_arm["no-index"]["tokens"]["total_tokens"] == 180
    for arm in ("full", "no-index"):
        assert by_arm[arm]["native_session_shared_across_trials"] is True
        assert result["arm_totals"][arm]["trials"] == 1
        _assert_shared_session_aggregate_is_unavailable(result["arm_totals"][arm])


@pytest.mark.parametrize("failed_first", [False, True])
def test_failed_trial_slice_keeps_global_session_overlap_with_successful_trial(tmp_path, failed_first):
    first = _codex("first")
    first["native_rollout_usage"]["dollar_cost"] = 0.5
    _comparison(tmp_path, [first])
    path = tmp_path / "supervisor.json"
    receipt = json.loads(path.read_text())
    receipt["trials"][0]["reward"]["reward"] = 0 if failed_first else 1
    later = _codex("later", inputs=150, cached=90, outputs=30,
                   records=2, artifact="snapshot-2")
    later["native_rollout_usage"]["dollar_cost"] = 0.75
    receipt["trials"].append({
        "trial": "second-authored-trial", "reward": {"reward": 1 if failed_first else 0},
        "supervisor": {"provider_invocations": [later], "unreceipted_provider_attempt": False},
    })
    receipt["trial_count"] = 2
    path.write_text(json.dumps(receipt))
    result = collect(baseline=tmp_path / "baseline.json", supervisors=[path])
    assert [row["tokens"]["total_tokens"] for row in result["rows"]] == [120, 180]
    assert all(row["native_session_shared_across_trials"] is True for row in result["rows"])
    _assert_shared_session_aggregate_is_unavailable(result["arm_totals"]["full"])
    failed = result["unsuccessful_attempt_costs"]["full"]
    assert failed["trials"] == 1
    assert failed["observed_official_passes"] == 0
    _assert_shared_session_aggregate_is_unavailable(failed)


def test_native_counter_regression_withholds_session_total(tmp_path):
    first = _codex("first", inputs=150, cached=90, outputs=30)
    second = _codex("second", records=2, artifact="snapshot-2")
    result, row = _comparison(tmp_path, [first, second])
    summary = _summary(row)
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] is None
    assert summary["sessions"][0]["status"] == "conflict"
    assert "native_snapshot_counter_regression" in summary["sessions"][0]["reasons"]
    assert [call["known_token_subtotals"]["total_tokens"] for call in row["invocations"]] == [180, 120]
    assert result["arm_totals"]["full"]["observed_efficiency"]["total_tokens_per_official_pass"] is None


@pytest.mark.parametrize("field,value", [
    ("usage_available", False),
    ("cache_included_in_input", False),
    ("counter_regression_observed", True),
])
def test_valid_later_snapshot_does_not_erase_invalid_earlier_native_evidence(tmp_path, field, value):
    early = _codex("early")
    early["native_rollout_usage"][field] = value
    later = _codex("later", inputs=150, cached=90, outputs=30, records=2,
                   artifact="snapshot-2")
    result, row = _comparison(tmp_path, [early, later])
    assert all(count is None for count in row["tokens"].values())
    assert all(count is None for count in row["known_token_subtotals"].values())
    session = _summary(row)["sessions"][0]
    assert session["status"] == "conflict"
    assert session["selected_observation_index"] is None
    assert "native_usage_observation_invalid" in session["reasons"]
    assert [call["known_token_subtotals"]["total_tokens"] for call in row["invocations"]] == [120, 180]
    totals = result["arm_totals"]["full"]
    assert totals["tokens"]["total_tokens"] is None
    assert totals["known_token_subtotals"]["total_tokens"] is None
    assert totals["observed_efficiency"]["total_tokens_per_official_pass"] is None


@pytest.mark.parametrize("marker", [_MISSING, None, True, {}, {"phase": "coding"}])
def test_only_explicit_no_missing_attempts_can_authorize_complete_total(tmp_path, marker):
    _, row = _comparison(tmp_path, [_codex()], marker=marker)
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert _summary(row)["complete_supplied_native_usage"] is False


@pytest.mark.parametrize("gap", ["thread_id", "rollout_sha256", "observed_token_count_records",
                                 "usage_available", "cache_included_in_input"])
def test_legacy_custody_gaps_preserve_raw_counters_without_complete_total(tmp_path, gap):
    invocation = _codex()
    invocation["native_rollout_usage"].pop(gap)
    _, row = _comparison(tmp_path, [invocation])
    assert row["tokens"]["total_tokens"] is None
    assert row["invocations"][0]["known_token_subtotals"]["total_tokens"] == 120
    assert _summary(row)["complete_native_total_tokens"] is None


@pytest.mark.parametrize("mutation", ["unavailable", "wrong_schema", "incoherent_total",
                                      "cache_above_input", "invalid_latest", "observed_regression"])
def test_contradictory_native_evidence_never_becomes_complete_usage(tmp_path, mutation):
    invocation = _codex()
    native = invocation["native_rollout_usage"]
    if mutation == "unavailable":
        native["usage_available"] = False
    elif mutation == "wrong_schema":
        native["schema"] = "unknown-usage@1"
    elif mutation == "incoherent_total":
        native["usage"]["total_tokens"] = 999
    elif mutation == "cache_above_input":
        native["usage"]["cached_input_tokens"] = 101
    elif mutation == "invalid_latest":
        native["latest_usage_snapshot_valid"] = False
    else:
        native["counter_regression_observed"] = True
    _, row = _comparison(tmp_path, [invocation])
    assert row["tokens"]["total_tokens"] is None
    assert row["invocations"][0]["known_token_subtotals"]["total_tokens"] == native["usage"]["total_tokens"]


def test_grok_missing_usage_completeness_keeps_reported_counters_only(tmp_path):
    invocation = _grok()
    invocation["native_rollout_usage"].pop("usage_complete_observed")
    _, row = _comparison(tmp_path, [invocation])
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 42
    assert _summary(row)["complete_native_total_tokens"] is None


def test_mixed_providers_keep_native_input_semantics_and_total_without_double_cache(tmp_path):
    codex = _codex(inputs=20, cached=15, outputs=2)
    _, row = _comparison(tmp_path, [codex, _grok()])
    assert row["tokens"] == dict(input_tokens=30, cached_input_tokens=45,
                                  output_tokens=4, total_tokens=64)
    assert row["cache_included_in_input"] is None
    assert [call["cache_included_in_input"] for call in row["invocations"]] == [True, False]
    summary = _summary(row)
    assert summary["complete_native_counts"]["input_tokens_including_cache"] == 60
    assert summary["complete_native_total_tokens"] == 64
    assert summary["delivery_bytes_added_to_native_tokens"] is False


@pytest.mark.parametrize("independent_cost", [None, 0.25])
def test_different_invocation_costs_in_one_native_session_have_unknown_scope(tmp_path, independent_cost):
    first, second = _codex("first"), _codex("second")
    first["native_rollout_usage"]["dollar_cost"] = 0.5
    second["native_rollout_usage"]["dollar_cost"] = 0.75
    invocations = [first, second]
    if independent_cost is not None:
        third = _codex("third", thread="other-session-123456789", artifact="other-snapshot")
        third["native_rollout_usage"]["dollar_cost"] = independent_cost
        invocations.append(third)
    _, row = _comparison(tmp_path, invocations)
    assert row["reported_cost_usd"] is None
    assert row["known_reported_cost_subtotal_usd"] == independent_cost
    assert [call["known_reported_cost_subtotal_usd"] for call in row["invocations"]][:2] == [0.5, 0.75]
    assert row["tokens"]["total_tokens"] == (120 if independent_cost is None else 240)


def test_identical_retained_invocation_cost_is_not_ambiguous_or_recharged(tmp_path):
    invocation = _codex()
    invocation["native_rollout_usage"]["dollar_cost"] = 0.25
    _, row = _comparison(tmp_path, [invocation, deepcopy(invocation)])
    assert len(row["invocations"]) == 1
    assert row["tokens"]["total_tokens"] == 120
    assert row["reported_cost_usd"] == 0.25
    assert row["known_reported_cost_subtotal_usd"] == 0.25


@pytest.mark.parametrize("context_gap", ["malformed_ledger", "missing_prompt_digest", "invalid_prompt_bytes"])
def test_optional_context_failure_does_not_erase_valid_native_accounting(tmp_path, context_gap):
    invocation = _codex()
    if context_gap == "malformed_ledger":
        invocation["native_context_delivery"] = {"schema": "unsupported-ledger@1",
                                                 "raw_source": "PRIVATE-CONTEXT-MARKER"}
    elif context_gap == "missing_prompt_digest":
        invocation.pop("model_prompt_sha256")
    else:
        invocation["model_prompt_bytes"] = True
    result, row = _comparison(tmp_path, [invocation])
    assert row["tokens"]["total_tokens"] == 120
    assert _summary(row)["complete_native_total_tokens"] == 120
    assert row["context_delivery_observation"]["status"] == "unavailable"
    assert "PRIVATE-CONTEXT-MARKER" not in json.dumps(result)


def test_private_receipt_content_is_not_exported_by_reconciliation(tmp_path):
    invocation = _codex()
    invocation.update(prompt="PRIVATE-PROMPT-MARKER", raw_model_text="PRIVATE-RESPONSE-MARKER",
                      arbitrary={"source": "PRIVATE-SOURCE-MARKER"},
                      error="PRIVATE-ERROR-MARKER")
    result, row = _comparison(tmp_path, [invocation], report_extra={
        "raw_source": "PRIVATE-REPORT-MARKER"})
    assert row["tokens"]["total_tokens"] == 120
    encoded = json.dumps(result)
    assert "PRIVATE-" not in encoded
    assert "/private/authored-workspace" not in encoded
    summary = _summary(row)
    assert summary["raw_content_exported"] is False
    assert summary["billing_total_verified"] is False
    assert summary["provider_ingestion_verified"] is False
    assert summary["campaign_call_population_complete"] is None
