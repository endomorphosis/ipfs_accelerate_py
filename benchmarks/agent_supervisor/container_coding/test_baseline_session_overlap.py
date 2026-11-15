"""Baseline transcript identities join existing supervisor overlap protection.

Native facts and verifier rewards are authored fixtures. The genuine receipt
producers and comparison reader run without a provider or task execution.
"""
import hashlib
import json

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as baseline
from benchmarks.agent_supervisor.container_coding.test_benchmark_native_session_accounting import campaign


def _baseline_from_shared_rollout(campaign, *, duplicate=False, unidentified=False):
    native = campaign["report"]["provider_invocations"][1]["native_rollout_usage"]
    thread = native["thread_id"]
    original = campaign["home"] / "sessions/2026/10/15" / ("rollout-" + thread + ".jsonl")
    raw = original.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == native["rollout_sha256"]
    if unidentified:
        lines = raw.splitlines(keepends=True)
        assert json.loads(lines[0])["type"] == "session_meta"
        raw = b"".join(lines[1:])
    agent = campaign["root"] / "baseline-agent"
    folder = agent / "sessions/2026/10/15"
    folder.mkdir(parents=True)
    (folder / ("rollout-" + thread + ".jsonl")).write_bytes(raw)
    if duplicate:
        (folder / "rollout-identical-copy.jsonl").write_bytes(raw)
    usage = baseline.rollout_usage(agent)
    receipt = json.loads(campaign["baseline"].read_text())
    receipt["trials"][0]["raw_usage"] = usage
    campaign["baseline"].write_text(json.dumps(receipt, sort_keys=True))
    campaign["full"].write_text(json.dumps(campaign["value"], sort_keys=True))
    result = comparison.collect(baseline=campaign["baseline"], supervisors=[campaign["full"]])
    rows = {row["arm"]: row for row in result["rows"]}
    return usage, result, rows


def test_authored_baseline_and_supervisor_reuse_withholds_both_arm_totals(campaign, record_property):
    usage, result, rows = _baseline_from_shared_rollout(campaign)
    native, full = rows["native-codex"], rows["full"]
    assert usage["source"] == "native_rollout_observed_cumulative_totals"
    assert usage["task_complete_observed"] is True
    assert len(usage["sessions"]) == 1 and usage["sessions"][0]["token_count_events"] == 2
    assert usage["usage"]["total_tokens"] == native["tokens"]["total_tokens"] == 170
    assert full["tokens"]["total_tokens"] == 247
    identity = native["baseline_native_session_identity_observation"]
    assert identity["schema"] == "terminal-baseline-native-session-identities@1"
    assert identity["status"] == "observed" and identity["inventory_complete"] is True
    assert identity["observed_session_count"] == 1
    assert identity["modern_source"] is True and identity["usage_population_coherent"] is True
    shared = hashlib.sha256(usage["sessions"][0]["session_id"].encode("utf-8")).hexdigest()
    assert identity["identity_sha256"] == [shared]
    for row in rows.values():
        assert row["native_sessions_reused_across_comparison_trials"] == [shared]
        assert row["official_reward"] == 1
        assert row["whole_interaction_qualified"] is False
    summary = full["native_session_accounting"]
    assert summary["native_session_count"] == 2 and summary["failed_invocation_count"] == 1
    assert summary["complete_native_total_tokens"] == 247
    assert summary["sessions"][0]["identity_sha256"] == shared
    assert [call["status"] for call in full["invocations"]] == ["provider_returned", "provider_returned", "failed"]
    read = full["context_delivery_observation"]["repository_source_mcp"]["read_scope_counts"]
    assert read["line_range_reads"] == 1 and read["exposed_content_utf8_bytes"] == 36
    assert result["retained_trial_count"] == 2 and result["all_declared_controls_match"] is True
    for arm in ("native-codex", "full"):
        totals = result["arm_totals"][arm]
        assert totals["native_session_overlap_count"] == 1
        assert totals["native_session_overlap_prevents_complete_total"] is True
        assert totals["tokens"] == dict.fromkeys(comparison.TOKENS)
        assert totals["reported_cost_usd"] is None
        assert totals["observed_efficiency"]["total_tokens_per_official_pass"] is None
        assert "native_session_reused_across_retained_trials" in totals["observed_efficiency"]["token_ratio_unavailable_reasons"]
    assert result["arm_totals"]["native-codex"]["known_token_subtotals"]["total_tokens"] == 170
    assert result["arm_totals"]["full"]["known_token_subtotals"]["total_tokens"] == 247
    assert result["causal_claim"] is result["runtime_control_enforcement_verified"] is False
    observation = dict(schema="baseline-session-overlap-observation@1",
        measurement_scope="authored_native_jsonl_baseline_and_supervisor", retained_trials=2,
        identified_baseline_sessions=1, supervisor_native_sessions=2, shared_native_sessions=1,
        baseline_reported_total_tokens=170, supervisor_reconciled_total_tokens=247,
        failed_supervisor_invocations=1, aggregate_complete_total_tokens_withheld=True,
        per_pass_ratios_withheld=True, baseline_identity_inventory_complete=True,
        provider_receipts_authenticity_verified=False, live_provider_execution=False,
        token_savings_measured=False, causal_claim=False, billing_total_verified=False,
        whole_interaction_qualified=False)
    record_property("baseline_session_overlap_observation", json.dumps(observation, sort_keys=True))


def test_actual_baseline_producer_deduplicates_identical_transcript_copies(campaign):
    usage, result, rows = _baseline_from_shared_rollout(campaign, duplicate=True)
    assert len(usage["sessions"]) == 1 and usage["usage"]["total_tokens"] == 170
    assert rows["native-codex"]["provider_calls"] == 1
    identity = rows["native-codex"]["baseline_native_session_identity_observation"]
    assert identity["status"] == "observed" and identity["inventory_complete"] is True
    assert identity["observed_session_count"] == len(identity["identity_sha256"]) == 1
    assert rows["full"]["native_session_accounting"]["complete_native_total_tokens"] == 247
    for arm in ("native-codex", "full"):
        assert result["arm_totals"][arm]["native_session_overlap_count"] == 1
        assert result["arm_totals"][arm]["tokens"]["total_tokens"] is None


def test_unidentified_baseline_fallback_never_becomes_a_native_session_identity(campaign):
    usage, result, rows = _baseline_from_shared_rollout(campaign, unidentified=True)
    assert len(usage["sessions"]) == 1
    assert usage["sessions"][0]["session_id"].startswith("unidentified:")
    assert usage["task_complete_observed"] is False and usage["usage"]["total_tokens"] == 170
    native = rows["native-codex"]
    identity = native["baseline_native_session_identity_observation"]
    assert identity["inventory_complete"] is False and identity["identity_sha256"] == []
    assert native["tokens"]["total_tokens"] is None
    assert native["known_token_subtotals"]["total_tokens"] == 170
    assert native["native_sessions_reused_across_comparison_trials"] == []
    assert result["arm_totals"]["native-codex"]["tokens"]["total_tokens"] is None
    assert result["arm_totals"]["full"]["native_session_overlap_count"] == 0
    assert result["arm_totals"]["full"]["tokens"]["total_tokens"] == 247
    assert result["causal_claim"] is False
