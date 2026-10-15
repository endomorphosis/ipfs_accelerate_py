"""Authored dialogue fixtures preserve charges and expose coverage gaps.

These receipts are local test data, not provider or Terminal Bench results.
"""

from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as terminal
from ipfs_accelerate_py.agent_supervisor.runtime import cumulative_usage_reconciliation as accounting
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_usage as usage


def _native(number=1, *, thread=None, inputs=100, outputs=20, complete=True, count=1):
    thread = thread or f"native-session-{number:016d}"
    return {
        "schema": accounting.INVOCATION_SCHEMA, "invocation_id": f"invocation-{number}",
        "provider": "codex_cli", "model": "fixture-model", "workspace": "/fixture/worktree",
        "purpose": "coding", "status": "provider_returned" if complete else "failed",
        "model_prompt_sha256": hashlib.sha256(f"prompt-{number}".encode()).hexdigest(),
        "usage": {"thread_id": thread, "model_id": "fixture-model"},
        "native_rollout_usage": {
            "schema": accounting.NATIVE_SCHEMA, "thread_id": thread,
            "rollout_sha256": hashlib.sha256(f"{number}:{count}:{inputs}:{outputs}".encode()).hexdigest(),
            "observed_token_count_records": count,
            "usage": {"input_tokens": inputs, "output_tokens": outputs,
                      "cached_input_tokens": inputs // 2, "reasoning_output_tokens": outputs // 2,
                      "total_tokens": inputs + outputs},
            "cache_included_in_input": True, "usage_available": True,
            "task_complete_observed": complete, "completion_basis": accounting.COMPLETION_BASIS,
            "lifecycle_malformed_records": 0, "usage_lifecycle_complete": complete,
        },
    }


def _attempt(native=None, *, number=1, provider="codex_cli", dispatched=True):
    prompt_sha = (native or {}).get("model_prompt_sha256", hashlib.sha256(f"prompt-{number}".encode()).hexdigest())
    receipt = {
        "schema": usage.PROVIDER_SCHEMA, "call_id": f"call-{number}", "session_id": "dialogue-session",
        "provider": provider, "model": "fixture-model", "workspace": "/fixture/worktree",
        "model_prompt_sha256": prompt_sha, "provider_called": dispatched,
        # These wrapper counters and local prompt estimates must not be added.
        "usage": {"prompt_tokens": 9999, "completion_tokens": 9999, "cached_tokens": 9999},
        "status": "provider_returned",
    }
    if native is not None:
        receipt["native_invocation"] = native
    return {
        "call_id": f"call-{number}", "model_prompt_sha256": prompt_sha,
        "provider_receipt": receipt, "provider_receipt_bound": True,
        "content_tokens": 9999, "status": "provider_returned",
    }


def _dialogue(*attempts, provider="codex_cli"):
    return {
        "schema": usage.DIALOGUE_SCHEMA, "session_id": "dialogue-session", "provider": provider,
        "model": "fixture-model", "workspace": "/fixture/worktree", "status": "returned",
        "attempts": list(attempts), "invocation_attempt_count": len(attempts),
        "provider_usage_complete": True, "missing_provider_receipt_count": 0,
        "codex_cumulative_usage": {"provider_total_tokens": 1},
    }


def _surfaces(report):
    before = deepcopy(report)
    terminal._final_usage_accounting(report)
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == before
    final = report["provider_usage_accounting"]
    measured = harbor.measured_usage(report)
    assert final["report_provider_total_tokens"] == measured["total_tokens"]
    for result in (final["semantic_decision_coverage"], measured["semantic_decision_coverage"]):
        assert result["local_prompt_estimates_added"] is False
        assert result["completion_authority"] is False
    return final, measured


@pytest.mark.parametrize("dialogue_first", [False, True])
@pytest.mark.parametrize("printed_phase", [False, True])
def test_collector_deduplicates_printed_and_nested_original_invocations_in_both_orders(dialogue_first, printed_phase):
    first, second = _native(), _native(2, inputs=50, outputs=10)
    dialogue = _dialogue(_attempt(first), _attempt(second, number=2))
    printed = [deepcopy(first), deepcopy(second)]
    if printed_phase:
        for row in printed:
            row["phase"] = "coding"
    observations = ([dialogue, *printed] if dialogue_first else [*printed, dialogue])
    report = {"provider_invocations": []}
    for item in observations + [deepcopy(dialogue)]:
        assert usage.collect_provider_usage_observation(report, item) is True
    assert len(report["semantic_decision_dialogues"]) == 1
    assert len(report["provider_invocations"]) == 2
    assert [row["invocation_id"] for row in report["provider_invocations"]] == ["invocation-1", "invocation-2"]
    final, measured = _surfaces(report)
    assert final["report_provider_total_tokens"] == 180
    assert measured["provider_calls"] == 2
    assert measured["all_invocations_receipted"] is True


def test_pure_projection_counts_shared_cumulative_session_once_and_does_not_trust_summary():
    first = _native(complete=False)
    second = _native(2, thread=first["native_rollout_usage"]["thread_id"], count=3, inputs=160, outputs=40)
    report = {"provider_invocations": [first, second],
              "semantic_decision_dialogues": [_dialogue(_attempt(first), _attempt(second, number=2))],
              "provider_usage_accounting": {"report_provider_total_tokens": 1}}
    report.pop("provider_usage_accounting")
    final, measured = _surfaces(report)
    assert final["report_provider_total_tokens"] == 200
    assert final["reconciliation"]["native_sessions_observed"] == 1
    assert final["reconciliation"]["observations_received"] == 2
    assert measured["provider_calls"] == 2
    assert measured["cached_input_tokens"] == 80
    assert measured["known_counter_subtotals"]["reasoning_output_tokens"] == 20
    report["provider_usage_accounting"] = {"report_provider_total_tokens": 1}
    assert harbor.measured_usage(report)["total_tokens"] == 200


def test_failed_fresh_expansion_retains_partial_usage_and_unknown_exact_total():
    first, failed = _native(), _native(2, inputs=70, outputs=10, complete=False)
    attempt = _attempt(failed, number=2)
    attempt["status"] = "provider_failed"
    attempt["provider_receipt"]["status"] = "failed"
    dialogue = _dialogue(_attempt(first), attempt)
    dialogue["status"] = "failed"
    report = {"provider_invocations": [], "semantic_decision_dialogues": [dialogue]}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["known_provider_token_subtotal"] == 200
    assert final["report_provider_total_tokens"] is None
    assert measured["provider_calls"] == 2
    assert measured["all_invocations_receipted"] is True


@pytest.mark.parametrize("failure,reason", [
    ("missing", "missing_dialogue_provider_receipt"),
    ("unbound", "unbound_dialogue_provider_receipt"),
    ("unknown_dispatch", "unknown_dialogue_provider_dispatch"),
    ("malformed", "invalid_dialogue_attempt"),
    ("missing_native", "missing_dialogue_native_invocation"),
    ("binding_conflict", "dialogue_native_invocation_binding_conflict"),
])
def test_complete_subset_cannot_hide_a_dialogue_coverage_gap(failure, reason):
    first = _native()
    later = _attempt(number=2)
    if failure == "missing":
        later["provider_receipt"] = None
    elif failure == "unbound":
        later["provider_receipt_bound"] = False
    elif failure == "unknown_dispatch":
        later["provider_receipt"]["provider_called"] = None
    elif failure == "malformed":
        later = "malformed attempt"
    elif failure == "binding_conflict":
        later = _attempt(_native(2), number=2)
        later["provider_receipt"]["native_invocation"]["workspace"] = "/another/worktree"
    report = {"provider_invocations": [first],
              "semantic_decision_dialogues": [_dialogue(_attempt(first), later)]}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["known_provider_token_subtotal"] >= 120
    assert final["report_provider_total_tokens"] is None
    assert measured["all_invocations_receipted"] is False
    assert reason in {row["reason"] for row in final["semantic_decision_coverage"]["issues"]}


@pytest.mark.parametrize("gap", ["unbound", "foreign_native_binding"])
def test_foreign_receipts_keep_call_count_unknown_and_retain_known_usage(gap):
    first, later_native = _native(), _native(2)
    later = _attempt(later_native, number=2)
    if gap == "unbound":
        later["provider_receipt_bound"] = False
    else:
        later["provider_receipt"]["native_invocation"]["workspace"] = "/foreign/worktree"
    report = {"provider_invocations": [first],
              "semantic_decision_dialogues": [_dialogue(_attempt(first), later)]}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["known_provider_token_subtotal"] >= 120
    assert final["report_provider_total_tokens"] is None
    assert measured["provider_calls"] is None
    assert measured["semantic_decision_coverage"]["provider_call_count_complete"] is False


def test_local_predispatch_refusal_does_not_invent_a_call_or_zero_tokens():
    attempt = _attempt(dispatched=False)
    attempt["status"] = "provider_failed"
    attempt["provider_receipt"].update(status="failed", usage=None)
    report = {"provider_invocations": [], "semantic_decision_dialogues": [_dialogue(attempt)]}
    final, measured = _surfaces(report)
    coverage = final["semantic_decision_coverage"]
    assert coverage["reported_local_refusals"] == 1
    assert coverage["reported_provider_dispatches"] == 0
    assert coverage["authoritative_usage_coverage_complete"] is True
    assert measured["provider_calls"] == 0
    assert measured["total_tokens"] is None
    assert measured["known_provider_token_subtotal"] is None


def test_dispatch_refusal_contradicting_native_evidence_retains_charge_and_blocks_total():
    native = _native()
    attempt = _attempt(native, dispatched=False)
    report = {"provider_invocations": [], "semantic_decision_dialogues": [_dialogue(attempt)]}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["known_provider_token_subtotal"] == 120
    assert final["report_provider_total_tokens"] is None
    assert measured["provider_calls"] is None
    assert "dialogue_provider_dispatch_contradiction" in {
        row["reason"] for row in final["semantic_decision_coverage"]["issues"]}


def test_changed_native_receipt_with_same_identity_is_preserved_as_conflict():
    first = _native()
    changed = _native(inputs=101)
    report = {"provider_invocations": []}
    usage.collect_provider_usage_observation(report, first)
    usage.collect_provider_usage_observation(report, _dialogue(_attempt(changed)))
    assert len(report["provider_invocations"]) == 2
    terminal._final_usage_accounting(report)
    result = report["provider_usage_accounting"]
    assert result["report_provider_total_tokens"] is None
    assert "invocation_identity_alias_conflict" in result["reconciliation"]["sessions"][0]["reasons"]
    with pytest.raises(ValueError, match="conflicting provider receipts"):
        harbor.measured_usage(report)


def test_grok_observations_are_retained_without_claiming_authoritative_totals():
    attempt = _attempt(provider="grok_cli")
    report = {"provider_invocations": [],
              "semantic_decision_dialogues": [_dialogue(attempt, provider="grok_cli")]}
    final, measured = _surfaces(report)
    assert measured["provider_calls"] == 1
    assert measured["all_invocations_receipted"] is True
    assert measured["total_tokens"] is None
    assert measured["known_provider_token_subtotal"] is None
    assert final["semantic_decision_coverage"]["reported_provider_dispatches"] == 1
    assert "unsupported_dialogue_provider_usage" in {
        row["reason"] for row in final["semantic_decision_coverage"]["issues"]}
    assert report["semantic_decision_dialogues"][0]["attempts"][0]["provider_receipt"]["usage"]["prompt_tokens"] == 9999


def test_changed_dialogue_attempts_or_population_cannot_claim_complete_usage():
    original = _dialogue(_attempt(_native()))
    changed = deepcopy(original)
    changed["attempts"][0]["provider_receipt_bound"] = False
    report = {"provider_invocations": [], "semantic_decision_dialogues": [original, changed]}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["known_provider_token_subtotal"] == 120
    assert measured["total_tokens"] is None
    assert "dialogue_session_identity_conflict" in {
        row["reason"] for row in final["semantic_decision_coverage"]["issues"]}
    original["invocation_attempt_count"] = 2
    rows, coverage = usage.semantic_decision_accounting_inputs({"semantic_decision_dialogues": [original]})
    assert len(rows) == 1
    assert "dialogue_attempt_count_conflict" in {row["reason"] for row in coverage["issues"]}


def test_coverage_summary_does_not_export_arbitrary_receipt_fields():
    native = _native()
    attempt = _attempt(native)
    attempt["provider_receipt"].update(prompt="sensitive prompt", credential="sensitive secret")
    report = {"provider_invocations": [], "semantic_decision_dialogues": [_dialogue(attempt)]}
    final, measured = _surfaces(report)
    assert measured["total_tokens"] == 120
    assert "sensitive" not in json.dumps(final)
    assert "sensitive" not in json.dumps(measured)


def test_legacy_report_keeps_existing_usage_surface_and_no_dialogue_fields():
    report = {"provider_invocations": [_native()]}
    before = deepcopy(report)
    terminal._final_usage_accounting(report)
    measured = harbor.measured_usage(report)
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == before
    assert report["provider_usage_accounting"]["report_provider_total_tokens"] == 120
    assert measured["provider_calls"] == 1 and measured["total_tokens"] == 120
    assert "semantic_decision_coverage" not in measured
    assert "semantic_decision_coverage" not in report["provider_usage_accounting"]
    assert usage.collect_provider_usage_observation(report, {"schema": "unrelated@1"}) is False


def test_interruption_before_expected_dialogue_receipt_keeps_known_native_subtotal():
    report = {"provider_invocations": [_native()], "semantic_decision_dialogue_expected": True}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["supplied_receipt_usage_complete"] is True
    assert final["reconciliation"]["known_provider_token_subtotal"] == 120
    assert final["report_provider_total_tokens"] is None
    assert measured["all_invocations_receipted"] is False
    assert "expected_dialogue_receipt_missing" in {
        row["reason"] for row in final["semantic_decision_coverage"]["issues"]}


def test_earlier_dialogue_does_not_cover_later_interrupted_native_call():
    first, later = _native(), _native(2)
    report = {"provider_invocations": [first, later], "semantic_decision_dialogue_expected": True,
              "semantic_decision_dialogues": [_dialogue(_attempt(first))]}
    final, measured = _surfaces(report)
    assert final["reconciliation"]["known_provider_token_subtotal"] == 240
    assert final["report_provider_total_tokens"] is None
    assert measured["all_invocations_receipted"] is False
    assert "uncovered_dialogue_native_invocation" in {
        row["reason"] for row in final["semantic_decision_coverage"]["issues"]}
