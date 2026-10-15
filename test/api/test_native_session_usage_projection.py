"""Retained native accounting risks, with authored receipts and a real cold CLI.

These checks qualify local projection and publication only. They call no model
and do not establish a complete dispatch population, bill, or native token cap.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_session_usage_projection as projection
from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
    COMPLETION_BASIS, reconcile_router_cumulative_usage,
)
from ipfs_accelerate_py.cli_runtime.grok_native_usage import (
    grok_outcome_receipt, grok_usage_receipt, native_grok_observation,
)

MAX_COUNTER = (1 << 63) - 1
PRIVATE = "private-native-projection-marker"
WORKSPACE = "/private/native-projection-workspace"
MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.native_session_usage_projection"


def codex(*, wrapper="allocation-a", thread="native-thread-aaaaaaaa", inputs=100,
          outputs=20, count=1, complete=True):
    usage = {"input_tokens": inputs, "cached_input_tokens": inputs // 2,
        "output_tokens": outputs, "reasoning_output_tokens": outputs // 2,
        "total_tokens": inputs + outputs}
    return {
        "schema": "router-implementation-invocation@1", "invocation_id": wrapper,
        "provider": "codex_cli", "model": "gpt-5.6-sol", "workspace": WORKSPACE,
        "status": "provider_returned" if complete else "failed", "phase": "coding",
        "usage": {"thread_id": thread, "model_id": "gpt-5.6-sol"},
        "native_rollout_usage": {"schema": "native-codex-cumulative-usage@1",
            "thread_id": thread, "usage": usage, "usage_available": True,
            "rollout_sha256": hashlib.sha256(f"{thread}:{count}:{inputs}:{outputs}".encode()).hexdigest(),
            "observed_token_count_records": count, "task_complete_observed": complete,
            "cache_included_in_input": True, "completion_basis": COMPLETION_BASIS,
            "usage_lifecycle_complete": complete, "lifecycle_malformed_records": 0,
            "billing_total_verified": False, "dollar_cost": None},
        "completion_authority": False,
    }


def grok(*, wrapper="grok-allocation-a", session="grok-session-a", inputs=7,
         cached=2, cache_write=1, outputs=3, complete=True, failed=False,
         semantic=False):
    payload = {"stopReason": "end_turn", "usage": {
        "input_tokens": inputs, "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": cache_write, "output_tokens": outputs,
        "total_tokens": inputs + cached + cache_write + outputs}}
    if complete is not None:
        payload["usage_is_incomplete"] = not complete
    if failed:
        payload["type"] = "error"
    native = native_grok_observation(payload, exit_code=1 if failed else 0)
    row = {"schema": "router-semantic-decision-provider@1" if semantic else "router-implementation-invocation@1",
        "call_id" if semantic else "invocation_id": wrapper,
        "provider": "grok_cli", "model": "grok-4.7", "workspace": WORKSPACE,
        "status": "failed" if failed else "provider_returned", "phase": "coding",
        "usage": {"model_id": "grok-4.7"},
        "native_rollout_usage": grok_usage_receipt(native),
        "native_provider_outcome": grok_outcome_receipt(native),
        "completion_authority": False}
    if session is not None:
        row["usage"]["session_id"] = session
    if semantic:
        row.update(session_id="requested-dialogue-session", generation=0,
            binding_sha256="a" * 64, model_prompt_sha256="b" * 64,
            provider_called=True)
    return row


def no_authority(value):
    assert value["schema"] == "retained-native-usage-observation@1"
    assert value["task_attempt_total"] is None
    assert value["campaign_usage_complete"] is None
    for name in ("billing_total_verified", "native_cumulative_cap_enforced",
                 "completion_authority", "execution_authority"):
        assert value[name] is False


def reconcile(rows):
    original = deepcopy(rows)
    result = projection.reconcile_native_usage_observations(rows)
    assert rows == original
    no_authority(result)
    return result


def test_codex_uses_latest_cumulative_session_without_adding_cache_reasoning_or_local_counts():
    first = codex(inputs=100, outputs=20, count=1)
    later = codex(wrapper="allocation-b", inputs=150, outputs=30, count=2)
    first.update(tool_tokens=999, prompt_selection={"selected_content_tokens": 888})
    result = reconcile([first, later])
    native = result["codex_cumulative"]
    assert native == reconcile_router_cumulative_usage([first, later])
    assert native["native_sessions_observed"] == 1
    assert native["known_provider_token_subtotal"] == native["provider_total_tokens"] == 180
    assert native["known_counter_subtotals"]["cached_input_tokens"] == 75
    assert native["known_counter_subtotals"]["reasoning_output_tokens"] == 15


@pytest.mark.parametrize("semantic", [False, True])
def test_exact_grok_wrapper_duplicate_ignores_collector_phase_and_keeps_one_charge(semantic):
    first = grok(semantic=semantic)
    printed_again = deepcopy(first)
    printed_again["phase"] = "planning"
    result = reconcile([first, printed_again])
    assert result["observations_received"] == 2
    assert result["duplicate_wrapper_observations"] == 1
    bucket = result["grok_final_envelopes"]
    assert len(bucket["rows"]) == 1
    assert bucket["rows"][0]["normalized_input_tokens"] == 10
    assert bucket["rows"][0]["normalized_total_tokens"] == 13
    assert bucket["diagnostic_subtotals"]["normalized_total_tokens"] == 13


@pytest.mark.parametrize("change", ["counter", "model", "workspace", "provider"])
def test_contradictory_same_wrapper_cannot_select_a_favorable_receipt(change):
    first = grok()
    changed = deepcopy(first)
    if change == "counter":
        changed["native_rollout_usage"]["usage"]["input_tokens"] = 8
        changed["native_rollout_usage"]["usage"]["total_tokens"] = 14
    elif change == "model":
        changed["model"] = "other-model"
    elif change == "workspace":
        changed["workspace"] = "/different/workspace"
    else:
        changed["provider"] = "codex_cli"
    result = reconcile([first, changed])
    assert result["issues"]
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False
    assert all(row["normalized_total_tokens"] is None
               for row in result["grok_final_envelopes"]["rows"])


def test_identical_counts_and_usage_digest_do_not_merge_distinct_reported_grok_sessions():
    first = grok(wrapper="wrapper-a", session="reported-session-a")
    second = grok(wrapper="wrapper-b", session="reported-session-b")
    assert first["native_rollout_usage"]["usage_observation_sha256"] == second["native_rollout_usage"]["usage_observation_sha256"]
    result = reconcile([first, second])
    rows = result["grok_final_envelopes"]["rows"]
    assert len(rows) == 2
    assert len({row["reported_session_identity_sha256"] for row in rows}) == 2
    assert [row["normalized_total_tokens"] for row in rows] == [13, 13]
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is True
    assert result["grok_final_envelopes"]["reported_identity_groups"] == 2
    assert result["grok_final_envelopes"]["diagnostic_subtotals"]["normalized_total_tokens"] == 26


def test_one_reported_grok_session_under_distinct_wrappers_is_ambiguous_even_at_equal_counts():
    first = grok(wrapper="wrapper-a")
    second = grok(wrapper="wrapper-b")
    result = reconcile([first, second])
    rows = result["grok_final_envelopes"]["rows"]
    assert len(rows) == 2
    assert len({row["reported_session_identity_sha256"] for row in rows}) == 1
    assert all(row["status"] == "conflict" and row["normalized_total_tokens"] is None for row in rows)
    assert result["grok_final_envelopes"]["ambiguous_reported_identity_groups"] == 1
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False


@pytest.mark.parametrize("semantic", [False, True])
def test_missing_native_reported_session_keeps_counters_but_not_an_aggregate(semantic):
    row = grok(session=None, semantic=semantic)
    # Semantic outer IDs and the usage digest are never native-session fallbacks.
    row.update(session_id="requested-session-only", call_id="requested-call-only")
    result = reconcile([row])
    bucket = result["grok_final_envelopes"]
    native = bucket["rows"][0]
    assert native["reported_session_identity_sha256"] is None
    assert native["status"] == "identity_unavailable"
    assert native["usage"]["input_tokens"] == 7
    assert native["usage"]["cached_input_tokens"] == 2
    assert bucket["missing_reported_identity_count"] == 1
    assert bucket["diagnostic_subtotal_available"] is False


@pytest.mark.parametrize("thread", ["different-reported-thread", True, "", "bad\nidentity"])
def test_conflicting_or_invalid_reported_native_session_aliases_are_not_guessed(thread):
    row = grok()
    row["usage"]["thread_id"] = thread
    result = reconcile([row])
    assert result["issues"] or result["grok_final_envelopes"]["rows"][0]["reasons"]
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False


@pytest.mark.parametrize("complete", [None, False, True])
def test_completeness_is_separate_from_observed_normalized_diagnostics(complete):
    result = reconcile([grok(complete=complete)])
    row = result["grok_final_envelopes"]["rows"][0]
    assert row["usage_complete_observed"] is complete
    assert row["normalized_input_tokens"] == 10 and row["normalized_total_tokens"] == 13
    assert row["usage"]["reasoning_output_tokens"] is None
    if complete is None:
        assert row["observed_native_scope_complete"] is None
    else:
        assert row["observed_native_scope_complete"] is complete
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is True


def test_known_failed_grok_charge_is_retained_without_completion_or_task_authority():
    result = reconcile([grok(complete=False, failed=True)])
    row = result["grok_final_envelopes"]["rows"][0]
    assert row["usage"]["total_tokens"] == row["normalized_total_tokens"] == 13
    assert row["task_complete_observed"] is False
    assert row["observed_native_scope_complete"] is False


@pytest.mark.parametrize("counter", ["input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "total_tokens", "reasoning_output_tokens"])
@pytest.mark.parametrize("invalid", [-1, True, 1 << 63, 1.5])
def test_invalid_counter_categories_cannot_escape_as_valid_diagnostic_totals(counter, invalid):
    row = grok()
    row["native_rollout_usage"]["usage"][counter] = invalid
    result = reconcile([row])
    native = result["grok_final_envelopes"]["rows"][0]
    assert native["usage"][counter] is None
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False
    assert native["reasons"] or result["issues"]


@pytest.mark.parametrize("category", ["input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "total_tokens"])
def test_missing_required_native_categories_remain_unknown(category):
    row = grok()
    del row["native_rollout_usage"]["usage"][category]
    result = reconcile([row])
    assert result["grok_final_envelopes"]["rows"][0]["usage"][category] is None
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False


@pytest.mark.parametrize("damage", ["missing", "invalid", "conflicting_total"])
def test_true_completeness_flags_cannot_cover_bad_required_counter_evidence(damage):
    row = grok(complete=True)
    if damage == "missing":
        del row["native_rollout_usage"]["usage"]["cache_write_input_tokens"]
    elif damage == "invalid":
        row["native_rollout_usage"]["usage"]["output_tokens"] = True
    else:
        row["native_rollout_usage"]["usage"]["total_tokens"] = 999
    result = reconcile([row])
    assert result["grok_final_envelopes"]["rows"][0]["observed_native_scope_complete"] is not True
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False


def test_huge_invalid_counter_is_closed_without_decimal_serialization_or_echo():
    row = grok()
    row["native_rollout_usage"]["usage"]["input_tokens"] = 1 << 20000
    result = reconcile([row])
    native = result["grok_final_envelopes"]["rows"][0]
    assert native["usage"]["input_tokens"] is None
    assert native["status"] == "conflict"
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False
    json.dumps(result, allow_nan=False)


def test_router_nested_self_reference_is_ignored_without_recursive_child_dispatch():
    row = grok()
    row["native_invocation"] = row
    result = projection.reconcile_native_usage_observations([row])
    no_authority(result)
    assert row["native_invocation"] is row
    assert result["observations_received"] == 1
    assert len(result["grok_final_envelopes"]["rows"]) == 1
    assert result["grok_final_envelopes"]["rows"][0]["normalized_total_tokens"] == 13
    json.dumps(result, allow_nan=False)


def test_codex_lone_surrogate_workspace_becomes_a_closed_conflict():
    row = codex()
    row["workspace"] = "/private/invalid-\ud800-workspace"
    result = reconcile([row])
    assert result["issues"] or result["codex_cumulative"]["sessions"][0]["reasons"]
    assert result["codex_cumulative"]["provider_total_tokens"] is None
    assert result["codex_cumulative"]["supplied_receipt_usage_complete"] is False
    assert "\\ud800" not in json.dumps(result, allow_nan=False)


def test_inconsistent_native_total_is_not_repaired_from_categories():
    row = grok()
    row["native_rollout_usage"]["usage"]["total_tokens"] = 999
    result = reconcile([row])
    native = result["grok_final_envelopes"]["rows"][0]
    assert native["normalized_total_tokens"] is None
    assert native["usage"]["total_tokens"] == 999
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False


def test_checked_grok_aggregate_cannot_overflow_even_when_each_envelope_is_valid():
    amount = MAX_COUNTER // 2 + 1
    rows = [grok(wrapper=f"wrapper-{index}", session=f"session-{index}", inputs=amount,
                 cached=0, cache_write=0, outputs=0) for index in range(2)]
    result = reconcile(rows)
    assert [row["normalized_total_tokens"] for row in result["grok_final_envelopes"]["rows"]] == [amount, amount]
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False
    assert result["issues"] or any(row["reasons"] for row in result["grok_final_envelopes"]["rows"])


def test_new_wrapper_quarantines_derived_codex_overflow_without_changing_existing_reconciler():
    amount = MAX_COUNTER // 2 + 1
    rows = [codex(wrapper=f"allocation-{index}", thread=f"native-thread-{index:016d}", inputs=amount,
                  outputs=0) for index in range(2)]
    original = deepcopy(rows)
    established = reconcile_router_cumulative_usage(rows)
    assert established["provider_total_tokens"] == amount * 2
    result = reconcile(rows)
    assert result["codex_cumulative"]["provider_total_tokens"] is None
    assert result["codex_cumulative"]["supplied_receipt_usage_complete"] is False
    assert result["issues"]
    assert rows == original
    assert reconcile_router_cumulative_usage(rows) == established


def test_mixed_provider_scopes_do_not_acquire_a_combined_task_total():
    result = reconcile([codex(), grok()])
    assert result["codex_cumulative"]["known_provider_token_subtotal"] == 120
    assert result["grok_final_envelopes"]["rows"][0]["normalized_total_tokens"] == 13
    no_authority(result)


def test_nested_semantic_codex_and_printed_native_copy_are_one_retained_charge():
    native = codex()
    provider = {"schema": "router-semantic-decision-provider@1", "call_id": "call-a",
        "provider": "codex_cli", "model": native["model"], "workspace": WORKSPACE,
        "provider_called": True, "native_invocation": deepcopy(native)}
    report = {"provider_invocations": [native], "semantic_decision_dialogues": [{
        "schema": "supervisor-semantic-decision-dialogue@1", "session_id": "dialogue-a",
        "attempts": [{"call_id": "call-a", "provider_receipt": provider}]}]}
    original = deepcopy(report)
    result = projection.project_retained_native_usage(report)
    no_authority(result)
    assert result["codex_cumulative"]["native_sessions_observed"] == 1
    assert result["codex_cumulative"]["known_provider_token_subtotal"] == 120
    assert result["report_observation"] == {"provider_invocations_received": 1,
        "dialogues_received": 1, "attempts_received": 1,
        "unreceipted_provider_attempt_reported": False}
    assert report == original


def test_semantic_grok_is_consumed_and_cached_summaries_never_fill_dispatch_holes():
    semantic = grok(semantic=True)
    report = {"semantic_decision_dialogues": [{
        "schema": "supervisor-semantic-decision-dialogue@1", "session_id": "dialogue-a",
        "attempts": [{"call_id": semantic["call_id"], "provider_receipt": semantic},
                     {"call_id": "missing-provider-receipt", "provider_receipt": None}]}],
        "unreceipted_provider_attempt": True,
        "provider_usage_accounting": {"total_tokens": 999999, "provider_usage_complete": True}}
    original = deepcopy(report)
    result = projection.project_retained_native_usage(report)
    no_authority(result)
    assert len(result["grok_final_envelopes"]["rows"]) == 1
    assert result["grok_final_envelopes"]["rows"][0]["normalized_total_tokens"] == 13
    assert result["report_observation"]["attempts_received"] == 2
    assert result["report_observation"]["unreceipted_provider_attempt_reported"] is True
    assert result["issues"]
    assert report == original


def test_private_receipt_subtrees_and_workspace_or_grok_ids_do_not_leave_projection():
    row = grok()
    row.update(prompt=PRIVATE, response=PRIVATE, auth={"key": PRIVATE}, tools=[PRIVATE])
    row["native_rollout_usage"]["provider_transcript"] = PRIVATE
    result = reconcile([row])
    public = json.dumps(result, allow_nan=False)
    for forbidden in (PRIVATE, WORKSPACE, "grok-session-a", "grok-allocation-a"):
        assert forbidden not in public


@pytest.mark.parametrize("bad", [None, True, {}, "rows", 1])
def test_reconcile_requires_an_explicit_sequence_population(bad):
    with pytest.raises(ValueError):
        projection.reconcile_native_usage_observations(bad)


def test_tuple_population_is_supported_without_mutation():
    result = reconcile((grok(),))
    assert len(result["grok_final_envelopes"]["rows"]) == 1


@pytest.mark.parametrize("bad", [None, True, [], "report"])
def test_report_requires_a_mapping_population(bad):
    with pytest.raises(ValueError):
        projection.project_retained_native_usage(bad)


@pytest.mark.parametrize("report", [{"provider_invocations": {}},
    {"semantic_decision_dialogues": {}},
    {"semantic_decision_dialogues": [{"attempts": {}}]}])
def test_malformed_declared_report_populations_refuse_instead_of_becoming_empty(report):
    with pytest.raises(ValueError):
        projection.project_retained_native_usage(report)


def test_individual_bad_rows_remain_closed_issues_instead_of_disappearing():
    result = reconcile([None, {}, {"schema": "unknown", "prompt": PRIVATE}, grok()])
    assert result["observations_received"] == 4
    assert result["issues"]
    assert len(result["grok_final_envelopes"]["rows"]) == 1
    assert PRIVATE not in json.dumps(result)


def test_raw_and_aggregate_observation_bounds_apply_before_deduplication():
    with pytest.raises(ValueError):
        projection.reconcile_native_usage_observations([grok()] * 10001)
    with pytest.raises(ValueError):
        projection.project_retained_native_usage({"provider_invocations": [{}] * 5001,
            "semantic_decision_dialogues": [{"attempts": [{}] * 5000}]})


def cli(tmp_path, report, output):
    checkout = Path(projection.__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(checkout), environment.get("PYTHONPATH")]))
    environment.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI="0")
    return subprocess.run([sys.executable, "-m", MODULE, "--report", str(report),
        "--output", str(output)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30, check=False)


def test_real_cold_cli_pins_exact_input_bytes_and_publishes_a_fresh_public_sidecar(tmp_path):
    report, output = tmp_path / "report.json", tmp_path / "projection.json"
    raw = (json.dumps({"provider_invocations": [grok()], "prompt": PRIVATE}, indent=2) + "\n").encode()
    report.write_bytes(raw)
    result = cli(tmp_path, report, output)
    assert result.returncode == 0, result.stderr
    sidecar = json.loads(output.read_bytes())
    summary = json.loads(result.stdout)
    assert sidecar["schema"] == "retained-native-usage-report@1"
    assert summary["schema"] == "retained-native-usage-cli-summary@1"
    for value in (sidecar, summary):
        assert value["input_report_sha256"] == hashlib.sha256(raw).hexdigest()
        assert value["input_report_bytes"] == len(raw)
    assert summary["observations_received"] == 1
    assert sidecar["projection"]["grok_final_envelopes"]["rows"][0]["normalized_total_tokens"] == 13
    no_authority(sidecar["projection"])
    assert report.read_bytes() == raw
    assert PRIVATE not in output.read_text() + result.stdout + result.stderr
    assert WORKSPACE not in output.read_text() + result.stdout + result.stderr


def test_real_cli_never_overwrites_an_existing_output(tmp_path):
    report, output = tmp_path / "report.json", tmp_path / "existing.json"
    report.write_text(json.dumps({"provider_invocations": [grok()]}))
    output.write_text(PRIVATE)
    result = cli(tmp_path, report, output)
    assert result.returncode != 0
    assert output.read_text() == PRIVATE
    assert PRIVATE not in result.stdout + result.stderr


@pytest.mark.parametrize("raw", [
    '{"provider_invocations":[],"provider_invocations":[{"prompt":"' + PRIVATE + '"}]}',
    '{"provider_invocations":[],"private":NaN}',
    '{"provider_invocations":[],"private":Infinity}',
    '{"provider_invocations":[],"private":1e999}',
    '{"provider_invocations":[],"private":"' + PRIVATE + '"',
])
def test_real_cli_refuses_duplicate_nonfinite_or_malformed_json_without_leaking_input(tmp_path, raw):
    report, output = tmp_path / "report.json", tmp_path / "absent.json"
    report.write_text(raw)
    result = cli(tmp_path, report, output)
    assert result.returncode != 0 and not output.exists()
    assert PRIVATE not in result.stdout + result.stderr


def test_real_cli_refuses_symlink_input_and_does_not_follow_symlink_output(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"provider_invocations": [grok()]}))
    linked = tmp_path / "linked.json"
    linked.symlink_to(source)
    absent = tmp_path / "absent.json"
    result = cli(tmp_path, linked, absent)
    assert result.returncode != 0 and not absent.exists()
    target, output = tmp_path / "target.json", tmp_path / "output.json"
    target.write_text(PRIVATE)
    output.symlink_to(target)
    result = cli(tmp_path, source, output)
    assert result.returncode != 0 and output.is_symlink() and target.read_text() == PRIVATE
    assert PRIVATE not in result.stdout + result.stderr


def test_real_cli_applies_32_mib_input_bound_before_json_parsing(tmp_path):
    report, output = tmp_path / "oversized.json", tmp_path / "absent.json"
    with report.open("wb") as stream:
        stream.seek(32 * 1024 * 1024)
        stream.write(b"x")
    result = cli(tmp_path, report, output)
    assert result.returncode != 0 and not output.exists()


@pytest.mark.parametrize("shape", ["depth", "nodes"])
def test_real_cli_bounds_json_structure_even_in_ignored_report_fields(tmp_path, shape):
    report, output = tmp_path / "structured.json", tmp_path / "absent.json"
    if shape == "depth":
        ignored = "[" * 40 + "0" + "]" * 40
    else:
        ignored = "[" + ",".join(["0"] * 1000001) + "]"
    report.write_text('{"provider_invocations":[],"ignored":' + ignored + '}')
    result = cli(tmp_path, report, output)
    assert result.returncode != 0 and not output.exists()


def test_empty_retained_report_is_unknown_not_a_provider_free_zero_total():
    result = projection.project_retained_native_usage({})
    no_authority(result)
    assert result["observations_received"] == 0
    assert result["grok_final_envelopes"]["diagnostic_subtotal_available"] is False
