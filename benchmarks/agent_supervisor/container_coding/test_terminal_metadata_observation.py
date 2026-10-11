"""Retained metadata treatment observations; no model or container calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task


MODE = "common-bindings@1"
FALLBACK = "complete_input_not_smaller_under_bytes_and_proxy"


def report_for(modes=(MODE,)):
    report = {"semantic_metadata_view": {"requested_mode": MODE}, "provider_invocations": [],
        "context_input_audit": {"status": "verified_all_observed_model_inputs", "read_errors": [],
            "coding_receipts_seen": len(modes), "usable_coding_receipts": len(modes),
            "all_observed_coding_inputs_verified": True, "invocations": [], "matches": []}}
    for index, mode in enumerate(modes):
        identity = "authored-coding-" + str(index)
        selected_hash = "b" * 64 if mode == MODE else "a" * 64
        selected_bytes = 100 if mode == MODE else 200
        selection = {"schema": "supervisor-semantic-metadata-selection@1", "selected_mode": mode,
            "fallback_reason": None if mode == MODE else FALLBACK, "candidate_only": True,
            "token_proxy": "utf8-bytes-ceil-div4@1", "view_receipt_sha256": "c" * 64,
            "native_complete_sha256": "a" * 64, "candidate_complete_sha256": "b" * 64,
            "selected_complete_sha256": selected_hash, "native_complete_bytes": 200,
            "candidate_complete_bytes": 100 if mode == MODE else 300, "selected_complete_bytes": selected_bytes,
            **{key: False for key in ("source_freshness_verified", "program_semantics_proved", "omission_authority",
                "proof_authority", "execution_authority", "completion_authority", "publication_authority")}}
        for prefix in ("native", "candidate", "selected"):
            selection[prefix + "_complete_proxy_tokens"] = (selection[prefix + "_complete_bytes"] + 3) // 4
        report["provider_invocations"].append({"phase": "coding", "invocation_id": identity,
            "model_prompt_sha256": selected_hash, "model_prompt_bytes": selected_bytes,
            "semantic_metadata_view": selection,
            "native_rollout_usage": {"usage": {"input_tokens": 29, "output_tokens": 3, "total_tokens": 32}}})
        report["context_input_audit"]["invocations"].append({"invocation_id": identity,
            "model_input_verified": True, "status": "verified_exact_model_input"})
        report["context_input_audit"]["matches"].append({"invocation_id": identity,
            "model_input_verified": True, "model_prompt_sha256": selected_hash,
            "model_prompt_bytes": selected_bytes,
            "model_input_checks": {"semantic_metadata_selection_receipt": True, "model_prompt_sha256": True}})
    return report


@pytest.mark.parametrize("modes,status,effective", [((MODE,), "verified_treatment", True),
    (("legacy",), "verified_fallback", False), ((MODE, "legacy"), "verified_mixed_selection", False)])
def test_treatment_fallback_and_mixed_observation_preserve_every_usage_row(modes, status, effective):
    report = report_for(modes)
    original = deepcopy(report)
    result = benchmark._metadata_dispatch_observation(report, MODE)
    assert result["requested_mode_matches_report"] is True
    assert result["coding_receipts_seen"] == len(modes)
    assert result["selected_modes"] == sorted(set(modes))
    assert result["status"] == status and result["effective_treatment_observed"] is effective
    assert result["all_selection_receipts_bound_to_model_inputs"] is True
    assert result["all_coding_inputs_verified_by_retained_audit"] is True
    assert result["fallback_reasons"] == ([FALLBACK] if "legacy" in modes else [])
    assert report == original  # Token counters and failed/retried attempts are never edited.
    assert result["source_audit_rerun"] is result["provider_usage_modified"] is result["token_savings_established"] is False


@pytest.mark.parametrize("declaration", [None, {}, {"requested_mode": "legacy"}, "common-bindings@1"])
def test_missing_or_changed_request_declaration_is_not_observed_treatment(declaration):
    report = report_for()
    report["semantic_metadata_view"] = declaration
    result = benchmark._metadata_dispatch_observation(report, MODE)
    assert result["requested_mode_matches_report"] is False
    assert result["effective_treatment_observed"] is False
    assert "requested_mode_report_mismatch" in result["problems"]


@pytest.mark.parametrize("change", ["missing_selection", "bad_mode", "bad_fallback", "selected_hash", "selected_bytes",
    "boolean_bytes", "same_receipt_only", "missing_audit", "wrong_audit_identity", "wrong_audit_digest",
    "wrong_audit_count", "missing_audit_check", "failed_audit_check", "conflicting_audit_match",
    "duplicate_invocation", "conflicting_duplicate", "unhashable_identity", "audit_read_error",
    "bad_mode_list", "boolean_proxy", "wrong_proxy", "boolean_audit_count", "unreceipted_attempt"])
def test_missing_conflicting_or_unverified_receipts_never_establish_treatment(change):
    report = report_for()
    row = report["provider_invocations"][0]
    selection = row["semantic_metadata_view"]
    audit = report["context_input_audit"]
    match = audit["matches"][0]
    if change == "missing_selection": row.pop("semantic_metadata_view")
    elif change == "bad_mode": selection["selected_mode"] = {}
    elif change == "bad_mode_list": selection["selected_mode"] = []
    elif change == "bad_fallback": selection["fallback_reason"] = FALLBACK
    elif change == "selected_hash": selection["selected_complete_sha256"] = "d" * 64
    elif change == "selected_bytes": selection["selected_complete_bytes"] = 123
    elif change == "boolean_bytes": selection["selected_complete_bytes"] = True
    elif change == "boolean_proxy": selection["selected_complete_proxy_tokens"] = True
    elif change == "wrong_proxy": selection["native_complete_proxy_tokens"] = 0
    elif change == "same_receipt_only":
        selection["candidate_complete_sha256"] = selection["selected_complete_sha256"] = row["model_prompt_sha256"] = "d" * 64
    elif change == "missing_audit": report.pop("context_input_audit")
    elif change == "wrong_audit_identity": audit["invocations"][0]["invocation_id"] = "foreign"
    elif change == "wrong_audit_digest": match["model_prompt_sha256"] = "d" * 64
    elif change == "wrong_audit_count": match["model_prompt_bytes"] = 99
    elif change == "missing_audit_check": match["model_input_checks"].pop("semantic_metadata_selection_receipt")
    elif change == "failed_audit_check": match["model_input_checks"]["semantic_metadata_selection_receipt"] = False
    elif change == "conflicting_audit_match":
        audit["matches"].append({**deepcopy(match), "model_prompt_sha256": "d" * 64})
    elif change == "duplicate_invocation": report["provider_invocations"].append(deepcopy(row))
    elif change == "conflicting_duplicate": report["provider_invocations"].append({**deepcopy(row), "model_prompt_bytes": 80})
    elif change == "unhashable_identity": row["invocation_id"] = []
    elif change == "audit_read_error": audit["read_errors"] = [{"kind": "receipt", "error_type": "InvalidReceipt"}]
    elif change == "boolean_audit_count": audit["usable_coding_receipts"] = True
    elif change == "unreceipted_attempt": report["unreceipted_provider_attempt"] = True
    result = benchmark._metadata_dispatch_observation(report, MODE)
    assert result["effective_treatment_observed"] is False
    assert result["all_coding_inputs_verified_by_retained_audit"] is False
    assert result["status"] == "unverified"


@pytest.mark.parametrize("population", [None, [], [{"phase": "planning"}], [True], [{}] * 4097])
def test_no_coding_population_is_unknown_instead_of_zero_savings(population):
    report = report_for()
    report["provider_invocations"] = population
    result = benchmark._metadata_dispatch_observation(report, MODE)
    assert result["effective_treatment_observed"] is False
    assert result["all_selection_receipts_bound_to_model_inputs"] is False
    assert "no_observed_coding_receipts" in result["problems"]


def retained_trial(tmp_path, report, mode=MODE):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", semantic_metadata_view=mode)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": benchmark.CLI_VERSION}
    if mode != "legacy": prepared["semantic_metadata_view"] = mode
    (output / "preparation.json").write_text(json.dumps(prepared))
    job = Path(config["jobs_dir"]) / config["job_name"]
    attempt = job / "authored-trial"
    (attempt / "agent").mkdir(parents=True)
    (job / "result.json").write_text("{}")
    (attempt / "result.json").write_text(json.dumps({"task_name": task.name,
        "config": {"task": {"path": str(task)}}, "verifier_result": {"rewards": {"reward": 0}}}))
    if report is not None:
        (attempt / "agent/supervisor-result.json").write_text(json.dumps(report))
    return output


@pytest.mark.parametrize("mode", [MODE, "legacy"])
def test_collect_keeps_valid_fallback_complete_and_original_attempt_usage(tmp_path, mode):
    report = report_for((mode,))
    result = benchmark.collect(retained_trial(tmp_path, report))
    assert result["complete_single_trial_receipt"] is True
    trial = result["trials"][0]
    assert trial["supervisor"] == report
    assert trial["semantic_metadata_dispatch"]["effective_treatment_observed"] is (mode == MODE)


@pytest.mark.parametrize("report", [None, {}, {"semantic_metadata_view": {"requested_mode": "legacy"}}])
def test_collect_refuses_opt_in_completion_on_missing_or_foreign_report_declaration(tmp_path, report):
    result = benchmark.collect(retained_trial(tmp_path, report))
    assert result["semantic_metadata_config_unchanged"] is True
    assert result["complete_single_trial_receipt"] is False
    assert result["trials"][0]["semantic_metadata_dispatch"]["requested_mode_matches_report"] is False


def test_legacy_collector_retains_historical_receipt_shape(tmp_path):
    result = benchmark.collect(retained_trial(tmp_path, {}, mode="legacy"))
    assert result["complete_single_trial_receipt"] is True
    assert "semantic_metadata_dispatch" not in result["trials"][0]
    assert "semantic_metadata_view" not in result
