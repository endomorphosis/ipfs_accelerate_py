"""Actual indexed/router comparison with independently pinned report controls."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_capsule_columns_experiment as experiment


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("columns-offline-fixtures")
    report = experiment.run_experiment(scratch_root=scratch)
    assert report["qualified"] is True, {key: report.get(key) for key in (
        "status", "tokenizer", "source_pins_unchanged", "private_fixture_cleanup_complete",
        "private_fixture_cleanup_reason", "private_fixture_observed_prefix_files",
        "private_fixture_observed_prefix_bytes")}
    assert list(scratch.iterdir()) == []
    return report


def _reseal(report):
    report.pop("report_sha256", None)
    report["report_sha256"] = experiment._sha(experiment._wire(report))
    return experiment.report_sha256(report)


def test_real_indexed_population_and_complete_router_pairs(measured):
    result = experiment.validate_report(measured, expected_report_sha256=experiment.report_sha256(measured))
    assert result["valid"] is True
    assert [(row["function_count"], row["capsule_budget"]) for row in measured["cases"]] == [
        (count, budget) for count in (4, 8) for budget in experiment.BUDGETS]
    assert all(all(row["checks"].values()) for row in measured["cases"])
    largest = next(row for row in measured["cases"] if row["function_count"] == 8 and row["capsule_budget"] == "legacy")
    assert largest["selected_capsule_count"] >= 4
    assert measured["private_fixture_cleanup_complete"] is True
    assert measured["live_provider_calls"] == measured["model_calls"] == 0
    assert all(arm["scripted_provider_calls"] == 1 for row in measured["cases"] for arm in row["arms"])


def test_same_canonical_sources_and_constraint_scope_across_budgets(measured):
    for count in (4, 8):
        rows = [row for row in measured["cases"] if row["function_count"] == count]
        assert all(row["source_sha256"] == rows[0]["source_sha256"] for row in rows)
        assert all(row["constraints_sha256"] == rows[0]["constraints_sha256"] for row in rows)
        assert all(row["semantic_root_cid"] == rows[0]["semantic_root_cid"] for row in rows)
        assert all(row["required_raw_paths"] == ["instruction.md", "smoke.py"] for row in rows)


def test_whole_authored_trace_charges_same_expansion_once(measured):
    for row in measured["cases"]:
        shared = row["host_authored_expansion_workload"]
        assert shared["message_count"] == 4
        for arm in row["arms"]:
            assert arm["complete_authored_trace"]["utf8_bytes"] == arm["complete_model_prompt"]["utf8_bytes"] + shared["utf8_bytes"]
            assert arm["complete_authored_trace"]["local_tokenizer_proxy_tokens"] == arm["complete_model_prompt"]["local_tokenizer_proxy_tokens"] + shared["local_tokenizer_proxy_tokens"]
        assert row["selected_token_proxy_delta"] <= 0
        if row["arms"][1]["selected_mode"] == "legacy":
            assert row["arms"][1]["complete_model_prompt"] == row["arms"][0]["complete_model_prompt"]
            assert row["selected_token_proxy_delta"] == 0


def test_fallback_is_not_reported_as_forced_gain(measured):
    for row in measured["cases"]:
        assert row["arms"][1]["selected_mode"] in {"legacy", "capsule-columns@1"}
        if row["arms"][1]["selected_mode"] == "capsule-columns@1":
            assert row["arms"][1]["complete_model_prompt"] == row["candidate_complete_model_prompt"]
            assert row["selected_token_proxy_delta"] < 0


def test_cached_tokenizer_is_explicit_proxy_and_output_has_no_fixture_text(measured):
    assert measured["tokenizer"]["encoding"] == "o200k_base"
    assert measured["tokenizer"]["qualification"] == "estimate"
    assert measured["tokenizer"]["download_performed"] is False
    assert measured["grok_tokenizer_compatibility_verified"] is False
    assert measured["deployed_worker_retrieval_verified"] is measured["provider_ingestion_verified"] is False
    text = json.dumps(measured, ensure_ascii=False)
    assert "def helper_" not in text and "return value +" not in text
    assert "Preserve exact defaults and café" not in text
    assert "provider_usage_measured" in text and measured["raw_content_exported"] is False


@pytest.mark.parametrize("change", ["drop_case", "change_metric", "change_checks", "claim_ingestion"])
def test_independent_report_pin_rejects_resealed_changes(measured, change):
    report = deepcopy(measured)
    if change == "drop_case": report["cases"].pop()
    elif change == "change_metric": report["cases"][0]["arms"][0]["complete_model_prompt"]["utf8_bytes"] += 1
    elif change == "change_checks": report["cases"][0]["checks"]["source_bytes_unchanged"] = False
    else: report["provider_ingestion_verified"] = True
    _reseal(report)
    with pytest.raises(ValueError, match="independent_report_pin"):
        experiment.validate_report(report, expected_report_sha256=experiment.report_sha256(measured))


@pytest.mark.parametrize("change", ["sum", "boolean_metric", "delta", "reorder_cases", "authority", "missing_cleanup", "dropped_checks"])
def test_rebound_report_still_rejects_inconsistent_scope_and_arithmetic(measured, change):
    report = deepcopy(measured)
    if change == "sum": report["cases"][0]["host_authored_expansion_workload"]["utf8_bytes"] += 1
    elif change == "boolean_metric": report["cases"][0]["candidate_complete_model_prompt"]["utf8_bytes"] = True
    elif change == "delta": report["cases"][0]["selected_token_proxy_delta"] += 1
    elif change == "reorder_cases": report["cases"].reverse()
    elif change == "authority": report["completion_authority"] = True
    elif change == "dropped_checks": report["cases"][0]["checks"] = {}
    else: report["private_fixture_cleanup_complete"] = False
    pin = _reseal(report)
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256=pin)


@pytest.mark.parametrize("count", [-1, True, 1 << 63])
def test_invalid_local_counter_outputs_refuse(count):
    with pytest.raises(ValueError, match="invalid_local_token_count"):
        experiment._measure("literal complete message", lambda _: count)


def test_no_proxy_counter_stays_unknown():
    measured = experiment._trace(["complete message"], None)
    assert measured["local_tokenizer_proxy_tokens"] is None


def test_measurement_bound_refuses_instead_of_truncating():
    with pytest.raises(ValueError, match="measurement_text_exceeds_bound"):
        experiment._measure("x" * (experiment.MAX_MESSAGE_BYTES + 1), lambda _: 1)


def test_private_fixture_cleanup_preserves_foreign_replacement(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir(); (foreign / "keep").write_text("foreign bytes")
    with experiment._private_fixture(tmp_path) as (fixture, cleanup):
        displaced = tmp_path / "displaced-owned"
        fixture.rename(displaced)
        fixture.symlink_to(foreign, target_is_directory=True)
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert (foreign / "keep").read_text() == "foreign bytes"
    assert fixture.is_symlink() and displaced.is_dir()


def test_private_fixture_cleanup_bound_keeps_unqualified_owned_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "MAX_FIXTURE_BYTES", 1)
    with experiment._private_fixture(tmp_path) as (fixture, cleanup):
        (fixture / "owned").write_text("bounded failure retained")
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert cleanup["private_fixture_cleanup_reason"] == "private_fixture_byte_bound_exceeded"
    assert cleanup["private_fixture_observed_prefix_bytes"] > 1
    assert (fixture / "owned").is_file()


def test_cleanup_failure_exports_closed_reason_without_exception_path(tmp_path, monkeypatch):
    def unavailable(_):
        raise ValueError("private foreign path must never be exported")
    monkeypatch.setattr(experiment.shutil, "rmtree", unavailable)
    with experiment._private_fixture(tmp_path) as (fixture, cleanup):
        (fixture / "owned").write_text("authored fixture")
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert cleanup["private_fixture_cleanup_reason"] == "owned_fixture_cleanup_unavailable"
    assert cleanup["private_fixture_observed_prefix_files"] == 1
    assert cleanup["private_fixture_observed_prefix_bytes"] == len("authored fixture")
    assert cleanup["private_fixture_storage_bound_bytes"] == 128 * 1024 * 1024
    assert "foreign path" not in json.dumps(cleanup)


def test_scratch_symlink_refuses_without_fixture_creation(tmp_path):
    alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical_scratch_parent"):
        with experiment._private_fixture(alias):
            pytest.fail("symlink scratch parent admitted")


def test_source_fixture_is_authored_and_bounded():
    for count in experiment.FUNCTION_COUNTS:
        source = experiment._source(count)
        assert source.count("def helper_") == count
        assert len(source.encode()) < 8192
        assert "unsupported, unknown, café" in source
