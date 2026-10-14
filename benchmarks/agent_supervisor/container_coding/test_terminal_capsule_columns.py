"""Frozen columns controls, exact audit joins and conservative observations."""
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import terminal_semantic_metadata_policy as policy
from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import inventory
from benchmarks.agent_supervisor.container_coding.test_terminal_metadata_observation import report_for


MODE = "capsule-columns@1"


def columns_report(modes=(MODE,)):
    report = report_for(tuple("common-bindings@1" if mode == MODE else mode for mode in modes))
    report["semantic_metadata_view"]["requested_mode"] = MODE
    for row, mode in zip(report["provider_invocations"], modes):
        row["semantic_metadata_view"].update(requested_mode=MODE, selected_mode=mode)
    return report


def test_opt_in_is_frozen_in_configuration_and_default_stays_identical(tmp_path):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    baseline = benchmark.config_for(*args)
    explicit_default = benchmark.config_for(*args, semantic_metadata_view="legacy")
    assert baseline == explicit_default
    configured = benchmark.config_for(*args, semantic_metadata_view=MODE)
    expected = deepcopy(baseline)
    expected["agents"][0]["kwargs"]["semantic_metadata_view"] = MODE
    assert configured == expected
    assert policy.CAPSULE_COLUMNS_METADATA_VIEW == MODE
    assert policy.semantic_metadata_observation(MODE) == {
        "semantic_metadata_view": MODE, "semantic_metadata_token_savings_claimed": False,
        "semantic_metadata_authority": False}


@pytest.mark.parametrize("options", [
    {"arm": "no-index"}, {"transport_schema": "supervisor-semantic-router-input@2"},
    {"provider": "provider_free"}, {"provider": "doctor_candidate"}, {"provider": True},
])
def test_unqualified_arm_transport_or_provider_rejected(options):
    with pytest.raises(ValueError):
        policy.validate_semantic_metadata_view(MODE, **options)


@pytest.mark.parametrize("source_policy", ["source-input-raw@1", "source-input-compact@1",
    "source-input-ast-ref-raw@1", "source-input-ast-ref-compact@1"])
def test_source_input_composition_refused_before_task_files_are_read(tmp_path, source_policy):
    with pytest.raises(ValueError, match="capsule columns require unchanged inline source"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out", archive=tmp_path / "missing",
            arm="full", semantic_metadata_view=MODE, source_input_policy=source_policy)


def test_query_reference_composition_refused_before_task_files_are_read(tmp_path):
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out", archive=tmp_path / "missing",
            arm="full", semantic_metadata_view=MODE, query_reference_policy="required-instruction-ref@1")


@pytest.mark.parametrize("relative", [
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_columns.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/native_context_delivery.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/context_delivery_accounting.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_delivery_observation.py",
])
def test_new_profile_requires_current_codec_and_accounting_archive_bodies(relative):
    manifest = inventory()
    assert relative in policy.SOURCE_PATHS
    policy.require_semantic_metadata_archive(manifest, MODE)
    row = next(row for row in manifest["files"] if row["path"] == "source/" + relative)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_semantic_metadata_archive(manifest, MODE)


def test_model_router_argv_forwards_columns_and_keeps_other_inputs_literal():
    base = dict(router=Path("/router"), model="authored-model", reasoning="high", timeout=30,
        semantic_repository=Path("/repository"), doctor=None)
    native = implementation_argv(**base)
    selected = implementation_argv(**base, semantic_metadata_view=MODE)
    assert selected == native + ["--semantic-metadata-view", MODE]


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_provider_free_candidate_cannot_silently_ignore_columns_treatment(route):
    with pytest.raises(ValueError, match="signed semantic model-router"):
        implementation_argv(router=Path("/router"), model="authored-model", reasoning="high", timeout=30,
            semantic_repository=Path("/repository"), semantic_metadata_view=MODE,
            doctor={"route": route, "status": "candidate_ready", "provider_calls": 0})


@pytest.mark.parametrize("modes,status,effective", [
    ((MODE,), "verified_treatment", True), (("legacy",), "verified_fallback", False),
    ((MODE, "legacy"), "verified_mixed_selection", False),
])
def test_delivery_collector_reports_declared_columns_and_keeps_usage(modes, status, effective):
    report = columns_report(modes)
    original = deepcopy(report)
    observed = benchmark._metadata_dispatch_observation(report, MODE)
    assert observed["status"] == status
    assert observed["effective_treatment_observed"] is effective
    assert report == original
    assert observed["provider_usage_modified"] is observed["token_savings_established"] is False
    for row in report["provider_invocations"]:
        projected = comparison._semantic_metadata_observation(row["semantic_metadata_view"])
        assert projected["status"] == "reported"
        assert projected["requested_mode"] == MODE
        assert projected["provider_usage_reported"] is False


@pytest.mark.parametrize("change", ["request", "schema_list", "boolean_size", "missing_audit", "unreceipted"])
def test_invalid_or_incomplete_columns_delivery_never_establishes_treatment(change):
    report = columns_report()
    selection = report["provider_invocations"][0]["semantic_metadata_view"]
    if change == "request": selection["requested_mode"] = "common-bindings@1"
    elif change == "schema_list": selection["schema"] = []
    elif change == "boolean_size": selection["selected_complete_bytes"] = True
    elif change == "missing_audit": report.pop("context_input_audit")
    else: report["unreceipted_provider_attempt"] = {"status": "unknown"}
    result = benchmark._metadata_dispatch_observation(report, MODE)
    assert result["status"] == "unverified"
    assert result["effective_treatment_observed"] is False


def test_comparison_refuses_columns_mode_without_matching_request():
    selection = columns_report()["provider_invocations"][0]["semantic_metadata_view"]
    selection.pop("requested_mode")
    assert comparison._semantic_metadata_observation(selection)["status"] == "unavailable"
