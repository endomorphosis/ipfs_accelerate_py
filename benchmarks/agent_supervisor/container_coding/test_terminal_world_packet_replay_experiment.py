"""Real finite issued-packet workload and independently retained input refusals."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_world_packet_replay_experiment as experiment
from benchmarks.agent_supervisor.container_coding.terminal_context_audit import collect_terminal_context_audit


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("world-replay-private-fixtures")
    artifacts = tmp_path_factory.mktemp("world-replay-retained")
    report = experiment.run_experiment(scratch_root=scratch, artifact_root=artifacts)
    assert report["qualified"] is True, {key: report.get(key) for key in (
        "status", "source_pins_unchanged", "private_fixture_cleanup_complete",
        "private_fixture_cleanup_reason", "private_fixture_observed_prefix_files",
        "private_fixture_observed_prefix_bytes", "checks", "audit")}
    retained = list(scratch.iterdir())
    assert len(retained) == 1 and retained[0].is_dir() and not retained[0].is_symlink()
    assert list(retained[0].iterdir()) == []
    assert report["private_fixture_cleanup_scope"] == experiment.CLEANUP_SCOPE
    assert report["private_fixture_root_removed"] is False
    return report, scratch, artifacts


def _receipts(measured):
    report, _, root = measured
    result = []
    for row in report["invocations"]:
        descriptor = row["retention"]["invocation"]
        raw = (root / descriptor["name"]).read_bytes()
        assert len(raw) == descriptor["bytes"] and hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
        stored = json.loads(raw)
        result.append({**stored["invocation"], "phase": "coding", "world_packet_artifacts": row["retention"]})
    return result


def _collect(measured, receipts, artifact_root):
    _, scratch, _ = measured
    return collect_terminal_context_audit(state=artifact_root / "absent-live-state",
        receipts=receipts, workspace_root=scratch,
        world_packet_artifact_root=artifact_root, timeout_seconds=90)


def _copy_artifacts(measured, tmp_path):
    root = tmp_path / "independently-selected-artifacts"
    root.mkdir(mode=0o700)
    for row in measured[0]["invocations"]:
        for key in ("packet", "invocation"):
            name = row["retention"][key]["name"]
            shutil.copyfile(measured[2] / name, root / name)
            (root / name).chmod(0o600)
    return root


def _reseal(report):
    report.pop("report_sha256", None)
    report["report_sha256"] = experiment._sha(experiment._wire(report))
    return experiment.report_sha256(report)


def test_actual_native_two_generation_dispatch_retention_and_terminal_audit(measured):
    report, _, _ = measured
    assert experiment.validate_report(report, expected_report_sha256=experiment.report_sha256(report))["valid"] is True
    assert set(report["checks"]) == experiment.CHECKS and all(report["checks"].values())
    assert report["counters"] == {"source_generations": 2, "native_signed_checks": 3,
        "real_child_mcp_edits": 1, "authored_provider_calls": 2, "stale_source_refusals": 1,
        "newer_owner_refusals": 1, "retained_successful_invocations": 2}
    assert report["audit"]["status"] == "verified_all_observed_model_inputs"
    assert report["audit"]["invocation_count"] == report["audit"]["match_count"] == 2
    assert report["source_edit_profile"] == "existing_guarded_full_file_read_and_explicit_authored_answer_symbol"
    assert report["live_provider_calls"] == report["model_calls"] == 0
    assert report["task_completed"] is False


def test_exact_artifact_reconstruction_survives_absent_owner_and_source(measured):
    receipts = _receipts(measured)
    assert all(not Path(row["workspace"]).exists() for row in receipts)
    audit = _collect(measured, receipts, measured[2])
    assert audit["all_observed_coding_inputs_verified"] is True
    for row in measured[0]["invocations"]:
        match = next(item for item in audit["matches"] if item["invocation_id"] == row["invocation_id"])
        assert match["model_prompt_sha256"] == row["model_prompt_sha256"]
        assert match["model_prompt_bytes"] == row["model_prompt_bytes"]
        assert match["verification_scope"] == "historical_world_packet_input_reconstruction"


def test_qualification_report_is_content_free_and_usage_unknown(measured):
    report = measured[0]
    text = json.dumps(report)
    assert "def answer" not in text and "Change answer()" not in text
    assert str(measured[1]) not in text and str(measured[2]) not in text
    assert report["tokenizer_measurement"] is report["whole_session_token_savings"] is None
    assert report["raw_content_exported"] is report["provider_usage_measured"] is False
    for key in ("proof_authority", "execution_authority", "completion_authority", "owner_signature_replayed",
                "source_currentness_verified", "token_savings_established", "provider_ingestion_verified"):
        assert report[key] is False


@pytest.mark.parametrize("change", ["drop", "counter", "check", "authority"])
def test_external_full_report_pin_refuses_resealed_changes(measured, change):
    report = deepcopy(measured[0])
    if change == "drop": report["invocations"].pop()
    elif change == "counter": report["counters"]["authored_provider_calls"] += 1
    elif change == "check": report["checks"]["stale_source_dispatch_refused"] = False
    else: report["completion_authority"] = True
    _reseal(report)
    with pytest.raises(ValueError, match="independent_report_pin_differs"):
        experiment.validate_report(report, expected_report_sha256=experiment.report_sha256(measured[0]))


@pytest.mark.parametrize("change", ["population", "checks", "authority", "savings", "cleanup", "boolean_counter", "duplicate_identity", "audit_hash", "audit_authority", "context_close"])
def test_rebound_report_cannot_assert_inconsistent_scope(measured, change):
    report = deepcopy(measured[0])
    if change == "population": report["invocations"].pop()
    elif change == "checks": report["checks"] = {}
    elif change == "authority": report["proof_authority"] = True
    elif change == "savings": report["whole_session_token_savings"] = 0
    elif change == "boolean_counter": report["counters"]["stale_source_refusals"] = True
    elif change == "duplicate_identity": report["invocations"][1]["invocation_id"] = report["invocations"][0]["invocation_id"]
    elif change == "audit_hash": report["audit"]["matches"][0]["model_prompt_sha256"] = "0" * 64
    elif change == "audit_authority": report["audit"]["execution_authority"] = True
    elif change == "context_close": report["private_fixture_native_contexts_closed"] = False
    else: report["private_fixture_cleanup_complete"] = False
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256=_reseal(report))


@pytest.mark.parametrize("artifact", ["packet", "invocation"])
def test_changed_artifact_body_refuses_exact_input_qualification(measured, tmp_path, artifact):
    root = _copy_artifacts(measured, tmp_path)
    descriptor = measured[0]["invocations"][0]["retention"][artifact]
    target = root / descriptor["name"]
    raw = bytearray(target.read_bytes()); raw[-1] ^= 1; target.write_bytes(raw)
    audit = _collect(measured, _receipts(measured), root)
    assert audit["all_observed_coding_inputs_verified"] is not True
    assert all(row.get("model_input_verified") is not True for row in audit["invocations"]
        if row["invocation_id"] == measured[0]["invocations"][0]["invocation_id"])


@pytest.mark.parametrize("change", ["name", "digest", "bytes", "boolean_bytes", "foreign_invocation"])
def test_foreign_or_modified_retention_descriptor_refuses(measured, change):
    receipts = deepcopy(_receipts(measured))
    descriptor = receipts[0]["world_packet_artifacts"]["invocation"]
    if change == "name": descriptor["name"] = "../foreign.json"
    elif change == "digest": descriptor["sha256"] = "0" * 64
    elif change == "bytes": descriptor["bytes"] += 1
    elif change == "boolean_bytes": descriptor["bytes"] = True
    else: receipts[0]["world_packet_artifacts"]["invocation"] = receipts[1]["world_packet_artifacts"]["invocation"]
    audit = _collect(measured, receipts, measured[2])
    assert audit["all_observed_coding_inputs_verified"] is not True


def test_unrelated_empty_artifact_root_does_not_search_receipt_workspace(measured, tmp_path):
    foreign = tmp_path / "foreign-root"; foreign.mkdir(mode=0o700)
    audit = _collect(measured, _receipts(measured), foreign)
    assert audit["all_observed_coding_inputs_verified"] is not True
    assert list(foreign.iterdir()) == []


def test_symlink_artifact_root_is_not_a_trust_boundary(measured, tmp_path):
    alias = tmp_path / "alias"; alias.symlink_to(measured[2], target_is_directory=True)
    audit = _collect(measured, _receipts(measured), alias)
    assert audit["all_observed_coding_inputs_verified"] is not True


@pytest.mark.parametrize("mode", ["symlink", "not_private", "nonempty"])
def test_experiment_requires_fresh_private_canonical_artifact_root(tmp_path, mode):
    root = tmp_path / "retained"; root.mkdir(mode=0o700)
    selected = root
    if mode == "symlink": selected = tmp_path / "alias"; selected.symlink_to(root, target_is_directory=True)
    elif mode == "not_private": root.chmod(0o755)
    else: (root / "foreign").write_text("do not overwrite")
    with pytest.raises(ValueError): experiment._artifact_root(selected)


def test_refusal_helper_never_swallows_success_or_infrastructure_failure():
    with pytest.raises(ValueError, match="missing_expected_refusal"):
        experiment._refused(lambda: None, "missing_expected_refusal")
    def infrastructure_failure(): raise OSError("retained infrastructure failure")
    with pytest.raises(OSError): experiment._refused(infrastructure_failure, "wrong")


def test_owned_readonly_package_cleanup_chmods_only_directory_inodes(tmp_path, monkeypatch):
    shared = tmp_path / "retained-hardlink-source"
    shared.write_text("outside fixture bytes"); shared.chmod(0o444)
    original = experiment.os.fchmod
    changed = []
    def directory_only(descriptor, mode):
        info = os.fstat(descriptor)
        assert stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
        assert mode == stat.S_IMODE(info.st_mode) | stat.S_IWUSR
        changed.append((info.st_dev, info.st_ino))
        return original(descriptor, mode)
    monkeypatch.setattr(experiment.os, "fchmod", directory_only)
    with experiment._world_private_fixture(tmp_path) as (fixture, cleanup):
        package = fixture / "immutable-package"; package.mkdir()
        nested = package / "nested"; nested.mkdir()
        (nested / "owned").write_text("authored package")
        os.link(shared, nested / "shared-readonly-file")
        nested.chmod(0o555); package.chmod(0o555)
    assert cleanup["private_fixture_cleanup_complete"] is True
    assert cleanup["private_fixture_cleanup_directories_made_owner_writable"] == len(changed) == 2
    assert fixture.is_dir() and list(fixture.iterdir()) == []
    assert cleanup["private_fixture_cleanup_scope"] == experiment.CLEANUP_SCOPE
    assert cleanup["private_fixture_root_removed"] is False
    assert shared.read_text() == "outside fixture bytes"
    assert stat.S_IMODE(shared.stat().st_mode) == 0o444


def test_symlink_refusal_preserves_foreign_and_readonly_owned_permissions(tmp_path):
    foreign = tmp_path / "foreign"; foreign.mkdir(mode=0o755)
    (foreign / "keep").write_text("foreign bytes")
    before = stat.S_IMODE(foreign.stat().st_mode)
    with experiment._world_private_fixture(tmp_path) as (fixture, cleanup):
        package = fixture / "readonly-package"; package.mkdir(); package.chmod(0o555)
        (fixture / "foreign-link").symlink_to(foreign, target_is_directory=True)
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert cleanup["private_fixture_cleanup_reason"] == "unexpected_private_fixture_symlink"
    assert cleanup["private_fixture_cleanup_directories_made_owner_writable"] == 0
    assert stat.S_IMODE(package.stat().st_mode) == 0o555
    assert stat.S_IMODE(foreign.stat().st_mode) == before
    assert (foreign / "keep").read_text() == "foreign bytes"


def test_complete_census_bound_refuses_before_any_directory_permission_change(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "MAX_FIXTURE_FILES", 1)
    with experiment._world_private_fixture(tmp_path) as (fixture, cleanup):
        package = fixture / "readonly-package"; package.mkdir()
        (package / "owned").write_text("keep on bounded refusal")
        package.chmod(0o555)
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert cleanup["private_fixture_cleanup_reason"] == "private_fixture_file_bound_exceeded"
    assert cleanup["private_fixture_cleanup_directories_made_owner_writable"] == 0
    assert stat.S_IMODE(package.stat().st_mode) == 0o555
    assert (package / "owned").is_file()


def test_native_context_close_proof_missing_preserves_readonly_fixture(tmp_path):
    with experiment._world_private_fixture(tmp_path) as (fixture, cleanup):
        package = fixture / "readonly-package"; package.mkdir(); package.chmod(0o555)
        cleanup["private_fixture_native_contexts_closed"] = False
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert cleanup["private_fixture_cleanup_reason"] == "private_fixture_native_contexts_not_closed"
    assert cleanup["private_fixture_cleanup_directories_made_owner_writable"] == 0
    assert stat.S_IMODE(package.stat().st_mode) == 0o555


def test_root_replacement_at_contents_delete_is_preserved_and_unqualified(tmp_path, monkeypatch):
    original = experiment._delete_fixture_contents
    held = {}
    def replace_then_delete(root_fd, parent_fd, root_name, records, observation):
        fixture = tmp_path / root_name
        preserved = tmp_path / "preserved-owned-root"
        fixture.rename(preserved)
        fixture.mkdir(mode=0o700)
        (fixture / "foreign-keep").write_text("foreign replacement bytes")
        held.update(replacement=fixture, preserved=preserved)
        return original(root_fd, parent_fd, root_name, records, observation)
    monkeypatch.setattr(experiment, "_delete_fixture_contents", replace_then_delete)
    with experiment._world_private_fixture(tmp_path) as (fixture, cleanup):
        (fixture / "owned").write_text("original owned bytes")
    assert cleanup["private_fixture_cleanup_complete"] is False
    assert cleanup["private_fixture_cleanup_reason"] == "private_fixture_identity_changed"
    assert cleanup["private_fixture_root_removed"] is False
    assert cleanup["private_fixture_cleanup_removed_files"] == 0
    assert (held["replacement"] / "foreign-keep").read_text() == "foreign replacement bytes"
    assert stat.S_IMODE(held["replacement"].stat().st_mode) == 0o700
    assert (held["preserved"] / "owned").read_text() == "original owned bytes"
