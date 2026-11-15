"""Real authored prelaunch scenario and bounded content-free report custody."""
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_world_prelaunch_experiment as experiment


def reseal(report):
    body = dict(report)
    body.pop("report_sha256", None)
    report["report_sha256"] = experiment._sha(experiment._wire(body))
    return experiment.report_sha256(report)


@pytest.fixture
def report():
    """Synthetic validator fixture, never native qualification evidence."""
    value = dict(schema=experiment.SCHEMA, qualified=True,
        scope="native_owner_prelaunch_assignment_only",
        transport_profile="authored_FakeQuackTransport_no_actual_TCP",
        deployment_profile="authored_host_metadata_no_actual_container_or_UID_transition",
        source_pins=dict(scope="listed_source_files_only_not_transitive_provenance",
            files={name: dict(bytes=10, sha256="a" * 64) for name in experiment.SOURCE_PATHS}),
        source_pins_unchanged=True, failure=None,
        scenario=dict(checks=dict.fromkeys(experiment.CHECKS, True),
            assignment=dict(gate_receipt_sha256="a" * 64, task_admission_sha256="b" * 64,
                allocation_sha256="c" * 64, canonical_repository_path_sha256="d" * 64,
                allocated_workspace_path_sha256="e" * 64, canonical_source_sha256="f" * 64,
                allocated_source_sha256="f" * 64, native_task_cid="bafynative",
                portal_task_cid="bafyportal", task_revision=3),
            observations=dict(retained_positive_gate_snapshots=2, stale_refusals=6,
                read_only_native_payload_bytes=100, lifecycle_fence_advanced=True,
                authored_fixture_source_provisioned_files=2, authored_fixture_source_bytes_unchanged=True,
                authored_fixture_executable_bits_unchanged=True,
                native_claim_release_observed=True, kernel_uid_transition_performed=False,
                renewed_gate_receipt_sha256="a" * 64),
            fault_scope="authored_metadata_faults_and_genuine_native_lifecycle_renewal_and_claim_release"),
        private_fixture_cleanup_complete=True, private_fixture_files=5,
        private_fixture_bytes=100, private_fixture_storage_bound_bytes=experiment.MAX_FIXTURE_BYTES,
        private_fixture_cleanup_directories_made_owner_writable=0,
        private_fixture_cleanup_scope=experiment.CLEANUP_SCOPE, private_fixture_root_removed=False,
        private_fixture_cleanup_removed_files=5, private_fixture_cleanup_removed_directories=2,
        private_fixture_native_contexts_closed=True, **experiment.FALSE)
    reseal(value)
    return value


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("prelaunch-measured")
    value = experiment.run_experiment(scratch_root=scratch)
    assert value["qualified"] is True, {
        "failure": value["failure"], "scenario_present": value["scenario"] is not None,
        "source_pins_unchanged": value["source_pins_unchanged"],
        "cleanup_complete": value["private_fixture_cleanup_complete"],
        "cleanup_reason": value.get("private_fixture_cleanup_reason")}
    return value, scratch


def test_measured_native_assignment_and_every_refusal(measured):
    value, _scratch = measured
    result = experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))
    assert result["valid"] is result["qualified"] is True
    assert set(value["scenario"]["checks"]) == experiment.CHECKS
    assert value["scenario"]["observations"]["stale_refusals"] == 6
    assert value["scenario"]["observations"]["read_only_native_payload_bytes"] > 0
    assert result["native_operations_independently_replayed"] is False


def test_measured_scope_exports_no_private_paths_source_or_authority(measured):
    value, scratch = measured
    wire = experiment._wire(value)
    assert str(scratch).encode() not in wire
    assert b"VALUE = 42" not in wire and b"claim_process_attestation" not in wire
    for key, expected in experiment.FALSE.items():
        assert type(value[key]) is type(expected) and value[key] == expected


def test_measured_cleanup_keeps_one_empty_private_root(measured):
    value, scratch = measured
    entries = list(scratch.iterdir())
    assert len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink()
    assert list(entries[0].iterdir()) == []
    assert value["private_fixture_root_removed"] is False


def test_full_independent_pin_remains_required_after_self_reseal(report):
    original = experiment.report_sha256(report)
    report["scenario"]["assignment"]["task_revision"] += 1
    reseal(report)
    with pytest.raises(ValueError, match="independent_report_hash"):
        experiment.validate_report(report, expected_report_sha256=original)


@pytest.mark.parametrize("pin", [None, "", "a" * 63, "g" * 64, True])
def test_invalid_external_report_pin(report, pin):
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256=pin)


@pytest.mark.parametrize("change", ["unknown_top", "unknown_source", "dropped_source",
    "false_check", "dropped_check", "source_mismatch", "path_alias", "task_namespace",
    "wrong_counter", "bool_counter", "renewal_hash", "world_ready", "target_execution",
    "container_claim", "UID_claim", "root_removed", "wrong_cleanup_scope",
    "private_cleanup_reason", "unknown_scenario", "full_source_body", "payload_bytes",
    "provisioning_counter", "provisioning_bytes", "provisioning_executable"])
def test_rehashed_contract_and_privacy_forgeries(report, change):
    if change == "unknown_top": report["foreign"] = "outside the closed report"
    elif change == "unknown_source": report["source_pins"]["files"]["foreign.py"] = dict(bytes=1, sha256="b" * 64)
    elif change == "dropped_source": report["source_pins"]["files"].pop(experiment.SOURCE_PATHS[0])
    elif change == "false_check": report["scenario"]["checks"][next(iter(experiment.CHECKS))] = False
    elif change == "dropped_check": report["scenario"]["checks"].pop(next(iter(experiment.CHECKS)))
    elif change == "source_mismatch": report["scenario"]["assignment"]["allocated_source_sha256"] = "0" * 64
    elif change == "path_alias": report["scenario"]["assignment"]["allocated_workspace_path_sha256"] = "d" * 64
    elif change == "task_namespace": report["scenario"]["assignment"]["portal_task_cid"] = "bafynative"
    elif change == "wrong_counter": report["scenario"]["observations"]["stale_refusals"] = 5
    elif change == "bool_counter": report["scenario"]["observations"]["retained_positive_gate_snapshots"] = True
    elif change == "renewal_hash": report["scenario"]["observations"]["renewed_gate_receipt_sha256"] = "b" * 64
    elif change == "world_ready": report["world_dispatch_ready"] = True
    elif change == "target_execution": report["target_code_executed"] = True
    elif change == "container_claim": report["container_boundary_verified"] = True
    elif change == "UID_claim": report["separate_worker_uid_verified"] = True
    elif change == "root_removed": report["private_fixture_root_removed"] = True
    elif change == "wrong_cleanup_scope": report["private_fixture_cleanup_scope"] = "remove_root"
    elif change == "private_cleanup_reason": report["private_fixture_cleanup_reason"] = "/private/foreign/path"
    elif change == "unknown_scenario": report["scenario"]["foreign"] = True
    elif change == "payload_bytes": report["scenario"]["observations"]["read_only_native_payload_bytes"] = True
    elif change == "provisioning_counter": report["scenario"]["observations"]["authored_fixture_source_provisioned_files"] = True
    elif change == "provisioning_bytes": report["scenario"]["observations"]["authored_fixture_source_bytes_unchanged"] = False
    elif change == "provisioning_executable": report["scenario"]["observations"]["authored_fixture_executable_bits_unchanged"] = False
    else: report["scenario"]["assignment"]["source_body"] = "VALUE = 42"
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256=reseal(report))


def test_failed_scratch_report_is_valid_but_unqualified(report):
    report.update(scenario=None, failure=dict(stage="authored_prelaunch_scenario", error_type="ValueError"),
        qualified=False, private_fixture_cleanup_complete=False,
        private_fixture_native_contexts_closed=False,
        private_fixture_cleanup_reason="private_fixture_native_contexts_not_closed")
    result = experiment.validate_report(report, expected_report_sha256=reseal(report))
    assert result["valid"] is True and result["qualified"] is False


@pytest.mark.parametrize("cause", ["failure", "pins", "cleanup", "contexts"])
def test_cannot_override_incomplete_evidence(report, cause):
    if cause == "failure": report["failure"] = dict(stage="authored_prelaunch_scenario", error_type="ValueError")
    elif cause == "pins": report["source_pins_unchanged"] = False
    elif cause == "cleanup": report["private_fixture_cleanup_complete"] = False
    else: report["private_fixture_native_contexts_closed"] = False
    with pytest.raises(ValueError, match="qualification_claim"):
        experiment.validate_report(report, expected_report_sha256=reseal(report))


def test_bound_refuses_before_wire_encoding(report, monkeypatch):
    report["extra"] = "x" * (experiment.MAX_REPORT_BYTES + 1)
    def unexpected(_value):
        pytest.fail("oversized string reached report serialization")
    monkeypatch.setattr(experiment, "_wire", unexpected)
    with pytest.raises(ValueError, match="exceeds_bound"):
        experiment.validate_report(report, expected_report_sha256="a" * 64)


def test_pin_failure_releases_process_local_lock(tmp_path, monkeypatch):
    def unavailable():
        raise ValueError("authored source unavailable")
    monkeypatch.setattr(experiment, "_pins", unavailable)
    with pytest.raises(ValueError):
        experiment.run_experiment(scratch_root=tmp_path)
    assert experiment._LOCK.acquire(blocking=False)
    experiment._LOCK.release()


def test_scenario_failure_preserves_private_evidence_and_typed_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_pins", lambda: dict(scope="listed_source_files_only_not_transitive_provenance",
        files={name: dict(bytes=1, sha256="a" * 64) for name in experiment.SOURCE_PATHS}))
    def fail(fixture):
        (fixture / "preserved").write_bytes(b"private authored evidence")
        raise ValueError("/private/path must not be exported")
    monkeypatch.setattr(experiment, "_scenario", fail)
    value = experiment.run_experiment(scratch_root=tmp_path)
    assert value["qualified"] is False
    assert value["failure"] == dict(stage="authored_prelaunch_scenario", error_type="ValueError")
    assert value["private_fixture_cleanup_complete"] is False
    assert next(tmp_path.iterdir()).joinpath("preserved").read_bytes() == b"private authored evidence"
    assert b"/private/path" not in experiment._wire(value)
    assert experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))["qualified"] is False
