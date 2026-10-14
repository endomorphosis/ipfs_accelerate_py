"""Offline report custody; actual UID crossing is run separately by root."""
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_world_worker_handoff_experiment as experiment


def reseal(report):
    body = dict(report); body.pop("report_sha256", None)
    report["report_sha256"] = experiment._sha(experiment._wire(body))
    return experiment.report_sha256(report)


@pytest.fixture
def report():
    """Synthetic contract fixture, never real-worker qualification evidence."""
    value = dict(schema=experiment.SCHEMA, qualified=True,
        scope="real_worker_readonly_handoff_and_authenticated_assignment_metadata_only",
        native_transport="authored_FakeQuackTransport_no_actual_TCP",
        inner_resource_accounting_scope="container_local_fixture_authority_only",
        source_pins=dict(scope="listed_source_files_only_not_transitive_provenance",
            files={name: dict(bytes=10, sha256="a" * 64) for name in experiment.SOURCE_PATHS}),
        source_pins_unchanged=True, failure=None,
        profile=dict(owner_uid=1000, owner_gid=1000, worker_uid=1001, worker_gid=1000,
            launcher_sha256=experiment._sha(experiment.WORKER_ENTRY_SOURCE.encode()),
            literal_popen_argv_sha256=experiment._sha(experiment._wire(list(experiment.WORKER_ARGV))),
            worker_peer_argv_sha256=experiment._sha(experiment._wire(list(experiment.WORKER_PEER_ARGV))),
            namespaces=dict(pid="pid:[1]", mnt="mnt:[2]", net="net:[3]")),
        cycle=dict(checks=dict.fromkeys(experiment.CHECKS, True),
            measurement=dict(positive_response_sha256="a" * 64, negative_response_sha256="b" * 64,
                handoff_observation_sha256="c" * 64, observer_close_sha256="d" * 64,
                handoff_close_sha256="e" * 64, successful_metadata_observations=1,
                stale_metadata_refusals=1, allocated_root_mode_before=0o700,
                allocated_root_mode_after=0o750, actual_worker_birth_verified=True,
                actual_worker_uid_gid_verified=True,
                observation_scope="authenticated_worker_assignment_metadata_only",
                worker_peer=dict(pid=11, uid=1001, gid=1000, start_time_ticks=100,
                    root_pid=10, root_start_time_ticks=99, groups_sha256="f" * 64, group_count=1))),
        private_fixture_cleanup_complete=True, private_fixture_files=5,
        private_fixture_bytes=100, private_fixture_storage_bound_bytes=experiment.MAX_FIXTURE_BYTES,
        private_fixture_cleanup_directories_made_owner_writable=0,
        private_fixture_cleanup_scope=experiment.CLEANUP_SCOPE, private_fixture_root_removed=False,
        private_fixture_cleanup_removed_files=5, private_fixture_cleanup_removed_directories=2,
        private_fixture_native_contexts_closed=True, **experiment.FALSE)
    reseal(value)
    return value


def test_full_pinned_synthetic_contract_validates_without_native_replay(report):
    result = experiment.validate_report(report, expected_report_sha256=experiment.report_sha256(report))
    assert result["valid"] is True and result["qualified"] is True
    assert result["native_operations_independently_replayed"] is False
    assert result["container_image_independently_verified"] is False


def test_self_reseal_cannot_replace_independent_pin(report):
    pin = experiment.report_sha256(report)
    report["cycle"]["measurement"]["worker_peer"]["pid"] += 1
    reseal(report)
    with pytest.raises(ValueError, match="independent_worker_report_pin"):
        experiment.validate_report(report, expected_report_sha256=pin)


@pytest.mark.parametrize("change", ["same_uid", "wrong_gid", "birth_bool", "root_pid_missing",
    "group_count_bool", "namespace_forgery", "launcher_forgery", "argv_forgery",
    "world_ready", "execution", "database", "tcp", "container_image", "raw_body",
    "raw_peer", "wrong_mode", "bool_count", "unknown_cycle", "missing_check",
    "wrong_scope", "missing_source", "extra_source", "source_bool_bytes", "cleanup_claim"])
def test_rehashed_scope_population_and_authority_forgeries_refused(report, change):
    measure = report["cycle"]["measurement"]
    if change == "same_uid": measure["worker_peer"]["uid"] = 1000
    elif change == "wrong_gid": measure["worker_peer"]["gid"] = 1001
    elif change == "birth_bool": measure["worker_peer"]["start_time_ticks"] = True
    elif change == "root_pid_missing": measure["worker_peer"].pop("root_pid")
    elif change == "group_count_bool": measure["worker_peer"]["group_count"] = True
    elif change == "namespace_forgery": report["profile"]["namespaces"]["pid"] = "/private/path"
    elif change == "launcher_forgery": report["profile"]["launcher_sha256"] = "a" * 64
    elif change == "argv_forgery": report["profile"]["literal_popen_argv_sha256"] = "a" * 64
    elif change == "world_ready": report["world_dispatch_ready"] = True
    elif change == "execution": report["execution_authority"] = True
    elif change == "database": report["database_access_granted"] = True
    elif change == "tcp": report["actual_quack_tcp_verified"] = True
    elif change == "container_image": report["container_image_independently_verified"] = True
    elif change == "raw_body": measure["source_body"] = "private source"
    elif change == "raw_peer": measure["worker_peer"]["argv"] = ["foreign"]
    elif change == "wrong_mode": measure["allocated_root_mode_after"] = 0o777
    elif change == "bool_count": measure["successful_metadata_observations"] = True
    elif change == "unknown_cycle": report["cycle"]["foreign"] = True
    elif change == "missing_check": report["cycle"]["checks"].pop(next(iter(experiment.CHECKS)))
    elif change == "wrong_scope": measure["observation_scope"] = "provider_dispatch"
    elif change == "missing_source": report["source_pins"]["files"].pop(experiment.SOURCE_PATHS[0])
    elif change == "extra_source": report["source_pins"]["files"]["foreign.py"] = dict(bytes=1, sha256="a" * 64)
    elif change == "source_bool_bytes": report["source_pins"]["files"][experiment.SOURCE_PATHS[0]]["bytes"] = True
    else: report["private_fixture_cleanup_complete"] = False
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256=reseal(report))


def test_unavailable_actual_worker_profile_is_explicitly_unqualified(report):
    report.update(profile=None, cycle=None, qualified=False,
        failure=dict(stage="trusted_worker_handoff_fixture", error_type="ValueError", reason_code="actual_owner_identity_required"),
        private_fixture_native_contexts_closed=False, private_fixture_cleanup_complete=False)
    result = experiment.validate_report(report, expected_report_sha256=reseal(report))
    assert result["valid"] is True and result["qualified"] is False


def test_unavailable_profile_never_starts_handoff_or_worker(tmp_path, monkeypatch, report):
    monkeypatch.setattr(experiment, "_pins", lambda: report["source_pins"])
    def refuse():
        raise ValueError("actual owner or protected launcher unavailable")
    def forbidden(_fixture):
        pytest.fail("unavailable profile reached handoff")
    monkeypatch.setattr(experiment, "_profile", refuse)
    monkeypatch.setattr(experiment, "_cycle", forbidden)
    value = experiment.run_experiment(scratch_root=tmp_path)
    assert value["qualified"] is False and value["cycle"] is None
    assert value["failure"] == dict(stage="trusted_worker_handoff_fixture", error_type="ValueError", reason_code="unspecified_failure")
    assert experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))["qualified"] is False


def test_oversized_input_refused_before_serialization(report, monkeypatch):
    report["extra"] = "x" * (256 * 1024 + 1)
    monkeypatch.setattr(experiment, "_wire", lambda _: pytest.fail("oversized input serialized"))
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256="a" * 64)


def test_pin_failure_releases_only_process_local_fixture_lock(tmp_path, monkeypatch):
    def fail(): raise ValueError("source unavailable")
    monkeypatch.setattr(experiment, "_pins", fail)
    with pytest.raises(ValueError): experiment.run_experiment(scratch_root=tmp_path)
    assert experiment._LOCK.acquire(blocking=False)
    experiment._LOCK.release()


@pytest.mark.parametrize("message,reason", [("actual_owner_identity_required", "actual_owner_identity_required"),
    ("/private/path has data", "unspecified_failure"), ("secret=" + "a" * 64, "unspecified_failure")])
def test_failure_reason_exports_only_bounded_closed_code(tmp_path, monkeypatch, report, message, reason):
    monkeypatch.setattr(experiment, "_pins", lambda: report["source_pins"])
    def refuse(): raise ValueError(message)
    monkeypatch.setattr(experiment, "_profile", refuse)
    value = experiment.run_experiment(scratch_root=tmp_path)
    assert value["failure"]["reason_code"] == reason
    assert message == reason or message.encode() not in experiment._wire(value)
    assert experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))["qualified"] is False


def test_worker_launcher_and_sudo_argv_are_fixed_without_payload_args():
    assert experiment.WORKER_ARGV == ("/usr/bin/sudo", "-n", "-u", "benchmarkworker",
        "-g", "supervisor", "--", "/usr/local/libexec/ipfs-world-handoff-observe-worker")
    assert "sys.path[:0]" in experiment.WORKER_ENTRY_SOURCE
    assert "libc.prctl(38,1,0,0,0)" in experiment.WORKER_ENTRY_SOURCE
    assert "repository_world_worker_observer import worker_observer_client_main" in experiment.WORKER_ENTRY_SOURCE
    assert "terminal_world_worker_handoff_experiment" not in experiment.WORKER_ENTRY_SOURCE
    assert experiment.WORKER_PEER_ARGV == ("/opt/ipfs-supervisor/venv/bin/python", "-I",
        "/usr/local/libexec/ipfs-world-handoff-observe-worker")
