"""Report/measurement contracts; synthetic rows never qualify real delivery."""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_world_worker_packet_experiment as experiment


def reseal(report):
    body = dict(report); body.pop("report_sha256", None)
    report["report_sha256"] = experiment._sha(experiment._wire(body))
    return experiment.report_sha256(report)


def cleanup():
    return dict(private_fixture_cleanup_complete=True, private_fixture_files=2,
        private_fixture_bytes=100, private_fixture_storage_bound_bytes=experiment.MAX_FIXTURE_BYTES,
        private_fixture_cleanup_directories_made_owner_writable=0,
        private_fixture_cleanup_scope=experiment.CLEANUP_SCOPE, private_fixture_root_removed=False,
        private_fixture_cleanup_removed_files=2, private_fixture_cleanup_removed_directories=1,
        private_fixture_native_contexts_closed=True)


def cycle(count):
    source = experiment._source(count)
    return dict(checks=dict.fromkeys(experiment.CHECKS, True), measurement=dict(
        source_sha256=experiment._sha(source), selected_source_files=1,
        selected_source_bytes=len(source), authored_function_count=count,
        required_reference_count=5, structural_reference_count=count, native_validation_count=0,
        native_validation_coverage="no_native_validation_commands_declared",
        task_projection_sha256="a" * 64, binding_metadata_sha256="b" * 64,
        packet_text=experiment._measurement("P" * 100, None),
        required_source_text=experiment._measurement(source.decode(), None),
        positive_receipt_sha256="c" * 64, negative_receipt_sha256="d" * 64,
        worker_peer=dict(pid=11, uid=1001, gid=1000, start_time_ticks=100,
            root_pid=10, root_start_time_ticks=99, groups_sha256="e" * 64, group_count=1),
        ipc=dict(accepted_connections=3, sent_responses=3, refused_responses=1,
            sent_packet_responses=1, acknowledged_packets=1, sent_packet_bytes=100,
            sent_response_bytes=300, sent_response_wire_bytes=312,
            helper_stdin_bytes=30, helper_stdout_bytes=50, packet_body_counted_once=True,
            byte_scope="actual_service_canonical_response_frames_and_exact_helper_transcript"),
        comparison=experiment._comparison("A" * 50, "B" * 100, "TRACE\n", None)))


@pytest.fixture
def report():
    """Synthetic observation contract only, not an experiment result."""
    value = dict(schema=experiment.SCHEMA, scope=experiment.SCOPE, qualified=True,
        source_pins=dict(scope="listed_source_files_only_not_transitive_provenance",
            files={name: dict(bytes=10, sha256="a" * 64) for name in experiment.SOURCE_PATHS}),
        source_pins_unchanged=True,
        profile=dict(owner_uid=1000, owner_gid=1000, worker_uid=1001, worker_gid=1000,
            launcher_sha256=experiment._sha(experiment.WORKER_ENTRY_SOURCE.encode()),
            literal_popen_argv_sha256=experiment._sha(experiment._wire(list(experiment.WORKER_ARGV))),
            worker_peer_argv_sha256=experiment._sha(experiment._wire(list(experiment.WORKER_PEER_ARGV))),
            namespaces=dict(pid="pid:[1]", mnt="mnt:[2]", net="net:[3]")),
        tokenizer=dict(available=False, encoding=None, ranks_sha256=None,
            reason="verified_local_o200k_cache_unavailable",
            count_scope="exact_local_encoding_of_authored_text_only", provider_compatibility_verified=False,
            provider_usage=False, message_framing_included=False, download_performed=False),
        cases=[dict(case_id=f"authored-functions-{count}", function_count=count, status="completed",
            failure=None, cycle=cycle(count), cleanup=cleanup()) for count in experiment.CASE_COUNTS],
        failure=None, private_fixture_cleanup_complete=True,
        offline_comparison_counts=dict(compiled_smaller=0, raw_comparison_preferred_not_smaller=0, token_count_unavailable=3),
        storage_scope="three_sequential_private_fixtures_each_bounded_and_cleaned_before_next",
        inner_resource_accounting_scope="container_local_fixture_authority_only",
        native_transport="authored_FakeQuackTransport_no_actual_TCP",
        provider_usage=dict(complete_native_total_tokens=None, billed_tokens=None, provider_session_population_complete=None),
        **experiment.FALSE)
    reseal(value)
    return value


def test_independently_pinned_report_does_not_replay_native_work(report):
    checked = experiment.validate_report(report, expected_report_sha256=experiment.report_sha256(report))
    assert checked["valid"] is checked["qualified"] is True
    assert checked["native_operations_independently_replayed"] is False
    assert checked["provider_calls"] == 0 and checked["world_dispatch_ready"] is False
    for field in ("private_fixture_files", "private_fixture_bytes"):
        for invalid in (None, True):
            changed = deepcopy(report)
            changed["cases"][0]["cleanup"][field] = invalid
            with pytest.raises(ValueError, match="cleanup_counter"):
                experiment.validate_report(changed, expected_report_sha256=reseal(changed))


def test_self_reseal_does_not_replace_independent_report_pin(report):
    pin = experiment.report_sha256(report)
    report["cases"][0]["cycle"]["measurement"]["worker_peer"]["pid"] += 1
    reseal(report)
    with pytest.raises(ValueError, match="independent_packet_report_pin"):
        experiment.validate_report(report, expected_report_sha256=pin)


@pytest.mark.parametrize("fault", ["missing_case", "reordered_case", "extra_case", "bool_population",
    "missing_required", "source_forgery", "source_bytes", "source_digest", "same_uid", "wrong_gid", "bool_birth",
    "fake_ack", "double_packet", "wire_prefix", "bool_ipc", "raw_content", "native_validation", "selected_scope",
    "missing_pin", "extra_pin", "provider_usage", "native_tokens", "wrong_profile", "argv_forgery", "namespace_path",
    "fake_saving", "route_fallback", "trace_double_count", "token_available_forgery", "root_removed", "cleanup_live",
    "completion", "compiler_signature", "semantic_truth", "kernel_cwd", "whole_repo", "TCP", "partial_clean_claim"])
def test_rehashed_population_scope_and_authority_changes_refused(report, fault):
    measurement = report["cases"][0]["cycle"]["measurement"]
    comparison = measurement["comparison"]
    if fault == "missing_case": report["cases"].pop()
    elif fault == "reordered_case": report["cases"].reverse()
    elif fault == "extra_case": report["cases"].append(deepcopy(report["cases"][0]))
    elif fault == "bool_population": report["cases"][0]["function_count"] = True
    elif fault == "missing_required": measurement["required_reference_count"] = 4
    elif fault == "source_forgery": measurement["source_sha256"] = "f" * 64
    elif fault == "source_bytes": measurement["selected_source_bytes"] += 1
    elif fault == "source_digest": measurement["required_source_text"]["sha256"] = "f" * 64
    elif fault == "same_uid": measurement["worker_peer"]["uid"] = 1000
    elif fault == "wrong_gid": measurement["worker_peer"]["gid"] = 1001
    elif fault == "bool_birth": measurement["worker_peer"]["start_time_ticks"] = True
    elif fault == "fake_ack": measurement["ipc"]["acknowledged_packets"] = 0
    elif fault == "double_packet": measurement["ipc"]["sent_packet_bytes"] *= 2
    elif fault == "wire_prefix": measurement["ipc"]["sent_response_wire_bytes"] += 4
    elif fault == "bool_ipc": measurement["ipc"]["sent_packet_responses"] = True
    elif fault == "raw_content": measurement["raw_source"] = "private"
    elif fault == "native_validation": measurement["native_validation_count"] = 1
    elif fault == "selected_scope": measurement["structural_reference_count"] = 9
    elif fault == "missing_pin": report["source_pins"]["files"].pop(experiment.SOURCE_PATHS[0])
    elif fault == "extra_pin": report["source_pins"]["files"]["foreign.py"] = dict(bytes=1, sha256="a" * 64)
    elif fault == "provider_usage": report["provider_usage_verified"] = True
    elif fault == "native_tokens": report["provider_usage"]["complete_native_total_tokens"] = 0
    elif fault == "wrong_profile": report["profile"]["worker_uid"] = 1000
    elif fault == "argv_forgery": report["profile"]["worker_peer_argv_sha256"] = "a" * 64
    elif fault == "namespace_path": report["profile"]["namespaces"]["pid"] = "/private"
    elif fault == "fake_saving": comparison["offline_comparison_disposition"] = "compiled_smaller"
    elif fault == "route_fallback": comparison["actual_model_route_fallback"] = True
    elif fault == "trace_double_count": comparison["measurements"]["candidate_authored_trace"]["bytes"] += 100
    elif fault == "token_available_forgery": comparison["measurements"]["candidate_initial"]["local_o200k_tokens"] = 1
    elif fault == "root_removed": report["cases"][0]["cleanup"]["private_fixture_root_removed"] = True
    elif fault == "cleanup_live": report["cases"][0]["cleanup"]["private_fixture_native_contexts_closed"] = False
    elif fault == "completion": report["completion_authority"] = True
    elif fault == "compiler_signature": report["compiler_signature_verified"] = True
    elif fault == "semantic_truth": report["semantic_correctness_verified"] = True
    elif fault == "kernel_cwd": report["worker_workspace_verified"] = True
    elif fault == "whole_repo": report["whole_repository_observed"] = True
    elif fault == "TCP": report["actual_quack_tcp_verified"] = True
    else: report["cases"][0]["cleanup"]["private_fixture_cleanup_complete"] = False
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256=reseal(report))


@pytest.mark.parametrize("old,new,disposition", [("A" * 100, "B" * 50, "compiled_smaller"),
    ("A" * 50, "B" * 100, "raw_comparison_preferred_not_smaller"),
    ("A" * 50, "B" * 50, "raw_comparison_preferred_not_smaller")])
def test_offline_disposition_does_not_activate_route_fallback(old, new, disposition):
    value = experiment._comparison(old, new, "same authored trace\n", len)
    experiment._validate_comparison(value)
    assert value["offline_comparison_disposition"] == disposition
    assert value["actual_model_route_fallback"] is False
    assert value["service_delivery_mode"] == "compiled_packet_only"
    assert value["complete_native_total_tokens"] is None


def test_missing_tokenizer_preserves_null_measurements_and_uncertainty():
    value = experiment._comparison("A", "B", "TRACE", None)
    experiment._validate_comparison(value)
    assert value["offline_comparison_disposition"] == "token_count_unavailable"
    assert all(row["local_o200k_tokens"] is None for row in value["measurements"].values())


def test_full_trace_counter_sees_each_complete_trace_once():
    seen = []
    def counter(text):
        seen.append(text)
        return len(text)
    experiment._comparison("OLD", "NEW", "COMMON", counter)
    assert seen == ["OLD", "NEW", "COMMON", "OLDCOMMON", "NEWCOMMON"]


@pytest.mark.parametrize("counter", [lambda _: True, lambda _: -1, lambda _: 1 << 63])
def test_invalid_token_counter_is_refused(counter):
    with pytest.raises(ValueError, match="local_token_count"):
        experiment._measurement("literal", counter)


def test_no_broad_source_or_target_execution_in_fixture():
    for count in experiment.CASE_COUNTS:
        text = experiment._source(count).decode()
        assert text.count("def helper_") == count
        assert "never imported or executed" in text
        assert "import " not in text and "exec(" not in text


def test_fixed_worker_imports_only_delivery_client():
    assert "from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_packet_delivery import worker_packet_client_main" in experiment.WORKER_ENTRY_SOURCE
    assert "benchmarks" not in experiment.WORKER_ENTRY_SOURCE
    assert "get_packet" not in experiment.WORKER_ARGV


def test_unavailable_profile_never_runs_cycle_and_retains_all_three_records(tmp_path, monkeypatch, report):
    monkeypatch.setattr(experiment, "_pins", lambda: report["source_pins"])
    def refused(): raise ValueError("actual_owner_identity_required")
    def forbidden(*args): pytest.fail("unavailable profile reached native fixture")
    monkeypatch.setattr(experiment, "_profile", refused)
    monkeypatch.setattr(experiment, "_cycle", forbidden)
    observed = experiment.run_experiment(scratch_root=tmp_path)
    assert [row["status"] for row in observed["cases"]] == ["not_attempted"] * 3
    assert observed["failure"]["reason_code"] == "actual_owner_identity_required"
    assert observed["qualified"] is False
    assert experiment.validate_report(observed, expected_report_sha256=experiment.report_sha256(observed))["qualified"] is False


def test_failed_native_case_preserves_population_and_prevents_next_case(tmp_path, monkeypatch, report):
    monkeypatch.setattr(experiment, "_pins", lambda: report["source_pins"])
    monkeypatch.setattr(experiment, "_profile", lambda: report["profile"])
    monkeypatch.setattr(experiment, "_tokenizer", lambda _: (None, report["tokenizer"]))
    calls = []
    def refused(fixture, count, counter):
        calls.append(count)
        raise ValueError("native_context_refused")
    monkeypatch.setattr(experiment, "_cycle", refused)
    observed = experiment.run_experiment(scratch_root=tmp_path)
    assert calls == [1]
    assert [row["status"] for row in observed["cases"]] == ["failed", "not_attempted", "not_attempted"]
    assert observed["cases"][0]["cleanup"]["private_fixture_cleanup_complete"] is False
    assert observed["cases"][0]["cleanup"]["private_fixture_files"] is None
    assert observed["cases"][0]["cleanup"]["private_fixture_bytes"] is None
    assert observed["qualified"] is False
    assert experiment.validate_report(observed, expected_report_sha256=experiment.report_sha256(observed))["valid"] is True


def test_source_pin_drift_makes_otherwise_complete_report_unqualified(report):
    report["source_pins_unchanged"] = report["qualified"] = False
    assert experiment.validate_report(report, expected_report_sha256=reseal(report))["qualified"] is False


@pytest.mark.parametrize("message,expected", [("native_context_refused", "native_context_refused"),
    ("/private/path: secret", "unspecified_failure"), ("secret=" + "a" * 64, "unspecified_failure")])
def test_failure_reason_never_exports_paths_or_credentials(message, expected):
    value = experiment._failure(ValueError(message))
    assert value["reason_code"] == expected
    assert message == expected or message.encode() not in experiment._wire(value)


def test_structure_budget_precedes_serialization(report, monkeypatch):
    report["extra"] = "x" * (256 * 1024 + 1)
    monkeypatch.setattr(experiment, "_wire", lambda _: pytest.fail("oversized object serialized"))
    with pytest.raises(ValueError):
        experiment.validate_report(report, expected_report_sha256="a" * 64)


def test_local_fixture_lock_is_released_after_pin_failure(tmp_path, monkeypatch):
    def refused(): raise ValueError("source_pin_refused")
    monkeypatch.setattr(experiment, "_pins", refused)
    with pytest.raises(ValueError): experiment.run_experiment(scratch_root=tmp_path)
    assert experiment._LOCK.acquire(blocking=False)
    experiment._LOCK.release()
