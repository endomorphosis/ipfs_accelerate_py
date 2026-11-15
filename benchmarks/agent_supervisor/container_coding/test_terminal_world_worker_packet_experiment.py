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


def with_presentation(report, *, counter=None, baseline="R" * 400, original="O" * 300,
        proposed="P" * 100, codec_mode=experiment.PRESENTATION_PROFILE):
    """Synthetic closed observation only; no codec/native/kernel operation."""
    report["presentation_profile"] = experiment.PRESENTATION_PROFILE
    report["source_pins"]["files"].update({name: dict(bytes=10, sha256="a" * 64)
        for name in experiment.PRESENTATION_SOURCE_PATHS})
    if counter is not None:
        report["tokenizer"].update(available=True, encoding="o200k_base",
            ranks_sha256=experiment.TOKENIZER_RANKS_SHA256, reason=None)
    for row in report["cases"]:
        value = row["cycle"]["measurement"]
        initial = experiment._presentation_initial(baseline=baseline, original=original,
            proposed=proposed, codec_mode=codec_mode, counter=counter)
        delivered = proposed if initial["selected_mode"] == "presented_view" else original
        value["packet_text"] = experiment._measurement(delivered, counter)
        value["required_source_text"] = experiment._measurement(experiment._source(row["function_count"]).decode(), counter)
        value["ipc"].update(sent_packet_bytes=len(delivered.encode()),
            sent_response_bytes=len(delivered.encode()) + 100, sent_response_wire_bytes=len(delivered.encode()) + 112)
        observed, comparison = experiment._presentation_finish(initial=initial, baseline=baseline,
            original=original, proposed=proposed, common_trace="COMMON\n", counter=counter,
            codec_mode=codec_mode, codec_receipt_sha256="a" * 64,
            codec_census=dict.fromkeys(experiment.PRESENTATION_CENSUS_FIELDS, 0), sidecar_bytes=b"HOST ONLY",
            inverse_original=original, native_prompt=original, proposed_prompt=proposed, delivered_prompt=delivered)
        value["presentation"], value["comparison"] = observed, comparison
    report["offline_comparison_counts"] = {kind: sum(row["cycle"]["measurement"]["comparison"]["offline_comparison_disposition"] == kind
        for row in report["cases"]) for kind in report["offline_comparison_counts"]}
    report["representation_efficiency_population"] = experiment._representation_population(report["cases"])
    reseal(report)
    return report


def test_default_report_retains_original_field_and_source_population(report):
    assert "presentation_profile" not in report and "representation_efficiency_population" not in report
    assert set(report["source_pins"]["files"]) == set(experiment.SOURCE_PATHS)
    assert all("presentation" not in row["cycle"]["measurement"] for row in report["cases"])
    assert experiment.validate_report(report, expected_report_sha256=experiment.report_sha256(report))["valid"] is True


@pytest.mark.parametrize("profile", [True, "legacy", "lossless_model_view@2", "", 1])
def test_unsupported_profile_refuses_before_source_or_native_access(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(experiment, "_pins", lambda: pytest.fail("unsupported profile read source"))
    with pytest.raises(ValueError, match="unsupported_presentation_profile"):
        experiment.run_experiment(scratch_root=tmp_path, presentation_profile=profile)


@pytest.mark.parametrize("baseline,original,proposed,mode,counter,reason", [
    ("R" * 400, "O" * 300, "P" * 100, experiment.PRESENTATION_PROFILE, len, "strictly_smaller_initial_inputs"),
    ("R" * 400, "O" * 300, "P" * 300, experiment.PRESENTATION_PROFILE, len, "not_smaller_than_original"),
    ("R" * 100, "O" * 300, "P" * 100, experiment.PRESENTATION_PROFILE, len, "not_smaller_than_raw_baseline"),
    ("R" * 400, "O" * 300, "P" * 100, experiment.PRESENTATION_PROFILE, None, "tokenizer_unavailable"),
    ("R" * 400, "O" * 300, "O" * 300, "original_fallback", len, "unsupported_codec"),
    ("R" * 400, "O" * 300, "P" * 100, experiment.PRESENTATION_PROFILE, lambda _: 10, "not_smaller_than_original"),
])
def test_prelaunch_requires_both_complete_inputs_bytes_and_tokens(baseline, original, proposed, mode, counter, reason):
    value = experiment._presentation_initial(baseline=baseline, original=original,
        proposed=proposed, codec_mode=mode, counter=counter)
    assert value["reason"] == reason
    assert value["efficiency_qualified"] is (reason == "strictly_smaller_initial_inputs")
    assert value["selected_mode"] == ("presented_view" if value["efficiency_qualified"] else "original_packet")


@pytest.mark.parametrize("counter", [None, len])
def test_optin_three_arm_report_keeps_structural_and_efficiency_scope_separate(report, counter):
    value = with_presentation(report, counter=counter)
    checked = experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))
    assert checked["qualified"] is True
    population = value["representation_efficiency_population"]
    assert population["completed_measurement_cases"] == 3
    assert population["presented_deliveries"] == (0 if counter is None else 3)
    assert population["original_fallbacks"] == (3 if counter is None else 0)
    assert population["unavailable_efficiency_cases"] == (3 if counter is None else 0)
    assert checked["provider_usage_verified"] is False


def test_supported_but_not_smaller_proposal_records_genuine_original_delivery(report):
    value = with_presentation(report, counter=len, proposed="P" * 600)
    assert experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))["qualified"] is True
    for row in value["cases"]:
        observed = row["cycle"]["measurement"]["presentation"]
        assert observed["prelaunch"]["selected_mode"] == "original_packet"
        assert observed["delivered_prompt"] == observed["original_packet"]
        assert observed["proposed_view"] != observed["delivered_prompt"]
    assert value["representation_efficiency_population"]["post_delivery_negative_cases"] == 3


def test_unknown_codec_fallback_preserves_original_and_proposal_identity(report):
    value = with_presentation(report, counter=len, proposed="O" * 300, codec_mode="original_fallback")
    assert experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))["valid"] is True
    assert all(row["cycle"]["measurement"]["presentation"]["prelaunch"]["reason"] == "unsupported_codec" for row in value["cases"])


def test_full_trace_negative_does_not_retroactively_claim_original_delivery(report):
    def counter(text):
        return 1000 if text.startswith("P") and text.endswith("COMMON\n") else len(text)
    value = with_presentation(report, counter=counter)
    checked = experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))
    assert checked["qualified"] is True
    assert value["representation_efficiency_population"]["presented_deliveries"] == 3
    assert value["representation_efficiency_population"]["post_delivery_negative_cases"] == 3
    for row in value["cases"]:
        observed = row["cycle"]["measurement"]["presentation"]
        assert observed["prelaunch"]["efficiency_qualified"] is True
        assert observed["post_delivery"]["efficiency_qualified"] is False
        assert observed["delivered_prompt"] == observed["proposed_view"]
        assert row["cycle"]["measurement"]["comparison"]["service_delivery_mode"] == "presented_packet_view"


@pytest.mark.parametrize("fault", ["missing_arm", "extra_arm", "bool_mode", "missing_profile", "missing_sourcepin",
    "wrong_original_inverse", "inverse_false", "actual_delivery_lie", "fake_initial_gain", "fake_trace_gain",
    "double_trace_bytes", "missing_tokens", "bool_population", "wrong_population", "sidecar_tokens", "default_authority",
    "provider_authority", "host_scope", "census_bool", "census_extra", "unknown_census_scope", "fake_comparison"])
def test_optin_rehashed_population_delivery_and_scientific_forgeries_refuse(report, fault):
    value = with_presentation(report, counter=len)
    measured = value["cases"][0]["cycle"]["measurement"]
    observed = measured["presentation"]
    if fault == "missing_arm": observed["renderings"].pop("original_compiled_initial")
    elif fault == "extra_arm": observed["renderings"]["foreign"] = dict(observed["renderings"]["raw_baseline_initial"])
    elif fault == "bool_mode": value["presentation_profile"] = True
    elif fault == "missing_profile": value.pop("presentation_profile")
    elif fault == "missing_sourcepin": value["source_pins"]["files"].pop(experiment.PRESENTATION_SOURCE_PATHS[0])
    elif fault == "wrong_original_inverse": observed["inverse"]["sha256"] = "b" * 64
    elif fault == "inverse_false": observed["inverse"]["exact_original_recovered"] = False
    elif fault == "actual_delivery_lie": observed["delivered_prompt"] = dict(observed["original_packet"])
    elif fault == "fake_initial_gain": observed["prelaunch"]["efficiency_qualified"] = False
    elif fault == "fake_trace_gain": observed["post_delivery"]["efficiency_qualified"] = False
    elif fault == "double_trace_bytes": observed["renderings"]["proposed_presentation_authored_trace"]["bytes"] += 100
    elif fault == "missing_tokens": observed["renderings"]["original_compiled_authored_trace"]["local_o200k_tokens"] = None
    elif fault == "bool_population": value["representation_efficiency_population"]["declared_cases"] = True
    elif fault == "wrong_population": value["representation_efficiency_population"]["presented_deliveries"] = 0
    elif fault == "sidecar_tokens": observed["sidecar_added_to_model_tokens"] = True
    elif fault == "default_authority": observed["default_adoption_authority"] = True
    elif fault == "provider_authority": observed["provider_usage_verified"] = True
    elif fault == "host_scope": observed["sidecar"]["count_scope"] = "model_input"
    elif fault == "census_bool": observed["codec_census"]["counts"]["shared_fact_count"] = True
    elif fault == "census_extra": observed["codec_census"]["counts"]["raw_source"] = "hidden"
    elif fault == "unknown_census_scope": observed["codec_census"]["scope"] = "measured_token_causality"
    else: measured["comparison"]["measurements"]["candidate_initial"] = dict(observed["renderings"]["original_compiled_initial"])
    with pytest.raises(ValueError):
        experiment.validate_report(value, expected_report_sha256=reseal(value))


def test_optin_independent_report_pin_rejects_resealed_sidecar_change(report):
    value = with_presentation(report, counter=len)
    pin = experiment.report_sha256(value)
    value["cases"][0]["cycle"]["measurement"]["presentation"]["sidecar"]["sha256"] = "f" * 64
    reseal(value)
    with pytest.raises(ValueError, match="independent_packet_report_pin"):
        experiment.validate_report(value, expected_report_sha256=pin)


def test_three_full_traces_are_counted_directly_not_by_adding_local_counts():
    seen = []
    def counter(text):
        seen.append(text)
        return len(text)
    initial = experiment._presentation_initial(baseline="RAW", original="ORIGINAL", proposed="P",
        codec_mode=experiment.PRESENTATION_PROFILE, counter=counter)
    experiment._presentation_finish(initial=initial, baseline="RAW", original="ORIGINAL", proposed="P",
        common_trace="COMMON", counter=counter, codec_mode=experiment.PRESENTATION_PROFILE,
        codec_receipt_sha256="a" * 64, codec_census=dict.fromkeys(experiment.PRESENTATION_CENSUS_FIELDS, 0),
        sidecar_bytes=b"HOST", inverse_original="ORIGINAL", native_prompt="ORIGINAL", proposed_prompt="P", delivered_prompt="P")
    assert seen == ["RAW", "ORIGINAL", "P", "COMMON", "RAWCOMMON", "ORIGINALCOMMON", "PCOMMON"]


def test_inverse_mismatch_cannot_make_a_presentation_observation():
    initial = experiment._presentation_initial(baseline="RAW", original="ORIGINAL", proposed="P",
        codec_mode=experiment.PRESENTATION_PROFILE, counter=len)
    with pytest.raises(ValueError, match="presentation_inverse_original"):
        experiment._presentation_finish(initial=initial, baseline="RAW", original="ORIGINAL", proposed="P",
            common_trace="COMMON", counter=len, codec_mode=experiment.PRESENTATION_PROFILE,
            codec_receipt_sha256="a" * 64, codec_census=dict.fromkeys(experiment.PRESENTATION_CENSUS_FIELDS, 0),
            sidecar_bytes=b"HOST", inverse_original="FOREIGN", native_prompt="ORIGINAL", proposed_prompt="P", delivered_prompt="P")


def test_optin_missing_profile_still_retains_closed_unmeasured_population(tmp_path, monkeypatch, report):
    pins = deepcopy(report["source_pins"])
    pins["files"].update({name: dict(bytes=10, sha256="a" * 64) for name in experiment.PRESENTATION_SOURCE_PATHS})
    def fake_pins(**kwargs):
        assert kwargs == dict(presentation_profile=experiment.PRESENTATION_PROFILE)
        return pins
    monkeypatch.setattr(experiment, "_pins", fake_pins)
    def refused(): raise ValueError("actual_owner_identity_required")
    monkeypatch.setattr(experiment, "_profile", refused)
    value = experiment.run_experiment(scratch_root=tmp_path, presentation_profile=experiment.PRESENTATION_PROFILE)
    assert value["representation_efficiency_population"]["unmeasured_cases"] == 3
    assert value["representation_efficiency_population"]["presented_deliveries"] == 0
    assert experiment.validate_report(value, expected_report_sha256=experiment.report_sha256(value))["qualified"] is False
