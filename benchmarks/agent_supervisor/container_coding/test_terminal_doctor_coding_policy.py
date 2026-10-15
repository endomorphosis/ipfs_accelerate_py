"""Explicit Doctor attempt policy reaches the signed launch without extra calls."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding import terminal_doctor_coding_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import declaration
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

SELECTED = dict(candidate_rounds=2, doctor_coding_policy=policy.DOCTOR_CODING_POLICY,
    source_view_profile="identity@1")


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name, "bytes": (root / name).stat().st_size,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in policy.SOURCE_PATHS]}


def manifest():
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
    return _supervisor_manifest({**inventory(), "archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "learned_requirements": []})


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_keeps_configuration_and_controls_identical(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    baseline = benchmark.config_for(*args)
    explicit = benchmark.config_for(*args, candidate_rounds=1, doctor_coding_policy=None)
    assert baseline == explicit and declaration(baseline) == declaration(explicit)
    assert "candidate_rounds" not in baseline["agents"][0]["kwargs"]
    assert "doctor_coding_policy" not in baseline["agents"][0]["kwargs"]
    policy.require_doctor_coding_archive({}, 1, None)


def test_exact_opt_in_is_frozen_in_closed_comparison_controls(tmp_path):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full", **SELECTED)
    kwargs = config["agents"][0]["kwargs"]
    assert all(kwargs[key] == value for key, value in SELECTED.items())
    frozen = declaration(config)
    assert frozen["schema"] == controls.DOCTOR_CODING_SCHEMA and controls.validate_controls(frozen)
    assert set(frozen["experimental_controls_sha256"]) == {"candidate_rounds", "doctor_coding_policy"}
    assert controls.validate_frozen_candidate_rounds(dict(arm="full", **SELECTED), config) == 2
    for key in ("candidate_rounds", "doctor_coding_policy"):
        missing = deepcopy(frozen)
        missing["experimental_controls_sha256"].pop(key)
        assert not controls.validate_controls(missing)


@pytest.mark.parametrize("change", ["missing_policy", "wrong_policy", "one_round", "no_index", "no_source", "mapped_edits", "numerical"])
def test_invalid_pair_refuses_before_assets_or_provider_selection(tmp_path, monkeypatch, change):
    selected, arm = dict(SELECTED), "full"
    if change == "missing_policy": selected.pop("doctor_coding_policy")
    elif change == "wrong_policy": selected["doctor_coding_policy"] = "doctor-coding-two-attempts@2"
    elif change == "one_round": selected["candidate_rounds"] = 1
    elif change == "no_index": arm = "no-index"
    elif change == "mapped_edits": selected["source_edit_mode"] = "mapped-span@2"
    elif change == "numerical": selected["numerical_formalization_profile"] = "python-numerical-control-structure@3"
    else: selected["source_view_profile"] = "disabled"
    monkeypatch.setattr(benchmark, "resolve_provider_profile", lambda *_: pytest.fail("provider discovery reached"))
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing-archive", arm=arm, **selected)
    with pytest.raises(ValueError):
        benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", arm, **selected)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("operation", ["execute", "collect"])
@pytest.mark.parametrize("where", ["prepared", "config"])
@pytest.mark.parametrize("change", ["omit_policy", "change_policy", "omit_rounds", "omit_both", "add_numerical"])
def test_saved_trial_cannot_omit_or_substitute_policy(tmp_path, monkeypatch, operation, where, change):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full", **SELECTED)
    prepared = dict(arm="full", **SELECTED)
    target = prepared if where == "prepared" else config["agents"][0]["kwargs"]
    if change in {"omit_policy", "omit_both"}: target.pop("doctor_coding_policy")
    if change in {"omit_rounds", "omit_both"}: target.pop("candidate_rounds")
    if change == "change_policy": target["doctor_coding_policy"] = "foreign"
    if change == "add_numerical": target["numerical_formalization_profile"] = "python-numerical-control-structure@3"
    (tmp_path / "preparation.json").write_text(json.dumps(prepared))
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(benchmark, "_prepared_task", lambda *_a, **_k: pytest.fail("trial execution reached"))
    with pytest.raises(ValueError):
        getattr(benchmark, operation)(tmp_path)


@pytest.mark.parametrize("change", ["missing", "hash", "size", "duplicate"])
def test_selected_archive_binds_current_runtime_and_benchmark_sources(change):
    selected = inventory()
    policy.require_doctor_coding_archive(selected, 2, policy.DOCTOR_CODING_POLICY)
    if change == "missing": selected["files"].pop()
    elif change == "hash": selected["files"][0]["sha256"] = "f" * 64
    elif change == "size": selected["files"][0]["bytes"] = True
    else: selected["files"].append(selected["files"][0])
    with pytest.raises(ValueError, match="newly qualified"):
        policy.require_doctor_coding_archive(selected, 2, policy.DOCTOR_CODING_POLICY)


@pytest.mark.parametrize("change", ["missing", "hash", "size"])
def test_selected_archive_binds_candidate_prompt_codec(change):
    selected = inventory()
    codec = next(row for row in selected["files"]
        if row["path"].endswith("/candidate_prompt_projection.py"))
    if change == "missing": selected["files"].remove(codec)
    elif change == "hash": codec["sha256"] = "f" * 64
    else: codec["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified"):
        policy.require_doctor_coding_archive(selected, 2, policy.DOCTOR_CODING_POLICY)


def test_dry_prepare_freezes_the_pair_and_attempt_scope(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    value = manifest()
    value["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **_kw:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry run", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", **SELECTED)
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert prepared["provider_calls"] == 0 and len(calls) == 1 and "--dry-run" in calls[0]
    assert controls.validate_frozen_candidate_rounds(prepared, config) == 2
    assert prepared["doctor_coding_policy_observation"]["provider_tokens_and_cost_observation_only"] is True
    assert prepared["comparison_controls"]["schema"] == controls.DOCTOR_CODING_SCHEMA


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
def test_driver_reaches_actual_admitted_command_builder_with_selection(tmp_path, monkeypatch, route):
    from test.api.test_terminal_doctor_dispatch import (
        test_driver_selects_before_owner_and_preserves_provider_accounting as exercise,
    )
    original, seen = doctor.implementation_argv, []
    def capture(**kwargs):
        argv = original(**kwargs)
        seen.append((kwargs, argv))
        return argv
    monkeypatch.setattr(doctor, "implementation_argv", capture)
    exercise(tmp_path, monkeypatch, "full", route, False,
        "supervisor-semantic-router-input@1", controls=SELECTED)
    assert len(seen) == 1
    kwargs, argv = seen[0]
    assert kwargs["candidate_rounds"] == 2 and kwargs["doctor_coding_policy"] == policy.DOCTOR_CODING_POLICY
    if route == "model_router":
        for flag, value in (("--candidate-rounds", "2"), ("--doctor-coding-policy", policy.DOCTOR_CODING_POLICY),
                ("--doctor-residual-sha256", "d" * 64), ("--doctor-residual-task-cid", "task-cid")):
            assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    else:
        assert "--candidate-rounds" not in argv and "--doctor-coding-policy" not in argv
        assert "--model" not in argv


@pytest.mark.parametrize("operation", ["setup", "run"])
@pytest.mark.parametrize("change", ["policy", "rounds", "remove_both", "numerical"])
def test_harbor_rejects_changed_policy_before_environment_access(operation, change):
    agent = object.__new__(harbor.FullSupervisorAgent)
    agent.arm = "full"
    agent.source_view_profile = "identity@1"
    agent.candidate_rounds = agent._frozen_candidate_rounds = 2
    agent.doctor_coding_policy = agent._frozen_doctor_coding_policy = policy.DOCTOR_CODING_POLICY
    if change == "policy": agent.doctor_coding_policy = None
    elif change == "rounds": agent.candidate_rounds = 1
    elif change == "numerical": agent.numerical_formalization_profile = "python-numerical-control-structure@3"
    else:
        del agent.candidate_rounds
        del agent.doctor_coding_policy
    with pytest.raises(ValueError):
        asyncio.run(agent.setup(None) if operation == "setup" else agent.run("authored", None, None))


@pytest.mark.parametrize("profile", benchmark.NUMERICAL_PROFILES)
@pytest.mark.parametrize("entrypoint", ["driver", "harbor"])
def test_numerical_continuation_refuses_before_planning_or_base_setup(tmp_path, monkeypatch, profile, entrypoint):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    monkeypatch.setattr(driver.preparation, "prepare", lambda *a, **kw: pytest.fail("planning reached"))
    monkeypatch.setattr(harbor.BaseAgent, "__init__", lambda *a, **kw: pytest.fail("base setup reached"))
    with pytest.raises(ValueError, match="numerical formalization broker"):
        if entrypoint == "driver":
            driver.run(instruction=tmp_path / "missing", state=tmp_path / "state", arm="full",
                numerical_formalization_profile=profile, **SELECTED)
        else:
            harbor.FullSupervisorAgent(runtime_archive=str(tmp_path / "missing"),
                numerical_formalization_profile=profile, **SELECTED)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("profile", benchmark.NUMERICAL_PROFILES)
def test_frozen_controls_cannot_add_numerical_broker_to_two_attempts(tmp_path, profile):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full", **SELECTED)
    config["agents"][0]["kwargs"]["numerical_formalization_profile"] = profile
    with pytest.raises(ValueError, match="numerical formalization broker"):
        declaration(config)


@pytest.mark.parametrize("profile", benchmark.NUMERICAL_PROFILES)
@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_one_attempt_numerical_config_and_controls_are_unchanged(tmp_path, arm, profile):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    kwargs = dict(numerical_formalization_profile=profile)
    default = benchmark.config_for(*args, **kwargs)
    explicit = benchmark.config_for(*args, **kwargs, candidate_rounds=1, doctor_coding_policy=None)
    assert default == explicit and declaration(default) == declaration(explicit)
    assert controls.validate_frozen_candidate_rounds(dict(arm=arm, **kwargs), default) == 1


def test_harbor_forwards_pair_and_reports_declaration_with_observed_usage(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr(harbor, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(harbor, "export_public_outputs", AsyncMock(return_value={}))
    calls = []
    class Environment:
        async def upload_file(self, *_): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            report = {"task_completed": False, "seconds": 1, "phases": {},
                **policy.doctor_coding_observation(2, policy.DOCTOR_CODING_POLICY),
                "provider_invocations": [{"invocation_id": "authored", "phase": "coding", "purpose": "coding",
                    "native_rollout_usage": {"usage": {"input_tokens": 2, "cached_input_tokens": 0,
                        "output_tokens": 1, "total_tokens": 3}}}]}
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    agent = harbor.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
        runtime_archive=str(tmp_path), **SELECTED)
    context = AgentContext()
    asyncio.run(agent.run("Authored task", Environment(), context))
    assert len(calls) == 1 and context.metadata["usage"]["total_tokens"] == 3
    argv = calls[0]
    for flag, value in (("--candidate-rounds", "2"), ("--doctor-coding-policy", policy.DOCTOR_CODING_POLICY)):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert context.metadata["doctor_coding_runtime_declared_matches"] is True


def test_runtime_declaration_flags_require_exact_boolean_types():
    report = policy.doctor_coding_observation(2, policy.DOCTOR_CODING_POLICY)
    assert policy.doctor_coding_runtime_declared_matches(report, 2, policy.DOCTOR_CODING_POLICY)
    report["doctor_coding_policy_observation"]["native_token_cap_enforced"] = 0
    assert not policy.doctor_coding_runtime_declared_matches(report, 2, policy.DOCTOR_CODING_POLICY)


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_provider_free_selected_policy_needs_no_unused_semantic_repository(route):
    argv = doctor.implementation_argv(router=Path("/router"), model="unused", reasoning="high", timeout=30,
        semantic_repository=None, candidate_rounds=2, doctor_coding_policy=policy.DOCTOR_CODING_POLICY,
        doctor=dict(route=route, status="candidate_ready", provider_calls=0, artifact="/authored",
            sha256="a" * 64, task_cid="authored-task"))
    assert "--model" not in argv and "--candidate-rounds" not in argv and "--doctor-coding-policy" not in argv


@pytest.mark.parametrize("missing", ["residual_context", "artifact", "sha256", "task_cid"])
def test_selected_residual_requires_complete_context_binding(missing):
    dispatch = dict(route="model_router", status="residual", residual_context={
        "artifact": "/authored", "sha256": "a" * 64, "task_cid": "authored-task"})
    if missing == "residual_context": dispatch.pop(missing)
    else: dispatch["residual_context"].pop(missing)
    with pytest.raises(ValueError, match="exact residual context"):
        doctor.implementation_argv(router=Path("/router"), model="authored", reasoning="high", timeout=30,
            semantic_repository=Path("/repository"), doctor=dispatch, **SELECTED)


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
@pytest.mark.parametrize("change", [None, "missing_policy", "missing_rounds", "requested_drift",
    "missing_observation", "fake_cap", "missing_materialization", "mapped_edits"])
def test_observed_provider_free_materialization_accepts_only_exact_selected_pair(route, change):
    from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import _provider_free_report
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import provider_free_coding_application_observed
    report = _provider_free_report(route)
    report["source_edit_mode"] = report["coding_control_application"]["requested"]["source_edit_mode"] = "disabled"
    report.update(policy.doctor_coding_observation(2, policy.DOCTOR_CODING_POLICY))
    report["coding_control_application"]["requested"].update(policy.doctor_coding_selection(2, policy.DOCTOR_CODING_POLICY))
    if change == "missing_policy": report.pop("doctor_coding_policy")
    elif change == "missing_rounds": report.pop("candidate_rounds")
    elif change == "requested_drift": report["coding_control_application"]["requested"]["candidate_rounds"] = 1
    elif change == "missing_observation": report.pop("doctor_coding_policy_observation")
    elif change == "fake_cap": report["doctor_coding_policy_observation"]["native_token_cap_enforced"] = 0
    elif change == "missing_materialization": report.pop("doctor_invocations")
    elif change == "mapped_edits": report["source_edit_mode"] = "mapped-span@2"
    assert provider_free_coding_application_observed(report) is (change is None)


def _optional_provider_free(route, optional, selected):
    from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import _provider_free_report
    from benchmarks.agent_supervisor.container_coding.terminal_semantic_working_set_policy import semantic_working_set_observation
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE, resolve_provider_profile
    report = _provider_free_report(route)
    requested = report["coding_control_application"]["requested"]
    report["source_edit_mode"] = requested["source_edit_mode"] = "disabled"
    report["semantic_metadata_view"]["requested_mode"] = requested["semantic_metadata_view"] = "legacy"
    if selected:
        report.update(policy.doctor_coding_observation(2, policy.DOCTOR_CODING_POLICY))
        requested.update(policy.doctor_coding_selection(2, policy.DOCTOR_CODING_POLICY))
    if optional in {"working", "both"}:
        report.update(semantic_working_set_observation("capsule-references@1"))
        requested["semantic_working_set"] = "capsule-references@1"
    if optional in {"grok", "both"}:
        report["provider_profile"] = resolve_provider_profile(GROK_PROFILE)
        report["grok_coding_max_turns"] = requested["grok_coding_max_turns"] = 16
    return report


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
@pytest.mark.parametrize("optional", ["working", "grok", "both"])
@pytest.mark.parametrize("selected", [False, True])
def test_observed_materialization_accepts_valid_optional_frozen_controls(route, optional, selected):
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import provider_free_coding_application_observed
    assert provider_free_coding_application_observed(_optional_provider_free(route, optional, selected))


@pytest.mark.parametrize("change", ["working", "working_missing", "working_scope", "working_authority",
    "working_false_type", "requested_working", "turns_type", "requested_turns_type", "turns_missing",
    "provider_missing", "provider_changed", "provider_substituted", "extra_request", "materialization_missing"])
def test_optional_controls_cannot_hide_malformed_or_conflicting_materialization(change):
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import provider_free_coding_application_observed
    report = _optional_provider_free("doctor_candidate", "both", True)
    requested = report["coding_control_application"]["requested"]
    if change == "working": report["semantic_working_set"] = "foreign"
    elif change == "working_missing": report.pop("semantic_working_set")
    elif change == "working_scope": report["semantic_metadata_view"]["requested_mode"] = "common-bindings@1"
    elif change == "working_authority": report["semantic_working_set_proof_authority"] = True
    elif change == "working_false_type": report["semantic_working_set_proof_authority"] = 0
    elif change == "requested_working": requested["semantic_working_set"] = "legacy"
    elif change == "turns_type": report["grok_coding_max_turns"] = True
    elif change == "requested_turns_type": requested["grok_coding_max_turns"] = 16.0
    elif change == "turns_missing": report.pop("grok_coding_max_turns")
    elif change == "provider_missing": report.pop("provider_profile")
    elif change == "provider_changed": report["provider_profile"]["provider"] = "codex_cli"
    elif change == "provider_substituted": report["provider_profile"]["model"] = "foreign"
    elif change == "extra_request": requested["authority"] = True
    else: report.pop("doctor_invocations")
    assert not provider_free_coding_application_observed(report)


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
def test_grok_capsule_profile_keeps_attempt_policy_on_the_actual_admitted_command(tmp_path, monkeypatch, route):
    from benchmarks.agent_supervisor.container_coding import benchmark_provider_profile as profiles
    from test.api.test_terminal_doctor_dispatch import (
        test_driver_selects_before_owner_and_preserves_provider_accounting as exercise,
    )
    selected = profiles.resolve_provider_profile(profiles.GROK_PROFILE)
    monkeypatch.setattr(profiles, "resolve_provider_profile", lambda _name=None: dict(selected))
    exercise(tmp_path, monkeypatch, "full", route, False, "supervisor-semantic-router-input@1",
        controls={**SELECTED, "semantic_working_set": "capsule-references@1", "grok_coding_max_turns": 16})
