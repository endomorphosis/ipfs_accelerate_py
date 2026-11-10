"""The opt-in public acceptance policy is frozen with its exact task profile."""
import asyncio
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_coding_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import declaration
from benchmarks.agent_supervisor.container_coding.test_terminal_doctor_coding_policy import inventory, manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_public_acceptance_profile import (
    INSTRUCTION_TEXT, public_profile,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

SELECTED = dict(candidate_rounds=2, doctor_coding_policy=policy.PUBLIC_VALIDATION_POLICY,
    source_view_profile="identity@1")


def test_new_policy_is_explicit_and_freezes_exact_profile_without_changing_old_controls(tmp_path):
    arguments = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    profile = public_profile()
    selected = benchmark.config_for(*arguments, task_profile=profile, **SELECTED)
    old = benchmark.config_for(*arguments, task_profile=profile,
        **{**SELECTED, "doctor_coding_policy": policy.DOCTOR_CODING_POLICY})
    assert selected["agents"][0]["kwargs"] == {
        **old["agents"][0]["kwargs"], "doctor_coding_policy": policy.PUBLIC_VALIDATION_POLICY}
    new_controls, old_controls = declaration(selected), declaration(old)
    assert new_controls["schema"] == old_controls["schema"] == controls.DOCTOR_CODING_SCHEMA
    assert new_controls["experimental_controls_sha256"]["doctor_coding_policy"] != old_controls["experimental_controls_sha256"]["doctor_coding_policy"]
    assert new_controls["configuration_sha256"] != old_controls["configuration_sha256"]
    assert controls.validate_frozen_candidate_rounds(dict(arm="full", **SELECTED), selected) == 2


@pytest.mark.parametrize("change", ["missing", "legacy", "wrong_profile", "instruction", "extra", "output"])
@pytest.mark.parametrize("entrypoint", ["config", "prepare", "harbor", "driver"])
def test_profile_mismatch_refuses_before_setup_assets_or_provider(tmp_path, monkeypatch, change, entrypoint):
    profile = public_profile()
    if change == "missing": profile = None
    elif change == "legacy":
        profile["schema"] = "terminal-public-task-profile@1"
        profile.pop("public_validation_profile")
    elif change == "wrong_profile": profile["public_validation_profile"] = "authored-unreviewed"
    elif change == "instruction": profile["instruction_sha256"] = "a" * 64
    elif change == "extra": profile["command"] = ["authored-command"]
    else: profile["outputs"][0]["path"] = "foreign.py"
    path = None
    if profile is not None:
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(profile))
    monkeypatch.setattr(benchmark, "resolve_provider_profile", lambda *_: pytest.fail("provider selection reached"))
    monkeypatch.setattr(harbor.BaseAgent, "__init__", lambda *a, **k: pytest.fail("base setup reached"))
    monkeypatch.setattr(driver.preparation, "prepare", lambda *a, **k: pytest.fail("planning reached"))
    with pytest.raises(ValueError):
        if entrypoint == "config":
            benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
                task_profile=profile, **SELECTED)
        elif entrypoint == "prepare":
            benchmark.prepare(dataset=tmp_path / "missing-data", output=tmp_path / "out",
                archive=tmp_path / "missing-archive", arm="full", task_profile=path, **SELECTED)
        elif entrypoint == "harbor":
            harbor.FullSupervisorAgent(runtime_archive=str(tmp_path / "missing-archive"),
                task_profile=profile, **SELECTED)
        else:
            driver.run(instruction=tmp_path / "missing", state=tmp_path / "state", arm="full",
                task_profile=path, **SELECTED)
    assert not (tmp_path / "out").exists() and not (tmp_path / "state").exists()


@pytest.mark.parametrize("kind", ["symlink", "duplicate", "oversize", "nonfinite"])
def test_early_profile_file_loader_is_bounded_and_closed(tmp_path, kind):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(public_profile()))
    if kind == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    elif kind == "duplicate": path.write_text('{"schema":"foreign",' + path.read_text()[1:])
    elif kind == "oversize": path.write_text(" " * 65537)
    else: path.write_text('{"schema":NaN}')
    with pytest.raises(ValueError):
        controls.validate_full_supervisor_candidate_rounds(2, arm="full", task_profile=path,
            doctor_coding_policy=policy.PUBLIC_VALIDATION_POLICY, source_view_profile="identity@1")


@pytest.mark.parametrize("operation", ["execute", "collect"])
@pytest.mark.parametrize("change", ["omit", "legacy", "profile", "policy"])
def test_frozen_public_validation_selection_refuses_changed_binding(tmp_path, monkeypatch, operation, change):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        task_profile=public_profile(), **SELECTED)
    prepared = dict(arm="full", **SELECTED)
    kwargs = config["agents"][0]["kwargs"]
    if change == "omit": kwargs.pop("task_profile")
    elif change == "legacy":
        kwargs["task_profile"]["schema"] = "terminal-public-task-profile@1"
        kwargs["task_profile"].pop("public_validation_profile")
    elif change == "profile": kwargs["task_profile"]["public_validation_profile"] = "foreign"
    else: kwargs["doctor_coding_policy"] = policy.DOCTOR_CODING_POLICY
    (tmp_path / "preparation.json").write_text(json.dumps(prepared))
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(benchmark, "_prepared_task", lambda *a, **k: pytest.fail("trial reached"))
    with pytest.raises(ValueError):
        getattr(benchmark, operation)(tmp_path)


@pytest.mark.parametrize("operation", ["setup", "run"])
def test_harbor_rechecks_public_profile_before_environment_access(operation):
    agent = object.__new__(harbor.FullSupervisorAgent)
    agent.arm = "full"
    agent.source_view_profile = "identity@1"
    agent.candidate_rounds = agent._frozen_candidate_rounds = 2
    agent.doctor_coding_policy = agent._frozen_doctor_coding_policy = policy.PUBLIC_VALIDATION_POLICY
    agent.task_profile = public_profile()
    agent.task_profile["public_validation_profile"] = "foreign"
    with pytest.raises(ValueError, match="signed @4"):
        asyncio.run(agent.setup(None) if operation == "setup" else agent.run("authored", None, None))


@pytest.mark.parametrize("relative", [
    "ipfs_accelerate_py/agent_supervisor/runtime/candidate_public_validation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/candidate_execution.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_source_partition.py",
])
def test_selected_archive_pins_public_contract_checker_and_verifier(relative):
    value = inventory()
    policy.require_doctor_coding_archive(value, 2, policy.PUBLIC_VALIDATION_POLICY)
    row = next(row for row in value["files"] if row["path"] == "source/" + relative)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified"):
        policy.require_doctor_coding_archive(value, 2, policy.PUBLIC_VALIDATION_POLICY)


def test_dry_prepare_freezes_new_selection_and_profile(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    (task / "instruction.md").write_text(INSTRUCTION_TEXT)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(public_profile()))
    archive = tmp_path / "archive"
    archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    value = manifest()
    value["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kw:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry run", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", task_profile=profile_path, **SELECTED)
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert prepared["provider_calls"] == 0 and len(calls) == 1 and "--dry-run" in calls[0]
    assert controls.validate_frozen_candidate_rounds(prepared, config) == 2
    assert config["agents"][0]["kwargs"]["task_profile"] == public_profile()
    assert prepared["doctor_coding_policy_observation"]["policy"] == policy.PUBLIC_VALIDATION_POLICY


def test_harbor_actual_command_forwards_new_policy_and_profile(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr(harbor, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(harbor, "export_public_outputs", AsyncMock(return_value={}))
    calls, uploaded = [], {}
    class Environment:
        async def upload_file(self, source, target):
            uploaded[target] = Path(source).read_bytes()
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            report = {"task_completed": False, "seconds": 1, "phases": {}, "provider_invocations": [],
                **policy.doctor_coding_observation(2, policy.PUBLIC_VALIDATION_POLICY)}
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    agent = harbor.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
        runtime_archive=str(tmp_path), task_profile=public_profile(), **SELECTED)
    context = AgentContext()
    asyncio.run(agent.run(INSTRUCTION_TEXT, Environment(), context))
    assert len(calls) == 1
    argv = calls[0]
    for flag, value in (("--candidate-rounds", "2"), ("--doctor-coding-policy", policy.PUBLIC_VALIDATION_POLICY)):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert json.loads(uploaded[argv[argv.index("--task-profile") + 1]]) == public_profile()
    assert context.metadata["doctor_coding_runtime_declared_matches"] is True


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
def test_owner_argv_preserves_selection_and_provider_free_routes(route):
    dispatch = dict(route=route, status="residual" if route == "model_router" else "candidate_ready",
        provider_calls=0, artifact="/authored", sha256="a" * 64, task_cid="authored-task",
        residual_context=dict(artifact="/authored", sha256="a" * 64, task_cid="authored-task"))
    argv = doctor.implementation_argv(router=Path("/router"), model="authored", reasoning="high", timeout=30,
        semantic_repository=Path("/repository"), doctor=dispatch, **SELECTED)
    if route == "model_router":
        assert argv[argv.index("--doctor-coding-policy") + 1] == policy.PUBLIC_VALIDATION_POLICY
    else:
        assert "--doctor-coding-policy" not in argv and "--model" not in argv


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
def test_actual_driver_preserves_new_policy_and_public_profile_in_report(tmp_path, monkeypatch, route):
    from test.api.test_terminal_doctor_dispatch import (
        test_driver_selects_before_owner_and_preserves_provider_accounting as exercise,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(public_profile()))
    (tmp_path / "instruction").write_text(INSTRUCTION_TEXT)
    original, reports = driver.run, []
    def selected(**kwargs):
        result = original(**kwargs, task_profile=profile_path)
        reports.append(result)
        return result
    monkeypatch.setattr(driver, "run", selected)
    exercise(tmp_path, monkeypatch, "full", route, False,
        "supervisor-semantic-router-input@1", controls=SELECTED)
    assert len(reports) == 1 and reports[0]["task_profile"] == public_profile()


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
@pytest.mark.parametrize("change", [None, "missing", "foreign", "no_materialization"])
def test_provider_free_observation_requires_exact_public_profile_and_materialization(route, change):
    from benchmarks.agent_supervisor.container_coding.test_terminal_doctor_coding_policy import _optional_provider_free
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import provider_free_coding_application_observed
    report = _optional_provider_free(route, "both", True)
    report.update(policy.doctor_coding_observation(2, policy.PUBLIC_VALIDATION_POLICY))
    report["coding_control_application"]["requested"].update(
        policy.doctor_coding_selection(2, policy.PUBLIC_VALIDATION_POLICY))
    report["task_profile"] = public_profile()
    if change == "missing": report.pop("task_profile")
    elif change == "foreign": report["task_profile"]["public_validation_profile"] = "foreign"
    elif change == "no_materialization": report.pop("doctor_invocations")
    assert provider_free_coding_application_observed(report) is (change is None)


@pytest.mark.parametrize("change", [None, "authority", "owner_code", "unknown", "bool_round", "extra"])
def test_driver_retains_only_closed_new_owner_lifecycle_markers(change):
    marker = dict(schema="owner-candidate-rounds-observation@2", rounds=[dict(round_number=1,
        process_started=True, session_id="a" * 32, descriptor_sha256="b" * 64,
        returncode=1, wait_completed=True, retirement_observed=True)], max_rounds=2,
        elapsed_seconds=1.0, outcome="failed", failure_type="ValueError",
        completion_authority=False, proof_authority=False, source_obligations_discharged=0,
        accepted_index_published=False, public_validation_observations=[],
        owner_validation_outcome=None, owner_returncode=None, owner_lifecycle_retention_unavailable=True)
    if change == "authority": marker["completion_authority"] = True
    elif change == "owner_code": marker["owner_returncode"] = 0
    elif change == "unknown": marker["schema"] = "owner-candidate-rounds-observation@3"
    elif change == "bool_round": marker["rounds"][0]["round_number"] = True
    elif change == "extra": marker["command"] = "unreviewed"
    assert driver._owner_candidate_round_marker_valid(marker) is (change is None)
    assert driver._owner_candidate_round_marker_valid(dict(schema="owner-candidate-rounds-observation@1"))


@pytest.mark.parametrize("entrypoint", ["benchmark", "driver"])
def test_cli_parser_accepts_only_explicit_policy_selector_and_forwards_profile(tmp_path, monkeypatch, entrypoint):
    class Reached(Exception): pass
    observed = {}
    def capture(**kwargs):
        observed.update(kwargs)
        raise Reached()
    profile = str(tmp_path / "profile.json")
    common = ["--candidate-rounds", "2", "--doctor-coding-policy", policy.PUBLIC_VALIDATION_POLICY,
        "--source-view-profile", "identity@1", "--task-profile", profile]
    if entrypoint == "benchmark":
        monkeypatch.setattr(benchmark, "prepare", capture)
        argv = ["benchmark", "prepare", "--dataset", str(tmp_path), "--output", str(tmp_path / "out"),
            "--archive", str(tmp_path / "archive"), "--arm", "full", *common]
        entry = benchmark.main
    else:
        monkeypatch.setattr(driver, "run", capture)
        argv = ["driver", "--instruction", str(tmp_path / "instruction"), "--state", str(tmp_path / "state"),
            "--arm", "full", *common]
        entry = driver.main
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(Reached): entry()
    assert observed["doctor_coding_policy"] == policy.PUBLIC_VALIDATION_POLICY
    assert observed["task_profile"] == Path(profile) and observed["candidate_rounds"] == 2
