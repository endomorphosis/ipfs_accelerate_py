"""Authored pruning declarations and outer plumbing; no model/container calls."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ipfs_accelerate_py.cli_runtime import grok_pruning as pruning
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE, resolve_provider_profile
from benchmarks.agent_supervisor.container_coding.test_grok_coding_budget import manifest as base_manifest, seal
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task


OFF = "grok-pruning-off@1"
ON = "grok-pruning-on@1"
PLAN = "declared-contract@1"


def manifest():
    from benchmarks.agent_supervisor.container_coding import terminal_planning_policy as planning
    root = Path(benchmark.__file__).resolve().parents[3]
    value = base_manifest()
    rows = {row["path"]: row for row in value["files"]}
    for relative in {*pruning.SOURCE_PATHS, *planning.SOURCE_PATHS}:
        path = root / relative
        rows["source/" + relative] = {"path": "source/" + relative,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    value["files"] = list(rows.values())
    return value


def frozen_trial(tmp_path, profile=ON):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        provider_profile=GROK_PROFILE, planning_mode=PLAN, grok_pruning_profile=profile)
    (output / "config.json").write_text(json.dumps(config))
    selected = resolve_provider_profile(GROK_PROFILE)
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "provider_profile": GROK_PROFILE, "planning_mode": PLAN,
        **pruning.grok_pruning_selection(profile),
        **{key: selected[key] for key in ("model", "reasoning_effort", "cli_version")}}
    (output / "preparation.json").write_text(json.dumps(prepared))
    return output, prepared, config


def retained_report(output, prepared, config, report):
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored"; (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"task_name": benchmark.TASK,
        "config": {"task": {"path": str(Path(prepared["dataset"]) / benchmark.TASK)}}}))
    (job / "result.json").write_text("{}")
    (trial / "agent/supervisor-result.json").write_text(json.dumps(report))


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("provider", [None, GROK_PROFILE])
def test_legacy_keeps_configuration_and_declaration_exact(tmp_path, arm, provider):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    original = benchmark.config_for(*args, provider_profile=provider)
    explicit = benchmark.config_for(*args, provider_profile=provider, grok_pruning_profile="legacy")
    assert original == explicit
    assert "grok_pruning_profile" not in explicit["agents"][0]["kwargs"]
    assert seal(original) == seal(explicit)
    assert seal(explicit)["schema"] == controls.SCHEMA


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("profile", [OFF, ON])
def test_explicit_profiles_are_sealed_independently(tmp_path, arm, profile):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    base = benchmark.config_for(*args, provider_profile=GROK_PROFILE, planning_mode=PLAN)
    chosen = benchmark.config_for(*args, provider_profile=GROK_PROFILE,
        planning_mode=PLAN, grok_pruning_profile=profile)
    assert chosen["agents"][0]["kwargs"]["grok_pruning_profile"] == profile
    original, selected = seal(base), seal(chosen)
    assert selected["schema"] == controls.GROK_PRUNING_SCHEMA
    assert controls.validate_controls(selected)
    observe = lambda value: dict(status="observed", configuration_unchanged=True, declared=value)
    assert controls.compare_controls(observe(original), observe(selected))["differences"] == [
        "experimental_controls_sha256.grok_pruning_profile"]
    changed = deepcopy(selected)
    changed["experimental_controls_sha256"]["planning_mode"] = controls._digest("model")
    changed["sha256"] = controls._digest({key: value for key, value in changed.items() if key != "sha256"})
    assert controls.validate_controls(changed) is False


@pytest.mark.parametrize("extra", [
    {"provider": "codex_cli"}, {"metadata_census_only": True}, {"working_set_census_only": True},
    {"candidate_rounds": 2}, {"candidate_rounds": True}, {"doctor_coding_policy": "authored"},
    {"source_edit_mode": "mapped-span@2"}, {"planning_mode": "model"},
])
def test_nonordinary_routes_are_rejected(extra):
    options = {"provider": "grok_cli", "planning_mode": PLAN, **extra}
    with pytest.raises(ValueError):
        controls.validate_full_supervisor_grok_pruning(ON, **options)


@pytest.mark.parametrize("profile", [None, True, 1, {}, "invalid"])
def test_malformed_profile_rejected_before_prepare_files(tmp_path, profile):
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "absent", output=tmp_path / "out",
            archive=tmp_path / "absent", arm="full", provider_profile=GROK_PROFILE,
            planning_mode=PLAN, grok_pruning_profile=profile)


def test_non_grok_and_model_planning_rejected_before_prepare_files(tmp_path):
    for options in ({"planning_mode": PLAN}, {"provider_profile": GROK_PROFILE}):
        with pytest.raises(ValueError):
            benchmark.prepare(dataset=tmp_path / "absent", output=tmp_path / "out",
                archive=tmp_path / "absent", arm="full", grok_pruning_profile=ON, **options)


@pytest.mark.parametrize("profile", ["legacy", OFF, ON])
def test_doctor_only_forwards_explicit_profile_to_model_coding(profile):
    argv = doctor.implementation_argv(router=Path("/router"), model="grok-4.7", reasoning="high",
        timeout=30, semantic_repository=None, doctor=None, provider="grok_cli", grok_pruning_profile=profile)
    if profile == "legacy":
        assert "--grok-pruning-profile" not in argv
    else:
        assert argv.count("--grok-pruning-profile") == 1
        assert argv[argv.index("--grok-pruning-profile") + 1] == profile


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_provider_free_doctor_cannot_silently_ignore_explicit_pruning(route):
    with pytest.raises(ValueError, match="ordinary"):
        doctor.implementation_argv(router=Path("/router"), model="grok-4.7", reasoning="high",
            timeout=30, semantic_repository=None, provider="grok_cli", grok_pruning_profile=ON,
            doctor={"route": route, "status": "candidate_ready", "provider_calls": 0,
                "artifact": "/fixture", "sha256": "a" * 64, "task_cid": "task"})


@pytest.mark.parametrize("where", ["preparation", "config"])
@pytest.mark.parametrize("mutation", ["remove", "change"])
def test_collect_and_execute_detect_profile_drift(tmp_path, monkeypatch, where, mutation):
    output, prepared, config = frozen_trial(tmp_path)
    target = prepared if where == "preparation" else config["agents"][0]["kwargs"]
    if mutation == "remove": target.pop("grok_pruning_profile")
    else: target["grok_pruning_profile"] = OFF
    (output / "config.json").write_text(json.dumps(config))
    prepared["config_sha256"] = benchmark._hash(output / "config.json")
    (output / "preparation.json").write_text(json.dumps(prepared))
    receipt = benchmark.collect(output)
    assert receipt["grok_pruning_config_unchanged"] is False
    assert receipt["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_a, **_k: pytest.fail("changed selection executed"))
    with pytest.raises(ValueError, match="prepared Grok pruning profile changed"):
        benchmark.execute(output)


@pytest.mark.parametrize("declared", [None, OFF, ON])
def test_collect_cannot_qualify_a_declaration_without_native_preflight(tmp_path, declared):
    output, prepared, config = frozen_trial(tmp_path)
    report = {"provider_invocations": [{"purpose": "coding", "phase": "coding", "provider": "grok_cli",
        "provider_invocation_policy": {}, "native_rollout_usage": {"usage": {"total_tokens": 33}}}]}
    if declared is not None: report["grok_pruning_profile"] = declared
    retained_report(output, prepared, config, report)
    result = benchmark.collect(output)
    assert result["grok_pruning_runtime_declared_matches"] is (declared == ON)
    assert result["grok_pruning_runtime_policy_matches"] is False
    assert result["complete_single_trial_receipt"] is False
    assert result["trials"][0]["supervisor"] == report


def test_missing_coding_observation_stays_unknown(tmp_path):
    output, prepared, config = frozen_trial(tmp_path)
    retained_report(output, prepared, config, {"grok_pruning_profile": ON, "provider_invocations": []})
    result = benchmark.collect(output)
    assert result["grok_pruning_runtime_policy_matches"] is None
    assert result["complete_single_trial_receipt"] is False


@pytest.mark.parametrize("mutation", ["removed", "changed", "introduced", "planning"])
def test_harbor_frozen_selection_cannot_drift(tmp_path, mutation):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7", runtime_archive=str(tmp_path),
        provider_profile=GROK_PROFILE, planning_mode=PLAN,
        grok_pruning_profile="legacy" if mutation == "introduced" else ON)
    if mutation == "removed": del agent.grok_pruning_profile
    elif mutation == "planning": agent.planning_mode = "model"
    else: agent.grok_pruning_profile = ON if mutation == "introduced" else OFF
    with pytest.raises(ValueError): agent._checked_grok_pruning_profile()


@pytest.mark.parametrize("profile", ["legacy", OFF, ON])
def test_harbor_setup_declares_profile_only_to_worker_deployment(tmp_path, monkeypatch, profile):
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7",
        runtime_archive=str(tmp_path), provider_profile=GROK_PROFILE, planning_mode=PLAN,
        grok_pruning_profile=profile)
    deployment, boundary = AsyncMock(), AsyncMock(return_value={})
    monkeypatch.setattr(adapter, "deploy_supervisor", deployment)
    monkeypatch.setattr(adapter, "deploy_worker_boundary", boundary)
    monkeypatch.setattr(grok, "require_grok_credential_readiness", AsyncMock())
    asyncio.run(agent.setup(object()))
    assert "grok_pruning_profile" not in deployment.call_args.kwargs
    assert boundary.call_args.kwargs.get("grok_pruning_profile", "legacy") == profile
    assert ("grok_pruning_profile" in boundary.call_args.kwargs) is (profile != "legacy")


@pytest.mark.parametrize("profile", ["legacy", OFF, ON])
def test_harbor_run_forwards_profile_and_preserves_missing_policy_as_mismatch(tmp_path, monkeypatch, profile):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={}))
    calls = []
    class Environment:
        async def upload_file(self, *_): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            report = {"task_completed": False, "seconds": 1, "phases": {},
                **pruning.grok_pruning_selection(profile), "provider_invocations": [{
                    "invocation_id": "authored", "phase": "coding", "purpose": "coding", "provider": "grok_cli",
                    "provider_invocation_policy": {}, "native_rollout_usage": {"usage": {
                        "input_tokens": 2, "cached_input_tokens": 0, "output_tokens": 1, "total_tokens": 3}}}]}
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7",
        runtime_archive=str(tmp_path), provider_profile=GROK_PROFILE, planning_mode=PLAN,
        grok_pruning_profile=profile)
    context = AgentContext()
    asyncio.run(agent.run("Authored task", Environment(), context))
    assert len(calls) == 1 and context.metadata["usage"]["total_tokens"] == 3
    if profile == "legacy":
        assert "--grok-pruning-profile" not in calls[0]
        assert not any(key.startswith("grok_pruning_") for key in context.metadata)
    else:
        assert calls[0].count("--grok-pruning-profile") == 1
        assert calls[0][calls[0].index("--grok-pruning-profile") + 1] == profile
        assert context.metadata["grok_pruning_runtime_declared_matches"] is True
        assert context.metadata["grok_pruning_runtime_policy_matches"] is False


@pytest.mark.parametrize("profile", [OFF, ON])
def test_dry_preparation_freezes_profile(tmp_path, monkeypatch, profile):
    from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"): pass
    value = manifest(); value["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(value))
    monkeypatch.setattr(grok, "verify_grok_archive", lambda *_a, **_k: None)
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry run", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", provider_profile=GROK_PROFILE, planning_mode=PLAN, grok_pruning_profile=profile)
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert prepared["provider_calls"] == 0 and len(calls) == 1 and "--dry-run" in calls[0]
    assert prepared["grok_pruning_profile"] == config["agents"][0]["kwargs"]["grok_pruning_profile"] == profile
    assert prepared["comparison_controls"]["schema"] == controls.GROK_PRUNING_SCHEMA


def test_prepare_cli_selects_profile_and_later_operations_cannot_override(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path), "--output", str(tmp_path / "out"), "--provider-profile", GROK_PROFILE,
        "--planning-mode", PLAN, "--grok-pruning-profile", ON])
    benchmark.main()
    assert calls[0]["grok_pruning_profile"] == ON
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path),
            "--grok-pruning-profile", ON])
        with pytest.raises(SystemExit) as error: benchmark.main()
        assert error.value.code == 2


def test_driver_cli_forwards_selected_profile(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    calls = []
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": False, "arm": "full", "seconds": 0, "provider_invocations": []})
    monkeypatch.setattr(sys, "argv", ["driver", "--instruction", str(tmp_path / "instruction"),
        "--state", str(tmp_path / "state"), "--arm", "full", "--provider-profile", GROK_PROFILE,
        "--planning-mode", PLAN, "--grok-pruning-profile", ON])
    assert driver.main() == 1
    assert calls[0]["grok_pruning_profile"] == ON
