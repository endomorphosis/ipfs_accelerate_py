"""Provider-free Harbor census selection, custody, and measurement-only reporting."""
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

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_semantic_working_set_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE, resolve_provider_profile
from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import SOURCE_PATHS as PLANNING_PATHS
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

OPTIONS = dict(planning_mode="declared-contract@1", semantic_working_set="capsule-references@1",
               source_view_profile="identity@1", working_set_census_only=True)


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import SOURCE_PATHS as GROK_PATHS
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in sorted(set(
            policy.SOURCE_PATHS + PLANNING_PATHS + GROK_PATHS))]}


def manifest(tmp_path):
    value = _supervisor_manifest({**inventory(), "archive_sha256": "a" * 64,
        "codex_version": benchmark.CLI_VERSION, "learned_requirements": []})
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    return value


def seal(config, task_hashes=None):
    selected = resolve_provider_profile(config["agents"][0]["kwargs"].get("provider_profile"))
    return controls.build_controls(config, task_input_sha256=task_hashes or {"instruction.md": "a" * 64},
        task=benchmark.TASK, **{key: selected[key] for key in ("model", "reasoning_effort", "cli_version")})


@pytest.mark.parametrize("provider,turns", [(None, 128), (GROK_PROFILE, 8)])
def test_census_config_is_frozen_separately_and_verification_disabled(tmp_path, provider, turns):
    from harbor.models.job.config import JobConfig
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        **OPTIONS, provider_profile=provider, grok_coding_max_turns=turns)
    JobConfig.model_validate(config, extra="forbid")
    assert config["verifier"]["disable"] is True
    assert config["agents"][0]["kwargs"]["working_set_census_only"] is True
    record = seal(config)
    assert record["schema"] == controls.WORKING_SET_CENSUS_SCHEMA
    assert controls.validate_controls(record)
    previous = deepcopy(record); previous["schema"] = controls.GROK_CODING_SCHEMA
    previous["sha256"] = controls._digest({key: value for key, value in previous.items() if key != "sha256"})
    assert not controls.validate_controls(previous)
    changed = deepcopy(config); changed["verifier"]["disable"] = False
    with pytest.raises(ValueError, match="disabled official verification"):
        seal(changed)


def test_unselected_mode_keeps_existing_serialized_configuration(tmp_path):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    assert benchmark.config_for(*args) == benchmark.config_for(*args, working_set_census_only=False)
    assert policy.working_set_census_selection(False) == policy.working_set_census_observation(False) == {}


@pytest.mark.parametrize("changes", [
    {"working_set_census_only": 1}, {"working_set_census_only": "true"},
    {"arm": "no-index"}, {"planning_mode": "model"},
    {"semantic_working_set": "legacy"}, {"source_view_profile": "disabled"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"semantic_metadata_view": "common-bindings@1"},
    {"source_edit_mode": "mapped-edits@1"}, {"candidate_rounds": 2},
])
def test_incompatible_census_fails_before_preparation_io(tmp_path, changes):
    selected = dict(arm="full", **OPTIONS); selected.update(changes)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing-archive", **selected)
    assert not (tmp_path / "out").exists()


def test_census_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        policy.validate_working_set_census(True, arm="full", planning_mode="declared-contract@1",
            semantic_working_set="capsule-references@1", source_view_profile="identity@1",
            metadata_census_only=True)


def prepared(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    data = manifest(archive)
    data["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(data))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", **OPTIONS)
    assert calls and all("--dry-run" in call for call in calls)
    return tmp_path / "out", task, result


def test_prepare_and_collect_never_promote_measurement_to_task_receipt(tmp_path, monkeypatch):
    output, task, prep = prepared(tmp_path, monkeypatch)
    assert prep["provider_calls"] == 0 and prep["measurement_only"] is True
    config = json.loads((output / "config.json").read_text())
    assert config["verifier"]["disable"] is True
    empty = benchmark.collect(output)
    assert empty["complete_single_trial_receipt"] is False
    assert empty["working_set_census_receipt_complete"] is False
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored"; (trial / "agent").mkdir(parents=True)
    native = {"task_name": task.name, "config": {"task": {"path": str(task)}}}
    (trial / "result.json").write_text(json.dumps(native)); (job / "result.json").write_text("{}")
    report = {**policy.working_set_census_observation(True),
        **policy.semantic_working_set_observation("capsule-references@1"),
        "task_completed": False, "working_set_census_succeeded": True, "provider_invocations": [],
        "coding_dispatch_possible": False, "official_verifier_executed": False}
    report_path = trial / "agent/supervisor-result.json"; report_path.write_text(json.dumps(report))
    value = benchmark.collect(output)
    assert value["working_set_census_receipt_complete"] is True
    assert value["complete_single_trial_receipt"] is False and value["task_reward_claimed"] is False
    for field, changed in (("working_set_census_succeeded", False), ("task_completed", True),
                           ("provider_invocations", [{"invocation_id": "unexpected"}])):
        report_path.write_text(json.dumps({**report, field: changed}))
        assert benchmark.collect(output)["working_set_census_receipt_complete"] is False
    report_path.write_text(json.dumps(report))
    (trial / "result.json").write_text(json.dumps({**native, "verifier_result": {"rewards": {"reward": 1}}}))
    assert benchmark.collect(output)["working_set_census_receipt_complete"] is False


@pytest.mark.parametrize("change", ["config_flag", "prepared_flag", "verifier"])
def test_frozen_census_drift_refuses_before_collect_or_execution(tmp_path, monkeypatch, change):
    output, _, prep = prepared(tmp_path, monkeypatch)
    config = json.loads((output / "config.json").read_text())
    if change == "config_flag": config["agents"][0]["kwargs"]["working_set_census_only"] = False
    elif change == "prepared_flag": prep["working_set_census_only"] = False
    else: config["verifier"]["disable"] = False
    (output / "config.json").write_text(json.dumps(config))
    (output / "preparation.json").write_text(json.dumps(prep))
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("drift executed"))
    for call in (benchmark.collect, benchmark.execute):
        with pytest.raises(ValueError, match="working set census"):
            call(output)
    assert not (output / "invocation.json").exists()


def test_harbor_forwards_census_and_retains_measurement_status(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    manifest(tmp_path)
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={}))
    calls = []
    class Environment:
        async def upload_file(self, source, target): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1, "phases": {},
                "provider_invocations": [], "working_set_census_succeeded": True,
                "coding_dispatch_possible": False, "official_verifier_executed": False,
                **policy.semantic_working_set_observation("capsule-references@1"),
                **policy.working_set_census_observation(True)} if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", **OPTIONS)
    context = AgentContext(); asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1 and calls[0].count("--working-set-census-only") == 1
    assert context.metadata["working_set_census_runtime_matches"] is True
    assert context.metadata["task_completed"] is False


@pytest.mark.parametrize("selected,mutated", [(True, False), (False, True), (True, 1)])
def test_harbor_refuses_census_selection_drift_before_environment_actions(tmp_path, selected, mutated):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    manifest(tmp_path)
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", **{**OPTIONS, "working_set_census_only": selected})
    agent.working_set_census_only = mutated
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (env,)), (agent.run, ("Authored", env, SimpleNamespace()))):
        with pytest.raises(ValueError): asyncio.run(method(*args))
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited(); env.download_file.assert_not_awaited()


def test_census_setup_qualifies_namespace_without_source384_or_provider_catalog(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
    data = manifest(tmp_path); data["grok_cli_assets"] = grok.grok_binding()
    data["files"].append({"path": grok.GROK_PATH, "bytes": grok.GROK_BYTES,
        "sha256": grok.GROK_SHA256, "mode": 0o755})
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    monkeypatch.setattr(adapter, "deploy_supervisor", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "deploy_worker_boundary", AsyncMock(return_value={"authored": "qualification"}))
    forbidden = AsyncMock(side_effect=AssertionError("provider catalog called"))
    monkeypatch.setattr(grok, "require_grok_credential_readiness", forbidden)
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7",
        runtime_archive=str(tmp_path), arm="full", provider_profile=GROK_PROFILE,
        grok_coding_max_turns=8, **OPTIONS)
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))
    asyncio.run(agent.setup(env))
    forbidden.assert_not_awaited()
    assert env.upload_file.await_count == 1
    assert env.upload_file.call_args.args[1].endswith("census-worker-qualification.json")
    assert "chmod 0444" in env.exec.call_args.kwargs["command"]


def test_cli_selection_is_preparation_only(tmp_path, monkeypatch):
    calls = []; monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path), "--output", str(tmp_path / "out"), "--working-set-census-only"])
    benchmark.main(); assert calls[0]["working_set_census_only"] is True
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path),
            "--working-set-census-only"])
        with pytest.raises(SystemExit) as exc: benchmark.main()
        assert exc.value.code == 2
