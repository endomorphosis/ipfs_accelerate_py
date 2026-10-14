"""Declared planning propagation without model calls, credentials or Docker."""
import asyncio
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
from benchmarks.agent_supervisor.container_coding import terminal_planning_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

MODE = policy.DECLARED_PLANNING_MODE
DEFAULT = policy.DEFAULT_PLANNING_MODE


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.SOURCE_PATHS]}


def seal(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


def observed(config):
    return dict(status="observed", configuration_unchanged=True, declared=seal(config))


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("numerical", [None, "python-numerical-control-structure@3"])
def test_default_preserves_configuration_and_controls(tmp_path, arm, numerical):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    legacy = benchmark.config_for(*args, numerical_formalization_profile=numerical)
    explicit = benchmark.config_for(*args, numerical_formalization_profile=numerical, planning_mode=DEFAULT)
    assert explicit == legacy and seal(explicit) == seal(legacy)
    assert "planning_mode" not in legacy["agents"][0]["kwargs"]
    assert seal(legacy)["schema"] == (controls.SCHEMA if numerical is None else controls.NUMERICAL_SCHEMA)
    assert policy.planning_selection(DEFAULT) == policy.planning_observation(DEFAULT) == {}
    policy.require_planning_archive(None, DEFAULT)


@pytest.mark.parametrize("numerical", [None, "python-numerical-control-structure@3"])
def test_comparison_distinguishes_declared_planning_and_matches_both_arms(tmp_path, numerical):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive")
    baseline = benchmark.config_for(*args, "full", numerical_formalization_profile=numerical)
    full = benchmark.config_for(*args, "full", numerical_formalization_profile=numerical, planning_mode=MODE)
    no_index = benchmark.config_for(*args, "no-index", numerical_formalization_profile=numerical, planning_mode=MODE)
    assert full["agents"][0]["kwargs"] == {**baseline["agents"][0]["kwargs"], "planning_mode": MODE}
    for config in (full, no_index):
        assert seal(config)["schema"] == controls.PLANNING_SCHEMA and controls.validate_controls(seal(config))
    assert controls.compare_controls(observed(full), observed(no_index))["matches"] is True
    result = controls.compare_controls(observed(full), observed(baseline))
    assert result["matches"] is False
    assert result["differences"] == ["experimental_controls_sha256.planning_mode"]


@pytest.mark.parametrize("mutation", ["missing", "unknown", "bool", "digest", "v2", "v1"])
def test_controls_reject_edited_planning_declarations(tmp_path, mutation):
    record = seal(benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full", planning_mode=MODE))
    if mutation == "missing": del record["experimental_controls_sha256"]["planning_mode"]
    elif mutation == "unknown": record["experimental_controls_sha256"]["other"] = "a" * 64
    elif mutation == "bool": record["experimental_controls_sha256"]["planning_mode"] = True
    elif mutation == "digest": record["experimental_controls_sha256"]["planning_mode"] = "0" * 64
    elif mutation == "v2": record["schema"] = controls.NUMERICAL_SCHEMA
    else: record["schema"] = controls.SCHEMA
    assert controls.validate_controls(record) is False


@pytest.mark.parametrize("value", [None, True, 1, {}, [], "", "declared", "declared-contract@2", "model\x00"])
def test_bad_mode_refuses_before_file_access(tmp_path, value):
    with pytest.raises(ValueError, match="supported planning mode"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", planning_mode=value)


def test_explicit_intent_and_multitask_cannot_be_silently_replaced(tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import MULTITASK_SCHEMA
    for contract in ({}, {"schema": "intent-plan-requirement-contract@2"}):
        with pytest.raises(ValueError, match="explicit Intent"):
            benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
                planning_mode=MODE, intent_requirement_contract=contract)
    with pytest.raises(ValueError, match="explicit Intent"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out", archive=tmp_path / "missing",
            arm="full", planning_mode=MODE, intent_requirement_contract=tmp_path / "missing-contract")
    with pytest.raises(ValueError, match="single task"):
        benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
            planning_mode=MODE, task_profile={"schema": MULTITASK_SCHEMA})


@pytest.mark.parametrize("path", policy.SOURCE_PATHS)
def test_runtime_requires_every_current_planning_producer(path):
    manifest = inventory()
    next(row for row in manifest["files"] if row["path"] == "source/" + path)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_planning_archive(manifest, MODE)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool_bytes", "wrong_bytes"])
def test_runtime_inventory_cannot_hide_edited_producer(mutation):
    manifest = inventory(); row = manifest["files"][0]
    if mutation == "missing": manifest["files"].pop(0)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "bool_bytes": row["bytes"] = True
    else: row["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_planning_archive(manifest, MODE)


@pytest.mark.parametrize("manifest", [None, True, {}, {"files": {}}, {"files": "other"}])
def test_malformed_runtime_inventory_refuses(manifest):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_planning_archive(manifest, MODE)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_prepare_freezes_mode_and_declaration_without_dispatch(tmp_path, monkeypatch, arm):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"): pass
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": benchmark._hash(archive / "runtime.tar.gz"), "codex_version": CLI_VERSION})))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive, arm=arm, planning_mode=MODE)
    assert result["planning_mode"] == MODE and result["planning_strategy"] == "declared_contract"
    assert result["comparison_controls"]["schema"] == controls.PLANNING_SCHEMA
    assert result["provider_calls"] == 0 and calls and all("--dry-run" in c for c in calls)
    assert result["planning_mode_authority"] is False and result["planning_mode_token_savings_claimed"] is False
    assert benchmark.collect(tmp_path / "out")["planning_mode_config_unchanged"] is True


@pytest.mark.parametrize("selected", [DEFAULT, MODE])
@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_harbor_forwards_frozen_selection(tmp_path, monkeypatch, selected, arm):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []})))
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={"status": "captured"}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={"status": "captured"}))
    calls = []
    class Environment:
        async def upload_file(self, source, target): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1, "phases": {},
                "provider_invocations": []} if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm=arm, planning_mode=selected)
    context = AgentContext(); asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    if selected == DEFAULT:
        assert "--planning-mode" not in calls[0] and "planning_mode" not in context.metadata
        assert context.metadata["planning_strategy"] == "direct"
    else:
        assert calls[0].count("--planning-mode") == 1
        assert calls[0][calls[0].index("--planning-mode") + 1] == MODE
        assert context.metadata["planning_mode"] == MODE
        assert context.metadata["planning_strategy"] == "declared_contract"


@pytest.mark.parametrize("failure", ["archive", "mode_drift", "contract_drift", "multitask_drift"])
def test_adapter_refuses_before_environment_effects(tmp_path, failure):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import MULTITASK_SCHEMA
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", planning_mode=MODE)
    if failure == "mode_drift": agent.planning_mode = DEFAULT
    elif failure == "contract_drift": agent.intent_requirement_contract = {}
    elif failure == "multitask_drift": agent.task_profile = {"schema": MULTITASK_SCHEMA}
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (env,)), (agent.run, ("Authored instruction", env, SimpleNamespace()))):
        with pytest.raises(ValueError): asyncio.run(method(*args))
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited(); env.download_file.assert_not_awaited()


@pytest.mark.parametrize("where", ["preparation", "config"])
def test_collect_and_execute_detect_selection_drift(tmp_path, monkeypatch, where):
    dataset, task = authored_task(tmp_path); output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", planning_mode=MODE)
    if where == "config": config["agents"][0]["kwargs"].pop("planning_mode")
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION}
    if where == "config": prepared["planning_mode"] = MODE
    (output / "preparation.json").write_text(json.dumps(prepared))
    result = benchmark.collect(output)
    assert result["planning_mode_config_unchanged"] is False
    assert result["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("changed selection executed"))
    with pytest.raises(ValueError, match="prepared planning mode changed"): benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_cli_selection_is_frozen_by_prepare(tmp_path, monkeypatch):
    calls = []; monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path), "--archive", str(tmp_path),
        "--output", str(tmp_path / "out"), "--planning-mode", MODE])
    benchmark.main(); assert calls[0]["planning_mode"] == MODE
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--planning-mode", MODE])
        with pytest.raises(SystemExit) as exc: benchmark.main()
        assert exc.value.code == 2


@pytest.mark.parametrize("where", ["constructor", "run_drift"])
def test_live_metadata_census_cannot_invent_planner_delivery(tmp_path, where):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    options = dict(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", planning_mode=MODE)
    if where == "constructor":
        with pytest.raises(ValueError, match="live metadata census requires model planning"):
            FullSupervisorAgent(**options, metadata_census_only=True)
    else:
        (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
            "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
        agent = FullSupervisorAgent(**options)
        agent.metadata_census_only = True
        with pytest.raises(ValueError, match="live metadata census requires model planning"):
            agent._checked_planning_mode()
