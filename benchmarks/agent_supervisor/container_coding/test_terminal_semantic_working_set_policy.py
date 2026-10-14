"""Frozen capsule-reference selection without providers or changed write authority."""
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
from benchmarks.agent_supervisor.container_coding import terminal_semantic_working_set_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

MODE = "capsule-references@1"
DEFAULT = policy.DEFAULT_SEMANTIC_WORKING_SET


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.SOURCE_PATHS]}


def seal(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_legacy_is_byte_compatible_and_unselected(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    legacy = benchmark.config_for(*args)
    explicit = benchmark.config_for(*args, semantic_working_set=DEFAULT)
    assert legacy == explicit and seal(legacy) == seal(explicit)
    assert "semantic_working_set" not in explicit["agents"][0]["kwargs"]
    assert policy.semantic_working_set_selection(DEFAULT) == policy.semantic_working_set_observation(DEFAULT) == {}
    policy.require_semantic_working_set_archive(None, DEFAULT)


def test_controls_freeze_selected_mode_and_reject_relabeling(tmp_path):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    base = benchmark.config_for(*args, source_view_profile="identity@1", capsule_budget="one-capsule@1")
    selected = benchmark.config_for(*args, source_view_profile="identity@1", capsule_budget="one-capsule@1",
        semantic_working_set=MODE)
    assert selected["agents"][0]["kwargs"] == {**base["agents"][0]["kwargs"], "semantic_working_set": MODE}
    record = seal(selected)
    assert record["schema"] == controls.SEMANTIC_WORKING_SET_SCHEMA and controls.validate_controls(record)
    obs = lambda config: dict(status="observed", configuration_unchanged=True, declared=seal(config))
    assert controls.compare_controls(obs(selected), obs(base))["differences"] == [
        "experimental_controls_sha256.semantic_working_set"]
    record["schema"] = controls.CAPSULE_BUDGET_SCHEMA
    assert controls.validate_controls(record) is False


@pytest.mark.parametrize("options", [
    {"arm": "no-index", "source_view_profile": "identity@1"},
    {"source_view_profile": "disabled"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"semantic_metadata_view": "common-bindings@1"},
    {"semantic_working_set": "capsule-references@2"},
    {"semantic_working_set": True},
])
def test_incompatible_selection_refuses_before_preparation_io(tmp_path, options):
    selected = dict(arm="full", source_view_profile="identity@1", semantic_working_set=MODE)
    selected.update(options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", **selected)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "stale", "wrong_size"])
def test_archive_binds_expansion_and_projection_sources(mutation):
    manifest = inventory()
    policy.require_semantic_working_set_archive(manifest, MODE)
    row = next(row for row in manifest["files"] if row["path"].endswith("worker_semantic_evidence.py"))
    if mutation == "missing": manifest["files"].remove(row)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "stale": row["sha256"] = "0" * 64
    else: row["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified runtime"):
        policy.require_semantic_working_set_archive(manifest, MODE)


def test_signed_implementation_adds_only_explicit_flag_and_keeps_candidate_route():
    options = dict(router=Path("/router"), model="pinned", reasoning="high", timeout=30,
        semantic_repository=Path("/app"), doctor=None, source_view_profile="identity@1")
    original = dispatch.implementation_argv(**options)
    assert original == dispatch.implementation_argv(**options, semantic_working_set=DEFAULT)
    selected = dispatch.implementation_argv(**options, semantic_working_set=MODE)
    assert selected == original + ["--semantic-working-set", MODE]
    candidate = dict(route="doctor_candidate", status="candidate_ready", provider_calls=0,
        artifact="/artifact", sha256="a" * 64, task_cid="task")
    assert dispatch.implementation_argv(**{**options, "doctor": candidate}, semantic_working_set=MODE) == [
        "/router", "--doctor-candidate-artifact", "/artifact", "--doctor-candidate-sha256", "a" * 64,
        "--doctor-task-cid", "task"]


def test_prepare_freezes_selection_without_provider_dispatch(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"): pass
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": benchmark._hash(archive / "runtime.tar.gz"), "codex_version": CLI_VERSION})))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive, arm="full",
        source_view_profile="identity@1", semantic_working_set=MODE)
    assert result["semantic_working_set"] == MODE
    assert result["comparison_controls"]["schema"] == controls.SEMANTIC_WORKING_SET_SCHEMA
    assert result["provider_calls"] == 0 and calls and all("--dry-run" in c for c in calls)
    observed = benchmark.collect(tmp_path / "out")
    assert observed["semantic_working_set_config_unchanged"] is True
    assert observed["semantic_working_set_runtime_declared_matches"] is None
    assert observed["semantic_working_set_token_savings_claimed"] is False


@pytest.mark.parametrize("selected", [DEFAULT, MODE])
def test_harbor_forwards_frozen_selection_and_reports_declaration(tmp_path, monkeypatch, selected):
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
                "provider_invocations": [], **policy.semantic_working_set_observation(selected)}
                if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile="identity@1", semantic_working_set=selected)
    context = AgentContext(); asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    if selected == DEFAULT:
        assert "--semantic-working-set" not in calls[0] and "semantic_working_set" not in context.metadata
    else:
        assert calls[0].count("--semantic-working-set") == 1
        assert calls[0][calls[0].index("--semantic-working-set") + 1] == selected
        assert context.metadata["semantic_working_set_runtime_declared_matches"] is True


@pytest.mark.parametrize("where", ["preparation", "config"])
def test_collect_and_execute_reject_selection_drift(tmp_path, monkeypatch, where):
    dataset, task = authored_task(tmp_path); output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        source_view_profile="identity@1", semantic_working_set=MODE)
    if where == "config": config["agents"][0]["kwargs"].pop("semantic_working_set")
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True, "source_view_profile": "identity@1",
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION}
    if where == "config": prepared["semantic_working_set"] = MODE
    (output / "preparation.json").write_text(json.dumps(prepared))
    result = benchmark.collect(output)
    assert result["semantic_working_set_config_unchanged"] is False
    assert result["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("changed selection executed"))
    with pytest.raises(ValueError, match="prepared semantic working set changed"): benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_cli_selection_is_frozen_and_container_forwards_it(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    calls = []; monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path), "--archive", str(tmp_path),
        "--output", str(tmp_path / "out"), "--source-view-profile", "identity@1", "--semantic-working-set", MODE])
    benchmark.main(); assert calls[0]["semantic_working_set"] == MODE
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--semantic-working-set", MODE])
        with pytest.raises(SystemExit) as exc: benchmark.main()
        assert exc.value.code == 2
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": False, "arm": "full", "seconds": 0, "provider_invocations": []})
    monkeypatch.setattr(sys, "argv", ["driver", "--instruction", str(tmp_path / "instruction"),
        "--state", str(tmp_path / "state"), "--arm", "full", "--source-view-profile", "identity@1",
        "--semantic-working-set", MODE])
    assert driver.main() == 1 and calls[-1]["semantic_working_set"] == MODE


@pytest.mark.parametrize("change", ["mode", "source_view", "transport", "metadata", "census"])
def test_harbor_refuses_drift_before_environment_actions(tmp_path, change):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile="identity@1", semantic_working_set=MODE)
    if change == "mode": agent.semantic_working_set = DEFAULT
    elif change == "source_view": agent.source_view_profile = "disabled"
    elif change == "transport": agent.semantic_transport_schema = "supervisor-semantic-router-input@2"
    elif change == "metadata": agent.semantic_metadata_view = "common-bindings@1"
    else: agent.metadata_census_only = True
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (env,)), (agent.run, ("Authored instruction", env, SimpleNamespace()))):
        with pytest.raises(ValueError): asyncio.run(method(*args))
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited(); env.download_file.assert_not_awaited()


@pytest.mark.parametrize("report_state,expected", [("missing", None), ("legacy", False), ("selected", True)])
def test_collect_distinguishes_unavailable_mismatched_and_matching_runtime_declarations(tmp_path, report_state, expected):
    dataset, task = authored_task(tmp_path); output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        source_view_profile="identity@1", semantic_working_set=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "source_view_profile": "identity@1", **policy.semantic_working_set_observation(MODE),
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION}
    (output / "preparation.json").write_text(json.dumps(prepared))
    job = Path(config["jobs_dir"]) / config["job_name"]; trial = job / "authored"; trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"task_name": task.name, "config": {"task": {"path": str(task)}}}))
    (job / "result.json").write_text("{}")
    if report_state != "missing":
        (trial / "agent").mkdir()
        (trial / "agent/supervisor-result.json").write_text(json.dumps(
            policy.semantic_working_set_observation(MODE) if report_state == "selected" else {}))
    result = benchmark.collect(output)
    assert result["semantic_working_set_runtime_declared_matches"] is expected
    assert result["complete_single_trial_receipt"] is (expected is True)
