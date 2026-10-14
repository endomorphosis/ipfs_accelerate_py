"""Explicit capsule bounds remain frozen through outer controls and runtime wiring."""
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
from benchmarks.agent_supervisor.container_coding import terminal_capsule_selection_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

DEFAULT = policy.DEFAULT_CAPSULE_BUDGET
MODES = tuple(value for value in policy.CAPSULE_BUDGETS if value != DEFAULT)


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.BUDGET_SOURCE_PATHS]}


def seal(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


def observed(config):
    return dict(status="observed", configuration_unchanged=True, declared=seal(config))


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_legacy_preserves_default_configuration_and_controls(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    original = benchmark.config_for(*args)
    assert benchmark.config_for(*args, capsule_budget=DEFAULT) == original
    assert "capsule_budget" not in original["agents"][0]["kwargs"]
    assert seal(original)["schema"] == controls.SCHEMA
    assert seal(benchmark.config_for(*args, capsule_budget=DEFAULT)) == seal(original)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("numerical", [None, "python-numerical-control-structure@3"])
def test_control_comparison_distinguishes_selected_budget(tmp_path, mode, numerical):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    baseline = benchmark.config_for(*args, numerical_formalization_profile=numerical)
    selected = benchmark.config_for(*args, numerical_formalization_profile=numerical, capsule_budget=mode)
    assert selected["agents"][0]["kwargs"] == {**baseline["agents"][0]["kwargs"], "capsule_budget": mode}
    assert seal(selected)["schema"] == controls.CAPSULE_BUDGET_SCHEMA
    assert controls.validate_controls(seal(selected))
    comparison = controls.compare_controls(observed(baseline), observed(selected))
    assert comparison["matches"] is False
    assert comparison["differences"] == ["experimental_controls_sha256.capsule_budget"]
    assert controls.compare_controls(observed(selected), observed(selected))["matches"] is True


def test_controls_preserve_combined_experimental_profile_hashes(tmp_path):
    selected = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        capsule_budget=MODES[0], planning_mode="declared-contract@1",
        numerical_formalization_profile="python-numerical-control-structure@3",
        source_view_profile="identity@1", source_edit_mode="mapped-span@2")
    declared = seal(selected)
    assert controls.validate_controls(declared)
    assert set(declared["experimental_controls_sha256"]) == {
        "capsule_budget", "planning_mode", "numerical_formalization_profile", "source_edit_mode"}


@pytest.mark.parametrize("mutation", ["missing", "extra", "bool", "schema", "digest"])
def test_edited_budget_declarations_never_validate(tmp_path, mutation):
    record = seal(benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        capsule_budget=MODES[0]))
    if mutation == "missing": record["experimental_controls_sha256"].pop("capsule_budget")
    elif mutation == "extra": record["experimental_controls_sha256"]["other"] = "a" * 64
    elif mutation == "bool": record["experimental_controls_sha256"]["capsule_budget"] = True
    elif mutation == "schema": record["schema"] = controls.SOURCE_EDIT_SCHEMA
    else: record["experimental_controls_sha256"]["capsule_budget"] = "f" * 64
    assert not controls.validate_controls(record)


@pytest.mark.parametrize("mode", MODES)
def test_no_index_refuses_before_file_or_provider_access(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    with pytest.raises(ValueError, match="full indexed arm"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="no-index", capsule_budget=mode)
    with pytest.raises(ValueError, match="full indexed arm"):
        driver.run(instruction=tmp_path / "missing", state=tmp_path / "state", arm="no-index", capsule_budget=mode)


@pytest.mark.parametrize("mode", MODES)
def test_grok_route_remains_explicit(tmp_path, mode):
    result = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        provider_profile=GROK_PROFILE, capsule_budget=mode)
    assert result["agents"][0]["kwargs"]["provider_profile"] == GROK_PROFILE
    assert result["agents"][0]["kwargs"]["capsule_budget"] == mode


@pytest.mark.parametrize("mode", [DEFAULT, *MODES])
def test_preparation_dry_run_seals_budget_before_execution(tmp_path, monkeypatch, mode):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"): pass
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": benchmark._hash(archive / "runtime.tar.gz"), "codex_version": CLI_VERSION})))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr=""))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", capsule_budget=mode)
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert result["provider_calls"] == 0 and len(calls) == 1 and "--dry-run" in calls[0]
    if mode == DEFAULT:
        assert not any(key.startswith("capsule_budget") for key in result)
        assert "capsule_budget" not in config["agents"][0]["kwargs"]
    else:
        assert result["capsule_budget"] == mode and result["capsule_budget_authority"] is False
        assert result["capsule_budget_token_savings_claimed"] is False
        assert result["comparison_controls"]["schema"] == controls.CAPSULE_BUDGET_SCHEMA
        assert config["agents"][0]["kwargs"]["capsule_budget"] == mode
        receipt = benchmark.collect(tmp_path / "out")
        assert receipt["capsule_budget_config_unchanged"] is True
        assert receipt["capsule_budget_runtime_declared_matches"] is None


@pytest.mark.parametrize("mode", [DEFAULT, *MODES])
@pytest.mark.parametrize("report_state", ["matching", "missing", "changed"])
def test_harbor_driver_cli_and_metadata_retain_frozen_budget(tmp_path, monkeypatch, mode, report_state):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []})))
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={"status": "captured"}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={"status": "captured"}))
    calls = []
    declared = (policy.capsule_budget_observation(mode) if report_state == "matching" else {}
        if report_state == "missing" else policy.capsule_budget_observation(
            MODES[1] if mode == MODES[0] else MODES[0]))
    class Environment:
        async def upload_file(self, source, target): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1, "phases": {},
                "provider_invocations": [{"invocation_id": "authored", "native_rollout_usage": {
                    "usage": {"input_tokens": 2, "cached_input_tokens": 0, "output_tokens": 1, "total_tokens": 3}}}],
                **declared}
                if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", capsule_budget=mode)
    context = AgentContext(); asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    assert context.metadata["usage"]["total_tokens"] == 3
    if mode == DEFAULT:
        assert "--capsule-budget" not in calls[0]
        assert not any(key.startswith("capsule_budget") for key in context.metadata)
    else:
        assert calls[0].count("--capsule-budget") == 1
        assert calls[0][calls[0].index("--capsule-budget") + 1] == mode
        assert context.metadata["capsule_budget"] == mode
        assert context.metadata["capsule_budget_runtime_declared_matches"] is (report_state == "matching")


@pytest.mark.parametrize("failure", ["archive", "removed", "changed", "introduced"])
def test_harbor_refuses_drift_before_environment_actions(tmp_path, failure):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", capsule_budget=DEFAULT if failure == "introduced" else MODES[0])
    if failure == "removed": del agent.capsule_budget
    elif failure == "changed": agent.capsule_budget = MODES[1]
    elif failure == "introduced": agent.capsule_budget = MODES[0]
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (environment,)),
            (agent.run, ("Authored instruction", environment, SimpleNamespace()))):
        with pytest.raises(ValueError, match="capsule|bounded capsules"): asyncio.run(method(*args))
    environment.upload_file.assert_not_awaited(); environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


def frozen_trial(tmp_path, mode):
    dataset, task = authored_task(tmp_path); output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", capsule_budget=mode)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        **policy.capsule_budget_observation(mode)}
    (output / "preparation.json").write_text(json.dumps(prepared))
    return output, prepared, config


@pytest.mark.parametrize("where", ["preparation", "config"])
@pytest.mark.parametrize("mutation", ["remove", "change"])
def test_collection_and_execution_refuse_budget_mutation(tmp_path, monkeypatch, where, mutation):
    output, prepared, config = frozen_trial(tmp_path, MODES[0])
    target = prepared if where == "preparation" else config["agents"][0]["kwargs"]
    if mutation == "remove": target.pop("capsule_budget")
    else: target["capsule_budget"] = MODES[1]
    (output / "config.json").write_text(json.dumps(config))
    prepared["config_sha256"] = benchmark._hash(output / "config.json")
    (output / "preparation.json").write_text(json.dumps(prepared))
    receipt = benchmark.collect(output)
    assert receipt["capsule_budget_config_unchanged"] is False
    assert receipt["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("changed budget executed"))
    with pytest.raises(ValueError, match="prepared capsule budget changed"): benchmark.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("report_mode,expected", [(None, False), ("missing", None), (DEFAULT, False),
    (MODES[0], True), (MODES[1], False)])
def test_collector_marks_only_matching_runtime_declaration(tmp_path, report_mode, expected):
    output, _, config = frozen_trial(tmp_path, MODES[0])
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored-trial"; trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"task_name": benchmark.TASK}))
    if report_mode != "missing":
        (trial / "agent").mkdir()
        report = {} if report_mode is None else policy.capsule_budget_observation(report_mode)
        (trial / "agent/supervisor-result.json").write_text(json.dumps(report))
    receipt = benchmark.collect(output)
    assert receipt["capsule_budget_runtime_declared_matches"] is expected
    assert receipt["complete_single_trial_receipt"] is False


@pytest.mark.parametrize("mode", MODES)
def test_cli_budget_can_only_be_selected_during_prepare(tmp_path, monkeypatch, mode):
    calls = []; monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path), "--output", str(tmp_path / "out"), "--capsule-budget", mode])
    benchmark.main(); assert calls[0]["capsule_budget"] == mode
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--capsule-budget", mode])
        with pytest.raises(SystemExit) as exc: benchmark.main()
        assert exc.value.code == 2


@pytest.mark.parametrize("stage", ["initial_context", "context"])
def test_driver_forwards_budget_to_preparation_and_refuses_dropped_context(tmp_path, monkeypatch, stage):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import admission_environment
    # Load cleanup imports before guarding native provider process launches.
    from benchmarks.agent_supervisor.container_coding import terminal_numerical_retrieval_observation  # noqa: F401
    state = tmp_path / "state/run"; state.mkdir(parents=True); calls = []
    for key, value in admission_environment(None).items(): monkeypatch.setenv(key, value)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda value: 0)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *a, **k: None)
    def prepare(**kwargs):
        assert kwargs["capsule_budget"] == MODES[0]
        calls.append("prepare"); return {"intent_preplanning": {}}
    monkeypatch.setattr(driver.preparation, "prepare", prepare)
    def initial(**kwargs):
        calls.append("initial_context"); assert kwargs["capsule_budget"] == MODES[0]
        return {} if stage == "initial_context" else {"capsule_budget": MODES[0]}
    monkeypatch.setattr(driver.preparation, "initial_context", initial)
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: calls.append("planning") or {"qualified": True})
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs: calls.append("context") or {"context_bundle": {}})
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: pytest.fail("a provider was invoked"))
    report = driver.run(instruction=tmp_path / "unread", state=state, arm="full",
        source384_config=tmp_path / "unread-source384", capsule_budget=MODES[0])
    assert report["error_phase"] == stage and report["error"]["type"] == "ValueError"
    assert report["provider_invocations"] == [] and report["capsule_budget"] == MODES[0]
    assert calls == (["prepare", "initial_context"] if stage == "initial_context"
        else ["prepare", "initial_context", "planning", "context"])


@pytest.mark.parametrize("mode", MODES)
def test_multitask_budget_refuses_configuration_and_declared_controls(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import multitask_profile
    profile = multitask_profile()
    with pytest.raises(ValueError, match="one independent signed task"):
        benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
            capsule_budget=mode, task_profile=profile)
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        task_profile=profile)
    config["agents"][0]["kwargs"]["capsule_budget"] = mode
    with pytest.raises(ValueError, match="one independent signed task"):
        seal(config)


@pytest.mark.parametrize("mode", MODES)
def test_multitask_prepare_refuses_before_archive_or_dry_run(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import multitask_profile
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import instruction_sha256
    dataset, task = authored_task(tmp_path)
    profile = multitask_profile()
    profile["instruction_sha256"] = instruction_sha256((task / "instruction.md").read_text())
    profile_path = tmp_path / "profile.json"; profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("dry run dispatched"))
    with pytest.raises(ValueError, match="one independent signed task"):
        benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=tmp_path / "missing",
            arm="full", capsule_budget=mode, task_profile=profile_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", MODES)
def test_multitask_harbor_constructor_refuses_before_archive_reads(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import multitask_profile
    with pytest.raises(ValueError, match="one independent signed task"):
        FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
            runtime_archive=str(tmp_path / "missing"), arm="full", capsule_budget=mode,
            task_profile=multitask_profile())


@pytest.mark.parametrize("mode", MODES)
def test_multitask_harbor_late_change_refuses_before_environment_actions(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import multitask_profile
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", capsule_budget=mode)
    agent.task_profile = multitask_profile()
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (environment,)),
            (agent.run, ("Authored instruction", environment, SimpleNamespace()))):
        with pytest.raises(ValueError, match="one independent signed task"):
            asyncio.run(method(*args))
    environment.upload_file.assert_not_awaited(); environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


@pytest.mark.parametrize("mode", MODES)
def test_multitask_driver_refuses_before_container_initialization(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import admission_environment
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import multitask_profile
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Implement public_source and write result.py.\n")
    profile_path = tmp_path / "profile.json"; profile_path.write_text(json.dumps(multitask_profile()))
    for key, value in admission_environment(None).items(): monkeypatch.setenv(key, value)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: pytest.fail("index setup called"))
    with pytest.raises(ValueError, match="one independent signed task"):
        driver.run(instruction=instruction, state=tmp_path / "state", arm="full", task_profile=profile_path,
            capsule_budget=mode)
    assert not (tmp_path / "state").exists()


def test_multitask_legacy_configuration_still_uses_historical_schema(tmp_path):
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import multitask_profile
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        task_profile=multitask_profile(), capsule_budget=DEFAULT)
    assert "capsule_budget" not in config["agents"][0]["kwargs"]
    assert seal(config)["schema"] == controls.SCHEMA
