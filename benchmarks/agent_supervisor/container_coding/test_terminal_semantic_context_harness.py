"""Frozen capsule ablation controls; authored fixtures, no provider or Docker calls."""
import asyncio
import hashlib
import json
import tarfile
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_semantic_context_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

MODE = "indexed-compact@1"
DEFAULT = "legacy-bounded@1"


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.SOURCE_PATHS]}


def write_archive(root, *, current=True):
    root.mkdir(parents=True, exist_ok=True)
    # Numerical deployment inspects the archive even when no assets are selected.
    # A real empty gzip/tar models that absence without executable fixture content.
    with tarfile.open(root / "runtime.tar.gz", "w:gz"):
        pass
    raw = (root / "runtime.tar.gz").read_bytes()
    manifest = _supervisor_manifest({**(inventory() if current else {"files": []}),
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "codex_version": CLI_VERSION,
        "learned_requirements": []})
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_preserves_historical_config_shape(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    assert benchmark.config_for(*args) == benchmark.config_for(*args, semantic_context_profile=DEFAULT)
    assert "semantic_context_profile" not in benchmark.config_for(*args)["agents"][0]["kwargs"]


@pytest.mark.parametrize("baseline_profile", [DEFAULT, "indexed-default@1"])
def test_matched_config_changes_only_capsule_profile(tmp_path, baseline_profile):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    baseline = benchmark.config_for(*args, semantic_context_profile=baseline_profile)
    indexed = benchmark.config_for(*args, semantic_context_profile=MODE)
    assert indexed["agents"][0]["kwargs"].pop("semantic_context_profile") == MODE
    baseline["agents"][0]["kwargs"].pop("semantic_context_profile", None)
    assert indexed == baseline


@pytest.mark.parametrize("value", [None, True, {}, "full", "indexed-compact@2"])
def test_invalid_profile_rejected_before_task_or_archive_reads(tmp_path, value):
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", semantic_context_profile=value)


def test_compact_no_index_rejected_before_reads(tmp_path):
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="no-index", semantic_context_profile=MODE)


@pytest.mark.parametrize("value", [None, True, "unknown@1"])
def test_declared_comparison_controls_reject_invalid_profile(tmp_path, value):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    config["agents"][0]["kwargs"]["semantic_context_profile"] = value
    with pytest.raises(ValueError, match="supported semantic context profile"):
        benchmark.benchmark_controls.build_controls(config,
            task_input_sha256={"instruction.md": "a" * 64}, task=benchmark.TASK,
            model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


def test_prepare_records_profile_and_preserves_default(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"
    write_archive(archive)
    calls = []
    def dry_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry_run)
    selected = benchmark.prepare(dataset=dataset, output=tmp_path / "selected", archive=archive,
        arm="full", semantic_context_profile=MODE)
    config = json.loads((tmp_path / "selected/config.json").read_text())
    assert selected["semantic_context_profile"] == MODE
    assert config["agents"][0]["kwargs"]["semantic_context_profile"] == MODE
    assert selected["provider_calls"] == 0
    default = benchmark.prepare(dataset=dataset, output=tmp_path / "legacy", archive=archive, arm="full")
    assert "semantic_context_profile" not in default
    assert len(calls) == 2 and all("--dry-run" in argv for argv in calls)


def test_profile_drift_blocks_execution_and_receipt_qualification(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", semantic_context_profile=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "semantic_context_profile": MODE}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["semantic_context_config_unchanged"] is True
    prepared["semantic_context_profile"] = DEFAULT
    (output / "preparation.json").write_text(json.dumps(prepared))
    result = benchmark.collect(output)
    assert result["semantic_context_config_unchanged"] is False
    assert result["complete_single_trial_receipt"] is False
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="prepared semantic context profile selection changed"):
        benchmark.execute(output)
    assert calls == []
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("reported", [None, DEFAULT, "indexed-default@1", MODE])
def test_collect_requires_observed_runtime_profile(tmp_path, monkeypatch, reported):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", semantic_context_profile=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "semantic_context_profile": MODE}
    (output / "preparation.json").write_text(json.dumps(prepared))
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored-trial"
    (trial / "agent").mkdir(parents=True)
    (job / "result.json").write_text("{}")
    (trial / "result.json").write_text(json.dumps({"task_name": benchmark.TASK}))
    report = {} if reported is None else {"semantic_context_profile": reported}
    (trial / "agent/supervisor-result.json").write_text(json.dumps(report))
    monkeypatch.setattr(benchmark, "_trial_task_matches", lambda *args: True)
    result = benchmark.collect(output)
    assert result["trials"][0]["semantic_context_profile_matches"] is (reported == MODE)
    assert result["complete_single_trial_receipt"] is (reported == MODE)


def test_adapter_refuses_unbound_runtime_before_setup_or_run_io(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    write_archive(tmp_path, current=False)
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", semantic_context_profile=MODE)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for call in (agent.setup(environment), agent.run("Authored instruction", environment, SimpleNamespace())):
        with pytest.raises(ValueError):
            asyncio.run(call)
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


@pytest.mark.parametrize("mode", [DEFAULT, "indexed-default@1", MODE])
def test_adapter_forwards_only_explicit_compact_selection(tmp_path, monkeypatch, mode):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    write_archive(tmp_path)
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
        runtime_archive=str(tmp_path), arm="full", semantic_context_profile=mode)
    context = AgentContext()
    asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    assert "--intent-requirement-contract" not in calls[0]
    assert context.metadata["planning_strategy"] == "direct"
    if mode == DEFAULT:
        assert "--semantic-context-profile" not in calls[0]
        assert "semantic_context_profile" not in context.metadata
    else:
        assert calls[0].count("--semantic-context-profile") == 1
        assert calls[0][calls[0].index("--semantic-context-profile") + 1] == mode
        assert context.metadata["semantic_context_profile"] == mode


def test_cli_selection_is_frozen_by_preparation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--semantic-context-profile", MODE])
    benchmark.main()
    assert calls[0]["semantic_context_profile"] == MODE
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path),
            "--semantic-context-profile", MODE])
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 2
