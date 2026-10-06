"""Opt-in context transport is frozen before execution; no provider or Docker calls."""
import asyncio
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_semantic_transport_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls, observe_controls
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION


from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest

LEGACY = policy.DEFAULT_SEMANTIC_TRANSPORT_SCHEMA
COMPACT = policy.CONTROLLER_DICTIONARY_TRANSPORT_SCHEMA


def _current_inventory():
    checkout = Path(policy.__file__).resolve().parents[3]
    paths = [path for path in Path(policy.__file__).parent.glob("*.py") if not path.name.startswith("test_")]
    paths += [checkout / "ipfs_accelerate_py/agent_supervisor/runtime" / name for name in
        ("semantic_router_translation.py", "router_implementation_runner.py")]
    return {"files": [{"path": "source/" + path.relative_to(checkout).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}


@pytest.mark.parametrize("change", [None, "missing", "modified", "duplicate"])
def test_opt_in_archive_requires_every_current_transport_source(change):
    manifest = _current_inventory()
    if change == "missing":
        manifest["files"] = [row for row in manifest["files"] if not row["path"].endswith("router_implementation_runner.py")]
    elif change == "modified":
        next(row for row in manifest["files"] if row["path"].endswith("terminal_doctor_dispatch.py"))["sha256"] = "0" * 64
    elif change == "duplicate":
        manifest["files"].append(next(row for row in manifest["files"] if row["path"].endswith("terminal_container_supervisor.py")))
    if change is None:
        policy.require_semantic_transport_archive(manifest, COMPACT)
    else:
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            policy.require_semantic_transport_archive(manifest, COMPACT)


def test_default_configuration_stays_legacy_and_opt_in_is_sealed(tmp_path):
    args = (tmp_path, tmp_path / "output", tmp_path / "archive", "full")
    original = benchmark.config_for(*args)
    explicit_legacy = benchmark.config_for(*args, semantic_transport_schema=LEGACY)
    compact = benchmark.config_for(*args, semantic_transport_schema=COMPACT)
    assert original == explicit_legacy
    assert "semantic_transport_schema" not in original["agents"][0]["kwargs"]
    assert compact["agents"][0]["kwargs"]["semantic_transport_schema"] == COMPACT
    assert "intent_requirement_contract" not in compact["agents"][0]["kwargs"]
    def controls(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
            task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)
    old, new = controls(original), controls(compact)
    assert old["configuration_sha256"] != new["configuration_sha256"]
    assert old["job_controls_sha256"] == new["job_controls_sha256"]
    assert old["agent_controls_sha256"] == new["agent_controls_sha256"]
    compact["agents"][0]["kwargs"]["semantic_transport_schema"] = LEGACY
    assert observe_controls({"comparison_controls": new}, compact,
        current_task_hashes={"instruction.md": "a" * 64})["configuration_unchanged"] is False


@pytest.mark.parametrize("value", [None, True, {}, "supervisor-semantic-router-input@3", "@2"])
def test_invalid_selection_is_rejected_at_configuration_boundary(tmp_path, value):
    with pytest.raises(ValueError, match="supported semantic transport"):
        benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
            semantic_transport_schema=value)


def test_no_index_opt_in_is_rejected_before_preparation_reads_files(tmp_path):
    with pytest.raises(ValueError, match="full indexed arm"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing-archive", arm="no-index", semantic_transport_schema=COMPACT)


def test_old_archive_is_rejected_before_preparation_dry_run(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / benchmark.TASK).mkdir(parents=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    raw = b"authored immutable archive"
    (archive / "runtime.tar.gz").write_bytes(raw)
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": hashlib.sha256(raw).hexdigest(),
        "codex_version": CLI_VERSION, "files": []})))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full",
            semantic_transport_schema=COMPACT)
    assert calls == []
    assert not output.exists()
    policy.require_semantic_transport_archive({}, LEGACY)


def test_adapter_rejects_old_archive_before_upload_or_execution(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", semantic_transport_schema=COMPACT)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        asyncio.run(agent.run("Public instruction", environment, SimpleNamespace()))
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


def test_adapter_forwards_explicit_transport_and_keeps_direct_planning_identity(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    manifest = {**_current_inventory(), "archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "learned_requirements": []}
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest(manifest)))
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={"status": "captured"}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={"status": "captured"}))
    calls = []
    class Environment:
        async def upload_file(self, source, target):
            pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1,
                "phases": {}, "provider_invocations": []} if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", semantic_transport_schema=COMPACT)
    context = AgentContext()
    asyncio.run(agent.run("Public instruction", Environment(), context))
    assert len(calls) == 1 and calls[0].count("--semantic-transport-schema") == 1
    assert calls[0][calls[0].index("--semantic-transport-schema") + 1] == COMPACT
    assert "--intent-requirement-contract" not in calls[0]
    assert context.metadata["semantic_transport_schema"] == COMPACT
    assert context.metadata["planning_strategy"] == "direct"


def test_preparation_cli_forwards_opt_in_without_selecting_an_intent_contract(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "output"),
        "--semantic-transport-schema", COMPACT])
    benchmark.main()
    assert len(calls) == 1
    assert calls[0]["semantic_transport_schema"] == COMPACT
    assert calls[0]["intent_requirement_contract"] is None


def test_collection_retains_frozen_transport_and_detects_preparation_drift(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Public instruction")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "environment/Dockerfile").write_text("FROM scratch\n")
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("exit 0\n")
    output = tmp_path / "output"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        semantic_transport_schema=COMPACT)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config))
    preparation = {"arm": "full", "dataset": str(dataset), "config_sha256": benchmark._hash(config_path),
        "task_input_sha256": benchmark._task_hashes(task), "model": benchmark.MODEL,
        "reasoning_effort": "high", "cli_version": CLI_VERSION, "semantic_transport_schema": COMPACT}
    preparation_path = output / "preparation.json"
    preparation_path.write_text(json.dumps(preparation))
    result = benchmark.collect(output)
    assert result["semantic_transport_schema"] == COMPACT
    assert result["semantic_transport_config_unchanged"] is True
    preparation["semantic_transport_schema"] = LEGACY
    preparation_path.write_text(json.dumps(preparation))
    drifted = benchmark.collect(output)
    assert drifted["semantic_transport_config_unchanged"] is False
    assert drifted["complete_single_trial_receipt"] is False


@pytest.mark.parametrize("repository,doctor", [
    (None, None), (Path("/app"), {"route": "doctor_candidate"}),
    (Path("/app"), {"route": "doctor_contract_candidate"}),
])
def test_route_builder_refuses_inapplicable_opt_in(repository, doctor):
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    with pytest.raises(ValueError, match="semantic model-router coding route"):
        implementation_argv(router=Path("/router"), model="pinned", reasoning="high", timeout=30,
            semantic_repository=repository, doctor=doctor, semantic_transport_schema=COMPACT)


def test_route_builder_forwards_only_explicit_compact_coding_selection():
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    kwargs = dict(router=Path("/router"), model="pinned", reasoning="high", timeout=30,
        semantic_repository=Path("/app"), doctor={"route": "model_router", "status": "residual"})
    original = implementation_argv(**kwargs)
    assert "--semantic-transport-schema" not in original
    assert implementation_argv(**kwargs, semantic_transport_schema=LEGACY) == original
    compact = implementation_argv(**kwargs, semantic_transport_schema=COMPACT)
    assert compact.count("--semantic-transport-schema") == 1
    assert compact[compact.index("--semantic-transport-schema") + 1] == COMPACT
