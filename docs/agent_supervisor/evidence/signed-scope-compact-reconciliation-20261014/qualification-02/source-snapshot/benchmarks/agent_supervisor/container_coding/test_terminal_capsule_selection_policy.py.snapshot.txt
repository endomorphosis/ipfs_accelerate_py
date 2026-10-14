"""Frozen ranking controls without providers, credentials, models or Docker."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_capsule_selection_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original

DEFAULT = policy.DEFAULT_CAPSULE_SELECTION
MODE = policy.RETRIEVAL_CAPSULE_SELECTION


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.SOURCE_PATHS]}


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_configuration_and_observation_shapes_are_unchanged(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    baseline = benchmark.config_for(*args)
    assert benchmark.config_for(*args, capsule_selection=DEFAULT) == baseline
    assert "capsule_selection" not in baseline["agents"][0]["kwargs"]
    assert policy.capsule_selection_options(DEFAULT) == policy.capsule_selection_observation(DEFAULT) == {}
    policy.require_capsule_selection_archive(None, DEFAULT)


@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
def test_ranked_configuration_and_control_digest_bind_selection(tmp_path, transport):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    baseline = benchmark.config_for(*args, semantic_transport_schema=transport)
    ranked = benchmark.config_for(*args, semantic_transport_schema=transport, capsule_selection=MODE)
    assert ranked["agents"][0]["kwargs"] == {**baseline["agents"][0]["kwargs"], "capsule_selection": MODE}
    def controls(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64}, task=benchmark.TASK,
            model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)
    assert controls(baseline)["configuration_sha256"] != controls(ranked)["configuration_sha256"]
    assert policy.capsule_selection_observation(MODE) == {"capsule_selection": MODE,
        "capsule_selection_token_savings_claimed": False, "capsule_selection_authority": False}


@pytest.mark.parametrize("value", [None, True, {}, "ranked", "verified-retrieval-symbol-rank@2"])
def test_invalid_selection_refuses_before_input_reads(tmp_path, value):
    with pytest.raises(ValueError, match="supported capsule selection"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", capsule_selection=value)
    with pytest.raises(ValueError, match="supported capsule selection"):
        preparation.initial_context(state=tmp_path / "missing", capsule_selection=value)


def test_no_index_refuses_before_file_reads_or_provider_calls(tmp_path):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    with pytest.raises(ValueError, match="full indexed arm"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="no-index", capsule_selection=MODE)
    with pytest.raises(ValueError, match="full indexed arm"):
        driver.run(instruction=tmp_path / "missing", state=tmp_path / "state",
            arm="no-index", capsule_selection=MODE)


def test_grok_route_keeps_explicit_provider_profile(tmp_path):
    result = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        provider_profile=GROK_PROFILE, capsule_selection=MODE)
    assert result["agents"][0]["kwargs"]["provider_profile"] == GROK_PROFILE
    assert result["agents"][0]["kwargs"]["capsule_selection"] == MODE


@pytest.mark.parametrize("path", policy.SOURCE_PATHS)
def test_archive_requires_every_current_ranking_producer(path):
    manifest = inventory()
    next(row for row in manifest["files"] if row["path"] == "source/" + path)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_capsule_selection_archive(manifest, MODE)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "boolean-bytes", "wrong-bytes"])
def test_archive_cannot_hide_missing_duplicate_or_malformed_ranking_source(mutation):
    manifest = inventory()
    row = manifest["files"][0]
    if mutation == "missing": manifest["files"].pop(0)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "boolean-bytes": row["bytes"] = True
    else: row["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_capsule_selection_archive(manifest, MODE)


def test_current_archive_inventory_is_accepted():
    policy.require_capsule_selection_archive(inventory(), MODE)


@pytest.mark.parametrize("mode", [DEFAULT, MODE])
def test_initial_preparation_forwards_only_explicit_ranking(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context
    from benchmarks.agent_supervisor.container_coding import terminal_program_population
    calls = []
    prepared = {"authored": True}
    monkeypatch.setattr(preparation, "_load_prepared", lambda state: prepared)
    monkeypatch.setattr(preparation, "_require_single_task_context", lambda value: None)
    monkeypatch.setattr(terminal_program_population, "terminal_program_partition",
        lambda value: {"program_paths": ["bottle.py"], "support_hashes": {}})
    monkeypatch.setattr(terminal_initial_context, "prepare_initial_context", lambda **kwargs: calls.append(kwargs) or {})
    preparation.initial_context(state=tmp_path, capsule_selection=mode)
    assert calls[0]["prepared"] is prepared
    assert calls[0]["required_raw_paths"] == [preparation.INSTRUCTION, preparation.SMOKE]
    if mode == DEFAULT:
        assert "retrieval_ranked_capsules" not in calls[0]
    else:
        assert calls[0]["retrieval_ranked_capsules"] is True


def test_ranked_legacy_profile_refuses_before_initial_indexing(original, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context
    root, instruction, state = original
    preparation.prepare(repository=root, instruction=instruction, state=state)
    calls = []
    monkeypatch.setattr(terminal_initial_context, "prepare_initial_context",
        lambda **kwargs: calls.append(kwargs) or {})
    with pytest.raises(ValueError, match="explicit public task profile.*signed program/support partition"):
        preparation.initial_context(state=state, capsule_selection=MODE)
    assert calls == []
    assert not (root / ".runtime/terminal-vectors").exists()
    assert not (state / "planner-invoked.json").exists()


def test_prepare_freezes_ranking_and_default_receipt_shape(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz"): pass
    raw = stream.getvalue()
    (archive / "runtime.tar.gz").write_bytes(raw)
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "codex_version": CLI_VERSION})))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr=""))
    ranked = benchmark.prepare(dataset=dataset, output=tmp_path / "ranked", archive=archive,
        arm="full", capsule_selection=MODE)
    assert ranked["capsule_selection"] == MODE and ranked["provider_calls"] == 0
    assert ranked["capsule_selection_token_savings_claimed"] is False
    assert json.loads((tmp_path / "ranked/config.json").read_text())["agents"][0]["kwargs"]["capsule_selection"] == MODE
    default = benchmark.prepare(dataset=dataset, output=tmp_path / "default", archive=archive, arm="full")
    assert not any(key.startswith("capsule_selection") for key in default)
    assert len(calls) == 2 and all("--dry-run" in argv for argv in calls)


def test_adapter_archive_guard_runs_before_environment_actions(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", capsule_selection=MODE)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for call in (agent.setup(environment), agent.run("Authored instruction", environment, SimpleNamespace())):
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            asyncio.run(call)
    environment.upload_file.assert_not_awaited(); environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


@pytest.mark.parametrize("mode", [DEFAULT, MODE])
def test_harbor_passes_ranking_to_driver_and_retains_metadata(tmp_path, monkeypatch, mode):
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
        runtime_archive=str(tmp_path), arm="full", capsule_selection=mode)
    context = AgentContext(); asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    if mode == DEFAULT:
        assert "--capsule-selection" not in calls[0]
        assert not any(key.startswith("capsule_selection") for key in context.metadata)
    else:
        assert calls[0].count("--capsule-selection") == 1
        assert calls[0][calls[0].index("--capsule-selection") + 1] == MODE
        assert context.metadata["capsule_selection"] == MODE
        assert context.metadata["capsule_selection_token_savings_claimed"] is False


def test_collect_and_execute_detect_capsule_selection_drift(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", capsule_selection=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "capsule_selection": MODE}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["capsule_selection_config_unchanged"] is True
    prepared.pop("capsule_selection")
    (output / "preparation.json").write_text(json.dumps(prepared))
    observed = benchmark.collect(output)
    assert observed["capsule_selection_config_unchanged"] is False
    assert observed["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("changed selection executed"))
    with pytest.raises(ValueError, match="prepared capsule selection changed"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_cli_selection_is_frozen_during_prepare(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--capsule-selection", MODE])
    benchmark.main()
    assert calls[0]["capsule_selection"] == MODE
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--capsule-selection", MODE])
        with pytest.raises(SystemExit) as error: benchmark.main()
        assert error.value.code == 2


@pytest.mark.parametrize("stage", ["initial_context", "context"])
def test_driver_rejects_a_dropped_ranking_before_planning_or_dispatch(tmp_path, monkeypatch, stage):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import admission_environment
    state = tmp_path / "state/run"; state.mkdir(parents=True)
    calls = []
    for key, value in admission_environment(None).items(): monkeypatch.setenv(key, value)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda value: 0)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *a, **k: None)
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    def initial(**kwargs):
        calls.append("initial_context")
        assert kwargs["capsule_selection"] == MODE
        return {} if stage == "initial_context" else {"capsule_selection": MODE}
    monkeypatch.setattr(driver.preparation, "initial_context", initial)
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: calls.append("planning") or {"qualified": True})
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs: calls.append("context") or {"context_bundle": {}})
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: pytest.fail("a provider was invoked"))
    report = driver.run(instruction=tmp_path / "unread", state=state, arm="full",
        source384_config=tmp_path / "unread-source384", capsule_selection=MODE)
    assert report["error_phase"] == stage and report["error"]["type"] == "ValueError"
    assert report["provider_invocations"] == []
    assert report["capsule_selection"] == MODE
    assert calls == (["initial_context"] if stage == "initial_context" else ["initial_context", "planning", "context"])
