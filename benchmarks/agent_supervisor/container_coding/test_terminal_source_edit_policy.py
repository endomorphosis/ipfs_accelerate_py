"""Mapped edit selection is explicit, frozen and never a new write grant."""
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
from benchmarks.agent_supervisor.container_coding import terminal_source_edit_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

MODE = policy.MAPPED_SOURCE_EDIT_MODE
REPEATED_MODE = policy.REPEATED_SOURCE_EDIT_MODE
DEFAULT = policy.DEFAULT_SOURCE_EDIT_MODE


@pytest.fixture(params=[MODE, REPEATED_MODE])
def mode(request):
    return request.param


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
@pytest.mark.parametrize("planning", ["model", "declared-contract@1"])
def test_disabled_preserves_configuration_and_controls(tmp_path, arm, numerical, planning):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    options = dict(numerical_formalization_profile=numerical, planning_mode=planning)
    legacy = benchmark.config_for(*args, **options)
    explicit = benchmark.config_for(*args, **options, source_edit_mode=DEFAULT)
    assert explicit == legacy and seal(explicit) == seal(legacy)
    assert "source_edit_mode" not in legacy["agents"][0]["kwargs"]
    assert seal(legacy)["schema"] == (controls.PLANNING_SCHEMA if planning != "model"
        else controls.NUMERICAL_SCHEMA if numerical else controls.SCHEMA)
    assert policy.source_edit_selection(DEFAULT) == policy.source_edit_observation(DEFAULT) == {}
    policy.require_source_edit_archive(None, DEFAULT)


@pytest.mark.parametrize("profile", ["identity@1", "python-whitespace@1", "python-comments@1"])
@pytest.mark.parametrize("planning", ["model", "declared-contract@1"])
def test_comparison_binds_selected_edit_mode(tmp_path, profile, planning, mode):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    options = dict(source_view_profile=profile, planning_mode=planning,
        numerical_formalization_profile="python-numerical-control-structure@3")
    baseline = benchmark.config_for(*args, **options)
    selected = benchmark.config_for(*args, **options, source_edit_mode=mode)
    assert selected["agents"][0]["kwargs"] == {**baseline["agents"][0]["kwargs"], "source_edit_mode": mode}
    record = seal(selected)
    assert record["schema"] == controls.SOURCE_EDIT_SCHEMA and controls.validate_controls(record)
    assert controls.compare_controls(observed(selected), observed(selected))["matches"] is True
    result = controls.compare_controls(observed(selected), observed(baseline))
    assert result["matches"] is False
    assert result["differences"] == ["experimental_controls_sha256.source_edit_mode"]


@pytest.mark.parametrize("mutation", ["missing", "unknown", "bool", "digest", "v3", "v2", "v1"])
def test_controls_reject_edited_declarations(tmp_path, mutation, mode):
    record = seal(benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        source_view_profile="identity@1", source_edit_mode=mode))
    if mutation == "missing": del record["experimental_controls_sha256"]["source_edit_mode"]
    elif mutation == "unknown": record["experimental_controls_sha256"]["other"] = "a" * 64
    elif mutation == "bool": record["experimental_controls_sha256"]["source_edit_mode"] = True
    elif mutation == "digest": record["experimental_controls_sha256"]["source_edit_mode"] = "0" * 64
    else: record["schema"] = {"v3": controls.PLANNING_SCHEMA, "v2": controls.NUMERICAL_SCHEMA,
                               "v1": controls.SCHEMA}[mutation]
    assert controls.validate_controls(record) is False


@pytest.mark.parametrize("value", [None, True, 1, {}, [], "", "mapped-span@3", "mapped-span@1\x00", "mapped-span@2\x00"])
def test_bad_mode_refuses_before_file_access(tmp_path, value):
    with pytest.raises(ValueError, match="supported source edit mode"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", source_edit_mode=value)


@pytest.mark.parametrize("arm,profile", [("no-index", "disabled"), ("full", "disabled"),
    ("full", None), ("full", True), ("full", "unsupported")])
def test_requires_full_enabled_source_views(tmp_path, arm, profile, mode):
    with pytest.raises(ValueError):
        benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", arm,
            source_view_profile=profile, source_edit_mode=mode)


def test_controls_refuse_invalid_direct_kwargs(tmp_path, mode):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "no-index")
    config["agents"][0]["kwargs"]["source_edit_mode"] = mode
    with pytest.raises(ValueError, match="full indexed arm"): seal(config)


@pytest.mark.parametrize("path", policy.SOURCE_PATHS)
def test_runtime_requires_every_current_edit_producer(path, mode):
    manifest = inventory()
    next(row for row in manifest["files"] if row["path"] == "source/" + path)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_source_edit_archive(manifest, mode)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool_bytes", "wrong_bytes"])
def test_inventory_cannot_hide_edited_producer(mutation, mode):
    manifest = inventory(); row = manifest["files"][0]
    if mutation == "missing": manifest["files"].pop(0)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "bool_bytes": row["bytes"] = True
    else: row["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_source_edit_archive(manifest, mode)


@pytest.mark.parametrize("manifest", [None, True, {}, {"files": {}}, {"files": "other"}])
def test_malformed_inventory_refuses(manifest, mode):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_source_edit_archive(manifest, mode)


def test_prepare_freezes_selection_without_dispatch(tmp_path, monkeypatch, mode):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"): pass
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": benchmark._hash(archive / "runtime.tar.gz"), "codex_version": CLI_VERSION})))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr=""))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive, arm="full",
        source_view_profile="identity@1", source_edit_mode=mode)
    assert result["source_edit_mode"] == mode
    assert result["comparison_controls"]["schema"] == controls.SOURCE_EDIT_SCHEMA
    assert result["provider_calls"] == 0 and calls and all("--dry-run" in c for c in calls)
    assert result["source_edit_authority"] is False and result["source_edit_token_savings_claimed"] is False
    assert benchmark.collect(tmp_path / "out")["source_edit_config_unchanged"] is True


@pytest.mark.parametrize("selected", [DEFAULT, MODE, REPEATED_MODE])
@pytest.mark.parametrize("profile", ["identity@1", "python-whitespace@1", "python-comments@1"])
def test_harbor_forwards_frozen_selection(tmp_path, monkeypatch, selected, profile):
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
        runtime_archive=str(tmp_path), arm="full", source_view_profile=profile, source_edit_mode=selected)
    context = AgentContext(); asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    if selected == DEFAULT:
        assert "--source-edit-mode" not in calls[0] and "source_edit_mode" not in context.metadata
    else:
        assert calls[0].count("--source-edit-mode") == 1
        assert calls[0][calls[0].index("--source-edit-mode") + 1] == selected
        assert context.metadata["source_edit_mode"] == selected
        assert context.metadata["source_edit_authority"] is False


@pytest.mark.parametrize("failure", ["archive", "mode_drift", "profile_drift", "arm_drift", "census_drift"])
def test_adapter_refuses_before_environment_effects(tmp_path, failure, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile="identity@1", source_edit_mode=mode)
    if failure == "mode_drift": agent.source_edit_mode = DEFAULT
    elif failure == "profile_drift": agent.source_view_profile = "disabled"
    elif failure == "arm_drift": agent.arm = "no-index"
    elif failure == "census_drift": agent.metadata_census_only = True
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (env,)), (agent.run, ("Authored instruction", env, SimpleNamespace()))):
        with pytest.raises(ValueError): asyncio.run(method(*args))
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited(); env.download_file.assert_not_awaited()


def test_census_constructor_refuses_edit_mode(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    with pytest.raises(ValueError, match="coding worker"):
        FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
            runtime_archive=str(tmp_path), arm="full", source_view_profile="identity@1", source_edit_mode=mode,
            metadata_census_only=True)


@pytest.mark.parametrize("where", ["preparation", "config"])
def test_collect_and_execute_detect_selection_drift(tmp_path, monkeypatch, where, mode):
    dataset, task = authored_task(tmp_path); output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        source_view_profile="identity@1", source_edit_mode=mode)
    if where == "config": config["agents"][0]["kwargs"].pop("source_edit_mode")
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True, "source_view_profile": "identity@1",
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION}
    if where == "config": prepared["source_edit_mode"] = mode
    (output / "preparation.json").write_text(json.dumps(prepared))
    result = benchmark.collect(output)
    assert result["source_edit_config_unchanged"] is False
    assert result["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("changed selection executed"))
    with pytest.raises(ValueError, match="prepared source edit mode changed"): benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_cli_selection_is_frozen_by_prepare(tmp_path, monkeypatch, mode):
    calls = []; monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path), "--archive", str(tmp_path),
        "--output", str(tmp_path / "out"), "--source-view-profile", "identity@1", "--source-edit-mode", mode])
    benchmark.main(); assert calls[0]["source_edit_mode"] == mode
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--source-edit-mode", mode])
        with pytest.raises(SystemExit) as exc: benchmark.main()
        assert exc.value.code == 2


@pytest.mark.parametrize("prepared_mode,configured_mode", [(MODE, REPEATED_MODE), (REPEATED_MODE, MODE)])
def test_changing_between_edit_versions_invalidates_collection_and_execution(tmp_path, monkeypatch,
                                                                           prepared_mode, configured_mode):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"; output.mkdir()
    options = dict(source_view_profile="identity@1")
    original = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        **options, source_edit_mode=prepared_mode)
    changed = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        **options, source_edit_mode=configured_mode)
    mismatch = controls.compare_controls(observed(original), observed(changed))
    assert mismatch["matches"] is False
    assert mismatch["differences"] == ["experimental_controls_sha256.source_edit_mode"]
    (output / "config.json").write_text(json.dumps(changed))
    # Even a recomputed config digest cannot turn an explicit selection change into admission.
    prepared = dict(arm="full", dataset=str(dataset), prepared=True,
        source_view_profile="identity@1", source_edit_mode=prepared_mode,
        config_sha256=benchmark._hash(output / "config.json"), task_input_sha256=benchmark._task_hashes(task),
        model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)
    (output / "preparation.json").write_text(json.dumps(prepared))
    result = benchmark.collect(output)
    assert result["source_edit_config_unchanged"] is False
    assert result["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("changed mode executed"))
    with pytest.raises(ValueError, match="prepared source edit mode changed"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("old,new", [(MODE, REPEATED_MODE), (REPEATED_MODE, MODE)])
def test_adapter_refuses_other_enabled_edit_version_before_environment_effects(tmp_path, old, new):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile="identity@1", source_edit_mode=old)
    agent.source_edit_mode = new
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for method, args in ((agent.setup, (env,)), (agent.run, ("Authored instruction", env, SimpleNamespace()))):
        with pytest.raises(ValueError, match="source edit mode changed"):
            asyncio.run(method(*args))
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited(); env.download_file.assert_not_awaited()


def test_repeated_delivery_transport_owner_is_required_and_cannot_be_omitted():
    relative = "ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_broker.py"
    assert relative in policy.SOURCE_PATHS
    manifest = inventory()
    manifest["files"] = [row for row in manifest["files"] if row["path"] != "source/" + relative]
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_source_edit_archive(manifest, REPEATED_MODE)
