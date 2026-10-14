"""Source-view treatment controls and adapter wiring; no provider/Docker calls."""
import asyncio
from copy import deepcopy
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
from benchmarks.agent_supervisor.container_coding import terminal_source_view_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import SOURCE_PATHS
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task

DEFAULT = policy.DEFAULT_SOURCE_VIEW_PROFILE
MODE = "python-whitespace@1"


def manifest():
    return _supervisor_manifest({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []})


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_preserves_configuration_and_observation_shapes(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    baseline = benchmark.config_for(*args)
    assert benchmark.config_for(*args, source_view_profile=DEFAULT) == baseline
    assert "source_view_profile" not in baseline["agents"][0]["kwargs"]
    assert policy.source_view_selection(DEFAULT) == policy.source_view_observation(DEFAULT) == {}
    policy.require_source_view_archive(None, DEFAULT)


@pytest.mark.parametrize("profile", policy.SOURCE_VIEW_PROFILES[1:])
@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
def test_enabled_configuration_changes_controls_and_keeps_explicit_profile(tmp_path, profile, transport):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    default = benchmark.config_for(*args, semantic_transport_schema=transport)
    enabled = benchmark.config_for(*args, semantic_transport_schema=transport, source_view_profile=profile)
    assert enabled["agents"][0]["kwargs"] == {**default["agents"][0]["kwargs"], "source_view_profile": profile}
    def controls(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64}, task=benchmark.TASK,
            model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)
    assert controls(default)["configuration_sha256"] != controls(enabled)["configuration_sha256"]
    assert policy.source_view_observation(profile) == dict(source_view_profile=profile,
        source_view_token_savings_claimed=False, source_view_authority=False)


@pytest.mark.parametrize("value", [None, True, {}, "minified", "python-whitespace@2"])
def test_invalid_profile_refuses_before_files_or_providers(tmp_path, value):
    with pytest.raises(ValueError, match="supported source view profile"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", source_view_profile=value)


@pytest.mark.parametrize("arm", ["no-index", "codex", "unknown"])
def test_enabled_profile_requires_full_indexed_arm(tmp_path, arm):
    with pytest.raises(ValueError, match="full indexed arm"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm=arm, source_view_profile=MODE)


def test_grok_selection_keeps_pinned_provider_and_source_profile(tmp_path):
    config = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        provider_profile=GROK_PROFILE, source_view_profile=MODE)
    assert config["agents"][0]["kwargs"]["provider_profile"] == GROK_PROFILE
    assert config["agents"][0]["kwargs"]["source_view_profile"] == MODE


@pytest.mark.parametrize("relative", SOURCE_PATHS)
def test_opt_in_reuses_worker_capability_and_refuses_each_changed_producer(relative):
    value = manifest()
    next(row for row in value["files"] if row["path"] == "source/" + relative)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rebuild runtime archive"):
        policy.require_source_view_archive(value, MODE)


def test_current_worker_capability_includes_all_source_view_producers():
    required = {"ipfs_accelerate_py/agent_supervisor/runtime/" + name + ".py" for name in (
        "worker_source_client", "worker_source_broker", "worker_source_views", "reversible_source_view")}
    required.update({"benchmarks/agent_supervisor/container_coding/terminal_source_view_policy.py",
        "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py"})
    assert required <= set(SOURCE_PATHS)
    policy.require_source_view_archive(manifest(), MODE)


@pytest.mark.parametrize("profile", policy.SOURCE_VIEW_PROFILES)
def test_model_router_command_only_forwards_explicit_enabled_profile(profile):
    kwargs = dict(router=Path("/router"), model="authored-model", reasoning="high", timeout=30,
        semantic_repository=Path("/repository"), doctor=None)
    original = implementation_argv(**kwargs)
    selected = implementation_argv(**kwargs, source_view_profile=profile)
    if profile == DEFAULT:
        assert original == selected and "--source-view-profile" not in selected
    else:
        assert selected == [*original, "--source-view-profile", profile]


@pytest.mark.parametrize("doctor", [{"route": "doctor_candidate"}, {"route": "doctor_contract_candidate"},
    {"route": "unknown"}])
def test_source_views_cannot_admit_missing_candidate_contract_or_unknown_route(doctor):
    with pytest.raises(ValueError, match="candidate required|semantic model-router"):
        implementation_argv(router=Path("/router"), model="authored", reasoning="high", timeout=30,
            semantic_repository=Path("/repository"), doctor=doctor, source_view_profile=MODE)


def test_source_views_require_semantic_repository_in_signed_command():
    with pytest.raises(ValueError, match="semantic model-router"):
        implementation_argv(router=Path("/router"), model="authored", reasoning="high", timeout=30,
            semantic_repository=None, doctor=None, source_view_profile=MODE)


def test_prepare_freezes_source_view_profile_and_default_receipt_shape(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz"):
        pass
    raw = stream.getvalue()
    (archive / "runtime.tar.gz").write_bytes(raw)
    value = manifest()
    value["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    (archive / "manifest.json").write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr=""))
    enabled = benchmark.prepare(dataset=dataset, output=tmp_path / "enabled", archive=archive,
        arm="full", source_view_profile=MODE)
    assert enabled["source_view_profile"] == MODE and enabled["provider_calls"] == 0
    assert enabled["source_view_token_savings_claimed"] is False
    assert json.loads((tmp_path / "enabled/config.json").read_text())["agents"][0]["kwargs"]["source_view_profile"] == MODE
    default = benchmark.prepare(dataset=dataset, output=tmp_path / "default", archive=archive, arm="full")
    assert not any(key.startswith("source_view") for key in default)
    assert len(calls) == 2 and all("--dry-run" in argv for argv in calls)


@pytest.mark.parametrize("profile", [DEFAULT, MODE])
def test_harbor_passes_profile_to_driver_and_retains_metadata(tmp_path, monkeypatch, profile):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
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
        runtime_archive=str(tmp_path), arm="full", source_view_profile=profile)
    context = AgentContext()
    asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    if profile == DEFAULT:
        assert "--source-view-profile" not in calls[0]
        assert not any(key.startswith("source_view") for key in context.metadata)
    else:
        assert calls[0].count("--source-view-profile") == 1
        assert calls[0][calls[0].index("--source-view-profile") + 1] == MODE
        assert context.metadata["source_view_profile"] == MODE
        assert context.metadata["source_view_token_savings_claimed"] is False


def test_adapter_archive_guard_runs_before_environment_actions(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    value = manifest()
    value["files"][0]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile=MODE)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for call in (agent.setup(environment), agent.run("Authored instruction", environment, SimpleNamespace())):
        with pytest.raises(ValueError, match="rebuild runtime archive"):
            asyncio.run(call)
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


def test_collect_and_execute_detect_dropped_prepared_profile(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", source_view_profile=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "source_view_profile": MODE}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["source_view_config_unchanged"] is True
    prepared.pop("source_view_profile")
    (output / "preparation.json").write_text(json.dumps(prepared))
    observed = benchmark.collect(output)
    assert observed["source_view_config_unchanged"] is False
    assert observed["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("changed profile executed"))
    with pytest.raises(ValueError, match="prepared source view profile changed"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_cli_profile_selection_is_frozen_during_prepare(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--source-view-profile", MODE])
    benchmark.main()
    assert calls[0]["source_view_profile"] == MODE
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--source-view-profile", MODE])
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 2
