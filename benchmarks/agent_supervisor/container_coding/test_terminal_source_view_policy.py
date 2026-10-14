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


"""Frozen source view preview selection, without provider or Docker calls."""
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
from benchmarks.agent_supervisor.container_coding import terminal_source_view_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest


@pytest.fixture(params=(policy.RAW_SOURCE_VIEW_POLICY, policy.COMPACT_SOURCE_VIEW_POLICY))
def mode(request):
    return request.param


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + relative,
                       "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                       "bytes": (root / relative).stat().st_size}
                      for relative in policy.SOURCE_PATHS]}


@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
def test_default_is_unchanged_and_explicit_mode_is_config_bound(tmp_path, transport, mode):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    old = benchmark.config_for(*args, semantic_transport_schema=transport)
    assert old == benchmark.config_for(*args, semantic_transport_schema=transport, source_view_advisory_policy="legacy")
    assert "source_view_advisory_policy" not in old["agents"][0]["kwargs"]
    new = benchmark.config_for(*args, semantic_transport_schema=transport, source_view_advisory_policy=mode)
    assert new["agents"][0]["kwargs"]["source_view_advisory_policy"] == mode
    assert "intent_requirement_contract" not in new["agents"][0]["kwargs"]
    def seal(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
                              task=benchmark.TASK, model=benchmark.MODEL,
                              reasoning_effort="high", cli_version=CLI_VERSION)
    assert seal(old)["configuration_sha256"] != seal(new)["configuration_sha256"]


@pytest.mark.parametrize("change", [None, "missing", "modified", "duplicate", "size", "boolean_size"])
def test_required_helper_body_is_bound_to_archive(change, mode):
    manifest = inventory()
    target = next(r for r in manifest["files"] if r["path"].endswith("/source_view_policy.py"))
    if change == "missing":
        manifest["files"].remove(target)
    elif change == "modified":
        target["sha256"] = "0" * 64
    elif change == "duplicate":
        manifest["files"].append(dict(target))
    elif change == "size":
        target["bytes"] += 1
    elif change == "boolean_size":
        target["bytes"] = True
    if change is None:
        policy.require_source_view_advisory_archive(manifest, mode)
    else:
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            policy.require_source_view_advisory_archive(manifest, mode)
    policy.require_source_view_advisory_archive({}, "legacy")


@pytest.mark.parametrize("value", [None, True, {}, "source-view-foreign@1"])
def test_unknown_mode_refused_before_any_preparation_reads(tmp_path, value):
    with pytest.raises(ValueError, match="supported source view policy"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
                          archive=tmp_path / "missing", arm="full", source_view_advisory_policy=value)


@pytest.mark.parametrize("options", [{"arm": "no-index"}, {"arm": "full", "provider_profile": GROK_PROFILE}])
def test_unqualified_route_refused_before_any_preparation_reads(tmp_path, options, mode):
    with pytest.raises(ValueError, match="source view policy requires"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
                          archive=tmp_path / "missing", source_view_advisory_policy=mode, **options)


def test_adapter_rejects_missing_archive_support_before_upload(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_advisory_policy=mode)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        asyncio.run(agent.run("Public fixture", environment, SimpleNamespace()))
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


def test_adapter_forwards_mode_and_keeps_planning(tmp_path, monkeypatch, mode):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    manifest = {**inventory(), "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []}
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
        runtime_archive=str(tmp_path), arm="full", source_view_advisory_policy=mode)
    context = AgentContext()
    asyncio.run(agent.run("Public fixture", Environment(), context))
    assert len(calls) == 1
    assert calls[0][calls[0].index("--source-view-advisory-policy") + 1] == mode
    assert context.metadata["source_view_advisory_policy"] == mode
    assert context.metadata["planning_strategy"] == "direct"


def test_cli_freezes_mode_only_at_prepare(tmp_path, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--source-view-advisory-policy", mode])
    benchmark.main()
    assert calls[0]["source_view_advisory_policy"] == mode
    assert calls[0]["intent_requirement_contract"] is None
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path), "--source-view-advisory-policy", mode])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_route_builder_selects_only_model_coding(mode):
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    kwargs = dict(router=Path("/router"), model="fixture", reasoning="high", timeout=30,
                  semantic_repository=Path("/app"), doctor={"route": "model_router", "status": "residual"})
    old = implementation_argv(**kwargs)
    assert implementation_argv(**kwargs, source_view_advisory_policy="legacy") == old
    new = implementation_argv(**kwargs, source_view_advisory_policy=mode)
    assert new[new.index("--source-view-advisory-policy") + 1] == mode
    with pytest.raises(ValueError, match="committed provider-free Doctor candidate required"):
        implementation_argv(**{**kwargs, "doctor": {"route": "doctor_candidate"}}, source_view_advisory_policy=mode)


def test_collect_and_execute_detect_policy_drift(tmp_path, mode):
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    for folder in (task, task / "environment", task / "tests"):
        folder.mkdir(parents=True, exist_ok=True)
    for name, text in {"instruction.md": "Fixture", "task.toml": 'version = "1.0"\n',
                       "environment/Dockerfile": "FROM scratch\n", "tests/test.sh": "exit 0\n"}.items():
        (task / name).write_text(text)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", source_view_advisory_policy=mode)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "source_view_advisory_policy": mode}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["source_view_advisory_config_unchanged"] is True
    prepared["source_view_advisory_policy"] = "legacy"
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["source_view_advisory_config_unchanged"] is False
    with pytest.raises(ValueError, match="prepared source view policy selection changed"):
        benchmark.execute(output)


@pytest.mark.parametrize("relative", policy.SOURCE_PATHS)
def test_every_selected_runtime_body_is_archive_bound(relative, mode):
    manifest = inventory()
    row = next(item for item in manifest["files"] if item["path"] == "source/" + relative)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_source_view_advisory_archive(manifest, mode)


@pytest.mark.parametrize("manifest", [None, True, {"files": True}, {"files": {}}, {"files": "foreign"}])
def test_malformed_archive_inventory_rejected(manifest, mode):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_source_view_advisory_archive(manifest, mode)


def test_prepare_freezes_output_policy_without_provider_calls(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    # Numerical deployment inspects the archive even when no assets are selected.
    # A real empty gzip/tar models that absence without executable fixture content.
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    raw = (archive / "runtime.tar.gz").read_bytes()
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "codex_version": CLI_VERSION})))
    commands = []
    def dry_run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry_run)
    selected = benchmark.prepare(dataset=dataset, output=tmp_path / "selected", archive=archive,
                                 arm="full", source_view_advisory_policy=mode)
    assert selected["source_view_advisory_policy"] == mode
    assert selected["provider_calls"] == 0
    assert json.loads((tmp_path / "selected/config.json").read_text())["agents"][0]["kwargs"]["source_view_advisory_policy"] == mode
    assert len(commands) == 1 and commands[0][-1] == "--dry-run"
    legacy = benchmark.prepare(dataset=dataset, output=tmp_path / "legacy", archive=archive, arm="full")
    assert "source_view_advisory_policy" not in legacy
    assert "source_view_advisory_policy" not in json.loads((tmp_path / "legacy/config.json").read_text())["agents"][0]["kwargs"]


@pytest.mark.parametrize("extra", [
    ["--purpose", "planning"], ["--preflight"], ["--provider", "grok_cli"],
    ["--semantic-metadata-view", "common-bindings@1", "--semantic-metadata-census-only"],
    ["--doctor-candidate-artifact", "/artifact"], ["--finite-repository-artifact", "/artifact"],
])
def test_worker_guard_refuses_non_coding_or_provider_free_routes(extra, monkeypatch, mode):
    import argparse
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    source = WORKER_ENTRY.split("parser=argparse.ArgumentParser()", 1)[1].split(
        "from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner", 1)[0]
    namespace = {"argparse": argparse, "pathlib": __import__("pathlib"), "selected_provider": "codex_cli",
                 "sys": SimpleNamespace(argv=["worker", "--source-view-advisory-policy", mode,
                    "--semantic-repository", "/app", *extra])}
    monkeypatch.setattr(sys, "argv", namespace["sys"].argv)
    with pytest.raises(SystemExit, match="source view policy requires|provider differs"):
        exec("parser=argparse.ArgumentParser()" + source, namespace)


def test_worker_guard_accepts_selected_semantic_coding(monkeypatch, mode):
    import argparse
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    source = WORKER_ENTRY.split("parser=argparse.ArgumentParser()", 1)[1].split(
        "from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner", 1)[0]
    namespace = {"argparse": argparse, "pathlib": __import__("pathlib"), "selected_provider": "codex_cli",
                 "sys": SimpleNamespace(argv=["worker", "--source-view-advisory-policy", mode,
                    "--semantic-repository", "/app"])}
    monkeypatch.setattr(sys, "argv", namespace["sys"].argv)
    exec("parser=argparse.ArgumentParser()" + source, namespace)
    assert namespace["args"].purpose == "coding"
    assert namespace["args"].source_view_advisory_policy == mode


def test_adapter_refuses_observation_only_census_combination(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    with pytest.raises(ValueError, match="executed coding route"):
        FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
            runtime_archive=str(tmp_path), arm="full", source_view_advisory_policy=mode,
            metadata_census_only=True, semantic_metadata_view="common-bindings@1",
            coding_reply_mode="ordinary-completion@1")


def dispatch_report(mode):
    from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import create_binding
    binding = create_binding(workspace=Path('/authored/candidate'), requested_policy=mode)
    identity = 'authored-coding-invocation'
    row = dict(phase='coding', purpose='coding', provider='codex_cli', invocation_id=identity,
        workspace='/authored/candidate', source_view_advisory_policy=binding,
        model_prompt_sha256='a' * 64, model_prompt_bytes=2300)
    return dict(source_view_advisory_policy=mode, provider_invocations=[{'phase': 'planning'}, row],
        context_input_audit=dict(status='verified_all_observed_model_inputs',
            all_observed_coding_inputs_verified=True, coding_receipts_seen=1, usable_coding_receipts=1,
            read_errors=[], invocations=[dict(invocation_id=identity,
                model_input_verified=True, status='verified_exact_model_input')],
            matches=[dict(invocation_id=identity, model_input_verified=True,
                model_prompt_sha256='a' * 64, model_prompt_bytes=2300,
                model_input_checks=dict(source_view_helper_source=True, model_prompt_sha256=True,
                    model_prompt_bytes=True, workspace_advisory_sha256=True, workspace_advisory_bytes=True))]))


def test_dispatch_qualifies_exact_advisory_without_claiming_adoption(mode):
    result = policy.source_view_dispatch_observation(dispatch_report(mode), mode)
    assert result['effective_advisory_observed'] is True
    assert result['status'] == 'verified_advisory'
    assert result['planning_policy_absent'] is True
    assert result['helper_adoption_established'] is False
    assert result['token_savings_established'] is False


@pytest.mark.parametrize('mutation', ['declared', 'missing_binding', 'schema', 'instruction',
    'duplicate_coding', 'duplicate_audit', 'missing_audit', 'helper_check', 'prompt_digest',
    'advisory_check', 'planning_policy', 'unreceipted', 'unknown_phase', 'malformed_phase',
    'malformed_id', 'malformed_hash', 'no_coding', 'read_error'])
def test_dispatch_refuses_unobserved_or_uncorrelated_treatment(mutation, mode):
    from copy import deepcopy
    report = dispatch_report(mode)
    coding = report['provider_invocations'][1]
    audit = report['context_input_audit']
    if mutation == 'declared': report['source_view_advisory_policy'] = 'legacy'
    elif mutation == 'missing_binding': coding.pop('source_view_advisory_policy')
    elif mutation == 'schema': coding['source_view_advisory_policy']['schema'] = 'foreign'
    elif mutation == 'instruction': coding['source_view_advisory_policy']['instruction_sha256'] = '0' * 64
    elif mutation == 'duplicate_coding': report['provider_invocations'].append(deepcopy(coding))
    elif mutation == 'duplicate_audit': audit['invocations'].append(deepcopy(audit['invocations'][0]))
    elif mutation == 'missing_audit': report.pop('context_input_audit')
    elif mutation == 'helper_check': audit['matches'][0]['model_input_checks']['source_view_helper_source'] = False
    elif mutation == 'prompt_digest': audit['matches'][0]['model_prompt_sha256'] = '0' * 64
    elif mutation == 'advisory_check': audit['matches'][0]['model_input_checks']['workspace_advisory_sha256'] = False
    elif mutation == 'planning_policy': report['provider_invocations'][0]['source_view_advisory_policy'] = None
    elif mutation == 'unreceipted': report['unreceipted_provider_attempt'] = True
    elif mutation == 'unknown_phase': report['provider_invocations'].append({'phase': 'foreign'})
    elif mutation == 'malformed_phase': report['provider_invocations'].append({'phase': []})
    elif mutation == 'malformed_id': coding['invocation_id'] = {}
    elif mutation == 'malformed_hash': coding['model_prompt_sha256'] = 'foreign'
    elif mutation == 'no_coding': report['provider_invocations'] = [{'phase': 'planning'}]
    elif mutation == 'read_error': audit['read_errors'] = ['authored']
    result = policy.source_view_dispatch_observation(report, mode)
    assert result['effective_advisory_observed'] is False
    assert result['status'] == 'unverified'


def test_explicit_preview_versions_have_distinct_frozen_controls(tmp_path):
    configs = [benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        source_view_advisory_policy=mode) for mode in (policy.RAW_SOURCE_VIEW_POLICY, policy.COMPACT_SOURCE_VIEW_POLICY)]
    receipts = [build_controls(config, task_input_sha256={"instruction.md": "a" * 64}, task=benchmark.TASK,
        model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION) for config in configs]
    assert receipts[0]["configuration_sha256"] != receipts[1]["configuration_sha256"]
    assert configs[0]["agents"][0]["kwargs"]["source_view_advisory_policy"] == "source-view-raw@1"
    assert configs[1]["agents"][0]["kwargs"]["source_view_advisory_policy"] == "source-view-compact@1"


def test_dispatch_cannot_relabel_another_preview_version(mode):
    other = policy.COMPACT_SOURCE_VIEW_POLICY if mode == policy.RAW_SOURCE_VIEW_POLICY else policy.RAW_SOURCE_VIEW_POLICY
    report = dispatch_report(other)
    report["source_view_advisory_policy"] = mode
    result = policy.source_view_dispatch_observation(report, mode)
    assert result["requested_policy_matches_report"] is True
    assert result["invocations"][0]["policy_binding_valid"] is False
    assert result["effective_advisory_observed"] is False


@pytest.mark.parametrize("selection", [
    {"source_view_advisory_policy": "source-view-compact@1"},
    {"repository_source_policy": "repository-source-compact@1"},
    {"source_input_policy": "source-input-compact@1"},
    {"query_reference_policy": "query-reference@1"},
])
def test_signed_source_profile_refuses_unqualified_parallel_projection(selection):
    with pytest.raises(ValueError, match="signed source views cannot combine"):
        policy.validate_source_context_combination(source_view_profile="identity@1", **selection)


def test_capsule_selectors_cannot_silently_override_each_other():
    with pytest.raises(ValueError, match="retrieval capsule selection cannot combine"):
        policy.validate_source_context_combination(
            capsule_selection="verified-retrieval-symbol-rank@1", semantic_context_profile="indexed-compact@1")
