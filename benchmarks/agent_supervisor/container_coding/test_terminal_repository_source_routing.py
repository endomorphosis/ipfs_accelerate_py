"""Repository MCP selection reaches the admitted coding worker without live calls."""
from __future__ import annotations

import argparse
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
from benchmarks.agent_supervisor.container_coding import terminal_repository_source_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task


@pytest.fixture(params=(policy.RAW_REPOSITORY_SOURCE_POLICY, policy.COMPACT_REPOSITORY_SOURCE_POLICY))
def mode(request):
    return request.param


def inventory():
    return policy.repository_source_inventory()


def archive_fixture(root):
    """Real member bodies; all deployment/provider operations are mocked."""
    root.mkdir()
    manifest = _supervisor_manifest({**inventory(), "codex_version": CLI_VERSION, "learned_requirements": []})
    sources = policy._current_sources()
    checkout = Path(policy.__file__).resolve().parents[3]
    for row in manifest["files"]:
        if row["path"] not in sources:
            sources[row["path"]] = (checkout / row["path"].removeprefix("source/")).read_bytes()
    archive = root / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, body in sources.items():
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(body), 0o644
            stream.addfile(entry, io.BytesIO(body))
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def prepared_fixture(root, mode):
    dataset, task = authored_task(root)
    output = root / "output"
    output.mkdir()
    config = benchmark.config_for(dataset, output, root / "archive", "full", repository_source_policy=mode)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "repository_source_policy": mode}
    (output / "preparation.json").write_text(json.dumps(prepared))
    return output, prepared, config, task


@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
def test_selection_changes_only_frozen_coding_control(tmp_path, mode, transport):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    original = benchmark.config_for(*args, semantic_transport_schema=transport)
    explicit = benchmark.config_for(*args, semantic_transport_schema=transport, repository_source_policy="legacy")
    assert json.dumps(original, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    selected = benchmark.config_for(*args, semantic_transport_schema=transport, repository_source_policy=mode)
    assert selected["agents"][0]["kwargs"] == {**original["agents"][0]["kwargs"], **policy.repository_source_selection(mode)}
    def seal(value):
        return build_controls(value, task_input_sha256={"instruction.md": "a" * 64}, task=benchmark.TASK,
            model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)
    assert seal(original)["configuration_sha256"] != seal(selected)["configuration_sha256"]


@pytest.mark.parametrize("options", [
    {"arm": "no-index"}, {"arm": "full", "provider_profile": GROK_PROFILE},
    {"arm": "full", "source_view_advisory_policy": "source-view-compact@1"},
])
def test_incompatible_prepare_refuses_before_files_or_dispatch(tmp_path, mode, options):
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out", archive=tmp_path / "missing",
                          repository_source_policy=mode, **options)
    assert not (tmp_path / "out").exists()


def test_preparation_and_cli_freeze_selection_without_provider_calls(tmp_path, monkeypatch, mode):
    dataset, _ = authored_task(tmp_path)
    archive = archive_fixture(tmp_path / "archive")
    commands = []
    def dry_run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry_run)
    selected = benchmark.prepare(dataset=dataset, output=tmp_path / "selected", archive=archive,
        arm="full", repository_source_policy=mode)
    assert selected["repository_source_policy"] == mode and selected["provider_calls"] == 0
    assert json.loads((tmp_path / "selected/config.json").read_text())["agents"][0]["kwargs"]["repository_source_policy"] == mode
    assert len(commands) == 1 and commands[0][-1] == "--dry-run"
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(dataset), "--archive", str(archive),
        "--output", str(tmp_path / "next"), "--repository-source-policy", mode])
    benchmark.main()
    assert calls[0]["repository_source_policy"] == mode
    for operation in ("execute", "collect"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--repository-source-policy", mode])
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 2


def test_collect_and_execute_refuse_policy_drift(tmp_path, mode):
    output, prepared, _, _ = prepared_fixture(tmp_path, mode)
    assert benchmark.collect(output)["repository_source_config_unchanged"] is True
    prepared["repository_source_policy"] = "legacy"
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["repository_source_config_unchanged"] is False
    with pytest.raises(ValueError, match="prepared repository source policy selection changed"):
        benchmark.execute(output)


def test_harbor_rejects_unsupported_archive_before_io(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", repository_source_policy=mode)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    with pytest.raises(ValueError):
        asyncio.run(agent.run("Public authored fixture", environment, SimpleNamespace()))
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


def test_harbor_forwards_selection_and_setup_probe_without_claiming_adoption(tmp_path, monkeypatch, mode):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    archive = archive_fixture(tmp_path / "archive")
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={"status": "captured"}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={"status": "captured"}))
    deployment = AsyncMock(return_value={})
    monkeypatch.setattr(adapter, "deploy_supervisor", deployment)
    monkeypatch.setattr(adapter, "deploy_worker_boundary", AsyncMock(return_value={}))
    calls = []
    class Environment:
        async def upload_file(self, source, target):
            pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1, "phases": {},
                "provider_invocations": []} if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(archive), arm="full", repository_source_policy=mode)
    context = AgentContext()
    asyncio.run(agent.setup(Environment()))
    assert deployment.call_args.kwargs["repository_source_policy"] == mode
    asyncio.run(agent.run("Public authored fixture", Environment(), context))
    assert len(calls) == 1
    assert calls[0][calls[0].index("--repository-source-policy") + 1] == mode
    assert context.metadata["repository_source_policy"] == mode
    assert context.metadata["planning_strategy"] == "direct"
    assert context.metadata.get("effective_repository_tools_observed") is not True


def test_container_cli_forwards_flag_and_census_refuses_before_io(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as container
    calls = []
    original = container.run
    monkeypatch.setattr(container, "run", lambda **kwargs: calls.append(kwargs) or
        {"task_completed": False, "arm": "full", "seconds": 0, "provider_invocations": []})
    monkeypatch.setattr(sys, "argv", ["supervisor", "--instruction", str(tmp_path / "missing"),
        "--state", str(tmp_path / "state"), "--arm", "full", "--repository-source-policy", mode])
    assert container.main() == 1
    assert calls[0]["repository_source_policy"] == mode
    with pytest.raises(ValueError, match="executed coding route"):
        original(instruction=tmp_path / "missing", state=tmp_path / "state", arm="full",
            repository_source_policy=mode, metadata_census_only=True,
            semantic_metadata_view="common-bindings@1", coding_reply_mode="ordinary-completion@1")
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("doctor", [{"route": "doctor_candidate"}, {"route": "doctor_contract_candidate"}])
def test_route_builder_refuses_provider_free_paths(mode, doctor):
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    kwargs = dict(router=Path("/router"), model="fixture", reasoning="high", timeout=30,
        semantic_repository=Path("/app"), doctor=doctor)
    with pytest.raises(ValueError, match="semantic model-router coding route"):
        implementation_argv(**kwargs, repository_source_policy=mode)


def worker_namespace(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    return {"argparse": argparse, "pathlib": __import__("pathlib"), "selected_provider": "codex_cli",
            "sys": sys, "artifact": Path("/private/boundary.json"), "digest": "a" * 64}


@pytest.mark.parametrize("extra", [
    ["--purpose", "planning"], ["--preflight"], ["--provider", "grok_cli"],
    ["--semantic-metadata-view", "common-bindings@1", "--semantic-metadata-census-only"],
    ["--doctor-candidate-artifact", "/artifact"], ["--doctor-contract-artifact", "/artifact"],
    ["--finite-repository-artifact", "/artifact"], ["--source-view-advisory-policy", "source-view-compact@1"],
])
def test_generated_worker_guard_refuses_incompatible_dispatch(monkeypatch, mode, extra):
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    source = WORKER_ENTRY.split("parser=argparse.ArgumentParser()", 1)[1].split(
        "from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner", 1)[0]
    namespace = worker_namespace(monkeypatch, ["worker", "--repository-source-policy", mode,
        "--semantic-repository", "/app", *extra])
    with pytest.raises(SystemExit, match="repository source policy requires|provider differs"):
        exec("parser=argparse.ArgumentParser()" + source, namespace)


def test_signed_coding_route_reaches_runtime_with_custody_and_current_workspace(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runtime
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_cli_subreaper as custody
    selected = tmp_path / "allocated"
    selected.mkdir()
    monkeypatch.chdir(selected)
    observed = []
    def main():
        observed.append({"argv": list(sys.argv), "workspace": Path.cwd()})
        return 0
    custody_calls = []
    def run_with_custody(callback):
        custody_calls.append(callback)
        return callback()
    monkeypatch.setattr(runtime, "main", main)
    monkeypatch.setattr(custody, "run_with_child_custody", run_with_custody)
    kwargs = dict(router=Path("/router"), model="fixture", reasoning="high", timeout=30,
        semantic_repository=Path("/app"), doctor={"route": "model_router", "status": "residual"})
    original = implementation_argv(**kwargs)
    assert implementation_argv(**kwargs, repository_source_policy="legacy") == original
    argv = implementation_argv(**kwargs, repository_source_policy=mode)
    namespace = worker_namespace(monkeypatch, argv)
    with pytest.raises(SystemExit) as stopped:
        exec("parser=argparse.ArgumentParser()" + WORKER_ENTRY.split("parser=argparse.ArgumentParser()", 1)[1], namespace)
    assert stopped.value.code == 0 and custody_calls == [main]
    assert len(observed) == 1 and observed[0]["workspace"] == selected
    forwarded = observed[0]["argv"]
    assert forwarded[forwarded.index("--repository-source-policy") + 1] == mode
    assert forwarded[forwarded.index("--semantic-repository") + 1] == "/app"
    assert forwarded[forwarded.index("--container-boundary-sha256") + 1] == "a" * 64
    assert "--source-view-advisory-policy" not in forwarded


@pytest.mark.parametrize("observed", [True, False])
def test_collection_qualifies_observed_tool_reads_not_selected_flag(tmp_path, mode, observed):
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_policy import dispatch_report
    output, _, config, task = prepared_fixture(tmp_path, mode)
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored-trial"
    (trial / "agent").mkdir(parents=True)
    (job / "result.json").write_text("{}")
    (trial / "result.json").write_text(json.dumps({"task_name": task.name,
        "config": {"task": {"path": str(task)}}, "verifier_result": {"rewards": {"reward": 1.0}}}))
    report = dispatch_report(mode)
    if not observed:
        # Initialization and reward cannot substitute for a source read.
        coding = next(row for row in report["provider_invocations"] if row["phase"] == "coding")
        coding["repository_source_policy"]["usage"]["successful_reads"] = 0
        coding["repository_source_policy"]["usage"]["compact_reads"] = 0
    (trial / "agent/supervisor-result.json").write_text(json.dumps(report))
    result = benchmark.collect(output)
    assert result["repository_source_config_unchanged"] is True
    assert result["trials"][0]["reward"] == {"reward": 1.0}
    assert result["trials"][0]["repository_source_dispatch"]["effective_repository_tools_observed"] is observed
    assert result["complete_single_trial_receipt"] is observed


def test_container_driver_routes_only_coding_selection_before_any_provider(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
    state = tmp_path / "state/run"
    state.mkdir(parents=True)
    (state / "admission.json").write_text("{}")
    plans, routed, cleanup = [], [], []
    class ReachedWorkerRoute(RuntimeError):
        pass
    original_builder = doctor.implementation_argv
    def capture_builder(**kwargs):
        routed.append(original_builder(**kwargs))
        raise ReachedWorkerRoute("authored stop before worker execution")
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda *_: None)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver.signal, "signal", lambda *_: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *_: None)
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver.subprocess, "run", lambda argv, **kwargs:
        cleanup.append(argv) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    monkeypatch.setattr(driver.preparation, "initial_context", lambda **kwargs: {})
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: plans.append(kwargs) or {"qualified": True})
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs: {"context_bundle": {}})
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", lambda *_args, **_kwargs:
        {"graph": SimpleNamespace(tasks=[SimpleNamespace(task_cid="authored-task", task_key="authored")])})
    monkeypatch.setattr(doctor, "prepare_terminal_doctor_dispatch", lambda **kwargs:
        {"route": "model_router", "status": "residual"})
    monkeypatch.setattr(doctor, "implementation_argv", capture_builder)
    report = driver.run(instruction=tmp_path / "instruction", state=state, arm="full",
        repository_source_policy=mode, source384_config=tmp_path / "authored-config")
    assert report["error_phase"] == "implementation_setup"
    assert report["error"]["type"] == "ReachedWorkerRoute"
    assert report["repository_source_policy"] == mode
    assert report["provider_invocations"] == []
    assert len(plans) == 1 and "repository_source_policy" not in plans[0]
    assert len(routed) == 1 and routed[0][routed[0].index("--repository-source-policy") + 1] == mode
    assert report["worker_cleanup_returncode"] == 0 and cleanup[0][-1] == "--cleanup"

def test_prepare_rejects_current_inventory_with_unqualified_tar(tmp_path, mode):
    dataset, _ = authored_task(tmp_path)
    archive = archive_fixture(tmp_path / "archive")
    policy.require_repository_source_archive(
        json.loads((archive / "manifest.json").read_text()), mode, archive=archive / "runtime.tar.gz")
    body = b"AUTHORED-NON-TAR"
    (archive / "runtime.tar.gz").write_bytes(body)
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["archive_sha256"] = hashlib.sha256(body).hexdigest()
    (archive / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        benchmark.prepare(dataset=dataset, output=tmp_path / "output", archive=archive,
            arm="full", repository_source_policy=mode)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("method", ["setup", "run"])
def test_harbor_rechecks_actual_archive_before_any_container_operation(tmp_path, mode, method):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    archive = archive_fixture(tmp_path / "archive")
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(archive), arm="full", repository_source_policy=mode)
    policy.require_repository_source_archive(
        json.loads((archive / "manifest.json").read_text()), mode, archive=archive / "runtime.tar.gz")
    (archive / "runtime.tar.gz").write_bytes(b"AUTHORED-CHANGED-AFTER-SELECTION")
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        asyncio.run(agent.setup(environment) if method == "setup" else
                    agent.run("Authored public fixture", environment, SimpleNamespace()))
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()
