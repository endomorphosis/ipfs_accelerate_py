"""Frozen source input preview selection, without provider or Docker calls."""
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
from benchmarks.agent_supervisor.container_coding import terminal_source_input_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest


@pytest.fixture(params=policy.SOURCE_INPUT_POLICIES[1:])
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
    assert old == benchmark.config_for(*args, semantic_transport_schema=transport, source_input_policy="legacy")
    assert "source_input_policy" not in old["agents"][0]["kwargs"]
    new = benchmark.config_for(*args, semantic_transport_schema=transport, source_input_policy=mode)
    assert new["agents"][0]["kwargs"]["source_input_policy"] == mode
    assert "intent_requirement_contract" not in new["agents"][0]["kwargs"]
    def seal(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
                              task=benchmark.TASK, model=benchmark.MODEL,
                              reasoning_effort="high", cli_version=CLI_VERSION)
    assert seal(old)["configuration_sha256"] != seal(new)["configuration_sha256"]


@pytest.mark.parametrize("change", [None, "missing", "modified", "duplicate", "size", "boolean_size"])
def test_required_helper_body_is_bound_to_archive(change, mode):
    manifest = inventory()
    target = next(r for r in manifest["files"] if r["path"].endswith("/source_input_policy.py"))
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
        policy.require_source_input_archive(manifest, mode)
    else:
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            policy.require_source_input_archive(manifest, mode)
    policy.require_source_input_archive({}, "legacy")


@pytest.mark.parametrize("value", [None, True, {}, "source-input-foreign@1"])
def test_unknown_mode_refused_before_any_preparation_reads(tmp_path, value):
    with pytest.raises(ValueError, match="supported source input policy"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
                          archive=tmp_path / "missing", arm="full", source_input_policy=value)


@pytest.mark.parametrize("options", [{"arm": "no-index"}, {"arm": "full", "provider_profile": GROK_PROFILE}])
def test_unqualified_route_refused_before_any_preparation_reads(tmp_path, options, mode):
    with pytest.raises(ValueError, match="source input policy requires"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
                          archive=tmp_path / "missing", source_input_policy=mode, **options)


def test_adapter_rejects_missing_archive_support_before_upload(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_input_policy=mode)
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
        runtime_archive=str(tmp_path), arm="full", source_input_policy=mode)
    context = AgentContext()
    asyncio.run(agent.run("Public fixture", Environment(), context))
    assert len(calls) == 1
    assert calls[0][calls[0].index("--source-input-policy") + 1] == mode
    assert context.metadata["source_input_policy"] == mode
    assert context.metadata["planning_strategy"] == "direct"


def test_cli_freezes_mode_only_at_prepare(tmp_path, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--source-input-policy", mode])
    benchmark.main()
    assert calls[0]["source_input_policy"] == mode
    assert calls[0]["intent_requirement_contract"] is None
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path), "--source-input-policy", mode])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_route_builder_selects_only_model_coding(mode):
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    kwargs = dict(router=Path("/router"), model="fixture", reasoning="high", timeout=30,
                  semantic_repository=Path("/app"), doctor={"route": "model_router", "status": "residual"})
    old = implementation_argv(**kwargs)
    assert implementation_argv(**kwargs, source_input_policy="legacy") == old
    new = implementation_argv(**kwargs, source_input_policy=mode)
    assert new[new.index("--source-input-policy") + 1] == mode
    with pytest.raises(ValueError, match="semantic model-router coding route"):
        implementation_argv(**{**kwargs, "doctor": {"route": "doctor_candidate"}}, source_input_policy=mode)


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
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", source_input_policy=mode)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "source_input_policy": mode}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["source_input_config_unchanged"] is True
    prepared["source_input_policy"] = "legacy"
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["source_input_config_unchanged"] is False
    with pytest.raises(ValueError, match="prepared source input policy selection changed"):
        benchmark.execute(output)


@pytest.mark.parametrize("relative", policy.SOURCE_PATHS)
def test_every_selected_runtime_body_is_archive_bound(relative, mode):
    manifest = inventory()
    row = next(item for item in manifest["files"] if item["path"] == "source/" + relative)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_source_input_archive(manifest, mode)


@pytest.mark.parametrize("manifest", [None, True, {"files": True}, {"files": {}}, {"files": "foreign"}])
def test_malformed_archive_inventory_rejected(manifest, mode):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_source_input_archive(manifest, mode)


def test_prepare_freezes_output_policy_without_provider_calls(tmp_path, monkeypatch, mode):
    from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    raw = b"AUTHORED-no-executable-archive"
    (archive / "runtime.tar.gz").write_bytes(raw)
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "codex_version": CLI_VERSION})))
    commands = []
    def dry_run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry_run)
    selected = benchmark.prepare(dataset=dataset, output=tmp_path / "selected", archive=archive,
                                 arm="full", source_input_policy=mode)
    assert selected["source_input_policy"] == mode
    assert selected["provider_calls"] == 0
    assert json.loads((tmp_path / "selected/config.json").read_text())["agents"][0]["kwargs"]["source_input_policy"] == mode
    assert len(commands) == 1 and commands[0][-1] == "--dry-run"
    legacy = benchmark.prepare(dataset=dataset, output=tmp_path / "legacy", archive=archive, arm="full")
    assert "source_input_policy" not in legacy
    assert "source_input_policy" not in json.loads((tmp_path / "legacy/config.json").read_text())["agents"][0]["kwargs"]


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
                 "sys": SimpleNamespace(argv=["worker", "--source-input-policy", mode,
                    "--semantic-repository", "/app", *extra])}
    monkeypatch.setattr(sys, "argv", namespace["sys"].argv)
    with pytest.raises(SystemExit, match="source input policy requires|provider differs"):
        exec("parser=argparse.ArgumentParser()" + source, namespace)


def test_worker_guard_accepts_selected_semantic_coding(monkeypatch, mode):
    import argparse
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    source = WORKER_ENTRY.split("parser=argparse.ArgumentParser()", 1)[1].split(
        "from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner", 1)[0]
    namespace = {"argparse": argparse, "pathlib": __import__("pathlib"), "selected_provider": "codex_cli",
                 "sys": SimpleNamespace(argv=["worker", "--source-input-policy", mode,
                    "--semantic-repository", "/app"])}
    monkeypatch.setattr(sys, "argv", namespace["sys"].argv)
    exec("parser=argparse.ArgumentParser()" + source, namespace)
    assert namespace["args"].purpose == "coding"
    assert namespace["args"].source_input_policy == mode


def test_adapter_refuses_observation_only_census_combination(tmp_path, mode):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    with pytest.raises(ValueError, match="executed coding route|AST reference v1 excludes metadata"):
        FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
            runtime_archive=str(tmp_path), arm="full", source_input_policy=mode,
            metadata_census_only=True, semantic_metadata_view="common-bindings@1",
            coding_reply_mode="ordinary-completion@1")



@pytest.mark.parametrize('selected', [policy.AST_RAW_SOURCE_INPUT_POLICY, policy.AST_COMPACT_SOURCE_INPUT_POLICY])
def test_ast_profile_refuses_metadata_before_preparation(tmp_path, selected):
    with pytest.raises(ValueError, match='AST reference v1 excludes metadata'):
        benchmark.prepare(dataset=tmp_path / 'missing', output=tmp_path / 'out', archive=tmp_path / 'missing',
            arm='full', source_input_policy=selected, semantic_metadata_view='common-bindings@1')


def test_four_policies_have_distinct_controls_with_same_nonpresentation_settings(tmp_path):
    configs = [benchmark.config_for(tmp_path, tmp_path / 'out', tmp_path / 'archive', 'full',
        source_input_policy=selected) for selected in policy.SOURCE_INPUT_POLICIES[1:]]
    receipts = [build_controls(config, task_input_sha256={'instruction.md': 'a' * 64}, task=benchmark.TASK,
        model=benchmark.MODEL, reasoning_effort='high', cli_version=CLI_VERSION) for config in configs]
    assert len({receipt['configuration_sha256'] for receipt in receipts}) == 4
    for config in configs:
        config['agents'][0]['kwargs'].pop('source_input_policy')
    assert all(config == configs[0] for config in configs)
