"""Frozen coding completion selection, without provider or Docker calls."""
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
from benchmarks.agent_supervisor.container_coding import terminal_coding_reply_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls


MODE = policy.ORDINARY_CODING_REPLY_MODE


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    files = [p for p in Path(policy.__file__).parent.glob("*.py") if not p.name.startswith("test_")]
    files += [root / "ipfs_accelerate_py/agent_supervisor/runtime" / name for name in
              ("coding_reply_contract.py", "router_implementation_runner.py", "semantic_router_translation.py")]
    files += [root / "ipfs_accelerate_py/llm_router.py"]
    files += [root / "ipfs_accelerate_py/cli_runtime/cli_metadata.py"]
    return {"files": [{"path": "source/" + p.relative_to(root).as_posix(),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}


@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
def test_default_is_unchanged_and_explicit_mode_is_config_bound(tmp_path, transport):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    old = benchmark.config_for(*args, semantic_transport_schema=transport)
    assert old == benchmark.config_for(*args, semantic_transport_schema=transport, coding_reply_mode="legacy")
    assert "coding_reply_mode" not in old["agents"][0]["kwargs"]
    new = benchmark.config_for(*args, semantic_transport_schema=transport, coding_reply_mode=MODE)
    assert new["agents"][0]["kwargs"]["coding_reply_mode"] == MODE
    assert "intent_requirement_contract" not in new["agents"][0]["kwargs"]
    def seal(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
                              task=benchmark.TASK, model=benchmark.MODEL,
                              reasoning_effort="high", cli_version=CLI_VERSION)
    assert seal(old)["configuration_sha256"] != seal(new)["configuration_sha256"]


@pytest.mark.parametrize("change", [None, "missing", "modified", "duplicate"])
def test_required_helper_body_is_bound_to_archive(change):
    manifest = inventory()
    target = next(r for r in manifest["files"] if r["path"].endswith("/coding_reply_contract.py"))
    if change == "missing":
        manifest["files"].remove(target)
    elif change == "modified":
        target["sha256"] = "0" * 64
    elif change == "duplicate":
        manifest["files"].append(dict(target))
    if change is None:
        policy.require_coding_reply_archive(manifest, MODE)
    else:
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            policy.require_coding_reply_archive(manifest, MODE)
    policy.require_coding_reply_archive({}, "legacy")


@pytest.mark.parametrize("value", [None, True, {}, "ordinary-completion@2"])
def test_unknown_mode_refused_before_any_preparation_reads(tmp_path, value):
    with pytest.raises(ValueError, match="supported coding reply"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
                          archive=tmp_path / "missing", arm="full", coding_reply_mode=value)


@pytest.mark.parametrize("options", [{"arm": "no-index"}, {"arm": "full", "provider_profile": GROK_PROFILE}])
def test_unqualified_route_refused_before_any_preparation_reads(tmp_path, options):
    with pytest.raises(ValueError, match="ordinary completion requires"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
                          archive=tmp_path / "missing", coding_reply_mode=MODE, **options)


def test_adapter_rejects_missing_archive_support_before_upload(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []}))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", coding_reply_mode=MODE)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        asyncio.run(agent.run("Public fixture", environment, SimpleNamespace()))
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


def test_adapter_forwards_mode_and_keeps_planning(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    manifest = {**inventory(), "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
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
        runtime_archive=str(tmp_path), arm="full", coding_reply_mode=MODE)
    context = AgentContext()
    asyncio.run(agent.run("Public fixture", Environment(), context))
    assert len(calls) == 1
    assert calls[0][calls[0].index("--coding-reply-mode") + 1] == MODE
    assert context.metadata["coding_reply_mode"] == MODE
    assert context.metadata["planning_strategy"] == "direct"


def test_cli_freezes_mode_only_at_prepare(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--coding-reply-mode", MODE])
    benchmark.main()
    assert calls[0]["coding_reply_mode"] == MODE
    assert calls[0]["intent_requirement_contract"] is None
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path), "--coding-reply-mode", MODE])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_route_builder_selects_only_model_coding():
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    kwargs = dict(router=Path("/router"), model="fixture", reasoning="high", timeout=30,
                  semantic_repository=Path("/app"), doctor={"route": "model_router", "status": "residual"})
    old = implementation_argv(**kwargs)
    assert implementation_argv(**kwargs, coding_reply_mode="legacy") == old
    new = implementation_argv(**kwargs, coding_reply_mode=MODE)
    assert new[new.index("--coding-reply-mode") + 1] == MODE
    with pytest.raises(ValueError, match="semantic model-router coding route"):
        implementation_argv(**{**kwargs, "doctor": {"route": "doctor_candidate"}}, coding_reply_mode=MODE)


def test_collect_and_execute_detect_reply_mode_drift(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    for folder in (task, task / "environment", task / "tests"):
        folder.mkdir(parents=True, exist_ok=True)
    for name, text in {"instruction.md": "Fixture", "task.toml": 'version = "1.0"\n',
                       "environment/Dockerfile": "FROM scratch\n", "tests/test.sh": "exit 0\n"}.items():
        (task / name).write_text(text)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", coding_reply_mode=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "coding_reply_mode": MODE}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["coding_reply_config_unchanged"] is True
    prepared["coding_reply_mode"] = "legacy"
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["coding_reply_config_unchanged"] is False
    with pytest.raises(ValueError, match="prepared coding reply selection changed"):
        benchmark.execute(output)
