"""Opt-in admission and forwarding, with real source custody and no providers.

The embedded worker parser/admission segment runs unchanged using standard
library dependencies. Native producer/broker fixtures validate sealed stdin;
these tests make no cross-UID isolation or native token-savings claim.
"""
import argparse
import ast
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pathlib
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding import terminal_semantic_working_set_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create
from test.api.test_semantic_router_integration import allocated, native  # noqa: F401
from test.api.test_worker_source_broker import service  # noqa: F401

MODE = "capsule-token-minimum@1"
OLD = "capsule-references@1"
EXTRA_SOURCES = (
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_working_set_selection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_working_set_dispatch.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/automatic_prompt_tokenizer.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_prompt_tokenizer.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_input_view.py",
    "benchmarks/agent_supervisor/container_coding/automatic_tokenizer_audit.py",
)


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.TOKEN_SOURCE_PATHS]}


def seal(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


def test_selected_config_controls_are_distinct_without_changing_existing_modes(tmp_path):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    base = benchmark.config_for(*args, source_view_profile="identity@1")
    old = benchmark.config_for(*args, source_view_profile="identity@1", semantic_working_set=OLD)
    selected = benchmark.config_for(*args, source_view_profile="identity@1", semantic_working_set=MODE)
    assert "semantic_working_set" not in base["agents"][0]["kwargs"]
    assert selected["agents"][0]["kwargs"] == {**old["agents"][0]["kwargs"], "semantic_working_set": MODE}
    observation = lambda config: dict(status="observed", configuration_unchanged=True, declared=seal(config))
    record = seal(selected)
    assert record["schema"] == controls.SEMANTIC_WORKING_SET_SCHEMA
    assert controls.validate_controls(record) is True
    assert controls.compare_controls(observation(selected), observation(old))["differences"] == [
        "experimental_controls_sha256.semantic_working_set"]
    assert policy.semantic_working_set_observation(MODE) == {
        "semantic_working_set": MODE, "semantic_working_set_proof_authority": False,
        "semantic_working_set_token_savings_claimed": False}
    assert policy.semantic_working_set_selection("legacy") == {}
    policy.require_semantic_working_set_archive(None, "legacy")


@pytest.mark.parametrize("change", [dict(arm="no-index"), dict(source_view_profile="disabled"),
    dict(semantic_transport_schema="supervisor-semantic-router-input@2"),
    dict(semantic_metadata_view="common-bindings@1"), dict(semantic_working_set=True),
    dict(semantic_working_set="capsule-token-minimum@2")])
def test_invalid_route_refuses_before_preparation_filesystem(tmp_path, change):
    options = dict(arm="full", source_view_profile="identity@1", semantic_working_set=MODE)
    options.update(change)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=tmp_path / "missing", archive=tmp_path / "missing",
            output=tmp_path / "output", **options)
    assert not (tmp_path / "output").exists()


def test_old_archive_subset_remains_valid_only_for_old_mode():
    manifest = inventory()
    assert policy.TOKEN_SOURCE_PATHS == policy.SOURCE_PATHS + EXTRA_SOURCES
    policy.require_semantic_working_set_archive(manifest, MODE)
    manifest["files"] = [row for row in manifest["files"]
        if row["path"] not in {"source/" + name for name in EXTRA_SOURCES}]
    policy.require_semantic_working_set_archive(manifest, OLD)
    with pytest.raises(ValueError, match="newly qualified runtime"):
        policy.require_semantic_working_set_archive(manifest, MODE)


@pytest.mark.parametrize("name", EXTRA_SOURCES)
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "sha", "size"])
def test_new_mode_pins_each_actual_counting_and_dispatch_dependency(name, mutation):
    manifest = inventory()
    row = next(row for row in manifest["files"] if row["path"] == "source/" + name)
    if mutation == "missing": manifest["files"].remove(row)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "sha": row["sha256"] = "0" * 64
    else: row["bytes"] += 1
    # Added mode-specific dependencies do not silently change the old gate.
    policy.require_semantic_working_set_archive(manifest, OLD)
    with pytest.raises(ValueError, match="newly qualified runtime"):
        policy.require_semantic_working_set_archive(manifest, MODE)


def test_harbor_forwards_selected_flag_once_and_retains_declaration(tmp_path, monkeypatch):
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
                "provider_invocations": [], **policy.semantic_working_set_observation(MODE)}
                if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile="identity@1", semantic_working_set=MODE)
    context = AgentContext()
    asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1 and calls[0].count("--semantic-working-set") == 1
    assert calls[0][calls[0].index("--semantic-working-set") + 1] == MODE
    assert context.metadata["semantic_working_set_runtime_declared_matches"] is True
    assert context.metadata["semantic_working_set_token_savings_claimed"] is False


def test_benchmark_and_driver_cli_forward_new_mode(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path), "--output", str(tmp_path / "out"), "--source-view-profile", "identity@1",
        "--semantic-working-set", MODE])
    benchmark.main()
    assert calls[-1]["semantic_working_set"] == MODE
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": False, "arm": "full", "seconds": 0, "provider_invocations": []})
    monkeypatch.setattr(sys, "argv", ["driver", "--instruction", str(tmp_path / "instruction"),
        "--state", str(tmp_path / "state"), "--arm", "full", "--source-view-profile", "identity@1",
        "--semantic-working-set", MODE])
    assert driver.main() == 1 and calls[-1]["semantic_working_set"] == MODE


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
def test_actual_driver_preserves_provider_free_nonapplication(tmp_path, monkeypatch, route):
    from test.api.test_terminal_doctor_dispatch import (
        test_driver_selects_before_owner_and_preserves_provider_accounting as exercise_driver,
    )
    original = dispatch.implementation_argv
    calls = []
    def capture(**kwargs):
        argv = original(**kwargs)
        calls.append((kwargs, argv))
        return argv
    monkeypatch.setattr(dispatch, "implementation_argv", capture)
    exercise_driver(tmp_path, monkeypatch, "full", route, False, "supervisor-semantic-router-input@1",
        controls={"source_view_profile": "python-comments@1", "semantic_working_set": MODE})
    assert len(calls) == 1 and calls[0][0]["semantic_working_set"] == MODE
    argv = calls[0][1]
    assert argv.count("--semantic-working-set") == (1 if route == "model_router" else 0)
    if route == "model_router": assert argv[argv.index("--semantic-working-set") + 1] == MODE


def _worker_arguments(monkeypatch, argv):
    # Run the real argparse plus every pre-router option guard, without replacing
    # imported runtime packages or claiming container admission was exercised.
    tree = ast.parse(WORKER_ENTRY)
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    end = next(i for i, node in enumerate(tree.body[start:], start) if isinstance(node, ast.ImportFrom)
        and node.module == "ipfs_accelerate_py.agent_supervisor.runtime")
    module = ast.Module(body=tree.body[start:end], type_ignores=[])
    namespace = dict(argparse=argparse, pathlib=pathlib, selected_provider="grok_cli")
    monkeypatch.setattr(sys, "argv", ["worker", *argv])
    exec(compile(module, "<deployed-worker-option-admission>", "exec"), namespace)
    return namespace["args"]


def _source_args():
    return ["--semantic-repository", "/app", "--source-view-profile", "identity@1",
        "--source-view-descriptor", "/owner/source.json", "--source-view-descriptor-sha256", "a" * 64,
        "--semantic-working-set", MODE]


def test_deployed_worker_parser_admits_new_mode_with_exact_owner_pins(monkeypatch):
    args = _worker_arguments(monkeypatch, _source_args())
    assert args.semantic_working_set == MODE
    assert args.source_view_descriptor == Path("/owner/source.json")
    assert args.source_view_descriptor_sha256 == "a" * 64
    assert args.semantic_transport_schema == "supervisor-semantic-router-input@1"
    assert args.semantic_metadata_view == "legacy"


@pytest.mark.parametrize("change", ["missing_digest", "missing_repository", "disabled", "planning",
    "preflight", "metadata", "transport", "census"])
def test_deployed_worker_rejects_incompatible_selection_before_router(monkeypatch, change):
    argv = _source_args()
    if change in ("missing_digest", "missing_repository"):
        flag = "--source-view-descriptor-sha256" if change == "missing_digest" else "--semantic-repository"
        index = argv.index(flag); del argv[index:index + 2]
    elif change == "disabled": argv[argv.index("--source-view-profile") + 1] = "disabled"
    else:
        argv += {"planning": ["--purpose", "planning"], "preflight": ["--preflight"],
            "metadata": ["--semantic-metadata-view", "common-bindings@1"],
            "transport": ["--semantic-transport-schema", "supervisor-semantic-router-input@2"],
            "census": ["--semantic-metadata-census-only"]}[change]
    with pytest.raises(SystemExit): _worker_arguments(monkeypatch, argv)


@pytest.mark.parametrize("mode", [OLD, MODE])
def test_owner_broker_keeps_selected_capsule_protocol_and_sealed_stdin(service, mode):
    fd = _validation_memfd_create("token-working-set", getattr(os, "MFD_ALLOW_SEALING", 2)
        | getattr(os, "MFD_CLOEXEC", 1))
    value = None
    try:
        raw = service.prompt.encode("utf-8")
        os.write(fd, raw); os.lseek(fd, 7, os.SEEK_SET)
        argv = ["--purpose", "coding", "--semantic-repository", str(service.repository),
            "--source-view-profile", "python-comments@1", "--semantic-working-set", mode, "--timeout", "60"]
        with pytest.raises(ValueError, match="sealed owner prompt"):
            broker.prepare_worker_source_broker(argv=argv, boundary=service.boundary,
                workspace=service.workspace, input_fd=fd)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 15)
        value = broker.prepare_worker_source_broker(argv=argv, boundary=service.boundary,
            workspace=service.workspace, input_fd=fd)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 7
        assert os.pread(fd, len(raw), 0) == raw
        assert value.prompt == service.prompt
        assert value.semantic_evidence_mode == value.descriptor["semantic_evidence_mode"] == "selected-capsules@1"
        assert value.descriptor["schema"] == "supervisor-worker-source-descriptor@4"
        assert value.descriptor["native_prompt_sha256"] == service.descriptor["native_prompt_sha256"]
    finally:
        if value is not None: value.close()
        os.close(fd)


@pytest.mark.parametrize("argv", [["--semantic-working-set", MODE],
    ["--semantic-working-set", MODE, "--source-view-profile", "identity@1"],
    ["--semantic-working-set", MODE, "--source-view-profile", "identity@1", "--purpose", "planning"]])
def test_owner_refuses_unbound_selection_before_reading_stdin(monkeypatch, argv):
    monkeypatch.setattr(broker.os, "pread", lambda *args: pytest.fail("invalid route read sealed prompt"))
    with pytest.raises(ValueError):
        broker.prepare_worker_source_broker(argv=argv, boundary="/missing", workspace="/missing", input_fd=-1)
