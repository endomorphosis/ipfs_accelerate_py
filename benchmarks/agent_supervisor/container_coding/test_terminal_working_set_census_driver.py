"""Authored census routing controls; no model, provider or container executes.

The actual driver branch is exercised with explicit preparation/worker doubles.
Worker tests execute the shipped parser and routing AST after the separately
qualified UID/namespace setup; they do not execute that privileged setup.
"""
import argparse
import ast
import builtins
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

# Real preparation loads this module before dispatch. The authored preparation
# doubles below skip that step; load it before installing the subprocess guard
# so optional-library initialization is not mistaken for model execution.
import ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context

import benchmarks.agent_supervisor.container_coding.container_worker_deployment as worker
import benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent as harbor
import benchmarks.agent_supervisor.container_coding.terminal_container_supervisor as driver
import benchmarks.agent_supervisor.container_coding.terminal_live_census_provenance as provenance
import benchmarks.agent_supervisor.container_coding.terminal_live_metadata_census as live
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import (
    CODEX_PROFILE, GROK_PROFILE, resolve_provider_profile,
)


def _forbidden(*args, **kwargs):
    pytest.fail("census must not enter execution")


def _census_options(root):
    return dict(instruction=root/"unread-instruction", state=root/"state/run", arm="full",
        provider_profile=GROK_PROFILE, planning_mode="declared-contract@1",
        working_set_census_only=True, semantic_working_set="capsule-references@1",
        source_view_profile="identity@1", grok_coding_max_turns=8,
        source384_config=root/"unread-source384-config")


@pytest.mark.parametrize("cleanup_code", [0, 1])
def test_declared_driver_census_stops_before_coding_and_revokes_success_on_cleanup_failure(
        tmp_path, monkeypatch, cleanup_code):
    root = tmp_path / "container"
    state = root / "state/run"
    state.mkdir(parents=True)
    (state / "admission.json").write_text("{}")
    calls, cleanups, alarms, imports = [], [], [], []

    def stage(name, result):
        def invoke(**kwargs):
            calls.append(name)
            if name == "initial_context":
                assert kwargs["train_autoencoder"] is False
                assert kwargs["source384_config"] == root / "unread-source384-config"
            if name == "planning":
                assert callable(kwargs["provider_callable"])
            return result
        return invoke

    def admitted(*args, **kwargs):
        assert calls == ["prepare", "initial_context", "planning", "context"]
        calls.append("admission")
        return {"graph": SimpleNamespace(tasks=[SimpleNamespace(task_cid="task", task_key="key")])}

    def census(**kwargs):
        assert calls == ["prepare", "initial_context", "planning", "context", "admission"]
        assert kwargs["state"] == state
        assert kwargs["provider_selection"] == resolve_provider_profile(GROK_PROFILE)
        assert 0 < kwargs["timeout_seconds"] <= 120
        assert kwargs["source_view_profile"] == "identity@1"
        assert kwargs["grok_coding_max_turns"] == 8
        calls.append("census")
        kwargs["report"].update(working_set_census_succeeded=True,
            working_set_census={"provider_calls": 0, "router_calls": 0}, coding_dispatch_possible=False)

    def cleanup(argv, **kwargs):
        assert argv[-1] == "--cleanup"
        assert kwargs["cwd"] == "/"
        cleanups.append(argv)
        return SimpleNamespace(returncode=cleanup_code)

    forbidden_imports = {
        "benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch",
        "benchmarks.agent_supervisor.container_coding.native_quack_qualification",
        "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime",
        "ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy",
    }
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name in forbidden_imports:
            imports.append(name)
            _forbidden()
        return original_import(name, *args, **kwargs)

    failure = ModuleType("benchmarks.agent_supervisor.container_coding.terminal_failure_observation")
    failure.collect = lambda *args, **kwargs: {"authored_observation": True}
    monkeypatch.setitem(sys.modules, failure.__name__, failure)
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "WORKTREES", root / "worktrees")
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda value: 0)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: alarms.append(args))
    monkeypatch.setattr(driver, "NativeProgress", lambda **kwargs: SimpleNamespace(
        report={"authored_control": True}, finish=lambda report: None))
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(provenance, "audit_planning_usage", lambda planning, receipts: {
        "authored_control": True, "planner_callable_attempts": planning["provider_calls"],
        "complete_total_tokens": None})
    monkeypatch.setattr(driver.preparation, "prepare", stage("prepare", {"intent_preplanning": {}}))
    monkeypatch.setattr(driver.preparation, "initial_context", stage("initial_context", {}))
    monkeypatch.setattr(driver.preparation, "plan", stage("planning", {"qualified": True, "provider_calls": 0}))
    monkeypatch.setattr(driver.preparation, "context", stage("context", {"context_bundle": {}}))
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", admitted)
    monkeypatch.setattr(driver, "_run_working_set_census", census)
    monkeypatch.setattr(driver.subprocess, "run", cleanup)
    monkeypatch.setattr(driver.subprocess, "Popen", _forbidden)
    monkeypatch.setattr(builtins, "__import__", import_guard)
    report = driver.run(**_census_options(root))
    assert calls == ["prepare", "initial_context", "planning", "context", "admission", "census"]
    assert imports == [] and len(cleanups) == 1
    assert report["working_set_census_succeeded"] is (cleanup_code == 0)
    assert report["worker_cleanup_returncode"] == cleanup_code
    assert report["task_completed"] is False and report["official_reward"] is None
    assert report["official_verifier_executed"] is report["coding_dispatch_possible"] is False
    assert report["provider_invocations"] == [] and "unreceipted_provider_attempt" not in report
    assert "start" not in report and "doctor_dispatch" not in report
    assert alarms[-1] == (driver.signal.ITIMER_REAL, 0)
    assert json.loads((state.parent / "run-result.json").read_text()) == report


@pytest.mark.parametrize("change", [
    {"working_set_census_only": 1}, {"working_set_census_only": "true"},
    {"planning_mode": "model"}, {"metadata_census_only": True},
    {"semantic_working_set": "legacy"}, {"source_view_profile": "disabled"},
    {"semantic_metadata_view": "common-bindings@1"}, {"source_edit_mode": "mapped-span@1"},
    {"arm": "no-index"}, {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
])
def test_declared_census_invalid_controls_refuse_before_reads(tmp_path, monkeypatch, change):
    for name in ("read_text", "read_bytes", "open", "mkdir"):
        monkeypatch.setattr(Path, name, _forbidden)
    monkeypatch.setattr(driver.os, "geteuid", _forbidden)
    monkeypatch.setattr(driver.preparation, "prepare", _forbidden)
    monkeypatch.setattr(driver.subprocess, "run", _forbidden)
    with pytest.raises(ValueError):
        driver.run(**(_census_options(tmp_path) | change))


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_current_owner_selects_pack_census_for_worker_and_replay_before_success(
        tmp_path, monkeypatch, cleanup_fails):
    from benchmarks.agent_supervisor.container_coding import terminal_working_set_census as preparation
    from benchmarks.agent_supervisor.container_coding import terminal_working_set_census_observation as replay
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as protocol
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_working_set_census import PACK_SCHEMA

    assert PACK_SCHEMA == "router-semantic-working-set-census@3"
    root = tmp_path / "container"
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    private = root / "private"
    private.mkdir()
    prompt = private / "prompt.txt"
    prompt.write_text("Authored owner prompt")
    planning = {"qualified": True, "provider_calls": 0}
    report = {"planning": planning, "provider_invocations": [], "working_set_census_succeeded": False}
    metadata = {"planning_metadata_sha256": live._identity(planning)}
    source_observation = {"complete_input_replayed": True, "closed_observation_replayed": True}
    observed = {"schema": PACK_SCHEMA, "provider_calls": 0}
    events = []
    prepared = dict(metadata=metadata, runner_kwargs={"semantic_repository": root / "repository"},
        workspace=workspace, prompt_artifact=prompt, private_artifact_dir=private)
    deployment = {"authored_deployment": True}

    @contextmanager
    def prepare(**kwargs):
        assert kwargs["coding_reply_mode"] == "legacy"
        assert kwargs["require_source384"] is False
        events.append("prepare")
        try:
            yield prepared
        finally:
            assert report["working_set_census_succeeded"] is False
            events.append("cleanup")
            if cleanup_fails:
                raise ValueError("authored cleanup failure")

    def worker_call(argv, **kwargs):
        assert argv.count("--semantic-working-set-census-only") == 1
        flag = "--semantic-working-set-census-schema"
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == PACK_SCHEMA
        assert argv[argv.index("--grok-coding-max-turns") + 1] == "8"
        assert kwargs["stdin"].read() == prompt.read_bytes()
        events.append("worker")
        return SimpleNamespace(returncode=0, stdout=protocol.canonical_json(observed), stderr=b"")

    def owner_replay(**kwargs):
        assert kwargs["prepared"] is prepared
        assert kwargs["observed"] == observed
        assert kwargs["deployment"] == deployment
        assert kwargs["census_schema"] == PACK_SCHEMA
        assert report["working_set_census_succeeded"] is False
        events.append("replay")
        return source_observation

    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "WORKTREES", root / "worktrees")
    monkeypatch.setattr(driver, "_live_census_boundary", lambda path: deployment)
    monkeypatch.setattr(preparation, "prepare_terminal_working_set_census", prepare)
    monkeypatch.setattr(replay, "replay_working_set_census", owner_replay)
    monkeypatch.setattr(driver.subprocess, "run", worker_call)
    args = dict(state=root / "state", report=report, provider_selection=resolve_provider_profile(GROK_PROFILE),
        timeout_seconds=120, source_view_profile="identity@1", source_view_policy=None,
        coding_reply_mode="legacy", grok_coding_max_turns=8, require_source384=False)
    if cleanup_fails:
        with pytest.raises(ValueError, match="authored cleanup failure"):
            driver._run_working_set_census(**args)
        assert report["working_set_census_succeeded"] is False
        assert "working_set_census" not in report
    else:
        driver._run_working_set_census(**args)
        assert report["working_set_census_succeeded"] is True
        assert report["working_set_census"] == observed
        assert report["working_set_census_source_observation"] == source_observation
        assert report["working_set_census_context"]["retained_owner_planning_receipt_join_verified"] is True
        assert report["coding_dispatch_possible"] is False
    assert events == ["prepare", "worker", "replay", "cleanup"]
