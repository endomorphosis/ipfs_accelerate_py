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

import benchmarks.agent_supervisor.container_coding.container_worker_deployment as worker
import benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent as harbor
import benchmarks.agent_supervisor.container_coding.terminal_container_supervisor as driver
import benchmarks.agent_supervisor.container_coding.terminal_live_census_provenance as provenance
import benchmarks.agent_supervisor.container_coding.terminal_live_metadata_census as live
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import (
    CODEX_PROFILE, GROK_PROFILE, resolve_provider_profile,
)


def _forbidden(*args, **kwargs):
    pytest.fail("authored control must not reach input reads or execution")


def _census_options(tmp_path):
    return dict(instruction=tmp_path / "unread-instruction", state=tmp_path / "state/run",
        arm="full", provider_profile=CODEX_PROFILE, metadata_census_only=True,
        semantic_metadata_view="common-bindings@1", coding_reply_mode="ordinary-completion@1",
        source384_config=tmp_path / "unread-source384-config")


@pytest.mark.parametrize("change", [
    {"metadata_census_only": "true"}, {"source384_config": None},
    {"semantic_metadata_view": "legacy"}, {"coding_reply_mode": "legacy"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"provider_profile": GROK_PROFILE}, {"arm": "no-index"},
])
def test_driver_invalid_census_selection_refuses_before_reads(tmp_path, monkeypatch, change):
    options = {**_census_options(tmp_path), **change}
    with monkeypatch.context() as guard:
        for name in ("read_text", "read_bytes", "open", "mkdir"):
            guard.setattr(Path, name, _forbidden)
        guard.setattr(driver.os, "geteuid", _forbidden)
        guard.setattr(driver.preparation, "prepare", _forbidden)
        guard.setattr(driver.subprocess, "run", _forbidden)
        guard.setattr(driver.subprocess, "Popen", _forbidden)
        with pytest.raises(ValueError):
            driver.run(**options)


@pytest.mark.parametrize("cleanup_code", [0, 1])
def test_actual_driver_census_branch_retains_stages_and_skips_active_execution(
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
        assert kwargs["provider_selection"] == resolve_provider_profile(CODEX_PROFILE)
        assert 0 < kwargs["timeout_seconds"] <= 120
        calls.append("census")
        kwargs["report"].update(metadata_census_succeeded=True,
            metadata_census={"provider_calls": 0, "router_calls": 0}, coding_dispatch_possible=False)

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
    monkeypatch.setattr(driver.preparation, "plan", stage("planning", {"qualified": True, "provider_calls": 1}))
    monkeypatch.setattr(driver.preparation, "context", stage("context", {"context_bundle": {}}))
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", admitted)
    monkeypatch.setattr(driver, "_run_live_metadata_census", census)
    monkeypatch.setattr(driver.subprocess, "run", cleanup)
    monkeypatch.setattr(driver.subprocess, "Popen", _forbidden)
    monkeypatch.setattr(builtins, "__import__", import_guard)
    report = driver.run(**_census_options(root))
    assert calls == ["prepare", "initial_context", "planning", "context", "admission", "census"]
    assert imports == [] and len(cleanups) == 1
    assert report["metadata_census_succeeded"] is (cleanup_code == 0)
    assert report["worker_cleanup_returncode"] == cleanup_code
    assert report["task_completed"] is False and report["official_reward"] is None
    assert report["official_verifier_executed"] is report["coding_dispatch_possible"] is False
    assert report["provider_invocations"] == [] and "unreceipted_provider_attempt" not in report
    assert "start" not in report and "doctor_dispatch" not in report
    assert report["planning_usage_audit"]["planner_callable_attempts"] == 1
    assert report["planning_usage_audit"]["complete_total_tokens"] is None
    assert alarms[-1] == (driver.signal.ITIMER_REAL, 0)
    assert json.loads((state.parent / "run-result.json").read_text()) == report


@pytest.fixture
def authored_isolated_census(tmp_path, monkeypatch):
    state = tmp_path / "state"
    private = state / "live-metadata-census"
    private.mkdir(parents=True)
    prompt = private / "worker-prompt.txt"
    prompt.write_bytes(b"authored native task context")
    root, workspace = tmp_path / "app", tmp_path / "worktrees/census"
    workspace.mkdir(parents=True)
    boundary_sha = "1" * 64
    prompt_sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
    metadata = {"native_prompt_sha256": prompt_sha,
                "post_yield_admission_task_revalidated": False, "allocated_worktree_removed": False}
    options = {"semantic_repository": root, "semantic_metadata_view": "common-bindings@1",
               "coding_reply_mode": "ordinary-completion@1"}
    for prefix in ("doctor_residual", "public_instruction"):
        options.update({prefix + "_artifact": root / (prefix + ".json"),
                        prefix + "_sha256": "2" * 64, prefix + "_task_cid": "authored-task-cid"})
    prepared = {"runner_kwargs": options, "workspace": workspace,
                "prompt_artifact": prompt, "metadata": metadata}
    report = {"planning": {"qualified": True}, "provider_invocations": [],
              "metadata_census_succeeded": False}
    metadata["planning_metadata_sha256"] = live._identity(report["planning"])
    selected = resolve_provider_profile(CODEX_PROFILE)
    receipt = {"schema": "router-semantic-metadata-census@1", "native_prompt_sha256": prompt_sha,
        "container_boundary_checked": True, "container_boundary_artifact_sha256": boundary_sha,
        "provider": selected["provider"], "model_sha256": hashlib.sha256(selected["model"].encode()).hexdigest(),
        "reasoning_effort": "high", "coding_reply_mode": "ordinary-completion@1",
        "provider_calls": 0, "router_calls": 0, "provider_dispatch_observed": False,
        "provider_token_usage": None, "doctor_residual_included": True, "public_instruction_included": True}
    control = SimpleNamespace(state=state, prepared=prepared, report=report, selected=selected,
        receipt=receipt, fail_after_yield=False, worker_returncode=0, calls=[], events=[],
        helper_calls=[], worker_inputs=[], formal_status_owner=None, provenance={
            "worker_boundary_manifest_sha256": boundary_sha, "authored_control": True})

    @contextmanager
    def native_helper(**kwargs):
        expected = {"state": state, "worktree_root": driver.WORKTREES,
                    "worker_writable": True, "worker_group_gid": driver.os.getegid()}
        if control.formal_status_owner is not None:
            expected.update(include_formal_status_census=True,
                            formal_status_owner=control.formal_status_owner)
        assert kwargs == expected
        control.helper_calls.append(kwargs)
        control.events.append("yield")
        yield prepared
        assert report["metadata_census_succeeded"] is False
        assert "metadata_census" not in report and "live_planning_provenance" not in report
        control.events.append("recheck")
        if control.fail_after_yield:
            raise ValueError("authored source drift after worker receipt")
        metadata.update(post_yield_admission_task_revalidated=True, allocated_worktree_removed=True)

    def validate(**kwargs):
        assert kwargs["planning"] is report["planning"]
        assert kwargs["invocations"] is report["provider_invocations"]
        assert kwargs["selected_provider"] == selected
        assert kwargs["deployment_boundary"] == {"authored_deployment": True}
        return control.provenance

    def worker_call(argv, **kwargs):
        assert argv[0] == str(driver.ROUTER)
        assert kwargs["cwd"] == workspace and kwargs["timeout"] == 60
        prompt_bytes = kwargs["stdin"].read()
        assert prompt_bytes == b"authored native task context"
        control.worker_inputs.append(prompt_bytes)
        assert "--semantic-metadata-census-only" in argv
        for flag, value in (("--semantic-repository", str(root)),
                ("--semantic-metadata-view", "common-bindings@1"),
                ("--coding-reply-mode", "ordinary-completion@1"), ("--purpose", "coding")):
            assert argv[argv.index(flag) + 1] == value
        for prefix in ("doctor_residual", "public_instruction"):
            for suffix in ("artifact", "sha256", "task_cid"):
                flag = "--" + (prefix + "_" + suffix).replace("_", "-")
                assert argv[argv.index(flag) + 1] == str(options[prefix + "_" + suffix])
        control.calls.append(argv)
        return SimpleNamespace(returncode=control.worker_returncode,
            stdout=json.dumps(control.receipt).encode(), stderr=b"")

    monkeypatch.setattr(live, "prepare_terminal_live_metadata_census", native_helper)
    monkeypatch.setattr(provenance, "validate_live_planning_provenance", validate)
    monkeypatch.setattr(driver, "_live_census_boundary", lambda root: {"authored_deployment": True})
    monkeypatch.setattr(driver.subprocess, "run", worker_call)
    monkeypatch.setattr(driver.subprocess, "Popen", _forbidden)
    return control


def _run_authored_census(control, **kwargs):
    driver._run_live_metadata_census(state=control.state, report=control.report,
        provider_selection=control.selected, timeout_seconds=60, **kwargs)


def test_census_success_is_published_only_after_native_exit_rechecks(authored_isolated_census):
    control = authored_isolated_census
    _run_authored_census(control)
    assert control.events == ["yield", "recheck"] and len(control.calls) == 1
    assert control.report["metadata_census_succeeded"] is True
    assert control.report["coding_dispatch_possible"] is False
    assert control.report["metadata_census"] == control.receipt
    assert control.report["metadata_census_context"]["post_yield_admission_task_revalidated"] is True
    assert control.report["metadata_census_context"]["allocated_worktree_removed"] is True
    assert control.report["metadata_census_context"]["retained_owner_planning_receipt_join_verified"] is True


@pytest.mark.parametrize("fail_after_yield", [False, True])
def test_formal_owner_stays_in_host_and_publication_waits_for_native_exit(
        authored_isolated_census, fail_after_yield):
    control = authored_isolated_census
    _run_authored_census(control)
    default_argv = control.calls[0].copy()
    default_input = control.worker_inputs[0]
    default_helper = control.helper_calls[0]
    assert "formal_status_owner" not in default_helper
    assert "include_formal_status_census" not in default_helper
    for name in ("metadata_census", "metadata_census_context", "live_planning_provenance",
                 "coding_dispatch_possible"):
        control.report.pop(name)
    control.report["metadata_census_succeeded"] = False
    control.events.clear()
    control.formal_status_owner = object()  # Inert forwarding control; no admission claim.
    control.fail_after_yield = fail_after_yield
    if fail_after_yield:
        with pytest.raises(ValueError, match="source drift"):
            _run_authored_census(control, formal_status_owner=control.formal_status_owner)
        assert control.report["metadata_census_succeeded"] is False
        assert "metadata_census" not in control.report
        assert "live_planning_provenance" not in control.report
    else:
        _run_authored_census(control, formal_status_owner=control.formal_status_owner)
        assert control.report["metadata_census_succeeded"] is True
        assert control.report["metadata_census"] == control.receipt
        assert control.report["metadata_census_context"]["post_yield_admission_task_revalidated"] is True
    forwarded = control.helper_calls[1]
    assert forwarded["formal_status_owner"] is control.formal_status_owner
    assert forwarded["include_formal_status_census"] is True
    assert control.events == ["yield", "recheck"]
    assert control.calls[1] == default_argv
    assert control.worker_inputs[1] == default_input


def test_retained_owner_planning_digest_mismatch_refuses_before_worker_launch(authored_isolated_census):
    control = authored_isolated_census
    control.prepared["metadata"]["planning_metadata_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="retained owner planning metadata"):
        _run_authored_census(control)
    assert control.calls == [] and control.events == ["yield"]
    assert control.report["metadata_census_succeeded"] is False
    assert "retained_owner_planning_receipt_join_verified" not in control.prepared["metadata"]
    assert "metadata_census" not in control.report and "live_planning_provenance" not in control.report


def test_native_post_yield_drift_prevents_success_publication(authored_isolated_census):
    control = authored_isolated_census
    control.fail_after_yield = True
    with pytest.raises(ValueError, match="source drift"):
        _run_authored_census(control)
    assert control.events == ["yield", "recheck"] and len(control.calls) == 1
    assert control.report["metadata_census_succeeded"] is False
    assert "metadata_census" not in control.report and "live_planning_provenance" not in control.report


@pytest.mark.parametrize("mutation", [
    {"provider_calls": 1}, {"router_calls": 1}, {"provider_dispatch_observed": True},
    {"provider_token_usage": {}}, {"native_prompt_sha256": "3" * 64},
    {"container_boundary_artifact_sha256": "3" * 64}, {"container_boundary_checked": False},
    {"coding_reply_mode": "legacy"}, {"doctor_residual_included": False},
    {"public_instruction_included": False}, {"provider": "grok_cli"},
])
def test_foreign_or_dispatching_worker_receipt_cannot_publish_census(authored_isolated_census, mutation):
    control = authored_isolated_census
    control.receipt.update(mutation)
    with pytest.raises(ValueError, match="native owner bindings"):
        _run_authored_census(control)
    assert control.report["metadata_census_succeeded"] is False
    assert "metadata_census" not in control.report


def _worker_parser_and_route(argv):
    """Execute the exact shipped routing suffix with inert authored imports."""
    tree = ast.parse(worker.WORKER_ENTRY)
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    module = ast.Module(body=tree.body[start:], type_ignores=[])
    fake_sys = SimpleNamespace(argv=["worker-entry", *argv])
    events = []

    class AuthoredArgumentParser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            # argparse's own sys module otherwise reads pytest's ambient argv.
            return super().parse_args(fake_sys.argv[1:] if args is None else args, namespace)

    def main():
        events.append(("router", list(fake_sys.argv)))
        return 0

    def custody(function):
        events.append(("custody",))
        return function()

    def authored_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ipfs_accelerate_py.agent_supervisor.runtime":
            return SimpleNamespace(router_implementation_runner=SimpleNamespace(MAX_IMPLEMENTATION_TIMEOUT_SECONDS=600))
        if name == "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner":
            return SimpleNamespace(main=main)
        if name == "ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper":
            return SimpleNamespace(run_with_child_custody=custody)
        pytest.fail("worker routing escaped into provider/candidate import: " + name)

    sandbox_builtins = dict(vars(builtins), __import__=authored_import)
    namespace = {"__builtins__": sandbox_builtins,
                 "argparse": SimpleNamespace(ArgumentParser=AuthoredArgumentParser), "pathlib": __import__("pathlib"),
                 "sys": fake_sys, "selected_provider": "codex_cli", "artifact": Path("/authored/boundary.json"),
                 "digest": "4" * 64}
    try:
        exec(compile(module, "<shipped-worker-routing>", "exec"), namespace)
    except SystemExit as error:
        return error.code, events
    pytest.fail("shipped worker route must terminate")


def _worker_census_argv():
    return ["--semantic-repository", "/app", "--semantic-metadata-view", "common-bindings@1",
            "--semantic-metadata-census-only", "--coding-reply-mode", "ordinary-completion@1"]


def test_shipped_worker_parser_forwards_both_metadata_flags_through_child_custody():
    code, events = _worker_parser_and_route(_worker_census_argv())
    assert code == 0 and events[0] == ("custody",)
    assert len(events) == 2 and events[1][0] == "router"
    argv = events[1][1]
    assert "--semantic-metadata-census-only" in argv
    assert argv[argv.index("--semantic-metadata-view") + 1] == "common-bindings@1"
    assert argv[argv.index("--coding-reply-mode") + 1] == "ordinary-completion@1"
    assert argv[argv.index("--container-boundary-sha256") + 1] == "4" * 64


@pytest.mark.parametrize("extra", [
    ["--purpose", "planning"], ["--semantic-transport-schema", "supervisor-semantic-router-input@2"],
    ["--preflight"], ["--doctor-candidate-artifact", "/authored/candidate.json"],
    ["--doctor-contract-artifact", "/authored/contract.json"],
    ["--finite-repository-artifact", "/authored/finite.json"],
    ["--semantic-metadata-view", "legacy"], ["--provider", "grok_cli"],
])
def test_shipped_worker_parser_refuses_census_route_misuse_before_candidate_import(extra):
    code, events = _worker_parser_and_route([*_worker_census_argv(), *extra])
    assert isinstance(code, str) and events == []


@pytest.mark.parametrize("change", [
    {"metadata_census_only": "true"}, {"semantic_metadata_view": "legacy"},
    {"coding_reply_mode": "legacy"}, {"arm": "no-index"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"provider_profile": GROK_PROFILE, "model_name": "grok-4.7"},
])
def test_agent_census_controls_are_explicit_and_refuse_before_archive_reads(tmp_path, monkeypatch, change):
    parameter = inspect.signature(harbor.FullSupervisorAgent.__init__).parameters["metadata_census_only"]
    assert parameter.default is False

    def base_init(self, *args, **kwargs):
        self.model_name = kwargs["model_name"]
        self.logs_dir = kwargs["logs_dir"]

    monkeypatch.setattr(harbor.BaseAgent, "__init__", base_init)
    monkeypatch.setattr(Path, "resolve", _forbidden)
    options = dict(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol", runtime_archive=str(tmp_path / "unread"),
        arm="full", provider_profile=CODEX_PROFILE, metadata_census_only=True,
        semantic_metadata_view="common-bindings@1", coding_reply_mode="ordinary-completion@1")
    with pytest.raises(ValueError):
        harbor.FullSupervisorAgent(**{**options, **change})
