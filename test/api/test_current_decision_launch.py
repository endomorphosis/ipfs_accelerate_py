"""Current generated handoffs preserve native custody without provider calls.

Reconciled peer routing uses genuine runtime modules with authored entry callbacks.
The positive owner fixture uses real producer capsules and a real Git worktree;
it does not qualify a container identity or establish benchmark acceptance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import pathlib
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_owner as owner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from test.api.test_semantic_router_integration import allocated, native

POLICY = {"semantic_decision_policy_artifact": "/owner/policy.json",
          "semantic_decision_policy_sha256": "a" * 64}
NOMINATION = {"semantic_decision_nomination_artifact": "/owner/nomination.json",
              "semantic_decision_nomination_sha256": "b" * 64}


def _command(**options):
    return doctor.implementation_argv(router=Path("/router"), model="pinned", reasoning="high",
        timeout=options.pop("timeout", 90), provider=options.pop("provider", "codex_cli"),
        doctor=options.pop("doctor", None),
        semantic_repository=options.pop("semantic_repository", Path("/app")), **options)


def test_default_command_is_unchanged_and_nomination_adds_signed_task_only():
    baseline = _command()
    assert baseline == ["/router", "--model", "pinned", "--reasoning-effort", "high",
        "--timeout", "90", "--max-output-tokens", "4096", "--semantic-repository", "/app"]
    assert not any(part.startswith("--semantic-decision") for part in baseline)
    assert _command(semantic_slot_policy_revision="legacy@1", semantic_context_projection=False,
        semantic_native_input=False) == baseline
    selected = _command(**NOMINATION, semantic_decision_task_id="ADMITTED")
    assert selected[:len(baseline)] == baseline
    assert selected[selected.index("--semantic-decision-task-id") + 1] == "ADMITTED"
    assert selected[selected.index("--semantic-decision-typed-nomination") + 1] == "enabled"
    assert "--semantic-retrieval-query-reference" not in selected


@pytest.mark.parametrize("extra", [
    {"semantic_repository": None}, {"provider": "grok_cli"}, {"timeout": 601},
    {"doctor": {"route": "doctor_candidate", "status": "candidate_ready", "provider_calls": 0}},
    {"doctor": {"route": "doctor_contract_candidate", "status": "candidate_ready", "provider_calls": 0}},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"semantic_metadata_view": "common-bindings@1"}, {"source_view_profile": "identity@1"},
    {"tool_output_policy": "artifact-preview@1"}, {"coding_reply_mode": "ordinary-completion@1"},
    {"semantic_decision_max_calls": True}, {"semantic_decision_max_total_input_tokens": 0},
    {"semantic_native_input": True}, {"semantic_slot_policy_revision": "closed-typed-identifiers@2"},
])
def test_selected_command_refuses_incompatible_current_routes(extra):
    with pytest.raises(ValueError):
        _command(**(POLICY | extra))


@pytest.mark.parametrize("options", [
    {"semantic_decision_policy_artifact": "/owner/policy.json"},
    {**POLICY, "semantic_decision_policy_sha256": "bad"},
    {**POLICY, **NOMINATION}, {"semantic_decision_max_calls": 2},
    {**POLICY, "source384_config": Path("/no/read")},
    {**POLICY, "numerical_selection_config": Path("/no/read")},
])
def test_driver_refuses_invalid_selection_before_identity_or_state(options, monkeypatch):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("identity access preceded selection refusal"))
    with pytest.raises(ValueError):
        driver.run(instruction=Path("/no/read"), state=Path("/no/state"), arm="full", **options)


def test_query_owner_arguments_require_explicit_admitted_bundle_group():
    options = {**NOMINATION, "disable_semantic_typed_nomination": True,
        "semantic_retrieval_query_reference": True, "semantic_decision_task_id": "ADMITTED"}
    with pytest.raises(ValueError):
        _command(**options)
    selected = _command(**options, semantic_decision_context_bundle_artifact=".runtime/context.json",
        semantic_decision_context_bundle_sha256="c" * 64, semantic_decision_task_cid="task:admitted")
    assert selected[selected.index("--semantic-decision-task-cid") + 1] == "task:admitted"
    assert selected[selected.index("--semantic-decision-context-bundle-sha256") + 1] == "c" * 64
    assert selected[selected.index("--semantic-decision-typed-nomination") + 1] == "disabled"
    assert "--semantic-retrieval-query-reference" not in selected


@pytest.mark.parametrize("nomination", [False, True])
def test_actual_driver_builds_selected_coding_route_from_admitted_task(tmp_path, monkeypatch, nomination):
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
    monkeypatch.setattr(driver, "_security_runtime_inputs", lambda **kwargs: {"train_autoencoder": False})
    monkeypatch.setattr(driver.subprocess, "run", lambda argv, **kwargs:
        cleanup.append(argv) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    monkeypatch.setattr(driver.preparation, "initial_context", lambda **kwargs: {})
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: plans.append(kwargs) or {"qualified": True})
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs:
        {"context_bundle": {"artifact": ".runtime/admitted-context.json", "sha256": "c" * 64}})
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", lambda *_args, **_kwargs:
        {"graph": SimpleNamespace(tasks=[SimpleNamespace(task_cid="task:admitted", task_key="ADMITTED")])})
    monkeypatch.setattr(doctor, "prepare_terminal_doctor_dispatch", lambda **kwargs:
        {"route": "model_router", "status": "residual"})
    monkeypatch.setattr(doctor, "implementation_argv", capture_builder)
    options = NOMINATION if nomination else POLICY
    report = driver.run(instruction=tmp_path / "instruction", state=state, arm="full", **options)
    assert report["error_phase"] == "implementation_setup"
    assert report["error"]["type"] == "ReachedWorkerRoute"
    assert report["provider_invocations"] == [] and report["task_completed"] is False
    assert report["semantic_decision_dialogue_expected"] is True
    assert len(plans) == 1 and not any("semantic_decision" in key for key in plans[0])
    assert len(routed) == 1
    command = routed[0]
    artifact_flag = "--semantic-decision-" + ("nomination" if nomination else "policy") + "-artifact"
    assert command[command.index(artifact_flag) + 1] == options[artifact_flag[2:].replace("-", "_")]
    assert 1 <= int(command[command.index("--timeout") + 1]) <= 600
    assert report["provider_coding_timeout_seconds"] <= 600 and "--budget-profile" not in command
    if nomination:
        assert command[command.index("--semantic-decision-task-id") + 1] == "ADMITTED"
        assert report["semantic_decision_owner_binding_expected"] is True
        assert report["semantic_decision_expected_task_id"] == "ADMITTED"
        assert report["semantic_decision_nomination_sha256"] == NOMINATION["semantic_decision_nomination_sha256"]
        assert report["semantic_decision_typed_nomination_expected"] is True
    else:
        assert "--semantic-decision-task-id" not in command
        assert "semantic_decision_owner_binding_expected" not in report
    assert report["worker_cleanup_returncode"] == 0 and cleanup[0][-1] == "--cleanup"
    retained = json.loads((state.parent / "run-result.json").read_text())
    assert retained["error"]["type"] == "ReachedWorkerRoute"
    assert retained["semantic_decision_dialogue_expected"] is True


@pytest.fixture
def worker_entry(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_cli_subreaper
    parsed = ast.parse(deployment.WORKER_ENTRY)
    start = next(index for index, node in enumerate(parsed.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    code = compile(ast.Module(body=parsed.body[start:], type_ignores=[]), "<current-worker-routing>", "exec")
    calls = []
    monkeypatch.setattr(worker, "main", lambda: calls.append(("decision", list(sys.argv))) or 0)
    monkeypatch.setattr(router, "main", lambda: calls.append(("router", list(sys.argv))) or 0)
    monkeypatch.setattr(native_cli_subreaper, "run_with_child_custody", lambda callback: callback())

    def run(arguments, provider="codex_cli"):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["worker-entry", *arguments])
        with pytest.raises(SystemExit) as error:
            exec(code, {"argparse": argparse, "pathlib": pathlib, "sys": sys,
                "selected_provider": provider, "artifact": Path("/owner/boundary.json"),
                "digest": "d" * 64, "evidence": {}})
        return error.value.code, list(calls)
    return run


def test_generated_worker_preserves_grok_and_selects_decision_with_custody(worker_entry):
    status, calls = worker_entry([], provider="grok_cli")
    assert status == 0 and calls[0][0] == "router"
    assert calls[0][1][calls[0][1].index("--provider") + 1] == "grok_cli"
    flags = ["--semantic-repository", "/app", "--semantic-decision-policy-artifact", "/owner/policy.json",
        "--semantic-decision-policy-sha256", "a" * 64, "--semantic-decision-max-calls", "2",
        "--semantic-decision-max-total-input-tokens", "10000", "--max-input-tokens", "8000"]
    status, calls = worker_entry(flags)
    assert status == 0 and len(calls) == 1 and calls[0][0] == "decision"
    actual = calls[0][1]
    for flag in ("--semantic-decision-policy-artifact", "--semantic-decision-policy-sha256",
                 "--semantic-decision-max-calls", "--semantic-decision-max-total-input-tokens", "--max-input-tokens"):
        assert actual[actual.index(flag) + 1] == flags[flags.index(flag) + 1]
    assert actual[actual.index("--container-boundary-sha256") + 1] == "d" * 64
    assert "--budget-profile" not in actual


@pytest.mark.parametrize("extra", [
    ["--provider", "grok_cli"], ["--purpose", "planning"], ["--preflight"], ["--timeout", "601"],
    ["--semantic-transport-schema", "supervisor-semantic-router-input@2"],
    ["--budget-profile", "formalization-source384@1"],
    ["--source-formalization-handoff", "/owner/formal.json", "--source-formalization-handoff-sha256", "f" * 64],
    ["--source-lookup-descriptor", "/owner/source.json", "--source-lookup-descriptor-sha256", "f" * 64],
    ["--source-input-policy", "source-input-raw@1"], ["--semantic-metadata-census-only"],
])
def test_generated_worker_refuses_selected_unsupported_modes_before_callback(worker_entry, extra):
    flags = ["--semantic-repository", "/app", "--semantic-decision-policy-artifact", "/owner/policy.json",
        "--semantic-decision-policy-sha256", "a" * 64]
    status, calls = worker_entry([*flags, *extra])
    assert status != 0 and calls == []


@pytest.mark.parametrize("edit_source", [False, True])
def test_genuine_owner_binding_replays_in_current_worker_without_provider(allocated, monkeypatch, capsys, edit_source):
    repository, workspace, prompt, _ = allocated
    repository.chmod(0o755)
    (repository / ".runtime").chmod(0o755)
    parent = json.loads((repository / ".runtime/semantic/worker-context.json").read_text())
    policy = SemanticDecisionPolicy(required_capsule_ids=tuple(row["capsule_cid"] for row in parent["capsules"]),
        optional_capsule_ids=())
    nomination = repository.parent / "owner-nomination.json"
    raw = json.dumps(owner.build_decision_nomination(policy), sort_keys=True, separators=(",", ":")).encode()
    nomination.write_bytes(raw)
    arguments = _command(semantic_repository=repository,
        semantic_decision_nomination_artifact=nomination,
        semantic_decision_nomination_sha256=hashlib.sha256(raw).hexdigest(),
        disable_semantic_typed_nomination=True, semantic_decision_task_id="TASK-1")[1:]
    # Execute the actual generated owner staging statements with a sealed
    # authored stdin descriptor. Only its fd number is mapped for pytest;
    # real owner authentication/publication and source verification run.
    import fcntl
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create
    descriptor = _validation_memfd_create("authored-owner-stdin", 0x0001 | 0x0002)
    parsed = ast.parse(deployment.OWNER_ENTRY)
    assignments = {node.targets[0].id: index for index, node in enumerate(parsed.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    stage = compile(ast.Module(body=parsed.body[assignments["worker_arguments"]:
        assignments["proof_query_guard"]], type_ignores=[]), "<current-owner-staging>", "exec")
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"handoff": {"argv": arguments}, "cwd": workspace, "root": Path("/opt/ipfs-supervisor"),
        "json": json, "sys": sys,
        "os": SimpleNamespace(pread=lambda fd, size, offset: os.pread(descriptor, size, offset))}
    try:
        os.write(descriptor, prompt.encode())
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, getattr(fcntl, "F_ADD_SEALS", 1033), 0x0001 | 0x0002 | 0x0004 | 0x0008)
        exec(stage, namespace)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
        assert os.read(descriptor, len(prompt.encode()) + 1) == prompt.encode()
    finally:
        os.close(descriptor)
    dispatch = namespace["dispatch"]
    assert namespace["command"][6:] == list(dispatch.worker_arguments)
    assert json.loads(capsys.readouterr().out) == dispatch.receipt
    assert dispatch.prompt_bytes == prompt.encode()
    assert dispatch.receipt["native_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert dispatch.receipt["provider_called"] is False
    artifact = Path(dispatch.receipt["launch_policy_artifact"])
    assert artifact.stat().st_mode & 0o777 == 0o444 and not artifact.is_relative_to(workspace)
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    counter = router.PromptTokenCounter(provider="codex_cli", model="pinned", tokenizer_id="authored-fixture",
        tokenizer_revision="1", special_token_policy="fixture", count_kind="estimate",
        compatibility_basis="authored fixture only", count_tokens=lambda text: len(text))
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer",
        lambda **kwargs: (counter, {"schema": "authored-tokenizer-fixture"}))
    if edit_source:
        (repository / "mod.py").write_text("changed before dispatch\n")
        with pytest.raises(ValueError):
            worker._prepare_launch(prompt=prompt, model="pinned", semantic_repository=repository,
                semantic_decision_policy_artifact=artifact,
                semantic_decision_policy_sha256=dispatch.receipt["launch_policy_sha256"])
    else:
        prepared, binding = worker._prepare_launch(prompt=prompt, model="pinned", semantic_repository=repository,
            semantic_decision_policy_artifact=artifact,
            semantic_decision_policy_sha256=dispatch.receipt["launch_policy_sha256"])
        assert prepared["native_prompt"] == prompt and prepared["policy"] == policy
        assert binding["original_native_prompt_sha256"] == dispatch.receipt["native_prompt_sha256"]
        assert binding["launch_policy_sha256"] == dispatch.receipt["launch_policy_sha256"]


@pytest.mark.parametrize("entry,real_uid,effective_uid", [
    ("owner", 1001, 1000), ("owner", 1000, 0),
    ("worker", 1000, 1001), ("worker", 1001, 0),
])
def test_generated_identity_refusal_precedes_argument_or_provider_dispatch(entry, real_uid, effective_uid):
    text = deployment.OWNER_ENTRY if entry == "owner" else deployment.WORKER_ENTRY
    parsed = ast.parse(text)
    message = entry + " identity required"
    guard = next(node for node in parsed.body if isinstance(node, ast.If)
        and any(isinstance(child, ast.Constant) and child.value == message for child in ast.walk(node)))
    code = compile(ast.Module(body=[guard], type_ignores=[]), "<current-entry-identity>", "exec")
    with pytest.raises(SystemExit, match=message):
        exec(code, {"os": SimpleNamespace(getuid=lambda: real_uid, geteuid=lambda: effective_uid),
            "boundary": {"owner_uid": 1000}})
