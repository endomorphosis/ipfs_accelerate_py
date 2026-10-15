"""Exercise full-index native-input option custody without model calls.

Generated parser/routing statements are executed unchanged against a stub router.
Authored owner/process fixtures check that the control changes coding input
selection, while retaining indexed preparation, planning, and signed bindings.
These are offline wiring checks, not provider/token-savings qualification.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from pathlib import Path
import pathlib
import shlex
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from test.api.test_terminal_prompt_tokenizer_forwarding import supervisor

ROUTER = "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner"
BOUND = ["--semantic-repository", "/owner/canonical repository"]
NATIVE = "--semantic-native-input"


def command(**changes):
    return doctor.implementation_argv(**{
        "router": Path("/router"), "model": "gpt-5.6-sol", "reasoning": "high",
        "timeout": 90, "semantic_repository": Path("/app"), "doctor": None, **changes,
    })


@pytest.fixture
def dispatch(monkeypatch):
    parsed = ast.parse(deployment.WORKER_ENTRY)
    start = next(i for i, node in enumerate(parsed.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets))
    code = compile(ast.Module(body=parsed.body[start:], type_ignores=[]), "<worker-routing>", "exec")
    calls, imports = [], []

    def imported(name, globals=None, locals=None, fromlist=(), level=0):
        # These are authored routing fixtures; actual worker custody is
        # qualified separately in the protected deployment process.
        if name == "ipfs_accelerate_py.agent_supervisor.runtime" and tuple(fromlist) == ("router_implementation_runner",):
            return SimpleNamespace(router_implementation_runner=SimpleNamespace(MAX_IMPLEMENTATION_TIMEOUT_SECONDS=300))
        if name == "ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper":
            return SimpleNamespace(run_with_child_custody=lambda callback: callback())
        imports.append(name)
        assert name == ROUTER and tuple(fromlist) == ("main",) and level == 0
        return SimpleNamespace(main=lambda: calls.append(list(sys.argv)) or 0)

    def run(arguments):
        calls.clear()
        imports.clear()
        monkeypatch.setattr(sys, "argv", ["worker-entry", *arguments])
        with pytest.raises(SystemExit) as result:
            exec(code, {"__builtins__": {**vars(builtins), "__import__": imported},
                        "argparse": argparse, "pathlib": pathlib, "sys": sys, "selected_provider": "codex_cli",
                        "artifact": Path("/fixture/boundary.json"), "digest": "a" * 64})
        return result.value.code, list(calls), list(imports)

    return run


def test_default_helper_and_signed_argv_remain_unchanged():
    assert deployment.semantic_context_arguments() == []
    assert deployment.semantic_context_arguments(semantic_native_input=False) == []
    assert deployment.semantic_context_arguments(semantic_native_input=True) == [NATIVE]
    assert command(semantic_native_input=False) == command()
    selected = command(semantic_native_input=True)
    assert selected == command() + [NATIVE]
    assert shlex.split(shlex.join(selected)) == selected
    assert selected[selected.index("--semantic-repository") + 1] == "/app"


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_native_control_requires_an_explicit_boolean_before_owner_access(monkeypatch, value):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("invalid control reached owner access"))
    with pytest.raises(ValueError, match="explicit Boolean"):
        deployment.semantic_context_arguments(semantic_native_input=value)
    with pytest.raises(ValueError, match="explicit Boolean"):
        command(semantic_native_input=value)
    with pytest.raises(ValueError, match="explicit Boolean"):
        driver.run(instruction=Path("/input"), state=Path("/state"), arm="full", semantic_native_input=value)


@pytest.mark.parametrize("options", [
    {"semantic_context_projection": True},
    {"semantic_slot_policy_revision": "closed-typed-identifiers@2"},
    {"semantic_slot_policy_revision": "closed-typed-identifiers@2", "semantic_context_projection": True},
    {"semantic_decision_policy_artifact": Path("/policy.json"), "semantic_decision_policy_sha256": "b" * 64},
    {"semantic_decision_nomination_artifact": Path("/nomination.json"),
     "semantic_decision_nomination_sha256": "b" * 64, "semantic_decision_task_id": "TASK"},
])
def test_native_control_excludes_projection_encoding_and_decision_before_preparation(monkeypatch, options):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("invalid control reached owner access"))
    with pytest.raises(ValueError):
        command(semantic_native_input=True, **options)
    driver_options = {key: value for key, value in options.items() if key != "semantic_decision_task_id"}
    with pytest.raises(ValueError):
        driver.run(instruction=Path("/input"), state=Path("/state"), arm="full",
                   semantic_native_input=True, **driver_options)


def test_no_index_control_refuses_before_owner_access(monkeypatch):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("no-index control reached owner access"))
    with pytest.raises(ValueError, match="full indexed arm"):
        driver.run(instruction=Path("/input"), state=Path("/state"), arm="no-index", semantic_native_input=True)


@pytest.mark.parametrize("changes", [
    {"semantic_repository": None},
    {"doctor": {"route": "doctor_candidate"}},
    {"doctor": {"route": "doctor_contract_candidate"}},
])
def test_signed_native_control_requires_bound_model_coding(changes):
    with pytest.raises(ValueError, match="bound model coding"):
        command(semantic_native_input=True, **changes)


def test_generated_default_and_explicit_native_argv_have_only_the_requested_difference(dispatch):
    status, baseline, imports = dispatch(BOUND)
    assert status == 0 and imports == [ROUTER]
    status, selected, imports = dispatch([*BOUND, NATIVE])
    assert status == 0 and imports == [ROUTER]
    assert selected == [baseline[0] + [NATIVE]]
    assert selected[0][selected[0].index("--semantic-repository") + 1] == BOUND[1]


def test_generated_native_control_preserves_tokenizer_residual_and_public_instruction_bindings(dispatch):
    literals = [
        "--prompt-tokenizer-artifact", "/owner/tokenizer.json", "--prompt-tokenizer-sha256", "b" * 64,
        "--max-input-tokens", "20000",
        "--doctor-residual-artifact", "/owner/residual.json", "--doctor-residual-sha256", "c" * 64,
        "--doctor-residual-task-cid", "task",
        "--public-instruction-artifact", "/owner/instruction.json", "--public-instruction-sha256", "d" * 64,
        "--public-instruction-task-cid", "task",
    ]
    status, calls, imports = dispatch([*BOUND, NATIVE, *literals])
    assert status == 0 and imports == [ROUTER]
    start = calls[0].index("--semantic-repository")
    assert calls[0][start:] == [*BOUND, NATIVE, *literals]


@pytest.mark.parametrize("extra", [
    [], ["--purpose", "planning"], ["--preflight"],
    ["--doctor-candidate-artifact", "/candidate.json", "--doctor-candidate-sha256", "c" * 64,
     "--doctor-task-cid", "task"],
    ["--doctor-contract-artifact", "/contract.json", "--doctor-contract-sha256", "c" * 64,
     "--doctor-contract-task-cid", "task"],
    ["--finite-repository-artifact", "/finite.json", "--finite-repository-sha256", "c" * 64,
     "--finite-repository-task-cid", "task"],
])
def test_generated_native_control_requires_semantic_coding_before_router_import(dispatch, extra):
    status, calls, imports = dispatch([NATIVE, *extra])
    assert status == "semantic context options require bound model coding"
    assert calls == imports == []


@pytest.mark.parametrize("extra", [
    ["--purpose", "planning"], ["--preflight"],
    ["--semantic-context-projection"],
    ["--semantic-slot-policy-revision", "closed-typed-identifiers@2"],
    ["--semantic-decision-policy-artifact", "/policy.json", "--semantic-decision-policy-sha256", "b" * 64],
])
def test_generated_bound_native_control_refuses_conflicts_before_router_import(dispatch, extra):
    status, calls, imports = dispatch([*BOUND, NATIVE, *extra])
    assert status != 0 and calls == imports == []


@pytest.mark.parametrize("native", [NATIVE, NATIVE + "=true"])
@pytest.mark.parametrize("decision", [
    "--semantic-decision-policy-artifact", "--semantic-decision-policy-sha256",
    "--semantic-decision-nomination-artifact", "--semantic-decision-nomination-sha256",
    "--semantic-decision-task-id",
])
def test_owner_refuses_native_decision_conflict_before_nomination_preparation(native, decision):
    parsed = ast.parse(deployment.OWNER_ENTRY)
    start = next(i for i, node in enumerate(parsed.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "worker_arguments" for t in node.targets))
    code = compile(ast.Module(body=parsed.body[start:start + 3], type_ignores=[]), "<owner-native-guard>", "exec")
    with pytest.raises(SystemExit, match="excludes semantic decision"):
        exec(code, {"sys": SimpleNamespace(argv=["owner", native, decision + "=value"])})


def test_actual_full_driver_keeps_cold_preparation_planning_and_admission(supervisor, monkeypatch):
    state, planned, signed = supervisor
    calls = []
    for name in ("prepare", "initial_context", "plan", "context"):
        original = getattr(driver.preparation, name)

        def tracked(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(driver.preparation, name, tracked)
    verify = driver.verify_local_benchmark_admission

    def admitted(*args, **kwargs):
        calls.append("admission")
        return verify(*args, **kwargs)

    monkeypatch.setattr(driver, "verify_local_benchmark_admission", admitted)
    monkeypatch.setattr(driver, "prepare_terminal_doctor_dispatch",
                        lambda **_: pytest.fail("model-only must not select a Doctor route"))
    result = driver.run(instruction=Path("/instruction"), state=state, arm="full",
                        semantic_native_input=True, doctor_dispatch_policy="model-only")
    assert result["task_completed"] is True
    assert calls[:5] == ["prepare", "initial_context", "plan", "context", "admission"]
    assert len(planned) == len(signed) == 1
    assert NATIVE not in planned[0] and "--semantic-repository" not in planned[0]
    assert signed[0].count(NATIVE) == 1
    assert signed[0][signed[0].index("--semantic-repository") + 1] == "/app"
    assert "--public-instruction-artifact" in signed[0]


def test_driver_cli_forwards_explicit_native_control(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["supervisor", "--instruction", "/instruction", "--state", "/state",
                                     "--arm", "full", NATIVE])
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": True, "arm": "full", "seconds": 0, "provider_invocations": []})
    assert driver.main() == 0
    assert calls[0]["semantic_native_input"] is True
