"""Exercise actual signed command construction and generated worker routing.

Provider/deployment behavior is authored. These checks establish option custody
and default compatibility, not a qualified container or a measured token gain.
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
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor_dispatch
from test.api.test_terminal_prompt_tokenizer_forwarding import supervisor

POLICY = Path("/opt/ipfs-supervisor/policies/task policy.json")
DIGEST = "c" * 64
OPTIONS = {"semantic_decision_policy_artifact": POLICY,
           "semantic_decision_policy_sha256": DIGEST,
           "semantic_decision_max_calls": 2,
           "semantic_decision_max_total_input_tokens": 18000}
FLAGS = ["--semantic-decision-policy-artifact", str(POLICY),
         "--semantic-decision-policy-sha256", DIGEST,
         "--semantic-decision-max-calls", "2",
         "--semantic-decision-max-total-input-tokens", "18000"]
ROUTER = "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner"
DECISION = "ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker"


@pytest.fixture
def worker_dispatch(monkeypatch):
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
        assert name in {ROUTER, DECISION} and fromlist == ("main",) and level == 0
        imports.append(name)
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


def command(**changes):
    return doctor_dispatch.implementation_argv(router=Path("/router"), model="gpt-5.6-sol",
        reasoning="high", timeout=90, doctor=changes.pop("doctor", None),
        semantic_repository=changes.pop("semantic_repository", Path("/app")), **changes)


def test_policy_is_signed_literal_argv_and_default_is_unchanged():
    assert deployment.semantic_decision_arguments() == []
    assert deployment.semantic_decision_arguments(**OPTIONS) == FLAGS
    selected = command(**OPTIONS)
    assert selected == command() + FLAGS
    assert shlex.split(shlex.join(selected)) == selected


@pytest.mark.parametrize("changes", [
    {"semantic_decision_policy_artifact": POLICY},
    {"semantic_decision_policy_sha256": DIGEST},
    {**OPTIONS, "semantic_decision_policy_artifact": "relative.json"},
    {**OPTIONS, "semantic_decision_policy_artifact": "/owner/../policy.json"},
    {**OPTIONS, "semantic_decision_policy_sha256": "invalid"},
    {**OPTIONS, "semantic_decision_max_calls": True},
    {**OPTIONS, "semantic_decision_max_calls": 5},
    {**OPTIONS, "semantic_decision_max_total_input_tokens": 0},
    {"semantic_decision_max_calls": 2},
    {"semantic_decision_max_total_input_tokens": 100},
])
def test_invalid_policy_or_budgets_refuse_before_owner_or_provider_access(monkeypatch, changes):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("must reject before identity/state access"))
    with pytest.raises(ValueError):
        command(**changes)
    with pytest.raises(ValueError):
        driver.run(instruction=Path("/instruction"), state=Path("/state"), arm="full", **changes)


@pytest.mark.parametrize("changes", [
    {"semantic_repository": None},
    {"semantic_slot_policy_revision": "closed-typed-identifiers@2"},
    {"doctor": {"route": "doctor_candidate"}},
    {"doctor": {"route": "doctor_contract_candidate"}},
])
def test_direct_policy_refuses_other_routes(changes):
    with pytest.raises(ValueError, match="exclusive bound model coding"):
        command(**OPTIONS, **changes)


def test_no_index_policy_refuses_before_owner_access(monkeypatch):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("must reject before identity/state access"))
    with pytest.raises(ValueError, match="exclusive full indexed arm"):
        driver.run(instruction=Path("/instruction"), state=Path("/state"), arm="no-index", **OPTIONS)


def test_actual_signed_launch_carries_policy_only_to_coding(supervisor, monkeypatch):
    state, planned, signed = supervisor
    result = driver.run(instruction=Path("/instruction"), state=state, arm="full",
                        doctor_dispatch_policy="model-only", **OPTIONS)
    assert result["task_completed"] is True
    assert len(planned) == len(signed) == 1
    assert all("--semantic-decision" not in token for token in planned[0])
    start = signed[0].index("--semantic-decision-policy-artifact")
    assert signed[0][start:start + len(FLAGS)] == FLAGS
    assert "--public-instruction-artifact" in signed[0]


def test_generated_route_is_opt_in_and_forwards_boundary_and_complete_budgets(worker_dispatch):
    status, calls, imports = worker_dispatch([])
    assert status == 0 and imports == [ROUTER]
    status, calls, imports = worker_dispatch(["--semantic-repository", "/app", *FLAGS,
        "--max-input-tokens", "15000", "--public-instruction-artifact", "/instruction.json",
        "--public-instruction-sha256", "e" * 64, "--public-instruction-task-cid", "task"])
    assert status == 0 and imports == [DECISION] and len(calls) == 1
    for index in range(0, len(FLAGS), 2):
        start = calls[0].index(FLAGS[index])
        assert calls[0][start:start + 2] == FLAGS[index:index + 2]
    assert "--container-boundary-sha256" in calls[0]
    assert calls[0][calls[0].index("--max-input-tokens") + 1] == "15000"
    assert "--public-instruction-artifact" in calls[0]


@pytest.mark.parametrize("extra", [
    ["--purpose", "planning"], ["--preflight"],
    ["--semantic-slot-policy-revision", "closed-typed-identifiers@2"],
    ["--doctor-candidate-artifact", "/doctor.json", "--doctor-candidate-sha256", "d" * 64,
     "--doctor-task-cid", "task"],
])
def test_generated_route_refuses_other_modes_before_import(worker_dispatch, extra):
    status, calls, imports = worker_dispatch(["--semantic-repository", "/app", *FLAGS, *extra])
    assert status == "semantic decision policy requires exclusive bound model coding"
    assert calls == imports == []


@pytest.mark.parametrize("flags", [
    FLAGS[:2], FLAGS[2:4], ["--semantic-decision-max-calls", "2"],
    ["--semantic-decision-max-total-input-tokens", "100"],
    [*FLAGS[:4], "--semantic-decision-max-calls", "5"],
    [*FLAGS[:4], "--semantic-decision-max-total-input-tokens", "0"],
])
def test_generated_route_refuses_incomplete_or_invalid_policy_before_import(worker_dispatch, flags):
    status, calls, imports = worker_dispatch(["--semantic-repository", "/app", *flags])
    assert status != 0 and calls == imports == []
