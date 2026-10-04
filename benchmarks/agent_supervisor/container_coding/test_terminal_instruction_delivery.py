"""Execute the deployed wrapper's actual parser/dispatch with a stubbed router.

Container UID and namespace qualification is covered separately. These checks
exercise the worker argument boundary after that qualification, without sudo.
"""
import argparse
import ast
import pathlib
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router


PINS = ["--public-instruction-artifact", "/app/.runtime/router-public-instruction/pinned.json",
        "--public-instruction-sha256", "a" * 64, "--public-instruction-task-cid", "cid:task"]


def dispatch(monkeypatch, args):
    tree = ast.parse(deployment.WORKER_ENTRY)
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    tree.body = tree.body[start:]
    calls = []
    monkeypatch.setattr(router, "main", lambda: calls.append(sys.argv[:]) or 0)
    monkeypatch.setattr(sys, "argv", ["worker-entry", *args])
    namespace = {"argparse": argparse, "pathlib": pathlib, "sys": sys,
        "artifact": pathlib.Path("/worker-boundary/manifest.json"), "digest": "b" * 64}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(tree, "deployed-worker-parser", "exec"), namespace)
    return stopped.value.code, calls


@pytest.mark.parametrize("extra", [[], ["--semantic-repository", "/app"],
    ["--semantic-repository", "/app", "--doctor-residual-artifact", "/app/.runtime/residual.json",
     "--doctor-residual-sha256", "c" * 64, "--doctor-residual-task-cid", "cid:task"]])
def test_router_worker_forwards_exact_pins_on_both_coding_routes(monkeypatch, extra):
    code, calls = dispatch(monkeypatch, ["--model", "pinned", *extra, *PINS])
    assert code == 0 and len(calls) == 1
    assert calls[0][-len(PINS):] == PINS
    assert calls[0][calls[0].index("--purpose") + 1] == "coding"
    assert calls[0][calls[0].index("--container-boundary-sha256") + 1] == "b" * 64


@pytest.mark.parametrize("args", [PINS[:2], PINS[2:4], PINS[4:],
    [*PINS, "--purpose", "planning"], [*PINS, "--preflight"],
    [*PINS, "--doctor-contract-artifact", "/candidate.json", "--doctor-contract-sha256", "c" * 64,
     "--doctor-contract-task-cid", "cid:task"],
    [*PINS, "--doctor-candidate-artifact", "/candidate.json", "--doctor-candidate-sha256", "c" * 64,
     "--doctor-task-cid", "cid:task"]])
def test_worker_rejects_partial_or_unrelated_instruction_routes_before_router(monkeypatch, args):
    code, calls = dispatch(monkeypatch, args)
    assert code == "exact public instruction binding requires router coding"
    assert calls == []


def test_legacy_worker_invocation_needs_no_instruction_flags(monkeypatch):
    code, calls = dispatch(monkeypatch, ["--model", "pinned"])
    assert code == 0 and len(calls) == 1
    assert not any(value.startswith("--public-instruction") for value in calls[0])
