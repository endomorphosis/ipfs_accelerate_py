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


def dispatch(monkeypatch, args, *, provider="codex_cli"):
    tree = ast.parse(deployment.WORKER_ENTRY)
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    tree.body = tree.body[start:]
    calls = []
    monkeypatch.setattr(router, "main", lambda: calls.append(sys.argv[:]) or 0)
    monkeypatch.setattr(sys, "argv", ["worker-entry", *args])
    namespace = {"argparse": argparse, "pathlib": pathlib, "sys": sys,
        "artifact": pathlib.Path("/worker-boundary/manifest.json"), "digest": "b" * 64,
        "selected_provider": provider}
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


def test_grok_worker_preserves_deployed_provider_and_instruction_binding(monkeypatch):
    code, calls = dispatch(monkeypatch, ["--provider", "grok_cli", "--model", "grok-4.7", *PINS], provider="grok_cli")
    assert code == 0 and len(calls) == 1
    assert calls[0][calls[0].index("--provider") + 1] == "grok_cli"
    assert calls[0][-len(PINS):] == PINS


@pytest.mark.parametrize("deployed,requested", [("grok_cli", "codex_cli"), ("codex_cli", "grok_cli")])
def test_worker_refuses_provider_change_before_dispatch(monkeypatch, deployed, requested):
    code, calls = dispatch(monkeypatch, ["--provider", requested, "--model", "pinned"], provider=deployed)
    assert code == "provider differs from deployed worker route"
    assert calls == []


def test_opt_in_transport_reaches_only_semantic_coding_runner(monkeypatch):
    schema = "supervisor-semantic-router-input@2"
    code, calls = dispatch(monkeypatch, ["--semantic-repository", "/app",
        "--semantic-transport-schema", schema, *PINS])
    assert code == 0 and len(calls) == 1
    assert calls[0].count("--semantic-transport-schema") == 1
    assert calls[0][calls[0].index("--semantic-transport-schema") + 1] == schema
    assert calls[0][calls[0].index("--purpose") + 1] == "coding"
    code, calls = dispatch(monkeypatch, ["--purpose", "planning"])
    assert code == 0 and len(calls) == 1
    assert "--semantic-transport-schema" not in calls[0]


@pytest.mark.parametrize("extra", [[], ["--semantic-repository", "/app", "--purpose", "planning"],
    ["--semantic-repository", "/app", "--preflight"],
    ["--semantic-repository", "/app", "--doctor-candidate-artifact", "/candidate.json"]])
def test_opt_in_worker_rejects_nonsemantic_or_nonrouter_routes(monkeypatch, extra):
    code, calls = dispatch(monkeypatch, ["--semantic-transport-schema", "supervisor-semantic-router-input@2", *extra])
    assert code == "controller dictionary transport requires semantic router coding"
    assert calls == []


def test_worker_schema_values_are_closed_and_legacy_default_is_omitted(monkeypatch):
    code, calls = dispatch(monkeypatch, ["--semantic-transport-schema", "supervisor-semantic-router-input@3"])
    assert code == 2 and calls == []
    code, calls = dispatch(monkeypatch, ["--semantic-repository", "/app",
        "--semantic-transport-schema", "supervisor-semantic-router-input@1"])
    assert code == 0 and len(calls) == 1
    assert "--semantic-transport-schema" not in calls[0]
