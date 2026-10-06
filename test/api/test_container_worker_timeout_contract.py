"""Execute the deployed worker parser after authored identity qualification.

These tests execute its actual parser and routing statements. They do not claim
UID/namespace qualification; the separate Docker probe exercises that boundary.
"""
import argparse
import ast
import io
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    PROFILES, CODING600_SOURCE384_PROFILE, coding_timeout_seconds,
    execution_budget, implementation_watchdog_seconds,
)
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


def _execute_worker(monkeypatch, args):
    tree = ast.parse(deployment.WORKER_ENTRY)
    start = next(index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    tree.body = tree.body[start:]
    monkeypatch.setattr(sys, "argv", ["worker-entry", "--provider", "grok_cli", *args])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"Authored no-provider probe")))
    namespace = {"argparse": argparse, "pathlib": pathlib, "sys": sys,
        "artifact": pathlib.Path("/authored-qualified-boundary.json"), "digest": "a" * 64,
        "selected_provider": "grok_cli"}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(tree, "actual-worker-parser-after-authored-boundary", "exec"), namespace)
    return stopped.value.code


@pytest.mark.parametrize("timeout", ["1", "180", "300", "600"])
def test_worker_reaches_actual_router_preflight_without_provider_calls(monkeypatch, capsys, timeout):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid-model probe must stop before provider discovery or invocation")
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    monkeypatch.setattr(intelligence_index, "select_efficient_route", forbidden)
    monkeypatch.setattr(llm_router, "get_llm_provider", forbidden)
    monkeypatch.setattr(llm_router, "generate_text", forbidden)
    code = _execute_worker(monkeypatch, ["--timeout", timeout, "--model", "invalid-authored-model"])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    value = runner.validate_runner_error_envelope(json.loads(captured.err))
    assert value["error_type"] == "ValueError"
    assert value["diagnostic"]["phase"] == "argument_validation"
    assert value["diagnostic"]["provider_dispatch_observed"] is None


@pytest.mark.parametrize("timeout", ["0", "-1", "601", "999999", "1.5", "true", "nan"])
def test_worker_refuses_out_of_range_or_noninteger_timeout_before_router(monkeypatch, capsys, timeout):
    monkeypatch.setattr(runner, "main", lambda: pytest.fail("worker timeout refusal must precede router"))
    code = _execute_worker(monkeypatch, ["--timeout", timeout, "--model", "invalid-authored-model"])
    if timeout in {"1.5", "true", "nan"}:
        assert code == 2
    else:
        assert code == "bounded timeout required"
    assert "router-implementation-invocation@1" not in capsys.readouterr().out


def test_worker_default_and_preflight_sleep_limit_are_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "main", lambda: calls.append(sys.argv[:]) or 0)
    assert _execute_worker(monkeypatch, ["--model", "grok-4.7"]) == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--timeout") + 1] == "90"
    calls.clear()
    assert _execute_worker(monkeypatch, ["--timeout", "600", "--preflight-sleep", "31"]) == "bounded timeout required"
    assert calls == []


@pytest.mark.parametrize("profile", [None, *PROFILES])
def test_worker_ceiling_does_not_change_signed_profile_or_outer_budgets(monkeypatch, profile):
    calls = []
    monkeypatch.setattr(runner, "main", lambda: calls.append(sys.argv[:]) or 0)
    cap = coding_timeout_seconds(profile)
    assert cap == (600 if profile == CODING600_SOURCE384_PROFILE else 300)
    assert implementation_watchdog_seconds(profile) == cap + 60
    assert _execute_worker(monkeypatch, ["--timeout", str(cap), "--model", "grok-4.7"]) == 0
    assert calls[0][calls[0].index("--timeout") + 1] == str(cap)
    budget = execution_budget(profile)
    assert budget["driver_seconds"] in {285, 900}
    assert budget["cleanup_seconds"] in {40, 60}
