"""Longer coding budgets cross the real driver and generated worker boundary.

The subprocess fixture retains real deadline signals with owned service doubles.
Worker parser tests stub the router after its existing container qualification;
none of these checks dispatch a provider or grant container authority.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.test_terminal_instruction_delivery import (
    PINS,
    dispatch,
)

from test.api.test_terminal_agent_work_deadline import SCRIPT


@pytest.mark.parametrize("timeout", [570, 600])
def test_generated_worker_forwards_extended_coding_budget_with_exact_pins(monkeypatch, timeout):
    code, calls = dispatch(monkeypatch, ["--model", "pinned", "--timeout", str(timeout), *PINS])
    assert code == 0 and len(calls) == 1
    assert calls[0][calls[0].index("--timeout") + 1] == str(timeout)
    assert calls[0][-len(PINS):] == PINS


@pytest.mark.parametrize("timeout", [0, 601])
def test_generated_worker_refuses_out_of_bound_coding_budget_before_router(monkeypatch, timeout):
    code, calls = dispatch(monkeypatch, ["--model", "pinned", "--timeout", str(timeout), *PINS])
    assert code == "bounded timeout required" and calls == []


def test_extended_timeout_retains_preflight_sleep_bound(monkeypatch):
    code, calls = dispatch(monkeypatch, ["--model", "pinned", "--timeout", "570",
                                         "--preflight", "--preflight-sleep", "31"])
    assert code == "bounded timeout required" and calls == []


def test_extended_timeout_retains_public_instruction_route_restrictions(monkeypatch):
    code, calls = dispatch(monkeypatch, ["--model", "pinned", "--timeout", "570",
                                         "--purpose", "planning", *PINS])
    assert code == "exact public instruction binding requires router coding" and calls == []


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
@pytest.mark.parametrize("timeout", [90, 570])
def test_actual_driver_passes_its_budget_to_preparation_and_keeps_cleanup_reserve(tmp_path, timeout):
    original_call = 'driver.run(instruction=instruction, state=state, arm="no-index", timeout_seconds=90)'
    original_receipt = 'assert result["reserved_cleanup_seconds"] == 40 and result["work_cutoff_seconds"] == 50'
    original_prepare = 'events.append("prepare")'
    assert all(SCRIPT.count(text) == 1 for text in (original_call, original_receipt, original_prepare))
    script = SCRIPT.replace(original_call, original_call.replace("timeout_seconds=90", f"timeout_seconds={timeout}"))
    script = script.replace(original_receipt,
        f'assert result["reserved_cleanup_seconds"] == 40 and result["work_cutoff_seconds"] == {timeout - 40}')
    script = script.replace(original_prepare,
        original_prepare + '\n    events.append({"prepare_budget": kwargs["max_total_agent_seconds"]})')
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path),
        "ordinary_optional_error", str(repository)], cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    observed = json.loads(completed.stdout.splitlines()[-1])
    assert observed["events"][1] == {"prepare_budget": timeout}
    assert observed["report"]["max_total_agent_seconds"] == timeout
    assert observed["report"]["reserved_cleanup_seconds"] == 40
    assert observed["report"]["provider_invocations"] == []


@pytest.mark.parametrize("saved, expected", [({}, 300), ({"max_total_agent_seconds": 570}, 570)])
def test_planner_receipt_budget_preserves_saved_or_legacy_value(saved, expected):
    assert preparation._prepared_agent_timeout(saved) == expected


@pytest.mark.parametrize("timeout", [True, 89, 601, 570.0])
def test_preparation_rejects_invalid_budget_before_reading_or_mutating_repository(tmp_path, timeout):
    state = tmp_path / "state"
    with pytest.raises(ValueError, match="driver timeout"):
        preparation.prepare(repository=tmp_path / "missing-repository",
            instruction=tmp_path / "missing-instruction", state=state, max_total_agent_seconds=timeout)
    assert not state.exists()
