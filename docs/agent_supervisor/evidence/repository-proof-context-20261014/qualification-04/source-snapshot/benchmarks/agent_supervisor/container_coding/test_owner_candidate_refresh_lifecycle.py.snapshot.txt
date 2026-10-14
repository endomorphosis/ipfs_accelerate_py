"""Authored native children qualify owner refresh ordering without model calls."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY, OWNER_MONITOR


PREFIX = "[owner-candidate-refresh] "
HELPER_MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.owner_candidate_round_checkpoint"


def _monitor(monkeypatch, *, returncode=7, failure=None, deadline=None, with_source=True):
    """Run production monitor with actual workers and an authored cleanup child."""
    events, processes, refreshes = [], [], []
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    handlers = {}
    original_run = subprocess.run

    def register(name, process):
        events.append(name + "_register")
        if not processes:
            processes.append(process)
        assert process is processes[0]
        if failure == name + "_register":
            raise RuntimeError("authored registration failure")
        if failure == "interrupted" and name == "source":
            raise InterruptedError("authored interrupted owner")
        if failure == "wait" and name == "source":
            original_wait = process.wait
            waits = []
            def wait(**kwargs):
                waits.append(kwargs)
                if len(waits) == 1:
                    raise RuntimeError("authored wait failure")
                return original_wait(**kwargs)
            process.wait = wait

    def close(name):
        events.append(name + "_close")
        assert all(handlers[signum] is signal.SIG_IGN for signum in signals)
        if failure == "custody" and name == "source":
            processes[0].poll = lambda: None
        if failure == name + "_close":
            raise RuntimeError("authored retirement failure")

    source = SimpleNamespace(note_spawned=lambda process: register("source", process),
        close=lambda: close("source")) if with_source else None
    numerical = SimpleNamespace(note_spawned=lambda process: register("numerical", process),
        close=lambda: close("numerical"))

    def cleanup(command, **kwargs):
        assert command[-1] == "--cleanup"
        events.append("cleanup")
        assert all(handlers[signum] is signal.SIG_IGN for signum in signals)
        if processes:
            assert processes[0].poll() is not None
        if failure == "cleanup_raises":
            raise RuntimeError("authored cleanup failure")
        if failure == "deadline_during_cleanup":
            monkeypatch.setattr(time, "monotonic", lambda: deadline + 1)
        # This is a real second process, representing only isolated cleanup.
        # It executes no provider and receives no benchmark source or prompt.
        return original_run([sys.executable, "-c", "raise SystemExit(1)" if failure == "cleanup" else "pass"],
            **kwargs)

    def refresh(**kwargs):
        events.append("refresh")
        refreshes.append(kwargs)
        assert kwargs["source_broker"] is source
        assert kwargs["prior_returncode"] == returncode
        assert kwargs["absolute_deadline"] == deadline
        assert processes[0].poll() == returncode
        assert events[-4:] == ["cleanup", "numerical_close", "source_close", "refresh"]
        assert all(callable(handlers[signum]) and handlers[signum].__name__ == "interrupted"
            for signum in signals)
        if failure == "interrupted_refresh":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        if failure == "refresh":
            raise ValueError("PRIVATE authored details must never appear")
        return {"schema": "owner-candidate-refresh-observation@1", "status": "prepared",
            "reason_code": "candidate_context_prepared", "provider_calls": 0,
            "completion_authority": False, "automatic_retry_admitted": False}

    helper = ModuleType(HELPER_MODULE)
    helper.persist_failed_worker_candidate_context = refresh
    monkeypatch.setitem(sys.modules, HELPER_MODULE, helper)
    monkeypatch.setattr(subprocess, "run", cleanup)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    worker_code = (f"import os,signal;os.kill(os.getpid(),{-returncode})" if returncode < 0
        else f"raise SystemExit({returncode})")
    namespace = dict(command=[sys.executable, "-c", worker_code],
        worker_source_broker=source, numerical_obligation_broker=numerical,
        owner_candidate_round_deadline=deadline)
    if failure == "spawn":
        def refuse_spawn(*args, **kwargs):
            raise RuntimeError("authored spawn failure")
        monkeypatch.setattr(subprocess, "Popen", refuse_spawn)
    try:
        exec(compile(OWNER_MONITOR, "<installed-owner-monitor>", "exec"), namespace)
    except BaseException as error:
        return error, events, processes, refreshes
    raise AssertionError("owner monitor must terminate")


def test_waited_failed_worker_refreshes_once_after_retirement(monkeypatch, capsys):
    error, events, processes, refreshes = _monitor(monkeypatch, deadline=time.monotonic() + 30)
    assert type(error) is SystemExit and error.code == 7
    assert len(processes) == len(refreshes) == 1
    assert events == ["numerical_register", "source_register", "cleanup",
        "numerical_close", "source_close", "refresh"]
    assert processes[0].returncode == 7
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith(PREFIX)
    observation = json.loads(lines[0][len(PREFIX):])
    assert observation["status"] == "prepared"
    assert observation["provider_calls"] == 0
    assert observation["completion_authority"] is False
    assert observation["automatic_retry_admitted"] is False


@pytest.mark.parametrize("failure", ["spawn", "source_register", "numerical_register",
    "interrupted", "wait", "cleanup", "cleanup_raises", "source_close", "numerical_close"])
def test_unknown_interrupted_or_unretired_worker_never_refreshes(monkeypatch, capsys, failure):
    error, events, processes, refreshes = _monitor(monkeypatch,
        deadline=time.monotonic() + 30, failure=failure)
    assert not refreshes and "refresh" not in events
    assert capsys.readouterr().out == ""
    if failure == "cleanup":
        assert type(error) is SystemExit and error.code == 1
    else:
        assert isinstance(error, (RuntimeError, InterruptedError))
    assert all(process.poll() is not None for process in processes)


@pytest.mark.parametrize("returncode,with_source,deadline", [
    (0, True, "future"), (7, False, "future"), (7, True, None), (-signal.SIGTERM, True, "future"),
])
def test_success_validation_or_missing_budget_does_not_refresh(monkeypatch, capsys,
        returncode, with_source, deadline):
    selected = time.monotonic() + 30 if deadline == "future" else deadline
    error, events, processes, refreshes = _monitor(monkeypatch, returncode=returncode,
        deadline=selected, with_source=with_source)
    assert type(error) is SystemExit and error.code == returncode
    assert not refreshes and "refresh" not in events
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", [None, "refresh", "interrupted_refresh"])
def test_refresh_refusal_keeps_failed_exit_and_emits_only_closed_metadata(monkeypatch, capsys, failure):
    deadline = time.monotonic() + (30 if failure else -1)
    error, _, _, refreshes = _monitor(monkeypatch, deadline=deadline, failure=failure)
    assert type(error) is SystemExit and error.code == 7
    assert len(refreshes) == (1 if failure else 0)
    output = capsys.readouterr().out
    assert "PRIVATE" not in output and output.startswith(PREFIX)
    assert "owner monitor interrupted" not in output
    observation = json.loads(output[len(PREFIX):])
    assert observation["status"] == ("failed" if failure else "skipped")
    assert observation["reason_code"] == ("refresh_failed" if failure else "deadline_exhausted")
    assert observation["provider_calls"] == observation["model_inference_calls"] == 0
    assert observation["source_obligations_discharged"] == 0
    for key in ("proof_authority", "execution_authority", "completion_authority",
            "retry_authority", "settlement_authority", "automatic_retry_admitted"):
        assert observation[key] is False


@pytest.mark.parametrize("failure", ["custody", "deadline_during_cleanup"])
def test_custody_and_budget_rechecked_after_cleanup(monkeypatch, capsys, failure):
    error, _, _, refreshes = _monitor(monkeypatch, deadline=time.monotonic() + 30,
        failure=failure)
    assert type(error) is SystemExit and error.code == 7
    assert not refreshes
    output = capsys.readouterr().out
    observation = json.loads(output[len(PREFIX):])
    assert observation["reason_code"] == ("refresh_failed" if failure == "custody" else "deadline_exhausted")
    assert observation["automatic_retry_admitted"] is False


@pytest.mark.parametrize("argv,expected", [
    ([], 90), (["--timeout", "45"], 45), (["--timeout=120"], 120),
    (["--timeout", "1800"], 1800), (["--timeout", "0"], None),
    (["--timeout=-1"], None), (["--timeout", "1801"], None),
    (["--timeout", "nan"], None), (["--timeout", "1.5"], None),
    (["--timeout"], None), (["--timeout", "30", "--timeout=30"], None),
])
def test_refresh_deadline_uses_original_owner_start_and_safe_timeout(monkeypatch, argv, expected):
    start = OWNER_ENTRY.index("# The fixed lease includes")
    end = OWNER_ENTRY.index("from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context")
    namespace = {"owner_candidate_round_started": 123.25, "sys": sys}
    monkeypatch.setattr(sys, "argv", ["router-worker", *argv])
    exec(compile(OWNER_ENTRY[start:end], "<installed-owner-deadline>", "exec"), namespace)
    assert namespace["owner_candidate_round_deadline"] == (None if expected is None else 123.25 + expected)
    assert OWNER_ENTRY.index("owner_candidate_round_started=time.monotonic()") < OWNER_ENTRY.index("root=pathlib.Path")
    assert OWNER_ENTRY.index("owner_candidate_round_deadline=") < OWNER_ENTRY.index("handoff=prepare_container_source_formalization_dispatch")
