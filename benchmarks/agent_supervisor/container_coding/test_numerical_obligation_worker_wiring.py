"""Actual owner monitor cleanup and worker-parser wiring, with authored children."""
import signal
import sys
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY, OWNER_MONITOR


@pytest.mark.parametrize("failure", [None, "spawn", "registration", "wait", "cleanup", "guard_abort"])
def test_owner_monitor_always_retires_numerical_broker(monkeypatch, failure):
    events = []

    class AuthoredFailure(RuntimeError):
        pass

    class Process:
        waits = 0

        def wait(self, **kwargs):
            self.waits += 1
            events.append("wait")
            if failure == "wait" and self.waits == 1:
                raise AuthoredFailure("authored wait failure")
            return 0

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    process = Process()

    def spawn(command):
        assert command == ["authored-worker"]
        events.append("spawn")
        if failure == "spawn":
            raise AuthoredFailure("authored spawn failure")
        return process

    def cleanup(command, **kwargs):
        assert command[-1] == "--cleanup"
        events.append("cleanup")
        if failure == "cleanup":
            raise AuthoredFailure("authored cleanup failure")
        return SimpleNamespace(returncode=0)

    def register(child):
        assert child is process
        events.append("register")
        if failure == "registration":
            raise AuthoredFailure("authored registration failure")

    def abort():
        events.append("guard_abort")
        if failure == "guard_abort":
            raise AuthoredFailure("authored guard failure")

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(subprocess, "run", cleanup)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    namespace = dict(command=["authored-worker"],
        numerical_obligation_broker=SimpleNamespace(note_spawned=register,
            close=lambda: events.append("broker_close")),
        proof_query_guard=SimpleNamespace(note_spawned=lambda child: events.append("guard_register"), abort=abort))
    with pytest.raises(SystemExit if failure is None else AuthoredFailure) as error:
        exec(compile(OWNER_MONITOR, "<installed-owner-monitor>", "exec"), namespace)
    if failure is None:
        assert error.value.code == 0
    assert events.count("broker_close") == 1 and events[-1] == "broker_close"
    assert "cleanup" in events and "guard_abort" in events
    if failure != "spawn":
        assert events.index("register") < events.index("cleanup")
    if failure in {"registration", "wait"}:
        assert "terminate" in events


@pytest.mark.parametrize("failure", ["factory", "extend", "extend_and_abort"])
def test_owner_startup_retires_guards_before_monitor(monkeypatch, failure):
    events = []

    class AuthoredFailure(RuntimeError):
        pass

    class Command(list):
        def extend(self, extra):
            events.append("extend")
            raise AuthoredFailure("authored argv extension failure")

    broker = SimpleNamespace(extra_argv=["--numerical-obligation-descriptor", "/selected"],
        close=lambda: events.append("broker_close"))

    def factory(**kwargs):
        assert kwargs == dict(argv=["--purpose", "coding"],
            boundary=Path("/owner/container-boundary.json"), workspace=Path("/work"), input_fd=0)
        events.append("factory")
        if failure == "factory":
            raise AuthoredFailure("authored factory failure")
        return broker

    def abort():
        events.append("guard_abort")
        if failure == "extend_and_abort":
            raise AuthoredFailure("authored guard abort failure")

    monkeypatch.setitem(sys.modules,
        "ipfs_accelerate_py.agent_supervisor.runtime.numerical_obligation_broker",
        SimpleNamespace(prepare_numerical_obligation_broker=factory))
    start = OWNER_ENTRY.index("numerical_obligation_broker=None")
    stop = OWNER_ENTRY.index(OWNER_MONITOR)
    namespace = dict(root=Path("/owner"), cwd=Path("/work"),
        handoff={"argv": ["--purpose", "coding"]}, command=Command(["worker"]),
        proof_query_guard=SimpleNamespace(abort=abort))
    with pytest.raises(AuthoredFailure):
        exec(compile(OWNER_ENTRY[start:stop], "<installed-owner-startup>", "exec"), namespace)
    assert events.count("guard_abort") == 1
    if failure == "factory":
        assert events == ["factory", "guard_abort"]
    else:
        assert events == ["factory", "extend", "guard_abort", "broker_close"]
