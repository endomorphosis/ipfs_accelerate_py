"""Actual children exercise owner ordering; admission is tested separately."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_monitor as monitor
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY, WORKER_ENTRY


def _run(monkeypatch, *, first=7, second=0, failure=None):
    from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as lifecycle
    events, children = [], []
    signals = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}

    class Broker:
        def __init__(self, number):
            self.number = number
            self._process = None
        def note_spawned(self, child):
            events.append(("registered", self.number))
            assert type(child) is subprocess.Popen
            self._process = child
            children.append(child)
            if failure == "registration" and self.number == 1:
                raise ValueError("authored registration failure")
            if failure == "interrupted" and self.number == 1:
                wait = child.wait
                calls = []
                def interrupt_once(**kwargs):
                    calls.append(kwargs)
                    if len(calls) == 1:
                        raise InterruptedError("authored interrupted owner")
                    return wait(**kwargs)
                child.wait = interrupt_once

    command = [sys.executable, "-I", "-c", f"raise SystemExit({first})"]
    if first < 0:
        command[-1] = f"import os;os.kill(os.getpid(), {-first})"
    class Launch:
        source_broker = Broker(1)
        remaining_seconds = 3
        def revalidate(self):
            events.append(("validated", 1))
        def validate_first_command(self, value):
            assert value == command
            if failure == "command":
                raise ValueError("authored command failure")

    launch = Launch()
    if failure == "timeout":
        command[-1] = "import time;time.sleep(5)"
        launch.remaining_seconds = .05

    def retire(*, launch, process, source_broker):
        number = source_broker.number
        events.append(("retired", number))
        assert all(signal.getsignal(value) is signal.SIG_IGN for value in signals)
        if process is not None:
            assert process.poll() is not None
        if failure == "cleanup" and number == 1 or failure == "second_cleanup" and number == 2:
            raise ValueError("authored cleanup failure")
        return (number, process)

    def prepare(*, launch, retired_worker):
        events.append(("prepared", 2))
        assert retired_worker[0] == 1 and retired_worker[1].poll() == first
        assert events[-2] == ("retired", 1)
        if failure == "prepare":
            raise ValueError("authored preparation failure")
        class Prepared:
            source_broker = Broker(2)
            handoff = SimpleNamespace(close=lambda: events.append(("handoff_closed", 2)))
            def start(self):
                events.append(("started", 2))
                if failure == "start":
                    raise ValueError("authored start failure")
                broker = self.source_broker
                child = subprocess.Popen([sys.executable, "-I", "-c", f"raise SystemExit({second})"])
                broker.note_spawned(child)
                if failure == "return_window":
                    raise InterruptedError("authored interruption after child registration")
                return SimpleNamespace(process=child, source_broker=broker)
            def close(self):
                events.append(("closed", 2))
                if failure == "second_cleanup":
                    raise ValueError("authored unretired second child")
        return Prepared()

    monkeypatch.setattr(lifecycle, "CapturedCodingLaunch", Launch)
    monkeypatch.setattr(lifecycle, "retire_coding_worker", retire)
    monkeypatch.setattr(lifecycle, "prepare_second_coding_worker", prepare)
    outcome = None
    try:
        outcome = monitor.run_candidate_coding_rounds(launch=launch, command=command)
    except BaseException as error:
        outcome = error
    finally:
        for child in children:
            if child.poll() is None:
                child.kill(); child.wait()
    assert all(signal.getsignal(number) == value for number, value in signals.items())
    return outcome, events, children


def _observation(capsys):
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("[owner-candidate-rounds] ")
    value = json.loads(lines[0].split(" ", 1)[1])
    assert value["completion_authority"] is value["proof_authority"] is False
    return value


def test_second_child_only_after_first_retirement(monkeypatch, capsys):
    result, events, children = _run(monkeypatch)
    assert result == 0 and len(children) == 2 and children[0].pid != children[1].pid
    assert events == [("validated", 1), ("registered", 1), ("retired", 1),
        ("prepared", 2), ("started", 2), ("registered", 2), ("retired", 2), ("closed", 2), ("handoff_closed", 2)]
    observation = _observation(capsys)
    assert [row["returncode"] for row in observation["rounds"]] == [7, 0]
    assert all(row["wait_completed"] and row["retirement_observed"] for row in observation["rounds"])


def test_success_does_not_spend_a_second_round(monkeypatch, capsys):
    result, events, children = _run(monkeypatch, first=0)
    assert result == 0 and len(children) == 1
    assert ("prepared", 2) not in events
    assert len(_observation(capsys)["rounds"]) == 1


def test_second_failure_is_terminal_without_third_round(monkeypatch, capsys):
    result, events, children = _run(monkeypatch, second=9)
    assert result == 9 and len(children) == 2
    assert events.count(("prepared", 2)) == 1
    assert [row["returncode"] for row in _observation(capsys)["rounds"]] == [7, 9]


@pytest.mark.parametrize("failure", ["registration", "command", "cleanup", "timeout", "interrupted", "prepare", "start", "return_window", "second_cleanup"])
def test_failure_cannot_trigger_extra_children_and_restores_handlers(monkeypatch, capsys, failure):
    result, events, children = _run(monkeypatch, failure=failure)
    assert isinstance(result, (ValueError, InterruptedError, subprocess.TimeoutExpired))
    assert len(children) <= (2 if failure in {"return_window", "second_cleanup"} else 1) and ("retired", 1) in events
    if failure == "return_window":
        assert events[-3:] == [("retired", 2), ("closed", 2), ("handoff_closed", 2)]
    if failure == "second_cleanup":
        assert str(result) == "authored cleanup failure"
        assert events[-1] == ("handoff_closed", 2)
    if failure in {"registration", "command", "cleanup", "timeout", "interrupted"}:
        assert ("prepared", 2) not in events
    if failure == "start":
        assert events[-2:] == [("closed", 2), ("handoff_closed", 2)]
    value = _observation(capsys)
    assert value["outcome"] == "failed" and "authored" not in json.dumps(value)


def test_signalled_worker_never_admits_continuation(monkeypatch, capsys):
    result, events, children = _run(monkeypatch, first=-signal.SIGTERM)
    assert result == -signal.SIGTERM and len(children) == 1
    assert ("prepared", 2) not in events
    assert _observation(capsys)["rounds"][0]["returncode"] == -signal.SIGTERM


def _worker_arguments(monkeypatch, extra):
    start = WORKER_ENTRY.index("parser=argparse.ArgumentParser()")
    end = WORKER_ENTRY.index("from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner")
    code = WORKER_ENTRY[start:end]
    monkeypatch.setattr(sys, "argv", ["worker-entry", "--semantic-repository", "/canonical", *extra])
    namespace = dict(argparse=argparse, pathlib=__import__("pathlib"),
        sys=sys,
        selected_provider="codex_cli")
    exec(compile(code, "<installed-candidate-worker-options>", "exec"), namespace)
    return namespace["args"]


def test_installed_worker_accepts_exact_new_handoff(monkeypatch):
    flags = ["--candidate-dispatch-handoff", "/owner/dispatch.json", "--candidate-dispatch-handoff-sha256", "a"*64,
        "--candidate-round-descriptor", "/owner/round.json", "--candidate-round-descriptor-sha256", "b"*64]
    args = _worker_arguments(monkeypatch, flags)
    assert args.candidate_dispatch_handoff == Path("/owner/dispatch.json")
    assert args.source_view_profile == "disabled"


@pytest.mark.parametrize("extra", [["--preflight"], ["--purpose", "planning"],
    ["--public-instruction-artifact", "/owner/instruction"],
    ["--doctor-residual-artifact", "/owner/residual"], ["--source-view-profile", "identity@1"],
    ["--coding-reply-mode", "ordinary-completion@1"], ["--semantic-working-set", "capsule-references@1"]])
def test_installed_worker_refuses_mixed_candidate_context(monkeypatch, extra):
    flags = ["--candidate-dispatch-handoff", "/owner/dispatch.json", "--candidate-dispatch-handoff-sha256", "a"*64,
        "--candidate-round-descriptor", "/owner/round.json", "--candidate-round-descriptor-sha256", "b"*64]
    with pytest.raises(SystemExit):
        _worker_arguments(monkeypatch, [*flags, *extra])


def test_owner_selection_is_removed_before_original_handoff_and_captured_before_monitor():
    compile(OWNER_ENTRY, "<installed-owner-entry>", "exec")
    compile(WORKER_ENTRY, "<installed-worker-entry>", "exec")
    assert OWNER_ENTRY.index("split_owner_candidate_rounds(list(sys.argv[1:]))") < OWNER_ENTRY.index("handoff=prepare_container_source_formalization_dispatch")
    assert "argv=owner_initial_argv" in OWNER_ENTRY
    assert OWNER_ENTRY.index("owner_candidate_launch=capture_original_coding_launch") < OWNER_ENTRY.index("raise SystemExit(run_candidate_coding_rounds")


@pytest.mark.parametrize("argv,optional_import", [(["--model", "authored"], False), (["--candidate-rounds", "2"], True)])
def test_default_owner_remains_usable_with_older_runtime_archives(argv, optional_import):
    import builtins
    code = OWNER_ENTRY[OWNER_ENTRY.index("owner_candidate_rounds=1"):OWNER_ENTRY.index("owner_candidate_round_deadline=None")]
    native_import = builtins.__import__
    def old_archive(name, *args, **kwargs):
        if name.endswith("candidate_round_policy"):
            raise ModuleNotFoundError("authored older runtime")
        return native_import(name, *args, **kwargs)
    namespace = dict(sys=SimpleNamespace(argv=["worker-entry", *argv]),
        __builtins__={**vars(builtins), "__import__": old_archive})
    if optional_import:
        with pytest.raises(ModuleNotFoundError):
            exec(compile(code, "<installed-owner-round-selection>", "exec"), namespace)
    else:
        exec(compile(code, "<installed-owner-round-selection>", "exec"), namespace)
        assert namespace["owner_candidate_rounds"] == 1 and namespace["owner_initial_argv"] == argv
