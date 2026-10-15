"""Exercise the real driver deadline in child processes with owned fake services."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = r'''
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

root, mode, repository = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path[:0] = [str(repository), str(repository.parent / "ipfs_datasets"),
               str(repository.parent / "ipfs_kit")]
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction

assert Path(driver.__file__).resolve() == repository / "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py"
assert not issubclass(driver._AgentWorkBudgetExpired, Exception)
driver.ROOT = root
driver.WORKTREES = root / "worktrees"
driver.os.geteuid = lambda: 1000
state = root / "state/task"
instruction = root / "instruction.md"
instruction.write_text("Authored public deadline fixture.\n")
events = []
previous_alarm = signal.getsignal(signal.SIGALRM)
previous_term = signal.getsignal(signal.SIGTERM)
real_monotonic = time.monotonic
advance_clock = False
driver.time.monotonic = lambda: real_monotonic() + (90 if advance_clock else 0)

def interrupt_optional_work():
    try:
        if mode == "term_during_prepare":
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            signal.setitimer(signal.ITIMER_REAL, .02)
            time.sleep(2)
    except Exception:
        events.append("deadline_swallowed")
    events.append("continued_after_deadline")

def prepare(**kwargs):
    global advance_clock
    events.append("prepare")
    if mode in {"alarm_during_prepare", "term_during_prepare"}:
        interrupt_optional_work()
    if mode in {"KeyboardInterrupt", "SystemExit"}:
        raise KeyboardInterrupt() if mode == "KeyboardInterrupt" else SystemExit(7)
    if mode == "ordinary_optional_error":
        try:
            raise ValueError("authored optional component unavailable")
        except Exception:
            events.append("ordinary_optional_error_handled")
    state.mkdir()
    (state / "admission.json").write_text("{}\n")
    if mode == "remaining_budget":
        advance_clock = True
    return {"intent_preplanning": {"status": "authored"}}

def plan(**kwargs):
    events.append("plan")
    return {"qualified": True}

task = SimpleNamespace(task_cid="authored-task", task_key="authored-key")
driver.preparation.prepare = prepare
driver.preparation.plan = plan
driver.verify_local_benchmark_admission = lambda *args, **kwargs: {
    "graph": SimpleNamespace(tasks=[task]),
    "manifest": {"repository_cid": "authored-repository", "sources": {
        driver.preparation.INSTRUCTION: {"sha256": "a" * 64}}}}
driver.implementation_argv = lambda **kwargs: ["authored-worker"]
router_public_instruction.prepare_public_instruction_context = lambda **kwargs: {
    "artifact": "authored-instruction", "sha256": "b" * 64}

class Runtime:
    state = root / "owned-runtime"
    profile = SimpleNamespace()
    process = SimpleNamespace(snapshot=lambda profile: SimpleNamespace(members=[]))

    def start(self):
        events.append("start")
        return SimpleNamespace(to_dict=lambda: {"status": "succeeded"})

    def observe(self):
        events.append("observe")
        return {"status": "authored"}

    def stop(self):
        events.append("stop")
        if mode == "alarm_during_stop":
            interrupt_optional_work()
        return SimpleNamespace(to_dict=lambda: {"status": "succeeded"})

    def close(self):
        events.append("close")

def get_task(task_cid):
    if mode == "alarm_during_coding":
        interrupt_optional_work()
    return SimpleNamespace(status="completed", revision=1)

owner = SimpleNamespace(server=object(), source=SimpleNamespace(get_task=get_task))
driver.open_existing_native_owner = lambda **kwargs: contextlib.nullcontext(owner)
driver.AdmittedBenchmarkRuntime.create = lambda *args, **kwargs: Runtime()

def cleanup(argv, **kwargs):
    assert argv[-1] == "--cleanup", "fixture refused an unexpected subprocess"
    events.append("worker_cleanup")
    return SimpleNamespace(returncode=0)

driver.subprocess.run = cleanup
propagated = None
try:
    result = driver.run(instruction=instruction, state=state, arm="no-index", timeout_seconds=90)
except (KeyboardInterrupt, SystemExit) as error:
    propagated = type(error).__name__
    result = json.loads((state.parent / "task-result.json").read_text())
persisted = json.loads((state.parent / "task-result.json").read_text())
assert persisted == result
assert signal.getsignal(signal.SIGALRM) is previous_alarm
assert signal.getsignal(signal.SIGTERM) is previous_term
assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
assert result["worker_cleanup_returncode"] == 0
assert result["provider_invocations"] == []
assert result["provider_usage_accounting"]["report_provider_total_tokens"] is None
assert result["reserved_cleanup_seconds"] == 40 and result["work_cutoff_seconds"] == 50
print(json.dumps({"report": result, "events": events, "propagated": propagated}))
'''


def _run_child(tmp_path, mode):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), mode, str(REPOSITORY)],
        cwd=REPOSITORY, env=env, text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout.splitlines()[-1])


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
@pytest.mark.parametrize("mode", ["alarm_during_prepare", "term_during_prepare"])
def test_signal_deadline_escapes_optional_handlers_and_persists_accounting(tmp_path, mode):
    observed = _run_child(tmp_path, mode)
    report = observed["report"]
    assert report["error"]["type"] == "_AgentWorkBudgetExpired"
    assert report["task_completed"] is False
    assert observed["events"] == ["prepare", "worker_cleanup"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
def test_remaining_budget_refuses_planning_and_still_finalizes(tmp_path):
    observed = _run_child(tmp_path, "remaining_budget")
    assert observed["report"]["error"]["type"] == "_AgentWorkBudgetExpired"
    assert observed["report"]["task_completed"] is False
    assert observed["events"] == ["prepare", "worker_cleanup"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
def test_work_deadline_during_coding_still_stops_closes_and_finalizes(tmp_path):
    observed = _run_child(tmp_path, "alarm_during_coding")
    report = observed["report"]
    assert report["error"]["type"] == "_AgentWorkBudgetExpired"
    assert report["task_completed"] is False
    assert report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0
    assert observed["events"] == ["prepare", "plan", "start", "stop", "close", "worker_cleanup"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
def test_cleanup_deadline_cannot_be_swallowed_or_suppress_final_report(tmp_path):
    observed = _run_child(tmp_path, "alarm_during_stop")
    report = observed["report"]
    assert report["error"]["type"] == "_AgentWorkBudgetExpired"
    assert report["task_completed"] is False
    assert observed["events"] == ["prepare", "plan", "start", "observe", "stop", "close", "worker_cleanup"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
def test_ordinary_optional_error_can_still_fail_open(tmp_path):
    observed = _run_child(tmp_path, "ordinary_optional_error")
    assert observed["report"]["task_completed"] is True
    assert "error" not in observed["report"]
    assert observed["events"] == ["prepare", "ordinary_optional_error_handled", "plan", "start",
                                  "observe", "stop", "close", "worker_cleanup"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX signals")
@pytest.mark.parametrize("exception", ["KeyboardInterrupt", "SystemExit"])
def test_unrelated_base_exceptions_propagate_after_finalization(tmp_path, exception):
    observed = _run_child(tmp_path, exception)
    assert observed["propagated"] == exception
    assert observed["report"]["task_completed"] is False
    assert "error" not in observed["report"]
    assert observed["events"] == ["prepare", "worker_cleanup"]
