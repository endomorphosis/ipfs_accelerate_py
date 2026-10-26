"""A shared provider lease cannot restart after child birth or callbacks."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime


def clock(monkeypatch):
    value = [100.0]
    proxy = SimpleNamespace(**{name: getattr(runtime.time, name) for name in dir(runtime.time)})
    proxy.monotonic = lambda: value[0]
    monkeypatch.setattr(runtime, "time", proxy)
    return value


def invoke(tmp_path, **kwargs):
    return runtime.run_process_group_stream(
        [sys.executable, "-c", "import time; time.sleep(0.15)"], cwd=tmp_path,
        stdout=subprocess.DEVNULL, timeout_seconds=20, termination_grace_seconds=0.1,
        **kwargs,
    )


@pytest.mark.parametrize("deadline", [True, False, "110", [], {}, float("nan"),
    float("inf"), float("-inf"), 10 ** 500, -(10 ** 500)])
def test_invalid_deadline_never_launches(tmp_path, monkeypatch, deadline):
    monkeypatch.setattr(runtime, "launch_process_child", lambda *a, **k: pytest.fail("invalid lease launched"))
    with pytest.raises(ValueError, match="finite monotonic"):
        invoke(tmp_path, absolute_deadline=deadline)


@pytest.mark.parametrize("deadline", [0, 99.0, 100.0])
def test_expired_deadline_never_launches(tmp_path, monkeypatch, deadline):
    clock(monkeypatch)
    monkeypatch.setattr(runtime, "launch_process_child", lambda *a, **k: pytest.fail("expired lease launched"))
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        invoke(tmp_path, absolute_deadline=deadline)
    assert caught.value.timeout_reason == "absolute_deadline"
    assert caught.value.absolute_deadline == deadline
    assert caught.value.elapsed_seconds == 0
    assert caught.value.progress_events == 0
    assert caught.value.max_timeout_seconds == 20


@pytest.mark.parametrize("mode", ["communicate", "progress", "observer"])
@pytest.mark.parametrize("bounded", [False, True])
def test_birth_callback_time_is_deducted_from_actual_wait(tmp_path, monkeypatch, mode, bounded):
    now = clock(monkeypatch)
    waits = []

    def born(process):
        now[0] = 107.0
        method = "communicate" if mode == "communicate" else "wait"
        original = getattr(process, method)
        def observed(*args, **kwargs):
            waits.append(kwargs["timeout"])
            return original(*args, **kwargs)
        setattr(process, method, observed)

    options = {"on_started": born}
    if bounded:
        options["absolute_deadline"] = 110.0
    if mode == "progress":
        options.update(progress_timeout_seconds=10, max_timeout_seconds=20, progress_poll_seconds=5)
    elif mode == "observer":
        options.update(on_progress=lambda observation: None, progress_poll_seconds=5)
    assert invoke(tmp_path, **options).returncode == 0
    assert waits
    assert waits[0] == (3.0 if bounded else 20.0 if mode == "communicate" else 5.0)


@pytest.mark.skipif(sys.platform != "linux", reason="waitid WNOWAIT observes exit without reaping")
@pytest.mark.parametrize("child_state", ["running", "exited", "reaped"])
@pytest.mark.parametrize("expiry_stage", ["launch", "birth_callback"])
def test_expiry_during_birth_refuses_success_and_reaps_real_child(
    tmp_path, monkeypatch, child_state, expiry_stage
):
    now = clock(monkeypatch)
    children = []
    original_launch = runtime.launch_process_child

    def launch(*args, **kwargs):
        process = original_launch(*args, **kwargs)
        children.append(process)
        if expiry_stage == "launch":
            now[0] = 111.0
        return process

    def born(process):
        if child_state != "running":
            # Waiting without reaping represents a CLI that exits while owner
            # birth persistence is still busy. It must not become late success.
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
            if child_state == "reaped":
                process.wait(timeout=5)
        else:
            until = time.monotonic() + 5
            while not (tmp_path / "ready").exists():
                assert time.monotonic() < until, "authored child failed to start"
                time.sleep(0.001)
        now[0] = 111.0

    monkeypatch.setattr(runtime, "launch_process_child", launch)
    code = "from pathlib import Path; import time; Path('ready').write_text('ready')"
    if child_state == "running":
        code += "; time.sleep(30)"
    original_terminate = runtime.terminate_pid_tree
    def terminate(pid, **kwargs):
        assert children[0].returncode is None, "reaped numeric PID must not be reused for cleanup"
        result = original_terminate(pid, **kwargs)
        now[0] = 125.0  # Cleanup time must not inflate provider elapsed time.
        return result
    monkeypatch.setattr(runtime, "terminate_pid_tree", terminate)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            runtime.run_process_group_stream(
                [sys.executable, "-c", code], cwd=tmp_path, stdout=subprocess.DEVNULL,
                timeout_seconds=20, absolute_deadline=110.0, on_started=born,
                termination_grace_seconds=0.1,
            )
        assert caught.value.timeout_reason == "absolute_deadline"
        assert caught.value.absolute_deadline == 110.0
        assert caught.value.elapsed_seconds == 11.0
        assert caught.value.progress_events == 0
        assert caught.value.max_timeout_seconds == 20
        assert len(children) == 1 and children[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(children[0].pid, 0)
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_progress_callback_expiry_after_reap_uses_owned_group_cleanup(tmp_path, monkeypatch):
    now = clock(monkeypatch)
    children = []
    observed = []
    def born(process):
        children.append(process)
        now[0] = 107.0
    def progress(observation):
        observed.append(observation)
        assert children[0].returncode == 0
        now[0] = 111.0
    monkeypatch.setattr(runtime, "terminate_pid_tree",
        lambda *a, **k: pytest.fail("leader was reaped before progress callback"))
    with (tmp_path / "child.log").open("w") as stream:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            runtime.run_process_group_stream(
                [sys.executable, "-c", "import time; time.sleep(0.15); print('authored completion', flush=True)"],
                cwd=tmp_path, stdout=stream, timeout_seconds=20, absolute_deadline=110.0,
                on_started=born, on_progress=progress, progress_poll_seconds=5,
                termination_grace_seconds=0.1,
            )
    assert observed
    assert caught.value.timeout_reason == "absolute_deadline"
    assert caught.value.absolute_deadline == 110.0
    assert caught.value.elapsed_seconds == 11.0
    assert caught.value.progress_events == len(observed) == 1
    assert observed[0]["elapsed_seconds"] == 7.0
    with pytest.raises(ProcessLookupError):
        os.killpg(children[0].pid, 0)


def test_successful_communicate_observed_after_deadline_is_not_success(tmp_path, monkeypatch):
    now = clock(monkeypatch)
    children = []
    def born(process):
        children.append(process)
        original = process.communicate
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            assert process.returncode == 0
            now[0] = 111.0
            return result
        process.communicate = observed
    monkeypatch.setattr(runtime, "terminate_pid_tree",
        lambda *a, **k: pytest.fail("successful communicate already reaped the leader"))
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        invoke(tmp_path, absolute_deadline=110.0, on_started=born)
    assert caught.value.timeout_reason == "absolute_deadline"
    assert caught.value.elapsed_seconds == 11.0
    assert caught.value.progress_events == 0
    with pytest.raises(ProcessLookupError):
        os.killpg(children[0].pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="owned Linux process-group fixture")
@pytest.mark.parametrize("mode", ["communicate", "observer", "idle_progress"])
def test_real_absolute_timeout_preserves_elapsed_and_progress_before_cleanup(tmp_path, mode):
    children = []
    observed = []
    options = {}
    if mode != "communicate":
        options.update(on_progress=observed.append, progress_poll_seconds=0.02)
    if mode == "idle_progress":
        options.update(progress_timeout_seconds=2, max_timeout_seconds=5)
    deadline = time.monotonic() + 0.75
    try:
        with (tmp_path / "progress.log").open("w") as stream:
            with pytest.raises(subprocess.TimeoutExpired) as caught:
                runtime.run_process_group_stream(
                    [sys.executable, "-u", "-c",
                     "import time; print('synthetic progress one',flush=True); "
                     "time.sleep(0.05); print('synthetic progress two',flush=True); time.sleep(30)"],
                    cwd=tmp_path, stdout=stream, timeout_seconds=5,
                    absolute_deadline=deadline, on_started=children.append,
                    termination_grace_seconds=0.1, **options,
                )
        failure = caught.value
        assert failure.timeout_reason == "absolute_deadline"
        assert failure.absolute_deadline == deadline
        assert 0.5 <= failure.elapsed_seconds < 3
        assert failure.max_timeout_seconds == 5
        assert failure.progress_timeout_seconds == (2 if mode == "idle_progress" else None)
        if mode == "communicate":
            assert failure.progress_events == 0
        else:
            assert failure.progress_events == len(observed) >= 1
            assert 0 < observed[-1]["elapsed_seconds"] <= failure.elapsed_seconds
        assert len(children) == 1 and children[0].returncode is not None
        assert "synthetic progress" in (tmp_path / "progress.log").read_text()
        with pytest.raises(ProcessLookupError):
            os.killpg(children[0].pid, 0)
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
