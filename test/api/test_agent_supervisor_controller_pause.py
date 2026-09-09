"""Real-process source qualification pause, child preservation, and failsafe."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.controller_pause import (
    ControllerPauseError,
    pause_exact_controller,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "pidfd_open"), reason="Linux pidfds required"
)


def _wait(predicate, bound=3):
    end = time.monotonic() + bound
    while not predicate():
        assert time.monotonic() < end, "bounded process observation timed out"
        time.sleep(0.01)


def _state(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return raw[raw.rfind(")") + 2 :].split()[0]


@pytest.fixture
def controller(tmp_path):
    child_code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\nwhile True:\n p.write_text(str(time.monotonic())); time.sleep(.02)"
    code = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); pathlib.Path(sys.argv[3]).write_text(str(p.pid));\nwhile True: time.sleep(.02)"
    heartbeat = tmp_path / "heartbeat"
    childfile = tmp_path / "child"
    process = subprocess.Popen(
        [sys.executable, "-c", code, child_code, str(heartbeat), str(childfile)]
    )
    _wait(lambda: childfile.exists() and heartbeat.exists())
    child = int(childfile.read_text())
    try:
        yield process, child, heartbeat
    finally:
        for pid in [process.pid, child]:
            try:
                os.kill(pid, signal.SIGCONT)
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        process.wait(timeout=3)


def test_pause_preserves_children_and_resumes_after_body_error(controller):
    process, child, heartbeat = controller
    identity = read_process_birth(process.pid)
    guard = pause_exact_controller(identity, max_seconds=3)
    with pytest.raises(RuntimeError, match="qualification failed"), guard as pause:
        pause.require_paused()
        assert _state(process.pid) == "T"
        old = heartbeat.read_text()
        _wait(lambda: heartbeat.read_text() != old)
        assert _state(child) != "T"
        raise RuntimeError("qualification failed")
    _wait(lambda: _state(process.pid) != "T")
    assert read_process_birth(process.pid) == identity


def test_birth_mismatch_never_signals_controller(controller):
    process, _, _ = controller
    identity = read_process_birth(process.pid)
    guard = pause_exact_controller(
        replace(identity, start_time_ticks=identity.start_time_ticks + 1)
    )
    with pytest.raises(ControllerPauseError, match="birth differs"), guard:
        pytest.fail("mismatched process was admitted")
    assert _state(process.pid) != "T"


def test_existing_pause_is_not_resumed(controller):
    process, _, _ = controller
    identity = read_process_birth(process.pid)
    os.kill(process.pid, signal.SIGSTOP)
    _wait(lambda: _state(process.pid) == "T")
    guard = pause_exact_controller(identity)
    with pytest.raises(ControllerPauseError, match="independently runnable"), guard:
        pytest.fail("foreign pause was admitted")
    assert _state(process.pid) == "T"


def test_independent_deadline_resumes_a_stalled_caller(controller):
    process, _, _ = controller
    with pause_exact_controller(
        read_process_birth(process.pid), max_seconds=0.4
    ) as pause:
        assert _state(process.pid) == "T"
        _wait(lambda: _state(process.pid) != "T")
        with pytest.raises(ControllerPauseError, match="deadline"):
            pause.require_paused()


def test_independent_resumer_survives_caller_sigkill(controller, tmp_path):
    process, child, heartbeat = controller
    proof = tmp_path / "paused"
    code = """import sys,time
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth
from ipfs_accelerate_py.agent_supervisor.runtime.controller_pause import pause_exact_controller
with pause_exact_controller(read_process_birth(int(sys.argv[1])),max_seconds=3):
 Path(sys.argv[2]).write_text('ready')
 time.sleep(20)
"""
    worker = subprocess.Popen(
        [sys.executable, "-c", code, str(process.pid), str(proof)]
    )
    try:
        _wait(proof.exists)
        assert _state(process.pid) == "T"
        old = heartbeat.read_text()
        _wait(lambda: heartbeat.read_text() != old)
        worker.kill()
        worker.wait(timeout=3)
        _wait(lambda: _state(process.pid) != "T")
        assert _state(child) != "T"
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=3)


def test_independent_scope_survives_whole_caller_cgroup_kill(controller, tmp_path):
    import json
    import uuid

    process, child, heartbeat = controller
    proof = tmp_path / "scope-proof"
    unit = "ipfs-controller-caller-test-" + uuid.uuid4().hex + ".scope"
    code = """import json,sys,time
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth
from ipfs_accelerate_py.agent_supervisor.runtime.controller_pause import pause_exact_controller
with pause_exact_controller(read_process_birth(int(sys.argv[1])),max_seconds=3) as pause:
 Path(sys.argv[2]).write_text(json.dumps({'caller_cgroup':Path('/proc/self/cgroup').read_text().strip(),'resumer_cgroup':pause.resumer_cgroup,'resumer_unit':pause.resumer_unit}))
 time.sleep(20)
"""
    worker = subprocess.Popen(
        [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--unit",
            unit,
            sys.executable,
            "-c",
            code,
            str(process.pid),
            str(proof),
        ]
    )
    try:
        _wait(proof.exists)
        data = json.loads(proof.read_text())
        assert data["caller_cgroup"].endswith("/" + unit)
        assert data["caller_cgroup"] != data["resumer_cgroup"]
        assert not data["resumer_cgroup"].startswith(data["caller_cgroup"] + "/")
        assert _state(process.pid) == "T"
        old = heartbeat.read_text()
        _wait(lambda: heartbeat.read_text() != old)
        subprocess.run(
            [
                "systemctl",
                "--user",
                "kill",
                "--kill-whom=all",
                "--signal=SIGKILL",
                unit,
            ],
            check=True,
            capture_output=True,
        )
        worker.wait(timeout=3)
        _wait(lambda: _state(process.pid) != "T")
        assert _state(child) != "T"
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=3)


def test_same_cgroup_resumer_is_refused_before_stop(controller, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import controller_pause

    process, _, _ = controller
    original = subprocess.Popen

    def same_cgroup_launcher(argv, **kwargs):
        python_index = argv.index(sys.executable)
        return original(argv[python_index:], **kwargs)

    monkeypatch.setattr(controller_pause.subprocess, "Popen", same_cgroup_launcher)
    guard = pause_exact_controller(read_process_birth(process.pid), max_seconds=1)
    with pytest.raises(ControllerPauseError, match="cgroup admission failed"), guard:
        pytest.fail("same-job resumer was admitted")
    assert _state(process.pid) != "T"


@pytest.mark.parametrize("delay_after_request", [False, True])
def test_delayed_caller_cannot_stop_after_guardian_expiry(
    controller, monkeypatch, delay_after_request
):
    from ipfs_accelerate_py.agent_supervisor.runtime import controller_pause

    process, _, _ = controller
    original = os.write

    def delayed_arm(fd, value):
        if value == b"A":
            if delay_after_request:
                result = original(fd, value)
                time.sleep(0.7)
                return result
            time.sleep(0.7)
        return original(fd, value)

    monkeypatch.setattr(controller_pause.os, "write", delayed_arm)
    guard = pause_exact_controller(read_process_birth(process.pid), max_seconds=0.3)
    with pytest.raises(ControllerPauseError, match="expired|deadline"), guard:
        pytest.fail("late caller acquired an expired pause")
    assert _state(process.pid) != "T"
    time.sleep(0.1)
    assert _state(process.pid) != "T"
