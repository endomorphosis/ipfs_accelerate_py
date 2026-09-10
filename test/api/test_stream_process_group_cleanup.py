"""A completed leader cannot turn unavailable group cleanup into success."""

from __future__ import annotations

import errno
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor_runtime as runtime,
)


def run_direct(tmp_path: Path):
    return runtime.run_process_group_stream(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        timeout_seconds=5,
        termination_grace_seconds=0,
    )


@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EIO, errno.ENOSYS])
@pytest.mark.parametrize("after_signal", [None, signal.SIGTERM, signal.SIGKILL])
def test_probe_errors_never_report_absence(
    tmp_path, monkeypatch, error_number, after_signal
):
    signals = []

    def killpg(pid, signum):
        if signum == 0:
            if after_signal is None or after_signal in signals:
                raise OSError(error_number, "private diagnostic must not be rendered")
        else:
            signals.append(signum)

    monkeypatch.setattr(runtime.os, "killpg", killpg)
    with pytest.raises(
        RuntimeError, match="process group cleanup unverified"
    ) as raised:
        run_direct(tmp_path)
    assert type(raised.value).__name__ == "ProcessGroupCleanupUnverified"
    assert "private" not in str(raised.value)
    expected = []
    if after_signal is not None:
        expected.append(signal.SIGTERM)
    if after_signal == signal.SIGKILL:
        expected.append(signal.SIGKILL)
    assert signals == expected


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EIO])
def test_signal_failure_never_reports_cleanup(
    tmp_path, monkeypatch, signum, error_number
):
    signals = []

    def killpg(pid, actual_signum):
        if actual_signum:
            signals.append(actual_signum)
            if actual_signum == signum:
                raise OSError(error_number, "private signal failure")

    monkeypatch.setattr(runtime.os, "killpg", killpg)
    with pytest.raises(
        RuntimeError, match="process group cleanup unverified"
    ) as raised:
        run_direct(tmp_path)
    assert type(raised.value).__name__ == "ProcessGroupCleanupUnverified"
    assert "private" not in str(raised.value)
    assert signals[-1] == signum


def test_post_kill_survivors_remain_unverified(tmp_path, monkeypatch):
    signals = []
    probes_after_kill = 0

    def killpg(pid, signum):
        nonlocal probes_after_kill
        if signum:
            signals.append(signum)
        elif signal.SIGKILL in signals:
            probes_after_kill += 1

    monkeypatch.setattr(runtime.os, "killpg", killpg)
    with pytest.raises(
        RuntimeError, match="process group cleanup unverified"
    ) as raised:
        run_direct(tmp_path)
    assert type(raised.value).__name__ == "ProcessGroupCleanupUnverified"
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert probes_after_kill > 0


def test_completed_direct_child_keeps_original_return_code(tmp_path):
    result = runtime.run_process_group_stream(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        timeout_seconds=5,
        termination_grace_seconds=0,
    )
    assert result.returncode == 7


@pytest.mark.parametrize("disappear_at", [0, signal.SIGTERM, signal.SIGKILL])
def test_exact_esrch_at_probe_or_signal_allows_return(
    tmp_path, monkeypatch, disappear_at
):
    signalled = []

    def killpg(pid, signum):
        if signum == disappear_at or disappear_at in signalled:
            raise ProcessLookupError(errno.ESRCH, "absent")
        if signum:
            signalled.append(signum)

    monkeypatch.setattr(runtime.os, "killpg", killpg)
    assert run_direct(tmp_path).returncode == 0


@pytest.mark.skipif(sys.platform != "linux", reason="isolated Linux subreaper fixture")
@pytest.mark.parametrize("ignore_term", [False, True])
def test_real_owned_descendant_is_reaped_before_return(tmp_path, ignore_term):
    # Keep subreaper policy inside a separate fixture process. Its only adopted
    # descendant is our own worker, so no unrelated process is waited/signalled.
    harness = textwrap.dedent(
        """
        import ctypes, json, os, pathlib, signal, subprocess, sys, threading, time
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import run_process_group_stream

        assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
        root = pathlib.Path(sys.argv[1])
        ignore_term = sys.argv[2] == 'True'
        marker = root / 'descendant.pid'
        child_code = ('import os,pathlib,signal,time; '
                      + ('signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore_term else '')
                      + 'pathlib.Path(' + repr(str(marker)) + ').write_text(str(os.getpid())); time.sleep(30)')
        leader_code = ('import pathlib,subprocess,sys,time; '
                       + 'subprocess.Popen([sys.executable,"-c",' + repr(child_code) + ']); '
                       + 'p=pathlib.Path(' + repr(str(marker)) + '); '
                       + '\\nwhile not p.exists(): time.sleep(0.001)')
        reaped = threading.Event()
        stop = threading.Event()
        child_pid = []
        leader_pid = []

        def reap():
            while not stop.is_set():
                if marker.exists():
                    try:
                        pid = int(marker.read_text())
                    except ValueError:
                        continue
                    if not child_pid:
                        child_pid.append(pid)
                    try:
                        got, _ = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        got = 0
                    if got == pid:
                        reaped.set()
                        return
                time.sleep(0.001)

        thread = threading.Thread(target=reap, daemon=True)
        thread.start()
        try:
            result = run_process_group_stream(
                [sys.executable, '-c', leader_code], cwd=root,
                stdout=subprocess.DEVNULL, timeout_seconds=5,
                termination_grace_seconds=0.2,
                on_started=lambda process: leader_pid.append(process.pid),
            )
            assert result.returncode == 0
            try:
                os.killpg(leader_pid[0], 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError('owned group still exists after returned cleanup')
            assert reaped.wait(0.05)
            print(json.dumps({'leader_returncode':result.returncode,'descendant_reaped':True,'group_absent':True}))
        finally:
            if child_pid and not reaped.is_set():
                try:
                    os.kill(child_pid[0], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                reaped.wait(2)
            stop.set()
            thread.join(2)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", harness, str(tmp_path), str(ignore_term)],
        cwd=Path(runtime.__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "leader_returncode": 0,
        "descendant_reaped": True,
        "group_absent": True,
    }
