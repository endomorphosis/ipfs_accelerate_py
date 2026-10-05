"""Git status bounds apply while receiving bytes, before recovery parsing."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import bounded_git_status as capture


def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


@pytest.mark.parametrize("bound", ["bytes", "records"])
def test_actual_git_status_refuses_capture_bound(tmp_path, monkeypatch, bound):
    root = repository(tmp_path)
    for index in range(4):
        (root / ("authored-file-" + str(index))).write_text("fixture\n")
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_MAX_STDOUT_BYTES", 16 if bound == "bytes" else 4096,
                        raising=False)
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_MAX_RECORDS", 1 if bound == "records" else 100,
                        raising=False)
    with pytest.raises(ValueError, match="status.*(byte|record).*bound"):
        runner._plan_bound_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")


def test_status_preserves_completed_process_encoding_and_environment(tmp_path, monkeypatch):
    root = repository(tmp_path)
    (root / "authored.txt").write_text("fixture\n")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "foreign-git"))
    text = runner._plan_bound_git(root, "status", "--porcelain=v1", "-z")
    raw = runner._plan_bound_git(root, "status", "--porcelain=v1", "-z", input_bytes=b"")
    assert text.returncode == raw.returncode == 0
    assert text.stdout == "?? authored.txt\0" and raw.stdout == text.stdout.encode()
    assert isinstance(text.stderr, str) and isinstance(raw.stderr, bytes)


@pytest.mark.parametrize("failure", ["stdout", "stderr", "records", "timeout"])
def test_capture_failure_reaps_only_its_exact_authored_child(tmp_path, monkeypatch, failure):
    children = []
    original = subprocess.Popen
    def authored_child(argv, **kwargs):
        assert argv[0] == "/usr/bin/git" and "status" in argv
        assert kwargs["env"] == runner._plan_bound_git_environment()
        assert kwargs["cwd"] == tmp_path
        payload = {
            "stdout": "import os,time; os.write(1,b'x'*4096); time.sleep(30)",
            "stderr": "import os,time; os.write(2,b'x'*4096); time.sleep(30)",
            "records": "import os,time; os.write(1,b'?? f\\0'*4); time.sleep(30)",
            "timeout": "import time; time.sleep(30)",
        }[failure]
        process = original([sys.executable, "-B", "-c", payload], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", authored_child)
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_MAX_STDOUT_BYTES", 64, raising=False)
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_MAX_STDERR_BYTES", 64, raising=False)
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_MAX_RECORDS", 1, raising=False)
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_TIMEOUT_SECONDS", .25, raising=False)
    try:
        expected = subprocess.TimeoutExpired if failure == "timeout" else ValueError
        with pytest.raises(expected):
            runner._plan_bound_git(tmp_path, "status", "--porcelain=v1", "-z")
        assert len(children) == 1
        assert children[0].returncode is not None
        assert all(stream is None or stream.closed for stream in (
            children[0].stdin, children[0].stdout, children[0].stderr))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_timeout_fences_owned_descendant_holding_pipe_after_leader_exit(tmp_path, monkeypatch):
    original, children = subprocess.Popen, []
    identity_path = tmp_path / "descendant.json"
    payload = "\n".join([
        "import json,os,pathlib,time",
        "pid = os.fork()",
        "if pid: os._exit(0)",
        "fields = pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()",
        "pathlib.Path(" + repr(str(identity_path)) + ").write_text(json.dumps({'pid':os.getpid(),'birth':int(fields[19])}))",
        "time.sleep(30)",
    ])
    def authored_child(argv, **kwargs):
        assert kwargs["start_new_session"] is True
        process = original([sys.executable, "-B", "-c", payload], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", authored_child)
    monkeypatch.setattr(runner, "_PLAN_BOUND_STATUS_TIMEOUT_SECONDS", .5)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            runner._plan_bound_git(tmp_path, "status", "--porcelain=v1", "-z")
        assert children[0].returncode == 0
        identity = json.loads(identity_path.read_text())
        # killpg queues SIGKILL; give the owned descendant a bounded interval
        # to be scheduled before checking that no same-identity worker lives.
        deadline = time.monotonic() + 1.0
        while True:
            try:
                fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                break
            if int(fields[19]) != identity["birth"] or fields[0] == "Z":
                break
            assert time.monotonic() < deadline, "owned descendant survived status timeout"
            time.sleep(.01)
    finally:
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            try:
                fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[19]) == identity["birth"] and fields[0] != "Z":
                    os.kill(identity["pid"], signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_decode_failure_never_signals_reaped_group_identity(tmp_path, monkeypatch):
    original, children = subprocess.Popen, []

    def authored_child(argv, **kwargs):
        process = original([sys.executable, "-B", "-c", "import os; os.write(1,b'\\xff')"], **kwargs)
        children.append(process)
        return process

    def unexpected_signal(*args):
        pytest.fail("a reaped process group no longer carries signaling authority")

    monkeypatch.setattr(subprocess, "Popen", authored_child)
    monkeypatch.setattr(capture.locale, "getencoding", lambda: "ascii")
    monkeypatch.setattr(capture.os, "killpg", unexpected_signal)
    with pytest.raises(UnicodeDecodeError):
        runner._plan_bound_git(tmp_path, "status", "--porcelain=v1", "-z")
    assert children[0].returncode == 0
    assert children[0].stdout.closed and children[0].stderr.closed
