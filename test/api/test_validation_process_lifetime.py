"""The real sealed validator owns orphaned children through terminal cleanup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    build_validation_environment, validation_environment_for_runner,
)


def _run(tmp_path, code, *, timeout=5):
    command = "python -c " + shlex.quote(code)
    runner = native.TodoImplementationDaemon._validation_command_runner
    environment = validation_environment_for_runner(build_validation_environment(), runner)
    return runner(spec=SimpleNamespace(command=command, raw_command=command),
        workspace_path=tmp_path, timeout_seconds=timeout, environment=environment)


def _gone(pid):
    deadline = time.monotonic() + 3
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return not Path(f"/proc/{pid}").exists()


@pytest.mark.parametrize("exitcode", [0, 7])
def test_reaped_setsid_double_fork_after_original_shell_exit(tmp_path, exitcode):
    code = f'''
import os, time
from pathlib import Path
ready = Path('orphan.pid')
if os.fork() == 0:
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    temporary = ready.with_suffix('.tmp')
    temporary.write_text(str(os.getpid()))
    print('orphan-holds-output', flush=True)
    temporary.replace(ready)
    time.sleep(120)
    os._exit(0)
deadline = time.monotonic() + 3
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(.01)
assert ready.exists()
raise SystemExit({exitcode})
'''
    started = time.monotonic()
    result = _run(tmp_path, code)
    assert time.monotonic() - started < 10
    assert result["returncode"] == exitcode
    assert "orphan-holds-output" in result["output"]
    assert result["validation_python_launcher"]["sealed"] is True
    assert _gone(int((tmp_path / "orphan.pid").read_text()))


@pytest.mark.parametrize("new_session", [False, True])
def test_timeout_reaps_term_ignoring_descendant_retaining_output(tmp_path, new_session):
    code = f'''
import os, signal, subprocess, sys, time
from pathlib import Path
child = "import os, signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path('child.pid').write_text(str(os.getpid())); print('holding-output', flush=True); time.sleep(120)"
process = subprocess.Popen([sys.executable, '-I', '-B', '-c', child], start_new_session={new_session})
Path('parent.pid').write_text(str(os.getpid()))
deadline=time.monotonic()+3
while not Path('child.pid').exists() and time.monotonic()<deadline: time.sleep(.01)
assert Path('child.pid').exists()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(120)
'''
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _run(tmp_path, code, timeout=1)
    assert time.monotonic() - started < 10
    assert raised.value.process_tree_quiesced is True
    assert "holding-output" in raised.value.output
    for filename in ("parent.pid", "child.pid"):
        assert _gone(int((tmp_path / filename).read_text()))


def test_unrelated_caller_child_survives_validation_cleanup(tmp_path):
    sentinel = subprocess.Popen([sys.executable, "-I", "-B", "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run(tmp_path, "import time; time.sleep(120)", timeout=.2)
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_raw_negative_shell_exit_code_is_preserved(tmp_path):
    runner = native.TodoImplementationDaemon._validation_command_runner
    environment = validation_environment_for_runner(build_validation_environment(), runner)
    result = runner(spec=SimpleNamespace(command="kill -TERM $$", raw_command="kill -TERM $$"),
        workspace_path=tmp_path, timeout_seconds=5, environment=environment)
    assert result["returncode"] == -signal.SIGTERM


def test_result_descriptor_is_not_inherited_by_candidate(tmp_path, monkeypatch):
    actual = native.run_process_group_stream
    identity_path = tmp_path / "result-identity.json"
    def observed(command, **kwargs):
        descriptor = int(command[command.index("--result-fd") + 1])
        info = os.fstat(descriptor)
        identity_path.write_text(json.dumps([info.st_dev, info.st_ino]))
        return actual(command, **kwargs)
    monkeypatch.setattr(native, "run_process_group_stream", observed)
    result = _run(tmp_path, '''
import json, os
from pathlib import Path
expected = json.loads(Path('result-identity.json').read_text())
for name in os.listdir('/proc/self/fd'):
    try:
        info = os.fstat(int(name))
    except OSError:
        continue
    assert [info.st_dev, info.st_ino] != expected
print('private-result-not-inherited')
''')
    assert result["returncode"] == 0
    assert "private-result-not-inherited" in result["output"]


@pytest.mark.parametrize("damage", ["missing", "wrong_exit", "wrong_type"])
def test_missing_or_unbound_cleanup_report_cannot_be_success(tmp_path, monkeypatch, damage):
    actual = native.run_process_group_stream
    def altered(command, **kwargs):
        completed = actual(command, **kwargs)
        descriptor = int(command[command.index("--result-fd") + 1])
        if damage == "missing":
            os.ftruncate(descriptor, 0)
        elif damage == "wrong_exit":
            completed.returncode = 7
        else:
            os.lseek(descriptor, 0, os.SEEK_SET)
            raw = os.read(descriptor, 1024)
            value = json.loads(raw)
            value["children_reaped"] = 1
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, json.dumps(value).encode())
        return completed
    monkeypatch.setattr(native, "run_process_group_stream", altered)
    result = _run(tmp_path, "print('ordinary-child-finished')")
    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert result["error"] == "validation_process_cleanup_unverified"
    assert result["validation_python_launcher"]["sealed"] is True
    assert "process_tree_quiesced" not in result


def test_output_capture_is_bounded_and_overflow_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "AUTHORITY_VALIDATION_OUTPUT_LIMIT_BYTES", 1024)
    result = _run(tmp_path, "print('x' * 4096)")
    assert result["returncode"] == 75
    assert result["error"] == "validation_output_limit_exceeded"
    assert len(result["output"]) <= 1024


def test_unverified_outer_group_cleanup_is_a_typed_failure(tmp_path, monkeypatch):
    def unverified(*args, **kwargs):
        raise native.ProcessGroupCleanupUnverified()
    monkeypatch.setattr(native, "run_process_group_stream", unverified)
    result = _run(tmp_path, "print('not-dispatched')")
    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert result["error"] == "validation_process_cleanup_unverified"
    assert "process_tree_quiesced" not in result


def test_existing_subreaper_cli_contract_is_unchanged(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_cli_subreaper
    result = subprocess.run([sys.executable, "-I", "-S", "-B", native_cli_subreaper.__file__, "--",
        sys.executable, "-I", "-B", "-c", "print('legacy-argv'); raise SystemExit(7)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 7
    assert result.stdout.strip() == "legacy-argv"


def test_real_sealed_failure_marker_classification_survives_custody(tmp_path):
    output = "Playwright host dependency preflight failed on Linux.\nbrowser bundle is not installed under /var/cache/ms-playwright\n"
    result = _run(tmp_path, "print(" + repr(output) + "); raise SystemExit(1)")
    assert result["returncode"] == 1
    assert result["infrastructure_failure"] is True
    assert result["error"] == "validation_environment_playwright_browsers_missing"
    assert result["validation_python_launcher"]["sealed"] is True


def test_non_linux_retains_canonical_direct_without_custody_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "darwin")
    def no_linux_wrapper(*args, **kwargs):
        pytest.fail("non-Linux compatibility path invoked Linux custody")
    monkeypatch.setattr(native, "run_process_group_stream", no_linux_wrapper)
    result = _run(tmp_path, "print('canonical-direct-still-works')")
    assert result["returncode"] == 0
    assert "canonical-direct-still-works" in result["output"]
    assert result["validation_python_launcher"]["sealed"] is False
    assert result["validation_process_custody"] == {"supported": False, "scope": "direct_process_only"}
