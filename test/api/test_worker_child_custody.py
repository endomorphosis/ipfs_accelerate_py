"""Per-worker cleanup owns only children created after an empty-child boundary."""
from pathlib import Path
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_cli_subreaper.py"


def _isolated(script, tmp_path):
    loader = f'''import importlib.util, json, os, signal, subprocess, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("worker_custody", {str(HELPER)!r})
custody = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custody)
'''
    # The existing separate wrapper guarantees cleanup even if a regression
    # makes the inner assertion fail before the callback helper can finish.
    process = subprocess.Popen([sys.executable, "-I", str(HELPER), "--",
        sys.executable, "-I", "-c", loader + script], cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        _stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.communicate(timeout=10)
        pytest.fail("dedicated worker custody fixture timed out")
    assert process.returncode == 0, stderr


@pytest.mark.parametrize("outcome", ["return", "exception", "system-exit"])
def test_callback_return_and_exceptions_reap_detached_and_double_fork_children(tmp_path, outcome):
    _isolated(f'''
children = []
def callback():
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children.append(child.pid)
    code = "import os,time;from pathlib import Path\\nif os.fork():os._exit(0)\\nos.setsid()\\nif os.fork():os._exit(0)\\nPath('grandchild').write_text(str(os.getpid()))\\ntime.sleep(120)"
    subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 3
    while not Path('grandchild').exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    children.append(int(Path('grandchild').read_text()))
    if {outcome!r} == "exception": raise RuntimeError("authored callback failure")
    if {outcome!r} == "system-exit": raise SystemExit(7)
    return 17
try:
    result = custody.run_with_child_custody(callback)
    assert {outcome!r} == "return" and result == 17
except RuntimeError as exc:
    assert {outcome!r} == "exception" and str(exc) == "authored callback failure"
except SystemExit as exc:
    assert {outcome!r} == "system-exit" and exc.code == 7
assert len(children) == 2
assert all(not Path(f"/proc/{{pid}}").exists() for pid in children)
assert not custody._direct_child_pids()
''', tmp_path)


def test_preexisting_child_is_not_killed_and_callback_is_not_invoked(tmp_path):
    _isolated('''
child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"], start_new_session=True)
called = []
try:
    try:
        custody.run_with_child_custody(lambda: called.append(True))
    except RuntimeError as exc:
        assert "empty stable process" in str(exc)
    else:
        raise AssertionError("preexisting child must refuse callback")
    assert called == [] and child.poll() is None
finally:
    child.kill()
    child.wait()
''', tmp_path)


def test_unavailable_subreaper_refuses_before_callback(tmp_path):
    _isolated('''
custody._enable_child_subreaper = lambda: False
called = []
try:
    custody.run_with_child_custody(lambda: called.append(True))
except RuntimeError as exc:
    assert str(exc) == "worker child custody unavailable"
else:
    raise AssertionError("unavailable custody must refuse callback")
assert called == []
''', tmp_path)
