"""Disposable process tree executing the retained native signal/adoption methods.

The method bodies are loaded from this checkout (unchanged since SAWM25f).
Only process launch/cleanup dependencies are replaced by owned local fixtures;
this does not represent full native board or callback qualification.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILE = Path(__file__).resolve()


def native_method(relative, name, namespace):
    source = (ROOT / relative).read_text()
    nodes = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(nodes) == 1
    node = nodes[0]
    module = ast.Module(body=[node], type_ignores=[])
    exec(  # noqa: S102 - exact repository native method in isolated fixture
        compile(ast.fix_missing_locations(module), str(ROOT / relative), "exec"),
        namespace,
    )
    return namespace[name]


@contextlib.contextmanager
def lock(path):
    method = native_method(
        "ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py",
        "serialized_lock_update",
        {
            "contextmanager": contextlib.contextmanager,
            "fcntl": fcntl,
            "msvcrt": None,
            "os": os,
            "time": time,
        },
    )
    with method(path):
        yield


def spawn(role, directory, mode):
    return subprocess.Popen(
        [sys.executable, "-I", "-S", str(FILE), role, str(directory), mode],
        start_new_session=True,
    )


def daemon(directory, mode):
    def terminate(*_):
        if mode in {"ancillary", "late_transient"}:
            child = spawn("ancillary", directory, mode)
            (directory / "ancillary.pid").write_text(str(child.pid))
        raise SystemExit(143)

    if mode == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, terminate)
    if mode in {"transient", "parent_progress"}:
        child = spawn("ancillary", directory, mode)
        (directory / "ancillary.pid").write_text(str(child.pid))
    (directory / "daemon-ready").write_text(str(os.getpid()))
    while True:
        (directory / "daemon-heartbeat").write_text(str(time.monotonic_ns()))
        time.sleep(0.01)


def ancillary(directory, mode):
    if mode in {"transient", "parent_progress"}:
        while not (directory / "helper-trigger").exists():
            time.sleep(0.01)
        if mode == "transient":
            time.sleep(0.12)
            return
        before = (directory / "daemon-heartbeat").read_text()
        (directory / "helper-parent-wait").touch()
        while (directory / "daemon-heartbeat").read_text() == before:
            time.sleep(0.01)
        return
    if mode == "late_transient":
        time.sleep(0.12)
        return
    while True:
        time.sleep(0.01)


def wrapper(directory, mode):
    child = None
    namespace = {
        "threading": threading,
        "signal": signal,
        "logger": logging.getLogger(__name__),
    }
    native_run = native_method(
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "run_forever",
        namespace,
    )
    adopt = native_method(
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "adopt_or_launch_supervised_child",
        {
            "serialized_lock_update": lock,
            "adopt_supervised_child": lambda _: None,
            "launch_supervised_child": lambda _: spawn("daemon", directory, mode),
        },
    )

    class NativeSignalFixture:
        def _run_forever_loop(self):
            nonlocal child
            while True:
                child = adopt(None, launch_lock_path=directory / "lane.lock")
                with (directory / "births").open("a") as stream:
                    stream.write(str(child.pid) + "\n")
                (directory / "roster.json").write_text(
                    json.dumps(
                        {
                            "controller": os.getppid(),
                            "supervisor": os.getpid(),
                            "daemon": child.pid,
                        }
                    )
                )
                child.wait()
                (directory / "attempted-relaunch").touch()

        def _terminate_managed_daemon_tree(self):
            # Exercise lock release ordering without ever signalling a survivor.
            with lock(directory / "lane.lock"):
                assert child is None or child.poll() is not None
                (directory / "supervisor-cleanup").touch()
            return {"quiesced": True}

        def _reconcile_interrupted_implementation_after_shutdown(self):
            return {"fixture_only": True}

        def _record_event(self, *args):
            pass

        def _write_signal_shutdown_status(self, **kwargs):
            pass

    try:
        native_run(NativeSignalFixture())
    except SystemExit:
        pass


def master(directory, mode):
    stopping = threading.Event()

    class SupervisorRunInterrupted(BaseException):
        pass

    source = (
        ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py"
    ).read_text()
    handler = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_handle_signal"
    )
    body = textwrap.dedent(ast.get_source_segment(source, handler))
    factory = (
        "def handler_factory():\n"
        "    teardown_transition = False\n"
        "    terminal_outcome_frozen = False\n"
        "    transition_signals = []\n"
        "    post_outcome_signals = []\n"
        + textwrap.indent(body, "    ")
        + "\n    return _handle_signal\n"
    )
    namespace = {"SupervisorRunInterrupted": SupervisorRunInterrupted}
    exec(factory, namespace)  # noqa: S102 - exact native signal handler fixture
    signal.signal(signal.SIGTERM, namespace["handler_factory"]())
    thread = threading.Thread(target=lambda: stopping.wait(), daemon=True)
    thread.start()
    directories = [directory / "lane-0", directory / "lane-1"]
    for lane in directories:
        lane.mkdir()
    children = [spawn("wrapper", lane, mode) for lane in directories]
    try:
        while not stopping.wait(0.01):
            (directory / "master-heartbeat").write_text(str(time.monotonic_ns()))
            for index, child in enumerate(children):
                if child.poll() is not None:
                    children[index] = spawn("wrapper", directories[index], mode)
    except SupervisorRunInterrupted:
        stopping.set()
    assert all(child.poll() is not None for child in children), (
        "fixture refuses master forced cleanup on survivors"
    )
    (directory / "master-cleanup").touch()


if __name__ == "__main__":
    globals()[sys.argv[1]](Path(sys.argv[2]), sys.argv[3])
