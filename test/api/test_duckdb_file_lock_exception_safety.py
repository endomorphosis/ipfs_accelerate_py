"""Execute the real lock functions in isolation; never open a database."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import errno
import fcntl
from pathlib import Path
import tempfile
import threading
import time
from typing import Iterator, TYPE_CHECKING
import unittest
from unittest.mock import patch


if TYPE_CHECKING:
    import ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state

SOURCE = (Path(__file__).resolve().parents[2]
          / "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py")


def load_lock_functions():
    """Avoid importing any optional owner, model, or database implementation."""
    tree = ast.parse(SOURCE.read_text())
    names = {"_thread_lock", "exclusive_file_lock"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    selected = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "Path": Path, "Iterator": Iterator, "threading": threading,
        "time": time, "fcntl": fcntl, "contextmanager": contextmanager,
        "DEFAULT_LOCK_TIMEOUT_SECONDS": .05,
        "_THREAD_LOCKS": {}, "_THREAD_LOCKS_GUARD": threading.Lock(),
    }
    exec(compile(selected, str(SOURCE), "exec"), namespace)
    return namespace["exclusive_file_lock"]


class FileLockExceptionSafety(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="duckdb-lock-proof-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "queue.lock"
        self.exclusive = load_lock_functions()

    def assert_other_thread_can_acquire(self):
        outcome = []

        def acquire():
            try:
                with self.exclusive(self.path, timeout_seconds=.05):
                    outcome.append("acquired")
            except BaseException as exc:
                outcome.append(type(exc).__name__)

        worker = threading.Thread(target=acquire, daemon=True)
        worker.start()
        worker.join(.5)
        self.assertFalse(worker.is_alive(), "lock acquisition exceeded its bound")
        self.assertEqual(outcome, ["acquired"])

    def test_open_failure_releases_thread_lock(self):
        original = Path.open

        def fail_selected(path, *args, **kwargs):
            if path == self.path:
                raise OSError(errno.EMFILE, "injected descriptor exhaustion")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", fail_selected):
            with self.assertRaisesRegex(OSError, "descriptor exhaustion"):
                with self.exclusive(self.path, timeout_seconds=.05):
                    self.fail("lock-file open should fail")
        self.assert_other_thread_can_acquire()

    def test_unlock_failure_still_closes_and_releases(self):
        original = fcntl.flock

        def fail_unlock(fd, operation):
            if operation == fcntl.LOCK_UN:
                raise OSError(errno.EIO, "injected unlock failure")
            return original(fd, operation)

        with patch.object(fcntl, "flock", fail_unlock):
            with self.assertRaisesRegex(OSError, "unlock failure"):
                with self.exclusive(self.path, timeout_seconds=.05):
                    pass
        self.assert_other_thread_can_acquire()

    def test_close_failure_still_releases_thread_lock(self):
        original = Path.open

        class FailAfterClose:
            def __init__(self, handle):
                self.handle = handle

            def fileno(self):
                return self.handle.fileno()

            def close(self):
                self.handle.close()
                raise OSError(errno.EIO, "injected close acknowledgement failure")

        def selected_proxy(path, *args, **kwargs):
            handle = original(path, *args, **kwargs)
            return FailAfterClose(handle) if path == self.path else handle

        with patch.object(Path, "open", selected_proxy):
            with self.assertRaisesRegex(OSError, "acknowledgement failure"):
                with self.exclusive(self.path, timeout_seconds=.05):
                    pass
        self.assert_other_thread_can_acquire()

    def test_body_exception_releases_both_locks(self):
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            with self.exclusive(self.path, timeout_seconds=.05):
                raise RuntimeError("callback failed")
        self.assert_other_thread_can_acquire()

    def test_real_flock_contention_timeout_then_recovery(self):
        with self.path.open("a+b") as holder:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(TimeoutError, "process lock"):
                with self.exclusive(self.path, timeout_seconds=.02):
                    self.fail("contended file lock should time out")
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        self.assert_other_thread_can_acquire()

    def test_other_thread_cannot_enter_until_holder_exits(self):
        outcome = []

        def contend():
            try:
                with self.exclusive(self.path, timeout_seconds=.02):
                    outcome.append("entered")
            except TimeoutError:
                outcome.append("bounded_timeout")

        with self.exclusive(self.path, timeout_seconds=.05):
            worker = threading.Thread(target=contend, daemon=True)
            worker.start()
            worker.join(.5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(outcome, ["bounded_timeout"])
        self.assert_other_thread_can_acquire()


if __name__ == "__main__":
    unittest.main()
