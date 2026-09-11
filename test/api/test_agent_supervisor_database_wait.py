"""Database polling honors its interval without delaying native shutdown."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import bounded_daemon_wait_timeout


class DatabaseWaitTests(unittest.TestCase):
    def test_configured_interval_and_shorter_retry_deadline_are_honored(self):
        daemon = object.__new__(module.DatabaseImplementationDaemon)
        for result, expected in (({}, 30.0), ({"next_wake_after_seconds": 2.5}, 2.5)):
            with self.subTest(result=result), patch.object(module.time, "sleep") as sleep:
                daemon.wait_for_wake(bounded_daemon_wait_timeout(result, default_timeout=30.0))
                sleep.assert_called_once_with(expected)
        with patch.object(module.time, "sleep") as sleep:
            daemon.wait_for_wake(0)
            daemon.wait_for_wake(-1)
            sleep.assert_not_called()

    def test_actual_runner_signals_interrupt_long_wait_and_close_runtime(self):
        # Exercise the real main loop, signal handler, and wait method in an
        # owned child. Only persistence/provider construction is replaced.
        source = textwrap.dedent("""
            from types import SimpleNamespace
            from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as m
            wait = m.DatabaseImplementationDaemon.wait_for_wake
            class TemporaryDaemon:
                def __init__(self, **kwargs): pass
                def run_once(self): return {"unchanged": True, "write_count": 0}
                def wait_for_wake(self, timeout):
                    print("WAIT", timeout, flush=True)
                    wait(self, timeout)
                def close_event_runtime(self): print("CLOSED", flush=True)
            m.DatabaseImplementationDaemon = TemporaryDaemon
            m.database_program_from_daemon_namespace = lambda args: None
            m.bind_database_portal_execution_from_args = lambda *args, **kwargs: None
            m.parse_args = lambda argv: SimpleNamespace(
                log_level="ERROR", llm_merge_resolver_command="",
                llm_merge_resolver_timeout_seconds=None, database_path="unused-test.duckdb",
                authority_mode="embedded", task_source_kind="duckdb", implement=False,
                reset_manual_completion_authority_renewal_task_id="",
                clear_protected_path_incident=False, once=False, interval=30.0)
            m.main([])
        """)
        env = dict(os.environ, IPFS_ACCEL_SKIP_CORE="1")
        package_root = str(Path(module.__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=stop_signal):
                process = subprocess.Popen([sys.executable, "-c", source], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    with selectors.DefaultSelector() as ready:
                        ready.register(process.stdout, selectors.EVENT_READ)
                        self.assertTrue(ready.select(timeout=10), "child never reached native wait")
                    self.assertEqual(process.stdout.readline().strip(), "WAIT 30.0")
                    process.send_signal(stop_signal)
                    stdout, stderr = process.communicate(timeout=3)
                    self.assertEqual(process.returncode, 128 + stop_signal, stderr)
                    self.assertIn("CLOSED", stdout)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
