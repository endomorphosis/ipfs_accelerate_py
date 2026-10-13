"""Native STOP receipt tests; no provider calls or shared resource mutation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "legal_resource_wrapper", ROOT / "scripts/run_legal_autoformalization_supervisor.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

# Match the actual runtime's self-identifying receipt representation.
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity


class NativeFenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="lac-fence-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "master.pid.terminal.json"
        self.child = SimpleNamespace(pid=os.getpid())
        self.launched = time.time_ns() - 1_000_000_000
        self.started = time.time_ns() - 500_000_000
        self.observer = wrapper.NativeFenceObserver(
            self.path, "lac-test", self.child, self.launched, content_identity)

    def write(self, *, active=False, **changes):
        payload = {
            "schema": self.observer.ACTIVE if active else self.observer.TERMINAL,
            "label": "lac-test", "master_pid": self.child.pid,
            "run_started_at_epoch_nanoseconds": self.started,
            "recorded_at": "2026-10-13T00:00:00+00:00",
            "task_completion_authority": False,
        }
        payload.update({"active": True} if active else {
            "all_trees_fenced": True, "active_binding_cid": "test-active-cid"})
        payload.update(changes)
        identity_field = "active_binding_cid" if active else "terminal_receipt_cid"
        payload[identity_field] = content_identity(payload)
        self.path.write_text(json.dumps(payload))
        return payload

    def test_fresh_current_master_terminal_checks_native_identity(self):
        payload = self.write()
        self.assertTrue(self.observer.observe())
        self.assertEqual(self.observer.proof["terminal_receipt_cid"], payload["terminal_receipt_cid"])
        self.assertEqual(self.observer.proof["master_start_time_ticks"], wrapper.process_birth(os.getpid())[1])

    def test_previous_run_terminal_is_not_shutdown_proof(self):
        self.write(run_started_at_epoch_nanoseconds=self.launched - 1)
        self.assertFalse(self.observer.observe())
        self.assertIsNone(self.observer.proof)

    def test_nonobject_native_artifacts_remain_unverified(self):
        for value in ([], None, "text", 3):
            with self.subTest(value=value):
                self.path.write_text(json.dumps(value))
                self.assertFalse(self.observer.observe())
                self.assertIsNone(self.observer.proof)

    def test_unbound_master_cannot_supply_stop_proof(self):
        self.write(master_pid=999_999_999)
        self.assertFalse(self.observer.observe())

    def test_unfenced_native_terminal_keeps_resource_monitor_pending(self):
        self.write(all_trees_fenced=False)
        self.assertFalse(self.observer.observe())
        self.assertEqual(self.observer.reason, "native_stop_did_not_fence_all_trees")

    def test_receipt_body_change_fails_native_cid(self):
        payload = self.write()
        payload["label"] = "changed-after-signing"
        self.path.write_text(json.dumps(payload))
        self.assertFalse(self.observer.observe())

    def test_active_generation_must_bind_terminal(self):
        active = self.write(active=True)
        self.assertFalse(self.observer.observe())
        self.write(active_binding_cid="different-generation")
        self.assertFalse(self.observer.observe())
        self.write(active_binding_cid=active["active_binding_cid"])
        self.assertTrue(self.observer.observe())

    def test_newer_master_generation_cannot_replace_observed_owner(self):
        active = self.write(active=True)
        self.assertFalse(self.observer.observe())
        self.write(active_binding_cid=active["active_binding_cid"],
                   run_started_at_epoch_nanoseconds=self.started + 1)
        self.assertFalse(self.observer.observe())


class ShutdownMonitorTests(unittest.TestCase):
    def test_empty_outer_group_waits_for_detached_tree_native_proof(self):
        class Observer:
            calls = 0
            reason = "native_master_active"
            proof = {"all_trees_fenced": True}

            def observe(self):
                self.calls += 1
                return self.calls >= 4

        observer = Observer()
        waited = []
        child = SimpleNamespace(pid=12345, poll=lambda: 0,
                                wait=lambda **kwargs: waited.append(True))
        receipt = {"status": "stopping_for_signal"}
        updates = []
        with patch.object(wrapper, "live_group", return_value=[]), \
                patch.object(wrapper.os, "killpg") as kill, \
                patch.object(wrapper.time, "sleep"):
            wrapper.terminate_group(child, 1, receipt,
                lambda: updates.append(dict(receipt)), observer, 1)
        kill.assert_not_called()
        self.assertGreaterEqual(observer.calls, 4)
        self.assertTrue(any(row["status"] == "resource_held_pending_native_fence" for row in updates))
        self.assertEqual(waited, [True])
        self.assertTrue(receipt["native_stop_verified"])
        self.assertTrue(receipt["owned_group_quiescent"])
        self.assertEqual(receipt["status"], "stopping_for_signal")

    def test_signal_and_timeout_only_fence_owned_outer_group(self):
        class Observer:
            calls = 0
            reason = "native_master_active"
            proof = {"all_trees_fenced": True}

            def observe(self):
                self.calls += 1
                return self.calls >= 4

        observer = Observer()
        child = SimpleNamespace(pid=12345, poll=lambda: 0, wait=lambda **kwargs: None)
        receipt = {"status": "stopping_for_duration_limit"}
        groups = [[12345], [12345], [12345], [], [], [], []]
        clock = iter(range(100, 2000, 100))
        with patch.object(wrapper, "live_group", side_effect=lambda pid: groups.pop(0) if groups else []), \
                patch.object(wrapper.os, "killpg") as kill, \
                patch.object(wrapper.time, "sleep"), \
                patch.object(wrapper.time, "monotonic", side_effect=lambda: next(clock)):
            wrapper.terminate_group(child, 1, receipt, lambda: None, observer, 1)
        self.assertIn(((12345, signal.SIGTERM),), [(call.args,) for call in kill.call_args_list])
        self.assertTrue(all(call.args[0] == child.pid for call in kill.call_args_list))
        self.assertTrue(receipt["native_stop_verified"])

    def test_missing_proof_never_finishes_cleanup_or_waits_child(self):
        class MonitorSentinel(Exception):
            pass

        class Observer:
            calls = 0
            reason = "native_terminal_missing"
            proof = None

            def observe(self):
                self.calls += 1
                if self.calls == 5:
                    raise MonitorSentinel()
                return False

        waited = []
        child = SimpleNamespace(pid=12345, poll=lambda: 0,
                                wait=lambda **kwargs: waited.append(True))
        receipt = {"status": "child_exited"}
        with patch.object(wrapper, "live_group", return_value=[]), \
                patch.object(wrapper.time, "sleep"), \
                self.assertRaises(MonitorSentinel):
            wrapper.terminate_group(child, 1, receipt, lambda: None, Observer(), 1)
        self.assertEqual(waited, [])
        self.assertFalse(receipt["native_stop_verified"])
        self.assertTrue(receipt["cleanup_pending"])
        self.assertEqual(receipt["status"], "resource_held_pending_native_fence")


if __name__ == "__main__":
    unittest.main()
