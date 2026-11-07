"""Storage admission never grants completion or launches recovery effects."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_watchdog as adapter


class ResourcePressureTests(unittest.TestCase):
    def test_actual_statvfs_evidence_and_distinct_state_repository_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = adapter.resource_observation(repository=root, state_root=root, framework_root=root,
                statvfs=lambda path: SimpleNamespace(f_bavail=100 * 1024**2, f_frsize=1))
            self.assertTrue(evidence["blocked"])
            self.assertEqual([row["reason"] for row in evidence["paths"]], ["sufficient", "insufficient_free_space", "insufficient_free_space"])
            self.assertEqual(evidence["paths"][1]["free_bytes"], 100 * 1024**2)

    def test_invalid_configured_thresholds_cannot_disable_admission_bounds(self):
        for value in (0, -1, True, float("nan"), float("inf"), "64", 1024**4 + 1):
            with self.subTest(value=value), self.assertRaises(adapter.WatchdogBridgeError):
                adapter.resource_observation(repository=Path("/"), state_root=Path("/"), config={"resource_pressure": {"minimum_state_free_bytes": value}})

    def test_storage_probe_error_remains_a_visible_external_blocker(self):
        def unavailable(path):
            raise OSError(28, "full")
        value = adapter.resource_observation(repository=Path("/"), state_root=Path("/"), statvfs=unavailable)
        self.assertTrue(value["blocked"])
        self.assertTrue(all(row["reason"] == "statvfs_unavailable" and row["error_errno"] == 28 for row in value["paths"]))

    def test_full_volume_observation_preserves_canonical_clocks_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, state = root / "site", root / "state"
            repository.mkdir(); state.mkdir()
            ready = {"schema": "wwl.watchdog.ready@1", "run_id": "pressure-test", "repository": str(repository), "policy_id": "policy", "native_revision": "revision", "admitted_plan_revision": "plan", "required_task_ids": ["WWL-000"], "bindings": {"lifecycle_restart": True, "repair_dispatch": True}}
            snapshot = {"schema": "wwl.watchdog.snapshot@1", "run_id": "pressure-test", "observed_at": 10000, "heartbeat_at": 9900, "progress_token": "actual-phase", "blocked_kind": "software_defect"}
            (state / "campaign-readiness.json").write_text(json.dumps(ready))
            (state / "campaign-status.json").write_text(json.dumps(snapshot))
            bridge = adapter.WorldwideWatchdogBridge(repository=repository, state_root=state, run_id="pressure-test", clock=lambda: 10000,
                statvfs=lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
            with patch.object(adapter, "publish_once", side_effect=AssertionError("observe cannot write under pressure")):
                observed = bridge.observe({"run_id": "pressure-test", "policy_id": "policy"})
                request = {"schema": "wwl.watchdog.action@1", "run_id": "pressure-test", "policy_id": "policy", "action_id": "a" * 64, "kind": "recover", "issued_at": 10000}
                self.assertEqual(bridge.dispatch(request, timeout_seconds=0)["status"], "unknown")
            self.assertEqual(observed["blocked_kind"], "resource_quota")
            self.assertEqual(observed["heartbeat_at"], 9900)
            self.assertEqual(observed["progress_token"], "actual-phase")
            self.assertEqual(sorted(path.name for path in state.iterdir()), ["campaign-readiness.json", "campaign-status.json"])


if __name__ == "__main__":
    unittest.main()
