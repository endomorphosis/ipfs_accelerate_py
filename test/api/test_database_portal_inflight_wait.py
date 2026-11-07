"""A live paid provider remains inside one native callback and deadline."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge, DatabasePortalProviderCustodyUnknown
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon


class NativeInflightWait(unittest.TestCase):
    def fixture(self):
        bridge = DatabasePortalExecutionBridge.__new__(DatabasePortalExecutionBridge)
        native = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
        native._implementation_timeout_policy = lambda _: SimpleNamespace(max_timeout_seconds=100)
        native._implementation_cancel_requested = lambda: False
        implementation = {"reason": "inflight_process", "dispatch_intent_retained": True,
            "provider_dispatched": True, "attempt_consumed": True, "task_id": "WWL-001",
            "canonical_task_cid": "sha256:portal", "canonical_task_key": "portal-key", "attempt": 1, "worktree_path": "/owned/worktree"}
        live = {key: implementation[key] for key in ("task_id", "canonical_task_cid", "canonical_task_key", "attempt", "worktree_path")}
        native._find_live_inflight_implementation = lambda: live
        return bridge, native, implementation, SimpleNamespace(started_at_ms=1000000), SimpleNamespace(task_projection=Path('/owned/projection.md'))

    def invoke(self, bridge, native, implementation, attempt, paths):
        return bridge._wait_for_native_inflight(attempt=attempt, paths=paths, binding={"task_alias": "WWL-001"},
            daemon=native, authority={"canonical_task_cid": "sha256:portal"}, implementation=implementation,
            summaries=[], merge_request_loader=None)

    def test_wait_accepts_only_the_native_completion_gate_without_reentering_run_once(self):
        bridge, native, implementation, attempt, paths = self.fixture()
        ready = "# Fixture\n\n## WWL-001 Task\n- Status: ready\n"
        done = ready.replace('ready', 'completed')
        with patch.object(bridge, '_verify_projection', side_effect=[ready, ready, done]), patch.object(bridge, '_has_completion_event', return_value=True), patch.object(bridge, '_acceptance_receipt', return_value={'accepted': True}) as gate, patch.object(module.time, 'time', return_value=1001), patch.object(module.time, 'sleep') as slept:
            self.assertEqual(self.invoke(bridge,native,implementation,attempt,paths), {'accepted':True})
            gate.assert_called_once()
            slept.assert_called_once()

    def test_expired_original_attempt_is_unknown_and_never_renews_deadline(self):
        bridge,native,implementation,attempt,paths = self.fixture()
        ready = "# Fixture\n\n## WWL-001 Task\n- Status: ready\n"
        with patch.object(bridge,'_verify_projection',return_value=ready), patch.object(module.time,'time',return_value=1200), patch.object(module.time,'sleep') as slept:
            with self.assertRaisesRegex(DatabasePortalProviderCustodyUnknown,'original deadline elapsed'):
                self.invoke(bridge,native,implementation,attempt,paths)
            slept.assert_not_called()

    def test_disappeared_provider_gets_observation_grace_but_no_redispatch(self):
        bridge,native,implementation,attempt,paths = self.fixture()
        native._find_live_inflight_implementation = lambda: None
        ready = "# Fixture\n\n## WWL-001 Task\n- Status: ready\n"
        clock=[0.0]
        def sleep(seconds): clock[0]+=seconds
        with patch.object(bridge,'_verify_projection',return_value=ready), patch.object(module.time,'time',return_value=1001), patch.object(module.time,'monotonic',side_effect=lambda:clock[0]), patch.object(module.time,'sleep',side_effect=sleep):
            with self.assertRaisesRegex(DatabasePortalProviderCustodyUnknown,'vanished without accepted terminal evidence'):
                self.invoke(bridge,native,implementation,attempt,paths)
            self.assertEqual(clock[0],5.0)

    def test_noncanonical_daemon_or_foreign_dispatch_never_authorizes_wait(self):
        bridge,native,implementation,attempt,paths = self.fixture()
        for wrong in (SimpleNamespace(),native):
            changed = dict(implementation)
            if wrong is native: changed['canonical_task_cid']='sha256:foreign'
            with self.assertRaisesRegex(DatabasePortalProviderCustodyUnknown,'exact retained dispatch authority'):
                self.invoke(bridge,wrong,changed,attempt,paths)
