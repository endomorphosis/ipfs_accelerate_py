"""Policy/unit checks. Synthetic metadata here never represents native admission."""
import copy
import os
import pickle
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import factual_candidate as facts
import native_acceptance as native

FIXTURE = Path(__file__).parent / "fixtures" / "bootstrap"


class NativePolicyTests(unittest.TestCase):
    def setUp(self):
        self.projection = {
            "reservation_id": "synthetic-policy-fixture-only", "scheduler_lane": "validation",
            "workload": "canonical_validation", "status": "active", "cleanup_error": None,
            "resource_lease": {"lane": "validation", "owner_pid": 5001, "parent_lease_id": None,
                               "requires_gpu": False, "released": False, "cancelled": False, "lease_id": "synthetic-lease"},
            "record": {"native_lease_id": "synthetic-lease", "owner_pid": 5001,
                       "validation_request_sha256": "1" * 64, "child": {"pid": 5003}},
        }

    def check(self, value):
        return native.check_native_projection(value, expected_owner_pid=5001, expected_worker_pid=5002,
                                              expected_request_sha256="1" * 64, expected_child_pid=5003)

    def test_metadata_predicate_does_not_mint_handle(self):
        self.assertEqual(self.check(self.projection)["child_pid"], 5003)
        self.assertEqual(len(native._HANDLES), 0)

    def test_wrong_purpose_or_nonlive_native_metadata_is_rejected(self):
        mutations = [
            ("scheduler_lane", "llm"), ("workload", "agent_implementation"), ("status", "released"),
            ("cleanup_error", "OwnerLost"),
            ("resource_lease.lane", "llm"), ("resource_lease.owner_pid", 5002),
            ("resource_lease.parent_lease_id", "borrowed-parent"), ("resource_lease.requires_gpu", True),
            ("resource_lease.released", True), ("resource_lease.cancelled", True),
            ("record.native_lease_id", "different"), ("record.owner_pid", 5002),
            ("record.validation_request_sha256", "2" * 64), ("record.child.pid", 5004),
        ]
        for spelling, replacement in mutations:
            with self.subTest(spelling=spelling):
                changed = copy.deepcopy(self.projection)
                keys = spelling.split(".")
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                with self.assertRaises(native.AcceptanceRefusal):
                    self.check(changed)

    def test_handle_cannot_be_constructed_from_mapping(self):
        with self.assertRaisesRegex(native.AcceptanceRefusal, "native_owner_only_handle_constructor"):
            native.FactualAcceptanceHandle(self.projection)

    def test_forged_handle_object_is_not_registered_authority(self):
        forged = object.__new__(native.FactualAcceptanceHandle)
        with self.assertRaisesRegex(native.AcceptanceRefusal, "genuine_unconsumed_native_handle_required"):
            native._entry(forged)
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with self.subTest(operation=operation.__name__), self.assertRaises(native.AcceptanceRefusal):
                operation(forged)

    def test_worker_json_is_not_host_context(self):
        with self.assertRaisesRegex(native.AcceptanceRefusal, "actual_host_context_object_required"):
            native._context({"native_validation_owner_bound": True})

    def test_handle_finite_lifetime_covers_censuses_but_refuses_expiry(self):
        self.assertEqual(native.HANDLE_LIFETIME_SECONDS, 900)
        handle = object.__new__(native.FactualAcceptanceHandle)
        entry = {"owner_pid": os.getpid(), "expires": 1000.0}
        with patch.object(native, "_HANDLES", {handle: entry}), patch.object(native.time, "monotonic", return_value=999.0):
            self.assertIs(native._entry(handle), entry)
        with patch.object(native, "_HANDLES", {handle: entry}), patch.object(native.time, "monotonic", return_value=1000.001):
            with self.assertRaisesRegex(native.AcceptanceRefusal, "native_handle_owner_or_deadline_changed"):
                native._entry(handle)

    def test_unexpired_handle_cannot_change_owner_process(self):
        handle = object.__new__(native.FactualAcceptanceHandle)
        entry = {"owner_pid": os.getpid() + 1, "expires": time.monotonic() + 900}
        with patch.object(native, "_HANDLES", {handle: entry}):
            with self.assertRaisesRegex(native.AcceptanceRefusal, "native_handle_owner_or_deadline_changed"):
                native._entry(handle)

    def test_legacy_completion_projection_changes_only_9001_status(self):
        sources, frozen = facts.load_source_bundle(FIXTURE.absolute())
        facts.derive_expected(sources, frozen)
        before = sources["execution_board"]
        after = native.completion_projection(before, expected_board_sha256=facts.sha(before))
        self.assertEqual(after, before.replace(b"- Status: todo\n", b"- Status: completed\n", 1))
        self.assertEqual(after.count(b"- Status: blocked\n"), 63)
        self.assertEqual(after.count(b"- Completion: manual\n"), 64)

    def test_stale_legacy_revision_and_future_task_unblocking_refuse(self):
        sources, _ = facts.load_source_bundle(FIXTURE.absolute())
        before = sources["execution_board"]
        with self.assertRaisesRegex(native.AcceptanceRefusal, "legacy_board_revision_conflict"):
            native.completion_projection(before, expected_board_sha256="0" * 64)
        changed = before.replace(b"- Status: blocked\n", b"- Status: todo\n", 1)
        with self.assertRaisesRegex(native.AcceptanceRefusal, "legacy_future_task_unblocked"):
            native.completion_projection(changed, expected_board_sha256=facts.sha(changed))

    def test_completion_cannot_replay_against_already_completed_board(self):
        sources, _ = facts.load_source_bundle(FIXTURE.absolute())
        before = sources["execution_board"]
        after = native.completion_projection(before, expected_board_sha256=facts.sha(before))
        with self.assertRaisesRegex(native.AcceptanceRefusal, "legacy_completion_status_or_policy_differs"):
            native.completion_projection(after, expected_board_sha256=facts.sha(after))


if __name__ == "__main__":
    unittest.main()
