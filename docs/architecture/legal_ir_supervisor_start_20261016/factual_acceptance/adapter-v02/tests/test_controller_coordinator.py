"""Controller policy tests only; no native lease or task-board write is mocked."""
import copy
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import controller_coordinator as coordinator
import factual_candidate as facts
import native_acceptance as native

FIXTURE = Path(__file__).parent / "fixtures" / "bootstrap"


class ControllerPolicyTests(unittest.TestCase):
    def envelope(self, identifier, parent):
        return {"lease_id": identifier, "parent_lease_id": parent, "owner_pid": 5001,
                "lane": "orchestration", "cpu_slots": 1, "memory_mb": 128,
                "child_process_slots": 1, "requires_gpu": False,
                "gpu_memory_mb": 0, "unified_memory_mb": 0, "released": False, "cancelled": False}

    def setUp(self):
        self.root = self.envelope("root", None)
        self.phase = self.envelope("phase", "root")
        child = self.envelope("disk", "phase")
        self.request = {"disk_reservation_id": "synthetic-disk", "root_lease_id": "root", "phase_lease_id": "phase",
                        "root_request_id": "synthetic-controller-root", "phase_request_id": "synthetic-controller-phase"}
        self.owner = {"status": "active", "scheduler_lane": "orchestration", "workload": "canonical_parented_phase",
                      "reservation_id": "synthetic-disk", "cleanup_error": None, "resource_lease": child,
                      "record": {"native_lease_id": "disk", "parent_lease_id": "phase", "owner_pid": 5001}}
        self.active = [dict(self.root, request_id="synthetic-controller-root"),
                       dict(self.phase, request_id="synthetic-controller-phase"), child]

    def check(self, owner=None, active=None):
        return coordinator.check_controller_projection(owner or self.owner, self.root, self.phase,
                                                        active or self.active, self.request, owner_pid=5001)

    def test_genuine_ancestry_projection_is_required_without_granting_authority(self):
        self.assertEqual(self.check()["disk_native_lease_id"], "disk")
        self.assertEqual(len(native._HANDLES), 0)

    def test_wrong_native_purpose_parent_or_active_record_is_rejected(self):
        for key, replacement in (("scheduler_lane", "validation"), ("workload", "canonical_validation"), ("status", "released")):
            with self.subTest(key=key):
                changed = dict(self.owner, **{key: replacement})
                with self.assertRaises(native.AcceptanceRefusal):
                    self.check(owner=changed)
        changed = copy.deepcopy(self.owner)
        changed["resource_lease"]["parent_lease_id"] = "foreign-phase"
        with self.assertRaises(native.AcceptanceRefusal):
            self.check(owner=changed)
        with self.assertRaises(native.AcceptanceRefusal):
            self.check(active=self.active[:-1])

    def test_mapping_cannot_be_supplied_as_native_controller_owner(self):
        with self.assertRaisesRegex(native.AcceptanceRefusal, "genuine_parented_controller_owner_required"):
            coordinator._verify_admission(self.owner, self.root, self.phase, self.request)

    def test_one_use_consumer_mechanics_and_exact_candidate_fence(self):
        # This sterile registry entry is only a unit fixture for consumption.
        # It bypasses no production API and supplies no native qualification.
        sources, frozen = facts.load_source_bundle(FIXTURE.absolute())
        report = facts.canonical(facts.candidate_template(sources, frozen))
        context = native.NativeCandidateContext(
            task_cid="synthetic-task-cid", task_revision=facts.sha(sources["execution_board"]),
            native_attempt_id="synthetic-attempt", candidate_commit="1" * 40,
            orchestration_namespace=facts.strict_json(sources["scheduler_config"])["board_namespace"],
            source_root=str(FIXTURE.absolute()), report_path="/synthetic-worker/output/candidate.json", report_sha256=facts.sha(report),
            provider_id="grok_cli", model_id="grok-4.7", reasoning_effort="", provider_trigger="primary",
            provider_execution_id="synthetic-execution", provider_worker_pid=os.getpid() + 1,
            provider_receipt_path="/synthetic-owner/provider.json", provider_receipt_sha256="2" * 64,
            controller_observation_path="/synthetic-owner/controller.json", controller_observation_sha256="3" * 64,
        )
        handle = object.__new__(native.FactualAcceptanceHandle)
        entry = {"owner_pid": os.getpid(), "expires": time.monotonic() + 60, "context": context,
                 "report": report, "sources": sources, "frozen": frozen, "summary": {"synthetic_policy_fixture_only": True}}
        with patch.object(native, "_HANDLES", {handle: entry}):
            kwargs = dict(current_context=context, board_raw=sources["execution_board"], report_raw=report,
                          source_documents=sources, frozen_files=frozen)
            with self.assertRaisesRegex(native.AcceptanceRefusal, "accepted_candidate_or_source_changed"):
                native.consume_for_manual_completion(handle, **dict(kwargs, report_raw=report + b"\n"))
            after, summary = native.consume_for_manual_completion(handle, **kwargs)
            self.assertIn(b"- Status: completed\n", after)
            self.assertEqual(summary["proposed_board_sha256"], facts.sha(after))
            with self.assertRaisesRegex(native.AcceptanceRefusal, "genuine_unconsumed_native_handle_required"):
                native.consume_for_manual_completion(handle, **kwargs)


if __name__ == "__main__":
    unittest.main()
