"""Semantic evidence freshness never resets the dependency-probe budget."""

from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.rescue.gate_recovery import (
    GATE_REPEAT_LIMIT,
    GATE_RETRY_BASE_MS,
    GATE_RETRY_MAX_MS,
    GateRecovery,
    gate_evidence,
    gate_source_contract,
)


class GateEvidenceFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="gate-evidence-refresh-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        self.artifact = self.root / "data" / "upstream.json"
        self.todo = self.root / "todo.md"
        self.todo.write_text("fictional pending board\n", encoding="utf-8")
        self.log = self.root / "lane.log"
        self.attempt = self.root / "attempts.json"
        self.attempt_bytes = b'{"TASK":3,"_retry_after_ms":{"TASK":900000}}\n'
        self.attempt.write_bytes(self.attempt_bytes)
        self.blocks = {
            "UPSTREAM": {
                "status": "completed",
                "outputs": "data/upstream.json",
                "validation": "test -s data/upstream.json",
            },
            "TASK": {
                "status": "pending",
                "depends_on": "UPSTREAM",
                "deterministic_command": "python producer.py",
                "validation": "test -s data/upstream.json",
            },
        }
        self.ledger = GateRecovery(self.attempt)

    def evidence(self) -> tuple[str, dict, str]:
        fingerprint, evidence = gate_evidence(
            "TASK", self.blocks, repo_root=self.root,
            todo_path=self.todo, log_path=self.log,
        )
        return fingerprint, evidence, gate_source_contract(evidence, "c" * 64)

    def write_artifact(self, value: dict) -> None:
        self.artifact.write_text(json.dumps(value), encoding="utf-8")

    def prior_validation(self) -> tuple[str, dict, str]:
        fingerprint, evidence, contract = self.evidence()
        for index in range(GATE_REPEAT_LIMIT):
            record = self.ledger.observe(
                "TASK", fingerprint, now_ms=1000 * (index + 1),
                source_contract=contract,
            )
        record.update(
            dependencies_revalidated=True,
            dependency_validation_fingerprint=fingerprint,
            dependency_validations=[{
                "task_id": "UPSTREAM", "executed": True, "exit_code": 75,
                "evidence_fingerprint": fingerprint, "resolved_gate": False,
            }],
            repair_attempted=True,
            deterministic_repair={"exit_code": 75, "command": ["python", "repair.py"]},
        )
        self.ledger.store("TASK", record)
        return fingerprint, evidence, contract

    def assert_budget_retained(self, record: dict, count: int) -> None:
        self.assertEqual(record["source_refusals"], count)
        self.assertEqual(record["source_first_seen_ms"], 1000)
        self.assertTrue(record["repair_attempted"])
        self.assertEqual(record["deterministic_repair"]["exit_code"], 75)
        self.assertEqual(self.attempt.read_bytes(), self.attempt_bytes)
        delay = min(
            GATE_RETRY_MAX_MS,
            GATE_RETRY_BASE_MS * 2 ** min(count - GATE_REPEAT_LIMIT, 8),
        )
        self.assertEqual(record["next_retry_at_ms"] - record["last_seen_ms"], delay)

    def packet(self, record: dict, evidence: dict) -> dict:
        value = json.loads(self.ledger.packet("TASK", record, evidence).read_text())
        self.assertFalse(value["work_complete"])
        self.assertFalse(value["completion_authority"])
        self.assertEqual(self.blocks["TASK"]["status"], "pending")
        return value

    def test_missing_to_present_marks_old_validation_stale_without_rearming(self) -> None:
        old, _evidence, contract = self.prior_validation()
        prior = self.ledger.records["TASK"]["dependency_validations"]
        self.write_artifact({"available": True})
        current, evidence, same_contract = self.evidence()
        self.assertNotEqual(old, current)
        self.assertEqual(contract, same_contract)
        record = self.ledger.observe("TASK", current, now_ms=4000, source_contract=contract)
        self.assertTrue(record["dependencies_revalidated"])
        self.assertEqual(record["dependency_validations"], prior)
        packet = self.packet(record, evidence)
        self.assertEqual(packet.get("dependency_validation_evidence_fingerprint"), old)
        self.assertIs(packet.get("dependency_validations_current"), False)
        self.assertIs(packet.get("dependency_validation_budget_spent"), True)
        self.assert_budget_retained(record, 4)

    def test_same_evidence_reports_current_failed_validation_without_acceptance(self) -> None:
        fingerprint, evidence, contract = self.prior_validation()
        record = self.ledger.observe("TASK", fingerprint, now_ms=4000, source_contract=contract)
        packet = self.packet(record, evidence)
        self.assertIs(packet.get("dependency_validations_current"), True)
        self.assertIs(packet.get("dependency_validation_budget_spent"), True)
        self.assertEqual(packet["dependency_validations"][0]["exit_code"], 75)
        self.assertFalse(packet["dependency_validations"][0]["resolved_gate"])
        self.assert_budget_retained(record, 4)

    def test_clock_churn_retains_probe_spend_and_current_cache(self) -> None:
        self.write_artifact({"available": False, "checked_at_ms": 1})
        fingerprint, _evidence, contract = self.prior_validation()
        self.write_artifact({"available": False, "checked_at_ms": 9999})
        current, evidence, same_contract = self.evidence()
        self.assertEqual(current, fingerprint)
        self.assertEqual(same_contract, contract)
        record = self.ledger.observe("TASK", current, now_ms=4000, source_contract=contract)
        self.assertIs(self.packet(record, evidence).get("dependency_validations_current"), True)
        self.assert_budget_retained(record, 4)

    def test_run_id_churn_retains_probe_spend_and_current_cache(self) -> None:
        value = {"schema": "fixture/v1", "decision": "withheld",
                 "gaps": [{"code": "MISSING_INPUT"}], "run_id": "one"}
        self.write_artifact(value)
        fingerprint, _evidence, contract = self.prior_validation()
        self.write_artifact({**value, "run_id": "two", "receipt_id": "new"})
        current, evidence, same_contract = self.evidence()
        self.assertEqual(current, fingerprint)
        self.assertEqual(same_contract, contract)
        record = self.ledger.observe("TASK", current, now_ms=4000, source_contract=contract)
        self.assertIs(self.packet(record, evidence).get("dependency_validations_current"), True)
        self.assert_budget_retained(record, 4)

    def test_restart_preserves_old_origin_and_spent_budget(self) -> None:
        old, _evidence, contract = self.prior_validation()
        self.write_artifact({"available": True})
        self.ledger = GateRecovery(self.attempt)
        current, evidence, same_contract = self.evidence()
        self.assertEqual(same_contract, contract)
        record = self.ledger.observe("TASK", current, now_ms=4000, source_contract=contract)
        packet = self.packet(record, evidence)
        self.assertEqual(packet.get("dependency_validation_evidence_fingerprint"), old)
        self.assertIs(packet.get("dependency_validations_current"), False)
        self.assertIs(packet.get("dependency_validation_budget_spent"), True)
        self.assertTrue(record["dependencies_revalidated"])
        self.assert_budget_retained(record, 4)

    def test_alternating_evidence_never_rearms_or_discards_prior_failures(self) -> None:
        old, _evidence, contract = self.prior_validation()
        previous = self.ledger.records["TASK"]["dependency_validations"]
        for index in range(16):
            self.write_artifact({"available": bool(index % 2)})
            current, evidence, same_contract = self.evidence()
            self.assertEqual(same_contract, contract)
            record = self.ledger.observe("TASK", current, now_ms=4000 + index * 1000,
                                         source_contract=contract)
            packet = self.packet(record, evidence)
            self.assertTrue(record["dependencies_revalidated"])
            self.assertEqual(record["dependency_validations"], previous)
            self.assertEqual(packet.get("dependency_validation_evidence_fingerprint"), old)
            self.assertIs(packet.get("dependency_validations_current"), False)
            self.assertIs(packet.get("dependency_validation_budget_spent"), True)
            self.assert_budget_retained(record, 4 + index)

    def test_legacy_probe_origin_is_unknown_without_refunding_the_probe(self) -> None:
        fingerprint, evidence, contract = self.prior_validation()
        record = self.ledger.records["TASK"]
        record.pop("dependency_validation_fingerprint", None)
        self.ledger.store("TASK", record)
        self.ledger = GateRecovery(self.attempt)
        record = self.ledger.observe("TASK", fingerprint, now_ms=4000, source_contract=contract)
        packet = self.packet(record, evidence)
        self.assertIsNone(packet.get("dependency_validation_evidence_fingerprint"))
        self.assertIs(packet.get("dependency_validations_current"), False)
        self.assertIs(packet.get("dependency_validation_budget_spent"), True)
        self.assertEqual(len(packet["dependency_validations"]), 1)
        self.assert_budget_retained(record, 4)

    def test_interrupted_probe_reservation_cannot_be_reused(self) -> None:
        fingerprint, evidence, contract = self.evidence()
        record = self.ledger.observe("TASK", fingerprint, now_ms=1000, source_contract=contract)
        record.update(dependencies_revalidated=True, dependency_validation_fingerprint=fingerprint)
        self.ledger.store("TASK", record)
        self.ledger = GateRecovery(self.attempt)
        record = self.ledger.observe("TASK", fingerprint, now_ms=2000, source_contract=contract)
        packet = self.packet(record, evidence)
        self.assertTrue(record["dependencies_revalidated"])
        self.assertIs(packet.get("dependency_validation_budget_spent"), True)
        self.assertIs(packet.get("dependency_validations_current"), False)
        self.assertEqual(packet["dependency_validations"], [])
        self.assertEqual(self.attempt.read_bytes(), self.attempt_bytes)

    def test_new_source_contract_keeps_existing_separate_contract_policy(self) -> None:
        _old, evidence, old_contract = self.prior_validation()
        new_contract = gate_source_contract(evidence, "d" * 64)
        self.assertNotEqual(new_contract, old_contract)
        fingerprint, _evidence, _contract = self.evidence()
        record = self.ledger.observe("TASK", fingerprint, now_ms=4000, source_contract=new_contract)
        packet = self.packet(record, evidence)
        self.assertIs(packet.get("dependency_validation_budget_spent"), False)
        self.assertIs(packet.get("dependency_validations_current"), False)
        self.assertIsNone(packet.get("dependency_validation_evidence_fingerprint"))
        self.assertEqual(record["source_refusals"], 1)
        self.assertNotIn("repair_attempted", record)
        self.assertEqual(self.attempt.read_bytes(), self.attempt_bytes)


"""Draft master-dispatch controls; no real recovery/admission is exercised.

The coordinator is a no-action fixture. An opaque prerequisite mapping is
checked for preservation only; its schema and authority are never validated.
"""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TestRetainedHomeAutonomousUnstallDispatchBoundary(unittest.TestCase):
    def setUp(self):
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor

        self.module = implementation_supervisor
        self.subject = object.__new__(implementation_supervisor.PortalImplementationSupervisor)
        self.subject.config = SimpleNamespace(
            repo_root=Path("fictional-unstall-repository"),
            state_dir=Path("fictional-unstall-state"),
            state_prefix="fictional-unstall",
        )
        self.state = SimpleNamespace(
            active_task_id="fixture-task",
            active_task_cid="fixture-task-cid",
            active_attempt=1,
            active_task_started_at="fixture-owned-start",
            active_worktree_path="fixture-workspace",
        )
        # This opaque token only lets the master's guard branch continue.
        # It is not a recovery context, revision, admission or budget proof.
        self.snapshot_token = object()
        self.snapshot = Mock(return_value=(self.snapshot_token, None))
        self.subject._stale_recovery_snapshot = self.snapshot
        self.strategy = {
            "repository_root_cid": "fixture-repository-root",
            "policy_root": "fixture-policy-root",
            "run_cid": "fixture-run",
            "autonomous_unstall_policy": {
                "rescue_preview_enabled": False,
                "allow_provider_calls": False,
            },
        }
        self.load_strategy = Mock(return_value=self.strategy)
        self.subject._load_strategy = self.load_strategy
        self.coordinator = Mock(name="no_action_dispatch_fixture")
        self.fixture_result = {
            "status": "blocked",
            "recovered": False,
            "quarantined": False,
            "work_complete": False,
            "completion_authority": False,
        }
        self.coordinator.unstall.return_value = self.fixture_result
        self.coordinator_factory = Mock(return_value=self.coordinator)
        self.subject.autonomous_unstall_coordinator_class = self.coordinator_factory
        self.events = []

        def record_event(kind, payload):
            self.events.append((kind, payload))

        # Fourth declared boundary: in-memory capture of the actual10670 call.
        self.subject._record_event = record_event

    def test_snapshot_refusals_defer_before_strategy_coordinator_or_request_factory(self):
        calls = []

        def request_factory(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("master invoked a request factory before admission")

        self.subject.autonomous_unstall_rescue_execution_request_factory = request_factory
        for reason in (
            "task_state_snapshot_changed",
            "protected_hold_preserved",
            "task_source_identity_changed",
            "task_hold_preserved",
            "managed_daemon_not_quiesced",
            "dispatch_claim_not_quiesced",
            "stale_recovery_observation_unavailable",
        ):
            with self.subTest(reason=reason):
                self.snapshot.return_value = (None, reason)
                result = self.subject._run_autonomous_unstall(self.state, "fixture stale validation")
                self.assertEqual(result, {
                    "status": "deferred", "reason": reason,
                    "recovered": False, "quarantined": False,
                    "completion_authority": False,
                })
                self.load_strategy.assert_not_called()
                self.coordinator_factory.assert_not_called()
                self.coordinator.unstall.assert_not_called()
                self.assertEqual(calls, [])
                self.assertEqual(self.events, [])

    def test_optional_execution_request_factory_is_forwarded_by_identity_without_call(self):
        calls = []

        def request_factory(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("master eagerly invoked the optional request factory")

        self.subject.autonomous_unstall_rescue_execution_request_factory = request_factory
        result = self.subject._run_autonomous_unstall(self.state, "fixture stale validation")
        self.snapshot.assert_called_once_with(self.state)
        self.load_strategy.assert_called_once_with()
        self.coordinator_factory.assert_called_once()
        kwargs = self.coordinator_factory.call_args.kwargs
        self.assertIs(kwargs["rescue_execution_request_factory"], request_factory)
        self.assertIsNone(kwargs["rescue_orchestrator"])
        self.assertEqual(calls, [])
        self.coordinator.unstall.assert_called_once()
        evidence = self.coordinator.unstall.call_args.kwargs["evidence"]
        self.assertEqual(evidence["validation"]["task_id"], self.state.active_task_id)
        self.assertNotIn("prerequisite_recovery", evidence)
        self.assertIs(result, self.fixture_result)
        self.assertIs(result["completion_authority"], False)
        self.assertEqual(self.events, [("autonomous_unstall_result", result)])

    def test_default_none_and_opaque_prerequisite_packet_are_returned_and_logged_only(self):
        self.assertNotIn("autonomous_unstall_rescue_execution_request_factory", self.subject.__dict__)
        self.assertIsNone(self.subject.autonomous_unstall_rescue_execution_request_factory)
        packet = {
            "schema": "ipfs_accelerate_py/agent-supervisor/prerequisite-recovery-plan@1",
            "steps": [],
        }
        self.fixture_result["prerequisite_recovery"] = packet
        result = self.subject._run_autonomous_unstall(self.state, "fixture stale validation")
        self.snapshot.assert_called_once_with(self.state)
        self.load_strategy.assert_called_once_with()
        self.coordinator_factory.assert_called_once()
        kwargs = self.coordinator_factory.call_args.kwargs
        self.assertIsNone(kwargs["rescue_execution_request_factory"])
        self.assertIsNone(kwargs["rescue_orchestrator"])
        self.assertEqual(set(kwargs["recovery_handlers"]), {
            self.module.RescueOperation.RETRY,
            self.module.RescueOperation.QUARANTINE,
        })
        # The no-action coordinator fixture never invokes these real closures.
        # No quarantine reprobe or handler-produced state_repair is observed.
        self.coordinator.unstall.assert_called_once()
        self.assertNotIn("state_repair", result)
        self.assertIs(result, self.fixture_result)
        self.assertIs(result["prerequisite_recovery"], packet)
        self.assertIs(result["recovered"], False)
        self.assertIs(result["quarantined"], False)
        self.assertIs(result["work_complete"], False)
        self.assertIs(result["completion_authority"], False)
        self.assertEqual(self.events, [("autonomous_unstall_result", result)])
        self.assertIs(self.events[0][1]["prerequisite_recovery"], packet)


# Typed control metadata recovery: authored fictional-owner controls.
import hashlib


import json


from pathlib import Path


import tempfile


import unittest


from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    IncidentKind, RecordStatus, RecoveryAttemptOutcome, RescueOperation, prompt_workflow_cid)


from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import (
    RecoveryEvidence, RecoveryEvidenceKind, diagnose_supervisor_incident)


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    ControlMetadataRecoveryContext, ProgrammaticRecoveryController, ProgrammaticRecoveryPolicy,
    RecoveryActionObservation)


class _FictionalMetadataOwner:
    def __init__(self, root):
        self.root = root
        self.roots = {key: prompt_workflow_cid({"fictional_owner": True, "role": key})
                      for key in ("repository", "policy", "run")}
        self.current = ("revision-4", "generation-2")
        self.remaining = 2
        self.reserved = []
        self.calls = 0
        self.refuse_budget = False
        self.expire_after_action = False
        self.clock = 1
        self.artifact = root / "fixture-control-metadata.json"
        self.artifact.write_text('{"schema":"fictional-control-metadata/v1","parent":"FIX-001"}')
        self.artifact_sha = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.requirements = [{"code": "MISSING_ARTIFACT_POINTER", "subject": "fixture-parent-context",
                              "expected_sha256": self.artifact_sha}]
        self.plan = {
            "schema": "ipfs_accelerate_py/agent-supervisor/prerequisite-recovery-plan@1",
            "parent_task_id": "FIX-001", "status": "recovery_required",
            "requirements": self.requirements,
            "steps": [{"task_id": "FIX-PR-fixture-1", "kind": "resolve_retained_artifact_reference",
                       "depends_on": [], "purpose": "control_metadata_analysis", "requires_admission": True,
                       "scope_policy": "inherit_selected_unit_only"},
                      {"task_id": "FIX-PR-fixture-2", "kind": "verify_artifact_identity",
                       "depends_on": ["FIX-PR-fixture-1"], "purpose": "metadata_validation",
                       "requires_admission": True, "scope_policy": "inherit_selected_unit_only"}],
            "recheck_after": ["FIX-PR-fixture-2"], "external_requirements": [],
            "work_complete": False, "completion_authority": False, "execution_authority": False,
            "admission_granted": False, "operator_hold_removal_allowed": False, "budget_reset_allowed": False,
        }
        self.packet = {
            "schema": "ipfs_accelerate_py/agent-supervisor/gate-recovery@1", "task_id": "FIX-001",
            "state": "degraded", "phase": "gate_recovery_needed", "gate_recovery_needed": True,
            "gate_evidence_fingerprint": "a" * 64, "source_contract": "b" * 64,
            "prerequisite_recovery": self.plan, "work_complete": False, "completion_authority": False,
        }
        self.handler = self._handle

    def diagnosis(self):
        return diagnose_supervisor_incident(
            repository_root="/fixture/repository", state_root="/fixture/state",
            repository_root_cid=self.roots["repository"], policy_root=self.roots["policy"],
            run_cid=self.roots["run"], observed_at_ms=1,
            task=RecoveryEvidence(kind=RecoveryEvidenceKind.TASK, value=self.packet, target_id="FIX-001"))

    def assert_current(self):
        if self.current != ("revision-4", "generation-2"):
            raise RuntimeError("fictional owner's current revision/generation changed")

    def assert_registered(self, handler):
        if handler is not self.handler:
            raise RuntimeError("not the fictional owner's registered handler")

    def reserve(self, key):
        journals = list(self.results_dir.glob("control-metadata-*.json"))
        if len(journals) != 1:
            raise RuntimeError("reservation was not durable before owner budget callback")
        row = json.loads(journals[0].read_text())["records"][-1]
        if row["state"] != "pending":
            raise RuntimeError("scope reservation not pending")
        if self.refuse_budget or self.remaining == 0 or key in self.reserved:
            raise RuntimeError("fictional remaining metadata budget refused")
        self.remaining -= 1
        self.reserved.append(key)
        return "fictional-owner-reservation:" + key

    def factory(self, diagnosis):
        return ControlMetadataRecoveryContext(
            parent_task_id="FIX-001", parent_context_id="revision-4:generation-2",
            repository_root_cid=self.roots["repository"], policy_root=self.roots["policy"], run_cid=self.roots["run"],
            source_contract="b" * 64, gate_evidence_fingerprint="a" * 64,
            requirements_sha256=hashlib.sha256(json.dumps(self.requirements, sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            budget_scope_id="fictional-existing-owner-metadata-scope", assert_current=self.assert_current,
            assert_registered_handler=self.assert_registered, reserve_attempt=self.reserve)

    def _handle(self, context):
        self.calls += 1
        if context.control_metadata is None:
            raise RuntimeError("the handler has no selected metadata scope")
        actual = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        if actual != self.artifact_sha:
            raise RuntimeError("fictional retained artifact changed")
        if self.expire_after_action:
            self.clock = 5000
        return RecoveryActionObservation(succeeded=True, partial=True,
            observed_effects=("control_metadata_verified",),
            post_action_health={"healthy": False, "control_metadata_verified": True,
                                "original_gate_pending": True, "work_complete": False,
                                "completion_authority": False})

    def controller(self, **kwargs):
        controller = ProgrammaticRecoveryController(self.root / "state",
            handlers={RescueOperation.RESOLVE_CONTROL_METADATA: self.handler},
            control_metadata_context_factory=self.factory,
            policy=ProgrammaticRecoveryPolicy(max_attempts_per_action=1, max_total_attempts=1,
                max_actions=1, cooldown_ms=0, deadline_ms=1000),
            clock_ms=lambda: self.clock, **kwargs)
        self.results_dir = controller.results_dir
        return controller


class TestControlMetadataPositiveContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fictional-metadata-owner-")
        self.addCleanup(self.directory.cleanup)
        self.owner = _FictionalMetadataOwner(Path(self.directory.name))

    def test_verified_metadata_is_partial_not_healthy_or_task_complete(self):
        diagnosis = self.owner.diagnosis()
        self.assertEqual(diagnosis.kind, IncidentKind.METADATA_PREREQUISITE)
        self.assertEqual(diagnosis.incident.status, RecordStatus.BLOCKED)
        result = self.owner.controller().recover(diagnosis)
        self.assertTrue(result.metadata_progress)
        self.assertFalse(result.recovered)
        self.assertFalse(result.quarantined)
        self.assertEqual(result.attempts[-1].outcome, RecoveryAttemptOutcome.PARTIAL_PROGRESS)
        self.assertIs(result.receipt.post_action_health["original_gate_pending"], True)
        self.assertIs(result.receipt.post_action_health["healthy"], False)
        self.assertEqual(self.owner.calls, 1)
        self.assertEqual(len(self.owner.reserved), 1)
        self.assertEqual(self.owner.remaining, 1)

    def test_restart_uses_verified_receipt_without_budget_or_handler_replay(self):
        diagnosis = self.owner.diagnosis()
        first = self.owner.controller().recover(diagnosis)
        self.assertTrue(first.metadata_progress)
        second = self.owner.controller().recover(diagnosis)
        self.assertTrue(second.metadata_progress)
        self.assertTrue(second.deduplicated)
        self.assertEqual(first.terminal_cid, second.terminal_cid)
        self.assertEqual(self.owner.calls, 1)
        self.assertEqual(len(self.owner.reserved), 1)

    def test_actual_owner_budget_refusal_stays_blocked_and_spent(self):
        self.owner.refuse_budget = True
        result = self.owner.controller().recover(self.owner.diagnosis())
        self.assertFalse(result.metadata_progress)
        self.assertFalse(result.recovered)
        self.assertEqual(result.exhaustion_receipt.status, RecordStatus.BLOCKED)
        self.assertEqual(self.owner.calls, 0)
        self.assertEqual(len(self.owner.reserved), 0)
        again = self.owner.controller().recover(self.owner.diagnosis())
        self.assertIn("reserved_budget_exhausted", again.exhaustion_receipt.exhaustion_reason)

    def test_deadline_never_converts_metadata_effect_into_recovery(self):
        self.owner.expire_after_action = True
        result = self.owner.controller().recover(self.owner.diagnosis())
        self.assertEqual(self.owner.calls, 1)
        self.assertFalse(result.recovered)
        self.assertFalse(result.metadata_progress)
        self.assertEqual(result.exhaustion_receipt.status, RecordStatus.BLOCKED)


from copy import deepcopy as _metadata_deepcopy


import hashlib as _metadata_hashlib


import json as _metadata_json


from pathlib import Path as _MetadataPath


from tempfile import TemporaryDirectory as _MetadataTemporaryDirectory


from types import SimpleNamespace as _MetadataNamespace


import unittest as _metadata_unittest


from ipfs_accelerate_py.agent_supervisor.prompt import prompt_workflow as _metadata_prompt


from ipfs_accelerate_py.agent_supervisor.rescue import recovery_diagnostics as _metadata_diagnostics


from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_recovery as _metadata_recovery


def _metadata_json_bytes(value):
    return _metadata_json.dumps(value, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=False, allow_nan=False).encode('utf-8')


def _metadata_sha(value):
    return _metadata_hashlib.sha256(_metadata_json_bytes(value)).hexdigest()


class _MetadataFixtureInterruption(BaseException):
    """Loss of an authored callback result, not an actual process crash."""


def _authored_metadata_fixture(root, label):
    root = _MetadataPath(root)
    repository, state_dir = root / 'repository', root / 'state'
    repository.mkdir()
    state_dir.mkdir()
    policy = _metadata_recovery.ProgrammaticRecoveryPolicy(
        max_attempts_per_action=3, max_total_attempts=3, max_actions=1,
        cooldown_ms=0, deadline_ms=1000)
    parent = 'FIX-001'
    repository_cid = _metadata_prompt.prompt_workflow_cid({
        'fixture': 'metadata-only-repository', 'path': str(repository)})
    run_cid = _metadata_prompt.prompt_workflow_cid({
        'fixture': label, 'repository_root_cid': repository_cid,
        'policy_cid': policy.policy_cid, 'run': 1})
    parent_context = _metadata_prompt.prompt_workflow_cid({
        'fixture': label, 'parent_task_id': parent, 'revision': 7, 'generation': 3})
    budget_scope = _metadata_prompt.prompt_workflow_cid({
        'fixture': label, 'parent_context_id': parent_context,
        'purpose': 'fictional-control-metadata-lookup'})
    requirements = [{'code': 'MISSING_ARTIFACT_POINTER',
                     'subject': 'fixture:retained-metadata-reference',
                     'expected_sha256': _metadata_sha({'fixture': 'expected retained metadata'})}]
    false_flags = {name: False for name in (
        'work_complete', 'completion_authority', 'execution_authority',
        'admission_granted', 'operator_hold_removal_allowed', 'budget_reset_allowed')}
    steps = [
        {'task_id': 'FIX-PR-authored-1', 'kind': 'resolve_retained_artifact_reference',
         'depends_on': [], 'requirement': dict(requirements[0]),
         'purpose': 'control_metadata_analysis', 'requires_admission': True,
         'scope_policy': 'inherit_selected_unit_only'},
        {'task_id': 'FIX-PR-authored-2', 'kind': 'verify_artifact_identity',
         'depends_on': ['FIX-PR-authored-1'], 'requirement': dict(requirements[0]),
         'purpose': 'metadata_validation', 'requires_admission': True,
         'scope_policy': 'inherit_selected_unit_only'}]
    plan = {
        'schema': 'ipfs_accelerate_py/agent-supervisor/prerequisite-recovery-plan@1',
        'parent_task_id': parent, 'status': 'recovery_required',
        'requirements': requirements, 'steps': steps,
        'recheck_after': ['FIX-PR-authored-2'], 'external_requirements': [], **false_flags}
    packet = {
        'schema': 'ipfs_accelerate_py/agent-supervisor/gate-recovery@1',
        'task_id': parent, 'state': 'degraded', 'phase': 'gate_recovery_needed',
        'gate_recovery_needed': True,
        'source_contract': _metadata_sha({'fixture': label, 'source_contract': 1}),
        'gate_evidence_fingerprint': _metadata_sha({'fixture': label, 'gate_evidence': 1}),
        'prerequisite_recovery': plan, **false_flags}
    evidence = _metadata_diagnostics.RecoveryEvidence(
        kind=_metadata_diagnostics.RecoveryEvidenceKind.TASK,
        value=packet, target_id=parent, observed_at_ms=100)
    diagnosis = _metadata_diagnostics.diagnose_supervisor_incident(
        repository_root=str(repository), state_root=str(state_dir),
        repository_root_cid=repository_cid, policy_root=policy.policy_cid,
        run_cid=run_cid, task=evidence, observed_at_ms=100)
    if diagnosis.kind is not _metadata_prompt.IncidentKind.METADATA_PREREQUISITE:
        raise AssertionError('the actual typed diagnostic must select metadata prerequisites')
    if diagnosis.live_fault or diagnosis.incident.status is not _metadata_prompt.RecordStatus.BLOCKED:
        raise AssertionError('metadata prerequisites must not claim a live child fault or health')
    return _MetadataNamespace(
        state_dir=state_dir, policy=policy, diagnosis=diagnosis, packet=packet,
        parent_task_id=parent, parent_context_id=parent_context,
        repository_root_cid=repository_cid, policy_root=policy.policy_cid,
        run_cid=run_cid, budget_scope_id=budget_scope,
        requirements_sha256=_metadata_sha(requirements),
        operation=_metadata_prompt.RescueOperation.RESOLVE_CONTROL_METADATA)


class _FictionalCurrentMetadataOwner:
    """Test-only current/registry/budget capabilities, never request approval flags."""

    def __init__(self, fixture, handler, *, ledger=None, changes=None):
        self.fixture, self.handler = fixture, handler
        self.ledger = [] if ledger is None else ledger
        self.changes = dict(changes or {})
        self.registry_generation = 1
        self.current = True
        self.factory_calls = 0
        self.current_checks = 0
        self.registry_checks = 0
        self.reserve_calls = 0
        self.contexts = []

    def factory(self, diagnosis):
        self.factory_calls += 1
        if diagnosis.incident_cid != self.fixture.diagnosis.incident_cid:
            raise _metadata_recovery.RecoveryIntegrityError('fictional owner received another incident')
        expected_registry_generation = self.registry_generation
        holder = {}

        def assert_current():
            self.current_checks += 1
            selected = holder['context']
            # Parent revision and work-budget scope are the supplying owner's
            # responsibility. Other mismatched fields deliberately reach the
            # controller's own real comparison, rather than a validation stub.
            if (not self.current or
                    selected.parent_context_id != self.fixture.parent_context_id or
                    selected.budget_scope_id != self.fixture.budget_scope_id):
                raise _metadata_recovery.RecoveryIntegrityError('fictional current revision/scope changed')

        def assert_registered_handler(offered):
            self.registry_checks += 1
            if (offered is not self.handler or
                    self.registry_generation != expected_registry_generation):
                raise _metadata_recovery.RecoveryIntegrityError('fictional registered handler changed')

        def reserve_attempt(key):
            self.reserve_calls += 1
            if type(key) is not str or not key.strip():
                raise _metadata_recovery.RecoveryIntegrityError('actual reservation key required')
            if any(row['key'] == key for row in self.ledger) or len(self.ledger) >= 3:
                raise _metadata_recovery.RecoveryBoundExceeded('fictional lookup spend is not refundable')
            ticket = _metadata_prompt.prompt_workflow_cid({
                'fixture': 'lookup-reservation', 'budget_scope_id': self.fixture.budget_scope_id,
                'key': key})
            self.ledger.append({'key': key, 'ticket': ticket})
            return ticket

        values = {
            'parent_task_id': self.fixture.parent_task_id,
            'parent_context_id': self.fixture.parent_context_id,
            'repository_root_cid': self.fixture.repository_root_cid,
            'policy_root': self.fixture.policy_root, 'run_cid': self.fixture.run_cid,
            'source_contract': self.fixture.packet['source_contract'],
            'gate_evidence_fingerprint': self.fixture.packet['gate_evidence_fingerprint'],
            'requirements_sha256': self.fixture.requirements_sha256,
            'budget_scope_id': self.fixture.budget_scope_id,
            'assert_current': assert_current,
            'assert_registered_handler': assert_registered_handler,
            'reserve_attempt': reserve_attempt}
        values.update(self.changes)
        selected = _metadata_recovery.ControlMetadataRecoveryContext(**values)
        holder['context'] = selected
        self.contexts.append(selected)
        return selected


def _authored_metadata_observation(*, healthy=False, complete=False):
    return _metadata_recovery.RecoveryActionObservation(
        succeeded=True, partial=True, observed_effects=('control_metadata_verified',),
        post_action_health={
            'healthy': healthy, 'control_metadata_verified': True,
            'original_gate_pending': True, 'work_complete': complete,
            'completion_authority': False, 'admission_granted': False,
            'execution_authority': False},
        reason='fictional metadata observation, not original gate or work completion')


def _metadata_controller(fixture, handler, factory, forbidden_calls, *, include_metadata=True):
    def forbidden(context):
        forbidden_calls.append(('generic', context.action.operation))
        raise AssertionError('metadata prerequisite must not retry, restart or replay a child')

    def forbidden_health(context, observation):
        forbidden_calls.append(('health_check', context.action.operation))
        raise AssertionError('metadata-only progress must not use health restoration')

    handlers = {operation: forbidden for operation in (
        _metadata_prompt.RescueOperation.RETRY,
        _metadata_prompt.RescueOperation.RESTART,
        _metadata_prompt.RescueOperation.RESTART_LANE,
        _metadata_prompt.RescueOperation.VALIDATION_REPLAY)}
    if include_metadata:
        handlers[fixture.operation] = handler
    return _metadata_recovery.ProgrammaticRecoveryController(
        fixture.state_dir, handlers=handlers, health_check=forbidden_health,
        policy=fixture.policy, clock_ms=lambda: 100,
        control_metadata_context_factory=factory)


def _metadata_journal(state_dir):
    paths = sorted((_MetadataPath(state_dir) / 'programmatic-recovery').glob('control-metadata-*.json'))
    if len(paths) != 1:
        raise AssertionError('one actual control-metadata budget journal required')
    return paths[0], _metadata_json.loads(paths[0].read_bytes())


class TestControlMetadataScopeAndPendingRefusals(_metadata_unittest.TestCase):
    def _assert_blocked(self, result):
        self.assertIsNone(result.receipt)
        self.assertIsNotNone(result.exhaustion_receipt)
        self.assertIs(result.exhaustion_receipt.status, _metadata_prompt.RecordStatus.BLOCKED)
        self.assertFalse(result.diagnosis.live_fault)

    def test_wrong_parent_revision_roots_source_requirements_and_scope_refuse_before_handler(self):
        damages = (
            'parent_task_id', 'parent_context_id', 'repository_root_cid', 'policy_root',
            'run_cid', 'source_contract', 'gate_evidence_fingerprint',
            'requirements_sha256', 'budget_scope_id')
        for damage in damages:
            with self.subTest(damage=damage):
                with _MetadataTemporaryDirectory(prefix='control-metadata-context-') as root:
                    fixture = _authored_metadata_fixture(root, 'mismatch-' + damage)
                    calls, forbidden_calls = [], []

                    def handler(context):
                        calls.append(context.attempt)
                        return _authored_metadata_observation()

                    changed = ('FIX-002' if damage == 'parent_task_id' else
                               _metadata_sha({'fixture': 'changed ' + damage}) if damage in (
                                   'source_contract', 'gate_evidence_fingerprint', 'requirements_sha256') else
                               _metadata_prompt.prompt_workflow_cid({'fixture': 'changed ' + damage}))
                    owner = _FictionalCurrentMetadataOwner(fixture, handler, changes={damage: changed})
                    result = _metadata_controller(fixture, handler, owner.factory, forbidden_calls).recover(
                        fixture.diagnosis)
                    self._assert_blocked(result)
                    self.assertTrue(result.exhaustion_receipt.exhaustion_reason.startswith(
                        'control_metadata_context_refused:'))
                    self.assertEqual(calls, [])
                    self.assertEqual(forbidden_calls, [])
                    self.assertEqual(owner.reserve_calls, 0)
                    self.assertEqual(owner.ledger, [])
                    self.assertEqual(list((fixture.state_dir / 'programmatic-recovery').glob(
                        'control-metadata-*.json')), [])

    def test_missing_factory_missing_handler_and_untyped_factory_never_use_generic_actions(self):
        for missing in ('factory', 'handler', 'untyped_factory'):
            with self.subTest(missing=missing):
                with _MetadataTemporaryDirectory(prefix='control-metadata-provider-') as root:
                    fixture = _authored_metadata_fixture(root, 'missing-' + missing)
                    calls, forbidden_calls, unknown_factory_calls = [], [], []

                    def handler(context):
                        calls.append(context.attempt)
                        return _authored_metadata_observation()

                    owner = _FictionalCurrentMetadataOwner(fixture, handler)

                    def untyped_factory(diagnosis):
                        unknown_factory_calls.append(diagnosis.incident_cid)
                        return object()

                    factory = (None if missing == 'factory' else
                               untyped_factory if missing == 'untyped_factory' else owner.factory)
                    result = _metadata_controller(
                        fixture, handler, factory, forbidden_calls,
                        include_metadata=missing != 'handler').recover(fixture.diagnosis)
                    self._assert_blocked(result)
                    expected = ('control_metadata_context_refused:' if missing == 'untyped_factory'
                                else 'trusted_control_metadata_provider_unavailable')
                    self.assertTrue(result.exhaustion_receipt.exhaustion_reason.startswith(expected))
                    self.assertEqual(calls, [])
                    self.assertEqual(forbidden_calls, [])
                    self.assertEqual(owner.reserve_calls, 0)
                    self.assertEqual(owner.factory_calls, 0)
                    self.assertEqual(len(unknown_factory_calls), int(missing == 'untyped_factory'))

    def test_healthy_completion_and_boolean_callback_results_fail_metadata_only_outcome(self):
        for malformed in ('healthy', 'completion', 'boolean_true'):
            with self.subTest(malformed=malformed):
                with _MetadataTemporaryDirectory(prefix='control-metadata-outcome-') as root:
                    fixture = _authored_metadata_fixture(root, 'outcome-' + malformed)
                    calls, forbidden_calls = [], []

                    def handler(context):
                        calls.append(context.attempt)
                        if malformed == 'boolean_true':
                            return True
                        return _authored_metadata_observation(
                            healthy=malformed == 'healthy', complete=malformed == 'completion')

                    owner = _FictionalCurrentMetadataOwner(fixture, handler)
                    result = _metadata_controller(fixture, handler, owner.factory, forbidden_calls).recover(
                        fixture.diagnosis)
                    self._assert_blocked(result)
                    self.assertTrue(result.exhaustion_receipt.exhaustion_reason.startswith(
                        'control_metadata_attempt_failed:'))
                    self.assertEqual(calls, [1])
                    self.assertEqual(forbidden_calls, [])
                    self.assertEqual(owner.reserve_calls, 1)
                    self.assertEqual(len(owner.ledger), 1)
                    _, journal = _metadata_journal(fixture.state_dir)
                    self.assertEqual(len(journal['records']), 1)
                    self.assertEqual(journal['records'][0]['state'], 'failed')
                    self.assertTrue(journal['records'][0]['reservation_id'])
                    self.assertIsNone(journal['records'][0]['receipt'])

    def test_interrupted_metadata_callback_is_not_replayed_with_fresh_factory_same_budget_scope(self):
        with _MetadataTemporaryDirectory(prefix='control-metadata-pending-') as root:
            fixture = _authored_metadata_fixture(root, 'pending-reconstruction')
            calls, forbidden_calls = [], []

            def interrupted(context):
                calls.append(context.attempt)
                self.assertIs(context.action.operation, fixture.operation)
                self.assertEqual(context.control_metadata.budget_scope_id, fixture.budget_scope_id)
                raise _MetadataFixtureInterruption('authored missing metadata callback result')

            owner = _FictionalCurrentMetadataOwner(fixture, interrupted)
            with self.assertRaises(_MetadataFixtureInterruption):
                _metadata_controller(fixture, interrupted, owner.factory, forbidden_calls).recover(
                    fixture.diagnosis)
            journal_path, journal = _metadata_journal(fixture.state_dir)
            self.assertEqual(fixture.policy.max_attempts_per_action, 3)
            self.assertEqual(fixture.policy.max_total_attempts, 3)
            self.assertEqual(calls, [1])
            self.assertEqual(owner.reserve_calls, 1)
            self.assertEqual(len(journal['records']), 1)
            self.assertEqual(journal['records'][0]['state'], 'pending')
            self.assertTrue(journal['records'][0]['reservation_id'])
            self.assertEqual(journal['scope']['budget_scope_id'], fixture.budget_scope_id)
            before = journal_path.read_bytes()
            # A different factory object supplies a fresh context with the SAME
            # current parent and spent owner scope. This is not a new budget.
            fresh_owner = _FictionalCurrentMetadataOwner(fixture, interrupted, ledger=owner.ledger)
            result = _metadata_controller(fixture, interrupted, fresh_owner.factory, forbidden_calls).recover(
                fixture.diagnosis)
            self._assert_blocked(result)
            self.assertEqual(result.exhaustion_receipt.exhaustion_reason,
                             'unfinished_control_metadata_reservation')
            self.assertEqual(calls, [1])
            self.assertEqual(forbidden_calls, [])
            self.assertEqual(fresh_owner.factory_calls, 1)
            self.assertEqual(len(fresh_owner.contexts), 1)
            self.assertEqual(fresh_owner.reserve_calls, 0)
            self.assertEqual(len(fresh_owner.ledger), 1)
            self.assertEqual(journal_path.read_bytes(), before)


from dataclasses import fields as _metadata_receipt_fields


import os as _metadata_os


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    ProgrammaticRecoveryReceipt, RecoveryIntegrityError,
)


class TestControlMetadataReviewRegressions(unittest.TestCase):
    def test_partial_receipt_constructor_and_wire_refuse_every_authority_flag(self):
        with tempfile.TemporaryDirectory(prefix="fictional-partial-receipt-") as directory:
            owner = _FictionalMetadataOwner(Path(directory))
            result = owner.controller().recover(owner.diagnosis())
            self.assertTrue(result.metadata_progress)
            self.assertFalse(result.recovered)
            receipt = result.receipt
            constructor = {
                item.name: getattr(receipt, item.name)
                for item in _metadata_receipt_fields(receipt)
            }
            constructor["receipt_cid"] = ""
            # Missing claimed IDs cannot turn a CID-mismatch check into a false
            # positive: the unchanged constructor and wire first roundtrip.
            self.assertEqual(
                ProgrammaticRecoveryReceipt(**constructor).receipt_cid,
                receipt.receipt_cid,
            )
            self.assertEqual(
                ProgrammaticRecoveryReceipt.from_dict(
                    receipt.to_dict(include_id=False)
                ).receipt_cid,
                receipt.receipt_cid,
            )
            for flag in (
                "work_complete", "completion_authority", "execution_authority",
                "admission_granted", "operator_hold_removal_allowed",
                "budget_reset_allowed",
            ):
                with self.subTest(authority_flag=flag):
                    health = dict(receipt.post_action_health)
                    health[flag] = True
                    self.assertIs(health["healthy"], False)
                    self.assertIs(health["control_metadata_verified"], True)
                    self.assertIs(health["original_gate_pending"], True)
                    offered = dict(constructor, post_action_health=health)
                    with self.assertRaisesRegex(
                        RecoveryIntegrityError, "metadata progress cannot grant"
                    ):
                        ProgrammaticRecoveryReceipt(**offered)
                    wire = receipt.to_dict(include_id=False)
                    wire["post_action_health"] = health
                    with self.assertRaisesRegex(
                        RecoveryIntegrityError, "metadata progress cannot grant"
                    ):
                        ProgrammaticRecoveryReceipt.from_dict(wire)
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)

    def test_cached_partial_rechecks_parent_and_registry_after_journal_read(self):
        for drift in ("parent_revision", "registered_handler"):
            with self.subTest(drift=drift):
                with tempfile.TemporaryDirectory(prefix="fictional-cached-drift-") as directory:
                    owner = _FictionalMetadataOwner(Path(directory))
                    diagnosis = owner.diagnosis()
                    first = owner.controller().recover(diagnosis)
                    self.assertTrue(first.metadata_progress)
                    journal_paths = list(owner.results_dir.glob("control-metadata-*.json"))
                    self.assertEqual(len(journal_paths), 1)
                    original_journal = journal_paths[0].read_bytes()
                    original_registry_guard = owner.assert_registered
                    guard_calls = 0

                    def replacement_handler(context):
                        owner.calls += 1
                        raise AssertionError("a cached result must not replay a handler")

                    def guard_then_change_owner(handler):
                        nonlocal guard_calls
                        # Exercise the original identity validator before the
                        # owner changes; never substitute a permissive result.
                        original_registry_guard(handler)
                        guard_calls += 1
                        if guard_calls == 2:
                            if drift == "parent_revision":
                                owner.current = ("revision-5", "generation-2")
                            else:
                                owner.handler = replacement_handler

                    owner.assert_registered = guard_then_change_owner
                    controller = owner.controller()
                    blocked = controller.recover(diagnosis)
                    self.assertEqual(guard_calls, 2)
                    self.assertFalse(blocked.metadata_progress)
                    self.assertFalse(blocked.recovered)
                    self.assertFalse(blocked.quarantined)
                    self.assertEqual(blocked.exhaustion_receipt.status, RecordStatus.BLOCKED)
                    self.assertIn("RuntimeError", blocked.exhaustion_receipt.exhaustion_reason)
                    self.assertEqual(owner.calls, 1)
                    self.assertEqual(len(owner.reserved), 1)
                    self.assertEqual(owner.remaining, 1)
                    self.assertEqual(journal_paths[0].read_bytes(), original_journal)
                    again = controller.recover(diagnosis)
                    self.assertFalse(again.metadata_progress)
                    self.assertFalse(again.recovered)
                    self.assertEqual(owner.calls, 1)
                    self.assertEqual(len(owner.reserved), 1)

    def test_primary_refusal_survives_real_failure_save_permission_error(self):
        self.assertTrue(hasattr(_metadata_os, "geteuid"), "POSIX permission fixture required")
        self.assertNotEqual(
            _metadata_os.geteuid(), 0,
            "this control needs an unprivileged runner; root bypasses directory write bits",
        )
        with tempfile.TemporaryDirectory(prefix="fictional-denial-persistence-") as directory:
            owner = _FictionalMetadataOwner(Path(directory))

            class PrimaryError(RuntimeError):
                pass

            def refusing_handler(context):
                owner.calls += 1
                self.assertIsNotNone(context.control_metadata)
                self.assertEqual(len(owner.reserved), 1)
                # This is the real directory of the public controller. No
                # persistence function, file API or identity check is mocked.
                owner.results_dir.chmod(0o500)
                self.assertEqual(owner.results_dir.stat().st_mode & 0o222, 0)
                raise PrimaryError("fictional owner refuses after its durable reservation")

            owner.handler = refusing_handler
            controller = owner.controller()
            diagnosis = owner.diagnosis()
            try:
                blocked = controller.recover(diagnosis)
            finally:
                if owner.results_dir.exists():
                    owner.results_dir.chmod(0o700)
            self.assertFalse(blocked.metadata_progress)
            self.assertFalse(blocked.recovered)
            self.assertFalse(blocked.quarantined)
            receipt = blocked.exhaustion_receipt
            self.assertEqual(receipt.status, RecordStatus.BLOCKED)
            self.assertTrue(
                receipt.exhaustion_reason.startswith("control_metadata_attempt_failed:PrimaryError"),
                receipt.exhaustion_reason,
            )
            reported = json.dumps(receipt.to_record(), sort_keys=True)
            self.assertIn("PrimaryError", reported)
            self.assertIn("PermissionError", reported)
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)
            self.assertEqual(owner.remaining, 1)
            journal_paths = list(controller.results_dir.glob("control-metadata-*.json"))
            self.assertEqual(len(journal_paths), 1)
            journal = json.loads(journal_paths[0].read_text())
            self.assertEqual(len(journal["records"]), 1)
            self.assertEqual(journal["records"][0]["state"], "pending")
            self.assertTrue(journal["records"][0]["reservation_id"])
            self.assertIsNone(journal["records"][0]["receipt"])
            again = controller.recover(diagnosis)
            self.assertFalse(again.metadata_progress)
            self.assertFalse(again.recovered)
            self.assertEqual(again.exhaustion_receipt.status, RecordStatus.BLOCKED)
            self.assertIn("unfinished_control_metadata_reservation", again.exhaustion_receipt.exhaustion_reason)
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)
            self.assertEqual(owner.remaining, 1)


import os as _metadata_journal_os


class TestControlMetadataJournalTailValidation(unittest.TestCase):
    def test_completed_incident_cannot_hide_later_same_scope_pending_reservation(self):
        with tempfile.TemporaryDirectory(prefix="fictional-metadata-pending-tail-") as directory:
            owner = _FictionalMetadataOwner(Path(directory))
            controller = ProgrammaticRecoveryController(
                owner.root / "state",
                handlers={RescueOperation.RESOLVE_CONTROL_METADATA: owner.handler},
                control_metadata_context_factory=owner.factory,
                # Two fixture records remain below this bound, so a capacity
                # refusal cannot stand in for pending-custody validation.
                policy=ProgrammaticRecoveryPolicy(
                    max_attempts_per_action=3, max_total_attempts=3,
                    max_actions=1, cooldown_ms=0, deadline_ms=1000,
                ),
                clock_ms=lambda: owner.clock,
            )
            owner.results_dir = controller.results_dir
            diagnosis = owner.diagnosis()
            first = controller.recover(diagnosis)
            self.assertTrue(first.metadata_progress)
            self.assertFalse(first.recovered)
            journal_paths = list(controller.results_dir.glob("control-metadata-*.json"))
            self.assertEqual(len(journal_paths), 1)
            path = journal_paths[0]
            journal = json.loads(path.read_text())
            self.assertEqual(len(journal["records"]), 1)
            self.assertEqual(journal["records"][0]["state"], "completed")
            self.assertEqual(journal["records"][0]["incident_cid"], diagnosis.incident_cid)
            scope_before = dict(journal["scope"])

            original_fingerprint = owner.packet["gate_evidence_fingerprint"]
            try:
                owner.packet["gate_evidence_fingerprint"] = "c" * 64
                other_incident = owner.diagnosis()
            finally:
                owner.packet["gate_evidence_fingerprint"] = original_fingerprint
            self.assertNotEqual(other_incident.incident_cid, diagnosis.incident_cid)
            # Model the exact writer state after its second local reservation,
            # before the owner-budget callback. The budget/parent scope stays
            # unchanged while the semantic incident differs.
            journal["records"].append({
                "state": "pending", "incident_cid": other_incident.incident_cid,
                "reservation_id": "", "receipt": None, "error": "",
            })
            self.assertEqual(journal["scope"], scope_before)
            self.assertLess(len(journal["records"]), controller.policy.max_total_attempts)
            retained = json.dumps(
                journal, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
            with path.open("wb") as stream:
                stream.write(retained)
                stream.flush()
                _metadata_journal_os.fsync(stream.fileno())
            calls, reservations, remaining = owner.calls, tuple(owner.reserved), owner.remaining

            for _ in range(2):
                blocked = controller.recover(diagnosis)
                self.assertFalse(blocked.metadata_progress)
                self.assertFalse(blocked.recovered)
                self.assertFalse(blocked.quarantined)
                self.assertEqual(blocked.exhaustion_receipt.status, RecordStatus.BLOCKED)
                self.assertIn(
                    "unfinished_control_metadata_reservation",
                    blocked.exhaustion_receipt.exhaustion_reason,
                )
                self.assertEqual(owner.calls, calls)
                self.assertEqual(tuple(owner.reserved), reservations)
                self.assertEqual(owner.remaining, remaining)
                self.assertEqual(path.read_bytes(), retained)

    def test_completed_incident_cannot_hide_a_malformed_trailing_journal_record(self):
        with tempfile.TemporaryDirectory(prefix="fictional-metadata-malformed-tail-") as directory:
            owner = _FictionalMetadataOwner(Path(directory))
            controller = ProgrammaticRecoveryController(
                owner.root / "state",
                handlers={RescueOperation.RESOLVE_CONTROL_METADATA: owner.handler},
                control_metadata_context_factory=owner.factory,
                policy=ProgrammaticRecoveryPolicy(
                    max_attempts_per_action=3, max_total_attempts=3,
                    max_actions=1, cooldown_ms=0, deadline_ms=1000,
                ),
                clock_ms=lambda: owner.clock,
            )
            owner.results_dir = controller.results_dir
            diagnosis = owner.diagnosis()
            first = controller.recover(diagnosis)
            self.assertTrue(first.metadata_progress)
            self.assertFalse(first.recovered)
            journal_paths = list(controller.results_dir.glob("control-metadata-*.json"))
            self.assertEqual(len(journal_paths), 1)
            path = journal_paths[0]
            journal = json.loads(path.read_text())
            self.assertEqual(len(journal["records"]), 1)
            self.assertEqual(journal["records"][0]["state"], "completed")
            self.assertEqual(journal["records"][0]["incident_cid"], diagnosis.incident_cid)
            scope_before = dict(journal["scope"])
            # Explicit persisted-corruption fixture: ordinary canonical JSON,
            # correct outer scope, but an invalid later reservation state.
            journal["records"].append({"state": "not-a-control-metadata-reservation"})
            self.assertEqual(journal["scope"], scope_before)
            self.assertLess(len(journal["records"]), controller.policy.max_total_attempts)
            retained = json.dumps(
                journal, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
            with path.open("wb") as stream:
                stream.write(retained)
                stream.flush()
                _metadata_journal_os.fsync(stream.fileno())
            calls, reservations, remaining = owner.calls, tuple(owner.reserved), owner.remaining

            for _ in range(2):
                blocked = controller.recover(diagnosis)
                self.assertFalse(blocked.metadata_progress)
                self.assertFalse(blocked.recovered)
                self.assertFalse(blocked.quarantined)
                self.assertEqual(blocked.exhaustion_receipt.status, RecordStatus.BLOCKED)
                self.assertIn("RecoveryIntegrityError", blocked.exhaustion_receipt.exhaustion_reason)
                self.assertEqual(owner.calls, calls)
                self.assertEqual(tuple(owner.reserved), reservations)
                self.assertEqual(owner.remaining, remaining)
                self.assertEqual(path.read_bytes(), retained)


import json as _pending_json


from pathlib import Path as _PendingPath


from tempfile import TemporaryDirectory as _PendingTemporaryDirectory


from types import SimpleNamespace as _PendingNamespace


import unittest as _pending_unittest


from ipfs_accelerate_py.agent_supervisor.prompt import prompt_workflow as _pending_prompt


from ipfs_accelerate_py.agent_supervisor.rescue import recovery_diagnostics as _pending_diagnostics


from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_recovery as _pending_recovery


class _PendingFixtureInterruption(BaseException):
    """An authored interruption after actual durable callback reservation."""


def _pending_budget_fixture(root, label):
    root = _PendingPath(root)
    repository = root / 'repository'
    state_dir = root / 'state'
    repository.mkdir()
    state_dir.mkdir()
    policy = _pending_recovery.ProgrammaticRecoveryPolicy(
        max_attempts_per_action=3, max_total_attempts=3, max_actions=1,
        cooldown_ms=0, deadline_ms=1000)
    repository_cid = _pending_prompt.prompt_workflow_cid({
        'fixture': 'pending-recovery-budget', 'repository_root': str(repository)})
    run_cid = _pending_prompt.prompt_workflow_cid({
        'fixture': label, 'repository_root_cid': repository_cid,
        'policy_cid': policy.policy_cid, 'run': 1})
    evidence = _pending_diagnostics.RecoveryEvidence(
        kind=_pending_diagnostics.RecoveryEvidenceKind.TASK,
        target_id='FIX-001',
        value={'task_id': 'FIX-001', 'state': 'degraded',
               'objective_gap': True, 'policy_permits': True},
        observed_at_ms=100)
    diagnosis = _pending_diagnostics.diagnose_supervisor_incident(
        repository_root=str(repository), state_root=str(state_dir),
        repository_root_cid=repository_cid, policy_root=policy.policy_cid,
        run_cid=run_cid, task=evidence, observed_at_ms=100)
    operation = _pending_prompt.RescueOperation.OBJECTIVE_RECONCILE

    def new_controller(handler):
        return _pending_recovery.ProgrammaticRecoveryController(
            state_dir, policy=policy, handlers={operation: handler},
            clock_ms=lambda: 100)

    return _PendingNamespace(
        state_dir=state_dir, policy=policy, diagnosis=diagnosis,
        operation=operation, new_controller=new_controller)


def _pending_reservation_document(state_dir):
    paths = sorted((_PendingPath(state_dir) / 'programmatic-recovery').glob('*.attempts.json'))
    if len(paths) != 1:
        raise AssertionError('one actual per-incident reservation journal required')
    raw = paths[0].read_bytes()
    return paths[0], _pending_json.loads(raw)


def _pending_canonical_json(value):
    # Write authored corruptions in the existing journal's canonical JSON shape,
    # so a scope/count negative is not accidentally only a formatting negative.
    return _pending_json.dumps(value, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False).encode('utf-8')


class TestProgrammaticRecoveryPendingReservationControls(_pending_unittest.TestCase):
    def test_max_three_pending_reservation_blocks_reconstruction_without_replay(self):
        with _PendingTemporaryDirectory(prefix='recovery-pending-max3-') as root:
            fixture = _pending_budget_fixture(root, 'pending-max-three')
            calls = []

            def interrupted(context):
                calls.append((context.action.operation, context.attempt))
                raise _PendingFixtureInterruption('authored first callback interruption')

            with self.assertRaises(_PendingFixtureInterruption):
                fixture.new_controller(interrupted).recover(fixture.diagnosis)
            journal, reserved = _pending_reservation_document(fixture.state_dir)
            self.assertEqual(fixture.policy.max_attempts_per_action, 3)
            self.assertEqual(fixture.policy.max_total_attempts, 3)
            self.assertEqual(calls, [(fixture.operation, 1)])
            self.assertEqual(reserved['counts'], {fixture.operation.value: 1})
            self.assertEqual(reserved['pending']['operation'], fixture.operation.value)
            self.assertEqual(reserved['pending']['attempt'], 1)
            self.assertEqual(reserved['completed_attempts'], [])
            reserved_bytes = journal.read_bytes()

            def forbidden_replay(context):
                calls.append((context.action.operation, context.attempt))
                raise AssertionError('an interrupted callback must not be replayed')

            result = fixture.new_controller(forbidden_replay).recover(fixture.diagnosis)
            self.assertEqual(calls, [(fixture.operation, 1)])
            self.assertIsNone(result.receipt)
            self.assertIsNotNone(result.exhaustion_receipt)
            self.assertIs(result.exhaustion_receipt.status, _pending_prompt.RecordStatus.BLOCKED)
            self.assertTrue(result.exhaustion_receipt.circuit_open)
            self.assertEqual(result.exhaustion_receipt.exhaustion_reason,
                             'unfinished_reserved_recovery_attempt')
            self.assertEqual(result.attempts, ())
            self.assertEqual(journal.read_bytes(), reserved_bytes)
            self.assertEqual(_pending_json.loads(journal.read_bytes())['counts'],
                             {fixture.operation.value: 1})

    def test_first_failed_attempt_survives_second_interruption_and_reconstruction(self):
        with _PendingTemporaryDirectory(prefix='recovery-pending-failed-first-') as root:
            fixture = _pending_budget_fixture(root, 'failed-then-interrupted')
            calls = []

            def failed_then_interrupted(context):
                calls.append((context.action.operation, context.attempt))
                if context.attempt == 1:
                    return _pending_recovery.RecoveryActionObservation(
                        succeeded=False, observed_effects=(),
                        post_action_health={'healthy': False},
                        reason='authored first callback failure')
                raise _PendingFixtureInterruption('authored second callback interruption')

            with self.assertRaises(_PendingFixtureInterruption):
                fixture.new_controller(failed_then_interrupted).recover(fixture.diagnosis)
            journal, reserved = _pending_reservation_document(fixture.state_dir)
            self.assertEqual(calls, [(fixture.operation, 1), (fixture.operation, 2)])
            self.assertEqual(reserved['counts'], {fixture.operation.value: 2})
            self.assertEqual(reserved['pending']['attempt'], 2)
            self.assertEqual(len(reserved['completed_attempts']), 1)
            completed = _pending_prompt.RecoveryAttempt.from_dict(reserved['completed_attempts'][0])
            self.assertIs(completed.operation, fixture.operation)
            self.assertIs(completed.outcome, _pending_prompt.RecoveryAttemptOutcome.FAILED)
            self.assertEqual(completed.attempt, 1)
            self.assertEqual(completed.target_id, fixture.diagnosis.target_ids[0])
            self.assertEqual(completed.failure_fingerprint,
                             fixture.diagnosis.incident.failure_fingerprint)
            reserved_bytes = journal.read_bytes()

            def forbidden_third_callback(context):
                calls.append((context.action.operation, context.attempt))
                raise AssertionError('remaining numeric budget does not refund unknown custody')

            result = fixture.new_controller(forbidden_third_callback).recover(fixture.diagnosis)
            self.assertEqual(calls, [(fixture.operation, 1), (fixture.operation, 2)])
            self.assertIsNone(result.receipt)
            self.assertIsNotNone(result.exhaustion_receipt)
            self.assertIs(result.exhaustion_receipt.status, _pending_prompt.RecordStatus.BLOCKED)
            self.assertEqual(result.exhaustion_receipt.exhaustion_reason,
                             'unfinished_reserved_recovery_attempt')
            self.assertEqual(result.attempts, (completed,))
            self.assertEqual(result.exhaustion_receipt.attempts, (completed,))
            self.assertEqual(journal.read_bytes(), reserved_bytes)
            reread = _pending_json.loads(journal.read_bytes())
            self.assertEqual(reread['counts'], {fixture.operation.value: 2})
            self.assertEqual(reread['pending']['attempt'], 2)
            self.assertEqual(reread['completed_attempts'], [completed.to_record()])

    def test_corrupt_scope_or_boolean_reservation_refuses_before_callback(self):
        for damage in ('malformed_json', 'changed_scope', 'boolean_count'):
            with self.subTest(damage=damage):
                with _PendingTemporaryDirectory(prefix='recovery-pending-corruption-') as root:
                    fixture = _pending_budget_fixture(root, 'corrupt-' + damage)
                    calls = []

                    def interrupted(context):
                        calls.append((context.action.operation, context.attempt))
                        raise _PendingFixtureInterruption('authored pending journal seed')

                    with self.assertRaises(_PendingFixtureInterruption):
                        fixture.new_controller(interrupted).recover(fixture.diagnosis)
                    journal, reserved = _pending_reservation_document(fixture.state_dir)
                    self.assertEqual(reserved['counts'], {fixture.operation.value: 1})
                    self.assertIsNotNone(reserved['pending'])
                    if damage == 'malformed_json':
                        corrupt_bytes = b'{'
                    elif damage == 'changed_scope':
                        reserved['scope']['run_cid'] = _pending_prompt.prompt_workflow_cid({
                            'fixture': 'a different run cannot inherit this reservation'})
                        corrupt_bytes = _pending_canonical_json(reserved)
                    else:
                        reserved['counts'][fixture.operation.value] = True
                        corrupt_bytes = _pending_canonical_json(reserved)
                    journal.write_bytes(corrupt_bytes)
                    calls_before = tuple(calls)

                    def forbidden_callback(context):
                        calls.append((context.action.operation, context.attempt))
                        raise AssertionError('corrupt reservations must refuse before callback')

                    with self.assertRaises(_pending_recovery.RecoveryIntegrityError):
                        fixture.new_controller(forbidden_callback).recover(fixture.diagnosis)
                    self.assertEqual(tuple(calls), calls_before)
                    self.assertEqual(journal.read_bytes(), corrupt_bytes)


import hashlib


import json


from pathlib import Path


import tempfile


import unittest


from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    IncidentKind, RecordStatus, RecoveryAttemptOutcome, RescueOperation, prompt_workflow_cid)


from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import (
    RecoveryEvidence, RecoveryEvidenceKind, diagnose_supervisor_incident)


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    ControlMetadataRecoveryContext, ProgrammaticRecoveryController, ProgrammaticRecoveryPolicy,
    RecoveryActionObservation)


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    AutonomousUnstallCoordinator, AutonomousUnstallPolicy)


class _CoordinatorTrackingMetadataOwner(_FictionalMetadataOwner):
    def __init__(self, root):
        super().__init__(root)
        self.current_checks = 0
        self.registration_checks = 0

    def assert_current(self):
        self.current_checks += 1
        return super().assert_current()

    def assert_registered(self, handler):
        self.registration_checks += 1
        return super().assert_registered(handler)


class _ForbiddenMetadataEscalation:
    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def forbidden(*args, **kwargs):
            self.calls.append(("provider", name))
            raise AssertionError("metadata-only recovery must not invoke a provider or rescue executor")
        return forbidden


def _coordinator_metadata_fixture(owner, forbidden_calls, *, configured=True):
    def forbidden_handler(context):
        forbidden_calls.append(("lifecycle_or_retry", context.action.operation.value))
        raise AssertionError("metadata-only recovery must not restart, requeue or replay a child")

    def forbidden_quarantine(*args, **kwargs):
        forbidden_calls.append(("quarantine",))
        raise AssertionError("a missing metadata provider is not quarantine authority")

    def forbidden_request(*args, **kwargs):
        forbidden_calls.append(("provider_request",))
        raise AssertionError("metadata-only recovery must not build a provider request")

    provider = _ForbiddenMetadataEscalation(forbidden_calls)
    handlers = {RescueOperation.RESOLVE_CONTROL_METADATA: owner.handler}
    for operation in (RescueOperation.REPAIR_LIFECYCLE_STATE, RescueOperation.RETRY,
                      RescueOperation.RESTART, RescueOperation.RESTART_LANE,
                      RescueOperation.VALIDATION_REPLAY):
        handlers[operation] = forbidden_handler
    options = dict(
        state_dir=owner.root / "coordinator-state",
        repository_root=owner.root / "repository",
        repository_root_cid=owner.roots["repository"],
        policy_root=owner.roots["policy"], run_cid=owner.roots["run"],
        policy=AutonomousUnstallPolicy(
            deterministic_max_attempts_per_action=1, deterministic_max_total_attempts=1,
            deterministic_max_actions=1, cooldown_ms=0, deadline_ms=1000,
            rescue_preview_enabled=True, rescue_execution_enabled=True,
            allow_provider_calls=True, operating_policy_id="fictional-metadata-escalation-sentinels"),
        recovery_handlers=handlers,
        health_probe=lambda: {"healthy": False, "original_gate_pending": True},
        root_probe=lambda: {"repository_root_cid": owner.roots["repository"],
                            "policy_root": owner.roots["policy"], "run_cid": owner.roots["run"]},
        quarantine_scope=forbidden_quarantine, rescue_planner=provider,
        rescue_request_factory=forbidden_request, rescue_orchestrator=provider,
        rescue_execution_request_factory=forbidden_request,
        clock_ms=lambda: owner.clock)
    if configured:
        options["control_metadata_context_factory"] = owner.factory
    coordinator = AutonomousUnstallCoordinator(**options)
    # These two actual constructors were inspected: coordinator passes this
    # state root to ProgrammaticRecoveryController, whose results_dir appends
    # 'programmatic-recovery'. This is not the unrelated 'results' directory.
    owner.results_dir = coordinator.recovery_state_dir / "programmatic-recovery"
    return coordinator


def _coordinator_metadata_evidence(owner):
    return {"task": RecoveryEvidence(kind=RecoveryEvidenceKind.TASK,
                                      value=owner.packet, target_id="FIX-001")}


def _coordinator_metadata_journal(owner):
    paths = list(owner.results_dir.glob("control-metadata-*.json"))
    if len(paths) != 1:
        raise AssertionError("exactly one actual coordinator metadata journal required")
    return paths[0], json.loads(paths[0].read_bytes())


class TestAutonomousUnstallControlMetadataIntegration(unittest.TestCase):
    def _assert_not_recovered_or_completed(self, result):
        self.assertIs(result["recovered"], False)
        self.assertIs(result["quarantined"], False)
        self.assertIs(result["completion_authority"], False)
        self.assertIs(result["work_complete"], False)
        self.assertIs(result["health"]["healthy"], False)
        self.assertFalse(result.get("healthy", False))
        self.assertIs(result["metadata_only"], True)
        self.assertIs(result["original_gate_pending"], True)
        self.assertIs(result["admission_granted"], False)
        self.assertIs(result["execution_authority"], False)
        self.assertIs(result["operator_hold_removal_allowed"], False)
        self.assertIs(result["budget_reset_allowed"], False)
        self.assertEqual(result["target_ids"], ["FIX-001"])

    def test_configured_metadata_operation_reports_progress_without_parent_recovery(self):
        with tempfile.TemporaryDirectory(prefix="coordinator-metadata-progress-") as directory:
            owner = _CoordinatorTrackingMetadataOwner(Path(directory))
            forbidden_calls = []
            coordinator = _coordinator_metadata_fixture(owner, forbidden_calls)
            result = coordinator.unstall(evidence=_coordinator_metadata_evidence(owner))
            self.assertEqual(result["status"], "metadata_progress")
            self._assert_not_recovered_or_completed(result)
            self.assertIs(result["metadata_verified"], True)
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)
            self.assertEqual(owner.remaining, 1)
            self.assertGreater(owner.current_checks, 0)
            self.assertGreater(owner.registration_checks, 0)
            self.assertEqual(forbidden_calls, [])
            _, journal = _coordinator_metadata_journal(owner)
            self.assertEqual(len(journal["records"]), 1)
            self.assertEqual(journal["records"][0]["state"], "completed")
            receipt = journal["records"][0]["receipt"]
            self.assertEqual(receipt["operation"], RescueOperation.RESOLVE_CONTROL_METADATA.value)
            self.assertEqual(receipt["disposition"], "partial_progress")
            self.assertIs(receipt["post_action_health"]["healthy"], False)

    def test_reconstruction_reuses_metadata_receipt_and_still_checks_current_owner(self):
        with tempfile.TemporaryDirectory(prefix="coordinator-metadata-reconstruction-") as directory:
            owner = _CoordinatorTrackingMetadataOwner(Path(directory))
            forbidden_calls = []
            first = _coordinator_metadata_fixture(owner, forbidden_calls).unstall(
                evidence=_coordinator_metadata_evidence(owner))
            self.assertEqual(first["status"], "metadata_progress")
            journal_path, _ = _coordinator_metadata_journal(owner)
            before = journal_path.read_bytes()
            current_before, registration_before = owner.current_checks, owner.registration_checks
            second = _coordinator_metadata_fixture(owner, forbidden_calls).unstall(
                evidence=_coordinator_metadata_evidence(owner))
            self.assertEqual(second["status"], "metadata_progress")
            self._assert_not_recovered_or_completed(second)
            self.assertIs(second["metadata_verified"], True)
            self.assertIs(second["deduplicated"], True)
            self.assertGreater(owner.current_checks, current_before)
            self.assertGreater(owner.registration_checks, registration_before)
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)
            self.assertEqual(owner.remaining, 1)
            self.assertEqual(journal_path.read_bytes(), before)
            # The same completed receipt cannot waive the current owner's
            # revision/generation check on a later coordinator construction.
            owner.current = ("revision-5", "generation-2")
            current_before = owner.current_checks
            third = _coordinator_metadata_fixture(owner, forbidden_calls).unstall(
                evidence=_coordinator_metadata_evidence(owner))
            self.assertEqual(third["status"], "metadata_blocked")
            self._assert_not_recovered_or_completed(third)
            self.assertIs(third["metadata_verified"], False)
            self.assertGreater(owner.current_checks, current_before)
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)
            self.assertEqual(owner.remaining, 1)
            self.assertEqual(journal_path.read_bytes(), before)
            self.assertEqual(forbidden_calls, [])

    def test_default_unconfigured_factory_blocks_without_lifecycle_provider_or_quarantine(self):
        with tempfile.TemporaryDirectory(prefix="coordinator-metadata-unconfigured-") as directory:
            owner = _CoordinatorTrackingMetadataOwner(Path(directory))
            forbidden_calls = []
            # Omit the new keyword entirely: this verifies its default None.
            coordinator = _coordinator_metadata_fixture(owner, forbidden_calls, configured=False)
            result = coordinator.unstall(evidence=_coordinator_metadata_evidence(owner))
            self.assertEqual(result["status"], "metadata_blocked")
            self._assert_not_recovered_or_completed(result)
            self.assertIs(result["metadata_verified"], False)
            self.assertEqual(owner.calls, 0)
            self.assertEqual(owner.current_checks, 0)
            self.assertEqual(owner.registration_checks, 0)
            self.assertEqual(owner.reserved, [])
            self.assertEqual(owner.remaining, 2)
            self.assertEqual(forbidden_calls, [])
            self.assertEqual(list(owner.results_dir.glob("control-metadata-*.json")), [])


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    AutonomousUnstallCoordinator, AutonomousUnstallPolicy,
)


class TestControlMetadataOnlyMode(unittest.TestCase):
    def test_metadata_only_intent_refuses_untyped_gate_before_generic_callbacks(self):
        with tempfile.TemporaryDirectory(prefix="fictional-metadata-only-mode-") as directory:
            root = Path(directory)
            owner = _FictionalMetadataOwner(root)
            # The ordinary gate schema, task and fingerprint remain valid.
            # Absence of the typed plan must not select generic repair/rescue.
            owner.packet = dict(owner.packet)
            owner.packet.pop("prerequisite_recovery")
            diagnosis = owner.diagnosis()
            self.assertEqual(diagnosis.kind, IncidentKind.UNKNOWN)
            evidence = RecoveryEvidence(
                kind=RecoveryEvidenceKind.TASK, value=owner.packet, target_id="FIX-001"
            )
            calls = {"legacy_handler": 0, "quarantine_scope": 0,
                     "rescue_request": 0, "rescue_planner": 0}

            def forbidden_legacy_handler(context):
                calls["legacy_handler"] += 1
                raise AssertionError("metadata-only mode entered a generic action")

            def forbidden_quarantine_scope(targets, incident, reason):
                calls["quarantine_scope"] += 1
                raise AssertionError("untyped metadata intent cannot quarantine work")

            def forbidden_rescue_request(diagnosis, exhaustion, roots):
                calls["rescue_request"] += 1
                raise AssertionError("metadata-only mode entered provider planning")

            class ForbiddenRescuePlanner:
                def plan(self, request):
                    calls["rescue_planner"] += 1
                    raise AssertionError("metadata-only mode invoked a rescue planner")

            roots = {"repository_root_cid": owner.roots["repository"],
                     "policy_root": owner.roots["policy"], "run_cid": owner.roots["run"]}
            coordinator = AutonomousUnstallCoordinator(
                state_dir=root / "mode-state", repository_root="/fixture/repository",
                **roots,
                policy=AutonomousUnstallPolicy(
                    enabled=True, rescue_preview_enabled=True, allow_provider_calls=True,
                    rescue_execution_enabled=False,
                    operating_policy_id="fictional-metadata-only-mode",
                    max_incidents=1, max_provider_calls=1, max_rescue_executions=1,
                    deterministic_max_attempts_per_action=1,
                    deterministic_max_total_attempts=1, deterministic_max_actions=1,
                    cooldown_ms=0, deadline_ms=1000,
                ),
                recovery_handlers={operation: forbidden_legacy_handler for operation in (
                    RescueOperation.RETRY, RescueOperation.RESTART_LANE,
                    RescueOperation.OBJECTIVE_RECONCILE, RescueOperation.BACKLOG_REFILL,
                    RescueOperation.QUARANTINE,
                )},
                health_probe=lambda: {"healthy": False},
                root_probe=lambda: dict(roots),
                quarantine_scope=forbidden_quarantine_scope,
                rescue_planner=ForbiddenRescuePlanner(),
                rescue_request_factory=forbidden_rescue_request,
                clock_ms=lambda: 1,
            )
            result = coordinator.unstall(
                evidence={"task": evidence}, control_metadata_only=True
            )
            self.assertEqual(result.get("status"), "metadata_blocked")
            self.assertIs(result.get("metadata_only"), True)
            self.assertIs(result.get("recovered"), False)
            self.assertIs(result.get("quarantined"), False)
            self.assertIs(result.get("metadata_verified"), False)
            self.assertIs(result.get("original_gate_pending"), True)
            for flag in ("work_complete", "completion_authority",
                         "admission_granted", "execution_authority"):
                self.assertIs(result.get(flag), False)
            reason = str(result.get("reason") or "")
            self.assertIn("metadata", reason)
            self.assertTrue(
                any(part in reason for part in ("diagnos", "typed", "kind")), reason
            )
            self.assertEqual(calls, {name: 0 for name in calls})
            # The original outer ownership lock may exist; no incident or
            # generic recovery state is allowed before the mismatch refusal.
            self.assertFalse(coordinator.state_path.exists())
            self.assertFalse(coordinator.recovery_state_dir.exists())


import hashlib


import json


from pathlib import Path


import tempfile


import unittest


from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    IncidentKind, RecordStatus, RecoveryAttemptOutcome, RescueOperation, prompt_workflow_cid)


from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import (
    RecoveryEvidence, RecoveryEvidenceKind, diagnose_supervisor_incident)


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    ControlMetadataRecoveryContext, ProgrammaticRecoveryController, ProgrammaticRecoveryPolicy,
    RecoveryActionObservation)


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    AutonomousUnstallCoordinator, AutonomousUnstallPolicy, SupervisorWatchdog,
    read_lane_manifest)


class _OuterMetadataForbiddenProvider:
    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def forbidden(*args, **kwargs):
            self.calls["provider"].append(name)
            raise AssertionError("metadata-only outer recovery must not invoke a provider")
        return forbidden


def _outer_metadata_fixture(root, *, malformed=False):
    root = Path(root)
    owner = _FictionalMetadataOwner(root)
    repo_root, state_dir = root / "repository", root / "lane-state"
    repo_root.mkdir()
    state_dir.mkdir()
    bundle_key, state_prefix = "fixture-metadata-bundle", "fixture-metadata"
    lane = {"bundle_key": bundle_key, "state_dir": str(state_dir.resolve()),
            "state_prefix": state_prefix}
    manifest = {
        "schema": "fixture-static-autonomous-unstall/v1", "authoritative": False,
        "tree_id": "fixture-retained-tree", "lanes": [lane],
        "repository_root_cid": owner.roots["repository"],
        "policy_root": owner.roots["policy"], "run_cid": owner.roots["run"]}
    manifest_path = root / "static-lane-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                             encoding="utf-8")
    calls = {"packet": [], "coordinator": [], "health_probe": [],
             "lifecycle": [], "provider": [], "quarantine": []}
    heartbeat = {"gate_recovery_needed": True, "active_task_id": "FIX-001",
                 "gate_evidence_fingerprint": "a" * 64, "state": "degraded",
                 "phase": "gate_recovery_needed", "stale": False,
                 "state_inconsistent": False, "age_seconds": 0}
    pid_check = {"alive": True}

    def packet_provider(observed_heartbeat, observed_roots):
        calls["packet"].append((dict(observed_heartbeat), dict(observed_roots)))
        expected_roots = {"repository_root_cid": owner.roots["repository"],
                          "policy_root": owner.roots["policy"], "run_cid": owner.roots["run"]}
        if observed_roots != expected_roots or observed_heartbeat != heartbeat:
            raise AssertionError("the actual outer caller must supply its current bound roots and gate")
        packet = json.loads(json.dumps(owner.packet))
        if malformed:
            del packet["prerequisite_recovery"]
        return RecoveryEvidence(kind=RecoveryEvidenceKind.TASK, value=packet,
                                target_id="FIX-001")

    def forbidden_lifecycle(*args, **kwargs):
        calls["lifecycle"].append(True)
        raise AssertionError("metadata-only outer recovery must not restart a lane")

    def forbidden_request(*args, **kwargs):
        calls["provider"].append("request")
        raise AssertionError("metadata-only outer recovery must not prepare model rescue")

    def coordinator_factory(**kwargs):
        # This is the documented factory hook, returning the real class, not a
        # replacement coordinator or a stubbed identity/reader implementation.
        calls["coordinator"].append(dict(kwargs))
        original_quarantine = kwargs["quarantine_scope"]

        def observed_quarantine(*args, **values):
            calls["quarantine"].append(True)
            return original_quarantine(*args, **values)

        def process_health_tripwire():
            calls["health_probe"].append(True)
            raise AssertionError("metadata-only mode must not select process/status observation")

        kwargs["quarantine_scope"] = observed_quarantine
        # Replace only the public health callback with a tripwire. The outer
        # current_health closure must never be selected in metadata-only mode;
        # invoking it would inspect process/status paths outside this fixture.
        kwargs["health_probe"] = process_health_tripwire
        kwargs["clock_ms"] = lambda: owner.clock
        coordinator = AutonomousUnstallCoordinator(**kwargs)
        if coordinator.recovery_state_dir / "programmatic-recovery" != owner.results_dir:
            raise AssertionError("actual outer/controller durable-state roots must agree")
        return coordinator

    policy = AutonomousUnstallPolicy(
        deterministic_max_attempts_per_action=1, deterministic_max_total_attempts=1,
        deterministic_max_actions=1, cooldown_ms=0, deadline_ms=1000,
        rescue_preview_enabled=True, rescue_execution_enabled=True,
        allow_provider_calls=True, operating_policy_id="fictional-outer-metadata-sentinels")
    provider = _OuterMetadataForbiddenProvider(calls)
    watchdog = SupervisorWatchdog(
        manifest_path=manifest_path, repo_root=repo_root,
        lifecycle_restart=forbidden_lifecycle,
        autonomous_unstall_policy=policy,
        autonomous_unstall_factory=coordinator_factory,
        rescue_planner=provider, rescue_request_factory=forbidden_request,
        rescue_orchestrator=provider, rescue_execution_request_factory=forbidden_request,
        control_metadata_packet_provider=packet_provider,
        control_metadata_context_factory=owner.factory,
        control_metadata_handler=owner.handler)
    owner.results_dir = (state_dir / "autonomous-unstall" / "autonomous-unstall-recovery"
                         / "programmatic-recovery")
    return {"owner": owner, "watchdog": watchdog, "manifest": manifest,
            "manifest_path": manifest_path, "lane": lane, "bundle_key": bundle_key,
            "state_dir": state_dir, "state_prefix": state_prefix,
            "heartbeat": heartbeat, "pid_check": pid_check, "calls": calls}


def _invoke_outer_metadata_fixture(fixture):
    # Use the actual static-manifest reader; no process/status files or owning
    # PIDs are supplied. Its dynamic lifecycle branch is therefore inapplicable.
    manifest = read_lane_manifest(fixture["manifest_path"])
    if manifest != fixture["manifest"]:
        raise AssertionError("static manifest identity must remain exact")
    return fixture["watchdog"]._watchdog_unstall(
        manifest=manifest, lane=fixture["lane"], lane_started=dict(fixture["lane"]),
        bundle_key=fixture["bundle_key"], state_dir=fixture["state_dir"],
        state_prefix=fixture["state_prefix"], pid_check=fixture["pid_check"],
        heartbeat_check=fixture["heartbeat"])


class TestSupervisorWatchdogControlMetadataOuterCall(unittest.TestCase):
    def _assert_metadata_only_not_parent_recovery(self, result, fixture):
        self.assertIs(result["metadata_only"], True)
        self.assertIs(result["recovered"], False)
        self.assertIs(result["quarantined"], False)
        self.assertIs(result["original_gate_pending"], True)
        self.assertIs(result["work_complete"], False)
        self.assertIs(result["completion_authority"], False)
        self.assertIs(result["admission_granted"], False)
        self.assertIs(result["execution_authority"], False)
        self.assertFalse(result.get("healthy", False))
        for name in ("lifecycle", "provider", "quarantine", "health_probe"):
            self.assertEqual(fixture["calls"][name], [], name)
        self.assertEqual(len(fixture["calls"]["packet"]), 1)
        self.assertEqual(len(fixture["calls"]["coordinator"]), 1)

    def test_outer_watchdog_executes_metadata_only_without_lifecycle_escalation(self):
        with tempfile.TemporaryDirectory(prefix="outer-watchdog-metadata-") as directory:
            fixture = _outer_metadata_fixture(Path(directory))
            before_manifest = fixture["manifest_path"].read_bytes()
            result = _invoke_outer_metadata_fixture(fixture)
            self.assertEqual(result["status"], "metadata_progress")
            self._assert_metadata_only_not_parent_recovery(result, fixture)
            self.assertIs(result["metadata_verified"], True)
            self.assertIs(result["health"]["healthy"], False)
            self.assertEqual(result["target_ids"], ["FIX-001"])
            owner = fixture["owner"]
            self.assertEqual(owner.calls, 1)
            self.assertEqual(len(owner.reserved), 1)
            self.assertEqual(owner.remaining, 1)
            journals = list(owner.results_dir.glob("control-metadata-*.json"))
            self.assertEqual(len(journals), 1)
            records = json.loads(journals[0].read_bytes())["records"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["state"], "completed")
            self.assertEqual(records[0]["receipt"]["operation"],
                             RescueOperation.RESOLVE_CONTROL_METADATA.value)
            self.assertEqual(records[0]["receipt"]["disposition"], "partial_progress")
            self.assertIs(records[0]["receipt"]["post_action_health"]["healthy"], False)
            self.assertEqual(fixture["manifest_path"].read_bytes(), before_manifest)

    def test_outer_watchdog_refuses_malformed_same_schema_without_generic_recovery(self):
        with tempfile.TemporaryDirectory(prefix="outer-watchdog-malformed-metadata-") as directory:
            fixture = _outer_metadata_fixture(Path(directory), malformed=True)
            before_manifest = fixture["manifest_path"].read_bytes()
            result = _invoke_outer_metadata_fixture(fixture)
            self.assertEqual(result["status"], "metadata_blocked")
            self.assertEqual(result["reason"], "control_metadata_diagnosis_kind_refused")
            self._assert_metadata_only_not_parent_recovery(result, fixture)
            self.assertIs(result["metadata_verified"], False)
            owner = fixture["owner"]
            self.assertEqual(owner.calls, 0)
            self.assertEqual(owner.reserved, [])
            self.assertEqual(owner.remaining, 2)
            self.assertEqual(list(owner.results_dir.glob("control-metadata-*.json")), [])
            self.assertEqual(fixture["manifest_path"].read_bytes(), before_manifest)


import json


from pathlib import Path


import tempfile


import unittest


from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import prompt_workflow_cid


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    AutonomousUnstallPolicy, SupervisorWatchdog, read_lane_manifest,
)


class TestSupervisorWatchdogDefaultLegacyCompatibility(unittest.TestCase):
    def test_fresh_gate_wait_without_metadata_hooks_keeps_legacy_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root, state_dir = root / "repository", root / "lane-state"
            repo_root.mkdir()
            state_dir.mkdir()
            bundle_key, state_prefix = "fixture-legacy-bundle", "fixture-legacy"
            lane = {"bundle_key": bundle_key, "state_dir": str(state_dir.resolve()),
                    "state_prefix": state_prefix}
            roots = {name: prompt_workflow_cid({"fictional_legacy_fixture": name})
                     for name in ("repository_root_cid", "policy_root", "run_cid")}
            manifest = {"schema": "fixture-static-autonomous-unstall/v1",
                        "authoritative": False, "tree_id": "fixture-retained-tree",
                        "lanes": [lane], **roots}
            manifest_path = root / "static-lane-manifest.json"
            original_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            manifest_path.write_text(original_manifest, encoding="utf-8")
            # No owning PID, dynamic lifecycle, status or process files exist.
            # The factory spy never invokes the supplied observation callbacks.
            heartbeat = {"gate_recovery_needed": True, "active_task_id": "FIX-LEGACY-001",
                         "gate_evidence_fingerprint": "a" * 64,
                         "state": "degraded", "phase": "gate_recovery_needed",
                         "state_reason": "fictional_existing_gate_wait",
                         "stale": False, "state_inconsistent": False,
                         "age_seconds": 0.125}
            pid_check = {"alive": True, "pid": None, "reason": "fictional_live_observation"}
            policy = AutonomousUnstallPolicy(
                enabled=True, rescue_preview_enabled=False,
                rescue_execution_enabled=False, allow_provider_calls=False,
                operating_policy_id="fictional-default-legacy-compatibility")
            factory_calls, dispatch_calls = [], []
            sentinel = {"recovered": False, "quarantined": False,
                        "reason": "fictional_legacy_coordinator_dispatch"}

            class FictionalCoordinatorSpy:
                def unstall(self, **kwargs):
                    dispatch_calls.append(kwargs)
                    return sentinel

            def coordinator_factory(**kwargs):
                factory_calls.append(kwargs)
                return FictionalCoordinatorSpy()

            # Omit all new capabilities, so this same constructor call is valid
            # for the original owner and selects all-three-None in the revision.
            watchdog = SupervisorWatchdog(
                manifest_path=manifest_path, repo_root=repo_root,
                autonomous_unstall_policy=policy,
                autonomous_unstall_factory=coordinator_factory)
            observed_manifest = read_lane_manifest(manifest_path)
            self.assertEqual(observed_manifest, manifest)
            result = watchdog._watchdog_unstall(
                manifest=observed_manifest, lane=lane, lane_started=dict(lane),
                bundle_key=bundle_key, state_dir=state_dir, state_prefix=state_prefix,
                pid_check=pid_check, heartbeat_check=heartbeat)

            self.assertIs(result, sentinel)
            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(len(dispatch_calls), 1)
            factory_values = factory_calls[0]
            self.assertEqual(set(factory_values), {
                "state_dir", "repository_root", "repository_root_cid", "policy_root",
                "run_cid", "policy", "recovery_handlers", "health_probe", "root_probe",
                "quarantine_scope", "event_publisher", "rescue_planner",
                "rescue_request_factory", "rescue_orchestrator",
                "rescue_execution_request_factory"})
            self.assertEqual(factory_values["state_dir"], state_dir / "autonomous-unstall")
            self.assertEqual(factory_values["repository_root"], repo_root)
            self.assertIs(factory_values["policy"], policy)
            for name, value in roots.items():
                self.assertEqual(factory_values[name], value)
            self.assertEqual(factory_values["recovery_handlers"], {})
            for name in ("health_probe", "root_probe", "quarantine_scope"):
                self.assertTrue(callable(factory_values[name]), name)
            for name in ("event_publisher", "rescue_planner", "rescue_request_factory",
                         "rescue_orchestrator", "rescue_execution_request_factory"):
                self.assertIsNone(factory_values[name], name)
            # Exact old call shape: no task evidence, new context or mode kwarg.
            self.assertEqual(dispatch_calls[0], {"evidence": {
                "status": {"lane_id": bundle_key, "state": "degraded",
                           "state_inconsistent": False},
                "health": {"lane_id": bundle_key, "healthy": False,
                           "reason": "fictional_existing_gate_wait"},
                "process": {"lane_id": bundle_key, "alive": True,
                            "failed": False, "pid": None},
                "heartbeat": {"lane_id": bundle_key, "stale": False, "age_ms": 125}}})
            self.assertEqual(list(state_dir.iterdir()), [])
            self.assertEqual(list(repo_root.iterdir()), [])
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), original_manifest)


import json


from pathlib import Path


import tempfile


import unittest


class TestControlMetadataClockBoundary(unittest.TestCase):
    def run_clock(self, value):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        owner = _FictionalMetadataOwner(root)
        owner.clock = value
        controller = owner.controller()
        caught = None
        result = None
        try:
            result = controller.recover(owner.diagnosis())
        except Exception as exc:
            caught = {"type": type(exc).__name__, "message": str(exc)}
        rows = []
        for path in sorted(controller.results_dir.glob("control-metadata-*.json")):
            payload = json.loads(path.read_text())
            rows.append({"filename": path.name, "payload": payload})
        return {
            "clock_value": value, "clock_type": type(value).__name__,
            "owner_budget_reservations": len(owner.reserved),
            "remaining_owner_budget": owner.remaining,
            "handler_calls": owner.calls, "local_journal_count": len(rows),
            "local_journals": rows, "exception": caught,
            "returned": result is not None,
            "metadata_progress": getattr(result, "metadata_progress", None),
        }

    def test_boolean_clock_rejects_before_any_budget_or_handler(self):
        x=self.run_clock(True)
        self.assertEqual((x['owner_budget_reservations'],x['handler_calls'],x['local_journal_count']),(0,0,0),x)
        self.assertEqual(x['exception'], {
            'type': 'RecoveryIntegrityError',
            'message': 'recovery clock returned an invalid timestamp'}, x)
        self.assertFalse(x['returned'], x)
        self.assertIsNone(x['metadata_progress'], x)

    def test_negative_clock_rejects_before_any_budget_or_handler(self):
        x=self.run_clock(-1)
        self.assertEqual((x['owner_budget_reservations'],x['handler_calls'],x['local_journal_count']),(0,0,0),x)
        self.assertEqual(x['exception'], {
            'type': 'RecoveryIntegrityError',
            'message': 'recovery clock returned an invalid timestamp'}, x)
        self.assertFalse(x['returned'], x)
        self.assertIsNone(x['metadata_progress'], x)

    def test_valid_clock_is_positive_control(self):
        x=self.run_clock(1)
        self.assertEqual((x['owner_budget_reservations'],x['handler_calls']),(1,1),x)
        self.assertTrue(x['metadata_progress'],x)
        self.assertEqual(x['local_journal_count'], 1, x)

    def test_noninteger_clocks_reject_before_any_budget_or_handler(self):
        for value in (0.5, '1', None):
            with self.subTest(clock_value=value, clock_type=type(value).__name__):
                x = self.run_clock(value)
                self.assertEqual(
                    (x['owner_budget_reservations'], x['handler_calls'], x['local_journal_count']),
                    (0, 0, 0), x)
                self.assertEqual(x['exception'], {
                    'type': 'RecoveryIntegrityError',
                    'message': 'recovery clock returned an invalid timestamp'}, x)
                self.assertFalse(x['returned'], x)
                self.assertIsNone(x['metadata_progress'], x)


import ast


import copy


import hashlib


import unittest


_AUTHENTICATED_WATCHDOG_PRIORITY_SOURCE = None


_AUTHENTICATED_WATCHDOG_PRIORITY_SHA256 = None


_BASELINE_FF94_SHA256 = 'ff94d44a03831ae92d3886b9110d73e5c76119910173fc708f55ca4bd6ec814c'


_BASELINE_PREFIX_AST_SHA256 = 'cbc5200e4cc0eebee5c774e56bc834eb11d0235856968c0971a8b1c7238d35ac'


def bind_watchdog_fault_priority_source(raw, expected_sha256):
    """Bind selected cached DATA; no I/O or implicit current-source lookup."""
    if (type(raw) is not bytes or type(expected_sha256) is not str or
            len(expected_sha256) != 64 or
            any(char not in '0123456789abcdef' for char in expected_sha256) or
            hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise AssertionError('exact authenticated private watchdog bytes/SHA256 required')
    global _AUTHENTICATED_WATCHDOG_PRIORITY_SOURCE, _AUTHENTICATED_WATCHDOG_PRIORITY_SHA256
    _AUTHENTICATED_WATCHDOG_PRIORITY_SOURCE = raw
    _AUTHENTICATED_WATCHDOG_PRIORITY_SHA256 = expected_sha256
    return ExactSlicePriorityControls


def _priority_ast_dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def _actual_watchdog_priority_slice(raw, expected_sha256):
    if (type(raw) is not bytes or
            hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise AssertionError('Root must bind the selected exact private source before tests')
    tree = ast.parse(raw, filename='<authenticated-private-watchdog-priority-input>')
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SupervisorWatchdog']
    if len(classes) != 1:
        raise AssertionError('one actual SupervisorWatchdog class required')
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == '_check_cycle']
    if len(methods) != 1:
        raise AssertionError('one actual _check_cycle required')
    lane_loop = next(n for n in methods[0].body if isinstance(n, ast.For))
    metadata_indices = []
    for index, node in enumerate(lane_loop.body):
        if not isinstance(node, ast.If):
            continue
        gate_predicate = any(isinstance(n, ast.Constant) and n.value == 'gate_recovery_needed'
                             for n in ast.walk(node.test))
        metadata_dispatch = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                and n.func.attr == '_watchdog_unstall' for n in ast.walk(node))
        metadata_report = any(isinstance(n, ast.Constant) and n.value == 'control_metadata_blocked'
                              for n in ast.walk(node))
        if gate_predicate and metadata_dispatch and metadata_report:
            metadata_indices.append(index)
    if len(metadata_indices) != 1 or metadata_indices[0] < 4:
        raise AssertionError('one exact early metadata dispatch prefix required')
    metadata_index = metadata_indices[0]
    # Root's ff94 reproduction selected these same five contiguous statements.
    # The corrected predicate may shift its last source line by one; no line
    # number is used as permission to substitute another statement or branch.
    prefix = lane_loop.body[metadata_index - 4:metadata_index + 1]
    if len(prefix) != 5:
        raise AssertionError('the selected early prefix must retain five actual nodes')
    if expected_sha256 == _BASELINE_FF94_SHA256:
        if (prefix[0].lineno != 2939 or prefix[-1].end_lineno != 2974 or
                hashlib.sha256(_priority_ast_dump(prefix).encode()).hexdigest()
                != _BASELINE_PREFIX_AST_SHA256):
            raise AssertionError('baseline selector does not reproduce the authenticated ff94 slice')
    conflict_indices = [i for i, n in enumerate(lane_loop.body)
                        if isinstance(n, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == 'terminal_process_conflict'
                            for target in n.targets)]
    if len(conflict_indices) != 1 or conflict_indices[0] <= metadata_index:
        raise AssertionError('one existing later terminal_process_conflict assignment required')
    conflict_index = conflict_indices[0]
    conflict_assignment = lane_loop.body[conflict_index]
    conflict_if = lane_loop.body[conflict_index + 1]
    if (not isinstance(conflict_if, ast.If) or not any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id == 'terminal_process_conflict' for n in ast.walk(conflict_if.test))):
        raise AssertionError('the exact adjacent existing terminal classifier If is required')
    return prefix, [conflict_assignment, conflict_if]


def _make_watchdog_priority_callable(raw, expected_sha256, projection_sink):
    prefix, conflict = _actual_watchdog_priority_slice(raw, expected_sha256)
    wrapper = ast.parse("""
def execute_selected_slice(self, fixture):
    manifest = fixture['manifest']
    lane = fixture['lane']
    lane_started = fixture['lane_started']
    bundle_key = fixture['bundle_key']
    state_dir = fixture['state_dir']
    state_prefix = fixture['state_prefix']
    pid_check = fixture['pid_check']
    heartbeat_check = fixture['heartbeat_check']
    canonical_status = fixture['canonical_status']
    observed_state = fixture['observed_state']
    status_pid = heartbeat_check.get('status_pid')
    alive = bool(pid_check.get('alive'))
    reports = []
    authored_fallthrough_markers = []
    authored_classifier_end_markers = []
    for authored_iteration in [None]:
        pass
    return {'reports': reports,
            'authored_fallthrough_markers': authored_fallthrough_markers,
            'authored_classifier_end_markers': authored_classifier_end_markers,
            'heartbeat_check': heartbeat_check,
            'terminal_process_conflict': locals().get('terminal_process_conflict')}
""")
    function = wrapper.body[0]
    loop = next(n for n in function.body if isinstance(n, ast.For))
    first_marker = ast.parse(
        "authored_fallthrough_markers.append('after-exact-metadata-prefix')").body
    last_marker = ast.parse(
        "authored_classifier_end_markers.append('after-exact-terminal-classifier-not-later-action')").body
    loop.body = copy.deepcopy(prefix) + first_marker + copy.deepcopy(conflict) + last_marker
    if _priority_ast_dump(loop.body[:5]) != _priority_ast_dump(prefix):
        raise AssertionError('metadata prefix AST changed in fixture composition')
    conflict_offset = 5 + len(first_marker)
    if _priority_ast_dump(loop.body[conflict_offset:conflict_offset + 2]) != _priority_ast_dump(conflict):
        raise AssertionError('terminal classifier AST changed in fixture composition')
    ast.fix_missing_locations(wrapper)
    namespace = {'lifecycle_status_projection': projection_sink}
    exec(compile(wrapper, '<exact-watchdog-priority-prefix-and-terminal-classifier>', 'exec'), namespace)
    return namespace['execute_selected_slice']


class _PriorityFictionalMetadataCallback:
    control_metadata_packet_provider = staticmethod(lambda: None)
    control_metadata_context_factory = staticmethod(lambda: None)
    control_metadata_handler = staticmethod(lambda: None)

    def __init__(self):
        self.calls = []

    def _watchdog_unstall(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return {'metadata_verified': False, 'reason': 'fictional-unhealthy-metadata-result'}


def observe(state, *, state_inconsistent=False):
    raw, digest = _AUTHENTICATED_WATCHDOG_PRIORITY_SOURCE, _AUTHENTICATED_WATCHDOG_PRIORITY_SHA256
    if raw is None or digest is None:
        raise AssertionError('Root must call bind_watchdog_fault_priority_source with authenticated private DATA')
    projection_calls = []

    def fictional_projection_output_sink(**kwargs):
        # This output never participates in either actual product predicate.
        projection_calls.append(copy.deepcopy(kwargs))
        return {'fictional_projection_output_sink': True}

    callback = _PriorityFictionalMetadataCallback()
    fixture = {
        'manifest': {}, 'lane': {}, 'lane_started': {},
        'bundle_key': 'FICTIONAL-LANE', 'state_dir': '/fictional-unused',
        'state_prefix': 'fixture', 'canonical_status': True,
        'observed_state': state,
        'pid_check': {'pid': 424242, 'alive': True},
        'heartbeat_check': {'gate_recovery_needed': True, 'stale': False,
                            'state': state, 'status_pid': 424242}}
    if state_inconsistent:
        fixture['heartbeat_check'].update(state_inconsistent=True,
                                         state_reason='fixture_preexisting_state_inconsistent')
    execute = _make_watchdog_priority_callable(raw, digest, fictional_projection_output_sink)
    result = execute(callback, copy.deepcopy(fixture))
    result.update(
        normalized_fixture_state=state,
        fictional_metadata_calls=len(callback.calls),
        fictional_projection_calls=len(projection_calls),
        # Same meaning/assertion as Root's original prefix reproduction. The
        # marker is placed immediately AFTER those five actual product nodes.
        actual_source_continue_taken=(len(result['reports']) == 1 and
                                      not result['authored_fallthrough_markers']),
        exact_terminal_classifier_executed=bool(result['authored_classifier_end_markers']),
        real_later_fault_action_executed=False,
        real_parser_or_schema_or_enum_verified=False,
        full_check_cycle_or_runtime_qualified=False)
    return result


class ExactSlicePriorityControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Engineering callers bind authenticated retained bytes explicitly.
        # Normal owning tests read their declared imported owner, under the
        # existing normal-test Source profile and compiled-origin checks.
        if _AUTHENTICATED_WATCHDOG_PRIORITY_SOURCE is not None:
            return
        import os
        import stat
        from pathlib import Path
        from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_watchdog
        path = Path(supervisor_watchdog.__file__)
        spec = supervisor_watchdog.__spec__
        if spec is None or Path(spec.origin).resolve() != path.resolve():
            raise AssertionError("declared watchdog source origin differs")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            first = os.fstat(descriptor)
            if not stat.S_ISREG(first.st_mode) or first.st_size > 1048576:
                raise AssertionError("bounded regular watchdog source required")
            chunks = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            last = os.fstat(descriptor)
            current = os.stat(path, follow_symlinks=False)
            fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
                      "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(first, key) != getattr(last, key)
                   or getattr(first, key) != getattr(current, key) for key in fields):
                raise AssertionError("watchdog source changed during test binding")
        finally:
            os.close(descriptor)
        body = b"".join(chunks)
        bind_watchdog_fault_priority_source(body, hashlib.sha256(body).hexdigest())
    def _assert_exact_terminal_fault(self, x):
        self.assertFalse(x['actual_source_continue_taken'], x)
        self.assertEqual(x['fictional_projection_calls'], 0, x)
        self.assertEqual(x['reports'], [], x)
        self.assertTrue(x['exact_terminal_classifier_executed'], x)
        self.assertIs(x['terminal_process_conflict'], True, x)
        self.assertIs(x['heartbeat_check']['state_inconsistent'], True, x)
        self.assertEqual(x['heartbeat_check']['state_reason'], 'terminal_state_pid_alive', x)

    def test_failed_live_pid_gate_wait_must_not_enter_metadata_before_fault(self):
        x = observe('failed')
        self.assertEqual(x['fictional_metadata_calls'], 0, x)
        self._assert_exact_terminal_fault(x)
        with self.subTest('preexisting state_inconsistent remains outside metadata'):
            inconsistent = observe('blocked', state_inconsistent=True)
            self.assertEqual(inconsistent['fictional_metadata_calls'], 0, inconsistent)
            self.assertFalse(inconsistent['actual_source_continue_taken'], inconsistent)
            self.assertEqual(inconsistent['fictional_projection_calls'], 0, inconsistent)
            self.assertTrue(inconsistent['exact_terminal_classifier_executed'], inconsistent)
            self.assertIs(inconsistent['heartbeat_check']['state_inconsistent'], True, inconsistent)
            self.assertEqual(inconsistent['heartbeat_check']['state_reason'],
                             'fixture_preexisting_state_inconsistent', inconsistent)

    def test_stopped_live_pid_gate_wait_must_not_enter_metadata_before_fault(self):
        x = observe('stopped')
        self.assertEqual(x['fictional_metadata_calls'], 0, x)
        self._assert_exact_terminal_fault(x)

    def test_legitimate_blocked_wait_reaches_metadata_and_continue(self):
        x = observe('blocked')
        self.assertEqual(x['fictional_metadata_calls'], 1, x)
        self.assertTrue(x['actual_source_continue_taken'], x)
        self.assertEqual(x['reports'][0]['action'], 'control_metadata_blocked')
        self.assertFalse(x['reports'][0]['status']['work_complete'])
        self.assertFalse(x['reports'][0]['status']['completion_authority'])


from pathlib import Path


from types import SimpleNamespace


import tempfile


import unittest


from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import RescueOperation


from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    SupervisorWatchdog, read_lane_manifest)


class TestSupervisorWatchdogTerminalHeartbeatMetadataExclusion(unittest.TestCase):
    def test_terminal_gate_heartbeat_uses_generic_unstall_without_metadata_selection(self):
        for terminal_state in ("failed", "stopped"):
            with self.subTest(state=terminal_state):
                with tempfile.TemporaryDirectory(prefix="terminal-watchdog-generic-") as directory:
                    # Existing fictional owner, three complete trusted metadata
                    # hooks, static private manifest and actual manifest reader.
                    fixture = _outer_metadata_fixture(Path(directory))
                    watchdog, owner = fixture["watchdog"], fixture["owner"]
                    fixture["heartbeat"]["state"] = terminal_state
                    self.assertIs(fixture["heartbeat"]["gate_recovery_needed"], True)
                    self.assertIs(fixture["heartbeat"]["stale"], False)
                    self.assertIs(fixture["heartbeat"]["state_inconsistent"], False)
                    self.assertIs(fixture["pid_check"]["alive"], True)
                    self.assertTrue(all(callable(hook) for hook in (
                        watchdog.control_metadata_packet_provider,
                        watchdog.control_metadata_context_factory,
                        watchdog.control_metadata_handler)))
                    before_manifest = fixture["manifest_path"].read_bytes()
                    before_remaining = owner.remaining
                    factory_calls, unstall_calls = [], []
                    sentinel = object()

                    def generic_unstall_spy(**kwargs):
                        unstall_calls.append(dict(kwargs))
                        return sentinel

                    def generic_coordinator_factory_spy(**kwargs):
                        factory_calls.append(dict(kwargs))
                        # Do not call health_probe, root_probe, quarantine,
                        # lifecycle, metadata or model/provider callbacks. This
                        # is the explicit fictional coordinator boundary only.
                        return SimpleNamespace(unstall=generic_unstall_spy)

                    watchdog.autonomous_unstall_factory = generic_coordinator_factory_spy
                    manifest = read_lane_manifest(fixture["manifest_path"])
                    self.assertEqual(manifest, fixture["manifest"])
                    result = SupervisorWatchdog._watchdog_unstall(
                        watchdog, manifest=manifest, lane=fixture["lane"],
                        lane_started=dict(fixture["lane"]),
                        bundle_key=fixture["bundle_key"], state_dir=fixture["state_dir"],
                        state_prefix=fixture["state_prefix"],
                        pid_check=fixture["pid_check"], heartbeat_check=fixture["heartbeat"])

                    self.assertIs(result, sentinel)
                    self.assertEqual(fixture["calls"]["packet"], [])
                    self.assertEqual(owner.calls, 0)
                    self.assertEqual(owner.remaining, before_remaining)
                    self.assertEqual(owner.reserved, [])
                    self.assertEqual(list(owner.results_dir.glob("control-metadata-*.json")), [])
                    for name in ("health_probe", "lifecycle", "provider", "quarantine"):
                        self.assertEqual(fixture["calls"][name], [], name)
                    self.assertEqual(len(factory_calls), 1)
                    self.assertEqual(len(unstall_calls), 1)
                    factory_kwargs = factory_calls[0]
                    self.assertNotIn("control_metadata_context_factory", factory_kwargs)
                    self.assertNotIn("control_metadata_only", factory_kwargs)
                    self.assertNotIn(RescueOperation.RESOLVE_CONTROL_METADATA,
                                     factory_kwargs["recovery_handlers"])
                    self.assertTrue(all(handler is not owner.handler for handler in
                                        factory_kwargs["recovery_handlers"].values()))
                    self.assertTrue(callable(factory_kwargs["health_probe"]))
                    self.assertEqual(set(unstall_calls[0]), {"evidence"})
                    evidence = unstall_calls[0]["evidence"]
                    self.assertEqual(set(evidence), {"status", "health", "process", "heartbeat"})
                    self.assertNotIn("task", evidence)
                    self.assertEqual(evidence["status"]["state"], terminal_state)
                    self.assertIs(evidence["process"]["alive"], True)
                    self.assertIs(evidence["process"]["failed"], False)
                    self.assertIs(evidence["health"]["healthy"], False)
                    self.assertIs(evidence["heartbeat"]["stale"], False)
                    self.assertEqual(fixture["manifest_path"].read_bytes(), before_manifest)



"""Eight planner-only tests for trusted, optional read-budget observations.

All ledgers, selected scopes and files here are AUTHORED private fixtures, not
observations or grants from the real Source ledger. Tests call actual planner,
packet, Source-contract and persistent record APIs. No capture provider, kernel
immutability API, model, handler, task dispatch or Source operation is mocked or
invoked. Reference SHAPES do not authenticate their bodies: cache candidates
remain subject to independent authentication/current admission.
"""

from copy import deepcopy as _read_budget_copy
import hashlib as _read_budget_hashlib
import json as _read_budget_json
from pathlib import Path as _ReadBudgetPath
import stat as _read_budget_stat
from tempfile import TemporaryDirectory as _ReadBudgetTemporaryDirectory
import unittest as _read_budget_unittest

from ipfs_accelerate_py.agent_supervisor.rescue import gate_recovery as _read_budget_gate


_READ_BUDGET_PARENT = 'FIX-001'
_READ_BUDGET_FIELDS = (
    'st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_nlink',
    'st_size', 'st_mtime_ns', 'st_ctime_ns')


def _read_budget_digest(value):
    return _read_budget_hashlib.sha256(_read_budget_json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode('utf-8')).hexdigest()


def _read_budget_reference(path):
    path = _ReadBudgetPath(path).resolve()
    raw = path.read_bytes()
    info = path.stat()
    if not _read_budget_stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not raw:
        raise AssertionError('ordinary nonempty authored private fixture required')
    return {'path': str(path), 'bytes': len(raw),
            'sha256': _read_budget_hashlib.sha256(raw).hexdigest(),
            'full9': {name: str(getattr(info, name)) for name in _READ_BUDGET_FIELDS}}


def _read_budget_fixture(root):
    root = _ReadBudgetPath(root)
    scope = {'parent_task_id': _READ_BUDGET_PARENT,
             'parent_context_id': 'fixture:revision-7:generation-3',
             'budget_scope_id': 'fixture:retained-cumulative-read-budget',
             'source_epoch': _read_budget_digest({'fixture': 'selected source epoch'}),
             'selection_sha256': _read_budget_digest({'fixture': 'immutable selection'})}
    artifact_path = root / 'retained-control-metadata.DATA'
    artifact_path.write_bytes(b'authored retained metadata, not a Source body\n')
    artifact = _read_budget_reference(artifact_path)
    provenance_path = root / 'source-provenance.DATA'
    provenance_path.write_bytes(_read_budget_json.dumps({
        'schema': 'fictional-source-provenance/v1', 'scope': scope,
        'artifact_sha256': artifact['sha256'], 'production_authority': False},
        sort_keys=True, separators=(',', ':')).encode('utf-8'))
    provenance = _read_budget_reference(provenance_path)
    observation = {
        'scope': _read_budget_copy(scope),
        'ledger': {'global_cap': 1000, 'used': 900, 'captures': 7,
                   'internal_cap': 800, 'internal_used': 700},
        'required_source_bytes': {'global': 20, 'internal': 10},
        'authenticated_evidence': {'artifact_refs': [_read_budget_copy(artifact)],
                                   'source_provenance': _read_budget_copy(provenance)},
        'cache_candidates': [{
            'operation': 'read_authenticated_private_artifact',
            'required_fresh_Source_bytes': 0, 'scope': _read_budget_copy(scope),
            'artifact_ref': _read_budget_copy(artifact),
            'source_provenance': _read_budget_copy(provenance)}]}
    evidence = {'tasks': {_READ_BUDGET_PARENT: {}, 'FIX-003': {}}, 'artifacts': {}}
    return {'scope': scope, 'observation': observation, 'evidence': evidence,
            'artifact_path': artifact_path, 'root': root}


def _read_budget_plan(fixture, observation=None, selected_scope=None, evidence=None):
    return _read_budget_gate.prerequisite_recovery_plan(
        _READ_BUDGET_PARENT,
        fixture['evidence'] if evidence is None else evidence,
        runtime_read_budget=fixture['observation'] if observation is None else observation,
        selected_scope=fixture['scope'] if selected_scope is None else selected_scope)


def _read_budget_codes(plan):
    return {item.get('code') for item in plan['external_requirements']}


class TestGateReadBudgetWorkaroundPlanning(_read_budget_unittest.TestCase):
    def _assert_non_authorizing(self, plan):
        self.assertEqual(plan['parent_task_id'], _READ_BUDGET_PARENT)
        for name in ('work_complete', 'completion_authority', 'execution_authority',
                     'admission_granted', 'operator_hold_removal_allowed', 'budget_reset_allowed'):
            self.assertIs(plan[name], False, name)
        self.assertIsNot(plan.get('ready'), True)
        self.assertIsNot(plan.get('requeue_allowed'), True)

    def _assert_cache_requires_authentication(self, candidate, original):
        self.assertEqual(candidate['operation'], 'read_authenticated_private_artifact')
        self.assertEqual(candidate['scope'], original['scope'])
        self.assertEqual(candidate['artifact_ref'], original['artifact_ref'])
        self.assertEqual(candidate['source_provenance'], original['source_provenance'])
        self.assertIs(type(candidate['required_fresh_Source_bytes']), int)
        self.assertEqual(candidate['required_fresh_Source_bytes'], 0)
        self.assertIs(candidate['authentication_required'], True)
        for name in ('admission_granted', 'execution_authority', 'completion_authority', 'work_complete'):
            self.assertIs(candidate[name], False, name)
        self.assertIsNot(candidate.get('ready'), True)
        self.assertIsNot(candidate.get('selectable'), True)

    def _assert_insufficient(self, fixture, observation, *, evidence=None):
        saved = _read_budget_copy(observation)
        plan = _read_budget_plan(fixture, observation, evidence=evidence)
        self.assertIsNotNone(plan)
        self.assertEqual(plan['status'], 'external_dependency')
        self.assertIn('INSUFFICIENT_READ_BUDGET', _read_budget_codes(plan))
        self._assert_non_authorizing(plan)
        self.assertEqual(plan['read_budget']['scope'], fixture['scope'])
        self.assertEqual(plan['read_budget']['ledger'], observation['ledger'])
        self.assertEqual(plan['read_budget']['required_source_bytes'], observation['required_source_bytes'])
        self.assertEqual(plan['read_budget']['remaining_source_bytes'], {
            'global': observation['ledger']['global_cap'] - observation['ledger']['used'],
            'internal': observation['ledger']['internal_cap'] - observation['ledger']['internal_used']})
        self.assertEqual(observation, saved)
        return plan

    def test_default_none_retains_legacy_plan_and_packet_without_new_keys(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-none-') as directory:
            fixture = _read_budget_fixture(directory)
            evidence = _read_budget_copy(fixture['evidence'])
            evidence['tasks'][_READ_BUDGET_PARENT]['prerequisite_requirements'] = _read_budget_json.dumps([
                {'code': 'MISSING_OWNER_POINTER', 'subject': 'fixture:existing owner pointer'}])
            before = _read_budget_gate.prerequisite_recovery_plan(_READ_BUDGET_PARENT, evidence)
            after = _read_budget_gate.prerequisite_recovery_plan(
                _READ_BUDGET_PARENT, evidence, runtime_read_budget=None, selected_scope=fixture['scope'])
            self.assertIsNotNone(before)
            self.assertEqual(after, before)
            self.assertNotIn('read_budget', after)
            self.assertNotIn('cache_candidates', after)
            self.assertIsNone(_read_budget_gate.prerequisite_recovery_plan(
                _READ_BUDGET_PARENT, fixture['evidence'], runtime_read_budget=None))
            recovery = _read_budget_gate.GateRecovery(_ReadBudgetPath(directory) / 'attempts.json')
            contract = _read_budget_gate.gate_source_contract(evidence, 'fixture:command-contract')
            record = recovery.observe(_READ_BUDGET_PARENT, 'a' * 64, now_ms=100, source_contract=contract)
            old_path = recovery.packet(_READ_BUDGET_PARENT, record, evidence)
            old_raw = old_path.read_bytes()
            new_path = recovery.packet(_READ_BUDGET_PARENT, record, evidence,
                                       runtime_read_budget=None, selected_scope=fixture['scope'])
            self.assertEqual(new_path.read_bytes(), old_raw)

    def test_internal_only_deficit_retains_cache_and_existing_requirements(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-internal-') as directory:
            fixture = _read_budget_fixture(directory)
            observation = _read_budget_copy(fixture['observation'])
            observation['ledger']['internal_used'] = 795
            evidence = _read_budget_copy(fixture['evidence'])
            evidence['tasks'][_READ_BUDGET_PARENT]['prerequisite_requirements'] = _read_budget_json.dumps([
                {'code': 'MISSING_OWNER_POINTER', 'subject': 'fixture:existing owner pointer'}])
            legacy = _read_budget_gate.prerequisite_recovery_plan(_READ_BUDGET_PARENT, evidence)
            plan = self._assert_insufficient(fixture, observation, evidence=evidence)
            self.assertGreaterEqual(1000 - 900, observation['required_source_bytes']['global'])
            self.assertLess(800 - 795, observation['required_source_bytes']['internal'])
            for requirement in legacy['requirements']:
                self.assertIn(requirement, plan['requirements'])
            self.assertEqual(len(plan['cache_candidates']), 1)
            self._assert_cache_requires_authentication(plan['cache_candidates'][0], observation['cache_candidates'][0])

    def test_global_only_deficit_retains_cache_without_borrowing_internal_headroom(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-global-') as directory:
            fixture = _read_budget_fixture(directory)
            observation = _read_budget_copy(fixture['observation'])
            observation['ledger']['used'] = 995
            plan = self._assert_insufficient(fixture, observation)
            self.assertLess(1000 - 995, observation['required_source_bytes']['global'])
            self.assertGreaterEqual(800 - 700, observation['required_source_bytes']['internal'])
            self.assertEqual(len(plan['cache_candidates']), 1)
            self._assert_cache_requires_authentication(plan['cache_candidates'][0], observation['cache_candidates'][0])

    def test_both_deficits_exact_boundaries_zero_cost_and_large_python_cost(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-boundaries-') as directory:
            fixture = _read_budget_fixture(directory)
            both = _read_budget_copy(fixture['observation'])
            both['ledger'].update(used=995, internal_used=795)
            self._assert_insufficient(fixture, both)
            with self.subTest('exact two-ledger boundary is not a budget prerequisite'):
                exact = _read_budget_copy(fixture['observation'])
                exact['ledger'].update(used=980, internal_used=790)
                self.assertIsNone(_read_budget_plan(fixture, exact))
            with self.subTest('zero cost remains zero even at fully spent counters'):
                zero = _read_budget_copy(fixture['observation'])
                zero['ledger'].update(used=1000, internal_used=800)
                zero['required_source_bytes'] = {'global': 0, 'internal': 0}
                self.assertIsNone(_read_budget_plan(fixture, zero))
            with self.subTest('large exact Python integer is insufficient rather than wrapped'):
                huge = _read_budget_copy(fixture['observation'])
                huge['required_source_bytes'] = {'global': 2**200, 'internal': 2**199}
                plan = self._assert_insufficient(fixture, huge)
                self.assertNotIn('INVALID_READ_BUDGET_CONTEXT', _read_budget_codes(plan))
                self.assertEqual(plan['read_budget']['required_source_bytes']['global'], 2**200)

    def test_malformed_cost_ledger_scope_and_reference_contexts_fail_closed(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-invalid-') as directory:
            fixture = _read_budget_fixture(directory)
            cases = []
            def add(label, observation, scope=None):
                cases.append((label, observation, fixture['scope'] if scope is None else scope))
            for field in ('global', 'internal'):
                for value in (None, True, 1.0, -1, 'UNKNOWN'):
                    obs = _read_budget_copy(fixture['observation'])
                    obs['required_source_bytes'][field] = value
                    add('bad-cost-' + field + '-' + repr(value), obs)
                obs = _read_budget_copy(fixture['observation'])
                del obs['required_source_bytes'][field]
                add('missing-cost-' + field, obs)
            obs = _read_budget_copy(fixture['observation'])
            del obs['required_source_bytes']
            add('missing-cost-object', obs)
            obs = _read_budget_copy(fixture['observation'])
            obs['required_source_bytes'] = {'global': 10, 'internal': 11}
            add('internal-cost-exceeds-global', obs)
            for field in ('global_cap', 'used', 'captures', 'internal_cap', 'internal_used'):
                for value in (True, 1.0, -1):
                    obs = _read_budget_copy(fixture['observation'])
                    obs['ledger'][field] = value
                    add('bad-ledger-' + field + '-' + repr(value), obs)
                obs = _read_budget_copy(fixture['observation'])
                del obs['ledger'][field]
                add('missing-ledger-' + field, obs)
            for field, value in (('used', 1001), ('internal_used', 801)):
                obs = _read_budget_copy(fixture['observation'])
                obs['ledger'][field] = value
                add('overspent-' + field, obs)
            for field in fixture['scope']:
                obs = _read_budget_copy(fixture['observation'])
                obs['scope'][field] = ('f' * 64 if field in ('selection_sha256', 'source_epoch')
                                       else 'fixture:another-' + field)
                add('mismatched-observation-' + field, obs)
                scope = _read_budget_copy(fixture['scope'])
                del scope[field]
                add('missing-selected-' + field, _read_budget_copy(fixture['observation']), scope)
            scope = _read_budget_copy(fixture['scope'])
            scope['parent_task_id'] = 'FIX-999'
            obs = _read_budget_copy(fixture['observation'])
            obs['scope'] = _read_budget_copy(scope)
            add('different-selected-parent-even-if-observation-agrees', obs, scope)
            for field in fixture['scope']:
                scope = _read_budget_copy(fixture['scope'])
                scope[field] = ''
                add('empty-selected-' + field, _read_budget_copy(fixture['observation']), scope)
            obs = _read_budget_copy(fixture['observation'])
            del obs['authenticated_evidence']
            add('missing-evidence', obs)
            for field in ('artifact_refs', 'source_provenance'):
                obs = _read_budget_copy(fixture['observation'])
                del obs['authenticated_evidence'][field]
                add('missing-evidence-' + field, obs)
            obs = _read_budget_copy(fixture['observation'])
            obs['authenticated_evidence']['artifact_refs'] = []
            add('empty-evidence-artifacts', obs)
            for field, value in (('path', 'relative.DATA'), ('bytes', True), ('bytes', 0), ('sha256', 'x' * 64)):
                obs = _read_budget_copy(fixture['observation'])
                obs['authenticated_evidence']['artifact_refs'][0][field] = value
                add('bad-reference-' + field + '-' + repr(value), obs)
            for change in ('missing-field', 'nonstring-ns', 'nonregular', 'multiple-links', 'size-mismatch'):
                obs = _read_budget_copy(fixture['observation'])
                ref = obs['authenticated_evidence']['source_provenance']
                if change == 'missing-field': del ref['full9']['st_ctime_ns']
                elif change == 'nonstring-ns': ref['full9']['st_mtime_ns'] = 2**53 + 1
                elif change == 'nonregular': ref['full9']['st_mode'] = str(_read_budget_stat.S_IFDIR | 0o700)
                elif change == 'multiple-links': ref['full9']['st_nlink'] = '2'
                else: ref['full9']['st_size'] = str(ref['bytes'] + 1)
                add('bad-full9-' + change, obs)
            for label, observation, selected_scope in cases:
                with self.subTest(case=label):
                    plan = _read_budget_gate.prerequisite_recovery_plan(
                        _READ_BUDGET_PARENT, fixture['evidence'],
                        runtime_read_budget=observation, selected_scope=selected_scope)
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan['status'], 'external_dependency')
                    self.assertIn('INVALID_READ_BUDGET_CONTEXT', _read_budget_codes(plan))
                    self.assertEqual(plan['cache_candidates'], [])
                    self._assert_non_authorizing(plan)
            with self.subTest('runtime observation without separately selected scope'):
                plan = _read_budget_gate.prerequisite_recovery_plan(
                    _READ_BUDGET_PARENT, fixture['evidence'],
                    runtime_read_budget=fixture['observation'], selected_scope=None)
                self.assertIn('INVALID_READ_BUDGET_CONTEXT', _read_budget_codes(plan))
                self.assertEqual(plan['cache_candidates'], [])

    def test_empty_independent_cache_frontier_is_task_local_deferral_not_completion(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-empty-frontier-') as directory:
            fixture = _read_budget_fixture(directory)
            observation = _read_budget_copy(fixture['observation'])
            observation['ledger'].update(used=995, internal_used=795)
            observation['cache_candidates'] = []
            before_evidence = _read_budget_copy(fixture['evidence'])
            plan = self._assert_insufficient(fixture, observation)
            self.assertEqual(plan['cache_candidates'], [])
            self._assert_non_authorizing(plan)
            self.assertEqual(fixture['evidence'], before_evidence)
            self.assertIsNone(_read_budget_gate.prerequisite_recovery_plan('FIX-003', fixture['evidence']))

    def test_packet_counter_snapshots_do_not_change_source_contract_or_spent_history(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-packet-history-') as directory:
            fixture = _read_budget_fixture(directory)
            evidence = _read_budget_copy(fixture['evidence'])
            evidence['tasks'][_READ_BUDGET_PARENT]['depends_on'] = 'FIX-002'
            evidence['tasks']['FIX-002'] = {'validation': 'test -f fixture-only.DATA'}
            command_contract = _read_budget_digest({'fixture': 'unchanged producer/validator contract'})
            contract = _read_budget_gate.gate_source_contract(evidence, command_contract)
            fingerprint = _read_budget_digest({'fixture': 'unchanged actual gate evidence'})
            recovery = _read_budget_gate.GateRecovery(_ReadBudgetPath(directory) / 'attempts.json')
            for ordinal in range(3):
                record = recovery.observe(_READ_BUDGET_PARENT, fingerprint,
                                          now_ms=100 + ordinal, source_contract=contract)
            record.update(dependencies_revalidated=True,
                          dependency_validation_fingerprint=fingerprint,
                          dependency_validations=[{'task_id': 'FIX-002', 'validation': 'test -f fixture-only.DATA',
                                                   'exit_code': 1, 'executed': True,
                                                   'validation_strength': 'existence_only', 'resolved_gate': False}],
                          repair_attempted=True,
                          deterministic_repair={'command': ['fixture:unexecuted-repair'], 'exit_code': 1})
            recovery.store(_READ_BUDGET_PARENT, record)
            records_before = _read_budget_copy(recovery.records)
            durable_before = recovery.path.read_bytes()
            snapshots = []
            for used, internal_used, captures in ((995, 795, 8), (999, 799, 17)):
                observation = _read_budget_copy(fixture['observation'])
                observation['ledger'].update(used=used, internal_used=internal_used, captures=captures)
                packet_path = recovery.packet(_READ_BUDGET_PARENT, record, evidence,
                    runtime_read_budget=observation, selected_scope=fixture['scope'])
                packet = _read_budget_json.loads(packet_path.read_bytes())
                snapshots.append(packet['prerequisite_recovery']['read_budget']['ledger'])
                self.assertEqual(packet['gate_evidence_fingerprint'], fingerprint)
                self.assertEqual(packet['source_contract'], contract)
                self.assertIs(packet['dependency_validation_budget_spent'], True)
                self.assertEqual(packet['dependency_validation_evidence_fingerprint'], fingerprint)
                self.assertEqual(packet['dependency_validations'], record['dependency_validations'])
                self.assertEqual(packet['deterministic_repair'], record['deterministic_repair'])
                self.assertEqual(packet['source_refusals'], record['source_refusals'])
                self.assertIs(packet['work_complete'], False)
                self.assertIs(packet['completion_authority'], False)
                self.assertEqual(_read_budget_gate.gate_source_contract(evidence, command_contract), contract)
                self.assertEqual(recovery.records, records_before)
                self.assertEqual(recovery.path.read_bytes(), durable_before)
            self.assertNotEqual(snapshots[0], snapshots[1])

    def test_cache_reference_shape_never_authenticates_changed_private_body(self):
        with _ReadBudgetTemporaryDirectory(prefix='read-budget-auth-required-') as directory:
            fixture = _read_budget_fixture(directory)
            observation = _read_budget_copy(fixture['observation'])
            observation['ledger'].update(used=995, internal_used=795)
            before = self._assert_insufficient(fixture, observation)
            old_bytes = fixture['artifact_path'].read_bytes()
            fixture['artifact_path'].write_bytes(b'X' * len(old_bytes))
            self.assertNotEqual(_read_budget_hashlib.sha256(fixture['artifact_path'].read_bytes()).hexdigest(),
                                observation['cache_candidates'][0]['artifact_ref']['sha256'])
            after = self._assert_insufficient(fixture, observation)
            # Planner is deliberately a pure metadata planner, not an artifact
            # reader/authenticator. Its same candidate remains UNVERIFIED.
            self.assertEqual(after, before)
            self.assertEqual(len(after['cache_candidates']), 1)
            self._assert_cache_requires_authentication(
                after['cache_candidates'][0], observation['cache_candidates'][0])


# Distinct dependency requests must retain distinct durable handoff files.
import json
import os
import tempfile
import unittest
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.rescue import gate_recovery as _handoff_gate


class TestGateHandoffPacketIdentity(unittest.TestCase):
    def packet_for(self, recovery, task_id, fingerprint="a" * 64):
        record = {"fingerprint": fingerprint, "source_contract": "b" * 64,
                  "source_refusals": 3, "unchanged_probes": 3,
                  "dependencies_revalidated": True, "dependency_validations": [],
                  "next_retry_at_ms": 1234}
        return recovery.packet(task_id, record, {"tasks": {}, "artifacts": {}})

    def test_long_distinct_parent_requests_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            recovery = _handoff_gate.GateRecovery(Path(directory) / "attempts.json")
            first_id, second_id = "LAW-" + "a" * 90 + "-owner", "LAW-" + "a" * 90 + "-test"
            first = self.packet_for(recovery, first_id)
            before = first.read_bytes()
            second = self.packet_for(recovery, second_id)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(json.loads(first.read_text())["task_id"], first_id)
            self.assertEqual(json.loads(second.read_text())["task_id"], second_id)

    def test_existing_two_stage_planner_ids_have_separate_handoffs(self):
        parent = "AUTOMATEDLEGALFORMALIZATION-02"
        evidence = {"tasks": {parent: {"prerequisite_requirements": json.dumps([
            {"code": "LOSSLESS_CACHE_UNAVAILABLE", "subject": "fictional retained owner body",
             "expected_sha256": "c" * 64}])}}, "artifacts": {}}
        plan = _handoff_gate.prerequisite_recovery_plan(parent, evidence)
        self.assertEqual(len(plan["steps"]), 2)
        ids = [step["task_id"] for step in plan["steps"]]
        self.assertGreater(len(ids[0]), 80)
        self.assertEqual(ids[0][:80], ids[1][:80])
        with tempfile.TemporaryDirectory() as directory:
            recovery = _handoff_gate.GateRecovery(Path(directory) / "attempts.json")
            paths = [self.packet_for(recovery, task) for task in ids]
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual([json.loads(path.read_text())["task_id"] for path in paths], ids)

    def test_legacy_short_safe_task_paths_and_false_authority_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            recovery = _handoff_gate.GateRecovery(Path(directory) / "attempts.json")
            for task in ("LIR-02", "LAW_101", "A" * 80):
                with self.subTest(task=task):
                    path = self.packet_for(recovery, task)
                    self.assertEqual(path.name, "gate-recovery-" + task + ".json")
                    payload = json.loads(path.read_text())
                    self.assertIs(payload["work_complete"], False)
                    self.assertIs(payload["completion_authority"], False)

    def test_restart_keeps_long_request_identity_and_spent_validation_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempts.json"
            task = "LAW-" + "p" * 100
            recovery = _handoff_gate.GateRecovery(attempt)
            record = recovery.observe(task, "a" * 64, now_ms=1, source_contract="b" * 64)
            record.update(dependencies_revalidated=True, repair_attempted=True)
            recovery.store(task, record)
            first = recovery.packet(task, record, {"tasks": {}, "artifacts": {}})
            retained = recovery.path.read_bytes()
            restarted = _handoff_gate.GateRecovery(attempt)
            second = restarted.packet(task, restarted.current(task, "a" * 64),
                                      {"tasks": {}, "artifacts": {}})
            self.assertEqual(first, second)
            self.assertEqual(restarted.path.read_bytes(), retained)
            self.assertIs(restarted.current(task, "a" * 64)["dependencies_revalidated"], True)
            self.assertIs(restarted.current(task, "a" * 64)["repair_attempted"], True)
            self.assertLessEqual(len(os.fsencode(second.name)), 255)

    def test_same_request_can_update_evidence_without_changing_identity_or_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            recovery = _handoff_gate.GateRecovery(Path(directory) / "attempts.json")
            task = "LAW-" + "q" * 100
            first = self.packet_for(recovery, task)
            second = self.packet_for(recovery, task, "d" * 64)
            self.assertEqual(first, second)
            payload = json.loads(second.read_text())
            self.assertEqual(payload["task_id"], task)
            self.assertEqual(payload["gate_evidence_fingerprint"], "d" * 64)
            self.assertIs(payload["work_complete"], False)
            self.assertIs(payload["completion_authority"], False)


import hashlib as _handoff_hashlib


class TestGateHandoffNormalizedIdentity(unittest.TestCase):
    def test_normalized_request_and_legacy_safe_digest_spelling_cannot_alias(self):
        task = "LAW-1/"
        legacy = "LAW-1--" + _handoff_hashlib.sha256(task.encode("utf-8")).hexdigest()
        self.assertLessEqual(len(legacy), 80)
        with tempfile.TemporaryDirectory() as directory:
            recovery = _handoff_gate.GateRecovery(Path(directory) / "attempts.json")
            first = TestGateHandoffPacketIdentity.packet_for(self, recovery, task)
            retained = first.read_bytes()
            second = TestGateHandoffPacketIdentity.packet_for(self, recovery, legacy)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), retained)
            self.assertEqual(json.loads(first.read_text())["task_id"], task)
            self.assertEqual(json.loads(second.read_text())["task_id"], legacy)
