"""Reconcile actual finite-controller receipts; never synthesize a success."""
from copy import deepcopy
import fcntl
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import prompt_workflow_cid
from ipfs_accelerate_py.agent_supervisor.rescue.campaign_watchdog import (
    CampaignProgressWatchdog, CampaignProgressWatchdogPolicy, OBSERVATION_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import RecoveryActionObservation


@pytest.fixture
def recovery(tmp_path):
    class Recovery:
        now = 1000
        calls = 0
        mode = "health_pending"
        scope = {key: prompt_workflow_cid({"scope": key}) for key in
            ("repository_root_cid", "policy_root", "run_cid")}
        value = {"schema": OBSERVATION_SCHEMA, "scope": scope,
            "status": "stalled", "healthy": False, "completion_verified": False,
            "manual_hold": False, "hold_reason": "", "progress_verified": False,
            "owner_binding_verified": True, "recovery_permitted": True,
            "pending_task_ids": ["TASK-1"], "active_task_id": "TASK-1",
            "active_task_cid": prompt_workflow_cid({"task": 1}), "attempt": 1,
            "progress_token": "before", "owner_token": "a" * 64, "grace_until_ms": 0}

        def observe(self):
            return deepcopy(self.value)

        def recover(self, context):
            self.calls += 1
            self.started = context.started_at_ms
            self.now += 1000
            if self.mode == "exception":
                raise RuntimeError("synthetic_private_callback_error")
            if self.mode == "timeout":
                self.now = context.deadline_at_ms + 1
            return RecoveryActionObservation(False,
                ("old_lane_fenced",) if self.mode == "missing_effect" else ("old_lane_fenced", "lane_restarted"),
                {"healthy": False}, partial=self.mode != "known_effects")

        def watchdog(self, **kwargs):
            return CampaignProgressWatchdog(tmp_path / "watchdog", repository_root=tmp_path,
                scope=self.scope, observe=self.observe, recover=self.recover,
                clock_ms=lambda: self.now, **kwargs)

        def make_uncertain(self, **kwargs):
            self.kwargs = kwargs
            assert self.watchdog(**kwargs).run_once()["status"] == "confirming"
            self.now += 31000
            self.watchdog(**kwargs).run_once()
            ledger = json.loads(self.watchdog(**kwargs).state_path.read_text())
            self.key, self.entry = next(iter(ledger["incidents"].items()))
            terminal_file, = (tmp_path / "watchdog/programmatic-recovery").glob("*.json")
            self.terminal_file = terminal_file
            self.terminal = json.loads(terminal_file.read_text())["terminal"]
            self.action = {"schema": "programmatic-recovery-action-observation@1",
                "incident_cid": self.terminal["incident_cid"], "operation": "restart_lane",
                "target_id": self.scope["run_cid"], "attempt": 1, "outcome": "failed",
                "observation": {"succeeded": False, "observed_effects": ["old_lane_fenced", "lane_restarted"],
                    "post_action_health": {"healthy": False}, "reason": "application_recovery_observed", "partial": True}}
            self.now += 10000
            self.value.update(status="active", healthy=True, progress_verified=True,
                owner_token="b" * 64, progress_token="actual_owned_later_work", recovery_permitted=False)

        def proof(self, context):
            return {"schema": "campaign-known-restart-effects-proof@1",
                **{key: deepcopy(context[key]) for key in ("scope", "incident_key", "terminal_cid", "recovery_ordinal", "action_receipt_cid")},
                "restart_owner_token": "b" * 64, "restart_started_at_ms": self.started + 500,
                "progress_observed_at_ms": self.started + 5000, "progress_task_id": "TASK-1",
                "progress_task_cid": self.value["active_task_cid"], "progress_attempt": 1,
                "current_owner_token": self.value["owner_token"],
                "evidence_cids": [prompt_workflow_cid({"actual_fixture_evidence": key}) for key in range(3)],
                "effects_verified": True, "later_work_verified": True}

        def reconcile(self, **changes):
            args = {"incident_key": self.key, "expected_terminal_cid": self.entry["terminal_cid"],
                "action_observation": self.action, "verify_effects": self.proof, **changes}
            return self.watchdog(**self.kwargs).reconcile_health_pending_restart(**args)

        def bytes(self):
            return {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*.json") if not p.is_symlink()}
    return Recovery()


def test_exact_receipt_and_actual_later_work_resolve_without_replay_or_refund(recovery):
    recovery.make_uncertain()
    terminal = recovery.terminal_file.read_bytes()
    original = deepcopy(recovery.entry)
    result = recovery.reconcile()
    assert result["status"] == "reconciled" and result["recovery_verified"] is False
    assert result["attempt_refunded"] is False and result["provider_calls_by_watchdog"] == 0
    ledger = json.loads(recovery.watchdog().state_path.read_text())
    entry = ledger["incidents"][recovery.key]
    for key in ("attempts", "last_attempt_ms", "terminal_cid", "confirmation_owner"):
        assert entry[key] == original[key]
    assert entry["attempts"] == 1 and entry["phase"] == "finished"
    assert entry["confirmations"] == 0 and entry["last_confirmation_ms"] == 0
    assert recovery.terminal_file.read_bytes() == terminal and recovery.calls == 1
    receipt_file, = (recovery.watchdog().state_dir / "reconciliations").glob("*.json")
    receipt = json.loads(receipt_file.read_text())
    cid = receipt.pop("content_id")
    assert prompt_workflow_cid(receipt) == cid == entry["reconciliation_cid"]
    assert receipt["terminal_cid"] == original["terminal_cid"] and receipt["previous_phase"] == "uncertain"
    before = recovery.bytes()
    assert recovery.reconcile()["reason"] == "known_restart_effects_already_reconciled"
    assert recovery.bytes() == before and recovery.calls == 1


@pytest.mark.parametrize("damage", ["scope", "owner", "stale", "effects", "work", "private_extra", "drift"])
def test_unbound_or_changed_application_proof_changes_no_bytes(recovery, damage):
    recovery.make_uncertain()
    before = recovery.bytes()
    count = 0
    def verify(context):
        nonlocal count
        count += 1
        proof = recovery.proof(context)
        if damage == "scope":
            proof["scope"]["run_cid"] = prompt_workflow_cid({"foreign": True})
        elif damage == "owner":
            proof["current_owner_token"] = "foreign"
        elif damage == "stale":
            proof["progress_observed_at_ms"] = recovery.started - 1
        elif damage == "effects":
            proof["effects_verified"] = False
        elif damage == "work":
            proof["later_work_verified"] = False
        elif damage == "private_extra":
            proof["raw_prompt"] = "synthetic_private_message"
        elif damage == "drift" and count > 1:
            proof["evidence_cids"][0] = prompt_workflow_cid({"drift": True})
        return proof
    result = recovery.reconcile(verify_effects=verify)
    assert result["status"] == "refused" and "synthetic_private" not in json.dumps(result)
    assert recovery.bytes() == before and recovery.calls == 1


@pytest.mark.parametrize("condition", ["hold", "unbound", "no_progress", "budget", "wrong_terminal", "payload", "missing_effect", "exception", "timeout"])
def test_unknown_effects_holds_or_exhaustion_never_reconcile(recovery, condition):
    if condition in {"missing_effect", "exception", "timeout"}:
        recovery.mode = condition
    policy = CampaignProgressWatchdogPolicy(max_recovery_attempts=1) if condition == "budget" else CampaignProgressWatchdogPolicy()
    recovery.make_uncertain(policy=policy)
    if condition == "hold":
        recovery.value.update(manual_hold=True, hold_reason="manual_hold")
    elif condition == "unbound":
        recovery.value["owner_binding_verified"] = False
    elif condition == "no_progress":
        recovery.value["progress_verified"] = False
    changes = {}
    if condition == "wrong_terminal":
        changes["expected_terminal_cid"] = prompt_workflow_cid({"different": True})
    elif condition == "payload":
        changes["action_observation"] = deepcopy(recovery.action)
        changes["action_observation"]["observation"]["partial"] = False
    before = recovery.bytes()
    assert recovery.reconcile(**changes)["status"] == "refused"
    assert recovery.bytes() == before and recovery.calls == 1


def test_real_second_ordinal_and_finite_exhaustion_after_reconciliation(recovery):
    recovery.make_uncertain()
    assert recovery.reconcile()["status"] == "reconciled"
    recovery.mode = "known_effects"
    recovery.value.update(status="stalled", healthy=False, progress_verified=False,
        progress_token="before", recovery_permitted=True)
    recovery.now += 61000
    assert recovery.watchdog().run_once()["status"] == "confirming"
    recovery.now += 31000
    assert recovery.watchdog().run_once()["status"] == "exhausted"
    assert recovery.calls == 2
    entry = json.loads(recovery.watchdog().state_path.read_text())["incidents"][recovery.key]
    assert entry["attempts"] == 2 and entry["phase"] == "finished"
    assert recovery.watchdog().run_once()["reason"] == "incident_recovery_budget_exhausted"
    assert recovery.calls == 2


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "fifo"])
def test_terminal_aliases_and_nonregular_files_refuse_without_effect(recovery, alias):
    recovery.make_uncertain()
    target = recovery.terminal_file.with_suffix(".preserved")
    recovery.terminal_file.rename(target)
    original = target.read_bytes()
    if alias == "symlink":
        recovery.terminal_file.symlink_to(target)
    elif alias == "hardlink":
        os.link(target, recovery.terminal_file)
    else:
        os.mkfifo(recovery.terminal_file, 0o600)
    ledger = recovery.watchdog().state_path.read_bytes()
    assert recovery.reconcile()["status"] == "refused"
    assert recovery.watchdog().state_path.read_bytes() == ledger
    assert target.read_bytes() == original and recovery.calls == 1


def test_reconciliation_respects_the_existing_native_timer_lock(recovery):
    recovery.make_uncertain()
    before = recovery.bytes()
    with (recovery.watchdog().state_dir / ".campaign-progress-watchdog.lock").open("r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert recovery.reconcile()["reason"] == "watchdog_owner_busy"
    assert recovery.bytes() == before and recovery.calls == 1
