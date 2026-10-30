"""Single-campaign recovery exercises the real finite Native controller."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.campaign_watchdog import (
    CampaignProgressWatchdog, CampaignProgressWatchdogPolicy, OBSERVATION_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    ProgrammaticRecoveryContext, RecoveryActionObservation,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import prompt_workflow_cid


SCOPE = {key: prompt_workflow_cid({"fixture": key}) for key in ("repository_root_cid", "policy_root", "run_cid")}


def observation(**changes):
    value = {"schema": OBSERVATION_SCHEMA, "scope": dict(SCOPE), "status": "stalled", "healthy": False,
        "completion_verified": False, "manual_hold": False, "hold_reason": "", "progress_verified": False,
        "owner_binding_verified": True, "recovery_permitted": True, "pending_task_ids": ["TASK-1"],
        "active_task_id": "TASK-1", "active_task_cid": "synthetic-task-cid", "attempt": 1,
        "progress_token": "native-progress-1", "owner_token": "owned-birth-1", "grace_until_ms": 0}
    value.update(changes)
    return value


@pytest.fixture
def campaign(tmp_path):
    class Campaign:
        value = observation()
        now = 1_000
        calls = []

        def observe(self):
            return deepcopy(self.value)

        def recover(self, context):
            assert isinstance(context, ProgrammaticRecoveryContext)
            self.calls.append(context)
            return RecoveryActionObservation(False, reason="synthetic_unrestored")

        def watchdog(self, **kwargs):
            return CampaignProgressWatchdog(tmp_path / "watchdog", repository_root=tmp_path, scope=SCOPE, observe=self.observe,
                recover=self.recover, clock_ms=lambda: self.now, **kwargs)
    return Campaign()


def test_verified_complete_creates_no_incident_or_action(campaign, tmp_path):
    campaign.value = observation(status="complete", healthy=True, completion_verified=True,
        pending_task_ids=[], active_task_id="", active_task_cid="", attempt=0)
    result = campaign.watchdog().run_once()
    assert result["status"] == "complete" and result["completion_authority"] is False
    assert result["provider_calls_by_watchdog"] == 0 and campaign.calls == []
    assert not (tmp_path / "watchdog").exists()


@pytest.mark.parametrize("hold", ["manual_hold", "awaiting_user", "cancelled", "capacity_quarantine",
    "protected_incident", "attempt_budget_exhausted", "dependency_unavailable"])
def test_manual_and_policy_waits_never_become_recovery_incidents(campaign, tmp_path, hold):
    campaign.value.update(hold_reason=hold)
    assert campaign.watchdog().run_once()["status"] == "held"
    assert campaign.calls == [] and not (tmp_path / "watchdog").exists()


def test_semantic_progress_is_healthy_and_heartbeat_only_is_not(campaign):
    campaign.value.update(status="active", healthy=True, progress_verified=True)
    assert campaign.watchdog().run_once()["status"] == "healthy"
    campaign.value["progress_verified"] = False
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    assert campaign.calls == []


def test_actual_native_controller_recovers_after_two_confirmations_across_restart(campaign, tmp_path):
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    def recover(context):
        campaign.calls.append(context)
        campaign.value.update(status="active", healthy=True, progress_verified=True,
            progress_token="native-progress-2", owner_token="owned-birth-2")
        return RecoveryActionObservation(True, ("old_lane_fenced", "lane_restarted"), {"healthy": True})
    campaign.recover = recover
    result = campaign.watchdog().run_once()
    assert result["status"] == "recovered" and result["recovery_attempts"] == 1
    assert len(campaign.calls) == 1 and isinstance(campaign.calls[0], ProgrammaticRecoveryContext)
    assert campaign.calls[0].incident.repository_root == str(tmp_path)
    assert list((tmp_path / "watchdog/programmatic-recovery").glob("*.json"))
    assert result["completion_authority"] is False and result["attempt_refunded"] is False


def test_cooldown_and_finite_budget_survive_owner_and_process_changes(campaign):
    watchdog = campaign.watchdog()
    assert watchdog.run_once()["status"] == "confirming"
    campaign.now += 30_001
    assert watchdog.run_once()["status"] == "cooldown"
    campaign.value["owner_token"] = "owned-birth-replacement"
    assert campaign.watchdog().run_once()["status"] == "cooldown"
    assert len(campaign.calls) == 1
    campaign.now += 60_001
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    assert campaign.watchdog().run_once()["status"] == "exhausted"
    campaign.now += 60_001
    assert campaign.watchdog().run_once()["status"] == "exhausted"
    assert len(campaign.calls) == 2
    campaign.value["progress_token"] = "real-work-advanced"
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001


def test_observation_drift_before_effect_refuses_without_callback(campaign):
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    count = 0
    def observe():
        nonlocal count
        count += 1
        current = campaign.observe()
        if count > 1:
            current["owner_token"] = "different-current-owner"
        return current
    watchdog = CampaignProgressWatchdog(campaign.watchdog().state_dir, repository_root=campaign.watchdog().repository_root, scope=SCOPE,
        observe=observe, recover=campaign.recover, clock_ms=lambda: campaign.now)
    result = watchdog.run_once()
    assert result["status"] == "refused" and result["reason"] == "observation_changed_before_effect"
    assert campaign.calls == []


def test_missing_typed_effects_or_fake_health_cannot_recover(campaign):
    campaign.recover = lambda context: RecoveryActionObservation(True, (), {"healthy": True})
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    result = campaign.watchdog().run_once()
    assert result["status"] == "error" and result["reason"] == "application_recovery_effects_unproven"


def test_startup_grace_is_fixed_and_cannot_slide_on_observation_polls(campaign):
    campaign.value.update(status="starting", grace_until_ms=9_999_999)
    assert campaign.watchdog().run_once()["status"] == "recovering"
    campaign.now += 180_001
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    assert campaign.calls == []


@pytest.mark.parametrize("field,value", [("pending_task_ids", ["TASK-1"]), ("active_task_id", "TASK-1"),
    ("attempt", 1), ("progress_verified", True), ("healthy", False)])
def test_completion_conflicts_are_errors_with_no_effect(campaign, tmp_path, field, value):
    campaign.value = observation(status="complete", healthy=True, completion_verified=True,
        pending_task_ids=[], active_task_id="", active_task_cid="", attempt=0)
    campaign.value[field] = value
    assert campaign.watchdog().run_once()["status"] == "error"
    assert campaign.calls == [] and not (tmp_path / "watchdog").exists()


@pytest.mark.parametrize("damage", ["missing", "wrong_scope", "wrong_type", "private_extra"])
def test_malformed_or_unbound_observation_never_counts_as_health(campaign, damage):
    if damage == "missing":
        del campaign.value["status"]
    elif damage == "wrong_scope":
        campaign.value["scope"]["run_cid"] = "different-campaign"
    elif damage == "wrong_type":
        campaign.value["healthy"] = "true"
    else:
        campaign.value["raw_log"] = "synthetic_forbidden_field"
    result = campaign.watchdog().run_once()
    assert result["status"] == "error" and campaign.calls == []
    assert "synthetic_forbidden_field" not in json.dumps(result)


def test_interrupted_effect_is_not_replayed_after_process_restart(campaign):
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    path = campaign.watchdog().state_path
    value = json.loads(path.read_bytes())
    entry = next(iter(value["incidents"].values()))
    entry.update(phase="in_flight", attempts=1, last_attempt_ms=campaign.now)
    path.write_text(json.dumps(value))
    campaign.value["owner_token"] = "next-owner"
    result = campaign.watchdog().run_once()
    assert result["status"] == "refused" and result["reason"] == "recovery_effects_uncertain"
    assert campaign.calls == []


def test_corrupt_ledger_fails_closed_without_action(campaign):
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    campaign.watchdog().state_path.write_text("{invalid")
    assert campaign.watchdog().run_once()["status"] == "error"
    assert campaign.calls == []


def test_late_success_does_not_hide_native_action_deadline(campaign):
    def recover(context):
        campaign.calls.append(context)
        campaign.now = context.deadline_at_ms + 1
        campaign.value.update(status="active", healthy=True, progress_verified=True)
        return RecoveryActionObservation(True, ("old_lane_fenced", "lane_restarted"), {"healthy": True})
    campaign.recover = recover
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    result = campaign.watchdog().run_once()
    assert result["status"] == "error" and result["reason"] == "recovery_deadline_exhausted"
    campaign.value.update(status="stalled", healthy=False, progress_verified=False, owner_token="new-owner")
    campaign.now += 60_001
    assert campaign.watchdog().run_once()["reason"] == "recovery_effects_uncertain"
    assert len(campaign.calls) == 1


def test_recovery_deadline_accepts_validation_plus_overhead_without_changing_provider_policy():
    assert CampaignProgressWatchdogPolicy(deadline_ms=1_020_000).deadline_ms == 1_020_000
    with pytest.raises(ValueError):
        CampaignProgressWatchdogPolicy(deadline_ms=1_800_001)


def test_recovery_callback_exception_preserves_uncertain_effects(campaign):
    def recover(context):
        campaign.calls.append(context)
        raise RuntimeError("synthetic_private_message_not_for_output")
    campaign.recover = recover
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    result = campaign.watchdog().run_once()
    assert result["status"] == "error" and result["reason"] == "application_recovery_effects_uncertain"
    assert "synthetic_private_message" not in json.dumps(result)
    campaign.now += 60_001
    assert campaign.watchdog().run_once()["reason"] == "recovery_effects_uncertain"
    assert len(campaign.calls) == 1


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "fifo"])
def test_state_file_aliases_and_nonregular_files_are_refused(campaign, tmp_path, alias):
    import os
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    path = campaign.watchdog().state_path
    preserved = path.with_name("preserved.json")
    path.rename(preserved)
    if alias == "symlink":
        path.symlink_to(preserved)
    elif alias == "hardlink":
        os.link(preserved, path)
    else:
        os.mkfifo(path, 0o600)
    before = preserved.read_bytes()
    assert campaign.watchdog().run_once()["status"] == "error"
    assert preserved.read_bytes() == before and campaign.calls == []


def test_concurrent_timer_owner_does_not_enter_recovery(campaign):
    import fcntl
    watchdog = campaign.watchdog()
    assert watchdog.run_once()["status"] == "confirming"
    with (watchdog.state_dir / ".campaign-progress-watchdog.lock").open("r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = campaign.watchdog().run_once()
        assert result["status"] == "refused" and result["reason"] == "watchdog_owner_busy"
    assert campaign.calls == []


def test_confirmation_requires_distinct_timed_observations(campaign):
    watchdog = campaign.watchdog()
    assert watchdog.run_once()["status"] == "confirming"
    assert watchdog.run_once()["status"] == "confirming"
    campaign.now += 29_999
    assert watchdog.run_once()["status"] == "confirming"
    assert campaign.calls == []
    campaign.now += 1
    assert watchdog.run_once()["status"] == "cooldown"
    assert len(campaign.calls) == 1


def test_new_owner_must_reconfirm_without_resetting_same_incident_budget(campaign):
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    campaign.value["owner_token"] = "replacement-before-action"
    assert campaign.watchdog().run_once()["status"] == "confirming"
    assert campaign.calls == []
    campaign.now += 30_001
    assert campaign.watchdog().run_once()["status"] == "cooldown"
    campaign.now += 60_001
    campaign.value["owner_token"] = "another-owner"
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    assert campaign.watchdog().run_once()["status"] == "exhausted"
    assert len(campaign.calls) == 2


@pytest.mark.parametrize("uncertainty", ["partial", "missing_effects", "bad_post_observation"])
def test_partial_or_unproven_effects_are_not_automatically_repeated(campaign, uncertainty):
    def recover(context):
        campaign.calls.append(context)
        if uncertainty == "bad_post_observation":
            campaign.value["healthy"] = "invalid"
        return RecoveryActionObservation(True,
            () if uncertainty == "missing_effects" else ("old_lane_fenced", "lane_restarted"),
            {"healthy": True}, partial=uncertainty == "partial")
    campaign.recover = recover
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    assert campaign.watchdog().run_once()["status"] == "error"
    campaign.value["healthy"] = False
    campaign.value["owner_token"] = "next-owner"
    campaign.now += 60_001
    assert campaign.watchdog().run_once()["reason"] == "recovery_effects_uncertain"
    assert len(campaign.calls) == 1


def test_verified_completion_can_win_reconciliation_without_inventing_restart(campaign):
    def recover(context):
        campaign.calls.append(context)
        campaign.value.update(status="complete", healthy=True, completion_verified=True,
            pending_task_ids=[], active_task_id="", active_task_cid="", attempt=0, progress_verified=False)
        return RecoveryActionObservation(True, ("old_lane_fenced",), {"healthy": True})
    campaign.recover = recover
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    result = campaign.watchdog().run_once()
    assert result["status"] == "complete" and result["completion_authority"] is False
    assert result["recovery_effects_uncertain"] is True and len(campaign.calls) == 1
    ledger = json.loads(campaign.watchdog().state_path.read_bytes())
    assert next(iter(ledger["incidents"].values()))["phase"] == "uncertain"
    campaign.value = observation(owner_token="next-owner")
    campaign.now += 60_001
    assert campaign.watchdog().run_once()["reason"] == "recovery_effects_uncertain"
    assert len(campaign.calls) == 1


@pytest.mark.parametrize("failure", ["timeout", "exception"])
def test_completion_does_not_hide_timed_out_or_raised_recovery(campaign, failure):
    def recover(context):
        campaign.value.update(status="complete", healthy=True, completion_verified=True,
            pending_task_ids=[], active_task_id="", active_task_cid="", attempt=0, progress_verified=False)
        if failure == "exception":
            raise RuntimeError("synthetic_uncertain_effect")
        campaign.now = context.deadline_at_ms + 1
        return RecoveryActionObservation(True, ("old_lane_fenced",), {"healthy": True})
    campaign.recover = recover
    assert campaign.watchdog().run_once()["status"] == "confirming"
    campaign.now += 30_001
    result = campaign.watchdog().run_once()
    assert result["status"] == "error"
    assert result["reason"] == ("recovery_deadline_exhausted" if failure == "timeout"
        else "application_recovery_effects_uncertain")
