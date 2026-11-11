"""Focused SOURCE regressions for finite-policy waits and main-PID heartbeat.

Not executed while preparing this handoff. All enrollment metadata is synthetic
unit data under tmp_path; fake socket transport never notifies a real service.
"""
from pathlib import Path
from unittest.mock import patch
import copy
import hashlib
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as authorization
from ipfs_accelerate_py.agent_supervisor.rescue.systemd_heartbeat import SystemdHeartbeat


def _authorization_fixture(tmp_path):
    state = tmp_path / "state"
    prior_path = state / "repairs" / "engineering" / "job.json"
    prior_path.parent.mkdir(mode=0o700, parents=True)
    prior = {"board_id": "engineering", "status": "verified_healthy",
             "escalation_reason": "repair_budget_exhausted", "attempts": 3,
             "latest_incident": {"signature": "synthetic-old-incident"}}
    raw = (json.dumps(prior, sort_keys=True) + "\n").encode()
    prior_path.write_bytes(raw)
    prior_path.chmod(0o400)
    prior_ref = {"path": str(prior_path), "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}
    policy = {"state_root": str(state), "namespace": "legal_ir_unattended_repair"}
    epoch = {"schema": authorization.SCHEMA, "epoch_id": "unit-epoch",
             "objective_id": "Synthetic finite-policy boundary regression",
             "board_id": "engineering", "board_namespace": policy["namespace"],
             "candidate_policy_sha256": authorization.digest(policy),
             "max_attempts": 1, "prior_job": prior_ref,
             "prior_incident_signature_sha256": hashlib.sha256(
                 prior["latest_incident"]["signature"].encode()).hexdigest(),
             "not_before_observed_at": 10.0, "expires_at": 20.0}
    board = {"id": "engineering", "max_repair_attempts": 3,
             "repair_authorization_epoch": epoch}
    config = {"state_dir": str(state), "boards": [board],
              "repair_worker": {"candidate_policy": policy}}
    return config, board, prior_path


def test_valid_finite_epoch_boundary_has_typed_wait_without_renewal(tmp_path):
    config, board, prior_path = _authorization_fixture(tmp_path)
    before = copy.deepcopy(config)
    prior_bytes = prior_path.read_bytes()
    for current, reason in ((9.0, "repair_authorization_not_yet_valid"),
                            (20.0, "repair_authorization_expired"),
                            (21.0, "repair_authorization_expired")):
        with pytest.raises(authorization.RepairAuthorizationWait,
                           match="repair epoch is not current") as captured:
            authorization.require_current_authorization(config, board, now=current)
        wait = captured.value
        assert isinstance(wait, ValueError)
        assert wait.policy_wait_result() == {
            "status": "policy_wait", "board_id": "engineering", "reason": reason,
            "repair_epoch_id": "unit-epoch", "not_before_observed_at": 10.0,
            "expires_at": 20.0, "coding_attempts_incremented": False}
    for current in (10.0, 19.999):
        epoch = authorization.require_current_authorization(config, board, now=current)
        assert epoch.value == board["repair_authorization_epoch"]
        assert not epoch.directory.exists()
    assert config == before
    assert prior_path.read_bytes() == prior_bytes


def test_invalid_clock_or_epoch_remains_generic_denial(tmp_path):
    config, board, prior_path = _authorization_fixture(tmp_path)
    before = copy.deepcopy(config)
    for current in (float("nan"), float("inf"), -float("inf"), True, "21"):
        with pytest.raises(ValueError) as captured:
            authorization.require_current_authorization(config, board, now=current)
        assert not isinstance(captured.value, authorization.RepairAuthorizationWait)
    invalid = copy.deepcopy(config)
    invalid_board = invalid["boards"][0]
    invalid_board["repair_authorization_epoch"]["max_attempts"] = True
    with pytest.raises(ValueError) as captured:
        authorization.require_current_authorization(invalid, invalid_board, now=21.0)
    assert not isinstance(captured.value, authorization.RepairAuthorizationWait)
    assert config == before
    assert authorization.descriptor(prior_path) == board["repair_authorization_epoch"]["prior_job"]
    assert not (Path(config["state_dir"]) / "repair-epochs").exists()


def _run_one_main_cycle(tmp_path, failure=None):
    state = tmp_path / "worker-state"
    state.mkdir(mode=0o700)
    config = {"state_dir": str(state), "boards": [{"id": "engineering"}],
              "repair_worker": {"job_unit": "unit-test-repair.service"}}
    config_path = tmp_path / "fleet.json"
    config_path.write_text(json.dumps(config))
    original_config = config_path.read_bytes()
    datagrams = []
    ticks = iter(range(0, 100, 2))
    class RecordingSocket:
        def settimeout(self, value):
            assert value == 0.2
        def sendto(self, payload, address):
            assert payload == b"WATCHDOG=1"
            assert address == "/unit-test-notify.sock"
            datagrams.append((payload, address))
            return len(payload)
        def close(self):
            pass
    heartbeat = SystemdHeartbeat(address="/unit-test-notify.sock",
        interval_seconds=1.0, owner_pid=os.getpid(), clock=lambda: next(ticks),
        socket_factory=lambda *_: RecordingSocket())
    def reconcile(*_args, **_kwargs):
        if failure is not None:
            raise failure
        return []
    no_external_effect = AssertionError("one idle/policy/error cycle attempted external work")
    with patch.object(fleet, "load_config", return_value=config), \
         patch.object(fleet.signal, "signal"), \
         patch.object(fleet, "_SERVICE_HEARTBEAT", None), \
         patch.object(fleet.SystemdHeartbeat, "from_environment", return_value=heartbeat), \
         patch.object(fleet, "reconcile_queued_jobs", side_effect=reconcile), \
         patch.object(fleet, "should_reclaim_repair_unit", return_value=False), \
         patch.object(fleet, "next_pending_candidate", return_value=None), \
         patch.object(fleet, "next_job", return_value=None), \
         patch.object(fleet, "queue_status", return_value={"status": "idle", "waiting": [], "held": []}), \
         patch.object(fleet, "command", side_effect=no_external_effect), \
         patch.object(fleet.subprocess, "Popen", side_effect=no_external_effect):
        code = fleet.main(["run", "--config", str(config_path), "--once"])
    assert code == 0
    assert config_path.read_bytes() == original_config
    assert not (state / "repairs").exists()
    return json.loads((state / "repair-worker.json").read_bytes()), datagrams


def test_main_expected_expiry_keeps_policy_wait_and_heartbeat(tmp_path):
    config, board, _ = _authorization_fixture(tmp_path / "authorization")
    with pytest.raises(authorization.RepairAuthorizationWait) as captured:
        authorization.require_current_authorization(config, board, now=20.0)
    status, datagrams = _run_one_main_cycle(tmp_path, captured.value)
    assert status["status"] == "policy_wait"
    assert status["reason"] == "repair_authorization_expired"
    assert status["repair_epoch_id"] == "unit-epoch"
    assert status["expires_at"] == 20.0
    assert status["coding_attempts_incremented"] is False
    assert "error" not in status
    assert len(datagrams) == 2  # Startup and finally after expected policy wait.


def test_main_same_message_generic_error_stays_error_and_keeps_heartbeat(tmp_path):
    status, datagrams = _run_one_main_cycle(
        tmp_path, ValueError("repair epoch is not current"))
    assert status["status"] == "error"
    assert status["error"] == "ValueError: repair epoch is not current"
    assert len(datagrams) == 2  # No string matching or false policy authorization.


def test_main_clean_idle_cycle_keeps_existing_status_and_heartbeat(tmp_path):
    status, datagrams = _run_one_main_cycle(tmp_path)
    assert status["status"] == "idle"
    assert status["waiting"] == status["held"] == []
    assert len(datagrams) == 3  # Startup, completed stage, finally.
