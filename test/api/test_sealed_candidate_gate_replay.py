"""A sealed finished candidate-gate rejection can leave an unknown callback.

The replay uses the same candidate classifier and retry ceiling as the live
provider. It does not accept a generic unknown callback, a declared validation
failure, or another attempt after the budget is exhausted.
"""

import hashlib
import json
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.runtime.event_log import _event_identity
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
    DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
    _canonical_json,
    _projection_immutable_digest,
    _sha256_bytes,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
    _BOUND_STATE_TARGETS,
    bind_task_state_control_plane,
    save_portal_task_state,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)

_CID = "baguqeeravrergj6lpwjy76bjc2udorzqglwaytqz5p6q24i5gohjaj7n7r2a"


def _finished(**updates):
    event = {
        "type": "implementation_finished",
        "sequence": 1,
        "previous_event_id": "",
        "attempt": 1,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "returncode": 78,
        "task_id": "WWL-084",
        "canonical_task_cid": _CID,
        "lifecycle_finalize": {"finalized": True},
        "validation_result": {
            "attempted": False,
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
        },
        "merge_result": {"merged": False},
        "board_completion": {"complete": False},
        "commit_result": {},
    }
    event.update(updates)
    event["event_id"] = _event_identity(event)
    return event


def _plant(tmp_path, events, *, state_updates=None):
    attempt_id = "attempt:sealed-gate"
    attempt = SimpleNamespace(
        attempt_id=attempt_id,
        claim_id="claim:sealed-gate",
        task_cid="sha256:" + "ab" * 32,
        task_alias="WWL-084",
        attempt_number=3,
    )
    root = tmp_path / "attempts"
    root.mkdir(mode=0o700)
    attempt_dir = root / hashlib.sha256(attempt_id.encode()).hexdigest()[:24]
    attempt_dir.mkdir(mode=0o700)
    projection = "## WWL-084\n- Status: in_progress\n"
    (attempt_dir / "task-projection.runtime.todo.md").write_text(projection)
    binding = {
        "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
        "interface": DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "goal_cid": "goal:sealed",
        "plan_cid": "plan:sealed",
        "task_revision": 7,
        "fencing_token": 3,
        "fence_epoch": 3,
        "lease_id": "lease:sealed",
        "task_body_digest": _sha256_bytes(b"task"),
        "projection_seed_digest": _sha256_bytes(b"seed"),
        "projection_immutable_digest": _projection_immutable_digest(projection),
        "authoritative_task_store": "duckdb",
        "projection_authority": False,
    }
    binding["binding_id"] = _sha256_bytes(_canonical_json(binding))
    (attempt_dir / "database-attempt-binding.json").write_text(json.dumps(binding))
    (attempt_dir / "portal-events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    state_path = attempt_dir / "portal-task-state.json"
    database = attempt_dir / "portal-state.duckdb"
    bind_task_state_control_plane(state_path, str(database))
    state = {
        "implementation_in_progress": False,
        "active_provider_runner": {},
        "active_attempt": 0,
        "active_task_id": "",
        "active_task_cid": "",
        "active_task_key": "",
        "active_branch": "",
        "active_worktree_path": "",
        "active_log_path": "",
        "last_implementation_task_id": "WWL-084",
        "last_implementation_task_cid": _CID,
        "last_implementation_returncode": 78,
        "completion_authority": False,
    }
    state.update(state_updates or {})
    save_portal_task_state("portal-task-state", state, state_path=state_path)
    bridge = DatabasePortalExecutionBridge(
        task_source=None,
        attempt_root=root,
        portal_factory=lambda paths, alias: None,
        max_task_attempts=3,
    )
    return bridge, attempt


def test_sealed_proposal_gate_rejection_is_replayable(tmp_path):
    bridge, attempt = _plant(tmp_path, [_finished()])
    try:
        assert bridge.sealed_candidate_gate_rejection(attempt) == {
            "reason": "proposal_gate_failed",
            "portal_attempt": 1,
            "returncode": 78,
        }
    finally:
        _BOUND_STATE_TARGETS.clear()


def test_declared_validation_and_completion_are_not_candidate_replays(tmp_path):
    declared = _finished()
    declared["validation_result"] = {
        **declared["validation_result"],
        "reason": "declared_validation_failed",
        "attempted": True,
    }
    declared["event_id"] = _event_identity(declared)
    bridge, attempt = _plant(tmp_path, [declared])
    try:
        assert bridge.sealed_candidate_gate_rejection(attempt) is None
    finally:
        _BOUND_STATE_TARGETS.clear()
    completed = _finished()
    follow = {
        "type": "task_completed",
        "sequence": 2,
        "previous_event_id": completed["event_id"],
        "task_id": "WWL-084",
    }
    follow["event_id"] = _event_identity(follow)
    other = tmp_path / "completed"
    other.mkdir()
    bridge, attempt = _plant(other, [completed, follow])
    try:
        assert bridge.sealed_candidate_gate_rejection(attempt) is None
    finally:
        _BOUND_STATE_TARGETS.clear()


def test_busy_portal_state_is_not_a_finished_rejection(tmp_path):
    bridge, attempt = _plant(
        tmp_path, [_finished()], state_updates={"implementation_in_progress": True},
    )
    try:
        assert bridge.sealed_candidate_gate_rejection(attempt) is None
    finally:
        _BOUND_STATE_TARGETS.clear()


def _daemon_with_unknown_callback(tmp_path, monkeypatch, *, max_task_attempts, proof):
    def boom(self, attempt):
        raise DatabasePortalBridgeError("portal_provider_failed")

    monkeypatch.setattr(DatabasePortalExecutionBridge, "run_provider", boom)
    bridge = DatabasePortalExecutionBridge(
        task_source=None,
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: None,
        max_task_attempts=max_task_attempts,
    )
    daemon = _open_daemon(
        tmp_path,
        session="session:lane",
        provider_fn=bridge.run_provider,
        max_task_attempts=max_task_attempts,
    )
    bridge.task_source = daemon.task_source
    monkeypatch.setattr(daemon, "_custody_gap_census_clear", lambda: True)
    monkeypatch.setattr(bridge, "sealed_candidate_gate_rejection", lambda attempt: proof)
    daemon.materialize_population(_population(1))
    first = daemon.run_once()
    assert first["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
    attempt = daemon.get_attempt(first["implementation_result"]["attempt_id"])
    assert attempt.status == "running"
    return daemon, attempt


def test_exhausted_sealed_rejection_blocks_without_another_attempt(tmp_path, monkeypatch):
    daemon, attempt = _daemon_with_unknown_callback(
        tmp_path,
        monkeypatch,
        max_task_attempts=1,
        proof={"reason": "proposal_gate_failed", "portal_attempt": 1, "returncode": 78},
    )
    try:
        settled = daemon.run_once()
        result = settled["implementation_result"]
        assert result["portal_terminal_failure"] is True
        assert result["reason"] == "proposal_gate_failed"
        assert result["status"] == "blocked"
        failed = daemon.get_attempt(attempt.attempt_id)
        assert failed.status == "failed"
        task = daemon.task_source.get(failed.task_cid)
        assert str(task.status) == "blocked"
        claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert str(getattr(claim.state, "value", claim.state)) == "released"
        again = daemon.run_once()
        assert again.get("implementation_result") is None or (
            again["implementation_result"].get("reason") != "provider_callback_outcome_unknown"
        )
        assert daemon.list_running_attempts() == []
        assert daemon.task_source.get(failed.task_cid).status == "blocked"
    finally:
        daemon.close()


def test_sealed_rejection_with_budget_retries_instead_of_blocking(tmp_path, monkeypatch):
    daemon, attempt = _daemon_with_unknown_callback(
        tmp_path,
        monkeypatch,
        max_task_attempts=3,
        proof={"reason": "proposal_gate_failed", "portal_attempt": 1, "returncode": 78},
    )
    try:
        settled = daemon.run_once()
        result = settled["implementation_result"]
        assert result["portal_retryable_failure"] is True
        assert result["reason"] == "proposal_gate_failed"
        assert result["status"] == "retrying"
        assert result["retry_budget_exhausted"] is False
        failed = daemon.get_attempt(attempt.attempt_id)
        assert failed.status == "failed"
        assert daemon.task_source.get(failed.task_cid).status == "retrying"
        assert daemon.list_running_attempts() == []
    finally:
        daemon.close()


def test_unknown_callback_without_sealed_rejection_stays_unknown(tmp_path, monkeypatch):
    daemon, attempt = _daemon_with_unknown_callback(
        tmp_path, monkeypatch, max_task_attempts=3, proof=None,
    )
    try:
        waiting = daemon.run_once()
        assert waiting["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
    finally:
        daemon.close()


def test_live_provider_child_blocks_sealed_replay(tmp_path, monkeypatch):
    daemon, attempt = _daemon_with_unknown_callback(
        tmp_path,
        monkeypatch,
        max_task_attempts=1,
        proof={"reason": "proposal_gate_failed", "portal_attempt": 1, "returncode": 78},
    )
    monkeypatch.setattr(daemon, "_custody_gap_census_clear", lambda: False)
    try:
        waiting = daemon.run_once()
        assert waiting["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        assert daemon.task_source.get(attempt.task_cid).status == "in_progress"
    finally:
        daemon.close()
