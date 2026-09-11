"""Historical failure reconciliation preserves the later task claim."""

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationStaleFenceError, TASK_CLAIM_FAILURE_REARM_OPERATION,
    TASK_CLAIM_FAILURE_REARMED_EVENT, TASK_CLAIM_FAILURE_SETTLED_EVENT,
)
from test.api.test_agent_supervisor_database_coordination import (
    _open, _task_claim_failure_receipt,
)


from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationAuthorityError
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def setup_failure(tmp_path, *, rearm=True, successor=True):
    coordinator, clock = _open(tmp_path)
    coordinator.register_task(task_cid="task:history", task_id="HISTORY")
    claim = coordinator.claim_task(task_cid="task:history", owner_session_id="session:old")
    receipt = _task_claim_failure_receipt(claim, control_expected_revision=2)
    coordinator.fail_task_claim(claim, failure_receipt=receipt)
    if rearm:
        coordinator.rearm_failed_task(failure_receipt=receipt, control_task_observation={
            "task_cid": claim.task_cid, "status": "retrying", "revision": 4,
            "body": {"completion_receipt": {
                "operation": TASK_CLAIM_FAILURE_REARM_OPERATION,
                "settlement_id": receipt["settlement_id"],
            }},
        })
    next_claim = coordinator.claim_task(task_cid=claim.task_cid, owner_session_id="session:new") if rearm and successor else None
    return coordinator, claim, receipt, next_claim


def snapshot(coordinator):
    connection = coordinator._require()
    return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()]
            for table in ("task_claims", "task_attempts", "fenced_leases", "token_history", "task_completions", "lease_events", "coordination_tasks")}


@pytest.mark.parametrize("successor", [False, True])
def test_native_historical_observer_is_read_only_before_and_after_successor(tmp_path, successor):
    coordinator, old, receipt, newer = setup_failure(tmp_path, successor=successor)
    try:
        before = snapshot(coordinator)
        first = coordinator.observe_rearmed_failed_task_claim(old, failure_receipt=receipt)
        second = coordinator.observe_rearmed_failed_task_claim(old, failure_receipt=receipt)
        assert first == second
        assert first["failure_settlement_id"] == receipt["settlement_id"]
        assert first["control_revision"] == 4
        assert snapshot(coordinator) == before
        if newer is not None:
            with pytest.raises(DatabaseCoordinationStaleFenceError):
                coordinator.fail_task_claim(old, failure_receipt=receipt)
            assert coordinator.get_task_claim(newer.claim_id) == newer
            assert snapshot(coordinator) == before
    finally:
        coordinator.close()


def test_absent_rearm_keeps_ordinary_failure_settlement(tmp_path):
    coordinator, old, receipt, _ = setup_failure(tmp_path, rearm=False)
    try:
        before = snapshot(coordinator)
        assert coordinator.observe_rearmed_failed_task_claim(old, failure_receipt=receipt) is None
        assert snapshot(coordinator) == before
        assert coordinator.fail_task_claim(old, failure_receipt=receipt).state.value == "released"
    finally:
        coordinator.close()


@pytest.mark.parametrize("mutation", [
    "failure_missing", "failure_duplicate", "failure_changed", "rearm_duplicate",
    "rearm_changed", "rearm_extra_field", "claim_changed", "lease_changed",
    "attempt_changed", "old_token_missing", "old_epoch_changed", "foreign_receipt",
])
def test_historical_observer_denies_incomplete_or_changed_native_history(tmp_path, mutation):
    coordinator, old, receipt, _ = setup_failure(tmp_path)
    try:
        c = coordinator._require()
        event_type = TASK_CLAIM_FAILURE_SETTLED_EVENT if mutation.startswith("failure") else TASK_CLAIM_FAILURE_REARMED_EVENT
        if mutation.endswith("duplicate"):
            c.execute("INSERT INTO lease_events SELECT event_id || '-duplicate', lease_id, scope_key, event_type, fencing_token, fence_epoch, observed_at_ms, body_json FROM lease_events WHERE event_type = ?", [event_type])
        elif mutation == "failure_missing":
            c.execute("DELETE FROM lease_events WHERE event_type = ?", [event_type])
        elif mutation in {"failure_changed", "rearm_changed", "rearm_extra_field"}:
            row = c.execute("SELECT event_id, body_json FROM lease_events WHERE event_type = ?", [event_type]).fetchone()
            body = json.loads(row[1])
            if mutation == "failure_changed": body["settlement_id"] = "foreign"
            elif mutation == "rearm_changed": body["control_revision"] += 1
            else: body["retry_authority"] = True
            c.execute("UPDATE lease_events SET body_json = ? WHERE event_id = ?", [json.dumps(body), row[0]])
        elif mutation == "claim_changed":
            c.execute("UPDATE task_claims SET state = 'accepted' WHERE claim_id = ?", [old.claim_id])
        elif mutation == "lease_changed":
            c.execute("UPDATE fenced_leases SET attempt_number = 99 WHERE lease_id = ?", [old.lease_id])
        elif mutation == "attempt_changed":
            c.execute("UPDATE task_attempts SET status = 'succeeded' WHERE attempt_id = ?", [old.attempt_id])
        elif mutation == "old_token_missing":
            c.execute("DELETE FROM token_history WHERE fencing_token = ?", [old.fencing_token])
        elif mutation == "old_epoch_changed":
            c.execute("UPDATE token_history SET fence_epoch = 99 WHERE fencing_token = ?", [old.fencing_token])
        else:
            receipt = {**receipt, "attempt_id": "attempt:foreign"}
        before = snapshot(coordinator)
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.observe_rearmed_failed_task_claim(old, failure_receipt=receipt)
        assert snapshot(coordinator) == before
    finally:
        coordinator.close()


def setup_daemon_history(tmp_path, monkeypatch):
    calls = []

    def fail_provider(attempt):
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("terminal historical test failure")

    daemon = _open_daemon(tmp_path, session="session:history", provider_fn=fail_provider)
    daemon.materialize_population(_population(1))
    record_event = daemon._record_event

    def omit_local_marker(event_type, **kwargs):
        if event_type != "terminal_portal_failure_settled":
            return record_event(event_type, **kwargs)

    monkeypatch.setattr(daemon, "_record_event", omit_local_marker)
    initial = daemon.run_once()
    monkeypatch.setattr(daemon, "_record_event", record_event)
    old = daemon.get_attempt(initial["attempt_id"])
    assert old.status == "failed"
    blocked = daemon.task_source.get(old.task_cid)
    receipt = daemon._portal_failure_phase_receipt(old)
    daemon.task_source.compare_and_set_status(
        old.task_cid, expected_revision=blocked.revision, status="retrying",
        receipt={"operation": TASK_CLAIM_FAILURE_REARM_OPERATION,
                 "settlement_id": receipt["settlement_id"]},
    )
    newer = daemon.claim_next()
    assert newer is not None and newer.attempt_number == old.attempt_number + 1
    return daemon, old, newer, receipt, calls


def execution_snapshot(daemon):
    c = daemon._require_connection()
    return {table: [tuple(row) for row in c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()]
            for table in ("database_task_attempts", "attempt_phases",
                          "provider_invocations", "effect_claims")}


def test_daemon_replays_only_missing_local_marker_and_preserves_successor(tmp_path, monkeypatch):
    daemon, old, newer, receipt, calls = setup_daemon_history(tmp_path, monkeypatch)
    try:
        coordination_before = snapshot(daemon.coordinator)
        execution_before = execution_snapshot(daemon)
        canonical_before = daemon.task_source.get(old.task_cid)
        old_history_before = daemon.phase_history(old.attempt_id)
        def forbidden(*args, **kwargs):
            pytest.fail("historical replay invoked a mutation or callback")
        monkeypatch.setattr(daemon, "_fail_coordination_claim", forbidden)
        monkeypatch.setattr(daemon, "_settle_portal_failure_control_task", forbidden)
        monkeypatch.setattr(daemon, "_provider_fn", forbidden)
        first = daemon.reconcile_terminal_portal_failures()
        assert len(first) == 1
        assert first[0]["attempt_id"] == old.attempt_id
        assert first[0]["control_superseded"] is True
        assert first[0]["historical_rearm_replay"]["control_revision"] == receipt["control_expected_revision"] + 2
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.get_attempt(newer.attempt_id) == newer
        assert daemon.task_source.get(old.task_cid) == canonical_before
        assert daemon.phase_history(old.attempt_id) == old_history_before
        assert execution_snapshot(daemon) == execution_before
        assert snapshot(daemon.coordinator) == coordination_before
        assert calls == [old.attempt_id]
        rows = daemon._require_connection().execute(
            "SELECT body_json FROM daemon_execution_events WHERE event_type = 'terminal_portal_failure_settled'"
        ).fetchall()
        assert len(rows) == 1 and json.loads(rows[0][0]) == first[0]
    finally:
        daemon.close()


@pytest.mark.parametrize("observation", ["missing", "foreign", "behind", "boolean", "status_missing"])
def test_daemon_denies_historical_marker_with_unqualified_control_observation(tmp_path, monkeypatch, observation):
    daemon, old, newer, receipt, calls = setup_daemon_history(tmp_path, monkeypatch)
    try:
        current = daemon.task_source.get(old.task_cid)
        replacement = {
            "missing": None,
            "foreign": replace(current, task_cid="task:foreign"),
            "behind": replace(current, revision=receipt["control_expected_revision"] + 1),
            "boolean": replace(current, revision=True),
            "status_missing": replace(current, status=""),
        }[observation]
        before = snapshot(daemon.coordinator), execution_snapshot(daemon)
        monkeypatch.setattr(daemon.task_source, "get", lambda _cid: replacement)
        with pytest.raises(DatabaseImplementationAuthorityError):
            daemon.reconcile_terminal_portal_failures()
        assert not daemon._terminal_portal_failure_event_exists(old.attempt_id, receipt["settlement_id"])
        assert (snapshot(daemon.coordinator), execution_snapshot(daemon)) == before
        assert calls == [old.attempt_id]
    finally:
        daemon.close()


def test_daemon_historical_local_marker_response_loss_is_idempotent(tmp_path, monkeypatch):
    daemon, old, newer, receipt, calls = setup_daemon_history(tmp_path, monkeypatch)
    try:
        record = daemon._record_event
        def record_then_lose(*args, **kwargs):
            record(*args, **kwargs)
            raise RuntimeError("local marker response lost")
        monkeypatch.setattr(daemon, "_record_event", record_then_lose)
        with pytest.raises(RuntimeError, match="response lost"):
            daemon.reconcile_terminal_portal_failures()
        monkeypatch.setattr(daemon, "_record_event", record)
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.get_attempt(newer.attempt_id) == newer
        assert calls == [old.attempt_id]
    finally:
        daemon.close()


@pytest.mark.parametrize("rearm", [False, True])
@pytest.mark.parametrize("fault", ["begin", "commit", "commit_response"])
def test_observer_transaction_failure_never_authorizes_local_replay(tmp_path, monkeypatch, rearm, fault):
    coordinator, old, receipt, newer = setup_failure(tmp_path, rearm=rearm)
    try:
        connection = coordinator._require()
        before = snapshot(coordinator)
        class FailingConnection:
            def execute(self, query, *args, **kwargs):
                if query == "BEGIN TRANSACTION" and fault == "begin":
                    raise RuntimeError("read transaction begin unavailable")
                if query == "COMMIT" and fault == "commit":
                    raise RuntimeError("read transaction commit unavailable")
                result = connection.execute(query, *args, **kwargs)
                if query == "COMMIT" and fault == "commit_response":
                    raise RuntimeError("read transaction commit response lost")
                return result
        monkeypatch.setattr(coordinator, "_require", lambda: FailingConnection())
        with pytest.raises(RuntimeError, match="read transaction"):
            coordinator.observe_rearmed_failed_task_claim(old, failure_receipt=receipt)
        assert snapshot(coordinator) == before
    finally:
        coordinator.close()
