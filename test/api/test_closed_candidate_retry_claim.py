"""A closed candidate-gate miss can be claimed again after its fence expires."""

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalCandidateRetry,
    closed_candidate_retry_permits_successor_claim,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


def _facts(**updates):
    phase = {
        "schema": "database-bounded-portal-retry@1",
        "reason": "proposal_gate_failed",
        "pre_effect": False,
        "provider_dispatched": True,
        "attempt_consumed": True,
        "retry_budget_exhausted": False,
    }
    receipt = {
        "operation": "database_portal_retry",
        "attempt_id": "attempt:1",
        "reason": "proposal_gate_failed",
        "owner_session_id": "session:lane",
    }
    value = {
        "task_cid": "task:1",
        "attempt_id": "attempt:1",
        "attempt_status": "failed",
        "attempt_task_cid": "task:1",
        "attempt_owner_session_id": "session:lane",
        "owner_session_id": "session:lane",
        "phases": [{"phase": "claimed", "body": {}}, {"phase": "failed", "body": phase}],
        "effect_claim_present": False,
        "task_status": "retrying",
        "control_receipt": receipt,
        "claim_state": "expired",
    }
    value.update(updates)
    return value


def test_closed_gate_miss_permits_one_successor_claim():
    assert closed_candidate_retry_permits_successor_claim(**_facts()) is True


@pytest.mark.parametrize(
    "updates",
    [
        {"attempt_status": "running"},
        {"effect_claim_present": True},
        {"task_status": "in_progress"},
        {"claim_state": "accepted"},
        {"phases": [{"phase": "provider", "body": {}}, {"phase": "failed", "body": _facts()["phases"][1]["body"]}]},
        {"phases": [{"phase": "failed", "body": {**_facts()["phases"][1]["body"], "reason": "provider_crashed"}}]},
        {"phases": [{"phase": "failed", "body": {**_facts()["phases"][1]["body"], "retry_budget_exhausted": True}}]},
        {"control_receipt": {**_facts()["control_receipt"], "attempt_id": "attempt:other"}},
        {"attempt_owner_session_id": "session:other"},
    ],
)
def test_closed_gate_miss_stays_unclaimable_without_exact_evidence(updates):
    assert closed_candidate_retry_permits_successor_claim(**_facts(**updates)) is False


def test_expired_candidate_gate_retry_is_claimed_again(tmp_path):
    calls = []

    def provider(attempt):
        calls.append(attempt.attempt_id)
        raise DatabasePortalCandidateRetry("proposal_gate_failed")

    now = {"ms": 1_700_000_000_000}
    daemon: DatabaseImplementationDaemon = _open_daemon(
        tmp_path,
        session="session:lane",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt_id = first["implementation_result"]["attempt_id"]
        task_cid = daemon.get_attempt(attempt_id).task_cid
        unknown = {
            "schema": "database-portal-callback-intent@1",
            "callback_state": "started_outcome_unknown",
            "provider_effect_state": "unknown_may_have_started",
            "attempt_id": attempt_id,
        }
        # A plain provider raises before the bridge records dispatch intent.
        # Insert the same unknown callback the portal leaves behind.
        connection = daemon._require_connection()
        existing = connection.execute(
            "SELECT invocation_id FROM provider_invocations WHERE attempt_id = ?",
            [attempt_id],
        ).fetchone()
        if existing is None:
            connection.execute(
                """INSERT INTO provider_invocations(
                    invocation_id, attempt_id, task_cid, idempotency_key,
                    owner_session_id, recorded_at_ms, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    "provider:test-unknown",
                    attempt_id,
                    task_cid,
                    f"provider:{attempt_id}",
                    "session:lane",
                    now["ms"],
                    json.dumps(unknown),
                ],
            )
        else:
            connection.execute(
                "UPDATE provider_invocations SET result_json = ? WHERE attempt_id = ?",
                [json.dumps(unknown), attempt_id],
            )
        assert task_cid in daemon._automatic_claim_exclusions()
        claim = daemon.coordinator.get_task_claim(daemon.get_attempt(attempt_id).claim_id)
        now["ms"] += 5_000
        daemon.coordinator.expire_task_claim(claim)
        assert task_cid not in daemon._automatic_claim_exclusions()
        successor = daemon.claim_next()
        assert successor is not None
        assert successor.attempt_id != attempt_id
        assert successor.task_cid == task_cid
        assert calls == [attempt_id]
    finally:
        daemon.close()


def test_accepted_candidate_gate_retry_releases_without_waiting_for_expiry(tmp_path):
    calls = []

    def provider(attempt):
        calls.append(attempt.attempt_id)
        raise DatabasePortalCandidateRetry("proposal_gate_failed")

    now = {"ms": 1_700_000_000_000}
    daemon: DatabaseImplementationDaemon = _open_daemon(
        tmp_path,
        session="session:lane",
        provider_fn=provider,
        lease_ms=90 * 60 * 1000,
        clock_ms=lambda: now["ms"],
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt_id = first["implementation_result"]["attempt_id"]
        task_cid = daemon.get_attempt(attempt_id).task_cid
        unknown = {
            "schema": "database-portal-callback-intent@1",
            "callback_state": "started_outcome_unknown",
            "provider_effect_state": "unknown_may_have_started",
            "attempt_id": attempt_id,
        }
        connection = daemon._require_connection()
        existing = connection.execute(
            "SELECT invocation_id FROM provider_invocations WHERE attempt_id = ?",
            [attempt_id],
        ).fetchone()
        if existing is None:
            connection.execute(
                """INSERT INTO provider_invocations(
                    invocation_id, attempt_id, task_cid, idempotency_key,
                    owner_session_id, recorded_at_ms, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    "provider:test-unknown-accepted",
                    attempt_id,
                    task_cid,
                    f"provider:{attempt_id}",
                    "session:lane",
                    now["ms"],
                    json.dumps(unknown),
                ],
            )
        else:
            connection.execute(
                "UPDATE provider_invocations SET result_json = ? WHERE attempt_id = ?",
                [json.dumps(unknown), attempt_id],
            )
        assert task_cid in daemon._automatic_claim_exclusions()
        successor = daemon.claim_next()
        assert successor is not None
        assert successor.attempt_id != attempt_id
        assert successor.task_cid == task_cid
        released = daemon.coordinator.get_task_claim(daemon.get_attempt(attempt_id).claim_id)
        assert str(getattr(released.state, "value", released.state)) == "released"
        stored = json.loads(connection.execute(
            "SELECT result_json FROM provider_invocations WHERE attempt_id = ?",
            [attempt_id],
        ).fetchone()[0])
        assert stored["callback_state"] == "started_outcome_unknown"
        assert calls == [attempt_id]
    finally:
        daemon.close()


def _plant_unknown_callback(daemon, attempt_id: str, task_cid: str, now_ms: int) -> None:
    unknown = {
        "schema": "database-portal-callback-intent@1",
        "callback_state": "started_outcome_unknown",
        "provider_effect_state": "unknown_may_have_started",
        "attempt_id": attempt_id,
        "task_cid": task_cid,
    }
    connection = daemon._require_connection()
    existing = connection.execute(
        "SELECT invocation_id FROM provider_invocations WHERE attempt_id = ?",
        [attempt_id],
    ).fetchone()
    if existing is None:
        connection.execute(
            """INSERT INTO provider_invocations(
                invocation_id, attempt_id, task_cid, idempotency_key,
                owner_session_id, recorded_at_ms, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                f"provider:test-unknown-{attempt_id}",
                attempt_id,
                task_cid,
                f"provider:{attempt_id}",
                "session:lane",
                now_ms,
                json.dumps(unknown),
            ],
        )
    else:
        connection.execute(
            "UPDATE provider_invocations SET result_json = ? WHERE attempt_id = ?",
            [json.dumps(unknown), attempt_id],
        )


def test_second_candidate_gate_retry_is_not_fenced_by_the_first_callback(tmp_path):
    calls = []

    def provider(attempt):
        calls.append(attempt.attempt_id)
        raise DatabasePortalCandidateRetry("proposal_gate_failed")

    now = {"ms": 1_700_000_000_000}
    daemon: DatabaseImplementationDaemon = _open_daemon(
        tmp_path,
        session="session:lane",
        provider_fn=provider,
        lease_ms=90 * 60 * 1000,
        clock_ms=lambda: now["ms"],
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt_one = first["implementation_result"]["attempt_id"]
        task_cid = daemon.get_attempt(attempt_one).task_cid
        _plant_unknown_callback(daemon, attempt_one, task_cid, now["ms"])
        second_claim = daemon.claim_next()
        assert second_claim is not None and second_claim.attempt_id != attempt_one
        second = daemon.run_once()
        attempt_two = second["implementation_result"]["attempt_id"]
        assert attempt_two == second_claim.attempt_id
        _plant_unknown_callback(daemon, attempt_two, task_cid, now["ms"])
        assert task_cid in daemon._automatic_claim_exclusions()
        third = daemon.claim_next()
        assert third is not None
        assert third.attempt_id not in {attempt_one, attempt_two}
        assert third.task_cid == task_cid
        assert calls == [attempt_one, attempt_two]
    finally:
        daemon.close()
