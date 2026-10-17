"""Ordinary recovery must leave administrative UNKNOWN closeout inert.

Deliberately incomplete/unaudited FAILED rows exist only in a disposable native
execution database. The real daemon must skip them before validating authority,
opening privileged adapters, changing another store or creating a generic event.
"""
from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinator
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon, DATABASE_PORTAL_FAILURE_SETTLEMENT_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy


def test_ordinary_native_reconciliation_skips_admin_kind_before_validation_or_mutation(tmp_path, monkeypatch):
    canonical, coordination, execution = (tmp_path / (role + ".duckdb") for role in ("canonical", "coordination", "execution"))
    repository = IntentRepository(canonical)
    repository.upsert_goal(goal_cid="test-goal", goal_alias="TEST-G", title="TEST ONLY")
    repository.upsert_task(task_cid="test-task", task_alias="TEST-T", goal_cid="test-goal",
                           status="in_progress", body={"objective": "TEST ONLY"})
    source = DatabaseTaskSource(intent=repository)
    coordinator = DatabaseCoordinator(coordination)
    daemon = DatabaseImplementationDaemon(database_path=canonical, coordination_path=coordination,
        execution_path=execution, authority_mode="embedded_exclusive", task_source_kind="duckdb",
        task_source=source, coordinator=coordinator, require_real_execution=True)
    daemon.open()
    connection = daemon._require_connection()
    # Fixture corruption is intentional; this body conveys no ADMIN authority.
    body = {"schema": DATABASE_PORTAL_FAILURE_SETTLEMENT_SCHEMA,
            "failure_kind": policy.FAILURE_KIND, "settlement_id": "TEST-ONLY-incomplete"}
    connection.execute("INSERT INTO database_task_attempts "
        "(attempt_id,claim_id,task_cid,task_alias,attempt_number,owner_session_id,fencing_token,fence_epoch,lease_id,"
        "committed_phase,status,started_at_ms,finished_at_ms,revision,body_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ["test-attempt2", "test-claim2", "test-task", "TEST-T", 2, "test-owner", 2, 2,
         "test-lease2", "failed", "failed", 1000, 2000, 3, "{}"])
    connection.execute("INSERT INTO attempt_phases "
        "(attempt_id,phase,committed_at_ms,fencing_token,fence_epoch,revision,body_json) VALUES (?,?,?,?,?,?,?)",
        ["test-attempt2", "failed", 2000, 2, 2, 3, json.dumps(body)])
    before_attempt = daemon.get_attempt("test-attempt2").to_dict()
    before_task = policy.plain(repository.get_task("test-task"))
    before_events = connection.execute("SELECT * FROM daemon_execution_events ORDER BY event_id").fetchall()

    def refuse_unexpected(*_args, **_kwargs):
        raise AssertionError("ordinary recovery crossed the explicit ADMIN boundary")
    monkeypatch.setattr(daemon, "_validate_portal_failure_settlement_receipt", refuse_unexpected)
    monkeypatch.setattr(daemon, "_fail_coordination_claim", refuse_unexpected)
    monkeypatch.setattr(daemon, "_settle_portal_failure_control_task", refuse_unexpected)
    try:
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.get_attempt("test-attempt2").to_dict() == before_attempt
        assert policy.plain(repository.get_task("test-task")) == before_task
        assert connection.execute("SELECT * FROM daemon_execution_events ORDER BY event_id").fetchall() == before_events
    finally:
        daemon.close()
        coordinator.close()
        source.close()
