"""Coordination must not acknowledge an unsuccessful durable commit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

duckdb = pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator, open_database_coordinator,
)


def test_failed_preparation_commit_propagates_and_rolls_back(tmp_path, monkeypatch):
    coordinator = open_database_coordinator(tmp_path / "coordination.duckdb")
    try:
        coordinator.register_task(task_cid="task:commit-failure", task_id="COMMIT-001")
        claim = coordinator.claim_task(task_cid="task:commit-failure", owner_session_id="owner:test")
        connection = coordinator._require()
        def failed_commit():
            raise duckdb.FatalException("injected commit-time index failure")
        monkeypatch.setattr(connection, "commit", failed_commit)
        with pytest.raises(duckdb.FatalException, match="commit-time index failure"):
            coordinator.prepare_task_completion(
                claim, control_expected_revision=2, control_expected_status="in_progress",
                evidence_digest="sha256:test-only-evidence")
        assert not connection.in_transaction
        assert connection.execute("SELECT COUNT(*) FROM task_completions").fetchone()[0] == 0
        assert connection.execute("SELECT ready FROM coordination_tasks").fetchone()[0] is True
        assert connection.execute(
            "SELECT COUNT(*) FROM lease_events WHERE event_type='task_completion_prepared'"
        ).fetchone()[0] == 0
    finally:
        coordinator.close()


@pytest.mark.parametrize("wrapped", [True, False])
def test_raw_and_fallback_commit_errors_are_not_suppressed(wrapped):
    def fail():
        raise duckdb.FatalException("durability failure")
    raw = SimpleNamespace(commit=fail)
    connection = SimpleNamespace(_connection=raw) if wrapped else raw
    with pytest.raises(duckdb.FatalException, match="durability failure"):
        DatabaseCoordinator._commit_if_idle(None, connection)


def test_real_duckdb_commit_without_transaction_is_a_noop():
    connection = duckdb.connect()
    try:
        DatabaseCoordinator._commit_if_idle(None, connection)
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        connection.close()
