"""Owner-side historical observations never constitute recovery authority."""
from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from test.api.test_agent_supervisor_database_portal_bridge import (
    _seed_terminal_repair_history, _TerminalRepairReceiptAuthority,
)


@pytest.fixture
def native(tmp_path):
    daemon = module.DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:observation",
        authority_mode="embedded_exclusive", task_source_kind="duckdb",
        require_real_execution=False,
    )
    receipts = _seed_terminal_repair_history(daemon, count=2)
    # The pagination fixture predates canonical producer serialization.
    # Seed disposable rows using the native writer's current encoding.
    connection = daemon._require_connection()
    for table, column in [("attempt_phases", "body_json"),
                          ("database_portal_terminal_reconciliations", "record_json")]:
        for row in connection.execute(f"SELECT attempt_id, {column} FROM {table}").fetchall():
            connection.execute(
                f"UPDATE {table} SET {column} = ? WHERE attempt_id = ?",
                [module._database_daemon_json(json.loads(row[1])), row[0]],
            )
    attempt = daemon.get_attempt("attempt:terminal-history:0000")
    try:
        yield daemon, attempt, receipts
    finally:
        daemon.close()


def snapshot(daemon):
    return {
        table: daemon._require_connection().execute(f'SELECT * FROM "{table}" ORDER BY ALL').fetchall()
        for table in ("daemon_execution_metadata", *module._DAEMON_EXECUTION_REQUIRED_COLUMNS)
    }


def set_phase(daemon, attempt, body):
    daemon._require_connection().execute(
        "UPDATE attempt_phases SET body_json = ? WHERE attempt_id = ? AND phase = 'failed'",
        [body if isinstance(body, str) else module._database_daemon_json(body), attempt.attempt_id],
    )


def test_exact_snapshot_is_bounded_private_and_read_only(native):
    daemon, attempt, _ = native
    phase = daemon.phase_history(attempt.attempt_id)[0]["body"]
    phase["private_result"] = "DO_NOT_EXPORT_CALLBACK_PAYLOAD"
    set_phase(daemon, attempt, phase)
    before = snapshot(daemon)
    observation = daemon.observe_terminal_attempt(attempt)
    assert snapshot(daemon) == before
    assert observation["available"] is True
    assert observation["recovery_authorized"] is False
    assert observation["completion_authorized"] is False
    assert observation["canonical_task_history_included"] is False
    assert observation["process_birth"] == daemon.process_birth.to_dict()
    assert observation["execution_store_identity"] == daemon.execution_store_identity
    assert observation["groups"]["database_task_attempts"]["count"] == 1
    assert observation["phase_dispositions"] == [{
        "phase": "failed", "actual": "terminalized_for_retry",
        "intended": "terminalized_for_retry", "attempt_consumed": None,
    }]
    assert "DO_NOT_EXPORT_CALLBACK_PAYLOAD" not in json.dumps(observation)
    digest = observation.pop("observation_digest")
    assert digest == daemon._database_canonical_digest(observation)


@pytest.mark.parametrize("change", ["owner", "revision", "body", "fence", "running"])
def test_replacement_or_nonterminal_attempt_is_rejected(native, change):
    daemon, attempt, _ = native
    candidate = {
        "owner": replace(attempt, owner_session_id="foreign"),
        "revision": replace(attempt, revision=attempt.revision + 1),
        "body": replace(attempt, body={"foreign": True}),
        "fence": replace(attempt, fencing_token=attempt.fencing_token + 1),
        "running": replace(attempt, status="running"),
    }[change]
    before = snapshot(daemon)
    with pytest.raises((module.DatabaseImplementationAuthorityError, module.DatabaseImplementationConflictError)):
        daemon.observe_terminal_attempt(candidate)
    assert snapshot(daemon) == before


@pytest.mark.parametrize("change", ["writer_fence", "birth", "callback", "metadata", "late_fence"])
def test_owner_must_remain_current_through_snapshot(native, monkeypatch, change):
    daemon, attempt, _ = native
    if change == "writer_fence":
        monkeypatch.setattr(daemon, "_retained_embedded_writer_fence_current", lambda: False)
    elif change == "birth":
        monkeypatch.setattr(module, "current_process_birth", lambda: None)
    elif change == "callback":
        monkeypatch.setattr(daemon, "_active_external_callbacks", 1)
    elif change == "metadata":
        daemon._require_connection().execute(
            "UPDATE daemon_execution_metadata SET value='foreign' WHERE key='execution_store_identity'",
        )
    else:
        observations = iter([True, False])
        monkeypatch.setattr(daemon, "_retained_embedded_writer_fence_current", lambda: next(observations))
    before = snapshot(daemon)
    with pytest.raises(module.DatabaseImplementationAuthorityError):
        daemon.observe_terminal_attempt(attempt)
    assert snapshot(daemon) == before


@pytest.mark.parametrize("body", ['{"database_disposition":', '{"x":1,"x":2}', '{"x":"' + 'a' * 262_144 + '"}'], ids=['malformed', 'duplicate', 'oversized'])
def test_invalid_or_oversized_history_is_unavailable_without_mutation(native, body):
    daemon, attempt, _ = native
    set_phase(daemon, attempt, body)
    before = snapshot(daemon)
    with pytest.raises(module.DatabaseImplementationConflictError):
        daemon.observe_terminal_attempt(attempt)
    assert snapshot(daemon) == before


def test_conflict_is_observed_without_rewriting_or_advancing_cursor(native):
    daemon, attempt, receipts = native
    phase = daemon.phase_history(attempt.attempt_id)[0]["body"]
    phase["database_disposition"] = "blocked_unknown_outcome"
    set_phase(daemon, attempt, phase)
    before = snapshot(daemon)
    for _ in range(2):
        result = daemon._repair_database_portal_terminal_receipts(
            bridge=_TerminalRepairReceiptAuthority(receipts), trigger="test", exact_attempt=attempt,
        )
        assert len(result) == 1 and result[0]["blocked"] is True
        assert result[0]["reconciled"] is False
        assert result[0]["error"] == "terminal phase changed its actual database disposition"
        observation = result[0]["terminal_attempt_observation"]
        assert observation["available"] is True
        assert observation["phase_dispositions"][0]["actual"] == "blocked_unknown_outcome"
        assert observation["phase_dispositions"][0]["intended"] == "terminalized_for_retry"
        assert snapshot(daemon) == before


def test_unavailable_observation_does_not_hide_original_failure(native, monkeypatch):
    daemon, attempt, receipts = native
    phase = daemon.phase_history(attempt.attempt_id)[0]["body"]
    phase["database_disposition"] = "blocked_unknown_outcome"
    set_phase(daemon, attempt, phase)
    def unavailable(*_):
        raise RuntimeError("DO_NOT_EXPORT_PRIVATE_ERROR")
    monkeypatch.setattr(daemon, "observe_terminal_attempt", unavailable)
    before = snapshot(daemon)
    result = daemon._repair_database_portal_terminal_receipts(
        bridge=_TerminalRepairReceiptAuthority(receipts), trigger="test", exact_attempt=attempt,
    )
    assert result[0]["blocked"] is True and result[0]["reconciled"] is False
    assert result[0]["terminal_attempt_observation"]["available"] is False
    assert "DO_NOT_EXPORT_PRIVATE_ERROR" not in json.dumps(result)
    assert snapshot(daemon) == before
