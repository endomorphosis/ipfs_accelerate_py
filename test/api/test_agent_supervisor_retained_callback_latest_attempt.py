"""The retained callback path must use the real latest-cursor denial gate."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)
from test.api.test_retained_callback_suffix import _physical_fixture


@pytest.mark.parametrize(
    "cursor", ["exact", "newer_foreign_owner", "equal_foreign_owner", "older", "other_task", "foreign_claim"]
)
def test_physical_suffix_rejects_superseding_cursor_without_stubbing_latest(tmp_path, cursor):
    daemon, task, attempts, _phases, _history = _physical_fixture()
    current = list(attempts.values())[-1]
    # The original fixture hid the missing production method with lambda True.
    del daemon._local_attempt_is_exact_latest
    daemon._uses_quack_command_gateway = lambda: False
    connection = open_duckdb_connection(tmp_path / "cursors.duckdb", prefer_quack=False)
    daemon._require_connection = lambda: connection
    try:
        connection.execute(
            "CREATE TABLE database_task_attempts (attempt_id VARCHAR, task_cid VARCHAR, "
            "attempt_number BIGINT, owner_session_id VARCHAR, status VARCHAR)"
        )
        for attempt in attempts.values():
            connection.execute(
                "INSERT INTO database_task_attempts VALUES (?, ?, ?, ?, ?)",
                [attempt.attempt_id, attempt.task_cid, attempt.attempt_number,
                 attempt.owner_session_id, attempt.status],
            )
        if cursor in {"newer_foreign_owner", "equal_foreign_owner", "older", "other_task"}:
            number = current.attempt_number + (1 if cursor in {"newer_foreign_owner", "other_task"} else 0)
            if cursor == "older":
                number -= 1
            connection.execute(
                "INSERT INTO database_task_attempts VALUES (?, ?, ?, ?, ?)",
                ["attempt:foreign", "task:other" if cursor == "other_task" else current.task_cid,
                 number, "session:foreign", "completed"],
            )
        if cursor == "foreign_claim":
            # Latest alone cannot substitute for the existing receipt binding.
            attempts[current.attempt_id] = replace(current, claim_id="claim:foreign")
        before = connection.execute("SELECT * FROM database_task_attempts ORDER BY attempt_id").fetchall()
        context = daemon._retained_callback_suffix_context(task)
        assert (context is not None) == (cursor in {"exact", "older", "other_task"})
        if context is not None:
            assert context["current_attempt"] == current
        assert connection.execute("SELECT * FROM database_task_attempts ORDER BY attempt_id").fetchall() == before
    finally:
        connection.close()


@pytest.mark.parametrize(
    "changed", [None, "absent", "attempt_id", "claim_id", "task_cid", "owner_session_id",
                "fencing_token", "fence_epoch", "attempt_number", "lease_id", "revision", "status"]
)
def test_typed_latest_cursor_requires_exact_owner_gateway_projection(changed):
    fixture, _task, attempts, _phases, _history = _physical_fixture()
    current = list(attempts.values())[-1]
    record = current.to_dict()
    if changed == "absent":
        record = None
    elif changed is not None:
        record[changed] = record[changed] + 1 if type(record[changed]) is int else "foreign"
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._uses_quack_command_gateway = lambda: True
    calls = []

    def get_attempt(attempt_id):
        calls.append(attempt_id)
        return record

    daemon._require_execution_repository = lambda: SimpleNamespace(get_attempt=get_attempt)
    daemon._require_connection = lambda: pytest.fail("typed observation opened a raw database")
    assert daemon._local_attempt_is_exact_latest(current) is (changed is None)
    assert calls == [current.attempt_id]


def test_typed_latest_cursor_authority_failure_does_not_fall_back():
    _fixture, _task, attempts, _phases, _history = _physical_fixture()
    current = list(attempts.values())[-1]
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._uses_quack_command_gateway = lambda: True

    def get_attempt(_attempt_id):
        raise DatabaseImplementationAuthorityError("current task lease differs")

    daemon._require_execution_repository = lambda: SimpleNamespace(get_attempt=get_attempt)
    daemon._require_connection = lambda: pytest.fail("refused typed observation opened a raw database")
    with pytest.raises(DatabaseImplementationAuthorityError, match="current task lease differs"):
        daemon._local_attempt_is_exact_latest(current)
