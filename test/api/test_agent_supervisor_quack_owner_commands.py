"""Typed, repository-backed Quack state-owner command tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceCompletionError,
    TaskSourceConflictError,
    execute_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
    QUACK_OWNER_COMMAND_MAX_BYTES,
    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
    QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
    QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
    QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
    QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    QuackOwnerCommandRemoteError,
    open_duckdb_connection,
    quack_owner_command_response,
    quack_owner_command_signature,
    submit_quack_owner_command,
    validate_quack_owner_command,
    validate_quack_owner_command_request,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
    IntentRepositoryConflictError,
)


def _materialize_one_task(path: Path) -> None:
    with DatabaseTaskSource(path) as source:
        source.materialize(
            {
                "repository_tree_id": "tree:typed-owner-test",
                "objectives": [
                    {
                        "goal_id": "G1",
                        "goal_cid": "goal:typed-owner-test",
                        "objective_id": "objective:typed-owner-test",
                        "title": "Typed owner commands",
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "T1",
                        "task_cid": "task:typed-owner-test",
                        "goal_cid": "goal:typed-owner-test",
                        "status": "ready",
                    }
                ],
            }
        )


def _signed_request(*, now_ms: int = 1_000_000) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": "a" * 32,
        "issued_at_ms": now_ms,
        "writer_identity": "supervisor-process:123",
        "store_id": "data/control.duckdb",
        "store_generation": "generation-1",
        "command": QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        "payload": {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("1" * 64),
            "body": {"producer": "pytest"},
        },
    }
    request["signature"] = quack_owner_command_signature(request, "token_value_123")
    return request


def test_owner_request_is_closed_typed_and_has_no_sql() -> None:
    request = _signed_request()
    command, payload = validate_quack_owner_command_request(
        request,
        token="token_value_123",
        expected_request_id="a" * 32,
        expected_store_id="data/control.duckdb",
        expected_store_generation="generation-1",
        now_ms=1_000_000,
    )
    assert command == QUACK_OWNER_COMMAND_RECORD_EVIDENCE
    assert payload["task_cid"] == "task:typed-owner-test"
    assert "sql" not in request
    assert "parameters" not in request
    with pytest.raises(DuckDBConnectionPolicyError, match="closed schema"):
        validate_quack_owner_command(
            command,
            {**payload, "sql": "DELETE FROM tasks"},
        )


def test_request_authentication_binds_generation_freshness_and_hmac() -> None:
    request = _signed_request()
    for changes, message in (
        ({"store_generation": "generation-2"}, "binding"),
        ({"issued_at_ms": 900_000}, "stale"),
        ({"signature": "0" * 64}, "authorization"),
    ):
        changed = {**request, **changes}
        with pytest.raises(DuckDBConnectionPolicyError, match=message):
            validate_quack_owner_command_request(
                changed,
                token="token_value_123",
                expected_request_id="a" * 32,
                expected_store_id="data/control.duckdb",
                expected_store_generation="generation-1",
                now_ms=1_000_000,
            )


def test_submit_round_trips_bound_typed_result(tmp_path: Path, monkeypatch) -> None:
    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token_value_123")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "data/control.duckdb")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "generation-1")

    def owner() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if requests:
                request_path = requests[0]
                request = json.loads(request_path.read_text(encoding="utf-8"))
                assert set(request).isdisjoint({"sql", "parameters"})
                response = quack_owner_command_response(
                    request,
                    token="token_value_123",
                    result={"schema": "typed-result@1", "value": 7},
                )
                request_path.with_name(
                    request_path.name.replace(".request.json", ".done.json")
                ).write_text(json.dumps(response), encoding="utf-8")
                return
            time.sleep(0.01)
        raise AssertionError("typed command request did not arrive")

    thread = threading.Thread(target=owner)
    thread.start()
    result = submit_quack_owner_command(
        QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("2" * 64),
        },
        timeout_seconds=2,
    )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not list(inbox.glob("*.request.json"))
    assert result == {"schema": "typed-result@1", "value": 7}


def test_submit_rejects_forged_owner_response(tmp_path: Path, monkeypatch) -> None:
    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token_value_123")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "data/control.duckdb")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "generation-1")

    def forger() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if requests:
                request_path = requests[0]
                request = json.loads(request_path.read_text(encoding="utf-8"))
                forged = {
                    "schema": QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
                    "request_id": request["request_id"],
                    "command": request["command"],
                    "store_id": request["store_id"],
                    "store_generation": request["store_generation"],
                    "ok": True,
                    "result": {"schema": "forged-result@1", "changed": True},
                    "signature": "0" * 64,
                }
                request_path.with_name(
                    request_path.name.replace(".request.json", ".done.json")
                ).write_text(json.dumps(forged), encoding="utf-8")
                return
            time.sleep(0.01)
        raise AssertionError("typed command request did not arrive")

    thread = threading.Thread(target=forger)
    thread.start()
    with pytest.raises(DuckDBConnectionPolicyError, match="authorization"):
        submit_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            {
                "task_cid": "task:typed-owner-test",
                "evidence_kind": "validation",
                "digest": "sha256:" + ("4" * 64),
            },
            timeout_seconds=2,
        )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(list(inbox.glob("*.request.json"))) == 1


@pytest.mark.parametrize("response_kind", ["symlink", "oversize"])
def test_submit_rejects_unsafe_owner_response_file(
    response_kind: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token_value_123")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "data/control.duckdb")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "generation-1")

    def unsafe_writer() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if requests:
                request_path = requests[0]
                done_path = request_path.with_name(
                    request_path.name.replace(".request.json", ".done.json")
                )
                if response_kind == "symlink":
                    target = tmp_path / "forged-response.json"
                    target.write_text("{}", encoding="utf-8")
                    done_path.symlink_to(target)
                else:
                    done_path.write_bytes(b"x" * (QUACK_OWNER_COMMAND_MAX_BYTES + 1))
                return
            time.sleep(0.01)
        raise AssertionError("typed command request did not arrive")

    thread = threading.Thread(target=unsafe_writer)
    thread.start()
    with pytest.raises(DuckDBConnectionPolicyError, match="response"):
        submit_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            {
                "task_cid": "task:typed-owner-test",
                "evidence_kind": "validation",
                "digest": "sha256:" + ("5" * 64),
            },
            timeout_seconds=2,
        )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(list(inbox.glob("*.request.json"))) == 1


def test_bound_repository_command_is_atomic_and_restart_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        first_repo = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:first",
        )
        kwargs = {
            "request_id": "b" * 32,
            "store_id": "data/control.duckdb",
            "store_generation": "generation-1",
        }
        payload = {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("3" * 64),
        }
        first = execute_quack_owner_command(
            first_repo,
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            payload,
            **kwargs,
        )
        first_repo.close()
        owner_connection.close()

        # A new connection and repository model an owner process restart while
        # the durable idempotency record remains in the authoritative database.
        owner_connection = open_duckdb_connection(path)
        second_repo = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:second",
        )
        replay = execute_quack_owner_command(
            second_repo,
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            payload,
            **kwargs,
        )
        assert replay == first
        event_count = owner_connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_type = ?",
            ["intent.evidence_recorded"],
        ).fetchone()
        assert event_count is not None and int(event_count[0]) == 1
        record_count = owner_connection.execute(
            "SELECT COUNT(*) FROM idempotency_records WHERE idempotency_key = ?",
            ["quack-owner-command:" + ("b" * 32)],
        ).fetchone()
        assert record_count is not None and int(record_count[0]) == 1
        second_repo.close()
        # Repository lifecycle never closes the owner's injected connection.
        assert owner_connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        owner_connection.close()


def test_owner_dispatches_exact_five_live_mutations(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:five-commands",
        )
        bindings = {
            "store_id": "data/control.duckdb",
            "store_generation": "generation-1",
        }
        calls = (
            (
                QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
                {
                    "task_cid": "task:typed-owner-test",
                    "delay_ms": 1,
                    "reason": "retry-later",
                    "selection_penalty": 1,
                },
            ),
            (
                QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
                {"task_cid": "task:typed-owner-test"},
            ),
            (
                QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
                {
                    "task_cid": "task:typed-owner-test",
                    "evidence_kind": "validation",
                    "digest": "sha256:" + ("6" * 64),
                },
            ),
            (
                QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
                {
                    "task_cid": "task:typed-owner-test",
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + ("7" * 64),
                    "argv": ["pytest", "-q"],
                },
            ),
            (
                QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
                {
                    "task_cid_or_alias": "task:typed-owner-test",
                    "expected_revision": 1,
                    "status": "running",
                },
            ),
        )
        results = [
            execute_quack_owner_command(
                repository,
                command,
                payload,
                request_id=f"{index:x}" * 32,
                **bindings,
            )
            for index, (command, payload) in enumerate(calls, start=1)
        ]
        assert all(result.get("schema") for result in results)
        assert results[-1]["task"]["status"] == "running"
        assert (
            len(
                owner_connection.execute(
                    "SELECT idempotency_key FROM idempotency_records "
                    "WHERE command_kind LIKE 'record_%' OR "
                    "command_kind = 'compare_and_set_status'"
                ).fetchall()
            )
            == 5
        )
        repository.close()
    finally:
        owner_connection.close()


def test_owner_rearms_blocked_task_without_client_revision(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    with IntentRepository(
        path,
        install_schema=False,
        owner_id="owner:typed-test",
        session_id="session:block",
    ) as repository:
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        repository.block_task(
            task_cid="task:typed-owner-test",
            blocker_kind="dependency",
            blocker_id="task:other",
            reason="waiting",
            expected_revision=int(task["revision"]),
        )
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:rearm",
        )
        result = execute_quack_owner_command(
            repository,
            QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
            {
                "task_cid_or_alias": "task:typed-owner-test",
                "receipt": {"operation": "database_declared_outputs_on_head_rearm"},
            },
            request_id="c" * 32,
            store_id="data/control.duckdb",
            store_generation="generation-1",
        )
        assert result["changed"] is True
        assert result["previous_status"] == "blocked"
        assert result["task"]["status"] == "retrying"
        replay = execute_quack_owner_command(
            repository,
            QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
            {"task_cid_or_alias": "task:typed-owner-test"},
            request_id="d" * 32,
            store_id="data/control.duckdb",
            store_generation="generation-1",
        )
        assert replay["changed"] is False
        assert replay["task"]["status"] == "retrying"
        repository.close()
    finally:
        owner_connection.close()


def test_quack_transport_sql_mutations_fail_closed() -> None:
    class NeverExecute:
        description = None

        def execute(self, *_args, **_kwargs):
            raise AssertionError("raw Quack mutation reached the connection")

        def close(self) -> None:
            return None

    connection = DuckDBConnection.wrap(NeverExecute())
    connection._default_catalog = "control_plane"  # noqa: SLF001
    with pytest.raises(DuckDBConnectionPolicyError, match="read-only"):
        connection.execute(
            "UPDATE tasks SET status = ? WHERE task_cid = ?",
            ["blocked", "task:typed-owner-test"],
        )
    with pytest.raises(DuckDBConnectionPolicyError, match="executemany"):
        connection.executemany("INSERT INTO tasks VALUES (?)", [["x"]])
    with pytest.raises(DuckDBConnectionPolicyError, match="scripts"):
        connection.executescript("DELETE FROM tasks")


def test_database_task_source_maps_typed_owner_failures(monkeypatch) -> None:
    source = DatabaseTaskSource(
        "quack:127.0.0.1:45123",
        install_schema=False,
    )

    def conflict(*_args, **_kwargs):
        raise QuackOwnerCommandRemoteError("conflict", "stale CAS")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.submit_quack_owner_command",
        conflict,
    )
    with pytest.raises(TaskSourceConflictError, match="stale CAS"):
        source.compare_and_set_status("task:1", 1, "running")

    def completion(*_args, **_kwargs):
        raise QuackOwnerCommandRemoteError("completion_refused", "current evidence required")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.submit_quack_owner_command",
        completion,
    )
    with pytest.raises(TaskSourceCompletionError, match="current evidence"):
        source.compare_and_set_status("task:1", 1, "completed")
    source.close()


def test_block_and_unblock_use_revision_predicates(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    with IntentRepository(path, install_schema=False) as repository:
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        revision = int(task["revision"])
        with pytest.raises(IntentRepositoryConflictError, match="blocking"):
            repository.block_task(
                task_cid="task:typed-owner-test",
                blocker_kind="dependency",
                blocker_id="task:other",
                reason="waiting",
                expected_revision=revision + 1,
            )
        repository.block_task(
            task_cid="task:typed-owner-test",
            blocker_kind="dependency",
            blocker_id="task:other",
            reason="waiting",
            expected_revision=revision,
        )
        blocked = repository.get_task("task:typed-owner-test")
        assert blocked is not None and blocked["status"] == "blocked"
        with pytest.raises(IntentRepositoryConflictError, match="unblocking"):
            repository.unblock_task(
                task_cid="task:typed-owner-test",
                expected_revision=revision,
            )
        repository.unblock_task(
            task_cid="task:typed-owner-test",
            expected_revision=int(blocked["revision"]),
        )
        ready = repository.get_task("task:typed-owner-test")
        assert ready is not None and ready["status"] == "ready"
