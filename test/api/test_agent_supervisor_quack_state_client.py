"""Tests for the typed Quack client, transactions, and retry adapter (DQP-007).

Evidence subset:

* parameter binding
* raw identifier rejection
* response loss (idempotent replay)
* duplicate command
* optimistic conflict
* stale generation
* reconnect
* cursor pagination

Acceptance:

* Independent processes commit non-conflicting work concurrently
* Same-row conflicts return/retry predictably
* Replay after lost response returns the one committed result
* Callers cannot interpolate identifiers or run arbitrary model-supplied SQL
"""

from __future__ import annotations

import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    ControlPlaneStoreIdentity,
    StateAuthorityClass,
    StateCommand,
    StoreGeneration,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    CASResult,
    FenceMismatchError,
    IdempotencyConflictError,
    OptimisticConflictError,
    RetryPolicy,
    StateTransaction,
    StaleGenerationError,
    TransactionConflictKind,
    classify_exception,
    default_retry_policy,
    is_retryable_exception,
    result_digest,
    run_with_retry,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    DEFAULT_STATEMENT_TEMPLATES,
    QUACK_STATE_CLIENT_INTERFACE,
    QuackClientIdentityError,
    QuackClientSQLError,
    QuackStateClient,
    StatementTemplate,
    TransportMode,
    open_embedded_client,
    resolve_endpoint,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for Quack state client hermetic tests",
)

_DIGEST = "sha256:" + ("ab" * 32)
_UUID = "123e4567-e89b-12d3-a456-426614174000"


def _install(db: Path) -> None:
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="quack-client-test",
    )


def _seed_goal_and_tasks(db: Path, *, count: int = 2) -> list[str]:
    task_cids: list[str] = []
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:root",
                "G-ROOT",
                "objective:test",
                "",
                1,
                "Root",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        for index in range(count):
            task_cid = f"task:cid:{index + 1:03d}"
            task_cids.append(task_cid)
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task_cid,
                    f"T-{index + 1:03d}",
                    "goal:root",
                    "",
                    "objective:test",
                    index + 1,
                    "ready",
                    0,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "{}",
                    "{}",
                ],
            )
    return task_cids


def _seed_generation(
    db: Path,
    *,
    generation: int = 1,
    fence_epoch: int = 1,
    revision: int = 0,
    database_uuid: str = _UUID,
    birth_id: str = "birth:server-1",
) -> None:
    with open_duckdb_connection(db) as connection:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (?, 1, ?, ?, ?, ?, ?)
            """,
            [
                generation,
                fence_epoch,
                revision,
                database_uuid,
                birth_id,
                "1970-01-01T00:00:00Z",
            ],
        )


def _client(db: Path, owner_id: str = "owner:test") -> QuackStateClient:
    client = QuackStateClient(owner_id=owner_id, store_id="control.duckdb")
    client.attach(db, mode=TransportMode.EMBEDDED, seed_generation=False)
    return client


def test_interface_identity_and_template_registry() -> None:
    assert QUACK_STATE_CLIENT_INTERFACE == "QuackStateClient@1"
    assert StateTransaction.INTERFACE == "StateTransaction@1"
    assert "list_tasks_page" in DEFAULT_STATEMENT_TEMPLATES
    assert "cas_task_status" in DEFAULT_STATEMENT_TEMPLATES
    client = QuackStateClient(owner_id="owner:x")
    assert "select_task_by_cid" in client.list_templates()
    template = client.get_template("select_task_by_cid")
    assert template.parameter_names == ("task_cid",)


def test_parameter_binding_and_raw_sql_rejection(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_goal_and_tasks(db, count=1)

    with _client(db) as client:
        rows = client.execute(
            "select_task_by_cid",
            {"task_cid": task_cids[0]},
        )
        assert len(rows) == 1
        assert rows[0]["task_cid"] == task_cids[0]
        # SQL injection payload is treated as a bound value, not SQL text.
        rows = client.execute(
            "select_task_by_cid",
            {"task_cid": "task:cid:001'; DROP TABLE tasks; --"},
        )
        assert rows == ()
        with pytest.raises(QuackClientSQLError):
            client.execute_sql("SELECT * FROM tasks")
        with pytest.raises(QuackClientSQLError):
            client.execute("select_task_by_cid'; DROP TABLE tasks")
        with pytest.raises(QuackClientSQLError):
            client.execute(
                "select_task_by_cid",
                {"task_cid": task_cids[0], "extra": "nope"},
            )
        with pytest.raises(QuackClientSQLError):
            StatementTemplate(
                name="evil",
                sql="SELECT * FROM tasks; DROP TABLE tasks",
                parameter_names=(),
            )


def test_raw_identifier_interpolation_rejected() -> None:
    with pytest.raises(QuackClientSQLError):
        StatementTemplate(
            name="bad_table",
            sql='SELECT * FROM "tasks; DROP TABLE tasks"',
            parameter_names=(),
        )
    # Dynamic table names are not accepted as templates without fixed SQL.
    with pytest.raises(QuackClientSQLError):
        StatementTemplate(
            name="not_an_ident!",
            sql="SELECT 1",
            parameter_names=(),
        )


def test_cursor_pagination(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    _seed_goal_and_tasks(db, count=5)

    with _client(db) as client:
        first = client.paginate(cursor=0, limit=2)
        assert len(first.items) == 2
        assert first.exhausted is False
        assert first.next_cursor == 2
        second = client.paginate(cursor=first.next_cursor or 0, limit=2)
        assert len(second.items) == 2
        assert second.next_cursor == 4
        third = client.paginate(cursor=second.next_cursor or 0, limit=2)
        assert len(third.items) == 1
        assert third.exhausted is True
        assert third.next_cursor is None
        seen = [item["task_cid"] for item in first.items + second.items + third.items]
        assert len(seen) == len(set(seen)) == 5


def test_optimistic_conflict_and_retry(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db, revision=0)
    task_cids = _seed_goal_and_tasks(db, count=1)
    task_cid = task_cids[0]

    sleeps: list[float] = []

    with _client(db) as client:
        live = client.load_generation()
        session = client.session
        assert session is not None

        # First writer advances task revision to 1 and store revision to 1.
        first = client.cas_task_status(
            task_cid=task_cid,
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:first",
            command_id="cmd:first",
        )
        assert first.outcome is CommandOutcome.ACCEPTED
        assert first.changed is True
        assert first.revision == 1

        # Stale expected store revision is an optimistic conflict; retry with
        # refresh should succeed once the command is refreshed against live.
        stale = StateCommand(
            command_id="cmd:second",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=0,  # stale store head
            fence_epoch=live.fence_epoch,
            idempotency_key="idem:second",
            parameters={
                "task_cid": task_cid,
                "expected_task_revision": 1,
                "status": "running",
            },
        )
        result = client.submit_command(
            stale,
            refresh_on_conflict=True,
        )
        assert result.outcome is CommandOutcome.ACCEPTED
        assert result.changed is True
        assert result.attempts >= 1
        assert result.revision >= 2

        # Same-row task revision conflict without refreshable domain CAS:
        # applying with a stale task revision raises optimistic conflict.
        def apply_stale_task(
            txn: StateTransaction,
            command: StateCommand,
            generation: StoreGeneration,
        ) -> dict[str, Any]:
            txn.cas_row_revision(
                table="tasks",
                key_column="task_cid",
                key_value=task_cid,
                expected_revision=0,  # already advanced
                assignments={"status": "failed", "updated_at": "now"},
            )
            return {"ok": False}

        attempts = {"n": 0}

        def operation(attempt: int) -> CASResult:
            attempts["n"] = attempt
            txn = client.transaction()
            live_now = client.load_generation()
            cmd = StateCommand(
                command_id=f"cmd:stale-task:{attempt}",
                command_kind=CommandKind.CLAIM,
                store_id="control.duckdb",
                session_id=session.session_id,
                expected_generation=live_now.generation,
                expected_revision=live_now.revision,
                fence_epoch=live_now.fence_epoch,
                idempotency_key=f"idem:stale-task:{attempt}",
                parameters={},
            )
            try:
                return txn.execute_command(cmd, apply=apply_stale_task)
            except OptimisticConflictError as exc:
                return CASResult(
                    outcome=CommandOutcome.CONFLICT,
                    changed=False,
                    revision=live_now.revision,
                    generation=live_now.generation,
                    fence_epoch=live_now.fence_epoch,
                    result={"error": str(exc)},
                    conflict_kind=TransactionConflictKind.OPTIMISTIC,
                    attempts=attempt,
                    idempotency_key=cmd.idempotency_key,
                    command_id=cmd.command_id,
                )

        policy = RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_ratio=0.0,
            seed=1,
        )
        conflicted = run_with_retry(
            operation,
            policy=policy,
            sleep=sleeps.append,
        )
        assert conflicted.outcome is CommandOutcome.CONFLICT
        assert conflicted.conflict_kind is TransactionConflictKind.OPTIMISTIC
        assert attempts["n"] == 3


def test_stale_generation_and_fence_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db, generation=2, fence_epoch=5, revision=3)

    with _client(db) as client:
        live = client.load_generation()
        assert live.generation == 2
        assert live.fence_epoch == 5
        session = client.session
        assert session is not None

        stale_gen = StateCommand(
            command_id="cmd:stale-gen",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=1,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key="idem:stale-gen",
            parameters={
                "task_cid": "task:missing",
                "expected_task_revision": 0,
                "status": "x",
            },
        )
        result = client.submit_command(stale_gen, refresh_on_conflict=False)
        assert result.outcome is CommandOutcome.STALE
        assert result.conflict_kind is TransactionConflictKind.STALE_GENERATION

        txn = client.transaction(
            expected_generation=StoreGeneration(
                store_id="control.duckdb",
                generation=2,
                schema_revision=1,
                fence_epoch=4,
                revision=3,
                database_uuid=_UUID,
                birth_id="birth:server-1",
            )
        )
        with pytest.raises(FenceMismatchError):
            txn.begin()
            try:
                txn.assert_expected_generation()
            finally:
                txn.rollback()


def test_idempotent_replay_after_response_loss(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_goal_and_tasks(db, count=1)
    task_cid = task_cids[0]

    with _client(db) as client:
        first = client.cas_task_status(
            task_cid=task_cid,
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:lost-response",
            command_id="cmd:lost-response",
        )
        assert first.outcome is CommandOutcome.ACCEPTED
        assert first.changed is True
        committed_digest = first.result_digest
        committed_revision = first.revision
        committed_body = dict(first.result)

        # Simulate lost response: caller retries the exact same command.
        replay = client.cas_task_status(
            task_cid=task_cid,
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:lost-response",
            command_id="cmd:lost-response",
        )
        assert replay.outcome is CommandOutcome.IDEMPOTENT_REPLAY
        assert replay.changed is False
        assert replay.result_digest == committed_digest
        assert dict(replay.result) == committed_body
        # Store revision must not advance again.
        assert client.load_generation().revision == committed_revision
        rows = client.execute("select_task_by_cid", {"task_cid": task_cid})
        assert rows[0]["status"] == "claimed"
        assert int(rows[0]["revision"]) == 1


def test_duplicate_command_different_payload_conflicts(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_goal_and_tasks(db, count=1)
    task_cid = task_cids[0]

    with _client(db) as client:
        session = client.session
        assert session is not None
        first = client.cas_task_status(
            task_cid=task_cid,
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:dup",
            command_id="cmd:dup-a",
        )
        assert first.accepted

        live = client.load_generation()
        conflicting = StateCommand(
            command_id="cmd:dup-b",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key="idem:dup",
            parameters={
                "task_cid": task_cid,
                "expected_task_revision": 1,
                "status": "running",
            },
        )
        with pytest.raises(IdempotencyConflictError):
            client.transaction().execute_command(
                conflicting,
                apply=client._default_task_status_apply,
            )


def test_reconnect_preserves_store_identity(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db, database_uuid=_UUID)
    _seed_goal_and_tasks(db, count=1)

    client = _client(db)
    try:
        before = client.load_generation()
        session_before = client.session
        assert session_before is not None
        client.reconnect()
        assert client.attached
        after = client.load_generation()
        assert after.database_uuid == before.database_uuid
        assert after.generation == before.generation
        assert client.session is not None
        assert client.session.session_id != session_before.session_id
        rows = client.execute("count_tasks")
        assert int(rows[0]["task_count"]) == 1
    finally:
        client.close()


def test_identity_mismatch_on_attach(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db, database_uuid=_UUID)
    expected = ControlPlaneStoreIdentity(
        repository_id="repository:sha256:test",
        database_uuid="00000000-0000-4000-8000-000000000099",
        store_id="control.duckdb",
        schema_revision=1,
        generation=1,
        schema_fingerprint=_DIGEST,
        authority_class=StateAuthorityClass.AUTHORITATIVE,
    )
    client = QuackStateClient(owner_id="owner:id", expected_identity=expected)
    with pytest.raises(QuackClientIdentityError):
        client.attach(db, mode=TransportMode.EMBEDDED, seed_generation=False)


def test_retry_policy_jitter_is_bounded() -> None:
    policy = RetryPolicy(
        max_attempts=5,
        base_delay_seconds=0.1,
        max_delay_seconds=1.0,
        jitter_ratio=0.25,
        seed=7,
    )
    rng = random.Random(7)
    delays = [policy.delay_for_attempt(i, rng=rng) for i in range(1, 5)]
    assert delays[0] == 0.0 or delays[0] >= 0.0
    for delay in delays[1:]:
        assert 0.0 <= delay <= 1.0
    # Deterministic with seed.
    rng2 = random.Random(7)
    delays2 = [policy.delay_for_attempt(i, rng=rng2) for i in range(1, 5)]
    assert delays == delays2
    assert is_retryable_exception(OptimisticConflictError())
    assert not is_retryable_exception(StaleGenerationError())
    assert classify_exception(OptimisticConflictError()) is (
        TransactionConflictKind.OPTIMISTIC
    )


def test_result_digest_stable() -> None:
    body = {"task_cid": "task:1", "status": "claimed", "n": 1}
    assert result_digest(body) == result_digest(dict(reversed(list(body.items()))))


def _worker_cas(payload: dict[str, Any], queue: Any) -> None:
    """Multiprocess worker: CAS a distinct task row once."""

    db = Path(payload["db"])
    task_cid = payload["task_cid"]
    owner_id = payload["owner_id"]
    status = payload["status"]
    try:
        client = open_embedded_client(
            db,
            owner_id=owner_id,
            seed_generation=False,
            connect_timeout_seconds=60.0,
            retry_policy=RetryPolicy(
                max_attempts=12,
                base_delay_seconds=0.01,
                max_delay_seconds=0.1,
                jitter_ratio=0.5,
                seed=hash(owner_id) & 0xFFFF,
            ),
        )
        try:
            result = client.cas_task_status(
                task_cid=task_cid,
                expected_task_revision=0,
                new_status=status,
                idempotency_key=f"idem:{owner_id}:{task_cid}",
                command_id=f"cmd:{owner_id}:{task_cid}",
            )
            queue.put(
                {
                    "ok": True,
                    "outcome": result.outcome.value,
                    "changed": result.changed,
                    "task_cid": task_cid,
                    "attempts": result.attempts,
                }
            )
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - surfaced via queue
        queue.put({"ok": False, "error": str(exc), "task_cid": task_cid})


def test_independent_processes_commit_nonconflicting_work(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_goal_and_tasks(db, count=3)

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    processes = []
    for index, task_cid in enumerate(task_cids):
        process = ctx.Process(
            target=_worker_cas,
            args=(
                {
                    "db": str(db),
                    "task_cid": task_cid,
                    "owner_id": f"owner:worker-{index}",
                    "status": "claimed",
                },
                queue,
            ),
        )
        processes.append(process)
        process.start()
    results = [queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    assert all(item.get("ok") for item in results), results
    assert {item["task_cid"] for item in results} == set(task_cids)
    assert all(item["outcome"] == CommandOutcome.ACCEPTED.value for item in results)

    with _client(db) as client:
        for task_cid in task_cids:
            rows = client.execute("select_task_by_cid", {"task_cid": task_cid})
            assert rows[0]["status"] == "claimed"
            assert int(rows[0]["revision"]) == 1
        # Store head advanced once per successful command.
        assert client.load_generation().revision == 3


def test_same_row_conflict_across_processes(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_goal_and_tasks(db, count=1)
    task_cid = task_cids[0]

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    processes = []
    for index in range(2):
        process = ctx.Process(
            target=_worker_cas,
            args=(
                {
                    "db": str(db),
                    "task_cid": task_cid,
                    "owner_id": f"owner:conflict-{index}",
                    "status": f"claimed-{index}",
                },
                queue,
            ),
        )
        processes.append(process)
        process.start()
    results = [queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    # Exactly one process wins the task-revision CAS at expected_revision=0.
    # The loser either surfaces a non-ok error or a conflict outcome depending
    # on retry refresh (domain task revision is not auto-refreshed).
    ok_results = [item for item in results if item.get("ok")]
    assert len(ok_results) >= 1
    accepted = [
        item
        for item in ok_results
        if item.get("outcome") == CommandOutcome.ACCEPTED.value
    ]
    assert len(accepted) == 1

    with _client(db) as client:
        rows = client.execute("select_task_by_cid", {"task_cid": task_cid})
        assert int(rows[0]["revision"]) == 1
        assert str(rows[0]["status"]).startswith("claimed")


def test_resolve_endpoint_and_loopback_policy() -> None:
    embedded = resolve_endpoint("/tmp/control.duckdb")
    assert embedded.mode is TransportMode.EMBEDDED
    quack = resolve_endpoint("quack:127.0.0.1:42100")
    assert quack.mode is TransportMode.QUACK
    # Non-loopback is rejected at attach time for quack transport.
    client = QuackStateClient(owner_id="owner:net")
    with pytest.raises(Exception):
        client.attach("quack:8.8.8.8:42100", mode=TransportMode.QUACK)


def test_open_embedded_client_helper(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    # seed_generation=True should insert generation when missing.
    client = open_embedded_client(db, owner_id="owner:helper", seed_generation=True)
    try:
        assert client.attached
        generation = client.load_generation()
        assert generation.generation >= 1
    finally:
        client.close()


def test_default_retry_policy_respects_bounds() -> None:
    policy = default_retry_policy()
    assert policy.max_attempts >= 1
    assert policy.base_delay_seconds >= 0.0
