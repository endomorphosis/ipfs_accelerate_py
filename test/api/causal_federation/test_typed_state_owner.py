from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_state_owner as typed_owner,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientTransportError,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    STATUS_BOOTSTRAP_ALLOWED_OPERATIONS,
    STATUS_BOOTSTRAP_CLIENT_ID,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    TypedStateOwnerGateway,
    TypedStateOwnerProtocolError,
    TypedStateOwnerRemoteError,
    build_control_plane_operation_catalog,
)


def _install(db: Path) -> None:
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="typed-owner-test",
    )
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:typed-owner",
                "G-TYPED",
                "objective:typed-owner",
                "",
                1,
                "Typed owner",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "task:typed-owner",
                "CASF-TYPED",
                "goal:typed-owner",
                "",
                "objective:typed-owner",
                1,
                "ready",
                0,
                "P0",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                "{}",
                "{}",
            ],
        )
    seed = QuackStateClient(
        owner_id="typed-owner-test:seed",
        store_id="control.duckdb",
    )
    try:
        seed.attach(db, seed_generation=True)
    finally:
        seed.close()


def _gateway(db: Path, socket_path: Path) -> tuple[TypedStateOwnerGateway, Any]:
    connection = open_duckdb_connection(db)
    row = connection.execute(
        """
        SELECT generation, schema_revision, fence_epoch, revision,
               database_uuid, birth_id
        FROM store_generations ORDER BY generation DESC LIMIT 1
        """
    ).fetchone()
    assert row is not None
    identity = {
        "server_id": "server:typed-owner-test",
        "store_id": "control.duckdb",
        "generation": int(row[0]),
        "schema_revision": int(row[1]),
        "fence_epoch": int(row[2]),
        "revision": int(row[3]),
        "database_uuid": str(row[4]),
        "process_birth_id": str(row[5] or "birth:typed-owner-test"),
    }
    gateway = TypedStateOwnerGateway(
        connection=connection,
        socket_path=socket_path,
        store_id="control.duckdb",
        identity=identity,
    )
    gateway.start()
    return gateway, connection


def _read_operations() -> tuple[str, ...]:
    catalog = build_control_plane_operation_catalog()
    return tuple(name for name, operation in catalog.items() if not operation.mutation)


def _task_grant_operations() -> tuple[str, ...]:
    return (
        *_read_operations(),
        "txn_advance_store_revision",
        "txn_record_idempotency",
        "txn_cas_task_status",
    )


def _attached_client(client_id: str) -> QuackStateClient:
    client = QuackStateClient(
        owner_id=client_id,
        store_id="control.duckdb",
    )
    client.attach(
        "quack:127.0.0.1:7777",
        server_id="server:typed-owner-test",
    )
    return client


def test_default_owner_socket_path_compacts_an_overlong_store_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TYPED_STATE_OWNER_SOCKET_ENV, raising=False)
    long_store = tmp_path.joinpath(*("long-segment" for _ in range(12)), "control.duckdb")

    first = typed_owner.typed_owner_socket_path(str(long_store))
    second = typed_owner.typed_owner_socket_path(str(long_store))

    assert first == second
    assert len(os.fsencode(first)) <= 100
    assert first.parent.name == f"ipfs-accelerate-typed-owner-{os.geteuid()}"
    assert first.suffix == ".sock"


def test_explicit_owner_socket_path_is_not_silently_rewritten(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "launcher-owned" / "owner.sock"
    assert typed_owner.typed_owner_socket_path("ignored", str(explicit)) == explicit


def test_closed_owner_command_is_atomic_and_rolls_back_callback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:typed-owner-test",
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id="client:typed-owner-test",
        store_id="control.duckdb",
    )
    try:
        client.attach("quack:127.0.0.1:7777", server_id="server:typed-owner-test")
        accepted = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:typed-owner:accepted",
            command_id="command:typed-owner:accepted",
        )
        assert accepted.changed is True
        before = client.load_generation()
        session = client.session
        assert session is not None
        command = StateCommand(
            command_id="command:typed-owner:rollback",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=before.generation,
            expected_revision=before.revision,
            fence_epoch=before.fence_epoch,
            idempotency_key="idem:typed-owner:rollback",
            parameters={
                "operation": "task.status.cas",
                "task_cid": "task:typed-owner",
                "expected_task_revision": 1,
                "status": "failed",
            },
        )

        def fail_after_state_mutation(transaction: Any, *_args: Any) -> dict[str, Any]:
            transaction.cas_row_revision(
                table="tasks",
                key_column="task_cid",
                key_value="task:typed-owner",
                expected_revision=1,
                assignments={
                    "status": "failed",
                    "updated_at": "1970-01-01T00:00:01Z",
                },
            )
            raise RuntimeError("do not commit")

        with pytest.raises(RuntimeError, match="do not commit"):
            client.submit_command(command, apply=fail_after_state_mutation)
        task = client.execute("select_task_by_cid", {"task_cid": "task:typed-owner"})[0]
        after = client.load_generation()
        assert task["status"] == "claimed"
        assert int(task["revision"]) == 1
        assert after.revision == before.revision
        assert client.execute("lookup_idempotency", ["idem:typed-owner:rollback"]) == ()
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_commit_observer_runs_after_owner_transaction_lock_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    client_id = "client:post-commit-lock"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    observed: list[tuple[bool, str, tuple[str, ...]]] = []

    def observe_commit(command: StateCommand, manifest: tuple[Any, ...]) -> None:
        lock_available = gateway._transaction_lock.acquire(blocking=False)
        if lock_available:
            gateway._transaction_lock.release()
        observed.append(
            (
                lock_available,
                str(command.parameters.get("operation") or ""),
                tuple(str(item[0]) for item in manifest),
            )
        )

    gateway.bind_commit_observer(observe_commit)
    client = _attached_client(client_id)
    try:
        accepted = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:post-commit-lock",
            command_id="command:post-commit-lock",
        )
        assert accepted.changed is True
        assert observed == [
            (
                True,
                "task.status.cas",
                (
                    "txn_cas_task_status",
                    "txn_advance_store_revision",
                    "txn_record_idempotency",
                ),
            )
        ]
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_blocked_downstream_observer_does_not_block_later_owner_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    client_id = "client:blocked-downstream"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    first_observer_entered = threading.Event()
    release_first_observer = threading.Event()
    observer_guard = threading.Lock()
    observer_lock_available: list[bool] = []
    observer_calls = 0

    def blocked_downstream_observer(
        _command: StateCommand,
        _manifest: tuple[Any, ...],
    ) -> None:
        nonlocal observer_calls
        lock_available = gateway._transaction_lock.acquire(blocking=False)
        if lock_available:
            gateway._transaction_lock.release()
        with observer_guard:
            observer_calls += 1
            call_number = observer_calls
            observer_lock_available.append(lock_available)
        if call_number == 1:
            first_observer_entered.set()
            release_first_observer.wait(timeout=10.0)

    gateway.bind_commit_observer(blocked_downstream_observer)
    first_client = _attached_client(client_id)
    second_client = _attached_client(client_id)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                first_client.cas_task_status,
                task_cid="task:typed-owner",
                expected_task_revision=0,
                new_status="claimed",
                idempotency_key="idem:blocked-downstream:first",
                command_id="command:blocked-downstream:first",
            )
            assert first_observer_entered.wait(timeout=5.0)
            try:
                second_call = executor.submit(
                    second_client.cas_task_status,
                    task_cid="task:typed-owner",
                    expected_task_revision=1,
                    new_status="running",
                    idempotency_key="idem:blocked-downstream:second",
                    command_id="command:blocked-downstream:second",
                )
                second = second_call.result(timeout=5.0)
                assert second.changed is True
                assert first.done() is False
            finally:
                release_first_observer.set()
            assert first.result(timeout=5.0).changed is True
        assert observer_calls == 2
        assert observer_lock_available == [True, True]
        task = second_client.execute(
            "select_task_by_cid",
            {"task_cid": "task:typed-owner"},
        )[0]
        assert task["status"] == "running"
        assert int(task["revision"]) == 2
    finally:
        release_first_observer.set()
        first_client.close()
        second_client.close()
        gateway.stop()
        connection.close()


def test_failing_downstream_observer_cannot_undo_or_disable_owner_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    client_id = "client:failing-downstream"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    observer_lock_available: list[bool] = []

    def failing_downstream_observer(
        _command: StateCommand,
        _manifest: tuple[Any, ...],
    ) -> None:
        lock_available = gateway._transaction_lock.acquire(blocking=False)
        if lock_available:
            gateway._transaction_lock.release()
        observer_lock_available.append(lock_available)
        raise RuntimeError("downstream DuckLake-style observer unavailable")

    gateway.bind_commit_observer(failing_downstream_observer)
    client = _attached_client(client_id)
    try:
        first = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:failing-downstream:first",
            command_id="command:failing-downstream:first",
        )
        second = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=1,
            new_status="running",
            idempotency_key="idem:failing-downstream:second",
            command_id="command:failing-downstream:second",
        )
        assert first.changed is True
        assert second.changed is True
        assert observer_lock_available == [True, True]
        assert gateway.capability()["last_observer_error_type"] == "RuntimeError"
        task = client.execute(
            "select_task_by_cid",
            {"task_cid": "task:typed-owner"},
        )[0]
        assert task["status"] == "running"
        assert int(task["revision"]) == 2
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_grant_rejects_self_admission_unknown_mutation_and_scope_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:bounded",
        allowed_operations=("whoami_metadata", "load_store_generation"),
        peer_pid=os.getpid(),
    )
    scoped_token, _scoped_grant = gateway.issue_grant(
        client_id="client:scoped",
        allowed_operations=("casf_select_federation",),
        peer_pid=os.getpid(),
        tenant_id="tenant:one",
        federation_id="federation:one",
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    try:
        wrong = QuackStateClient(owner_id="client:self-promoted")
        with pytest.raises(QuackClientTransportError):
            wrong.attach("quack:127.0.0.1:7777")
        raw = TypedStateOwnerConnection(
            socket_path=socket_path,
            token=token,
            client_id="client:bounded",
            process_birth_id="caller-asserted-and-nonauthoritative",
            store_id="control.duckdb",
        )
        try:
            assert raw.execute_operation("whoami_metadata").fetchone() is not None
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                raw.execute_operation(
                    "seed_client_session",
                    [
                        "session:forged",
                        "server:forged",
                        "client:admin",
                        "birth:forged",
                        "1970-01-01T00:00:00Z",
                        "1970-01-01T00:00:00Z",
                        1,
                        1,
                        "attached",
                        0,
                    ],
                )
            assert denied.value.error_code == "authorization_denied"
        finally:
            raw.close()
        scoped = TypedStateOwnerConnection(
            socket_path=socket_path,
            token=scoped_token,
            client_id="client:scoped",
            process_birth_id="birth:scoped",
            store_id="control.duckdb",
        )
        try:
            assert scoped.execute_operation(
                "casf_select_federation",
                ["federation:one", "tenant:one"],
            ).fetchone() is None
            with pytest.raises(TypedStateOwnerRemoteError) as wrong_scope:
                scoped.execute_operation(
                    "casf_select_federation",
                    ["federation:two", "tenant:two"],
                )
            assert wrong_scope.value.error_code == "authorization_denied"
        finally:
            scoped.close()
    finally:
        gateway.stop()
        connection.close()


def test_live_grant_revocation_is_checked_on_every_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, grant = gateway.issue_grant(
        client_id="client:revocable",
        allowed_operations=("whoami_metadata", "load_store_generation"),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(owner_id="client:revocable")
    try:
        client.attach("quack:127.0.0.1:7777")
        assert client.execute("whoami_metadata")
        gateway.revoke_grant(grant.grant_id)
        with pytest.raises(QuackClientTransportError) as denied:
            client.execute("whoami_metadata")
        assert isinstance(denied.value.__cause__, TypedStateOwnerRemoteError)
        assert denied.value.__cause__.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_transaction_idempotency_lookup_cannot_cross_admitted_command_key(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    with open_duckdb_connection(db) as seed:
        seed.execute(
            """
            INSERT INTO idempotency_records (
                idempotency_key, command_kind, command_id, store_id,
                session_id, result_digest, created_at, expires_at, body_json,
                tenant_id, federation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "idem:tenant-b:secret",
                "claim",
                "command:tenant-b",
                "control.duckdb",
                "session:tenant-b",
                "digest:tenant-b",
                "1970-01-01T00:00:00Z",
                "9999-12-31T23:59:59Z",
                '{"private":"tenant-b"}',
                "tenant:b",
                "federation:b",
            ],
        )
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:tenant-a-child",
        allowed_operations=(
            "load_store_generation",
            "txn_load_generation",
            "txn_lookup_idempotency",
        ),
        allowed_command_operations=("task.status.cas",),
        tenant_id="tenant:a",
        federation_id="federation:a",
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:tenant-a-child",
        process_birth_id="birth:tenant-a-child",
        store_id="control.duckdb",
    )
    try:
        generation = client.execute_operation("load_store_generation").fetchone()
        assert generation is not None
        command = StateCommand(
            command_id="command:tenant-a",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=client.session_id,
            expected_generation=int(generation[0]),
            expected_revision=int(generation[3]),
            fence_epoch=int(generation[2]),
            idempotency_key="idem:tenant-a:admitted",
            parameters={
                "operation": "task.status.cas",
                "task_cid": "task:typed-owner",
                "expected_task_revision": 0,
                "status": "claimed",
                "tenant_id": "tenant:a",
                "federation_id": "federation:a",
            },
        )
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        assert client.execute_operation(
            "txn_lookup_idempotency", [command.idempotency_key]
        ).fetchone() is None
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation(
                "txn_lookup_idempotency", ["idem:tenant-b:secret"]
            )
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        connection.close()
    with open_duckdb_connection(db) as check:
        row = check.execute(
            "SELECT body_json FROM idempotency_records WHERE idempotency_key = ?",
            ["idem:tenant-b:secret"],
        ).fetchone()
        assert row is not None
        assert row["body_json"] == '{"private":"tenant-b"}'


def test_live_grant_expiry_is_checked_on_every_request(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:expiring",
        allowed_operations=("whoami_metadata",),
        peer_pid=os.getpid(),
        ttl_seconds=1.0,
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:expiring",
        process_birth_id="birth:expiring",
        store_id="control.duckdb",
    )
    try:
        assert client.execute_operation("whoami_metadata").fetchone() is not None
        time.sleep(1.05)
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation("whoami_metadata")
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        connection.close()


def _independent_reader(
    receive: Any,
    result: Any,
    socket_path: str,
) -> None:
    token = receive.recv()
    try:
        connection = TypedStateOwnerConnection(
            socket_path=Path(socket_path),
            token=token,
            client_id="client:independent",
            process_birth_id="birth:independent",
            store_id="control.duckdb",
        )
        try:
            result.send(connection.execute_operation("whoami_metadata").fetchone())
        finally:
            connection.close()
    except BaseException as exc:  # pragma: no cover - parent reports exact type
        result.send(("error", type(exc).__name__))


def _independent_status_reader(
    receive: Any,
    result: Any,
    socket_path: str,
) -> None:
    token = receive.recv()
    try:
        connection = TypedStateOwnerConnection(
            socket_path=Path(socket_path),
            token=token,
            client_id=STATUS_BOOTSTRAP_CLIENT_ID,
            process_birth_id=f"birth:status:{os.getpid()}",
            store_id="control.duckdb",
            status_bootstrap=True,
        )
        try:
            count = connection.execute_operation("count_tasks").fetchone()
            denied: list[str] = []
            for operation, parameters in (
                ("casf_select_federation", ["federation:forged", "tenant:forged"]),
                (
                    "casf_select_supervisor_bootstrap_health",
                    [
                        "tenant:other",
                        "federation:other",
                        "supervisor:other",
                        "subscription:other",
                        "consumer:other",
                        "event:other",
                        "acknowledgement:other",
                        "attempt:other",
                        "9999-12-31T23:59:59Z",
                    ],
                ),
                (
                    "seed_client_session",
                    [
                        "session:forged",
                        "server:forged",
                        "client:forged",
                        "birth:forged",
                        "1970-01-01T00:00:00Z",
                        "1970-01-01T00:00:00Z",
                        1,
                        1,
                        "attached",
                        0,
                    ],
                ),
            ):
                try:
                    connection.execute_operation(operation, parameters)
                except TypedStateOwnerRemoteError as exc:
                    denied.append(exc.error_code)
            raw_sql_denied = False
            try:
                connection.execute("SELECT 1")
            except TypedStateOwnerProtocolError:
                raw_sql_denied = True
            result.send(
                {
                    "count": count,
                    "grant": dict(connection.grant),
                    "denied": denied,
                    "raw_sql_denied": raw_sql_denied,
                }
            )
        finally:
            connection.close()
    except BaseException as exc:  # pragma: no cover - parent reports exact type
        result.send({"error": type(exc).__name__})


def test_grant_is_kernel_bound_to_an_independent_process(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    context = multiprocessing.get_context("spawn")
    token_receive, token_send = context.Pipe(duplex=False)
    result_receive, result_send = context.Pipe(duplex=False)
    process = context.Process(
        target=_independent_reader,
        args=(token_receive, result_send, str(socket_path)),
    )
    process.start()
    try:
        token, _grant = gateway.issue_grant(
            client_id="client:independent",
            allowed_operations=("whoami_metadata",),
            peer_pid=process.pid,
        )
        token_send.send(token)
        assert result_receive.poll(10.0)
        assert result_receive.recv() is not None
    finally:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        gateway.stop()
        connection.close()
    assert process.exitcode == 0


def test_status_bootstrap_rebinds_each_distinct_process_read_only(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    connection.execute(
        """
        INSERT INTO federations VALUES (
            'federation:status', 'tenant:status', 'program:status',
            'objective:status', 1, 'policy:status', 1, 'catalog:status',
            1, 1, 'semantic:status', 'admitted', 1, 1, 1, 1,
            'issuer:status', 'evidence:status', '9999-12-31T23:59:59Z',
            '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z',
            'content:status', '{}'
        )
        """
    )
    connection.execute(
        """
        INSERT INTO supervisor_instances (
            supervisor_id, repository_id, process_birth_id, started_at,
            status, revision, tenant_id, federation_id, lifecycle_state
        ) VALUES (
            'supervisor:status', 'repository:status', 'logical:not-started',
            '1970-01-01T00:00:00Z', 'ADMITTED', 1,
            'tenant:status', 'federation:status', 'ADMITTED'
        )
        """
    )
    connection.execute(
        """
        INSERT INTO event_subscriptions (
            subscription_id, tenant_id, federation_id, consumer_id,
            supervisor_id, revision, maximum_batch, maximum_pending,
            maximum_fanout, retry_budget, status, created_at, updated_at
        ) VALUES (
            'subscription:status', 'tenant:status', 'federation:status',
            'consumer:status', 'supervisor:status', 1, 1, 1, 1, 1,
            'active', '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z'
        )
        """
    )
    bootstrap_token = gateway.configure_status_bootstrap()
    gateway.bind_status_bootstrap_scope()
    context = multiprocessing.get_context("spawn")
    observed: list[dict[str, Any]] = []
    try:
        for _index in range(2):
            token_receive, token_send = context.Pipe(duplex=False)
            result_receive, result_send = context.Pipe(duplex=False)
            process = context.Process(
                target=_independent_status_reader,
                args=(token_receive, result_send, str(socket_path)),
            )
            process.start()
            token_send.send(bootstrap_token)
            assert result_receive.poll(10.0)
            payload = result_receive.recv()
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            assert process.exitcode == 0
            assert payload.get("error") is None
            assert payload["count"] == (1,)
            assert payload["denied"] == [
                "authorization_denied",
                "authorization_denied",
                "authorization_denied",
            ]
            assert payload["raw_sql_denied"] is True
            grant = payload["grant"]
            assert grant["client_id"] == STATUS_BOOTSTRAP_CLIENT_ID
            assert grant["peer_pid"] == process.pid
            assert grant["peer_uid"] == os.getuid()
            assert set(grant["allowed_operations"]) == set(
                STATUS_BOOTSTRAP_ALLOWED_OPERATIONS
            )
            assert grant["allowed_command_operations"] == []
            assert grant["tenant_id"] == "tenant:status"
            assert grant["federation_id"] == "federation:status"
            assert grant["entity_scopes"] == {
                "consumer_id": "consumer:status",
                "subscription_id": "subscription:status",
                "supervisor_id": "supervisor:status",
            }
            assert grant["authority_profile"] == "dedicated_store_status_portfolio"
            assert 0 < int(grant["expires_at"]) - int(grant["issued_at"]) <= 60_000
            observed.append(payload)

        assert observed[0]["grant"]["peer_pid"] != observed[1]["grant"]["peer_pid"]
        # Connection-local grants are retired instead of accumulating as an
        # ambient same-UID capability after each status process exits.
        deadline = time.monotonic() + 2.0
        while gateway.capability()["active_grants"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gateway.capability()["active_grants"] == 0

        connection.execute(
            """
            INSERT INTO federations VALUES (
                'federation:other', 'tenant:other', 'program:other',
                'objective:other', 1, 'policy:other', 1, 'catalog:other',
                1, 1, 'semantic:other', 'admitted', 1, 1, 1, 1,
                'issuer:other', 'evidence:other', '9999-12-31T23:59:59Z',
                '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z',
                'content:other', '{}'
            )
            """
        )
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=socket_path,
                token=bootstrap_token,
                client_id=STATUS_BOOTSTRAP_CLIENT_ID,
                process_birth_id="birth:multiple-federations",
                store_id="control.duckdb",
                status_bootstrap=True,
            )
        connection.execute(
            "DELETE FROM federations WHERE federation_id = 'federation:other'"
        )

        for token, client_id, store_id in (
            ("0" * 64, STATUS_BOOTSTRAP_CLIENT_ID, "control.duckdb"),
            (bootstrap_token, "client:forged", "control.duckdb"),
            (bootstrap_token, STATUS_BOOTSTRAP_CLIENT_ID, "other.duckdb"),
        ):
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=socket_path,
                    token=token,
                    client_id=client_id,
                    process_birth_id="birth:rejected-status",
                    store_id=store_id,
                    status_bootstrap=True,
                )

        original_uid = gateway._status_bootstrap_uid
        gateway._status_bootstrap_uid = original_uid + 1
        try:
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=socket_path,
                    token=bootstrap_token,
                    client_id=STATUS_BOOTSTRAP_CLIENT_ID,
                    process_birth_id="birth:wrong-uid",
                    store_id="control.duckdb",
                    status_bootstrap=True,
                )
        finally:
            gateway._status_bootstrap_uid = original_uid
    finally:
        gateway.stop()
        connection.close()


def test_budget_policy_and_usage_query_are_tenant_scoped(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, grant = gateway.issue_grant(
        client_id="client:budget",
        allowed_operations=("casf_select_active_admission_budget_usage",),
        allowed_command_operations=(
            "budget.reserve",
            "budget.release",
            "federation.create",
        ),
        tenant_id="tenant:one",
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:budget",
        process_birth_id="birth:budget",
        store_id="control.duckdb",
    )
    try:
        assert client.execute_operation(
            "casf_select_active_admission_budget_usage",
            ["tenant:one", "9999-12-31T23:59:59Z"],
        ).fetchall() == []
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation(
                "casf_select_active_admission_budget_usage",
                ["tenant:two", "9999-12-31T23:59:59Z"],
            )
        assert denied.value.error_code == "authorization_denied"
        assert grant.federation_id == ""
        assert grant.allowed_command_operations == frozenset(
            {"budget.reserve", "budget.release", "federation.create"}
        )
        policies = typed_owner._COMMAND_MUTATION_CATALOG
        assert {
            "casf_insert_admission_budget_reservation",
            "casf_insert_admission_budget_dimension",
        }.issubset(policies["budget.reserve"])
        assert "casf_transition_admission_budget_reservation" in policies[
            "budget.release"
        ]
        assert "casf_transition_admission_budget_reservation" in policies[
            "federation.create"
        ]
    finally:
        client.close()
        gateway.stop()
        connection.close()


@pytest.mark.parametrize(
    ("operation", "include_partial_state"),
    (("federation.create", True), ("supervisor.register", False)),
)
def test_partial_federation_commands_cannot_commit(
    tmp_path: Path,
    operation: str,
    include_partial_state: bool,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, owner_connection = _gateway(db, socket_path)
    allowed = [
        "load_store_generation",
        "txn_advance_store_revision",
        "txn_record_idempotency",
    ]
    if include_partial_state:
        allowed.append("casf_insert_federation")
    token, _grant = gateway.issue_grant(
        client_id=f"client:partial:{operation}",
        allowed_operations=tuple(allowed),
        allowed_command_operations=(operation,),
        tenant_id="tenant:one",
        federation_id="federation:one",
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=f"client:partial:{operation}",
        process_birth_id=f"birth:partial:{operation}",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    command = StateCommand(
        command_id=f"command:partial:{operation}",
        command_kind=CommandKind.APPEND,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key=f"idempotency:partial:{operation}",
        parameters={
            "operation": operation,
            "tenant_id": "tenant:one",
            "federation_id": "federation:one",
        },
    )
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        if include_partial_state:
            client.execute_operation(
                "casf_insert_federation",
                [
                    "federation:one",
                    "tenant:one",
                    "program:one",
                    "objective:one",
                    1,
                    "policy:one",
                    1,
                    "catalog:one",
                    int(head[0]),
                    1,
                    "semantic:one",
                    "admitted",
                    12,
                    256,
                    1,
                    int(head[2]),
                    "issuer:one",
                    "evidence:one",
                    "9999-12-31T23:59:59Z",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "content:one",
                    "{}",
                ],
            )
        client.execute_operation(
            "txn_advance_store_revision",
            [int(head[3]) + 1, int(head[0]), int(head[3]), int(head[2])],
        )
        client.execute_operation(
            "txn_record_idempotency",
            [
                command.idempotency_key,
                command.command_kind.value,
                command.command_id,
                command.store_id,
                command.session_id,
                "sha256:partial",
                "1970-01-01T00:00:00Z",
                None,
                "{}",
            ],
        )
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.commit()
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
    assert owner_connection.execute(
        "SELECT revision FROM store_generations ORDER BY generation DESC LIMIT 1"
    ).fetchone()[0] == int(head[3])
    assert owner_connection.execute(
        "SELECT COUNT(*) FROM idempotency_records WHERE idempotency_key = ?",
        [command.idempotency_key],
    ).fetchone()[0] == 0
    assert owner_connection.execute(
        "SELECT COUNT(*) FROM federations WHERE federation_id = 'federation:one'"
    ).fetchone()[0] == 0
    gateway.stop()
    owner_connection.close()


@pytest.mark.parametrize("attack", ("duplicate", "out_of_order"))
def test_manifest_rejects_duplicate_and_out_of_order_mutations(
    tmp_path: Path,
    attack: str,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, owner_connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id=f"client:manifest:{attack}",
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=f"client:manifest:{attack}",
        process_birth_id=f"birth:manifest:{attack}",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    command = StateCommand(
        command_id=f"command:manifest:{attack}",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key=f"idempotency:manifest:{attack}",
        parameters={
            "operation": "task.status.cas",
            "task_cid": "task:typed-owner",
            "expected_task_revision": 0,
            "status": "claimed",
        },
    )
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        if attack == "duplicate":
            parameters = [
                "claimed",
                "1970-01-01T00:00:01Z",
                1,
                "task:typed-owner",
                0,
            ]
            client.execute_operation("txn_cas_task_status", parameters)
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.execute_operation("txn_cas_task_status", parameters)
        else:
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.execute_operation(
                    "txn_record_idempotency",
                    [
                        command.idempotency_key,
                        command.command_kind.value,
                        command.command_id,
                        command.store_id,
                        command.session_id,
                        "sha256:out-of-order",
                        "1970-01-01T00:00:00Z",
                        None,
                        "{}",
                    ],
                )
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
    task = owner_connection.execute(
        "SELECT status, revision FROM tasks WHERE task_cid = 'task:typed-owner'"
    ).fetchone()
    assert task is not None
    assert (task[0], task[1]) == ("ready", 0)
    gateway.stop()
    owner_connection.close()
