"""The actual legacy queue through an admitted owner, never a second writer."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.merge import merge_queue
from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue import (
    OwnerMergeQueueClient,
    OwnerMergeQueueError,
    SERVICE_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)


def enqueue(api, task="task:one", **overrides):
    return api.call(
        "enqueue",
        **{
            "branch_name": "work/one",
            "task_id": task,
            "priority": "P1",
            "lane_id": "lane:one",
            "commit_sha": "a" * 40,
            "canonical_task_id": task,
            "canonical_task_key": "",
            "metadata_json": '{"score": 0.25}',
            **overrides,
        },
    )


def request(result):
    return (
        json.loads(result["request_json"])
        if result["request_json"] is not None
        else None
    )


def claim_args(claim):
    return {
        key: claim[key] for key in ("request_id", "claim_token", "claim_generation")
    }


@pytest.fixture
def owner(tmp_path):
    queue_dir = tmp_path / "queue"
    legacy = merge_queue.MergeQueue(queue_dir, max_age_seconds=3600)
    unbound = legacy.enqueue(
        branch_name="legacy/unbound", task_id="task:unbound", commit_sha="b" * 40
    )
    other = legacy.enqueue(
        branch_name="other/work",
        task_id="task:other",
        commit_sha="c" * 40,
        target_repository_id="repo:other",
        target_branch="main",
    )
    legacy.bind_target("repo:one", "main", required=True)
    retained = legacy.enqueue(
        branch_name="legacy/retained", task_id="task:retained", commit_sha="d" * 40
    )
    retained = legacy.claim_pending_request(retained, consumer_id="consumer:old")
    db = queue_dir / "merge_queue.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    connection.execute(
        "UPDATE store_generations SET birth_id=?",
        [gateway.identity["process_birth_id"]],
    )
    original = [
        tuple(row[index] for index in range(len(row)))
        for row in connection.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    files = {
        str(path.relative_to(queue_dir)): path.read_bytes()
        for path in queue_dir.rglob("*.json")
    }

    def bind(**overrides):
        gateway.bind_legacy_merge_queue_service(
            **{
                "expected_identity": dict(gateway.identity),
                "repository_id": "repo:one",
                "target_branch": "main",
                "max_age_seconds": 3600,
                "max_queue_size": 100,
                "max_processing": 10,
                "max_attempts": 3,
                "max_worktree_bytes": None,
                **overrides,
            }
        )

    clients = []

    def attach(*, consumer="consumer:one", scopes=None, operations=SERVICE_OPERATIONS):
        scopes = (
            scopes
            if scopes is not None
            else {
                "repository_id": "repo:one",
                "target_branch": "main",
                "consumer_id": consumer,
            }
        )
        token, grant = gateway.issue_grant(
            client_id=consumer,
            process_birth_id="birth:" + consumer,
            allowed_operations=tuple(operations),
            entity_scopes=scopes,
        )
        client = TypedStateOwnerConnection(
            socket_path=gateway.socket_path,
            token=token,
            client_id=consumer,
            process_birth_id="birth:" + consumer,
            store_id="control.duckdb",
        )
        clients.append(client)
        return OwnerMergeQueueClient(
            client, repository_id="repo:one", target_branch="main", consumer_id=consumer
        ), grant

    yield (
        gateway,
        connection,
        bind,
        attach,
        (unbound, other, retained),
        original,
        files,
        queue_dir,
    )
    for client in clients:
        client.close()
    gateway.stop()
    connection.close()


def test_binds_existing_rows_without_import_open_or_projections(owner, monkeypatch):
    gateway, conn, bind, attach, rows, original, files, queue_dir = owner
    monkeypatch.setattr(
        merge_queue,
        "open_duckdb_connection",
        lambda *a, **k: pytest.fail("second file open"),
    )
    monkeypatch.setattr(
        merge_queue.MergeQueue,
        "__init__",
        lambda *a, **k: pytest.fail("constructor/import"),
    )
    bind()
    assert not gateway._grants  # binding does not bootstrap admission
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    api, _ = attach()
    assert (
        request(api.call("get", request_id=rows[2].request_id))["claim_token"]
        == rows[2].claim_token
    )
    added = request(enqueue(api))
    assert added["metadata"]["score"] == 0.25
    assert request(enqueue(api))["request_id"] == added["request_id"]
    claimed = request(api.call("claim", request_id=added["request_id"]))
    assert claimed["claim_generation"] == 1
    assert claimed["consumer_id"] == "consumer:one"
    assert api.call("owns_claim", **claim_args(claimed))["owns_claim"] is True
    completed = api.call(
        "complete", **claim_args(claimed), metadata_json='{"proof": "observational"}'
    )
    assert request(completed)["status"] == "completed"
    assert request(completed)["claim_generation"] == 2
    assert completed["completion_authority"] is False
    assert (
        request(api.call("complete", **claim_args(claimed), metadata_json="{}"))[
            "status"
        ]
        == "completed"
    )
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests WHERE task_id != 'task:one' ORDER BY request_id"
        ).fetchall()
    ]
    assert files == {
        str(path.relative_to(queue_dir)): path.read_bytes()
        for path in queue_dir.rglob("*.json")
    }
    assert conn.execute("SELECT 1").fetchone()[0] == 1
    assert not conn.in_transaction


def test_retries_preserve_claim_fences_and_terminal_rows(owner):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    added = request(enqueue(api))
    claimed = request(api.call("claim", request_id=added["request_id"]))
    retried = request(
        api.call(
            "requeue", **claim_args(claimed), reason="transient", metadata_json="{}"
        )
    )
    assert (retried["status"], retried["attempt"], retried["claim_generation"]) == (
        "pending",
        2,
        2,
    )
    with pytest.raises(TypedStateOwnerError):
        api.call("complete", **claim_args(claimed), metadata_json="{}")
    second = request(api.call("claim", request_id=added["request_id"]))
    assert (
        second["claim_generation"] == 3
        and second["claim_token"] != claimed["claim_token"]
    )
    with pytest.raises(TypedStateOwnerError):
        api.call("complete", **claim_args(claimed), metadata_json="{}")
    quarantined = request(
        api.call(
            "quarantine",
            **claim_args(second),
            reason="evidence missing",
            metadata_json="{}",
        )
    )
    assert (
        quarantined["status"] == "quarantined" and quarantined["claim_generation"] == 4
    )
    assert conn.execute("SELECT count(*) FROM merge_requests").fetchone()[0] == 4


@pytest.mark.parametrize(
    "violation",
    [
        "repository",
        "target",
        "consumer",
        "operation",
        "extra_scope",
        "generation",
        "store",
        "fence",
        "owner_birth",
        "sql",
        "expired",
        "revoked",
    ],
)
def test_closed_owner_scope_and_lease_fail_without_queue_mutation(owner, violation):
    gateway, conn, bind, attach, rows, original, *_ = owner
    bind()
    scopes = {
        "repository_id": "repo:one",
        "target_branch": "main",
        "consumer_id": "consumer:one",
    }
    if violation in {"repository", "target", "consumer"}:
        scopes[
            {
                "repository": "repository_id",
                "target": "target_branch",
                "consumer": "consumer_id",
            }[violation]
        ] = "other"
    if violation == "extra_scope":
        scopes["task_id"] = "task:unrelated"
    api, grant = attach(
        scopes=scopes, operations=() if violation == "operation" else SERVICE_OPERATIONS
    )
    if violation == "expired":
        with gateway._grants_lock:
            for token in tuple(gateway._grants):
                gateway._grants[token] = replace(grant, issued_at=1, expires_at=1001)
    if violation == "revoked":
        gateway.revoke_grant(grant.grant_id)
    if violation in {"generation", "store", "fence", "owner_birth"}:
        api.connection.identity = dict(api.connection.identity)
        key = {
            "generation": "generation",
            "store": "store_id",
            "fence": "fence_epoch",
            "owner_birth": "process_birth_id",
        }[violation]
        api.connection.identity[key] = (
            999 if key in {"generation", "fence_epoch"} else "other"
        )
    with pytest.raises(TypedStateOwnerError):
        api.call(
            "execute_sql" if violation == "sql" else "get",
            request_id=rows[2].request_id,
        )
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    assert not conn.in_transaction


@pytest.mark.parametrize("index", [0, 1])
def test_unbound_and_other_target_rows_cannot_be_read_or_claimed(owner, index):
    _, conn, bind, attach, rows, original, *_ = owner
    bind()
    api, _ = attach()
    for operation in ("get", "claim"):
        with pytest.raises(TypedStateOwnerError):
            api.call(operation, request_id=rows[index].request_id)
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]


def test_expired_claim_denied_without_reaping_other_consumers(owner):
    gateway, conn, bind, attach, rows, *_ = owner
    bind()
    api, _ = attach(consumer="consumer:old")
    service = gateway._legacy_merge_queue_service
    service._queue._clock = lambda: rows[2].claimed_at + 7200
    assert (
        api.call("owns_claim", **claim_args(rows[2].to_dict()))["owns_claim"] is False
    )
    with pytest.raises(TypedStateOwnerError):
        api.call("complete", **claim_args(rows[2].to_dict()), metadata_json="{}")
    added = request(enqueue(api))
    api.call("claim", request_id=added["request_id"])
    observed = conn.execute(
        "SELECT status, claim_token, claim_generation FROM merge_requests WHERE request_id=?",
        [rows[2].request_id],
    ).fetchone()
    assert tuple(observed[index] for index in range(3)) == (
        "processing",
        rows[2].claim_token,
        rows[2].claim_generation,
    )


@pytest.mark.parametrize(
    "violation",
    ["missing", "column", "view", "index", "generation", "stopped", "identity"],
)
def test_binding_requires_exact_migrated_schema_and_live_identity(owner, violation):
    gateway, conn, bind, _, *_ = owner
    if violation == "missing":
        conn.execute("DROP TABLE merge_requests")
    elif violation == "column":
        conn.execute("ALTER TABLE merge_requests ADD COLUMN unauthorized INTEGER")
    elif violation == "view":
        conn.execute("CREATE TABLE preserved AS SELECT * FROM merge_requests")
        conn.execute("DROP TABLE merge_requests")
        conn.execute("CREATE VIEW merge_requests AS SELECT * FROM preserved")
    elif violation == "index":
        conn.execute("DROP INDEX merge_requests_dedupe")
    elif violation == "generation":
        conn.execute("UPDATE store_generations SET fence_epoch=fence_epoch+1")
    elif violation == "stopped":
        gateway.stop()
    before = [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
    ]
    with pytest.raises((OwnerMergeQueueError, TypedStateOwnerError)):
        bind(
            **(
                {"expected_identity": {**gateway.identity, "database_uuid": "other"}}
                if violation == "identity"
                else {}
            )
        )
    assert gateway._legacy_merge_queue_service is None
    assert before == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
    ]
    assert not conn.in_transaction
    assert conn.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.parametrize("failure", ["begin", "sql", "commit", "rollback"])
def test_failure_never_acknowledges_or_closes_borrowed_owner_handle(
    owner, monkeypatch, failure
):
    gateway, conn, bind, attach, rows, original, *_ = owner
    bind()
    api, _ = attach()
    real_execute, real_rollback = conn._execute_once, conn.rollback
    rollback_calls = []

    def execute(sql, parameters=None):
        if failure == "commit" and sql.strip().upper() == "COMMIT":
            raise RuntimeError("injected commit failure")
        if failure == "begin" and sql.strip().upper() == "BEGIN IMMEDIATE":
            raise RuntimeError("injected begin failure")
        if (
            failure in {"sql", "rollback"}
            and "SELECT * FROM merge_requests WHERE request_id = ?" in sql
        ):
            raise RuntimeError("injected post-insert failure")
        return real_execute(sql, parameters)

    def rollback():
        rollback_calls.append(True)
        real_rollback()
        if failure == "rollback":
            raise RuntimeError("injected rollback failure")

    monkeypatch.setattr(conn, "_execute_once", execute)
    monkeypatch.setattr(conn, "rollback", rollback)
    with pytest.raises(TypedStateOwnerError):
        enqueue(api)
    assert original == [
        tuple(row)
        for row in conn._connection.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    assert not conn.in_transaction
    if failure in {"commit", "rollback"}:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            DuckDBOwnerRetiredError,
        )

        with pytest.raises(DuckDBOwnerRetiredError):
            conn.execute("SELECT 1")
    else:
        assert real_execute("SELECT 1", None).fetchone()[0] == 1
    assert gateway._legacy_merge_queue_service._retired is (
        failure in {"commit", "rollback"}
    )
    assert len(rollback_calls) == (0 if failure == "begin" else 1)


def test_inherited_transaction_is_not_committed_or_rolled_back(owner):
    _, conn, bind, attach, rows, original, *_ = owner
    bind()
    api, _ = attach()
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            "UPDATE merge_requests SET failure_reason='uncommitted caller change' WHERE request_id=?",
            [rows[2].request_id],
        )
        with pytest.raises(TypedStateOwnerError):
            api.call("get", request_id=rows[2].request_id)
        assert conn.in_transaction
        assert (
            conn.execute(
                "SELECT failure_reason FROM merge_requests WHERE request_id=?",
                [rows[2].request_id],
            ).fetchone()[0]
            == "uncommitted caller change"
        )
    finally:
        conn.rollback()
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]


@pytest.mark.parametrize(
    "violation",
    [
        "generation",
        "fence",
        "uuid",
        "birth",
        "session_status",
        "session_generation",
        "session_birth",
        "target_metadata",
        "wrong_consumer",
    ],
)
def test_current_store_session_and_claim_coordinates_cannot_drift(owner, violation):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    added = request(enqueue(api))
    claimed = request(api.call("claim", request_id=added["request_id"]))
    if violation in {"generation", "fence", "uuid", "birth"}:
        statement = {
            "generation": "generation=generation+1",
            "fence": "fence_epoch=fence_epoch+1",
            "uuid": "database_uuid='other'",
            "birth": "birth_id='other'",
        }[violation]
        conn.execute("UPDATE store_generations SET " + statement)
    if violation.startswith("session_"):
        statement = {
            "session_status": "status='detached'",
            "session_generation": "generation=generation+1",
            "session_birth": "process_birth_id='other'",
        }[violation]
        conn.execute(
            "UPDATE client_sessions SET " + statement + " WHERE session_id=?",
            [api.connection.session_id],
        )
    if violation == "wrong_consumer":
        api, _ = attach(consumer="consumer:other")
    before = [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    with pytest.raises(TypedStateOwnerError):
        api.call(
            "complete",
            **claim_args(claimed),
            metadata_json='{"target_branch": "other"}'
            if violation == "target_metadata"
            else "{}",
        )
    assert before == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    assert not conn.in_transaction


@pytest.mark.parametrize("denial", ["revoke", "expire", "detach"])
def test_admission_lost_after_insert_rolls_back_before_commit(
    owner, monkeypatch, denial
):
    gateway, conn, bind, attach, _, original, *_ = owner
    bind()
    api, grant = attach()
    real_execute = conn._execute_once
    commits = []

    def execute(sql, parameters=None):
        result = real_execute(sql, parameters)
        if sql.strip().upper().startswith("INSERT INTO MERGE_REQUESTS"):
            if denial == "revoke":
                gateway.revoke_grant(grant.grant_id)
            elif denial == "expire":
                with gateway._grants_lock:
                    for token in tuple(gateway._grants):
                        gateway._grants[token] = replace(
                            grant, issued_at=1, expires_at=1001
                        )
            else:
                real_execute(
                    "UPDATE client_sessions SET status='detached' WHERE session_id=?",
                    [api.connection.session_id],
                )
        if sql.strip().upper() == "COMMIT":
            commits.append(sql)
        return result

    monkeypatch.setattr(conn, "_execute_once", execute)
    with pytest.raises(TypedStateOwnerError):
        enqueue(api)
    assert not commits
    assert not conn.in_transaction
    assert not gateway._legacy_merge_queue_service._retired
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in real_execute(
            "SELECT * FROM merge_requests ORDER BY request_id", None
        ).fetchall()
    ]
    assert real_execute("SELECT 1", None).fetchone()[0] == 1


def test_revoke_is_serialized_after_an_already_admitted_commit(owner, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    gateway, conn, bind, attach, *_ = owner
    bind()
    api, grant = attach()
    real_execute = conn._execute_once
    at_commit = threading.Event()
    revoke_attempted = threading.Event()
    revoke_finished = threading.Event()

    def execute(sql, parameters=None):
        if sql.strip().upper() == "COMMIT":
            assert gateway._grants_lock.locked()
            at_commit.set()
            assert revoke_attempted.wait(3)
            assert not revoke_finished.is_set()
        return real_execute(sql, parameters)

    def revoke():
        assert at_commit.wait(3)
        revoke_attempted.set()
        gateway.revoke_grant(grant.grant_id)
        revoke_finished.set()

    monkeypatch.setattr(conn, "_execute_once", execute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(revoke)
        added = request(enqueue(api))
        pending.result(timeout=5)
    assert revoke_finished.is_set()
    assert (
        conn.execute(
            "SELECT request_id FROM merge_requests WHERE task_id='task:one'"
        ).fetchone()[0]
        == added["request_id"]
    )
    with pytest.raises(TypedStateOwnerError):
        api.call("claim", request_id=added["request_id"])


def test_revoked_idempotent_enqueue_cannot_acknowledge_post_rollback_lookup(
    owner, monkeypatch
):
    gateway, conn, bind, attach, *_ = owner
    bind()
    api, grant = attach()
    added = request(enqueue(api))
    real_execute = conn._execute_once
    commits = []

    def execute(sql, parameters=None):
        result = real_execute(sql, parameters)
        if (
            sql.strip().upper().startswith("SELECT * FROM MERGE_REQUESTS")
            and conn.in_transaction
        ):
            gateway.revoke_grant(grant.grant_id)
        if sql.strip().upper() == "COMMIT":
            commits.append(sql)
        return result

    monkeypatch.setattr(conn, "_execute_once", execute)
    with pytest.raises(TypedStateOwnerError):
        enqueue(api)
    assert not commits
    assert not conn.in_transaction
    assert not gateway._legacy_merge_queue_service._retired
    assert (
        real_execute(
            "SELECT request_id FROM merge_requests WHERE task_id='task:one'", None
        ).fetchone()[0]
        == added["request_id"]
    )


def test_stale_dedupe_foreign_target_cannot_be_acknowledged_or_rebound(owner):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    added = request(enqueue(api))
    metadata = dict(added["metadata"], target_repository_id="repo:other")
    conn.execute(
        "UPDATE merge_requests SET metadata_json=? WHERE request_id=?",
        [json.dumps(metadata), added["request_id"]],
    )
    before = [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    for operation in (
        lambda: enqueue(api),
        lambda: api.call("get", request_id=added["request_id"]),
    ):
        with pytest.raises(TypedStateOwnerError):
            operation()
    assert before == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    assert not conn.in_transaction


@pytest.mark.parametrize("violation", ["temporary_shadow", "schema_drift"])
def test_existing_binding_denies_subsequent_namespace_or_schema_drift(owner, violation):
    _, conn, bind, attach, rows, *_ = owner
    bind()
    api, _ = attach()
    if violation == "temporary_shadow":
        conn.execute(
            "CREATE TEMPORARY TABLE merge_requests AS SELECT * FROM merge_requests"
        )
    else:
        conn.execute("ALTER TABLE merge_requests ADD COLUMN surprise INTEGER")
    with pytest.raises(TypedStateOwnerError):
        api.call("get", request_id=rows[2].request_id)
    assert not conn.in_transaction


def test_owner_port_never_uses_implicit_connection_reconnect_or_generic_gateway_reopen(
    owner, monkeypatch
):
    gateway, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnectionPolicyError,
    )

    real_execute = conn._execute_once

    def execute(sql, parameters=None):
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            raise DuckDBConnectionPolicyError(
                "DuckDB connection is unusable after an uncertain transaction"
            )
        return real_execute(sql, parameters)

    monkeypatch.setattr(conn, "_execute_once", execute)
    monkeypatch.setattr(
        conn,
        "reconnect_exclusive_owner",
        lambda: pytest.fail("implicit file reconnect"),
    )
    monkeypatch.setattr(
        gateway,
        "_recover_exclusive_handle_after_native_fatal",
        lambda *a: pytest.fail("gateway reopen"),
    )
    with pytest.raises(TypedStateOwnerError):
        enqueue(api)
    assert not conn.in_transaction
    assert real_execute("SELECT 1", None).fetchone()[0] == 1


@pytest.mark.parametrize("policy", ["processing", "bytes", "usage_failure"])
def test_owner_supplied_capacity_policy_is_preserved(owner, policy):
    _, conn, bind, attach, *_ = owner

    def failed_usage():
        raise RuntimeError("usage unavailable")

    bind(
        **{"max_processing": 1}
        if policy == "processing"
        else {
            "max_worktree_bytes": 0 if policy == "bytes" else 100,
            "worktree_usage": failed_usage if policy == "usage_failure" else None,
        }
    )
    api, _ = attach()
    added = request(enqueue(api))
    assert request(api.call("claim", request_id=added["request_id"])) is None
    assert (
        conn.execute(
            "SELECT status FROM merge_requests WHERE request_id=?",
            [added["request_id"]],
        ).fetchone()[0]
        == "pending"
    )


@pytest.mark.parametrize(
    "failure", ["rollback_before_effect", "native_commit_poison", "native_begin_poison"]
)
def test_uncertain_queue_transaction_freezes_other_gateway_services_without_reopen(
    owner, monkeypatch, failure
):
    from concurrent.futures import ThreadPoolExecutor
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBOwnerRetiredError,
        open_duckdb_connection,
    )

    gateway, conn, bind, attach, _, original, _, queue_dir = owner
    bind()
    api, _ = attach()
    token, _ = gateway.issue_grant(
        client_id="other-service",
        process_birth_id="birth:other-service",
        allowed_operations=("whoami_metadata",),
    )
    observer = TypedStateOwnerConnection(
        socket_path=gateway.socket_path,
        token=token,
        client_id="other-service",
        process_birth_id="birth:other-service",
        store_id="control.duckdb",
    )
    real_execute = conn._execute_once
    raw = conn._connection
    rollbacks = []
    reopen_calls = []

    def execute(sql, parameters=None):
        if (
            failure == "rollback_before_effect"
            and "SELECT * FROM merge_requests WHERE request_id = ?" in sql
        ):
            raise RuntimeError("post-insert failure")
        if (
            failure in {"native_commit_poison", "native_begin_poison"}
            and sql.strip().upper()
            == ("COMMIT" if failure == "native_commit_poison" else "BEGIN IMMEDIATE")
            and not conn._owner_binding_retired
        ):
            with conn._execution_condition:
                conn._poison_locked()  # native wrapper's existing uncertain-commit behavior
            raise RuntimeError("native commit lost its handle")
        return real_execute(sql, parameters)

    def failed_rollback():
        rollbacks.append(True)
        raise RuntimeError("rollback outcome unavailable")

    def reopen():
        reopen_calls.append(True)
        pytest.fail("retired binding must not recover itself")

    monkeypatch.setattr(conn, "_execute_once", execute)
    if failure == "rollback_before_effect":
        monkeypatch.setattr(conn, "rollback", failed_rollback)
    monkeypatch.setattr(conn, "_recover_exclusive_handle_locked", reopen)
    with pytest.raises(TypedStateOwnerError):
        enqueue(api)
    assert conn._owner_binding_retired
    assert gateway._legacy_merge_queue_service._retired
    assert conn._connection is (raw if failure == "rollback_before_effect" else None)
    assert len(rollbacks) == (1 if failure == "rollback_before_effect" else 0)
    with pytest.raises(TypedStateOwnerError):
        observer.execute_operation("whoami_metadata", [])
    operations = [
        lambda: conn.execute("SELECT 1"),
        lambda: conn.execute(
            "UPDATE merge_requests SET failure_reason='other service'"
        ),
        conn.commit,
        lambda: conn.execute("COMMIT"),
        conn.reconnect_exclusive_owner,
    ]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for operation in operations:
            with pytest.raises(DuckDBOwnerRetiredError):
                pool.submit(operation).result(timeout=3)
    assert not reopen_calls
    # Only explicit simulated owner shutdown disposes of the frozen handle.
    # A fresh test connection verifies rollback preserved all original rows.
    observer.close()
    gateway.stop()
    conn.close()
    with open_duckdb_connection(queue_dir / "merge_queue.duckdb") as fresh:
        assert original == [
            tuple(row[index] for index in range(len(row)))
            for row in fresh.execute(
                "SELECT * FROM merge_requests ORDER BY request_id"
            ).fetchall()
        ]


def test_dequeue_claims_only_eligible_bound_work_without_reaping(owner, monkeypatch):
    gateway, conn, bind, attach, retained, original, files, queue_dir = owner
    bind()
    api, _ = attach()
    # Even expired retained claims cannot be recovered by the new operation.
    gateway._legacy_merge_queue_service._queue._clock = lambda: 2000000000.0
    delayed = request(enqueue(api, "task:delayed"))
    conn.execute(
        "UPDATE merge_requests SET retry_not_before=? WHERE request_id=?",
        [2100000000.0, delayed["request_id"]],
    )
    recovery = request(enqueue(api, "task:recovery"))
    # Retained queue-authored recovery metadata cannot be created by enqueue.
    metadata = dict(recovery["metadata"])
    metadata[merge_queue._FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY] = {"required": True}
    conn.execute(
        "UPDATE merge_requests SET metadata_json=? WHERE request_id=?",
        [json.dumps(metadata), recovery["request_id"]],
    )
    lower = request(enqueue(api, "task:lower", priority="P2"))
    eligible = request(enqueue(api, "task:eligible", priority="P0"))
    monkeypatch.setattr(
        merge_queue, "open_duckdb_connection", lambda *a, **k: pytest.fail("second file open")
    )
    claimed = request(api.call("dequeue"))
    assert claimed["request_id"] == eligible["request_id"]
    assert claimed["consumer_id"] == "consumer:one"
    assert claimed["claim_token"] and claimed["claim_generation"] == 1
    next_claim = request(api.call("dequeue"))
    assert next_claim["request_id"] == lower["request_id"]
    assert request(api.call("dequeue")) is None
    assert request(api.call("get", request_id=delayed["request_id"]))["status"] == "pending"
    assert request(api.call("get", request_id=recovery["request_id"]))["status"] == "pending"
    assert original == [
        tuple(row[index] for index in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests WHERE task_id IN "
            "('task:unbound','task:other','task:retained') ORDER BY request_id"
        ).fetchall()
    ]
    assert files == {
        str(path.relative_to(queue_dir)): path.read_bytes() for path in queue_dir.rglob("*.json")
    }


@pytest.mark.parametrize("policy", [{"max_processing": 1}, {"max_worktree_bytes": 1}])
def test_dequeue_preserves_owner_capacity_policy(owner, policy):
    _, _, bind, attach, *_ = owner
    bind(**policy)
    api, _ = attach()
    added = request(enqueue(api, metadata_json='{"worktree_bytes": 100}'))
    assert request(api.call("dequeue")) is None
    assert request(api.call("get", request_id=added["request_id"]))["status"] == "pending"


@pytest.mark.parametrize(
    "arguments", [{"consumer_id": "other"}, {"limit": 2}, {"request_id": "unknown"}]
)
def test_dequeue_has_no_client_policy_or_consumer_override(owner, arguments):
    _, _, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    added = request(enqueue(api))
    with pytest.raises(TypedStateOwnerError):
        api.call("dequeue", **arguments)
    assert request(api.call("get", request_id=added["request_id"]))["status"] == "pending"


def test_dequeue_requires_separately_admitted_operation(owner):
    _, _, bind, attach, *_ = owner
    bind()
    api, _ = attach(operations=SERVICE_OPERATIONS - {"legacy.merge_queue.dequeue"})
    added = request(enqueue(api))
    with pytest.raises(TypedStateOwnerError):
        api.call("dequeue")
    assert request(api.call("get", request_id=added["request_id"]))["status"] == "pending"


def test_dequeue_rejects_corrupt_identity_before_claim_commit(owner):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    added = request(enqueue(api))
    conn.execute(
        "UPDATE merge_requests SET dedupe_key='corrupt' WHERE request_id=?", [added["request_id"]]
    )

    def current_row():
        row = conn.execute(
            "SELECT * FROM merge_requests WHERE request_id=?", [added["request_id"]]
        ).fetchone()
        return tuple(row[index] for index in range(len(row)))

    before = current_row()
    with pytest.raises(TypedStateOwnerError):
        api.call("dequeue")
    assert current_row() == before
    assert not conn.in_transaction


@pytest.mark.parametrize("denial", ["revoke", "expire", "detach"])
def test_dequeue_admission_lost_after_claim_rolls_back(owner, monkeypatch, denial):
    gateway, conn, bind, attach, *_ = owner
    bind()
    api, grant = attach()
    added = request(enqueue(api))
    real_execute = conn._execute_once
    before = [
        tuple(row[i] for i in range(len(row)))
        for row in real_execute("SELECT * FROM merge_requests ORDER BY request_id", None).fetchall()
    ]
    commits = []

    def execute(sql, parameters=None):
        result = real_execute(sql, parameters)
        if sql.strip().upper().startswith("UPDATE MERGE_REQUESTS"):
            if denial == "revoke":
                gateway.revoke_grant(grant.grant_id)
            elif denial == "expire":
                with gateway._grants_lock:
                    for token in tuple(gateway._grants):
                        gateway._grants[token] = replace(grant, issued_at=1, expires_at=1001)
            else:
                real_execute(
                    "UPDATE client_sessions SET status='detached' WHERE session_id=?",
                    [api.connection.session_id],
                )
        if sql.strip().upper() == "COMMIT":
            commits.append(sql)
        return result

    monkeypatch.setattr(conn, "_execute_once", execute)
    with pytest.raises(TypedStateOwnerError):
        api.call("dequeue")
    assert not commits and not conn.in_transaction
    assert not gateway._legacy_merge_queue_service._retired
    assert before == [
        tuple(row[i] for i in range(len(row)))
        for row in real_execute("SELECT * FROM merge_requests ORDER BY request_id", None).fetchall()
    ]
    assert (
        real_execute(
            "SELECT status FROM merge_requests WHERE request_id=?", [added["request_id"]]
        ).fetchone()[0]
        == "pending"
    )
