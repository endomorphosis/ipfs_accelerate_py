"""Actual owner recovery state survives clients without filesystem authority."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import owner_recovery_runtime as recovery
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)
from test.api.test_agent_supervisor_owner_merge_queue import owner as _owner_fixture

owner = _owner_fixture

SCOPE = {
    "board_namespace": "SPAR",
    "config_cid": "sha256:" + "a" * 64,
    "plan_cid": "sha256:" + "b" * 64,
    "lane_id": "lane:0",
    "attempt_root": "/admitted/attempts/lane-0",
}


@pytest.fixture
def recovery_owner(owner):
    gateway, connection, bind, attach_queue, *rest = owner
    bind()
    migration = gateway.provision_legacy_merge_recovery_schema(
        expected_identity=dict(gateway.identity),
        repository_id="repo:one",
        target_branch="main",
        migration_id="migration:reviewed",
        scope_bindings=[SCOPE],
    )
    gateway.bind_legacy_merge_recovery_service(
        expected_identity=dict(gateway.identity),
        repository_id="repo:one",
        target_branch="main",
    )
    scope = migration["scope_cids"][0]
    clients = []

    def attach(
        *,
        consumer="consumer:one",
        scope_cid=scope,
        scopes=None,
        operations=recovery.SERVICE_OPERATIONS,
    ):
        if scopes is None:
            scopes = {
                "repository_id": "repo:one",
                "target_branch": "main",
                "consumer_id": consumer,
                "recovery_scope_cid": scope_cid,
            }
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
        return recovery.OwnerRecoveryRuntimeClient(
            client,
            repository_id="repo:one",
            target_branch="main",
            consumer_id=consumer,
            recovery_scope_cid=scope_cid,
        ), grant

    api, grant = attach()
    value = SimpleNamespace(
        gateway=gateway,
        connection=connection,
        api=api,
        grant=grant,
        attach=attach,
        attach_queue=attach_queue,
        scope_cid=scope,
        scope_binding=SCOPE,
        original_rows=rest[1],
        queue_dir=rest[-1],
    )
    try:
        yield value
    finally:
        for client in clients:
            client.close()


def advance(api, head=None, *, operation_id="cursor:advance", value="request:064"):
    head = api.load_cursors() if head is None else head
    cursors = dict(head["cursors"])
    cursors["completed_requests"] = value
    return api.cas_cursors(
        expected_revision=head["revision"],
        expected_state_cid=head["state_cid"],
        cursors=cursors,
        operation_id=operation_id,
    )


def test_durable_cursor_cas_conflict_replay_and_noop(recovery_owner):
    own = recovery_owner
    api = own.api
    assert api.describe_scope() == SCOPE
    initial = api.load_cursors()
    assert initial["revision"] == 0
    first = advance(api, initial)
    assert (
        first["revision"] == 1
        and first["cursors"]["completed_requests"] == "request:064"
    )
    assert advance(api, initial) == first
    with pytest.raises(TypedStateOwnerError):
        advance(api, initial, value="request:065")
    api2, _ = own.attach()
    assert api2.load_cursors()["state_cid"] == first["state_cid"]
    stale = advance(api2, initial, operation_id="cursor:stale", value="request:999")
    assert stale["conflict"] is True and stale["cursors"] == first["cursors"]
    noop = advance(api2, first, operation_id="cursor:noop")
    assert noop["changed"] is False and noop["revision"] == 1
    assert (
        own.connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_cursor_history"
        ).fetchone()[0]
        == 2
    )
    assert (
        own.connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_operations"
        ).fetchone()[0]
        == 1
    )
    actual = [
        tuple(row[i] for i in range(len(row)))
        for row in own.connection.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]
    assert actual == own.original_rows


def test_bind_never_creates_missing_schema(owner):
    gateway, connection, bind, *_ = owner
    bind()
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        gateway.bind_legacy_merge_recovery_service(
            expected_identity=dict(gateway.identity),
            repository_id="repo:one",
            target_branch="main",
        )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name LIKE 'legacy_merge_recovery_%'"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    "change", ["scope", "target", "repository", "consumer", "extra"]
)
def test_exact_recovery_grant_scope_denies_before_writes(recovery_owner, change):
    own = recovery_owner
    scopes = {
        "repository_id": "repo:one",
        "target_branch": "main",
        "consumer_id": "consumer:one",
        "recovery_scope_cid": own.scope_cid,
    }
    key = {
        "scope": "recovery_scope_cid",
        "target": "target_branch",
        "repository": "repository_id",
        "consumer": "consumer_id",
        "extra": "task_cid",
    }[change]
    scopes[key] = "foreign"
    api, _ = own.attach(scopes=scopes)
    with pytest.raises(TypedStateOwnerError):
        advance(api)
    assert own.api.load_cursors()["revision"] == 0


def test_lease_expiry_cannot_steal_and_receipts_are_versioned(recovery_owner):
    own = recovery_owner
    api = own.api
    lease = api.acquire_consumer_lease(operation_id="lease:one", ttl_seconds=1)
    assert lease["acquired"] is True
    own.connection.execute("UPDATE legacy_merge_recovery_leases SET expires_at=0")
    other, _ = own.attach(consumer="consumer:two")
    assert other.acquire_consumer_lease(operation_id="lease:steal")["acquired"] is False
    args = {"lease_id": lease["lease_id"], "fence_epoch": lease["fence_epoch"]}
    with pytest.raises(TypedStateOwnerError):
        other.release_consumer_lease(**args, operation_id="lease:foreign-release")
    first = api.publish_receipt(
        receipt_key="train:acceptance:one",
        receipt={"stage": "prepared"},
        expected_revision=0,
        expected_receipt_cid="",
        operation_id="receipt:one",
        **args,
    )
    assert first["head"]["revision"] == 1
    same = api.publish_receipt(
        receipt_key="train:acceptance:one",
        receipt={"stage": "prepared"},
        expected_revision=0,
        expected_receipt_cid="",
        operation_id="receipt:one",
        **args,
    )
    assert same == first
    second = api.publish_receipt(
        receipt_key="train:acceptance:one",
        receipt={"stage": "validated"},
        expected_revision=1,
        expected_receipt_cid=first["head"]["receipt_cid"],
        operation_id="receipt:two",
        **args,
    )
    assert second["head"]["revision"] == 2
    assert api.get_receipt("train:acceptance:one", revision=1) == first["head"]
    assert other.get_receipt("train:acceptance:one") == second["head"]
    api.release_consumer_lease(**args, operation_id="lease:release")
    successor = other.acquire_consumer_lease(operation_id="lease:next")
    assert successor["fence_epoch"] == lease["fence_epoch"] + 1
    with pytest.raises(TypedStateOwnerError):
        api.renew_consumer_lease(**args, operation_id="lease:stale")


@pytest.mark.parametrize("denial", ["revoke", "expire", "detach"])
def test_cursor_admission_lost_after_update_rolls_back(
    recovery_owner, monkeypatch, denial
):
    own = recovery_owner
    run = own.connection._execute_once
    commits = []
    initial = own.api.load_cursors()

    def execute(sql, parameters=None):
        result = run(sql, parameters)
        if sql.startswith("UPDATE legacy_merge_recovery_cursors SET"):
            if denial == "revoke":
                own.gateway.revoke_grant(own.grant.grant_id)
            elif denial == "expire":
                with own.gateway._grants_lock:
                    for token, candidate in tuple(own.gateway._grants.items()):
                        if candidate.grant_id == own.grant.grant_id:
                            own.gateway._grants[token] = replace(
                                candidate, issued_at=1, expires_at=1001
                            )
            else:
                run(
                    "UPDATE client_sessions SET status='detached' WHERE session_id=?",
                    [own.api.connection.session_id],
                )
        if sql == "COMMIT":
            commits.append(sql)
        return result

    monkeypatch.setattr(own.connection, "_execute_once", execute)
    with pytest.raises(TypedStateOwnerError):
        advance(own.api, initial)
    assert not commits and not own.connection.in_transaction
    assert (
        run("SELECT revision FROM legacy_merge_recovery_cursors", None).fetchone()[0]
        == 0
    )
    assert (
        run("SELECT COUNT(*) FROM legacy_merge_recovery_operations", None).fetchone()[0]
        == 0
    )
    assert not own.connection._owner_binding_retired
    assert run("SELECT 1", None).fetchone()[0] == 1


@pytest.mark.parametrize("failure", ["commit", "rollback"])
def test_uncertain_recovery_transaction_retires_shared_owner_without_reopen(
    recovery_owner, monkeypatch, failure
):
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBOwnerRetiredError,
    )

    own = recovery_owner
    run = own.connection._execute_once
    initial = own.api.load_cursors()
    rollbacks = []

    def execute(sql, parameters=None):
        if failure == "commit" and sql == "COMMIT":
            raise RuntimeError("injected unknown commit")
        if failure == "rollback" and sql.startswith(
            "INSERT INTO legacy_merge_recovery_cursor_history"
        ):
            raise RuntimeError("injected write failure")
        return run(sql, parameters)

    def rollback():
        rollbacks.append(True)
        raise RuntimeError("injected unknown rollback")

    monkeypatch.setattr(own.connection, "_execute_once", execute)
    if failure == "rollback":
        monkeypatch.setattr(own.connection, "rollback", rollback)
    monkeypatch.setattr(
        own.connection,
        "_recover_exclusive_handle_locked",
        lambda: pytest.fail("recovery may not reopen shared handle"),
    )
    with pytest.raises(TypedStateOwnerError):
        advance(own.api, initial)
    assert own.connection._owner_binding_retired
    for operation in (
        lambda: own.connection.execute("SELECT 1"),
        own.connection.commit,
        own.connection.reconnect_exclusive_owner,
    ):
        with pytest.raises(DuckDBOwnerRetiredError):
            operation()
    assert len(rollbacks) == (1 if failure == "rollback" else 0)


@pytest.mark.parametrize("denial", ["generation", "fence", "database", "session"])
def test_owner_generation_and_session_drift_deny_writes(recovery_owner, denial):
    own = recovery_owner
    initial = own.api.load_cursors()
    if denial == "session":
        own.connection.execute(
            "UPDATE client_sessions SET status='detached' WHERE session_id=?",
            [own.api.connection.session_id],
        )
    else:
        sql = {
            "generation": "UPDATE store_generations SET generation=generation+1",
            "fence": "UPDATE store_generations SET fence_epoch=fence_epoch+1",
            "database": "UPDATE store_generations SET database_uuid='foreign'",
        }[denial]
        own.connection.execute(sql)
    with pytest.raises(TypedStateOwnerError):
        advance(own.api, initial)
    assert (
        own.connection.execute(
            "SELECT revision FROM legacy_merge_recovery_cursors"
        ).fetchone()[0]
        == 0
    )


def test_explicit_migration_imports_receipt_history_without_silent_gaps(owner):
    gateway, connection, bind, *_ = owner
    bind()
    first = {"stage": "prepared"}
    second = {"stage": "validated"}
    imports = [
        {
            "receipt_key": "legacy:receipt",
            "revision": i,
            "receipt_cid": recovery._cid(value),
            "receipt": value,
        }
        for i, value in enumerate((first, second), 1)
    ]
    options = {
        "expected_identity": dict(gateway.identity),
        "repository_id": "repo:one",
        "target_branch": "main",
        "migration_id": "migration:preserved",
        "scope_bindings": [SCOPE],
        "receipt_imports": imports,
    }
    result = gateway.provision_legacy_merge_recovery_schema(**options)
    assert gateway.provision_legacy_merge_recovery_schema(**options)["replayed"] is True
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_receipt_versions"
        ).fetchone()[0]
        == 2
    )
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        gateway.provision_legacy_merge_recovery_schema(
            **{
                **options,
                "migration_id": "migration:gap",
                "receipt_imports": [{**imports[1], "revision": 4}],
            }
        )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_receipt_versions"
        ).fetchone()[0]
        == 2
    )
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        gateway.provision_legacy_merge_recovery_schema(
            **{**options, "receipt_imports": []}
        )
    assert result["scope_cids"]


def test_release_lost_response_replays_exact_holder_receipt(recovery_owner):
    own = recovery_owner
    lease = own.api.acquire_consumer_lease(operation_id="acquire:release")
    args = {
        "lease_id": lease["lease_id"],
        "fence_epoch": lease["fence_epoch"],
        "operation_id": "release:exact",
    }
    assert own.api.release_consumer_lease(**args) == own.api.release_consumer_lease(
        **args
    )


def test_old_acquire_reply_does_not_revive_released_custody(recovery_owner):
    own = recovery_owner
    lease = own.api.acquire_consumer_lease(operation_id="acquire:replay")
    assert own.api.acquire_consumer_lease(operation_id="acquire:replay") == lease
    own.api.release_consumer_lease(
        lease_id=lease["lease_id"],
        fence_epoch=lease["fence_epoch"],
        operation_id="release:replay",
    )
    with pytest.raises(TypedStateOwnerError):
        own.api.acquire_consumer_lease(operation_id="acquire:replay")
    assert (
        own.connection.execute(
            "SELECT state FROM legacy_merge_recovery_leases"
        ).fetchone()[0]
        == "released"
    )


def test_gateway_successor_preserves_cursor_without_adopting_old_custody(
    recovery_owner, tmp_path
):
    from test.api.causal_federation.test_typed_state_owner import _gateway

    own = recovery_owner
    initial = own.api.load_cursors()
    first = advance(own.api, initial)
    lease = own.api.acquire_consumer_lease(operation_id="lease:original", ttl_seconds=1)
    old_identity = dict(own.gateway.identity)
    # This is an isolated fixture's explicit owner-generation transition,
    # after transport shutdown, never an automatic lease takeover policy.
    own.gateway.stop()
    own.connection.execute(
        "UPDATE store_generations SET generation=generation+1,fence_epoch=fence_epoch+1,birth_id='birth:successor'"
    )
    own.connection.execute("UPDATE legacy_merge_recovery_leases SET expires_at=0")
    own.connection.close()
    successor, connection = _gateway(
        own.queue_dir / "merge_queue.duckdb", tmp_path / "successor.sock"
    )
    client = None
    try:
        assert successor.identity["generation"] == old_identity["generation"] + 1
        successor.bind_legacy_merge_queue_service(
            expected_identity=dict(successor.identity),
            repository_id="repo:one",
            target_branch="main",
            max_age_seconds=3600,
            max_queue_size=100,
            max_processing=10,
            max_attempts=3,
            max_worktree_bytes=None,
        )
        # No provisioning or import is performed during this restart.
        successor.bind_legacy_merge_recovery_service(
            expected_identity=dict(successor.identity),
            repository_id="repo:one",
            target_branch="main",
        )
        scopes = {
            "repository_id": "repo:one",
            "target_branch": "main",
            "consumer_id": "consumer:one",
            "recovery_scope_cid": own.scope_cid,
        }
        token, _ = successor.issue_grant(
            client_id="consumer:one",
            process_birth_id="birth:new-client",
            allowed_operations=tuple(recovery.SERVICE_OPERATIONS),
            entity_scopes=scopes,
        )
        client = TypedStateOwnerConnection(
            socket_path=successor.socket_path,
            token=token,
            client_id="consumer:one",
            process_birth_id="birth:new-client",
            store_id="control.duckdb",
        )
        api = recovery.OwnerRecoveryRuntimeClient(client, **scopes)
        assert api.describe_scope() == SCOPE
        assert api.load_cursors()["state_cid"] == first["state_cid"]
        assert advance(api, initial) == first  # durable response replay, no new write
        second = advance(
            api, first, operation_id="cursor:successor", value="request:128"
        )
        assert second["revision"] == 2
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM legacy_merge_recovery_cursor_history"
            ).fetchone()[0]
            == 3
        )
        with pytest.raises((TypedStateOwnerError, OSError)):
            own.api.load_cursors()
        assert (
            api.acquire_consumer_lease(operation_id="lease:new-owner")["acquired"]
            is False
        )
        with pytest.raises(TypedStateOwnerError):
            api.acquire_consumer_lease(operation_id="lease:original", ttl_seconds=1)
        for operation in (api.renew_consumer_lease, api.release_consumer_lease):
            with pytest.raises(TypedStateOwnerError):
                operation(
                    lease_id=lease["lease_id"],
                    fence_epoch=lease["fence_epoch"],
                    operation_id="lease:deny" + operation.__name__,
                )
        held = connection.execute(
            "SELECT lease_id,fence_epoch,state,owner_json FROM legacy_merge_recovery_leases"
        ).fetchone()
        assert tuple(held[index] for index in range(3)) == (
            lease["lease_id"],
            lease["fence_epoch"],
            "active",
        )
        assert json.loads(held[3]) == old_identity
    finally:
        if client is not None:
            client.close()
        successor.stop()
        connection.close()


@pytest.mark.parametrize(
    "corruption", ["missing_head", "missing_version", "substituted_version"]
)
def test_receipt_corruption_never_becomes_absence_or_replacement(
    recovery_owner, corruption
):
    own = recovery_owner
    lease = own.api.acquire_consumer_lease(operation_id="lease:history")
    args = {"lease_id": lease["lease_id"], "fence_epoch": lease["fence_epoch"]}
    own.api.publish_receipt(
        "train:history",
        {"stage": "preserved"},
        expected_revision=0,
        expected_receipt_cid="",
        operation_id="receipt:preserved",
        **args,
    )
    sql = {
        "missing_head": "DELETE FROM legacy_merge_recovery_receipt_heads",
        "missing_version": "DELETE FROM legacy_merge_recovery_receipt_versions",
        "substituted_version": "UPDATE legacy_merge_recovery_receipt_versions SET receipt_json='{}'",
    }[corruption]
    own.connection.execute(sql)
    with pytest.raises(TypedStateOwnerError):
        own.api.get_receipt("train:history")
    with pytest.raises(TypedStateOwnerError):
        own.api.publish_receipt(
            "train:history",
            {"stage": "replacement"},
            expected_revision=0,
            expected_receipt_cid="",
            operation_id="receipt:replace",
            **args,
        )
    assert (
        own.connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_operations WHERE operation_id='receipt:replace'"
        ).fetchone()[0]
        == 0
    )


def test_missing_cursor_history_denies_resume_and_advance(recovery_owner):
    own = recovery_owner
    initial = own.api.load_cursors()
    own.connection.execute("DELETE FROM legacy_merge_recovery_cursor_history")
    with pytest.raises(TypedStateOwnerError):
        own.api.load_cursors()
    with pytest.raises(TypedStateOwnerError):
        advance(own.api, initial)
    assert (
        own.connection.execute(
            "SELECT revision FROM legacy_merge_recovery_cursors"
        ).fetchone()[0]
        == 0
    )


def test_lane_scopes_share_target_receipts_and_exclude_parallel_consumer(
    recovery_owner,
):
    own = recovery_owner
    other_scope = {
        **SCOPE,
        "lane_id": "lane:1",
        "attempt_root": "/admitted/attempts/lane-1",
    }
    result = own.gateway.provision_legacy_merge_recovery_schema(
        expected_identity=dict(own.gateway.identity),
        repository_id="repo:one",
        target_branch="main",
        migration_id="migration:other-lane",
        scope_bindings=[other_scope],
    )
    other, _ = own.attach(scope_cid=result["scope_cids"][0])
    lease = own.api.acquire_consumer_lease(operation_id="lease:lane-zero")
    assert (
        other.acquire_consumer_lease(operation_id="lease:lane-one")["acquired"] is False
    )
    args = {"lease_id": lease["lease_id"], "fence_epoch": lease["fence_epoch"]}
    first = own.api.publish_receipt(
        "train:target-wide",
        {"stage": "validated"},
        expected_revision=0,
        expected_receipt_cid="",
        operation_id="receipt:shared",
        **args,
    )
    assert other.get_receipt("train:target-wide") == first["head"]
    with pytest.raises(TypedStateOwnerError):
        other.release_consumer_lease(**args, operation_id="lease:wrong-lane")
    advance(own.api)
    assert other.load_cursors()["revision"] == 0


def test_receipt_float_values_roundtrip_exactly_through_typed_socket(recovery_owner):
    api = recovery_owner.api
    lease = api.acquire_consumer_lease(operation_id="lease:float")
    value = {
        "timestamp": 1789074000.125,
        "score": 0.25,
        "nested": {"scores": [1.25, -0.0]},
    }
    result = api.publish_receipt(
        "train:float",
        value,
        expected_revision=0,
        expected_receipt_cid="",
        lease_id=lease["lease_id"],
        fence_epoch=lease["fence_epoch"],
        operation_id="receipt:float",
    )
    assert result["head"]["receipt"] == value
    assert result["head"]["receipt_cid"] == recovery._cid(value)
    assert api.get_receipt("train:float") == result["head"]
    assert api.get_receipt("train:float", revision=1) == result["head"]


@pytest.mark.parametrize("denial", [None, "scope", "hash"])
def test_explicit_migration_preserves_supplied_cursor_without_rebinding(owner, denial):
    gateway, connection, bind, *_ = owner
    bind()
    scope = recovery.recovery_scope_cid(
        store_id=gateway.store_id,
        repository_id="repo:one",
        target_branch="main",
        scope_binding=SCOPE,
    )
    cursors = {
        stage: "request:064" if stage == "completed_requests" else ""
        for stage in recovery.STAGES
    }
    item = {"scope_cid": scope, "cursors": cursors, "state_cid": recovery._cid(cursors)}
    if denial == "scope":
        item["scope_cid"] = "foreign:scope"
    if denial == "hash":
        item["state_cid"] = "sha256:" + "0" * 64
    options = {
        "expected_identity": dict(gateway.identity),
        "repository_id": "repo:one",
        "target_branch": "main",
        "migration_id": "migration:legacy-cursor",
        "scope_bindings": [SCOPE],
        "cursor_imports": [item],
    }
    if denial:
        with pytest.raises(recovery.OwnerRecoveryRuntimeError):
            gateway.provision_legacy_merge_recovery_schema(**options)
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name LIKE 'legacy_merge_recovery_%'"
            ).fetchone()[0]
            == 0
        )
        return
    gateway.provision_legacy_merge_recovery_schema(**options)
    assert gateway.provision_legacy_merge_recovery_schema(**options)["replayed"] is True
    row = connection.execute(
        "SELECT revision,state_cid,cursors_json FROM legacy_merge_recovery_cursors"
    ).fetchone()
    assert row[0] == 0 and row[1] == item["state_cid"] and json.loads(row[2]) == cursors
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        gateway.provision_legacy_merge_recovery_schema(
            **{**options, "migration_id": "migration:overwrite"}
        )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_cursor_history"
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "unknown"),
        ("fence_epoch", 0),
        ("lease_id", ""),
        ("consumer_id", ""),
        ("scope_cid", "foreign"),
        ("peer_json", "[1,0,0]"),
        ("owner_json", "{}"),
    ],
)
def test_malformed_lease_row_never_authorizes_takeover(recovery_owner, field, value):
    own = recovery_owner
    lease = own.api.acquire_consumer_lease(operation_id="lease:corruption")
    own.connection.execute(
        "UPDATE legacy_merge_recovery_leases SET " + field + "=?", [value]
    )
    before = own.connection.execute(
        "SELECT * FROM legacy_merge_recovery_leases"
    ).fetchone()
    expected = tuple(before[i] for i in range(len(before)))
    other, _ = own.attach(consumer="consumer:successor")
    with pytest.raises(TypedStateOwnerError):
        other.acquire_consumer_lease(operation_id="lease:no-takeover")
    with pytest.raises(TypedStateOwnerError):
        own.api.acquire_consumer_lease(operation_id="lease:corruption")
    after = own.connection.execute(
        "SELECT * FROM legacy_merge_recovery_leases"
    ).fetchone()
    assert tuple(after[i] for i in range(len(after))) == expected
    assert lease["acquired"]


def test_rewound_heads_do_not_hide_newer_immutable_history(recovery_owner):
    own = recovery_owner
    api = own.api
    initial = api.load_cursors()
    advance(api, initial)
    own.connection.execute(
        "UPDATE legacy_merge_recovery_cursors SET revision=?,state_cid=?,cursors_json=?",
        [initial["revision"], initial["state_cid"], recovery._json(initial["cursors"])],
    )
    with pytest.raises(TypedStateOwnerError):
        api.load_cursors()
    with pytest.raises(TypedStateOwnerError):
        advance(api, initial, operation_id="cursor:no-rewind")
    lease = api.acquire_consumer_lease(operation_id="lease:rewound-receipt")
    args = {"lease_id": lease["lease_id"], "fence_epoch": lease["fence_epoch"]}
    first = api.publish_receipt(
        "train:rewound",
        {"stage": "one"},
        expected_revision=0,
        expected_receipt_cid="",
        operation_id="receipt:one",
        **args,
    )["head"]
    api.publish_receipt(
        "train:rewound",
        {"stage": "two"},
        expected_revision=1,
        expected_receipt_cid=first["receipt_cid"],
        operation_id="receipt:two",
        **args,
    )
    own.connection.execute(
        "UPDATE legacy_merge_recovery_receipt_heads SET revision=?,receipt_cid=?",
        [1, first["receipt_cid"]],
    )
    with pytest.raises(TypedStateOwnerError):
        api.get_receipt("train:rewound")
    with pytest.raises(TypedStateOwnerError):
        api.publish_receipt(
            "train:rewound",
            {"stage": "replacement"},
            expected_revision=1,
            expected_receipt_cid=first["receipt_cid"],
            operation_id="receipt:no-rewind",
            **args,
        )
    assert api.get_receipt("train:rewound", revision=1) == first
    assert (
        own.connection.execute(
            "SELECT COUNT(*) FROM legacy_merge_recovery_receipt_versions"
        ).fetchone()[0]
        == 2
    )


@pytest.mark.parametrize("corruption", ["scope", "cursor_history", "receipt_version"])
def test_migration_replay_denies_missing_preserved_coordinates(owner, corruption):
    gateway, connection, bind, *_ = owner
    bind()
    receipt = {"stage": "imported", "score": 0.25}
    options = {
        "expected_identity": dict(gateway.identity),
        "repository_id": "repo:one",
        "target_branch": "main",
        "migration_id": "migration:durable-replay",
        "scope_bindings": [SCOPE],
        "receipt_imports": [
            {
                "receipt_key": "train:imported",
                "revision": 1,
                "receipt_cid": recovery._cid(receipt),
                "receipt": receipt,
            }
        ],
    }
    gateway.provision_legacy_merge_recovery_schema(**options)
    table = {
        "scope": "legacy_merge_recovery_scopes",
        "cursor_history": "legacy_merge_recovery_cursor_history",
        "receipt_version": "legacy_merge_recovery_receipt_versions",
    }[corruption]
    connection.execute("DELETE FROM " + table)
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        gateway.provision_legacy_merge_recovery_schema(**options)
    assert connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == 0


def test_extension_migration_replay_preserves_imported_cursor_after_progress(
    recovery_owner,
):
    own = recovery_owner
    binding = {
        **SCOPE,
        "lane_id": "lane:imported",
        "attempt_root": "/admitted/imported",
    }
    scope = recovery.recovery_scope_cid(
        store_id=own.gateway.store_id,
        repository_id="repo:one",
        target_branch="main",
        scope_binding=binding,
    )
    cursors = {
        stage: "request:064" if stage == "completed_requests" else ""
        for stage in recovery.STAGES
    }
    receipt = {"stage": "imported"}
    initial = {
        "expected_identity": dict(own.gateway.identity),
        "repository_id": "repo:one",
        "target_branch": "main",
        "migration_id": "migration:initial-import",
        "scope_bindings": [binding],
        "cursor_imports": [
            {
                "scope_cid": scope,
                "cursors": cursors,
                "state_cid": recovery._cid(cursors),
            }
        ],
        "receipt_imports": [
            {
                "receipt_key": "train:extension",
                "revision": 1,
                "receipt_cid": recovery._cid(receipt),
                "receipt": receipt,
            }
        ],
    }
    own.gateway.provision_legacy_merge_recovery_schema(**initial)
    api, _ = own.attach(scope_cid=scope)
    head = advance(api, operation_id="cursor:extension", value="request:128")
    lease = api.acquire_consumer_lease(operation_id="lease:extension")
    current = api.publish_receipt(
        "train:extension",
        {"stage": "later"},
        expected_revision=1,
        expected_receipt_cid=recovery._cid(receipt),
        lease_id=lease["lease_id"],
        fence_epoch=lease["fence_epoch"],
        operation_id="receipt:extension",
    )["head"]
    extension = {
        key: value
        for key, value in initial.items()
        if key not in {"receipt_imports", "cursor_imports"}
    }
    extension["migration_id"] = "migration:existing-scope"
    own.gateway.provision_legacy_merge_recovery_schema(**extension)
    assert (
        own.gateway.provision_legacy_merge_recovery_schema(**extension)["replayed"]
        is True
    )
    assert (
        own.gateway.provision_legacy_merge_recovery_schema(**initial)["replayed"]
        is True
    )
    assert api.load_cursors()["state_cid"] == head["state_cid"]
    assert api.get_receipt("train:extension") == current
    assert api.get_receipt("train:extension", revision=1)["receipt"] == receipt


@pytest.mark.parametrize("kind", ["cursor", "receipt"])
def test_current_head_denies_missing_intermediate_history(recovery_owner, kind):
    own = recovery_owner
    if kind == "cursor":
        advance(own.api, operation_id="cursor:first")
        advance(own.api, operation_id="cursor:second", value="request:128")
        own.connection.execute(
            "DELETE FROM legacy_merge_recovery_cursor_history WHERE revision=1"
        )
        with pytest.raises(TypedStateOwnerError):
            own.api.load_cursors()
    else:
        lease = own.api.acquire_consumer_lease(operation_id="lease:gapped-history")
        head = {"revision": 0, "receipt_cid": ""}
        for revision in range(1, 4):
            head = own.api.publish_receipt(
                "train:gapped",
                {"revision": revision},
                expected_revision=head["revision"],
                expected_receipt_cid=head["receipt_cid"],
                lease_id=lease["lease_id"],
                fence_epoch=lease["fence_epoch"],
                operation_id="receipt:gapped:" + str(revision),
            )["head"]
        own.connection.execute(
            "DELETE FROM legacy_merge_recovery_receipt_versions WHERE revision=2"
        )
        with pytest.raises(TypedStateOwnerError):
            own.api.get_receipt("train:gapped")
