"""Recovery grants coexist with main's exact one-use database-task sessions."""

import os

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import (
    OwnerRecoveryRuntimeClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from test.api.test_agent_supervisor_owner_recovery_runtime import (
    advance,
    owner as _owner_fixture,
    recovery_owner as _recovery_owner_fixture,
)

owner = _owner_fixture
recovery_owner = _recovery_owner_fixture


def _task_session(own):
    calls = []
    own.gateway.bind_database_task_command_handler(
        lambda command, payload, request_id, grant: (
            calls.append((command, request_id)) or {"observed": True}
        )
    )
    birth = typed.kernel_process_birth_id()
    token, grant = own.gateway.issue_grant(
        client_id="native:task-only",
        process_birth_id=birth,
        peer_pid=os.getpid(),
        allowed_database_task_commands=("record_queue_retry",),
    )

    def connect():
        return typed.TypedStateOwnerConnection(
            socket_path=own.gateway.socket_path,
            token=token,
            client_id=grant.client_id,
            process_birth_id=birth,
            store_id=own.gateway.store_id,
            timeout_seconds=2,
        )

    return connect(), connect, token, grant, calls


def test_repeated_recovery_and_one_use_task_sessions_stay_separate(recovery_owner):
    own = recovery_owner
    task, connect, token, grant, calls = _task_session(own)
    try:
        assert token not in own.gateway._grants
        with pytest.raises(typed.TypedStateOwnerError):
            connect()
        peer = (grant.peer_pid, grant.peer_uid, grant.peer_start_time_ticks)
        with pytest.raises(typed.TypedStateOwnerAuthorizationError):
            own.gateway._require_active_grant(
                grant, peer_identity=peer, session_id="foreign-session"
            )

        first = advance(own.api)
        assert own.api.load_cursors()["state_cid"] == first["state_cid"]
        lease = own.api.acquire_consumer_lease(operation_id="main:acquire")
        lease_args = {key: lease[key] for key in ("lease_id", "fence_epoch")}
        published = own.api.publish_receipt(
            "main:receipt",
            {"observational": True, "duration": 0.25},
            expected_revision=0,
            expected_receipt_cid="",
            operation_id="main:publish",
            **lease_args,
        )
        assert published["conflict"] is False
        assert own.api.get_receipt("main:receipt") == published["head"]
        own.api.release_consumer_lease(operation_id="main:release", **lease_args)

        # A task-only grant cannot borrow the recovery client's request scope.
        foreign = OwnerRecoveryRuntimeClient(
            task,
            repository_id="repo:one",
            target_branch="main",
            consumer_id="consumer:one",
            recovery_scope_cid=own.scope_cid,
        )
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            foreign.load_cursors()
        for request_id in ("a" * 32, "b" * 32):
            assert task.execute_database_task_command(
                "record_queue_retry", {}, command_request_id=request_id
            ) == {"observed": True}
        assert calls == [
            ("record_queue_retry", "a" * 32),
            ("record_queue_retry", "b" * 32),
        ]
        assert own.api.load_cursors()["revision"] == 1
        assert own.api.get_receipt("main:receipt")["receipt"]["duration"] == 0.25
        assert not own.gateway._legacy_merge_recovery_service._retired
    finally:
        task.close()
    with pytest.raises(typed.TypedStateOwnerError):
        connect()


def test_revoked_recovery_cannot_replay_committed_operation_on_main(recovery_owner):
    own = recovery_owner
    initial = own.api.load_cursors()
    first = advance(own.api, initial, operation_id="main:durable")
    own.gateway.revoke_grant(own.grant.grant_id)
    with pytest.raises(typed.TypedStateOwnerRemoteError) as denied:
        advance(own.api, initial, operation_id="main:durable")
    assert denied.value.error_code == "authorization_denied"
    fresh, _ = own.attach()
    assert fresh.load_cursors()["state_cid"] == first["state_cid"]
    assert advance(fresh, initial, operation_id="main:durable") == first
    with own.gateway._transaction_lock:
        assert (
            own.connection.execute(
                "SELECT COUNT(*) FROM legacy_merge_recovery_cursor_history"
            ).fetchone()[0]
            == 2
        )


def test_detached_recovery_session_denied_while_fresh_session_reads(recovery_owner):
    own = recovery_owner
    first = advance(own.api)
    with own.gateway._transaction_lock:
        own.connection.execute(
            "UPDATE client_sessions SET status='detached' WHERE owner_id=?",
            [own.grant.client_id],
        )
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        own.api.load_cursors()
    fresh, _ = own.attach()
    assert fresh.load_cursors()["state_cid"] == first["state_cid"]
    assert fresh.describe_scope() == own.scope_binding
