"""Native owner task sessions preserve capability boundaries and recovery liveness."""

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from test.api.test_agent_supervisor_owner_recovery_main_sessions import _task_session
from test.api.test_agent_supervisor_owner_recovery_runtime import (
    owner as _owner_fixture,
    recovery_owner as _recovery_owner_fixture,
)

owner = _owner_fixture
recovery_owner = _recovery_owner_fixture


def test_foreign_task_attach_does_not_consume_owner_capability(recovery_owner):
    own = recovery_owner
    birth = typed.kernel_process_birth_id()
    token, grant = own.gateway.issue_grant(
        client_id="native:task-only",
        process_birth_id=birth,
        allowed_database_task_commands=("record_queue_retry",),
    )
    peer = (grant.peer_pid, grant.peer_uid, grant.peer_start_time_ticks)
    admitted = dict(
        supplied_token=token,
        client_id=grant.client_id,
        process_birth_id=birth,
        peer_identity=peer,
        session_id="session:task:admitted",
    )
    for foreign in (
        {"client_id": "native:other"},
        {"process_birth_id": "birth:foreign"},
        {"peer_identity": (peer[0], peer[1], peer[2] + 1)},
    ):
        with pytest.raises(typed.TypedStateOwnerAuthorizationError):
            own.gateway._admit_open_grant(**{**admitted, **foreign})
        assert own.gateway._grants[token] == grant
        assert not own.gateway._database_task_session_grants
    assert own.gateway._admit_open_grant(**admitted) == grant
    assert token not in own.gateway._grants
    assert own.gateway._database_task_session_grants == {
        admitted["session_id"]: grant
    }
    own.gateway.revoke_grant(grant.grant_id)


def test_expired_task_session_is_retired_without_stalling_recovery(recovery_owner):
    own = recovery_owner
    task, connect, token, grant, calls = _task_session(own)
    try:
        expired = replace(
            grant,
            issued_at=grant.issued_at - 2_000,
            expires_at=grant.issued_at - 1_000,
        )
        with own.gateway._grants_lock:
            own.gateway._database_task_session_grants[task.session_id] = expired
        with pytest.raises(typed.TypedStateOwnerRemoteError) as denied:
            task.execute_database_task_command(
                "record_queue_retry", {}, command_request_id="c" * 32
            )
        assert denied.value.error_code == "authorization_denied"
        assert not calls
        assert task.session_id not in own.gateway._database_task_session_grants
        assert token not in own.gateway._grants
        assert own.api.load_cursors()["revision"] == 0
        with pytest.raises(typed.TypedStateOwnerError):
            connect()
    finally:
        task.close()


def test_task_revocation_after_initial_admission_prevents_handler_call(
    recovery_owner, monkeypatch
):
    own = recovery_owner
    task, _connect, _token, grant, calls = _task_session(own)
    original = own.gateway._require_active_grant
    revoked = False

    def revoke_after_admission(candidate, **kwargs):
        nonlocal revoked
        current = original(candidate, **kwargs)
        if candidate.grant_id == grant.grant_id and not revoked:
            revoked = True
            own.gateway.revoke_grant(grant.grant_id)
        return current

    monkeypatch.setattr(own.gateway, "_require_active_grant", revoke_after_admission)
    try:
        with pytest.raises(typed.TypedStateOwnerRemoteError) as denied:
            task.execute_database_task_command(
                "record_queue_retry", {}, command_request_id="d" * 32
            )
        assert denied.value.error_code == "authorization_denied"
        assert revoked and not calls
        assert task.session_id not in own.gateway._database_task_session_grants
        assert own.api.load_cursors()["revision"] == 0
    finally:
        task.close()


def test_recovery_grant_renewal_retains_native_current_window_lookup(
    recovery_owner, monkeypatch
):
    own = recovery_owner
    old = own.grant
    renewed = own.gateway.renew_grant(old.grant_id, ttl_seconds=7_200)
    peer = (old.peer_pid, old.peer_uid, old.peer_start_time_ticks)
    with monkeypatch.context() as clock:
        clock.setattr(typed.time, "time", lambda: old.expires_at / 1_000 + 1)
        assert own.gateway._require_active_grant(
            old, peer_identity=peer, session_id="session:unrelated-task"
        ) == renewed
    assert own.api.load_cursors()["revision"] == 0


def test_task_command_transport_loss_requires_outcome_reconciliation(
    recovery_owner, monkeypatch
):
    own = recovery_owner
    task, _connect, _token, _grant, _calls = _task_session(own)

    def lose_response(*args, **kwargs):
        raise OSError("response lost after possible dispatch")

    try:
        with monkeypatch.context() as transport:
            transport.setattr(task, "_request", lose_response)
            with pytest.raises(typed.TypedStateOwnerDatabaseTaskOutcomeUnknownError) as unknown:
                task.execute_database_task_command(
                    "record_queue_retry", {}, command_request_id="e" * 32
                )
        assert unknown.value.command_request_id == "e" * 32
    finally:
        task.close()
