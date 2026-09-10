"""Merged native command grants retain bounded expiry and dispatch admission."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as owner
from test.api.causal_federation.test_typed_state_owner import _gateway, _install


@pytest.mark.parametrize("database_task", [False, True])
def test_expiry_retires_both_maps_without_reentering_grant_lock(database_task):
    # This isolated grant table deliberately has no live owner or DB handle.
    # A daemon thread gives the old deadlock a bounded, nonblocking failure.
    gateway = object.__new__(owner.TypedStateOwnerGateway)
    gateway._grants_lock = threading.Lock()
    gateway._revoked_grants = set()
    now_ms = int(time.time() * 1000)
    grant = owner.OwnerClientGrant(
        grant_id="test-expired", client_id="test-client", process_birth_id="test-birth",
        allowed_operations=frozenset(), allowed_command_operations=frozenset(),
        allowed_database_task_commands=frozenset({"record_queue_retry"}) if database_task else frozenset(),
        peer_pid=os.getpid(), peer_uid=os.getuid(), peer_start_time_ticks=1,
        issued_at=now_ms - 2000, expires_at=now_ms - 1000,
    )
    gateway._grants = {"unused-test-token": grant}
    gateway._database_task_session_grants = {"test-session": grant}
    errors = []
    finished = threading.Event()
    def expire():
        try:
            gateway._require_active_grant(
                grant, peer_identity=(grant.peer_pid, grant.peer_uid, grant.peer_start_time_ticks),
                session_id="test-session",
            )
        except (owner.TypedStateOwnerError, OSError) as exc:
            errors.append(exc)
        finally:
            finished.set()
    thread = threading.Thread(target=expire, daemon=True)
    thread.start()
    assert finished.wait(1), "grant expiry deadlocked its nonreentrant lock"
    thread.join(1)
    assert len(errors) == 1 and isinstance(errors[0], owner.TypedStateOwnerAuthorizationError)
    assert "expired" in str(errors[0])
    assert not gateway._grants and not gateway._database_task_session_grants
    assert grant.grant_id in gateway._revoked_grants
    assert gateway._grants_lock.acquire(timeout=0.1)
    gateway._grants_lock.release()


@pytest.mark.parametrize("denial", ["revoked", "expired"])
def test_queued_database_command_rechecks_exact_session_before_handler(tmp_path, monkeypatch, denial):
    database = tmp_path / "isolated.duckdb"
    _install(database)
    gateway, connection = _gateway(database, tmp_path / "owner.sock")
    connection.execute("CREATE TABLE review_dispatch_effects (request_id VARCHAR)")
    calls = []
    def handler(command, payload, request_id, grant):
        calls.append(command)
        connection.execute("INSERT INTO review_dispatch_effects VALUES (?)", [request_id])
        return {"observed": True}
    gateway.bind_database_task_command_handler(handler)
    birth = owner.kernel_process_birth_id()
    token, grant = gateway.issue_grant(
        client_id="test-queued-command", process_birth_id=birth, peer_pid=os.getpid(),
        allowed_database_task_commands=("record_queue_retry",),
    )
    client = owner.TypedStateOwnerConnection(
        socket_path=gateway.socket_path, token=token, client_id=grant.client_id,
        process_birth_id=birth, store_id=gateway.store_id, timeout_seconds=3,
    )
    admitted = threading.Event()
    real_require = gateway._require_active_grant
    def require(current, **kwargs):
        result = real_require(current, **kwargs)
        admitted.set()
        return result
    monkeypatch.setattr(gateway, "_require_active_grant", require)
    results, errors = [], []
    def request():
        try:
            results.append(client.execute_database_task_command(
                "record_queue_retry", {}, command_request_id="a" * 32,
            ))
        except (owner.TypedStateOwnerError, OSError) as exc:
            errors.append(exc)
    thread = threading.Thread(target=request, daemon=True)
    try:
        with gateway._transaction_lock:
            thread.start()
            assert admitted.wait(2), "request did not reach pre-lock admission"
            if denial == "revoked":
                gateway.revoke_grant(grant.grant_id)
            else:
                now_ms = int(time.time() * 1000)
                expired = replace(grant, issued_at=now_ms - 2000, expires_at=now_ms - 1000)
                with gateway._grants_lock:
                    gateway._database_task_session_grants = {
                        session: expired for session in gateway._database_task_session_grants
                    }
        thread.join(4)
        assert not thread.is_alive(), "queued command failed to terminate"
        assert not results and len(errors) == 1
        assert isinstance(errors[0], owner.TypedStateOwnerRemoteError)
        assert errors[0].error_code == "authorization_denied"
        assert calls == []
        assert connection.execute("SELECT count(*) FROM review_dispatch_effects").fetchone()[0] == 0
    finally:
        client.close()
        thread.join(4)
        gateway.stop()
        connection.close()


def test_native_database_grant_is_one_use_exact_session_and_command_scoped(tmp_path):
    database = tmp_path / "isolated.duckdb"
    _install(database)
    gateway, connection = _gateway(database, tmp_path / "owner.sock")
    calls = []
    gateway.bind_database_task_command_handler(
        lambda command, payload, request_id, grant: calls.append(command) or {"observed": True}
    )
    birth = owner.kernel_process_birth_id()
    token, grant = gateway.issue_grant(
        client_id="test-native-session", process_birth_id=birth, peer_pid=os.getpid(),
        allowed_database_task_commands=("record_queue_retry",),
    )
    def connect(client_id=grant.client_id):
        return owner.TypedStateOwnerConnection(
            socket_path=gateway.socket_path, token=token, client_id=client_id,
            process_birth_id=birth, store_id=gateway.store_id, timeout_seconds=2,
        )
    client = None
    try:
        with pytest.raises(owner.TypedStateOwnerError):
            connect("different-client")
        assert token in gateway._grants
        client = connect()
        assert token not in gateway._grants
        with pytest.raises(owner.TypedStateOwnerError):
            connect()
        peer = (grant.peer_pid, grant.peer_uid, grant.peer_start_time_ticks)
        with pytest.raises(owner.TypedStateOwnerAuthorizationError):
            gateway._require_active_grant(grant, peer_identity=peer, session_id="different-session")
        assert client.execute_database_task_command(
            "record_queue_retry", {}, command_request_id="b" * 32,
        ) == {"observed": True}
        with pytest.raises(owner.TypedStateOwnerRemoteError) as denied:
            client.execute_database_task_command(
                "record_queue_backoff", {}, command_request_id="c" * 32,
            )
        assert denied.value.error_code == "authorization_denied"
        assert calls == ["record_queue_retry"]
        client.close()
        deadline = time.monotonic() + 2
        while gateway._database_task_session_grants and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not gateway._database_task_session_grants
        with pytest.raises(owner.TypedStateOwnerError):
            connect()
    finally:
        if client is not None:
            client.close()
        gateway.stop()
        connection.close()
