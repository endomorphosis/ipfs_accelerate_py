"""Broker-bound commands cannot silently revert to filesystem authority."""
from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source as source,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    duckdb_state as state,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_state_owner as typed,
)


@pytest.fixture
def native_route(tmp_path, monkeypatch):
    monkeypatch.setenv(typed.TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV, str(tmp_path / "broker.sock"))
    monkeypatch.setenv(typed.TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV, "99")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(tmp_path / "control.duckdb"))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "ambient-must-not-be-used")
    events = []
    monkeypatch.setattr(state, "quack_owner_command_dir", lambda *a: pytest.fail("native route fell through to legacy inbox"))
    monkeypatch.setattr(state, "resolve_quack_attach_token", lambda *a: pytest.fail("native route borrowed ambient token"))
    monkeypatch.setattr(state, "reset_quack_transport_cache", lambda: events.append("cache_reset"))
    monkeypatch.setattr(typed, "kernel_process_birth_id", lambda: "exact-test-birth")
    monkeypatch.setattr(typed, "typed_owner_socket_path", lambda store: tmp_path / "owner.sock")
    def credential(**kwargs):
        events.append(("credential", kwargs))
        return "test-issued-token"
    monkeypatch.setattr(typed, "request_database_task_command_credential", credential)
    class Connection:
        def __init__(self, **kwargs):
            events.append(("open", kwargs))
        def execute_database_task_command(self, command, payload, **kwargs):
            events.append(("execute", command, payload, kwargs))
            return {"accepted": True}
        def close(self):
            events.append("close")
    monkeypatch.setattr(typed, "TypedStateOwnerConnection", Connection)
    return events, Connection


def test_native_socket_route_preserves_explicit_request_identity(native_route):
    events, _ = native_route
    assert state.submit_quack_owner_command(
        "record_queue_retry", {"task_cid": "task:one"}, request_id="a" * 32,
    ) == {"accepted": True}
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "credential", "open", "execute", "close", "cache_reset",
    ]
    assert events[0][1]["process_birth_id"] == "exact-test-birth"
    assert events[1][1]["token"] == "test-issued-token"
    assert events[2][3] == {"command_request_id": "a" * 32}


@pytest.mark.parametrize("missing", [typed.TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
                                     typed.TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
                                     "IPFS_ACCELERATE_AGENT_STATE_STORE_ID"])
def test_partial_native_binding_denies_without_legacy_fallback(native_route, monkeypatch, missing):
    events, _ = native_route
    monkeypatch.delenv(missing)
    with pytest.raises(state.DuckDBConnectionPolicyError):
        state.submit_quack_owner_command("record_queue_retry", {"task_cid": "task:one"})
    assert events == []


@pytest.mark.parametrize("failure", ["credential", "remote", "unknown", "malformed"])
def test_native_denials_and_unknowns_never_fall_back(native_route, monkeypatch, failure):
    events, Connection = native_route
    def execute(self, *args, **kwargs):
        if failure == "remote":
            raise typed.TypedStateOwnerRemoteError("authorization_denied", "test-denial")
        if failure == "unknown":
            raise typed.TypedStateOwnerDatabaseTaskOutcomeUnknownError(kwargs["command_request_id"])
    monkeypatch.setattr(Connection, "execute_database_task_command", execute)
    if failure == "credential":
        def deny(**kwargs):
            raise typed.TypedStateOwnerAuthorizationError("test-denial")
        monkeypatch.setattr(typed, "request_database_task_command_credential", deny)
    expected = state.DuckDBConnectionPolicyError if failure == "credential" else state.QuackOwnerCommandRemoteError
    with pytest.raises(expected) as denied:
        state.submit_quack_owner_command(
            "record_queue_retry", {"task_cid": "task:one"}, request_id="b" * 32,
        )
    assert "cache_reset" not in events
    if failure != "credential":
        assert "close" in events
        assert denied.value.request_id == "b" * 32
        assert denied.value.code == ("authorization_denied" if failure == "remote" else "unknown_external_outcome")


@pytest.mark.parametrize("request_id", ["A" * 32, "too-short"])
def test_invalid_native_request_id_denies_before_credential(native_route, request_id):
    events, _ = native_route
    with pytest.raises(state.DuckDBConnectionPolicyError, match="request_id"):
        state.submit_quack_owner_command(
            "record_queue_retry", {"task_cid": "task:one"}, request_id=request_id,
        )
    assert events == []


def test_native_unknown_result_keeps_database_source_reconciliation_identity():
    error = state.QuackOwnerCommandRemoteError(
        "unknown_external_outcome", "reconcile exact native command", request_id="c" * 32,
    )
    with pytest.raises(source.TaskSourceUnknownOutcomeError) as unknown:
        source._raise_typed_owner_error(error)
    assert unknown.value.request_id == "c" * 32
