"""Focused tests for the supervisor's typed state-owner read handoff."""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source,
    quack_state_client,
    state_owner_bootstrap,
    typed_database_task_source,
    typed_state_owner,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
)

_ENDPOINT = "quack:127.0.0.1:41327"
_STORE_ID = "campaign.duckdb"


def _supervisor(listener: socket.socket, *, owner_session_id: str) -> Any:
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(
        plan_bound_dispatch=False,
        state_owner_bootstrap_fd=listener.fileno(),
        state_owner_bootstrap_store_id=_STORE_ID,
        database_owner_session_id=owner_session_id,
        database_program=SimpleNamespace(
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_endpoint=_ENDPOINT,
            store_id=_STORE_ID,
            store_generation="generation-1",
            schema_revision="1",
        ),
    )
    supervisor._state_owner_task_source = None
    supervisor._state_owner_task_source_binding = {}
    supervisor._state_owner_task_source_lock = threading.Lock()
    return supervisor


def _listener(tmp_path: Path) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(tmp_path / "state-owner-bootstrap.sock"))
    listener.listen(4)
    return listener


def _install_fake_typed_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    original_fd: int,
) -> tuple[list[dict[str, Any]], type[Any]]:
    requests: list[dict[str, Any]] = []

    class Credentials:
        endpoint = _ENDPOINT
        socket_path = "/tmp/fake-typed-owner.sock"
        store_id = _STORE_ID
        server_id = "server:test"
        client_id = ""
        process_birth_id = "birth:test"
        token = "typed-token-0123456789"

    def request(descriptor: int, *, client_id: str, store_id: str) -> Any:
        assert descriptor != original_fd
        duplicate = socket.socket(fileno=descriptor)
        assert duplicate.family == socket.AF_UNIX
        assert duplicate.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
        duplicate.close()
        Credentials.client_id = client_id
        requests.append(
            {
                "descriptor": descriptor,
                "client_id": client_id,
                "store_id": store_id,
            }
        )
        return Credentials()

    class FakeConnection:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.attached = False
            self.closed = False
            self.connection = None

        def attach(self, endpoint: str, *, server_id: str) -> None:
            assert endpoint == _ENDPOINT
            assert server_id == "server:test"
            self.connection = self.kwargs["connection_factory"](endpoint)
            self.attached = True

        def close(self) -> None:
            self.closed = True

    class FakeTaskSource:
        aliases = ("AUTO-001", "AUTO-002")

        def __init__(self, client: FakeClient) -> None:
            assert client.attached
            self.client = client
            self.closed = False

        def snapshot(self) -> Any:
            return SimpleNamespace(task_count=len(self.aliases))

        def list_tasks(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                tasks=tuple(
                    SimpleNamespace(task_alias=alias, task_cid=f"cid:{alias}")
                    for alias in self.aliases
                ),
                next_cursor="",
            )

        def get_task(self, task_id: str) -> Any:
            return SimpleNamespace(task_alias=task_id)

        def close(self) -> None:
            self.closed = True
            self.client.close()

    monkeypatch.setattr(
        state_owner_bootstrap,
        "request_state_owner_bootstrap",
        request,
    )
    monkeypatch.setattr(quack_state_client, "QuackStateClient", FakeClient)
    monkeypatch.setattr(
        typed_database_task_source,
        "TypedDatabaseTaskSource",
        FakeTaskSource,
    )
    monkeypatch.setattr(
        typed_state_owner,
        "TypedStateOwnerConnection",
        FakeConnection,
    )
    return requests, FakeTaskSource


def test_supervisor_bootstraps_typed_reader_over_duplicate_without_ambient_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener(tmp_path)
    try:
        supervisor = _supervisor(
            listener,
            owner_session_id="campaign:shard:1-of-3:track:abcdef012345",
        )
        requests, _source_class = _install_fake_typed_stack(
            monkeypatch,
            original_fd=listener.fileno(),
        )
        monkeypatch.setenv(
            typed_state_owner.TYPED_STATE_OWNER_TOKEN_ENV,
            "unrelated-sentinel-token",
        )
        monkeypatch.setenv(
            typed_state_owner.TYPED_STATE_OWNER_SOCKET_ENV,
            "/tmp/unrelated-sentinel.sock",
        )

        source = supervisor._typed_supervisor_task_source()
        assert supervisor._typed_supervisor_task_source() is source

        assert len(requests) == 1
        assert requests[0]["client_id"] == (
            "database-implementation-supervisor:"
            "campaign:shard:1-of-3:track:abcdef012345"
        )
        assert requests[0]["store_id"] == _STORE_ID
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
        assert os.environ[typed_state_owner.TYPED_STATE_OWNER_TOKEN_ENV] == (
            "unrelated-sentinel-token"
        )
        assert os.environ[typed_state_owner.TYPED_STATE_OWNER_SOCKET_ENV] == (
            "/tmp/unrelated-sentinel.sock"
        )
        assert source.client.connection.kwargs == {
            "socket_path": Path("/tmp/fake-typed-owner.sock"),
            "token": "typed-token-0123456789",
            "client_id": requests[0]["client_id"],
            "process_birth_id": "birth:test",
            "store_id": _STORE_ID,
            "timeout_seconds": 30.0,
        }
        assert (
            supervisor._state_owner_task_source_binding["credential_in_environment"]
            is False
        )

        supervisor.close()
        assert source.closed is True
        assert source.client.closed is True
    finally:
        listener.close()


def test_objective_count_uses_cached_typed_reader_not_direct_quack_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener(tmp_path)
    try:
        supervisor = _supervisor(listener, owner_session_id="campaign:lane:2")
        requests, _source_class = _install_fake_typed_stack(
            monkeypatch,
            original_fd=listener.fileno(),
        )
        monkeypatch.setattr(
            database_task_source,
            "DatabaseTaskSource",
            lambda *_args, **_kwargs: pytest.fail(
                "bootstrap path opened the direct database adapter"
            ),
        )
        todo_text = "# Board\n\n" + "".join(
            f"## AUTO-{index:03d} Task\n\n- Status: pending\n\n"
            for index in range(1, 4)
        )

        count, authority = supervisor._objective_refill_authoritative_task_count(
            todo_text,
            task_prefix="AUTO-",
        )

        assert count == 3
        assert authority["canonical_task_count"] == 2
        assert authority["projection_only_task_count"] == 1
        assert authority["task_source_transport"] == ("pid_bound_typed_state_owner")
        assert len(requests) == 1
        supervisor.close()
    finally:
        listener.close()


def test_supervisor_bootstrap_rejects_missing_owner_session_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener(tmp_path)
    try:
        supervisor = _supervisor(listener, owner_session_id="")
        monkeypatch.setattr(
            state_owner_bootstrap,
            "request_state_owner_bootstrap",
            lambda *_args, **_kwargs: pytest.fail(
                "missing owner session reached bootstrap broker"
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="requires an explicit database owner session",
        ):
            supervisor._typed_supervisor_task_source()
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        listener.close()


def test_plan_bound_supervisor_retains_existing_database_adapter_path(
    tmp_path: Path,
) -> None:
    listener = _listener(tmp_path)
    try:
        supervisor = _supervisor(listener, owner_session_id="campaign")
        supervisor.config.plan_bound_dispatch = True

        assert supervisor._uses_supervisor_state_owner_bootstrap() is False
    finally:
        listener.close()


def test_connect_inherited_listener_retries_blockingioerror_until_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener(tmp_path)
    attempts = {"count": 0}

    def blocked_connect(self: socket.socket, address: object) -> None:
        attempts["count"] += 1
        raise BlockingIOError(11, "Resource temporarily unavailable")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    try:
        duplicate = os.dup(listener.fileno())
        started = time.monotonic()
        with pytest.raises(
            state_owner_bootstrap.StateOwnerBootstrapError,
            match="could not be opened",
        ):
            state_owner_bootstrap._connect_inherited_listener(
                duplicate,
                timeout_seconds=0.3,
            )
        assert time.monotonic() - started >= 0.2
        assert attempts["count"] >= 2
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        listener.close()


def test_supervisor_backs_off_when_quack_owner_is_down(tmp_path: Path) -> None:
    listener = _listener(tmp_path)
    try:
        supervisor = _supervisor(listener, owner_session_id="campaign")
        supervisor.config.database_program.quack_endpoint = "quack:127.0.0.1:1"
        supervisor.config.check_interval = 20.0
        assert supervisor._quack_owner_reachable() is False
        assert supervisor._supervisor_loop_recovery_delay_seconds() >= 15.0
    finally:
        listener.close()
