"""Defaults: DuckDB + Quack is the control plane; DuckLake is not."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    is_quack_transport_target,
    open_quack_transport_connection,
    quack_attach_lock_path,
    quack_transport_uri,
    resolve_quack_attach_token,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)


def test_loopback_quack_uri_is_accepted() -> None:
    assert is_quack_transport_target("quack:127.0.0.1:45123") is True
    assert is_quack_transport_target("quack://127.0.0.1:9") is True
    assert is_quack_transport_target("control.duckdb") is False
    assert is_quack_transport_target("quack:8.8.8.8:45123") is False


def test_absolutized_quack_path_is_still_transport() -> None:
    uri = "quack:127.0.0.1:45123"
    prefixed = Path.cwd() / uri
    assert is_quack_transport_target(prefixed) is True
    assert quack_transport_uri(prefixed) == uri
    collapsed = Path.cwd() / "quack:/127.0.0.1:45123"
    assert quack_transport_uri(collapsed) == "quack://127.0.0.1:45123"


def test_quack_transport_rejects_non_loopback() -> None:
    with pytest.raises(DuckDBConnectionPolicyError, match="non-loopback"):
        open_quack_transport_connection("quack:10.0.0.1:45123")


def test_quack_attach_lock_path_follows_store_id(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "control.duckdb"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", raising=False)
    assert quack_attach_lock_path("quack:127.0.0.1:41327") == (
        store.resolve().parent / "quack-owner" / "attach.lock"
    )


def test_quack_attach_retries_transient_authentication_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(tmp_path / "control.duckdb"))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "retry-token")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr("random.random", lambda: 0.0)

    class _Result:
        def fetchall(self) -> list[object]:
            return []

    class _FakeConnection:
        attaches = 0

        def execute(self, sql: str) -> _Result:
            if str(sql).startswith("ATTACH"):
                type(self).attaches += 1
                if type(self).attaches < 3:
                    raise RuntimeError("Invalid Input Error: Authentication failed")
            return _Result()

        def close(self) -> None:
            return None

        description = None

        def fetchall(self) -> list[object]:
            return []

    class _FakeDuckDB:
        @staticmethod
        def connect(_target: object) -> _FakeConnection:
            return _FakeConnection()

    monkeypatch.setitem(__import__("sys").modules, "duckdb", _FakeDuckDB())
    wrapped = open_quack_transport_connection("quack:127.0.0.1:41327")
    assert _FakeConnection.attaches == 3
    assert wrapped._default_catalog == "control_plane"


def test_intent_repository_reuses_quack_read_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens: list[object] = []

    class _FakeConnection:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            return type("R", (), {"fetchall": lambda self: []})()

        def close(self) -> None:
            return None

        description = None

        def fetchall(self) -> list[object]:
            return []

    def _fake_open(path: object, **_kwargs: object) -> _FakeConnection:
        opens.append(path)
        return _FakeConnection()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository.open_duckdb_connection",
        _fake_open,
    )
    repo = IntentRepository("quack:127.0.0.1:45123", install_schema=False)
    with repo._connection():
        pass
    with repo._connection():
        pass
    assert opens == ["quack:127.0.0.1:45123"]
    repo.close()


def test_resolve_quack_attach_token_follows_env_secret_handle() -> None:
    assert (
        resolve_quack_attach_token(
            "",
            environment={
                "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": (
                    "env://QUACK_TOKEN"
                ),
                "QUACK_TOKEN": "handle-target-token",
            },
        )
        == "handle-target-token"
    )
    assert (
        resolve_quack_attach_token(
            "",
            environment={
                "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "direct-token",
                "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": (
                    "env://QUACK_TOKEN"
                ),
                "QUACK_TOKEN": "handle-target-token",
            },
        )
        == "direct-token"
    )
    assert resolve_quack_attach_token("explicit-token", environment={}) == (
        "explicit-token"
    )


def test_file_adapter_refuses_to_create_quack_named_database(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DuckDBConnectionPolicyError, match="cannot be opened as DuckDB files"):
        DuckDBConnection("quack:127.0.0.1:45123")
    assert not (tmp_path / "quack:127.0.0.1:45123").exists()


def test_database_task_source_keeps_quack_uri(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeIntent:
        database_path = "quack:127.0.0.1:45123"

        def close(self) -> None:
            return None

    def _fake_open(path, **_kwargs):
        seen["path"] = path
        return _FakeIntent()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.open_intent_repository",
        _fake_open,
    )
    source = DatabaseTaskSource("quack:127.0.0.1:45123", install_schema=False)
    assert seen["path"] == "quack:127.0.0.1:45123"
    assert source.database_path == "quack:127.0.0.1:45123"
    assert IntentRepository("quack:127.0.0.1:45123", install_schema=False)._quack_transport is True
    prefixed = Path.cwd() / "quack:127.0.0.1:45123"
    repo = IntentRepository(prefixed, install_schema=False)
    assert repo._quack_transport is True
    assert repo._open_target == "quack:127.0.0.1:45123"


def test_quack_mutation_dir_follows_store_id(tmp_path, monkeypatch) -> None:
    store = tmp_path / "control.duckdb"
    store.write_bytes(b"")
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        quack_owner_mutation_dir,
    )

    assert quack_owner_mutation_dir() == store.resolve().parent / "quack-owner" / "mutations"


def test_database_daemon_defaults_to_quack_and_refuses_file_open(tmp_path) -> None:
    with pytest.raises(DatabaseImplementationAuthorityError, match="loopback quack"):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            task_source_kind="duckdb",
        )
