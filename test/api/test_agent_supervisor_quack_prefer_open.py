from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_ENDPOINT_ENV,
    QUACK_PREFER_ENV,
    QUACK_REQUIRE_ENV,
    QUACK_STORE_ID_ENV,
    QUACK_TOKEN_ENV,
    DuckDBConnectionPolicyError,
    discover_live_quack_endpoint,
    open_duckdb_connection,
)


@pytest.fixture(autouse=True)
def _clear_quack_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        QUACK_ENDPOINT_ENV,
        QUACK_PREFER_ENV,
        QUACK_REQUIRE_ENV,
        QUACK_STORE_ID_ENV,
        QUACK_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _write_owner_status(
    database: Path,
    *,
    lifecycle: str = "ready",
    uri: str = "quack:127.0.0.1:41417",
    pid: int | None = None,
    identity_status: str = "ready",
) -> Path:
    status_dir = database.parent / "quack-owner"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "lifecycle": lifecycle,
        "database_path": str(database),
        "listen_uri": uri,
        "identity": {
            "status": identity_status,
            "listen_uri": uri,
            "process_birth": {"pid": os.getpid() if pid is None else pid},
        },
    }
    path = status_dir / "quack-state-server.status.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _dummy_quack_connection() -> SimpleNamespace:
    return SimpleNamespace()


def test_open_falls_back_to_file_when_no_quack_owner(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    database = tmp_path / "control.duckdb"
    caplog.set_level(logging.INFO, logger=duckdb_state.__name__)
    with open_duckdb_connection(database) as connection:
        assert getattr(connection, "_transport_mode", "") == "file"
        connection.execute("CREATE TABLE items (value INTEGER)")
    assert "falling back to exclusive DuckDB file" in caplog.text
    assert "no_live_owner" in caplog.text


def test_open_prefers_discovered_ready_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"")
    _write_owner_status(database)
    attached: list[tuple[str, str]] = []

    def _attach(uri: str, *, token: str = "") -> SimpleNamespace:
        attached.append((uri, token))
        return _dummy_quack_connection()

    monkeypatch.setattr(duckdb_state, "open_quack_transport_connection", _attach)
    caplog.set_level(logging.INFO, logger=duckdb_state.__name__)
    connection = open_duckdb_connection(database)
    assert attached == [("quack:127.0.0.1:41417", "")]
    assert getattr(connection, "_transport_mode", "") == "quack"
    assert "attached" in caplog.text


def test_stopped_owner_status_falls_back_to_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database = tmp_path / "control.duckdb"
    _write_owner_status(database, lifecycle="stopped")
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda *args, **kwargs: pytest.fail("stopped owner must not be attached"),
    )
    caplog.set_level(logging.WARNING, logger=duckdb_state.__name__)
    discovery = discover_live_quack_endpoint(database)
    assert discovery.found is False
    assert "lifecycle_stopped" in str(discovery.details)
    with open_duckdb_connection(database) as connection:
        assert getattr(connection, "_transport_mode", "") == "file"
    assert "lifecycle_stopped" in caplog.text


def test_attach_failure_logs_error_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database = tmp_path / "control.duckdb"
    _write_owner_status(database)

    def _attach(uri: str, *, token: str = "") -> SimpleNamespace:
        raise RuntimeError("LOAD quack PermissionException")

    monkeypatch.setattr(duckdb_state, "open_quack_transport_connection", _attach)
    caplog.set_level(logging.WARNING, logger=duckdb_state.__name__)
    with open_duckdb_connection(database) as connection:
        assert getattr(connection, "_transport_mode", "") == "file"
    assert "attach_failed" in caplog.text
    assert "PermissionException" in caplog.text
    assert "falling back to exclusive DuckDB file" in caplog.text


def test_require_quack_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "control.duckdb"
    monkeypatch.setenv(QUACK_REQUIRE_ENV, "1")
    with pytest.raises(DuckDBConnectionPolicyError, match="no_live_owner"):
        open_duckdb_connection(database)


def test_prefer_quack_false_skips_attach_even_with_ready_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "control.duckdb"
    _write_owner_status(database)
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda *args, **kwargs: pytest.fail("prefer_quack=False must not attach"),
    )
    with open_duckdb_connection(database, prefer_quack=False) as connection:
        assert getattr(connection, "_transport_mode", "") == "file"


def test_unbound_env_endpoint_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "control.duckdb"
    monkeypatch.setenv(QUACK_ENDPOINT_ENV, "quack:127.0.0.1:41417")
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda *args, **kwargs: pytest.fail("unbound env URI must not attach"),
    )
    discovery = discover_live_quack_endpoint(database)
    assert discovery.found is False
    assert "unbound_to_database" in str(discovery.details)
    with open_duckdb_connection(database) as connection:
        assert getattr(connection, "_transport_mode", "") == "file"


def test_env_endpoint_attaches_when_store_id_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"")
    monkeypatch.setenv(QUACK_ENDPOINT_ENV, "quack:127.0.0.1:41999")
    monkeypatch.setenv(QUACK_STORE_ID_ENV, str(database))
    attached: list[str] = []

    def _attach(uri: str, *, token: str = "") -> SimpleNamespace:
        attached.append(uri)
        return _dummy_quack_connection()

    monkeypatch.setattr(duckdb_state, "open_quack_transport_connection", _attach)
    connection = open_duckdb_connection(database)
    assert attached == ["quack:127.0.0.1:41999"]
    assert getattr(connection, "_transport_mode", "") == "quack"


def test_foreign_database_status_is_ignored(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    foreign = tmp_path / "other.duckdb"
    _write_owner_status(foreign, lifecycle="stopped")
    status_dir = database.parent / "quack-owner"
    # Point the sibling status at a different store.
    payload = json.loads((status_dir / "quack-state-server.status.json").read_text())
    payload["database_path"] = str(foreign)
    (status_dir / "quack-state-server.status.json").write_text(json.dumps(payload) + "\n")
    discovery = discover_live_quack_endpoint(database)
    assert discovery.found is False
    assert discovery.reason == "no_live_owner"
    assert "status_missing" in str(discovery.details)


def test_dead_owner_pid_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    _write_owner_status(database, pid=999999999)
    discovery = discover_live_quack_endpoint(database)
    assert discovery.found is False
    assert "owner_process_not_alive" in str(discovery.details)


def test_quack_owner_file_open_skips_prefer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "control.duckdb"
    _write_owner_status(database)
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda *args, **kwargs: pytest.fail("exclusive owner must not attach to itself"),
    )
    with open_duckdb_connection(database, quack_owner=True) as connection:
        assert getattr(connection, "_transport_mode", "") == "file"
