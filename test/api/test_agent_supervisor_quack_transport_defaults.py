"""Defaults: DuckDB + Quack is the control plane; DuckLake is not."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    QuackTransportContentionError,
    is_quack_transport_target,
    open_quack_transport_connection,
    quack_transport_uri,
    reset_quack_transport_cache,
    resolve_quack_attach_token,
    unstall_stale_in_progress_tasks,
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


class _FakeQuackRaw:
    def execute(self, sql, params=None):
        del sql, params
        return type("Result", (), {"fetchall": lambda self: [(1,)]})()

    def close(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def test_resolve_quack_attach_token_prefers_vault_over_stale_env(
    tmp_path, monkeypatch
) -> None:
    vault = tmp_path / "env___IPFS_ACCELERATE_AGENT_QUACK_TOKEN.quack-token"
    vault.write_text("vaultTok_value1234567890\n", encoding="utf-8")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", str(vault))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "staleEnv_token_value")
    assert resolve_quack_attach_token() == "vaultTok_value1234567890"
    assert resolve_quack_attach_token("explicit_token_ok") == "explicit_token_ok"


def test_quack_attach_retries_authentication_failed_contention(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    reset_quack_transport_cache()
    attempts = {"n": 0}

    def fake_attach(uri: str, secret: str):
        del uri, secret
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Invalid Input Error: Authentication failed")
        return _FakeQuackRaw()

    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    try:
        connection = open_quack_transport_connection("quack:127.0.0.1:41347")
        assert attempts["n"] == 3
        assert connection._pooled is True
    finally:
        reset_quack_transport_cache()


def test_quack_attach_reuses_cached_connection(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    reset_quack_transport_cache()
    attaches = {"n": 0}

    def fake_attach(uri: str, secret: str):
        del uri, secret
        attaches["n"] += 1
        return _FakeQuackRaw()

    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    try:
        first = open_quack_transport_connection("quack:127.0.0.1:41347")
        first.close()
        second = open_quack_transport_connection("quack:127.0.0.1:41347")
        assert first is second
        assert attaches["n"] == 1
    finally:
        reset_quack_transport_cache()


def test_unstall_stale_in_progress_tasks_retries_dead_gate(tmp_path) -> None:
    import duckdb
    from datetime import datetime, timedelta, timezone

    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    now = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        [
            "cid-021",
            "PCCE-021",
            "in_progress",
            9,
            (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        [
            "cid-022",
            "PCCE-022",
            "in_progress",
            1,
            (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    result = unstall_stale_in_progress_tasks(connection, now=now, stale_seconds=16_200)
    aliases = [item["task_alias"] for item in result["unstalled"]]
    assert aliases == ["PCCE-021"]
    rows = {
        str(row[0]): (row[1], row[2])
        for row in connection.execute(
            "SELECT task_alias, status, revision FROM tasks"
        ).fetchall()
    }
    assert rows["PCCE-021"] == ("retrying", 10)
    assert rows["PCCE-022"] == ("in_progress", 1)


def test_unstall_drops_status_indexes_that_fatal_status_updates(tmp_path) -> None:
    import duckdb
    from datetime import datetime, timedelta, timezone

    connection = duckdb.connect(str(tmp_path / "control.duckdb"))
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL UNIQUE,
            goal_cid VARCHAR NOT NULL,
            ordinal BIGINT NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX tasks_goal_idx ON tasks(goal_cid, status)")
    connection.execute("CREATE INDEX tasks_status_idx ON tasks(status, ordinal)")
    now = datetime.now(timezone.utc)
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            "cid-021",
            "PCCE-021",
            "goal:root",
            21,
            "in_progress",
            9,
            (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    result = unstall_stale_in_progress_tasks(connection, now=now, stale_seconds=16_200)
    assert [item["task_alias"] for item in result["unstalled"]] == ["PCCE-021"]
    assert any("tasks_status_idx" in sql for sql in result["status_indexes_rebuilt"])
    row = connection.execute(
        "SELECT status, revision FROM tasks WHERE task_alias = 'PCCE-021'"
    ).fetchone()
    assert tuple(row) == ("retrying", 10)
    names = {
        str(_row[0])
        for _row in connection.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
        ).fetchall()
    }
    assert "tasks_status_idx" in names
    assert "tasks_goal_idx" in names


def test_apply_owner_command_payload_unstalls_without_client_sql(tmp_path) -> None:
    import duckdb
    from datetime import datetime, timedelta, timezone

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        apply_owner_command_payload,
    )

    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    stale = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        [
            "cid-021",
            "PCCE-021",
            "in_progress",
            9,
            stale,
        ],
    )
    reply = apply_owner_command_payload(
        connection,
        {"op": "board_unstall", "stale_seconds": 16_200},
    )
    assert reply["ok"] is True
    assert reply["rowcount"] == 1
    assert reply["board_unstall"]["unstalled"][0]["task_alias"] == "PCCE-021"
    status = connection.execute(
        "SELECT status, revision FROM tasks WHERE task_alias = 'PCCE-021'"
    ).fetchone()
    assert tuple(status) == ("retrying", 10)


def test_request_owner_board_unstall_writes_inbox_without_waiting(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        request_owner_board_unstall,
    )

    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    result = request_owner_board_unstall(wait=False)
    assert result["ok"] is True
    assert result["requested"] is True
    requests = list(inbox.glob("*.request.json"))
    assert len(requests) == 1
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert payload == {"op": "board_unstall", "stale_seconds": 16_200}
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        clear_owner_board_unstall_bounce,
        owner_should_recycle_for_board_unstall,
    )

    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is True
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=10_000) is False
    clear_owner_board_unstall_bounce(inbox)
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is False


def test_quack_attach_exhausted_contention_raises_typed_error(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    reset_quack_transport_cache()

    def fake_attach(uri: str, secret: str):
        del uri, secret
        raise RuntimeError("Invalid Input Error: Authentication failed")

    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    monkeypatch.setattr(ds, "QUACK_ATTACH_ATTEMPTS", 3)
    try:
        with pytest.raises(QuackTransportContentionError, match="contended"):
            open_quack_transport_connection("quack:127.0.0.1:41347")
    finally:
        reset_quack_transport_cache()
