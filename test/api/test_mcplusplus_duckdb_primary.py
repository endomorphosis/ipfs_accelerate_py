"""Prove MCP++ durable/state stores default to DuckDB/Quack/DuckLake, not SQLite."""

from __future__ import annotations

import inspect
from pathlib import Path

from ipfs_accelerate_py.mcp_server.mcplusplus.durable.journal import (
    ADAPTER_ID,
    DurableJournal,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.state.sqlite_authority import (
    PROVIDER_ID,
    SqliteAuthorityState,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.storage import engine as engine_mod
from ipfs_accelerate_py.mcp_server.mcplusplus.storage.engine import (
    DEFAULT_ENGINE,
    connect_sql_engine,
    resolve_sql_engine,
)


def test_resolver_defaults_to_duckdb() -> None:
    assert DEFAULT_ENGINE == "duckdb"
    assert resolve_sql_engine() == "duckdb"


def test_resolver_aliases_quack_ducklake_and_sqlite() -> None:
    assert resolve_sql_engine("quack") == "duckdb"
    assert resolve_sql_engine("ducklake") == "duckdb"
    assert resolve_sql_engine("duckdb-quack") == "duckdb"
    assert resolve_sql_engine("sqlite") == "sqlite"


def test_extension_loader_uses_local_load_never_install() -> None:
    src = inspect.getsource(engine_mod._load_local_extensions)
    assert "LOAD" in src
    assert "execute" in src
    assert 'f"INSTALL' not in src and "INSTALL {" not in src
    assert engine_mod.PRIMARY_EXTENSIONS == ("quack", "ducklake")


def test_durable_journal_defaults_to_duckdb(tmp_path: Path) -> None:
    journal = DurableJournal.open(tmp_path / "journal.duckdb")
    try:
        assert journal.engine == "duckdb"
        assert journal.journal_mode() == "duckdb"
        assert ADAPTER_ID == "duckdb-quack-journal@1"
        assert journal.db_version() == DurableJournal.DB_VERSION
        assert isinstance(journal.loaded_extensions, tuple)
    finally:
        journal.close()


def test_single_authority_state_defaults_to_duckdb(tmp_path: Path) -> None:
    store = SqliteAuthorityState.open(tmp_path / "authority.duckdb")
    try:
        assert store.engine == "duckdb"
        assert store.journal_mode() == "duckdb"
        assert PROVIDER_ID == "duckdb-authority"
        record = store.create("state:test/duckdb-authority", {"ok": True})
        assert record["state_ref"]["provider"] == "duckdb-authority"
        assert record["state_ref"]["mode"] == "single_authority"
    finally:
        store.close()


def test_sqlite_remains_explicit_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(engine_mod.ENGINE_ENV, "sqlite")
    assert resolve_sql_engine() == "sqlite"
    journal = DurableJournal.open(tmp_path / "journal.sqlite", engine="sqlite")
    try:
        assert journal.engine == "sqlite"
        assert journal.journal_mode() == "wal"
    finally:
        journal.close()


def test_duckdb_rewrites_integer_ddl_for_epoch_ms(tmp_path: Path) -> None:
    connection = connect_sql_engine(tmp_path / "executions.duckdb")
    try:
        assert connection.engine == "duckdb"
        connection.execute(
            "CREATE TABLE IF NOT EXISTS executions ("
            "idempotency_key TEXT PRIMARY KEY, created_at INTEGER NOT NULL)"
        )
        epoch_ms = 1_725_000_000_000
        connection.execute(
            "INSERT INTO executions VALUES (?, ?)", ("k", epoch_ms)
        )
        row = connection.execute(
            "SELECT created_at FROM executions WHERE idempotency_key=?",
            ("k",),
        ).fetchone()
        assert int(row["created_at"]) == epoch_ms
    finally:
        connection.close()
