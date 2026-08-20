"""Host-admitted EAAEF daemon gateway loads pinned extensions without INSTALL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
    _admitted_home_directory,
    _admitted_httpfs_extension,
    _connect_admitted_duckdb,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: str, parameters: Any = None) -> None:
        del parameters
        self.statements.append(statement)


class _FakeDuckDB:
    def __init__(self) -> None:
        self.connection = _FakeConnection()

    def connect(self, database: str) -> _FakeConnection:
        assert database == ":memory:"
        return self.connection


def _pin_extension_pair(tmp_path: Path) -> tuple[Path, Path]:
    directory = (
        tmp_path / ".duckdb" / "extensions" / "v1.5.5" / "linux_arm64"
    )
    directory.mkdir(parents=True)
    quack = directory / "quack.duckdb_extension"
    httpfs = directory / "httpfs.duckdb_extension"
    quack.write_bytes(b"quack")
    httpfs.write_bytes(b"httpfs")
    return quack, httpfs


def test_admitted_httpfs_is_the_pinned_quack_sibling(tmp_path: Path) -> None:
    quack, httpfs = _pin_extension_pair(tmp_path)
    assert _admitted_httpfs_extension(quack) == httpfs


def test_admitted_httpfs_rejects_a_missing_sibling(tmp_path: Path) -> None:
    quack = tmp_path / "quack.duckdb_extension"
    quack.write_bytes(b"quack")
    with pytest.raises(QuackDaemonGatewayError, match="httpfs"):
        _admitted_httpfs_extension(quack)


def test_admitted_home_directory_is_the_duckdb_dotdir_parent(tmp_path: Path) -> None:
    quack, _httpfs = _pin_extension_pair(tmp_path)
    assert _admitted_home_directory(quack) == tmp_path


def test_connect_admitted_duckdb_loads_httpfs_then_quack_without_install(
    tmp_path: Path,
) -> None:
    quack, httpfs = _pin_extension_pair(tmp_path)
    duckdb = _FakeDuckDB()
    connection = _connect_admitted_duckdb(duckdb, quack)
    assert connection is duckdb.connection
    escaped_home = str(tmp_path).replace("'", "''")
    escaped_httpfs = str(httpfs).replace("'", "''")
    escaped_quack = str(quack).replace("'", "''")
    assert connection.statements == [
        f"SET home_directory='{escaped_home}'",
        "SET autoinstall_known_extensions=false",
        f"LOAD '{escaped_httpfs}'",
        f"LOAD '{escaped_quack}'",
    ]
    assert all("INSTALL" not in statement for statement in connection.statements)
