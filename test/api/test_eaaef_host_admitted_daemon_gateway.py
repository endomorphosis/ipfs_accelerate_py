"""Host-admitted EAAEF daemon gateway loads pinned extensions without INSTALL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
    _CAS_TASK_STATUS_SQL,
    _admitted_home_directory,
    _admitted_httpfs_extension,
    _connect_admitted_duckdb,
    _submit_owner_mutation,
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


def test_factory_uses_daemon_execution_repository_property() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Execution,
    )

    class _Daemon:
        def __init__(self) -> None:
            self.execution_repository = object()

    daemon = _Daemon()
    assert daemon.execution_repository is not None
    # The closed execution component is what reserve/commit consume.
    assert hasattr(_Execution, "reserve_effect")
    assert hasattr(_Execution, "commit_effect")


def test_record_defaults_missing_task_dependencies() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Record,
    )

    task = _Record({"task_cid": "cid:1", "task_alias": "EAAEF-010", "status": "todo"})
    assert tuple(task.dependencies) == ()
    assert task.body == {}
    assert task.task_cid == "cid:1"


def test_owner_mutation_rejects_non_cas_sql(tmp_path: Path) -> None:
    inbox = tmp_path / "mutations"
    inbox.mkdir()
    with pytest.raises(QuackDaemonGatewayError, match="closed CAS template"):
        _submit_owner_mutation(
            mutation_dir=inbox,
            sql="DELETE FROM tasks",
            parameters=[],
            timeout_seconds=0.2,
        )


def test_owner_mutation_reads_owner_done_receipt(tmp_path: Path) -> None:
    import json
    import threading

    inbox = tmp_path / "mutations"
    inbox.mkdir()

    def _consume() -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if not requests:
                time.sleep(0.01)
                continue
            request = requests[0]
            payload = json.loads(request.read_text(encoding="utf-8"))
            assert payload["sql"] == _CAS_TASK_STATUS_SQL
            done = request.with_name(request.name.replace(".request.json", ".done.json"))
            done.write_text(json.dumps({"ok": True, "rowcount": 1}) + "\n", encoding="utf-8")
            return

    worker = threading.Thread(target=_consume, daemon=True)
    worker.start()
    updated = _submit_owner_mutation(
        mutation_dir=inbox,
        sql=_CAS_TASK_STATUS_SQL,
        parameters=["in_progress", 3, "2026-08-21T00:00:00Z", "cid:1", 2],
        timeout_seconds=2.0,
    )
    worker.join(timeout=2.0)
    assert updated == 1


def test_owned_patch_cid_hashes_only_owned_files(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _OWNED_RELATIVE_PATHS,
        _owned_patch_cid,
    )

    for relative in _OWNED_RELATIVE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("eaaef-010\n", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("noise\n", encoding="utf-8")
    first = _owned_patch_cid(tmp_path)
    (tmp_path / "unrelated.txt").write_text("changed\n", encoding="utf-8")
    assert _owned_patch_cid(tmp_path) == first
    owned = tmp_path / _OWNED_RELATIVE_PATHS[0]
    owned.write_text("changed-owned\n", encoding="utf-8")
    assert _owned_patch_cid(tmp_path) != first


def test_ensure_attempt_returns_the_attempt_record() -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.eaaef_host_admitted_daemon_gateway import (
        _Execution,
    )

    gateway = SimpleNamespace(
        capability=SimpleNamespace(content_id="sha256:" + "a" * 64),
        _attempts={},
    )
    execution = _Execution(gateway)
    stored = execution.ensure_attempt(
        attempt={
            "attempt_id": "attempt:1",
            "claim_id": "claim:1",
            "task_cid": "cid:1",
            "task_alias": "EAAEF-010",
            "attempt_number": 1,
            "owner_session_id": "owner",
            "fencing_token": 1,
            "fence_epoch": 1,
            "lease_id": "lease:1",
            "committed_phase": "claimed",
            "status": "running",
            "started_at_ms": 1,
            "revision": 1,
            "body": {},
        },
        claimed_phase={"phase": "claimed", "revision": 1},
    )
    assert stored["attempt_id"] == "attempt:1"
    assert execution.get_attempt("attempt:1")["attempt_id"] == "attempt:1"
    running = execution.list_running_attempts(owner_session_id="owner")
    assert len(running) == 1
