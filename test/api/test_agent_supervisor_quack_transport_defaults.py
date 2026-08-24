"""Defaults: DuckDB + Quack is the control plane; DuckLake is not."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    QuackTransportContentionError,
    defer_owner_inbox_until_recycle,
    is_quack_transport_target,
    mark_owner_mutation_bounce,
    open_quack_transport_connection,
    persist_quack_attach_token_vault,
    quack_attach_error_is_contention,
    quack_owner_mutation_error_is_retryable,
    quack_token_vault_path,
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


def test_resolve_quack_attach_token_persists_missing_vault(
    tmp_path, monkeypatch
) -> None:
    store = tmp_path / "control.duckdb"
    store.write_bytes(b"")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "liveTok_value1234567890")
    assert resolve_quack_attach_token() == "liveTok_value1234567890"
    vault = quack_token_vault_path()
    assert vault is not None
    assert vault.is_file()
    assert vault.stat().st_mode & 0o777 == 0o600
    assert vault.read_text(encoding="utf-8").strip() == "liveTok_value1234567890"


def test_persist_quack_attach_token_vault_does_not_overwrite(
    tmp_path, monkeypatch
) -> None:
    vault = tmp_path / "env___IPFS_ACCELERATE_AGENT_QUACK_TOKEN.quack-token"
    vault.write_text("vaultTok_value1234567890\n", encoding="utf-8")
    vault.chmod(0o600)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", str(vault))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "staleEnv_token_value")
    assert persist_quack_attach_token_vault() == vault
    assert vault.read_text(encoding="utf-8").strip() == "vaultTok_value1234567890"


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


def test_apply_owner_command_payload_reports_losing_cas_zero() -> None:
    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        apply_owner_command_payload,
    )

    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE tasks (task_cid VARCHAR PRIMARY KEY, revision BIGINT NOT NULL)"
    )
    connection.execute("INSERT INTO tasks VALUES ('cid-043', 47)")

    reply = apply_owner_command_payload(
        connection,
        {
            "sql": (
                "UPDATE tasks SET revision = 47 "
                "WHERE task_cid = ? AND revision = ?"
            ),
            "parameters": ["cid-043", 46],
        },
    )

    assert reply == {"ok": True, "rowcount": 0}


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
    skipped = request_owner_board_unstall(wait=False)
    assert skipped == {
        "ok": True,
        "requested": False,
        "skipped": "bounce_already_pending",
        "waited": False,
    }
    assert list(inbox.glob("*.request.json")) == requests
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        BOARD_UNSTALL_COOLDOWN_NAME,
        clear_owner_board_unstall_bounce,
        owner_should_recycle_for_board_unstall,
    )

    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is True
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=10_000) is False
    clear_owner_board_unstall_bounce(inbox)
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is True
    for leftover in inbox.glob("*.request.json"):
        leftover.unlink()
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is False
    cooled = request_owner_board_unstall(wait=False)
    assert cooled == {
        "ok": True,
        "requested": False,
        "skipped": "bounce_cooldown",
        "waited": False,
    }
    assert (inbox / BOARD_UNSTALL_COOLDOWN_NAME).is_file()


def test_owner_mutation_bounce_is_not_reset(tmp_path) -> None:
    inbox = tmp_path / "mutations"
    inbox.mkdir()
    first = mark_owner_mutation_bounce(inbox, request_id="one")
    mtime = first.stat().st_mtime
    second = mark_owner_mutation_bounce(inbox, request_id="two")
    assert second == first
    assert first.stat().st_mtime == mtime


def test_defer_owner_inbox_leaves_request_and_marks_bounce(tmp_path) -> None:
    inbox = tmp_path / "mutations"
    inbox.mkdir()
    request = inbox / "abc.request.json"
    request.write_text('{"sql":"UPDATE tasks SET status = \'retrying\'"}\n', encoding="utf-8")
    assert defer_owner_inbox_until_recycle(inbox) == 1
    assert request.is_file()
    assert (inbox / "board-unstall.bounce").is_file()
    assert list(inbox.glob("*.done.json")) == []


def test_owner_recycles_for_pending_request_without_bounce(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        owner_should_recycle_for_board_unstall,
    )

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    request = inbox / "abc.request.json"
    request.write_text("{}\n", encoding="utf-8")
    old = time.time() - 20
    os.utime(request, (old, old))
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=15) is True


def test_owner_mutation_wait_outlives_recycle_cooldown() -> None:
    """One logical CAS must not time out into a second claim before recycle."""

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    assert ds.QUACK_OWNER_MUTATION_TIMEOUT_SECONDS >= (
        ds.OWNER_BOARD_UNSTALL_COOLDOWN_SECONDS + 60.0
    )


def _write_ready_owner_status(
    owner_dir: Path,
    *,
    generation: int,
    fence_epoch: int | None = None,
    server_id: str | None = None,
    database_uuid: str = "8e9f113c-24cb-4f76-a881-5dbca48fa001",
    lifecycle: str = "ready",
) -> None:
    identity_status = "ready" if lifecycle == "ready" else lifecycle
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-state-server@1",
        "interface": "QuackStateServer@1",
        "lifecycle": lifecycle,
        "identity": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/state-server-identity@1"
            ),
            "interface": "StateServerIdentity@1",
            "status": identity_status,
            "server_id": server_id or f"server:g{generation}",
            "store_id": "control.duckdb",
            "database_uuid": database_uuid,
            "schema_revision": 1,
            "schema_fingerprint": "sha256:" + "a" * 64,
            "generation": generation,
            "fence_epoch": generation if fence_epoch is None else fence_epoch,
            "process_birth_id": f"birth:g{generation}",
            "listen_uri": "quack:127.0.0.1:41347",
        },
    }
    status = owner_dir / "quack-state-server.status.json"
    status.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    status.chmod(0o600)


def test_owner_mutation_preserves_zero_rowcount(tmp_path, monkeypatch) -> None:
    """A losing queued CAS remains a conflict, never an unknown success."""

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=7)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 2.0)

    def fake_sleep(_seconds: float) -> None:
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        request = pending[0]
        done = request.with_name(request.name.replace(".request.json", ".done.json"))
        done.write_text('{"ok": true, "rowcount": 0}\n', encoding="utf-8")
        _write_ready_owner_status(tmp_path, generation=8)

    monkeypatch.setattr(ds.time, "sleep", fake_sleep)
    cursor = ds._execute_quack_owner_mutation(
        "UPDATE tasks SET revision = 47 WHERE task_cid = ? AND revision = 46",
        ["cid-043"],
        dml=True,
    )

    assert cursor.rowcount == 0


def test_owner_mutation_retries_after_listen_handle_reject(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=7)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 2.0)
    sleeps = {"n": 0}

    def fake_sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        request = pending[0]
        done = request.with_name(request.name.replace(".request.json", ".done.json"))
        if sleeps["n"] == 1:
            done.write_text(
                '{"ok": false, "error": "FatalException: mutation rejected"}\n',
                encoding="utf-8",
            )
            return
        done.write_text('{"ok": true, "rowcount": 1}\n', encoding="utf-8")
        _write_ready_owner_status(tmp_path, generation=8)

    monkeypatch.setattr(ds.time, "sleep", fake_sleep)
    cursor = ds._execute_quack_owner_mutation(
        "UPDATE tasks SET status = 'retrying' WHERE task_cid = ?",
        ["cid-021"],
        dml=True,
    )
    assert cursor.rowcount == 1
    assert sleeps["n"] >= 2


def test_owner_mutation_waits_for_new_ready_owner_then_resets_transport(
    tmp_path, monkeypatch
) -> None:
    """A pre-listen done marker cannot release the next SQL statement."""

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=11)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 2.0)
    events: list[str] = []

    def fake_reset() -> None:
        events.append("reset")

    def fake_sleep(_seconds: float) -> None:
        if "done" in events:
            if "stopped" not in events:
                _write_ready_owner_status(
                    tmp_path,
                    generation=11,
                    lifecycle="stopped",
                )
                events.append("stopped")
                return
            _write_ready_owner_status(tmp_path, generation=12)
            events.append("ready")
            return
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        request = pending[0]
        done = request.with_name(request.name.replace(".request.json", ".done.json"))
        if not done.is_file():
            done.write_text('{"ok": true, "rowcount": 1}\n', encoding="utf-8")
            events.append("done")

    monkeypatch.setattr(ds, "reset_quack_transport_cache", fake_reset)
    monkeypatch.setattr(ds.time, "sleep", fake_sleep)

    cursor = ds._execute_quack_owner_mutation(
        "DELETE FROM evidence_nodes WHERE evidence_id = ?",
        ["evidence:043"],
        dml=True,
    )

    assert cursor.rowcount == 1
    assert events == ["done", "stopped", "ready", "reset"]


def test_same_quack_wrapper_rebinds_for_insert_owner_dml_insert(
    tmp_path, monkeypatch
) -> None:
    """The live validation shape survives an owner generation transition."""

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=51)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 2.0)
    uri = "quack:127.0.0.1:41347"
    durable_inserts: list[tuple[int, str]] = []
    raw_connections: list[object] = []
    owner_generation = {"value": 51}
    ordering = {"fresh_begin_before_publish": False}

    class _RecyclingQuackRaw:
        def __init__(self, generation: int) -> None:
            self.generation = generation
            self.description = None
            self.closed = False

        def execute(self, sql, params=None):
            del params
            if self.closed:
                raise RuntimeError("Connection already closed")
            normalized = " ".join(str(sql).strip().upper().split())
            if self.generation != owner_generation["value"]:
                raise RuntimeError("Invalid connection id")
            self.description = None
            if normalized == "BEGIN TRANSACTION" and self.generation == 52:
                assert ds._QUACK_TRANSPORT_CACHE[uri] is connection
                assert connection._connection is raw_connections[0]
                ordering["fresh_begin_before_publish"] = True
            if normalized.startswith("INSERT "):
                durable_inserts.append((self.generation, normalized))
            return self

        def fetchall(self):
            return []

        def close(self) -> None:
            self.closed = True

        def commit(self) -> None:
            if self.closed:
                raise RuntimeError("Invalid connection id")

        def rollback(self) -> None:
            return None

    def fake_attach(_uri: str, _secret: str):
        raw = _RecyclingQuackRaw(owner_generation["value"])
        raw_connections.append(raw)
        return raw

    transition = {"done": False, "ready": False}

    def fake_sleep(_seconds: float) -> None:
        if transition["done"] and not transition["ready"]:
            owner_generation["value"] = 52
            _write_ready_owner_status(tmp_path, generation=52)
            transition["ready"] = True
            return
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        done = pending[0].with_name(
            pending[0].name.replace(".request.json", ".done.json")
        )
        done.write_text('{"ok": true, "rowcount": 1}\n', encoding="utf-8")
        _write_ready_owner_status(tmp_path, generation=51, lifecycle="stopped")
        transition["done"] = True

    reset_quack_transport_cache()
    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    monkeypatch.setattr(ds.time, "sleep", fake_sleep)
    try:
        connection = open_quack_transport_connection(uri)
        wrapper_identity = id(connection)
        connection.execute("BEGIN TRANSACTION")
        connection.execute("INSERT INTO validation_runs VALUES ('before')")
        connection.execute(
            "DELETE FROM evidence_nodes WHERE evidence_id = ?",
            ["evidence:043"],
        )
        assert id(connection) == wrapper_identity
        assert connection._transaction_active is True
        assert connection._connection is raw_connections[1]
        connection.execute("INSERT INTO evidence_nodes VALUES ('after')")
        connection.execute("COMMIT")

        assert len(raw_connections) == 2
        assert raw_connections[0].closed is True
        assert ordering["fresh_begin_before_publish"] is True
        assert durable_inserts == [
            (51, "INSERT INTO VALIDATION_RUNS VALUES ('BEFORE')"),
            (52, "INSERT INTO EVIDENCE_NODES VALUES ('AFTER')"),
        ]
        assert open_quack_transport_connection(uri) is connection
    finally:
        reset_quack_transport_cache()


def test_quack_rebind_begin_failure_is_targeted_and_fail_closed(
    monkeypatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    first_uri = "quack:127.0.0.1:41401"
    other_uri = "quack:127.0.0.1:41402"
    attaches: dict[str, list[object]] = {first_uri: [], other_uri: []}

    class _Raw:
        def __init__(self, *, fail_begin: bool = False) -> None:
            self.fail_begin = fail_begin
            self.description = None
            self.closed = False

        def execute(self, sql, params=None):
            del params
            if self.closed:
                raise RuntimeError("connection closed")
            normalized = " ".join(str(sql).strip().upper().split())
            if self.fail_begin and normalized == "BEGIN TRANSACTION":
                raise RuntimeError("begin rejected")
            self.description = None
            return self

        def fetchall(self):
            return []

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    def fake_attach(uri: str, _secret: str):
        raw = _Raw(fail_begin=uri == first_uri and bool(attaches[uri]))
        attaches[uri].append(raw)
        return raw

    reset_quack_transport_cache()
    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    try:
        first = open_quack_transport_connection(first_uri)
        other = open_quack_transport_connection(other_uri)
        first.execute("BEGIN TRANSACTION")

        with pytest.raises(
            DuckDBConnectionPolicyError,
            match="could not restore transaction",
        ):
            ds._rebind_quack_transport_connection(first)

        assert first._closed is True
        assert first_uri not in ds._QUACK_TRANSPORT_CACHE
        assert attaches[first_uri][0].closed is True
        assert attaches[first_uri][1].closed is True
        assert ds._QUACK_TRANSPORT_CACHE[other_uri] is other
        assert attaches[other_uri][0].closed is False
        assert open_quack_transport_connection(other_uri) is other
    finally:
        reset_quack_transport_cache()


def test_quack_rebind_serializes_same_wrapper_open(monkeypatch) -> None:
    import threading

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41403"
    begin_entered = threading.Event()
    release_begin = threading.Event()
    opener_done = threading.Event()
    raws: list[object] = []

    class _Raw:
        def __init__(self, *, block_begin: bool) -> None:
            self.block_begin = block_begin
            self.description = None
            self.closed = False

        def execute(self, sql, params=None):
            del params
            if self.closed:
                raise RuntimeError("connection closed")
            normalized = " ".join(str(sql).strip().upper().split())
            if self.block_begin and normalized == "BEGIN TRANSACTION":
                begin_entered.set()
                assert release_begin.wait(timeout=2.0)
            self.description = None
            return self

        def fetchall(self):
            return []

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    def fake_attach(_uri: str, _secret: str):
        raw = _Raw(block_begin=bool(raws))
        raws.append(raw)
        return raw

    reset_quack_transport_cache()
    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    errors: list[BaseException] = []
    opened: list[DuckDBConnection] = []
    try:
        connection = open_quack_transport_connection(uri)
        connection.execute("BEGIN TRANSACTION")

        def rebind() -> None:
            try:
                ds._rebind_quack_transport_connection(connection)
            except BaseException as exc:
                errors.append(exc)

        def open_same() -> None:
            try:
                opened.append(open_quack_transport_connection(uri))
            except BaseException as exc:
                errors.append(exc)
            finally:
                opener_done.set()

        rebind_thread = threading.Thread(target=rebind)
        rebind_thread.start()
        assert begin_entered.wait(timeout=2.0)
        opener_thread = threading.Thread(target=open_same)
        opener_thread.start()
        assert opener_done.wait(timeout=0.05) is False
        release_begin.set()
        rebind_thread.join(timeout=2.0)
        opener_thread.join(timeout=2.0)

        assert rebind_thread.is_alive() is False
        assert opener_thread.is_alive() is False
        assert errors == []
        assert opened == [connection]
        assert connection._connection is raws[1]
        assert raws[0].closed is True
    finally:
        release_begin.set()
        reset_quack_transport_cache()


def test_owner_mutation_handoff_times_out_on_stale_ready_owner(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=20)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    ticks = iter((0.0, 0.0, 0.1, 0.2, 0.3))
    monkeypatch.setattr(ds.time, "monotonic", lambda: next(ticks, 1.0))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 0.25)

    def fake_sleep(_seconds: float) -> None:
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        done = pending[0].with_name(
            pending[0].name.replace(".request.json", ".done.json")
        )
        done.write_text('{"ok": true, "rowcount": 1}\n', encoding="utf-8")

    monkeypatch.setattr(ds.time, "sleep", fake_sleep)
    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="timed out waiting for matching newer ready",
    ):
        ds._execute_quack_owner_mutation(
            "DELETE FROM evidence_nodes WHERE evidence_id = ?",
            ["evidence:timeout"],
            dml=True,
        )


def test_owner_mutation_handoff_rejects_database_fence_drift(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=30)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 2.0)

    def fake_sleep(_seconds: float) -> None:
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        done = pending[0].with_name(
            pending[0].name.replace(".request.json", ".done.json")
        )
        done.write_text('{"ok": true, "rowcount": 1}\n', encoding="utf-8")
        _write_ready_owner_status(
            tmp_path,
            generation=31,
            database_uuid="4af3d1ae-a739-47c2-a48b-b846ae327dfa",
        )

    monkeypatch.setattr(ds.time, "sleep", fake_sleep)
    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="handoff fence mismatch: database_uuid",
    ):
        ds._execute_quack_owner_mutation(
            "DELETE FROM evidence_nodes WHERE evidence_id = ?",
            ["evidence:drift"],
            dml=True,
        )


def test_owner_mutation_rejects_status_for_different_transport(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=35)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    connection = DuckDBConnection.wrap(_FakeQuackRaw())
    connection._transport_uri = "quack:127.0.0.1:41348"

    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="status does not fence the active transport",
    ):
        ds._execute_quack_owner_mutation(
            "DELETE FROM evidence_nodes WHERE evidence_id = ?",
            ["evidence:wrong-endpoint"],
            dml=True,
            transport_connection=connection,
        )
    assert list(inbox.glob("*.request.json")) == []


def test_owner_mutation_nonretryable_error_does_not_wait_or_reset(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    inbox = tmp_path / "mutations"
    inbox.mkdir()
    _write_ready_owner_status(tmp_path, generation=40)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setattr(ds, "QUACK_OWNER_MUTATION_TIMEOUT_SECONDS", 2.0)
    reset = {"called": False}

    def fake_sleep(_seconds: float) -> None:
        pending = list(inbox.glob("*.request.json"))
        if not pending:
            return
        done = pending[0].with_name(
            pending[0].name.replace(".request.json", ".done.json")
        )
        done.write_text(
            '{"ok": false, "error": "Constraint Error: stale fence"}\n',
            encoding="utf-8",
        )

    def fake_reset() -> None:
        reset["called"] = True

    monkeypatch.setattr(ds.time, "sleep", fake_sleep)
    monkeypatch.setattr(ds, "reset_quack_transport_cache", fake_reset)
    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="owner mutation failed: Constraint Error: stale fence",
    ):
        ds._execute_quack_owner_mutation(
            "DELETE FROM evidence_nodes WHERE evidence_id = ?",
            ["evidence:stale"],
            dml=True,
        )
    assert reset["called"] is False


def test_attach_connect_failure_is_contention() -> None:
    assert quack_attach_error_is_contention(
        RuntimeError("IO Error: Could not connect to server error for HTTP POST")
    )
    assert quack_attach_error_is_contention(
        RuntimeError("Failed to send message: IO Error")
    )
    assert quack_owner_mutation_error_is_retryable(
        "FatalException: mutation rejected"
    )


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
