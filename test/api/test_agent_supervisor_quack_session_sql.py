"""Real loopback Quack session SQL; every database/listener belongs to the test.

No model, paper experiment, shared runtime database, or external network is used.
Tests skip only if DuckDB or its locally installed Quack extension is unavailable.
"""
from __future__ import annotations

import secrets
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport,
    _allocate_loopback_port,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_quack_transport_connection,
)


@pytest.fixture
def live_quack(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    owner = duckdb.connect(
        str(tmp_path / "session-owner.duckdb"),
        config={"autoinstall_known_extensions": False},
    )
    try:
        owner.execute("LOAD quack")
    except Exception as exc:
        owner.close()
        pytest.skip(f"locally installed Quack extension unavailable: {type(exc).__name__}")
    owner.execute("CREATE TABLE tasks(task_id VARCHAR)")
    owner.execute("INSERT INTO tasks VALUES ('only-temporary-session-test')")
    owner.execute("CREATE TABLE records(id INTEGER PRIMARY KEY, value VARCHAR, revision INTEGER)")
    owner.execute("INSERT INTO records VALUES (1, 'initial', 1)")
    transport = InProcessQuackTransport()
    port = _allocate_loopback_port()
    token = secrets.token_urlsafe(32)
    clients = []
    identity = SimpleNamespace(
        server_id="session-test", store_id="temporary-session-store",
        database_uuid="temporary-session-uuid", schema_revision=1,
        schema_fingerprint="temporary-session-schema", generation=1,
        process_birth_id="temporary-session-birth",
    )
    try:
        transport.start(owner, host="127.0.0.1", port=port, token=token, identity=identity)
        def connect():
            try:
                client = open_quack_transport_connection(f"quack:127.0.0.1:{port}", token=token)
            except Exception as exc:
                pytest.fail(f"temporary authenticated attach failed: {type(exc).__name__}", pytrace=False)
            clients.append(client)
            return client
        yield SimpleNamespace(connect=connect, owner=owner)
    finally:
        for client in reversed(clients):
            try:
                client.close()
            except Exception:
                pass
        try:
            transport.stop(owner)
        finally:
            owner.close()


def value(client):
    return client.execute("SELECT value, revision FROM records WHERE id = ?", [1]).fetchone()


def test_remote_transaction_rolls_back_and_commits_across_calls(live_quack):
    writer = live_quack.connect()
    observer = live_quack.connect()
    writer.execute("BEGIN IMMEDIATE")
    assert writer.in_transaction
    assert writer.execute("UPDATE records SET value = ? WHERE id = ?", ["rolled back", 1]).rowcount == 1
    assert value(writer)[0] == "rolled back"
    assert value(observer)[0] == "initial"
    writer.rollback()
    assert not writer.in_transaction
    assert value(writer)[0] == "initial"
    writer.execute("BEGIN TRANSACTION")
    writer.execute("UPDATE records SET value = ?, revision = ? WHERE id = ?", ["committed", 2, 1])
    assert value(observer)[0] == "initial"
    writer.commit()
    assert not writer.in_transaction
    assert tuple(value(observer)._values) == ("committed", 2)
    assert live_quack.owner.execute("SELECT value FROM records WHERE id = 1").fetchone()[0] == "committed"


def test_two_clients_isolate_uncommitted_writes_and_fence_stale_cas(live_quack):
    first = live_quack.connect()
    second = live_quack.connect()
    first.execute("BEGIN TRANSACTION")
    second.execute("BEGIN TRANSACTION")
    assert first.execute(
        "UPDATE records SET value = ?, revision = revision + 1 WHERE id = ? AND revision = ?",
        ["winner", 1, 1],
    ).rowcount == 1
    assert value(second)[0] == "initial"
    with pytest.raises(Exception, match="(?i)(conflict|transaction)"):
        second.execute(
            "UPDATE records SET value = ?, revision = revision + 1 WHERE id = ? AND revision = ?",
            ["loser", 1, 1],
        )
    second.rollback()
    first.commit()
    assert second.execute(
        "UPDATE records SET value = ?, revision = revision + 1 WHERE id = ? AND revision = ?",
        ["stale", 1, 1],
    ).rowcount == 0
    assert second.execute(
        "UPDATE records SET value = ?, revision = revision + 1 WHERE id = ? AND revision = ?",
        ["next", 1, 2],
    ).rowcount == 1
    row = value(first)
    assert row[0] == "next" and row[1] == 3


def test_update_delete_and_executemany_return_affected_rowcounts(live_quack):
    client = live_quack.connect()
    inserted = client.executemany(
        "INSERT INTO records VALUES (?, ?, ?)",
        ((i, f"row {i}", 1) for i in (2, 3, 4)),
    )
    assert inserted.rowcount == 3
    assert client.execute("UPDATE records SET revision = ? WHERE id >= ?", [2, 3]).rowcount == 2
    assert client.execute("UPDATE records SET revision = ? WHERE id = ?", [3, 99]).rowcount == 0
    assert client.execute("DELETE FROM records WHERE id >= ?", [3]).rowcount == 2
    assert client.execute("DELETE FROM records WHERE id = ?", [99]).rowcount == 0
    assert client.executemany("INSERT INTO records VALUES (?, ?, ?)", []).rowcount == 0
    assert client.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 2


@pytest.mark.parametrize("text", [
    "quoted 'value'; DELETE FROM records; -- still data",
    "Unicode: Δ 法律 café 🦆",
    "before\x00after\n? $named /* not SQL */",
])
def test_bound_strings_roundtrip_without_sql_interpretation(live_quack, text):
    client = live_quack.connect()
    assert client.execute("UPDATE records SET value = ? WHERE id = ?", [text, 1]).rowcount == 1
    assert value(client)[0] == text
    row = client.execute("SELECT '?' AS literal, ? AS value /* ? is only a comment */", [text]).fetchone()
    assert row["literal"] == "?" and row["value"] == text
    assert client.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1


def test_named_parameters_bind_by_name_reuse_and_support_reserved_words(live_quack):
    client = live_quack.connect()
    row = client.execute(
        "SELECT $first AS a, $second AS b, $first AS repeated",
        {"second": "二", "first": "one's"},
    ).fetchone()
    assert row["a"] == "one's" and row["b"] == "二" and row["repeated"] == "one's"
    assert client.execute("SELECT $select AS value", {"select": "reserved-name"}).fetchone()[0] == "reserved-name"


def test_prepared_query_with_trailing_line_comment_executes_bound_value(live_quack):
    client = live_quack.connect()
    row = client.execute("SELECT ? AS value -- question ? and semicolon ; in comment", ["kept"]).fetchone()
    assert row is not None and row["value"] == "kept"


def test_executemany_and_close_rollback_preserve_remote_owner(live_quack):
    writer = live_quack.connect()
    observer = live_quack.connect()
    writer.execute("BEGIN TRANSACTION")
    assert writer.executemany("INSERT INTO records VALUES (?, ?, ?)",
                              [(2, "two", 1), (3, "three", 1)]).rowcount == 2
    assert writer.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 3
    assert observer.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    writer.close()
    writer.close()  # idempotent cleanup must not re-run a closed transaction.
    assert observer.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    observer.execute("BEGIN TRANSACTION")
    observer.executemany("INSERT INTO records VALUES (?, ?, ?)", [(4, "four", 1), (5, "five", 1)])
    observer.commit()
    assert live_quack.owner.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 3
