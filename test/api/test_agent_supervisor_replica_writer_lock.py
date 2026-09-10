"""Replica refresh must retain the real DuckDB writer's kernel file lock."""
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server as quack
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import QuackCapabilityStatus


@pytest.fixture
def writer(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    database = tmp_path / "control.duckdb"
    connection = duckdb.connect(str(database), config={"threads": "1"})
    connection.execute("CREATE TABLE values_to_copy AS SELECT 42 AS value")
    connection.execute("CHECKPOINT")
    server = quack.build_server(database_path=database, state_dir=tmp_path / "state",
                                 transport=quack.FakeQuackTransport())
    server._connection = connection
    server._open_database_parent_anchor()
    server._bind_database_inode_after_migration()
    try:
        yield server, connection, database
    finally:
        connection.close()
        server._connection = None
        server._close_database_namespace_anchor()


def assert_other_process_writer_blocked(database):
    # Opening the canonical file in this test process would itself release the
    # writer's POSIX lock on close. Only the competing child opens that path.
    code = """
import duckdb, sys
try:
    connection = duckdb.connect(sys.argv[1], config={'threads': '1'})
except duckdb.IOException as exc:
    sys.exit(69 if 'lock' in str(exc).lower() else 2)
else:
    connection.close()
    sys.exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code, str(database)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 69, (result.returncode, result.stderr)


def test_repeated_replica_copy_keeps_writer_lock_and_anchor_offset(writer):
    server, connection, database = writer
    descriptor = server._database_namespace_anchor.database_descriptor
    assert_other_process_writer_blocked(database)
    os.lseek(descriptor, 17, os.SEEK_SET)
    for value in (43, 44):
        digest, size = server._copy_authoritative_read_replica()
        assert_other_process_writer_blocked(database)
        replica = server.read_replica_path().read_bytes()
        assert len(replica) == size
        assert digest == "sha256:" + hashlib.sha256(replica).hexdigest()
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 17
        connection.execute("INSERT INTO values_to_copy VALUES (?)", [value])
    assert connection.execute("SELECT count(*) FROM values_to_copy").fetchone() == (3,)


@pytest.mark.parametrize("failure", ["write", "timeout"])
def test_failed_copy_keeps_canonical_lock_and_allows_next_write(writer, monkeypatch, failure):
    server, connection, database = writer
    assert_other_process_writer_blocked(database)
    if failure == "write":
        def fail(*args, **kwargs):
            raise OSError("injected replica disk write failure")
        monkeypatch.setattr(quack.os, "write", fail)
    else:
        monkeypatch.setattr(quack, "READ_REPLICA_COPY_TIMEOUT_SECONDS", -1)
    with pytest.raises(quack.QuackStateServerReadyError):
        server._copy_authoritative_read_replica()
    assert_other_process_writer_blocked(database)
    assert not server.read_replica_path().exists()
    assert not list(database.parent.glob(".*.tmp"))
    connection.execute("INSERT INTO values_to_copy VALUES (43)")
    assert connection.execute("SELECT count(*) FROM values_to_copy").fetchone() == (2,)


def test_real_quack_owner_keeps_writer_exclusion_across_start_and_checkpoint(tmp_path):
    pytest.importorskip("duckdb")
    capability = quack.probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip("reviewed preinstalled Quack extension unavailable")
    database = tmp_path / "control.duckdb"
    server = quack.build_server(database_path=database, state_dir=tmp_path / "owner",
                                secret_handle="handle:replica-lock-regression",
                                allow_legacy_board_unstall=False)
    server.start()
    try:
        assert_other_process_writer_blocked(database)
        assert server.checkpoint()["checkpointed"] is True
        assert_other_process_writer_blocked(database)
        assert server.ready()["ready"] is True
    finally:
        server.stop()


def test_temp_name_collision_preserves_existing_file_and_writer_lock(writer, monkeypatch):
    server, _connection, database = writer
    monkeypatch.setattr(quack.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    temporary = database.parent / f".{server.read_replica_path().name}.{os.getpid()}.collision.tmp"
    temporary.write_bytes(b"preexisting file must survive")
    with pytest.raises(quack.QuackStateServerReadyError):
        server._copy_authoritative_read_replica()
    assert temporary.read_bytes() == b"preexisting file must survive"
    assert_other_process_writer_blocked(database)


def test_failed_copy_does_not_unlink_replacement_temp_inode(writer, monkeypatch):
    server, _connection, database = writer
    monkeypatch.setattr(quack.uuid, "uuid4", lambda: SimpleNamespace(hex="replacement"))
    temporary = database.parent / f".{server.read_replica_path().name}.{os.getpid()}.replacement.tmp"
    def replace_then_fail(*args, **kwargs):
        temporary.unlink()
        temporary.write_bytes(b"replacement belongs to another operation")
        raise OSError("injected copy failure after temp replacement")
    monkeypatch.setattr(quack.os, "write", replace_then_fail)
    with pytest.raises(quack.QuackStateServerReadyError):
        server._copy_authoritative_read_replica()
    assert temporary.read_bytes() == b"replacement belongs to another operation"
    assert_other_process_writer_blocked(database)
