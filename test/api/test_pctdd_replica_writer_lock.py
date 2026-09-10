"""Replica refresh must retain the real DuckDB writer's kernel file lock."""
import hashlib
import os
import stat
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
    if hasattr(server, "_open_replica_source_anchor"):
        server._open_replica_source_anchor()
    try:
        yield server, connection, database
    finally:
        connection.close()
        server._connection = None
        if hasattr(server, "_close_replica_source_anchor"):
            server._replica_source_close_unproven = False
            server._close_replica_source_anchor()


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
    descriptor = getattr(server, "_replica_source_descriptor", None)
    assert_other_process_writer_blocked(database)
    if descriptor is not None:
        os.lseek(descriptor, 17, os.SEEK_SET)
    for value in (43, 44):
        digest, size = server._copy_authoritative_read_replica()
        assert_other_process_writer_blocked(database)
        replica = server.read_replica_path().read_bytes()
        assert len(replica) == size
        assert digest == "sha256:" + hashlib.sha256(replica).hexdigest()
        if descriptor is not None:
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
                                secret_handle="handle:replica-lock-regression")
    server.start()
    try:
        assert_other_process_writer_blocked(database)
        assert server.checkpoint()["checkpointed"] is True
        assert_other_process_writer_blocked(database)
        assert server.ready()["ready"] is True
    finally:
        descriptor = server._replica_source_descriptor
        server.stop()
    assert descriptor is not None
    assert server._replica_source_descriptor is None
    with pytest.raises(OSError):
        os.fstat(descriptor)


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


def test_source_path_replacement_rejected_without_dropping_original_lock(writer):
    server, connection, database = writer
    original = database.with_name("preserved.duckdb")
    database.rename(original)
    database.write_bytes(b"replacement is not the admitted database")
    with pytest.raises(quack.QuackStateServerReadyError, match="anchor identity"):
        server._copy_authoritative_read_replica()
    assert_other_process_writer_blocked(original)
    assert not server.read_replica_path().exists()
    assert database.read_bytes() == b"replacement is not the admitted database"


def test_repeated_failed_close_preserves_anchor_and_writer_exclusion(writer):
    server, connection, database = writer
    class FailedClose:
        def close(self):
            raise OSError("injected unproven close")
    server._connection = FailedClose()
    descriptor = server._replica_source_descriptor
    server._emergency_cleanup()
    server._emergency_cleanup()
    assert server._replica_source_descriptor == descriptor
    os.fstat(descriptor)
    assert_other_process_writer_blocked(database)


@pytest.mark.parametrize("interference", ["foreign_regular", "symlink", "truncated_owned"])
def test_promotion_requires_owned_complete_temp_entry(writer, monkeypatch, interference):
    server, connection, database = writer
    monkeypatch.setattr(quack.uuid, "uuid4", lambda: SimpleNamespace(hex="promotion"))
    replica = server.read_replica_path()
    replica.write_bytes(b"previous replica must survive")
    temporary = database.parent / f".{replica.name}.{os.getpid()}.promotion.tmp"
    displaced = temporary.with_suffix(".preserved")
    foreign = database.parent / "foreign-work"
    foreign.write_bytes(b"foreign work must survive")
    original_fsync = os.fsync
    fired = False

    def interfere_after_sync(descriptor):
        nonlocal fired
        original_fsync(descriptor)
        if fired:
            return
        fired = True
        if interference == "truncated_owned":
            os.ftruncate(descriptor, 1)
        else:
            temporary.rename(displaced)
            if interference == "symlink":
                temporary.symlink_to(foreign.name)
            else:
                temporary.write_bytes(b"foreign work must survive")

    monkeypatch.setattr(quack.os, "fsync", interfere_after_sync)
    with pytest.raises(quack.QuackStateServerReadyError):
        server._copy_authoritative_read_replica()
    assert fired
    assert replica.read_bytes() == b"previous replica must survive"
    assert foreign.read_bytes() == b"foreign work must survive"
    if interference == "truncated_owned":
        assert not temporary.exists()
    else:
        assert displaced.exists()
        assert temporary.read_bytes() == b"foreign work must survive"
        assert temporary.is_symlink() == (interference == "symlink")
    assert_other_process_writer_blocked(database)
    connection.execute("INSERT INTO values_to_copy VALUES (43)")
    assert connection.execute("SELECT count(*) FROM values_to_copy").fetchone() == (2,)


@pytest.mark.parametrize("interference", ["replacement", "same_size_rewrite"])
def test_changed_replica_during_directory_sync_does_not_report_success(writer, monkeypatch, interference):
    server, connection, database = writer
    replica = server.read_replica_path()
    original_fsync = os.fsync
    fired = False

    def interfere_during_directory_sync(descriptor):
        nonlocal fired
        original_fsync(descriptor)
        if fired or not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            return
        fired = True
        if interference == "replacement":
            replica.rename(replica.with_suffix(".preserved"))
            replica.write_bytes(b"foreign replacement")
        else:
            with replica.open("r+b") as handle:
                handle.write(b"changed!")

    monkeypatch.setattr(quack.os, "fsync", interfere_during_directory_sync)
    with pytest.raises(quack.QuackStateServerReadyError):
        server._copy_authoritative_read_replica()
    assert fired
    assert_other_process_writer_blocked(database)
    connection.execute("INSERT INTO values_to_copy VALUES (43)")
    assert connection.execute("SELECT count(*) FROM values_to_copy").fetchone() == (2,)
