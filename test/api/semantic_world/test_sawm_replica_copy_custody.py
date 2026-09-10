"""Replica copies must preserve the real writer's process-scoped database lock."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from test.api.semantic_world.test_semantic_addressed_world_model_board import _load


@pytest.fixture
def operator():
    return _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_replica_copy_custody_operator_test",
    )


def _canonical_write_lock_held(database: Path) -> bool:
    observed = database.stat()
    identity = (os.major(observed.st_dev), os.minor(observed.st_dev), observed.st_ino)
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        if (
            len(parts) != 8
            or parts[1:5] != ["POSIX", "ADVISORY", "WRITE", str(os.getpid())]
            or parts[6:] != ["0", "EOF"]
        ):
            continue
        major, minor, inode = parts[5].split(":")
        if (int(major, 16), int(minor, 16), int(inode)) == identity:
            return True
    return False


def _independent_writer_can_open(database: Path) -> bool:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", """
import json, sys
# The test environment can install DuckDB in user site, which -I omits.
# Admit the exact already-imported test dependency instead of ambient paths.
sys.path.insert(0, sys.argv[2])
import duckdb
try:
    connection = duckdb.connect(sys.argv[1], config={"threads": 1})
except duckdb.IOException:
    print(json.dumps({"opened": False}))
else:
    connection.close()
    print(json.dumps({"opened": True}))
""", str(database), str(Path(duckdb.__file__).resolve().parent.parent)],
        capture_output=True, text=True, check=True, timeout=10,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
    )
    return json.loads(completed.stdout)["opened"]


def _request(database: Path, target: Path):
    observed = database.lstat()
    return {
        "source": str(database), "target": str(target), "nonce": "a" * 32,
        "source_identity": [observed.st_dev, observed.st_ino, observed.st_size,
                            observed.st_mtime_ns, observed.st_ctime_ns, observed.st_uid],
    }


def _child_copy(operator, request):
    return subprocess.run(
        [sys.executable, "-I", str(Path(operator.__file__).with_name("sawm_replica_copy.py"))],
        input=json.dumps(request), capture_output=True, text=True, timeout=10,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
    )


@pytest.fixture
def canonical_writer(tmp_path):
    database = tmp_path / "control.duckdb"
    connection = duckdb.connect(str(database), config={"threads": 1})
    try:
        connection.execute("CREATE TABLE progress (value INTEGER)")
        connection.execute("INSERT INTO progress VALUES (1)")
        connection.execute("CHECKPOINT")
        assert _canonical_write_lock_held(database)
        yield database, connection
    finally:
        connection.close()


def test_actual_replica_copy_retains_writer_lock_and_later_mutation(operator, canonical_writer):
    database, writer = canonical_writer
    target = database.with_name("control.read-replica.duckdb")
    assert not _independent_writer_can_open(database)
    result = operator._SawmQuackTransport._copy_replica(database, target)

    # POSIX locks belong to the process: even closing an unrelated read-only fd
    # for this inode would silently drop DuckDB's existing exclusive lock.
    assert _canonical_write_lock_held(database), "replica copying dropped the canonical writer lock"
    assert not _independent_writer_can_open(database), "replica copying admitted a second DuckDB writer"
    assert result["authority"] == "non_authoritative_read_replica"
    assert result["size_bytes"] == target.stat().st_size
    assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.stat().st_ino != database.stat().st_ino
    replica = duckdb.connect(str(target), read_only=True, config={"threads": 1})
    try:
        assert replica.execute("SELECT value FROM progress").fetchall() == [(1,)]
        writer.execute("INSERT INTO progress VALUES (2)")
        writer.execute("CHECKPOINT")
        assert _canonical_write_lock_held(database)
        assert writer.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 3
        assert not _independent_writer_can_open(database)
        assert replica.execute("SELECT value FROM progress").fetchall() == [(1,)]
    finally:
        replica.close()
    assert _canonical_write_lock_held(database)


@pytest.mark.parametrize("bad_target", ["same", "foreign_parent"])
def test_invalid_replica_target_preserves_canonical_lock(operator, canonical_writer, tmp_path, bad_target):
    database, writer = canonical_writer
    if bad_target == "same":
        target = database
    else:
        directory = tmp_path / "foreign"
        directory.mkdir()
        target = directory / "replica.duckdb"
    with pytest.raises(RuntimeError, match="bounded native replica copy unavailable"):
        operator._SawmQuackTransport._copy_replica(database, target)
    assert _canonical_write_lock_held(database)
    assert writer.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 1


@pytest.mark.parametrize("symlink_side", ["source", "target"])
def test_replica_copy_rejects_symlink_without_rebinding_or_losing_lock(
    operator, canonical_writer, symlink_side,
):
    database, writer = canonical_writer
    target = database.with_name("control.read-replica.duckdb")
    source = database
    preserved = database.with_name("unrelated.bin")
    preserved.write_bytes(b"existing unrelated content")
    if symlink_side == "source":
        source = database.with_name("alias.duckdb")
        source.symlink_to(database.name)
    else:
        target.symlink_to(preserved.name)
    with pytest.raises(RuntimeError, match="bounded native replica copy unavailable"):
        operator._SawmQuackTransport._copy_replica(source, target)
    assert preserved.read_bytes() == b"existing unrelated content"
    assert _canonical_write_lock_held(database)
    assert writer.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 1
    assert (source if symlink_side == "source" else target).is_symlink()


def test_child_source_cas_denial_preserves_target_and_retained_writer(operator, canonical_writer):
    database, writer = canonical_writer
    target = database.with_name("control.read-replica.duckdb")
    target.write_bytes(b"prior projection")
    request = _request(database, target)
    writer.execute("INSERT INTO progress VALUES (2)")
    writer.execute("CHECKPOINT")
    assert request["source_identity"] != _request(database, target)["source_identity"]
    result = _child_copy(operator, request)
    assert result.returncode != 0
    assert result.stdout == ""
    assert target.read_bytes() == b"prior projection"
    assert _canonical_write_lock_held(database)
    assert writer.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 3
    assert not list(database.parent.glob(f".{target.name}.*.tmp"))


def test_child_temporary_collision_preserves_existing_work(operator, canonical_writer):
    database, writer = canonical_writer
    target = database.with_name("control.read-replica.duckdb")
    target.write_bytes(b"prior projection")
    request = _request(database, target)
    collision = target.with_name(f".{target.name}.{request['nonce']}.tmp")
    collision.write_bytes(b"another copy's preserved temporary work")
    result = _child_copy(operator, request)
    assert result.returncode != 0
    assert result.stdout == ""
    assert target.read_bytes() == b"prior projection"
    assert collision.read_bytes() == b"another copy's preserved temporary work"
    assert _canonical_write_lock_held(database)
    assert writer.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 1


def _injected_copy_child(operator, request, injection):
    script = """
import json, os, runpy, sys
from pathlib import Path
module = runpy.run_path(sys.argv[1], run_name='isolated_copy_test')
request = json.loads(sys.stdin.read())
""" + injection + """
try:
    result = module['_copy'](request)
except Exception:
    sys.exit(1)
print(json.dumps(result))
"""
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", script,
         str(Path(operator.__file__).with_name("sawm_replica_copy.py"))],
        input=json.dumps(request), capture_output=True, text=True, timeout=10, check=False,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
    )


@pytest.mark.parametrize(
    "replacement", ["foreign_regular", "foreign_same_size", "symlink", "truncated_owned"],
)
def test_child_promotion_requires_exact_owned_regular_entry(
    operator, canonical_writer, replacement,
):
    database, connection = canonical_writer
    target = database.with_name("control.read-replica.duckdb")
    target.write_bytes(b"previous replica")
    request = _request(database, target)
    temporary = target.with_name(f".{target.name}.{request['nonce']}.tmp")
    displaced = temporary.with_suffix(".preserved")
    foreign = target.with_name("foreign-entry")
    foreign.write_bytes(b"foreign work")
    injection = """
original_fsync = os.fsync
fired = False
def interfere(fd):
    global fired
    original_fsync(fd)
    if fired:
        return
    fired = True
    temporary = Path(request['target']).with_name('.' + Path(request['target']).name + '.' + request['nonce'] + '.tmp')
    mode = MODE
    if mode == 'truncated_owned':
        os.ftruncate(fd, 1)
    else:
        temporary.rename(temporary.with_suffix('.preserved'))
        if mode == 'foreign_regular':
            temporary.write_bytes(b'foreign work')
        elif mode == 'foreign_same_size':
            temporary.write_bytes(b'x' * request['source_identity'][2])
        else:
            temporary.symlink_to('foreign-entry')
os.fsync = interfere
""".replace("MODE", repr(replacement))
    result = _injected_copy_child(operator, request, injection)
    assert result.returncode == 1, "copier promoted an entry without its exact ownership"
    assert result.stdout == ""
    assert target.read_bytes() == b"previous replica"
    assert foreign.read_bytes() == b"foreign work"
    if replacement == "truncated_owned":
        assert not temporary.exists()
    else:
        assert displaced.exists()
        expected = b"x" * request["source_identity"][2] if replacement == "foreign_same_size" else b"foreign work"
        assert temporary.read_bytes() == expected
        assert temporary.is_symlink() == (replacement == "symlink")
    assert _canonical_write_lock_held(database)
    assert not _independent_writer_can_open(database)
    connection.execute("INSERT INTO progress VALUES (2)")
    assert connection.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 3


def test_child_receipt_rejects_promoted_content_changed_during_directory_fsync(
    operator, canonical_writer,
):
    database, connection = canonical_writer
    target = database.with_name("control.read-replica.duckdb")
    request = _request(database, target)
    injection = """
import stat
original_fsync = os.fsync
fired = False
def interfere(fd):
    global fired
    original_fsync(fd)
    if fired or not stat.S_ISDIR(os.fstat(fd).st_mode):
        return
    fired = True
    target = Path(request['target'])
    before = target.lstat()
    target.write_bytes(b'x' * before.st_size)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
os.fsync = interfere
"""
    result = _injected_copy_child(operator, request, injection)
    assert result.returncode == 1, "copier acknowledged a digest for changed promoted bytes"
    assert result.stdout == ""
    assert target.read_bytes() == b"x" * request["source_identity"][2]
    assert _canonical_write_lock_held(database)
    assert not _independent_writer_can_open(database)
    connection.execute("INSERT INTO progress VALUES (2)")
    assert connection.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 3
