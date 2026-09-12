"""Actual inode/lock boundaries; no live board or source admission is claimed."""
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import socket
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import empty_owner_wal as ew
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import acquire_exclusive_owner_lock
from scripts import run_agent_supervisor_efficiency_state_hardening as op

SOURCE = {"candidate_head": "a" * 40, "candidate_tree": "b" * 40, "store_id": "data/aseh/control.duckdb"}


def database(root):
    path = root / "control.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE evidence(value VARCHAR)")
        connection.execute("INSERT INTO evidence VALUES ('committed, unknown callback stays unknown')")
        connection.execute("CHECKPOINT")
    path.chmod(0o600)
    return path


def empty_wal(db, mode=0o664):
    path = db.with_name(db.name + ".wal")
    path.write_bytes(b"")
    path.chmod(mode)
    return path


def binding(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NOATIME)
    try:
        return {**ew._stat(os.fstat(fd)), "sha256": ew._hash(fd)}
    finally:
        os.close(fd)


def journal(db):
    return db.parent / ("." + db.name + ".empty-wal-preservation")


@contextmanager
def native_locks(db):
    with ExitStack() as stack:
        lease = acquire_exclusive_owner_lock(db.with_name("." + db.name + ".state-owner.lock"))
        stack.callback(lease.close)
        descriptors = [lease.fileno()]
        for lock_class, suffix in (("migration", ".migration.lock"), ("intent", ".intent.lock"), ("database", ".lock")):
            stack.enter_context(op._r21_nonblocking_named_lock(lease.directory_fileno(),
                name="." + db.name + suffix, lock_class=lock_class, retained_descriptors=descriptors))
        yield lease, descriptors
    assert len(descriptors) == 1


def preserve(db, lease, descriptors, guard=lambda: None, source=SOURCE):
    return ew.preserve_empty_owner_wal(directory_fd=lease.directory_fileno(), directory_path=db.parent,
        database_name=db.name, lock_descriptors=descriptors, source_identity=source, native_guard=guard)


@pytest.mark.parametrize("mode", [0o600, 0o644, 0o664])
def test_preserves_original_inode_bytes_metadata_and_restart_is_idempotent(tmp_path, mode):
    db = database(tmp_path); wal = empty_wal(db, mode)
    db_before, wal_before = binding(db), binding(wal)
    with native_locks(db) as (lease, descriptors):
        result = preserve(db, lease, descriptors)
    entry = journal(db) / result["prepared_sha256"]
    prepared = json.loads((entry / "prepared.json").read_bytes())
    completed = json.loads((entry / "completed.json").read_bytes())
    assert prepared["database"] == db_before and prepared["wal"] == wal_before
    after = binding(entry / "original.wal")
    assert {k: v for k, v in after.items() if k != "ctime_ns"} == {k: v for k, v in wal_before.items() if k != "ctime_ns"}
    assert completed["archived_wal"] == after and completed["completion_authority"] is False
    assert not wal.exists() and binding(db) == db_before
    with native_locks(db) as (lease, descriptors):
        again = preserve(db, lease, descriptors)
    assert again == {"preserved": False, "reason": "wal_already_preserved", "completion_authority": False}
    assert binding(db) == db_before and binding(entry / "original.wal") == after
    assert len(list(journal(db).iterdir())) == 1
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT value FROM evidence").fetchall() == [("committed, unknown callback stays unknown",)]


@pytest.mark.parametrize("kind", ["nonempty", "wal-hardlink", "wal-symlink", "fifo", "wal-mode", "db-hardlink", "db-symlink", "db-mode", "checkpoint", "recovery"])
def test_unsafe_inputs_refuse_without_any_retirement(tmp_path, kind):
    db = database(tmp_path); wal = empty_wal(db)
    if kind == "nonempty": wal.write_bytes(b"opaque recovery input")
    if kind == "wal-hardlink": os.link(wal, tmp_path / "external-wal")
    if kind == "wal-symlink":
        wal.rename(tmp_path / "kept-wal"); wal.symlink_to(tmp_path / "kept-wal")
    if kind == "fifo": wal.unlink(); os.mkfifo(wal, 0o600)
    if kind == "wal-mode": wal.chmod(0o666)
    if kind == "db-mode": db.chmod(0o644)
    if kind == "db-hardlink": os.link(db, tmp_path / "external-db")
    if kind == "db-symlink":
        db.rename(tmp_path / "kept-db"); db.symlink_to(tmp_path / "kept-db")
    if kind in {"checkpoint", "recovery"}: (tmp_path / (wal.name + "." + kind)).write_bytes(b"opaque")
    before = os.lstat(wal)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors)
    assert ew._stat(os.lstat(wal)) == ew._stat(before)
    assert not journal(db).exists()


@pytest.mark.parametrize("ordinal", range(4))
def test_borrowed_but_unlocked_or_readonly_lock_cannot_authorize(tmp_path, ordinal):
    db = database(tmp_path); wal = empty_wal(db); before = binding(wal)
    with native_locks(db) as (lease, descriptors):
        fcntl.flock(descriptors[ordinal], fcntl.LOCK_SH)
        with pytest.raises(ew.EmptyOwnerWalError, match="exclusive flock"):
            preserve(db, lease, descriptors)
    assert binding(wal) == before and not journal(db).exists()


@contextmanager
def duckdb_writer(db):
    script = "import duckdb,sys; c=duckdb.connect(sys.argv[1]); print('ready',flush=True); sys.stdin.read(1); c.close()"
    child = subprocess.Popen([sys.executable, "-B", "-c", script, str(db)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ)
            assert ready.select(20), "writer fixture failed to start"
            assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        child.stdin.close()
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate(); child.wait(timeout=5)
        assert child.returncode == 0, child.stderr.read()


def test_real_duckdb_writer_with_all_four_named_locks_refuses_before_wal_move(tmp_path):
    db = database(tmp_path)
    with duckdb_writer(db):
        wal = empty_wal(db); before = binding(wal)
        with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError, match="writer fence is contended"):
            preserve(db, lease, descriptors)
        assert binding(wal) == before and not journal(db).exists()


@pytest.mark.parametrize("phase", ["prepared", "completed", "completed-after-publish"])
def test_durable_audit_failure_blocks_and_fresh_locked_retry_recovers(tmp_path, monkeypatch, phase):
    db = database(tmp_path); wal = empty_wal(db); original = binding(wal)
    publish = ew._publish
    def fail(parent, name, value):
        if phase == "completed-after-publish" and name == "completed.json":
            publish(parent, name, value)
            raise OSError("injected post-publication audit interruption")
        if name == phase + ".json": raise OSError("injected audit failure")
        return publish(parent, name, value)
    monkeypatch.setattr(ew, "_publish", fail)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors)
    assert wal.exists() == (phase == "prepared")
    if wal.exists(): assert binding(wal) == original
    else:
        archived = next(journal(db).glob("*/original.wal"))
        assert binding(archived)["inode"] == original["inode"]
    monkeypatch.setattr(ew, "_publish", publish)
    with native_locks(db) as (lease, descriptors):
        result = preserve(db, lease, descriptors)
    assert result["completion_authority"] is False and not wal.exists()
    assert len(list(journal(db).glob("*/completed.json"))) == 1


@pytest.mark.parametrize("change", ["source", "database", "archive", "wal-reappeared"])
def test_pending_audit_never_reconstructs_authority_or_ignores_drift(tmp_path, monkeypatch, change):
    db = database(tmp_path); wal = empty_wal(db)
    publish = ew._publish
    def fail(parent, name, value):
        if name == "completed.json": raise OSError("audit unavailable")
        return publish(parent, name, value)
    monkeypatch.setattr(ew, "_publish", fail)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors)
    monkeypatch.setattr(ew, "_publish", publish)
    source = SOURCE
    if change == "source": source = {**SOURCE, "candidate_head": "c" * 40}
    if change == "database": os.utime(db, ns=(db.stat().st_atime_ns, db.stat().st_mtime_ns + 1))
    if change == "archive":
        archived = next(journal(db).glob("*/original.wal")); archived.rename(archived.with_name("outside-original")); archived.write_bytes(b""); archived.chmod(0o664)
    if change == "wal-reappeared": empty_wal(db)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors, source=source)
    assert not list(journal(db).glob("*/completed.json"))


@pytest.mark.parametrize("change", ["wal-replace", "wal-link", "wal-grow", "db-replace", "lock-replace", "parent-replace", "guard-failure"])
def test_race_after_prepared_evidence_refuses_before_retirement(tmp_path, monkeypatch, change):
    root = tmp_path / "canonical"; root.mkdir(); db = database(root); wal = empty_wal(db)
    publish = ew._publish
    changed = []
    def drift(parent, name, value):
        publish(parent, name, value)
        if name != "prepared.json": return
        changed.append(True)
        if change == "wal-replace": wal.rename(root / "retained-wal"); empty_wal(db)
        if change == "wal-link": os.link(wal, tmp_path / "external")
        if change == "wal-grow": wal.write_bytes(b"new bytes")
        if change == "db-replace": db.rename(root / "retained-db"); db.write_bytes(b"replacement"); db.chmod(0o600)
        if change == "lock-replace":
            lock = root / ".control.duckdb.intent.lock"; lock.rename(root / "retained-lock"); lock.write_bytes(b""); lock.chmod(0o600)
        if change == "parent-replace": root.rename(tmp_path / "retained-parent"); root.mkdir()
    monkeypatch.setattr(ew, "_publish", drift)
    def guard():
        if change == "guard-failure" and changed: raise ew.EmptyOwnerWalError("native source/endpoint refusal")
    with pytest.raises((ew.EmptyOwnerWalError, op.OperatorError)), native_locks(db) as (lease, descriptors):
        preserve(db, lease, descriptors, guard=guard)
    actual_root = tmp_path / "retained-parent" if change == "parent-replace" else root
    assert list(actual_root.glob(".control.duckdb.empty-wal-preservation/*/prepared.json"))
    assert not list(actual_root.glob(".control.duckdb.empty-wal-preservation/*/original.wal"))
    assert not list(actual_root.glob(".control.duckdb.empty-wal-preservation/*/completed.json"))


def test_post_rename_guard_failure_keeps_inode_and_resumes_with_fresh_gates(tmp_path, monkeypatch):
    db = database(tmp_path); wal = empty_wal(db); before = binding(wal)
    def guard():
        if not wal.exists(): raise ew.EmptyOwnerWalError("endpoint changed")
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError, match="endpoint changed"):
        preserve(db, lease, descriptors, guard=guard)
    archived = next(journal(db).glob("*/original.wal"))
    assert archived.stat().st_ino == before["inode"] and not wal.exists()
    assert not list(journal(db).glob("*/completed.json"))
    calls = []
    with native_locks(db) as (lease, descriptors):
        preserve(db, lease, descriptors, guard=lambda: calls.append("fresh"))
    assert len(calls) >= 4


def native_fixture(tmp_path):
    db = database(tmp_path); owner = tmp_path / "owner"; owner.mkdir(mode=0o700)
    channel = socket.socket(); channel.bind(("127.0.0.1", 0)); port = channel.getsockname()[1]; channel.close()
    server = SimpleNamespace(lifecycle=SimpleNamespace(value="created"), identity=None,
        config=SimpleNamespace(host="127.0.0.1", port=port),
        owner_marker_path=lambda: db.with_name(".control.duckdb.state-owner.json"),
        typed_command_socket_path=lambda: owner / "typed-owner.sock",
        typed_command_token_path=lambda: owner / "typed-owner.token")
    paths = {"database": db, "owner": owner, "owner_start_permission_receipts": tmp_path / "permissions"}
    context = {**SOURCE, "candidate_authorization_witness": {"fixture": "not live admission"},
        "bootstrap_receipt_id": "sha256:" + "1" * 64, "repair_transition_receipt_cid": "sha256:" + "2" * 64,
        "materialized_launch_admission_cid": "sha256:" + "3" * 64}
    return paths, server, context


@pytest.mark.parametrize("mode", [0o644, 0o664])
def test_native_r23_composition_starts_absent_and_keeps_wal_mutated_false(tmp_path, monkeypatch, mode):
    paths, server, context = native_fixture(tmp_path)
    db = paths["database"]; wal = empty_wal(db, mode); before = binding(db)
    monkeypatch.setattr(op, "_assert_candidate_authorization_witness", lambda *a, **k: None)
    observation = op._r21_owner_start_contention_observation(paths=paths, server=server,
        expected_lifecycle="created", r23_permission_context=context)
    assert observation["wal"]["availability"] == "absent" and not wal.exists()
    receipt = json.loads(next(paths["owner_start_permission_receipts"].glob("*.json")).read_bytes())
    assert receipt["wal_mutated"] is False
    assert binding(db) == before
    assert list(journal(db).glob("*/original.wal"))


@pytest.mark.parametrize("gate", ["source", "owner", "listener"])
def test_native_r23_existing_gates_still_precede_retirement(tmp_path, monkeypatch, gate):
    paths, server, context = native_fixture(tmp_path); db = paths["database"]; wal = empty_wal(db)
    def source_guard(*a, **k):
        if gate == "source": raise op.OperatorError("source rejected")
    monkeypatch.setattr(op, "_assert_candidate_authorization_witness", source_guard)
    if gate == "owner": server.identity = {"active": True}
    if gate == "listener":
        def present(*a, **k): raise op.OperatorError("listener present")
        monkeypatch.setattr(op, "_r21_listener_absence_samples", present)
    before = binding(wal)
    with pytest.raises(op.OperatorError):
        op._r21_owner_start_contention_observation(paths=paths, server=server,
            expected_lifecycle="created", r23_permission_context=context)
    assert binding(wal) == before and not journal(db).exists()


def test_actual_process_exit_after_retirement_releases_locks_and_restarts_audit(tmp_path):
    db = database(tmp_path); wal = empty_wal(db); before = binding(wal)
    # The disposable child exits without Python finalizers after the inode was
    # preserved. The parent then reacquires actual fresh native locks.
    script = """
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('fixture', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
publish = m.ew._publish
def exit_before_complete(parent, name, value):
    if name == 'completed.json': os._exit(73)
    return publish(parent, name, value)
m.ew._publish = exit_before_complete
db = m.Path(sys.argv[2])
with m.native_locks(db) as (lease, descriptors): m.preserve(db, lease, descriptors)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, __file__, str(db)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, timeout=30)
    assert result.returncode == 73, result.stderr.decode()
    assert not wal.exists() and not list(journal(db).glob("*/completed.json"))
    assert next(journal(db).glob("*/original.wal")).stat().st_ino == before["inode"]
    with native_locks(db) as (lease, descriptors):
        assert preserve(db, lease, descriptors)["preserved"] is True
    assert list(journal(db).glob("*/completed.json"))


def test_partial_audit_file_is_retained_as_diagnostic_and_retry_is_atomic(tmp_path, monkeypatch):
    db = database(tmp_path); empty_wal(db)
    write = os.write
    partial = []
    def interrupted(fd, data):
        path = os.readlink(f"/proc/self/fd/{fd}")
        if 'empty-wal-preservation' in path and b'aseh-empty-wal-completed@1' in data:
            write(fd, data[:11]); partial.append(path)
            raise OSError("injected partial disk write")
        return write(fd, data)
    monkeypatch.setattr(ew.os, "write", interrupted)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors)
    assert len(partial) == 1 and Path(partial[0]).stat().st_size == 11
    diagnostic = binding(Path(partial[0]))
    monkeypatch.setattr(ew.os, "write", write)
    with native_locks(db) as (lease, descriptors): preserve(db, lease, descriptors)
    assert binding(Path(partial[0])) == diagnostic
    assert len(list(journal(db).glob("*/completed.json"))) == 1


def test_lost_canonical_ofd_fence_is_detected_before_retirement(tmp_path, monkeypatch):
    db = database(tmp_path); wal = empty_wal(db); before = binding(wal)
    publish = ew._publish
    def unlock(parent, name, value):
        publish(parent, name, value)
        if name == "prepared.json":
            for item in Path('/proc/self/fd').iterdir():
                try:
                    if os.readlink(item) == str(db) and "OFDLCK" in Path('/proc/self/fdinfo', item.name).read_text():
                        fcntl.fcntl(int(item.name), fcntl.F_OFD_SETLK,
                            struct.pack('hhqqi4x', fcntl.F_UNLCK, os.SEEK_SET, 0, 0, 0))
                except FileNotFoundError: pass
    monkeypatch.setattr(ew, "_publish", unlock)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError, match="no longer held"):
        preserve(db, lease, descriptors)
    assert binding(wal) == before and not list(journal(db).glob("*/original.wal"))


def test_no_replace_publication_cannot_overwrite_existing_original(tmp_path, monkeypatch):
    db = database(tmp_path); wal = empty_wal(db); original = binding(wal)
    rename = ew._rename; collision = []
    def collide(source_fd, source, target_fd, target):
        if source == wal.name:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=target_fd)
            try: os.write(fd, b'foreign opaque bytes')
            finally: os.close(fd)
            collision.append(True)
        return rename(source_fd, source, target_fd, target)
    monkeypatch.setattr(ew, "_rename", collide)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors)
    assert collision and binding(wal) == original
    assert next(journal(db).glob("*/original.wal")).read_bytes() == b'foreign opaque bytes'
    assert not list(journal(db).glob("*/completed.json"))


def test_post_publish_directory_fsync_failure_requires_fresh_durability_barriers(tmp_path, monkeypatch):
    db = database(tmp_path); empty_wal(db)
    fsync = os.fsync
    failed = []
    def fail_after_publication(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if path.is_dir() and (path / "completed.json").exists():
            failed.append(str(path)); raise OSError("directory fsync unavailable")
        return fsync(fd)
    monkeypatch.setattr(ew.os, "fsync", fail_after_publication)
    with native_locks(db) as (lease, descriptors), pytest.raises(ew.EmptyOwnerWalError):
        preserve(db, lease, descriptors)
    assert failed and list(journal(db).glob("*/completed.json"))
    flushed = []
    def observe(fd):
        flushed.append(os.readlink(f"/proc/self/fd/{fd}")); return fsync(fd)
    monkeypatch.setattr(ew.os, "fsync", observe)
    with native_locks(db) as (lease, descriptors):
        assert preserve(db, lease, descriptors)["reason"] == "wal_already_preserved"
    for suffix in ('prepared.json', 'completed.json', 'original.wal'):
        assert any(path.endswith('/' + suffix) for path in flushed)
    assert str(journal(db)) in flushed and str(tmp_path) in flushed


def test_new_recovery_module_is_in_native_capsule_source_scope():
    from ipfs_accelerate_py import agent_implementation_route as route
    root = Path(__file__).resolve().parents[2]
    files = route._agent_control_plane_source_files(root, verify_loaded_origins=True)
    assert Path(ew.__file__).resolve() in files
    assert root / 'scripts/run_agent_supervisor_efficiency_state_hardening.py' in files
