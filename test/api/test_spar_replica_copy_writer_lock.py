"""The native replica copier must leave the real parent writer lock intact."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server as quack


def writer_lock(database):
    value = database.stat()
    identity = (os.major(value.st_dev), os.minor(value.st_dev), value.st_ino)
    for line in Path('/proc/locks').read_text().splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[1:5] == ['POSIX', 'ADVISORY', 'WRITE', str(os.getpid())] and parts[6:] == ['0', 'EOF']:
            major, minor, ino = parts[5].split(':')
            if (int(major, 16), int(minor, 16), int(ino)) == identity:
                return True
    return False


def assert_other_writer_blocked(database):
    import duckdb
    result = subprocess.run([sys.executable, '-I', '-c', '''
import sys
sys.path.insert(0, sys.argv[2])
import duckdb
try:
    connection = duckdb.connect(sys.argv[1], config={'threads': 1})
except duckdb.IOException as error:
    sys.exit(69 if 'lock' in str(error).lower() else 2)
else:
    connection.close()
    sys.exit(0)
''', str(database), str(Path(duckdb.__file__).resolve().parent.parent)],
        env={'PATH': os.defpath, 'LANG': 'C.UTF-8'}, capture_output=True, timeout=15)
    assert result.returncode == 69


@pytest.fixture
def writer(tmp_path):
    duckdb = pytest.importorskip('duckdb')
    database = tmp_path / 'control.duckdb'
    connection = duckdb.connect(str(database), config={'threads': 1})
    connection.execute('CREATE TABLE progress AS SELECT 1 AS value')
    connection.execute('CHECKPOINT')
    server = quack.build_server(database_path=database, state_dir=tmp_path / 'owner', transport=quack.FakeQuackTransport())
    server._connection = connection
    server._open_database_parent_anchor()
    server._bind_database_inode_after_migration()
    try:
        assert writer_lock(database)
        yield server, connection, database
    finally:
        connection.close()
        server._connection = None
        server._close_database_namespace_anchor()


def test_repeated_native_copy_preserves_real_writer_and_later_mutation(writer):
    server, connection, database = writer
    assert_other_writer_blocked(database)
    for value in (2, 3):
        digest, size = server._copy_authoritative_read_replica()
        assert writer_lock(database), 'native replica copy dropped parent POSIX writer lock'
        assert_other_writer_blocked(database)
        data = server.read_replica_path().read_bytes()
        assert len(data) == size and digest == 'sha256:' + hashlib.sha256(data).hexdigest()
        connection.execute('INSERT INTO progress VALUES (?)', [value])
    assert connection.execute('SELECT SUM(value) FROM progress').fetchone()[0] == 6


def test_native_temp_collision_preserves_all_existing_work_and_lock(writer, monkeypatch):
    server, _connection, database = writer
    nonce = 'a' * 32
    monkeypatch.setattr(quack.uuid, 'uuid4', lambda: SimpleNamespace(hex=nonce))
    original = server.read_replica_path().with_name(f'.{server.read_replica_path().name}.{os.getpid()}.{nonce}.tmp')
    child = server.read_replica_path().with_name(f'.{server.read_replica_path().name}.{nonce}.tmp')
    original.write_bytes(b'prior native temporary belongs to its owner')
    child.write_bytes(b'prior child temporary belongs to its owner')
    with pytest.raises(quack.QuackStateServerReadyError):
        server._copy_authoritative_read_replica()
    assert original.read_bytes() == b'prior native temporary belongs to its owner'
    assert child.read_bytes() == b'prior child temporary belongs to its owner'
    assert writer_lock(database)
    assert_other_writer_blocked(database)


def test_real_quack_native_start_and_refresh_keep_writer_exclusion(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import QuackCapabilityStatus
    capability = quack.probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip('real preinstalled Quack is unavailable')
    database = tmp_path / 'control.duckdb'
    server = quack.build_server(database_path=database, state_dir=tmp_path / 'owner',
        secret_handle='handle:spar-replica-regression', allow_legacy_board_unstall=False)
    try:
        server.start()
        assert writer_lock(database), 'native startup replica dropped parent POSIX writer lock'
        assert_other_writer_blocked(database)
        # This native runtime's checkpoint API does not refresh the replica.
        # Exercise the actual refresh under its existing two serialization gates.
        with server._owner_transaction_lock:
            with server._lock:
                server._refresh_read_replica()
        assert writer_lock(database)
        assert_other_writer_blocked(database)
        assert server.ready()['ready'] is True
        assert server.checkpoint()['checkpointed'] is True
        assert writer_lock(database)
    finally:
        server.stop()
    assert not writer_lock(database)


def child_request(database, target):
    from ipfs_accelerate_py.agent_supervisor.runtime import replica_file_copy as copier
    return {"source": str(database), "target": str(target), "nonce": "b" * 32,
            "source_identity": copier._identity(database.lstat())}


def run_copy_child(request, injection=""):
    from ipfs_accelerate_py.agent_supervisor.runtime import replica_file_copy as copier
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
    return subprocess.run([sys.executable, '-I', '-S', '-c', script, copier.__file__],
        input=json.dumps(request), capture_output=True, text=True, timeout=10,
        env={'PATH': os.defpath, 'LANG': 'C.UTF-8'})


@pytest.mark.parametrize('replacement', ['foreign_regular', 'same_size_foreign', 'symlink', 'truncated_owned'])
def test_child_promotion_requires_exact_owned_regular_entry(writer, replacement):
    server, connection, database = writer
    target = server.read_replica_path()
    target.write_bytes(b'previous replica')
    request = child_request(database, target)
    temporary = target.with_name(f".{target.name}.{request['nonce']}.tmp")
    displaced = temporary.with_suffix('.preserved')
    foreign = target.with_name('foreign-entry')
    foreign.write_bytes(b'foreign work')
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
        if mode in ('foreign_regular', 'same_size_foreign'):
            temporary.write_bytes(b'F' * request['source_identity'][2] if mode == 'same_size_foreign' else b'foreign work')
        else:
            temporary.symlink_to('foreign-entry')
os.fsync = interfere
""".replace('MODE', repr(replacement))
    result = run_copy_child(request, injection)
    assert result.returncode == 1, 'copier promoted an entry without its exact ownership'
    assert result.stdout == ''
    assert target.read_bytes() == b'previous replica'
    assert foreign.read_bytes() == b'foreign work'
    if replacement == 'truncated_owned':
        assert not temporary.exists()
    else:
        assert displaced.exists()
        assert temporary.read_bytes() == (b'F' * request['source_identity'][2] if replacement == 'same_size_foreign' else b'foreign work')
        assert temporary.is_symlink() == (replacement == 'symlink')
    assert writer_lock(database)
    assert_other_writer_blocked(database)
    connection.execute('INSERT INTO progress VALUES (2)')
    assert connection.execute('SELECT SUM(value) FROM progress').fetchone()[0] == 3


def test_child_rejects_stale_source_identity_without_disturbing_writer(writer):
    server, connection, database = writer
    target = server.read_replica_path()
    target.write_bytes(b'previous replica')
    request = child_request(database, target)
    connection.execute('INSERT INTO progress VALUES (2)')
    connection.execute('CHECKPOINT')
    assert request['source_identity'] != child_request(database, target)['source_identity']
    result = run_copy_child(request)
    assert result.returncode == 1 and result.stdout == ''
    assert target.read_bytes() == b'previous replica'
    assert not list(database.parent.glob(f'.{target.name}.*.tmp'))
    assert writer_lock(database)
    assert_other_writer_blocked(database)
    assert connection.execute('SELECT SUM(value) FROM progress').fetchone()[0] == 3


@pytest.mark.parametrize('symlink_side', ['source', 'target'])
def test_parent_rejects_symlinks_and_preserves_existing_bytes(writer, symlink_side):
    from ipfs_accelerate_py.agent_supervisor.runtime import replica_file_copy as copier
    server, _connection, database = writer
    source, target = database, server.read_replica_path()
    foreign = database.with_name('foreign-content')
    foreign.write_bytes(b'foreign content')
    if symlink_side == 'source':
        source = database.with_name('database-alias')
        source.symlink_to(database.name)
    else:
        target.symlink_to(foreign.name)
    with pytest.raises(copier.ReplicaCopyUnavailable):
        copier.copy_replica(source, target)
    assert (source if symlink_side == 'source' else target).is_symlink()
    assert foreign.read_bytes() == b'foreign content'
    assert writer_lock(database)


@pytest.mark.parametrize('options', [
    {'timeout_seconds': 0}, {'timeout_seconds': -1}, {'timeout_seconds': 31},
    {'timeout_seconds': float('nan')}, {'timeout_seconds': float('inf')},
    {'timeout_seconds': True}, {'max_database_bytes': 0},
    {'max_database_bytes': 1}, {'max_database_bytes': 8 * 1024**3 + 1},
    {'max_database_bytes': True},
])
def test_invalid_copy_budget_never_launches_or_replaces_replica(writer, monkeypatch, options):
    from ipfs_accelerate_py.agent_supervisor.runtime import replica_file_copy as copier
    server, _connection, database = writer
    target = server.read_replica_path()
    target.write_bytes(b'previous replica')
    def forbidden(*args, **kwargs):
        pytest.fail('invalid budget launched copier child')
    monkeypatch.setattr(copier.subprocess, 'Popen', forbidden)
    with pytest.raises(copier.ReplicaCopyUnavailable):
        copier.copy_replica(database, target, **options)
    assert target.read_bytes() == b'previous replica'
    assert writer_lock(database)


def test_checkpoint_consumes_native_copy_deadline_before_child_launch(writer, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import replica_file_copy as copier
    server, _connection, database = writer
    target = server.read_replica_path()
    target.write_bytes(b'previous replica')
    clock = iter((50.0, 81.0))
    def forbidden(*args, **kwargs):
        pytest.fail('expired native checkpoint budget launched copier child')
    monkeypatch.setattr(copier.subprocess, 'Popen', forbidden)
    with monkeypatch.context() as context:
        context.setattr(quack.time, 'monotonic', lambda: next(clock))
        with pytest.raises(quack.QuackStateServerReadyError):
            server._copy_authoritative_read_replica()
    assert target.read_bytes() == b'previous replica'
    assert writer_lock(database)


def test_parent_never_opens_canonical_inode_or_inherits_ambient_authority(writer, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import replica_file_copy as copier
    server, _connection, database = writer
    original_open, original_popen = os.open, subprocess.Popen
    launches = []
    def guarded_open(path, *args, **kwargs):
        assert Path(path) != database, 'parent acquired another canonical descriptor'
        return original_open(path, *args, **kwargs)
    def checked_launch(argv, **kwargs):
        assert argv[:3] == [sys.executable, '-I', '-S']
        assert kwargs['env'] == {'PATH': os.defpath, 'LANG': 'C.UTF-8'}
        assert kwargs['close_fds'] is True and not kwargs.get('pass_fds')
        launches.append(True)
        return original_popen(argv, **kwargs)
    monkeypatch.setenv('QUACK_TEST_AMBIENT_AUTHORITY', 'must-not-be-inherited')
    monkeypatch.setattr(copier.os, 'open', guarded_open)
    monkeypatch.setattr(copier.subprocess, 'Popen', checked_launch)
    # Exercise the standalone older-lineage copier. The current owner uses
    # its stronger retained namespace anchor, covered by the native tests.
    copier.copy_replica(database, server.read_replica_path())
    assert launches == [True]
    assert writer_lock(database)


def test_child_write_failure_removes_only_its_owned_temp(writer):
    server, connection, database = writer
    target = server.read_replica_path()
    target.write_bytes(b'previous replica')
    request = child_request(database, target)
    preserved = target.with_name('.other-copier.tmp')
    preserved.write_bytes(b'another copier work')
    result = run_copy_child(request, """
def failed_write(*args):
    raise OSError('injected write failure')
os.write = failed_write
""")
    assert result.returncode == 1 and result.stdout == ''
    assert target.read_bytes() == b'previous replica'
    temporary = target.with_name(f".{target.name}.{request['nonce']}.tmp")
    assert not temporary.exists()
    assert preserved.read_bytes() == b'another copier work'
    assert writer_lock(database)
    assert_other_writer_blocked(database)
    connection.execute('INSERT INTO progress VALUES (2)')
    assert connection.execute('SELECT SUM(value) FROM progress').fetchone()[0] == 3


@pytest.mark.parametrize('mutation', ['truncate', 'same_size_write'])
def test_child_directory_flush_drift_never_emits_success_receipt(writer, mutation):
    server, _connection, database = writer
    target = server.read_replica_path()
    target.write_bytes(b'previous replica')
    request = child_request(database, target)
    injection = """
import stat
original_fsync = os.fsync
def interfere(fd):
    original_fsync(fd)
    if stat.S_ISDIR(os.fstat(fd).st_mode):
        with open(request['target'], 'r+b') as changed:
            if MODE == 'truncate':
                changed.truncate(1)
            else:
                changed.write(b'X')
os.fsync = interfere
""".replace('MODE', repr(mutation))
    result = run_copy_child(request, injection)
    assert result.returncode == 1, 'copier accepted post-promotion file drift'
    assert result.stdout == ''
    assert target.exists(), 'uncertain promoted artifact must not be guessed-deleted'
    assert writer_lock(database)
    assert_other_writer_blocked(database)
