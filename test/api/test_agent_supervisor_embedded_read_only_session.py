"""Source/factory contracts and genuine private physical-read-only checks.

No test establishes a managed executor, native task admission or legal truth.
The real-engine cases require the parent's explicit private-DuckDB profile;
no skip or substitute connection supplies physical read-only acceptance.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository as intent


@pytest.fixture
def private_store(tmp_path, monkeypatch):
    """Create the real engine only when the selected CALL phase invokes us.

    The fixture only caps memory when forwarding to the same genuine factory;
    it never substitutes a connection or the physical read_only flag.
    """
    used = False
    def create():
        nonlocal used
        assert used is False
        used = True
        import duckdb
        path = tmp_path / 'readonly-fixture.duckdb'
        assert not path.exists()
        native = state.connect_duckdb_with_policy(duckdb, path,
            configuration={'threads': 1, 'memory_limit': '128MB'})
        try:
            native.execute('CREATE TABLE sentinel(value INTEGER)')
            native.execute('INSERT INTO sentinel VALUES (7)')
        finally:
            native.close()
        original_factory = intent.open_duckdb_connection
        def memory_bounded_factory(path, **kwargs):
            assert 'memory_limit' not in kwargs
            return original_factory(path, memory_limit='128MB', **kwargs)
        monkeypatch.setattr(intent, 'open_duckdb_connection', memory_bounded_factory)
        monkeypatch.setenv(state.QUACK_PREFER_ENV, '0')
        return duckdb, path
    return create


def _metadata(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('explicit', [False, True], ids=['default', 'explicit-false'])
def test_default_factory_preserves_original_file_route(tmp_path, monkeypatch, explicit):
    marker = object(); calls = []
    def original(path, **kwargs):
        calls.append((path, kwargs)); return marker
    monkeypatch.setattr(state, '_open_file_duckdb_connection', original)
    kwargs = {'prefer_quack': False}
    if explicit: kwargs['read_only'] = False
    path = tmp_path / 'not-created.duckdb'
    assert state.open_duckdb_connection(path, **kwargs) is marker
    assert calls == [(path, {'timeout_seconds': state.DEFAULT_LOCK_TIMEOUT_SECONDS,
        'memory_limit': state.DEFAULT_MEMORY_LIMIT, 'threads': 1, 'owner_extension_load': False})]
    assert not path.exists()


@pytest.mark.parametrize('flag', [None, 0, 1, 'true'])
def test_physical_read_only_factory_requires_exact_boolean_before_work(tmp_path, monkeypatch, flag):
    def forbidden(*args, **kwargs): pytest.fail('invalid flag reached a backend route')
    monkeypatch.setattr(state, 'is_quack_transport_target', forbidden)
    with pytest.raises(TypeError, match='boolean'):
        state.open_duckdb_connection(tmp_path / 'unused.duckdb', read_only=flag)


@pytest.mark.parametrize('mode', ['memory', 'quack', 'extension-owner', 'prefer-quack'])
def test_physical_read_only_factory_refuses_other_authority_modes_before_work(tmp_path, monkeypatch, mode):
    path = tmp_path / 'ordinary-source.bin'; path.write_bytes(b'not a database, never opened')
    kwargs = {'read_only': True}
    if mode == 'memory': path = ':memory:'
    elif mode == 'quack': path = 'quack://127.0.0.1:1'
    elif mode == 'extension-owner': kwargs['owner_extension_load'] = True
    else: kwargs['prefer_quack'] = True
    def forbidden(*args, **kwargs): pytest.fail('unsupported authority reached native connection')
    monkeypatch.setattr(state, 'connect_duckdb_with_policy', forbidden)
    with pytest.raises(state.DuckDBConnectionPolicyError):
        state.open_duckdb_connection(path, **kwargs)


@pytest.mark.parametrize('kind', ['relative', 'missing', 'directory'])
def test_physical_read_only_factory_never_creates_a_database(tmp_path, monkeypatch, kind):
    path = 'relative.duckdb' if kind == 'relative' else tmp_path / 'missing.duckdb' if kind == 'missing' else tmp_path
    def forbidden(*args, **kwargs): pytest.fail('invalid physical file reached native connection')
    monkeypatch.setattr(state, 'connect_duckdb_with_policy', forbidden)
    with pytest.raises(state.DuckDBConnectionPolicyError, match='existing literal file'):
        state.open_duckdb_connection(path, read_only=True)
    assert not (tmp_path / 'missing.duckdb').exists()


def test_genuine_private_factory_is_physically_read_only_without_python_writer_lock(private_store, monkeypatch):
    duckdb, path = private_store(); before = _metadata(path); seen = []
    original_policy = state.connect_duckdb_with_policy
    def policy(*args, **kwargs):
        seen.append(kwargs.get('read_only')); return original_policy(*args, **kwargs)
    def forbidden(*args, **kwargs): pytest.fail('physical reader entered Python writer-lock constructor')
    monkeypatch.setattr(state, 'connect_duckdb_with_policy', policy)
    monkeypatch.setattr(state, 'exclusive_file_lock', forbidden)
    monkeypatch.setattr(state.DuckDBConnection, '__init__', forbidden)
    monkeypatch.setenv(state.QUACK_PREFER_ENV, '1')
    reader = state.open_duckdb_connection(path, read_only=True, memory_limit='128MB')
    try:
        assert reader._lock_context is None and reader.path is None
        assert reader.execute('SELECT value FROM sentinel').fetchone()[0] == 7
        with pytest.raises(duckdb.Error, match='read.only'):
            reader.execute('INSERT INTO sentinel VALUES (8)')
        with pytest.raises(state.DuckDBConnectionPolicyError, match='file-owning'):
            reader.reconnect_exclusive_owner()
        assert reader.execute('SELECT COUNT(*) FROM sentinel').fetchone()[0] == 1
    finally:
        reader.close()
    assert seen == [True] and _metadata(path) == before
    assert not path.with_name('.' + path.name + '.lock').exists()
    assert not Path(str(path) + '.wal').exists()


def test_genuine_private_read_session_reuses_only_the_outer_physical_reader(private_store):
    _, path = private_store(); before = _metadata(path)
    repo = intent.IntentRepository(path, install_schema=False)
    try:
        with repo.read_session(read_only=True):
            reader = repo._read_session_state.connection
            assert reader._lock_context is None and repo._read_session_state.read_only is True
            with repo.read_session():
                assert repo._read_session_state.connection is reader
            with repo.read_session(read_only=True):
                assert repo._read_session_state.connection is reader
            with repo._connection(write=False) as current:
                assert current is reader
                assert current.execute('SELECT value FROM sentinel').fetchone()[0] == 7
            with pytest.raises(intent.IntentRepositoryError, match='forbidden'):
                with repo._connection(write=True): pytest.fail('repository write escaped read session')
        assert repo._read_session_state.connection is None and repo._read_session_state.read_only is False
        assert reader._closed is True
    finally:
        repo.close()
    assert _metadata(path) == before and not path.with_name('.' + path.name + '.lock').exists()


@pytest.mark.parametrize('flag', [None, 0, 1, 'true'])
def test_read_session_new_flag_is_exact_boolean(tmp_path, monkeypatch, flag):
    # A source-only invalid-mode gate. Availability is not native engine credit.
    monkeypatch.setattr(intent, '_require_duckdb', lambda: None)
    repo = intent.IntentRepository(tmp_path / 'never-opened.bin', install_schema=False)
    def forbidden(*args, **kwargs): pytest.fail('invalid session flag opened a connection')
    monkeypatch.setattr(intent, 'open_duckdb_connection', forbidden)
    try:
        with pytest.raises(TypeError, match='boolean'):
            with repo.read_session(read_only=flag): pytest.fail('invalid session entered')
    finally: repo.close()


@pytest.mark.parametrize('mode', ['bound-writer', 'quack', 'nested-legacy'])
def test_physical_read_session_cannot_borrow_writer_or_transport_authority(private_store, tmp_path, monkeypatch, mode):
    repo = None
    try:
        if mode == 'nested-legacy':
            _, path = private_store()
            repo = intent.IntentRepository(path, install_schema=False)
            with repo.read_session():
                existing = repo._read_session_state.connection
                with pytest.raises(intent.IntentRepositoryError, match='legacy writer-mode'):
                    with repo.read_session(read_only=True): pytest.fail('borrowed writer was relabelled read-only')
                assert repo._read_session_state.connection is existing
                assert existing.execute('SELECT value FROM sentinel').fetchone()[0] == 7
        else:
            # Unit-only unsafe-authority markers, never a claimed native handle.
            monkeypatch.setattr(intent, '_require_duckdb', lambda: None)
            if mode == 'bound-writer': repo = intent.IntentRepository(bound_connection=object())
            else: repo = intent.IntentRepository('quack://127.0.0.1:1', install_schema=False)
            def forbidden(*args, **kwargs): pytest.fail('unsafe existing authority opened a second connection')
            monkeypatch.setattr(intent, 'open_duckdb_connection', forbidden)
            with pytest.raises(intent.IntentRepositoryError, match='unbound embedded'):
                with repo.read_session(read_only=True): pytest.fail('unsafe session entered')
    finally:
        if repo is not None: repo.close()


def test_physical_read_session_closes_and_clears_mode_after_body_failure(private_store):
    _, path = private_store(); repo = intent.IntentRepository(path, install_schema=False); reader = None
    try:
        with pytest.raises(ValueError, match='explicit body failure'):
            with repo.read_session(read_only=True):
                reader = repo._read_session_state.connection
                raise ValueError('explicit body failure')
        assert repo._read_session_state.connection is None and repo._read_session_state.read_only is False
        assert reader is not None and reader._closed is True
    finally: repo.close()
