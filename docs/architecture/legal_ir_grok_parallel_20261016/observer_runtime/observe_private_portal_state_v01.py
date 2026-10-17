"""Read the genuine one-attempt Portal state without any database writes.

The canonical Quack task claim and this private Portal state remain separate.
The private target is derived from the pinned native portal_factory source,
then opened read-only under the writer's existing shared-file-lock protocol.
No endpoint discovery, credential transfer, schema installation, DDL, state
binding mutation, or fallback to the canonical task database occurs here.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat

SOURCE_PATHS = (
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/portal_task_state_control_plane.py',
    'ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py',
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_DATABASE_BYTES = 64 * 1024 * 1024


class PrivatePortalStatePending(ValueError):
    """The native producer has not published an available committed row yet."""


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
            info.st_gid, info.st_nlink)


def _witness(info):
    return (*_identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _regular(path, *, maximum, allow_empty=False):
    _need(path.is_absolute() and path.resolve(strict=True) == path,
          'private Portal input must have an exact non-aliased path')
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
          and info.st_nlink == 1 and not info.st_mode & 0o002
          and (not info.st_mode & 0o020 or info.st_gid == os.getegid())
          and (0 <= info.st_size <= maximum if allow_empty else 0 < info.st_size <= maximum),
          'private Portal file custody differs')
    return info


def _source_bytes(path, expected):
    before = _regular(path, maximum=MAX_SOURCE_BYTES)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw = bytearray()
        while len(raw) <= MAX_SOURCE_BYTES:
            chunk = os.read(fd, min(65536, MAX_SOURCE_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        _need(_witness(before) == _witness(os.fstat(fd)) == _witness(path.lstat())
              and len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == expected,
              'private Portal native source changed or pin differs')
        return bytes(raw)
    finally:
        os.close(fd)


def _native_target_basename(runner_source):
    """Extract the actual reviewed nested portal_factory literal binding."""
    tree = ast.parse(runner_source)
    outer = [item for item in tree.body if isinstance(item, ast.FunctionDef)
             and item.name == 'bind_database_portal_execution_from_args']
    _need(len(outer) == 1, 'native Portal factory owner differs')
    factory = [item for item in outer[0].body if isinstance(item, ast.FunctionDef)
               and item.name == 'portal_factory']
    _need(len(factory) == 1, 'native Portal factory differs')
    assignments = [item for item in factory[0].body if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'state_database'
                           for target in item.targets)]
    _need(len(assignments) == 1, 'native private state assignment differs')
    value = assignments[0].value
    _need(isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
          and ast.dump(value.left) == ast.dump(ast.Attribute(value=ast.Name(id='paths', ctx=ast.Load()),
                                                            attr='root', ctx=ast.Load()))
          and isinstance(value.right, ast.Constant) and type(value.right.value) is str,
          'native private state target is not a literal child of the attempt root')
    basename = value.right.value
    _need(re.fullmatch(r'[A-Za-z0-9_.-]{1,96}\.duckdb', basename) is not None,
          'native private state basename differs')
    calls = [item for item in factory[0].body if isinstance(item, ast.Expr)
             and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name)
             and item.value.func.id == 'bind_task_state_control_plane']
    _need(len(calls) == 1 and ast.dump(calls[0].value) == ast.dump(ast.Call(
        func=ast.Name(id='bind_task_state_control_plane', ctx=ast.Load()),
        args=[ast.Attribute(value=ast.Name(id='paths', ctx=ast.Load()), attr='state', ctx=ast.Load()),
              ast.Call(func=ast.Name(id='str', ctx=ast.Load()),
                       args=[ast.Name(id='state_database', ctx=ast.Load())], keywords=[])], keywords=[])),
        'native private state binding differs')
    return basename


def observe_private_portal_state(*, source_root, projection_observation,
                                 expected_claim, source_sha256):
    """Return a committed native state snapshot and a non-authoritative receipt.

    A missing DB/row or busy native writer is pending. Custody, source, projection
    or identity mismatches are refusals. All SQL is SELECT through the genuine
    public configuration-locked read-only DuckDB connection API.
    """
    root = Path(source_root)
    _need(root.is_absolute() and root.resolve(strict=True) == root, 'exact source root required')
    _need(type(source_sha256) is dict and set(source_sha256) == set(SOURCE_PATHS),
          'closed private Portal owner source pins required')
    sources = {relative: _source_bytes(root / relative, source_sha256[relative])
               for relative in SOURCE_PATHS}
    basename = _native_target_basename(sources[SOURCE_PATHS[1]])
    modules = [importlib.import_module(relative[:-3].replace('/', '.')) for relative in SOURCE_PATHS]
    for module, relative in zip(modules, SOURCE_PATHS):
        _need(Path(module.__file__).resolve(strict=True) == root / relative,
              'private Portal native owner import origin differs')
    bridge, _runner, state_owner, connection_owner = modules
    owner = bridge.DatabasePortalExecutionBridge
    projection = Path(projection_observation['projection_file']['path'])
    paths = owner._direct_attempt_paths(projection.parent)
    _need(paths.task_projection == projection, 'private state projection path differs')
    binding = owner._strict_binding(paths.binding)
    for key in ('task_cid', 'attempt_id', 'claim_id', 'lease_id', 'task_revision',
                'fencing_token', 'fence_epoch'):
        _need(binding.get(key) == expected_claim.get(key), 'private state native claim differs')
    _need(binding['binding_id'] == projection_observation['native_binding_id']
          and binding['task_alias'] == 'LIR-9001', 'private state native binding differs')
    projection_text = owner._verify_projection(paths, binding)
    _need(hashlib.sha256(projection_text.encode()).hexdigest()
          == projection_observation['projection_file']['sha256'], 'private state projection changed')
    directory = paths.root.lstat()
    _need(paths.root.resolve(strict=True) == paths.root and stat.S_ISDIR(directory.st_mode)
          and directory.st_uid == os.geteuid() and not directory.st_mode & 0o022,
          'private state attempt directory custody differs')
    database = paths.root / basename
    lock = database.with_name('.' + database.name + '.lock')
    try:
        before = _regular(database, maximum=MAX_DATABASE_BYTES)
        lock_before = _regular(lock, maximum=65536, allow_empty=True)
    except FileNotFoundError as error:
        raise PrivatePortalStatePending('private native state has not been published') from error
    lock_fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    connection = None
    try:
        _need(_witness(lock_before) == _witness(os.fstat(lock_fd)), 'private state lock changed')
        try:
            # The native writer takes LOCK_EX on this exact existing inode.
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PrivatePortalStatePending('private native state writer is active') from error
        locked_database = _regular(database, maximum=MAX_DATABASE_BYTES)
        _need(_identity(before) == _identity(locked_database),
              'private state database identity changed before the native lock was acquired')
        # A native commit may complete between the initial path check and lock
        # acquisition. Observe its committed file witness only after locking.
        before = locked_database
        wal = database.with_name(database.name + '.wal')
        if os.path.lexists(wal):
            _regular(wal, maximum=MAX_DATABASE_BYTES, allow_empty=True)
        import duckdb
        connection = connection_owner.connect_duckdb_with_policy(
            duckdb, database, read_only=True,
            configuration={'threads': 1, 'memory_limit': '64MB'})
        _need(state_owner.TABLE == 'portal_task_state', 'native private state table differs')
        tables = connection.execute(
            'SELECT table_type FROM information_schema.tables '
            'WHERE table_catalog = current_database() AND table_schema = ? '
            'AND table_name = ?',
            ['main', 'portal_task_state']).fetchall()
        if not tables:
            raise PrivatePortalStatePending('private native state table has not been published')
        _need(tables == [('BASE TABLE',)], 'private native state table identity differs')
        # Once the exact native table exists, missing columns or invalid SQL
        # reflect schema drift and must refuse rather than wait indefinitely.
        row = connection.execute(
            'SELECT body_json, revision, completion_authority, updated_at '
            'FROM portal_task_state WHERE lane_key = ? '
            'AND octet_length(encode(body_json)) BETWEEN 1 AND ?',
            [paths.state.stem, MAX_STATE_BYTES]).fetchone()
        if row is None:
            raise PrivatePortalStatePending('private native state row has not been published')
        _need(type(row[0]) is str and type(row[1]) is int and row[1] > 0
              and row[2] is False and type(row[3]) is str and len(row[3]) <= 128,
              'private native state row shape differs')
        def pairs(items):
            result = {}
            for key, value in items:
                _need(key not in result, 'private native state has duplicate JSON keys')
                result[key] = value
            return result
        def invalid_constant(_value):
            raise ValueError('private native state has nonfinite JSON')
        body = json.loads(row[0], object_pairs_hook=pairs, parse_constant=invalid_constant)
        _need(type(body) is dict and body.get('completion_authority') is False,
              'private state body cannot grant completion authority')
        raw = _canonical(body)
        _need(len(raw) <= MAX_STATE_BYTES, 'private native state exceeds its bound')
        _need(_witness(before) == _witness(_regular(database, maximum=MAX_DATABASE_BYTES))
              and _identity(directory) == _identity(paths.root.lstat())
              and _witness(lock_before) == _witness(os.fstat(lock_fd)) == _witness(lock.lstat()),
              'private native state custody changed during read')
        for relative in SOURCE_PATHS:
            _need(_source_bytes(root / relative, source_sha256[relative]) == sources[relative],
                  'private native owner source changed during read')
        _need(owner._strict_binding(paths.binding) == binding
              and owner._verify_projection(paths, binding) == projection_text,
              'private native projection changed during read')
        receipt = {'schema': 'legal-ir-private-native-portal-state-observation/v1',
            'database_path': str(database), 'lane_key': paths.state.stem,
            'native_binding_id': binding['binding_id'], 'database_task_cid': binding['task_cid'],
            'native_attempt_id': binding['attempt_id'], 'row_revision': row[1],
            'row_updated_at': row[3], 'body_bytes': len(raw),
            'body_sha256': hashlib.sha256(raw).hexdigest(),
            'read_only': True, 'native_shared_lock_observed': True,
            'schema_installation_attempted': False, 'canonical_database_opened': False,
            'completion_authority': False, 'claim_authority': False,
            'owner_source_sha256': source_sha256}
        receipt['observation_id'] = hashlib.sha256(_canonical(receipt)).hexdigest()
        return raw, receipt
    finally:
        if connection is not None:
            connection.close()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
