"""Opt-in, MainPID-bound health after an original native executor cycle.

No database, Portal, migration, provider, claim or arbitrary command runs here.
The native owner retains execution authority; this module verifies its closed
artifacts and distinguishes current executor health from task completion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time

PROFILE_SCHEMA = 'agent-supervisor/native-executor-health-profile@1'
RECEIPT_SCHEMA = 'agent-supervisor/native-executor-main-loop-health@1'
CYCLE_SCHEMA = 'agent-supervisor/native-executor-original-cycle-links@1'
DIAGNOSTIC_SCHEMA = 'native-catalogue-diagnostic/v1'
IDENTITY_KEYS = frozenset({'board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256', 'database_uuid', 'store_generation'})
PROFILE_KEYS = frozenset({'schema', 'catalogue', 'unit', 'receipt_path', 'source_descriptors', 'main_entrypoint', 'health_module', 'owner_source', 'driver_source', 'task_scope', 'max_age_seconds', 'max_cycle_seconds', 'proof_root', 'cycle_contract'})
CYCLE_KEYS = frozenset({'schema', 'kind', 'sequence', 'started_at', 'finished_at', 'started_monotonic_ns', 'finished_monotonic_ns', 'process', 'request_ref', 'execution_request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref', 'owner_observation'})
RECEIPT_KEYS = frozenset({'schema', 'profile_ref', 'sequence', 'identity', 'observed_at', 'observed_monotonic_ns', 'cycle_proof_ref', 'catalogue', 'origins', 'previous_receipt_sha256'})
MAX_JSON = 4 * 1024**2
MAX_SOURCE = 4 * 1024**2
MAX_INSTALLED_TOOL = 64 * 1024**2
MAX_DATABASE_DATA_BYTES = 128 * 1024**2
MAX_EPOCH_ARCHIVES = 64
MAX_ARCHIVE_DIRECTORY_ENTRIES = 4096
ENV_PROFILE_SHA = 'IPFS_NATIVE_EXECUTOR_PROFILE_SHA256'
SAFE_ENV = frozenset({'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'IPFS_DATASETS_PY_MINIMAL_IMPORTS', 'IPFS_DATASETS_AUTO_INSTALL', 'IPFS_AUTO_INSTALL', 'PYTHONNOUSERSITE'})

_CYCLES = {}
_USED_REQUESTS = set()

class _CycleHandle:
    __slots__ = ()

class ExecutorHealthError(ValueError):
    pass

def _need(value, code):
    if not value:
        raise ExecutorHealthError(code)

def _keys(value, expected, code):
    _need(type(value) is dict and set(value) == set(expected), code)

def _integer(value, minimum=0, maximum=2**63-1):
    return type(value) is int and minimum <= value <= maximum

def _number(value):
    return type(value) in (int, float) and math.isfinite(value)

def _text(value):
    return type(value) is str and 0 < len(value) <= 8192 and '\0' not in value and '\n' not in value and '\r' not in value

def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def _decode(raw):
    def pairs(rows):
        out = {}
        for key, value in rows:
            _need(key not in out, 'duplicate_json_key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ExecutorHealthError('nonfinite_json')))

def _path(value):
    _need(_text(value), 'invalid_path')
    path = Path(value)
    _need(path.is_absolute() and str(path) == os.path.abspath(value) and not any(p.is_symlink() for p in (path, *path.parents)), 'aliased_or_relative_path')
    return path

def _raw(path, maximum=MAX_JSON):
    path = _path(str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        first = os.fstat(stream.fileno())
        _need(stat.S_ISREG(first.st_mode) and first.st_nlink == 1 and first.st_uid in (0, os.geteuid()) and not first.st_mode & 0o022 and 0 <= first.st_size <= maximum, 'unsafe_or_oversized_file')
        raw = stream.read(maximum + 1)
        last = os.fstat(stream.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _need(len(raw) == first.st_size and stamp(first) == stamp(last) == stamp(path.lstat()), 'file_changed_during_read')
    return raw

def descriptor(path, *, maximum=MAX_JSON):
    raw = _raw(path, maximum)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def database_descriptor(path):
    """Describe bounded mutable catalogue DATA, without opening a database.

    Group-writable owner data is allowed. Control documents, project source and
    installed tools continue using their existing readers and mode checks. The
    streaming hash certifies one stable file observation, not a SQL snapshot.
    """
    path = _path(str(path))
    before = path.lstat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid,
                       s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        first = os.fstat(fd)
        _need(stat.S_ISREG(first.st_mode) and first.st_nlink == 1 and
              first.st_uid == os.geteuid() and not first.st_mode & 0o002 and
              0 < first.st_size <= MAX_DATABASE_DATA_BYTES,
              'unsafe_or_oversized_database_data')
        _need(stamp(before) == stamp(first), 'database_data_changed_before_open')
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            _need(total <= first.st_size and total <= MAX_DATABASE_DATA_BYTES,
                  'database_data_grew_during_read')
            digest.update(chunk)
        last = os.fstat(fd)
        current = _path(str(path)).lstat()
        _need(total == first.st_size and stamp(first) == stamp(last) == stamp(current),
              'database_data_changed_during_read')
        return dict(path=str(path), bytes=total, sha256=digest.hexdigest())
    finally:
        os.close(fd)

def installed_tool_descriptor(path):
    """Describe an installed ELF interpreter, outside the project-source class.

    Only the exact unit.interpreter slot receives this larger bound in profiles;
    project sources and imported module origins retain their ordinary limit.
    """
    path = _path(str(path))
    raw = _raw(path, MAX_INSTALLED_TOOL)
    _need(raw.startswith(b'\x7fELF'), 'installed_tool_not_elf')
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def _bound(ref, maximum=MAX_JSON):
    _keys(ref, {'path', 'bytes', 'sha256'}, 'descriptor_shape')
    _need(_integer(ref['bytes'], 0, maximum) and type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'descriptor_bounds')
    raw = _raw(ref['path'], maximum)
    _need(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'descriptor_changed')
    return raw

def _bound_installed_tool(ref):
    raw = _bound(ref, MAX_INSTALLED_TOOL)
    _need(raw.startswith(b'\x7fELF'), 'installed_tool_not_elf')
    return raw

def _bound_project_source(ref, profile):
    _keys(ref, {'path', 'bytes', 'sha256'}, 'descriptor_shape')
    _need(ref['path'] != profile['unit']['interpreter'], 'installed_tool_is_not_project_source')
    return _bound(ref, MAX_SOURCE)

def _document(ref):
    value = _decode(_bound(ref))
    _need(type(value) is dict, 'metadata_document_not_object')
    return value

def _profile(ref):
    profile = _document(ref)
    _keys(profile, PROFILE_KEYS, 'profile_shape')
    _need(profile['schema'] == PROFILE_SCHEMA, 'profile_schema')
    _keys(profile['catalogue'], IDENTITY_KEYS, 'catalogue_binding_shape')
    _need(all(_text(profile['catalogue'][k]) for k in IDENTITY_KEYS - {'store_generation'}) and _integer(profile['catalogue']['store_generation'], 1), 'catalogue_binding_values')
    _need(re.fullmatch('[0-9a-f]{64}', str(profile['catalogue']['population_sha256'])), 'population_digest')
    _keys(profile['unit'], {'name', 'argv', 'interpreter', 'working_directory', 'environment'}, 'unit_binding_shape')
    unit = profile['unit']
    _need(_text(unit['name']) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@:-]*\.service', unit['name']), 'unit_name')
    _need(type(unit['argv']) is list and 1 <= len(unit['argv']) <= 128 and all(_text(v) for v in unit['argv']), 'unit_argv')
    _path(unit['interpreter']); _path(unit['working_directory'])
    _need(type(unit['environment']) is dict and 0 < len(unit['environment']) <= 32 and all(_text(k) and _text(v) for k, v in unit['environment'].items()), 'unit_environment')
    _need(set(unit['environment']) <= SAFE_ENV, 'unapproved_environment_field')
    _need(type(profile['source_descriptors']) is list and 4 <= len(profile['source_descriptors']) <= 4096, 'source_population')
    # Classify by the already validated exact role/path before reading content.
    # No required project role may borrow the installed interpreter allowance.
    for field in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'):
        _keys(profile[field], {'path', 'bytes', 'sha256'}, 'required_source_shape')
        _need(profile[field]['path'] != unit['interpreter'], 'project_role_aliases_installed_tool')
    sources = {}
    total = 0
    for pin in profile['source_descriptors']:
        _keys(pin, {'path', 'bytes', 'sha256'}, 'descriptor_shape')
        if pin['path'] == unit['interpreter']:
            _bound_installed_tool(pin)
        else:
            _bound_project_source(pin, profile)
        _need(pin['path'] not in sources, 'duplicate_source')
        sources[pin['path']] = pin
        total += pin['bytes']
        _need(total <= 128 * 1024**2, 'source_total_bound')
    _need(unit['interpreter'] in sources, 'interpreter_source_not_pinned')
    for field in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'):
        pin = profile[field]
        _need(type(pin) is dict and sources.get(pin.get('path')) == pin, 'required_source_not_pinned')
    _need(profile['main_entrypoint']['path'] in unit['argv'], 'entrypoint_not_in_unit_argv')
    _need(type(profile['task_scope']) is list and 0 < len(profile['task_scope']) <= 1000 and len(set(profile['task_scope'])) == len(profile['task_scope']) and all(_text(v) for v in profile['task_scope']), 'task_scope')
    _need(_integer(profile['max_age_seconds'], 1, 300) and _integer(profile['max_cycle_seconds'], 1, 300), 'finite_health_limits')
    _path(profile['receipt_path']); _path(profile['proof_root'])
    _need(Path(profile['proof_root']) in Path(profile['receipt_path']).parents, 'receipt_outside_proof_root')
    _keys(profile['cycle_contract'], {'plan_schema', 'plan_operation'}, 'cycle_contract_shape')
    _need(all(_text(v) for v in profile['cycle_contract'].values()), 'cycle_contract_values')
    return profile

def _manager(unit):
    argv = ['/usr/bin/systemctl', '--user', 'show', unit, *['--property=' + key for key in ('Id', 'MainPID', 'ActiveState', 'SubState', 'InvocationID', 'ExecMainStartTimestampMonotonic', 'ControlGroup')]]
    result = subprocess.run(argv, capture_output=True, timeout=5, check=False)
    _need(result.returncode == 0 and len(result.stdout) <= 65536 and len(result.stderr) <= 65536, 'manager_unavailable')
    return dict(line.split('=', 1) for line in result.stdout.decode().splitlines() if '=' in line)

def _process(pid):
    root = Path('/proc') / str(pid)
    first = (root / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    argv = (root / 'cmdline').read_bytes()
    environ = (root / 'environ').read_bytes()
    _need(len(argv) <= 65536 and len(environ) <= 1024**2, 'process_metadata_bound')
    environment = dict(part.decode().split('=', 1) for part in environ.split(b'\0') if b'=' in part)
    value = dict(pid=pid, birth_ticks=int(first), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), argv=[v.decode() for v in argv.split(b'\0') if v], exe=os.readlink(root / 'exe'), cwd=os.readlink(root / 'cwd'), environment=environment)
    _need((root / 'stat').read_text().rsplit(') ', 1)[1].split()[19] == first, 'process_birth_changed')
    return value

def _identity(profile_ref, profile, manager_reader, process_reader):
    unit = profile['unit']
    manager = manager_reader(unit['name'])
    _need(type(manager) is dict, 'manager_observation_shape')
    _need(manager.get('Id') == unit['name'] and manager.get('ActiveState') == 'active' and manager.get('SubState') == 'running', 'unit_not_running')
    pid_text = str(manager.get('MainPID', ''))
    _need(pid_text.isdecimal() and int(pid_text) > 1, 'main_pid_unavailable')
    process = process_reader(int(pid_text))
    _need(type(process) is dict and process.get('pid') == int(pid_text) and _integer(process.get('birth_ticks'), 1), 'process_identity')
    _need(process.get('argv') == unit['argv'] and process.get('exe') == unit['interpreter'] and process.get('cwd') == unit['working_directory'], 'process_entrypoint_changed')
    tools = [pin for pin in profile['source_descriptors'] if pin['path'] == process['exe']]
    _need(len(tools) == 1, 'process_installed_tool_not_pinned')
    _bound_installed_tool(tools[0])
    invocation = manager.get('InvocationID')
    _need(type(invocation) is str and re.fullmatch('[0-9a-f]{32}', invocation), 'invocation_unavailable')
    environment = process.get('environment', {})
    _need(type(environment) is dict, 'process_environment_shape')
    _need(str(manager.get('ExecMainStartTimestampMonotonic', '')).isdecimal() and int(manager['ExecMainStartTimestampMonotonic']) > 0 and _text(manager.get('ControlGroup')), 'manager_start_or_cgroup_unavailable')
    required = {**unit['environment'], 'INVOCATION_ID': invocation, ENV_PROFILE_SHA: profile_ref['sha256']}
    _need(all(environment.get(k) == v for k, v in required.items()), 'process_environment_changed')
    _need(_text(process.get('boot_id')), 'boot_identity_unavailable')
    return dict(unit=unit['name'], pid=int(pid_text), birth_ticks=process['birth_ticks'], boot_id=process['boot_id'], invocation_id=invocation, argv=unit['argv'], exe=unit['interpreter'], cwd=unit['working_directory'], environment=required)

def _native(value, profile, *, now=None):
    _need(type(value) is dict and value.get('schema') == DIAGNOSTIC_SCHEMA, 'native_diagnostic_schema')
    _need(value.get('fresh_read') is True and value.get('completion_authority') is False and value.get('database_and_wal_metadata_unchanged') is True, 'native_diagnostic_not_fresh_readonly')
    epoch = time.time() if now is None else now
    _need(_number(value.get('checked_at_epoch')) and 0 <= epoch - value['checked_at_epoch'] <= profile['max_age_seconds'], 'native_diagnostic_stale_or_future')
    _need(_integer(value.get('store_generation'), 1), 'native_generation_type')
    identity = {key: value.get(key) for key in IDENTITY_KEYS}
    _need(identity == profile['catalogue'], 'native_catalogue_identity_changed')
    _need(_integer(value.get('event_cursor')), 'native_event_cursor')
    tasks = value.get('tasks')
    _need(type(tasks) is list and 0 < len(tasks) <= 1000 and all(type(t) is dict and _text(t.get('task_cid')) and _integer(t.get('revision'), 1) and _text(t.get('status')) for t in tasks), 'native_task_population')
    by_cid = {t['task_cid']: t for t in tasks}
    _need(len(by_cid) == len(tasks) and set(profile['task_scope']) <= set(by_cid), 'native_task_scope_changed')
    return dict(**identity, event_cursor=value['event_cursor'], task_scope=list(profile['task_scope']))

def _linked_native(ref, profile, *, now=None):
    value = _document(ref)
    if value.get('schema') == 'agent-supervisor/native-catalogue-probe@1':
        value = value.get('details', {}).get('native_catalogue')
    return value, _native(value, profile, now=now)

def _within(ref, root):
    path = _path(ref['path'])
    _need(Path(root) in path.parents, 'cycle_artifact_outside_root')
    return path

def _origins(value, profile):
    _need(type(value) is dict and value.get('schema') == 'legal-ir-pre-execution-project-import-origins/v1' and value.get('project_origins_checked_before_module_execution') is True and value.get('denials') == [], 'original_import_origins_missing')
    rows = value.get('project_imports')
    _need(type(rows) is list and 1 <= len(rows) <= 4096, 'original_import_origins_population')
    sources = {p['path']: p for p in profile['source_descriptors']}
    seen = set()
    for row in rows:
        _need(type(row) is dict and _text(row.get('module')) and row['module'] not in seen, 'original_import_origin_duplicate')
        seen.add(row['module'])
        _need(sources.get(row.get('path')) == {k: row.get(k) for k in ('path', 'bytes', 'sha256')} and row.get('execution_profile') == 'compiled_exact_authenticated_source_bytes', 'original_import_origin_unbound')
        _bound_project_source(sources[row['path']], profile)
    return rows

def _embedded_idle_result(result, request, plan, origins, profile, native):
    """Join genuine embedded read-only observation to original source/custody.

    This is a separate backend contract. It never relabels an embedded source as
    TypedDatabaseTaskSource, and it cannot grant task or provider authority.
    Physical descriptors are checked as metadata; the mutable DB is never opened.
    """
    base = {'schema', 'ready_task_count', 'selection_idle_reason', 'unchanged',
            'write_count', 'provider_dispatched', 'attempt_consumed', 'completion_authority'}
    _keys(result, base | {'backend', 'database_path', 'task_scope', 'observer_source',
                         'observer_callable', 'catalogue_binding'}, 'embedded_idle_result_shape')
    _need(result['backend'] == 'embedded' and result['observer_callable'] ==
          'observe_embedded_database_without_dispatch', 'embedded_idle_backend_or_callable')
    _need(result['task_scope'] == profile['task_scope'] and
          result['selection_idle_reason'] == 'no_scoped_ready_tasks', 'embedded_idle_scope_or_selection')
    _keys(plan, {'schema', 'operation', 'accelerate_root', 'task_scope', 'native_metadata_cli'},
          'embedded_idle_plan_shape')
    _need(plan['schema'] == 'native-embedded-scoped-observation-plan/v1' and
          plan['operation'] == 'native_embedded_scoped_observation_v1', 'embedded_idle_plan_contract')
    command = plan['native_metadata_cli']
    _keys(command, {'schema', 'runtime_root', 'cwd', 'attempt_directory', 'temporary_directory',
                    'project_import_plan', 'pre_body_guard', 'result_path', 'source_sha256',
                    'observation_binding', 'import_head_capsule_Git'}, 'embedded_idle_cli_shape')
    _need(command['schema'] == 'native-embedded-scoped-observation-cli/v2' and
          _text(command['runtime_root']) and command['runtime_root'] == plan['accelerate_root'],
          'embedded_idle_cli_contract')
    git = command['import_head_capsule_Git']
    _keys(git, {'blobs', 'ls_tree_tail', 'payload_bytes', 'required_relative_files', 'route_source',
                'scope', 'selected_file_count', 'source_head', 'source_root', 'unique_blob_count'},
          'embedded_idle_import_git_shape')
    route = git['route_source']
    _keys(route, {'path', 'bytes', 'sha256'}, 'embedded_idle_import_route_descriptor')
    _need(git['source_root'] == command['runtime_root'] and type(git['source_head']) is str and
          re.fullmatch('[0-9a-f]{40}', git['source_head']) and route in profile['source_descriptors'] and
          route['path'] == str(Path(git['source_root']) / 'ipfs_accelerate_py/agent_implementation_route.py') and
          request.get('source_sha256', {}).get(route['path']) == route['sha256'] and
          type(command['source_sha256']) is dict and command['source_sha256'].get(route['path']) == route['sha256'],
          'embedded_idle_import_source_binding')
    _need(any(row.get('module') == 'ipfs_accelerate_py.agent_implementation_route' and
              {key: row.get(key) for key in ('path', 'bytes', 'sha256')} == route for row in _origins(origins, profile)),
          'embedded_idle_import_route_origin_missing')

    binding = command['observation_binding']
    _keys(binding, {'schema', 'database_preimage', 'task_scope', 'plan_root_cid',
                    'repository_tree_id', 'observer_source', 'source_schema'}, 'embedded_idle_plan_binding_shape')
    _need(binding['schema'] == 'native-embedded-daemon-observation-binding@1', 'embedded_idle_plan_binding')
    source = result['observer_source']
    _keys(source, {'path', 'bytes', 'sha256'}, 'embedded_idle_observer_descriptor')
    _need(_integer(source['bytes'], 1, MAX_JSON) and source in profile['source_descriptors'] and
          binding['observer_source'] == source and
          request.get('source_sha256', {}).get(source['path']) == source['sha256'] and
          type(command['source_sha256']) is dict and
          command['source_sha256'].get(source['path']) == source['sha256'], 'embedded_idle_observer_source')
    rows = _origins(origins, profile)
    _need(any(row.get('module') == 'ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap'
              and {key: row.get(key) for key in ('path', 'bytes', 'sha256')} == source for row in rows),
          'embedded_idle_observer_origin_missing')
    db = request.get('database_preimage')
    _keys(db, {'path', 'bytes', 'sha256'}, 'embedded_idle_database_descriptor')
    _need(_integer(db['bytes'], 1, 128 * 1024**2) and type(db['sha256']) is str and
          re.fullmatch('[0-9a-f]{64}', db['sha256']), 'embedded_idle_database_descriptor_bounds')
    _path(db['path'])
    _need(binding['database_preimage'] == db and result['database_path'] == db['path'],
          'embedded_idle_database_join')
    external = request.get('external_files')
    _need(type(external) is list and 1 <= len(external) <= 64, 'embedded_idle_external_declarations')
    for row in external:
        _keys(row, {'path', 'max_bytes'}, 'embedded_idle_external_declaration_shape')
        _path(row['path'])
        _need(_integer(row['max_bytes'], 1, 128 * 1024**2), 'embedded_idle_external_declaration_bounds')
    declared_db = [row for row in external if row['path'] == db['path']]
    _need(len(declared_db) == 1 and declared_db[0]['max_bytes'] >= db['bytes'],
          'embedded_idle_database_not_declared')
    catalogue = result['catalogue_binding']
    _keys(catalogue, {'database_preimage', 'plan_root_cid', 'repository_tree_id',
                     'source_schema', 'event_cursor', 'task_statuses'}, 'embedded_idle_catalogue_shape')
    _need(catalogue['database_preimage'] == db and binding['task_scope'] == profile['task_scope']
          and catalogue['plan_root_cid'] == binding['plan_root_cid'] == profile['catalogue']['plan_root_cid']
          and catalogue['repository_tree_id'] == binding['repository_tree_id'] == profile['catalogue']['repository_tree_id']
          and catalogue['source_schema'] == binding['source_schema'] ==
              'ipfs_accelerate_py/agent-supervisor/database-task-source@1', 'embedded_idle_catalogue_binding')
    _need(_integer(catalogue['event_cursor']) and catalogue['event_cursor'] == native['event_cursor'],
          'embedded_idle_catalogue_cursor_changed')
    expected = {row['task_cid']: {key: row[key] for key in ('task_cid', 'status', 'revision')}
                for row in native['tasks'] if row['task_cid'] in profile['task_scope']}
    tasks = catalogue['task_statuses']
    _need(type(tasks) is list and len(tasks) == len(profile['task_scope']) and
          all(type(row) is dict and set(row) == {'task_cid', 'status', 'revision'} and
              _integer(row.get('revision'), 1) and row.get('task_cid') in expected and
              row == expected[row['task_cid']] for row in tasks)
          and len({row['task_cid'] for row in tasks}) == len(tasks), 'embedded_idle_catalogue_tasks_changed')

def _cycle(cycle_ref, profile, identity, native, *, now, monotonic_ns):
    _within(cycle_ref, profile['proof_root'])
    proof = _document(cycle_ref)
    _keys(proof, CYCLE_KEYS, 'cycle_link_shape')
    _need(proof['schema'] == CYCLE_SCHEMA and proof['kind'] in ('native_execution', 'genuine_scoped_idle'), 'cycle_kind_or_schema')
    _need(_integer(proof['sequence'], 1), 'cycle_sequence')
    _keys(proof['process'], {'pid', 'birth_ticks', 'boot_id', 'invocation_id'}, 'cycle_process_shape')
    _need(proof['process'] == {k: identity[k] for k in proof['process']}, 'cycle_process_not_current_main')
    _need(all(_number(proof[k]) for k in ('started_at', 'finished_at')) and proof['started_at'] <= proof['finished_at'] <= now and now - proof['finished_at'] <= profile['max_age_seconds'], 'cycle_wall_clock')
    _need(all(_integer(proof[k], 1) for k in ('started_monotonic_ns', 'finished_monotonic_ns')) and proof['started_monotonic_ns'] <= proof['finished_monotonic_ns'] <= monotonic_ns and proof['finished_monotonic_ns'] - proof['started_monotonic_ns'] <= profile['max_cycle_seconds'] * 10**9 and monotonic_ns - proof['finished_monotonic_ns'] <= profile['max_age_seconds'] * 10**9, 'cycle_monotonic_clock')
    refs = ('request_ref', 'execution_request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref')
    for key in refs:
        _within(proof[key], profile['proof_root'])
    request, execution, terminal, result, plan, origins = (_document(proof[k]) for k in refs[:-1])
    _need(request.get('schema') == 'legal-ir-original-native-control-request/v1' and request.get('purpose') == 'native_control_metadata', 'original_request_schema')
    _need(request.get('source_sha256', {}).get(profile['owner_source']['path']) == profile['owner_source']['sha256'], 'original_owner_source_not_selected')
    _need(execution.get('schema') == 'native-metadata-resource-owner-execution-request/v1' and execution.get('request') == proof['request_ref'] and execution.get('owner_source') == profile['owner_source'], 'original_owner_request_join')
    _need(terminal.get('schema') == 'legal-ir-original-native-control-terminal/v1' and terminal.get('request_sha256') == proof['request_ref']['sha256'] and terminal.get('execution_request') == proof['execution_request_ref'], 'original_terminal_request_join')
    _need(terminal.get('completed') is True and terminal.get('source_READ_closed') is True and terminal.get('original_native_compute_closed') is True and terminal.get('child_reaped') is True and type(terminal.get('child_returncode')) is int and terminal['child_returncode'] == 0 and terminal.get('storage_status') == 'released' and terminal.get('cleanup_errors') == [] and terminal.get('received_signals') == [] and terminal.get('partial_or_failed_control_must_be_reconciled') is False, 'original_cycle_not_closed')
    reservation = terminal.get('reservation', {})
    _need(type(reservation) is dict and type(reservation.get('record')) is dict and type(reservation.get('resource_lease')) is dict, 'original_reservation_shape')
    _need(reservation.get('status') == 'released' and reservation.get('record', {}).get('status') == 'released' and reservation.get('resource_lease', {}).get('released') is True, 'original_storage_native_release_missing')
    observation = proof['owner_observation']
    _keys(observation, {'pid', 'birth_ticks', 'boot_id', 'argv', 'ancestry', 'captured_at', 'captured_monotonic_ns'}, 'original_owner_observation_shape')
    _need(_integer(observation['pid'], 2) and _integer(observation['birth_ticks'], 1) and observation['boot_id'] == identity['boot_id'] and terminal.get('owner_pid') == observation['pid'] and terminal.get('owner_birth') == str(observation['birth_ticks']), 'original_owner_process_join')
    _need(_number(observation['captured_at']) and proof['started_at'] <= observation['captured_at'] <= proof['finished_at'] and _integer(observation['captured_monotonic_ns'], 1) and proof['started_monotonic_ns'] <= observation['captured_monotonic_ns'] <= proof['finished_monotonic_ns'], 'original_owner_capture_outside_cycle')
    ancestry = observation['ancestry']
    _need(type(ancestry) is list and 2 <= len(ancestry) <= 32 and ancestry[0]['pid'] == observation['pid'] and ancestry[0]['birth_ticks'] == observation['birth_ticks'] and ancestry[-1]['pid'] == identity['pid'] and ancestry[-1]['birth_ticks'] == identity['birth_ticks'], 'original_owner_ancestry_changed')
    for i, row in enumerate(ancestry):
        _keys(row, {'pid', 'birth_ticks', 'parent_pid'}, 'original_owner_ancestor_shape')
        _need(_integer(row['pid'], 1) and _integer(row['birth_ticks'], 1) and _integer(row['parent_pid']), 'original_owner_ancestor_values')
        if i + 1 < len(ancestry):
            _need(row['parent_pid'] == ancestry[i + 1]['pid'], 'original_owner_ancestry_disconnected')
    expected_owner_argv = [profile['unit']['argv'][0], '-B', profile['owner_source']['path'], '--request', proof['request_ref']['path'], '--expected-request-sha256', proof['request_ref']['sha256']]
    _need(observation['argv'] == expected_owner_argv, 'original_owner_argv_changed')
    timing = reservation.get('record', {})
    _need(execution.get('disk_reservation_id') == reservation.get('reservation_id') == timing.get('reservation_id') and _text(execution.get('disk_reservation_id')) and timing.get('owner_pid') == observation['pid'] and reservation.get('resource_lease', {}).get('owner_pid') == observation['pid'] and timing.get('native_lease_id') == reservation.get('resource_lease', {}).get('lease_id'), 'original_native_reservation_join')
    _need(all(_number(timing.get(k)) for k in ('created_at', 'released_at')) and proof['started_at'] <= timing['created_at'] <= timing['released_at'] <= proof['finished_at'], 'original_cycle_timing_not_current')
    _need(_number(terminal.get('elapsed_seconds')) and 0 <= terminal['elapsed_seconds'] <= profile['max_cycle_seconds'] and terminal['elapsed_seconds'] <= proof['finished_at'] - proof['started_at'] + 1, 'original_cycle_elapsed_bound')
    custody = terminal.get('sampled_process_custody_final', {})
    stopped = terminal.get('process_spawn_receipts', {})
    _need(type(custody) is dict and type(stopped) is dict, 'original_process_custody_shape')
    _need(custody.get('scope') == 'sampled_birth_bound_descendants' and custody.get('live_identities') == [] and type(custody.get('live_processes')) is int and custody['live_processes'] == 0 and type(custody.get('rss_bytes')) is int and custody['rss_bytes'] == 0 and custody.get('tracking_error') == '' and stopped.get('schema') == 'native-metadata-stopped-process-receipts/v1' and stopped.get('all_recorded_original_births_stopped') is True, 'original_process_closure_missing')
    _need(plan.get('schema') == profile['cycle_contract']['plan_schema'] and plan.get('operation') == profile['cycle_contract']['plan_operation'], 'original_cycle_plan_changed')
    _need(type(request.get('argv')) is list and request['argv'] and request['argv'][0] in (profile['unit']['argv'][0], profile['unit']['interpreter']), 'original_driver_interpreter_changed')
    expected = [request['argv'][0], '-B', profile['driver_source']['path'], '--plan', proof['plan_ref']['path'], '--plan-sha256', proof['plan_ref']['sha256']]
    _need(request.get('argv') == expected and request.get('source_sha256', {}).get(profile['driver_source']['path']) == profile['driver_source']['sha256'], 'original_driver_source_or_argv_changed')
    _need(plan.get('native_metadata_cli', {}).get('result_path') == proof['result_ref']['path'], 'original_result_path_not_bound')
    _origins(origins, profile)
    cycle_native_value, cycle_native = _linked_native(proof['native_diagnostic_ref'], profile, now=now)
    _need(cycle_native['event_cursor'] <= native['event_cursor'], 'native_event_cursor_regressed')
    _need(_number(cycle_native_value.get('checked_at_epoch')) and proof['started_at'] <= cycle_native_value['checked_at_epoch'] <= proof['finished_at'], 'native_cycle_report_not_current')
    if proof['kind'] == 'native_execution':
        _need(result.get('schema') == 'native-metadata-cold-cli-result/v1' and result.get('completed') is True and type(result.get('original_cli_calls')) is int and result['original_cli_calls'] == 1 and type(result.get('original_main_returncode')) is int and result['original_main_returncode'] == 0 and result.get('error') is None and result.get('provider_authority') is False and result.get('completion_authority') is False, 'original_cli_result_not_positive')
        command = plan.get('native_metadata_cli', {})
        cid = plan.get('task_cid')
        _need(cid in profile['task_scope'] and command.get('task_cid') == cid and _integer(plan.get('expected_ready_revision'), 1) and command.get('expected_revision') == plan['expected_ready_revision'], 'original_native_execution_scope')
        argv = command.get('cli_argv', [])
        _need(type(argv) is list and '--once' in argv and '--implement' in argv, 'original_native_cli_not_once')
        for key, value in (('--native-metadata-execution-task-cid', cid), ('--execution-slice-task-cid', cid), ('--native-metadata-execution-task-revision', str(plan['expected_ready_revision']))):
            _need(argv.count(key) == 1 and argv[argv.index(key) + 1] == value, 'original_native_cli_task_binding')
    else:
        _need(plan.get('task_scope') == profile['task_scope'], 'original_idle_task_scope_changed')
        if result.get('schema') == 'native-embedded-daemon-observation@1':
            _embedded_idle_result(result, request, plan, origins, profile, cycle_native_value)
        else:
            _keys(result, {'schema', 'ready_task_count', 'selection_idle_reason', 'unchanged', 'write_count', 'provider_dispatched', 'attempt_consumed', 'completion_authority'}, 'original_idle_result_shape')
            _need(result['schema'] == 'native-typed-daemon-observation@1', 'original_idle_result_schema')
        _need(type(result['ready_task_count']) is int and result['ready_task_count'] == 0 and result['unchanged'] is True and type(result['write_count']) is int and result['write_count'] == 0 and result['provider_dispatched'] is False and result['attempt_consumed'] is False and result['completion_authority'] is False and _text(result['selection_idle_reason']), 'original_idle_result_not_quiet')
        tasks = {t['task_cid']: t for t in cycle_native_value['tasks']}
        _need(all(tasks[c]['status'] in ('blocked', 'completed', 'complete', 'done', 'skipped') for c in profile['task_scope']), 'scoped_idle_has_runnable_work')
    return proof

def _receipt(ref, profile_ref, profile, native, identity, *, now, monotonic_ns):
    receipt = _document(ref)
    _keys(receipt, RECEIPT_KEYS, 'receipt_shape')
    _need(receipt['schema'] == RECEIPT_SCHEMA and receipt['profile_ref'] == profile_ref and receipt['identity'] == identity, 'receipt_process_or_profile_changed')
    _need(_integer(receipt['sequence'], 1) and _number(receipt['observed_at']) and 0 <= now - receipt['observed_at'] <= profile['max_age_seconds'], 'receipt_stale_or_future')
    _need(_integer(receipt['observed_monotonic_ns'], 1) and 0 <= monotonic_ns - receipt['observed_monotonic_ns'] <= profile['max_age_seconds'] * 10**9, 'receipt_monotonic_stale_or_future')
    _keys(receipt['catalogue'], IDENTITY_KEYS | {'event_cursor', 'task_scope'}, 'receipt_catalogue_shape')
    expected = {k: native[k] for k in IDENTITY_KEYS}
    _need({k: receipt['catalogue'].get(k) for k in IDENTITY_KEYS} == expected and receipt['catalogue'].get('task_scope') == profile['task_scope'] and _integer(receipt['catalogue'].get('event_cursor')) and receipt['catalogue']['event_cursor'] <= native['event_cursor'], 'receipt_catalogue_or_scope_changed')
    proof = _cycle(receipt['cycle_proof_ref'], profile, identity, native, now=now, monotonic_ns=monotonic_ns)
    cycle_native_value, cycle_native = _linked_native(proof['native_diagnostic_ref'], profile, now=now)
    _need(cycle_native['event_cursor'] <= receipt['catalogue']['event_cursor'], 'receipt_cycle_cursor_regressed')
    _need(receipt['previous_receipt_sha256'] is None or (type(receipt['previous_receipt_sha256']) is str and re.fullmatch('[0-9a-f]{64}', receipt['previous_receipt_sha256'])), 'receipt_previous_digest')
    _need(proof['sequence'] == receipt['sequence'] and proof['finished_monotonic_ns'] <= receipt['observed_monotonic_ns'] and proof['finished_at'] <= receipt['observed_at'], 'receipt_does_not_follow_cycle')
    if receipt['sequence'] == 1 and receipt['previous_receipt_sha256'] is not None:
        archive = _epoch_archive_path(profile['receipt_path'], receipt['previous_receipt_sha256'])
        archive_ref = descriptor(archive)
        _need(archive_ref['sha256'] == receipt['previous_receipt_sha256'], 'previous_epoch_archive_digest_changed')
        archived = _document(archive_ref)
        _keys(archived, RECEIPT_KEYS, 'previous_epoch_archive_shape')
        _stored_identity(archived['identity'], profile_ref, profile)
        _need(archived['profile_ref'] == profile_ref and archived['identity']['invocation_id'] != identity['invocation_id'], 'previous_epoch_archive_binding_changed')
    sources = {p['path']: p for p in profile['source_descriptors']}
    _need(type(receipt['origins']) is dict and 1 <= len(receipt['origins']) <= 4096 and all(type(pin) is dict and sources.get(pin.get('path')) == pin and pin['path'] != profile['unit']['interpreter'] for pin in receipt['origins'].values()), 'producer_import_origins_unbound')
    _need(receipt['origins'].get('__main__') == profile['main_entrypoint'] and profile['health_module'] in receipt['origins'].values(), 'producer_entrypoint_or_health_origin_missing')
    return receipt, proof

def verify_bound_executor(profile_ref, native_diagnostic, *, now=None, manager_reader=None, process_reader=None):
    """Independently verify current process, all source pins and original closure."""
    result = dict(verified=False, reason='executor_unverified', scope='none', completion_authority=False, dispatch_authority=False)
    try:
        epoch = time.time() if now is None else now
        _need(_number(epoch), 'verification_clock')
        mono = time.monotonic_ns()
        profile = _profile(profile_ref)
        native = _native(native_diagnostic, profile, now=epoch)
        manager_reader = manager_reader or _manager
        process_reader = process_reader or _process
        opening = _identity(profile_ref, profile, manager_reader, process_reader)
        receipt_ref = descriptor(profile['receipt_path'])
        epoch = time.time() if now is None else now
        mono = time.monotonic_ns()
        receipt, proof = _receipt(receipt_ref, profile_ref, profile, native, opening, now=epoch, monotonic_ns=mono)
        _need(_profile(profile_ref) == profile and descriptor(profile['receipt_path']) == receipt_ref, 'profile_or_receipt_changed_during_verification')
        _need(_identity(profile_ref, profile, manager_reader, process_reader) == opening, 'executor_identity_changed_during_verification')
        closing_epoch = time.time() if now is None else now
        _need(0 <= closing_epoch - receipt['observed_at'] <= profile['max_age_seconds'] and 0 <= time.monotonic_ns() - receipt['observed_monotonic_ns'] <= profile['max_age_seconds'] * 10**9, 'receipt_expired_during_verification')
        result.update(verified=True, reason='verified_original_native_cycle', scope=proof['kind'], profile_ref=profile_ref, receipt_ref=receipt_ref, cycle_proof_ref=receipt['cycle_proof_ref'], sequence=receipt['sequence'], actual_identity=opening, catalogue=receipt['catalogue'], source_attestation='source_pinned_main_process_and_recorded_project_origins', observed_at=receipt['observed_at'])
    except (ExecutorHealthError, OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, subprocess.SubprocessError) as exc:
        result['reason'] = str(exc) if isinstance(exc, ExecutorHealthError) else 'executor_evidence_unavailable'
    return result

def _stat_identity(pid):
    raw = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, birth_ticks=int(raw[19]), parent_pid=int(raw[1]))

def _same_birth(pid, birth):
    try:
        return _stat_identity(pid)['birth_ticks'] == birth
    except (OSError, ValueError, IndexError):
        return False

def track_original_cycle(profile_ref, owner_process, request_ref, *, sequence):
    """Capture a real live original owner while it descends from this MainPID.

    Keep the returned opaque handle in memory; it cannot be restored from JSON.
    The existing original owner performs all execution and resource admission.
    """
    _need(threading.current_thread() is threading.main_thread(), 'producer_not_main_thread')
    _need(type(owner_process) is subprocess.Popen and owner_process.poll() is None, 'original_owner_not_live_popen')
    profile = _profile(profile_ref)
    identity = _identity(profile_ref, profile, _manager, _process)
    _need(identity['pid'] == os.getpid() and _integer(sequence, 1), 'producer_not_manager_main_pid')
    _within(request_ref, profile['proof_root']); _bound(request_ref)
    _need(request_ref['sha256'] not in _USED_REQUESTS and not any(c['request_ref']['sha256'] == request_ref['sha256'] for c in _CYCLES.values()), 'original_request_replayed')
    owner = _process(owner_process.pid)
    expected = [profile['unit']['argv'][0], '-B', profile['owner_source']['path'], '--request', request_ref['path'], '--expected-request-sha256', request_ref['sha256']]
    _need(owner['argv'] == expected and owner['exe'] == profile['unit']['interpreter'] and owner['boot_id'] == identity['boot_id'], 'original_owner_source_argv_changed')
    ancestry = []
    pid = owner_process.pid
    for _ in range(32):
        row = _stat_identity(pid); ancestry.append(row)
        if pid == os.getpid():
            break
        _need(row['parent_pid'] > 1 and row['parent_pid'] not in {x['pid'] for x in ancestry}, 'original_owner_not_main_descendant')
        pid = row['parent_pid']
    _need(ancestry[-1]['pid'] == os.getpid() and ancestry[0]['birth_ticks'] == owner['birth_ticks'] and ancestry[-1]['birth_ticks'] == identity['birth_ticks'], 'original_owner_ancestry_changed')
    now, mono = time.time(), time.monotonic_ns()
    ticks = os.sysconf('SC_CLK_TCK')
    start_mono = owner['birth_ticks'] * 10**9 // ticks
    _need(0 <= mono - start_mono <= profile['max_cycle_seconds'] * 10**9 and owner_process.poll() is None and _same_birth(owner_process.pid, owner['birth_ticks']), 'original_owner_capture_too_late')
    observation = dict(pid=owner_process.pid, birth_ticks=owner['birth_ticks'], boot_id=owner['boot_id'], argv=owner['argv'], ancestry=ancestry, captured_at=now, captured_monotonic_ns=mono)
    handle = _CycleHandle()
    _CYCLES[handle] = dict(profile_ref=profile_ref, request_ref=request_ref, sequence=sequence, identity=identity, process=owner_process, owner_observation=observation, started_at=now - (mono - start_mono) / 10**9, started_monotonic_ns=start_mono)
    return handle

def cycle_start_fields(cycle_handle):
    """Return the exact captured link fields, without converting a handle to authority."""
    _need(type(cycle_handle) is _CycleHandle and cycle_handle in _CYCLES, 'original_cycle_capability_missing')
    value = _CYCLES[cycle_handle]
    return json.loads(json.dumps(dict(sequence=value['sequence'], process={k: value['identity'][k] for k in ('pid', 'birth_ticks', 'boot_id', 'invocation_id')}, owner_observation=value['owner_observation'], started_at=value['started_at'], started_monotonic_ns=value['started_monotonic_ns'])))

def _stored_identity(value, profile_ref, profile):
    _keys(value, {'unit', 'pid', 'birth_ticks', 'boot_id', 'invocation_id', 'argv', 'exe', 'cwd', 'environment'}, 'previous_identity_shape')
    unit = profile['unit']
    _need(value['unit'] == unit['name'] and _integer(value['pid'], 2) and _integer(value['birth_ticks'], 1) and _text(value['boot_id']) and type(value['invocation_id']) is str and re.fullmatch('[0-9a-f]{32}', value['invocation_id']), 'previous_identity_values')
    expected_env = {**unit['environment'], 'INVOCATION_ID': value['invocation_id'], ENV_PROFILE_SHA: profile_ref['sha256']}
    _need(value['argv'] == unit['argv'] and value['exe'] == unit['interpreter'] and value['cwd'] == unit['working_directory'] and value['environment'] == expected_env, 'previous_identity_profile_changed')

def _definitely_dead_identity(previous, current):
    # A process from a different kernel boot cannot still be alive. PID/birth
    # coincidences in the new boot must not resurrect its old identity.
    if previous['boot_id'] != current['boot_id']:
        return True
    try:
        observed = _stat_identity(previous['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return True
    except (OSError, ValueError, IndexError, OverflowError) as exc:
        raise ExecutorHealthError('previous_process_custody_unknown') from exc
    return observed['birth_ticks'] != previous['birth_ticks']

def _epoch_archive_path(receipt_path, digest):
    _need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'previous_receipt_digest')
    path = Path(receipt_path)
    return path.with_name(path.name + '.previous-' + digest + '.json')

def _previous_receipt_transition(profile_ref, profile, native, identity, proof, *, now, monotonic_ns):
    """Validate the prior epoch at its own clocks; this function never writes."""
    _need(_integer(proof.get('sequence'), 1), 'cycle_sequence')
    path = _path(profile['receipt_path'])
    transition = dict(previous_receipt_sha256=None, previous_ref=None, archive_path=None, archive_raw=None)
    if not os.path.lexists(path):
        _need(proof['sequence'] == 1, 'initial_cycle_sequence')
        return transition
    old_ref = descriptor(path)
    raw = _bound(old_ref)
    old = _decode(raw)
    _keys(old, RECEIPT_KEYS, 'previous_receipt_shape')
    _stored_identity(old['identity'], profile_ref, profile)
    _need(_number(old['observed_at']) and old['observed_at'] <= now and _integer(old['observed_monotonic_ns'], 1), 'previous_receipt_clock')
    previous, old_proof = _receipt(old_ref, profile_ref, profile, native, old['identity'], now=old['observed_at'], monotonic_ns=old['observed_monotonic_ns'])
    _need(proof['request_ref']['sha256'] != old_proof['request_ref']['sha256'], 'original_request_replayed')
    transition.update(previous_receipt_sha256=old_ref['sha256'], previous_ref=old_ref)
    if old['identity'] == identity:
        _need(proof['sequence'] == previous['sequence'] + 1 and proof['started_monotonic_ns'] >= previous['observed_monotonic_ns'], 'cycle_replay_or_sequence_gap')
    else:
        _need(old['identity']['invocation_id'] != identity['invocation_id'], 'restart_invocation_not_changed')
        _need(proof['sequence'] == 1 and _definitely_dead_identity(old['identity'], identity), 'previous_executor_still_alive_or_epoch_sequence')
        archive = _epoch_archive_path(path, old_ref['sha256'])
        _within(dict(path=str(archive)), profile['proof_root'])
        transition.update(archive_path=str(archive), archive_raw=raw)
    return transition

def _archive_previous_receipt(path, raw):
    """One exact immutable archive per replaced epoch; no deletion or overwrite."""
    path = _path(str(path))
    _need(len(raw) <= MAX_JSON and path.name.endswith('.previous-' + hashlib.sha256(raw).hexdigest() + '.json'), 'previous_archive_name_or_bound')
    prefix = path.name.rsplit('.previous-', 1)[0] + '.previous-'
    count = 0
    entries = 0
    with os.scandir(path.parent) as directory:
        for entry in directory:
            entries += 1
            _need(entries <= MAX_ARCHIVE_DIRECTORY_ENTRIES, 'archive_directory_population_bound')
            if entry.name.startswith(prefix) and entry.name.endswith('.json'):
                _need(re.fullmatch(re.escape(prefix) + r'[0-9a-f]{64}\.json', entry.name), 'foreign_epoch_archive_name')
                count += 1
                _need(count <= MAX_EPOCH_ARCHIVES, 'epoch_archive_population_bound')
    exists = os.path.lexists(path)
    _need(exists or count < MAX_EPOCH_ARCHIVES, 'epoch_archive_population_bound')
    if exists:
        _need(_raw(path) == raw, 'previous_archive_collision')
    else:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o400)
        except FileExistsError:
            _need(_raw(path) == raw, 'previous_archive_collision')
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    _need(_raw(path) == raw, 'previous_archive_postimage_changed')
    return descriptor(path)

def publish_after_native_cycle(profile_ref, cycle_proof_ref, native_diagnostic, *, sequence, cycle_handle):
    """Call synchronously from the unit MainPID after genuine original closure.

    NativeOwnerHeartbeat remains advisory background owner-read liveness. It
    never calls this function. New health sequences require new cycle evidence.
    """
    _need(threading.current_thread() is threading.main_thread(), 'producer_not_main_thread')
    _need(type(cycle_handle) is _CycleHandle and cycle_handle in _CYCLES, 'original_cycle_capability_missing')
    capability = _CYCLES[cycle_handle]
    profile = _profile(profile_ref)
    _need(Path(__file__).resolve() == Path(profile['health_module']['path']), 'health_module_origin_changed')
    identity = _identity(profile_ref, profile, _manager, _process)
    _need(identity['pid'] == os.getpid(), 'producer_not_manager_main_pid')
    now, mono = time.time(), time.monotonic_ns()
    native = _native(native_diagnostic, profile)
    proof = _cycle(cycle_proof_ref, profile, identity, native, now=now, monotonic_ns=mono)
    _need(_integer(sequence, 1) and sequence == proof['sequence'] == capability['sequence'] and profile_ref == capability['profile_ref'] and identity == capability['identity'], 'producer_cycle_sequence_changed')
    _need(proof['request_ref'] == capability['request_ref'] and proof['owner_observation'] == capability['owner_observation'] and proof['started_at'] == capability['started_at'] and proof['started_monotonic_ns'] == capability['started_monotonic_ns'], 'cycle_capability_join_changed')
    _need(type(capability['process'].poll()) is int and capability['process'].returncode == 0, 'original_owner_wait_not_positive')
    _need(not _same_birth(capability['owner_observation']['pid'], capability['owner_observation']['birth_ticks']), 'original_owner_not_reaped')
    _need(proof['request_ref']['sha256'] not in _USED_REQUESTS, 'original_request_replayed')
    path = _path(profile['receipt_path'])
    transition = _previous_receipt_transition(profile_ref, profile, native, identity, proof, now=now, monotonic_ns=mono)
    previous_sha = transition['previous_receipt_sha256']
    sources = {p['path']: p for p in profile['source_descriptors']}
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != '__main__' and not name.startswith(('ipfs_accelerate_py', 'ipfs_datasets_py')) and module is not sys.modules.get(__name__):
            continue
        origin = getattr(module, '__file__', None)
        _need(origin is not None and str(Path(origin).resolve()) in sources, 'live_producer_import_origin_unbound')
        pin = sources[str(Path(origin).resolve())]
        _bound_project_source(pin, profile)
        origins[name] = pin
    _need(origins.get('__main__') == profile['main_entrypoint'] and profile['health_module'] in origins.values(), 'producer_main_origin_unbound')
    receipt = dict(schema=RECEIPT_SCHEMA, profile_ref=profile_ref, sequence=sequence, identity=identity, observed_at=now, observed_monotonic_ns=mono, cycle_proof_ref=cycle_proof_ref, catalogue=native, origins=origins, previous_receipt_sha256=previous_sha)
    parent = path.parent
    info = parent.lstat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700, 'receipt_parent_not_private')
    _need(_identity(profile_ref, profile, _manager, _process) == identity and _profile(profile_ref) == profile, 'producer_binding_changed')
    raw = _encoded(receipt)
    _need(len(raw) <= MAX_JSON, 'receipt_output_bound')
    _need((descriptor(path) if os.path.lexists(path) else None) == transition['previous_ref'], 'previous_receipt_changed_before_publish')
    # Consume before archive creation or replacement: uncertain effects never retry.
    _USED_REQUESTS.add(proof['request_ref']['sha256'])
    del _CYCLES[cycle_handle]
    if transition['archive_path'] is not None:
        _archive_previous_receipt(Path(transition['archive_path']), transition['archive_raw'])
    _need((descriptor(path) if os.path.lexists(path) else None) == transition['previous_ref'], 'previous_receipt_changed_after_archive')
    fd, temporary = tempfile.mkstemp(prefix='.native-executor-health-', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return descriptor(path)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('verify', 'publish'))
    parser.add_argument('--profile-ref', required=True, help='Exact JSON path/bytes/SHA descriptor')
    parser.add_argument('--native-report-ref', required=True, help='Exact JSON descriptor of existing native diagnostic')
    parser.add_argument('--cycle-proof-ref', help='Required only for MainPID-bound publish')
    parser.add_argument('--sequence', type=int)
    args = parser.parse_args(argv)
    try:
        profile_ref = _decode(args.profile_ref.encode())
        profile = _profile(profile_ref)
        native, _ = _linked_native(_decode(args.native_report_ref.encode()), profile)
        if args.operation == 'publish':
            raise ExecutorHealthError('producer_requires_in_process_original_cycle_capability')
        _need(args.cycle_proof_ref is None and args.sequence is None, 'verifier_producer_arguments_forbidden')
        result = verify_bound_executor(profile_ref, native)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if args.operation == 'publish' or result.get('verified') is True else 2
    except (ExecutorHealthError, OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
        print(json.dumps(dict(verified=False, reason=str(exc) if isinstance(exc, ExecutorHealthError) else 'executor_evidence_unavailable', completion_authority=False)))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
