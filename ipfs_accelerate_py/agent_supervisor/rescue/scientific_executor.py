"""Finite, source-bound service around the existing original native owner.

Only closed, genuine idle observations may poll. Unknown or failed cycles leave
a persistent hold and exit 78; restarting this service never retries them.
The original owner retains all compute, storage and task execution authority.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

CONFIG_SCHEMA = 'agent-supervisor/scientific-executor-service@1'
TEMPLATE_SCHEMA = 'agent-supervisor/scientific-executor-cycle-template@1'
CONFIG_KEYS = frozenset({'schema', 'profile_path', 'catalogue', 'idle_template_ref',
    'execution_templates', 'poll_seconds', 'max_cycles', 'max_proof_bytes'})
CATALOGUE_KEYS = frozenset({'database_path', 'population_path', 'board_id',
    'board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256',
    'max_age_seconds', 'timeout_seconds'})
TEMPLATE_KEYS = frozenset({'schema', 'kind', 'seed_root', 'request_ref', 'plan_ref', 'links'})
HOLD_EXIT = 78
QUIET_STATUSES = frozenset({'blocked', 'completed', 'complete', 'done', 'skipped'})


class ExecutorHold(ValueError):
    """A cycle cannot be replayed or silently discarded."""


def health_owner():
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health
    return native_executor_health


def need(value, reason):
    if not value:
        raise ExecutorHold(reason)


def put(path, value):
    """Create one immutable private artifact; never replace an existing one."""
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    need(len(raw) <= 4 * 1024**2, 'artifact_bound')
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return health_owner().descriptor(path)


def load_configuration(config_ref):
    h = health_owner()
    value = h._document(config_ref)
    h._keys(value, CONFIG_KEYS, 'service_config_shape')
    need(value['schema'] == CONFIG_SCHEMA, 'service_config_schema')
    profile_ref = h.descriptor(h._path(value['profile_path']))
    profile = h._profile(profile_ref)
    need(config_ref in profile['source_descriptors'], 'service_config_not_source_pinned')
    h._keys(value['catalogue'], CATALOGUE_KEYS, 'service_catalogue_shape')
    catalogue = value['catalogue']
    h._path(catalogue['database_path']); h._path(catalogue['population_path'])
    need(catalogue['board_id'] == 'legal-ir-scientific', 'service_board_scope')
    need(all(catalogue[k] == profile['catalogue'][k] for k in
        ('board_namespace', 'plan_root_cid', 'repository_tree_id', 'population_sha256')), 'service_catalogue_binding')
    need(h._integer(catalogue['max_age_seconds'], 1, profile['max_age_seconds']) and
        h._integer(catalogue['timeout_seconds'], 1, min(30, profile['max_cycle_seconds'])), 'service_observation_bounds')
    need(h._integer(value['poll_seconds'], 1, 60) and h._integer(value['max_cycles'], 2, 10000)
        and h._integer(value['max_proof_bytes'], 1024**2, 16 * 1024**3), 'service_finite_limits')
    idle = h._document(value['idle_template_ref'])
    validate_template(idle, profile, 'genuine_scoped_idle')
    templates = value['execution_templates']
    need(type(templates) is list and len(templates) <= len(profile['task_scope']), 'service_execution_population')
    seen = set()
    for row in templates:
        h._keys(row, {'task_cid', 'expected_revision', 'template_ref'}, 'service_execution_template_shape')
        need(row['task_cid'] in profile['task_scope'] and row['task_cid'] not in seen and
            h._integer(row['expected_revision'], 1), 'service_execution_binding')
        seen.add(row['task_cid'])
        validate_template(h._document(row['template_ref']), profile, 'native_execution')
    return value, profile_ref, profile


def validate_template(template, profile, kind):
    h = health_owner()
    h._keys(template, TEMPLATE_KEYS, 'cycle_template_shape')
    need(template['schema'] == TEMPLATE_SCHEMA and template['kind'] == kind, 'cycle_template_kind')
    seed = h._path(template['seed_root'])
    need(Path(profile['proof_root']) in seed.parents, 'seed_outside_proof_root')
    for field in ('request_ref', 'plan_ref'):
        h._bound(template[field]); need(seed in Path(template[field]['path']).parents, 'template_artifact_outside_seed')
    h._keys(template['links'], {'execution_request', 'result', 'origins'}, 'cycle_template_links')
    for value in template['links'].values():
        need(type(value) is str and value and not Path(value).is_absolute() and '..' not in Path(value).parts,
            'template_relative_link')
    need(len(set(template['links'].values())) == 3, 'template_link_alias')


def observe_native(config):
    from ipfs_accelerate_py.agent_supervisor.rescue.native_catalogue_probe import observe_native_catalogue
    binding = dict(config['catalogue'])
    database, population = Path(binding.pop('database_path')), Path(binding.pop('population_path'))
    report = observe_native_catalogue(database, population, **binding)
    value = report.get('details', {}).get('native_catalogue')
    need(type(value) is dict, 'native_observation_missing')
    return value


def select_template(config, profile, native):
    """Never interpret zero globally ready tasks as authority to ignore scope."""
    health_owner()._native(native, profile)
    tasks = {row['task_cid']: row for row in native['tasks']}
    scoped = [tasks[cid] for cid in profile['task_scope']]
    if all(row['status'] in QUIET_STATUSES for row in scoped):
        return config['idle_template_ref'], 'genuine_scoped_idle'
    permitted = {(row['task_cid'], row['expected_revision']): row['template_ref']
                 for row in config['execution_templates']}
    for row in sorted(scoped, key=lambda item: item['task_cid']):
        if row['status'] == 'ready' and (row['task_cid'], row['revision']) in permitted:
            return permitted[(row['task_cid'], row['revision'])], 'native_execution'
    raise ExecutorHold('scoped_work_has_no_explicit_execution_binding')


def relocate(value, seed, root):
    """Rewrite only paths rooted in the sealed seed; preserve external pins."""
    if isinstance(value, str):
        if value == str(seed) or value.startswith(str(seed) + os.sep):
            return str(root) + value[len(str(seed)):]
        return value
    if type(value) is list:
        return [relocate(item, seed, root) for item in value]
    if type(value) is dict:
        return {relocate(key, seed, root): relocate(item, seed, root) for key, item in value.items()}
    return value


def check_proof_budget(root, maximum):
    total = count = 0
    for path in root.rglob('*'):
        info = path.lstat()
        need(not stat.S_ISLNK(info.st_mode), 'proof_symlink')
        if stat.S_ISDIR(info.st_mode):
            continue
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'proof_nonordinary_artifact')
        total += info.st_size; count += 1
        need(count <= 20000 and total <= maximum, 'proof_storage_bound')


def prepare_cycle(template_ref, kind, config, profile, root):
    h = health_owner(); template = h._document(template_ref)
    validate_template(template, profile, kind)
    seed = Path(template['seed_root'])
    plan = relocate(h._document(template['plan_ref']), seed, root)
    request = relocate(h._document(template['request_ref']), seed, root)
    need(plan.get('schema') == profile['cycle_contract']['plan_schema'] and
        plan.get('operation') == profile['cycle_contract']['plan_operation'], 'prepared_plan_contract')
    if kind == 'genuine_scoped_idle':
        need(plan.get('task_scope') == profile['task_scope'], 'prepared_idle_scope')
    else:
        need(plan.get('task_cid') in profile['task_scope'] and any(row['task_cid'] == plan.get('task_cid')
            and row['expected_revision'] == plan.get('expected_ready_revision')
            and row['template_ref'] == template_ref for row in config['execution_templates']), 'prepared_execution_scope')
    need(request.get('schema') == 'legal-ir-original-native-control-request/v1' and
        request.get('purpose') == 'native_control_metadata', 'prepared_original_request_schema')
    need(h._integer(request.get('timeout_seconds'), 1, profile['max_cycle_seconds']), 'prepared_owner_timeout')
    sources = request.get('source_sha256')
    need(type(sources) is dict and all(sources.get(profile[field]['path']) == profile[field]['sha256']
        for field in ('owner_source', 'driver_source')), 'prepared_original_source_binding')
    # Refresh only the read-only DB preimage. Nothing changes task admission.
    preimage = request.get('database_preimage')
    need(type(preimage) is dict and preimage.get('path') == config['catalogue']['database_path'], 'prepared_database_binding')
    current_db = h.database_descriptor(config['catalogue']['database_path'])
    request['database_preimage'] = current_db
    command = plan.get('native_metadata_cli', {})
    if kind == 'genuine_scoped_idle':
        binding = command.get('observation_binding', {})
        need(binding.get('database_preimage', {}).get('path') == current_db['path'], 'prepared_observation_database')
        binding['database_preimage'] = current_db
    attempt = h._path(request['attempt_directory'])
    need(root in attempt.parents and not attempt.exists(), 'prepared_attempt_directory')
    attempt.mkdir(mode=0o700, parents=True)
    plan_ref = put(root / 'operation-plan.json', plan)
    old_plan = str(root / Path(template['plan_ref']['path']).relative_to(seed))
    sources.pop(old_plan, None); sources[plan_ref['path']] = plan_ref['sha256']
    request['argv'] = [profile['unit']['argv'][0], '-B', profile['driver_source']['path'],
        '--plan', plan_ref['path'], '--plan-sha256', plan_ref['sha256']]
    request['request_id'] = 'scientific-executor:' + root.name
    request_ref = put(root / 'request.json', request)
    links = {key: root / value for key, value in template['links'].items()}
    need(command.get('result_path') == str(links['result']), 'prepared_result_link')
    return request_ref, plan_ref, links


def original_terminal(raw, request_ref):
    h = health_owner(); candidates = []
    try:
        candidates.append(h._decode(raw))
    except (ValueError, TypeError):
        pass
    for line in raw.splitlines():
        try:
            candidate = h._decode(line)
        except (ValueError, TypeError):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    matches = [row for row in candidates if type(row) is dict and
        row.get('schema') == 'legal-ir-original-native-control-terminal/v1' and
        row.get('request_sha256') == request_ref['sha256']]
    need(len(matches) == 1, 'original_terminal_missing_or_ambiguous')
    return matches[0]


def persistent_hold(proof_root, cycle_root, reason):
    path = Path(proof_root) / 'executor-held.json'
    if not os.path.lexists(path):
        put(path, {'schema': 'agent-supervisor/scientific-executor-persistent-hold@1',
            'cycle_root': str(cycle_root), 'reason': str(reason)[:2048],
            'recorded_at': time.time(), 'automatic_retry_admitted': False})


def require_main_process(profile_ref, profile):
    """Authenticate the producer before spawning any original owner."""
    h = health_owner()
    need(threading.current_thread() is threading.main_thread(), 'service_not_main_thread')
    need(Path(__file__).resolve() == Path(profile['main_entrypoint']['path']) and
        Path(h.__file__).resolve() == Path(profile['health_module']['path']), 'service_live_source_origin_changed')
    identity = h._identity(profile_ref, profile, h._manager, h._process)
    need(identity['pid'] == os.getpid(), 'service_not_systemd_main_pid')
    return identity


def run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop):
    h = health_owner(); root = Path(profile['proof_root']) / ('cycle-%08d-%s' % (sequence, uuid.uuid4().hex))
    root.mkdir(mode=0o700)
    process = None
    try:
        request_ref, plan_ref, links = prepare_cycle(template_ref, kind, config, profile, root)
        put(root / 'intent.json', {'kind': kind, 'sequence': sequence, 'request_ref': request_ref,
            'automatic_retry_admitted': False})
        stdout_path, stderr_path = root / 'owner.stdout', root / 'owner.stderr'
        with os.fdopen(os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as out, \
             os.fdopen(os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as err:
            argv = [profile['unit']['argv'][0], '-B', profile['owner_source']['path'],
                '--request', request_ref['path'], '--expected-request-sha256', request_ref['sha256']]
            process = subprocess.Popen(argv, cwd=profile['unit']['working_directory'],
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True)
            handle = h.track_original_cycle(profile_ref, process, request_ref, sequence=sequence)
            deadline = time.monotonic() + profile['max_cycle_seconds']
            while process.poll() is None:
                need(not stop.is_set() and time.monotonic() < deadline, 'original_cycle_interrupted_or_timeout')
                need(stdout_path.stat().st_size <= h.MAX_JSON and stderr_path.stat().st_size <= h.MAX_JSON,
                    'original_owner_output_bound')
                stop.wait(0.1)
            need(process.wait() == 0, 'original_owner_failed')
        terminal_ref = put(root / 'terminal.json', original_terminal(h._raw(stdout_path), request_ref))
        native = observe_native(config)
        native_ref = put(root / 'native-diagnostic.json', native)
        proof = dict(h.cycle_start_fields(handle), schema=h.CYCLE_SCHEMA, kind=kind,
            finished_at=time.time(), finished_monotonic_ns=time.monotonic_ns(),
            request_ref=request_ref, plan_ref=plan_ref, terminal_ref=terminal_ref,
            execution_request_ref=h.descriptor(links['execution_request']),
            result_ref=h.descriptor(links['result']), origins_ref=h.descriptor(links['origins']),
            native_diagnostic_ref=native_ref)
        proof_ref = put(root / 'cycle-proof.json', proof)
        receipt_ref = h.publish_after_native_cycle(profile_ref, proof_ref, native,
            sequence=sequence, cycle_handle=handle)
        put(root / 'closed.json', {'cycle_proof_ref': proof_ref, 'receipt_ref': receipt_ref})
        return receipt_ref
    except BaseException as exc:
        # A signal is handed to the original owner, which owns cancellation and
        # custody. No synthetic closure, SIGKILL, reservation release or replay.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        persistent_hold(profile['proof_root'], root, str(exc))
        raise


def serve(config_ref, stop):
    h = health_owner(); config, profile_ref, profile = load_configuration(config_ref)
    root = h._path(profile['proof_root']); info = root.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and
        stat.S_IMODE(info.st_mode) == 0o700, 'service_proof_root_not_private')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not os.path.lexists(root / 'executor-held.json'), 'persistent_hold_requires_original_reconciliation')
        need(not any(not (path / 'closed.json').exists() for path in root.glob('cycle-*')), 'unfinished_cycle_requires_original_reconciliation')
        opening_identity = require_main_process(profile_ref, profile)
        sequence = 1
        if os.path.lexists(profile['receipt_path']):
            previous = h._document(h.descriptor(profile['receipt_path']))
            if previous.get('identity') == opening_identity:
                sequence = previous['sequence'] + 1
            # A new manager invocation starts sequence1; the canonical helper
            # requires the previous process definitely dead and archives its proof.
        for _ in range(config['max_cycles']):
            if stop.is_set():
                return 0
            try:
                need(h._document(config_ref) == config and h._profile(profile_ref) == profile,
                    'service_binding_changed')
                need(require_main_process(profile_ref, profile) == opening_identity, 'service_main_identity_changed')
                check_proof_budget(root, config['max_proof_bytes'])
                native = observe_native(config)
                template_ref, kind = select_template(config, profile, native)
                run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop)
                sequence += 1
                # An approved execution is one finite action. A fresh explicit
                # owner activation is needed before another task dispatch.
                if kind != 'genuine_scoped_idle':
                    persistent_hold(root, root, 'approved_execution_closed_no_automatic_followup')
                    return HOLD_EXIT
            except Exception as exc:
                persistent_hold(root, root, str(exc))
                return HOLD_EXIT
            if stop.wait(config['poll_seconds']):
                return 0
        persistent_hold(root, root, 'finite_cycle_budget_exhausted')
        return HOLD_EXIT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['serve'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    # This is set before any project import, including file-path invocation.
    os.environ['IPFS_ACCEL_SKIP_CORE'] = '1'
    if __package__ in (None, ''):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        need(sys.dont_write_bytecode and __debug__, 'service_requires_B_and_assertions')
        config_ref = health_owner().descriptor(args.config)
        need(config_ref['sha256'] == args.config_sha256, 'service_config_digest')
        return serve(config_ref, stop)
    except Exception as exc:
        print(json.dumps({'schema': 'agent-supervisor/scientific-executor-held@1',
            'reason': str(exc), 'automatic_retry_admitted': False}), file=sys.stderr)
        return HOLD_EXIT


if __name__ == '__main__':
    raise SystemExit(main())
