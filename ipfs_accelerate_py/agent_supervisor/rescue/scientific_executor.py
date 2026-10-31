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
RECOVERY_SCHEMA = 'agent-supervisor/scientific-known-nonstarted-recovery@1'
RECOVERY_SHA_ENV = 'IPFS_SCIENTIFIC_KNOWN_NONSTARTED_RECOVERY_SHA256'
RECOVERY_KEYS = frozenset({'schema', 'profile_ref', 'prior_profile_ref', 'prior_profile_snapshot_ref',
    'activation_capture_ref', 'activation_plan_ref', 'negative_audit_ref', 'memory_correction_ref',
    'source_commitment_ref', 'hold_ref', 'intent_ref', 'request_ref', 'owner_stdout_ref',
    'failed_cycle_root', 'scheduler_snapshot_ref', 'max_cycles', 'authorization', 'recorded_at', 'manager_before'})
RECOVERY_ARCHIVE_FIELDS = ('prior_profile_snapshot_ref', 'activation_capture_ref', 'activation_plan_ref',
    'negative_audit_ref', 'memory_correction_ref', 'source_commitment_ref', 'scheduler_snapshot_ref')


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


def _same_payload(left, right):
    """A durable historical snapshot keeps bytes/hash, not its temporary name."""
    return type(left) is dict and type(right) is dict and all(left.get(key) == right.get(key) for key in ('bytes', 'sha256'))


def _recovery_scheduler(request, snapshot_ref):
    """Read the original state; never allocate, normalize, prune or persist it."""
    h = health_owner(); expected = request['expected_scheduler_policy_sha256']
    family = {request['request_id'], request['request_id'] + ':root', request['request_id'] + ':phase'}
    archived = h._document(snapshot_ref)
    # The live scheduler is operational DATA; group-write permission is not
    # executable authority. The archived snapshot retains strict control rules.
    path = Path(request['scheduler_state_path']); before = path.lstat()
    need(0 < before.st_size <= h.MAX_JSON, 'recovery_scheduler_data_bound')
    live_ref = h.database_descriptor(path)
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid,
                       s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        first = os.fstat(stream.fileno()); raw = stream.read(h.MAX_JSON + 1)
        last = os.fstat(stream.fileno())
    need(stamp(before) == stamp(first) == stamp(last) == stamp(path.lstat())
         and len(raw) == live_ref['bytes'] <= h.MAX_JSON
         and hashlib.sha256(raw).hexdigest() == live_ref['sha256']
         and h.database_descriptor(path) == live_ref, 'recovery_scheduler_data_changed')
    for state in (archived, h._decode(raw)):
        need(type(state.get('config')) is dict, 'recovery_scheduler_policy_missing')
        digest = hashlib.sha256(json.dumps(state['config'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        need(digest == expected, 'recovery_scheduler_policy_changed')
        for field in ('waiters', 'leases'):
            rows = state.get(field)
            need(type(rows) is dict and all(type(row) is dict for row in rows.values()), 'recovery_scheduler_rows_unknown')
            need(not any(row.get('request_id') in family for row in rows.values()), 'recovery_original_request_still_active')


def _validate_nonstarted_recovery(value, profile_ref, profile, config):
    """Closed, idle-only negative reconciliation; this cannot establish health."""
    h = health_owner(); h._keys(value, RECOVERY_KEYS, 'recovery_shape')
    need(h._profile(profile_ref) == profile, 'recovery_current_profile_changed')
    need(value['schema'] == RECOVERY_SCHEMA and value['authorization'] == 'resume_genuine_scoped_idle'
         and value['profile_ref'] == profile_ref and value['max_cycles'] == config['max_cycles']
         and config['execution_templates'] == [], 'recovery_profile_or_authority')
    root = Path(profile['proof_root']); archive = root / 'reconciliation-v01'
    total = 0
    for field in RECOVERY_ARCHIVE_FIELDS:
        h._within(value[field], archive); h._bound(value[field]); total += value[field]['bytes']
    need(total <= 1024**2, 'recovery_archive_bound')
    prior = h._document(value['prior_profile_snapshot_ref'])
    need(_same_payload(value['prior_profile_ref'], value['prior_profile_snapshot_ref'])
         and value['prior_profile_ref']['path'] == profile_ref['path'], 'recovery_prior_profile_snapshot')
    for field in ('catalogue', 'task_scope', 'proof_root', 'receipt_path', 'max_age_seconds', 'max_cycle_seconds',
                  'health_module', 'owner_source', 'driver_source'):
        need(prior.get(field) == profile[field], 'recovery_prior_profile_scope')
    need(all(prior['unit'][key] == profile['unit'][key] for key in
         ('name', 'interpreter', 'working_directory', 'environment')), 'recovery_prior_unit_scope')
    need(prior['main_entrypoint']['path'] == profile['main_entrypoint']['path'], 'recovery_main_source_path')
    source = h._document(value['source_commitment_ref'])
    source_root = Path(profile['owner_source']['path']).parents[3]
    need(source.get('schema') == 'legal-ir-scientific-source-commitment/v1'
         and source.get('runtime_activation_or_task_authority') is False
         and source.get('source_root') == str(source_root),
         'recovery_source_commitment')
    inventory_ref = source['source_inventory']
    need(Path(profile['proof_root']).parent in Path(inventory_ref['path']).parents,
         'recovery_inventory_not_durable')
    inventory = h._document(inventory_ref)
    need(inventory.get('schema') == 'legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(key) == source.get(key) for key in ('source_root', 'source_commit', 'source_tree'))
         and type(source.get('source_commit')) is str and len(source['source_commit']) == 40
         and set(source['source_commit']) <= set('0123456789abcdef')
         and type(source.get('source_tree')) is str and len(source['source_tree']) == 40
         and set(source['source_tree']) <= set('0123456789abcdef'), 'recovery_inventory_commitment_join')
    rows = inventory.get('files')
    need(type(rows) is list and 0 < len(rows) <= 4096 and inventory.get('file_count') == len(rows)
         and all(type(row) is dict and type(row.get('path')) is str for row in rows), 'recovery_inventory_shape')
    indexed = {row['path']: row for row in rows}
    need(len(indexed) == len(rows), 'recovery_inventory_duplicate_paths')
    selected = [pin for pin in profile['source_descriptors'] if Path(pin['path']).is_relative_to(source_root)
        and Path(pin['path']).suffix == '.py']
    need(all(profile[field] in selected for field in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'))
         and all(all(indexed.get(pin['path'], {}).get(key) == pin[key] for key in ('path', 'bytes', 'sha256'))
             for pin in selected), 'recovery_inventory_current_source_mismatch')
    hold = h._document(value['hold_ref']); failed = h._path(value['failed_cycle_root'])
    need(value['hold_ref']['path'] == str(root / 'executor-held.json') and failed.parent == root
         and hold.get('schema') == 'agent-supervisor/scientific-executor-persistent-hold@1'
         and hold.get('cycle_root') == str(failed) and hold.get('reason') == 'original_owner_failed'
         and hold.get('automatic_retry_admitted') is False, 'recovery_exact_original_hold')
    need(value['intent_ref']['path'] == str(failed / 'intent.json')
         and value['request_ref']['path'] == str(failed / 'request.json')
         and value['owner_stdout_ref']['path'] == str(failed / 'owner.stdout'), 'recovery_original_artifact_paths')
    intent = h._document(value['intent_ref']); request = h._document(value['request_ref'])
    need(intent.get('kind') == 'genuine_scoped_idle' and intent.get('automatic_retry_admitted') is False
         and intent.get('request_ref') == value['request_ref'] and h._integer(intent.get('sequence'), 1),
         'recovery_original_idle_intent')
    need(request.get('schema') == 'legal-ir-original-native-control-request/v1'
         and request.get('purpose') == 'native_control_metadata'
         and all(request['source_sha256'].get(prior[field]['path']) == prior[field]['sha256']
             for field in ('owner_source', 'driver_source')), 'recovery_original_owner_source')
    terminal = original_terminal(h._bound(value['owner_stdout_ref']), value['request_ref'])
    need(terminal.get('completed') is False and terminal.get('child_launch_attempted') is False
         and terminal.get('child_reaped') is False and terminal.get('child_returncode') is None
         and terminal.get('control_effects') == 'not_started'
         and terminal.get('storage_status') == 'not_requested'
         and terminal.get('original_native_compute_closed') is True and terminal.get('source_READ_closed') is True
         and terminal.get('absolute_resource_operation_deadline_closed') is True
         and terminal.get('cleanup_errors') == [] and terminal.get('received_signals') == []
         and terminal.get('error_type') == 'ResourceAdmissionDeadlineError'
         and terminal.get('task_completion_authority_granted_by_resource_owner') is False
         and not any(field in terminal for field in ('reservation', 'execution_request', 'external_precharge', 'process_spawn_receipts')),
         'recovery_not_proven_known_nonstarted')
    need(terminal.get('database_preimage_before_grants') == request.get('database_preimage')
         and request['database_preimage']['path'] == config['catalogue']['database_path'],
         'recovery_original_database_join')
    attempt = h._path(request['attempt_directory'])
    need(failed in attempt.parents and attempt.is_dir() and not any(attempt.iterdir())
         and not (failed / 'closed.json').exists() and not (failed / 'cycle-proof.json').exists(),
         'recovery_attempt_or_receipt_not_empty')
    capture = h._document(value['activation_capture_ref']); plan = h._document(value['activation_plan_ref'])
    need(capture.get('schema') == 'legal-ir-scientific-first-install-actual-capture/v1'
         and capture.get('accepted') is False and capture.get('applied') is True
         and capture.get('source_READ_closed_before_start') is True and capture.get('start_requested') is True
         and _same_payload(capture['plan'], value['activation_plan_ref'])
         and _same_payload(plan['files'][1]['after'], value['prior_profile_snapshot_ref']), 'recovery_original_activation_join')
    need(all(config[key] == plan['main_loop_limits'][key]
         for key in ('max_cycles', 'poll_seconds', 'max_proof_bytes')), 'recovery_original_budget_changed')
    old_main = capture['opening_actual_main_identity']
    h._stored_identity(old_main, value['prior_profile_ref'], prior)
    negative = h._document(value['negative_audit_ref'])
    need(negative.get('schema') == 'scientific-first-activation-independent-negative-audit/v1'
         and negative.get('accepted_as_runtime_activation') is False
         and negative.get('negative_terminal_and_hold_verified') is True
         and negative.get('attempt_directory_empty') is True and negative.get('cycle_receipts_absent') is True
         and negative.get('original_terminal') == terminal and negative.get('persistent_hold') == hold
         and negative['failed_initial_activation']['opening_identity'] == old_main, 'recovery_independent_negative_join')
    correction = h._document(value['memory_correction_ref'])
    need(correction.get('schema') == 'scientific-service-memory-context-actual-repair/v1'
         and correction.get('accepted') is True and correction.get('native_envelope_and_policy_changed') is False
         and correction.get('restart_or_cycle_replay') is False
         and correction['scientific_after'].get('Id') == profile['unit']['name']
         and correction['scientific_after'].get('MainPID') == '0'
         and correction['scientific_after'].get('MemoryHigh') == 'infinity', 'recovery_cause_not_corrected')
    dropin = correction['persistent_manager_dropin']
    h._bound({key: dropin[key] for key in ('path', 'bytes', 'sha256')})
    need(dropin.get('text') and 'MemoryHigh=infinity' in dropin['text'].splitlines(), 'recovery_durable_memory_correction')
    observation = value['manager_before']
    need(type(observation) is dict and observation.get('Id') == profile['unit']['name']
         and observation.get('MainPID') == '0' and observation.get('ActiveState') in ('failed', 'inactive')
         and h._number(value['recorded_at']) and value['recorded_at'] <= time.time(), 'recovery_admin_main_absence')
    current_boot = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    need(h._definitely_dead_identity(old_main, current_boot), 'recovery_old_main_still_alive')
    need(type(terminal.get('owner_birth')) is str and terminal['owner_birth'].isdecimal(), 'recovery_owner_birth_unknown')
    owner = dict(pid=terminal['owner_pid'], birth_ticks=int(terminal['owner_birth']), boot_id=old_main['boot_id'])
    need(h._integer(owner['pid'], 2) and h._integer(owner['birth_ticks'], 1)
         and h._definitely_dead_identity(owner, current_boot), 'recovery_old_owner_still_alive')
    _recovery_scheduler(request, value['scheduler_snapshot_ref'])
    return failed


def authorize_known_nonstarted_recovery(spec_ref, output):
    """Explicit admin-only metadata step while the actual unit has no MainPID.

    Source/config adoption precedes this call; unit ENV pins the returned digest
    afterward. Evidence snapshots are data under the SAME proof-root archive.
    This function never starts a service, allocates resources or removes a hold.
    """
    h = health_owner(); spec = h._document(spec_ref)
    h._keys(spec, RECOVERY_KEYS - {'schema', 'recorded_at', 'manager_before'}, 'recovery_authorization_spec')
    profile = h._profile(spec['profile_ref'])
    config_ref = next(pin for pin in profile['source_descriptors'] if pin['path'] in profile['unit']['argv']
        and pin['path'].endswith('.json') and h._document(pin).get('schema') == CONFIG_SCHEMA)
    config, _, _ = load_configuration(config_ref)
    root = Path(profile['proof_root']); output = h._path(str(output))
    need((root / 'reconciliation-v01') in output.parents, 'recovery_authorization_outside_archive')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_NOFOLLOW), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = h._manager(profile['unit']['name'])
        need(manager.get('MainPID') == '0' and manager.get('ActiveState') in ('failed', 'inactive'), 'recovery_admin_requires_no_main')
        value = dict(spec, schema=RECOVERY_SCHEMA, recorded_at=time.time(), manager_before=manager)
        _validate_nonstarted_recovery(value, spec['profile_ref'], profile, config)
        return put(output, value)


def reconcile_known_nonstarted(recovery_ref, profile_ref, profile, config, opening_identity):
    """One exact negative acknowledgement; no closed.json or health receipt."""
    h = health_owner(); h._within(recovery_ref, Path(profile['proof_root']) / 'reconciliation-v01')
    value = h._document(recovery_ref)
    need(os.environ.get(RECOVERY_SHA_ENV) == recovery_ref['sha256'], 'recovery_receipt_not_unit_pinned')
    argv = profile['unit']['argv']; flag = '--known-nonstarted-recovery'
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
         and argv[argv.index(flag) + 1] == recovery_ref['path'], 'recovery_receipt_not_in_unit_argv')
    failed = _validate_nonstarted_recovery(value, profile_ref, profile, config)
    ack_path = failed / 'nonstarted-reconciliation.json'
    ack = dict(schema='agent-supervisor/scientific-nonstarted-reconciliation@1',
        recovery_ref=recovery_ref, hold_ref=value['hold_ref'], failed_cycle_root=str(failed),
        consuming_main_identity=opening_identity, health_credit=False, completion_authority=False,
        dispatch_authority=False)
    numbers = []
    for cycle in Path(profile['proof_root']).glob('cycle-*'):
        intent = h._document(h.descriptor(cycle / 'intent.json'))
        number = intent.get('cycle_number', intent.get('sequence'))
        need(h._integer(number, 1, config['max_cycles']), 'recovery_cycle_counter_unknown')
        need(cycle.name.startswith('cycle-%08d-' % number), 'recovery_cycle_number_path')
        need(cycle == failed or (cycle / 'closed.json').is_file(), 'recovery_other_unfinished_cycle')
        numbers.append(number)
    need(sorted(numbers) == list(range(1, len(numbers) + 1)), 'recovery_cycle_counter_gap_or_duplicate')
    need(len(numbers) < config['max_cycles'], 'recovery_cycle_budget_exhausted')
    if os.path.lexists(ack_path):
        need(h._document(h.descriptor(ack_path)) == ack, 'recovery_authorization_already_consumed_by_other_main')
    else:
        put(ack_path, ack)
    return len(numbers)


def _recovery_failure(root, cycle_number, sequence, reason):
    """Retain a later failure even while the original global hold is immutable."""
    matches = list(root.glob('cycle-%08d-*' % cycle_number))
    need(len(matches) <= 1, 'recovery_duplicate_cycle_number')
    if matches:
        cycle = matches[0]
    else:
        cycle = root / ('cycle-%08d-%s' % (cycle_number, uuid.uuid4().hex)); cycle.mkdir(mode=0o700)
        put(cycle / 'intent.json', dict(kind='not_selected', sequence=sequence,
            cycle_number=cycle_number, automatic_retry_admitted=False))
    if not (cycle / 'cycle-held.json').exists():
        put(cycle / 'cycle-held.json', dict(schema='agent-supervisor/scientific-cycle-held@1',
            reason=str(reason)[:2048], recorded_at=time.time(), automatic_retry_admitted=False))


def require_main_process(profile_ref, profile):
    """Authenticate the producer before spawning any original owner."""
    h = health_owner()
    need(threading.current_thread() is threading.main_thread(), 'service_not_main_thread')
    need(Path(__file__).resolve() == Path(profile['main_entrypoint']['path']) and
        Path(h.__file__).resolve() == Path(profile['health_module']['path']), 'service_live_source_origin_changed')
    identity = h._identity(profile_ref, profile, h._manager, h._process)
    need(identity['pid'] == os.getpid(), 'service_not_systemd_main_pid')
    return identity


def run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop, *, cycle_number=None):
    h = health_owner()
    need(cycle_number is None or h._integer(cycle_number, 1), 'cycle_number')
    root = Path(profile['proof_root']) / ('cycle-%08d-%s' % (sequence if cycle_number is None else cycle_number, uuid.uuid4().hex))
    root.mkdir(mode=0o700)
    process = None
    try:
        request_ref, plan_ref, links = prepare_cycle(template_ref, kind, config, profile, root)
        intent = {'kind': kind, 'sequence': sequence, 'request_ref': request_ref, 'automatic_retry_admitted': False}
        if cycle_number is not None: intent['cycle_number'] = cycle_number
        put(root / 'intent.json', intent)
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
        if cycle_number is not None: _recovery_failure(Path(profile['proof_root']), cycle_number, sequence, exc)
        raise


def serve(config_ref, stop, *, recovery_ref=None):
    h = health_owner(); config, profile_ref, profile = load_configuration(config_ref)
    root = h._path(profile['proof_root']); info = root.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and
        stat.S_IMODE(info.st_mode) == 0o700, 'service_proof_root_not_private')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reconciled_count = 0
        if recovery_ref is None:
            need(not os.path.lexists(root / 'executor-held.json'), 'persistent_hold_requires_original_reconciliation')
            need(not any(not (path / 'closed.json').exists() for path in root.glob('cycle-*')), 'unfinished_cycle_requires_original_reconciliation')
            opening_identity = require_main_process(profile_ref, profile)
        else:
            opening_identity = require_main_process(profile_ref, profile)
            reconciled_count = reconcile_known_nonstarted(recovery_ref, profile_ref, profile, config, opening_identity)
        sequence = 1
        if os.path.lexists(profile['receipt_path']):
            previous = h._document(h.descriptor(profile['receipt_path']))
            if previous.get('identity') == opening_identity:
                sequence = previous['sequence'] + 1
            # A new manager invocation starts sequence1; the canonical helper
            # requires the previous process definitely dead and archives its proof.
        cycle_number = reconciled_count + 1
        for _ in range(config['max_cycles'] - reconciled_count):
            if stop.is_set():
                return 0
            try:
                need(h._document(config_ref) == config and h._profile(profile_ref) == profile,
                    'service_binding_changed')
                need(require_main_process(profile_ref, profile) == opening_identity, 'service_main_identity_changed')
                check_proof_budget(root, config['max_proof_bytes'])
                native = observe_native(config)
                template_ref, kind = select_template(config, profile, native)
                if recovery_ref is None:
                    run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop)
                else:
                    run_cycle(config, profile_ref, profile, template_ref, kind, sequence, stop, cycle_number=cycle_number)
                sequence += 1
                cycle_number += 1
                # An approved execution is one finite action. A fresh explicit
                # owner activation is needed before another task dispatch.
                if kind != 'genuine_scoped_idle':
                    persistent_hold(root, root, 'approved_execution_closed_no_automatic_followup')
                    return HOLD_EXIT
            except Exception as exc:
                persistent_hold(root, root, str(exc))
                if recovery_ref is not None: _recovery_failure(root, cycle_number, sequence, exc)
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
    parser.add_argument('--known-nonstarted-recovery')
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
        recovery_ref = None
        if args.known_nonstarted_recovery:
            recovery_ref = health_owner().descriptor(args.known_nonstarted_recovery)
        return serve(config_ref, stop, recovery_ref=recovery_ref)
    except Exception as exc:
        print(json.dumps({'schema': 'agent-supervisor/scientific-executor-held@1',
            'reason': str(exc), 'automatic_retry_admitted': False}), file=sys.stderr)
        return HOLD_EXIT


if __name__ == '__main__':
    raise SystemExit(main())
