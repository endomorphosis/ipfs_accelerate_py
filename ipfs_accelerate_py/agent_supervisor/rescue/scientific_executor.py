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


CLOSED_IDLE_SCHEMA = 'agent-supervisor/scientific-known-closed-idle-recovery@1'
CLOSED_IDLE_SHA_ENV = 'IPFS_SCIENTIFIC_KNOWN_CLOSED_IDLE_RECOVERY_SHA256'
CLOSED_IDLE_KEYS = frozenset({'schema', 'recovery_kind', 'profile_ref', 'prior_profile_ref',
    'prior_profile_snapshot_ref', 'prior_authorization_ref', 'prior_ack_ref',
    'prior_administrative_capture_ref', 'negative_audit_ref', 'source_commitment_ref',
    'scheduler_snapshot_ref', 'current_native_ref', 'global_hold_ref', 'cycle_hold_ref',
    'cycle_proof_ref', 'max_cycles', 'authorization', 'recorded_at', 'manager_before'})
CLOSED_IDLE_ARCHIVE_FIELDS = ('prior_profile_snapshot_ref', 'prior_administrative_capture_ref',
    'negative_audit_ref', 'source_commitment_ref', 'scheduler_snapshot_ref', 'current_native_ref')


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


def _closed_idle_source_inventory(value, profile):
    """Authenticate the adopted generation; historical owner/driver may differ."""
    h = health_owner(); source = h._document(value['source_commitment_ref'])
    source_root = Path(profile['owner_source']['path']).parents[3]
    need(source.get('schema') == 'legal-ir-scientific-source-commitment/v1'
         and source.get('runtime_activation_or_task_authority') is False
         and source.get('source_root') == str(source_root), 'closed_idle_source_commitment')
    inventory_ref = source['source_inventory']
    need(Path(profile['proof_root']).parent in Path(inventory_ref['path']).parents,
         'closed_idle_inventory_not_durable')
    inventory = h._document(inventory_ref); rows = inventory.get('files')
    need(inventory.get('schema') == 'legal-ir-scientific-source-inventory/v1'
         and all(inventory.get(key) == source.get(key) for key in ('source_root', 'source_commit', 'source_tree'))
         and all(type(source.get(key)) is str and len(source[key]) == 40
                 and set(source[key]) <= set('0123456789abcdef') for key in ('source_commit', 'source_tree'))
         and type(rows) is list and 0 < len(rows) <= 4096 and inventory.get('file_count') == len(rows)
         and all(type(row) is dict and type(row.get('path')) is str for row in rows),
         'closed_idle_inventory_shape')
    indexed = {row['path']: row for row in rows}
    selected = [pin for pin in profile['source_descriptors']
                if Path(pin['path']).is_relative_to(source_root) and Path(pin['path']).suffix == '.py']
    need(len(indexed) == len(rows)
         and all(profile[field] in selected for field in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source'))
         and all(all(indexed.get(pin['path'], {}).get(key) == pin[key]
                     for key in ('path', 'bytes', 'sha256')) for pin in selected),
         'closed_idle_inventory_current_source_mismatch')


def _closed_idle_origins(origins, prior, request, plan, negative):
    """Historical incomplete execution provenance remains unlabelled; no health credit."""
    h = health_owner()
    need(type(origins) is dict and origins.get('schema') == 'legal-ir-pre-execution-project-import-origins/v1'
         and origins.get('project_origins_checked_before_module_execution') is True
         and origins.get('denials') == [], 'closed_idle_origin_guard_unknown')
    rows = origins.get('project_imports'); pins = {pin['path']: pin for pin in prior['source_descriptors']}
    need(type(rows) is list and 1 <= len(rows) <= 4096
         and len({row.get('module') for row in rows if type(row) is dict}) == len(rows),
         'closed_idle_origin_population')
    missing = []
    for row in rows:
        need(type(row) is dict and h._text(row.get('module')), 'closed_idle_origin_shape')
        pin = {key: row.get(key) for key in ('path', 'bytes', 'sha256')}
        need(pins.get(row.get('path')) == pin and request['source_sha256'].get(pin['path']) == pin['sha256'],
             'closed_idle_origin_unpinned')
        h._bound_project_source(pin, prior)
        if 'execution_profile' not in row:
            missing.append(row)
        else:
            need(row['execution_profile'] == 'compiled_exact_authenticated_source_bytes',
                 'closed_idle_origin_other_execution_state')
    need(len(missing) == 1 and missing[0].get('module') == 'ipfs_datasets_py.ipfs_backend_router'
         and missing[0] == negative.get('exact_unlabeled_origin')
         and negative.get('actual_origin_rows') == len(rows)
         and negative.get('compiled_authenticated_origin_rows') == len(rows) - 1,
         'closed_idle_not_exact_unlabelled_origin_refusal')
    command = plan['native_metadata_cli']; declared = command['project_import_plan']['source_pins']
    need(type(declared) is list and missing[0]['path'] == str(Path(command['project_import_plan']['datasets_root']) /
         'ipfs_datasets_py/ipfs_backend_router.py') and pins[missing[0]['path']] in declared,
         'closed_idle_resolution_source_binding')
    # Do not add execution_profile, rewrite origins, or call a health publisher.
    return rows


def _closed_idle_observation(result, request, plan, origins, prior, native, terminal):
    """Validate the read-only outcome separately from rejected health evidence."""
    h = health_owner(); command = plan.get('native_metadata_cli', {}); binding = command.get('observation_binding', {})
    h._keys(result, {'schema', 'backend', 'database_path', 'task_scope', 'observer_source', 'observer_callable',
        'catalogue_binding', 'ready_task_count', 'selection_idle_reason', 'unchanged', 'write_count',
        'provider_dispatched', 'attempt_consumed', 'completion_authority'}, 'closed_idle_result_shape')
    need(plan.get('schema') == 'native-embedded-scoped-observation-plan/v1'
         and plan.get('operation') == 'native_embedded_scoped_observation_v1'
         and result['schema'] == 'native-embedded-daemon-observation@1' and result['backend'] == 'embedded'
         and result['observer_callable'] == 'observe_embedded_database_without_dispatch'
         and result['selection_idle_reason'] == 'no_scoped_ready_tasks'
         and result['task_scope'] == plan.get('task_scope') == prior['task_scope']
         and type(result['ready_task_count']) is int and result['ready_task_count'] == 0
         and type(result['write_count']) is int and result['write_count'] == 0
         and result['unchanged'] is True and all(result[key] is False for key in
             ('provider_dispatched', 'attempt_consumed', 'completion_authority')),
         'closed_idle_outcome_not_readonly')
    db = request['database_preimage']
    h._keys(db, {'path', 'bytes', 'sha256'}, 'closed_idle_database_descriptor')
    need(h._integer(db['bytes'], 1, 128 * 1024**2) and type(db['sha256']) is str
         and len(db['sha256']) == 64 and set(db['sha256']) <= set('0123456789abcdef'), 'closed_idle_database_descriptor_bounds')
    h._path(db['path'])
    need(terminal.get('database_preimage_before_grants') == db == terminal.get('database_preimage_observed')
         == terminal.get('database_postimage') == binding.get('database_preimage')
         and result['database_path'] == db['path'] and h.database_descriptor(db['path']) == db,
         'closed_idle_database_history_or_current_changed')
    source = result['observer_source']; route = command.get('import_head_capsule_Git', {}).get('route_source')
    need(command.get('schema') == 'native-embedded-scoped-observation-cli/v2'
         and command.get('runtime_root') == plan.get('accelerate_root')
         and binding.get('schema') == 'native-embedded-daemon-observation-binding@1'
         and source == binding.get('observer_source') and source in prior['source_descriptors']
         and route in prior['source_descriptors']
         and all(request['source_sha256'].get(pin['path']) == command['source_sha256'].get(pin['path']) == pin['sha256']
                 for pin in (source, route)), 'closed_idle_observer_or_route_source')
    rows = origins['project_imports']
    for name, pin in (('ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap', source),
                      ('ipfs_accelerate_py.agent_implementation_route', route)):
        need(any(row.get('module') == name and all(row.get(key) == pin[key] for key in ('path', 'bytes', 'sha256'))
                 and row.get('execution_profile') == 'compiled_exact_authenticated_source_bytes' for row in rows),
             'closed_idle_observer_body_not_compiled')
    catalogue = result['catalogue_binding']
    h._keys(catalogue, {'database_preimage', 'plan_root_cid', 'repository_tree_id', 'source_schema',
        'event_cursor', 'task_statuses'}, 'closed_idle_catalogue_shape')
    need(catalogue['database_preimage'] == db and binding.get('task_scope') == prior['task_scope']
         and catalogue['source_schema'] == binding.get('source_schema') ==
             'ipfs_accelerate_py/agent-supervisor/database-task-source@1'
         and all(catalogue[key] == binding.get(key) == prior['catalogue'][key]
                 for key in ('plan_root_cid', 'repository_tree_id'))
         and h._integer(catalogue['event_cursor']) and catalogue['event_cursor'] == native['event_cursor'], 'closed_idle_catalogue_binding')
    expected = {row['task_cid']: {key: row[key] for key in ('task_cid', 'status', 'revision')}
                for row in native['tasks'] if row['task_cid'] in prior['task_scope']}
    tasks = catalogue['task_statuses']
    need(type(tasks) is list and len(tasks) == len(expected) == len(prior['task_scope'])
         and len({row.get('task_cid') for row in tasks if type(row) is dict}) == len(tasks)
         and all(type(row) is dict and set(row) == {'task_cid', 'status', 'revision'}
                 and h._integer(row.get('revision'), 1)
                 and row == expected.get(row['task_cid']) and row['status'] in QUIET_STATUSES for row in tasks),
         'closed_idle_catalogue_tasks_changed')


def _validate_closed_idle_recovery(value, profile_ref, profile, config):
    """Explicit negative reconciliation of exactly two spent idle observations."""
    h = health_owner(); h._keys(value, CLOSED_IDLE_KEYS, 'closed_idle_recovery_shape')
    need(value['schema'] == CLOSED_IDLE_SCHEMA and value['recovery_kind'] == 'known_closed_idle_health_rejection'
         and value['authorization'] == 'resume_genuine_scoped_idle' and value['profile_ref'] == profile_ref
         and h._profile(profile_ref) == profile and value['max_cycles'] == config['max_cycles']
         and config['max_cycles'] > 2 and config['execution_templates'] == [], 'closed_idle_recovery_authority')
    root = Path(profile['proof_root']); archive = root / 'reconciliation-v02'; total = 0
    for key in CLOSED_IDLE_ARCHIVE_FIELDS:
        h._within(value[key], archive); h._bound(value[key]); total += value[key]['bytes']
    need(total <= 1024**2, 'closed_idle_archive_bound')
    prior = h._document(value['prior_profile_snapshot_ref'])
    need(_same_payload(value['prior_profile_ref'], value['prior_profile_snapshot_ref'])
         and value['prior_profile_ref']['path'] == profile_ref['path'], 'closed_idle_prior_profile_snapshot')
    for key in ('catalogue', 'task_scope', 'proof_root', 'receipt_path', 'max_age_seconds', 'max_cycle_seconds', 'cycle_contract'):
        need(prior.get(key) == profile[key], 'closed_idle_prior_scope_changed')
    need(all(prior['unit'][key] == profile['unit'][key] for key in
         ('name', 'interpreter', 'working_directory', 'environment')), 'closed_idle_prior_unit_changed')
    _closed_idle_source_inventory(value, profile)
    old_auth = h._document(value['prior_authorization_ref']); old_ack = h._document(value['prior_ack_ref'])
    capture = h._document(value['prior_administrative_capture_ref']); negative = h._document(value['negative_audit_ref'])
    need(old_auth.get('schema') == RECOVERY_SCHEMA and old_auth.get('profile_ref') == value['prior_profile_ref']
         and old_auth.get('authorization') == 'resume_genuine_scoped_idle'
         and old_auth.get('max_cycles') == config['max_cycles']
         and old_ack.get('schema') == 'agent-supervisor/scientific-nonstarted-reconciliation@1'
         and old_ack.get('recovery_ref') == value['prior_authorization_ref']
         and all(old_ack.get(key) is False for key in ('health_credit', 'completion_authority', 'dispatch_authority')),
         'closed_idle_prior_authorization_or_ack')
    original_plan = h._document(old_auth['activation_plan_ref'])
    need(all(config[key] == original_plan['main_loop_limits'][key] for key in
             ('max_cycles', 'poll_seconds', 'max_proof_bytes')), 'closed_idle_original_budget_changed')
    first_request = h._document(old_auth['request_ref'])
    first_terminal = original_terminal(h._bound(old_auth['owner_stdout_ref']), old_auth['request_ref'])
    first_negative = h._document(old_auth['negative_audit_ref'])
    need(first_terminal.get('completed') is False and first_terminal.get('child_launch_attempted') is False
         and first_terminal.get('child_reaped') is False and first_terminal.get('child_returncode') is None
         and first_terminal.get('control_effects') == 'not_started' and first_terminal.get('storage_status') == 'not_requested'
         and first_terminal.get('source_READ_closed') is True and first_terminal.get('original_native_compute_closed') is True
         and first_terminal.get('absolute_resource_operation_deadline_closed') is True
         and first_terminal.get('cleanup_errors') == [] and first_terminal.get('received_signals') == []
         and first_terminal.get('task_completion_authority_granted_by_resource_owner') is False
         and first_terminal.get('database_preimage_before_grants') == first_request.get('database_preimage')
         and first_negative.get('negative_terminal_and_hold_verified') is True
         and first_negative.get('original_terminal') == first_terminal
         and first_negative.get('persistent_hold') == h._document(value['global_hold_ref']),
         'closed_idle_first_nonstarted_history_changed')
    first_capture = h._document(old_auth['activation_capture_ref'])
    first_main = first_capture['opening_actual_main_identity']
    first_prior = h._document(old_auth['prior_profile_snapshot_ref'])
    h._stored_identity(first_main, old_auth['prior_profile_ref'], first_prior)
    old_main = capture.get('opening_identity'); h._stored_identity(old_main, value['prior_profile_ref'], prior)
    need(capture.get('schema') == 'scientific-explicit-known-nonstarted-recovery-actual-capture/v1'
         and capture.get('accepted') is False and capture.get('start_requested') is True
         and capture.get('source_leases_closed_before_start') is True and capture.get('hold_removed') is False
         and capture.get('native_or_task_or_epoch_or_policy_reset') is False
         and capture.get('recovery_authorization_ref') == value['prior_authorization_ref']
         and old_ack.get('consuming_main_identity') == old_main, 'closed_idle_original_main_lineage')
    need(negative.get('schema') == 'scientific-native-completed-idle-health-failure-independent-negative-audit/v1'
         and _same_payload(negative.get('actual_administrative_capture'), value['prior_administrative_capture_ref'])
         and negative.get('actual_profile_ref') == value['prior_profile_ref']
         and negative.get('authorization_ref') == value['prior_authorization_ref']
         and negative.get('negative_ack_ref') == value['prior_ack_ref']
         and negative.get('accepted_as_executor_health_or_activation') is False
         and negative.get('completed_native_read_only_cycle_independently_verified') is True
         and negative.get('native_source_process_storage_closed') is True
         and negative.get('task_or_provider_or_completion_effects') is False
         and negative.get('database_unchanged_during_original_cycle') is True
         and negative.get('exact_original_cycle_number') == 2 and negative.get('health_sequence_attempted') == 1,
         'closed_idle_independent_negative_join')
    hold = h._document(value['global_hold_ref']); cycle_hold = h._document(value['cycle_hold_ref'])
    failed = Path(value['cycle_proof_ref']['path']).parent
    need(value['global_hold_ref'] == old_auth['hold_ref'] == old_ack['hold_ref'] == negative['global_original_hold_ref']
         and value['global_hold_ref']['path'] == str(root / 'executor-held.json')
         and hold.get('reason') == 'original_owner_failed' and hold.get('automatic_retry_admitted') is False
         and value['cycle_hold_ref']['path'] == str(failed / 'cycle-held.json')
         and value['cycle_hold_ref'] == negative['new_cycle_hold_ref']
         and cycle_hold.get('schema') == 'agent-supervisor/scientific-cycle-held@1'
         and cycle_hold.get('reason') == 'original_import_origin_unbound'
         and cycle_hold.get('automatic_retry_admitted') is False, 'closed_idle_exact_two_holds')
    cycles = sorted(root.glob('cycle-*'))
    need(len(cycles) == 2 and failed.parent == root and failed.name.startswith('cycle-00000002-')
         and cycles[0] == Path(old_auth['failed_cycle_root'])
         and cycles[1] == failed and value['prior_ack_ref']['path'] == str(cycles[0] / 'nonstarted-reconciliation.json')
         and all(not (cycle / 'closed.json').exists() for cycle in cycles)
         and not os.path.lexists(profile['receipt_path']), 'closed_idle_cycle_history_or_receipt')
    intent = h._document(h.descriptor(failed / 'intent.json')); proof = h._document(value['cycle_proof_ref'])
    need(intent.get('kind') == 'genuine_scoped_idle' and intent.get('automatic_retry_admitted') is False
         and type(intent.get('cycle_number')) is int and intent['cycle_number'] == 2
         and type(intent.get('sequence')) is int and intent['sequence'] == 1
         and intent.get('request_ref') == proof.get('request_ref')
         and proof.get('kind') == 'genuine_scoped_idle' and proof.get('sequence') == 1,
         'closed_idle_original_intent')
    for key in ('request_ref', 'execution_request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref'):
        need(proof[key] == negative.get(key), 'closed_idle_negative_artifact_join')
    request, terminal, result, plan, origins, native = (h._document(proof[key]) for key in
         ('request_ref', 'terminal_ref', 'result_ref', 'plan_ref', 'origins_ref', 'native_diagnostic_ref'))
    need(original_terminal(h._raw(failed / 'owner.stdout'), proof['request_ref']) == terminal,
         'closed_idle_original_stdout_terminal')
    native_identity = h._native(native, prior, now=proof['finished_at'])
    need(proof['started_at'] <= native['checked_at_epoch'] <= proof['finished_at'], 'closed_idle_native_capture_outside_cycle')
    execution = h._document(proof['execution_request_ref'])
    need(execution.get('completion_authority') is False and execution.get('provider_authority') is False,
         'closed_idle_execution_authority')
    # Reuse every original terminal/process/native/storage check preceding the
    # historical label refusal. This is an expected rejection, never health.
    try:
        h._cycle(value['cycle_proof_ref'], prior, old_main, native_identity,
                 now=proof['finished_at'], monotonic_ns=proof['finished_monotonic_ns'])
    except h.ExecutorHealthError as exc:
        need(str(exc) == 'original_import_origin_unbound', 'closed_idle_other_original_cycle_refusal')
    else:
        raise ExecutorHold('closed_idle_historical_health_was_not_rejected')
    _closed_idle_origins(origins, prior, request, plan, negative)
    _closed_idle_observation(result, request, plan, origins, prior, native, terminal)
    need(terminal.get('absolute_resource_operation_deadline_closed') is True
         and terminal.get('child_launch_attempted') is True
         and terminal.get('control_effects') == 'actor_exit_zero_independent_task_receipt_still_required'
         and terminal.get('task_completion_authority_granted_by_resource_owner') is False,
         'closed_idle_original_terminal_authority')
    current = h._document(value['current_native_ref']); h._native(current, profile)
    def scoped_tasks(report):
        return sorted(({key: row[key] for key in ('task_cid', 'status', 'revision')}
                       for row in report['tasks'] if row['task_cid'] in profile['task_scope']), key=lambda row: row['task_cid'])
    need(current['event_cursor'] == native['event_cursor'] and scoped_tasks(current) == scoped_tasks(native),
         'closed_idle_current_tasks_changed')
    boot = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    need(h._definitely_dead_identity(first_main, boot), 'closed_idle_first_main_alive')
    need(type(first_terminal.get('owner_birth')) is str and first_terminal['owner_birth'].isdecimal(),
         'closed_idle_first_owner_birth_unknown')
    first_owner = dict(pid=first_terminal['owner_pid'], birth_ticks=int(first_terminal['owner_birth']), boot_id=first_main['boot_id'])
    need(h._integer(first_owner['pid'], 2) and h._integer(first_owner['birth_ticks'], 1)
         and h._definitely_dead_identity(first_owner, boot), 'closed_idle_first_owner_alive')
    need(h._definitely_dead_identity(old_main, boot), 'closed_idle_prior_main_alive')
    for old in terminal['sampled_process_custody_final']['observed_identities']:
        need(type(old) is dict and h._integer(old.get('pid'), 2) and h._integer(old.get('birth_ticks'), 1)
             and h._definitely_dead_identity(dict(old, boot_id=old_main['boot_id']), boot), 'closed_idle_prior_descendant_alive')
    owner = proof['owner_observation']
    need(h._definitely_dead_identity(owner, boot), 'closed_idle_prior_owner_alive')
    manager = value['manager_before']
    need(manager.get('Id') == profile['unit']['name'] and manager.get('MainPID') == '0'
         and manager.get('ActiveState') in ('failed', 'inactive')
         and h._number(value['recorded_at']) and value['recorded_at'] <= time.time(), 'closed_idle_admin_main_absence')
    _recovery_scheduler(request, value['scheduler_snapshot_ref'])
    _recovery_scheduler(h._document(old_auth['request_ref']), value['scheduler_snapshot_ref'])
    # The replacement source must bind the corrected guard plus new owners;
    # historical role hashes are evidence and are never imposed on the new run.
    future = h._document(h._document(config['idle_template_ref'])['plan_ref'])['native_metadata_cli']
    guard = future.get('pre_body_guard'); prior_guard = plan['native_metadata_cli'].get('pre_body_guard')
    need(type(guard) is dict and guard in profile['source_descriptors'] and guard != prior_guard
         and all(profile[key]['sha256'] != prior[key]['sha256'] for key in ('owner_source', 'driver_source')),
         'closed_idle_cause_source_not_adopted')
    h._bound(guard)
    return failed


def authorize_known_closed_idle_recovery(spec_ref, output):
    """Existing administrative protocol, distinct typed one-use authorization."""
    h = health_owner(); spec = h._document(spec_ref)
    h._keys(spec, CLOSED_IDLE_KEYS - {'schema', 'recorded_at', 'manager_before'}, 'closed_idle_authorization_spec')
    profile = h._profile(spec['profile_ref'])
    config_ref = next(pin for pin in profile['source_descriptors'] if pin['path'] in profile['unit']['argv']
        and pin['path'].endswith('.json') and h._document(pin).get('schema') == CONFIG_SCHEMA)
    config, _, _ = load_configuration(config_ref); root = Path(profile['proof_root']); output = h._path(str(output))
    need(output.parent == root / 'reconciliation-v02' and output.name == 'known-closed-idle-authorization.json',
         'closed_idle_authorization_versioned_path')
    with os.fdopen(os.open(root / 'scientific-executor.lock', os.O_RDWR | os.O_NOFOLLOW), 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = h._manager(profile['unit']['name'])
        need(manager.get('MainPID') == '0' and manager.get('ActiveState') in ('failed', 'inactive'),
             'closed_idle_admin_requires_no_main')
        value = dict(spec, schema=CLOSED_IDLE_SCHEMA, recorded_at=time.time(), manager_before=manager)
        _validate_closed_idle_recovery(value, spec['profile_ref'], profile, config)
        return put(output, value)


def reconcile_known_closed_idle(recovery_ref, profile_ref, profile, config, opening_identity):
    """Retain both holds and consumed old AUTH/ACK; spend neither a third run nor health."""
    h = health_owner(); root = Path(profile['proof_root'])
    need(recovery_ref['path'] == str(root / 'reconciliation-v02/known-closed-idle-authorization.json'),
         'closed_idle_recovery_versioned_path')
    value = h._document(recovery_ref)
    need(os.environ.get(CLOSED_IDLE_SHA_ENV) == recovery_ref['sha256'], 'closed_idle_receipt_not_unit_pinned')
    argv = profile['unit']['argv']; flag = '--known-closed-idle-recovery'
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv) and argv[argv.index(flag) + 1] == recovery_ref['path']
         and '--known-nonstarted-recovery' not in argv, 'closed_idle_receipt_not_in_unit_argv')
    failed = _validate_closed_idle_recovery(value, profile_ref, profile, config)
    ack_path = failed / 'closed-idle-reconciliation-v01.json'
    ack = dict(schema='agent-supervisor/scientific-closed-idle-reconciliation@1', recovery_ref=recovery_ref,
        retained_global_hold_ref=value['global_hold_ref'], retained_cycle_hold_ref=value['cycle_hold_ref'],
        prior_authorization_ref=value['prior_authorization_ref'], prior_ack_ref=value['prior_ack_ref'],
        rejected_cycle_proof_ref=value['cycle_proof_ref'], consuming_main_identity=opening_identity,
        spent_cycle_count=2, next_cycle_number=3, first_health_sequence=1,
        health_credit=False, completion_authority=False, dispatch_authority=False, historical_origins_relabelled=False)
    if os.path.lexists(ack_path):
        need(h._document(h.descriptor(ack_path)) == ack, 'closed_idle_authorization_consumed_by_other_main')
    else:
        put(ack_path, ack)
    return 2


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
            kind = h._document(recovery_ref).get('schema')
            if kind == RECOVERY_SCHEMA:
                reconciled_count = reconcile_known_nonstarted(recovery_ref, profile_ref, profile, config, opening_identity)
            else:
                need(kind == CLOSED_IDLE_SCHEMA, 'unknown_typed_recovery')
                reconciled_count = reconcile_known_closed_idle(recovery_ref, profile_ref, profile, config, opening_identity)
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
    recovery_flags = parser.add_mutually_exclusive_group()
    recovery_flags.add_argument('--known-nonstarted-recovery')
    recovery_flags.add_argument('--known-closed-idle-recovery')
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
        recovery_path = args.known_nonstarted_recovery or args.known_closed_idle_recovery
        if recovery_path:
            recovery_ref = health_owner().descriptor(recovery_path)
        return serve(config_ref, stop, recovery_ref=recovery_ref)
    except Exception as exc:
        print(json.dumps({'schema': 'agent-supervisor/scientific-executor-held@1',
            'reason': str(exc), 'automatic_retry_admitted': False}), file=sys.stderr)
        return HOLD_EXIT


if __name__ == '__main__':
    raise SystemExit(main())
