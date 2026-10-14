"""Artifact-only readout of an explicitly registered identity/comments live pair.

Reads retained preparation/configuration/result metadata only. The external
prelaunch lock owns execution custody; this readout never launches a provider.
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

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding.native_usage_accounting import native_usage_contract

TOKENS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens', 'total_tokens')
PROFILES = {'identity': 'identity@1', 'comments': 'python-comments@1'}
CONFIG_DIFFS = {'/agents/0/kwargs/source_view_profile', '/jobs_dir'}
SOURCE_COUNTS = ('sessions', 'accepted_requests', 'prepared_pages', 'delivered_pages', 'delivered_responses', 'response_bytes', 'delivered_response_bytes', 'prepared_source_bytes', 'delivered_source_bytes', 'request_hash_count', 'response_hash_count', 'unique_request_hash_count', 'unique_response_hash_count')
SESSION_SOURCE_COUNTS = tuple(k for k in SOURCE_COUNTS if k not in {'sessions', 'unique_request_hash_count', 'unique_response_hash_count'})
EDIT_COUNTS = ('prepared_mapping_pages', 'delivered_mapping_pages', 'prepared_confirmation_pages', 'delivered_confirmation_pages')
STATUS_CODES = ('ok', 'invalid_request', 'peer_refused', 'stale_context', 'budget_exhausted', 'session_closed', 'retrieval_refused')
MISSING = object()


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def obj(value):
    return value if type(value) is dict else {}


def count(value):
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def tag(value):
    return value if type(value) is str and re.fullmatch(r'[A-Za-z0-9_./:@+-]{1,256}', value) else None


def digest(value):
    return value if type(value) is str and re.fullmatch('[0-9a-f]{64}', value) else None


def read(path, *, optional=False):
    path = Path(path).absolute()
    try:
        before = path.lstat()
    except FileNotFoundError:
        if optional:
            return None, {'path': str(path), 'availability': 'missing'}
        raise
    need(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode) and before.st_nlink == 1
         and before.st_size <= 64 * 1024**2, 'bounded canonical regular metadata required')
    def identity(info):
        return tuple(getattr(info, key) for key in ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        need(identity(before) == identity(os.fstat(handle.fileno())), 'metadata changed before read')
        raw = handle.read(64 * 1024**2 + 1)
        after = os.fstat(handle.fileno())
    need(identity(before) == identity(after) == identity(path.lstat()) and len(raw) == before.st_size, 'metadata changed during read')
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'duplicate metadata field'); result[key] = value
        return result
    def nonfinite(_):
        raise ValueError('nonfinite metadata')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    need(type(value) is dict, 'object metadata required')
    return value, {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def source_binding(path):
    path = Path(path).absolute()
    before = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode) and before.st_nlink == 1
         and before.st_size <= 8 * 1024**2, 'bounded canonical pinned source required')
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    identity = lambda value: tuple(getattr(value, key) for key in fields)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as handle:
        need(identity(before) == identity(os.fstat(handle.fileno())), 'pinned source changed before read')
        raw = handle.read(8 * 1024**2 + 1)
        after = os.fstat(handle.fileno())
    need(identity(before) == identity(after) == identity(path.lstat()) and len(raw) == before.st_size, 'pinned source changed during read')
    return sha(raw)


def differences(left, right, path=''):
    if type(left) is dict and type(right) is dict:
        return sum((differences(left.get(k, MISSING), right.get(k, MISSING), path + '/' + k)
                    for k in sorted(left.keys() | right.keys())), [])
    if type(left) is list and type(right) is list and len(left) == len(right):
        return sum((differences(a, b, path + '/' + str(i)) for i, (a, b) in enumerate(zip(left, right))), [])
    return [] if type(left) is type(right) and left == right else [path]


def counters(value, keys):
    if value is None:
        return None
    need(type(value) is dict and set(value) <= set(keys), 'unexpected counter fields')
    need(all(count(v) is not None for v in value.values()), 'invalid counter')
    return dict(value)


def usage(report):
    raw_rows = report.get('provider_invocations', [])
    need(type(raw_rows) is list and len(raw_rows) <= 256, 'bounded invocation list required')
    identities, rows = set(), []
    for raw in raw_rows:
        need(type(raw) is dict and type(raw.get('invocation_id')) is str and re.fullmatch('[0-9a-f]{32}', raw['invocation_id']), 'invocation identity required')
        need(raw['invocation_id'] not in identities, 'duplicate invocation identity')
        identities.add(raw['invocation_id'])
        native = obj(raw.get('native_rollout_usage'))
        complete, scope, cache = native_usage_contract(raw, native)
        observed = {k: count(obj(native.get('usage')).get(k)) for k in TOKENS}
        if scope == 'native_session_cumulative':
            coherent = all(observed[k] is not None for k in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')) and observed['total_tokens'] == observed['input_tokens'] + observed['output_tokens'] and observed['cached_input_tokens'] <= observed['input_tokens']
        elif scope == 'native_final_envelope':
            coherent = all(v is not None for v in observed.values()) and observed['total_tokens'] == sum(observed[k] for k in TOKENS[:-1])
        else:
            coherent = False
        if not coherent:
            complete = None
        dispatch = obj(raw.get('source_view_retrieval'))
        rows.append(dict(invocation_id=raw['invocation_id'], phase=tag(raw.get('phase')), purpose=tag(raw.get('purpose')),
            provider=tag(raw.get('provider')), model=tag(raw.get('model')), status=tag(raw.get('status')), error_type=tag(raw.get('error_type')),
            seconds=number(raw.get('seconds')), timeout_seconds=number(raw.get('timeout_seconds')),
            model_prompt_bytes=count(raw.get('model_prompt_bytes')), model_prompt_sha256=digest(raw.get('model_prompt_sha256')),
            native_usage_complete_observed=complete, token_count_scope=scope, cache_included_in_input=cache,
            native_accounting_coherent=coherent, reported_native_counts=observed,
            tokens={k: value if complete is True else None for k, value in observed.items()},
            source_dispatch=None if not dispatch else dict(schema=tag(dispatch.get('schema')), profile=tag(dispatch.get('profile')),
                source_edit_mode=tag(dispatch.get('source_edit_mode')), descriptor_sha256=digest(dispatch.get('descriptor_sha256')),
                session_id_sha256=sha(dispatch['session_id'].encode()) if type(dispatch.get('session_id')) is str else None,
                guidance_sha256=digest(dispatch.get('guidance_sha256')), guidance_bytes=count(dispatch.get('guidance_bytes')))))
    unreceipted = report.get('unreceipted_provider_attempt') is True
    complete = bool(rows) and not unreceipted and all(r['native_usage_complete_observed'] is True for r in rows)
    def sums(key, selected):
        return sum(r['tokens'][key] for r in selected) if selected and all(r['tokens'][key] is not None for r in selected) else None
    return dict(invocations=rows, observed_router_sessions=len(rows), unreceipted_provider_attempt=unreceipted,
        native_usage_complete_observed=complete,
        tokens={k: sums(k, rows) if complete else None for k in TOKENS},
        phase_tokens={phase: {k: sums(k, [r for r in rows if r['phase'] == phase]) for k in TOKENS} for phase in ('planning', 'coding')},
        failed_or_incomplete_invocation_count=sum(r['error_type'] is not None or r['native_usage_complete_observed'] is not True for r in rows),
        accounting_scope='All retained unique supervisor invocation receipts; native internal turns and unreported auxiliary calls are not separate observations.',
        cache_categories_added_to_reported_total=False, billing_total_verified=False)


def source_usage(report, invocations, profile):
    value = report.get('source_view_retrieval_observation')
    if value is None:
        return dict(status='unavailable', helper_preparation_observed=None, helper_confirmation_delivery_observed=None,
                    ordinary_tool_usage=None, reason='source_session_observation_missing')
    need(type(value) is dict, 'source observation object required')
    if value.get('status') != 'observed':
        return dict(status='unavailable', helper_preparation_observed=None, helper_confirmation_delivery_observed=None,
                    ordinary_tool_usage=None, reason='source_session_observation_not_complete')
    need(value.get('schema') == 'terminal-source-view-observation@1' and value.get('source_edit_mode') == 'mapped-span@2', 'unexpected source observation mode')
    need(all(value.get(key) is False for key in ('proof_authority', 'execution_authority', 'completion_authority', 'raw_queries_exported', 'raw_sources_exported')), 'source observation authority differs')
    totals = counters(value.get('counts'), SOURCE_COUNTS)
    edits = counters(value.get('source_edit_counts'), EDIT_COUNTS)
    need(totals is not None and set(totals) == set(SOURCE_COUNTS) and edits is not None and set(edits) == set(EDIT_COUNTS), 'complete selected source counters required')
    sessions = value.get('sessions')
    need(type(sessions) is list and 1 <= len(sessions) <= 16, 'bounded source sessions required')
    dispatches = {sha(r['invocation_id'].encode()): r['source_dispatch'] for r in invocations if r['source_dispatch'] is not None}
    seen, projected = set(), []
    for session in sessions:
        identity = session.get('invocation_id_sha256')
        need(identity in dispatches and identity not in seen, 'source session dispatch join missing or duplicate'); seen.add(identity)
        dispatched = dispatches[identity]
        need(session.get('profile') == dispatched['profile'] == profile and session.get('source_edit_mode') == dispatched['source_edit_mode'] == 'mapped-span@2', 'source dispatch selection differs')
        need(session.get('descriptor_sha256') == dispatched['descriptor_sha256'] and session.get('session_id_sha256') == dispatched['session_id_sha256'], 'source session identity differs')
        need(type(session.get('counts')) is dict and set(session['counts']) == set(SESSION_SOURCE_COUNTS)
             and type(session.get('source_edit_counts')) is dict and set(session['source_edit_counts']) == set(EDIT_COUNTS), 'complete session counters required')
        need(digest(session.get('audit_sha256')) is not None and dispatched['descriptor_sha256'] is not None and dispatched['session_id_sha256'] is not None, 'source session digests required')
        projected.append(dict(invocation_id_sha256=identity, profile=profile, source_edit_mode='mapped-span@2',
            audit_sha256=digest(session.get('audit_sha256')), descriptor_sha256=dispatched['descriptor_sha256'],
            counts=counters(session.get('counts'), SESSION_SOURCE_COUNTS), source_edit_counts=counters(session.get('source_edit_counts'), EDIT_COUNTS),
            status_counts=counters(session.get('status_counts'), STATUS_CODES)))
    need(set(dispatches) == seen and len(sessions) == totals.get('sessions'), 'incomplete selected source session coverage')
    for key in EDIT_COUNTS:
        need(edits[key] == sum(row['source_edit_counts'][key] for row in projected), 'edit aggregate differs')
    for key in SESSION_SOURCE_COUNTS:
        need(totals.get(key) == sum(row['counts'][key] for row in projected), 'source aggregate differs')
    need(all(sum(row['status_counts'].values()) == row['counts']['accepted_requests'] for row in projected), 'session statuses incomplete')
    need(totals['unique_request_hash_count'] <= totals['request_hash_count'] and totals['unique_response_hash_count'] <= totals['response_hash_count'], 'unique count exceeds observations')
    need(edits['delivered_confirmation_pages'] <= edits['prepared_confirmation_pages'] <= edits['prepared_mapping_pages']
         and edits['delivered_mapping_pages'] <= edits['prepared_mapping_pages'], 'impossible edit page accounting')
    return dict(status='observed', profile=profile, source_edit_mode='mapped-span@2', counts=totals, source_edit_counts=edits, sessions=projected,
        helper_preparation_observed=edits['delivered_mapping_pages'] > 0,
        helper_confirmation_delivery_observed=edits['delivered_confirmation_pages'] > 0,
        ordinary_tool_usage=None, ordinary_tool_usage_reason='Native ordinary shell/edit tool census is not retained by these metadata receipts.',
        zero_helper_use_establishes_no_efficiency_advantage=edits['delivered_mapping_pages'] == 0,
        owner_response_byte_scope='Canonical JSON response bodies, excluding frame prefixes, CLI text, commands and conversation history.',
        source_byte_scope='Source slices including repeated reads; does not distinguish compact from original expansion in retained aggregate counters.',
        source_proof_or_final_task_acceptance_established=False)


def collect(manifest):
    need(set(manifest) == {'schema', 'pair_lock', 'attempts'} and manifest['schema'] == 'compact-source-live-readout-input@1', 'closed readout manifest required')
    lock, lock_binding = read(manifest['pair_lock']['path'])
    need(lock_binding['sha256'] == manifest['pair_lock']['sha256'], 'external pair lock differs')
    need(manifest == input_from_lock(lock, lock_binding), 'readout attempt registration differs from actual lock')
    need(lock.get('frozen_before_launch') is True and set(lock.get('allowed_config_diff_paths', [])) == CONFIG_DIFFS, 'prelaunch control scope differs')
    pins = lock.get('host_source_sha256'); need(type(pins) is dict and 0 < len(pins) <= 4096, 'bounded source closure required')
    imported_root = Path(comparison.__file__).resolve().parents[3]
    for path, expected in pins.items():
        need(Path(path).is_relative_to(imported_root) and digest(expected) is not None, 'source pin outside reporting checkout')
        need(source_binding(path) == expected, 'frozen host source changed')
    initial_failures = []
    if lock.get('retained_prelaunch_failure') is not None:
        failure = preparation_failure(lock['retained_prelaunch_failure']['path'])
        need(failure['binding']['sha256'] == lock['retained_prelaunch_failure']['sha256'], 'retained preparation failure changed')
        initial_failures.append(failure)
    attempts = manifest['attempts']; need(type(attempts) is list and 2 <= len(attempts) <= 32, 'bounded registered pair required')
    need({row.get('label') for row in attempts} == set(PROFILES), 'both source profiles required')
    reference = refprep = None; rows = []; directories = set(); invocation_ids = set()
    for registered in attempts:
        need(set(registered) == {'label', 'directory', 'preparation_sha256', 'config_sha256'}, 'closed registered attempt required')
        label = registered['label']; base = Path(registered['directory']).absolute(); need(str(base) not in directories, 'duplicate attempt'); directories.add(str(base))
        prep, pb = read(base / 'preparation.json'); config, cb = read(base / 'config.json')
        need(pb['sha256'] == registered['preparation_sha256'] and cb['sha256'] == registered['config_sha256'] == prep.get('config_sha256'), 'frozen attempt input differs')
        need(prep.get('prepared') is True and prep.get('arm') == 'full' and prep.get('source_view_profile') == PROFILES[label] and prep.get('source_edit_mode') == 'mapped-span@2', 'prepared selection differs')
        need(type(config.get('agents')) is list and len(config['agents']) == 1 and config.get('n_attempts') == 1, 'one-attempt full supervisor configuration required')
        kwargs = obj(config['agents'][0].get('kwargs'))
        need(kwargs.get('arm') == 'full' and kwargs.get('source_view_profile') == PROFILES[label] and kwargs.get('source_edit_mode') == 'mapped-span@2' and kwargs.get('coding_reply_mode', 'legacy') == 'legacy', 'configured selection differs')
        need(controls.validate_controls(prep.get('comparison_controls')), 'prepared controls invalid')
        need(controls.build_controls(config, task_input_sha256=prep['task_input_sha256'], task=prep['task'], model=prep['model'], reasoning_effort=prep['reasoning_effort'], cli_version=prep['cli_version']) == prep['comparison_controls'], 'prepared controls differ from configuration')
        if reference is None:
            reference, refprep = config, prep
        changed = differences(reference, config)
        need(set(changed) <= CONFIG_DIFFS, 'pair differs beyond source profile and output directory')
        need(all(prep.get(k) == refprep.get(k) for k in ('task', 'task_input_sha256', 'archive_sha256', 'manifest_sha256', 'host_source_sha256', 'provider_profile', 'model', 'reasoning_effort', 'cli_version', 'resource_profile')), 'common preparation differs')
        receipt, rb = read(base / 'receipt.json', optional=True); launch, lb = read(base / 'launch-result.json', optional=True)
        invocation, ib = read(base / 'invocation.json', optional=True)
        started, sb = read(base / 'launch-start.json', optional=True)
        if any(item is not None for item in (receipt, launch, invocation)):
            need(started is not None, 'executed attempt lacks locked launch receipt')
        if started is not None:
            need(started.get('schema') == 'repeated-compact-live-launch@1' and started.get('label') == label
                 and started.get('pair_lock_sha256') == lock_binding['sha256']
                 and started.get('preparation_sha256') == pb['sha256']
                 and started.get('runner_sha256') == lock['launch_inputs']['runner']['sha256']
                 and started.get('recipe_sha256') == lock['launch_inputs']['recipe']['sha256']
                 and started.get('environment_manifest_sha256') == lock['launch_inputs']['environment']['sha256'], 'launch start differs from locked identities')
        attempt = dict(label=label, directory=str(base), preparation=pb, configuration=cb, receipt=rb, launch_result=lb, invocation=ib,
            configuration_difference_paths=changed,
            generic_control_comparison=controls.compare_controls(
                dict(status='observed', configuration_unchanged=True, declared=refprep['comparison_controls']),
                dict(status='observed', configuration_unchanged=True, declared=prep['comparison_controls'])),
            launch_start=sb, trials=[], status='awaiting_receipt' if receipt is None else 'observed',
            launch_seconds=number(obj(launch).get('seconds')), launch_exit_code=count(obj(launch).get('exit_code')),
            comparison_control_integrity=False)
        if receipt is not None:
            need(receipt.get('schema') == comparison.SUPERVISOR and receipt.get('arm') == 'full', 'supervisor receipt required')
            need(all(receipt.get(k) == prep.get(k) for k in ('task', 'model', 'provider_profile', 'reasoning_effort', 'cli_version')), 'receipt task/route differs')
            selected = obj(receipt.get('comparison_controls'))
            flags = {'source_view_config_unchanged', 'source_edit_config_unchanged'} | {key for key in receipt if key.endswith('_config_unchanged')}
            attempt['comparison_control_integrity'] = (selected.get('status') == 'observed' and selected.get('declared') == prep['comparison_controls']
                and selected.get('configuration_unchanged') is True and receipt.get('original_task_inputs_unchanged') is True and all(receipt.get(key) is True for key in flags))
            attempt['harbor_returncode'] = count(receipt.get('harbor_returncode'))
            trial_rows = receipt.get('trials'); need(type(trial_rows) is list and len(trial_rows) <= 16, 'bounded native trial list required')
            for raw in trial_rows:
                need(raw.get('exact_trial_task_matches') is True and raw.get('task') in (prep['task'], 'terminal-bench/' + prep['task']), 'exact native task binding required')
                report = obj(raw.get('supervisor')); row = comparison._trial(receipt, raw, source=rb)
                measured = usage(report)
                provider = obj(report.get('provider_profile'))
                if report:
                    need(provider.get('id') == prep['provider_profile'] and provider.get('model') == prep['model']
                         and provider.get('reasoning_effort') == prep['reasoning_effort'] and report.get('arm') == 'full', 'observed supervisor route differs')
                for call in measured['invocations']:
                    need(call['invocation_id'] not in invocation_ids, 'invocation reused across pair'); invocation_ids.add(call['invocation_id'])
                    need(call['model'] == prep['model'] and call['provider'] == provider.get('provider'), 'observed model/provider differs')
                trial = dict(trial=tag(raw.get('trial')), official_reward=number(row['official_reward']), outcome=tag(row['outcome']),
                    exception_type=tag(row['exception_type']), supervisor_error_type=tag(row['supervisor_error_type']),
                    native_task_completed=row['native_task_completed'], durations_seconds={k: number(v) for k, v in row['durations_seconds'].items()},
                    native_total_seconds=number(row['native_total_seconds']), supervisor_seconds=number(row['supervisor_seconds']),
                    supervisor_phases_seconds={k: number(v) for k, v in obj(row['supervisor_phases_seconds']).items()},
                    native_result_sha256=digest(raw.get('native_result_sha256')),
                    runtime_archive_sha256=digest(row.get('runtime_archive_sha256')),
                    runtime_matches_preparation=row.get('runtime_archive_sha256') == prep['archive_sha256'] if row.get('runtime_archive_sha256') is not None else None, usage=measured,
                    source_use=source_usage(report, measured['invocations'], PROFILES[label]))
                attempt['trials'].append(trial)
            if launch is not None and launch.get('receipt_sha256') is not None:
                need(launch['receipt_sha256'] == rb['sha256'], 'launch receipt differs')
        if launch is not None:
            need(launch.get('schema') == 'repeated-compact-live-launch-result@1' and launch.get('label') == label, 'launch result identity differs')
            attempt['comparison_control_integrity'] = attempt['comparison_control_integrity'] and all(launch.get(key) is True for key in ('preparation_unchanged', 'configuration_unchanged', 'host_source_unchanged'))
        else:
            attempt['comparison_control_integrity'] = False
        rows.append(attempt)
    aggregates = {}
    for label in PROFILES:
        selected = [r for r in rows if r['label'] == label]; trials = [t for r in selected for t in r['trials']]
        complete = all(r['status'] == 'observed' and r['trials'] for r in selected) and all(t['usage']['native_usage_complete_observed'] for t in trials)
        aggregates[label] = dict(registered_attempts=len(selected), retained_trials=len(trials),
            absent_trial_attempts=sum(not r['trials'] for r in selected), official_passes=sum(t['official_reward'] == 1 for t in trials),
            official_nonpasses=sum(t['official_reward'] is not None and t['official_reward'] != 1 for t in trials), unknown_rewards=sum(t['official_reward'] is None for t in trials),
            observed_router_sessions=sum(t['usage']['observed_router_sessions'] for t in trials),
            failed_or_incomplete_invocations=sum(t['usage']['failed_or_incomplete_invocation_count'] for t in trials),
            complete_native_usage_for_registered_attempts=complete,
            tokens={k: sum(t['usage']['tokens'][k] for t in trials) if complete and all(t['usage']['tokens'][k] is not None for t in trials) else None for k in TOKENS})
    matched = all(r['comparison_control_integrity'] and r['trials'] and all(t['runtime_matches_preparation'] is True for t in r['trials']) for r in rows)
    return dict(schema='compact-source-live-readout@1', pair_lock=lock_binding, attempts=rows, aggregates=aggregates,
        frozen_host_source_count=len(pins), frozen_host_source_pins_verified=True, configuration_preparation_failures=initial_failures,
        generic_comparator_scope='The existing comparator omits source_view_profile; its actual match result is retained alongside the full configuration difference audit.',
        configuration_comparison_scope='Full/full, only identity@1 versus python-comments@1 source display; both mapped-span@2 and legacy coding reply.',
        all_receipt_controls_match_preparation=matched,
        comments_minus_identity_tokens={k: aggregates['comments']['tokens'][k] - aggregates['identity']['tokens'][k]
            if matched and all(aggregates[label]['tokens'][k] is not None for label in PROFILES) else None for k in TOKENS},
        limitations=['The separately audited external lock owns prelaunch custody; this reader validates its hash, not launch authenticity.',
            'Totals cover all attempts explicitly registered here; no unregistered host run or unreported auxiliary provider usage is inferred.',
            'Supervisor phase times overlap enclosing agent times and must not be added again.',
            'Helper preparation/confirmation delivery is separate from source semantics, proof, final validation and official reward.',
            'Zero mapped-helper requests is no evidence of an efficiency advantage; native ordinary tool use remains unknown.',
            'No billing, causal advantage, source proof or total-token benefit is established by this readout.'],
        provider_calls_by_collector=0, raw_model_transcripts_read=False, hidden_verifier_or_solution_contents_read=False,
        billing_total_verified=False, causal_advantage_claimed=False)


def input_from_lock(lock, binding):
    need(lock.get('schema') == 'repeated-compact-live-pair-lock@1', 'unsupported prelaunch lock schema')
    attempts = lock.get('attempts'); need(type(attempts) is list and 2 <= len(attempts) <= 32, 'bounded locked attempts required')
    return dict(schema='compact-source-live-readout-input@1', pair_lock={k: binding[k] for k in ('path', 'sha256')},
        attempts=[dict(label=row['label'], directory=row['directory'],
            preparation_sha256=row['preparation']['sha256'], config_sha256=row['configuration']['sha256']) for row in attempts])


def preparation_failure(path):
    value, binding = read(path)
    need(count(value.get('returncode')) is not None and value['returncode'] != 0,
         'failed preparation result required')
    need(value.get('model_calls') == 0 and type(value.get('model_calls')) is int
         and value.get('provider_generation_calls') == 0 and type(value.get('provider_generation_calls')) is int,
         'preparation generation accounting missing')
    argv = value.get('argv')
    need(type(argv) is list and all(type(item) is str for item in argv)
         and 'prepare' in argv and 'execute' not in argv, 'prepare-only command required')
    return dict(binding=binding, label=tag(value.get('label')), provider_profile=tag(value.get('provider_profile')),
        returncode=value['returncode'], seconds=number(value.get('seconds')), reported_model_calls=0,
        reported_provider_generation_calls=0, scope='Retained configuration preparation failure; separate from executed native trials and token totals.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preparation-failure', type=Path, action='append', default=[])
    args = parser.parse_args(); manifest, binding = read(args.attempts)
    if manifest.get('schema') == 'repeated-compact-live-pair-lock@1':
        manifest = input_from_lock(manifest, binding)
    result = collect(manifest); result['registered_attempt_manifest'] = binding
    need(len(args.preparation_failure) <= 16, 'bounded preparation failures required')
    for path in args.preparation_failure:
        failure = preparation_failure(path)
        if failure['binding']['sha256'] not in {row['binding']['sha256'] for row in result['configuration_preparation_failures']}:
            result['configuration_preparation_failures'].append(failure)
    result['implementation_sha256'] = {str(path): sha(path.read_bytes()) for path in (Path(__file__).resolve(), Path(comparison.__file__).resolve(), Path(__import__(native_usage_contract.__module__, fromlist=['']).__file__).resolve())}
    with args.output.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'path': str(args.output), 'sha256': sha(args.output.read_bytes())}, sort_keys=True))


if __name__ == '__main__':
    main()
