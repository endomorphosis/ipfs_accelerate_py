"""Read retained pilot artifacts only; no runtime imports, credentials or providers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from safe_diagnostics import failure_observations, grok_progress

A = Path(__file__).resolve().parent
PRIOR = Path('/home/barberb/lift_coding/artifacts/doctor-turns8-live-20261014/live-trials/grok-two-attempts-turns8-01')
CATEGORIES = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens', 'total_tokens')
REASONING = 'reasoning_output_tokens'
CONTROL_DEFAULTS = dict(candidate_rounds=1, doctor_coding_policy=None, source_view_profile='disabled',
    source_edit_mode='disabled', numerical_formalization_profile=None, semantic_metadata_view='legacy',
    semantic_working_set='legacy', capsule_budget='legacy', coding_reply_mode='legacy',
    grok_coding_max_turns=None, resource_profile=None, provider_profile=None,
    semantic_transport_schema='supervisor-semantic-router-input@1')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def count(value):
    return value if type(value) is int and value >= 0 else None


def tag(value):
    return value if type(value) is str and re.fullmatch(r'[A-Za-z0-9_.:@/+-]{1,160}', value) else None


def hex_id(value, size):
    return type(value) is str and re.fullmatch('[0-9a-f]{' + str(size) + '}', value) is not None


def fields(value, names):
    value = value if type(value) is dict else {}
    return {key: item if type(item) in (bool, int, float) or item is None else tag(item)
            for key in names if key in value for item in [value[key]]}


def read(path, bindings):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError('invalid_retained_artifact')
    raw = path.read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_json_field')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite_json')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError('object_artifact_required')
    bindings.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return value


def distinct_receipts(report):
    rows = report.get('provider_invocations', [])
    if type(rows) is not list or len(rows) > 128:
        raise ValueError('invalid_receipt_set')
    unique = {}
    for row in rows:
        identity = row.get('invocation_id') if type(row) is dict else None
        if type(identity) is not str or not re.fullmatch(r'[0-9a-f]{32}', identity):
            raise ValueError('invalid_invocation_identity')
        if identity in unique and canonical(unique[identity]) != canonical(row):
            raise ValueError('conflicting_invocation_receipts')
        unique[identity] = row
    return list(unique.values()), len(rows) - len(unique)


def native(row):
    raw = row.get('native_rollout_usage')
    raw = raw if type(raw) is dict else {}
    usage = raw.get('usage')
    usage = usage if type(usage) is dict else {}
    counters = {key: count(usage.get(key)) for key in (*CATEGORIES, REASONING)}
    grok = row.get('provider') == 'grok_cli' and raw.get('schema') == 'native-grok-final-usage@1'
    codex = row.get('provider') == 'codex_cli' and raw.get('schema') == 'native-codex-cumulative-usage@1'
    valid = (grok and raw.get('usage_available') is True and raw.get('totals_scope') == 'native_final_envelope'
        and raw.get('cache_included_in_input') is False
        and all(counters[k] is not None for k in CATEGORIES)
        and sum(counters[k] for k in CATEGORIES[:-1]) == counters['total_tokens'])
    complete = None
    flags = [raw.get('task_complete_observed'), raw.get('usage_complete_observed')]
    if valid:
        complete = False if any(v is False for v in flags) else True if all(v is True for v in flags) else None
    elif codex and raw.get('cache_included_in_input', True) is True:
        complete = raw.get('task_complete_observed') if type(raw.get('task_complete_observed')) is bool else None
    return dict(counters=counters, usage_complete_under_native_contract=complete,
        arithmetic_valid=bool(valid) if grok else None,
        retained_native_fields=fields(raw, ('schema', 'usage_available', 'usage_complete_observed',
            'task_complete_observed', 'totals_scope', 'cache_included_in_input', 'billing_total_verified',
            'observed_final_usage_records', 'usage_observation_sha256')))


def aggregate(rows, *, unreceipted=False, aggregate_agrees=False):
    observations = [native(row) for row in rows]
    keys = (*CATEGORIES, REASONING)
    missing = {key: sum(o['counters'][key] is None for o in observations) for key in keys}
    subtotal = {key: sum(values) if values else None for key in keys
        for values in [[o['counters'][key] for o in observations if o['counters'][key] is not None]]}
    native_complete = bool(rows) and not unreceipted and all(
        o['usage_complete_under_native_contract'] is True for o in observations)
    complete = native_complete and aggregate_agrees is True
    return dict(receipts=len(rows), outer_router_calls=(sum(row['router_calls'] for row in rows)
        if rows and all(count(row.get('router_calls')) is not None for row in rows) else None),
        observed_subtotals=subtotal, missing_categories=missing,
        complete_native_totals={key: subtotal[key] if complete and missing[key] == 0 else None for key in keys},
        all_native_contracts_complete=native_complete, retained_aggregate_completeness_agrees=aggregate_agrees is True,
        unreceipted_attempt_reported=bool(unreceipted),
        billing_total_verified=False, reasoning_is_separate_not_added_to_total=True)


def audit(value):
    value = value if type(value) is dict else {}
    result = fields(value, ('schema', 'status', 'reason', 'all_observed_coding_inputs_verified',
        'coding_receipts_seen', 'usable_coding_receipts', 'session_scope', 'all_coding_sessions_observed',
        'candidate_sessions_excluded', 'provider_tokens', 'total_token_savings_measured'))
    for name in ('counts', 'preflight_counts', 'candidate_context_counts'):
        if type(value.get(name)) is dict:
            result[name] = {key: count(item) for key, item in value[name].items() if tag(key)}
    result['prior_receipt_matches_reported'] = [row.get('prior_router_receipt_matched') is True
        for row in value.get('sessions', []) if type(row) is dict]
    return result


def continuation(report, rows):
    coding = [r for r in rows if r.get('phase') == 'coding']
    second = [r for r in coding if type(r.get('candidate_round_dispatch')) is dict]
    accounting = report.get('candidate_attempt_accounting')
    accounting = accounting if type(accounting) is dict else {}
    kept = fields(accounting, ('schema', 'status', 'reason', 'potential_unreceipted_attempt',
        'complete_usage_established', 'doctor_coding_policy', 'doctor_coding_policy_sha256',
        'owner_lifecycle_sha256', 'total_tokens_bounded', 'total_cost_bounded', 'provider_internal_calls_bounded'))
    reserved = accounting.get('doctor_attempts_reserved')
    kept['doctor_attempts_reserved'] = reserved if type(reserved) is list and all(type(n) is int for n in reserved) and reserved in ([], [1], [1, 2]) else None
    kept['rounds'] = [fields(row, ('round_number', 'process_started', 'receipt_observed', 'possible_unreceipted_attempt'))
        for row in accounting.get('rounds', []) if type(row) is dict]
    joins = False
    if len(second) == 1:
        row = second[0]; d = row['candidate_round_dispatch']
        first = [r for r in coding if type(r.get('source_view_retrieval')) is dict
            and r['source_view_retrieval'].get('session_id') == d.get('prior_session_id')]
        doctor = d.get('doctor_candidate_dispatch') or {}
        if len(first) == 1 and type(doctor) is dict:
            prior = first[0]['source_view_retrieval']
            original = first[0].get('doctor_residual_context') or {}
            joins = (len(coding) == 2 and d.get('schema') == 'supervisor-candidate-dispatch-observation@2'
                and type(d.get('round_number')) is int and d['round_number'] == 2
                and hex_id(d.get('session_id'), 32) and hex_id(d.get('prior_session_id'), 32)
                and d.get('session_id') != d.get('prior_session_id')
                and hex_id(d.get('prior_descriptor_sha256'), 64)
                and d.get('prior_descriptor_sha256') == prior.get('descriptor_sha256')
                and all(tag(d.get(k)) is not None and d[k] == prior.get(k) for k in ('task_id', 'native_prompt_sha256'))
                and doctor.get('coding_policy_sha256') == accounting.get('doctor_coding_policy_sha256')
                and hex_id(doctor.get('coding_policy_sha256'), 64)
                and doctor.get('policy_id') == accounting.get('doctor_coding_policy') == 'doctor-coding-two-attempts@1'
                and type(original) is dict
                and all(hex_id(doctor.get('doctor_' + k), 64) and doctor['doctor_' + k] == original.get(k)
                    for k in ('artifact_sha256', 'advisory_sha256'))
                and type(row.get('router_calls')) is int and row['router_calls'] == 1
                and type(first[0].get('router_calls')) is int and first[0]['router_calls'] == 1)
    reported = (accounting.get('schema') == 'terminal-candidate-attempt-accounting@1'
        and accounting.get('status') == 'matched_round_receipts'
        and accounting.get('potential_unreceipted_attempt') is False
        and kept['doctor_attempts_reserved'] == [1, 2] and len(kept['rounds']) == 2
        and all(type(r.get('round_number')) is int and r['round_number'] == i and r.get('process_started') is True and r.get('receipt_observed') is True
            for i, r in enumerate(kept['rounds'], 1)))
    return dict(candidate_receipts=len(second), two_coding_router_calls_joined=bool(joins),
        in_container_accounting_agrees=reported, continuation_supported_by_retained_report=bool(joins and reported),
        independent_owner_replay_performed=False, native_accounting=kept)


def summarize_report(report, aggregate_usage=None):
    rows, duplicates = distinct_receipts(report)
    unreceipted = report.get('unreceipted_provider_attempt') not in (None, False)
    accounting = report.get('candidate_attempt_accounting')
    if type(accounting) is dict and accounting.get('potential_unreceipted_attempt') is True:
        unreceipted = True
    aggregate_usage = aggregate_usage if type(aggregate_usage) is dict else {}
    observed = aggregate(rows, unreceipted=unreceipted)
    aggregate_fields = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')
    if any(r.get('provider') == 'grok_cli' for r in rows):
        aggregate_fields += ('cache_write_input_tokens',)
    aggregate_agrees = (bool(rows) and aggregate_usage.get('native_scoped_usage_complete_observed') is True
        and aggregate_usage.get('all_invocations_receipted') is True
        and all(count(aggregate_usage.get(k)) is not None and observed['missing_categories'][k] == 0
            and aggregate_usage[k] == observed['observed_subtotals'][k] for k in aggregate_fields))
    invocations = []
    for row in rows:
        item = fields(row, ('invocation_id', 'phase', 'provider', 'model', 'status', 'router_calls',
            'failure_phase', 'timeout_seconds', 'requested_output_tokens', 'native_prompt_bytes',
            'prompt_bytes', 'router_prompt_bytes', 'model_prompt_bytes'))
        item['native_grok_progress'] = grok_progress(row.get('native_grok_progress'))
        item.update(native_usage=native(row), native_cli=fields(row.get('usage'), ('num_turns', 'stop_reason', 'exit_code')))
        item['candidate_preflight'] = fields(row.get('candidate_round_preflight'),
            ('schema', 'requests', 'artifact', 'response_bytes', 'provider_input_bytes', 'child_custody_checked',
             'source_freshness_checked', 'producer_freshness_checked'))
        item['input_representation'] = {key: fields(row[key], ('schema', 'selected', 'selected_mode',
            'selection_basis', 'native_complete_bytes', 'candidate_complete_bytes', 'selected_complete_bytes',
            'native_complete_tokens', 'candidate_complete_tokens', 'selected_complete_tokens', 'token_counter_status',
            'complete_run_token_savings_claimed')) for key in ('semantic_metadata_view', 'semantic_working_set') if key in row}
        invocations.append(item)
    phases = {phase: aggregate([r for r in rows if r.get('phase') == phase], unreceipted=unreceipted,
        aggregate_agrees=aggregate_agrees)
        for phase in ('planning', 'coding', 'review') if any(r.get('phase') == phase for r in rows)}
    return dict(status='retained_report_read', duplicate_receipts_ignored=duplicates,
        native_failure_observations=failure_observations(report.get('native_failure_observations')),
        invocations=invocations, usage_by_phase=phases,
        all_phase_usage=aggregate(rows, unreceipted=unreceipted, aggregate_agrees=aggregate_agrees),
        retained_aggregate_usage=fields(aggregate_usage, (*CATEGORIES, 'native_scoped_usage_complete_observed',
            'usage_complete_observed', 'observed_complete_sessions', 'all_invocations_receipted',
            'harbor_usage_exportable', 'token_count_scope', 'cache_included_in_input', 'billing_total_verified')),
        continuation=continuation(report, rows), task_completed_reported=report.get('task_completed') is True,
        seconds=count(report.get('seconds')) if type(report.get('seconds')) is int else report.get('seconds') if type(report.get('seconds')) is float else None,
        phase_seconds={key: value for key in ('prepare_seconds', 'initial_context_seconds', 'planning_seconds',
            'context_seconds', 'metadata_census_seconds', 'doctor_seconds', 'post_publication_refresh_seconds')
            for value in [(report.get('phases') or {}).get(key)]
            if type(value) in (int, float) and math.isfinite(value) and value >= 0},
        input_audit=audit(report.get('context_input_audit')),
        baseline_source_audit=audit(report.get('source_view_retrieval_observation')),
        candidate_source_audit=audit(report.get('candidate_round_retrieval_observation')))


def read_run(directory):
    directory = Path(directory); bindings = []
    config = read(directory / 'config.json', bindings) or {}
    agents = config.get('agents') or []
    kwargs = agents[0].get('kwargs', {}) if len(agents) == 1 and type(agents[0]) is dict else {}
    controls = {key: kwargs.get(key, default) for key, default in CONTROL_DEFAULTS.items()}
    controls = fields(controls, controls)
    parents = sorted({p.parent for p in directory.glob('jobs/supervisor-full-largest-eigenval/largest-eigenval__*/result.json')}
        | {p.parent.parent for p in directory.glob('jobs/supervisor-full-largest-eigenval/largest-eigenval__*/agent/supervisor-result.json')})
    trials = []
    for parent in parents:
        result = read(parent / 'result.json', bindings) or {}
        report = read(parent / 'agent/supervisor-result.json', bindings)
        reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        row = dict(trial=parent.name, official_reward=reward if type(reward) in (int, float) else None,
            verifier_result_retained='verifier_result' in result,
            status='report_unavailable' if report is None else 'retained_report_read')
        if report is not None:
            try:
                usage = ((result.get('agent_result') or {}).get('metadata') or {}).get('usage')
                row.update(summarize_report(report, usage))
            except (ValueError, TypeError, KeyError) as error:
                row.update(status='invalid_receipt_set', error_code=tag(str(error)) or type(error).__name__)
        trials.append(row)
    return dict(directory=str(directory), status='observed' if trials else 'no_retained_trial',
        controls=controls, trials=trials, artifact_bindings=bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=A / 'recipe.json')
    parser.add_argument('--include-prior', action='store_true')
    args = parser.parse_args(); bindings = []
    recipe = read(args.recipe, bindings)
    if recipe is None or recipe.get('schema') != 'doctor-dispatch-live-pilot@1':
        raise ValueError('known_pilot_recipe_required')
    pilot = read_run(recipe['output']); prior = read_run(PRIOR) if args.include_prior else None
    differences = ({key: dict(pilot=pilot['controls'][key], prior=prior['controls'][key])
        for key in CONTROL_DEFAULTS if pilot['controls'][key] != prior['controls'][key]} if prior else None)
    print(json.dumps(dict(schema='doctor-dispatch-pilot-readout@1', pilot=pilot, prior=prior,
        differing_configuration_controls=differences, recipe_bindings=bindings,
        runtime_comparison='fresh_follow_up_after_code_changes_not_matched_runtime',
        reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        causal_savings_claimed=False, total_token_savings_established=False, provider_calls_by_reader=0,
        raw_prompts_or_provider_text_exported=False, owner_private_audits_exported=False,
        independent_host_replay_performed=False,
        limitations=['Native in-container audit outcomes are retained reports, not independently replayed owner artifacts.',
            'Observed subtotals include known counters; missing categories and native completeness remain unknown.',
            'Reasoning output is reported separately and is not added again to native totals.',
            'Input and retrieval bytes are not provider tokens or evidence of causal run savings.']),
        indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
