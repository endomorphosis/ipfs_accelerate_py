"""Export closed controller/native metadata; never verifier or model bodies."""
import argparse
import hashlib
import json
import math
import pathlib
import re

R = pathlib.Path('/home/barberb/lift_coding/artifacts/terminal-expansion-20261005')
TASKS = ('tune-mjcf', 'largest-eigenval')
TRIAL_NAME = re.compile(r'grok-(tune-mjcf|largest-eigenval)-([0-9]{2})')
HEAD_PATHS = {
    'source': '/home/barberb/lift_coding/.worktrees/terminal-expansion-20261005',
    'datasets': '/home/barberb/lift_coding/.worktrees/ir-supervisor-contracts-datasets-20261004',
    'kit': '/home/barberb/lift_coding/.worktrees/terminal-supervisor-kit-20261005',
}
TOKENS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
          'output_tokens', 'reasoning_output_tokens', 'total_tokens')
BOOLS = frozenset('original_task_inputs_unchanged task_completed coding_dispatch_possible '
    'qualified initial_indexes_reused neural_inference_replayed proof_authority completion_authority '
    'all_invocations_receipted billing_total_verified cache_included_in_input observed_complete_sessions '
    'usage_complete_observed task_complete_observed envelope_observed error_observed raw_provider_data_exported '
    'exact_trial_task_matches effective_toolset_verified attempted'.split())
INTS = frozenset(('provider_calls', 'indexed_symbols', 'full_capsules', 'worker_capsules',
    'worker_semantic_bytes', 'training_steps', 'source_files', 'program_source_files',
    'inference_python_files', 'harness_support_files', 'goals', 'tasks', 'remaining_processes',
    'worker_cleanup_returncode', 'max_turns', 'http_status', 'request_bytes', 'response_bytes',
    'latency_ms', 'timeout_ms'))
FLOATS = frozenset(('seconds', 'invocation_seconds', 'timeout_seconds', 'dollar_cost', 'cost_usd'))
HASHES = frozenset(('native_result_sha256', 'checkpoint_sha256', 'inference_sha256'))
ENUMS = {
    'task': set(TASKS), 'arm': {'full', 'no-index'}, 'model': {'grok-4.7'},
    'cli_version': {'1.0.46'}, 'provider_profile': {'grok-4.7-cli-1.0.46@1'},
    'provider': {'grok_cli'}, 'reasoning_effort': {'none', 'low', 'medium', 'high', 'xhigh'},
    'planning_provider': {'grok_cli', 'disabled'}, 'planning_model': {'grok-4.7', 'none'},
    'planning_strategy': {'direct', 'intent_symbolic', 'intent_coverage'},
    # PromptGoalProviderReceipt differs from native-process outcome metadata.
    'planner_provider_status': {'over_budget', 'disabled', 'unavailable', 'succeeded',
                                'malformed', 'timeout', 'failed'},
    'planner_provider_reason_code': {'request_over_budget', 'policy_disabled',
        'capability_unavailable', 'provider_graph_accepted', 'deterministic_baseline_selected',
        'response_over_budget', 'graph_over_budget', 'output_too_large', 'timeout', 'unavailable',
        'malformed', 'failed', 'invalid_path', 'invalid_schema', 'forbidden_instruction',
        'duplicate_value', 'unknown_or_missing_field', 'invalid_request_context', 'identity_mismatch',
        'prose_wrapper', 'duplicate_key', 'protected_path', 'orphan_reference',
        'unsupported_capability', 'cycle', 'orphan_node', 'missing_validation',
        'unsupported_resource', 'invalid_graph'},
    # OperationResult.status is separate from the daemon lifecycle state.
    'lifecycle_operation_status': {'succeeded', 'failed', 'denied', 'conflict', 'not_found',
                                   'cancelled', 'timed_out', 'unavailable'},
    'resource_profile': {'source384-5cpu-12gib@1', 'source384-5cpu-16gib-extended@1',
                         'source384-5cpu-16gib-planner180@1'},
    'implementation_route': {'model_router', 'doctor_candidate', 'doctor_contract_candidate'},
    'route': {'model_router', 'doctor_candidate', 'doctor_contract_candidate'},
    'phase': {'planning', 'coding', 'provider_invocation', 'semantic_response_decode'},
    'purpose': {'planning', 'coding'},
    'error_phase': {'prepare', 'initial_context', 'planning', 'context', 'admission', 'doctor',
                    'implementation_setup', 'native_execution'},
    'status': {'succeeded', 'failed', 'provider_returned', 'residual', 'candidate_ready', 'unavailable'},
    'analysis_status': {'supported', 'unsupported', 'residual', 'candidate_ready', 'no_findings'},
    'permission_mode': {'dontAsk', 'bypassPermissions'}, 'tools_profile': {'none', 'isolated_coding'},
    'process_outcome': {'returned', 'failed', 'timeout', 'unknown'},
    'stop_reason': {'end_turn', 'stop', 'max_tokens', 'max_turn_requests', 'refusal', 'cancelled',
                    'tool_use', 'pause_turn', 'stop_sequence', 'unknown'},
    'error_subtype': {'error_max_turns', 'error_during_execution',
                      'error_max_structured_output_retries', 'unknown'},
    'reason_code': {'end_turn', 'max_turns', 'max_tokens', 'refusal', 'cancelled',
        'structured_output_retries', 'execution_error', 'process_error', 'timeout', 'other_stop',
        'unknown', 'provider_error', 'rate_limit', 'authentication', 'unavailable'},
    'classification_source': {'stop_reason', 'error_subtype', 'native_type', 'message_marker',
                               'process', 'none', 'unknown'},
    'source': {'observed_native_final_totals', 'observed_native_cumulative_totals'},
    'totals_scope': {'native_final_envelope'},
    'schema': {'native-grok-final-usage@1', 'native-grok-outcome@1'},
}
EXCEPTIONS = {'RuntimeError', 'ValueError', 'TimeoutError', 'ProviderInvocationError',
              'CalledProcessError', 'AgentTimeoutError', 'AgentSetupTimeoutError',
              'PromptGoalPlannerError'}
TOOLS = {'read_file', 'search_replace', 'grep', 'list_dir', 'todo_write', 'run_terminal_cmd',
         'search_tool', 'use_tool', '*'}


def read_metadata(root, path, *, max_bytes=4_194_304):
    path = pathlib.Path(path)
    if (path.resolve(strict=True) != path.absolute() or path.is_symlink()
            or not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > max_bytes):
        raise ValueError('bounded canonical artifact metadata required')
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError('metadata changed beyond byte bound')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate metadata field')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
    if type(value) is not dict:
        raise ValueError('metadata object required')
    return value, dict(path=path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def scalar(key, value):
    if value is None:
        return None
    if key in BOOLS:
        return value if type(value) is bool else None
    if key in INTS or key in TOKENS or key in {'n_input_tokens', 'n_cache_tokens', 'n_output_tokens'}:
        return value if type(value) is int and 0 <= value < 2**63 else None
    if key in FLOATS:
        return value if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e9 else None
    if key in HASHES:
        return value if type(value) is str and re.fullmatch('[0-9a-f]{64}', value) else None
    if key in ('error_type', 'exception_type'):
        return value if type(value) is str and value in EXCEPTIONS else 'unknown'
    allowed = ENUMS.get(key)
    if allowed is None:
        raise ValueError('unreviewed export field: ' + key)
    return value if type(value) is str and value in allowed else 'unknown'


def pick(value, keys):
    value = value if type(value) is dict else {}
    return {key: scalar(key, value.get(key)) for key in keys.split()}


def usage(value, *, native=False):
    if value is None:
        return None
    value = value if type(value) is dict else {}
    keys = ('schema task_complete_observed usage_complete_observed totals_scope billing_total_verified '
            'cache_included_in_input') if native else (
            'source all_invocations_receipted observed_complete_sessions usage_complete_observed '
            'billing_total_verified cache_included_in_input provider_calls dollar_cost')
    result = pick(value, keys)
    counters = value.get('usage') if native else value
    checked = pick(counters, ' '.join(TOKENS))
    return {**result, **({'usage': checked} if native else checked)}


def selection(root, receipt, attempt_dir):
    prepared, reference = read_metadata(root, attempt_dir / 'preparation.json')
    fields = 'task arm model cli_version provider_profile reasoning_effort resource_profile'.split()
    selected, origins = {}, {}
    for key in fields:
        values = [(source, obj[key]) for source, obj in [('receipt', receipt), ('preparation', prepared)]
                  if key in obj]
        if not values or any(value != values[0][1] for _, value in values):
            raise ValueError('missing or inconsistent recorded selection: ' + key)
        selected[key] = scalar(key, values[0][1])
        if selected[key] == 'unknown':
            raise ValueError('unreviewed recorded selection: ' + key)
        origins[key] = '+'.join(source for source, _ in values)
    archive = pathlib.Path(prepared.get('archive', ''))
    manifest, manifest_ref = read_metadata(root, archive / 'manifest.json', max_bytes=8_388_608)
    if (manifest_ref['sha256'] != prepared.get('manifest_sha256')
            or manifest.get('archive_sha256') != prepared.get('archive_sha256')):
        raise ValueError('prepared runtime archive binding differs')
    revisions, source_evidence = None, None
    for path in sorted((root / 'grok-container').glob('archive-review*.json')):
        review, review_ref = read_metadata(root, path)
        if (review.get('qualified') is not True or review.get('manifest_sha256') != manifest_ref['sha256']
                or review.get('archive_sha256_declared') != prepared['archive_sha256']):
            continue
        heads = review.get('source_heads') or {}
        candidate = {label: heads.get(name) for label, name in HEAD_PATHS.items()}
        if any(type(head) is not str or re.fullmatch('[0-9a-f]{40}', head) is None
               for head in candidate.values()):
            continue
        if revisions is not None and candidate != revisions:
            raise ValueError('conflicting archive-bound source revisions')
        revisions, source_evidence = candidate, review_ref
    return selected, dict(preparation=reference, runtime_manifest=manifest_ref,
        archive_sha256=prepared['archive_sha256'], selection_origins=origins,
        source_revisions=revisions, source_revision_evidence=source_evidence,
        source_revision_basis='qualified_review_matching_prepared_archive_digests' if revisions else 'unknown',
        archive_content_hash_recomputed_by_this_export=False)


def outcome(value):
    return None if value is None else pick(value,
        'schema envelope_observed process_outcome stop_reason error_subtype error_observed '
        'reason_code classification_source raw_provider_data_exported completion_authority')


def invocation(value):
    if type(value) is not dict:
        raise ValueError('bounded invocation object required')
    policy = value.get('provider_invocation_policy')
    if policy is not None:
        tool_fields = {}
        for field in ('disallowed_tools', 'native_tool_allowlist', 'native_tool_denylist'):
            values = policy.get(field) if type(policy) is dict else None
            tool_fields[field] = (values if type(values) is list and len(values) <= 16
                and all(type(tool) is str and tool in TOOLS for tool in values) else None)
        policy = dict(pick(policy, 'max_turns tools_profile permission_mode effective_toolset_verified'), **tool_fields)
    return dict(pick(value, 'provider model purpose phase status seconds error_type timeout_seconds'),
        provider_failure=pick(value.get('provider_failure'), 'phase reason_code http_status'),
        provider_invocation_policy=policy, native_provider_outcome=outcome(value.get('native_provider_outcome')),
        native_usage=usage(value.get('native_rollout_usage'), native=True),
        requested_policy_is_actual_tool_observation=False)


def planning(value):
    value = value if type(value) is dict else {}
    provider = value.get('provider_receipt')
    provider = provider if type(provider) is dict else None
    # Failures contain PromptGoalProviderReceipt directly; success contains
    # PromptGoalPlanningReceipt with the provider receipt in its provider field.
    shape = 'flat_provider_receipt' if provider is not None else None
    if provider is not None and type(provider.get('provider')) is dict:
        provider = provider['provider']
        shape = 'planning_receipt.provider'
    projected = None if provider is None else {
        **pick(provider, 'attempted request_bytes response_bytes latency_ms timeout_ms'),
        'status': scalar('planner_provider_status', provider.get('status')),
        'reason_code': scalar('planner_provider_reason_code', provider.get('reason_code'))}
    failure = value.get('failure')
    failure = failure if type(failure) is dict else {}
    return {**pick(value, 'qualified provider_calls goals tasks planning_strategy'),
        'provider': scalar('planning_provider', value.get('provider')),
        'model': scalar('planning_model', value.get('model')),
        'provider_receipt': projected, 'provider_receipt_shape': shape,
        'failure_exception_type': scalar('exception_type', failure.get('type'))}


def collect(root):
    root = pathlib.Path(root).resolve(strict=True)
    rows = []
    paths = sorted(path for task in TASKS for path in root.glob('grok-' + task + '-*/receipt.json'))
    if len(paths) > 198:
        raise ValueError('trial population exceeds bound')
    for path in paths:
        match = TRIAL_NAME.fullmatch(path.parent.name)
        if match is None:
            raise ValueError('unreviewed trial directory')
        receipt, reference = read_metadata(root, path)
        trials = receipt.get('trials')
        if type(trials) is not list or len(trials) != 1 or type(trials[0]) is not dict:
            raise ValueError('exactly one native trial per attempt required')
        trial = trials[0]
        supervisor = trial.get('supervisor') or {}
        context = trial.get('agent_context') or {}
        metadata = context.get('metadata') or {}
        initial = supervisor.get('initial_context') or {}
        source = initial.get('source384_context') or {}
        calls = supervisor.get('provider_invocations') or []
        if type(calls) is not list or len(calls) > 32:
            raise ValueError('invocation population exceeds bound')
        selected, bindings = selection(root, receipt, path.parent)
        if selected['task'] != match[1]:
            raise ValueError('trial directory differs from recorded task selection')
        reward = trial.get('reward')
        reward = reward.get('reward') if type(reward) is dict and set(reward) == {'reward'} else reward
        reward = reward if type(reward) in (int, float) and math.isfinite(reward) and 0 <= reward <= 1 else None
        durations = trial.get('durations_seconds') or {}
        reasons = (supervisor.get('doctor_dispatch') or {}).get('reason_codes')
        reason_count = len(reasons) if type(reasons) is list and len(reasons) <= 256 else None
        rows.append(dict(trial_name=path.parent.name, task=match[1], attempt=match[2],
            receipt=reference, selection=selected, selection_binding=bindings,
            trial={**pick(trial, 'exception_type exact_trial_task_matches native_result_sha256'), 'reward': reward,
                'durations_seconds': {key: scalar('seconds', durations.get(key)) for key in
                    ('agent_execution', 'agent_setup', 'environment_setup', 'verifier')}},
            supervisor=pick(supervisor, 'task_completed seconds remaining_processes worker_cleanup_returncode '
                'implementation_route coding_dispatch_possible error_phase'),
            lifecycle={key + '_status': scalar('lifecycle_operation_status', (supervisor.get(key) or {}).get('status'))
                for key in ('start', 'stop')},
            planning=planning(supervisor.get('planning')),
            initial_context=pick(initial, 'indexed_symbols full_capsules provider_calls seconds'),
            admitted_context=pick(supervisor.get('context'), 'indexed_symbols full_capsules worker_capsules '
                'worker_semantic_bytes initial_indexes_reused'),
            doctor={**pick(supervisor.get('doctor_dispatch'), 'status route analysis_status provider_calls'),
                'unexported_reason_code_count': reason_count},
            source384={**pick(source, 'checkpoint_sha256 inference_sha256 neural_inference_replayed training_steps seconds'),
                'summary': pick(source.get('summary'), 'source_files program_source_files inference_python_files '
                    'harness_support_files provider_calls proof_authority completion_authority')},
            native_usage=usage(metadata.get('usage')),
            harbor_usage=pick(context, 'n_input_tokens n_cache_tokens n_output_tokens cost_usd'),
            router_invocations=[invocation(call) for call in calls],
            benchmark_advantage_claimed=False, billing_total_verified=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=pathlib.Path, default=R)
    parser.add_argument('--output', default='live-trial-summary.json')
    parser.add_argument('--require-attempt', action='append', default=[])
    parser.add_argument('--require-trial-name', action='append', default=[])
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    output = root / args.output
    if (pathlib.Path(args.output).is_absolute() or '..' in pathlib.Path(args.output).parts
            or output.resolve() != output.absolute() or not output.is_relative_to(root)):
        raise ValueError('canonical new artifact output required')
    rows = collect(root)
    for attempt in args.require_attempt:
        if sum(row['attempt'] == attempt for row in rows) != 1:
            raise ValueError('required attempt is absent or ambiguous; use --require-trial-name')
    if (any(TRIAL_NAME.fullmatch(name) is None for name in args.require_trial_name)
            or not set(args.require_trial_name) <= {row['trial_name'] for row in rows}):
        raise ValueError('required exact completed trial receipt is absent')
    with output.open('x') as stream:
        json.dump(dict(schema='terminal-expansion-live-summary@2', trials=rows,
            raw_provider_verifier_or_credential_bodies_exported=False,
            native_usage_completeness_inferred=False), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([dict(trial_name=row['trial_name'], reward=row['trial']['reward'],
        task_completed=row['supervisor']['task_completed']) for row in rows]))


if __name__ == '__main__':
    main()
