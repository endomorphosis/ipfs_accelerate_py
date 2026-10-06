"""Export closed controller/native metadata; never verifier or model bodies."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

R = pathlib.Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
TASKS = ('tune-mjcf', 'largest-eigenval')
TRIAL_NAME = re.compile(r'grok-(tune-mjcf|largest-eigenval)-([0-9]{2})')
HEAD_PATHS = {
    'source': '/home/barberb/lift_coding/.worktrees/grok-recovery-20261006',
    'datasets': '/home/barberb/lift_coding/.worktrees/terminal-bounded-header-datasets-20261005',
    'kit': '/home/barberb/lift_coding/.worktrees/terminal-supervisor-kit-20261005',
}
TOKENS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
          'output_tokens', 'reasoning_output_tokens', 'total_tokens')
BOOLS = frozenset('original_task_inputs_unchanged task_completed coding_dispatch_possible '
    'qualified initial_indexes_reused neural_inference_replayed proof_authority completion_authority '
    'all_invocations_receipted billing_total_verified cache_included_in_input observed_complete_sessions '
    'usage_complete_observed task_complete_observed envelope_observed error_observed raw_provider_data_exported '
    'exact_trial_task_matches effective_toolset_verified attempted native_schema_requested '
    'response_schema_validated plan_admitted runtime_close_attempted runtime_close_succeeded '
    'observations_truncated complete_single_trial_receipt native_job_result_present '
    'top_level_id_omitted canonical_validation_preserved task_contract_authority'.split())
INTS = frozenset(('provider_calls', 'indexed_symbols', 'full_capsules', 'worker_capsules',
    'worker_semantic_bytes', 'training_steps', 'source_files', 'program_source_files',
    'inference_python_files', 'harness_support_files', 'goals', 'tasks', 'remaining_processes',
    'worker_cleanup_returncode', 'max_turns', 'http_status', 'request_bytes', 'response_bytes',
    'latency_ms', 'timeout_ms', 'response_schema_bytes', 'revision', 'subgoals',
    'bootstrap_receipt_count', 'bootstrap_error_count', 'start_timeout_ms', 'stop_timeout_ms', 'count',
    'canonical_schema_bytes', 'native_wire_schema_bytes', 'task_contract_bytes', 'task_count'))
FLOATS = frozenset(('seconds', 'invocation_seconds', 'timeout_seconds', 'dollar_cost', 'cost_usd',
    'provider_coding_timeout_cap_seconds', 'provider_coding_timeout_seconds'))
HASHES = frozenset(('native_result_sha256', 'checkpoint_sha256', 'inference_sha256', 'response_schema_sha256',
    'canonical_schema_sha256', 'native_wire_schema_sha256', 'task_contract_sha256'))
ENUMS = {
    'projection_id': {'canonical-prompt-goal-native-id-omission@1', 'identity@1', 'canonical-prompt-goal-task-contract@1'},
    'schema_error_code': {'schema_id_invalid'},
    'schema_error_path': {'/$id'},
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
                         'source384-5cpu-16gib-planner180@1', 'source384-5cpu-20gib-planner180@1',
                         'source384-5cpu-20gib-coding600@1'},
    'implementation_route': {'model_router', 'doctor_candidate', 'doctor_contract_candidate'},
    'route': {'model_router', 'doctor_candidate', 'doctor_contract_candidate'},
    'phase': {'planning', 'coding', 'provider_invocation', 'semantic_response_decode', 'provider_result_validation'},
    'purpose': {'planning', 'coding'},
    'error_phase': {'prepare', 'initial_context', 'planning', 'context', 'admission', 'doctor',
                    'implementation_setup', 'native_execution'},
    'status': {'succeeded', 'failed', 'provider_returned', 'residual', 'candidate_ready', 'unavailable'},
    'analysis_status': {'supported', 'unsupported', 'residual', 'candidate_ready', 'no_findings', 'available', 'unavailable'},
    'permission_mode': {'dontAsk', 'bypassPermissions'}, 'tools_profile': {'none', 'isolated_coding'},
    'process_outcome': {'returned', 'failed', 'timeout', 'unknown'},
    'stop_reason': {'end_turn', 'stop', 'max_tokens', 'max_turn_requests', 'refusal', 'cancelled',
                    'tool_use', 'pause_turn', 'stop_sequence', 'unknown'},
    'error_subtype': {'error_max_turns', 'error_during_execution',
                      'error_max_structured_output_retries', 'unknown'},
    'reason_code': {'end_turn', 'max_turns', 'max_tokens', 'refusal', 'cancelled',
        'structured_output_retries', 'execution_error', 'process_error', 'timeout', 'other_stop',
        'unknown', 'provider_error', 'rate_limit', 'authentication', 'unavailable', 'schema_rejected'},
    'classification_source': {'stop_reason', 'error_subtype', 'native_type', 'message_marker',
                               'process', 'none', 'unknown', 'native_schema_error'},
    'source': {'observed_native_final_totals', 'observed_native_cumulative_totals'},
    'totals_scope': {'native_final_envelope'},
    'schema': {'native-grok-final-usage@1', 'native-grok-outcome@1', 'grok-native-json-schema@1'},
}
EXCEPTIONS = {'RuntimeError', 'ValueError', 'TimeoutError', 'TimeoutExpired', 'LLMRouterError', 'ProviderInvocationError',
              'CalledProcessError', 'AgentTimeoutError', 'AgentSetupTimeoutError',
              'PromptGoalPlannerError', 'LocalPlanningError', 'OSError', 'PermissionError', 'ProcessLookupError',
              'InterruptedError', 'KeyboardInterrupt', 'SystemExit', 'other'}
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
        archive_content_hash_recomputed_by_this_export=False,
        grok_binary=provider_binary(manifest.get('grok_cli_assets')))


def provider_binary(value):
    value = value if type(value) is dict else {}
    digest = value.get('sha256')
    return dict(version=closed(value.get('version'), {'1.0.46'}),
        sha256=digest if type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) else None,
        bytes=scalar('response_schema_bytes', value.get('bytes')))


def outcome(value):
    return None if value is None else pick(value,
        'schema envelope_observed process_outcome stop_reason error_subtype error_observed '
        'reason_code classification_source raw_provider_data_exported completion_authority '
        'http_status schema_error_code schema_error_path')


def structured_output(value):
    if value is None:
        return None
    value = value if type(value) is dict else {}
    projection = value.get('native_schema_projection')
    return {**pick(value, 'schema response_schema_sha256 response_schema_bytes native_schema_requested '
                       'response_schema_validated plan_admitted'),
        'native_schema_projection': None if projection is None else pick(projection,
            'projection_id top_level_id_omitted canonical_schema_sha256 canonical_schema_bytes '
            'native_wire_schema_sha256 native_wire_schema_bytes canonical_validation_preserved '
            'task_contract_sha256 task_contract_bytes task_count task_contract_authority')}


BOOTSTRAP_PHASES = {'validation', 'peer', 'request', 'process_tree', 'duplicate_birth', 'lease', 'grant', 'response'}
BOOTSTRAP_REASONS = {'validation_failed', 'peer_mismatch', 'request_mismatch', 'process_tree_unavailable',
    'process_root_missing', 'process_root_ambiguous', 'process_child_missing', 'process_child_ambiguous',
    'process_child_is_root', 'process_parent_mismatch', 'process_scope_mismatch', 'duplicate_birth',
    'lease_unavailable', 'lease_mismatch', 'lease_expired', 'grant_unavailable',
    'grant_lifetime_exceeded', 'response_failed', 'unknown'}
TASK_STATUSES = {'completed', 'failed', 'blocked', 'cancelled', 'ready', 'pending', 'in_progress', 'retrying'}
PROGRESS_REASONS = {'native_task_blocked', 'native_task_failed', 'native_task_completed', 'native_task_cancelled', 'repeated_post_start_bootstrap_failure', 'expired_attempt_settlement_unavailable',
    'all_selectable_ready_tasks_reached_max_task_attempts', 'unsettled_portal_failure_quarantine',
    'completed', 'failed', 'blocked', 'cancelled', 'work_budget_exhausted'}


def closed(value, allowed):
    return value if type(value) is str and value in allowed else ('unknown' if value is not None else None)


def task_state(value):
    value = value if type(value) is dict else {}
    return dict(status=closed(value.get('status'), TASK_STATUSES), revision=scalar('revision', value.get('revision')))


def startup(value):
    if value is None:
        return None
    if type(value) is not dict or value.get('schema') != 'admitted-native-startup-observation@1':
        return {'observation': 'malformed'}
    result = pick(value, 'bootstrap_receipt_count bootstrap_error_count start_timeout_ms stop_timeout_ms observations_truncated')
    result['schema'] = 'admitted-native-startup-observation@1'
    phases = value.get('observations')
    if type(phases) is list and len(phases) <= 16:
        result['observations'] = [dict(phase=closed(row.get('phase'), {'control_validation', 'launch_validation', 'bootstrap_validation'}),
            status=closed(row.get('status'), {'running', 'completed', 'failed'}), seconds=scalar('seconds', row.get('seconds')))
            for row in phases if type(row) is dict]
    else:
        result['observations'] = None
    failures = value.get('bootstrap_failure_counts')
    if type(failures) is list and len(failures) <= len(BOOTSTRAP_PHASES) * len(BOOTSTRAP_REASONS):
        result['bootstrap_failure_counts'] = [dict(phase=closed(row.get('phase'), BOOTSTRAP_PHASES),
            reason=closed(row.get('reason'), BOOTSTRAP_REASONS), count=scalar('count', row.get('count')))
            for row in failures if type(row) is dict]
    else:
        result['bootstrap_failure_counts'] = None
    return result


def progress(value):
    value = value if type(value) is dict else {}
    latest = value.get('latest') if type(value.get('latest')) is dict else {}
    return dict(stop_reason=closed(value.get('stop_reason'), PROGRESS_REASONS),
        latest_task=task_state(latest.get('task')), elapsed_seconds=scalar('seconds', latest.get('seconds')),
        completion_authority=False, settlement_authority=False, retry_authority=False)


def custody(supervisor):
    error = supervisor.get('error') if type(supervisor.get('error')) is dict else {}
    closure = supervisor.get('runtime_close') if type(supervisor.get('runtime_close')) is dict else {}
    return dict(pick(supervisor, 'remaining_processes worker_cleanup_returncode'),
        runtime_close_attempted=scalar('runtime_close_attempted', closure.get('attempted')),
        runtime_close_succeeded=scalar('runtime_close_succeeded', closure.get('succeeded')),
        runtime_close_error_type=scalar('error_type', closure.get('error_type')),
        driver_error_type=scalar('error_type', error.get('type')),
        closure_inferred_from_stop=False)


def shutdown_failures(value, *, source_revision=None):
    """Retain only source-bound, closed STOP and close diagnostic slots."""
    if value is None:
        return None
    malformed = {'observation': 'malformed'}
    unavailable = {'observation': 'unavailable', 'reason': 'validator_source_unavailable'}
    if type(value) is not dict or not value or not set(value) <= {'stop', 'runtime_close'}:
        return malformed
    if type(source_revision) is not str or re.fullmatch('[0-9a-f]{40}', source_revision) is None:
        return unavailable
    source = pathlib.Path(HEAD_PATHS['source'])
    relative = 'benchmarks/agent_supervisor/container_coding/terminal_shutdown_observation.py'
    try:
        expected = subprocess.check_output(['git', '-C', str(source), 'show', source_revision + ':' + relative],
            stderr=subprocess.DEVNULL, timeout=5)
        path = source / relative
        if path.resolve(strict=True) != path.absolute() or path.read_bytes() != expected:
            return unavailable
        # Execute the already source-bound stdlib-only module bytes, avoiding
        # cached modules from another checkout or a second filesystem read.
        namespace = {'__file__': str(path), '__name__': 'closed_shutdown_observation'}
        exec(compile(expected, str(path), 'exec'), namespace)
        result = {}
        for slot, row in value.items():
            parsed = namespace['validate'](row)
            if parsed is None or ((parsed['phase'] == 'runtime_close') != (slot == 'runtime_close')):
                return malformed
            result[slot] = parsed
        return result
    except Exception:
        return unavailable


def start_cleanup(value):
    """Keep exact closed repair observations; absent legacy evidence is unknown."""
    if value is None:
        return None
    malformed = {'observation': 'malformed'}
    fields = {'schema', 'status', 'reason', 'proof_observation', 'control_observation',
        'lifecycle_phase', 'control_phase', 'marker_bound_process_tree_absent', 'start_succeeded',
        'absence_scope', 'completion_authority', 'retry_authority', 'execution_authority'}
    if (type(value) is not dict or set(value) != fields
            or value['schema'] != 'terminal-start-cleanup-observation@1'
            or value['absence_scope'] != 'recorded_marker_bound_tree'
            or any(value[key] is not False for key in
                ('completion_authority', 'retry_authority', 'execution_authority'))):
        return malformed
    observed = {'observed', 'missing', 'invalid', 'unavailable'}
    if any(type(value[key]) is not str or value[key] not in observed
           for key in ('proof_observation', 'control_observation')):
        return malformed
    proof = value['proof_observation'] == 'observed'
    control = value['control_observation'] == 'observed'
    if (value['lifecycle_phase'] != ('failed' if proof else None)
            or value['control_phase'] != ('repaired' if control else None)
            or value['marker_bound_process_tree_absent'] is not (True if proof else None)
            or value['start_succeeded'] is not (False if proof else None)):
        return malformed
    count = int(proof) + int(control)
    status = 'available' if count == 2 else 'partial' if count else 'unavailable'
    reasons = ({'bound_receipts'} if count == 2 else {'partial_evidence'} if count else
        {'evidence_unavailable', 'no_failed_start', 'original_start_unavailable',
         'collection_unavailable', 'runtime_not_created'})
    if (value['status'] != status or type(value['reason']) is not str or value['reason'] not in reasons
            or (count and 'unavailable' in (value['proof_observation'], value['control_observation']))
            or (value['reason'] != 'evidence_unavailable' and not count
                and (value['proof_observation'] != 'unavailable' or value['control_observation'] != 'unavailable'))):
        return malformed
    return dict(value)


def native_failures(value, *, source_revision=None):
    """Reuse archive-bound production validators; no bodies or authority survive."""
    if value is None:
        return None
    malformed = {'observation': 'malformed'}
    unavailable = {'observation': 'unavailable', 'reason': 'validator_source_unavailable'}
    base = {'schema', 'observation_only', 'completion_authority', 'retry_authority', 'settlement_authority'}
    if (type(value) is not dict or value.get('schema') != 'terminal-native-failure-observations@1'
            or value.get('observation_only') is not True
            or any(value.get(key) is not False for key in
                   ('completion_authority', 'retry_authority', 'settlement_authority'))):
        return malformed
    if set(value) == base | {'status'}:
        return dict(value) if value['status'] == 'unavailable' else malformed
    if set(value) != base | {'bridge', 'planner_child'}:
        return malformed
    # The selected trial's independently audited archive binds this revision.
    # Module bytes must still match it before any producer validator is used.
    source = pathlib.Path(HEAD_PATHS['source'])
    paths = (
        'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
        'ipfs_accelerate_py/agent_supervisor/todo_daemon/bridge_failure_diagnostics.py',
    )
    if type(source_revision) is not str or re.fullmatch('[0-9a-f]{40}', source_revision) is None:
        return unavailable
    try:
        for relative in paths:
            expected = subprocess.check_output(['git', '-C', str(source), 'show', source_revision + ':' + relative],
                                              stderr=subprocess.DEVNULL, timeout=5)
            path = source / relative
            if path.resolve(strict=True) != path.absolute() or path.read_bytes() != expected:
                return unavailable
        sys.path.insert(0, str(source))
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
            from ipfs_accelerate_py.agent_supervisor.todo_daemon import bridge_failure_diagnostics as bridge
        finally:
            sys.path.remove(str(source))
        if (pathlib.Path(router.__file__).resolve() != source / paths[0]
                or pathlib.Path(bridge.__file__).resolve() != source / paths[1]):
            return unavailable
        observations = {}
        for key, scopes, validator in (
            ('bridge', {'exact_admitted_task_and_attempt'}, bridge.validate_bridge_failure_diagnostic),
            ('planner_child', {'planner_child_stderr'}, router.validate_runner_error_envelope),
        ):
            item = value[key]
            fields = {'status', 'diagnostic', 'scope', 'matched_record_count', 'observation_only',
                      'provider_dispatch_observed', 'completion_authority', 'retry_authority', 'settlement_authority'}
            if (type(item) is not dict or set(item) != fields
                    or type(item['status']) is not str
                    or item['status'] not in {'observed', 'missing', 'ambiguous', 'invalid', 'unavailable'}
                    or type(item['scope']) is not str or item['scope'] not in scopes
                    or type(item['matched_record_count']) is not int
                    or not 0 <= item['matched_record_count'] <= (1 if key == 'bridge' else 4096)
                    or item['observation_only'] is not True or item['provider_dispatch_observed'] is not None
                    or any(item[name] is not False for name in
                           ('completion_authority', 'retry_authority', 'settlement_authority'))):
                return malformed
            diagnostic = None
            if item['status'] == 'observed':
                if item['matched_record_count'] < 1 or (key == 'planner_child' and item['matched_record_count'] != 1):
                    return malformed
                diagnostic = validator(item['diagnostic'])
                if diagnostic is None:
                    return malformed
            elif item['diagnostic'] is not None:
                return malformed
            observations[key] = {**item, 'diagnostic': diagnostic}
        return {**value, **observations}
    except (ValueError, TypeError, RecursionError):
        return malformed
    except Exception:
        return unavailable


def admission_failure(value):
    if type(value) is not dict:
        return None
    return dict(status=closed(value.get('status'), {'unavailable', 'observed', 'rejected'}),
        reason=closed(value.get('reason'), {'no_native_admission_error', 'native_admission_rejected', 'collection_unavailable'}),
        causal_proof=False, complete_admission_decision=False)


def invocation(value):
    if type(value) is not dict:
        raise ValueError('bounded invocation object required')
    policy = value.get('provider_invocation_policy')
    if policy is not None:
        policy = policy if type(policy) is dict else {}
        tool_fields = {}
        for field in ('disallowed_tools', 'native_tool_allowlist', 'native_tool_denylist'):
            values = policy.get(field) if type(policy) is dict else None
            tool_fields[field] = (values if type(values) is list and len(values) <= 16
                and all(type(tool) is str and tool in TOOLS for tool in values) else None)
        policy = dict(pick(policy, 'max_turns tools_profile permission_mode effective_toolset_verified'),
            structured_output=structured_output(policy.get('structured_output')), **tool_fields)
    return dict(pick(value, 'provider model purpose phase status seconds error_type timeout_seconds'),
        provider_failure=pick(value.get('provider_failure'), 'phase reason_code http_status'),
        provider_invocation_policy=policy, native_provider_outcome=outcome(value.get('native_provider_outcome')),
        native_usage=usage(value.get('native_rollout_usage'), native=True),
        requested_policy_is_actual_tool_observation=False)


TASK_CONTRACT_FIELDS = {
    'scope_paths': ('unordered_equal', ()),
    'outputs': ('unordered_equal', ('path', 'effect', 'media_type')),
    'validations': ('unordered_equal', ('validation_key', 'argv', 'cwd', 'expected_exit_codes', 'policy_cid')),
    'acceptance': ('unordered_equal', ('criterion_key', 'criterion', 'evidence_cids', 'validation_keys')),
    'dependencies': ('unordered_equal', ()),
    'assumptions': ('empty', ()),
    'evidence_cids': ('allowed_subset', ()),
    'policy_roots': ('ordered_equal', ()),
}


def task_contract_mismatch(value):
    """Preserve the producer's exact closed taxonomy, never proposed values."""
    if value is None:
        return None
    malformed = {'observation': 'malformed'}
    if (type(value) is not dict or set(value) != {'schema', 'fields'}
            or value['schema'] != 'supervisor-local-task-contract-mismatch@1'
            or type(value['fields']) is not list or not 1 <= len(value['fields']) <= 8):
        return malformed
    rows, seen = [], set()
    for row in value['fields']:
        if (type(row) is not dict or set(row) != {'field', 'comparison', 'expected_count',
                'observed_count', 'counts_capped', 'changed_members'} or type(row['field']) is not str):
            return malformed
        definition = TASK_CONTRACT_FIELDS.get(row['field'])
        if definition is None or row['field'] in seen:
            return malformed
        comparison, members = definition
        if (type(row['comparison']) is not str or row['comparison'] != comparison
                or type(row['counts_capped']) is not bool
                or any(type(row[key]) is not int or not 0 <= row[key] <= 65535
                       for key in ('expected_count', 'observed_count'))
                or (row['counts_capped'] and 65535 not in (row['expected_count'], row['observed_count']))
                or type(row['changed_members']) is not list or len(row['changed_members']) > len(members)
                or any(type(member) is not str or member not in members for member in row['changed_members'])
                or len(set(row['changed_members'])) != len(row['changed_members'])):
            return malformed
        seen.add(row['field'])
        rows.append(dict(row))
    return {'schema': 'supervisor-local-task-contract-mismatch@1', 'fields': rows}


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
    return {**pick(value, 'qualified provider_calls goals subgoals tasks planning_strategy'),
        'provider': scalar('planning_provider', value.get('provider')),
        'model': scalar('planning_model', value.get('model')),
        'provider_receipt': projected, 'provider_receipt_shape': shape,
        'independently_admitted_plan': scalar('qualified', value.get('qualified')),
        'admission_basis': 'terminal_indexed_preparation_qualified_after_local_admission_and_materialization',
        'failure_exception_type': scalar('exception_type', failure.get('type')),
        'task_contract_mismatch': task_contract_mismatch(value.get('task_contract_mismatch'))}


def collect(root, *, selected_names=None):
    root = pathlib.Path(root).resolve(strict=True)
    rows = []
    paths = sorted(path for task in TASKS for path in root.glob('grok-' + task + '-*/receipt.json')
        if selected_names is None or path.parent.name in selected_names)
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
        reward = reward if type(reward) in (int, float) and math.isfinite(reward) and reward in (0, 1) else None
        durations = trial.get('durations_seconds') or {}
        reasons = (supervisor.get('doctor_dispatch') or {}).get('reason_codes')
        reason_count = len(reasons) if type(reasons) is list and len(reasons) <= 256 else None
        rows.append(dict(trial_name=path.parent.name, task=match[1], attempt=match[2],
            receipt=reference, selection=selected, selection_binding=bindings,
            trial={**pick(trial, 'exception_type exact_trial_task_matches native_result_sha256'), 'reward': reward,
                'reward_basis': 'official_original_verifier_receipt' if reward is not None else 'unknown',
                'durations_seconds': {key: scalar('seconds', durations.get(key)) for key in
                    ('agent_execution', 'agent_setup', 'environment_setup', 'verifier')}},
            supervisor=pick(supervisor, 'task_completed seconds remaining_processes worker_cleanup_returncode '
                'implementation_route coding_dispatch_possible error_phase '
                'provider_coding_timeout_cap_seconds provider_coding_timeout_seconds'),
            lifecycle={key + '_status': scalar('lifecycle_operation_status', (supervisor.get(key) or {}).get('status'))
                for key in ('start', 'stop')},
            task_state=task_state(supervisor.get('task_state')),
            native_startup=startup(supervisor.get('native_startup')),
            native_progress=progress(supervisor.get('native_progress')),
            custody=custody(supervisor),
            shutdown_failures=shutdown_failures(supervisor.get('shutdown_failures'),
                source_revision=(bindings.get('source_revisions') or {}).get('source')),
            start_cleanup=start_cleanup(supervisor.get('start_cleanup')),
            native_failure_observations=native_failures(supervisor.get('native_failure_observations'),
                source_revision=(bindings.get('source_revisions') or {}).get('source')),
            admission_failure=admission_failure(supervisor.get('failure_admission')),
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
            benchmark_advantage_claimed=False, billing_total_verified=False,
            missing_usage_is_unknown=True))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=pathlib.Path, default=R)
    parser.add_argument('--output', default='live-trial-summary.json')
    parser.add_argument('--require-attempt', action='append', default=[])
    parser.add_argument('--trial-name', action='append', required=True,
        help='Only these exact new trial names are exported; no historical aggregate.')
    parser.add_argument('--require-trial-name', action='append', default=[])
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    output = root / args.output
    if (pathlib.Path(args.output).is_absolute() or '..' in pathlib.Path(args.output).parts
            or output.resolve() != output.absolute() or not output.is_relative_to(root)):
        raise ValueError('canonical new artifact output required')
    if any(TRIAL_NAME.fullmatch(name) is None for name in args.trial_name):
        raise ValueError('exact supported trial names required')
    rows = collect(root, selected_names=set(args.trial_name))
    if {row['trial_name'] for row in rows} != set(args.trial_name):
        raise ValueError('each selected trial needs its final receipt')
    for attempt in args.require_attempt:
        if sum(row['attempt'] == attempt for row in rows) != 1:
            raise ValueError('required attempt is absent or ambiguous; use --require-trial-name')
    if (any(TRIAL_NAME.fullmatch(name) is None for name in args.require_trial_name)
            or not set(args.require_trial_name) <= {row['trial_name'] for row in rows}):
        raise ValueError('required exact completed trial receipt is absent')
    with output.open('x') as stream:
        json.dump(dict(schema='terminal-grok-recovery-live-summary@1', trials=rows,
            raw_provider_verifier_or_credential_bodies_exported=False,
            native_usage_completeness_inferred=False), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([dict(trial_name=row['trial_name'], reward=row['trial']['reward'],
        task_completed=row['supervisor']['task_completed']) for row in rows]))


if __name__ == '__main__':
    main()
