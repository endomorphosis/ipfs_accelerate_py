"""Content-free final summary of already completed native trials; no execution.

Read only the observer's authoritative receipt projections, exact frozen configs,
preparations, launch results, and official emitted verifier output summaries.
Do not open task source, hidden test source, solutions, model transcripts or auth.
"""
from pathlib import Path
import hashlib
import json
import re

R = Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013')
OUT = R / 'native-observations'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 64 * 1024 * 1024
    return json.loads(path.read_text())

def binding(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def write(name, data):
    path = OUT / name
    assert not path.exists(), 'Never silently overwrite final evidence'
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return binding(path)

MISSING = object()

def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((differences(a.get(k, MISSING), b.get(k, MISSING), path + '/' + k)
                    for k in sorted(a.keys() | b.keys())), [])
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum((differences(x, y, path + '/' + str(i)) for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [{'path': path,
                              'baseline': '<absent>' if a is MISSING else a,
                              'treatment': '<absent>' if b is MISSING else b}]

def verifier(base):
    dirs = list(base.glob('jobs/*/largest-eigenval__*/verifier'))
    assert len(dirs) == 1
    directory = dirs[0]
    stdout = directory / 'test-stdout.txt'
    ctrf = directory / 'ctrf.json'
    # Only scalar summary counts are retained. No test-source files are opened.
    lines = [line for line in stdout.read_text().splitlines()
             if re.fullmatch(r'=+\s+(?:(?:\d+ failed, )?\d+ passed) in [\d.]+s\s+=+', line)]
    assert len(lines) == 1
    line = lines[0]
    matched = re.fullmatch(r'=+\s+(?:(\d+) failed, )?(\d+) passed in ([\d.]+)s\s+=+', line)
    assert matched
    counts = read(ctrf)['results']['summary']
    return {'parameterized_passed': int(matched[2]), 'parameterized_failed': int(matched[1] or 0),
            'reported_test_seconds': float(matched[3]),
            'ctrf_group_counts': {k: counts[k] for k in ('tests', 'passed', 'failed', 'skipped', 'pending', 'other')},
            'ctrf_counts_are_parameterized_case_counts': False,
            'emitted_stdout': binding(stdout), 'ctrf': binding(ctrf),
            'test_or_fixture_source_read': False}

observation_path = OUT / 'observation-03.json'
observation = read(observation_path)
names = ['codex-legacy-01', 'codex-projected-01']
bases = [R / 'live-trials' / name for name in names]
configs = [read(base / 'config.json') for base in bases]
preps = [read(base / 'preparation.json') for base in bases]
config_diff = differences(*configs)
assert [x['path'] for x in config_diff] == ['/agents/0/kwargs/semantic_metadata_view', '/jobs_dir']
assert config_diff[0]['baseline'] == '<absent>' and config_diff[0]['treatment'] == 'common-bindings@1'
same_keys = ['agent_timeout_seconds', 'archive', 'archive_sha256', 'arm', 'cli_version', 'dataset',
             'host_source_sha256', 'intent_action_384', 'intent_requirement_contract_cid',
             'intent_requirement_contract_sha256', 'manifest_sha256', 'model',
             'planning_and_cold_index_charged_to_agent_time', 'planning_strategy',
             'provider_profile', 'reasoning_effort', 'resource_profile', 'source384',
             'source384_enabled', 'source_view_authority', 'source_view_profile', 'task', 'task_input_sha256']
same = {key: preps[0][key] == preps[1][key] for key in same_keys}
assert all(same.values())
assert observation['codex_declared_controls_comparison']['matches'] is True
control_checks = {}
for name in names:
    arm = observation['arms'][name]
    assert len(arm['attempts']) == 1
    checks = {key: arm[key] for key in ('config_matches_preparation', 'prepared_controls_valid', 'host_source_closure_unchanged')}
    checks.update({key: arm['receipt_summary'][key] for key in ('original_task_inputs_unchanged', 'source_view_config_unchanged', 'intent_selection_config_unchanged')})
    checks['all_observed_coding_inputs_verified'] = arm['attempts'][0]['context_input_audit']['all_observed_coding_inputs_verified']
    if name == names[1]:
        checks['semantic_metadata_config_unchanged'] = arm['receipt_summary']['semantic_metadata_config_unchanged']
    assert all(checks.values())
    control_checks[name] = checks
control_audit = {
    'schema': 'source-context-native-exact-control-audit@1',
    'observer': binding(observation_path),
    'configurations': {name: binding(base / 'config.json') for name, base in zip(names, bases)},
    'preparations': {name: binding(base / 'preparation.json') for name, base in zip(names, bases)},
    'exact_config_differences': config_diff,
    'only_intentional_metadata_and_output_location_differences': True,
    'preparation_common_fields_equal': same,
    'existing_compare_controls_result': observation['codex_declared_controls_comparison'],
    'interpretation': 'Existing compare_controls matched the shared declared controls. Metadata mode is the intentional treatment; jobs_dir isolates outputs. Full configuration digests differ as expected. Exact JSON comparison establishes no other configuration differences. Runtime input auditing establishes the selected coding prompts; configuration equality alone does not establish identical live conditions or generated plans.',
    'observed_integrity_checks': control_checks,
    'common_profile': {key: preps[0][key] for key in ('provider_profile', 'model', 'reasoning_effort', 'cli_version', 'archive_sha256', 'manifest_sha256', 'arm', 'source384_enabled', 'source_view_profile', 'resource_profile', 'agent_timeout_seconds', 'planning_strategy')},
    'shared_source384_checkpoint_sha256': observation['arms'][names[0]]['attempts'][0]['source384']['context']['checkpoint_sha256'],
    'source_pins': {'accelerate': '2c220ae4b7c79ad9980122c3329ebdf9ef4131a3',
                    'datasets': 'f47161b757190380e9898aff155067c1ad609217',
                    'kit': '8ac00bb1974b1378e12091d546d8425af34049e0'},
    'native_plans_and_provider_sessions_generated_independently': True,
    'identical_runtime_or_candidate_output_claimed': False,
    'provider_calls_by_auditor': 0,
}
control_binding = write('exact-control-audit.json', control_audit)

arms = {}
for name, base in zip(names, bases):
    arm = observation['arms'][name]
    trial = arm['attempts'][0]
    receipt = read(base / 'receipt.json')
    launch = read(base / 'launch-result.json')
    report = receipt['trials'][0]['supervisor']
    tokens = dict(trial['tokens'])
    assert trial['native_usage_complete_observed'] is True
    assert trial['cache_included_in_input'] is True and trial['unreceipted_provider_attempt'] is False
    tokens['uncached_input_tokens'] = tokens['input_tokens'] - tokens['cached_input_tokens']
    assert tokens['total_tokens'] == tokens['input_tokens'] + tokens['output_tokens']
    invocations = []
    for row in trial['invocations']:
        token = dict(row['tokens'])
        assert row['native_usage_complete_observed'] is True and row['cache_included_in_input'] is True
        token['uncached_input_tokens'] = token['input_tokens'] - token['cached_input_tokens']
        inp = row['input_observation']
        selected = inp.get('semantic_metadata_view')
        projection = {'status': 'not_requested' if row['phase'] == 'coding' else 'not_applicable_planning',
                      'selected_mode': 'legacy' if row['phase'] == 'coding' else None,
                      'counterfactual_bytes_receipted': False,
                      'byte_reduction_is_measured_provider_tokens': False}
        if selected:
            saved = selected['native_complete_bytes'] - selected['selected_complete_bytes']
            assert selected['selected_complete_bytes'] == inp['model_prompt_bytes']
            assert selected['selected_complete_sha256'] == inp['model_prompt_sha256']
            projection.update(status='selected_rendered_replacement', selected_mode=selected['selected_mode'],
                              counterfactual_bytes_receipted=True, native_complete_bytes=selected['native_complete_bytes'],
                              selected_complete_bytes=selected['selected_complete_bytes'], byte_reduction=saved,
                              byte_reduction_percent=100 * saved / selected['native_complete_bytes'],
                              native_complete_sha256=selected['native_complete_sha256'],
                              selected_complete_sha256=selected['selected_complete_sha256'],
                              selection_basis=selected['selection_basis'],
                              tokenizer_compatibility_verified=selected['tokenizer_compatibility_verified'],
                              view_receipt_sha256=selected['view_receipt_sha256'],
                              displacement=row['metadata_displacement'])
        invocations.append({'invocation_id': row['invocation_id'], 'phase': row['phase'],
                            'provider': row['provider'], 'model': row['model'], 'status': row['status'],
                            'seconds': row['seconds'], 'tokens': token,
                            'native_usage_complete': row['native_usage_complete_observed'],
                            'token_count_scope': row['token_count_scope'], 'cache_included_in_input': row['cache_included_in_input'],
                            'model_prompt_bytes': inp['model_prompt_bytes'], 'model_prompt_sha256': inp['model_prompt_sha256'],
                            'source_view_profile': row['source_view_profile'], 'metadata_projection': projection})
    declared = report['declared_output_observation']
    assert declared['outputs'] == []
    successor = report['post_publication_context']['results'][0]['source384_refresh']
    arms[name] = {
        'profile': arm['declared_profile'], 'trial': trial['trial'], 'official_reward': trial['official_reward'],
        'supervisor_task_completed': trial['native_task_completed'], 'outcome': trial['outcome'],
        'harbor_exception_type': trial['exception_type'], 'supervisor_error_type': trial['supervisor_error_type'],
        'verifier': verifier(base), 'tokens': tokens, 'native_usage_complete': True,
        'cache_included_in_input': True, 'billing_total_verified': False, 'reported_cost_usd': trial['reported_cost_usd'],
        'provider_invocations': len(invocations), 'invocations': invocations,
        'durations_seconds': {**trial['durations_seconds'], 'driver_wall': launch['seconds'],
                             'receipt_invocation': arm['receipt_summary']['invocation_seconds'],
                             'supervisor': trial['supervisor_seconds'],
                             'provider_invocations_sum': sum(row['seconds'] for row in invocations)},
        'supervisor_phases_seconds': trial['supervisor_phases_seconds'],
        'durations_are_nested_and_not_additive': True,
        'source384': trial['source384'], 'intent_ir': trial['intent_ir'],
        'source_unit_preplanning': trial['source_unit_preplanning'],
        'initial_indexed_context': trial['initial_indexed_context_observation'],
        'indexed_context': trial['indexed_context_observation'], 'source_reader': trial['source_reader'],
        'source384_successor_inference': {key: successor[key] for key in ('status', 'checkpoint_sha256', 'inference_sha256', 'inference_execution', 'inference_activity_scope', 'training_steps', 'text_generation_calls', 'source_semantics_verified', 'proof_authority')},
        'declared_output_observation': {key: declared[key] for key in ('schema', 'status', 'reason', 'outputs', 'observation_boundary', 'patch_application_established', 'writer_identity_established')},
        'postimage_byte_identity_established': False,
        'evidence': {key: binding(base / (key + '.json')) for key in ('receipt', 'launch-result', 'preparation', 'config')},
    }

grok_base = R / 'live-trials/grok-legacy-02'
grok = observation['grok_failed_setup_attempts'][0]
grok_launch = read(grok_base / 'launch-result.json')
grok_diagnosis = read(R / 'live-preparation/grok-readiness-diagnosis.json')
assert grok['invocations'] == [] and grok['exception_type'] == 'GrokCredentialReadinessError'
assert grok_diagnosis['provider_generation_calls'] == 0
delta = {key: arms[names[1]]['tokens'][key] - arms[names[0]]['tokens'][key] for key in arms[names[0]]['tokens']}
assert delta['total_tokens'] == 1
summary = {
    'schema': 'source-context-native-pair-summary@1', 'status': 'completed', 'task': 'largest-eigenval',
    'observer': binding(observation_path), 'summarizer': binding(Path(__file__)), 'exact_control_audit': control_binding,
    'shared_controls_matched': True, 'intended_metadata_modes': {'baseline': 'legacy', 'treatment': 'common-bindings@1'},
    'arms': arms, 'treatment_minus_baseline_tokens': delta,
    'total_provider_token_reduction_observed': False,
    'total_provider_tokens_percent_change': 100 * delta['total_tokens'] / arms[names[0]]['tokens']['total_tokens'],
    'causal_token_savings_or_reward_advantage_established': False,
    'executed_attempt_ledger': [
        {'name': 'grok-legacy-02', 'trial': grok['trial'], 'provider': 'grok_cli', 'model': 'grok-4.7',
         'status': 'failed_credential_readiness_before_generation', 'exception_type': grok['exception_type'],
         'official_reward': None, 'provider_generation_calls': 0, 'observed_provider_invocations': 0,
         'native_catalog_metadata_commands': 1, 'tokens': grok['tokens'],
         'unknown_usage_not_coerced_to_zero': True,
         'durations_seconds': {**grok['durations_seconds'], 'driver_wall': grok_launch['seconds']},
         'receipt': binding(grok_base / 'receipt.json'), 'launch_result': binding(grok_base / 'launch-result.json')},
        *[{'name': name, 'trial': arms[name]['trial'], 'provider': 'codex_cli', 'model': 'gpt-6.1-sol',
           'status': 'completed', 'provider_generation_calls': arms[name]['provider_invocations'],
           'official_reward': arms[name]['official_reward'], 'receipt': arms[name]['evidence']['receipt']} for name in names],
    ],
    'executed_benchmark_attempts': 3,
    'completed_codex_trials': 2, 'native_provider_generation_invocations_across_completed_trials': 4,
    'excluded_model_attempts': 0, 'unreceipted_provider_attempts': 0,
    'metadata_only_diagnostic_probes': {
        'commands': observation['grok_metadata_probe_count'], 'provider_generation_calls': observation['grok_metadata_probe_generation_calls'],
        'evidence': observation['grok_metadata_probe_evidence'],
        'scope': 'Two fresh isolated credential-copy model-catalog probes; separate from benchmark generation attempts.',
    },
    'preparation_failures': [{'name': 'grok-legacy-01', 'stage': 'dry_run_import', 'error_type': 'TypeError',
                              'provider_generation_calls': 0, 'benchmark_executed': False,
                              'repair': 'Use existing compatible Harbor venv packages first in Python path.'}],
    'not_executed': ['grok-projected-02'],
    'official_verifier_interpretation': 'Legacy completed implementation with 26 passing parameterized cases and one speed assertion failure. Projected completed with all 27 passing. One sample per arm and timing-sensitive scoring do not establish a causal metadata effect or output byte equivalence.',
    'scope_limits': [
        'The treatment saved 882 rendered input bytes within its own coding invocation, using a byte proxy; that is not a tokenizer measurement. Independent planning means cross-arm prompt lengths are not a controlled projection counterfactual.',
        'Native cumulative usage is counted once per unique invocation. Cached input is a subset of Codex input, not added again. Total tokens increased by one; input decreased by 198 and output increased by 199. Billing cost is unverified.',
        'Both trials used full indexed supervisors with the same identity@1 source reader. This is a legacy-versus-common-bindings metadata ablation, not a native Codex harness or no-index comparison.',
        'Both owner source-reader sessions recorded zero requests and zero delivered bytes. This pair does not establish savings from source retrieval or whole-codebase reversible minification.',
        'Pinned Source384 inference ran, but all four selected function candidates remained unsupported and unverified. No source obligations were discharged, no model training occurred, and whole-program proof was not established.',
        'IntentIR ran before goal decomposition but failed open without a checkpoint. Optional source-unit preplanning was disabled. Learned-index embedding reporting was false. These are coverage gaps, not formal proof success.',
        'Post-publication Source384 successor inference executed once per arm. Its existence is not proof of task correctness; both direct declared-output observations lacked usable file bindings, so candidate output byte identity remains unknown.',
        'Agent setup took 256.701212 seconds versus 734.253863 seconds. Provider invocation time is nested within agent execution; no attribution of setup or throughput differences to metadata is established.',
        'The separate scripted Docker pair saved 885 bytes with matched source operations. Its byte result is distinct from this live pair and is not provider token evidence.',
    ],
    'reader_actions': {'provider_calls': 0, 'new_trials': 0, 'production_files_changed': False,
                       'hidden_test_or_solution_source_read': False, 'auth_contents_read': False,
                       'raw_prompt_or_model_transcript_read': False},
}
result = write('pair-summary.json', summary)
print(json.dumps({'summary': result, 'controls': control_binding, 'total_tokens': {name: arms[name]['tokens']['total_tokens'] for name in names}, 'delta': delta, 'rewards': {name: arms[name]['official_reward'] for name in names}}, sort_keys=True))
