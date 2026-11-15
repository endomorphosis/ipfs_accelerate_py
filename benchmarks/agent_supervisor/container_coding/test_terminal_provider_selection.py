"""Selected Grok planning, signed public text and provider-specific accounting."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_provider_profile as profiles
from benchmarks.agent_supervisor.container_coding import benchmark_controls, full_supervisor_benchmark as full
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import (
    measured_usage, populate_harbor_usage, AgentContext)
from benchmarks.agent_supervisor.container_coding.terminal_planner_instruction import bind_public_instruction
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_harbor_session_usage import (
    _codex_receipt, _grok_receipt,
)


def test_grok_profile_config_and_frozen_collection_identity(tmp_path):
    selected = profiles.resolve_provider_profile(profiles.GROK_PROFILE)
    config = full.config_for(Path('/dataset'), tmp_path, tmp_path / 'archive', 'full',
                            provider_profile=profiles.GROK_PROFILE)
    identity = {key: selected[key] for key in ('model', 'reasoning_effort', 'cli_version')}
    controls = benchmark_controls.build_controls(config, task_input_sha256={'instruction.md': 'a' * 64},
                                                  task='authored', **identity)
    prepared = {**identity, 'comparison_controls': controls, 'provider_profile': profiles.GROK_PROFILE}
    assert profiles.prepared_provider_identity(prepared, config) == identity
    assert controls['identity']['model'] == 'grok-4.7'
    prepared['provider_profile'] = profiles.CODEX_PROFILE
    with pytest.raises(ValueError, match='provider selection'):
        profiles.prepared_provider_identity(prepared, config)
    with pytest.raises(ValueError, match='cache'):
        full.config_for(Path('/dataset'), tmp_path, tmp_path / 'archive', 'full',
            provider_profile=profiles.GROK_PROFILE, setup_cache_selection={})


@pytest.mark.parametrize('version_output', ['grok 1.0.46 (2765805b9442)',
                                         'grok 1.0.46 (2765805b9442) [stable]'])
def test_verbatim_fenced_instruction_enters_selected_signed_grok_planner(original, monkeypatch, version_output):
    root, instruction, state = original
    text = 'Inspect bottle.py. Example output:\n```json\n{"file_path":"/app/bottle.py","cwe_id":[]}\n```\n'
    instruction.write_text(text)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
                            provider_profile=profiles.GROK_PROFILE)
    request = json.loads((state / 'provider-request.json').read_text())
    assert request['terminal_public_instruction']['text'] == text
    assert request['terminal_public_instruction']['source_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert not any('```' in row for row in request['constraints']['constraint_summaries'])
    actual = prep.subprocess.run
    def command(argv, *args, **kwargs):
        if argv == ['/opt/ipfs-supervisor/provider-bin/grok', '--version']:
            return SimpleNamespace(stdout=version_output + '\n')
        return actual(argv, *args, **kwargs)
    monkeypatch.setattr(prep.subprocess, 'run', command)
    calls = []
    def provider(prompt, **kwargs):
        calls.append(kwargs)
        bound = json.loads(prompt)['terminal_public_instruction']
        assert bound['text'] == text and bound['execution_authority'] is False
        assert kwargs['provider'] == 'grok_cli' and kwargs['model'] == 'grok-4.7'
        return {'text': _proposal_json(prepared), 'observation': {}, 'execution_receipt': None}
    result = prep.plan(state=state, provider_callable=provider)
    assert result['qualified'] and result['provider'] == 'grok_cli' and len(calls) == 1
    assert prep._load_prepared(state)['request']['planning_policy']['provider_preferences'] == ['grok_cli']


@pytest.mark.parametrize('mutation', ['profile', 'model', 'signed_preference'])
def test_prepared_route_cannot_be_changed_after_signing(original, mutation):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    if mutation == 'profile':
        prepared['provider_profile'] = profiles.GROK_PROFILE
    elif mutation == 'model':
        prepared['model'] = 'grok-4.7'
    else:
        prepared['request']['planning_policy']['provider_preferences'] = ['grok_cli']
    (state / 'prepared.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError):
        prep._load_prepared(state)


def test_instruction_envelope_rejects_digest_drift_overwrite_and_oversize(original):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    with pytest.raises(ValueError, match='byte budget'):
        bind_public_instruction('{}', prepared=prepared, maximum_bytes=1)
    with pytest.raises(ValueError, match='unmodified'):
        bind_public_instruction('{"terminal_public_instruction":{}}', prepared=prepared)
    with pytest.raises(ValueError, match='duplicate'):
        bind_public_instruction('{"a":1,"a":2}', prepared=prepared)
    changed = deepcopy(prepared)
    changed['query'] += '\nChanged task.'
    with pytest.raises(ValueError, match='signed source'):
        bind_public_instruction('{}', prepared=changed)


@pytest.mark.parametrize('bound', ['max_prompt_tokens', 'max_serialized_bytes'])
def test_instruction_envelope_preserves_signed_request_budget(original, bound):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    if bound == 'max_serialized_bytes':
        request = prep.PromptWorkflowRequest.from_dict(prepared['request'])
        prepared['request'] = replace(request,
            budget=replace(request.budget, max_serialized_bytes=100000)).to_dict()
    budget = prepared['request']['budget']
    limit = min(budget['max_serialized_bytes'], budget['max_prompt_tokens'] * 4)
    # The original request fits; the appended instruction would cross the
    # signed bound while remaining below the adapter's separate 262144 cap.
    prompt = json.dumps({'padding': 'x' * (limit - 100)}, separators=(',', ':'))
    assert len(prompt.encode()) < limit < 262144
    with pytest.raises(ValueError, match='byte budget'):
        bind_public_instruction(prompt, prepared=prepared)


@pytest.mark.parametrize('complete', [True, False, None])
def test_grok_counters_preserve_native_cache_and_completeness_semantics(complete):
    row = _grok_receipt(complete=complete, writes=0)
    observed = measured_usage({'provider_invocations': [row, row]})
    assert observed['input_tokens'] == (10 if complete is True else None)
    assert observed['total_tokens'] == (42 if complete is True else None)
    assert observed['known_token_subtotals']['total_tokens'] == 42
    assert observed['cache_included_in_input'] is False and observed['provider_calls'] == 1
    assert observed['usage_complete_observed'] is complete
    assert observed['source'] == 'observed_native_final_totals'
    missing = measured_usage({'provider_invocations': [row], 'unreceipted_provider_attempt': {'phase': 'coding'}})
    assert missing['total_tokens'] is None and not missing['observed_complete_sessions']
    assert missing['usage_complete_observed'] is (False if complete is False else None)


@pytest.mark.parametrize('alteration', [None, 'missing_write', 'unknown_semantics', 'conflicting_total',
    'unreceipted', 'missing_completeness', 'incomplete', 'boolean_lookalike', 'unfinished', 'unknown_schema'])
def test_grok_harbor_input_includes_cache_only_with_complete_consistent_counters(alteration):
    row = _grok_receipt()
    receipt = row['native_rollout_usage']
    report = {'provider_invocations': [row]}
    if alteration == 'missing_write':
        del receipt['usage']['cache_write_input_tokens']
    elif alteration == 'unknown_semantics':
        del receipt['cache_included_in_input']
    elif alteration == 'conflicting_total':
        receipt['usage']['total_tokens'] += 1
    elif alteration == 'unreceipted':
        report['unreceipted_provider_attempt'] = {'phase': 'coding'}
    elif alteration == 'missing_completeness':
        del receipt['usage_complete_observed']
    elif alteration == 'incomplete':
        receipt['usage_complete_observed'] = False
    elif alteration == 'boolean_lookalike':
        receipt['usage_complete_observed'] = 1
    elif alteration == 'unfinished':
        receipt['task_complete_observed'] = False
    elif alteration == 'unknown_schema':
        receipt['schema'] = 'unknown'
    usage = measured_usage(report)
    context = AgentContext(n_input_tokens=123, n_cache_tokens=100, n_output_tokens=23)
    populate_harbor_usage(context, usage)
    assert context.n_input_tokens == (45 if alteration is None else None)
    assert context.n_cache_tokens == (30 if alteration is None else None)
    assert context.n_output_tokens == (2 if alteration is None else None)
    assert usage['input_tokens'] == (10 if alteration is None else None)
    assert usage['billing_total_verified'] is False


@pytest.mark.parametrize('complete', [True, False, None, 1])
def test_harbor_codex_cumulative_scope_preserves_native_completion_requirement(complete):
    row = _codex_receipt(complete=complete)
    usage = measured_usage({'provider_invocations': [row, deepcopy(row)], 'task_completed': True})
    context = AgentContext()
    populate_harbor_usage(context, usage)
    assert usage['total_tokens'] == (110 if complete is True else None)
    assert usage['provider_calls'] == 1
    assert context.n_input_tokens == (100 if complete is True else None)
    assert context.n_cache_tokens == (80 if complete is True else None)
    assert context.n_output_tokens == (10 if complete is True else None)
    assert usage['token_count_scope'] == (None if type(complete) is int else 'native_session_cumulative')


def test_harbor_explicit_grok_route_preserves_cache_semantics():
    row = _grok_receipt()
    usage = measured_usage({'provider_invocations': [row]})
    context = AgentContext()
    populate_harbor_usage(context, usage)
    assert usage['cache_included_in_input'] is False
    assert usage['cache_write_input_tokens'] == 5
    assert usage['source'] == 'observed_native_final_totals'
    assert context.n_input_tokens == 45 and context.n_cache_tokens == 30 and context.n_output_tokens == 2
    row['provider'] = 'codex_cli'
    contradictory = measured_usage({'provider_invocations': [row]})
    populate_harbor_usage(context, contradictory)
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    assert contradictory['token_count_scope'] is None


def test_harbor_does_not_invent_provider_route_from_native_schema():
    row = _grok_receipt()
    del row['provider']
    usage = measured_usage({'provider_invocations': [row]})
    context = AgentContext()
    populate_harbor_usage(context, usage)
    assert usage['total_tokens'] is None
    assert usage['provider_calls'] == 1
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
