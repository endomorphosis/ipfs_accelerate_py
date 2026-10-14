"""Read-only capsule experiments retain control and all-attempt accounting gaps."""
import hashlib
import json
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_capsule_budget_comparison as comparison
from benchmarks.agent_supervisor.container_coding.terminal_capsule_selection_policy import CAPSULE_BUDGETS, capsule_budget_observation
from benchmarks.agent_supervisor.container_coding.test_benchmark_comparison import _invocation, _grok_invocation


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt(tmp_path, budget, name=None, *, tokens=100, reward=1, mutate_config=None,
            mutate_trial=None, mutate_receipt=None):
    name = name or budget
    root = tmp_path / name
    root.mkdir()
    config = benchmark.config_for(tmp_path / 'dataset', root, tmp_path / 'archive', 'full', capsule_budget=budget)
    if mutate_config:
        mutate_config(config)
    config_sha = write(root / 'config.json', config)
    identity = dict(task=benchmark.TASK, model=benchmark.MODEL,
                    reasoning_effort='high', cli_version=benchmark.CLI_VERSION)
    hashes = {'instruction.md': 'a' * 64}
    declared = controls.build_controls(config, task_input_sha256=hashes, **identity)
    preparation = dict(schema='terminal-full-supervisor-preparation@1', prepared=True, arm='full',
        config_sha256=config_sha, task_input_sha256=hashes, comparison_controls=declared,
        archive_sha256='b' * 64, manifest_sha256='c' * 64, host_source_sha256={'fixture.py': 'd' * 64},
        **identity, **capsule_budget_observation(budget))
    write(root / 'preparation.json', preparation)
    call = _invocation(name)
    call.update(phase='coding', model=benchmark.MODEL, model_prompt_bytes=1000,
                model_prompt_sha256='e' * 64)
    call['native_rollout_usage']['usage'].update(input_tokens=tokens - 2, cached_input_tokens=0,
                                               output_tokens=2, total_tokens=tokens)
    report = dict(task_completed=True, provider_invocations=[call],
        context=dict(worker_capsules={'legacy': 4, 'one-capsule@1': 1, 'two-capsules@1': 2}[budget],
                     full_capsules=8, worker_semantic_bytes=2000), **capsule_budget_observation(budget))
    trial = dict(trial=name, exact_trial_task_matches=True, reward={'reward': reward}, supervisor=report,
                 agent_context={'metadata': {'runtime_archive_sha256': 'b' * 64}})
    if mutate_trial:
        mutate_trial(trial)
    receipt = dict(schema=comparison.native.SUPERVISOR, arm='full', trial_count=1, trials=[trial],
        original_task_inputs_unchanged=True, complete_single_trial_receipt=True,
        comparison_controls=controls.observe_controls(preparation, config, current_task_hashes=hashes),
        **identity, **capsule_budget_observation(budget))
    if budget != 'legacy':
        receipt.update(capsule_budget_config_unchanged=True, capsule_budget_runtime_declared_matches=True)
    if mutate_receipt:
        mutate_receipt(receipt)
    write(root / 'receipt.json', receipt)
    return root / 'receipt.json'


def campaign(tmp_path):
    return dict(zip(('legacy', 'one', 'two'), ([attempt(tmp_path, budget, tokens=tokens)]
        for budget, tokens in zip(CAPSULE_BUDGETS, (100, 60, 80)))))


def mutate(path, callback):
    value = json.loads(path.read_text())
    callback(value)
    write(path, value)


def test_budget_only_comparison_groups_all_three_and_preserves_generic_mismatch(tmp_path):
    result = comparison.collect(**campaign(tmp_path))
    assert result['matched_official_success'] is True
    assert result['contrasts']['one-capsule@1']['candidate_minus_legacy_tokens'] == -40
    assert result['contrasts']['two-capsules@1']['candidate_minus_legacy_tokens'] == -20
    assert result['budgets']['legacy']['selected_capsule_counts'] == [4]
    assert result['attempts'][1]['experiment_control_comparison']['generic_control_comparison']['matches'] is False
    assert result['attempts'][1]['experiment_control_comparison']['matches'] is True
    assert result['token_savings_claimed'] is result['billing_total_verified'] is False
    assert result['planning_outputs_held_fixed'] is False
    assert 'MUST NOT APPEAR' not in json.dumps(result)
    assert 'Prompt bytes are not token savings' in comparison.markdown(result)


@pytest.mark.parametrize('group', ['legacy', 'one', 'two'])
@pytest.mark.parametrize('gap', ['missing', 'no-trial', 'partial', 'unreceipted'])
def test_registered_unknown_attempts_never_become_zero_or_a_savings_conclusion(tmp_path, group, gap):
    paths = campaign(tmp_path)
    path = paths[group][0]
    if gap == 'missing':
        path.unlink()
    elif gap == 'no-trial':
        mutate(path, lambda value: value.update(trials=[], trial_count=0))
    elif gap == 'partial':
        mutate(path, lambda value: value['trials'][0]['supervisor']['provider_invocations'][0]
               ['native_rollout_usage'].update(task_complete_observed=False))
    else:
        mutate(path, lambda value: value['trials'][0]['supervisor'].update(unreceipted_provider_attempt={'phase': 'coding'}))
    result = comparison.collect(**paths)
    budget = dict(zip(('legacy', 'one', 'two'), CAPSULE_BUDGETS))[group]
    assert result['budgets'][budget]['aggregate']['tokens']['total_tokens'] is None
    assert result['budgets'][budget]['complete_registered_native_usage'] is False
    relevant = CAPSULE_BUDGETS[1:] if group == 'legacy' else [budget]
    for name in relevant:
        assert result['contrasts'][name]['candidate_minus_legacy_tokens'] is None
    assert len(result['attempts']) == 3


def test_failed_trial_costs_are_counted_but_success_is_not_matched(tmp_path):
    paths = campaign(tmp_path)
    paths['one'].append(attempt(tmp_path, 'one-capsule@1', 'failed', tokens=30, reward=0,
        mutate_trial=lambda trial: trial['supervisor']['provider_invocations'][0].update(status='failed', error_type='TimeoutError')))
    result = comparison.collect(**paths)
    one = result['budgets']['one-capsule@1']['aggregate']
    assert one['trials'] == 2 and one['observed_official_passes'] == 1
    assert one['tokens']['total_tokens'] == 90
    assert one['observed_efficiency']['total_tokens_per_official_pass'] == 90
    assert result['matched_official_success'] is False
    assert result['contrasts']['one-capsule@1']['candidate_minus_legacy_tokens'] is None


def test_internal_completion_without_reward_does_not_establish_success(tmp_path):
    paths = campaign(tmp_path)
    mutate(paths['two'][0], lambda value: value['trials'][0].update(reward={}))
    result = comparison.collect(**paths)
    assert result['rows'][2]['native_task_completed'] is True
    assert result['matched_official_success'] is False
    assert result['budgets']['two-capsules@1']['aggregate']['observed_efficiency']['total_tokens_per_official_pass'] is None


@pytest.mark.parametrize('change', ['metadata', 'resource', 'archive', 'runtime-budget', 'count', 'config-hash', 'sidecar'])
def test_other_treatments_or_unverified_bindings_prevent_qualification(tmp_path, change):
    paths = campaign(tmp_path)
    path = paths['one'][0]
    if change == 'sidecar':
        (path.parent / 'config.json').unlink()
    elif change == 'config-hash':
        mutate(path.parent / 'config.json', lambda config: config.update(timeout_multiplier=2))
    elif change in ('metadata', 'resource'):
        # A valid separately sealed experiment is still not the same treatment.
        shutil.rmtree(path.parent)
        def alter(config):
            if change == 'metadata':
                config['agents'][0]['kwargs']['semantic_metadata_view'] = 'common-bindings@1'
            else:
                config['environment']['override_memory_mb'] = 32768
        paths['one'][0] = attempt(tmp_path, 'one-capsule@1', mutate_config=alter)
    elif change == 'archive':
        mutate(path, lambda value: value['trials'][0]['agent_context']['metadata'].update(runtime_archive_sha256='f' * 64))
    elif change == 'runtime-budget':
        mutate(path, lambda value: value['trials'][0]['supervisor'].update(capsule_budget='two-capsules@1'))
    else:
        mutate(path, lambda value: value['trials'][0]['supervisor']['context'].update(worker_capsules=8))
    result = comparison.collect(**paths)
    assert result['all_experiment_controls_match'] is False
    assert result['matched_official_success'] is False
    assert result['contrasts']['one-capsule@1']['comparable'] is False


def test_missing_receipt_and_sidecars_remain_registered(tmp_path):
    paths = campaign(tmp_path)
    paths['one'].append(tmp_path / 'not-run' / 'receipt.json')
    result = comparison.collect(**paths)
    assert len(result['attempts']) == 4
    assert result['attempts'][2]['status'] == 'missing_receipt'
    assert result['budgets']['one-capsule@1']['aggregate']['known_token_subtotals']['total_tokens'] == 60
    assert result['budgets']['one-capsule@1']['aggregate']['observed_efficiency']['total_tokens_per_official_pass'] is None


def test_duplicate_observation_is_not_charged_twice(tmp_path):
    paths = campaign(tmp_path)
    copy = tmp_path / 'copy'
    shutil.copytree(paths['one'][0].parent, copy)
    paths['one'].append(copy / 'receipt.json')
    result = comparison.collect(**paths)
    assert result['budgets']['one-capsule@1']['aggregate']['tokens']['total_tokens'] == 60
    assert result['budgets']['one-capsule@1']['aggregate']['trials'] == 1
    assert result['attempts'][2]['status'] == 'duplicate_trial_observation'


@pytest.mark.parametrize('failure', ['same-path', 'wrong-budget', 'trial-reuse', 'call-reuse', 'population', 'symlink'])
def test_ambiguous_or_unsafe_inventories_refuse(tmp_path, failure):
    paths = campaign(tmp_path)
    if failure == 'same-path':
        paths['one'].append(paths['one'][0])
    elif failure == 'wrong-budget':
        paths['one'], paths['two'] = paths['two'], paths['one']
    elif failure == 'trial-reuse':
        mutate(paths['one'][0], lambda value: value['trials'][0].update(trial='legacy'))
    elif failure == 'call-reuse':
        mutate(paths['one'][0], lambda value: value['trials'][0]['supervisor']['provider_invocations'][0].update(invocation_id='legacy'))
    elif failure == 'population':
        mutate(paths['one'][0], lambda value: value.update(trial_count=2))
    else:
        target = paths['one'][0]
        link = target.parent / 'link.json'
        link.symlink_to(target)
        paths['one'][0] = link
    with pytest.raises(ValueError):
        comparison.collect(**paths)


def test_grok_reported_counts_and_event_history_do_not_override_missing_completeness(tmp_path):
    paths = campaign(tmp_path)
    mutate(paths['one'][0], lambda value: value['trials'][0]['supervisor'].update(provider_invocations=[_grok_invocation(complete=None)]))
    result = comparison.collect(**paths)
    one = result['budgets']['one-capsule@1']['aggregate']
    assert one['known_token_subtotals']['total_tokens'] == 42
    assert one['tokens']['total_tokens'] is None
    assert result['contrasts']['one-capsule@1']['candidate_minus_legacy_tokens'] is None


def test_complete_provider_usage_is_independent_of_prompt_byte_reduction(tmp_path):
    paths = campaign(tmp_path)
    mutate(paths['one'][0], lambda value: value['trials'][0]['supervisor']['provider_invocations'][0].update(model_prompt_bytes=999999))
    result = comparison.collect(**paths)
    assert result['contrasts']['one-capsule@1']['candidate_minus_legacy_tokens'] == -40
    assert result['token_savings_claimed'] is False


def test_cli_writes_closed_reports_without_dispatch(tmp_path, capsys):
    paths = campaign(tmp_path)
    output = tmp_path / 'comparison'
    comparison.main(['--legacy', str(paths['legacy'][0]), '--one', str(paths['one'][0]),
                     '--two', str(paths['two'][0]), '--output', str(output)])
    result = json.loads((output / 'comparison.json').read_text())
    assert result['provider_calls_by_collector'] == 0
    assert result['matched_official_success'] is True
    assert (output / 'comparison.md').is_file()
    assert json.loads(capsys.readouterr().out)['schema'] == comparison.SCHEMA
    with pytest.raises(ValueError, match='fresh exact'):
        comparison.main(['--legacy', str(paths['legacy'][0]), '--one', str(paths['one'][0]),
                         '--two', str(paths['two'][0]), '--output', str(output)])


def test_all_three_bounded_groups_are_required(tmp_path):
    paths = campaign(tmp_path)
    with pytest.raises(ValueError, match='requires 1..16'):
        comparison.collect(**{**paths, 'one': []})
    with pytest.raises(ValueError, match='requires 1..16'):
        comparison.collect(**{**paths, 'two': paths['two'] * 17})


@pytest.mark.parametrize('change', ['arithmetic', 'missing-total', 'wrong-model', 'wrong-provider'])
def test_complete_flags_do_not_qualify_inconsistent_usage_or_foreign_runtime(tmp_path, change):
    paths = campaign(tmp_path)
    def alter(value):
        call = value['trials'][0]['supervisor']['provider_invocations'][0]
        if change == 'arithmetic':
            call['native_rollout_usage']['usage']['total_tokens'] += 99
        elif change == 'missing-total':
            call['native_rollout_usage']['usage'].pop('total_tokens')
        elif change == 'wrong-model':
            call['model'] = 'foreign'
        else:
            call['provider'] = 'foreign'
    mutate(paths['one'][0], alter)
    result = comparison.collect(**paths)
    assert result['contrasts']['one-capsule@1']['candidate_minus_legacy_tokens'] is None
    if change in ('arithmetic', 'missing-total'):
        assert result['budgets']['one-capsule@1']['complete_registered_native_usage'] is False


@pytest.mark.parametrize('kind', ['valid', 'failed-provider', 'malformed', 'historical', 'unselected', 'unverified'])
def test_comparison_retains_content_free_validated_retrieval_and_unknown_history(tmp_path, kind):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_policy import dispatch_report

    mode = 'repository-source-raw@1'
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    journal = private / 'operations.jsonl'
    audit = mcp._Audit(journal)
    audit.record('initialize')
    payload = dict(generation=1, semantic_root_cid='PRIVATE-SEMANTIC-ROOT',
        semantic_state_cid='PRIVATE-SEMANTIC-STATE', source_tree_sha256='1' * 64,
        capsules=[{'source_text': 'PRIVATE SOURCE TEXT MUST NOT APPEAR'}])
    audit.record('tool', tool='repository_capsule_expand', returned_bytes=128,
        retrieval=mcp._retrieval_event('repository_capsule_expand', {'generation': 1}, payload, True))
    audit.close()
    usage = {**mcp.summarize_repository_audit(journal),
             'store_cleanup_complete': True, 'artifacts_retained_after_invocation': False}
    if kind == 'malformed':
        usage['retrieval_observation']['capsules_returned'] = 33
    elif kind == 'historical':
        usage.pop('retrieval_observation')

    def configure(config):
        if kind != 'unselected':
            config['agents'][0]['kwargs']['repository_source_policy'] = mode

    paths = dict(zip(('legacy', 'one', 'two'), ([attempt(tmp_path, budget, mutate_config=configure)]
        for budget in CAPSULE_BUDGETS)))
    fixture = dispatch_report(mode)
    fixture_call = fixture['provider_invocations'][-1]
    fixture_call['repository_source_policy']['usage'] = usage
    if kind == 'failed-provider':
        fixture_call['status'] = 'failed'

    def inject(receipt):
        report = receipt['trials'][0]['supervisor']
        report.update(repository_source_policy=mode, context_input_audit=fixture['context_input_audit'],
                      source_text=payload['capsules'][0]['source_text'])
        report['provider_invocations'][0].update(fixture_call)
    mutate(paths['one'][0], inject)
    if kind == 'unverified':
        mutate(paths['one'][0].parent / 'config.json', lambda config: config.update(timeout_multiplier=2))
    result = comparison.collect(**paths)
    observation = result['rows'][1]['repository_source_observation']
    if kind in ('unselected', 'unverified'):
        assert observation['status'] == 'unavailable'
        assert observation['dispatch_observation'] is None
    else:
        call = observation['dispatch_observation']['invocations'][0]
        if kind == 'historical':
            assert observation['status'] == 'unavailable'
            assert call['retrieval_observation'] is call['retrieval_observation_valid'] is None
            assert call['capsule_retrieval_observed'] is None
        elif kind == 'malformed':
            assert observation['status'] == 'partial'
            assert call['retrieval_observation_valid'] is call['capsule_retrieval_observed'] is False
            assert call['retrieval_observation'] is None
        else:
            assert observation['status'] == 'observed'
            assert call['retrieval_observation_valid'] is call['capsule_retrieval_observed'] is True
            assert call['retrieval_observation']['capsules_returned'] == 1
            assert call['retrieval_observation']['omitted_capsule_recovery_verified'] is False
            assert call['provider_attempt_status'] == ('failed' if kind == 'failed-provider' else 'provider_returned')
    encoded = json.dumps(result)
    for secret in ('PRIVATE SOURCE TEXT MUST NOT APPEAR', 'PRIVATE-SEMANTIC-ROOT',
                   'PRIVATE-SEMANTIC-STATE', '/authored/candidate', 'config_path', 'state_dir'):
        assert secret not in encoded
