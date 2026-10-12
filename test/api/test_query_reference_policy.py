"""Actual query-reference coding input and exact retained audit with local providers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding.terminal_source_input_experiment import _native_prompt
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_query_reference_policy as benchmark_policy
from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke
from test.api.test_source_input_policy import _retained_source_state

POLICY = 'required-instruction-ref@1'


@pytest.fixture
def native(tmp_path):
    root = tmp_path / 'repository'
    root.mkdir()
    source = 'def calculate_total(values):\n    return sum(values)\n'
    instruction = 'Inspect calculate_total, preserve its public signature, and validate behavior with the existing tests.\n' * 24
    (root / 'mod.py').write_text(source)
    (root / 'required.md').write_text(instruction)
    subprocess.run(['git', '-C', str(root), 'init', '-q'], check=True)
    subprocess.run(['git', '-C', str(root), 'add', 'required.md'], check=True)
    artifact = root / '.runtime/semantic/worker-context.json'
    prepare_semantic_context(repository=root, paths=['mod.py', 'required.md'], required_raw_paths=['required.md'],
        program_paths=['mod.py'], objective='Inspect calculations', task_id='QUERY-REFERENCE-TEST',
        output=artifact.parent, worker_query=instruction, worker_selector='duckdb-query-symbol-name@1',
        worker_capsule_limit=8, worker_max_bytes=32768, context_input_tokens=8192)
    return root, artifact, _native_prompt(root, artifact, 'QUERY-REFERENCE-TEST', ['mod.py', 'required.md']), source


def _replay(case, receipt):
    parsed = audit._receipt({**receipt, 'phase': 'coding'}, case['workspace'].parent)
    reconstructed, _, checks = audit._model_projection(rendered=case['prompt'], receipt=parsed,
        repository=case['repository'])
    assert reconstructed == case['calls'][-1][0] and all(checks.values())
    assert checks['query_reference_projection'] and checks['query_reference_selection']
    return parsed, checks


@pytest.mark.parametrize('transport', ['supervisor-semantic-router-input@1', 'supervisor-semantic-router-input@2'])
def test_actual_reference_input_preserves_canonical_source_and_replays(dispatch_case, monkeypatch, transport):
    case = dispatch_case
    artifact = case['repository'] / '.runtime/semantic/worker-context.json'
    before = artifact.read_bytes()
    _, receipt = invoke(case, query_reference_policy=POLICY, semantic_transport_schema=transport)
    assert receipt['query_reference_selected'] is receipt['query_reference_dispatched'] is True
    assert receipt['query_reference_provider_attempted'] is True
    assert receipt['query_reference_view']['reference_count'] == 1
    assert receipt['query_reference_view']['source_transport_schema'] == transport
    assert '$required_raw_source' in case['calls'][-1][0]
    assert receipt['native_prompt_sha256'] == hashlib.sha256(case['prompt'].encode()).hexdigest()
    parsed, _ = _replay(case, receipt)
    subprocess.run(['git', '-C', str(case['repository']), 'worktree', 'remove', '--force', str(case['workspace'])],
        check=True, capture_output=True)
    monkeypatch.chdir(case['repository'])
    (case['repository'] / 'required.md').write_text('Historical source has changed.\n')
    reconstructed, _, checks = audit._model_projection(rendered=case['prompt'], receipt=parsed,
        repository=case['repository'])
    assert reconstructed == case['calls'][-1][0] and all(checks.values())
    assert artifact.read_bytes() == before


def test_native_tokenizer_fallback_is_audited_without_claiming_reference_treatment(dispatch_case):
    counter = runner.PromptTokenCounter(provider='codex_cli', model='authored-model', tokenizer_id='authored',
        tokenizer_revision='tie', special_token_policy='none', count_tokens=lambda _: 1)
    _, receipt = invoke(dispatch_case, query_reference_policy=POLICY, prompt_token_counter=counter)
    assert receipt['prompt_selection']['selected_representation'] == 'native'
    assert receipt['semantic_translation'] is None
    assert receipt['query_reference_selected'] is receipt['query_reference_dispatched'] is False
    assert receipt['query_reference_provider_attempted'] is True
    assert '$required_raw_source' not in dispatch_case['calls'][-1][0]
    _replay(dispatch_case, receipt)


@pytest.mark.parametrize('options', [{'semantic_metadata_view': 'common-bindings@1'},
    {'source_input_policy': 'source-input-raw@1'}, {'source_input_policy': 'source-input-ast-ref-compact@1'},
    {'purpose': 'planning'}, {'census_only': True}])
def test_incompatible_routes_refuse_before_provider(dispatch_case, options):
    with pytest.raises(ValueError, match='query reference'):
        runner.run(prompt=dispatch_case['prompt'], provider='codex_cli', model='authored-model', timeout=30,
            max_output_tokens=4096, semantic_repository=dispatch_case['repository'], query_reference_policy=POLICY,
            **options)
    assert dispatch_case['calls'] == []


@pytest.mark.parametrize('location', ['repository', 'workspace'])
@pytest.mark.parametrize('source_path', ['required.md', 'mod.py'])
@pytest.mark.parametrize('fallback', [False, True])
def test_complete_manifest_mutation_during_provider_setup_refuses_dispatch(
        dispatch_case, monkeypatch, capsys, location, source_path, fallback):
    from ipfs_accelerate_py import llm_router
    def provider(*_args, **_kwargs):
        (dispatch_case[location] / source_path).write_text('Changed outside the agent after projection.\n')
        return object()
    monkeypatch.setattr(llm_router, 'get_llm_provider', provider)
    options = {}
    if fallback:
        options['prompt_token_counter'] = runner.PromptTokenCounter(provider='codex_cli', model='authored-model',
            tokenizer_id='authored', tokenizer_revision='tie', special_token_policy='none', count_tokens=lambda _: 1)
    with pytest.raises(ValueError):
        invoke(dispatch_case, query_reference_policy=POLICY, **options)
    assert dispatch_case['calls'] == []
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert receipt['query_reference_selected'] is (not fallback)
    assert receipt['query_reference_dispatched'] is False
    assert receipt['query_reference_provider_attempted'] is False
    assert receipt['failure_phase'] == 'query_reference_freshness'
    parsed = audit._receipt({**receipt, 'phase': 'coding'}, dispatch_case['workspace'].parent)
    _, _, checks = audit._model_projection(rendered=dispatch_case['prompt'], receipt=parsed,
        repository=dispatch_case['repository'])
    assert checks['model_prompt_sha256'] is True
    assert checks['query_reference_provider_attempted'] is False
    assert not all(checks.values())


def test_full_model_budget_is_enforced_and_tokenizer_side_effect_cannot_bypass_freshness(dispatch_case):
    def changed(_text):
        (dispatch_case['workspace'] / 'required.md').write_text('Changed by external tokenizer code.\n')
        return 1
    counter = runner.PromptTokenCounter(provider='codex_cli', model='authored-model', tokenizer_id='authored',
        tokenizer_revision='mutation', special_token_policy='none', count_tokens=changed)
    with pytest.raises(ValueError):
        invoke(dispatch_case, query_reference_policy=POLICY, prompt_token_counter=counter)
    assert dispatch_case['calls'] == []


def test_complete_input_budget_refuses_before_provider(dispatch_case):
    counter = runner.PromptTokenCounter(provider='codex_cli', model='authored-model', tokenizer_id='authored',
        tokenizer_revision='bytes', special_token_policy='none', count_tokens=lambda text: len(text.encode()))
    with pytest.raises(ValueError):
        invoke(dispatch_case, query_reference_policy=POLICY, prompt_token_counter=counter, max_input_tokens=10)
    assert dispatch_case['calls'] == []


def test_complete_native_fallback_byte_bound_includes_workspace_and_reply(dispatch_case):
    case = dispatch_case
    counter = runner.PromptTokenCounter(provider='codex_cli', model='authored-model', tokenizer_id='authored',
        tokenizer_revision='tie', special_token_policy='none', count_tokens=lambda _: 1)
    padding = 255_000 - len(case['prompt'].encode())
    with pytest.raises(ValueError, match='model prompt.*byte bound'):
        runner.run(prompt=case['prompt'] + ' ' * padding, provider='codex_cli', model='authored-model',
            timeout=30, max_output_tokens=4096, semantic_repository=case['repository'],
            query_reference_policy=POLICY, coding_reply_mode='ordinary-completion@1', prompt_token_counter=counter)
    assert case['calls'] == []


def test_provider_failure_records_attempted_reference_without_claiming_success(dispatch_case, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    attempts = []
    def fail(model_prompt, **_kwargs):
        attempts.append(model_prompt)
        raise TimeoutError('Authored provider timeout')
    monkeypatch.setattr(llm_router, 'generate_text', fail)
    with pytest.raises(TimeoutError):
        invoke(dispatch_case, query_reference_policy=POLICY)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert len(attempts) == 1
    assert receipt['query_reference_provider_attempted'] is receipt['query_reference_dispatched'] is True
    assert receipt['status'] == 'failed' and receipt['completion_authority'] is False


def test_default_policy_preserves_legacy_model_bytes(dispatch_case):
    _, before = invoke(dispatch_case)
    _, after = invoke(dispatch_case, query_reference_policy='legacy')
    assert before['model_prompt_sha256'] == after['model_prompt_sha256']
    assert not any(key.startswith('query_reference') for key in after)


def _collect_trial(report, tmp_path):
    from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
    from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
    dataset, task = authored_task(tmp_path)
    output = tmp_path / 'collection'
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / 'archive', 'full', query_reference_policy=POLICY)
    (output / 'config.json').write_text(json.dumps(config))
    prepared = {'arm': 'full', 'dataset': str(dataset), 'prepared': True,
        'config_sha256': benchmark._hash(output / 'config.json'), 'task_input_sha256': benchmark._task_hashes(task),
        'model': benchmark.MODEL, 'reasoning_effort': 'high', 'cli_version': CLI_VERSION,
        'query_reference_policy': POLICY}
    (output / 'preparation.json').write_text(json.dumps(prepared))
    job = Path(config['jobs_dir']) / config['job_name']
    trial = job / 'authored-trial'
    (trial / 'agent').mkdir(parents=True)
    (job / 'result.json').write_text('{}')
    (trial / 'result.json').write_text(json.dumps({'task_name': benchmark.TASK,
        'config': {'task': {'path': str(task)}}, 'verifier_result': {'rewards': {'reward': 1.0}}}))
    (trial / 'agent/supervisor-result.json').write_text(json.dumps(report))
    return benchmark.collect(output)


@pytest.mark.parametrize('fallback', [False, True])
def test_actual_retained_audit_binds_reference_receipt_and_native_fallback(dispatch_case, tmp_path, monkeypatch, fallback):
    case = dispatch_case
    state = _retained_source_state(case, tmp_path)
    options = {}
    if fallback:
        options['prompt_token_counter'] = runner.PromptTokenCounter(provider='codex_cli', model='authored-model',
            tokenizer_id='authored', tokenizer_revision='tie', special_token_policy='none', count_tokens=lambda _: 1)
    _, receipt = invoke(case, query_reference_policy=POLICY, **options)
    receipt['phase'] = 'coding'
    subprocess.run(['git', '-C', str(case['repository']), 'worktree', 'remove', '--force', str(case['workspace'])],
        check=True, capture_output=True)
    monkeypatch.chdir(case['repository'])
    observed = audit.collect_terminal_context_audit(state=state, receipts=[receipt],
        workspace_root=case['workspace'].parent, timeout_seconds=15)
    assert observed['status'] == 'verified_all_observed_model_inputs', observed
    assert observed['matches'][0]['query_reference_audit_binding_sha256'] == audit.query_reference_audit_binding_sha256(receipt)
    assert all(type(value) is bool and value for value in observed['matches'][0]['model_input_checks'].values())
    report = dict(query_reference_policy=POLICY, provider_invocations=[receipt], context_input_audit=observed)
    observation = benchmark_policy.query_reference_dispatch_observation(report, POLICY)
    assert observation['effective_input_observed'] is (not fallback)
    assert observation['status'] == ('verified_native_fallback' if fallback else 'verified_input')
    collected = _collect_trial(report, tmp_path)
    assert collected['complete_single_trial_receipt'] is (not fallback)
    assert collected['trial_count'] == 1
    assert collected['trials'][0]['reward'] == {'reward': 1.0}
    assert collected['trials'][0]['supervisor']['provider_invocations'] == [receipt]
    assert collected['trials'][0]['supervisor']['provider_invocations'][0]['usage'] == receipt['usage']
    if fallback:
        changed = deepcopy(report)
        changed['provider_invocations'][0]['query_reference_selected'] = True
        changed['provider_invocations'][0]['query_reference_dispatched'] = True
        assert benchmark_policy.query_reference_dispatch_observation(changed, POLICY)['effective_input_observed'] is False
        changed = deepcopy(report)
        changed['provider_invocations'][0]['query_reference_provider_attempted'] = False
        refusal = benchmark_policy.query_reference_dispatch_observation(changed, POLICY)
        assert refusal['native_fallback_inputs'] == 0 and refusal['unverified_inputs'] == 1
        assert refusal['effective_input_observed'] is False
        return
    for mutation in ('source', 'workspace', 'schema', 'provider', 'purpose', 'translation', 'reply', 'dispatch', 'digest', 'not_attempted', 'missing_attempt'):
        changed = deepcopy(report)
        row = changed['provider_invocations'][0]
        if mutation == 'source': row['query_reference_view']['source_sha256'] = '0' * 64
        elif mutation == 'workspace': row['workspace'] = str(case['workspace'].parent / 'other-worker')
        elif mutation == 'schema': row['schema'] = 'foreign@1'
        elif mutation == 'provider': row['provider'] = 'other'
        elif mutation == 'purpose': row['purpose'] = 'planning'
        elif mutation == 'translation': row['semantic_translation']['translation_cid'] = 'cid:changed'
        elif mutation == 'reply': row['coding_reply_contract']['response_validated'] = not row['coding_reply_contract']['response_validated']
        elif mutation == 'dispatch': row['query_reference_dispatched'] = False
        elif mutation == 'digest': changed['context_input_audit']['matches'][0].pop('query_reference_audit_binding_sha256')
        elif mutation == 'not_attempted': row['query_reference_provider_attempted'] = False
        elif mutation == 'missing_attempt': row.pop('query_reference_provider_attempted')
        assert row['model_prompt_sha256'] == receipt['model_prompt_sha256']
        assert row['query_reference_view']['reference_count'] == 1
        assert benchmark_policy.query_reference_dispatch_observation(changed, POLICY)['effective_input_observed'] is False, mutation
