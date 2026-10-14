"""Actual coding input inclusion and retained replay with authored local providers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.source_input_policy import POLICIES
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding.terminal_source_input_experiment import _native_prompt
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_source_input_policy as benchmark_policy
from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke


@pytest.fixture
def native(tmp_path):
    repository = tmp_path / 'repository'
    repository.mkdir()
    source = '''def calculate_total(values, limit=10):
    # This explanatory comment remains retrievable from the canonical source.
    total = 0
    for value in values:
        if value <= limit:
            total += value
    return total


def adjustment(value):
    # Preserve the public arithmetic contract and spelling.
    return value + 1
'''
    (repository / 'mod.py').write_text(source)
    (repository / 'required.md').write_text('Preserve calculation behavior and validate changes.\n')
    subprocess.run(['git', '-C', str(repository), 'init', '-q'], check=True)
    subprocess.run(['git', '-C', str(repository), 'add', 'required.md'], check=True)
    artifact = repository / '.runtime/semantic/worker-context.json'
    prepare_semantic_context(repository=repository, paths=['mod.py', 'required.md'], required_raw_paths=['required.md'],
        program_paths=['mod.py'], objective='Inspect calculations', task_id='SOURCE-INPUT-TEST',
        output=artifact.parent, worker_query='calculate_total adjustment',
        worker_selector='duckdb-query-symbol-name@1', worker_capsule_limit=8,
        worker_max_bytes=32768, context_input_tokens=8192)
    prompt = _native_prompt(repository, artifact, 'SOURCE-INPUT-TEST', ['mod.py', 'required.md'])
    return repository, artifact, prompt, source


def replay(case, receipt):
    parsed = audit._receipt({**receipt, 'phase': 'coding'}, case['workspace'].parent)
    text, _, checks = audit._model_projection(rendered=case['prompt'], receipt=parsed, repository=case['repository'])
    assert text == case['calls'][-1][0] and all(checks.values())
    assert checks['source_input_projection'] and checks['source_input_positive_inclusion']
    return parsed, checks


@pytest.mark.parametrize('policy', POLICIES[1:])
def test_actual_source_input_requires_no_helper_adoption_and_replays_after_deletion(dispatch_case, monkeypatch, policy):
    case = dispatch_case
    artifact = case['repository'] / '.runtime/semantic/worker-context.json'
    canonical = artifact.read_bytes()
    _, receipt = invoke(case, source_input_policy=policy)
    binding = receipt['source_input_view']
    assert receipt['source_input_dispatched'] is True
    assert binding['views_included'] > 0 and binding['block_bytes'] > 0
    assert binding['factor_ast'] is ('-ast-ref-' in policy)
    if 'compact' in policy:
        assert binding['compact_views_included'] > 0
    assert 'calculate_total' in case['calls'][-1][0]
    assert 'source_view_advisory_policy' not in receipt
    assert 'helper_operations_observed' not in binding
    assert artifact.read_bytes() == canonical
    assert receipt['native_prompt_sha256'] == hashlib.sha256(case['prompt'].encode()).hexdigest()
    parsed, _ = replay(case, receipt)
    (case['repository'] / 'mod.py').write_text('def calculate_total(values): return -1\n')
    subprocess.run(['git', '-C', str(case['repository']), 'worktree', 'remove', '--force', str(case['workspace'])],
        check=True, capture_output=True)
    monkeypatch.chdir(case['repository'].parent)
    reconstructed, _, checks = audit._model_projection(rendered=case['prompt'], receipt=parsed, repository=case['repository'])
    assert reconstructed == case['calls'][-1][0] and all(checks.values())
    assert artifact.read_bytes() == canonical


@pytest.mark.parametrize('policy', POLICIES[1:3])
def test_additive_composes_with_metadata_reply_and_optional_helpers(dispatch_case, policy):
    case = dispatch_case
    _, receipt = invoke(case, source_input_policy=policy, semantic_metadata_view='common-bindings@1',
        source_view_advisory_policy='source-view-raw@1', tool_output_policy='artifact-preview@2')
    replay(case, receipt)
    assert receipt['source_input_view']['views_included'] > 0
    assert receipt['source_view_advisory_policy']['usage']['helper_operations_observed'] is False
    assert receipt['tool_output_policy']['usage']['helper_operations_observed'] is False
    assert receipt['coding_reply_contract']['model_prompt_after_sha256'] == receipt['model_prompt_sha256']


def counter(count):
    return runner.PromptTokenCounter(provider='codex_cli', model='authored-model', tokenizer_id='authored',
        tokenizer_revision='local', special_token_policy='none', count_tokens=count)


@pytest.mark.parametrize('metadata', ['legacy', 'common-bindings@1'])
def test_tokenizer_native_fallback_retains_same_source_block_and_exact_audit(dispatch_case, metadata):
    case = dispatch_case
    _, receipt = invoke(case, source_input_policy=POLICIES[2], semantic_metadata_view=metadata,
        prompt_token_counter=counter(lambda _text: 1))
    assert receipt['prompt_selection']['selected_representation'] == 'native'
    assert receipt['semantic_translation'] is None
    assert receipt['source_input_view']['compact_views_included'] > 0
    replay(case, receipt)


@pytest.mark.parametrize('policy', POLICIES[3:])
@pytest.mark.parametrize('option', ['metadata', 'tokenizer', 'budget'])
def test_ast_profile_explicitly_refuses_unsupported_composition(dispatch_case, policy, option):
    kwargs = {'metadata': {'semantic_metadata_view': 'common-bindings@1'},
              'tokenizer': {'prompt_token_counter': counter(lambda _: 1)}, 'budget': {'max_input_tokens': 100}}[option]
    with pytest.raises(ValueError, match='AST reference v1 excludes metadata and tokenizer selection'):
        invoke(dispatch_case, source_input_policy=policy, **kwargs)
    assert dispatch_case['calls'] == []


@pytest.mark.parametrize('location', ['repository', 'workspace'])
@pytest.mark.parametrize('policy', [POLICIES[1], POLICIES[4]])
def test_late_source_change_before_dispatch_is_refused(dispatch_case, monkeypatch, location, policy, capsys):
    from ipfs_accelerate_py import llm_router
    case = dispatch_case
    def initialize(*_a, **_kw):
        (case[location] / 'mod.py').write_text('def calculate_total(values): return -1\n')
        return object()
    monkeypatch.setattr(llm_router, 'get_llm_provider', initialize)
    with pytest.raises(ValueError):
        invoke(case, source_input_policy=policy)
    assert case['calls'] == []
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert receipt['source_input_dispatched'] is False
    assert receipt['failure_phase'] == 'source_input_freshness'


@pytest.mark.parametrize('mutation', ['selection', 'format', 'projector', 'zero', 'historical', 'model_hash', 'dispatch'])
def test_source_receipt_tampering_cannot_qualify_actual_input(dispatch_case, mutation):
    case = dispatch_case
    _, original = invoke(case, source_input_policy=POLICIES[2])
    receipt = deepcopy(original)
    view = receipt['source_input_view']
    if mutation == 'selection': view['views'][0]['symbol'] = 'foreign'
    elif mutation == 'format': receipt['source_input_policy'] = POLICIES[1]
    elif mutation == 'projector': view['projector_sha256'] = '0' * 64
    elif mutation == 'zero': view['compact_views_included'] = 0
    elif mutation == 'historical': view['freshness_checked'] = False
    elif mutation == 'dispatch': receipt['source_input_dispatched'] = 1
    else: receipt['model_prompt_sha256'] = '0' * 64
    try:
        parsed = audit._receipt({**receipt, 'phase': 'coding'}, case['workspace'].parent)
        _, _, checks = audit._model_projection(rendered=case['prompt'], receipt=parsed, repository=case['repository'])
    except ValueError:
        return
    assert not all(checks.values())


def test_legacy_preserves_original_model_input_without_source_policy(dispatch_case):
    _, before = invoke(dispatch_case)
    observed = dispatch_case['calls'][-1][0]
    _, after = invoke(dispatch_case, source_input_policy='legacy')
    assert dispatch_case['calls'][-1][0] == observed
    assert before['model_prompt_sha256'] == after['model_prompt_sha256']
    assert 'source_input_policy' not in after and 'source_input_view' not in after


def test_complete_native_fallback_budget_counts_inline_code_and_wrappers(dispatch_case):
    case = dispatch_case
    padding = 255_000 - len(case['prompt'].encode())
    assert padding > 0
    with pytest.raises(ValueError, match='model prompt.*byte bound'):
        runner.run(prompt=case['prompt'] + ' ' * padding, provider='codex_cli', model='authored-model',
            timeout=30, max_output_tokens=4096, semantic_repository=case['repository'],
            source_input_policy=POLICIES[2], coding_reply_mode='ordinary-completion@1',
            prompt_token_counter=counter(lambda _: 1))
    assert case['calls'] == []


def test_tokenizer_side_effect_cannot_dispatch_stale_input(dispatch_case):
    case = dispatch_case
    def count(_text):
        (case['workspace'] / 'required.md').write_text('Changed required dependency after projection.\n')
        return 1
    with pytest.raises(ValueError):
        invoke(case, source_input_policy=POLICIES[2], prompt_token_counter=counter(count))
    assert case['calls'] == []


def test_collection_requires_positive_input_and_exact_correlated_audit(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, source_input_policy=POLICIES[2])
    parsed, checks = replay(case, receipt)
    identity = receipt['invocation_id']
    report = dict(source_input_policy=POLICIES[2], provider_invocations=[{'phase': 'planning'}, {**receipt, 'phase': 'coding'}],
        context_input_audit=dict(schema=audit.SCHEMA, status='verified_all_observed_model_inputs', all_observed_coding_inputs_verified=True,
            coding_receipts_seen=1, usable_coding_receipts=1, read_errors=[],
            invocations=[dict(invocation_id=identity, model_input_verified=True, status='verified_exact_model_input')],
            matches=[dict(invocation_id=identity, model_input_verified=True, model_input_checks=checks,
                source_input_audit_binding_sha256=audit.source_input_audit_binding_sha256({**receipt, 'phase': 'coding'}),
                model_prompt_sha256=receipt['model_prompt_sha256'], model_prompt_bytes=receipt['model_prompt_bytes'])]))
    observation = benchmark_policy.source_input_dispatch_observation(report, POLICIES[2])
    assert observation['effective_input_observed'] is True
    assert observation['helper_adoption_established'] is False
    assert observation['model_attention_established'] is False
    assert observation['token_savings_established'] is False
    for mutation in ('no_audit', 'no_inclusion', 'duplicate', 'planning', 'zero', 'other_format', 'unknown_phase', 'no_projection', 'not_dispatched', 'missing_dispatch', 'missing_binding', 'planning_dispatch', 'audit_schema', 'unknown_audit_match'):
        changed = deepcopy(report)
        if mutation == 'no_audit': changed.pop('context_input_audit')
        elif mutation == 'no_inclusion': changed['context_input_audit']['matches'][0]['model_input_checks']['source_input_positive_inclusion'] = False
        elif mutation == 'duplicate': changed['provider_invocations'].append(deepcopy(changed['provider_invocations'][1]))
        elif mutation == 'planning': changed['provider_invocations'][0]['source_input_view'] = None
        elif mutation == 'zero': changed['provider_invocations'][1]['source_input_view']['compact_views_included'] = 0
        elif mutation == 'other_format': changed['provider_invocations'][1]['source_input_policy'] = POLICIES[1]
        elif mutation == 'unknown_phase': changed['provider_invocations'][0]['phase'] = 'unknown'
        elif mutation == 'not_dispatched': changed['provider_invocations'][1]['source_input_dispatched'] = False
        elif mutation == 'missing_dispatch': changed['provider_invocations'][1].pop('source_input_dispatched')
        elif mutation == 'missing_binding': changed['context_input_audit']['matches'][0].pop('source_input_audit_binding_sha256')
        elif mutation == 'planning_dispatch': changed['provider_invocations'][0]['source_input_dispatched'] = True
        elif mutation == 'audit_schema': changed['context_input_audit']['schema'] = 'foreign@1'
        elif mutation == 'unknown_audit_match': changed['context_input_audit']['matches'].append({'invocation_id': 'unknown'})
        else: changed['context_input_audit']['matches'][0]['model_input_checks']['source_input_projection'] = False
        assert benchmark_policy.source_input_dispatch_observation(changed, POLICIES[2])['effective_input_observed'] is False


def test_failed_provider_boundary_keeps_attempted_source_input_distinct_from_success(dispatch_case, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    seen = []
    def failed(model_prompt, **_kwargs):
        seen.append(model_prompt)
        raise TimeoutError('authored provider failure')
    monkeypatch.setattr(llm_router, 'generate_text', failed)
    with pytest.raises(TimeoutError, match='authored provider failure'):
        invoke(dispatch_case, source_input_policy=POLICIES[2])
    assert len(seen) == 1
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert receipt['source_input_dispatched'] is True
    assert receipt['status'] == 'failed' and receipt['failure_phase'] == 'provider_invocation'
    assert receipt['completion_authority'] is False


def _retained_source_state(case, tmp_path):
    """Persist real native context and the corresponding historical audit inputs."""
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import (
        PROFILE, instruction_sha256, task_profile_bytes,
    )
    root = case['repository']
    artifact = root / '.runtime/semantic/worker-context.json'
    semantic = json.loads(artifact.read_text())
    task = semantic['task_id']
    query = 'Inspect calculations'
    sources = {path: {'sha256': item['sha256']} for path, item in semantic['manifest'].items()}
    profile = {'schema': 'terminal-public-task-profile@1', 'instruction_sha256': instruction_sha256(query),
        'input_paths': ['mod.py'], 'outputs': [{'path': 'mod.py', 'effect': 'modify', 'media_type': 'text/x-python'}]}
    sources[PROFILE] = {'sha256': hashlib.sha256(task_profile_bytes(profile)).hexdigest()}
    expected = {'task_cid': 'cid:task', 'semantic_root_cid': semantic['semantic_root_cid'],
        'world_snapshot_cid': 'cid:world', 'index_id': 'cid:vectors'}
    prepared = {'repository': str(root), 'query': query, 'spec': {'task_key': task},
        'worker_inputs': sorted(semantic['manifest']), 'task_profile': profile,
        'manifest': {'payload': {'sources': sources}}}
    retrieval = {'task_id': task, 'index_id': expected['index_id'], 'result_id': 'cid:result',
        'query_id': 'cid:query', 'query_text': query, 'source_sha256': {'mod.py': sources['mod.py']['sha256']},
        'status': 'current', 'nomination_only': True, 'semantic_authority': False,
        'execution_authority': False, 'completion_authority': False}
    world = {'semantic_root_cid': semantic['semantic_root_cid'], 'world_snapshot_cid': expected['world_snapshot_cid'],
        'plan_projection_cid': 'cid:projection', 'intent_freshness_checked': False,
        'tasks': [{'task_cid': expected['task_cid'], 'body': {'local_planning_contract_cid': 'cid:contract'}}],
        'execution_authority': False, 'completion_authority': False}
    refs = []
    for kind, payload in [('semantic-context', semantic), ('code-retrieval-context', retrieval),
                          ('intent-world-context', world)]:
        text = artifact.read_text() if kind == 'semantic-context' else json.dumps(payload, ensure_ascii=False)
        refs.extend(build_text_context_references(text, reference_prefix=kind, kind=kind,
            path=artifact.relative_to(root).as_posix() if kind == 'semantic-context' else '',
            repository_id='repo:test', tree_id='tree:test', required=True, chunk_bytes=1201))
    compiled = ContextCompiler(ContextBudget(max_input_tokens=65536, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(repository_id='repo:test',
        tree_id='tree:test', objective_id=task, objective_revision='sha256:task', policy_id='policy:test',
        policy_revision='sha256:policy', caller='supervisor:test', stage='implementation', goal={'id': task},
        authority={'mode': 'candidate_only', 'completion_authority': False},
        scope={'allowed_paths': sorted(semantic['manifest'])}, acceptance={'criteria': ['pending native validation']},
        evidence=refs)
    case['prompt'] = render_context_capsule(compiled.capsule)
    state = tmp_path / 'retained-state'
    folder = state / 'launch/implementation-logs'
    folder.mkdir(parents=True)
    (folder / 'source-input-base-context-capsule.json').write_text(compiled.capsule.to_json())
    (state / 'prepared.json').write_text(json.dumps(prepared))
    (state / 'context-result.json').write_text(json.dumps(expected))
    return state


@pytest.mark.parametrize('policy', [POLICIES[2], POLICIES[4]])
def test_retained_audit_subprocess_binds_complete_receipt_against_stale_match_reuse(
        dispatch_case, tmp_path, monkeypatch, policy):
    case = dispatch_case
    state = _retained_source_state(case, tmp_path)
    options = {'semantic_metadata_view': 'common-bindings@1'} if policy == POLICIES[2] else {}
    _, receipt = invoke(case, source_input_policy=policy, **options)
    receipt['phase'] = 'coding'
    subprocess.run(['git', '-C', str(case['repository']), 'worktree', 'remove', '--force', str(case['workspace'])],
        check=True, capture_output=True)
    monkeypatch.chdir(case['repository'])
    observed = audit.collect_terminal_context_audit(state=state, receipts=[receipt],
        workspace_root=case['workspace'].parent, timeout_seconds=15)
    assert observed['status'] == 'verified_all_observed_model_inputs', observed
    assert observed['provider_calls'] == 0 and observed['raw_prompts_exported'] is False
    for match in observed['matches']:
        assert match['source_input_audit_binding_sha256'] == audit.source_input_audit_binding_sha256(receipt)
        assert all(type(value) is bool and value for value in match['model_input_checks'].values())
    report = dict(source_input_policy=policy, provider_invocations=[receipt], context_input_audit=observed)
    assert benchmark_policy.source_input_dispatch_observation(report, policy)['effective_input_observed'] is True
    original_counts = (receipt['source_input_view']['views_included'], receipt['source_input_view']['compact_views_included'])
    for mutation in ('content', 'selection', 'workspace', 'schema', 'provider', 'purpose',
                     'policy', 'translation', 'reply', 'dispatch', 'missing_digest', 'other_digest'):
        changed = deepcopy(report)
        row = changed['provider_invocations'][0]
        if mutation == 'content': row['source_input_view']['views'][0]['content_sha256'] = '0' * 64
        elif mutation == 'selection': row['source_input_view']['ordered_selection_sha256'] = '0' * 64
        elif mutation == 'workspace': row['workspace'] = str(case['workspace'].parent / 'different-worker')
        elif mutation == 'schema': row['schema'] = 'unrelated@1'
        elif mutation == 'provider': row['provider'] = 'other'
        elif mutation == 'purpose': row['purpose'] = 'planning'
        elif mutation == 'policy':
            changed['source_input_policy'] = row['source_input_policy'] = POLICIES[2] if policy == POLICIES[4] else POLICIES[4]
            view = row['source_input_view']
            view['factor_ast'] = not view['factor_ast']
            view['ast_slots_factored'] = len(view['views']) if view['factor_ast'] else 0
            view['ast_bytes_factored'] = sum(item['ast_bytes'] for item in view['views']) if view['factor_ast'] else 0
            row.pop('semantic_metadata_view', None)
        elif mutation == 'translation': row['semantic_translation']['translation_cid'] = 'cid:changed'
        elif mutation == 'reply': row['coding_reply_contract']['response_validated'] = not row['coding_reply_contract']['response_validated']
        elif mutation == 'dispatch': row['source_input_dispatched'] = False
        elif mutation == 'missing_digest': changed['context_input_audit']['matches'][0].pop('source_input_audit_binding_sha256')
        elif mutation == 'other_digest': changed['context_input_audit']['matches'][0]['source_input_audit_binding_sha256'] = '0' * 64
        assert row['model_prompt_sha256'] == receipt['model_prompt_sha256']
        assert row['model_prompt_bytes'] == receipt['model_prompt_bytes']
        assert (row['source_input_view']['views_included'], row['source_input_view']['compact_views_included']) == original_counts
        actual = benchmark_policy.source_input_dispatch_observation(changed, changed['source_input_policy'])
        assert actual['effective_input_observed'] is False, mutation
        if mutation in ('content', 'selection', 'workspace', 'policy', 'translation', 'reply'):
            # These still meet local shape/eligibility checks. The audit digest
            # must be what prevents reuse of the earlier successful audit.
            assert actual['invocations'][0]['policy_binding_valid'] is True, mutation
