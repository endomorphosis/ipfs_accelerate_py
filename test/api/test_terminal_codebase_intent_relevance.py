"""Synthetic post-fit relevance controls; no fit or model/checker execution.

The original structural fixture supplies actual native IntentIR and passive AST
features. The scores, labels, and lexical hits here are authored unit controls,
not measurements of trained ranking utility or semantic correctness.
"""
from copy import deepcopy
import ast
import hashlib

import pytest

from benchmarks.agent_supervisor.container_coding import (
    terminal_codebase_intent_relevance as relevance,
    terminal_codebase_intent_training_join as join,
)
from test.api.test_terminal_codebase_intent_training_join import (
    ae, digest, inputs, reseal_artifacts,  # noqa: F401
)


def _expanded(arguments):
    """Keep an irrelevant highest-scoring function outside the authored pool."""
    arguments = deepcopy(arguments)
    raw = (arguments['source_records'][0]['source_text'].encode()
        + b'\ndef unrelated(value):\n    return value + 2\n\nclass Bucket:\n    pass\n')
    source_hash = hashlib.sha256(raw).hexdigest()
    ledger = {'bottle.py': source_hash}
    arguments['source_records'][0].update(source_text=raw.decode(),
        source_sha256=source_hash, bytes=len(raw))
    rows, unsupported = ae._features({'bottle.py': raw}, ['bottle.py'], 1024)
    for owner in (arguments['features'], arguments['training_receipt'], arguments['learner']):
        owner['source_hashes'] = deepcopy(ledger)
    arguments['features'].update(rows=rows, unsupported=unsupported)
    errors = {'unrelated': 0.9, '_hkey': 0.2, '_hval': 0.1}
    ranks = sorted([{key: row[key] for key in ('row_id', 'path', 'symbol', 'line')}
        | {'reconstruction_error': errors[row['symbol']], 'latent': [0.25, 0.25]}
        for row in rows], key=lambda row: (-row['reconstruction_error'], row['row_id']))
    by_id = {row['row_id']: row for row in ranks}
    mean = sum(by_id[row['row_id']]['reconstruction_error'] for row in rows) / len(rows)
    arguments['learned_index'].update(ranks=ranks, mean_reconstruction_error=mean)
    for owner in (arguments['training_receipt'], arguments['learner']):
        owner['sample_count'] = len(rows)
        owner['metrics']['after_reconstruction_loss'] = mean
    arguments['source_context'].update(source_sha256=source_hash,
        source_bytes=len(raw), training_source_hashes=deepcopy(ledger))
    arguments['match_result']['current_source_snapshot'].update(
        source_sha256=source_hash, source_bytes=len(raw))
    zero = next(control for control in arguments['model_controls']
        if control['name'] == 'zero_heads')
    zero['ranking'] = sorted([
        {key: row[key] for key in ('row_id', 'path', 'symbol', 'line')}
        | {'latent': [0.0, 0.0], 'reconstruction_error':
            sum(value * value for value in row['features']) / len(ae.FEATURES)}
        for row in rows], key=lambda row: (-row['reconstruction_error'], row['row_id']))
    reseal_artifacts(arguments)
    units = {unit['symbol']: unit for unit in relevance._inventory(arguments)}
    arguments['fixed_candidates']['lexical'] = {
        'schema': 'supervisor-code-retrieval-context@1', 'status': 'current',
        'stale_paths': [], 'query_text': arguments['match_result']['intent_source']['text'],
        'source_sha256': deepcopy(ledger), 'hits': [_hit(units['_hkey'], 1, 0.9)]}
    return arguments


def _hit(unit, rank, score):
    return {'row_id': 'synthetic-lexical:' + unit['symbol'], 'path': unit['path'],
        'symbol': 'bottle.' + unit['symbol'], 'line_start': unit['line'],
        'line_end': unit['end_line'], 'rank': rank, 'score': score}


def _manifest(expected, arguments, relevant=('_hkey', '_hval')):
    units = {unit['symbol']: unit for unit in relevance._inventory(arguments)}
    return {'schema': relevance.LABEL_SCHEMA,
        'label_policy': 'postfit_authored_development',
        'review_ref': 'synthetic:authored-after-inert-artifacts-no-runtime-fit',
        'join_sha256': expected['join_sha256'],
        'source_context_sha256': arguments['match_result']['current_source_snapshot']['source_context_sha256'],
        'candidate_universe_sha256': expected['full_trained_row_inventory_sha256'],
        'cases': [
            {'case_id': 'header-navigation', 'statement_ids': ['repair-goal'],
             'label_status': 'positive_units',
             'relevant_units': [deepcopy(units[symbol]) for symbol in relevant]},
            {'case_id': 'retain-runtime-guards', 'statement_ids': ['retain-guards'],
             'label_status': 'unjudged', 'relevant_units': []}]}


@pytest.fixture
def evaluation_inputs(inputs):
    arguments = _expanded(inputs)
    expected = join.build_terminal_intent_training_join(**arguments)
    return {'expected_join': expected, 'original_arguments': arguments,
        'label_manifest': _manifest(expected, arguments)}


def _evaluate(values):
    return relevance.evaluate_terminal_codebase_intent_relevance(**values)


def _positive(receipt):
    return next(case for case in receipt['cases'] if case['case_id'] == 'header-navigation')


def _reseal(receipt):
    receipt['evaluation_sha256'] = 'sha256:' + digest(
        {key: value for key, value in receipt.items() if key != 'evaluation_sha256'})
    return receipt


def test_complete_universe_includes_irrelevant_function_before_authored_focus(evaluation_inputs):
    result = _evaluate(evaluation_inputs)
    assert result['schema'] == relevance.SCHEMA
    assert result['candidate_universe_count'] == 3
    assert result['authored_focus_pool_count'] == 2
    assert {unit['symbol'] for unit in result['candidate_universe']} == {'_hkey', '_hval', 'unrelated'}
    arguments = evaluation_inputs['original_arguments']
    row_ids = {row['row_id'] for row in arguments['features']['rows']}
    assert {unit['feature_row_id'] for unit in result['candidate_universe']} == row_ids
    units = {unit['symbol']: unit for unit in result['candidate_universe']}
    raw = arguments['source_records'][0]['source_text'].encode()
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    nodes = {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
    for name, node in nodes.items():
        unit = units[name]
        assert (unit['path'], unit['line'], unit['end_line']) == ('bottle.py', node.lineno, node.end_lineno)
        assert unit['source_sha256'] == hashlib.sha256(raw).hexdigest()
        assert unit['source_ast_sha256'] == hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        span = unit['source_span']
        assert (span['start_byte'], span['end_byte']) == (
            offsets[node.lineno - 1] + node.col_offset,
            offsets[node.end_lineno - 1] + node.end_col_offset)
        assert span['sha256'] == hashlib.sha256(raw[span['start_byte']:span['end_byte']]).hexdigest()
    methods = result['ranking_methods']
    for name in ('trained', 'zero_heads', 'shuffled_order'):
        assert methods[name]['ranking_count'] == 3 and methods[name]['ranking_complete'] is True
        assert {row['feature_row_id'] for row in methods[name]['rows']} == row_ids
    assert methods['trained']['rows'][0]['feature_row_id'] == units['unrelated']['feature_row_id']
    metrics = _positive(result)['metrics']
    assert metrics['trained']['first_relevant_rank'] == 2
    assert metrics['trained']['reciprocal_first_rank'] == 0.5
    assert metrics['trained']['recall_at']['1']['value'] == 0.0
    assert metrics['trained']['recall_at']['5']['value'] == metrics['trained']['recall_at']['10']['value'] == 1.0
    assert metrics['shuffled_order']['first_relevant_rank'] == 1
    assert metrics['shuffled_order']['recall_at']['1']['value'] == 0.5


def test_positive_labels_can_reference_units_outside_the_authored_focus(evaluation_inputs):
    values = deepcopy(evaluation_inputs)
    values['label_manifest'] = _manifest(values['expected_join'], values['original_arguments'], ('unrelated',))
    result = _evaluate(values)
    assert result['authored_focus_pool_count'] == 2
    assert _positive(result)['metrics']['trained']['reciprocal_first_rank'] == 1.0
    assert _positive(result)['metrics']['shuffled_order']['reciprocal_first_rank'] == pytest.approx(1 / 3)
    assert result['ranking_methods'] == _evaluate(evaluation_inputs)['ranking_methods']
    assert result['rankings_conditioned_on_case_labels'] is False


def test_model_off_and_unjudged_requirements_have_null_metrics(evaluation_inputs):
    result = _evaluate(evaluation_inputs)
    assert result['ranking_methods']['model_off']['rows'] == []
    off = _positive(result)['metrics']['model_off']
    assert off['ranking_count'] == 0 and off['ranking_complete'] is False
    assert off['first_relevant_rank'] is off['reciprocal_first_rank'] is None
    assert all(row['value'] is None and row['relevant_found'] is None
        and row['cutoff_covered'] is False for row in off['recall_at'].values())
    unjudged = next(case for case in result['cases'] if case['label_status'] == 'unjudged')
    assert unjudged['statement_ids'] == ['retain-guards'] and unjudged['relevant_units'] == []
    for metric in unjudged['metrics'].values():
        assert metric['first_relevant_rank'] is metric['reciprocal_first_rank'] is None
        assert all(row['value'] is None and row['positive_label_count'] is None
            for row in metric['recall_at'].values())


def test_nonfunction_lexical_hit_keeps_original_position_and_censored_tail(evaluation_inputs):
    values = deepcopy(evaluation_inputs)
    arguments = values['original_arguments']
    units = {unit['symbol']: unit for unit in relevance._inventory(arguments)}
    bucket = next(node for node in ast.parse(arguments['source_records'][0]['source_text']).body
        if isinstance(node, ast.ClassDef))
    arguments['fixed_candidates']['lexical']['hits'] = [
        {'row_id': 'synthetic-lexical:Bucket', 'path': 'bottle.py', 'symbol': 'bottle.Bucket',
         'line_start': bucket.lineno, 'line_end': bucket.end_lineno, 'rank': 1, 'score': 0.95},
        _hit(units['_hkey'], 2, 0.9)]
    values['expected_join'] = join.build_terminal_intent_training_join(**arguments)
    values['label_manifest'] = _manifest(values['expected_join'], arguments)
    result = _evaluate(values)
    lexical = result['ranking_methods']['lexical']
    assert lexical['ranking_count'] == 2 and lexical['ranking_complete'] is False
    assert lexical['rows'][0]['feature_row_id'] is None
    assert lexical['rows'][0]['disposition'] == 'outside_function_feature_universe'
    assert lexical['rows'][1]['position'] == 2
    metric = _positive(result)['metrics']['lexical']
    assert metric['first_relevant_rank'] == 2 and metric['reciprocal_first_rank'] == 0.5
    assert metric['recall_at']['1']['value'] == 0.0
    assert metric['recall_at']['5']['value'] is metric['recall_at']['10']['value'] is None
    assert metric['recall_at']['5']['cutoff_covered'] is False
    assert result['candidate_universe_count'] == 3


def test_missing_positive_in_short_lexical_prefix_is_unknown_not_failure(evaluation_inputs):
    values = deepcopy(evaluation_inputs)
    values['label_manifest'] = _manifest(values['expected_join'], values['original_arguments'], ('_hval',))
    lexical = _positive(_evaluate(values))['metrics']['lexical']
    assert lexical['ranking_count'] == 1 and lexical['ranking_complete'] is False
    assert lexical['first_relevant_rank'] is lexical['reciprocal_first_rank'] is None
    assert lexical['reciprocal_first_rank_bounds'] == [0.0, 0.5]
    assert lexical['recall_at']['1']['value'] == 0.0
    assert lexical['recall_at']['5']['value'] is lexical['recall_at']['10']['value'] is None


def test_replay_detaches_inputs_and_keeps_every_native_residual_and_authority(evaluation_inputs):
    pristine = deepcopy(evaluation_inputs)
    result = _evaluate(evaluation_inputs)
    assert evaluation_inputs == pristine
    assert result['original_native_match'] == pristine['expected_join']['original_native_match']
    assert result['residual_requirements'] == pristine['expected_join']['residual_requirements']
    assert len(result['residual_requirements']) == 2
    assert all(result[name] is False for name in join._AUTHORITY)
    assert result['blind_holdout_evaluated'] is result['case_independence_established'] is False
    assert result['live_checkout_or_native_storage_verified_here'] is result['ranking_gain_generalizes'] is False
    assert result['trained_inference_replayed_here'] is False
    assert result['training_inference_search_SQL_checker_or_worker_operations_here'] == 0
    assert result['label_manifest_sha256'] == digest(pristine['label_manifest'])
    assert result['evaluation_sha256'] == 'sha256:' + digest(
        {key: value for key, value in result.items() if key != 'evaluation_sha256'})
    assert relevance.validate_terminal_codebase_intent_relevance(result, **evaluation_inputs) == result
    result['candidate_universe'][0]['source_span']['sha256'] = '0' * 64
    result['residual_requirements'].clear()
    result['original_native_match']['intent_source']['text'] = 'forged instruction'
    assert evaluation_inputs == pristine


def test_new_development_labels_change_identity_without_changing_rankings(evaluation_inputs):
    first = _evaluate(evaluation_inputs)
    values = deepcopy(evaluation_inputs)
    values['label_manifest'] = _manifest(values['expected_join'], values['original_arguments'], ('_hval',))
    second = _evaluate(values)
    assert second['label_manifest_sha256'] != first['label_manifest_sha256']
    assert second['evaluation_sha256'] != first['evaluation_sha256']
    assert second['ranking_methods'] == first['ranking_methods']
    assert second['residual_requirements'] == first['residual_requirements']
    with pytest.raises(relevance.RelevanceEvaluationError, match='original-input replay'):
        relevance.validate_terminal_codebase_intent_relevance(second, **evaluation_inputs)


@pytest.mark.parametrize('part', ['metric', 'source_unit', 'residuals', 'authority'])
def test_resealed_evaluation_cannot_replace_independently_supplied_original_inputs(evaluation_inputs, part):
    forged = _evaluate(evaluation_inputs)
    if part == 'metric':
        _positive(forged)['metrics']['trained']['reciprocal_first_rank'] = 1.0
    elif part == 'source_unit':
        forged['candidate_universe'][0]['line'] += 1
    elif part == 'residuals':
        forged['residual_requirements'].clear()
    else:
        forged['proof_authority'] = True
    _reseal(forged)
    with pytest.raises(relevance.RelevanceEvaluationError, match='original-input replay'):
        relevance.validate_terminal_codebase_intent_relevance(forged, **evaluation_inputs)


@pytest.mark.parametrize('part', ['heldout_policy', 'extra_field', 'unknown_statement',
    'duplicate_case', 'duplicate_positive', 'join_root', 'context_root', 'universe_root'])
def test_labels_require_postfit_policy_unique_cases_and_original_native_roots(evaluation_inputs, part):
    values = deepcopy(evaluation_inputs)
    labels = values['label_manifest']
    if part == 'heldout_policy':
        labels['label_policy'] = 'blind_holdout'
    elif part == 'extra_field':
        labels['heldout'] = True
    elif part == 'unknown_statement':
        labels['cases'][0]['statement_ids'] = ['unmentioned-new-requirement']
    elif part == 'duplicate_case':
        labels['cases'].append(deepcopy(labels['cases'][0]))
    elif part == 'duplicate_positive':
        labels['cases'][0]['relevant_units'].append(deepcopy(labels['cases'][0]['relevant_units'][0]))
    else:
        field = {'join_root': 'join_sha256', 'context_root': 'source_context_sha256',
            'universe_root': 'candidate_universe_sha256'}[part]
        labels[field] = 'sha256:' + '0' * 64
    with pytest.raises(relevance.RelevanceEvaluationError):
        _evaluate(values)


@pytest.mark.parametrize('part', ['path', 'line', 'ast', 'span', 'line_bool', 'start_byte_bool'])
def test_positive_source_identity_cannot_be_reassigned_under_same_feature_id(evaluation_inputs, part):
    values = deepcopy(evaluation_inputs)
    unit = values['label_manifest']['cases'][0]['relevant_units'][0]
    if part == 'path':
        unit['path'] = 'different/bottle.py'
    elif part == 'line':
        unit['line'] += 1
    elif part == 'ast':
        unit['source_ast_sha256'] = '0' * 64
    elif part == 'line_bool':
        assert unit['line'] == 1
        unit['line'] = True
    elif part == 'start_byte_bool':
        assert unit['source_span']['start_byte'] == 0
        unit['source_span']['start_byte'] = False
    else:
        unit['source_span']['end_byte'] -= 1
    with pytest.raises(relevance.RelevanceEvaluationError, match='source/AST/feature binding'):
        _evaluate(values)


@pytest.mark.parametrize('part', ['source', 'checkpoint', 'rank_identity'])
def test_evaluator_refuses_stale_source_or_model_before_scoring(evaluation_inputs, part):
    values = deepcopy(evaluation_inputs)
    arguments = values['original_arguments']
    if part == 'source':
        arguments['source_records'][0]['source_text'] += '\n# stale after join\n'
    elif part == 'checkpoint':
        arguments['checkpoint']['weights'][0][0][0] += 0.1
    else:
        ranks = arguments['learned_index']['ranks']
        ranks[0]['symbol'], ranks[1]['symbol'] = ranks[1]['symbol'], ranks[0]['symbol']
        reseal_artifacts(arguments)
    with pytest.raises(ValueError):
        _evaluate(values)


def test_resealed_same_feature_operator_change_cannot_reuse_old_function_rows(evaluation_inputs):
    values = deepcopy(evaluation_inputs)
    arguments = values['original_arguments']
    raw = arguments['source_records'][0]['source_text'].replace('>=', '>').encode()
    rows, _ = ae._features({'bottle.py': raw}, ['bottle.py'], 1024)
    assert [row['features'] for row in rows] == [row['features'] for row in arguments['features']['rows']]
    assert [row['ast_sha256'] for row in rows] != [row['ast_sha256'] for row in arguments['features']['rows']]
    source_hash = hashlib.sha256(raw).hexdigest()
    ledger = {'bottle.py': source_hash}
    arguments['source_records'][0].update(source_text=raw.decode(), source_sha256=source_hash, bytes=len(raw))
    for owner in (arguments['features'], arguments['training_receipt'], arguments['learner']):
        owner['source_hashes'] = deepcopy(ledger)
    arguments['source_context'].update(source_sha256=source_hash, source_bytes=len(raw), training_source_hashes=ledger)
    arguments['match_result']['current_source_snapshot'].update(source_sha256=source_hash, source_bytes=len(raw))
    reseal_artifacts(arguments)
    with pytest.raises(ValueError, match='current source AST replay'):
        _evaluate(values)


def test_resealed_zero_control_cannot_choose_a_label_friendly_order(evaluation_inputs):
    values = deepcopy(evaluation_inputs)
    arguments = values['original_arguments']
    zero = next(control for control in arguments['model_controls'] if control['name'] == 'zero_heads')
    zero['ranking'].reverse()
    values['expected_join'] = join.build_terminal_intent_training_join(**arguments)
    values['label_manifest'] = _manifest(values['expected_join'], arguments)
    with pytest.raises(relevance.RelevanceEvaluationError, match='zero diagnostic ordering'):
        _evaluate(values)


@pytest.mark.parametrize('part', ['module_prefix', 'end_line', 'duplicate_row'])
def test_lexical_mapping_refuses_wrong_module_span_or_duplicate_hit(evaluation_inputs, part):
    values = deepcopy(evaluation_inputs)
    arguments = values['original_arguments']
    hits = arguments['fixed_candidates']['lexical']['hits']
    if part == 'module_prefix':
        hits[0]['symbol'] = 'other_module._hkey'
    elif part == 'end_line':
        hits[0]['line_end'] += 1
    else:
        hits.append({**deepcopy(hits[0]), 'rank': 2, 'score': 0.8})
    values['expected_join'] = join.build_terminal_intent_training_join(**arguments)
    values['label_manifest'] = _manifest(values['expected_join'], arguments)
    with pytest.raises(relevance.RelevanceEvaluationError):
        _evaluate(values)


def test_distinct_lexical_alias_rows_cannot_count_one_positive_function_twice(evaluation_inputs):
    values = deepcopy(evaluation_inputs)
    arguments = values['original_arguments']
    hits = arguments['fixed_candidates']['lexical']['hits']
    hits.append({**deepcopy(hits[0]), 'row_id': 'synthetic-lexical:alias',
        'rank': 2, 'score': 0.8})
    assert hits[0]['row_id'] != hits[1]['row_id']
    values['expected_join'] = join.build_terminal_intent_training_join(**arguments)
    values['label_manifest'] = _manifest(values['expected_join'], arguments)
    with pytest.raises(relevance.RelevanceEvaluationError, match='duplicate lexical feature association'):
        _evaluate(values)
