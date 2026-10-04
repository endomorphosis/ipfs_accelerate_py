"""Small real scratch-head fit plus pure replay and split exclusion controls.

The one module-scoped synthetic fit is unit evidence, not a benchmark score.
The public experiment owns its separately frozen corpus and fit artifacts.
"""
from copy import deepcopy
import math

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_corpus as corpus
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_ranker_training as head
from benchmarks.agent_supervisor.container_coding.terminal_codebase_intent_training_join import _digest
from test.api.test_terminal_codebase_intent_corpus import (
    make_inputs, _arguments, _source, _query, _judgment)


@pytest.fixture(scope='module')
def fitted():
    originals = make_inputs()
    frozen = corpus.build_terminal_intent_relevance_corpus(**originals)
    result = head.train_terminal_codebase_intent_ranker(
        corpus_receipt=frozen, original_inputs=originals, epochs=16)
    return originals, frozen, result


def _validate(result, originals, frozen, **kwargs):
    return head.validate_terminal_codebase_intent_ranker(
        result, corpus_receipt=frozen, original_inputs=originals, **kwargs)


def test_real_scratch_fit_decreases_pair_objective_and_keeps_full_unknown_bank(fitted):
    originals, frozen, result = fitted
    trace = result['training_receipt']['trace']
    assert len(trace) == 17 and trace[-1]['objective'] < trace[0]['objective']
    assert trace[-1]['gradient_norm'] < trace[0]['gradient_norm']
    assert all(right['objective'] <= left['objective'] for left, right in zip(trace, trace[1:]))
    assert any(weight != 0 for weight in result['checkpoint']['weights'])
    assert result['new_autoencoder_fit'] is False
    query = result['query_results'][0]
    assert query['candidate_count'] == 4
    for method, rows in query['rankings'].items():
        assert len(rows) == (0 if method == 'model_off' else 4)
        assert query['metrics'][method]['first_positive_rank'] is None if method == 'model_off' else True
    assert query['other_candidate_label'] == 'unjudged_not_implicit_negative'
    assert frozen['judgment_summary'][0]['unjudged_count'] == 2
    assert all(item['status'] == 'unknown' for item in query['residual_requirements'])
    assert result['asymptotic_optimizer_convergence_proved'] is False
    assert _validate(result, originals, frozen,
        expected_checkpoint_sha256=_digest(result['checkpoint']),
        expected_training_receipt_sha256=result['training_receipt']['receipt_sha256']) == result


def test_pair_gradient_matches_independent_central_difference():
    weights = [0.17, -0.09, 0.31]
    differences = [[0.2, 0.7, -0.1], [-0.4, 0.3, 0.5]]
    observation, gradient = head._objective(weights, differences)
    epsilon = 1e-6
    for index in range(len(weights)):
        lower, upper = list(weights), list(weights)
        lower[index] -= epsilon
        upper[index] += epsilon
        numeric = (head._objective(upper, differences)[0]['objective']
                   - head._objective(lower, differences)[0]['objective']) / (2 * epsilon)
        assert math.isclose(numeric, gradient[index], rel_tol=1e-6, abs_tol=1e-9)
    assert observation['objective'] == observation['pair_logistic_loss'] + observation['L2_penalty']


def test_validation_labels_do_not_change_training_membership_or_features():
    originals = make_inputs()
    heldout_source = _source('def compose(left, right):\n    return (left, right)\n\ndef finish(items):\n    return tuple(items)\n',
        'synthetic-heldout', 'heldout.py')
    heldout_query = _query('synthetic-heldout', 'tuple-inspection',
        instruction='Inspect a distinct tuple composition interface in the heldout initial source.',
        navigation='Find the tuple composition helper for two input arguments.')
    originals['source_records'].append(heldout_source)
    originals['query_records'].append(heldout_query)
    originals['split_assignments'][heldout_query['query_id']] = 'validation'
    originals['reviewed_judgments'].append(_judgment(heldout_query, heldout_source, 'compose', 'positive'))
    # Build full exact history from the enlarged independent originals.
    from test.api.test_terminal_codebase_intent_corpus import _history
    originals['historical_exposure'] = _history(originals['source_records'], originals['query_records'])
    frozen = corpus.build_terminal_intent_relevance_corpus(**originals)
    changed = deepcopy(originals)
    changed['reviewed_judgments'][-1] = _judgment(heldout_query, heldout_source, 'finish', 'positive')
    alternate = corpus.build_terminal_intent_relevance_corpus(**changed)
    first = head._prepare(frozen, originals)
    second = head._prepare(alternate, changed)
    assert first[3:] == second[3:]
    assert all(pair['task_role'] == 'train' for pair in first[3])
    assert heldout_query['query_id'] not in {pair['query_id'] for pair in first[3]}


@pytest.mark.parametrize('tamper', ['truncate', 'metric', 'weight', 'authority', 'train_pair', 'epoch_bool'])
def test_resealed_tampering_refuses_original_replay(fitted, tamper):
    originals, frozen, result = fitted
    bad = deepcopy(result)
    if tamper == 'truncate':
        bad['query_results'][0]['rankings']['trained'].pop()
    elif tamper == 'metric':
        bad['query_results'][0]['metrics']['trained']['first_positive_rank'] = 99
    elif tamper == 'weight':
        bad['checkpoint']['weights'][0] += 0.25
        bad['checkpoint']['weights_sha256'] = _digest(bad['checkpoint']['weights'])
    elif tamper == 'authority':
        bad['proof_authority'] = True
    elif tamper == 'train_pair':
        bad['training_receipt']['train_pairs'].append({'query_id': 'heldout', 'task_role': 'test'})
    else:
        bad['training_receipt']['trace'][0]['epoch'] = False
        content = {k: v for k, v in bad['training_receipt'].items() if k != 'receipt_sha256'}
        bad['training_receipt']['receipt_sha256'] = _digest(content)
    bad['result_sha256'] = _digest({k: v for k, v in bad.items() if k != 'result_sha256'})
    with pytest.raises((ValueError, KeyError)):
        _validate(bad, originals, frozen)


def test_external_actual_fit_pins_refuse_another_checkpoint_or_receipt(fitted):
    originals, frozen, result = fitted
    with pytest.raises(ValueError):
        _validate(result, originals, frozen, expected_checkpoint_sha256='0' * 64)
    with pytest.raises(ValueError):
        _validate(result, originals, frozen, expected_training_receipt_sha256='0' * 64)


def test_unjudged_queries_remain_null_with_complete_positions(fitted):
    originals, frozen, result = fitted
    prepared = head._prepare(frozen, originals)
    unknown = deepcopy(prepared[0])
    unknown['reviewed_judgments'] = []
    passive = head._output((unknown, *prepared[1:]), result['checkpoint'], result['training_receipt'])
    query = passive['query_results'][0]
    for method, metrics in query['metrics'].items():
        assert metrics['first_positive_rank'] is None
        assert metrics['reciprocal_first_rank'] is None
        assert all(value is None for value in metrics['recall_at'].values())
        assert len(query['rankings'][method]) == (0 if method == 'model_off' else 4)


@pytest.mark.parametrize('epochs', [True, 0, 129])
def test_invalid_epoch_limits_refuse_before_optimizer(fitted, epochs):
    originals, frozen, _ = fitted
    with pytest.raises(ValueError):
        head.train_terminal_codebase_intent_ranker(corpus_receipt=frozen,
            original_inputs=originals, epochs=epochs)
