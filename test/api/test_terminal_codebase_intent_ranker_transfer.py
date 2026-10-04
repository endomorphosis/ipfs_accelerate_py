"""Synthetic fixed-head transfer controls; no public artifacts or native SQL.

Exactly one module-scoped four-epoch scratch fit supplies the old checkpoint.
Every transfer call performs passive inference/replay. Metadata declarations
are authored unit fixtures, not evidence of a real DuckDB/DuckLake readback.
"""
from copy import deepcopy
import hashlib
import math

import pytest

from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata
from benchmarks.agent_supervisor.container_coding import terminal_codebase_analysis_order as analysis
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_corpus as corpus
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_ranker_training as head
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_ranker_transfer as transfer
from benchmarks.agent_supervisor.container_coding.terminal_codebase_intent_training_join import _digest
from test.api.test_terminal_codebase_intent_corpus import (
    make_inputs, _history, _judgment, _query, _ref, _source,
)
from test.api.test_terminal_codebase_analysis_order import _metadata_context


def _training_inputs():
    original = make_inputs()
    # Authored prior-exposure labels let removal controls exercise a real
    # nonempty declared history. They do not represent extra unit fit calls.
    original['historical_exposure']['source_history'][0]['prior_fit_refs'] = [
        'synthetic:declared-prior-exposure']
    original['historical_exposure']['query_history'][0]['prior_review_refs'] = [
        'synthetic:declared-prior-review']
    return original


def _expanded_inputs(original):
    expanded = deepcopy(original)
    source = _source(
        'def assemble(first, second):\n'
        '    return {"first": first, "second": second}\n\n'
        'def bracket(text):\n'
        '    return "[" + text + "]"\n\n'
        'def integer_flag(values):\n'
        '    return any(isinstance(item, int) for item in values)\n',
        'synthetic-transfer-mapping', 'mapping.py')
    query = _query(source['codebase_id'], 'mapping-navigation',
        instruction='Inspect the separate mapping assembly interface and retain its entire unresolved request.',
        navigation='Locate dictionary assembly of first and second mapping entries.')
    unjudged = _query(source['codebase_id'], 'mapping-unjudged',
        instruction='Review the integer predicate in the independent mapping source without assuming program correctness.',
        navigation='Navigate the boolean generator checking integer values.')
    expanded['source_records'].append(source)
    expanded['query_records'].extend([query, unjudged])
    expanded['split_assignments'].update({query['query_id']: 'validation', unjudged['query_id']: 'validation'})
    expanded['reviewed_judgments'].extend([
        _judgment(query, source, 'assemble', 'positive'),
        _judgment(query, source, 'bracket', 'negative_navigation')])
    added = _history([source], [query, unjudged])
    expanded['historical_exposure']['source_history'].extend(added['source_history'])
    expanded['historical_exposure']['query_history'].extend(added['query_history'])
    return expanded


@pytest.fixture(scope='module')
def fitted():
    original = _training_inputs()
    training = corpus.build_terminal_intent_relevance_corpus(**original)
    # One explicit four-update synthetic optimizer invocation for the module.
    ranker = head.train_terminal_codebase_intent_ranker(
        corpus_receipt=training, original_inputs=original, epochs=4)
    expanded = _expanded_inputs(original)
    evaluation = corpus.build_terminal_intent_relevance_corpus(**expanded)
    return {'original': original, 'training': training, 'ranker': ranker,
        'expanded': expanded, 'evaluation': evaluation}


def _arguments(fitted, **overrides):
    arguments = {'training_corpus_receipt': fitted['training'],
        'training_original_inputs': fitted['original'],
        'training_ranker_receipt': fitted['ranker'],
        'evaluation_corpus_receipt': fitted['evaluation'],
        'evaluation_original_inputs': fitted['expanded'],
        'expected_training_corpus_sha256': fitted['training']['corpus_sha256'],
        'expected_evaluation_corpus_sha256': fitted['evaluation']['corpus_sha256'],
        'expected_checkpoint_sha256': _digest(fitted['ranker']['checkpoint']),
        'expected_training_receipt_sha256': fitted['ranker']['training_receipt']['receipt_sha256']}
    arguments.update(overrides)
    return deepcopy(arguments)


def _evaluate(fitted, **overrides):
    return transfer.infer_terminal_codebase_intent_transfer(**_arguments(fitted, **overrides))


def _query_result(receipt, query_id):
    return next(row for row in receipt['query_results'] if row['query_id'] == query_id)


def _evaluation_arguments(fitted, expanded):
    frozen = corpus.build_terminal_intent_relevance_corpus(**expanded)
    return _arguments(fitted, evaluation_original_inputs=expanded,
        evaluation_corpus_receipt=frozen, expected_evaluation_corpus_sha256=frozen['corpus_sha256'])


def _replace_training_source(expanded, new_text):
    """Reseal a valid native expanded corpus, rather than stale label hashes."""
    source = expanded['source_records'][0]
    source['source_text'] = new_text
    source['source_sha256'] = hashlib.sha256(new_text.encode()).hexdigest()
    for row in expanded['historical_exposure']['source_history']:
        if (row['codebase_id'], row['path']) == (source['codebase_id'], source['path']):
            row['source_sha256'] = source['source_sha256']
    for judgment in expanded['reviewed_judgments']:
        reference = judgment['candidate_ref']
        if (reference['codebase_id'], reference['path']) == (source['codebase_id'], source['path']):
            judgment['candidate_ref'] = _ref(source, reference['symbol'])
    return source


def test_fixed_head_keeps_original_fit_identity_and_performs_no_optimizer(fitted, monkeypatch):
    old = deepcopy(fitted['ranker'])
    def forbidden_fit(*args, **kwargs):
        raise AssertionError('transfer must not fit a checkpoint')
    monkeypatch.setattr(head, 'train_terminal_codebase_intent_ranker', forbidden_fit)
    result = _evaluate(fitted)
    assert result['schema'] == transfer.SCHEMA
    assert result['training_corpus_sha256'] == fitted['training']['corpus_sha256']
    assert result['evaluation_corpus_sha256'] == fitted['evaluation']['corpus_sha256']
    assert result['training_corpus_sha256'] != result['evaluation_corpus_sha256']
    assert result['checkpoint'] == old['checkpoint']
    assert result['training_receipt'] == old['training_receipt']
    assert result['checkpoint']['corpus_sha256'] == result['training_corpus_sha256']
    assert result['training_receipt']['corpus_sha256'] == result['training_corpus_sha256']
    assert result['new_fit'] is False
    assert result['new_fit_receipt_created'] is False
    assert result['new_autoencoder_fit'] is False
    assert result['new_optimizer_steps'] == 0
    assert fitted['ranker'] == old
    assert transfer.validate_terminal_codebase_intent_transfer(result, **_arguments(fitted)) == result


@pytest.mark.parametrize('field', ['expected_training_corpus_sha256', 'expected_evaluation_corpus_sha256',
    'expected_checkpoint_sha256', 'expected_training_receipt_sha256'])
@pytest.mark.parametrize('invalid', ['wrong_digest', 'missing_value'])
def test_all_four_external_pins_are_mandatory_and_exact(fitted, field, invalid):
    arguments = _arguments(fitted)
    arguments[field] = None if invalid == 'missing_value' else (
        ('sha256:' if 'corpus' in field else '') + '0' * 64)
    with pytest.raises(transfer.IntentRankerTransferError):
        transfer.infer_terminal_codebase_intent_transfer(**arguments)


@pytest.mark.parametrize('mutation', ['operator', 'source_bank', 'original_prompt',
    'train_labels', 'added_train_query'])
def test_resealed_evaluation_corpus_cannot_change_original_training_population(fitted, mutation):
    expanded = deepcopy(fitted['expanded'])
    source = expanded['source_records'][0]
    if mutation == 'operator':
        from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae
        before, _ = ae._features({source['path']: source['source_text'].encode()}, [source['path']], 1024)
        edited = source['source_text'].replace('>=', '>')
        assert edited != source['source_text']
        source = _replace_training_source(expanded, edited)
        after, _ = ae._features({source['path']: source['source_text'].encode()}, [source['path']], 1024)
        assert [row['features'] for row in before] == [row['features'] for row in after]
        assert [row['ast_sha256'] for row in before] != [row['ast_sha256'] for row in after]
    elif mutation == 'source_bank':
        added = _source('def additional_unit(value):\n    return {"extra": [value]}\n',
            source['codebase_id'], 'additional.py')
        expanded['source_records'].append(added)
        expanded['historical_exposure']['source_history'].extend(_history([added], [])['source_history'])
        # Old file, judgments and numerical train-pair rows remain untouched.
        assert expanded['source_records'][0] == fitted['original']['source_records'][0]
    elif mutation == 'original_prompt':
        old_query = expanded['query_records'][0]
        updated = _query(old_query['codebase_id'], old_query['query_id'],
            instruction=old_query['instruction_text'] + ' A changed unresolved constraint.',
            navigation=old_query['navigation_text'])
        expanded['query_records'][0] = updated
        for row in expanded['historical_exposure']['query_history']:
            if row['query_id'] == updated['query_id']:
                row['instruction_sha256'] = updated['instruction_sha256']
        assert updated['navigation_text'] == old_query['navigation_text']
    elif mutation == 'train_labels':
        query_id = fitted['original']['query_records'][0]['query_id']
        for row in expanded['reviewed_judgments']:
            if row['query_id'] == query_id:
                row['label'] = 'negative_navigation' if row['label'] == 'positive' else 'positive'
    else:
        added = _query(source['codebase_id'], 'extra-train-navigation',
            instruction='Review the original Unicode source through an additional unjudged development task.',
            navigation='Inspect the existing marker wrapper in this training codebase.')
        expanded['query_records'].append(added)
        expanded['split_assignments'][added['query_id']] = 'train'
        expanded['historical_exposure']['query_history'].extend(_history([], [added])['query_history'])
    arguments = _evaluation_arguments(fitted, expanded)
    # The expanded corpus is valid/resealed; refusal must preserve old fit
    # lineage rather than rely on a stale source or candidate-reference hash.
    assert corpus.validate_terminal_intent_relevance_corpus(
        arguments['evaluation_corpus_receipt'], **expanded) == arguments['evaluation_corpus_receipt']
    if mutation in ('source_bank', 'original_prompt', 'added_train_query'):
        assert head._prepare(fitted['training'], fitted['original'])[3:] == head._prepare(
            arguments['evaluation_corpus_receipt'], expanded)[3:]
    with pytest.raises(transfer.IntentRankerTransferError):
        transfer.infer_terminal_codebase_intent_transfer(**arguments)


def test_monotonic_declared_exposure_history_addition_keeps_head_and_scores(fitted):
    original = _evaluate(fitted)
    expanded = deepcopy(fitted['expanded'])
    source_row = expanded['historical_exposure']['source_history'][0]
    query_row = expanded['historical_exposure']['query_history'][0]
    source_row['prior_fit_refs'] = sorted(source_row['prior_fit_refs'] + [
        'synthetic:current-retained-fit:' + _digest(fitted['ranker']['checkpoint'])])
    query_row['prior_review_refs'] = sorted(query_row['prior_review_refs'] + [
        'synthetic:subsequent-navigation-review'])
    revised = transfer.infer_terminal_codebase_intent_transfer(**_evaluation_arguments(fitted, expanded))
    assert revised['checkpoint'] == original['checkpoint']
    assert revised['training_receipt'] == original['training_receipt']
    assert revised['query_results'] == original['query_results']
    assert revised['historical_source_exposure'] != original['historical_source_exposure']
    assert revised['transfer_lineage']['historical_references_policy'] == (
        'monotonic_supersets_for_exact_original_identities')


@pytest.mark.parametrize('family,field', [
    ('source_history', 'prior_fit_refs'), ('query_history', 'prior_review_refs')])
def test_resealed_history_cannot_remove_declared_old_exposure(fitted, family, field):
    expanded = deepcopy(fitted['expanded'])
    assert expanded['historical_exposure'][family][0][field]
    expanded['historical_exposure'][family][0][field] = []
    with pytest.raises(transfer.IntentRankerTransferError, match='removed prior exposure'):
        transfer.infer_terminal_codebase_intent_transfer(**_evaluation_arguments(fitted, expanded))


def test_new_validation_label_changes_only_metrics_not_weights_features_logits_or_orders(fitted):
    original = _evaluate(fitted)
    expanded = deepcopy(fitted['expanded'])
    query_id = 'mapping-navigation'
    for row in expanded['reviewed_judgments']:
        if row['query_id'] == query_id:
            row['label'] = 'negative_navigation' if row['label'] == 'positive' else 'positive'
    arguments = _evaluation_arguments(fitted, expanded)
    changed = transfer.infer_terminal_codebase_intent_transfer(**arguments)
    assert original['checkpoint'] == changed['checkpoint'] == fitted['ranker']['checkpoint']
    assert original['training_receipt'] == changed['training_receipt'] == fitted['ranker']['training_receipt']
    assert original['feature_profile'] == changed['feature_profile']
    old_prepared = head._prepare(fitted['evaluation'], fitted['expanded'])
    new_prepared = head._prepare(arguments['evaluation_corpus_receipt'], expanded)
    assert old_prepared[1] == new_prepared[1]  # Complete fixed 80D feature maps.
    assert old_prepared[3:] == new_prepared[3:]  # Original train pairs/update profile.
    for query in original['query_results']:
        assert query['rankings'] == _query_result(changed, query['query_id'])['rankings']
    assert _query_result(original, query_id)['metrics']['trained'] != _query_result(changed, query_id)['metrics']['trained']
    assert _query_result(original, query_id)['metrics']['model_off'] == _query_result(changed, query_id)['metrics']['model_off']


def _independent_features(query, candidate):
    import re
    token_pattern = re.compile(r'[^\W_]+', re.UNICODE)
    query_tokens = set(token_pattern.findall(query['navigation_text'].casefold()))
    code_tokens = set(token_pattern.findall(candidate['normalized_body'].casefold()))
    shared = query_tokens & code_tokens
    def bucket(token, count):
        return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big') % count
    shared_buckets = {bucket(token, 32) for token in shared}
    query_buckets = {bucket(token, 16) for token in query_tokens}
    qn, cn, sn = len(query_tokens), len(code_tokens), len(shared)
    return ([float(index in shared_buckets) for index in range(32)]
        + [sn / max(1, qn), sn / max(1, cn), sn / max(1, len(query_tokens | code_tokens)),
            sn / max(1, qn + cn)]
        + [float(value) * float(index % 16 in query_buckets)
            for index, value in enumerate(candidate['features'])])


def test_every_new_bank_has_complete_fixed_head_controls_and_all_unknown_original_guards(fitted):
    result = _evaluate(fitted)
    weights = fitted['ranker']['checkpoint']['weights']
    assert fitted['evaluation']['complete_candidate_count'] == 7
    assert len(result['query_results']) == 3
    for query in fitted['evaluation']['queries']:
        ranked = _query_result(result, query['query_id'])
        bank = fitted['evaluation']['candidate_banks'][query['codebase_id']]
        ids = {row['candidate_id'] for row in bank}
        assert ranked['candidate_count'] == len(bank)
        assert set(ranked['rankings']) == {'trained', 'zero', 'reverse', 'lexical', 'model_off'}
        for method, rows in ranked['rankings'].items():
            if method == 'model_off':
                assert rows == []
                assert ranked['metrics'][method]['first_positive_rank'] is None
                assert ranked['metrics'][method]['reciprocal_first_rank'] is None
                assert all(value is None for value in ranked['metrics'][method]['recall_at'].values())
            else:
                assert len(rows) == len(ids) == len({row['candidate_id'] for row in rows})
                assert {row['candidate_id'] for row in rows} == ids
                assert [row['position'] for row in rows] == list(range(1, len(bank) + 1))
        expected_scores = {row['candidate_id']: math.fsum(
            weight * value for weight, value in zip(weights, _independent_features(query, row))) for row in bank}
        assert {row['candidate_id']: row['score'] for row in ranked['rankings']['trained']} == expected_scores
        assert [row['candidate_id'] for row in ranked['rankings']['zero']] == sorted(ids)
        assert [row['candidate_id'] for row in ranked['rankings']['reverse']] == list(reversed(
            [row['candidate_id'] for row in ranked['rankings']['trained']]))
        assert ranked['residual_requirements'] == query['residual_requirements']
        assert len(query['intent_document']['sources']) == 2
        guard = next(row for row in ranked['residual_requirements'] if row['statement']['kind'] == 'guard')
        assert guard['statement']['normalized_text'] == query['instruction_text']
        assert all(row['status'] == 'unknown' and row['behavioral_satisfaction'] is False
            for row in ranked['residual_requirements'])
    unjudged = _query_result(result, 'mapping-unjudged')
    assert unjudged['reviewed_positive_ids'] == unjudged['reviewed_negative_navigation_ids'] == []
    assert all(metrics['first_positive_rank'] is None and metrics['reciprocal_first_rank'] is None
        and all(value is None for value in metrics['recall_at'].values()) for metrics in unjudged['metrics'].values())
    assert result['planning_handoff'] == 'abstained'
    assert result['complete_prompt_interpretation_qualified'] is False
    assert result['generalized_ranking_gain_qualified'] is False
    assert all(result[name] is False for name in corpus._AUTHORITY)


@pytest.mark.parametrize('mutation', ['trim_ranking', 'duplicate_ranking', 'original_guard', 'authority'])
def test_self_resealed_transfer_cannot_replace_original_rank_residual_or_authority(fitted, mutation):
    result = _evaluate(fitted)
    query = _query_result(result, 'mapping-navigation')
    if mutation == 'trim_ranking':
        query['rankings']['trained'].pop()
    elif mutation == 'duplicate_ranking':
        query['rankings']['trained'][1] = deepcopy(query['rankings']['trained'][0])
        query['rankings']['trained'][1]['position'] = 2
    elif mutation == 'original_guard':
        query['residual_requirements'] = [row for row in query['residual_requirements']
            if row['statement']['kind'] != 'guard']
    else:
        result['execution_authority'] = True
    result['transfer_sha256'] = corpus._digest({key: value for key, value in result.items()
        if key != 'transfer_sha256'})
    with pytest.raises(transfer.IntentRankerTransferError, match='independent lineage/inference replay'):
        transfer.validate_terminal_codebase_intent_transfer(result, **_arguments(fitted))


def test_thirteen_metadata_families_keep_old_fit_roots_and_new_source_query_rank_bindings(fitted):
    result = _evaluate(fitted)
    records = transfer.project_terminal_intent_transfer_metadata(result, **_arguments(fitted))
    assert len(records) == 13
    assert set(records) == {'sources', 'ast', 'kg', 'vectors', 'contracts', 'queries', 'judgments',
        'split_history', 'ranker_checkpoint', 'ranker_training', 'ranking', 'coverage_status', 'ranker_transfer'}
    assert records['ranker_checkpoint'] == [fitted['ranker']['checkpoint']]
    assert records['ranker_training'] == [fitted['ranker']['training_receipt']]
    assert records['sources'] == fitted['evaluation']['sources']
    assert records['queries'] == fitted['evaluation']['queries']
    assert len(records['ast']) == len(records['vectors']) == 7
    assert records['contracts'] == []
    assert len(records['ranking']) == 4 * (4 + 3 + 3) == 40
    expected = {key: result['transfer_lineage'][key] for key in (
        'training_corpus_sha256', 'evaluation_corpus_sha256', 'checkpoint_sha256', 'training_receipt_sha256')}
    for row in records['ranking']:
        assert {key: row[key] for key in expected} == expected
        assert row['corpus_sha256'] == result['evaluation_corpus_sha256']
        assert row['proof_authority'] is False
    summary, = records['ranker_transfer']
    assert summary['transfer_sha256'] == result['transfer_sha256']
    assert summary['transfer_lineage'] == result['transfer_lineage']
    assert 'query_results' not in summary
    assert summary['new_fit'] is False and summary['new_optimizer_steps'] == 0
    coverage, = records['coverage_status']
    assert coverage['cross_corpus_transfer_inference_only'] is True
    assert coverage['new_fit_receipt_created'] is False
    assert coverage['whole_repository_metadata_coverage'] is False
    source_digest = 'sha256:' + corpus._digest({'synthetic_row_root_comparison': True})
    old_records = analysis.project_terminal_intent_metadata(fitted['training'], fitted['ranker'], fitted['original'])
    old_rows = metadata._families(old_records, source_digest)
    new_rows = metadata._families(records, source_digest)
    assert metadata._row_root(old_rows) != metadata._row_root(new_rows)
    assert old_rows['ranker_checkpoint'] == new_rows['ranker_checkpoint']
    assert old_rows['ranker_training'] == new_rows['ranker_training']
    assert sum(map(len, new_rows.values())) > sum(map(len, old_rows.values()))


@pytest.mark.parametrize('mutation', ['trim', 'old_corpus_rank_binding'])
def test_resealed_native_metadata_cannot_omit_transfer_rank_or_substitute_fit_corpus(fitted, tmp_path, mutation):
    result = _evaluate(fitted)
    expected = transfer.project_terminal_intent_transfer_metadata(result, **_arguments(fitted))
    changed = deepcopy(expected)
    if mutation == 'trim':
        changed['ranking'].pop()
    else:
        changed['ranking'][0]['evaluation_corpus_sha256'] = result['training_corpus_sha256']
    declared = _metadata_context(changed, tmp_path / 'metadata')
    # Exact pure native payload joining, not a claim of native SQL validation.
    with pytest.raises(analysis.TerminalAnalysisOrderError, match='metadata rows differ'):
        analysis._metadata(expected, declared['rows'], declared['report'], corpus._digest(declared['report']))


def test_existing_consumer_refuses_transfer_checkpoint_as_a_new_evaluation_fit(fitted):
    result = _evaluate(fitted)
    with pytest.raises(analysis.TerminalAnalysisOrderError, match='corpus/profile'):
        analysis.project_terminal_intent_metadata(fitted['evaluation'], result, fitted['expanded'])
    assert result['checkpoint']['corpus_sha256'] == fitted['training']['corpus_sha256']


def test_closed_twelve_family_consumer_does_not_silently_accept_transfer_metadata(fitted, tmp_path):
    result = _evaluate(fitted)
    records = transfer.project_terminal_intent_transfer_metadata(result, **_arguments(fitted))
    declared = _metadata_context(records, tmp_path / 'metadata')
    arguments = {'corpus_receipt': fitted['training'], 'original_inputs': fitted['original'],
        'ranker_receipt': fitted['ranker'], 'expected_corpus_sha256': fitted['training']['corpus_sha256'],
        'expected_checkpoint_sha256': _digest(fitted['ranker']['checkpoint']),
        'expected_training_receipt_sha256': fitted['ranker']['training_receipt']['receipt_sha256'],
        'query_binding': analysis.bind_terminal_analysis_query(
            fitted['training'], fitted['original']['query_records'][0]['query_id']),
        'current_source_records': deepcopy(fitted['original']['source_records']),
        'metadata_rows': declared['rows'], 'metadata_report': declared['report'],
        'expected_metadata_report_sha256': corpus._digest(declared['report']), 'enabled': True}
    with pytest.raises(analysis.TerminalAnalysisOrderError, match='metadata rows differ'):
        analysis.build_terminal_codebase_analysis_order(**deepcopy(arguments))


def test_transfer_replay_detaches_originals_checkpoint_and_complete_query_results(fitted):
    arguments = _arguments(fitted)
    before = deepcopy(arguments)
    result = transfer.infer_terminal_codebase_intent_transfer(**arguments)
    clean = deepcopy(result)
    arguments['evaluation_original_inputs']['source_records'][0]['source_text'] = 'changed caller source'
    arguments['training_ranker_receipt']['checkpoint']['weights'][0] = 99.0
    assert result == clean
    result['checkpoint']['weights'][0] = -99.0
    result['query_results'][0]['residual_requirements'][0]['status'] = 'satisfied'
    assert _arguments(fitted) == before
    assert transfer.infer_terminal_codebase_intent_transfer(**before) == clean
