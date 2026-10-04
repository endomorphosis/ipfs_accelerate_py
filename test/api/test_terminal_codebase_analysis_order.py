"""Synthetic advisory-consumer controls, not native storage qualification.

One module-scoped scratch-head fit performs four epochs on authored unit data.
Metadata wrappers/report values are constructed in memory without SQL, a
DuckLake process, a checker, or a worker. The public run has separate evidence.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata
from benchmarks.agent_supervisor.container_coding import terminal_codebase_analysis_order as consumer
from benchmarks.agent_supervisor.container_coding import terminal_codebase_analysis_receiver as receiver
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_corpus as corpus
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_ranker_training as head
from benchmarks.agent_supervisor.container_coding.terminal_codebase_intent_training_join import (
    _digest as head_digest,
)
from test.api.test_terminal_codebase_intent_corpus import make_inputs, _history, _query


def _digest(value):
    """Raw hex identity of the consumer's finite canonical JSON profile."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _originals():
    inputs = make_inputs()
    second = _query(inputs['source_records'][0]['codebase_id'], 'nested-navigation',
        instruction='Synthetic second request: inspect nested processing and preserve every unresolved constraint.',
        navigation='Locate the nested inner method that prefixes the lambda character.')
    inputs['query_records'].append(second)
    inputs['split_assignments'][second['query_id']] = 'train'
    inputs['historical_exposure'] = _history(inputs['source_records'], inputs['query_records'])
    return inputs


def _query_row(frozen, query_id):
    return next(row for row in frozen['queries'] if row['query_id'] == query_id)


def _result_query(result, query_id):
    return next(row for row in result['query_results'] if row['query_id'] == query_id)


def _pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(metadata._wire(value))
    return _pin(path)


def _metadata_context(payloads, output):
    """Construct native-shaped in-memory declarations; no SQL claim is tested.

    The historical verification fields below are authored synthetic input to
    the pure consumer. They are not receipts from an actual native process.
    Actual row identities, packets, export bytes and hashes use native helpers.
    """
    snapshot = {'schema': 'synthetic-analysis-order-source-snapshot@1',
        'fixture_scope': 'authored_unit_only_no_native_SQL'}
    source_digest = metadata._digest(metadata._wire(snapshot))
    rows = metadata._families(payloads, source_digest)
    families = {}
    export_pins = {}
    for name, values in rows.items():
        raw = b''.join(metadata._wire(row) + b'\n' for row in values)
        path = output / 'exports' / (name + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        export_pins[name] = _pin(path)
        families[name] = {'count': len(values),
            'digest': metadata._digest(metadata._row_array(values)),
            'view': 'family_' + name,
            'export': {'relative_path': 'exports/' + name + '.jsonl',
                'sha256': metadata._digest(raw), 'bytes': len(raw)}}
    packets = metadata._packets(rows)
    receipts = []
    for index in range(0, len(packets), 10):
        event_ids = [packet['packet_id'] for packet in packets[index:index + 10]]
        receipts.append({'schema': 'autoencoder-ducklake-commit-v1',
            'scope': 'isolated_history', 'admitted': False,
            'production_activated': False, 'event_payloads_verified': True,
            'source_id': 'codebase-ir-experiment:' + source_digest[7:],
            'history_id': 'sha256:' + _digest({'synthetic_history': True}),
            'batch_id': 'sha256:' + _digest({'synthetic_event_ids': event_ids}),
            'event_count': len(event_ids), 'event_ids': event_ids,
            'snapshot_id': len(receipts) + 1})
    manifest = {'schema': metadata.SCHEMA, 'output': str(output),
        'created_at': '2026-10-04T00:00:00Z', 'source_snapshot': snapshot,
        'source_snapshot_sha256': source_digest, 'families': families,
        'row_count': sum(map(len, rows.values())), 'row_root_sha256': metadata._row_root(rows),
        'catalog_schema_sha256': 'sha256:' + _digest({'synthetic_catalog': True}),
        'native_runtime': {'duckdb': 'synthetic_unit_not_loaded',
            'platform': 'synthetic_unit_no_SQL',
            'extensions': {name: hashlib.sha256(('synthetic:' + name).encode()).hexdigest()
                for name in ('ducklake', 'httpfs', 'quack')}},
        'limits': deepcopy(metadata.LIMITS), 'lake_layout': deepcopy(metadata.LAKE_LAYOUT),
        'lake_packet_count': len(packets),
        'lake_snapshot_digest': metadata._digest(metadata._wire(receipts)),
        'batches': [{'receipt': receipt} for receipt in receipts],
        'qualification': deepcopy(metadata.QUALIFICATION)}
    manifest_pin = _write(output / 'manifest.json', manifest)
    report = metadata._report(manifest, 'sha256:' + manifest_pin['sha256'])
    report['fresh_process_readback'] = {
        'method': 'new_python_process_native_duckdb_and_ducklake_readback', 'verified': True,
        **{key: report[key] for key in (
            'manifest_sha256', 'row_root_sha256', 'row_count', 'lake_snapshot_digest')}}
    report_pin = _write(output / 'readback.json', report)
    return {'rows': rows, 'report': report, 'manifest': manifest,
        'manifest_pin': manifest_pin, 'report_pin': report_pin, 'exports': export_pins}


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    originals = _originals()
    frozen = corpus.build_terminal_intent_relevance_corpus(**originals)
    # Exactly one real four-epoch synthetic optimizer invocation in this module.
    result = head.train_terminal_codebase_intent_ranker(
        corpus_receipt=frozen, original_inputs=originals, epochs=4)
    payloads = consumer.project_terminal_intent_metadata(frozen, result, originals)
    context = _metadata_context(payloads, tmp_path_factory.mktemp('analysis-order-unit') / 'metadata')
    return {'originals': originals, 'corpus': frozen, 'ranker': result,
        'payloads': payloads, 'metadata': context, 'query_id': 'unicode-navigation'}


def _arguments(fitted, **overrides):
    result = fitted['ranker']
    value = {'corpus_receipt': fitted['corpus'], 'original_inputs': fitted['originals'],
        'ranker_receipt': result, 'expected_corpus_sha256': fitted['corpus']['corpus_sha256'],
        'expected_checkpoint_sha256': head_digest(result['checkpoint']),
        'expected_training_receipt_sha256': result['training_receipt']['receipt_sha256'],
        'query_binding': consumer.bind_terminal_analysis_query(fitted['corpus'], fitted['query_id']),
        'current_source_records': deepcopy(fitted['originals']['source_records']),
        'metadata_rows': fitted['metadata']['rows'], 'metadata_report': fitted['metadata']['report'],
        'expected_metadata_report_sha256': _digest(fitted['metadata']['report']),
        'enabled': True, 'method': 'trained'}
    value.update(overrides)
    return deepcopy(value)


def _build(fitted, **overrides):
    return consumer.build_terminal_codebase_analysis_order(**_arguments(fitted, **overrides))


@pytest.mark.parametrize('method', ['trained', 'lexical', 'zero', 'reverse'])
def test_complete_order_preserves_native_doc_unknown_residuals_and_unjudged_units(fitted, method):
    receipt = _build(fitted, method=method)
    query = _query_row(fitted['corpus'], fitted['query_id'])
    bank = fitted['corpus']['candidate_banks'][query['codebase_id']]
    expected = _result_query(fitted['ranker'], fitted['query_id'])['rankings'][method]
    assert receipt['order'] == expected
    assert receipt['ordered_candidate_ids'] == [row['candidate_id'] for row in expected]
    assert len(receipt['ordered_candidates']) == len(bank) == 4
    assert {row['candidate_id']: row for row in receipt['ordered_candidates']} == {
        row['candidate_id']: row for row in bank}
    reviewed = {row['candidate_id'] for row in fitted['corpus']['reviewed_judgments']}
    assert len(set(receipt['ordered_candidate_ids']) - reviewed) == 2
    assert receipt['intent_document'] == query['intent_document']
    assert len(receipt['intent_document']['sources']) == 2
    assert receipt['original_prompt_residuals'] == query['original_prompt_residuals']
    assert receipt['residual_requirements'] == query['residual_requirements']
    assert all(row['status'] == 'unknown' and row['behavioral_satisfaction'] is False
        for row in receipt['residual_requirements'])
    assert receipt['planning_handoff'] == 'abstained'
    assert receipt['full_prompt_interpreted'] is False
    assert receipt['native_stores_reopened_here'] is False
    assert receipt['fit_executed_here'] is False
    assert all(receipt[name] is False for name in corpus._AUTHORITY)
    assert consumer.validate_terminal_codebase_analysis_order(
        receipt, **_arguments(fitted, method=method)) == receipt


@pytest.mark.parametrize('enabled,method', [(False, 'trained'), (True, 'model_off')])
def test_disabled_and_model_off_use_exact_full_original_bank(fitted, enabled, method):
    receipt = _build(fitted, enabled=enabled, method=method)
    query = _query_row(fitted['corpus'], fitted['query_id'])
    bank = fitted['corpus']['candidate_banks'][query['codebase_id']]
    assert receipt['ordered_candidates'] == bank
    assert receipt['order'] == [{'candidate_id': row['candidate_id'], 'position': i, 'score': None}
        for i, row in enumerate(bank, 1)]
    assert receipt['effective_method'] == 'original_full_bank_order'
    assert receipt['residual_requirements'] == query['residual_requirements']
    assert receipt['planning_handoff'] == 'abstained'


@pytest.mark.parametrize('field', ['expected_corpus_sha256', 'expected_checkpoint_sha256',
    'expected_training_receipt_sha256', 'expected_metadata_report_sha256'])
def test_wrong_independent_external_pin_refuses(fitted, field):
    arguments = _arguments(fitted)
    arguments[field] = ('sha256:' if field == 'expected_corpus_sha256' else '') + '0' * 64
    with pytest.raises(consumer.TerminalAnalysisOrderError):
        consumer.build_terminal_codebase_analysis_order(**arguments)


def test_same_codebase_query_swap_cannot_reuse_other_native_source_binding(fitted):
    arguments = _arguments(fitted)
    second = _query_row(fitted['corpus'], 'nested-navigation')
    assert second['codebase_id'] == arguments['query_binding']['codebase_id']
    arguments['query_binding']['query_id'] = second['query_id']
    with pytest.raises(consumer.TerminalAnalysisOrderError, match='query binding differs'):
        consumer.build_terminal_codebase_analysis_order(**arguments)


def test_all_unjudged_second_query_keeps_complete_bank_and_full_opaque_guard(fitted):
    binding = consumer.bind_terminal_analysis_query(fitted['corpus'], 'nested-navigation')
    receipt = _build(fitted, query_binding=binding)
    assert receipt['candidate_count'] == 4
    assert not any(row['query_id'] == 'nested-navigation'
        for row in fitted['corpus']['reviewed_judgments'])
    guard = next(row for row in receipt['residual_requirements'] if row['statement']['kind'] == 'guard')
    assert guard['statement']['normalized_text'] == _query_row(
        fitted['corpus'], 'nested-navigation')['instruction_text']
    assert guard['status'] == 'unknown'
    assert receipt['planning_handoff'] == 'abstained'


def test_current_operator_edit_refuses_even_when_all_44_native_features_match(fitted):
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae
    arguments = _arguments(fitted)
    source = arguments['current_source_records'][0]
    changed = source['source_text'].replace('>=', '>')
    assert changed != source['source_text']
    old, _ = ae._features({source['path']: source['source_text'].encode()}, [source['path']], 1024)
    new, _ = ae._features({source['path']: changed.encode()}, [source['path']], 1024)
    assert [row['features'] for row in old] == [row['features'] for row in new]
    assert [row['ast_sha256'] for row in old] != [row['ast_sha256'] for row in new]
    source['source_text'] = changed
    source['source_sha256'] = hashlib.sha256(changed.encode()).hexdigest()
    with pytest.raises(consumer.TerminalAnalysisOrderError, match='current sources differ'):
        consumer.build_terminal_codebase_analysis_order(**arguments)


@pytest.mark.parametrize('mutation', ['trim', 'duplicate', 'query_swap'])
def test_resealed_metadata_projection_cannot_trim_duplicate_or_swap_rank_rows(fitted, tmp_path, mutation):
    payloads = deepcopy(fitted['payloads'])
    if mutation == 'trim':
        payloads['ranking'].pop()
    elif mutation == 'duplicate':
        # Same candidate under an alias position yields a distinct native row,
        # so native row uniqueness alone cannot catch this projection attack.
        extra = deepcopy(payloads['ranking'][0])
        extra['position'] = 99
        payloads['ranking'].append(extra)
    else:
        names = [fitted['query_id'], 'nested-navigation']
        for row in payloads['ranking']:
            row['query_id'] = names[1] if row['query_id'] == names[0] else names[0]
    altered = _metadata_context(payloads, tmp_path / 'metadata')
    arguments = _arguments(fitted, metadata_rows=altered['rows'], metadata_report=altered['report'],
        expected_metadata_report_sha256=_digest(altered['report']))
    with pytest.raises(consumer.TerminalAnalysisOrderError, match='metadata rows differ'):
        consumer.build_terminal_codebase_analysis_order(**arguments)


def test_removed_opaque_original_guard_refuses_before_ordering(fitted):
    arguments = _arguments(fitted)
    query = next(row for row in arguments['original_inputs']['query_records']
        if row['query_id'] == fitted['query_id'])
    query['intent_document']['statements'] = [row for row in query['intent_document']['statements']
        if row['kind'] != 'guard']
    with pytest.raises(consumer.TerminalAnalysisOrderError, match='whole original opaque guard'):
        consumer.build_terminal_codebase_analysis_order(**arguments)


def test_resealed_report_must_bind_actual_canonical_lake_receipts(fitted):
    arguments = _arguments(fitted)
    report = arguments['metadata_report']
    report['lake_snapshot_digest'] = 'sha256:' + '0' * 64
    report['fresh_process_readback']['lake_snapshot_digest'] = report['lake_snapshot_digest']
    arguments['expected_metadata_report_sha256'] = _digest(report)
    with pytest.raises(consumer.TerminalAnalysisOrderError, match='snapshot receipt digest'):
        consumer.build_terminal_codebase_analysis_order(**arguments)


@pytest.mark.parametrize('mutation', ['authority', 'planning', 'omit_unjudged'])
def test_receipt_reseal_cannot_upgrade_authority_or_omit_candidates(fitted, mutation):
    arguments = _arguments(fitted)
    receipt = consumer.build_terminal_codebase_analysis_order(**arguments)
    if mutation == 'authority':
        receipt['proof_authority'] = True
    elif mutation == 'planning':
        receipt['planning_handoff'] = 'accepted'
    else:
        reviewed = {row['candidate_id'] for row in fitted['corpus']['reviewed_judgments']}
        index = next(i for i, row in enumerate(receipt['ordered_candidates'])
            if row['candidate_id'] not in reviewed)
        receipt['ordered_candidates'].pop(index)
        receipt['ordered_candidate_ids'].pop(index)
        receipt['order'].pop(index)
        receipt['candidate_count'] -= 1
    receipt['analysis_order_sha256'] = _digest({key: value for key, value in receipt.items()
        if key != 'analysis_order_sha256'})
    with pytest.raises(consumer.TerminalAnalysisOrderError, match='independent replay'):
        consumer.validate_terminal_codebase_analysis_order(receipt, **arguments)


def test_consumer_detaches_originals_and_returned_full_candidates(fitted):
    arguments = _arguments(fitted)
    before = deepcopy(arguments)
    receipt = consumer.build_terminal_codebase_analysis_order(**arguments)
    clean = deepcopy(receipt)
    arguments['original_inputs']['source_records'][0]['source_text'] = 'changed caller state'
    assert receipt == clean
    receipt['ordered_candidates'][0]['source_binding']['start_byte'] += 1
    receipt['intent_document']['sources'][0]['source_uri'] = 'changed returned state'
    assert _arguments(fitted) == before
    assert consumer.build_terminal_codebase_analysis_order(**before) == clean


def _receiver_arguments(fitted, directory, **overrides):
    original = deepcopy(fitted['originals'])
    envelope_pin = _write(directory / 'frozen.json', {
        'schema': 'terminal-intent-relevance-frozen-envelope@1',
        'corpus': fitted['corpus'], 'original_inputs': original})
    ranker_pin = _write(directory / 'ranker-result.json', fitted['ranker'])
    context = _metadata_context(fitted['payloads'], directory / 'metadata')
    source_files = []
    for index, source in enumerate(original['source_records']):
        path = directory / 'live-sources' / str(index) / source['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source['source_text'].encode())
        source_files.append({'codebase_id': source['codebase_id'], 'path': source['path'], 'file': _pin(path)})
    instruction_files = []
    for index, query in enumerate(original['query_records']):
        path = directory / 'instructions' / (str(index) + '.md')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(query['instruction_text'].encode())
        instruction_files.append({'query_id': query['query_id'], 'file': _pin(path)})
    value = {'frozen_envelope': envelope_pin, 'ranker_result': ranker_pin,
        'metadata_report': context['report_pin'], 'metadata_manifest': context['manifest_pin'],
        'metadata_exports': context['exports'], 'source_files': source_files,
        'instruction_files': instruction_files,
        'expected_corpus_sha256': fitted['corpus']['corpus_sha256'],
        'expected_checkpoint_sha256': head_digest(fitted['ranker']['checkpoint']),
        'expected_training_receipt_sha256': fitted['ranker']['training_receipt']['receipt_sha256'],
        'query_binding': consumer.bind_terminal_analysis_query(fitted['corpus'], fitted['query_id']),
        'enabled': True, 'method': 'trained'}
    value.update(overrides)
    return deepcopy(value)


def test_receiver_visits_real_unicode_decorated_spans_in_order_without_losing_population(fitted, tmp_path):
    arguments = _receiver_arguments(fitted, tmp_path)
    trained = receiver.inspect_terminal_codebase_analysis(**arguments)
    baseline = receiver.inspect_terminal_codebase_analysis(**{**arguments, 'enabled': False})
    assert [row['candidate_id'] for row in trained['visits']] == trained['analysis_order']['ordered_candidate_ids']
    assert [row['candidate_id'] for row in baseline['visits']] == baseline['analysis_order']['ordered_candidate_ids']
    assert trained['visit_population_sha256'] == baseline['visit_population_sha256']
    assert len(trained['visits']) == len(baseline['visits']) == 4
    assert all(row['span_verified'] is True and row['behavioral_facts_emitted'] == 0
        for row in trained['visits'])
    assert any(row['symbol'].endswith('inner') for row in trained['visits'])
    assert any(row['symbol'].endswith('méthode') for row in trained['visits'])
    for receipt in (trained, baseline):
        assert receipt['files_observed_before_and_after'] is True
        assert receipt['all_candidates_inspected'] is True
        assert receipt['atomic_checkout_snapshot'] is False
        assert receipt['whole_repository_coverage'] is False
        assert receipt['planning_handoff'] == 'abstained_unresolved_original_intent'
        assert all(receipt[name] is False for name in corpus._AUTHORITY)
        assert all(receipt[name] == 0 for name in (
            'native_SQL_calls_here', 'new_training_calls', 'new_prover_calls', 'PlanCreate_calls'))
        assert all(receipt[name] == [] for name in ('planner_tasks', 'facts', 'effects', 'outputs'))


@pytest.mark.parametrize('target', ['ranker_file_pin', 'checkpoint_pin'])
def test_receiver_wrong_independent_pin_refuses_before_any_source_visit(fitted, tmp_path, monkeypatch, target):
    arguments = _receiver_arguments(fitted, tmp_path)
    if target == 'ranker_file_pin':
        arguments['ranker_result']['sha256'] = '0' * 64
    else:
        arguments['expected_checkpoint_sha256'] = '0' * 64
    entered = []
    monkeypatch.setattr(receiver, '_inspect_candidate', lambda *args: entered.append(args))
    with pytest.raises((receiver.AnalysisReceiverError, consumer.TerminalAnalysisOrderError)):
        receiver.inspect_terminal_codebase_analysis(**arguments)
    assert entered == []


@pytest.mark.parametrize('target', ['source_files', 'instruction_files', 'ranker_result'])
def test_receiver_change_during_first_actual_visit_withholds_result(fitted, tmp_path, monkeypatch, target):
    from pathlib import Path
    arguments = _receiver_arguments(fitted, tmp_path)
    original_visit = receiver._inspect_candidate
    visits = []
    descriptor = arguments[target][0]['file'] if target != 'ranker_result' else arguments[target]
    path = Path(descriptor['path'])
    def inspect_and_change(candidate, raw, position):
        value = original_visit(candidate, raw, position)
        visits.append(value)
        if len(visits) == 1:
            path.write_bytes(path.read_bytes() + b'\n')
        return value
    monkeypatch.setattr(receiver, '_inspect_candidate', inspect_and_change)
    with pytest.raises(receiver.AnalysisReceiverError, match='changed before final observation'):
        receiver.inspect_terminal_codebase_analysis(**arguments)
    assert len(visits) == 4  # Real visits happened; final refusal is not a no-effect claim.
    assert path.read_bytes().endswith(b'\n')


@pytest.mark.parametrize('target', ['source', 'instruction'])
def test_receiver_updated_live_file_pin_does_not_replace_admitted_original(fitted, tmp_path, monkeypatch, target):
    from pathlib import Path
    arguments = _receiver_arguments(fitted, tmp_path)
    group = 'source_files' if target == 'source' else 'instruction_files'
    descriptor = arguments[group][0]['file']
    path = Path(descriptor['path'])
    if target == 'source':
        path.write_bytes(path.read_bytes().replace(b'>=', b'>'))
    else:
        path.write_bytes(path.read_bytes() + b' Invented narrowed task.')
    arguments[group][0]['file'] = _pin(path)
    entered = []
    monkeypatch.setattr(receiver, '_inspect_candidate', lambda *args: entered.append(args))
    with pytest.raises((receiver.AnalysisReceiverError, consumer.TerminalAnalysisOrderError)):
        receiver.inspect_terminal_codebase_analysis(**arguments)
    assert entered == []


def test_receiver_missing_export_refuses_before_source_visitation(fitted, tmp_path, monkeypatch):
    arguments = _receiver_arguments(fitted, tmp_path)
    del arguments['metadata_exports']['ranking']
    entered = []
    monkeypatch.setattr(receiver, '_inspect_candidate', lambda *args: entered.append(args))
    with pytest.raises(receiver.AnalysisReceiverError, match='complete metadata export population'):
        receiver.inspect_terminal_codebase_analysis(**arguments)
    assert entered == []
