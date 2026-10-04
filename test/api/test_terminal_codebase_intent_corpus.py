"""Synthetic native corpus controls; no fits, retrieval, SQL or proof jobs.

The source, navigation judgments and prior-exposure references are authored unit
inputs. Native IntentIR and passive AST features are exercised; these cases do
not qualify held-out usefulness, code behavior or model convergence.
"""
from copy import deepcopy
import ast
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_corpus as corpus
from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae
from ipfs_datasets_py.logic.intent_ir.schema import (
    IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
    ReviewStatus, SourceRef, SourceSpan, StatementKind,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode())


def _source(text, codebase_id='synthetic-decorated', path='module.py'):
    return {'codebase_id': codebase_id, 'path': path, 'source_text': text,
        'source_sha256': _sha(text.encode()),
        'source_uri': 'synthetic://' + codebase_id + '.invalid/' + path}


def _query(codebase_id, query_id, instruction=None, navigation=None):
    instruction = instruction or ('Synthetic original request ' + query_id
        + ': preserve every existing behavior and unresolved requirement. μ\n')
    navigation = navigation or ('Find the decorated Unicode processing function for ' + query_id + '.')
    instruction_uri = 'synthetic://' + query_id + '.invalid/instruction.md'
    instruction_sha = _sha(instruction.encode())
    navigation_sha = _sha(navigation.encode())
    original = SourceRef(ref_id='original-' + query_id, source_uri=instruction_uri,
        source_id=instruction_sha, source_revision=instruction_sha,
        content_sha256=instruction_sha, span=SourceSpan(0, len(instruction)),
        review_status=ReviewStatus.TRUSTED_FIXTURE)
    derived = SourceRef(ref_id='navigation-' + query_id,
        source_uri=instruction_uri + '#navigation:' + query_id,
        source_id=navigation_sha, source_revision=navigation_sha,
        content_sha256=navigation_sha, span=SourceSpan(0, len(navigation)),
        review_status=ReviewStatus.TRUSTED_FIXTURE)
    guard = IntentStatement(statement_id='opaque-' + query_id,
        kind=StatementKind.GUARD, modality=IntentModality.ASSERTED,
        normalized_text=instruction, source_ref_ids=(original.ref_id,),
        predicate='opaque_request', arguments=(), confidence=0.0,
        grounding=NodeGrounding.GROUNDED, review_status=ReviewStatus.TRUSTED_FIXTURE)
    goal = IntentStatement(statement_id='navigate-' + query_id,
        kind=StatementKind.GOAL, modality=IntentModality.REQUIRED,
        normalized_text=navigation, source_ref_ids=(derived.ref_id,),
        predicate='navigate', arguments=('source_unit',), confidence=0.0,
        grounding=NodeGrounding.INFERRED, review_status=ReviewStatus.MACHINE_EXTRACTED)
    document = IntentIRDocument(document_id='corpus-' + query_id,
        title='Synthetic navigation-only corpus control', intent_kind=IntentKind.DECLARATIVE,
        sources=(original, derived), statements=(guard, goal)).to_dict()
    return {'query_id': query_id, 'codebase_id': codebase_id,
        'instruction_text': instruction, 'instruction_sha256': instruction_sha,
        'instruction_uri': instruction_uri, 'intent_document': document,
        'navigation_text': navigation, 'statement_ids': [goal.statement_id]}


def _history(sources, queries):
    return {'schema': corpus.HISTORY_SCHEMA,
        'source_history': [{'codebase_id': item['codebase_id'], 'path': item['path'],
            'source_sha256': item['source_sha256'], 'prior_fit_refs': [],
            'prior_review_refs': []} for item in sources],
        'query_history': [{'query_id': item['query_id'],
            'instruction_sha256': item['instruction_sha256'], 'prior_fit_refs': [],
            'prior_review_refs': []} for item in queries]}


def _ref(source, symbol):
    rows, unsupported = ae._features({source['path']: source['source_text'].encode()},
        [source['path']], 1024)
    assert unsupported == []
    row = next(row for row in rows if row['symbol'] == symbol)
    return {'codebase_id': source['codebase_id'], 'path': source['path'],
        'symbol': symbol, 'line': row['line'], 'source_sha256': source['source_sha256'],
        'ast_sha256': row['ast_sha256']}


def _judgment(query, source, symbol, label):
    return {'query_id': query['query_id'], 'candidate_ref': _ref(source, symbol),
        'label': label, 'reason': 'Synthetic reviewed navigation disposition; no semantic claim.',
        'review_ref': 'synthetic:unit-review:' + symbol}


def _arguments(sources, queries, assignments=None, judgments=None):
    return {'source_records': deepcopy(sources), 'query_records': deepcopy(queries),
        'reviewed_judgments': deepcopy(judgments or []),
        'split_assignments': assignments or {query['query_id']: 'train' for query in queries},
        'historical_exposure': _history(sources, queries)}


def make_inputs():
    """Fresh pure native fixture shared by corpus and synthetic ranker controls."""
    source = _source(
        '# Synthetic Unicode source: 雪\n'
        'def marker(function):\n'
        '    return function\n\n'
        '@marker\n'
        'def accented(δ):\n'
        '    note = "雪μ"\n'
        '    if len(δ) >= 1:\n'
        '        return note + δ\n'
        '    return note\n\n'
        'class Box:\n'
        '    @marker\n'
        '    def méthode(self, value):\n'
        '        @marker\n'
        '        def inner(Ω):\n'
        '            return "λ" + Ω\n'
        '        return inner(value)\n')
    query = _query(source['codebase_id'], 'unicode-navigation')
    return _arguments([source], [query], judgments=[
        _judgment(query, source, 'accented', 'positive'),
        _judgment(query, source, 'marker', 'negative_navigation')])


@pytest.fixture
def corpus_inputs():
    return make_inputs()


def _build(arguments):
    return corpus.build_terminal_intent_relevance_corpus(**arguments)


def _different_roles(left, right):
    """Distinct query origins keep code-body leakage independent of prompt reuse."""
    a = _source(left, 'synthetic-left', 'left.py')
    b = _source(right, 'synthetic-right', 'right.py')
    qa = _query(a['codebase_id'], 'arithmetic-query',
        instruction='Compute the additive quantity for an arithmetic navigation exercise.',
        navigation='Locate arithmetic addition over numeric arguments.')
    qb = _query(b['codebase_id'], 'stream-query',
        instruction='Inspect the streaming protocol in a separate source-navigation exercise.',
        navigation='Find streaming cursor consumption during iteration.')
    return _arguments([a, b], [qa, qb],
        assignments={qa['query_id']: 'train', qb['query_id']: 'validation'})


@pytest.mark.parametrize('mutation', ['duplicate', 'contradiction', 'unknown_unit',
    'unknown_query', 'wrong_path', 'wrong_ast', 'boolean_line', 'invalid_label'])
def test_judgments_require_unique_exact_native_candidates(corpus_inputs, mutation):
    arguments = deepcopy(corpus_inputs)
    judgment = arguments['reviewed_judgments'][0]
    if mutation in {'duplicate', 'contradiction'}:
        extra = deepcopy(judgment)
        if mutation == 'contradiction':
            extra['label'] = 'negative_navigation'
        arguments['reviewed_judgments'].append(extra)
    elif mutation == 'unknown_unit':
        judgment['candidate_ref']['symbol'] = 'unknown_function'
    elif mutation == 'unknown_query':
        judgment['query_id'] = 'not-in-query-population'
    elif mutation == 'wrong_path':
        judgment['candidate_ref']['path'] = 'other.py'
    elif mutation == 'wrong_ast':
        judgment['candidate_ref']['ast_sha256'] = '0' * 64
    elif mutation == 'boolean_line':
        judgment['candidate_ref']['line'] = True
    else:
        judgment['label'] = 'semantically_false'
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


@pytest.mark.parametrize('mutation', ['removed', 'shortened', 'inferred', 'navigation_bound'])
def test_original_instruction_remains_whole_grounded_opaque_guard(corpus_inputs, mutation):
    arguments = deepcopy(corpus_inputs)
    query = arguments['query_records'][0]
    statements = query['intent_document']['statements']
    guard = next(item for item in statements if item['kind'] == 'guard')
    if mutation == 'removed':
        statements.remove(guard)
    elif mutation == 'shortened':
        guard['normalized_text'] = 'Only the selected navigation phrase.'
    elif mutation == 'inferred':
        guard['grounding'] = 'inferred'
    else:
        guard['source_ref_ids'] = ['navigation-' + query['query_id']]
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


@pytest.mark.parametrize('mutation', ['opaque_selection', 'unknown_selection', 'wrong_navigation'])
def test_navigation_selection_cannot_replace_or_reinterpret_original_request(corpus_inputs, mutation):
    arguments = deepcopy(corpus_inputs)
    query = arguments['query_records'][0]
    if mutation == 'opaque_selection':
        query['statement_ids'] = ['opaque-' + query['query_id']]
    elif mutation == 'unknown_selection':
        query['statement_ids'] = ['not-in-native-document']
    else:
        query['navigation_text'] += ' Invented additional instruction.'
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


def test_shared_codebase_cannot_receive_two_query_roles(corpus_inputs):
    arguments = deepcopy(corpus_inputs)
    original = arguments['query_records'][0]
    second = _query(original['codebase_id'], 'separate-stream-navigation',
        instruction='Inspect event-loop ordering in this navigation unit exercise.',
        navigation='Locate event-loop yielding and streaming cursor use.')
    arguments['query_records'].append(second)
    arguments['split_assignments'][second['query_id']] = 'test'
    arguments['historical_exposure'] = _history(
        arguments['source_records'], arguments['query_records'])
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


def test_same_native_instruction_family_cannot_cross_roles():
    arguments = _different_roles('def add(v):\n    return v + 7\n',
        'def stream(xs):\n    for item in xs:\n        yield item\n')
    first, second = arguments['query_records']
    replacement = _query(second['codebase_id'], second['query_id'],
        instruction=first['instruction_text'], navigation=second['navigation_text'])
    arguments['query_records'][1] = replacement
    arguments['historical_exposure'] = _history(
        arguments['source_records'], arguments['query_records'])
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


def test_near_duplicate_navigation_family_cannot_cross_roles():
    arguments = _different_roles('def add(v):\n    return v + 7\n',
        'def stream(xs):\n    for item in xs:\n        yield item\n')
    text = ('Locate amber birch cobalt dune ember fir grove harbor island jade kelp larch '
        'maple north olive pine quartz river spruce tide umber violet willow xenon yellow '
        'zinc acorn brook cedar delta elm forest granite heath iris juniper knoll lake moss.')
    for index, suffix in enumerate((' moss.', ' reed.')):
        query = arguments['query_records'][index]
        arguments['query_records'][index] = _query(query['codebase_id'], query['query_id'],
            instruction=query['instruction_text'], navigation=text[:-6] + suffix)
    assert arguments['query_records'][0]['navigation_text'] != arguments['query_records'][1]['navigation_text']
    with pytest.raises(corpus.IntentCorpusError, match='native query split guard'):
        _build(arguments)


@pytest.mark.parametrize('right', [
    'def add(v):\n    return v + 7\n',
    'def renamed(q):\n    return q + 901\n',
], ids=['identical_body', 'alpha_literal_renamed_body'])
def test_complete_candidate_bodies_cannot_leak_across_roles(right):
    arguments = _different_roles('def add(v):\n    return v + 7\n',
        '# Separate synthetic module provenance.\n' + right)
    assert arguments['query_records'][0]['instruction_sha256'] != arguments['query_records'][1]['instruction_sha256']
    assert arguments['source_records'][0]['codebase_id'] != arguments['source_records'][1]['codebase_id']
    collision = 'normalized_body_sha256' if 'def add(' in right else 'alpha_shape_sha256'
    with pytest.raises(corpus.IntentCorpusError, match=collision):
        _build(arguments)


@pytest.mark.parametrize('mutation', ['missing_source', 'missing_query', 'duplicate_source',
    'wrong_source_hash', 'wrong_query_hash', 'unbound_history_field'])
def test_exposure_history_must_cover_exact_original_populations(corpus_inputs, mutation):
    arguments = deepcopy(corpus_inputs)
    history = arguments['historical_exposure']
    if mutation == 'missing_source':
        history['source_history'].clear()
    elif mutation == 'missing_query':
        history['query_history'].clear()
    elif mutation == 'duplicate_source':
        history['source_history'].append(deepcopy(history['source_history'][0]))
    elif mutation == 'wrong_source_hash':
        history['source_history'][0]['source_sha256'] = '0' * 64
    elif mutation == 'wrong_query_hash':
        history['query_history'][0]['instruction_sha256'] = '0' * 64
    else:
        history['frozen_before_fit'] = True
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


def test_source_hash_refuses_guard_change_even_with_identical_autoencoder_features(corpus_inputs):
    original = corpus_inputs['source_records'][0]
    changed = original['source_text'].replace('>=', '>')
    old_rows, _ = ae._features({original['path']: original['source_text'].encode()}, [original['path']], 1024)
    new_rows, _ = ae._features({original['path']: changed.encode()}, [original['path']], 1024)
    assert [row['features'] for row in old_rows] == [row['features'] for row in new_rows]
    assert [row['ast_sha256'] for row in old_rows] != [row['ast_sha256'] for row in new_rows]
    arguments = deepcopy(corpus_inputs)
    arguments['source_records'][0]['source_text'] = changed
    with pytest.raises(corpus.IntentCorpusError):
        _build(arguments)


@pytest.mark.parametrize('separator', ['\r', '\v', '\f', '\x00'], ids=['CR', 'VT', 'FF', 'NUL'])
def test_unsupported_source_line_separators_refuse_before_span_derivation(corpus_inputs, separator):
    arguments = deepcopy(corpus_inputs)
    source = arguments['source_records'][0]
    source['source_text'] = source['source_text'].replace('雪\n', '雪' + separator + '\n', 1)
    source['source_sha256'] = _sha(source['source_text'].encode())
    with pytest.raises(corpus.IntentCorpusError, match='source line mapping'):
        _build(arguments)


def _nodes(tree):
    found = {}

    def visit(node, prefix=''):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[prefix + node.name] = node
            prefix += node.name + '.'
        elif isinstance(node, ast.ClassDef):
            prefix += node.name + '.'
        for child in ast.iter_child_nodes(node):
            visit(child, prefix)

    visit(tree)
    return found


def _reseal(receipt):
    receipt['candidate_banks_sha256'] = _digest(receipt['candidate_banks'])
    receipt['corpus_sha256'] = 'sha256:' + _digest(
        {key: value for key, value in receipt.items() if key != 'corpus_sha256'})
    return receipt


def test_full_bank_keeps_decorated_nested_unicode_bytes_ast_and_native_features(corpus_inputs):
    receipt = _build(corpus_inputs)
    assert receipt['schema'] == corpus.SCHEMA
    assert receipt['complete_candidate_count'] == 4
    source = corpus_inputs['source_records'][0]
    raw = source['source_text'].encode()
    units = receipt['candidate_banks'][source['codebase_id']]
    by_symbol = {unit['symbol']: unit for unit in units}
    assert set(by_symbol) == {'marker', 'accented', 'Box.méthode', 'Box.méthode.inner'}
    rows, unsupported = ae._features({source['path']: raw}, [source['path']], 1024)
    assert unsupported == []
    assert {unit['native_feature_row_id'] for unit in units} == {row['row_id'] for row in rows}
    native = {row['symbol']: row for row in rows}
    nodes = _nodes(ast.parse(raw))
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    for symbol, unit in by_symbol.items():
        node = nodes[symbol]
        first_line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        start = offsets[first_line - 1]
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        binding = unit['source_binding']
        assert (unit['line'], unit['end_line']) == (node.lineno, node.end_lineno)
        assert unit['source_sha256'] == _sha(raw)
        assert unit['ast_sha256'] == _sha(ast.dump(node, include_attributes=False).encode())
        assert (binding['start_byte'], binding['end_byte']) == (start, end)
        assert binding['span_sha256'] == _sha(raw[start:end])
        chunks = []
        source_cursor, body_cursor = start, 0
        for mapping in binding['line_byte_map']:
            source_start, source_end = mapping['source_start_byte'], mapping['source_end_byte']
            body_start, body_end = mapping['normalized_start_byte'], mapping['normalized_end_byte']
            assert start <= source_cursor <= source_start < source_end <= end
            assert all(byte in b' \t' for byte in raw[source_cursor:source_start])
            assert body_start == body_cursor and body_end - body_start == source_end - source_start
            chunks.append(raw[source_start:source_end])
            source_cursor, body_cursor = source_end, body_end
        body = b''.join(chunks)
        assert source_cursor == end and body_cursor == len(body)
        assert unit['normalized_body'].encode() == body
        assert unit['normalized_body_sha256'] == binding['normalized_body_sha256'] == _sha(body)
        if node.decorator_list:
            assert body.startswith(b'@marker\n')
        assert unit['native_feature_row'] == native[symbol]
        assert unit['features'] == native[symbol]['features']
        assert len(unit['features']) == len(ae.FEATURES)
        assert all(unit[flag] is False for flag in corpus._AUTHORITY)
    assert '雪μ' in by_symbol['accented']['normalized_body']
    assert 'λ' in by_symbol['Box.méthode.inner']['normalized_body']


def test_explicit_navigation_judgments_leave_entire_unreviewed_population_unjudged(corpus_inputs):
    receipt = _build(corpus_inputs)
    query = receipt['queries'][0]
    bank = receipt['candidate_banks'][query['codebase_id']]
    assert query['candidate_ids'] == [unit['candidate_id'] for unit in bank]
    assert len(set(query['candidate_ids'])) == 4
    summary, = receipt['judgment_summary']
    assert summary == {'query_id': query['query_id'], 'task_role': 'train',
        'positive_count': 1, 'negative_navigation_count': 1, 'unjudged_count': 2}
    judgments = {row['candidate_ref']['symbol']: row for row in receipt['reviewed_judgments']}
    assert set(judgments) == {'accented', 'marker'}
    assert judgments['marker']['label'] == 'negative_navigation'
    pair, = receipt['pairs']
    assert pair['positive_candidate_id'] == judgments['accented']['candidate_id']
    assert pair['negative_candidate_id'] == judgments['marker']['candidate_id']
    assert not {unit['candidate_id'] for unit in bank if unit['symbol'].startswith('Box.')} & {
        pair['positive_candidate_id'], pair['negative_candidate_id']}
    arguments = deepcopy(corpus_inputs)
    arguments['reviewed_judgments'] = []
    unjudged = _build(arguments)
    assert unjudged['candidate_banks'] == receipt['candidate_banks']
    assert unjudged['queries'] == receipt['queries']
    assert unjudged['reviewed_judgments'] == unjudged['pairs'] == []
    assert unjudged['judgment_summary'][0]['unjudged_count'] == 4


def test_native_full_original_and_authored_navigation_remain_unknown_residuals(corpus_inputs):
    receipt = _build(corpus_inputs)
    query = receipt['queries'][0]
    original = corpus_inputs['query_records'][0]
    assert query['intent_document'] == original['intent_document']
    assert query['instruction_text'].encode() == original['instruction_text'].encode()
    residuals = query['residual_requirements']
    assert query['original_prompt_residuals'] == residuals
    assert {row['requirement_id'] for row in residuals} == {
        row['statement_id'] for row in original['intent_document']['statements']}
    opaque = next(row for row in residuals if row['statement']['kind'] == 'guard')
    assert opaque['statement']['normalized_text'] == original['instruction_text']
    assert opaque['statement']['grounding'] == 'grounded'
    assert opaque['selected_for_navigation'] is False
    assert opaque['original_source_refs'][0]['span'] == {
        'start_char': 0, 'end_char': len(original['instruction_text'])}
    selected = next(row for row in residuals if row['selected_for_navigation'])
    assert selected['statement']['normalized_text'] == original['navigation_text']
    assert selected['statement']['grounding'] == 'inferred'
    assert selected['statement']['review_status'] == 'machine_extracted'
    assert all(row['status'] == 'unknown' and row['behavioral_satisfaction'] is False for row in residuals)
    assert all(query[flag] is False and receipt[flag] is False for flag in corpus._AUTHORITY)


def test_known_prior_fit_and_review_stay_declared_development_lineage(corpus_inputs):
    arguments = deepcopy(corpus_inputs)
    history = arguments['historical_exposure']
    history['source_history'][0]['prior_fit_refs'] = ['synthetic:prior-unit-fit-not-runtime']
    history['query_history'][0]['prior_review_refs'] = ['synthetic:prior-unit-review-not-blind']
    receipt = _build(arguments)
    assert receipt['historical_exposure'] == history
    assert receipt['task_role_query_counts'] == {'train': 1, 'validation': 0, 'test': 0}
    assert receipt['historical_exposure_independently_verified'] is False
    assert receipt['frozen_before_new_fit_verified_here'] is False
    assert receipt['blind_holdout_evaluated'] is False
    assert receipt['source_generalization_established'] is False
    assert receipt['ranking_gain_generalizes'] is False
    assert receipt['training_inference_SQL_filesystem_checker_or_worker_operations_here'] == 0


def test_disjoint_source_and_query_families_admit_explicit_development_roles():
    arguments = _different_roles('def add(v):\n    return v + 7\n',
        'def stream(xs):\n    for item in xs:\n        yield item\n')
    receipt = _build(arguments)
    assert receipt['complete_candidate_count'] == 2
    assert receipt['task_role_query_counts'] == {'train': 1, 'validation': 1, 'test': 0}
    assert receipt['split_assignments'] == arguments['split_assignments']
    assert receipt['native_split_guard']['scope'] == 'declared_development_task_roles_not_blind_holdout'
    assert receipt['blind_holdout_evaluated'] is False
    assert all(summary['unjudged_count'] == 1 for summary in receipt['judgment_summary'])


def test_same_role_body_duplicates_remain_visible_without_removing_candidates():
    arguments = _different_roles('def add(v):\n    return v + 7\n',
        '# Separate synthetic module provenance.\ndef add(v):\n    return v + 7\n')
    arguments['split_assignments'] = {query['query_id']: 'train' for query in arguments['query_records']}
    receipt = _build(arguments)
    assert receipt['complete_candidate_count'] == 2
    duplicates = receipt['native_split_guard']['same_role_duplicates']
    assert {row['kind'] for row in duplicates} == {'normalized_body_sha256', 'alpha_shape_sha256'}
    assert all(row['task_role'] == 'train' and len(row['candidate_ids']) == 2 for row in duplicates)
    assert sum(len(bank) for bank in receipt['candidate_banks'].values()) == 2


def test_corpus_replay_uses_independent_originals_and_detaches_all_nested_inputs(corpus_inputs):
    pristine = deepcopy(corpus_inputs)
    receipt = _build(corpus_inputs)
    assert corpus_inputs == pristine
    assert receipt['corpus_sha256'] == 'sha256:' + _digest(
        {key: value for key, value in receipt.items() if key != 'corpus_sha256'})
    assert corpus.validate_terminal_intent_relevance_corpus(receipt, **corpus_inputs) == receipt
    receipt['queries'][0]['intent_document']['statements'].clear()
    receipt['candidate_banks'][corpus_inputs['source_records'][0]['codebase_id']][0]['features'][0] = 123.0
    receipt['historical_exposure']['source_history'][0]['prior_fit_refs'].append('synthetic:forged')
    assert corpus_inputs == pristine
    _reseal(receipt)
    with pytest.raises(corpus.IntentCorpusError, match='independent original-input replay'):
        corpus.validate_terminal_intent_relevance_corpus(receipt, **corpus_inputs)


@pytest.mark.parametrize('mutation', ['source', 'judgment', 'span', 'role', 'authority', 'unknown_eligibility'])
def test_resealed_corpus_cannot_replace_sources_labels_spans_roles_or_authority(corpus_inputs, mutation):
    receipt = _build(corpus_inputs)
    codebase = corpus_inputs['source_records'][0]['codebase_id']
    if mutation == 'source':
        receipt['sources'][0]['source_text'] = receipt['sources'][0]['source_text'].replace('>=', '>')
        receipt['sources'][0]['source_sha256'] = _sha(receipt['sources'][0]['source_text'].encode())
    elif mutation == 'judgment':
        judgment = receipt['reviewed_judgments'][0]
        judgment['label'] = 'negative_navigation' if judgment['label'] == 'positive' else 'positive'
    elif mutation == 'span':
        receipt['candidate_banks'][codebase][0]['source_binding']['start_byte'] += 1
    elif mutation == 'role':
        receipt['queries'][0]['task_role'] = 'test'
        receipt['split_assignments'][receipt['queries'][0]['query_id']] = 'test'
    elif mutation == 'authority':
        receipt['proof_authority'] = True
    else:
        receipt['eligible_for_plan'] = True
    _reseal(receipt)
    with pytest.raises(corpus.IntentCorpusError, match='independent original-input replay'):
        corpus.validate_terminal_intent_relevance_corpus(receipt, **corpus_inputs)
