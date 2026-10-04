"""Pinned public batching-grounding controls with no optimizer or checkers.

Only the exact retained public-source envelope is read. Native IntentIR,
source accounting, projection and finite witness construction are passive;
these tests do not run the baseline, Lean, solvers, workers or native SQL.
"""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_corpus as corpus
from benchmarks.agent_supervisor.container_coding import terminal_codebase_requirement_grounding as grounding


_ENVELOPE_SHA = 'baf3ea631e06ff366768f489d67026d0caabd26a87d5561818db6d03b40eb508'
_CORPUS_SHA = 'sha256:43da7b53f322e21b495d0352b6ca90378399ffe0fc49864e729a9edf2a8306c6'
_BASELINE_SHA = '547d230d5e6d93197803480cb81cab0ec6ac63fcfa0a24b246f549523f216c4e'
_INSTRUCTION_SHA = '0817374767533fbf51bb93345ee1bba266263c216c6cc55c096e71b4a1d37dfc'
_QUERY_ID = 'batching-build-plan'
_REVIEW_REF = 'reviewed-public-batching-constraints@1'
_BASELINE_PATH = 'environment/task_file/scripts/baseline_packer.py'


@pytest.fixture(scope='module')
def public_inputs():
    supplied = os.environ.get('TERMINAL_BATCHING_GROUNDING_FROZEN_ENVELOPE')
    if supplied is not None:
        path = Path(supplied)
        assert path.is_absolute(), 'captured public fixture pointer must be absolute'
    else:
        path = (Path(__file__).resolve().parents[4] / 'artifacts' / 'codebase_ir_terminal_bench'
            / 'terminal-codebase-ir-expanded-transfer-20261004-01' / 'frozen-corpus'
            / (_ENVELOPE_SHA + '.json'))
        if not path.is_file():
            pytest.skip('exact retained public fixture unavailable; set TERMINAL_BATCHING_GROUNDING_FROZEN_ENVELOPE')
    assert path.resolve(strict=True) == path
    raw = path.read_bytes()
    assert len(raw) == 2248025 and hashlib.sha256(raw).hexdigest() == _ENVELOPE_SHA
    envelope = json.loads(raw)
    assert envelope['schema'] == 'terminal-intent-relevance-frozen-envelope@1'
    assert envelope['corpus']['corpus_sha256'] == _CORPUS_SHA
    return deepcopy(envelope)


def _arguments(public_inputs, **overrides):
    value = {'corpus_receipt': public_inputs['corpus'],
        'original_inputs': public_inputs['original_inputs'], 'expected_corpus_sha256': _CORPUS_SHA,
        'query_id': _QUERY_ID, 'review_ref': _REVIEW_REF}
    value.update(overrides)
    return deepcopy(value)


def _build(public_inputs, **overrides):
    return grounding.build_terminal_batching_requirement_grounding(**_arguments(public_inputs, **overrides))


def _original_query(arguments):
    return next(row for row in arguments['original_inputs']['query_records'] if row['query_id'] == _QUERY_ID)


def _baseline(arguments):
    return next(row for row in arguments['original_inputs']['source_records']
        if row['codebase_id'] == 'batching' and row['path'] == _BASELINE_PATH)


def _resealed_arguments(arguments):
    frozen = corpus.build_terminal_intent_relevance_corpus(**arguments['original_inputs'])
    arguments['corpus_receipt'] = frozen
    arguments['expected_corpus_sha256'] = frozen['corpus_sha256']
    return arguments


def _change_baseline_and_native_references(arguments):
    """Make a new valid corpus; its source still fails the fixed public profile."""
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae
    source = _baseline(arguments)
    original_text = source['source_text']
    edited = original_text.replace('>=', '>')
    assert edited != original_text
    before, unsupported = ae._features({source['path']: original_text.encode()}, [source['path']], 1024)
    after, other_unsupported = ae._features({source['path']: edited.encode()}, [source['path']], 1024)
    assert unsupported == other_unsupported == []
    assert [row['features'] for row in before] == [row['features'] for row in after]
    assert [row['ast_sha256'] for row in before] != [row['ast_sha256'] for row in after]
    source['source_text'] = edited
    source['source_sha256'] = hashlib.sha256(edited.encode()).hexdigest()
    native_rows = {(row['symbol'], row['line']): row for row in after}
    for row in arguments['original_inputs']['historical_exposure']['source_history']:
        if (row['codebase_id'], row['path']) == ('batching', _BASELINE_PATH):
            row['source_sha256'] = source['source_sha256']
    for row in arguments['original_inputs']['reviewed_judgments']:
        ref = row['candidate_ref']
        if (ref['codebase_id'], ref['path']) == ('batching', _BASELINE_PATH):
            native = native_rows[(ref['symbol'], ref['line'])]
            ref['source_sha256'] = source['source_sha256']
            ref['ast_sha256'] = native['ast_sha256']
    return _resealed_arguments(arguments)


def _change_instruction_and_native_references(arguments):
    query = _original_query(arguments)
    query['instruction_text'] += '\nA changed development-only requirement.'
    query['instruction_sha256'] = hashlib.sha256(query['instruction_text'].encode()).hexdigest()
    for source in query['intent_document']['sources']:
        if source['source_uri'] == query['instruction_uri']:
            for key in ('content_sha256', 'source_id', 'source_revision'):
                source[key] = query['instruction_sha256']
            source['span']['end_char'] = len(query['instruction_text'])
    for statement in query['intent_document']['statements']:
        if statement['kind'] == 'guard':
            statement['normalized_text'] = query['instruction_text']
    for row in arguments['original_inputs']['historical_exposure']['query_history']:
        if row['query_id'] == query['query_id']:
            row['instruction_sha256'] = query['instruction_sha256']
    return _resealed_arguments(arguments)


@pytest.fixture(scope='module')
def grounded(public_inputs):
    return _build(public_inputs)


@pytest.mark.parametrize('query_id', ['batching-build-plan', 'batching-representative'])
def test_either_admitted_query_preserves_its_entire_dual_source_native_request(public_inputs, query_id):
    arguments = _arguments(public_inputs, query_id=query_id)
    result = grounding.build_terminal_batching_requirement_grounding(**arguments)
    expected = next(row for row in public_inputs['corpus']['queries'] if row['query_id'] == query_id)
    assert result['original_query'] == expected
    assert result['original_query']['instruction_sha256'] == _INSTRUCTION_SHA
    assert len(result['original_query']['intent_document']['sources']) == 2
    assert len(result['original_query']['residual_requirements']) == 2
    assert result['original_query']['original_prompt_residuals'] == expected['original_prompt_residuals']
    assert all(row['status'] == 'unknown' and row['behavioral_satisfaction'] is False
        for row in result['original_query']['residual_requirements'])
    guard = next(row for row in result['original_query']['intent_document']['statements'] if row['kind'] == 'guard')
    assert guard['normalized_text'] == expected['instruction_text']
    assert result['planning_handoff'] == 'abstained'
    assert result['checker_calls'] == 0
    assert all(result[name] == [] for name in ('canonical_tasks', 'facts', 'effects'))
    assert all(result[name] is False for name in grounding.AUTHORITY)
    assert grounding.validate_terminal_batching_requirement_grounding(result, **arguments) == result


@pytest.mark.parametrize('pin', [None, 'sha256:' + '0' * 64])
def test_external_corpus_pin_is_mandatory_and_exact(public_inputs, pin):
    with pytest.raises(grounding.RequirementGroundingError):
        _build(public_inputs, expected_corpus_sha256=pin)


@pytest.mark.parametrize('review_ref', ['', None])
def test_reviewed_candidates_require_an_explicit_review_reference(public_inputs, review_ref):
    with pytest.raises(grounding.RequirementGroundingError, match='review reference'):
        _build(public_inputs, review_ref=review_ref)


@pytest.mark.parametrize('mutation', ['resealed_code', 'resealed_instruction', 'resealed_query_hash'])
def test_resealed_source_instruction_or_native_query_hash_cannot_change_public_profile(public_inputs, mutation):
    arguments = _arguments(public_inputs)
    if mutation == 'resealed_code':
        arguments = _change_baseline_and_native_references(arguments)
        # New native corpus is well formed, including every new judgment hash.
        assert corpus.validate_terminal_intent_relevance_corpus(arguments['corpus_receipt'],
            **arguments['original_inputs']) == arguments['corpus_receipt']
    elif mutation == 'resealed_instruction':
        arguments = _change_instruction_and_native_references(arguments)
        assert corpus.validate_terminal_intent_relevance_corpus(arguments['corpus_receipt'],
            **arguments['original_inputs']) == arguments['corpus_receipt']
    else:
        query = next(row for row in arguments['corpus_receipt']['queries'] if row['query_id'] == _QUERY_ID)
        query['native_intent_sha256'] = '0' * 64
        core = {key: value for key, value in arguments['corpus_receipt'].items() if key != 'corpus_sha256'}
        arguments['corpus_receipt']['corpus_sha256'] = 'sha256:' + corpus._digest(core)
        arguments['expected_corpus_sha256'] = arguments['corpus_receipt']['corpus_sha256']
    with pytest.raises(grounding.RequirementGroundingError):
        grounding.build_terminal_batching_requirement_grounding(**arguments)


@pytest.mark.parametrize('mutation', ['wrong_codebase', 'native_residual_state'])
def test_wrong_codebase_or_resealed_native_state_cannot_supply_grounding(public_inputs, mutation):
    arguments = _arguments(public_inputs)
    if mutation == 'wrong_codebase':
        other = next(row for row in arguments['original_inputs']['query_records'] if row['codebase_id'] != 'batching')
        arguments['query_id'] = other['query_id']
    else:
        query = next(row for row in arguments['corpus_receipt']['queries'] if row['query_id'] == _QUERY_ID)
        query['residual_requirements'][0]['status'] = 'satisfied'
        core = {key: value for key, value in arguments['corpus_receipt'].items() if key != 'corpus_sha256'}
        arguments['corpus_receipt']['corpus_sha256'] = 'sha256:' + corpus._digest(core)
        arguments['expected_corpus_sha256'] = arguments['corpus_receipt']['corpus_sha256']
    with pytest.raises(grounding.RequirementGroundingError):
        grounding.build_terminal_batching_requirement_grounding(**arguments)


@pytest.mark.parametrize('mutation', ['remove_guard', 'wrong_source_ref', 'shorten_original_span'])
def test_original_opaque_guard_and_exact_full_source_ref_cannot_be_replaced(public_inputs, mutation):
    arguments = _arguments(public_inputs)
    query = _original_query(arguments)
    document = query['intent_document']
    guard = next(row for row in document['statements'] if row['kind'] == 'guard')
    if mutation == 'remove_guard':
        document['statements'].remove(guard)
    elif mutation == 'wrong_source_ref':
        guard['source_ref_ids'] = ['authored-navigation']
    else:
        original_ref = next(row for row in document['sources'] if row['source_uri'] == query['instruction_uri'])
        original_ref['span']['end_char'] -= 1
    with pytest.raises(grounding.RequirementGroundingError, match='native corpus replay refused'):
        grounding.build_terminal_batching_requirement_grounding(**arguments)


def test_native_ledger_accounts_for_every_character_and_utf8_byte_including_performance_frontiers(grounded):
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import validate_intent_requirement_ledger
    original = grounded['original_query']['instruction_text']
    ledger = grounded['requirement_ledger']
    assert len(original) == 4346 and len(original.encode()) == 4365
    assert ledger['source']['characters'] == len(original)
    assert ledger['source']['bytes'] == len(original.encode())
    assert ledger['source']['sha256'] == _INSTRUCTION_SHA
    units = ledger['source_units']
    cursor = 0
    for unit in units:
        assert unit['start_char'] == cursor
        assert unit['text'] == original[unit['start_char']:unit['end_char']]
        assert unit['start_byte'] == len(original[:unit['start_char']].encode())
        assert unit['end_byte'] == len(original[:unit['end_char']].encode())
        assert unit['sha256'] == hashlib.sha256(unit['text'].encode()).hexdigest()
        cursor = unit['end_char']
    assert cursor == len(original)
    assert ''.join(row['text'] for row in units) == original
    assert sum(row['end_byte'] - row['start_byte'] for row in units) == 4365
    unresolved = [row for row in units if row['disposition'] == 'unsupported']
    assert unresolved and all(not row['requirement_ids'] for row in unresolved)
    unresolved_text = ''.join(row['text'] for row in unresolved)
    assert '3.0e11' in unresolved_text and '4.8e10' in unresolved_text
    assert 'input_data files unchanged' in unresolved_text
    assert ledger['source_accounting_complete'] is True and ledger['semantic_support_complete'] is False
    assert grounded['coverage']['performance_thresholds_and_file_immutability'] == 'unknown_unreported'
    assert all(row['status'] == 'unknown' for row in grounded['coverage']['unreported_source_units'])
    assert validate_intent_requirement_ledger(ledger, source_text=original,
        source_report=grounded['reviewed_source_report']) == ledger


def test_four_original_bullets_link_eight_native_required_atomic_candidates(grounded):
    report = grounded['reviewed_source_report']
    document = grounded['reviewed_native_intent']
    ledger = grounded['requirement_ledger']
    links = grounded['requirement_links']
    assert len(report['units']) == len(report['candidates']) == len(document['sources']) == 4
    assert len(document['statements']) == len(ledger['requirements']) == len(links) == 8
    expected_predicates = {'included_exactly_once', 'heads_align_equals', 'hidden_align_equals',
        'seq_align_covers_prompt', 'seq_align_multiple_of', 'global_unique_shapes_at_most',
        'one_record_per_request', 'identical_shapes_within_batch'}
    assert {row['predicate'] for row in document['statements']} == expected_predicates
    assert all(row['kind'] == 'goal' and row['modality'] == 'required'
        and row['grounding'] == 'inferred' and row['review_status'] == 'machine_extracted'
        and row['confidence'] == 0.0 for row in document['statements'])
    assert sorted(len(row['candidate_intent_ir']['statements']) for row in report['candidates']) == [1, 1, 2, 4]
    requirements = {row['requirement_id']: row for row in ledger['requirements']}
    for link in links:
        requirement = requirements[link['requirement_id']]
        assert requirement['kind'] == 'goal' and requirement['modality'] == 'required'
        assert requirement['representation'] == 'native_statement'
        assert requirement['statement_ids'] == [link['statement_id']]
        assert link['source_unit']['text'] == grounded['original_query']['instruction_text'][
            link['source_unit']['start_char']:link['source_unit']['end_char']]
        assert link['original_source_ref']['span'] == {
            'start_char': link['source_unit']['start_char'], 'end_char': link['source_unit']['end_char']}
        assert link['status'] == 'unknown'
        assert link['link_status'] == 'reviewed_syntactic_candidate_not_satisfaction'
        assert all(link[name] is False for name in grounding.AUTHORITY)


def test_all_six_source_functions_and_nested_helper_are_current_syntactic_context_only(grounded, public_inputs):
    context = grounded['source_context']
    source = context['source_record']
    expected = public_inputs['corpus']['candidate_banks']['batching']
    assert source['path'] == _BASELINE_PATH and source['source_sha256'] == _BASELINE_SHA
    assert len(source['source_text'].encode()) == 4380
    assert hashlib.sha256(source['source_text'].encode()).hexdigest() == _BASELINE_SHA
    assert context['complete_candidate_bank'] == expected
    assert {row['symbol'] for row in expected} == {'load_requests', '_plan_for_requests',
        '_plan_for_requests.assign_rep', '_write_plan', 'build_plan', 'main'}
    assert {row['symbol'] for row in context['selected_source_units']} == {
        '_plan_for_requests', '_plan_for_requests.assign_rep', 'build_plan'}
    nodes = {}
    class Functions(ast.NodeVisitor):
        def __init__(self):
            self.prefix = []
        def visit_FunctionDef(self, node):
            self.prefix.append(node.name)
            nodes['.'.join(self.prefix)] = node
            self.generic_visit(node)
            self.prefix.pop()
    Functions().visit(ast.parse(source['source_text']))
    for row in expected:
        assert hashlib.sha256(ast.dump(nodes[row['symbol']], include_attributes=False).encode()).hexdigest() == row['ast_sha256']
    assert context['syntactic_observations']['independent_bucket_planner_call_arguments'] == [
        ['reqs1', 'GRAN', 'MAX_SHAPES'], ['reqs2', 'GRAN', 'MAX_SHAPES']]
    assert context['syntactic_observations']['representative_helper_is_nested'] is True
    assert context['generic_program_derivation_status'] == 'unsupported_fragment_not_invoked'
    assert context['facts_are_syntactic_only'] is True
    assert context['source_runtime_equivalence_verified'] is False


def test_sixteen_family_views_preserve_unmodeled_frontiers_and_grant_no_proof(grounded):
    projection = grounded['family_projection']
    reports = projection['projections']
    assert len(projection['requested_families']) == 16
    assert len(reports) == 17
    assert len({row['family_id'] for row in reports}) == 16
    assert {row['profile_id'] for row in reports if row['family_id'] == 'transition_system'} == {None, 'tla_plus'}
    unsupported_families = {'separation_logic', 'hyperproperty', 'epistemic', 'doxastic',
        'refinement', 'concurrency', 'session_process', 'cryptographic_protocol'}
    assert all(row['status'] == 'unsupported' for row in reports if row['family_id'] in unsupported_families)
    assert unsupported_families <= {row['family_id'] for row in reports}
    assert projection['all_requested_families_projected'] is False
    assert projection['training_executed'] is False
    assert projection['provider_calls'] == projection['external_backend_calls'] == 0
    for row in reports:
        assert row['authority'] == 'unverified_candidate_only'
        assert all(row[name] is False for name in (
            'proof_authority', 'execution_authority', 'completion_authority', 'omission_authority', 'source_semantics_verified'))
    assert grounded['coverage']['native_family_count'] == 16
    assert grounded['coverage']['complete_original_meaning_formalized'] is False


def test_conditional_two_bucket_model_has_eight_members_each_and_sixteen_global_shapes(grounded):
    witness = grounded['finite_witness']
    buckets = witness['buckets']
    assert len(buckets) == 2
    all_shapes, all_ids = [], []
    for bucket in buckets:
        sequences = bucket['aligned_sequences']
        assert len(sequences) == len(set(sequences)) == bucket['representative_count'] == 8
        assert sequences == sorted(sequences) == bucket['unique_sequence_values'] == bucket['representatives']
        assert bucket['no_reduction_branch'] is True
        assert bucket['shape_triples'] == [[value, 32, 4096] for value in sequences]
        for value in sequences:
            assert value > 0 and value % 64 == 0 and ((value + 63) // 64) * 64 == value
            assert next(rep for rep in bucket['representatives'] if rep >= value) == value
        all_shapes.extend(bucket['shape_triples'])
        all_ids.extend(bucket['request_ids'])
    assert len(all_ids) == len(set(all_ids)) == 16
    assert len(all_shapes) == len({tuple(row) for row in all_shapes}) == 16
    assert all_shapes == witness['all_shape_triples']
    assert witness['shape_components'] == ['seq_align', 'heads_align', 'hidden_align']
    assert witness['global_unique_shape_count'] == 16 and witness['required_global_cap'] == 8
    assert witness['model_global_cap_satisfied'] is False and witness['model_counterexample_present'] is True
    assert witness['actual_program_violation_established'] is False
    assert witness['actual_benchmark_inputs_used'] is False
    assert witness['dependency_semantics_verified'] is False
    assert witness['checker_executed'] is False and witness['lean_validation_status'] == 'not_run'
    assert any('cost_model.align' in text and 'unverified' in text for text in witness['assumptions'])
    assert any('conditional model' in text for text in witness['assumptions'])
    sources = grounded['lean_sources']
    assert set(sources) == {'intent', 'finite_countermodel', 'false_global_cap_control'}
    assert 'distinctShapes.length = 16' in sources['finite_countermodel']
    assert '¬ distinctShapes.length ≤ 8' in sources['finite_countermodel']
    assert 'incorrect_global_shape_cap : distinctShapes.length ≤ 8' in sources['false_global_cap_control']
    assert sources['finite_countermodel'] != sources['false_global_cap_control']
    assert grounded['lean_source_sha256'] == {key: hashlib.sha256(value.encode()).hexdigest()
        for key, value in sources.items()}


@pytest.mark.parametrize('mutation', ['ledger_requirement', 'ledger_source_gap', 'projection', 'witness',
    'original_residual', 'source_bank', 'authority', 'lean_source'])
def test_receipt_reseal_cannot_omit_native_evidence_or_upgrade_its_truth(public_inputs, grounded, mutation):
    result = deepcopy(grounded)
    if mutation in ('ledger_requirement', 'ledger_source_gap'):
        ledger = result['requirement_ledger']
        if mutation == 'ledger_requirement':
            ledger['requirements'].pop()
        else:
            gap = next(row for row in ledger['source_units'] if row['disposition'] == 'unsupported')
            ledger['source_units'].remove(gap)
        ledger['ledger_sha256'] = hashlib.sha256(json.dumps(
            {key: value for key, value in ledger.items() if key != 'ledger_sha256'},
            sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    elif mutation == 'projection':
        projection = result['family_projection']
        removed = projection['projections'].pop()
        projection['requested_families'].remove(removed['family_id'])
        projection['report_sha256'] = corpus._digest({key: value for key, value in projection.items()
            if key != 'report_sha256'})
    elif mutation == 'witness':
        result['finite_witness']['global_unique_shape_count'] = 8
        result['finite_witness']['model_global_cap_satisfied'] = True
    elif mutation == 'original_residual':
        for key in ('residual_requirements', 'original_prompt_residuals'):
            result['original_query'][key][0]['status'] = 'satisfied'
            result['original_query'][key][0]['behavioral_satisfaction'] = True
    elif mutation == 'source_bank':
        context = result['source_context']
        context['complete_candidate_bank'] = [row for row in context['complete_candidate_bank']
            if row['symbol'] != '_plan_for_requests.assign_rep']
    elif mutation == 'authority':
        result['proof_authority'] = True
        result['coverage']['proof_authority'] = True
    else:
        result['lean_sources']['finite_countermodel'] = 'import Std\nexample : (1 : Nat) = 2 := by decide\n'
        result['lean_source_sha256']['finite_countermodel'] = hashlib.sha256(
            result['lean_sources']['finite_countermodel'].encode()).hexdigest()
    result['grounding_sha256'] = corpus._digest({key: value for key, value in result.items()
        if key != 'grounding_sha256'})
    with pytest.raises(grounding.RequirementGroundingError, match='exact source/native replay'):
        grounding.validate_terminal_batching_requirement_grounding(result, **_arguments(public_inputs))


def test_metadata_delta_keeps_every_family_target_full_source_bank_and_original_query(public_inputs, grounded):
    records = grounding.project_terminal_batching_grounding_metadata(grounded, **_arguments(public_inputs))
    assert set(records) == {'batching_requirement_ledger', 'batching_native_intent', 'batching_requirement_links',
        'batching_projection_reports', 'batching_native_targets', 'batching_source_context',
        'batching_finite_witness', 'batching_grounding_coverage'}
    assert records['batching_requirement_ledger'] == [grounded['requirement_ledger']]
    assert records['batching_requirement_links'] == grounded['requirement_links']
    assert records['batching_projection_reports'] == grounded['family_projection']['projections']
    assert len(records['batching_projection_reports']) == 17
    assert records['batching_native_intent'][0]['original_query'] == grounded['original_query']
    assert records['batching_source_context'][0]['complete_candidate_bank'] == public_inputs['corpus']['candidate_banks']['batching']
    assert len(records['batching_source_context'][0]['complete_candidate_bank']) == 6
    assert [row['projection_index'] for row in records['batching_native_targets']] == list(
        range(len(grounded['family_projection']['native_targets']['projections'])))
    coverage, = records['batching_grounding_coverage']
    recovered = {**coverage['family_projection_envelope'],
        'projections': records['batching_projection_reports'],
        'native_targets': {**coverage['native_targets_envelope'],
            'projections': [row['target'] for row in records['batching_native_targets']]}}
    assert recovered == grounded['family_projection']
    assert coverage['coverage']['contracts'] == 'unchanged_unknown_empty'
    assert coverage['coverage']['planning_handoff'] == 'abstained'
    assert all(coverage['coverage'][name] == 0 for name in ('lean_checker_calls', 'training_calls', 'inference_calls', 'SQL_calls'))
    assert not {'contracts', 'tasks', 'facts', 'effects'} & set(records)
    assert all(len(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode()) <= 262144 for rows in records.values() for row in rows)


def test_partial_native_ledger_refuses_actual_symbolic_operation_gate_before_task_construction(grounded):
    from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import (
        IntentRequirementAdapterError, validate_symbolic_operations)
    ledger = grounded['requirement_ledger']
    assert ledger['source_accounting_complete'] is True
    assert any(row['disposition'] == 'unsupported' for row in ledger['source_units'])
    contract = {'schema': 'intent-symbolic-operation-contract@1',
        'ledger_sha256': ledger['ledger_sha256'], 'review_ref': 'review:partial-public-batching-ledger',
        'interpretation_scope': 'administrative_requirement_task_coverage', 'operations': [],
        'semantic_alignment_verified': False, 'proof_authority': False,
        'execution_authority': False, 'completion_authority': False}
    with pytest.raises(IntentRequirementAdapterError,
            match='^symbolic planning does not support unresolved source units$'):
        validate_symbolic_operations(contract, ledger=ledger, requirements=[])


def test_inputs_receipt_and_metadata_are_independently_detached(public_inputs, grounded):
    arguments = _arguments(public_inputs)
    before = deepcopy(arguments)
    result = grounding.build_terminal_batching_requirement_grounding(**arguments)
    pristine = deepcopy(result)
    _baseline(arguments)['source_text'] = 'changed caller state'
    assert result == pristine
    result['source_context']['complete_candidate_bank'][0]['source_binding']['start_byte'] += 1
    result['reviewed_native_intent']['statements'][0]['predicate'] = 'changed returned state'
    assert _arguments(public_inputs) == before
    assert grounded == pristine
    projected = grounding.project_terminal_batching_grounding_metadata(grounded, **before)
    projected['batching_requirement_ledger'][0]['requirements'].clear()
    assert len(grounded['requirement_ledger']['requirements']) == 8
