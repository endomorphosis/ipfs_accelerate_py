"""Pure advisory join of owner-verified training artifacts and native IntentIR.

Reconstruction scores order an explicitly authored source focus. They do not
interpret the prompt, generate formulas, remove requirements or grant proof or
runtime authority. The caller must first replay the real learner artifacts;
this consumer checks inert bytes/AST identities without running inference.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import PurePosixPath

SCHEMA = 'terminal-codebase-intent-training-join@1'
MAX_BYTES = 32 * 1024 * 1024
_AUTHORITY = {name: False for name in ('semantic_alignment_verified',
    'source_semantics_verified', 'proof_authority', 'formalization_authority',
    'execution_authority', 'completion_authority', 'mutation_authority',
    'omission_authority', 'behavioral_satisfaction', 'whole_program_proved',
    'asymptotic_optimizer_convergence_proved')}
_POLICIES = {'trained': 'trained_checkpoint_reconstruction',
    'model_off': 'model_off_no_inference',
    'zero_heads': 'all_parameters_zero_diagnostic',
    'shuffled_order': 'same_trained_rank_reverse_permutation'}


class TerminalIntentTrainingJoinError(ValueError):
    """Source, native intent, artifact or advisory control identity differs."""


def _need(value, message):
    if value is not True:
        raise TerminalIntentTrainingJoinError(message)


def _wire(value, *, ascii=True):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=ascii, allow_nan=False).encode()


def _digest(value, *, ascii=True):
    return hashlib.sha256(_wire(value, ascii=ascii)).hexdigest()


def _json(value):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop(); count += 1
        _need(depth <= 48 and count <= 1_000_000, 'bounded JSON structure required')
        if type(item) is dict:
            _need(all(type(key) is str for key in item), 'string JSON keys required')
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _need(math.isfinite(item), 'finite JSON numbers required')
        else:
            _need(type(item) in (str, int, bool, type(None)), 'exact JSON values required')
    raw = _wire(value)
    _need(len(raw) <= MAX_BYTES, 'bounded advisory receipt input required')
    return json.loads(raw)


def _fields(value, names, label):
    _need(type(value) is dict and set(value) == set(names), 'exact '+label+' fields required')


def _zero(value):
    if type(value) is list:
        return [_zero(item) for item in value]
    _need(type(value) in (int, float) and math.isfinite(value), 'finite weight tensor required')
    return 0.0


def _source_units(raw, symbols):
    lines = raw.splitlines(keepends=True); offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result = {}
    for node in ast.parse(raw).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
            _need(node.name not in result, 'ambiguous top-level source symbol')
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            result[node.name] = {'line': node.lineno, 'end_line': node.end_lineno,
                'source_ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                'source_span': {'start_byte': start, 'end_byte': end,
                    'sha256': hashlib.sha256(raw[start:end]).hexdigest()}}
    return result


def build_terminal_intent_training_join(*, match_result, source_records,
        source_context, learner, training_receipt, checkpoint, features,
        learned_index, fixed_candidates, model_controls):
    """Rebuild the native match and bind advisory rows to exact current AST units.

The owner-verification prerequisite is not authenticated by a caller flag.
Actual trained inference, optimizer history and persistence are not replayed
here. The zero-parameter control checks finite shapes and feature MSE within
1e-6; that numerical check is not a theorem about Adam or program behavior.
"""
    from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import match_intent_codebase
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae

    supplied = _json({'match': match_result, 'sources': source_records,
        'context': source_context, 'learner': learner, 'receipt': training_receipt,
        'checkpoint': checkpoint, 'features': features, 'index': learned_index,
        'fixed': fixed_candidates, 'controls': model_controls})
    match, sources, context = supplied['match'], supplied['sources'], supplied['context']
    learner, receipt, checkpoint = supplied['learner'], supplied['receipt'], supplied['checkpoint']
    features, index, fixed, controls = supplied['features'], supplied['index'], supplied['fixed'], supplied['controls']
    rebuilt = match_intent_codebase(intent_document=match['intent_document'],
        source_text=match['intent_source']['text'], source_identity=match['intent_source']['identity'],
        query=match['query'], evidence_rows=match['evidence_rows'],
        current_source_snapshot=match['current_source_snapshot'])
    _need(rebuilt == match, 'original native match envelope differs from pure replay')
    _need(type(sources) is list and 1 <= len(sources) <= 256, 'complete bounded source records required')
    source_rows, source_bytes = {}, {}
    for row in sources:
        path = row['path']; relative = PurePosixPath(path)
        _need(type(path) is str and path and not relative.is_absolute()
            and str(relative) == path and '..' not in relative.parts
            and path not in source_rows, 'unique canonical source path required')
        _need(type(row['source_text']) is str, 'exact source text required')
        raw = row['source_text'].encode()
        _need(type(row['bytes']) is int and len(raw) == row['bytes']
            and hashlib.sha256(raw).hexdigest() == row['source_sha256'], 'source record bytes differ')
        source_rows[path], source_bytes[path] = row, raw
    _need(sum(len(raw) for raw in source_bytes.values()) <= 4_000_000, 'native source byte budget exceeded')
    snapshot = match['current_source_snapshot']; path = snapshot['source_path']
    _need(path in source_rows and snapshot['source_sha256'] == source_rows[path]['source_sha256']
        and snapshot['source_bytes'] == source_rows[path]['bytes'], 'native match current source differs')
    instruction = match['intent_source']['text'].encode()
    _need(any(raw == instruction for raw in source_bytes.values()), 'entire native intent source must equal an original public source record')
    ledger = learner['source_hashes']
    _need(type(ledger) is dict and ledger and all(p in source_rows
        and source_rows[p]['source_sha256'] == digest for p, digest in ledger.items()), 'current trained source ledger differs')
    _need(receipt['source_hashes'] == features['source_hashes'] == ledger
        and receipt['paths'] == features['paths'] and set(receipt['paths']) <= set(ledger), 'feature/training source selection differs')
    _need(learner['schema'] == ae.SCHEMA and receipt['schema'] == 'supervisor-code-autoencoder-training@1'
        and learner['domain'] == receipt['domain'] == features['domain'] == checkpoint['domain'] == ae.DOMAIN,
        'native code-only model namespace required')
    _need(learner['repository'] == receipt['repository'] and learner['output'] == receipt['output']
        and learner['metrics'] == receipt['metrics'] and learner['epochs_completed'] == receipt['metrics']['epochs'],
        'learner descriptor and original training receipt differ')
    _need(learner['receipt_sha256'] == _digest(receipt)
        and learner['checkpoint_sha256'] == receipt['checkpoint_sha256'] == _digest(checkpoint)
        and receipt['features_sha256'] == _digest(features) and receipt['index_sha256'] == _digest(index),
        'exact canonical native artifact or receipt digest differs')
    _need(checkpoint['schema'] == ae.CHECKPOINT_SCHEMA
        and checkpoint['architecture'] == 'tanh-linear-autoencoder@1'
        and checkpoint['feature_manifest_sha256'] == receipt['features_sha256']
        and features['schema'] == ae.FEATURE_SCHEMA and features['features'] == list(ae.FEATURES),
        'supported native checkpoint/feature profile required')
    _need(all(receipt[name] is False for name in ('proof_authority', 'formalization_authority', 'legal_state_mutated'))
        and learner['proof_authority'] is False and learner['authority'] == receipt['authority'] == index['authority'] == 'unverified_candidate_only'
        and index['ranking_role'] == 'nomination_order_only'
        and all(index[name] is False for name in ('omission_authority', 'formalization_authority', 'proof_authority')),
        'reconstruction artifacts grant no authority')
    current_rows, unsupported = ae._features({p: source_bytes[p] for p in ledger},
        receipt['paths'], receipt['max_functions'])
    _need(features['rows'] == current_rows and features['unsupported'] == unsupported,
        'complete feature rows differ from current source AST replay')
    rows = {row['row_id']: row for row in current_rows}
    _need(len(rows) == len(current_rows) == learner['sample_count'] == receipt['sample_count'], 'unique complete feature row population required')
    _need(learner['source_count'] == receipt['source_count'] == len(ledger), 'trained source population differs')
    ranks = index['ranks']; by_id = {row['row_id']: row for row in ranks}
    _need(len(by_id) == len(ranks) == len(rows) and set(by_id) == set(rows), 'unique complete learned rank population required')
    width = checkpoint['latent_width']
    _need(type(width) is int and 2 <= width <= 16 and checkpoint['input_width'] == len(ae.FEATURES), 'native model dimensions differ')
    def check_rank(row):
        feature = rows[row['row_id']]
        _fields(row, {'row_id', 'path', 'symbol', 'line', 'reconstruction_error', 'latent'}, 'learned rank')
        _need(type(row['line']) is int and all(row[name] == feature[name] for name in ('path', 'symbol', 'line')), 'learned row identity reassigned')
        _need(type(row['reconstruction_error']) in (int, float) and math.isfinite(row['reconstruction_error'])
            and row['reconstruction_error'] >= 0 and type(row['latent']) is list and len(row['latent']) == width
            and all(type(x) in (int, float) and math.isfinite(x) for x in row['latent']), 'finite learned reconstruction required')
    for row in ranks:
        check_rank(row)
    _need(ranks == sorted(ranks, key=lambda row: (-row['reconstruction_error'], row['row_id']))
        and index['mean_reconstruction_error'] == sum(by_id[row['row_id']]['reconstruction_error'] for row in current_rows)/len(ranks)
        and index['mean_reconstruction_error'] == receipt['metrics']['after_reconstruction_loss'],
        'native reconstruction score or canonical ordering differs')
    _need(learner['ranks'] == [{key: row[key] for key in ('row_id', 'path', 'symbol', 'line', 'reconstruction_error')} for row in ranks], 'descriptor learned rank rows differ')
    _fields(context, {'schema', 'source_path', 'source_sha256', 'source_bytes', 'model_domain', 'training_source_hashes', 'checkpoint'}, 'training source context')
    expected_context = {'schema': 'terminal-codebase-training-source-context@1',
        'source_path': path, 'source_sha256': source_rows[path]['source_sha256'], 'source_bytes': source_rows[path]['bytes'],
        'model_domain': ae.DOMAIN, 'training_source_hashes': ledger,
        'checkpoint': {'checkpoint_sha256': learner['checkpoint_sha256'], 'receipt_sha256': learner['receipt_sha256'],
            'features_sha256': receipt['features_sha256'], 'index_sha256': receipt['index_sha256']}}
    _need(context == expected_context and snapshot['source_context_sha256'] == _digest(context, ascii=False),
        'independent current source/checkpoint/context root differs')
    for row in match['evidence_rows']:
        _need(row['key_relationship']['dimensions']['source'] == context, 'conditional evidence refers to another model context')
    _fields(fixed, {'lexical', 'kg'}, 'fixed lexical/KG candidates')
    _need(type(fixed['kg']) is list, 'complete passive KG rows required')
    actual_units = _source_units(source_bytes[path], {unit['symbol'] for unit in snapshot['source_unit_bindings']}); pool = []
    for unit in snapshot['source_unit_bindings']:
        _need(unit['symbol'] in actual_units and all(unit[name] == actual_units[unit['symbol']][name]
            for name in ('line', 'end_line', 'source_ast_sha256', 'source_span')), 'native source unit no longer matches exact AST/span')
        selected = [row for row in current_rows if row['path'] == path and row['symbol'] == unit['symbol']]
        _need(len(selected) == 1 and selected[0]['line'] == unit['line']
            and selected[0]['ast_sha256'] == unit['source_ast_sha256'], 'native source unit has no unique trained AST row')
        if unit['symbol'] in match['query']['symbols']:
            row = selected[0]
            identity = {'source_context_sha256': snapshot['source_context_sha256'], 'source_unit': unit, 'feature_row_id': row['row_id']}
            pool.append({'candidate_id': 'sha256:'+_digest(identity), **identity,
                'feature_row': row, 'trained_rank': by_id[row['row_id']],
                'matching_rule': 'explicit_authored_reviewed_source_focus_only', **_AUTHORITY})
    pool.sort(key=lambda row: row['candidate_id'])
    _need(type(controls) is list and len(controls) == 4, 'four separately declared controls required')
    outcomes, seen = [], set()
    for control in controls:
        name = control['name']; _need(name in _POLICIES and name not in seen, 'unique closed control names required'); seen.add(name)
        fields = {'name', 'ranking', 'input_features_sha256', 'source_hashes', 'checkpoint_sha256', 'inference_policy', 'training_executed'}
        _fields(control, fields | ({'control_weights_sha256'} if name == 'zero_heads' else set()), 'model control')
        _need(control['input_features_sha256'] == receipt['features_sha256'] and control['source_hashes'] == ledger
            and control['checkpoint_sha256'] == learner['checkpoint_sha256'] and control['training_executed'] is False
            and control['inference_policy'] == _POLICIES[name], 'control changes original source/model/feature binding')
        ranking = control['ranking']
        if name == 'model_off':
            _need(ranking == [], 'model-off cannot claim inference rows')
        elif name in ('trained', 'shuffled_order'):
            _need(ranking == (ranks if name == 'trained' else list(reversed(ranks))), 'control is not the declared actual rank ordering')
        else:
            _need(control['control_weights_sha256'] == _digest(_zero(checkpoint['weights'])), 'zero diagnostic parameters differ')
            _need(type(ranking) is list and len(ranking) == len(rows) and {row['row_id'] for row in ranking} == set(rows), 'complete zero control rows required')
            for row in ranking:
                check_rank(row); expected = sum(x*x for x in rows[row['row_id']]['features'])/len(ae.FEATURES)
                _need(all(x == 0 for x in row['latent']) and abs(row['reconstruction_error']-expected) <= 1e-6,
                    'zero control is not zero-latent feature reconstruction')
        selected_ids = {row['feature_row_id']: row['candidate_id'] for row in pool}
        outcomes.append({'control': control, 'fixed_candidate_ids': [row['candidate_id'] for row in pool],
            'learned_candidate_order': [selected_ids[row['row_id']] for row in ranking if row['row_id'] in selected_ids],
            'residual_requirements': match['residual_requirements'], 'fixed_candidates_sha256': _digest(fixed), **_AUTHORITY})
    outcomes.sort(key=lambda row: row['control']['name'])
    result = {'schema': SCHEMA, 'original_native_match': match, 'source_records': sources,
        'source_context': context, 'fixed_candidates': fixed, 'source_unit_nominations': pool,
        'model_control_outcomes': outcomes, 'residual_requirements': match['residual_requirements'],
        'artifact_roots': context['checkpoint'], 'full_trained_row_inventory_sha256': _digest(current_rows),
        'features_replayed_from_current_AST_here': True, 'trained_inference_replayed_here': False,
        'owner_artifact_validation': 'required_caller_prerequisite_not_authenticated_here',
        'learned_ranking_role': 'reconstruction_nomination_order_only', 'intent_similarity_or_formula_generation': False,
        'training_inference_SQL_filesystem_or_worker_operations_here': 0,
        'persistence_verified_here': False, 'zero_control_absolute_error_tolerance': 1e-6, **_AUTHORITY}
    result['join_sha256'] = 'sha256:'+_digest(result)
    return _json(result)


def validate_terminal_intent_training_join(receipt, **original_arguments):
    """Require exact replay against independently supplied original owner inputs."""
    rebuilt = build_terminal_intent_training_join(**original_arguments)
    _need(_json(receipt) == rebuilt, 'entire advisory join differs from original-input replay')
    return rebuilt


__all__ = ['SCHEMA', 'TerminalIntentTrainingJoinError',
    'build_terminal_intent_training_join', 'validate_terminal_intent_training_join']
