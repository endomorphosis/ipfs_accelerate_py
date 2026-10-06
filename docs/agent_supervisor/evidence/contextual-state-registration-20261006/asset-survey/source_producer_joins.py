#!/usr/bin/env python3
"""Read-only exact cached-vector joins to preserved original producer receipts."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_retained_custody_survey_metadata_helpers', ROOT / 'survey.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, pin, digest = helper.read, helper.pin, helper.digest
BASE = helper.BASE
LOGS = BASE / 'external/ipfs_datasets/workspace/test-logs'


def join_lookup(records, source_key, vector_key):
    result = {}
    for row in records:
        source = row[source_key]
        result.setdefault(source, set()).add(digest(row[vector_key]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    pin(ROOT / 'survey.py')
    custody_path = ROOT / 'contextual-state-custody-survey.json'
    custody = read(custody_path)
    manifest_path = LOGS / 'decoder-selected-continuation-20261004/training-manifest.json'
    manifest = read(manifest_path)
    native_inputs_path = LOGS / 'decoder-native-dimensions-20261003/preparation-r2/results/dimension-inputs.json'
    native_inputs = read(native_inputs_path)
    assert pin(native_inputs_path)[0]['sha256'] == manifest['inputs'][str(native_inputs_path)]
    inventory_path = LOGS / 'decoder-multiplicative-source-training-20261003/clause-cache-inventory.json'
    inventory = read(inventory_path)
    assert pin(inventory_path)[0]['sha256'] == manifest['inputs'][str(inventory_path)]
    small_producer_path = LOGS / 'decoder-distillation-limits-20261002/smoke-r2/results/embedding-results.json'
    small_producer = read(small_producer_path)
    assert pin(small_producer_path)[0]['sha256'] == manifest['inputs'][str(small_producer_path)]
    clauses384 = {}
    for split in ('train', 'validation'):
        path = LOGS / f'decoder-distillation-limits-20261002/prepared-r1/prepared384-{split}.json'
        declared = inventory['artifacts'][str(path)]
        clauses384[split] = read(path, {'path': str(path), **declared})
    metadata384_path = LOGS / 'decoder-distillation-limits-20261002/prepared-r1/metadata384.json'
    metadata384 = read(metadata384_path, {'path': str(metadata384_path), **inventory['artifacts'][str(metadata384_path)]})
    assert metadata384['embedding_provenance'] == inventory['recorded_clause_embedding_provenance']
    paragraphs_path = LOGS / 'decoder-distillation-limits-20261002/smoke-r2/results/paragraphs.json'
    paragraphs = read(paragraphs_path, {'path': str(paragraphs_path), **inventory['artifacts'][str(paragraphs_path)]})
    paragraph_by_id = {row['id']: row for split in ('train', 'validation') for row in paragraphs[split]}
    producer768_pin = native_inputs['native768_cached_report']
    producer768 = read(producer768_pin['path'], producer768_pin)
    assert producer768_pin['sha256'] == manifest['inputs'][producer768_pin['path']]
    bounded768_path = LOGS / 'decoder-native-dimensions-20261003/preparation-r2/results/bounded768-production.json'
    bounded768 = read(bounded768_path)
    assert pin(bounded768_path)[0]['sha256'] == manifest['inputs'][str(bounded768_path)]
    assert digest(bounded768) == native_inputs['bounded768_production_sha256']
    assert bounded768['experiment_token_limit'] == 512 and bounded768['historical_profile_token_limit'] == 8192
    small_by_id = {row['input_id']: row for row in small_producer['rows']}
    old_native_by_source = join_lookup(producer768['receipts'], 'source_sha256', 'embedding')
    bounded_native_by_source = join_lookup(bounded768['receipts'], 'source_sha256', 'embedding')
    native_by_source = join_lookup(producer768['receipts'] + bounded768['receipts'], 'source_sha256', 'embedding')
    small_info = small_producer['producer']
    assert small_info['model_id'] == 'thenlper/gte-small'
    assert small_info['revision'] == '17e1f347d17fe144873b1201da91788898c639cd'
    assert small_info['encoder_context_tokens'] == 512
    lanes = []
    for lane in custody['lanes']:
        dimension = lane['dimension']
        rows_pin = lane['exact_cached_file_pins']['original_training_rows']
        contexts_pin = lane['exact_cached_file_pins']['original_contexts']
        rows = read(rows_pin['path'], rows_pin)
        contexts = read(contexts_pin['path'], contexts_pin)
        original = native_inputs['dimensions'][str(dimension)]
        assert original['source_contexts'] == contexts
        paragraphs_matched = clauses_matched = 0
        origins = {'paragraph': {'original_complete_receipt': 0, 'bounded512_receipt': 0},
                   'clause': {'original_complete_receipt': 0, 'bounded512_receipt': 0}}
        for split in ('train', 'validation'):
            assert original[split] == [{key: row[key] for key in ('id', 'input', 'source_text')} for row in rows[split]]
            clause_lookup = join_lookup(clauses384[split], 'source_text', 'input') if dimension == 384 else native_by_source
            for row in rows[split]:
                if dimension == 384:
                    receipt = small_by_id[row['id']]
                    assert receipt['status'] == 'embedded' and receipt['vector'] == row['input']
                    source = paragraph_by_id[row['id']]
                    assert source['source_text'] == row['source_text']
                    assert source['source_sha256'] == hashlib.sha256(row['source_text'].encode()).hexdigest()
                else:
                    source_sha = hashlib.sha256(row['source_text'].encode()).hexdigest()
                    assert digest(row['input']) in native_by_source[source_sha]
                    origin = 'original_complete_receipt' if digest(row['input']) in old_native_by_source.get(source_sha, set()) else 'bounded512_receipt'
                    origins['paragraph'][origin] += 1
                paragraphs_matched += 1
                for segment in contexts[split][row['id']]['segments']:
                    key = segment['source_text'] if dimension == 384 else segment['source_sha256']
                    assert digest(segment['vector']) in clause_lookup[key]
                    if dimension == 768:
                        origin = 'original_complete_receipt' if digest(segment['vector']) in old_native_by_source.get(key, set()) else 'bounded512_receipt'
                        origins['clause'][origin] += 1
                    clauses_matched += 1
        if dimension == 384:
            paragraph_producer = {'receipt_pin': pin(small_producer_path)[0],
                'paragraph_source_pin': pin(paragraphs_path)[0],
                'producer_metadata': small_info, 'scope': '96exactparagraphvectors joined byoriginalID/sourceSHA/vector; original CPUfloat32 GTE-small512 producer receipt, no currentencoderexecution'}
            clause_producer = {'metadata_receipt_pin': pin(metadata384_path)[0],
                'inventory_receipt_pin': pin(inventory_path)[0],
                'producer_metadata': inventory['recorded_clause_embedding_provenance'],
                'scope': '360clauseoccurrences joined to original TRAIN180/VALIDATION60 cached clause rows byliteraltext andexactvectorSHA; original CUDAfloat32 GTE-small recordedprovenance, no currentencoderexecution'}
        else:
            paragraph_producer = clause_producer = {'receipt_pin': producer768_pin,
                'profile_id': producer768['profile_id'], 'execution_profile': producer768['execution_profile'],
                'asset_model_revision': producer768['assets']['model_revision'],
                'asset_code_revision': producer768['assets']['code_revision'],
                'asset_manifest_sha256': producer768['assets']['manifest_sha256'],
                'scope': '96paragraphvectors and360clauseoccurrences joined to recordedcomplete-native768 receipt byexactsourceSHA andvectorSHA; historical8192profile with experiment512admission, no currentencoderexecution',
                'maximum_context_numerics_verified': producer768['maximum_context_numerics_verified']}
            paragraph_producer = {**paragraph_producer, 'bounded512_producer_receipt_pin': pin(bounded768_path)[0],
                'bounded512_execution_profile': bounded768['execution_profile'],
                'bounded512_profile_id': bounded768['native_profile_id'],
                'bounded512_actual_forward_width_checked': bounded768['all_actual_forward_tokens_checked'],
                'cache_origin_counts': origins['paragraph'],
                'scope': '96paragraph vectors joined to exact originalcomplete or bounded512 historicalreceipts bysourceSHA+vectorSHA; separate producer scopes retained without reenactment.'}
            clause_producer = {**clause_producer, 'bounded512_producer_receipt_pin': pin(bounded768_path)[0],
                'bounded512_execution_profile': bounded768['execution_profile'],
                'cache_origin_counts': origins['clause'],
                'scope': '360clause occurrences joined to exact originalcomplete or bounded512 historicalreceipts bysourceSHA+vectorSHA; separate producer scopes retained without reenactment.'}
        lanes.append({'dimension': dimension, 'original_checkpoint_pin': lane['original_checkpoint_pin'],
            'original_native_input_pin': pin(native_inputs_path)[0],
            'native_input_paragraph_vectors_and_clause_contexts_exact_original_cache': True,
            'paragraph_vectors_matched': paragraphs_matched, 'clause_occurrences_matched': clauses_matched,
            'paragraph_producer_evidence': paragraph_producer, 'clause_producer_evidence': clause_producer,
            'canonical_input_representation_profile_id': original['representation']['profile_id'],
            'profile_id_none_is_preserved': dimension == 384,
            'decoder_profile_id_still_none': True,
            'experiment_encoder_budget_tokens': 512, 'decoder_output_budget_tokens': 512,
            'runtime_admitted': False, 'teacher_qualified': False, 'proof_authority': False})
    before = [helper.PINS[path] for path in sorted(helper.PINS)]
    after = [pin(path)[0] for path in sorted(helper.PINS)]
    assert before == after
    result = {'schema': 'retained-contextual-source-producer-joins/v1', 'completed': True,
        'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_pin': pin(Path(__file__).resolve())[0], 'custody_survey_pin': pin(custody_path)[0],
        'lanes': lanes, 'file_pins_before': before, 'file_pins_after': after,
        'bound_files_unchanged': True,
        'qualification_scope': 'Fresh exact source/cache/vector joins and immutable local original producer receipts; no current model/provenance reenactment, sourcefidelity,8192capability,teacher or runtimequalification.',
        'operations': {'model_loaded': False, 'encoder_invoked': False, 'new_embeddings_generated': False,
            'training_executed': False, 'database_access': False, 'network_calls': False,
            'asset_or_catalog_mutations': False}}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'lanes': len(lanes),
        'paragraph_vectors_matched_per_lane': 96, 'clause_occurrences_matched_per_lane': 360,
        'bound_files': len(after), 'files_unchanged': True}))


if __name__ == '__main__':
    main()
