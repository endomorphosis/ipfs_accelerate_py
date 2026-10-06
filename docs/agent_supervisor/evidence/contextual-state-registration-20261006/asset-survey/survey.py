#!/usr/bin/env python3
"""Survey retained contextual states, cached input contracts and local Hub receipts.

Only a fresh --output JSON file is written. No project owner, model, ML package,
database, network client or provider is imported or invoked. Historical numerical
and remote observations are retained as historical evidence, not rerun here.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys

BASE = Path('/home/barberb/lift_coding')
DATASETS = BASE / '.worktrees/decoder-profile-recovery-datasets-20261006'
DATASETS_HEAD = '3b3b994407b2fcfef955ce5153afc2eea01e4eff'
REPLAY = BASE / 'artifacts/ir-semantic-reconstruction-replay-20261004/contextual384-768-v1'
PREVIOUS = BASE / 'artifacts/supervisor-decoder-contract-20261006/retained-decoder-contract-survey.json'
PUBLICATION = BASE / 'artifacts/autoformalization-publication-20261004'
MAX_JSON_BYTES = 16 * 1024 * 1024
PINS = {}
SHAS = {384: '0ac5c21656187d1d8040a02bcc0b4716a8093db17f05adc9a22e99d5b6b2cc00',
        768: '8892a3261c0750a6247ba5400069ed18cad3354426e2095ed1d295287e3559b4'}
TARGET_FIELDS = {'modality', 'actor', 'action', 'object', 'conditions', 'exceptions', 'temporal'}


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def file_witness(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('regular local file required: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def pin(path, expected=None, expected_sha=None):
    path = Path(path)
    before = file_witness(path)
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(b'blob ' + str(before[3]).encode() + b'\0')
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha256.update(chunk)
            git_sha1.update(chunk)
    if before != file_witness(path):
        raise ValueError('local file changed while hashed')
    result = {'path': str(path), 'bytes': before[3], 'sha256': sha256.hexdigest()}
    if expected is not None and result != expected:
        raise ValueError('historical pin mismatch: ' + str(path))
    if expected_sha is not None and result['sha256'] != expected_sha:
        raise ValueError('historical SHA mismatch: ' + str(path))
    original = PINS.setdefault(str(path), result)
    if original != result:
        raise ValueError('local input generation changed')
    return result, git_sha1.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key] = value
    return result


def read(path, expected=None):
    receipt, _ = pin(path, expected)
    if not 0 < receipt['bytes'] <= MAX_JSON_BYTES:
        raise ValueError('bounded JSON file required')
    with Path(path).open(encoding='utf-8') as handle:
        value = json.load(handle, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('finite JSON required')))
    if pin(path)[0] != receipt:
        raise ValueError('JSON changed while read')
    return value


def metadata(value):
    return {key: item for key, item in value.items()
            if item is None or type(item) in (str, int, float, bool)}


def validate_vector(vector, dimension):
    assert type(vector) is list and len(vector) == dimension
    assert all(type(value) in (int, float) and math.isfinite(value) for value in vector)
    assert abs(sum(float(value) ** 2 for value in vector) - 1.0) <= 1e-4


def validate_contexts(rows, contexts, dimension):
    assert len(rows) == len(contexts) == 48
    masks, clause_sources = [], set()
    for row in rows:
        descriptor = contexts[row['id']]
        source = row['source_text']
        pieces = source.split('\n\n')
        assert 1 <= len(pieces) <= 8 and all(piece.strip() for piece in pieces)
        assert descriptor['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
        assert set(descriptor) == {'source_sha256', 'segments'}
        assert len(pieces) == len(descriptor['segments'])
        char = byte = 0
        for piece, segment in zip(pieces, descriptor['segments']):
            assert set(segment) == {'source_text', 'source_sha256', 'embedding_sha256', 'vector',
                                    'char_start', 'char_end', 'byte_start', 'byte_end'}
            assert segment['source_text'] == piece
            assert segment['source_sha256'] == hashlib.sha256(piece.encode()).hexdigest()
            assert segment['embedding_sha256'] == digest(segment['vector'])
            assert segment['char_start'] == char and segment['char_end'] == char + len(piece)
            assert segment['byte_start'] == byte and segment['byte_end'] == byte + len(piece.encode())
            validate_vector(segment['vector'], dimension)
            clause_sources.add(segment['source_sha256'])
            char += len(piece) + 2
            byte += len(piece.encode()) + 2
        validate_vector(row['input'], dimension)
        masks.append({'id': row['id'], 'mask': [True] * len(pieces) + [False] * (8 - len(pieces))})
    return {'row_count': len(rows), 'clause_occurrences': sum(sum(row['mask']) for row in masks),
            'unique_clause_source_count': len(clause_sources), 'source_SHA_set_sha256': digest(sorted(clause_sources)),
            'source_padding_masks_derived_from_pinned_original_rule': masks,
            'masks_sha256': digest(masks),
            'ordered_IDs': [row['id'] for row in rows],
            'source_text_order_sha256': digest([row['source_text'] for row in rows]),
            'vector_payload_order_sha256': digest([row['input'] for row in rows]),
            'context_payload_sha256': digest(contexts),
            'clause_count_distribution': dict(collections.Counter(str(sum(row['mask'])) for row in masks))}


def target_contract(rows, vocabulary):
    counts = collections.Counter()
    target_lengths = []
    target_docs = []
    for row in rows:
        tokens = row['target_ids']
        assert tokens[0] == 1 and tokens[-1] == 2 and len(tokens) <= 514
        document = json.loads(''.join(vocabulary[index] for index in tokens[1:-1]))
        assert set(document) == {'rules'} and type(document['rules']) is list
        assert 1 <= len(document['rules']) <= 8
        for rule in document['rules']:
            assert set(rule) == TARGET_FIELDS
            assert all(type(rule[key]) is str for key in ('actor', 'action', 'modality', 'object'))
            assert rule['modality'] in ('O', 'P', 'F')
            assert all(rule[key] == [] for key in ('conditions', 'exceptions', 'temporal'))
        counts[str(len(document['rules']))] += 1
        target_lengths.append(len(tokens) - 2)
        target_docs.append(document)
    return {'output_shape': 'JSON object with ordered rules array; each rule has exactly seven canonical fields',
            'observed_fields': sorted(TARGET_FIELDS), 'observed_qualifier_fields_empty': True,
            'rows': len(rows), 'rule_count_distribution': dict(counts),
            'total_rules': sum(int(key) * value for key, value in counts.items()),
            'target_token_lengths_excluding_BOS_EOS': {'minimum': min(target_lengths), 'maximum': max(target_lengths)},
            'ordered_target_ids_sha256': digest([row['target_ids'] for row in rows]),
            'canonical_target_documents_sha256': digest(target_docs),
            'native_IR_schema_version': None,
            'schema_version_unknown_reason': 'Observed seven-field target shape and pinned validator are not an explicit native schema version declaration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    source_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=DATASETS, text=True).strip()
    if source_head != DATASETS_HEAD:
        raise ValueError('selected recovered datasets head differs')
    before_git = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=DATASETS, text=True)
    preparation = read(REPLAY / 'preparation.json')
    historical_score = read(REPLAY / 'scored-result.json')
    previous = read(PREVIOUS)
    training_manifest = read(preparation['training_manifest']['path'], preparation['training_manifest'])
    training_plan = read(preparation['training_plan']['path'], preparation['training_plan'])
    read(preparation['evaluation_manifest']['path'], preparation['evaluation_manifest'])
    pin(preparation['driver']['path'], preparation['driver'])
    pin(REPLAY / 'replay.py')
    pin(REPLAY.parent / 'score_contextual.py')
    selected_sources = {}
    for path, expected_sha in training_manifest['inputs'].items():
        if any(name in path for name in ('/clause_source_context.py', '/decoder_distillation_experiment.py',
            '/ordered_clause_recurrent_decoder_experiment.py', '/benchmark_clause_context_source_training.py',
            '/benchmark_ordered_clause_recurrent_training.py')):
            selected_sources[path] = pin(path, expected_sha=expected_sha)[0]
    assert len(selected_sources) == 5
    clause_owner = next(Path(path) for path in selected_sources if path.endswith('/clause_source_context.py'))
    clause_text = clause_owner.read_text()
    assert 'MAX_CLAUSES = 8' in clause_text and 'mask[index, :len(values)] = True' in clause_text
    source_files = {}
    for name in ('ir_model_manager_import.py', 'ir_model_hub_publish.py', 'ir_decoder_profile_inventory.py',
                 'ir_decoder_format_runtime.py', 'checkpoint_hub.py'):
        path = DATASETS / 'ipfs_datasets_py/logic/formalization/autoencoder' / name
        source_files[name] = pin(path)[0]
    for name in ('ir_progress_recovery.md', 'progress_integration.md'):
        pin(DATASETS / 'docs/autoencoders' / name)
    publication_path = PUBLICATION / 'huggingface/additional-decoder-upload-01/publication.json'
    publication = read(publication_path)
    plan = read(PUBLICATION / 'huggingface/additional_decoder_publication_plan.json')
    read(PUBLICATION / 'huggingface/additional-decoder-upload-01/commit-created.json', publication['commit_creation_binding'])
    assert pin(PUBLICATION / 'huggingface/additional_decoder_publication_plan.json')[0] == publication['plan_binding']
    coverage = read(PUBLICATION / 'all_weights/prior-coverage-01.json')
    remote_history = read(PUBLICATION / 'all_weights/prior-remote-verification-01.json')
    assert publication['status'] == 'published_and_verified' and publication['all_remote_files_verified'] is True
    assert publication['repo_id'] == 'Publicus/legal-ir-autoencoder'
    assert publication['commit_oid'] == '49839a5e55e2ef8ab3815f4e2650dc69ee72d5b8'
    assert remote_history['coverage_binding'] == pin(PUBLICATION / 'all_weights/prior-coverage-01.json')[0]
    lanes = []
    for item in preparation['lanes']:
        dimension = item['dimension']
        assert dimension in SHAS and item['checkpoint']['sha256'] == SHAS[dimension]
        checkpoint = read(item['checkpoint']['path'], item['checkpoint'])
        summary = read(item['summary']['path'], item['summary'])
        rows = read(item['original_training_rows']['path'], item['original_training_rows'])
        contexts = read(item['original_contexts']['path'], item['original_contexts'])
        preprocessing = read(item['original_preprocessing']['path'], item['original_preprocessing'])
        inputs = read(item['source_only_inputs']['path'], item['source_only_inputs'])
        source_contexts = read(item['source_only_contexts']['path'], item['source_only_contexts'])
        read(item['historical_evaluation']['path'], item['historical_evaluation'])
        assert inputs == [{key: row[key] for key in ('id', 'source_text', 'input')} for row in rows['validation']]
        assert source_contexts == contexts['validation']
        assert checkpoint['input_transform'] == preprocessing['input_transform']
        assert checkpoint['architecture']['normalization'] == preprocessing['paragraph_normalization']
        assert checkpoint['architecture']['clause_normalization'] == preprocessing['clause_normalization']
        assert checkpoint['lineage']['input_provenance_sha256'] == preprocessing['source_inputs_sha256']
        assert summary['states']['selected']['sha256'] == SHAS[dimension]
        split_observations = {split: validate_contexts(rows[split], contexts[split], dimension) for split in ('train', 'validation')}
        assert set(split_observations['train']['ordered_IDs']).isdisjoint(split_observations['validation']['ordered_IDs'])
        vocabulary = checkpoint['codec']['target_vocabulary']
        assert len(vocabulary) == 32 and digest(checkpoint['codec']) == checkpoint['architecture']['codec_sha256']
        target_observations = {split: target_contract(rows[split], vocabulary) for split in ('train', 'validation')}
        pub_matches = [row for row in publication['remote_files'] if row['sha256'] == SHAS[dimension]]
        plan_matches = [row for row in plan['files'] if row['sha256'] == SHAS[dimension]]
        history_matches = [row for row in remote_history['remote_files'] if row['sha256'] == SHAS[dimension]]
        state_matches = [row for row in coverage['saved_state_coverage'] if row.get('file_sha256') == SHAS[dimension]]
        assert len(pub_matches) == len(plan_matches) == len(history_matches) == len(state_matches) == 1
        public = pub_matches[0]
        copy_path = Path(plan['directory']) / plan_matches[0]['path']
        copy_pin, copy_git = pin(copy_path)
        assert copy_pin['sha256'] == SHAS[dimension] and copy_pin['bytes'] == item['checkpoint']['bytes']
        assert copy_git == public['git_blob_oid'] == history_matches[0]['git_blob_oid']
        assert public['path'] == history_matches[0]['path_in_repository']
        assert state_matches[0]['coverage_kind'] == 'complete_original_JSON_container'
        state_hash = digest(checkpoint['model_state'])
        assert state_hash == checkpoint['weights_sha256'] == state_matches[0]['canonical_saved_model_state_sha256']
        assert len(checkpoint['model_state']) == 32
        current_matches = previous['current_selected_model_manager_catalog']['current_contextual_exact_sha_match_counts'][SHAS[dimension]]
        assert current_matches == 0
        score = next(row for row in historical_score['lanes'] if row['dimension'] == dimension)
        architecture = metadata(checkpoint['architecture'])
        architecture.update(source_fields=checkpoint['architecture']['source_fields'], state_layout=checkpoint['architecture']['state_layout'])
        lanes.append({'dimension': dimension, 'ir_family_id': 'legal_ir', 'dimension_role': 'input_embedding',
            'original_checkpoint_pin': item['checkpoint'], 'serialization_schema': checkpoint['schema'],
            'architecture': architecture, 'codec': checkpoint['codec'],
            'codec_sha256': digest(checkpoint['codec']), 'complete_model_state_entry_count': 32,
            'saved_model_state_canonical_JSON_sha256': state_hash,
            'saved_model_state_hash_matches_full_publication_container': True,
            'input_transform_metadata': metadata(checkpoint['input_transform']),
            'input_transform_content_sha256': digest(checkpoint['input_transform']),
            'normalization': {key: {'metadata': metadata(preprocessing[key]),
                'content_sha256': digest(preprocessing[key]),
                'matches_checkpoint_architecture': True} for key in ('paragraph_normalization', 'clause_normalization')},
            'representation_declaration': preprocessing['representation'],
            'source_binding_content_sha256': digest(preprocessing['source_binding']),
            'exact_cached_file_pins': {key: item[key] for key in ('original_training_rows', 'original_contexts', 'original_preprocessing', 'source_only_inputs', 'source_only_contexts')},
            'split_observations': split_observations, 'target_contract_observations': target_observations,
            'source_only48_copies_exact_original_validation_subset': True,
            'mask_contract': {'standalone_cached_mask_file': False, 'source_owner_pin': selected_sources[str(clause_owner)],
                'batch_shape': ['batch', 8, dimension], 'mask_shape': ['batch', 8], 'mask_dtype': 'bool',
                'true_positions': 'ordered literal source-clause positions; false pads to8',
                'padding_order': 'transform actual clause vectors first; zero padding afterward',
                'torch_float32_transform_not_recomputed_by_this_survey': True},
            'historical_replay_metrics': score['metrics'],
            'historical_replay_source_pin': pin(REPLAY / 'scored-result.json')[0],
            'encoder_experiment_limit_tokens': 512, 'decoder_output_limit_tokens': 512,
            'new_numerical_replay_this_survey': False,
            'historical_complete_state_HF_publication': {'repository_id': publication['repo_id'],
                'immutable_revision': publication['commit_oid'], 'file': public,
                'local_publication_copy_pin': copy_pin,
                'local_publication_copy_git_blob_SHA1': copy_git,
                'publication_receipt_pin': pin(publication_path)[0],
                'later_historical_remote_observation': history_matches[0],
                'later_historical_remote_receipt_pin': pin(PUBLICATION / 'all_weights/prior-remote-verification-01.json')[0],
                'fresh_remote_observation_this_survey': False},
            'selected_catalog_observation': {'source_receipt_pin': pin(PREVIOUS)[0],
                'observed_at_utc': previous['observed_at_utc'], 'binding_count': 668,
                'exact_full_state_SHA_match_count': current_matches, 'fresh_catalog_load_this_survey': False},
            'append_only_registration_contract': {'ir_family_id': 'legal_ir', 'dimension': dimension,
                'dimension_role': 'input_embedding', 'task_id': 'semantic_IR_reconstruction',
                'task_identity_provenance': 'literal decoder_task declaration in historical retained replay comparison; sourceparagraph+sourceclause inputs -> canonical rule array',
                'schema_version': None, 'profile_id': None, 'format_id': None,
                'known_checkpoint_schema_is_not_native_IR_schema_version': True,
                'complete_runtime_io_contract': False,
                'input_contract': ['native paragraph vector', 'exact literal-source clause vectors', 'derived boolsourcepadding mask', 'source text/IDs for cache validation'],
                'output_contract': 'ordered canonical rules JSON with seven fields; observed empty qualifiers and bounded inherited vocabulary',
                'runtime_ready': False, 'teacher_qualified': False, 'proof_authority': False,
                'record_prefix': 'ir-model-asset-binding/v1:',
                'native_registered_format_prefix_forbidden_for_this_legacy_contract': True}})
    after = [pin(path)[0] for path in sorted(PINS)]
    assert after == [PINS[path] for path in sorted(PINS)]
    after_git = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=DATASETS, text=True)
    assert after_git == before_git
    result = {'schema': 'retained-contextual-decoder-cache-publication-survey/v1', 'completed': True,
        'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_pin': pin(Path(__file__).resolve())[0],
        'datasets_source': {'root': str(DATASETS), 'head': source_head, 'tracked_state_preserved': True,
            'status_before': before_git.splitlines(), 'status_after': after_git.splitlines(),
            'relevant_new_main_scope': '3b3b994 restores original384 cached/Intent/Securityfragment interfaces; it does not admit contextualprivate wrappers or infer unknownnativeoutputschema.',
            'owner_source_pins': source_files},
        'lanes': lanes, 'source_owner_selected_historical_pins': list(selected_sources.values()),
        'publication_scope': 'Fresh local exact file/state/cache hashes and historical receipt joins. Historical exactstate publication now established; remote currentness remains a separate fresh network observation.',
        'importer_api': 'import_ir_model_manager_records(manifest_pin, *, release_receipts, manager, readback, ...)',
        'importer_required_publication_schema': 'ir-model-hub-publication-receipt/v1',
        'historical_publication_schema': publication['schema'],
        'receipt_adapter_needed': 'Historical append-only-HF-publication/v1 differs from importerreceipt schema. Fresh checked remote file hashes/GitOIDs may support an explicit transparent adapter, not relabeling unsupported history.',
        'publisher_api': 'publish_ir_model_hub_release(plan, *, api=None, operation_factory=None)',
        'new_asset_upload_not_needed_to_establish_historical_complete_state_bytes': True,
        'new_dimension_repo_or_profile_metadata_publication_requires_separate_explicit_receipt': True,
        'quality_scope': 'Exposed authored48paragraph regression and cachedliteralclause inputs; no sourceparagraphsinglevector, independentholdout, nonemptyqualifier, arbitrarynewsource, originalprose,8192span, teacher or proof qualification.',
        'bound_file_count': len(after), 'file_pins_before': [PINS[path] for path in sorted(PINS) if path != str(Path(__file__).resolve())],
        'file_pins_after': after, 'bound_files_unchanged': True,
        'operations': {'model_loaded': False, 'model_invoked': False, 'training_executed': False,
            'new_embeddings_generated': False, 'source_modules_imported': False, 'database_access': False,
            'registrations': False, 'network_calls': False, 'Hub_mutations': False, 'asset_mutations': False},
        'authority': {'runtime_admitted': False, 'teacher_qualified': False, 'proof_authority': False,
            'source_fidelity_qualified': False, 'fresh_quality_measured': False}}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'lanes': len(lanes), 'bound_files': len(after),
        'complete_state_publication_receipt_matches': 2, 'cached_validation_rows_per_lane': 48,
        'cached_clause_occurrences_per_validation_lane': 180, 'files_unchanged': True,
        'new_models_or_network_calls': 0}))


if __name__ == '__main__':
    main()
