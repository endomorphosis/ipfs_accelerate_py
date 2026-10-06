"""Prepare two original contextual decoder states for explicit metadata registration.

No model, encoder, optimizer or ModelManager is constructed. HF payloads must
already be freshly downloaded at the immutable revision. Unknown native IR
schema/profile/format identities remain null. This is an asset registration,
with explicit contextual IO declarations, not a new runtime admission.
"""
import importlib.util
import json
from pathlib import Path
import sys

from custody import capture, read, require, source, write

BASE = Path('/home/barberb/lift_coding')
OUT = Path(__file__).resolve().parent
DATASETS = BASE / '.worktrees/decoder-profile-recovery-datasets-20261006'
RELATIVE = 'ipfs_datasets_py/logic/formalization/autoencoder/ir_model_manager_import.py'
REVISION = '49839a5e55e2ef8ab3815f4e2650dc69ee72d5b8'
REPOSITORY = 'Publicus/legal-ir-autoencoder'
EXPECTED = {384: (2799098, '0ac5c21656187d1d8040a02bcc0b4716a8093db17f05adc9a22e99d5b6b2cc00'),
            768: (4781613, '8892a3261c0750a6247ba5400069ed18cad3354426e2095ed1d295287e3559b4')}


def main():
    owner = source(DATASETS, RELATIVE)
    spec = importlib.util.spec_from_file_location('retained_native_importer', DATASETS / RELATIVE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    historical_path = BASE / 'artifacts/autoformalization-publication-20261004/huggingface/additional-decoder-upload-01/publication.json'
    historical_pin = capture(historical_path)
    historical = json.loads(read(historical_pin))
    require(historical['repo_id'] == REPOSITORY and historical['commit_oid'] == REVISION
            and historical['status'] == 'published_and_verified', 'exact historical publication required')
    survey_path = OUT / 'asset-survey/contextual-state-custody-survey.json'
    survey_pin = capture(survey_path)
    survey = json.loads(read(survey_pin))
    require(survey['completed'] is True and len(survey['lanes']) == 2
            and {row['dimension'] for row in survey['lanes']} == {384, 768},
            'exactly two unique contextual cache custody lanes required')
    records, files, controls = [], [], []
    for dimension, (count, sha) in EXPECTED.items():
        run = f'{dimension}-selected-followup-lr0001-1729'
        original = BASE / f'external/ipfs_datasets/workspace/test-logs/decoder-selected-continuation-20261004/training-r1/results/{run}/selected-state.json'
        original_pin = capture(original)
        require((original_pin['bytes'], original_pin['sha256']) == (count, sha), 'original selected state changed')
        remote_path = f'releases/20261004-additional-decoder-cutoff-v1/checkpoints/decoder-selected-continuation-20261004/{run}/selected-state.json'
        downloaded = OUT / 'remote-download' / remote_path
        download_pin = capture(downloaded)
        require((download_pin['bytes'], download_pin['sha256']) == (count, sha), 'fresh remote download differs from original')
        matches = [row for row in historical['remote_files'] if row['path'] == remote_path
                   and row['bytes'] == count and row['sha256'] == sha]
        require(len(matches) == 1, 'unique exact historical remote file required')
        blob = __import__('hashlib').sha1(f'blob {count}\0'.encode() + read(download_pin)).hexdigest()
        require(blob == matches[0]['git_blob_oid'], 'fresh remote Git blob differs')
        state = json.loads(read(original_pin))
        require(state['schema'] == 'private-native-dimension-source-state/v1'
                and state['dimension'] == dimension and state['selected'] is True
                and state['lineage']['domain'] == 'legal_ir', 'original contextual state identity differs')
        require(state['proof_authority'] is False and state['admitted'] is False,
                'historical state must remain unadmitted')
        lane = next(row for row in survey['lanes'] if row['dimension'] == dimension)
        require(lane['original_checkpoint_pin'] == original_pin,
                'contextual cache survey belongs to different selected checkpoint')
        role = 'selected_contextual_semantic_decoder_state'
        model_id = native.ir_model_asset_record_id('legal_ir', dimension, 'input_embedding', role, sha)
        identity = {'record_id': model_id, 'ir_family_id': 'legal_ir', 'dimension': dimension,
            'dimension_role': 'input_embedding', 'role': role, 'schema_version': None,
            'task_id': lane['append_only_registration_contract']['task_id'], 'profile_id': None,
            'format_id': None, 'original_checkpoint_pin': original_pin, 'trained': True,
            'initialization_only': False, 'donor': None, 'runtime_ready': False,
            'teacher_qualified': False, 'proof_authority': False}
        config = {'ir_checkpoint': identity, 'complete_runtime_io_contract': False,
            'checkpoint_serialization_schema': state['schema'],
            'native_output_schema_status': 'unknown_not_inferred_from_checkpoint_serialization',
            'contextual_input_custody': lane,
            'contextual_custody_survey_pin': survey_pin,
            'decoder_codec_schema': state['codec']['schema'],
            'ordered_codec_sha256': state['lineage']['teacher_codec_sha256'],
            'declared_decoder_output_token_limit': state['lineage']['teacher_output_limit'],
            'source_text_reconstruction_qualified': False, 'fresh_holdout_qualified': False,
            'long_context_8192_qualified': False,
            'registration_task_id_scope': lane['append_only_registration_contract']['task_identity_provenance'],
            'release': {'repository_id': REPOSITORY, 'revision': REVISION, 'path_in_repo': remote_path,
                        'checkpoint_sha256': sha}}
        metadata = {'model_id': model_id, 'model_name': f'LegalIR {dimension}D retained contextual selected decoder',
            'model_type': 'decoder_only', 'architecture': state['architecture']['schema'],
            'inputs': [{'name': 'native_paragraph_embedding', 'data_type': 'embeddings', 'shape': [-1, dimension],
                        'description': 'Original cached paragraph vectors from the explicitly bound producer.'},
                       {'name': 'native_clause_embeddings', 'data_type': 'embeddings', 'shape': [-1, 8, dimension],
                        'description': 'Original ordered literal-source-clause vectors, with the original producer padding to eight slots after normalization; mandatory extra conditioning.'},
                       {'name': 'clause_mask', 'data_type': 'features', 'shape': [-1, 8], 'dtype': 'bool',
                        'description': 'Boolean mask constructed from the original cached clause counts by the bound producer; true on clauses, false on zero-padded slots.'}],
            'outputs': [{'name': 'canonical_ir_tokens', 'data_type': 'tokens', 'shape': [-1, -1], 'dtype': 'int64',
                         'description': 'Restricted ordered typed JSON lexical rule tokens; native output schema qualification pending.'}],
            'huggingface_config': config, 'model_revision': sha, 'revision_id': sha,
            'source_url': f'https://huggingface.co/{REPOSITORY}/blob/{REVISION}/{remote_path}',
            'tags': ['legal_ir', f'{dimension}d', 'contextual', 'retained-selected-state', 'runtime-unqualified'],
            'description': 'Original selected contextual semantic decoder state, registered as a separate asset. Requires paragraph and ordered clause vectors plus masks, original normalization and codec. Historical exposed development exact canonical IR does not establish originating prose reconstruction, fresh holdout or 8192-token decoding.'}
        records.append({'model_metadata': metadata, 'checkpoint_pin': original_pin, 'release': config['release']})
        files.append({'path_in_repo': remote_path, 'file_pin': original_pin, 'verified': True,
            'bytes': count, 'sha256': sha, 'remote_identity': {'scheme': 'git-blob-sha1', 'bytes': count, 'blob_id': blob}})
        controls.append({'dimension': dimension, 'original_pin': original_pin,
            'fresh_download_pin': download_pin, 'remote_path': remote_path, 'git_blob_sha1': blob,
            'historical_receipt_pin': historical_pin, 'exact_remote_bytes_equal_original': True})
    release = {'schema': native.PUBLICATION_SCHEMA, 'repository_id': REPOSITORY, 'revision': REVISION,
        'files_verified': True, 'files': files, 'observation_scope': 'Two immutable-revision HF CLI downloads freshly hashed and compared with original retained bytes; no new upload.',
        'historical_append_publication_pin': historical_pin, 'fresh_remote_verifications': controls,
        'runtime_ready': False, 'proof_authority': False}
    release_pin = write(OUT / 'verified-contextual-publication-receipt.json', release)
    plan_pin = write(OUT / 'contextual-model-manager-import-plan.json', {'schema': native.SCHEMA, 'models': records})
    native._prepare(plan_pin, [release_pin], native.MAX_REFERENCE_BYTES)
    for pin in [historical_pin, survey_pin, *(row['fresh_download_pin'] for row in controls),
                *(row['original_pin'] for row in controls), *survey['file_pins_before']]:
        require(capture(pin['path']) == pin, 'preparation closing file fence differs')
    require(source(DATASETS, RELATIVE) == owner, 'importer source changed')
    result = {'schema': 'contextual-state-registration-preparation/v1', 'completed': True,
        'importer_owner': owner, 'import_plan_pin': plan_pin, 'publication_receipt_pin': release_pin,
        'contextual_custody_survey_pin': survey_pin, 'model_count': 2,
        'model_ids': [row['model_metadata']['model_id'] for row in records],
        'native_preparation_passed': True, 'original_assets_preserved': True,
        'native_schema_profile_format_remain_null': True, 'runtime_ready': False, 'proof_authority': False,
        'model_manager_constructed': False, 'database_mutated': False, 'model_loaded': False,
        'inference_executed': False, 'training_executed': False, 'huggingface_uploaded': False}
    result_pin = write(OUT / 'registration-preparation.json', result)
    print(json.dumps({'preparation_pin': result_pin, 'models': result['model_ids']}))


if __name__ == '__main__':
    main()
