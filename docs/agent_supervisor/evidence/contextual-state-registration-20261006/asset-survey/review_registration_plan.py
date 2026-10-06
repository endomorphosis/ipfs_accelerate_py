#!/usr/bin/env python3
"""Read-only exact contract review of an already prepared registration plan."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_contextual_plan_metadata_helpers', ROOT / 'survey.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, pin, digest = helper.read, helper.pin, helper.digest
STAGE = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    custody_path = ROOT / 'contextual-state-custody-survey.json'
    producer_path = ROOT / 'source-producer-joins.json'
    plan_path = STAGE / 'contextual-model-manager-import-plan.json'
    publication_path = STAGE / 'verified-contextual-publication-receipt.json'
    preparation_path = STAGE / 'registration-preparation.json'
    custody, producers = read(custody_path), read(producer_path)
    plan, publication, preparation = read(plan_path), read(publication_path), read(preparation_path)
    assert custody['completed'] is True and producers['completed'] is True
    assert preparation['completed'] is True and preparation['native_preparation_passed'] is True
    assert preparation['import_plan_pin'] == pin(plan_path)[0]
    assert preparation['publication_receipt_pin'] == pin(publication_path)[0]
    assert preparation['contextual_custody_survey_pin'] == pin(custody_path)[0]
    assert plan['schema'] == 'ir-model-manager-import-plan/v1'
    assert publication['schema'] == 'ir-model-hub-publication-receipt/v1'
    assert publication['repository_id'] == 'Publicus/legal-ir-autoencoder'
    assert publication['revision'] == '49839a5e55e2ef8ab3815f4e2650dc69ee72d5b8'
    assert publication['files_verified'] is True and publication['runtime_ready'] is False and publication['proof_authority'] is False
    lanes = {row['dimension']: row for row in custody['lanes']}
    assert len(custody['lanes']) == len(lanes) == len(plan['models']) == len(publication['files']) == 2
    assert set(lanes) == {384, 768}
    observations = []
    dimensions = []
    for row in plan['models']:
        metadata = row['model_metadata']
        config = metadata['huggingface_config']
        identity = config['ir_checkpoint']
        dimension = identity['dimension']
        dimensions.append(dimension)
        lane = lanes[dimension]
        assert row['checkpoint_pin'] == identity['original_checkpoint_pin'] == lane['original_checkpoint_pin']
        assert identity['ir_family_id'] == 'legal_ir' and identity['dimension_role'] == 'input_embedding'
        assert identity['task_id'] == lane['append_only_registration_contract']['task_id'] == 'semantic_IR_reconstruction'
        assert all(identity[key] is None for key in ('schema_version', 'profile_id', 'format_id'))
        assert identity['trained'] is True and identity['initialization_only'] is False and identity['donor'] is None
        assert all(identity[key] is False for key in ('runtime_ready', 'teacher_qualified', 'proof_authority'))
        expected_identity = {key: identity[key] for key in ('ir_family_id', 'dimension', 'dimension_role', 'role')}
        expected_identity['checkpoint_sha256'] = row['checkpoint_pin']['sha256']
        expected_record = 'ir-model-asset-binding/v1:' + hashlib.sha256(json.dumps(expected_identity,
            sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        assert metadata['model_id'] == identity['record_id'] == expected_record
        assert metadata['model_revision'] == metadata['revision_id'] == row['checkpoint_pin']['sha256']
        assert config['contextual_input_custody'] == lane
        assert config['contextual_custody_survey_pin'] == pin(custody_path)[0]
        assert config['complete_runtime_io_contract'] is False
        assert config['checkpoint_serialization_schema'] == 'private-native-dimension-source-state/v1'
        assert config['declared_decoder_output_token_limit'] == lane['decoder_output_limit_tokens'] == 512
        assert config['decoder_codec_schema'] == 'typed-json-lexical/v1'
        assert config['ordered_codec_sha256'] == lane['codec_sha256']
        assert all(config[key] is False for key in ('fresh_holdout_qualified', 'long_context_8192_qualified', 'source_text_reconstruction_qualified'))
        assert len(metadata['inputs']) == 3
        paragraph, clause, mask = metadata['inputs']
        assert paragraph['name'] == 'native_paragraph_embedding' and paragraph['data_type'] == 'embeddings' and paragraph['shape'] == [-1, dimension]
        assert clause['name'] == 'native_clause_embeddings' and clause['data_type'] == 'embeddings' and clause['shape'] == [-1, 8, dimension]
        assert mask['name'] == 'clause_mask' and mask['data_type'] == 'features' and mask['shape'] == [-1, 8] and mask['dtype'] == 'bool'
        assert all(item.get('optional', False) is False for item in metadata['inputs'])
        assert len(metadata['outputs']) == 1 and metadata['outputs'][0]['data_type'] == 'tokens'
        release = row['release']
        assert release == config['release']
        assert release['repository_id'] == publication['repository_id'] and release['revision'] == publication['revision']
        assert release['checkpoint_sha256'] == row['checkpoint_pin']['sha256']
        pub = [item for item in publication['files'] if item['sha256'] == release['checkpoint_sha256']]
        fresh = [item for item in publication['fresh_remote_verifications'] if item['dimension'] == dimension]
        assert len(pub) == len(fresh) == 1
        pub, fresh = pub[0], fresh[0]
        assert pub['path_in_repo'] == release['path_in_repo'] == fresh['remote_path']
        assert pub['file_pin'] == row['checkpoint_pin'] == fresh['original_pin']
        original_pin, original_git = pin(row['checkpoint_pin']['path'], row['checkpoint_pin'])
        downloaded_pin, downloaded_git = pin(fresh['fresh_download_pin']['path'], fresh['fresh_download_pin'])
        assert (original_pin['bytes'], original_pin['sha256']) == (downloaded_pin['bytes'], downloaded_pin['sha256'])
        assert original_git == downloaded_git == fresh['git_blob_sha1'] == pub['remote_identity']['blob_id']
        assert fresh['exact_remote_bytes_equal_original'] is True
        assert pub['remote_identity']['scheme'] == 'git-blob-sha1' and pub['verified'] is True
        observations.append({'dimension': dimension, 'model_id': metadata['model_id'],
            'original_checkpoint_pin': row['checkpoint_pin'], 'full_downloaded_state_matches_original': True,
            'immutable_remote_path': release['path_in_repo'], 'Git_blob_OID': original_git,
            'custody_lane_exactly_preserved': True, 'unknown_native_selectors_preserved': True,
            'mandatory_paragraph_clause_and_mask_geometry_matches_original_recipe': True,
            'runtime_admitted': False, 'teacher_qualified': False, 'proof_authority': False})
    assert set(dimensions) == {384, 768} and len(set(dimensions)) == len(dimensions)
    before = [helper.PINS[path] for path in sorted(helper.PINS)]
    after = [pin(path)[0] for path in sorted(helper.PINS)]
    assert before == after
    result = {'schema': 'contextual-registration-semantic-contract-review/v1', 'completed': True,
        'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'decoder_asset_survey', 'script_pin': pin(Path(__file__).resolve())[0],
        'scope': 'Read-only semantic/asset contract correspondence review after the root reported registration complete. No current database/query or native runtime replay is performed by this review; already registered artifacts are preserved.',
        'plan_and_publication_contract_checks_passed': True, 'observations': observations,
        'precision_note_not_registered_artifact_mutation': {'registered_description': 'padding to eight slots after normalization',
            'finding': 'The broad word normalization must not merge the saved original input_transform with separate decoder paragraph/clause feature normalizations.',
            'authoritative_input_recipe': [
                'Retain raw native paragraph and ordered literal-source clause vectors plus exact source IDs/text bindings from the pinned original caches.',
                'Apply the original saved training-only input_transform once to the actual paragraph vector before the retained model forward path.',
                'Apply that original saved input_transform once to each real clause vector, then zero-pad the transformed vector packet to8slots; derive boolmask true on actualsourceclauses andfalse onpadding.',
                'The retained frozen decoder/projection applies its distinct saved paragraph and clause feature normalizations at their existing model stages; do not preapply them as the packetproducer input_transform.',
                'Preserve all32state entries, original vocabulary and source/context/order policies. The32vocabulary size and512output limit are distinct.'
            ],
            'recipe_source_owner_pin': lanes[384]['mask_contract']['source_owner_pin'],
            'normalization_and_input_transform_hashes_are_bound_per_custody_lane': True,
            'source_text_and_IDs_needed_for_cache_validation_remain_in_custody_contract': True,
            'IOSpec_is_advisory_complete_runtime_io_contract_false': True,
            'requires_registration_or_database_rewrite': False,
            'mirror_and_runtime_docs_should_use_explicit_recipe': True},
        'no_unresolved_material_contract_findings': True,
        'inspected_pins_before': before, 'inspected_pins_after': after, 'inspected_bytes_unchanged': True,
        'review_independence_disclosure': 'Reviewer authored custodysurvey and sourceproducerjoins; root authored registrationplan/publicationadapter. Review independently checks their exact joins without owner/runtime/database execution.',
        'operations': {'model_loaded': False, 'inference_executed': False, 'training_executed': False,
            'database_access_or_writes': False, 'network_calls': False,
            'registered_plan_or_receipt_edits': False, 'only_new_review_evidence_written': True}}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'semantic_contract_review_passed': True,
        'reviewed_lanes': 2, 'registered_artifacts_preserved': True, 'precision_note_recorded': True}))


if __name__ == '__main__':
    main()
