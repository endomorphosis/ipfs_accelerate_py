"""Prepare exact retained states and explicit plans; no Hub operations occur."""
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

BASE = Path('/home/barberb/lift_coding')
OUT = Path(__file__).resolve().parent
DATASETS = BASE / '.worktrees/decoder-profile-recovery-datasets-20261006'
SOURCE_REVISION = '3b3b994407b2fcfef955ce5153afc2eea01e4eff'
PUBLISHER = 'ipfs_datasets_py/logic/formalization/autoencoder/ir_model_hub_publish.py'
SURVEY = OUT.parent / 'asset-survey/contextual-state-custody-survey.json'
AGGREGATE_RECEIPT = OUT.parent / 'verified-contextual-publication-receipt.json'
IMPORT_PLAN = OUT.parent / 'contextual-model-manager-import-plan.json'
PREFIX = 'releases/20261006-contextual-selected-v1'
PINS = {}
FORBIDDEN = {'torch', 'transformers', 'numpy', 'sentence_transformers', 'duckdb',
             'huggingface_hub', 'requests', 'httpx', 'socket', 'urllib', 'http'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def pinned(path, expected=None, retain=False):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and 0 < before.st_size <= 16 * 1024 * 1024, 'bounded independent regular file required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    chunks, count, digest = [], 0, hashlib.sha256()
    try:
        require(identity(os.fstat(descriptor)) == identity(before), 'file changed before read')
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, before.st_size - count + 1))
            if not chunk:
                break
            count += len(chunk)
            require(count <= before.st_size, 'file grew during read')
            digest.update(chunk)
            if retain:
                chunks.append(chunk)
        require(identity(os.fstat(descriptor)) == identity(before) == identity(path.lstat()),
                'file identity changed during read')
    finally:
        os.close(descriptor)
    require(count == before.st_size, 'file size changed')
    value = {'path': str(path), 'bytes': count, 'sha256': digest.hexdigest()}
    require(expected is None or value == expected, 'original file pin differs')
    require(str(path) not in PINS or PINS[str(path)] == value, 'file generation changed')
    PINS[str(path)] = value
    return value, b''.join(chunks) if retain else None


def read(path):
    _, payload = pinned(path, retain=True)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('finite JSON required')))


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def create(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.resolve() == path and not path.is_symlink(), 'canonical owned stage required')
    if path.exists():
        require(path.read_bytes() == payload, 'refusing to overwrite a different staged artifact')
    else:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return pinned(path)[0]


survey = read(SURVEY)
aggregate = read(AGGREGATE_RECEIPT)
import_plan = read(IMPORT_PLAN)
require(survey['completed'] is True and survey['bound_files_unchanged'] is True, 'completed custody survey required')
require(aggregate['files_verified'] is True and aggregate['revision'] == '49839a5e55e2ef8ab3815f4e2650dc69ee72d5b8',
        'exact original aggregate publication receipt required')
source_path = DATASETS / PUBLISHER
source_pin, source_bytes = pinned(source_path, retain=True)
committed = subprocess.run(['git', '-C', str(DATASETS), 'show', SOURCE_REVISION + ':' + PUBLISHER],
                           check=True, capture_output=True).stdout
require(source_bytes == committed, 'native publisher differs from committed source')
for item in survey['source_owner_selected_historical_pins']:
    pinned(item['path'], item)

standard_import = builtins.__import__
attempts, before_modules = [], set(sys.modules)


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN or name == 'ipfs_accelerate_py.model_manager':
        attempts.append(name)
        raise AssertionError('staging imported forbidden owner')
    return standard_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('dimension_mirror_native_publisher', source_path)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
lanes, plans = [], []
for lane in survey['lanes']:
    width = lane['dimension']
    require(type(width) is int and width in (384, 768), 'exact authorized dimension required')
    repo = 'Publicus/legal-ir-autoencoder-' + str(width) + 'd'
    info_path = OUT.parent / ('hf-legal' + str(width) + '-repository-info.json')
    info = read(info_path)
    require(info['id'] == repo and info['private'] is False, 'existing public dimension repository required')
    checkpoint_pin, checkpoint_bytes = pinned(lane['original_checkpoint_pin']['path'], lane['original_checkpoint_pin'], retain=True)
    require(lane['complete_model_state_entry_count'] == 32 and lane['saved_model_state_hash_matches_full_publication_container'] is True,
            'complete retained selected state required')
    for item in lane['exact_cached_file_pins'].values():
        pinned(item['path'], item)
    require(lane['decoder_output_limit_tokens'] == lane['encoder_experiment_limit_tokens'] == 512,
            'historical experimental budgets differ')
    contract = lane['append_only_registration_contract']
    require(contract['ir_family_id'] == 'legal_ir' and contract['dimension'] == width
            and contract['dimension_role'] == 'input_embedding'
            and contract['schema_version'] is contract['profile_id'] is contract['format_id'] is None,
            'exact existing unknown format identities must be preserved')
    matching = [row for row in import_plan['models']
                if row['checkpoint_pin']['sha256'] == checkpoint_pin['sha256']]
    require(len(matching) == 1, 'exact original ModelManager record reference required')
    metadata = matching[0]['model_metadata']
    checkpoint_relative = ('checkpoints/decoder-selected-continuation-20261004/' + str(width)
                           + '-selected-followup-lr0001-1729/selected-state.json')
    stage = OUT / (str(width) + 'd')
    staged_checkpoint_pin = create(stage / checkpoint_relative, checkpoint_bytes)
    historical = lane['historical_complete_state_HF_publication']
    aggregate_matches = [row for row in aggregate['files']
                         if row['sha256'] == checkpoint_pin['sha256'] and row['bytes'] == checkpoint_pin['bytes']]
    require(len(aggregate_matches) == 1 and aggregate_matches[0]['path_in_repo'] == historical['file']['path'],
            'fresh original aggregate receipt differs')
    for path in (PREFIX + '/' + checkpoint_relative, PREFIX + '/manifest.json', PREFIX + '/README.md'):
        require(not any(row['rfilename'] == path for row in info['siblings']), 'staged release path already exists in captured repository')
    normalizations = {name: {'content_sha256': value['content_sha256'],
        'receipt_sha256': value['metadata']['receipt_sha256'], 'scope': value['metadata']['scope'],
        'validation_rows_used_for_fitting': value['metadata']['validation_rows_used_for_fitting']}
        for name, value in lane['normalization'].items()}
    manifest = {
        'schema': 'retained-contextual-legal-ir-dimension-mirror/v1',
        'release_prefix': PREFIX, 'repository_id': repo, 'ir_family_id': 'legal_ir',
        'dimension': width, 'dimension_role': 'input_embedding',
        'task_id': contract['task_id'], 'native_IR_schema_version': None,
        'decoder_profile_id': None, 'decoder_format_id': None,
        'checkpoint_serialization_schema': lane['serialization_schema'],
        'checkpoint': {'path_in_release': checkpoint_relative, 'bytes': checkpoint_pin['bytes'],
                       'sha256': checkpoint_pin['sha256'], 'model_state_entries': 32,
                       'model_state_canonical_JSON_sha256': lane['saved_model_state_canonical_JSON_sha256']},
        'original_aggregate_release': {'repository_id': aggregate['repository_id'], 'revision': aggregate['revision'],
            'path_in_repo': historical['file']['path'], 'bytes': checkpoint_pin['bytes'], 'sha256': checkpoint_pin['sha256']},
        'original_model_manager_record': {'record_id': metadata['model_id'],
            'original_checkpoint_sha256': checkpoint_pin['sha256'], 'primary_publication_unchanged': True,
            'record_rewritten_by_this_mirror': False},
        'input_contract': {'paragraph_vector_width': width, 'clause_tensor_shape': ['batch', 8, width],
            'mask_shape': ['batch', 8], 'mask_dtype': 'bool', 'mask_recipe_source_pin': lane['mask_contract']['source_owner_pin'],
            'mask_recipe': 'Transform raw literal-source clause vectors with the original saved TRAIN-only input_transform first; '
                'then zero-pad to 8 positions and derive true clause positions plus false padding. '
                'Saved paragraph/clause feature normalizations are separate decoder stages and must not be applied twice.',
            'standalone_cached_mask_file': False, 'source_binding_content_sha256': lane['source_binding_content_sha256'],
            'saved_input_transform': {'content_sha256': lane['input_transform_content_sha256'],
                'metadata': lane['input_transform_metadata'],
                'container_pin': lane['exact_cached_file_pins']['original_preprocessing']},
            'separate_decoder_feature_normalizations': normalizations,
            'cached_file_references': lane['exact_cached_file_pins'],
            'cached_payloads_included_in_this_release': False,
            'representation_declaration': lane['representation_declaration'],
            'encoder_experiment_limit_tokens': 512, 'qualified_source_token_limit': None},
        'output_contract': {'shape': 'JSON object containing an ordered rules array',
            'rule_fields': ['modality', 'actor', 'action', 'object', 'conditions', 'exceptions', 'temporal'],
            'observed_qualifier_fields_empty': True, 'codec_schema': lane['codec']['schema'],
            'codec_content_sha256': lane['codec_sha256'], 'target_vocabulary_size': len(lane['codec']['target_vocabulary']),
            'decoder_output_limit_tokens': 512, 'qualified_target_token_limit': None},
        'historical_regression': {'metrics': lane['historical_replay_metrics'],
            'source_receipt_pin': lane['historical_replay_source_pin'],
            'scope': 'Previously exposed authored 48-paragraph regression with 180 expected rules; '
                'both selected checkpoints had 48/48 exact canonical IR. Source-only48 input artifacts copy '
                'the original validation subset. This mirror performs no new numerical replay.',
            'fresh_independent_holdout': False, 'original_prose_reconstruction_qualified': False,
            'single_paragraph_vector_only_qualified': False, 'larger_source_spans_qualified': False},
        'historical_decoder_recipe_source_pins': survey['source_owner_selected_historical_pins'],
        'authority': {'complete_runtime_io_contract': False, 'runtime_ready': False,
            'teacher_qualified': False, 'source_fidelity_qualified': False, 'proof_authority': False},
        'operations': {'original_weights_modified': False, 'new_embeddings_generated': False,
            'new_training_executed': False, 'numerical_inference_executed': False,
            'model_manager_record_rewritten': False},
    }
    manifest_pin = create(stage / 'manifest.json', raw(manifest))
    model_profile = ('The historical native 768D encoder profile declares an 8192-token ceiling; this experiment '
        'admits 512 source tokens and the decoder output limit is 512 tokens. The original profile is preserved.\n\n'
        if width == 768 else 'The encoder experiment admits 512 source tokens and the decoder output limit is 512 tokens.\n\n')
    readme = ('# Retained contextual LegalIR ' + str(width) + 'D selected checkpoint\n\n'
        'This release mirrors the exact complete original selected state, with SHA256 `' + checkpoint_pin['sha256']
        + '`, already published in [the aggregate repository](https://huggingface.co/' + aggregate['repository_id']
        + '/blob/' + aggregate['revision'] + '/' + historical['file']['path'] + '). It preserves all 32 saved model-state entries.\n\n'
        'The retained input contract uses a paragraph vector plus ordered literal-source clause vectors. '
        'Apply the original saved TRAIN-only input transform to raw clause vectors first, then zero-pad to eight '
        'positions with a boolean mask. Paragraph and clause feature normalizations are separate saved decoder stages; '
        'do not apply those transforms twice. Exact original cache, preprocessing and producer pins are in `manifest.json`. '
        'This mirror includes checkpoint bytes and metadata; it references the existing caches without uploading them.\n\n'
        + model_profile
        + 'The output is an ordered canonical rules array with modality, actor, action, object, conditions, exceptions '
        'and temporal fields. The observed qualifier fields are empty. Historical replay reconstructed 48/48 previously '
        'exposed authored paragraphs exactly as canonical IR, covering 180 rules. Those results describe the retained '
        'regression cohort; this publication performs no new inference or evaluation.\n\n'
        'Native IR schema, decoder profile and decoder format identities remain unknown (`null`). Runtime, source '
        'fidelity, teacher and proof qualifications remain false. Original-prose reconstruction, independent held-out '
        'accuracy, paragraph-vector-only decoding and larger text spans require their own evaluation.\n\n'
        'This release adds explicit versioned files. Existing repository paths, root model card, defaults and visibility '
        'are preserved. The original ModelManager record retains its immutable aggregate publication binding.\n')
    readme_pin = create(stage / 'README.md', readme.encode('utf-8'))
    plan = {'schema': publisher.PLAN_SCHEMA, 'manifest_pin': manifest_pin, 'repository_id': repo,
            'private_new': False, 'operations': [
                {'file_pin': staged_checkpoint_pin, 'path_in_repo': PREFIX + '/' + checkpoint_relative},
                {'file_pin': manifest_pin, 'path_in_repo': PREFIX + '/manifest.json'},
                {'file_pin': readme_pin, 'path_in_repo': PREFIX + '/README.md'}]}
    publisher._freeze_files(publisher._capture_plan(plan))
    plan_pin = create(stage / 'publication-plan.json', raw(plan))
    lanes.append({'dimension': width, 'repository_id': repo, 'observed_initial_revision': info['sha'],
        'observed_initial_visibility_private': info['private'], 'observed_initial_file_count': len(info['siblings']),
        'repository_info_pin': PINS[str(info_path)], 'original_checkpoint_pin': checkpoint_pin,
        'staged_checkpoint_pin': staged_checkpoint_pin, 'manifest_pin': manifest_pin, 'readme_pin': readme_pin,
        'publication_plan_pin': plan_pin, 'planned_operations': plan['operations'],
        'original_model_manager_record_id': metadata['model_id']})
    plans.append(plan)

require(attempts == [] and not (FORBIDDEN & (set(sys.modules) - before_modules)), 'staging crossed a forbidden owner boundary')
require(pinned(source_path)[0] == source_pin, 'native publisher changed during staging')
for original in list(PINS.values()):
    pinned(original['path'], original)
preparation = {'schema': 'retained-contextual-dimension-mirror-preparation/v1', 'completed': True,
    'native_publisher_source': {'revision': SOURCE_REVISION, 'file_pin': source_pin},
    'asset_survey_pin': PINS[str(SURVEY)], 'verified_aggregate_publication_receipt_pin': PINS[str(AGGREGATE_RECEIPT)],
    'original_model_manager_import_plan_pin': PINS[str(IMPORT_PLAN)], 'release_prefix': PREFIX,
    'lanes': lanes, 'planned_repository_count': 2, 'planned_new_files': 6,
    'planned_checkpoint_bytes': sum(lane['original_checkpoint_pin']['bytes'] for lane in lanes),
    'all_original_and_staged_file_pins': list(PINS.values()), 'file_pins_unchanged': True,
    'native_local_plan_validation_completed': True, 'remote_calls': 0, 'Hub_mutations': 0,
    'model_loaded': False, 'database_access': False, 'training_executed': False,
    'new_embeddings_generated': False, 'runtime_admitted': False, 'proof_authority': False,
    'publication_authorization_pending_root_review': True, 'forbidden_import_attempts': attempts,
    'continuation': 'After root reviews these exact pins, use the current native public append-only publisher '
        'against the two existing repositories and verify every preexisting path identity and visibility. '
        'Retain mirror receipts as equivalent-byte links without rewriting original ModelManager records.'}
print(json.dumps(preparation, sort_keys=True, indent=2, allow_nan=False))
