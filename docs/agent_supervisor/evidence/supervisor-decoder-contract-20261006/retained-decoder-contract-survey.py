#!/usr/bin/env python3
"""Read-only retained checkpoint metadata survey; only --output is written.

No decoder is imported or invoked. Checkpoint JSON is inspected as bounded
metadata; tensor arrays are never included in the result. The native catalog
owner performs exactly one read-only load of the explicitly selected store.
The multi-gigabyte store has cooperative stat witnesses, not a full-file hash.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

BASE = Path('/home/barberb/lift_coding')
DEFAULT_SOURCE = BASE / '.worktrees/supervisor-decoder-contract-20261006'
NATIVE_DATASETS = BASE / '.worktrees/ir-model-manager-hf-sync-20261004'
PINNED_DATASETS = BASE / '.worktrees/ir-supervisor-contracts-datasets-20261004'
NATIVE_HEAD = '73db2c8f3edb9fbdfc5decb0e8f12bb4e86e10f3'
PINNED_HEAD = '987cf856b2b902aa68c4587bb492b19b932b5d30'
STORE = BASE / 'external/ipfs_accelerate/model_manager.duckdb'
REPLAY_ROOT = BASE / 'artifacts/ir-semantic-reconstruction-replay-20261004'
INVENTORY_PATH = BASE / 'artifacts/ir-decoder-profile-inventory-20261004/controls/runs/actual-inventory-v3/profile-inventory.json'
INVENTORY_PIN = {'path': str(INVENTORY_PATH), 'bytes': 125540,
                 'sha256': '7124e191e7f6decc204b449dd9d8db3720f123278ba2cf8b7d4f1517bf63d641'}
MAX_JSON_BYTES = 128 * 1024 * 1024
SELECTORS = ('record_id', 'ir_family_id', 'dimension', 'dimension_role',
             'schema_version', 'task_id', 'profile_id', 'format_id',
             'checkpoint_sha256', 'role')
BOUND_PINS: dict[str, dict] = {}


def witness(path: Path):
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    return {'device': value.st_dev, 'inode': value.st_ino, 'mode': value.st_mode,
            'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'ctime_ns': value.st_ctime_ns}


def pin(path: Path):
    before = witness(path)
    if before is None or not stat.S_ISREG(before['mode']):
        raise ValueError('existing regular file required: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if witness(path) != before:
        raise ValueError('file changed during streamed hash: ' + str(path))
    return {'path': str(path), 'bytes': before['bytes'], 'sha256': digest.hexdigest()}


def bind(path: Path, expected=None):
    found = pin(path)
    if expected is not None and found != expected:
        raise ValueError('recorded file pin mismatch: ' + str(path))
    previous = BOUND_PINS.setdefault(str(path), found)
    if previous != found:
        raise ValueError('conflicting source/file generation: ' + str(path))
    return found


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON field: ' + key)
        result[key] = value
    return result


def read_json(path: Path, expected=None):
    receipt = bind(path, expected)
    if not 0 < receipt['bytes'] <= MAX_JSON_BYTES:
        raise ValueError('JSON metadata file outside survey byte bound')
    with path.open('r', encoding='utf-8') as handle:
        value = json.load(handle, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if pin(path) != receipt:
        raise ValueError('JSON changed while read: ' + str(path))
    return value


def git_observation(root: Path):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    rows = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=root, text=True).splitlines()
    return {'root': str(root), 'head': head, 'status_entry_count': len(rows),
            'status_code_counts': dict(collections.Counter(row[:2] for row in rows)),
            'tracked_clean': not any(row[:2] != '??' for row in rows)}


def fields(value, names):
    return {name: value[name] for name in names if name in value}


def checkpoint_metadata(asset, value):
    family, dimension = asset['ir_family_id'], asset['dimension']
    result = {'top_level_keys': list(value), 'metadata': {},
              'catalog_unknown_fields_remain_unknown': True,
              'new_contract_required_for_runtime_admission': True}
    if family == 'codebase_ir' and dimension == 8:
        result['metadata'] = {'contract': value['contract'],
            'state': fields(value['state'], ('schema', 'latent_width', 'contract_sha256', 'feature_space_sha256')),
            'report': fields(value['report'], ('schema', 'training_target_count', 'qualified', 'admitted', 'formalized'))}
        result['measurement'] = '53 compiler-derived structural features -> latent8 -> structural feature reconstruction; no original-text decoder'
        result['existing_native_format_binding'] = False
    elif dimension == 8:
        space, report = value['space'], value['report']
        result['metadata'] = {'checkpoint_schema': value['schema'],
            'space': fields(space, ('schema', 'domain_id', 'normalization', 'training_reports_sha256', 'producer_pins')),
            'feature_column_count': len(space['columns']),
            'projection_ids': list(space['projections']),
            'report': fields(report, ('schema', 'latent_width', 'training_rows', 'validation_rows',
                'training_executed', 'source_text_decoder_trained', 'required_families', 'trained_logic_families'))}
        result['measurement'] = 'native compiler/projection feature MSE and cosine losses; no original-text reconstruction or textual token budget'
        result['existing_native_format_binding'] = False
    elif family == 'codebase_ir':
        checkpoint = value['checkpoint']
        result['metadata'] = {'schema': value['schema'], 'profile': value['profile'],
            'embedded_checkpoint': fields(checkpoint, ('schema', 'domain_id', 'dimension', 'projection_width', 'parent_binding')),
            'target_schema_kind': checkpoint['target_schema']['schema'],
            'authority': value['authority'],
            'fit': fields(value['fit'], ('schema', 'training_rows', 'qualified', 'source_semantics_verified'))}
        result['measurement'] = 'SecurityIR fixed-tree/ridge payload inside Codebase owner envelope; native Codebase decoder incomplete and child unpromoted'
        result['existing_native_format_binding'] = False
    elif family == 'legal_ir' and dimension == 384:
        result['metadata'] = {'schema': value['schema'], 'binding': value['binding'],
            'projection_id': value['projection_id'], 'config': value['config'],
            'codec_schema': value['codec']['schema'], 'codec_policy': value['codec']['policy'],
            'training_count': value['training_count'], 'training_manifest_sha256': value['training_manifest_sha256']}
        result['measurement'] = 'one CanonicalRoundTripIR@1 deontic rule/seven facets; max source/target codec64; source_input provenance_only_not_neural_input; not originating legal prose'
        result['existing_native_format_binding'] = False
    elif family in ('intent_ir', 'security_ir') and dimension == 384:
        config = value['config']
        provenance = config['embedding_provenance']
        result['metadata'] = {'schema': value['schema'], 'architecture': value['architecture'],
            'domain_id': value['domain_id'], 'dimension': value['dimension'],
            'codec_schema': value['codec']['schema'], 'ordered_target_vocabulary_count': len(value['codec']['target_vocabulary']),
            'config': fields(config, ('hidden_size', 'projection_width', 'token_embedding_dim', 'max_target_tokens')),
            'embedding_provenance': fields(provenance, ('model_id', 'revision', 'dimension', 'device', 'dtype', 'normalized', 'truncated')),
            'training': fields(value['training'], ('selected_epoch', 'selected_validation', 'training', 'training_tokens'))}
        result['measurement'] = 'source embedding -> family-specific native fragment, not full-document or prose reconstruction; release panel0/2 exact2/2 valid'
        result['existing_native_format_binding'] = True
        result['new_contract_required_for_runtime_admission'] = False
    elif family == 'legal_ir' and dimension == 768:
        result['metadata'] = fields(value, ('schema', 'architecture', 'dimension', 'domain_id', 'profile_id',
            'representation_id', 'aligned_representation_id', 'initialization_representation_id',
            'donor_pins', 'codec_sha256', 'mode_policy', 'native768_inputs_used',
            'all_26_inherited_tensors_unchanged', 'distillation_executed', 'training_executed',
            'reference_supervised_training_executed', 'encoder_inference_executed',
            'production_kd_eligible', 'source_fidelity_qualified', 'proof_authority'))
        result['measurement'] = 'trained aligned dual-donor interfaces, retained primary512/grammar64 output budgets; encoder profile8192 is independent; no source-fidelity or distillation qualification'
        result['existing_native_format_binding'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    if args.output.exists():
        raise ValueError('use a fresh output file; retained evidence must not be overwritten')
    evidence = source_root / 'docs/agent_supervisor/evidence/supervisor-task-context-20261005'
    nominations_path = evidence / 'retained-ir-catalog-nominations.json'
    witnesses_path = evidence / 'retained-checkpoint-file-witnesses.json'
    nominations = read_json(nominations_path)
    asset_witnesses = read_json(witnesses_path)
    native_inventory = read_json(INVENTORY_PATH, INVENTORY_PIN)
    handoff = read_json(NATIVE_DATASETS / 'docs/autoencoders/pilots/intent_codebase_grounding/decoder_profile_inventory_handoff.json')
    for relative in ('ir_decoder_profile_inventory.py', 'ir_decoder_format_runtime.py'):
        bind(NATIVE_DATASETS / 'ipfs_datasets_py/logic/formalization/autoencoder' / relative)
    bind(BASE / 'external/ipfs_datasets/docs/autoencoders/inference_and_qualification.md')
    replay_path = REPLAY_ROOT / 'comparison-and-next-gaps-v1.json'
    replay = read_json(replay_path)
    constraints = read_json(REPLAY_ROOT / 'latent-input-contract-constraints-v1.json')
    discovery = read_json(REPLAY_ROOT / 'discovery-registration-handoff-v1.json')
    before_git = [git_observation(root) for root in (PINNED_DATASETS, NATIVE_DATASETS, BASE / 'external/ipfs_datasets')]
    if before_git[0]['head'] != PINNED_HEAD or not before_git[0]['tracked_clean']:
        raise ValueError('pinned dataset checkout differs')
    if before_git[1]['head'] != NATIVE_HEAD or not before_git[1]['tracked_clean']:
        raise ValueError('committed original-profile dataset checkout differs')
    assets = []
    requests = {item['record_id']: item for item in nominations['requests']}
    for asset in asset_witnesses['assets']:
        checkpoint = read_json(Path(asset['pin']['path']), asset['pin'])
        assets.append({'record_id': asset['record_id'], 'ir_family_id': asset['ir_family_id'],
            'dimension': asset['dimension'], 'dimension_role': asset['dimension_role'],
            'original_pin_before': asset['pin'], 'catalog_request_from_prior_receipt': requests[asset['record_id']],
            **checkpoint_metadata(asset, checkpoint)})
    contextual = []
    for profile in replay['checkpoint_profiles']:
        # Stream-check all three better replay arms, but report the two complete
        # contextual states separately from the structured single-rule arm.
        bind(Path(profile['checkpoint']['path']), profile['checkpoint'])
        if profile['serialization_schema'] == 'private-native-dimension-source-state/v1':
            contextual.append({'checkpoint_pin_before': profile['checkpoint'],
                'historical_claim_source_pin': BOUND_PINS[str(replay_path)],
                'historical_profile': profile,
                'fresh_replay_performed_this_survey': False})
    sys.path.insert(0, str(source_root))
    from ipfs_accelerate_py.model_catalog.sources.ir_persistent import IRPersistentCatalogSource
    native_module_pins = []
    for name, module in sorted(sys.modules.items()):
        if name.startswith('ipfs_accelerate_py') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve(strict=True)
            if not path.is_relative_to(source_root):
                raise ValueError('foreign Accelerate import origin: ' + str(path))
            native_module_pins.append(bind(path))
    store_before = witness(STORE)
    wal_path = Path(str(STORE) + '.wal')
    wal_before = witness(wal_path)
    catalog = IRPersistentCatalogSource(path=STORE).load()
    bindings = catalog.ir_bindings
    targets = [item['checkpoint_pin_before']['sha256'] for item in contextual]
    targets.extend(asset['original_pin_before']['sha256'] for asset in assets)
    matches = {sha: [{key: binding[key] for key in SELECTORS} for binding in bindings
                     if binding['checkpoint_sha256'] == sha] for sha in targets}
    store_after, wal_after = witness(STORE), witness(wal_path)
    if store_after != store_before or wal_after != wal_before:
        raise ValueError('selected store/WAL changed across read-only observation')
    loaded_forbidden = [name for name in sys.modules if name == 'torch' or name.startswith(('torch.', 'transformers', 'huggingface_hub')) or name == 'ipfs_accelerate_py.model_manager']
    if loaded_forbidden:
        raise ValueError('unexpected ML/manager implementation import: ' + repr(loaded_forbidden))
    for item in contextual:
        item['current_selected_catalog_exact_sha_matches'] = matches[item['checkpoint_pin_before']['sha256']]
        item['current_presence_scope'] = 'One native read-only load of the explicit selected store; this is not all-host inventory or remote-Hub discovery.'
    for asset in assets:
        asset['current_selected_catalog_exact_sha_matches'] = matches[asset['original_pin_before']['sha256']]
    after_pins = [pin(Path(path)) for path in sorted(BOUND_PINS)]
    if any(item != BOUND_PINS[item['path']] for item in after_pins):
        raise ValueError('bound checkpoint/source/evidence bytes changed during survey')
    after_git = [git_observation(Path(item['root'])) for item in before_git]
    result = {'schema': 'retained-decoder-contract-readonly-survey/v1',
        'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_pin': pin(Path(__file__).resolve()), 'python_executable': sys.executable,
        'source_root': str(source_root), 'source_head_observation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(),
        'scope': 'Bounded metadata and independent streamed file witnesses; no decoder/runtime execution or original prose reconstruction claim.',
        'original_nominated_assets': assets,
        'native_original_profile_inventory': {'pin': INVENTORY_PIN, 'dataset_head': NATIVE_HEAD,
            'native_supported_binding_count': len(native_inventory['checkpoints']),
            'supported_bindings': handoff['supported_bindings'],
            'supported_profile_identities': native_inventory['profiles'],
            'only_two_original_intent_security_384_fragment_contracts_supported': True,
            'seven_nominated_other_assets_have_no_supported_native_format_binding': True,
            'legacy_recovery_policy': 'Recover literal metadata as evidence; preserve null catalog selectors. New unsupported family/task/schema/format contracts require separate reviewed append-only records, not inferred defaults.'},
        'better_retained_contextual_states': contextual,
        'historical_replay_constraints': constraints,
        'historical_registration_handoff': fields(discovery, ('schema', 'status', 'ModelManager_storage_path', 'publisher_API', 'importer_API', 'repositories', 'identity_policy', 'contextual_model_IO', 'work_needed', 'live_network_or_database_checked')),
        'reconstruction_measurements_are_distinct': ['embedding/structural feature loss', 'learned canonical semantic IR generation', 'deterministic compiler/decompiler roundtrip', 'original source-text reconstruction', 'syntax and theorem/proof qualification'],
        'current_selected_model_manager_catalog': {'path': str(STORE), 'native_source': 'IRPersistentCatalogSource',
            'native_load_count': 1, 'read_only': True, 'binding_count': len(bindings),
            'binding_snapshot_revision': catalog.binding_snapshot_revision, 'catalog_revision': catalog.revision,
            'store_stat_before': store_before, 'store_stat_after': store_after,
            'wal_stat_before': wal_before, 'wal_stat_after': wal_after,
            'store_stat_unchanged': True, 'whole_store_sha256_performed': False,
            'store_consistency_scope': 'Cooperative metadata endpoint checks, not atomic snapshot or full store-byte authentication.',
            'native_import_source_pins': native_module_pins,
            'catalog_authority': catalog.to_dict()['authority'],
            'current_contextual_exact_sha_match_counts': {sha: len(matches[sha]) for sha in targets[:len(contextual)]}},
        'file_evidence_before': [BOUND_PINS[path] for path in sorted(BOUND_PINS)],
        'file_evidence_after': after_pins,
        'bound_source_checkpoint_evidence_bytes_unchanged': True,
        'git_observations_before': before_git, 'git_observations_after': after_git,
        'source_checkouts_modified_by_survey': False,
        'forbidden_imports_observed': loaded_forbidden,
        'operations': {'model_manager_constructed': False, 'model_loaded': False, 'inference_executed': False,
            'training_executed': False, 'new_embeddings_generated': False, 'database_writes': False,
            'registrations': False, 'network_or_provider_calls': False, 'HF_writes': False},
        'authority': {'new_quality_measured': False, 'teacher_qualified': False, 'proof_authority': False,
            'runtime_admitted': False, 'original_prose_reconstruction_qualified': False,
            'independent_holdout': False, '8192_span_reconstruction_qualified': False}}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'original_asset_count': len(assets),
        'native_supported_profiles': len(native_inventory['profiles']),
        'contextual_states': len(contextual), 'current_catalog_binding_count': len(bindings),
        'current_contextual_exact_sha_match_counts': result['current_selected_model_manager_catalog']['current_contextual_exact_sha_match_counts'],
        'bound_file_count': len(BOUND_PINS), 'bound_bytes_unchanged': True}))


if __name__ == '__main__':
    main()
