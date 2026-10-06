"""Rebind retained profile custody through the already restored current owner.

This writes only a new detached metadata export next to this script. Existing
registries, checkpoints, vectors, corpora, databases and Hub state are preserved.
The original and prior rebound registries are inspected without repairing them.
"""
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OWNER = Path('/home/barberb/lift_coding/.worktrees/decoder-profile-recovery-datasets-20261006')
OWNER_HEAD = '3b3b994407b2fcfef955ce5153afc2eea01e4eff'
RETAINED_OWNER = Path('/home/barberb/lift_coding/.worktrees/ir-model-manager-hf-sync-20261004')
RETAINED_HEAD = '73db2c8f3edb9fbdfc5decb0e8f12bb4e86e10f3'
ORIGINAL = Path('/home/barberb/lift_coding/artifacts/ir-decoder-profile-inventory-20261004/controls/runs/actual-inventory-v3/profile-inventory.json')
ORIGINAL_SHA = '7124e191e7f6decc204b449dd9d8db3720f123278ba2cf8b7d4f1517bf63d641'
PRIOR_REBOUND = Path('/home/barberb/lift_coding/artifacts/supervisor-decoder-contract-20261006/rebound-original-profile-inventory.json')
PRIOR_REBOUND_SHA = '45cd178e5d23f33c5d04116d8cae08633046ac526b6e5ad1248d9141b9fcc8e1'
OUTPUT = Path('/home/barberb/lift_coding/artifacts/decoder-profile-recovery-20261006/native-owner/profile-inventory-current-3b3.json')
METADATA_NAMES = ('ir_decoder_profile_inventory.py', 'ir_decoder_format_runtime.py',
    'ir_original_corpus_runtime.py', 'ir_cell_runtime.py', 'ir_cell_routing.py',
    'ir_cell_target_compatibility.py')
TEST_NAMES = ('ir_decoder_profile_inventory', 'ir_decoder_format_runtime',
    'ir_original_corpus_runtime', 'ir_cell_runtime', 'ir_cell_routing',
    'ir_cell_target_compatibility')
PRODUCER_PATHS = ('ipfs_datasets_py/optimizers/logic_theorem_optimizer/domain_384_autoencoder.py',
    'ipfs_datasets_py/logic/intent_ir/formalize/rich_grammar.py',
    'ipfs_datasets_py/logic/software_verification/program.py')
FORBIDDEN = {'torch', 'transformers', 'numpy', 'sentence_transformers',
    'huggingface_hub', 'duckdb', 'requests', 'httpx', 'socket', 'urllib',
    'http', 'aiohttp', 'ftplib'}
NETWORK_ROOTS = {'socket', 'urllib', 'http', 'aiohttp', 'ftplib'}


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True).stdout


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def stable_git_sources(root, head, paths):
    assert git(root, 'rev-parse', 'HEAD').decode().strip() == head
    assert not git(root, 'status', '--porcelain', '--untracked-files=no').strip()
    result = []
    for relative in paths:
        path = root / relative
        assert path.read_bytes() == git(root, 'show', head + ':' + relative)
        result.append(pin(path))
    return result


def difference(left, right, path=''):
    if type(left) is not type(right):
        return [{'path': path, 'before_type': type(left).__name__, 'after_type': type(right).__name__}]
    if type(left) is dict:
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append({'path': path + '/' + key, 'before_present': key in left,
                               'after_present': key in right})
            else:
                result.extend(difference(left[key], right[key], path + '/' + key))
        return result
    if type(left) is list:
        result = []
        if len(left) != len(right):
            result.append({'path': path + '/length', 'before': len(left), 'after': len(right)})
        for index, (first, second) in enumerate(zip(left, right)):
            result.extend(difference(first, second, path + '/' + str(index)))
        return result
    return [] if left == right else [{'path': path, 'before': left, 'after': right}]


metadata_paths = ['ipfs_datasets_py/logic/formalization/autoencoder/' + name for name in METADATA_NAMES]
test_paths = ['tests/unit/logic/formalization/autoencoder/test_' + name + '.py' for name in TEST_NAMES]
hub_path = 'ipfs_datasets_py/logic/formalization/autoencoder/checkpoint_hub.py'
paths = [*metadata_paths, hub_path, *PRODUCER_PATHS, *test_paths]
current_source_pins = stable_git_sources(OWNER, OWNER_HEAD, paths)
retained_source_pins = stable_git_sources(RETAINED_OWNER, RETAINED_HEAD, paths)
comparisons = [{'path': relative, 'same_bytes': (OWNER / relative).read_bytes() ==
                (RETAINED_OWNER / relative).read_bytes()} for relative in paths]
assert all(row['same_bytes'] for row in comparisons if row['path'] != hub_path)
assert not next(row['same_bytes'] for row in comparisons if row['path'] == hub_path)

original_pin, prior_pin = pin(ORIGINAL), pin(PRIOR_REBOUND)
assert original_pin['bytes'] == 125540 and original_pin['sha256'] == ORIGINAL_SHA
assert prior_pin['bytes'] == 125524 and prior_pin['sha256'] == PRIOR_REBOUND_SHA
original = json.loads(ORIGINAL.read_bytes())
prior = json.loads(PRIOR_REBOUND.read_bytes())
checkpoint_pins = [row['checkpoint_receipt'] for row in original['checkpoints']]
assert all(pin(Path(item['path'])) == item for item in checkpoint_pins)

os.environ['IPFS_DATASETS_PY_MINIMAL_IMPORTS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWNER))
standard_import = builtins.__import__
before_native_modules = set(sys.modules)
attempts = []


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN or name == 'ipfs_accelerate_py.model_manager':
        attempts.append(name)
        raise AssertionError('forbidden numerical/database/network owner: ' + name)
    return standard_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
from ipfs_datasets_py.logic.formalization.autoencoder import ir_decoder_profile_inventory as native

rebuilt = native.build_ir_decoder_profile_inventory(original['directory_plan_receipt'],
    original['inventory_receipts'], original['bindings'])
original_delta, prior_delta = difference(original, rebuilt), difference(prior, rebuilt)
assert len(original_delta) == len(prior_delta) == 8
assert original['formats'] == prior['formats'] == rebuilt['formats']
assert original['profiles'] == prior['profiles'] == rebuilt['profiles']
assert [row['record_id'] for row in original['checkpoints']] == [row['record_id'] for row in rebuilt['checkpoints']]
assert original['bindings'] == rebuilt['bindings']
assert original['cells'] == rebuilt['cells'] and len(rebuilt['cells']) == 12
assert not any(rebuilt['authority'].values())
wire = (json.dumps(rebuilt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
assert len(wire) <= 1024 * 1024
assert OUTPUT.is_absolute() and OUTPUT.resolve() == OUTPUT and not OUTPUT.is_symlink()
if OUTPUT.exists():
    assert OUTPUT.read_bytes() == wire, 'Refusing to overwrite a different detached export'
    export_status = 'already_identical_verified'
else:
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(wire)
        stream.flush()
        os.fsync(stream.fileno())
    export_status = 'created_new_detached_export'
output_pin = pin(OUTPUT)
outcomes = []
historical_refusals = []
for record in original['checkpoints']:
    request = {'ir_family_id': record['request']['ir_family_id'],
        'record_id': record['record_id'], 'profile_id': record['profile_id'],
        'format_id': record['format_id'], 'run_id': record['run_id'],
        'request': record['request'], 'format_request': record['format_request']}
    for old_pin in (original_pin, prior_pin):
        refused = {**request, 'registry_pin': old_pin}
        try:
            stale = native.resolve_ir_decoder_profile_route(old_pin, record['request'],
                format_request=record['format_request'], profile_id=record['profile_id'], run_id=record['run_id'])
        except native.DecoderProfileInventoryError as error:
            refused.update(status='refused', reason=str(error), error_type=type(error).__name__)
        else:
            refused.update(status='unexpectedly_resolved', authority=stale['authority'])
        historical_refusals.append(refused)
    outcome = dict(request)
    try:
        resolution = native.resolve_ir_decoder_profile_route(output_pin, record['request'],
            format_request=record['format_request'], profile_id=record['profile_id'], run_id=record['run_id'])
    except native.DecoderProfileInventoryError as error:
        outcome.update(status='refused', reason=str(error), error_type=type(error).__name__)
    else:
        assert resolution['selected_checkpoint']['record_id'] == record['record_id']
        assert resolution['selected_format'] == next(row for row in original['formats']
                                                     if row['format_id'] == record['format_id'])
        assert resolution['selected_profile'] == next(row for row in original['profiles']
                                                       if row['profile_id'] == record['profile_id'])
        assert not any(resolution['authority'].values())
        outcome.update(status='resolved', native_resolution=resolution)
    outcomes.append(outcome)

closing = native.build_ir_decoder_profile_inventory(original['directory_plan_receipt'],
    original['inventory_receipts'], original['bindings'])
assert closing == rebuilt
assert pin(ORIGINAL) == original_pin and pin(PRIOR_REBOUND) == prior_pin and pin(OUTPUT) == output_pin
assert all(pin(Path(item['path'])) == item for item in checkpoint_pins)
assert stable_git_sources(OWNER, OWNER_HEAD, paths) == current_source_pins
assert stable_git_sources(RETAINED_OWNER, RETAINED_HEAD, paths) == retained_source_pins
assert all(name.split('.')[0] in NETWORK_ROOTS for name in attempts)
new_forbidden = FORBIDDEN & (set(sys.modules) - before_native_modules)
assert not new_forbidden
assert 'ipfs_accelerate_py.model_manager' not in sys.modules

receipt = {
    'schema': 'decoder-profile-current-native-owner-qualification@1',
    'owner': {'worktree': str(OWNER), 'head': OWNER_HEAD, 'tracked_clean_before_and_after': True,
              'source_test_pins': current_source_pins},
    'retained_owner': {'worktree': str(RETAINED_OWNER), 'head': RETAINED_HEAD,
                       'tracked_clean_before_and_after': True, 'source_test_pins': retained_source_pins},
    'source_test_comparison': comparisons,
    'source_recovery_needed': False, 'repository_sources_modified': False,
    'loader_compatibility_scope': 'Current optimized=True Legal loaders remain intact. '
        'The six restored metadata owners and six test files exactly match retained committed source.',
    'original_registry_pin': original_pin, 'prior_rebound_registry_pin': prior_pin,
    'historical_registries_preserved': True,
    'new_registry_pin': output_pin, 'export_status': export_status,
    'new_registry_only_mutation': True, 'native_closing_rebuild_equal': True,
    'lane_count': len(rebuilt['cells']), 'all_lane_declarations_preserved': True,
    'format_identity_preserved': True, 'profile_identity_preserved': True,
    'checkpoint_record_ids_preserved': True, 'original_binding_inputs_preserved': True,
    'original_to_current_leaf_difference_count': len(original_delta),
    'original_to_current_leaf_differences': original_delta,
    'prior_to_current_leaf_difference_count': len(prior_delta),
    'prior_to_current_leaf_differences': prior_delta,
    'original_checkpoint_pins_preserved': checkpoint_pins,
    'historical_registry_exact_route_refusals': historical_refusals,
    'native_current_exact_route_outcomes': outcomes,
    'successful_exact_current_routes': sum(row['status'] == 'resolved' for row in outcomes),
    'failed_exact_current_routes': sum(row['status'] == 'refused' for row in outcomes),
    'native_authority': rebuilt['authority'],
    'forbidden_import_roots': sorted(FORBIDDEN), 'denied_import_attempts': attempts,
    'preexisting_forbidden_roots': sorted(FORBIDDEN & before_native_modules),
    'new_forbidden_owner_modules_imported': sorted(new_forbidden),
    'import_fence_scope': 'Network imports are denied and recorded, including optional package '
        'bootstrap refusals. Preexisting urllib belongs to stdlib path bootstrap. '
        'No numerical, database, ModelManager or network root was newly imported.',
    'scope': 'Real original-asset metadata qualification on current committed datasets source. '
        'Two exact supported original IntentIR/SecurityIR 384D serialized native fragment routes '
        'reopen through a newly pinned current-source custody export. Historical snapshots '
        'still refuse under this checkout; no mismatch is waived or original asset migrated.',
    'limitations': 'No decoder model was loaded or executed, no null format identity was '
        'filled, and no runtime, token/span, source-text reconstruction, full-document/logic '
        'decoding, quality, teacher, proof, database inventory or remote availability was qualified. '
        'All source/file checks are cooperative endpoints, not an atomic snapshot.',
    'model_manager_constructed': False, 'database_mutated': False, 'huggingface_mutated': False,
    'model_loaded': False, 'training_executed': False, 'inference_executed': False,
    'download_calls': 0, 'provider_calls': 0, 'runtime_admitted': False,
    'proof_authority': False, 'completion_authority': False,
}
print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
