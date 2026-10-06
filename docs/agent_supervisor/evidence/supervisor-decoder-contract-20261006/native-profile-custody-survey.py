"""Read-only reproduction of the retained decoder profile source-custody gap.

The native rebuild is detached and remains in memory. This script does not
write registries, create databases, import numerical owners or load models.
Run with Python and redirect stdout to the adjacent JSON evidence artifact.
"""
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OWNER = Path('/home/barberb/lift_coding/.worktrees/ir-model-manager-hf-sync-20261004')
OWNER_HEAD = '73db2c8f3edb9fbdfc5decb0e8f12bb4e86e10f3'
REGISTRY = Path('/home/barberb/lift_coding/artifacts/ir-decoder-profile-inventory-20261004/controls/runs/actual-inventory-v3/profile-inventory.json')
REGISTRY_SHA = '7124e191e7f6decc204b449dd9d8db3720f123278ba2cf8b7d4f1517bf63d641'
SOURCE_NAMES = ('ir_decoder_profile_inventory.py', 'ir_decoder_format_runtime.py',
                'ir_original_corpus_runtime.py', 'ir_cell_runtime.py',
                'ir_cell_routing.py', 'ir_cell_target_compatibility.py', 'checkpoint_hub.py')
FORBIDDEN = {'torch', 'transformers', 'numpy', 'sentence_transformers',
             'huggingface_hub', 'duckdb', 'requests', 'httpx', 'socket',
             'urllib', 'http', 'aiohttp', 'ftplib'}


def git(*args):
    return subprocess.run(['git', '-C', str(OWNER), *args], check=True,
                          capture_output=True).stdout


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def differences(left, right, path=''):
    if type(left) is not type(right):
        return [{'path': path, 'before_type': type(left).__name__,
                 'after_type': type(right).__name__}]
    if type(left) is dict:
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append({'path': path + '/' + key,
                               'before_present': key in left, 'after_present': key in right})
            else:
                result.extend(differences(left[key], right[key], path + '/' + key))
        return result
    if type(left) is list:
        result = []
        if len(left) != len(right):
            result.append({'path': path + '/length', 'before': len(left), 'after': len(right)})
        for index, (first, second) in enumerate(zip(left, right)):
            result.extend(differences(first, second, path + '/' + str(index)))
        return result
    return [] if left == right else [{'path': path, 'before': left, 'after': right}]


assert git('rev-parse', 'HEAD').decode().strip() == OWNER_HEAD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
source_pins = []
for name in SOURCE_NAMES:
    relative = Path('ipfs_datasets_py/logic/formalization/autoencoder') / name
    raw = (OWNER / relative).read_bytes()
    assert raw == git('show', OWNER_HEAD + ':' + relative.as_posix())
    source_pins.append(pin(OWNER / relative))
registry_before = pin(REGISTRY)
assert registry_before['bytes'] == 125540 and registry_before['sha256'] == REGISTRY_SHA
registry = json.loads(REGISTRY.read_bytes())

os.environ['IPFS_DATASETS_PY_MINIMAL_IMPORTS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWNER))
original_import = builtins.__import__
forbidden_attempts = []
modules_before_native = set(sys.modules)


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN:
        forbidden_attempts.append(name)
        raise AssertionError('metadata-only probe attempted forbidden owner import: ' + name)
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
from ipfs_datasets_py.logic.formalization.autoencoder import ir_decoder_profile_inventory as native

rebuilt = native.build_ir_decoder_profile_inventory(registry['directory_plan_receipt'],
                                                   registry['inventory_receipts'], registry['bindings'])
leaf_differences = differences(registry, rebuilt)
outcomes = []
for record in registry['checkpoints']:
    selected = {'ir_family_id': record['request']['ir_family_id'],
                'record_id': record['record_id'], 'profile_id': record['profile_id'],
                'format_id': record['format_id'], 'run_id': record['run_id'],
                'request': record['request'], 'format_request': record['format_request'],
                'checkpoint_receipt': record['checkpoint_receipt']}
    try:
        result = native.resolve_ir_decoder_profile_route(registry_before, record['request'],
            format_request=record['format_request'], profile_id=record['profile_id'],
            run_id=record['run_id'])
    except native.DecoderProfileInventoryError as error:
        selected.update(status='refused', error_type=type(error).__name__, reason=str(error))
    else:
        selected.update(status='resolved', authority=result['authority'])
    outcomes.append(selected)

registry_after = pin(REGISTRY)
assert registry_after == registry_before
assert source_pins == [pin(Path(item['path'])) for item in source_pins]
assert git('rev-parse', 'HEAD').decode().strip() == OWNER_HEAD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
assert all(name.split('.')[0] in {'socket', 'urllib', 'http', 'aiohttp', 'ftplib'}
           for name in forbidden_attempts)
assert not (FORBIDDEN & (set(sys.modules) - modules_before_native))
report = {
    'schema': 'supervisor-native-decoder-profile-custody-survey@1',
    'owner_worktree': str(OWNER), 'owner_head': OWNER_HEAD,
    'owner_introduced_commit': '323da19d1c5a43749e1677204760941b69f83e43',
    'owner_tracked_clean_before_and_after': True, 'owner_source_pins': source_pins,
    'registry_pin': registry_before, 'registry_unchanged': True,
    'native_build_succeeded': True, 'native_rebuild_persisted': False,
    'native_rebuild_cells': len(rebuilt['cells']),
    'format_identity_preserved': registry['formats'] == rebuilt['formats'],
    'profile_identity_preserved': registry['profiles'] == rebuilt['profiles'],
    'record_ids_preserved': [item['record_id'] for item in registry['checkpoints']] ==
                            [item['record_id'] for item in rebuilt['checkpoints']],
    'leaf_difference_count': len(leaf_differences), 'leaf_differences': leaf_differences,
    'exact_saved_registry_route_outcomes': outcomes,
    'native_rebuild_authority': rebuilt['authority'],
    'forbidden_import_roots': sorted(FORBIDDEN), 'forbidden_import_attempts': forbidden_attempts,
    'preexisting_forbidden_roots': sorted(FORBIDDEN & modules_before_native),
    'forbidden_owner_modules_imported': sorted(FORBIDDEN & (set(sys.modules) - modules_before_native)),
    'import_fence_scope': 'Network-root import attempts are denied. Optional package bootstrap '
        'may catch an import refusal; attempted roots are retained above. Preexisting urllib '
        'comes from stdlib path bootstrap. No forbidden root was newly imported.',
    'interpretation': 'Native metadata rebuild preserves both existing format/profile/record identities. '
        'Saved snapshot resolution refuses because source-file locators and their concrete format-contract '
        'digests belong to a different checkout. Rebinding must be explicit; no mismatch was ignored.',
    'limitations': 'This is one retained two-format inventory and one committed metadata owner. '
        'It authenticates neither decoder numerical operation, training execution, token/span capability, '
        'source fidelity, model quality, proof status nor remote availability.',
    'model_manager_constructed': False, 'database_mutated': False,
    'model_loaded': False, 'training_executed': False, 'inference_executed': False,
    'download_calls': 0, 'provider_calls': 0, 'runtime_admitted': False,
    'completion_authority': False, 'proof_authority': False,
}
print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
