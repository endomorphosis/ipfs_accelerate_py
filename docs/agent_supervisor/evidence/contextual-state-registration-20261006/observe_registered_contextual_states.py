"""Read-only supervisor nomination/byte check of the newly persisted original states."""
import json
import os
from pathlib import Path
import sys

from custody import capture, read, require, source, write

BASE = Path('/home/barberb/lift_coding')
OUT = Path(__file__).resolve().parent
ACCELERATE = BASE / '.worktrees/contextual-state-registration-accelerate-20261006'
STORE = BASE / 'external/ipfs_accelerate/model_manager.duckdb'


def main():
    for key, value in {'IPFS_ACCEL_SKIP_CORE': '1', 'IPFS_ACCEL_AUTO_INSTALL': '0',
        'IPFS_DATASETS_ENABLED': '0', 'IPFS_KIT_DISABLE': '1', 'STORAGE_FORCE_LOCAL': '1'}.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ACCELERATE))
    preparation_pin = capture(OUT / 'registration-preparation.json')
    preparation = json.loads(read(preparation_pin))
    plan = json.loads(read(preparation['import_plan_pin']))
    owners = [source(ACCELERATE, relative) for relative in (
        'ipfs_accelerate_py/agent_supervisor/runtime/task_ir_checkpoint.py',
        'ipfs_accelerate_py/agent_supervisor/runtime/task_ir_selection.py',
        'ipfs_accelerate_py/model_catalog/sources/ir_persistent.py',
        'ipfs_accelerate_py/model_catalog/sources/persistent.py')]
    import duckdb
    from ipfs_accelerate_py.model_catalog.sources.ir_persistent import IRPersistentCatalogSource
    from ipfs_accelerate_py.agent_supervisor.runtime.task_ir_checkpoint import authenticate_task_ir_checkpoints
    selectors = []
    keys = ('record_id', 'ir_family_id', 'dimension', 'dimension_role',
            'schema_version', 'task_id', 'profile_id', 'format_id', 'role')
    for record in plan['models']:
        declaration = record['model_metadata']['huggingface_config']['ir_checkpoint']
        selectors.append({**{key: declaration[key] for key in keys},
                          'checkpoint_sha256': declaration['original_checkpoint_pin']['sha256']})
    catalog = IRPersistentCatalogSource(path=STORE).load()
    outcomes = authenticate_task_ir_checkpoints(catalog_path=STORE, selections=selectors)
    require(len(outcomes) == 2 and all(row['checkpoint_bytes_authenticated'] is True for row in outcomes),
            'two original checkpoint byte authentications required')
    require(all(row['native_resolution']['authority']['runtime_admitted'] is False for row in outcomes),
            'metadata/byte observation must not grant runtime')
    closing = IRPersistentCatalogSource(path=STORE).load()
    require(catalog.binding_snapshot_revision == closing.binding_snapshot_revision,
            'registered catalog generation changed during supervisor observation')
    require(owners == [source(ACCELERATE, row['relative_path']) for row in owners], 'supervisor owners changed')
    result = {'schema': 'registered-contextual-states-supervisor-observation/v1',
        'completed': True, 'preparation_pin': preparation_pin, 'source_owners': owners,
        'catalog_revision': closing.revision, 'binding_snapshot_revision': closing.binding_snapshot_revision,
        'exact_selectors': selectors, 'observations': outcomes, 'successful_byte_authentications': 2,
        'checkpoint_bytes_unchanged': True, 'model_manager_constructed': False, 'database_mutated': False,
        'model_loaded': False, 'inference_executed': False, 'training_executed': False,
        'runtime_admitted': False, 'proof_authority': False,
        'scope': 'Existing explicit supervisor nominations and checkpoint byte authentication only; contextual runtime wrapper is pending.'}
    print(json.dumps({'receipt_pin': write(OUT / 'supervisor-registered-state-observation.json', result)}))


if __name__ == '__main__':
    main()
