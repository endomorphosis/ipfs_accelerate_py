"""Replay byte authentication of nine existing exact catalog nominations only."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-decoder-contract-20261006')
OUT = Path(__file__).parent
STORE = Path('/home/barberb/lift_coding/external/ipfs_accelerate/model_manager.duckdb')


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'sentence_transformers',
                                      'huggingface_hub', 'tensorflow'} or fullname == 'ipfs_accelerate_py.model_manager':
            raise ImportError('model/runtime imports are forbidden in this byte-only replay')


def pin(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def witness(path):
    if not path.exists():
        return None
    info = path.stat()
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def main():
    sys.path.insert(0, str(ROOT))
    sys.meta_path.insert(0, NoModels())
    from ipfs_accelerate_py.agent_supervisor.runtime.task_ir_checkpoint import authenticate_task_ir_checkpoints
    source = ROOT / 'docs/agent_supervisor/evidence/supervisor-task-context-20261005/retained-ir-catalog-nominations.json'
    retained = json.loads(source.read_text())
    requests = retained['requests']
    if isinstance(requests, dict):
        requests = list(requests.values())
    assert len(requests) == 9
    before = {str(p): witness(p) for p in [STORE, Path(str(STORE) + '.wal')]}
    source_before = pin(source)
    original_batch_refusal = None
    try:
        authenticate_task_ir_checkpoints(catalog_path=STORE, selections=requests)
    except ValueError as error:
        original_batch_refusal = {'type': type(error).__name__, 'message': str(error)}
    assert original_batch_refusal is not None
    resolutions = retained['native_resolutions']
    refused_aliases = []
    admitted_requests = []
    for request, resolution in zip(requests, resolutions):
        original = resolution['selected_binding']['declaration']['original_checkpoint_pin']
        path = Path(original['path'])
        if path.resolve(strict=True) != path:
            refused_aliases.append({'selectors': request, 'original_checkpoint_pin': original,
                'resolved_target': str(path.resolve(strict=True)),
                'reason': 'Original catalog locator traverses a mutable ancestor alias; requires a separate reviewed canonical binding, never an implicit target substitution.'})
        else:
            admitted_requests.append(request)
    assert len(refused_aliases) == 1 and refused_aliases[0]['selectors']['ir_family_id'] == 'codebase_ir'
    observed = authenticate_task_ir_checkpoints(catalog_path=STORE, selections=admitted_requests)
    after = {str(p): witness(p) for p in [STORE, Path(str(STORE) + '.wal')]}
    assert before == after and pin(source) == source_before
    assert len(observed) == 8 and sum(row['original_checkpoint_pin']['bytes'] for row in observed) == 105868568
    assert all(row['checkpoint_bytes_authenticated'] is True and not any(row['authority'].values())
               and row['native_resolution']['authority']['checkpoint_bytes_authenticated'] is False for row in observed)
    result = {'schema': 'supervisor-retained-checkpoint-authentication-replay@1',
        'store': str(STORE), 'source_nomination_receipt': source_before,
        'original_nine_batch_refusal': original_batch_refusal, 'refused_noncanonical_catalog_locators': refused_aliases,
        'observations': observed, 'authenticated_existing_files': len(observed),
        'authenticated_existing_bytes': sum(row['original_checkpoint_pin']['bytes'] for row in observed),
        'store_stat_witnesses_before': before, 'store_stat_witnesses_after': after,
        'store_wal_stat_witnesses_equal': True,
        'scope': 'Eight exact existing canonical retained catalog selections authenticated. Original nine-selection batch refuses its Codebase384 ancestor alias. No decoder shape/ABI/quality/runtime or Hub authentication and no model regeneration.',
        'new_embeddings': False, 'model_loaded': False, 'training': False,
        'decoder_runtime_admitted': False, 'proof_authority': False, 'completion_authority': False}
    (OUT / 'retained-checkpoint-authentication.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('authenticated_existing_files', 'authenticated_existing_bytes', 'store_wal_stat_witnesses_equal', 'decoder_runtime_admitted')}))


if __name__ == '__main__':
    main()
