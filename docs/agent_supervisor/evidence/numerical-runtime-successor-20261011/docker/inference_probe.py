"""Offline container qualification of cached NumPy selection inference.

Authored source fixtures are scanned as text, never imported or executed.
No provider, training, model download or proof discharge is requested.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path[:0] = ["/code/source", "/code/datasets", "/code/kit"]

from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as owner
from ipfs_datasets_py.logic.software_contracts.codebase_ir import (
    CodebaseScanLimits, RepositoryCodebaseIndex, StaleCodebaseError,
)
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
import duckdb
import torch

OUT = Path('/qualification-output')
SNAPSHOT = '/models/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd'
SOURCES = {
    'matching.py': b'import numpy as array_api\n\ndef choose_index(scores, buffer, options, token):\n    return array_api.argmax(scores)\n',
    'mismatching.py': b'import numpy as np\n\ndef select_offset(values, other, context, marker):\n    return np.argmax(values)\n',
    'over512.py': b'# ' + b'word ' * 900 + b'\nimport numpy as np\ndef index_of(values):\n    return np.argmax(values)\n',
    'unsupported.py': b'raise RuntimeError("source fixture must never execute")\ndef f(x): return x+1\n',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def forbidden(*args, **kwargs):
    raise AssertionError('cold native operation replay invoked an inference or embedding seam')


def main():
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
    assert not torch.cuda.is_available()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='numerical-container-') as temporary:
        root = Path(temporary)
        repo = root / 'repo'
        repo.mkdir()
        for name, source in SOURCES.items():
            (repo / name).write_bytes(source)
        for args in (('init', '-q'), ('add', '.'),
                ('-c', 'user.name=Qualification', '-c', 'user.email=qualification@example.invalid',
                 'commit', '-qm', 'authored offline numerical source fixtures')):
            subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
        cx = duckdb.connect(str(root / 'source.duckdb'), config={'threads': 1, 'memory_limit': '64MB'})
        store = DuckDBASTStore(connection=cx)
        cas = ImmutableCAS(root / 'cas')
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=cas,
            catalog=CodebaseCatalog(store, cas))
        head = index.prepare_current(repo, repository_id='offline-numerical-container', operation_id='capture',
            expected_head=None, limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1048576),
            memory_mb=512).head
        registry = AutoencoderRegistry(root / 'models.duckdb', root / 'models')
        assets = dict(checkpoint_path='/models/checkpoint.json',
            checkpoint_sha256=owner.RELEASE['checkpoint_file_sha256'], embedding_snapshot=SNAPSHOT)
        options = dict(expected_head=head, paths=sorted(SOURCES))
        before = time.monotonic()
        result = owner.infer_numerical_selection(index, repo, registry=registry,
            **options, **assets, timeout_seconds=180, memory_mb=4096)
        inference_seconds = time.monotonic() - before
        write('native-result.json', result)
        assert result['native_worker_executed'] is result['inference_executed'] is True
        report = result['report']
        output = report['output']
        assert output['model_loads'] == 1
        assert report['key']['head_sha256'] == 'd22acf5a95d2ceb9340e6ad741928c536b4d52eb13ce846c040a4778f62b33d8'
        assert report['key']['projection_sha256'] == '4e90ec5a3ff3978d3f6d40fb366ea01f2a056fe6c59b3fddcf7c50f66366afd7'
        assert report['worker_receipt']['provider_calls'] == 0
        assert report['worker_receipt']['device'] == 'cpu'
        assert report['retention'] == 'unknown_not_evaluated'
        rows = {row['path']: row for row in output['rows']}
        assert set(rows) == {'matching.py', 'mismatching.py', 'over512.py'}
        matched = rows['matching.py']['candidate']
        mismatched = rows['mismatching.py']['candidate']
        assert matched['source_candidate_match'] is True
        assert matched['prediction']['candidate_ir']['document']['operator'] == 'numpy.argmax'
        assert matched['prediction']['candidate_ir']['document']['operand_ids'][-1] == 'expr:parameter:0'
        assert mismatched['source_candidate_match'] is False
        assert mismatched['status'] == 'fail_open_source_mismatch'
        assert mismatched['candidate_replaced'] is False
        assert mismatched['continue_planning'] is True
        assert mismatched['matched_projection_refs'] is None
        assert mismatched['prediction']['candidate_ir'] is not None
        families = {row['family'] for row in matched['matched_projection_refs']['artifacts']}
        assert {'program', 'first_order', 'higher_order'} <= families
        assert matched['matched_projection_refs']['pending_obligation_ids']
        oversized = rows['over512.py']
        assert oversized['tokens'] > 512 and oversized['candidate'] is None
        assert oversized['status'] == 'deferred_gte_token_limit'
        prepared = {row['path']: row for row in report['preparation']['files']}
        assert prepared['unsupported.py']['disposition'] == 'unsupported_profile'
        assert prepared['unsupported.py']['reason']
        assert report['coverage']['source_aligned_fragments'] == 1
        assert report['coverage']['source_mismatches'] == 1
        assert report['coverage']['token_deferred'] == 1
        receipt = output['embedding_receipt']
        assert receipt['execution']['kind'] == 'native' and receipt['execution']['truncation'] is False
        assert receipt['status_counts'] == dict(embedded=2, token_limit_exceeded=1, missing_input=0)
        for row in (rows['matching.py'], rows['mismatching.py']):
            candidate = row['candidate']
            assert candidate['source_sha256'] == sha(SOURCES[row['path']])
            assert candidate['model_inference_executed'] is True
            assert candidate['candidate_replaced'] is False
            assert candidate['proof_authority'] is candidate['source_semantics_verified'] is False
            assert candidate['source_obligations_discharged'] == 0
        for value in (result, report):
            assert all(type(value[key]) is type(expected) and value[key] == expected
                for key, expected in owner.FALSE.items())
        assert all((repo / name).read_bytes() == source for name, source in SOURCES.items())
        # Close/open persisted owner registry before replay and prohibit every
        # expensive native seam. Exact report equality excludes regeneration.
        owner._worker = forbidden
        owner.embeddings.produce_native_embedding_receipt = forbidden
        owner.profile.NumericalSelectionRuntime384.infer = forbidden
        registry.close()
        registry = AutoencoderRegistry(root / 'models.duckdb', root / 'models')
        before = time.monotonic()
        replay = owner.infer_numerical_selection(index, repo, registry=registry, **options, **assets)
        assert replay['native_worker_executed'] is replay['inference_executed'] is False
        assert replay['artifact'] == result['artifact'] and replay['report'] == result['report']
        assert owner.validate_numerical_selection(index, repo, result, registry=registry, **assets) == result
        replay_seconds = time.monotonic() - before
        write('cached-replay.json', replay)
        # A cache hit must not permit stale source while the model seams remain
        # disabled. Restoring the exact bytes makes the retained report valid.
        (repo / 'matching.py').write_bytes(SOURCES['matching.py'].replace(b'argmax', b'argmin'))
        try:
            owner.validate_numerical_selection(index, repo, result, registry=registry, **assets)
        except StaleCodebaseError:
            stale_source_refused = True
        else:
            raise AssertionError('cached validation accepted changed original source')
        finally:
            (repo / 'matching.py').write_bytes(SOURCES['matching.py'])
        assert owner.validate_numerical_selection(index, repo, result, registry=registry, **assets) == result
        registry.close()
        cx.close()
        packages = {name: importlib.metadata.version(name) for name in (
            'torch', 'numpy', 'sentence-transformers', 'transformers', 'tokenizers', 'safetensors', 'duckdb')}
        observation = dict(schema='numerical-container-inference-qualification@1', passed=True,
            python=sys.version, packages=packages, torch_runtime=torch.__version__, device='cpu',
            checkpoint_file_sha256=assets['checkpoint_sha256'], producer_pins=owner.pins(),
            source_hashes={name: sha(source) for name, source in SOURCES.items()}, coverage=report['coverage'],
            projection_families=sorted(families), model_loads=output['model_loads'],
            inference_executed=True, cold_cache_replay_inference_executed=False,
            cold_cache_replay_exact=True, over512_tokens=oversized['tokens'],
            stale_source_refused=stale_source_refused, restored_source_validated=True,
            inference_seconds=inference_seconds, replay_validation_seconds=replay_seconds,
            total_seconds=time.monotonic() - started, provider_calls=0, training_steps=0,
            proof_authority=False, source_obligations_discharged=0)
        write('observation.json', observation)
        print(json.dumps(observation, sort_keys=True))


if __name__ == '__main__':
    main()
