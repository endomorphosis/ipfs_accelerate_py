"""Actual native DuckDB/DuckLake persistence and current-code AE hydration."""
import hashlib
import json

import pytest

from test.api.test_codebase_autoencoder import inputs  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime.codebase_autoencoder import train_codebase_autoencoder
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder_index as index


@pytest.fixture
def catalog(inputs, tmp_path):
    learner = train_codebase_autoencoder(**inputs)
    request = dict(repository=inputs['repository'], learner=learner, output=tmp_path / 'code-index')
    receipt = index.build_codebase_autoencoder_index(**request)
    return request, receipt


def test_real_native_world_metadata_and_ducklake_hydration(catalog):
    import duckdb
    request, receipt = catalog
    assert receipt['hydrated'] is True
    assert receipt['status'] == 'hydrated'
    assert receipt['hydration']['catalog_count'] == receipt['hydration']['linked_count'] == 4
    assert receipt['metadata_ducklake']['stored_catalogs'] == receipt['metadata_ducklake']['stored_links'] == 4
    assert receipt['world_ducklake']['stored_records'] == 1
    checked = index.validate_codebase_autoencoder_index(repository=request['repository'],
        learner=request['learner'], expected_receipt=receipt)
    assert checked['status'] == 'verified' and checked['hydrated'] is True
    assert checked['proof_authority'] is checked['completion_authority'] is checked['formalization_authority'] is False
    # Independently open the real persisted lake, not a mocked projection report.
    connection = duckdb.connect(':memory:')
    try:
        connection.execute('LOAD ducklake')
        path = str(request['output'] / 'metadata-lake' / 'metadata.ducklake').replace("'", "''")
        connection.execute("ATTACH 'ducklake:" + path + "' AS ae_lake (READ_ONLY)")
        assert connection.execute('SELECT count(*), bool_or(completion_authority) FROM ae_lake.catalogs').fetchone() == (4, False)
        assert connection.execute('SELECT count(*) FROM ae_lake.identity_links').fetchone()[0] == 4
    finally:
        connection.close()


@pytest.mark.parametrize('kind', ['source', 'weights', 'database', 'parquet', 'extra_artifact', 'receipt'])
def test_stale_or_tampered_learning_or_catalog_never_hydrates_current(catalog, kind):
    request, receipt = catalog
    if kind == 'source':
        path = request['repository'] / 'authored.py'
    elif kind == 'weights':
        from pathlib import Path
        path = Path(request['learner']['output']) / 'checkpoint.json'
    elif kind == 'database':
        path = request['output'] / 'world.duckdb'
    elif kind == 'parquet':
        path = next(request['output'].rglob('*.parquet'))
    elif kind == 'extra_artifact':
        path = request['output'] / 'extra.txt'
        path.write_text('foreign index artifact')
    else:
        path = request['output'] / 'receipt.json'
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        index.validate_codebase_autoencoder_index(repository=request['repository'], learner=request['learner'], expected_receipt=receipt)


def test_reusing_learner_namespace_or_existing_catalog_refused(catalog):
    from pathlib import Path
    request, receipt = catalog
    with pytest.raises(ValueError, match='namespace'):
        index.build_codebase_autoencoder_index(**request)
    with pytest.raises(ValueError, match='namespace'):
        index.build_codebase_autoencoder_index(**{**request, 'output': Path(request['learner']['output']) / 'nested'})
    assert index.validate_codebase_autoencoder_index(repository=request['repository'], learner=request['learner'], expected_receipt=receipt)['status'] == 'verified'


def test_unavailable_lake_projection_is_not_reported_hydrated(inputs, tmp_path, monkeypatch):
    learner = train_codebase_autoencoder(**inputs)
    monkeypatch.setattr(index.ProgramWorldDatabase, 'project_ducklake', lambda *_args, **_kwargs: {'status': 'unavailable'})
    output = tmp_path / 'code-index'
    with pytest.raises(ValueError, match='projection unavailable'):
        index.build_codebase_autoencoder_index(repository=inputs['repository'], learner=learner, output=output)
    assert not (output / 'receipt.json').exists()


@pytest.mark.parametrize('kind', ['authority', 'hydrated', 'status', 'projection'])
def test_even_repinned_receipt_cannot_forge_authority_or_hydration(catalog, kind):
    request, receipt = catalog
    if kind == 'authority':
        receipt['proof_authority'] = True
    elif kind == 'hydrated':
        receipt['hydrated'] = False
    elif kind == 'status':
        receipt['status'] = 'unavailable'
    else:
        receipt['world_ducklake']['status'] = 'unavailable'
    raw = index._json({key: value for key, value in receipt.items() if key != 'receipt_sha256'})
    path = request['output'] / 'receipt.json'
    path.chmod(0o644)
    path.write_bytes(raw)
    receipt['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match='binding differs'):
        index.validate_codebase_autoencoder_index(repository=request['repository'], learner=request['learner'], expected_receipt=receipt)
