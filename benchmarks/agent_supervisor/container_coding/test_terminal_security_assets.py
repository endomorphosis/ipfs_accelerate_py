"""Real portable weight/export contracts at the benchmark deployment boundary."""
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _security_learning_inputs, run


@pytest.fixture
def fork(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.codebase_autoencoder_transfer import fork_legal_shared_weights

    source = tmp_path / 'legal-source' / 'teacher.json'
    source.parent.mkdir()
    source.write_text(json.dumps({
        'feature_embedding_weights': {'token:header': [0.25, 0.5], 'token:value': [-0.25, 0.75]},
        'legal_heads': {'retained_only_in_source': 'AUTHORED_LEGAL_SOURCE_NOT_FOR_CONTAINER'},
    }))
    raw = source.read_bytes()
    descriptor = fork_legal_shared_weights(source_checkpoint=source,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        output=tmp_path / 'fork' / 'security-code-initializer')
    return descriptor, source, raw


def test_only_portable_exact_weights_and_rebased_descriptor_enter_archive(tmp_path, fork):
    descriptor, source, original = fork
    result = deployment.build_runtime_archive(output=tmp_path / 'bundle',
        security_initializer=descriptor, **_inputs(tmp_path))
    with tarfile.open(tmp_path / 'bundle/runtime.tar.gz') as archive:
        asset_names = [name for name in archive.getnames() if name.startswith('models/')]
        assert sorted(asset_names) == sorted([
            deployment.SECURITY_INITIALIZER_PATH + '/initializer.json',
            deployment.SECURITY_INITIALIZER_PATH + '/manifest.json',
            deployment.SECURITY_INITIALIZER_DESCRIPTOR,
        ])
        rebound = json.load(archive.extractfile(deployment.SECURITY_INITIALIZER_DESCRIPTOR))
        assert rebound == {**descriptor, 'output': deployment.ROOT + '/' + deployment.SECURITY_INITIALIZER_PATH}
        for name in asset_names:
            raw = archive.extractfile(name).read()
            assert b'AUTHORED_LEGAL_SOURCE_NOT_FOR_CONTAINER' not in raw
            assert b'source.checkpoint' not in raw
            entry = next(item for item in result['files'] if item['path'] == name)
            assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    assert source.read_bytes() == original
    assert (Path(descriptor['output']) / 'source.checkpoint').read_bytes() == original
    assert result['security_initializer']['host_source_replayed'] is True
    assert result['security_initializer']['source_checkpoint_included'] is False
    assert result['security_initializer']['source_lineage']['source_checkpoint_sha256'] == descriptor['source_checkpoint_sha256']
    assert result['learned_requirements'] == []
    assert result['security_training_requirements'] == ['numpy==1.26.4']
    assert result['torch_cpu_requirement'] == 'torch==2.13.0+cpu'


@pytest.mark.parametrize('name', ['initializer.json', 'manifest.json', 'source.checkpoint'])
def test_modified_initializer_or_host_snapshot_refused_before_archive(tmp_path, fork, name):
    descriptor, _, _ = fork
    target = Path(descriptor['output']) / name
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises(ValueError):
        deployment.build_runtime_archive(output=tmp_path / 'bundle',
            security_initializer=descriptor, **_inputs(tmp_path))
    assert not (tmp_path / 'bundle').exists()


@pytest.mark.parametrize('name', ['initializer.json', 'manifest.json', 'source.checkpoint'])
def test_symlinked_portable_or_host_evidence_refused(tmp_path, fork, name):
    descriptor, _, _ = fork
    target = Path(descriptor['output']) / name
    moved = tmp_path / ('moved-' + name)
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError):
        deployment.build_runtime_archive(output=tmp_path / 'bundle',
            security_initializer=descriptor, **_inputs(tmp_path))


def test_old_profile_remains_asset_free(tmp_path):
    result = deployment.build_runtime_archive(output=tmp_path / 'bundle', **_inputs(tmp_path))
    assert result['security_initializer'] is result['canonical_cve_training'] is None
    assert result['security_training_requirements'] == [] and result['torch_cpu_requirement'] == ''
    assert len(result['files']) == 7
    assert _security_learning_inputs(security_initializer=None, canonical_cve_export=None,
        canonical_cve_manifest_sha256=None) == {'weight_transfer': None, 'canonical_cve_training': None}


def test_portable_runtime_loads_without_host_legal_snapshot(tmp_path, fork):
    descriptor, _, _ = fork
    portable = tmp_path / 'portable' / 'security-code-initializer'
    portable.mkdir(parents=True)
    for name in ('initializer.json', 'manifest.json'):
        (portable / name).write_bytes((Path(descriptor['output']) / name).read_bytes())
    rebound = {**descriptor, 'output': str(portable)}
    descriptor_path = portable.parent / 'descriptor.json'
    descriptor_path.write_text(json.dumps(rebound))
    result = _security_learning_inputs(security_initializer=descriptor_path,
        canonical_cve_export=None, canonical_cve_manifest_sha256=None)
    assert result == {'weight_transfer': rebound, 'canonical_cve_training': None}
    assert not (portable / 'source.checkpoint').exists()
    assert rebound['runtime_validation_scope'] == 'admitted_initializer_integrity_not_original_legal_state_replay'


def test_runtime_descriptor_symlink_is_refused(tmp_path, fork):
    descriptor, _, _ = fork
    original = tmp_path / 'descriptor.json'
    original.write_text(json.dumps(descriptor))
    link = tmp_path / 'linked-descriptor.json'
    link.symlink_to(original)
    with pytest.raises(ValueError):
        _security_learning_inputs(security_initializer=link,
            canonical_cve_export=None, canonical_cve_manifest_sha256=None)
    with pytest.raises(ValueError):
        deployment._load_initializer_descriptor(link)


@pytest.mark.parametrize('export,pin', [(Path('/missing'), None), (None, 'a' * 64)])
def test_canonical_export_and_independent_pin_are_required_together(tmp_path, export, pin):
    with pytest.raises(ValueError, match='together'):
        deployment.build_runtime_archive(output=tmp_path / 'bundle', canonical_cve_export=export,
            canonical_cve_manifest_sha256=pin, **_inputs(tmp_path))
    with pytest.raises(ValueError, match='together'):
        _security_learning_inputs(security_initializer=None, canonical_cve_export=export,
            canonical_cve_manifest_sha256=pin)


def test_no_index_cli_cannot_silently_accept_training_assets(tmp_path):
    with pytest.raises(ValueError, match='full indexed arm'):
        run(instruction=tmp_path / 'instruction', state=tmp_path / 'state', arm='no-index',
            security_initializer=tmp_path / 'initializer.json')


@pytest.fixture
def canonical_export(tmp_path, monkeypatch):
    from test.api.test_security_cve_canonical_export import _fixture_export

    root, receipt, _calls = _fixture_export(tmp_path, monkeypatch)
    return root, receipt


def test_canonical_export_packages_only_three_derived_files_with_source_refs(tmp_path, canonical_export):
    root, receipt = canonical_export
    (root.parent / 'raw-original-not-an-export.json').write_text('AUTHORED_RAW_ORIGINAL_MUST_STAY_OUTSIDE_CONTAINER')
    result = deployment.build_runtime_archive(output=tmp_path / 'bundle', canonical_cve_export=root,
        canonical_cve_manifest_sha256=receipt['manifest_sha256'], **_inputs(tmp_path))
    with tarfile.open(tmp_path / 'bundle/runtime.tar.gz') as archive:
        names = [name for name in archive.getnames() if name.startswith('training/')]
        assert sorted(names) == sorted(deployment.CANONICAL_CVE_PATH + '/' + name
            for name in ('manifest.json', 'canonical-records.json', 'training-pairs.json'))
        for name in names:
            raw = archive.extractfile(name).read()
            assert b'AUTHORED_RAW_ORIGINAL' not in raw and b'return max' not in raw
            assert hashlib.sha256(raw).hexdigest() == next(item['sha256'] for item in result['files'] if item['path'] == name)
    binding = result['canonical_cve_training']
    assert binding['manifest_sha256'] == receipt['manifest_sha256']
    assert binding['canonical_dataset_cid'] == receipt['canonical_dataset_cid']
    assert binding['training_pair_count'] == 2 and binding['raw_source_included'] is False
    assert binding['source_export_refs']['selected_rows'][0]['native_source_cid_preimage_verified'] is True
    assert _security_learning_inputs(security_initializer=None, canonical_cve_export=root,
        canonical_cve_manifest_sha256=receipt['manifest_sha256']) == {
            'weight_transfer': None,
            'canonical_cve_training': {'output': str(root), 'manifest_sha256': receipt['manifest_sha256']},
        }


def test_initializer_and_canonical_training_keep_independent_pins_in_one_archive(tmp_path, fork, canonical_export):
    descriptor, source, original = fork
    root, receipt = canonical_export
    result = deployment.build_runtime_archive(output=tmp_path / 'bundle',
        security_initializer=descriptor, canonical_cve_export=root,
        canonical_cve_manifest_sha256=receipt['manifest_sha256'], **_inputs(tmp_path))
    assert len(result['files']) == 13
    assert len({item['path'] for item in result['files']}) == 13
    assert result['security_initializer']['descriptor']['manifest_sha256'] == descriptor['manifest_sha256']
    assert result['canonical_cve_training']['manifest_sha256'] == receipt['manifest_sha256']
    assert source.read_bytes() == original
    with tarfile.open(tmp_path / 'bundle/runtime.tar.gz') as archive:
        assert not any(name.endswith('source.checkpoint') for name in archive.getnames())


@pytest.mark.parametrize('name', ['manifest.json', 'canonical-records.json', 'training-pairs.json'])
def test_canonical_export_tampering_refused_before_archive(tmp_path, canonical_export, name):
    root, receipt = canonical_export
    target = root / name
    target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises(ValueError):
        deployment.build_runtime_archive(output=tmp_path / 'bundle', canonical_cve_export=root,
            canonical_cve_manifest_sha256=receipt['manifest_sha256'], **_inputs(tmp_path))
    assert not (tmp_path / 'bundle').exists()


def test_raw_original_inside_export_directory_is_refused(tmp_path, canonical_export):
    root, receipt = canonical_export
    (root / 'raw-original.json').write_text('AUTHORED_RAW_ORIGINAL_MUST_NOT_BECOME_A_PORTABLE_ASSET')
    with pytest.raises(ValueError, match='three public artifacts'):
        deployment.build_runtime_archive(output=tmp_path / 'bundle', canonical_cve_export=root,
            canonical_cve_manifest_sha256=receipt['manifest_sha256'], **_inputs(tmp_path))
    assert not (tmp_path / 'bundle').exists()


@pytest.mark.parametrize('name', ['manifest.json', 'canonical-records.json', 'training-pairs.json'])
def test_canonical_export_symlinks_refused_before_archive(tmp_path, canonical_export, name):
    root, receipt = canonical_export
    target = root / name
    moved = tmp_path / ('moved-' + name)
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError):
        deployment.build_runtime_archive(output=tmp_path / 'bundle', canonical_cve_export=root,
            canonical_cve_manifest_sha256=receipt['manifest_sha256'], **_inputs(tmp_path))


def test_repinned_raw_export_declaration_is_not_portable_training(tmp_path, canonical_export):
    root, receipt = canonical_export
    path = root / 'manifest.json'
    manifest = json.loads(path.read_bytes())
    manifest['raw_bodies_persisted'] = True
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        deployment.build_runtime_archive(output=tmp_path / 'bundle', canonical_cve_export=root,
            canonical_cve_manifest_sha256=hashlib.sha256(raw).hexdigest(), **_inputs(tmp_path))


def test_weight_change_between_native_validation_and_capture_is_rejected(tmp_path, fork, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder_transfer as transfer

    descriptor, _, _ = fork
    actual_validate = transfer.validate_legal_shared_weight_fork
    def change_after_validation(**kwargs):
        result = actual_validate(**kwargs)
        path = Path(descriptor['output']) / 'initializer.json'
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b' ')
        return result
    monkeypatch.setattr(transfer, 'validate_legal_shared_weight_fork', change_after_validation)
    with pytest.raises(ValueError, match='changed after validation'):
        deployment.build_runtime_archive(output=tmp_path / 'bundle',
            security_initializer=descriptor, **_inputs(tmp_path))
