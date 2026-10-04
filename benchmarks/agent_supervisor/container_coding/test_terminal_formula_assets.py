"""Frozen production decoder archive boundary and full-arm runtime forwarding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import pytest

from tests.unit.logic.formalization.autoencoder.test_security_formula_decoder import formula_checkpoint  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_frozen_security import checkpoint, teacher, fork, joint_inputs  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import security_asset_arguments
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _security_runtime_inputs
from ipfs_datasets_py.logic.formalization.autoencoder.security import security_formula_decoder as decoder_api

PROTOCOL = {'review_ref': 'authored:explicit-http-header-contract', 'callback_parameter': 'start_response'}


def _arguments(tmp_path, checkpoint, formula_checkpoint):
    return {'output': tmp_path / 'runtime-archive', **_inputs(tmp_path),
        'security_checkpoint': Path(checkpoint['output']),
        'security_checkpoint_manifest_sha256': checkpoint['manifest_sha256'],
        'formula_decoder': formula_checkpoint, 'header_protocol': dict(PROTOCOL)}


def test_real_decoder_package_relocates_and_infers_offline_after_archive(tmp_path, checkpoint, formula_checkpoint, monkeypatch):
    args = _arguments(tmp_path, checkpoint, formula_checkpoint)
    built = deployment.build_runtime_archive(**args)
    binding = built['formula_decoder']
    assert binding['descriptor'] == {**formula_checkpoint, 'output': deployment.ROOT + '/' + deployment.SECURITY_FORMULA_PATH}
    assert built['torch_cpu_requirement'] == 'torch==2.13.0+cpu'
    assert built['security_training_requirements'] == []
    assert built['security_inference_requirements'] == ['numpy==1.26.4']
    assert binding['runtime_training_steps'] == binding['runtime_download_calls'] == 0
    offline = tmp_path / 'offline-decoder'; offline.mkdir()
    with tarfile.open(args['output'] / 'runtime.tar.gz') as archive:
        names = [name for name in archive.getnames() if name.startswith(deployment.SECURITY_FORMULA_PATH + '/')]
        assert {Path(name).name for name in names} == decoder_api.FILES
        for name in names:
            raw = archive.extractfile(name).read()
            (offline / Path(name).name).write_bytes(raw)
            assert raw == Path(formula_checkpoint['output'], Path(name).name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == next(x['sha256'] for x in built['files'] if x['path'] == name)
            assert b'def ' not in raw  # No source/teacher bodies enter the task runtime package.
        relocated = json.load(archive.extractfile(deployment.SECURITY_FORMULA_DESCRIPTOR))
        assert relocated == binding['descriptor']
        protocol_raw = archive.extractfile(deployment.SECURITY_HEADER_PROTOCOL).read()
        assert hashlib.sha256(protocol_raw).hexdigest() == built['header_protocol']['sha256']
    selected = {**relocated, 'output': str(offline)}
    decoder_api.load_security_formula_decoder(selected)
    monkeypatch.setattr(decoder_api, 'train_security_formula_decoder', lambda **kw: pytest.fail('offline inference retrained'))
    result = decoder_api.decode_security_formula(source_bytes=b'def bounded(value):\n    return (value + 7) * (value - 4)\n', source_path='authored.py', checkpoint=selected)
    assert result['status'] == 'accepted' and result['learned_formula_count'] == 1
    assert result['provider_calls'] == result['download_calls'] == result['training_steps'] == 0
    assert result['proof_authority'] is False
    descriptor_path = tmp_path / 'relocated-decoder.json'; descriptor_path.write_text(json.dumps(selected))
    protocol_path = tmp_path / 'protocol.json'; protocol_path.write_bytes(protocol_raw)
    runtime = _security_runtime_inputs(security_checkpoint=Path(checkpoint['output']),
        security_checkpoint_manifest_sha256=checkpoint['manifest_sha256'], security_initializer=None,
        canonical_cve_export=None, canonical_cve_manifest_sha256=None,
        formula_decoder_descriptor=descriptor_path, header_protocol_descriptor=protocol_path)
    assert runtime['formula_decoder'] == selected and runtime['header_protocol'] == PROTOCOL
    assert runtime['train_autoencoder'] is False
    argv, observation = security_asset_arguments(built, 'full')
    assert argv[-4:] == ['--formula-decoder-descriptor', deployment.ROOT + '/' + deployment.SECURITY_FORMULA_DESCRIPTOR,
        '--header-protocol-descriptor', deployment.ROOT + '/' + deployment.SECURITY_HEADER_PROTOCOL]
    assert observation['formula_decoder']['weights_sha256'] == formula_checkpoint['weights_sha256']
    assert observation['header_protocol']['security_specification_inferred'] is False
    assert security_asset_arguments(built, 'no-index') == ([], {})


@pytest.mark.parametrize('change', ['without_checkpoint', 'protocol_without_decoder', 'unknown_protocol', 'training_mix'])
def test_invalid_formula_selection_refused_before_archive(tmp_path, checkpoint, formula_checkpoint, change):
    args = _arguments(tmp_path, checkpoint, formula_checkpoint)
    if change == 'without_checkpoint':
        args['security_checkpoint'] = None; args['security_checkpoint_manifest_sha256'] = None
    elif change == 'protocol_without_decoder': args['formula_decoder'] = None
    elif change == 'unknown_protocol': args['header_protocol']['proof_authority'] = True
    else: args['security_initializer'] = {'output': '/unselected-legal-state'}
    with pytest.raises(ValueError): deployment.build_runtime_archive(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('name', ['weights.json', 'manifest.json', 'config.json', 'training.json', 'unexpected.py'])
def test_package_drift_or_executable_extras_refuse_archive(tmp_path, checkpoint, formula_checkpoint, name):
    clone = tmp_path / 'changed-model'; shutil.copytree(formula_checkpoint['output'], clone)
    descriptor = {**formula_checkpoint, 'output': str(clone)}
    path = clone / name
    if path.exists(): path.chmod(0o644)  # Damage only the independently copied fixture.
    path.write_bytes((path.read_bytes() if path.exists() else b'') + b' ')
    args = _arguments(tmp_path, checkpoint, descriptor)
    with pytest.raises(ValueError): deployment.build_runtime_archive(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('change', ['decoder_path', 'descriptor_path', 'protocol_path', 'protocol_sha', 'source_training', 'mode'])
def test_harbor_refuses_retargeted_formula_or_protocol_bindings(tmp_path, checkpoint, formula_checkpoint, change):
    built = deployment.build_runtime_archive(**_arguments(tmp_path, checkpoint, formula_checkpoint))
    bad = deepcopy(built)
    if change == 'decoder_path': bad['formula_decoder']['descriptor']['output'] = '/unrelated/weights'
    elif change == 'descriptor_path': bad['formula_decoder']['descriptor_path'] = '../unrelated.json'
    elif change == 'protocol_path': bad['header_protocol']['path'] = '../unrelated.json'
    elif change == 'protocol_sha': bad['header_protocol']['sha256'] = '0' * 64
    elif change == 'source_training': bad['formula_decoder']['source_training_data_included'] = True
    else: bad['formula_decoder']['mode'] = 'training'
    with pytest.raises(ValueError): security_asset_arguments(bad, 'full')


def test_prior_asset_free_profile_keeps_manifest_and_argument_defaults(tmp_path):
    built = deployment.build_runtime_archive(output=tmp_path / 'old-profile', **_inputs(tmp_path))
    assert 'formula_decoder' not in built and 'header_protocol' not in built
    assert built['torch_cpu_requirement'] == ''
    assert security_asset_arguments(built, 'full') == ([], {})


def test_bundle_cli_reads_explicit_formula_and_protocol_descriptors(tmp_path, checkpoint, formula_checkpoint, monkeypatch, capsys):
    inputs = _inputs(tmp_path)
    descriptor = tmp_path / 'formula-selection.json'; descriptor.write_text(json.dumps(formula_checkpoint))
    protocol = tmp_path / 'protocol-selection.json'; protocol.write_text(json.dumps(PROTOCOL))
    output = tmp_path / 'cli-bundle'
    argv = ['terminal_deployment', 'bundle', '--output', str(output)]
    for key, value in inputs.items(): argv += ['--' + key.replace('_', '-'), str(value)]
    argv += ['--security-checkpoint', checkpoint['output'], '--security-checkpoint-manifest-sha256', checkpoint['manifest_sha256'],
        '--formula-decoder-descriptor', str(descriptor), '--header-protocol-descriptor', str(protocol)]
    monkeypatch.setattr(sys, 'argv', argv)
    deployment.main()
    status = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / 'manifest.json').read_bytes())
    assert status['archive_sha256'] == manifest['archive_sha256']
    assert manifest['formula_decoder']['manifest_sha256'] == formula_checkpoint['manifest_sha256']
    assert manifest['header_protocol']['protocol'] == PROTOCOL
