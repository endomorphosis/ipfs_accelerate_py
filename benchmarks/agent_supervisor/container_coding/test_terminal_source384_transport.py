"""Source384 transport boundaries; tiny model bytes are explicit test doubles.

These tests do not measure inference, source coverage or a benchmark score.
"""
import asyncio
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as baseline
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    SOURCE384_PROFILE, SOURCE384_ENVIRONMENT, apply_resource_profile,
)
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import (
    FullSupervisorAgent, source384_asset_arguments,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import observation
from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls
from ipfs_accelerate_py.agent_supervisor.runtime import source384_config as selection
from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384_v2 as compatibility
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding


@pytest.fixture
def selected(tmp_path, monkeypatch):
    snapshot = tmp_path / 'models--thenlper--gte-small' / 'snapshots' / embedding.PINNED_REVISION
    (snapshot/'1_Pooling').mkdir(parents=True)
    data = {'config.json': b'{"transport_only":true}', 'model.safetensors': b'fake-transport-weights',
            '1_Pooling/config.json': b'{"transport_pooling":true}'}
    for name, raw in data.items():
        (snapshot / name).write_bytes(raw)
    pins = {name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in data.items()}
    monkeypatch.setattr(embedding, '_PINNED_ASSETS', pins)
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text('{"domain_id":"security_ir","transport_only":true}')
    calls = []
    def native_load(path, *, expected_sha256):
        calls.append((str(path), expected_sha256))
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_sha256
        if json.loads(Path(path).read_text())['domain_id'] != 'security_ir':
            raise ValueError('Security structured checkpoint required')
    monkeypatch.setattr(compatibility, 'load_source_program_decoder_384_v2', native_load)
    config = dict(schema=selection.SCHEMA, mode='pinned_parent', checkpoint_path=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        embedding_snapshot=str(snapshot), embedding_revision=embedding.PINNED_REVISION,
        embedding_assets=[dict(name=name, bytes=size, sha256=sha) for name,(size,sha) in sorted(pins.items())],
        training_steps=0, download_calls=0)
    path = tmp_path / 'selected.json'
    path.write_text(json.dumps(config))
    return config, path, calls


def build(tmp_path, selected):
    return deployment.build_runtime_archive(output=tmp_path/'bundle', source384_config=selected[0], **_inputs(tmp_path))


def test_offline_selection_uses_native_compatibility_and_relocates_exact_assets(tmp_path, selected):
    assert selection.load_source384_config(selected[1]) == selected[0]
    assert selected[2] == [(selected[0]['checkpoint_path'], selected[0]['checkpoint_sha256'])]
    manifest = build(tmp_path, selected)
    binding = deployment.validate_source384_binding(manifest)
    assert manifest['learned_requirements'] == []
    assert manifest['source384_requirements'] == list(deployment.LEARNED_REQUIREMENTS)
    assert manifest['torch_cpu_requirement'] and manifest['task_inputs_in_archive'] is False
    assert binding['source_training_data_included'] is False
    assert binding['config']['checkpoint_path'] == deployment.ROOT + '/' + deployment.SOURCE384_PATH
    deployment.verify_source384_archive(tmp_path/'bundle/runtime.tar.gz', manifest)
    with tarfile.open(tmp_path/'bundle/runtime.tar.gz') as bundle:
        assert json.load(bundle.extractfile(deployment.SOURCE384_CONFIG)) == binding['config']
        assert all(member.isfile() for member in bundle)
    args, observed = source384_asset_arguments(manifest, 'full', SOURCE384_PROFILE)
    assert args == ['--source384-config', deployment.ROOT + '/' + deployment.SOURCE384_CONFIG]
    assert observed['enabled'] and not observed['execution_authority']
    args, observed = source384_asset_arguments(manifest, 'no-index', SOURCE384_PROFILE)
    assert args == [] and not observed['enabled'] and observed['disabled_reason'] == 'no_index_ablation'
    with pytest.raises(ValueError, match='common'): source384_asset_arguments(manifest, 'full')


@pytest.mark.parametrize('mutation', ['mode','training','boolean_training','downloads','extra','revision','assets',
    'changed_checkpoint','wrong_domain','missing_snapshot','relative_checkpoint','checkpoint_symlink'])
def test_invalid_selection_never_creates_an_archive(tmp_path, selected, mutation):
    config = deepcopy(selected[0])
    if mutation == 'mode': config['mode'] = 'required_training'
    elif mutation == 'training': config['training_steps'] = 1
    elif mutation == 'boolean_training': config['training_steps'] = False
    elif mutation == 'downloads': config['download_calls'] = 1
    elif mutation == 'extra': config['fallback'] = True
    elif mutation == 'revision': config['embedding_revision'] = 'f'*40
    elif mutation == 'assets': config['embedding_assets'][0]['sha256'] = 'f'*64
    elif mutation == 'changed_checkpoint': Path(config['checkpoint_path']).write_text('{}')
    elif mutation == 'wrong_domain':
        path = Path(config['checkpoint_path']); path.write_text('{"domain_id":"legal_ir"}')
        config['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == 'missing_snapshot': config['embedding_snapshot'] += '-missing'
    elif mutation == 'relative_checkpoint': config['checkpoint_path'] = 'checkpoint.json'
    else:
        path = tmp_path/'link.json'; path.symlink_to(config['checkpoint_path']); config['checkpoint_path'] = str(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        deployment.build_runtime_archive(output=tmp_path/'bundle', source384_config=config, **_inputs(tmp_path))
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('kind', ['duplicate','nonfinite','symlink','fifo','oversize'])
def test_config_reader_rejects_unbounded_or_ambiguous_inputs(tmp_path, selected, kind):
    path = selected[1]
    if kind == 'duplicate': path.write_text('{"schema":"a","schema":"b"}')
    elif kind == 'nonfinite': path.write_text('{"value":NaN}')
    elif kind == 'symlink': path.unlink(); path.symlink_to(selected[0]['checkpoint_path'])
    elif kind == 'fifo': path.unlink(); os.mkfifo(path)
    else: path.write_bytes(b' '* (selection.MAX_CONFIG_BYTES+1))
    with pytest.raises(ValueError): selection.load_source384_config(path)


@pytest.mark.parametrize('mutation', ['config_path','config_hash','checkpoint_hash','extra_member','missing_member',
                                     'duplicate_member','oversize_checkpoint','training_mix'])
def test_modified_manifest_is_refused_before_harbor_invocation(tmp_path, selected, mutation):
    manifest = build(tmp_path, selected); binding = manifest['source384']
    if mutation == 'config_path': binding['config_path'] = '../../bad'
    elif mutation == 'config_hash': binding['config_sha256'] = 'f'*64
    elif mutation == 'checkpoint_hash': binding['config']['checkpoint_sha256'] = 'f'*64
    elif mutation == 'extra_member': manifest['files'].append(dict(path='models/source384/extra',sha256='f'*64,bytes=1))
    elif mutation == 'missing_member': manifest['files'] = [x for x in manifest['files'] if x['path'] != deployment.SOURCE384_PATH]
    elif mutation == 'duplicate_member': manifest['files'].append(deepcopy(manifest['files'][-1]))
    elif mutation == 'training_mix': manifest['canonical_cve_training'] = {'unexpected': True}
    else: next(x for x in manifest['files'] if x['path'] == deployment.SOURCE384_PATH)['bytes'] = 2**40
    with pytest.raises(ValueError): source384_asset_arguments(manifest, 'full', SOURCE384_PROFILE)


def test_tampered_archive_is_refused_before_container_access_even_with_new_outer_hash(tmp_path, selected):
    manifest = build(tmp_path, selected); path = tmp_path/'bundle/runtime.tar.gz'
    with tarfile.open(path) as bundle: rows = [(m,bundle.extractfile(m).read()) for m in bundle]
    with tarfile.open(path,'w:gz') as bundle:
        for member,raw in rows:
            if member.name == deployment.SOURCE384_PATH: raw = b'x'*len(raw)
            bundle.addfile(member,io.BytesIO(raw))
    manifest['archive_sha256'] = deployment._sha(path)
    (tmp_path/'bundle/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='digest'):
        asyncio.run(deployment.deploy_supervisor(None,archive_dir=tmp_path/'bundle',output=tmp_path/'deployment'))


@pytest.mark.timeout(10)
@pytest.mark.parametrize('replacement', ['fifo','changed_bytes'])
def test_checkpoint_change_during_asset_capture_is_rechecked_without_blocking(tmp_path, selected, monkeypatch, replacement):
    original=embedding._snapshot_assets
    calls=0
    def observed(path):
        nonlocal calls
        result=original(path); calls+=1
        if calls==3:
            checkpoint=Path(selected[0]['checkpoint_path'])
            if replacement=='fifo': checkpoint.unlink(); os.mkfifo(checkpoint)
            else: checkpoint.write_bytes(b'{}')
        return result
    monkeypatch.setattr(embedding,'_snapshot_assets',observed)
    with pytest.raises(ValueError): build(tmp_path,selected)
    assert calls==3 and not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('key', ['security_initializer','canonical_cve_export','canonical_cve_manifest_sha256',
    'security_checkpoint','security_checkpoint_manifest_sha256','security_checkpoint_hub_descriptor',
    'security_checkpoint_cache','formula_decoder','header_protocol'])
def test_source384_refuses_legacy_security_selection(tmp_path, selected, key):
    with pytest.raises(ValueError,match='mutually exclusive'):
        deployment.build_runtime_archive(output=tmp_path/'bundle', source384_config=selected[0],
            **{key: {}}, **_inputs(tmp_path))
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('key', ['security_initializer','canonical_cve_training','security_checkpoint',
                                'formula_decoder','header_protocol'])
def test_manifest_rejects_all_legacy_security_bindings_even_empty(tmp_path, selected, key):
    manifest=build(tmp_path,selected);manifest[key]={}
    with pytest.raises(ValueError,match='mutually exclusive'):
        deployment.validate_source384_binding(manifest)


@pytest.mark.parametrize('arm', ['full','no-index'])
def test_actual_harbor_adapter_forwards_only_full_selection(tmp_path, selected, monkeypatch, arm):
    build(tmp_path, selected)
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    monkeypatch.setattr(adapter,'capture_public_inputs',AsyncMock(return_value={}))
    monkeypatch.setattr(adapter,'export_public_outputs',AsyncMock(return_value={}))
    commands=[]
    class Environment:
        async def upload_file(self,*args): pass
        async def exec(self,**kwargs):
            commands.append(shlex.split(kwargs['command']))
            return SimpleNamespace(stdout='',stderr='',return_code=0)
        async def download_file(self,*args): raise FileNotFoundError
    agent = FullSupervisorAgent(logs_dir=tmp_path/'logs',model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path/'bundle'),arm=arm,resource_profile=SOURCE384_PROFILE)
    context = AgentContext()
    asyncio.run(agent.run('Public instruction.',Environment(),context))
    assert ('--source384-config' in commands[0]) == (arm == 'full')
    assert context.metadata['source384_assets']['enabled'] == (arm == 'full')
    assert context.metadata['resource_profile'] == SOURCE384_PROFILE


def test_preparation_rejects_implicit_resources_and_binds_exact_config(tmp_path, selected, monkeypatch):
    manifest=build(tmp_path,selected)
    dataset=tmp_path/'dataset'; task=dataset/benchmark.TASK; task.mkdir(parents=True)
    (task/'instruction.md').write_text('Public instruction.')
    (task/'task.toml').write_text('version="1.0"')
    (task/'environment').mkdir(); (task/'tests').mkdir()
    monkeypatch.setattr(benchmark.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='',stderr=''))
    with pytest.raises(ValueError,match='common'):
        benchmark.prepare(dataset=dataset,output=tmp_path/'implicit',archive=tmp_path/'bundle',arm='full')
    assert not (tmp_path/'implicit').exists()
    result=benchmark.prepare(dataset=dataset,output=tmp_path/'trial',archive=tmp_path/'bundle',arm='full',
        source384_config=selected[1],resource_profile=SOURCE384_PROFILE)
    assert result['source384']==manifest['source384'] and result['source384_enabled']
    assert result['provider_calls']==0 and not result['benchmark_advantage_claimed']
    changed=deepcopy(selected[0]); changed['checkpoint_sha256']='f'*64; selected[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=dataset,output=tmp_path/'wrong',archive=tmp_path/'bundle',arm='no-index',
            source384_config=selected[1],resource_profile=SOURCE384_PROFILE)


def test_harbor_normalized_three_arm_limits_match_and_defaults_stay_original():
    from harbor.models.job.config import JobConfig
    normalized=lambda config: JobConfig.model_validate(config,extra='forbid').model_dump(mode='json')
    original=baseline.config_for(Path('/dataset'),Path('/base'))
    assert original['environment']==dict(type='docker',force_build=True,delete=True)
    native=normalized(baseline.config_for(Path('/dataset'),Path('/base'),resource_profile=SOURCE384_PROFILE))
    for key,value in SOURCE384_ENVIRONMENT.items(): assert native['environment'][key]==value
    for arm in ('full','no-index'):
        other=normalized(benchmark.config_for(Path('/dataset'),Path('/supervisor'),Path('/archive'),arm,
                                             resource_profile=SOURCE384_PROFILE))
        assert compare_controls(observation(native),observation(other))['matches'] is True
        assert compare_controls(observation(normalized(original)),observation(other))['matches'] is False


@pytest.mark.parametrize('field,value', [('override_cpus',4),('override_memory_mb',2048),
    ('cpu_enforcement_policy','request'),('memory_enforcement_policy','auto')])
def test_declared_profile_cannot_conceal_resource_mismatch(field,value):
    config=benchmark.config_for(Path('/dataset'),Path('/supervisor'),Path('/archive'),'full',resource_profile=SOURCE384_PROFILE)
    config['environment'][field]=value
    with pytest.raises(ValueError,match='declared resource'): observation(config)
    with pytest.raises(ValueError,match='conflicts'): apply_resource_profile(config,SOURCE384_PROFILE)
