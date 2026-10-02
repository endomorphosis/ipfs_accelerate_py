"""Closed Intent384 deployment transport; tiny assets are explicit test doubles."""
from copy import deepcopy
import asyncio
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import (
    FullSupervisorAgent, intent_asset_arguments,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding


@pytest.fixture
def selected(tmp_path, monkeypatch):
    snapshot = tmp_path / 'models--thenlper--gte-small' / 'snapshots' / embedding.PINNED_REVISION
    (snapshot / '1_Pooling').mkdir(parents=True)
    data = {'config.json': b'{"test_only":true}', 'model.safetensors': b'transport-only-weights',
            '1_Pooling/config.json': b'{"pooling_fixture":true}'}
    for name, raw in data.items():
        (snapshot / name).write_bytes(raw)
    pins = {name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in data.items()}
    monkeypatch.setattr(embedding, '_PINNED_ASSETS', pins)
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text('{"domain":"intent_ir","transport_fixture_only":true}')
    def native_load(path, *, expected_sha256, expected_domain):
        assert expected_domain == 'intent_ir'
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_sha256
        if json.loads(Path(path).read_text())['domain'] != expected_domain:
            raise ValueError('checkpoint domain mismatch')
    monkeypatch.setattr(structured, 'load_checkpoint', native_load)
    config = dict(schema='supervisor-intent-action-384-config/v1', checkpoint_path=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        embedding_snapshot_path=str(snapshot))
    config_path = tmp_path / 'selected.json'
    config_path.write_text(json.dumps(config))
    return config, config_path


def _build(tmp_path, selected):
    return deployment.build_runtime_archive(output=tmp_path / 'bundle',
        intent_action_384_config=selected[0], **_inputs(tmp_path))


def test_closed_assets_relocate_and_leave_retrieval_disabled(tmp_path, selected):
    manifest = _build(tmp_path, selected)
    bound = deployment.validate_intent_action_384_binding(manifest)
    assert deployment.load_intent_action_384_config(selected[1]) == selected[0]
    assert manifest['learned_requirements'] == []
    assert manifest['model_snapshot_revision'] is None
    assert manifest['intent_action_384_requirements'] == list(deployment.LEARNED_REQUIREMENTS)
    assert manifest['torch_cpu_requirement']
    assert bound['runtime_download_calls'] == bound['runtime_training_steps'] == 0
    assert bound['source_training_data_included'] is False
    assert bound['checkpoint_sha256'] == selected[0]['checkpoint_sha256']
    assert bound['config']['checkpoint_path'] == deployment.ROOT + '/' + deployment.INTENT_ACTION_384_PATH
    deployment.verify_intent_action_384_archive(tmp_path / 'bundle/runtime.tar.gz', manifest)
    args, observed = intent_asset_arguments(manifest)
    assert args == ['--intent-action-384-config', deployment.ROOT + '/' + deployment.INTENT_ACTION_384_CONFIG]
    assert observed['execution_authority'] is False
    assert intent_asset_arguments(manifest, enabled=False)[0] == ['--disable-intent-autoencoder']
    with tarfile.open(tmp_path / 'bundle/runtime.tar.gz') as bundle:
        archived = json.load(bundle.extractfile(deployment.INTENT_ACTION_384_CONFIG))
        assert archived == bound['config']
        assert len([x for x in bundle if x.name.startswith('models/intent-action-384/')]) == 5
        assert all(not member.issym() and not member.islnk() for member in bundle)


@pytest.mark.parametrize('mutation', ['null_snapshot','missing_snapshot','changed_checkpoint','missing_checkpoint','wrong_domain','extra_field','relative_checkpoint'])
def test_invalid_local_selection_is_rejected_before_archive(tmp_path, selected, mutation):
    config = deepcopy(selected[0])
    if mutation == 'null_snapshot': config['embedding_snapshot_path'] = None
    if mutation == 'missing_snapshot': config['embedding_snapshot_path'] += '-missing'
    if mutation == 'changed_checkpoint': Path(config['checkpoint_path']).write_text('{}')
    if mutation == 'missing_checkpoint': Path(config['checkpoint_path']).unlink()
    if mutation == 'wrong_domain':
        path = Path(config['checkpoint_path']); path.write_text('{"domain":"security_ir"}')
        config['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if mutation == 'extra_field': config['download'] = True
    if mutation == 'relative_checkpoint': config['checkpoint_path'] = 'checkpoint.json'
    with pytest.raises((ValueError, FileNotFoundError)):
        deployment.build_runtime_archive(output=tmp_path/'bundle', intent_action_384_config=config, **_inputs(tmp_path))
    assert not (tmp_path/'bundle').exists()


def test_duplicate_config_fields_and_config_symlink_are_rejected(tmp_path, selected):
    selected[1].write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match='duplicate'): deployment.load_intent_action_384_config(selected[1])
    link=tmp_path/'link.json';link.symlink_to(selected[1])
    with pytest.raises(ValueError, match='canonical'): deployment.load_intent_action_384_config(link)


def test_cached_blob_symlinks_are_captured_but_external_targets_refused(tmp_path, selected):
    config=selected[0];snapshot=Path(config['embedding_snapshot_path']);asset=snapshot/'model.safetensors'
    blobs=snapshot.parent.parent/'blobs';blobs.mkdir()
    target=blobs/'weights';asset.rename(target);asset.symlink_to(target)
    assets, _=deployment._intent_action_384_assets(config)
    assert any(raw==target.read_bytes() for _,raw in assets)
    asset.unlink();external=tmp_path/'external';external.write_bytes(target.read_bytes());asset.symlink_to(external)
    with pytest.raises(ValueError, match='inside'): deployment._intent_action_384_assets(config)


@pytest.mark.parametrize('change', ['path','config_path','checkpoint_sha256','embedding_path','embedding_assets','extra_member','missing_member','member_hash','duplicate_member'])
def test_tampered_manifest_selection_rejected(tmp_path, selected, change):
    manifest=_build(tmp_path,selected);bound=manifest['intent_action_384']
    if change in ['path','config_path','embedding_path']:bound[change]='../../escape'
    elif change=='checkpoint_sha256':bound[change]='f'*64
    elif change=='embedding_assets':bound[change][0]['sha256']='f'*64
    elif change=='extra_member':manifest['files'].append(dict(path='models/intent-action-384/extra.json',sha256='f'*64,bytes=1))
    elif change=='duplicate_member':manifest['files'].append(deepcopy(manifest['files'][-1]))
    else:
        row=next(x for x in manifest['files'] if x['path']==deployment.INTENT_ACTION_384_PATH)
        if change=='missing_member':manifest['files'].remove(row)
        else:row['sha256']='f'*64
    with pytest.raises(ValueError):intent_asset_arguments(manifest)


def test_changed_archive_bytes_refused_even_when_outer_hash_is_updated(tmp_path, selected):
    manifest=_build(tmp_path,selected);path=tmp_path/'bundle/runtime.tar.gz'
    with tarfile.open(path) as bundle: rows=[(m,bundle.extractfile(m).read()) for m in bundle]
    with tarfile.open(path,'w:gz') as bundle:
        for member,raw in rows:
            if member.name==deployment.INTENT_ACTION_384_PATH:raw=b'x'*len(raw)
            bundle.addfile(member,io.BytesIO(raw))
    manifest['archive_sha256']=deployment._sha(path)
    with pytest.raises(ValueError,match='digest'):deployment.verify_intent_action_384_archive(path,manifest)


def test_legacy_and_new_selection_cannot_mix(tmp_path, selected):
    sources = _inputs(tmp_path)
    with pytest.raises(ValueError,match='mutually exclusive'):
        deployment.build_runtime_archive(output=tmp_path/'bundle',intent_action_384_config=selected[0],
            intent_checkpoint={}, **sources)
    manifest=deployment.build_runtime_archive(output=tmp_path/'bundle',intent_action_384_config=selected[0], **sources)
    manifest['intent_checkpoint']={}
    with pytest.raises(ValueError,match='mutually exclusive'):intent_asset_arguments(manifest)


@pytest.mark.parametrize('arm',['full','no-index'])
def test_real_harbor_entrypoint_forwards_packaged_config(tmp_path, selected, monkeypatch, arm):
    manifest=_build(tmp_path,selected)
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as agent_module
    monkeypatch.setattr(agent_module,'capture_public_inputs',AsyncMock(return_value={}))
    monkeypatch.setattr(agent_module,'export_public_outputs',AsyncMock(return_value={}))
    commands=[]
    class Environment:
        async def upload_file(self,*args): pass
        async def exec(self,**kwargs):
            commands.append(shlex.split(kwargs['command']))
            return SimpleNamespace(stdout='',stderr='',return_code=0)
        async def download_file(self,*args):raise FileNotFoundError
    agent=FullSupervisorAgent(logs_dir=tmp_path/'logs',model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path/'bundle'),arm=arm)
    context=AgentContext()
    asyncio.run(agent.run('the agent must compute result; requires true; ensures result = old(left) + old(right) and returned.',Environment(),context))
    assert '--intent-action-384-config' in commands[0]
    assert '--model-snapshot' not in commands[0]
    assert context.metadata['intent_model_assets']['sha256']==selected[0]['checkpoint_sha256']


def test_container_cli_forwards_exact_selection_without_model_call(tmp_path,monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    captured={}
    def run(**kwargs):
        captured.update(kwargs)
        return dict(task_completed=True,arm=kwargs['arm'],seconds=0,provider_invocations=[])
    monkeypatch.setattr(driver,'run',run)
    monkeypatch.setattr(sys,'argv',['driver','--instruction',str(tmp_path/'instruction'),'--state',str(tmp_path/'state'),
        '--arm','no-index','--intent-action-384-config',str(tmp_path/'config.json')])
    assert driver.main()==0
    assert captured['intent_action_384_config']==tmp_path/'config.json'


def test_benchmark_preparation_retains_config_binding_and_rejects_wrong_selection(tmp_path,selected,monkeypatch):
    manifest=_build(tmp_path,selected)
    dataset=tmp_path/'dataset';task=dataset/benchmark.TASK;task.mkdir(parents=True)
    (task/'instruction.md').write_text('Public instruction outside scalar grammar.')
    (task/'task.toml').write_text('version = "1.0"')
    monkeypatch.setattr(benchmark.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='',stderr=''))
    result=benchmark.prepare(dataset=dataset,output=tmp_path/'trial',archive=tmp_path/'bundle',arm='full',
        intent_action_384_config=selected[1])
    assert result['provider_calls']==0
    assert result['intent_action_384']==manifest['intent_action_384']
    changed=deepcopy(selected[0]);changed['checkpoint_sha256']='f'*64;selected[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError):benchmark.prepare(dataset=dataset,output=tmp_path/'bad-trial',archive=tmp_path/'bundle',
        arm='no-index',intent_action_384_config=selected[1])
