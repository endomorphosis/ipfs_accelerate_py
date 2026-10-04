"""Qualification orchestration controls; Docker execution is explicitly mocked."""
import asyncio
from copy import deepcopy
from pathlib import Path
import json
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deploy
from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualify
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_PROFILE, SOURCE384_ENVIRONMENT
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_transport import selected, build


def observation():
    return dict(schema='terminal-source384-cgroup-observation@1', cgroup_path='/sys/fs/cgroup',
        cpu_max='500000 100000', memory_max=str(12288*1024*1024), detected_cpu_slots=5,
        detected_total_memory_mb=12288, available_memory_mb=7500)


@pytest.mark.parametrize('field,value', [('cpu_max','max 100000'), ('cpu_max','100000 100000'),
    ('cpu_max','500000 0'), ('memory_max','max'), ('memory_max',str(2048*1024*1024)),
    ('detected_cpu_slots',4), ('detected_cpu_slots',True), ('detected_total_memory_mb',6144),
    ('available_memory_mb',12289), ('available_memory_mb',True), ('extra',0)])
def test_cgroup_limits_require_exact_finite_profile(field,value):
    record=observation(); record[field]=value
    with pytest.raises(ValueError): qualify.validate_resource_observation(record,SOURCE384_PROFILE)


def test_exact_cgroup_limits_and_default_options():
    assert qualify.validate_resource_observation(observation(),SOURCE384_PROFILE)==observation()
    assert qualify.resource_options(None)=={}
    assert qualify.resource_options(SOURCE384_PROFILE)==SOURCE384_ENVIRONMENT
    with pytest.raises(ValueError): qualify.resource_options('different')
    with pytest.raises(ValueError): qualify.resource_options('source384-5cpu-8gib@1')
    with pytest.raises(ValueError): qualify.validate_resource_observation(observation(),None)


@pytest.fixture
def docker_boundary(tmp_path,monkeypatch,selected):
    build(tmp_path,selected)
    task=tmp_path/'task'; task.mkdir(); (task/'environment').mkdir(); (task/'tests').mkdir()
    (task/'tests/test.sh').write_text('#!/bin/sh\nexit 0\n')
    (task/'instruction.md').write_text('Inspect and repair public code.')
    (task/'task.toml').write_text('[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=10240\n')
    environment=SimpleNamespace(start=AsyncMock(),stop=AsyncMock())
    calls=[]
    def constructor(**kwargs): calls.append(kwargs); return environment
    import harbor.environments.docker.docker as docker
    monkeypatch.setattr(docker,'DockerEnvironment',constructor)
    deployed=dict(original_inputs=dict(files={'bottle.py':dict(sha256='a'*64)}),qualified=True)
    monkeypatch.setattr(deploy,'deploy_supervisor',AsyncMock(return_value=deployed))
    monkeypatch.setattr(qualify,'observe_resources',AsyncMock(return_value=observation()))
    monkeypatch.setattr(qualify,'qualify_context',AsyncMock(return_value=dict(signed_source_hashes={'bottle.py':'a'*64})))
    return dict(task_dir=task,archive_dir=tmp_path/'bundle',output=tmp_path/'qualification'),calls,environment,deployed


def test_default_qualification_keeps_original_harbor_task_resources(docker_boundary):
    args,calls,env,deployed=docker_boundary
    assert asyncio.run(deploy.qualify_original_container(**args,install_codex=False))==deployed
    assert not set(SOURCE384_ENVIRONMENT).intersection(calls[0])
    assert calls[0]['task_env_config'].cpus==1 and calls[0]['task_env_config'].memory_mb==2048
    env.stop.assert_awaited_once_with(delete=True)
    qualify.qualify_context.assert_not_called(); qualify.observe_resources.assert_not_called()


def test_selected_qualification_enforces_limits_and_complete_source_receipt(docker_boundary):
    args,calls,env,_=docker_boundary
    result=asyncio.run(deploy.qualify_original_container(**args,install_codex=False,
        resource_profile=SOURCE384_PROFILE,source384_context=True))
    assert result['qualified'] and result['original_container_sources_consumed']
    assert {key:calls[0][key] for key in SOURCE384_ENVIRONMENT}==SOURCE384_ENVIRONMENT
    assert qualify.observe_resources.await_count==2
    assert json.loads((args['output']/'qualification.json').read_text())==result
    env.stop.assert_awaited_once_with(delete=True)


@pytest.mark.parametrize('failure',['source_omitted','source_changed','native_refusal'])
def test_failed_qualification_is_retained_and_container_closed(docker_boundary,failure):
    args,_,env,_=docker_boundary
    if failure=='native_refusal': qualify.qualify_context.side_effect=ValueError('native admission refused')
    else: qualify.qualify_context.return_value={'signed_source_hashes':{} if failure=='source_omitted' else {'bottle.py':'b'*64}}
    with pytest.raises(ValueError):
        asyncio.run(deploy.qualify_original_container(**args,resource_profile=SOURCE384_PROFILE,source384_context=True))
    report=json.loads((args['output']/'qualification.json').read_text())
    assert not report['qualified'] and report['provider_calls']==0 and not report['benchmark_result']
    env.stop.assert_awaited_once_with(delete=True)


def test_receipt_persistence_failure_still_closes_container(docker_boundary,monkeypatch):
    args,_,env,_=docker_boundary
    path_type=type(args['output']); original=path_type.write_text
    def failing_write(path,*arguments,**kwargs):
        if path.name=='qualification.json': raise OSError('simulated output volume full')
        return original(path,*arguments,**kwargs)
    monkeypatch.setattr(path_type,'write_text',failing_write)
    with pytest.raises(OSError,match='volume full'):
        asyncio.run(deploy.qualify_original_container(**args,resource_profile=SOURCE384_PROFILE,source384_context=True))
    env.stop.assert_awaited_once_with(delete=True)


@pytest.mark.parametrize('selection',['no_profile','no_assets','bad_switch'])
def test_invalid_selection_is_refused_before_docker(docker_boundary,selection):
    args,calls,_,_=docker_boundary
    options=dict(resource_profile=SOURCE384_PROFILE,source384_context=True)
    if selection=='no_profile': options['resource_profile']=None
    elif selection=='bad_switch': options['source384_context']='yes'
    else:
        path=args['archive_dir']/'manifest.json'; value=json.loads(path.read_text()); value.pop('source384')
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError): asyncio.run(deploy.qualify_original_container(**args,**options))
    assert not calls and not args['output'].exists()


def test_embedded_probes_compile_and_do_not_invoke_provider_or_verifier():
    compile(qualify.CONTEXT_PROBE,'context-probe','exec')
    compile(qualify.RESOURCE_PROBE,'resource-probe','exec')
    assert 'prep.prepare(' in qualify.CONTEXT_PROBE and 'prep.initial_context(' in qualify.CONTEXT_PROBE
    assert "models/source384/config.json" in qualify.CONTEXT_PROBE
    assert "context['nonoverlapping_seconds']" in qualify.CONTEXT_PROBE
    assert 'prep.plan(' not in qualify.CONTEXT_PROBE and 'run_verifier(' not in qualify.CONTEXT_PROBE


@pytest.mark.parametrize('mutation',[None,'checkpoint','config','producer','provider','verifier','refused','export_changed','report_oversize','report_malformed'])
@pytest.mark.parametrize('profile', ['source384-5cpu-12gib@1', 'source384-5cpu-16gib-extended@1'])
def test_probe_receipt_matches_relocated_assets_and_actual_archive_producers(tmp_path,selected,mutation,profile):
    manifest=build(tmp_path,selected)
    task=tmp_path/'task'; task.mkdir(); (task/'instruction.md').write_text('Public input only.')
    output=tmp_path/'output'; output.mkdir()
    producer=dict(consumer='a'*64, config='b'*64, byte_reader='c'*64,
        source_owners={'ipfs_datasets_py.logic.software_contracts.content':'d'*64})
    for key,name in [('consumer','source384_repository_context'),('config','source384_config'),('byte_reader','security_autoencoder_advisor')]:
        manifest['files'].append(dict(path='source/ipfs_accelerate_py/agent_supervisor/runtime/'+name+'.py',sha256=producer[key]))
    manifest['files'].append(dict(path='datasets/ipfs_datasets_py/logic/software_contracts/content.py',sha256='d'*64))
    inference_raw=json.dumps({'report':{'key':{'transport_only':True}}}).encode()
    result=dict(qualified=True,inference_sha256=hashlib.sha256(inference_raw).hexdigest(),native_inference_key={'transport_only':True},provider_calls=0,official_verifier_executed=False,benchmark_result=False,
        checkpoint_sha256=manifest['source384']['config']['checkpoint_sha256'],
        config_sha256=manifest['source384']['config_sha256'],producer=producer)
    if mutation in ('checkpoint','config'): result[mutation+'_sha256']='e'*64
    elif mutation=='producer': result['producer']['source_owners']['ipfs_datasets_py.logic.software_contracts.content']='e'*64
    elif mutation=='provider': result['provider_calls']=1
    elif mutation=='verifier': result['official_verifier_executed']=True
    elif mutation=='refused': result.update(qualified=False,error='native source bound exceeded')
    async def download(source,destination):
        if source==qualify.RESULT_PATH:
            destination.write_bytes(b'x'*(qualify.MAX_RESULT_BYTES+1) if mutation=='report_oversize' else b'{' if mutation=='report_malformed' else json.dumps(result).encode())
            return
        assert source.endswith('/source384-context/inference.json')
        destination.write_bytes(inference_raw if mutation!='export_changed' else b'{}')
    environment=SimpleNamespace(download_file=download,upload_file=AsyncMock(),exec=AsyncMock(return_value=SimpleNamespace(
        stdout='diagnostic output before JSON\n'+json.dumps(result),stderr='native diagnostic',return_code=int(mutation=='refused'))))
    call=qualify.qualify_context(environment,task_dir=task,output=output,manifest=manifest,profile=profile)
    if mutation:
        with pytest.raises(ValueError): asyncio.run(call)
    else: assert asyncio.run(call)==result
    if mutation not in {'report_oversize','report_malformed'}:
        assert json.loads((output/'source384-context.json').read_text())==result
    assert (output/'public-instruction.md').read_text()=='Public input only.'
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import execution_budget, admission_environment
    assert environment.exec.call_args.kwargs['timeout_sec']==execution_budget(profile)['qualification_exec_seconds']
    env=environment.exec.call_args.kwargs['env']
    assert env['IPFS_SUPERVISOR_BENCHMARK_PROFILE']==profile
    for key,value in admission_environment(profile).items(): assert env[key]==value
    assert environment.exec.call_args.kwargs['user']=='supervisor'


@pytest.mark.parametrize('selection', ['no_source384', 'ordinary_profile', 'explicit'])
def test_warm_recovery_selection_at_canonical_caller(docker_boundary, selection):
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import EXTENDED_SOURCE384_PROFILE
    from benchmarks.agent_supervisor.container_coding.terminal_source384_warm_recovery import POLICY
    args, calls, env, _ = docker_boundary
    options = dict(resource_profile=EXTENDED_SOURCE384_PROFILE, source384_context=True,
        source384_warm_recovery=POLICY)
    if selection == 'no_source384': options['source384_context'] = False
    elif selection == 'ordinary_profile': options['resource_profile'] = SOURCE384_PROFILE
    else:
        record = observation()
        record.update(memory_max=str(16384*1024*1024), detected_total_memory_mb=16384)
        qualify.observe_resources.return_value = record
    if selection == 'explicit':
        asyncio.run(deploy.qualify_original_container(**args, **options))
        assert qualify.qualify_context.call_args.kwargs['source384_warm_recovery'] == POLICY
        env.stop.assert_awaited_once_with(delete=True)
    else:
        with pytest.raises(ValueError):
            asyncio.run(deploy.qualify_original_container(**args, **options))
        assert not calls and not args['output'].exists()


@pytest.mark.parametrize('mutation', [None, 'policy', 'producer', 'status', 'missing'])
def test_canonical_probe_checks_selected_warm_policy_and_producer(tmp_path, selected, mutation):
    from benchmarks.agent_supervisor.container_coding import terminal_source384_warm_recovery as warm
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import EXTENDED_SOURCE384_PROFILE
    manifest = build(tmp_path, selected)
    task = tmp_path/'task'; task.mkdir(); (task/'instruction.md').write_text('Public input only.')
    output = tmp_path/'output'; output.mkdir()
    digest = hashlib.sha256(Path(warm.__file__).read_bytes()).hexdigest()
    manifest['files'].append(dict(path='source/benchmarks/agent_supervisor/container_coding/terminal_source384_warm_recovery.py', sha256=digest))
    receipt = dict(schema=warm.SCHEMA, policy=warm.POLICY, resource_profile=EXTENDED_SOURCE384_PROFILE,
        policy_source_sha256=digest, selected_seconds=180., effective_seconds=180., max_attempts=2,
        per_attempt_max_seconds=90., backoff_max_seconds=5., backoff_seconds=0., backoff_requested_seconds=0.,
        attempts=[dict(attempt=1, timeout_seconds=90., elapsed_seconds=1., status='validated')],
        status='validated', elapsed_seconds=1., timing_scope='complete_validation_calls_and_explicit_backoff',
        admission_execution_split_measured=False, inference_replayed=False)
    if mutation == 'policy': receipt['policy'] = None
    elif mutation == 'producer': receipt['policy_source_sha256'] = '0'*64
    elif mutation == 'status': receipt['status'] = 'failed'
    result = dict(qualified=True, warm_observation=receipt)
    if mutation == 'missing': result.pop('warm_observation')
    async def download(source, destination):
        assert source == qualify.RESULT_PATH
        destination.write_text(json.dumps(result))
    environment = SimpleNamespace(download_file=download, upload_file=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(stdout='', stderr='', return_code=0)))
    # A valid warm receipt reaches the independent identity check; malformed warm
    # receipts must be rejected before that later check, not accepted as success.
    with pytest.raises(ValueError, match='identity or scope differs' if mutation is None else
            'warm|Source384|producer'):
        asyncio.run(qualify.qualify_context(environment, task_dir=task, output=output,
            manifest=manifest, profile=EXTENDED_SOURCE384_PROFILE, source384_warm_recovery=warm.POLICY))
    assert environment.exec.call_args.kwargs['env'][warm.ENVIRONMENT_KEY] == warm.POLICY
