"""Actual Doctor dispatch and the deployed full-arm route without model calls."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shlex
import stat
import subprocess
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import _prepare, _prepare_analysis_guard_fixture, _provers, SOURCE
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import native_quack_qualification as native_owner
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime


def test_real_automatic_dispatch_publishes_only_candidate_handoff(scenario, tmp_path, monkeypatch):
    inputs = _provers(_prepare(scenario, tmp_path))
    monkeypatch.setenv('DOCTOR_COMPOSITION_LEAN', str(inputs['kernel_executable']))
    before = scenario['intent'].get_task(inputs['task_cid'])
    result = dispatch.prepare_terminal_doctor_dispatch(repository=scenario['repository'], state=tmp_path,
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    assert result['status'] == 'candidate_ready' and result['route'] == 'doctor_candidate'
    artifact = Path(result['artifact'])
    assert artifact.parent == scenario['repository'] / '.runtime/doctor-handoffs'
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['sha256']
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    assert all(stat.S_IMODE(path.stat().st_mode) & 0o022 == 0 for path in [artifact.parent, artifact.parent.parent])
    private = json.loads((tmp_path / 'doctor-workflow-result.json').read_text())
    assert raw == Path(private['handoff_path']).read_bytes()
    assert sorted(path.name for path in artifact.parent.iterdir()) == [artifact.name]
    assert private['stages']['proof']['mutation_capable'] and private['transaction']['committed']
    assert scenario['intent'].get_task(inputs['task_cid']) == before
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
    argv = dispatch.implementation_argv(router=Path('/router'), model='unused', reasoning='high', timeout=30,
        semantic_repository=scenario['repository'], doctor=result)
    assert argv == ['/router', '--doctor-candidate-artifact', str(artifact),
        '--doctor-candidate-sha256', result['sha256'], '--doctor-task-cid', inputs['task_cid']]


@pytest.mark.parametrize('kind', ['unsupported', 'missing_prover'])
def test_real_residual_retains_model_route_and_native_successors(scenario, tmp_path, monkeypatch, kind):
    source = 'import os\n' + SOURCE if kind == 'unsupported' else SOURCE
    inputs = _prepare(scenario, tmp_path, text=source)
    monkeypatch.setattr(dispatch, '_installed_provers', lambda: (Path('/unavailable/z3'), Path('/unavailable/lean')))
    result = dispatch.prepare_terminal_doctor_dispatch(repository=scenario['repository'], state=tmp_path,
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    assert result['status'] == 'residual' and result['route'] == 'model_router'
    assert result['residual_successors'] + result['residual_work_proposals'] > 0
    if kind == 'missing_prover':
        assert result['residual_work_proposals'] == 1
    assert not (scenario['repository'] / '.runtime/doctor-handoffs').exists()
    assert not (tmp_path / 'doctor-workflow').exists()
    assert (scenario['repository'] / 'answer.py').read_text() == source
    assert scenario['intent'].get_task(inputs['task_cid'])['status'] == 'ready'
    argv = dispatch.implementation_argv(router=Path('/router'), model='model', reasoning='high', timeout=30,
        semantic_repository=scenario['repository'], doctor=result)
    assert '--model' in argv and '--semantic-repository' in argv
    assert '--doctor-candidate-artifact' not in argv


def test_secret_guard_unavailability_retains_existing_signed_router_advisory(scenario, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import load_doctor_residual_advisory
    inputs = _prepare_analysis_guard_fixture(scenario, tmp_path)
    task = scenario['intent'].get_task(inputs['task_cid'])
    result = dispatch.prepare_terminal_doctor_dispatch(repository=scenario['repository'], state=tmp_path,
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    assert result['status'] == 'residual' and result['route'] == 'model_router'
    assert result['analysis_status'] == 'unavailable' and result['analysis_observation_cid']
    assert result['provider_calls'] == 0 and result['residual_work_proposals'] == 1
    assert not (scenario['repository'] / '.runtime/doctor-handoffs').exists()
    context = result['residual_context']
    advisory, observed = load_doctor_residual_advisory(artifact=Path(context['artifact']),
        expected_sha256=context['sha256'], repository=scenario['repository'],
        task_cid=inputs['task_cid'], prompt=json.dumps({'objective_id': task['task_alias']}))
    assert 'doctor_analysis_secret_screen_refused' in advisory
    assert 'doctor:screened-source-analysis' in advisory
    assert 'docs/example.rst' not in advisory and 'AUTHORED-DUMMY' not in advisory
    assert observed['extra_provider_calls'] == 0 and observed['derived_runtime_admitted'] is False
    argv = dispatch.implementation_argv(router=Path('/router'), model='model', reasoning='high', timeout=30,
        semantic_repository=scenario['repository'], doctor=result)
    assert '--model' in argv and '--doctor-residual-artifact' in argv
    assert argv[argv.index('--doctor-residual-sha256') + 1] == context['sha256']
    assert '--doctor-candidate-artifact' not in argv
    assert scenario['intent'].get_task(inputs['task_cid']) == task
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE


@pytest.mark.parametrize('kind', ['symlink', 'writable', 'digest'])
def test_publication_refuses_unsafe_directory_or_rebound_handoff(tmp_path, kind):
    root = tmp_path / 'repository'
    root.mkdir()
    source = tmp_path / 'private.json'
    handoff = {'repository': str(root), 'provider_calls': 0,
        'completion_authority': False, 'publication_authority': False}
    raw = json.dumps(handoff).encode()
    source.write_bytes(raw)
    result = {'handoff_path': str(source), 'handoff': handoff,
        'handoff_sha256': hashlib.sha256(raw).hexdigest()}
    if kind == 'symlink':
        (root / '.runtime').symlink_to(tmp_path, target_is_directory=True)
    elif kind == 'writable':
        (root / '.runtime').mkdir()
        (root / '.runtime').chmod(0o777)
    else:
        source.write_bytes(raw + b' ')
    with pytest.raises(ValueError):
        dispatch._publish_handoff(repository=root, result=result)
    assert not (tmp_path / 'doctor-handoffs').exists()


@pytest.mark.parametrize('arm,route,learned,transport', [('full', 'doctor_candidate', False, 'supervisor-semantic-router-input@1'),
    ('full', 'model_router', False, 'supervisor-semantic-router-input@1'),
    ('no-index', 'model_router', False, 'supervisor-semantic-router-input@1'),
    ('full', 'model_router', True, 'supervisor-semantic-router-input@1'),
    ('full', 'model_router', False, 'supervisor-semantic-router-input@2')])
def test_driver_selects_before_owner_and_preserves_provider_accounting(tmp_path, monkeypatch, arm, route, learned, transport):
    root = tmp_path / 'deployment'
    state = root / 'state/trial'
    model = root / 'models' / ('a' * 40)
    alias = root / 'models/embedding'
    if learned:
        model.mkdir(parents=True)
        alias.symlink_to(model.name, target_is_directory=True)
    events = []
    task = SimpleNamespace(task_cid='task-cid', task_key='TASK', status='completed', revision=4)
    # This driver fixture authors its context and mocks the policy-selection
    # boundary too; real authenticated retrieval is covered by the native
    # published-empty-task-context tests.
    authored_bundle = {'bound': True}
    expected_bundle = authored_bundle if arm == 'full' else None
    expected_policy = ('local-safetensors-symbols@1' if learned else
        'lexical-tfidf-symbols@1' if arm == 'full' else None)
    expected_artifacts = {'result': '.runtime/terminal-vectors/result.json',
        'manifest': '.runtime/terminal-vectors/model-manifest.json',
        'model_snapshot': str(model)} if learned else None
    monkeypatch.setattr(driver, 'ROOT', root)
    monkeypatch.setattr(driver.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(driver.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(driver.signal, 'setitimer', lambda *_: None)
    monkeypatch.setattr(driver.subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    def prepare(**_):
        state.mkdir(parents=True)
        (state / 'admission.json').write_text('{}')
        events.append('prepare')
        return {'intent_preplanning': {}}
    monkeypatch.setattr(driver.preparation, 'prepare', prepare)
    def initial_context(**_):
        events.append('initial_context')
        return {'planning_context': 'owner-verified'}
    monkeypatch.setattr(driver.preparation, 'initial_context', initial_context)
    def plan(**kwargs):
        events.append('planning')
        assert callable(kwargs['provider_callable'])
        return {'qualified': True}
    monkeypatch.setattr(driver.preparation, 'plan', plan)
    def context(**_):
        events.append('context')
        return {'context_bundle': authored_bundle}
    monkeypatch.setattr(driver.preparation, 'context', context)
    def retrieval_options(*, repository, bundle, task, model_snapshot):
        events.append('retrieval_options')
        assert repository == Path('/app')
        assert bundle is expected_bundle
        assert task is authored_task
        assert model_snapshot == (alias if learned else None)
        return dict(published_retrieval_policy=expected_policy,
                    published_learned_artifacts=expected_artifacts)
    authored_task = task
    monkeypatch.setattr(driver, '_published_retrieval_options', retrieval_options)
    monkeypatch.setattr(driver, 'verify_local_benchmark_admission', lambda *_args, **_kwargs:
        {'graph': SimpleNamespace(tasks=[task]), 'manifest': {'repository_cid': 'repository',
            'sources': {driver.preparation.INSTRUCTION: {'sha256': 'b' * 64}}}})
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction
    monkeypatch.setattr(router_public_instruction, 'prepare_public_instruction_context', lambda **_:
        {'artifact': '/app/.runtime/public-instruction.json', 'sha256': 'c' * 64})
    def doctor(**kwargs):
        events.append('doctor')
        assert kwargs['task_cid'] == task.task_cid
        return {'route': route, 'status': 'candidate_ready' if route == 'doctor_candidate' else 'residual',
            'provider_calls': 0, 'artifact': '/app/.runtime/doctor-handoffs/candidate.json',
            'sha256': 'a' * 64, 'task_cid': task.task_cid}
    monkeypatch.setattr(dispatch, 'prepare_terminal_doctor_dispatch', doctor)
    @contextmanager
    def owner(**_):
        events.append('owner')
        yield SimpleNamespace(server=object(), source=SimpleNamespace(get_task=lambda _: task))
    monkeypatch.setattr(native_owner, 'open_existing_native_owner', owner)
    response = SimpleNamespace(to_dict=lambda: {'status': 'succeeded'})
    def runtime(_state, **kwargs):
        events.append('runtime')
        argv = shlex.split(kwargs['implementation_command'])
        assert ('--doctor-candidate-artifact' in argv) == (route == 'doctor_candidate')
        assert ('--semantic-repository' in argv) == (arm == 'full' and route == 'model_router')
        assert ('--semantic-transport-schema' in argv) == (transport == 'supervisor-semantic-router-input@2')
        if '--semantic-transport-schema' in argv:
            assert argv[argv.index('--semantic-transport-schema') + 1] == transport
        assert kwargs['context_bundle'] is expected_bundle
        assert kwargs['published_retrieval_policy'] == expected_policy
        assert kwargs['published_learned_artifacts'] == expected_artifacts
        return SimpleNamespace(state=state, start=lambda: response, stop=lambda: response,
            observe=lambda: {}, close=lambda: None, profile=object(),
            process=SimpleNamespace(snapshot=lambda _: SimpleNamespace(members=[])))
    monkeypatch.setattr(AdmittedBenchmarkRuntime, 'create', runtime)
    monkeypatch.setattr(driver, '_native_diagnostics', lambda _: {})
    monkeypatch.setattr(driver, '_final_context_audit', lambda *_args, **_kwargs: None)
    result = driver.run(instruction=tmp_path / 'instruction', state=state, arm=arm,
        model_snapshot=alias if learned else None, model_revision=model.name if learned else '',
        semantic_transport_schema=transport)
    assert result['task_completed'], result
    assert result['implementation_route'] == route
    assert result['provider_invocations'] == []
    assert ('unreceipted_provider_attempt' in result) == (route == 'model_router')
    assert events.count('retrieval_options') == 1
    assert events.count('planning') == 1
    assert result.get('semantic_transport_schema', 'supervisor-semantic-router-input@1') == transport
    assert events.index('owner') < events.index('retrieval_options') < events.index('runtime')
    if arm == 'full':
        assert events.index('initial_context') < events.index('context')
        assert events.index('context') < events.index('doctor') < events.index('owner') < events.index('runtime')
    else:
        assert 'initial_context' not in events and 'context' not in events and 'doctor' not in events
