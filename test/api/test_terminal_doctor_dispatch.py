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
def test_driver_selects_before_owner_and_preserves_provider_accounting(
        tmp_path, monkeypatch, arm, route, learned, transport, controls=None):
    controls = controls or {}
    provider_free = route in {'doctor_candidate', 'doctor_contract_candidate'}
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
        result = {'route': route, 'status': 'candidate_ready' if provider_free else 'residual',
            'provider_calls': 0, 'artifact': '/app/.runtime/doctor-handoffs/candidate.json',
            'sha256': 'a' * 64, 'task_cid': task.task_cid}
        if controls.get('doctor_coding_policy') is not None and not provider_free:
            result['residual_context'] = {'artifact': '/app/.runtime/doctor-residuals/residual.json',
                'sha256': 'd' * 64, 'task_cid': task.task_cid}
        return result
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
        assert ('--doctor-contract-artifact' in argv) == (route == 'doctor_contract_candidate')
        assert ('--semantic-transport-schema' in argv) == (
            transport == 'supervisor-semantic-router-input@2' and not provider_free)
        if '--semantic-transport-schema' in argv:
            assert argv[argv.index('--semantic-transport-schema') + 1] == transport
        for option in controls:
            flag = '--' + option.replace('_', '-')
            assert (flag in argv) is (not provider_free)
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
        semantic_transport_schema=transport, **controls)
    assert result['task_completed'], result
    assert result['implementation_route'] == route
    assert result['provider_invocations'] == []
    # The mocked native runtime emits no materialization or round lifecycle.
    # A selected continuation cannot infer zero coding usage from its route.
    assert ('unreceipted_provider_attempt' in result) == (
        route == 'model_router' or controls.get('candidate_rounds') == 2)
    assert events.count('retrieval_options') == 1
    assert events.count('planning') == 1
    assert result.get('semantic_transport_schema', 'supervisor-semantic-router-input@1') == transport
    if provider_free:
        observation = result['coding_control_application']
        assert observation['schema'] == 'terminal-coding-control-application@1'
        assert observation['status'] == 'not_applicable_provider_free'
        assert observation['implementation_route'] == route
        assert observation['model_coding_dispatched'] is False
        assert observation['token_savings_established'] is observation['completion_authority'] is False
        assert observation['requested'] == {
            'semantic_transport_schema': transport, 'semantic_metadata_view': 'legacy',
            'coding_reply_mode': 'legacy', 'source_view_profile': 'disabled',
            'source_edit_mode': 'disabled', 'source_view_policy': None, **controls}
    else:
        assert 'coding_control_application' not in result
    for option, value in controls.items():
        if option == 'semantic_metadata_view':
            assert result[option]['requested_mode'] == value
        else:
            assert result[option] == value
    assert events.index('owner') < events.index('retrieval_options') < events.index('runtime')
    if arm == 'full':
        assert events.index('initial_context') < events.index('context')
        assert events.index('context') < events.index('doctor') < events.index('owner') < events.index('runtime')
    else:
        assert 'initial_context' not in events and 'context' not in events and 'doctor' not in events


@pytest.mark.parametrize('route', ['doctor_candidate', 'doctor_contract_candidate', 'model_router'])
@pytest.mark.parametrize('transport,controls', [
    ('supervisor-semantic-router-input@1', {
        'source_view_profile': 'python-comments@1', 'source_edit_mode': 'mapped-span@2',
        'semantic_metadata_view': 'common-bindings-instruction@1', 'coding_reply_mode': 'ordinary-completion@1'}),
    ('supervisor-semantic-router-input@2', {'source_view_profile': 'identity@1'}),
])
def test_checked_doctor_precedes_model_only_controls_with_explicit_nonapplication(
        tmp_path, monkeypatch, route, transport, controls):
    test_driver_selects_before_owner_and_preserves_provider_accounting(
        tmp_path, monkeypatch, 'full', route, False, transport, controls=controls)


@pytest.mark.parametrize('route', ['doctor_candidate', 'doctor_contract_candidate'])
@pytest.mark.parametrize('controls', [
    {'source_view_profile': 'unknown'},
    {'source_view_profile': 'disabled', 'source_edit_mode': 'mapped-span@2'},
    {'semantic_metadata_view': 'common-bindings-instruction@1',
     'semantic_transport_schema': 'supervisor-semantic-router-input@2'},
    {'coding_reply_mode': 'unknown'},
])
def test_provider_free_route_does_not_hide_unsupported_model_control_combinations(route, controls):
    candidate = dict(route=route, status='candidate_ready', provider_calls=0,
        artifact='/authored-candidate', sha256='a' * 64, task_cid='authored-task')
    with pytest.raises(ValueError):
        dispatch.implementation_argv(router=Path('/router'), model='unused', reasoning='high',
            timeout=30, semantic_repository=Path('/app'), doctor=candidate, **controls)


def test_compact_controls_keep_real_doctor_publication_validation_and_completion(scenario, tmp_path, monkeypatch):
    """Real proof, route, materialization and typed-owner gates; no daemon START.

    Publication uses the existing deterministic native merge fixture. Its bytes
    come from the actual Doctor materialization, not an authored repair result.
    """
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import TaskSourceConflictError
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import TransactionError
    from test.api.test_local_completion_bridge import native_published_transition, complete
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon

    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail('provider-free Doctor reached a model entrypoint')
    monkeypatch.setattr(llm_router, 'generate_text', forbidden)
    monkeypatch.setattr(llm_router, 'get_llm_provider', forbidden)
    inputs = _provers(_prepare(scenario, tmp_path))
    monkeypatch.setenv('DOCTOR_COMPOSITION_LEAN', str(inputs['kernel_executable']))
    result = dispatch.prepare_terminal_doctor_dispatch(repository=scenario['repository'], state=tmp_path,
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    assert result['status'] == 'candidate_ready' and result['provider_calls'] == 0
    kwargs = dict(router=Path('/authored-unused-router'), model='unused', reasoning='high', timeout=30,
        semantic_repository=scenario['repository'], doctor=result)
    controls = dict(source_view_profile='python-comments@1', source_edit_mode='mapped-span@2',
        semantic_metadata_view='common-bindings-instruction@1', coding_reply_mode='ordinary-completion@1')
    argv = dispatch.implementation_argv(**kwargs, **controls)
    assert argv == dispatch.implementation_argv(**kwargs)
    selected = dict(zip(argv[1::2], argv[2::2]))
    handoff = json.loads(Path(selected['--doctor-candidate-artifact']).read_bytes())
    workspace = tmp_path / 'native-candidate'
    subprocess.run(['git', '-C', str(scenario['repository']), 'worktree', 'add', '--detach',
        str(workspace), handoff['base_commit']], check=True, capture_output=True)
    observed = materialize_doctor_candidate(artifact=Path(selected['--doctor-candidate-artifact']),
        expected_sha256=selected['--doctor-candidate-sha256'], task_cid=selected['--doctor-task-cid'],
        prompt=json.dumps({'objective_id': handoff['task_id']}), workspace=workspace)
    assert observed['status'] == 'candidate_materialized' and observed['provider_calls'] == 0
    assert observed['publication_authority'] is observed['completion_authority'] is False
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
    scenario.update(graph=local.PromptGoalGraph.from_dict(inputs['admission']['graph']),
        manifest=inputs['admission']['manifest'], task_cid=inputs['task_cid'])
    @contextmanager
    def claim_prepared():
        # Doctor preparation already materialized this exact admitted task.
        # Open the real owner and claim it without rematerializing the plan.
        with native_owner.open_existing_native_owner(database=scenario['intent'].database_path,
                checkout=scenario['repository'], state_dir=tmp_path / 'owner',
                repository_id=scenario['manifest']['payload']['repository_cid'],
                execution_routes={handoff['task_id']: GROK_CODEX_EXECUTION_MODE}) as owner:
            daemon = DatabaseImplementationDaemon(database_path=owner.database,
                coordination_path=tmp_path / 'coordination.duckdb', execution_path=tmp_path / 'execution.duckdb',
                authority_mode='quack', task_source_kind='duckdb', owner_session_id='session:doctor-controls',
                process_instance_id=owner.identity.process_birth_id, quack_uri=owner.identity.listen_uri,
                task_source=owner.source, close_task_source=False, state_owner_bootstrap_credentials=owner.credentials,
                strict_task_sharding=True, max_task_attempts=1, lease_ms=60_000, require_real_execution=True).open()
            try:
                attempt = daemon.claim_next()
                assert attempt is not None and attempt.task_cid == inputs['task_cid']
                yield owner, attempt
            finally:
                daemon.close()
    with claim_prepared() as (owner, attempt):
        task = owner.source.get_task(attempt.task_cid)
        failed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision)
        assert failed['passed'] is False
        with pytest.raises((TaskSourceConflictError, TransactionError)):
            complete(owner, attempt, failed['results'][0]['evidence_digest'])
        transition = native_published_transition(scenario, tmp_path, owner, attempt,
            modified_outputs={'answer.py': (workspace / 'answer.py').read_text()})
        with pytest.raises((TaskSourceConflictError, TransactionError)):
            complete(owner, attempt, local.content_identity(transition))
        passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed['passed'] is True
        complete(owner, attempt, passed['results'][0]['evidence_digest'])
        assert owner.source.get_task(task.task_cid).status == 'completed'
        assert (scenario['repository'] / 'answer.py').read_bytes() == (workspace / 'answer.py').read_bytes()
    assert calls == []
