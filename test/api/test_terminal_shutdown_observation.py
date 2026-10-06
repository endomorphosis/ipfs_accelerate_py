"""STOP failures survive later close failures without changing cleanup authority."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_shutdown_observation as observation
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime import IsolatedBenchmarkRuntime


def _close_guard():
    IsolatedBenchmarkRuntime._require_no_live_launched_children(
        SimpleNamespace(_children=[SimpleNamespace(poll=lambda:None)]))


def _exercise(tmp_path, monkeypatch, *, fail_phase=None, failure=None, close_failure=None,
              observe_failure=False, cancellation=None):
    from test.api.test_terminal_doctor_dispatch import test_driver_selects_before_owner_and_preserves_provider_accounting as exercise
    run = driver.run
    calls, captured = [], []

    def fail():
        if callable(failure):
            failure()
        raise failure

    def capture(**kwargs):
        create = AdmittedBenchmarkRuntime.create
        def injected(path, **options):
            runtime = create(path, **options)
            response = SimpleNamespace(to_dict=lambda:{'status':'succeeded'})
            def stop():
                calls.append('stop')
                if fail_phase == 'stop_request':fail()
                if fail_phase == 'stop_response_serialization':
                    return SimpleNamespace(to_dict=fail)
                return response
            def snapshot(_):
                if fail_phase == 'post_stop_process_observation':fail()
                return SimpleNamespace(members=[])
            def close():
                calls.append('close')
                if close_failure == 'guard':_close_guard()
                elif close_failure is not None:raise close_failure
            runtime.stop, runtime.close, runtime.process.snapshot = stop, close, snapshot
            return runtime
        monkeypatch.setattr(AdmittedBenchmarkRuntime,'create',injected)
        if fail_phase == 'post_stop_context_refresh':
            monkeypatch.setattr(driver,'_refresh_completed_context',lambda *_a,**_k:fail())
        if observe_failure:
            monkeypatch.setattr(observation,'observe',lambda *_a,**_k:(_ for _ in ()).throw(OSError('private-observer-body')))
        result = run(**kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(driver,'run',capture)
    failed = fail_phase is not None or close_failure is not None
    if cancellation is not None:
        with pytest.raises(type(cancellation)) as caught:
            exercise(tmp_path,monkeypatch,'no-index','model_router',False,'supervisor-semantic-router-input@1')
        assert caught.value is cancellation
    elif failed:
        # The shared authored environment ends with an intentional success
        # assertion; inspect its captured failed driver result independently.
        with pytest.raises(AssertionError):
            exercise(tmp_path,monkeypatch,'no-index','model_router',False,'supervisor-semantic-router-input@1')
        assert len(captured) == 1
    else:
        exercise(tmp_path,monkeypatch,'no-index','model_router',False,'supervisor-semantic-router-input@1')
    path = tmp_path/'deployment/state/trial-result.json'
    result = json.loads(path.read_text())
    assert calls == ['stop','close']
    assert result['worker_cleanup_returncode'] == 0
    if captured:assert result == captured[0]
    return result


def test_original_stop_diagnostic_survives_later_live_child_close_failure(tmp_path,monkeypatch):
    result = _exercise(tmp_path,monkeypatch,fail_phase='stop_request',
        failure=TimeoutError('private-stop-body /private/path'),close_failure='guard')
    failures = result['shutdown_failures']
    stop, close = failures['stop'],failures['runtime_close']
    assert stop['phase'] == 'stop_request' and stop['reason_code'] == 'timeout'
    assert stop['exceptions'][0]['exception_type'] == 'TimeoutError'
    assert close['phase'] == 'runtime_close' and close['reason_code'] == 'live_launched_children'
    assert close['exceptions'][0]['exception_type'] == 'RuntimeError'
    assert any(frame['file'] == 'isolated_benchmark_runtime.py' for frame in close['exceptions'][0]['frames'])
    assert result['error']['type'] == 'RuntimeError'  # Existing propagation preserved.
    assert result['runtime_close'] == {'attempted':True,'succeeded':False,'error_type':'RuntimeError'}
    assert 'stop' not in result and result['task_completed'] is False
    assert 'private' not in json.dumps(failures)
    assert all(observation.validate(value) == value for value in failures.values())


def test_native_target_denial_preserves_stale_tree_type_and_source_frame(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.control.control_plane import SupervisorControlService
    def deny_exact_target():
        repository, state = tmp_path/'repository', tmp_path/'native-state'
        service = SimpleNamespace(_repository_roots={repository},_state_roots={state},
            _identity_validator=lambda _:False,
            _invoke_validator=SupervisorControlService._invoke_validator)
        SupervisorControlService._check_target(service,
            SimpleNamespace(repository_root=repository,state_root=state))
    result = _exercise(tmp_path,monkeypatch,fail_phase='stop_request',
        failure=deny_exact_target,close_failure='guard')
    first = result['shutdown_failures']['stop']['exceptions'][0]
    assert first['exception_type'] == 'StaleTreeError'
    assert any(frame['file'] == 'control_plane.py' for frame in first['frames'])
    assert result['shutdown_failures']['runtime_close']['reason_code'] == 'live_launched_children'
    assert str(tmp_path) not in json.dumps(result['shutdown_failures'])


@pytest.mark.parametrize('phase', ['stop_request','stop_response_serialization',
    'post_stop_process_observation','post_stop_context_refresh'])
def test_exact_shutdown_phase_and_successful_close_retained(tmp_path,monkeypatch,phase):
    result = _exercise(tmp_path,monkeypatch,fail_phase=phase,failure=ValueError('private-stage-body'))
    assert set(result['shutdown_failures']) == {'stop'}
    assert result['shutdown_failures']['stop']['phase'] == phase
    assert result['error']['type'] == 'ValueError'
    assert result['runtime_close'] == {'attempted':True,'succeeded':True}
    assert ('stop' in result) is phase.startswith('post_stop_')
    assert result['task_completed'] is False


def test_successful_shutdown_has_no_failure_observation(tmp_path,monkeypatch):
    result = _exercise(tmp_path,monkeypatch)
    assert result['task_completed'] is True and 'shutdown_failures' not in result
    assert result['stop']['status'] == 'succeeded' and result['remaining_processes'] == 0


def test_diagnostic_failure_does_not_suppress_close_worker_cleanup_or_report(tmp_path,monkeypatch):
    result = _exercise(tmp_path,monkeypatch,fail_phase='stop_request',failure=ValueError('stop'),
        close_failure='guard',observe_failure=True)
    assert result['error']['type'] == 'RuntimeError' and result['task_completed'] is False
    assert result['shutdown_failures'] == {
        'stop':observation.unavailable('stop_request'),
        'runtime_close':observation.unavailable('runtime_close')}


@pytest.mark.parametrize('phase',['stop_request','runtime_close'])
@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_cancellation_propagates_with_cleanup_and_durable_diagnostic(tmp_path,monkeypatch,phase,kind):
    failure = kind('private-cancellation-body')
    result = _exercise(tmp_path,monkeypatch,
        fail_phase='stop_request' if phase == 'stop_request' else None,
        failure=failure,close_failure=failure if phase == 'runtime_close' else None,cancellation=failure)
    value = result['shutdown_failures']['stop' if phase == 'stop_request' else 'runtime_close']
    assert value['phase'] == phase and value['reason_code'] == 'cancellation'
    assert value['exceptions'][0]['exception_type'] == kind.__name__
    assert 'private' not in json.dumps(value)


def test_unknown_exception_names_messages_and_foreign_frames_are_not_exported():
    PrivateException = type('AUTHORED_PRIVATE_CLASS',(Exception,),{})
    try:
        raise PrivateException('/private/path body-secret')
    except PrivateException as error:
        value = observation.observe(error,phase='stop_request')
    assert value['exceptions'] == [{'exception_type':'other','frames':[]}]
    assert not any(text in json.dumps(value) for text in ('AUTHORED','private','secret'))


def test_known_literal_without_installed_origin_is_not_a_custody_classification():
    value = observation.observe(RuntimeError('stop every live launched child before releasing runtime custody'),phase='runtime_close')
    assert value['reason_code'] == 'other'


def test_long_cyclic_exception_chain_is_bounded():
    errors = [RuntimeError('private-'+str(i)) for i in range(12)]
    for current,cause in zip(errors,errors[1:]+errors[:1]):current.__cause__=cause
    value = observation.observe(errors[0],phase='stop_request')
    assert len(value['exceptions']) == 4 and value['chain_truncated'] is True
    assert observation.validate(value) == value


@pytest.mark.parametrize('mutate',[
    lambda x:x.update(message='private'), lambda x:x.update(phase='arbitrary'),
    lambda x:x.update(completion_authority=True),lambda x:x.update(observation_only=1),
    lambda x:x.update(reason_code='arbitrary'),lambda x:x.update(chain_truncated=0),
    lambda x:x.update(exceptions=[]),lambda x:x['exceptions'][0].update(exception_type='PrivateError'),
    lambda x:x['exceptions'][0]['frames'].append({'file':'/private/path','line':1}),
    lambda x:x['exceptions'][0]['frames'].append({'file':'control_plane.py','line':True}),
    lambda x:x['exceptions'][0]['frames'].append({'file':'control_plane.py','line':10000001}),
    lambda x:x['exceptions'].extend([{'exception_type':'RuntimeError','frames':[]}] * 4),
    lambda x:x.update(reason_code='cancellation'),lambda x:x.update(reason_code='timeout'),
    lambda x:x.update(reason_code='live_launched_children'),
])
def test_malformed_or_unbounded_projection_is_rejected(mutate):
    value = observation.observe(ValueError('private'),phase='stop_request')
    mutate(value)
    assert observation.validate(value) is None


def test_validator_returns_independent_copy():
    value = observation.observe(ValueError('private'),phase='stop_request')
    copied = observation.validate(value)
    copied['exceptions'][0]['frames'].append({'file':'control_plane.py','line':1})
    assert value['exceptions'][0]['frames'] == []
