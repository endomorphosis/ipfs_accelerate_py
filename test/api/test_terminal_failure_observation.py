"""Retained native failure diagnostics remain scoped, bounded and advisory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_failure_observation as observation
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

TASK = 'task:authored-owned'
ATTEMPT = 'attempt:authored-owned'
ATTEMPT_HASH = hashlib.sha256(ATTEMPT.encode()).hexdigest()


def _diagnostic(**changes):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.bridge_failure_diagnostics import observe_bridge_failure
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeError
    diagnostic = observe_bridge_failure(DatabasePortalBridgeError('portal_provider_failed'), {}, phase='terminal_failure')
    return {**diagnostic, **changes}


def _sidecar(tmp_path, **changes):
    path = tmp_path / 'run/bridge-failure-observation.json'
    path.parent.mkdir(exist_ok=True)
    value = {'schema':'database-bridge-failure-observation@1', 'task_cid_sha256':hashlib.sha256(TASK.encode()).hexdigest(),
        'attempt_id_sha256':ATTEMPT_HASH,'diagnostic':_diagnostic(),'observation_only':True,**changes}
    path.write_text(json.dumps(value) + '\n')
    return path


def _bridge(tmp_path):
    value = observation.bridge_failure(tmp_path, task_cid=TASK,
        attempt_id_sha256=ATTEMPT_HASH, phase='terminal_failure')
    assert value['provider_dispatch_observed'] is None
    assert value['completion_authority'] is value['retry_authority'] is value['settlement_authority'] is False
    assert value['scope'] == 'exact_admitted_task_and_attempt'
    assert TASK not in json.dumps(value) and ATTEMPT not in json.dumps(value)
    return value


def _report():
    return {'task_state': {'task_cid':TASK,'status':'blocked','revision':4}, 'native_progress':{
        'latest':{'task':{'task_cid':TASK,'status':'blocked','revision':4,'completion_receipt':{
            'failure_kind':'terminal_portal_bridge_error','attempt_id_sha256':ATTEMPT_HASH}}}}}


def _runner_envelope():
    return {'schema': 'router-implementation-error@2', 'error_type': 'ValueError',
            'diagnostic': runner._runner_failure_diagnostic(ValueError('private-error-canary'))}


def test_exact_terminal_sidecar_with_json_projection_absent(tmp_path):
    _sidecar(tmp_path)
    assert not (tmp_path / 'run/admitted_events.jsonl').exists()
    value = _bridge(tmp_path)
    assert value['status'] == 'observed' and value['matched_record_count'] == 1
    assert value['diagnostic']['reason_code'] == 'portal_provider_failed'
    assert value['diagnostic']['phase'] == 'terminal_failure'


@pytest.mark.parametrize('change', [
    {'task_cid_sha256':'a'*64}, {'attempt_id_sha256':'b'*64},
    {'diagnostic':{'phase':'unknown_callback'}},
])
def test_foreign_task_attempt_or_phase_never_selected(tmp_path, change):
    if 'diagnostic' in change:
        change = {'diagnostic':_diagnostic(**change['diagnostic'])}
    _sidecar(tmp_path, **change)
    assert _bridge(tmp_path)['status'] == 'missing'


@pytest.mark.parametrize('change', [
    {'schema':'foreign'}, {'task_cid_sha256':[]}, {'attempt_id_sha256':'bad'},
    {'body':'private-error-canary'}, {'observation_only':False},
])
def test_malformed_sidecar_is_not_promoted(tmp_path, change):
    _sidecar(tmp_path, **change)
    value = _bridge(tmp_path)
    assert value['status'] == 'invalid' and value['diagnostic'] is None
    assert 'private' not in json.dumps(value)


@pytest.mark.parametrize('change', [
    {'schema':'foreign'}, {'reason_code':'private-error-canary'}, {'raw_body':'private-error-canary'},
    {'completion_authority':True}, {'retry_authority':True}, {'settlement_authority':True},
])
def test_malformed_diagnostic_is_not_promoted(tmp_path, change):
    _sidecar(tmp_path, diagnostic=_diagnostic(**change))
    assert _bridge(tmp_path)['status'] == 'invalid'


@pytest.mark.parametrize('kind', ['missing','symlink','fifo','oversized','duplicate-key','malformed'])
def test_sidecar_storage_bounds(tmp_path, kind):
    path = _sidecar(tmp_path)
    status = 'unavailable'
    if kind == 'missing':
        path.unlink(); status='missing'
    elif kind == 'symlink':
        target=tmp_path/'other';path.rename(target);path.symlink_to(target)
    elif kind == 'fifo':
        path.unlink();os.mkfifo(path)
    elif kind == 'oversized':
        path.write_bytes(b' '*(observation.MAX_BRIDGE_BYTES+1))
    elif kind == 'duplicate-key':
        path.write_bytes(b'{"schema":"first","schema":"last"}');status='invalid'
    else:
        path.write_bytes(b'{truncated');status='invalid'
    assert _bridge(tmp_path)['status'] == status


def test_sidecar_replacement_during_read_is_unavailable(tmp_path, monkeypatch):
    path = _sidecar(tmp_path)
    original = os.fstat
    calls = 0
    def replace_before_final_stat(descriptor):
        nonlocal calls
        calls += 1
        value = original(descriptor)
        if calls == 2:
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return value
    monkeypatch.setattr(os,'fstat',replace_before_final_stat)
    assert _bridge(tmp_path)['status'] == 'unavailable'


def test_terminal_sidecar_retains_child_wrapper_without_dispatch_inference(tmp_path):
    _sidecar(tmp_path, diagnostic=_diagnostic(child_reported_router_failure={
        'status':'observed','diagnostic':_runner_envelope(),'observation_only':True,'scope':'bounded_native_log_tail'}))
    value = _bridge(tmp_path)
    assert value['status']=='observed'
    assert value['diagnostic']['child_reported_router_failure']['diagnostic']['error_type']=='ValueError'
    assert value['provider_dispatch_observed'] is None
    assert 'private' not in json.dumps(value)


def test_actual_planner_preflight_stderr_retained_without_invocation_receipt(tmp_path):
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-m',
        'ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner',
        '--provider', 'grok_cli', '--model', 'wrong-model'], input='private-input-canary',
        capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 1 and result.stdout == ''
    (tmp_path / 'planner-trace.stderr').write_text(result.stderr)
    value = observation.planner_failure(tmp_path)
    assert value['status'] == 'observed' and value['diagnostic']['diagnostic']['phase'] == 'argument_validation'
    assert value['diagnostic']['error_type'] == 'ValueError'
    assert value['provider_dispatch_observed'] is None
    assert 'private' not in json.dumps(value)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'invalid', 'partial', 'symlink', 'oversized'])
def test_planner_child_report_bounds(tmp_path, kind):
    path = tmp_path / 'planner-trace.stderr'
    value = _runner_envelope()
    status = 'missing'
    if kind == 'missing':
        pass
    elif kind == 'duplicate':
        path.write_text((json.dumps(value) + '\n') * 2); status = 'ambiguous'
    elif kind == 'invalid':
        path.write_text(json.dumps({**value, 'raw': 'private-canary'}) + '\n'); status = 'invalid'
    elif kind == 'partial':
        path.write_text(json.dumps(value)); status = 'invalid'
    elif kind == 'symlink':
        target = tmp_path / 'other'; target.write_text(json.dumps(value) + '\n')
        path.symlink_to(target); status = 'unavailable'
    else:
        path.write_bytes(b' ' * (observation.MAX_STDERR_BYTES + 1)); status = 'unavailable'
    result = observation.planner_failure(tmp_path)
    assert result['status'] == status and result['diagnostic'] is None
    assert 'private' not in json.dumps(result)


@pytest.mark.parametrize('line', ['{"schema":"router-implementation-error@2","schema":"other"}\n', '{truncated\n'])
def test_malformed_object_cannot_be_hidden_beside_valid_planner_diagnostic(tmp_path, line):
    (tmp_path / 'planner-trace.stderr').write_text(line + json.dumps(_runner_envelope()) + '\n')
    result = observation.planner_failure(tmp_path)
    assert result['status'] == 'invalid' and result['diagnostic'] is None


@pytest.mark.parametrize('stale', [False, True])
def test_collect_uses_terminal_receipt_attempt_identity(tmp_path, stale):
    _sidecar(tmp_path)
    report = {'task_state': {'task_cid': TASK, 'status': 'blocked', 'revision': 4}, 'native_progress': {
        'latest': {'task': {'task_cid': TASK, 'status': 'blocked', 'revision': 3 if stale else 4,
                           'completion_receipt': {'failure_kind': 'terminal_portal_bridge_error',
                                                'attempt_id_sha256': ATTEMPT_HASH}}}}}
    result = observation.collect(tmp_path, report, native_state=tmp_path)
    assert result['bridge']['status'] == ('unavailable' if stale else 'observed')
    assert result['planner_child']['status'] == 'missing'
    assert TASK not in json.dumps(result) and ATTEMPT not in json.dumps(result)


def test_collection_failure_after_cleanup_does_not_suppress_driver_report(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from test.api.test_terminal_doctor_dispatch import test_driver_selects_before_owner_and_preserves_provider_accounting as exercise
    results = []
    run = driver.run

    def capture(**kwargs):
        from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        create = AdmittedBenchmarkRuntime.create
        def nested(path, **options):
            runtime = create(path, **options)
            runtime.state = path / 'state'
            return runtime
        monkeypatch.setattr(AdmittedBenchmarkRuntime, 'create', nested)
        result = run(**kwargs); results.append(result); return result

    def unavailable(state, report, *, native_state):
        assert native_state == state / 'launch/state'
        assert report['runtime_close'] == {'attempted': True, 'succeeded': True}
        assert report['worker_cleanup_returncode'] == 0
        raise OSError('private-error-canary')

    monkeypatch.setattr(driver, 'run', capture)
    monkeypatch.setattr(observation, 'collect', unavailable)
    exercise(tmp_path, monkeypatch, 'full', 'model_router', False, 'supervisor-semantic-router-input@1')
    value = results[0]['native_failure_observations']
    assert value['status'] == 'unavailable' and value['observation_only'] is True
    assert 'private' not in json.dumps(value)


def test_collect_separates_outer_planner_and_actual_nested_native_state(tmp_path):
    native = tmp_path / 'launch' / 'state'
    native.mkdir(parents=True)
    _sidecar(native)
    _sidecar(tmp_path, diagnostic=_diagnostic(reason_code='portal_validation_failed'))
    (tmp_path / 'planner-trace.stderr').write_text(json.dumps(_runner_envelope()) + '\n')
    report = _report()
    result = observation.collect(tmp_path, report, native_state=native)
    assert result['bridge']['status'] == 'observed'
    assert result['bridge']['diagnostic']['reason_code'] == 'portal_provider_failed'
    assert result['planner_child']['status'] == 'observed'
    absent = observation.collect(tmp_path, report)
    assert absent['bridge']['status'] == 'unavailable'
    assert absent['bridge']['diagnostic'] is None
    assert absent['planner_child']['status'] == 'observed'


def test_planner_failure_before_native_creation_still_publishes_closed_observation(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from test.api.test_terminal_doctor_dispatch import test_driver_selects_before_owner_and_preserves_provider_accounting as exercise
    results = []
    run = driver.run
    def capture(**kwargs):
        def fail(**_):
            (kwargs['state'] / 'planner-trace.stderr').write_text(json.dumps(_runner_envelope()) + '\n')
            raise ValueError('private-planner-failure')
        monkeypatch.setattr(driver.preparation, 'plan', fail)
        result = run(**kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(driver, 'run', capture)
    # Reuse the existing driver fixture's complete environment. Its success
    # assertion intentionally fails when the authored planning failure occurs.
    with pytest.raises(AssertionError):
        exercise(tmp_path, monkeypatch, 'full', 'model_router', False, 'supervisor-semantic-router-input@1')
    assert len(results) == 1
    result = results[0]
    assert result['error_phase'] == 'planning' and result['task_completed'] is False
    assert result['worker_cleanup_returncode'] == 0 and 'runtime_close' not in result
    observed = result['native_failure_observations']
    assert observed['bridge']['status'] == 'unavailable'
    assert observed['planner_child']['status'] == 'observed'
    assert 'private' not in json.dumps(observed)
    path = tmp_path / 'deployment/state/trial-result.json'
    assert json.loads(path.read_text())['native_failure_observations'] == observed
