"""Long preparation has its own deadline and cannot spend the coding allowance."""
from contextlib import nullcontext
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor

PROFILE = profiles.FORMALIZATION_SOURCE384_PROFILE


def test_formalization_limits_reserve_every_phase_and_preserve_historical_profiles():
    budget = profiles.execution_budget(PROFILE)
    assert budget['driver_seconds'] == sum(budget[key] for key in (
        'preparation_seconds', 'planning_seconds', 'native_setup_seconds',
        'implementation_seconds', 'cleanup_seconds')) == 6420
    assert budget['driver_seconds'] < budget['exec_seconds'] < budget['harbor_seconds']
    assert budget['native_lifetime_seconds'] == sum(budget[key] for key in (
        'native_setup_seconds', 'implementation_seconds', 'cleanup_seconds')) == 2160
    assert profiles.coding_timeout_seconds(PROFILE) == 1800
    assert profiles.planner_timeout_seconds(PROFILE) == 600
    assert profiles.implementation_watchdog_seconds(PROFILE) == 1860
    assert profiles.execution_budget(profiles.CODING600_SOURCE384_PROFILE)['driver_seconds'] == 900
    assert profiles.coding_timeout_seconds(profiles.CODING600_SOURCE384_PROFILE) == 600
    assert profiles.planner_timeout_seconds(profiles.PLANNER180_20GIB_SOURCE384_PROFILE) == 180
    assert profiles.execution_budget()['driver_seconds'] == 285


def test_cumulative_preparation_cannot_be_refilled_by_planning():
    budget = profiles.ReservedPhaseBudget(started=100., profile=PROFILE)
    assert budget.enter('preparation', now=100.) == 3700.
    assert budget.enter('planning', now=3600.) == 4260.
    assert budget.enter('preparation', now=4250.) == 4350.
    assert budget.enter('native_setup', now=4349.) == 4529.
    assert budget.enter('implementation', now=4528.) == 6388.
    assert budget.remaining(now=6328.) == 60.
    with pytest.raises(TimeoutError, match='implementation'):
        budget.remaining(now=6388.)


def test_initial_driver_setup_is_preparation_and_cannot_shorten_last_phase():
    budget = profiles.ReservedPhaseBudget(started=1000., profile=PROFILE)
    assert budget.enter('preparation', now=1010.) == 4600.
    assert budget.enter('planning', now=4599.) == 5259.
    assert budget.enter('preparation', now=5258.) == 5259.
    assert budget.enter('native_setup', now=5258.5) == 5438.5
    assert budget.enter('implementation', now=5438.) == 7298.
    assert budget.remaining(now=5438.) == 1860


def test_initial_driver_setup_cannot_overrun_its_preparation_reserve():
    budget = profiles.ReservedPhaseBudget(started=1000., profile=PROFILE)
    with pytest.raises(TimeoutError, match='preparation'):
        budget.enter('preparation', now=4600.)


def _simulate(tmp_path, monkeypatch, *, arm='full', before_plan=3300., planning=600.,
              after_plan=290., startup=150., coding=1799., expire_coding=False,
              real_alarm_in_prepare=False, continuation=None):
    now = [1000.]
    real_signal, real_timer, real_sleep = driver.signal.signal, driver.signal.setitimer, driver.time.sleep
    alarms, callbacks, creates, contexts, plans, cleanup = [], {}, [], [], [], []
    for key, value in profiles.admission_environment(PROFILE).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(driver.os, 'umask', lambda *args: None)
    monkeypatch.setattr(driver.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(driver.signal, 'signal', lambda kind, handler: callbacks.setdefault(kind, handler))
    monkeypatch.setattr(driver.signal, 'setitimer', lambda kind, seconds: alarms.append((now[0], seconds)))
    monkeypatch.setattr(driver, '_failure_diagnostics', lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, '_final_context_audit', lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, '_native_diagnostics', lambda *args: {})
    monkeypatch.setattr(driver, '_start_cleanup_observation', lambda *args: {'status': 'unavailable'})
    def worker_cleanup(argv, **kwargs):
        cleanup.append(('worker', kwargs['timeout']))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(driver.subprocess, 'run', worker_cleanup)
    def prepare(**kwargs):
        if real_alarm_in_prepare:
            previous = real_signal(driver.signal.SIGALRM, callbacks[driver.signal.SIGALRM])
            real_timer(driver.signal.ITIMER_REAL, .02)
            try:
                try:
                    real_sleep(1)
                except Exception:
                    continuation.append('optional component swallowed expiry')
                continuation.append('continued after expiry')
            finally:
                real_timer(driver.signal.ITIMER_REAL, 0)
                real_signal(driver.signal.SIGALRM, previous)
        now[0] += before_plan
        return {'intent_preplanning': {}}
    monkeypatch.setattr(driver.preparation, 'prepare', prepare)
    monkeypatch.setattr(driver.preparation, 'initial_context', lambda **kwargs: contexts.append(kwargs) or {})
    def plan(**kwargs):
        plans.append(kwargs)
        now[0] += planning
        return {'qualified': True}
    monkeypatch.setattr(driver.preparation, 'plan', plan)
    monkeypatch.setattr(driver.preparation, 'context', lambda **kwargs: {'context_bundle': {}})
    task = SimpleNamespace(task_cid='task', task_key='task-key')
    def verify(*args, **kwargs):
        now[0] += after_plan
        return {'graph': SimpleNamespace(tasks=[task]), 'manifest': {
            'repository_cid': 'repository', 'sources': {driver.preparation.INSTRUCTION: {'sha256': 'b' * 64}}}}
    monkeypatch.setattr(driver, 'verify_local_benchmark_admission', verify)
    monkeypatch.setattr(doctor, 'prepare_terminal_doctor_dispatch', lambda **kwargs: {
        'route': 'model_router', 'status': 'residual'})
    monkeypatch.setattr(driver, '_published_retrieval_options', lambda **kwargs: {})
    class Progress:
        def __init__(self, **kwargs): self.report = {}
        def begin_post_start(self, **kwargs): pass
        def sample(self, **kwargs): return None
        def finish(self, report): pass
    monkeypatch.setattr(driver, 'NativeProgress', Progress)
    def get_task(_):
        now[0] += coding
        if expire_coding:
            callbacks[driver.signal.SIGALRM](driver.signal.SIGALRM, None)
        return SimpleNamespace(status='blocked', revision=4)
    owner = SimpleNamespace(server=object(), source=SimpleNamespace(get_task=get_task))
    state = tmp_path / 'state/run'
    state.mkdir(parents=True)
    (state / 'admission.json').write_text('{}')
    def create(*args, **kwargs):
        creates.append(kwargs)
        now[0] += 30.
        def start():
            now[0] += startup - 30.
            return SimpleNamespace(to_dict=lambda: {'status': 'succeeded'})
        def stop():
            cleanup.append(('stop', now[0]))
            now[0] += 20.
            return SimpleNamespace(to_dict=lambda: {'status': 'succeeded'})
        return SimpleNamespace(start=start, stop=stop, close=lambda: cleanup.append(('close', now[0])),
            state=state, profile=object(), startup_diagnostics=lambda: None, observe=lambda: {},
            process=SimpleNamespace(snapshot=lambda _: SimpleNamespace(members=[])))
    for name, attrs in (
        ('benchmarks.agent_supervisor.container_coding.native_quack_qualification',
            {'open_existing_native_owner': lambda **kwargs: nullcontext(owner)}),
        ('ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime',
            {'AdmittedBenchmarkRuntime': SimpleNamespace(create=create)}),
        ('ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy',
            {'GROK_CODEX_EXECUTION_MODE': 'route'}),
        ('ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction',
            {'prepare_public_instruction_context': lambda **kwargs: {'artifact': 'instruction', 'sha256': 'c' * 64}}),
    ):
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    report = driver.run(instruction=tmp_path / 'instruction', state=state, arm=arm,
        resource_profile=PROFILE, **({'source384_config': tmp_path / 'source384.json'} if arm == 'full' else {}))
    assert json.loads((state.parent / 'run-result.json').read_text()) == report
    assert report['worker_cleanup_returncode'] == 0
    assert cleanup[-1][0] == 'worker' and 0 < cleanup[-1][1] <= 15
    return report, creates, contexts, plans, alarms, cleanup


@pytest.mark.parametrize('arm', ['full', 'no-index'])
def test_long_preparation_preserves_full_coding_window_and_native_owner_lifetime(tmp_path, monkeypatch, arm):
    report, creates, contexts, plans, alarms, cleanup = _simulate(tmp_path, monkeypatch, arm=arm)
    assert 'error' not in report
    assert plans[0]['timeout_seconds'] == 600
    assert report['provider_coding_timeout_seconds'] == 1800
    assert creates[0]['implementation_timeout_seconds'] == 1860
    assert creates[0]['lifetime_seconds'] == 2160 and creates[0]['budget_profile'] == PROFILE
    command = shlex.split(creates[0]['implementation_command'])
    assert command[command.index('--timeout') + 1] == '1800'
    assert command[command.index('--budget-profile') + 1] == PROFILE
    assert creates[0]['start_timeout_ms'] == 120000
    assert report['stop']['status'] == 'succeeded' and report['remaining_processes'] == 0
    assert report['runtime_close']['succeeded'] is True
    assert report['seconds'] > 5700  # much longer than the old enclosing900s
    assert alarms[0][1] == 3600  # scope/import setup is already under the preparation alarm
    assert (any(seconds == 1860 for _, seconds in alarms)
            and any(seconds == 118 for _, seconds in alarms))
    assert [row[0] for row in cleanup] == ['stop', 'close', 'worker']
    if arm == 'full':
        assert contexts[0]['source384_budget_profile'] == PROFILE
        assert contexts[0]['source384_timeout_seconds'] == 300
    else:
        assert not contexts


@pytest.mark.parametrize('change, phase, native_created', [
    ({'before_plan': 3601.}, 'prepare', False),
    ({'planning': 661.}, 'planning', False),
    ({'after_plan': 301.}, 'native_execution', False),
    ({'startup': 181.}, 'native_execution', True),
    ({'coding': 1860., 'expire_coding': True}, 'native_execution', True),
])
def test_expired_phase_cannot_borrow_coding_or_skip_bounded_cleanup(tmp_path, monkeypatch, change, phase, native_created):
    report, creates, _, _, alarms, cleanup = _simulate(tmp_path, monkeypatch, **change)
    assert report['error']['type'] == 'TimeoutError'
    assert report['error_phase'] == phase or (phase == 'prepare' and report['error_phase'] == 'initial_context')
    assert bool(creates) is native_created
    assert report['task_completed'] is False
    if native_created:
        assert [row[0] for row in cleanup] == ['stop', 'close', 'worker']
        assert any(seconds == 118 for _, seconds in alarms)
    assert alarms[-1][1] == 0


def test_reserved_profile_cannot_be_silently_shortened(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.os, 'umask', lambda *args: pytest.fail('invalid budget reached setup'))
    with pytest.raises(ValueError, match='full declared driver budget'):
        driver.run(instruction=tmp_path / 'instruction', state=tmp_path / 'state', arm='full',
            resource_profile=PROFILE, timeout_seconds=900)


@pytest.mark.skipif(not hasattr(driver.signal, 'setitimer'), reason='requires POSIX phase alarms')
def test_real_phase_alarm_escapes_optional_exception_handlers_and_reaches_cleanup(tmp_path, monkeypatch):
    continuation = []
    report, creates, _, _, alarms, cleanup = _simulate(tmp_path, monkeypatch,
        real_alarm_in_prepare=True, continuation=continuation)
    assert continuation == []
    assert creates == [] and [row[0] for row in cleanup] == ['worker']
    assert report['error']['type'] == 'TimeoutError' and report['error_phase'] == 'prepare'
    assert report['task_completed'] is False and alarms[-1][1] == 0
