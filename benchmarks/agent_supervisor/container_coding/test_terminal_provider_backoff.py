"""Deterministic provider admission, persistence, and one-shot execute checks."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest

from benchmarks.agent_supervisor.container_coding.terminal_provider_backoff import ProviderBackoff, BackoffStateError
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark


class Clock:
    def __init__(self):
        self.value = 1000.
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def gate(tmp_path, clock):
    return ProviderBackoff(tmp_path / 'state', wall_clock=clock, monotonic=clock,
                           jitter=lambda: 0, boot_id='fixture-boot')


def stop(reason='rate_limit', delay=None, **extra):
    return dict(schema='terminal-provider-stop@1', observation_valid=True,
                provider='codex_cli', reason_code=reason, retry_after_seconds=delay, **extra)


def limit(gate, **kwargs):
    permit = gate.acquire('codex_cli')
    assert permit.allowed
    try:
        return gate.settle(permit, stop=stop(**kwargs))
    finally:
        permit.close()


def test_missing_hint_defers_new_work_and_probe_success_clears(gate, clock):
    assert limit(gate)['retry_after_seconds'] == 60
    denied = gate.acquire('codex_cli')
    assert not denied.allowed and denied.reason == 'cooldown'
    assert gate.acquire('grok_cli').allowed
    clock.advance(60)
    probe = gate.acquire('codex_cli')
    assert probe.allowed and probe.recovery_probe
    try:
        assert gate.acquire('codex_cli').reason == 'recovery_probe_in_progress'
        with pytest.raises(BackoffStateError, match='recovery_probe_in_progress'):
            gate.reset('codex_cli')
        assert gate.settle(probe, provider_succeeded=True)['recovery_required'] is False
    finally:
        probe.close()
    assert gate.acquire('codex_cli').reason == 'ready'


def test_repeated_rate_limits_back_off_to_cap_and_honor_longer_provider_hint(gate, clock):
    for expected in (60, 120, 240, 480, 960, 1920, 3600, 3600):
        assert limit(gate)['retry_after_seconds'] == expected
        clock.advance(expected)
    assert limit(gate, delay=86400)['retry_after_seconds'] == 86400


def test_positive_jitter_is_bounded(gate):
    gate.jitter = lambda: .5
    assert limit(gate)['retry_after_seconds'] == 66


@pytest.mark.parametrize('reason', ['billing', 'authentication', 'quota'])
def test_hard_stop_does_not_time_expire_or_downgrade(gate, clock, reason):
    old = gate.acquire('codex_cli')
    assert limit(gate, reason=reason)['blocked'] is True
    gate.settle(old, stop=stop())
    clock.advance(1000000)
    assert gate.acquire('codex_cli').reason == reason
    gate.reset('codex_cli')
    assert gate.acquire('codex_cli').allowed


@pytest.mark.parametrize('delay', [True, -1, 0, float('nan'), float('inf'), '3', 604801, 10**400])
def test_malformed_stop_hint_is_manual_block(gate, delay):
    assert limit(gate, delay=delay)['blocked'] is True
    assert gate.acquire('codex_cli').reason == 'invalid_provider_stop'


def test_unbounded_hint_is_not_clamped_into_early_retry(gate, clock):
    assert limit(gate, retry_after_unbounded=True)['blocked'] is True
    clock.advance(10000000)
    assert gate.acquire('codex_cli').reason == 'provider_delay_unbounded'


def test_old_success_cannot_clear_new_rate_limit(gate, clock):
    older = gate.acquire('codex_cli')
    limit(gate)
    clock.advance(60)
    probe = gate.acquire('codex_cli')
    gate.settle(older, stop=stop(delay=900))
    try:
        result = gate.settle(probe, provider_succeeded=True)
        assert result['recovery_required'] and result['retry_after_seconds'] == 900
    finally:
        probe.close()


def test_dead_probe_requires_explicit_recovery_and_no_time_based_lease_steal(gate, clock):
    limit(gate)
    clock.advance(60)
    probe = gate.acquire('codex_cli')
    probe.close()  # Process death closes flock but leaves the durable pending marker.
    clock.advance(100000)
    assert gate.acquire('codex_cli').reason == 'unsettled_recovery_probe'
    gate.reset('codex_cli')
    assert gate.acquire('codex_cli').allowed


def test_unknown_probe_result_does_not_clear_cooldown(gate, clock):
    limit(gate)
    clock.advance(60)
    probe = gate.acquire('codex_cli')
    try:
        assert gate.settle(probe)['reason'] == 'recovery_unverified'
    finally:
        probe.close()
    assert not gate.acquire('codex_cli').allowed


def test_shared_monotonic_deadline_survives_wall_clock_jump(gate, clock):
    limit(gate)
    gate.wall_clock = lambda: 10**10
    assert gate.acquire('codex_cli').retry_after_seconds == 60
    restarted = ProviderBackoff(gate.directory, monotonic=clock, wall_clock=clock,
                                boot_id='fixture-boot', jitter=lambda: 0)
    assert restarted.acquire('codex_cli').retry_after_seconds == 60


def test_wait_rechecks_deadline_and_never_holds_state_lock(gate, clock):
    limit(gate)
    clock.advance(58)
    calls = []
    def sleep(seconds):
        calls.append(seconds)
        # Reentrant read would raise state_busy if wait held the lock.
        assert not gate.acquire('codex_cli').allowed
        clock.advance(seconds)
    permit = gate.acquire_with_wait('codex_cli', max_wait_seconds=3, sleeper=sleep)
    try:
        assert permit.allowed and calls == [1, 1]
        gate.abandon_before_dispatch(permit)
    finally:
        permit.close()


def test_wait_cancellation_and_long_delay_never_sleep(gate, clock):
    limit(gate, delay=500)
    sleeps = []
    assert not gate.acquire_with_wait('codex_cli', max_wait_seconds=10, sleeper=sleeps.append).allowed
    event = threading.Event()
    event.set()
    assert gate.acquire_with_wait('codex_cli', cancel_event=event).reason == 'cancelled_before_dispatch'
    assert sleeps == []


@pytest.mark.parametrize('seconds', [-1, True, 61, '60', float('inf'), float('nan')])
def test_wait_parameter_is_strict(gate, seconds):
    with pytest.raises(ValueError):
        gate.acquire_with_wait('codex_cli', max_wait_seconds=seconds)


def test_failed_settlement_installs_blocking_sentinel(gate, monkeypatch):
    permit = gate.acquire('codex_cli')
    with monkeypatch.context() as patch:
        patch.setattr(gate, '_write', lambda *_: (_ for _ in ()).throw(OSError('fixture-write-failure')))
        with pytest.raises(OSError):
            gate.settle(permit, stop=stop())
    assert gate.acquire('codex_cli').reason == 'settlement_unavailable'
    assert (gate.directory / 'codex_cli.unsettled').read_bytes() == b''
    gate.reset('codex_cli')
    assert gate.acquire('codex_cli').allowed


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / 'trial'
    output.mkdir()
    (output / 'preparation.json').write_text('{}')
    return output


def invocation(*, failed=False):
    return dict(schema='router-implementation-invocation@1', invocation_id='authored-1',
                status='failed' if failed else 'provider_returned', provider='codex_cli',
                model='gpt-6.1-sol', phase='planning', failure_phase='provider_invocation',
                **({'provider_failure': {'phase': 'provider_invocation', 'reason_code': 'rate_limit',
                                          'retry_after_seconds': 180}} if failed else {}))


def test_execute_stops_before_marker_container_or_call_and_does_not_replay(gate, prepared, monkeypatch):
    import benchmarks.agent_supervisor.container_coding.terminal_provider_backoff as module
    monkeypatch.setattr(module, 'ProviderBackoff', lambda *_: gate)
    limit(gate)
    called = []
    monkeypatch.setattr(benchmark, '_execute_once', lambda *_a, **_kw: called.append(True))
    result = benchmark.execute(prepared)
    assert result['state'] == 'provider_deferred'
    assert result['provider_calls'] == result['containers_launched'] == 0
    assert not (prepared / 'invocation.json').exists() and called == []
    assert not (prepared / 'receipt.json').exists()
    assert (prepared / 'provider-deferral.json').is_file()


@pytest.mark.parametrize('failed', [False, True])
def test_execute_settles_actual_router_status_and_keeps_unknown_usage(gate, clock, prepared, monkeypatch, failed):
    import benchmarks.agent_supervisor.container_coding.terminal_provider_backoff as module
    monkeypatch.setattr(module, 'ProviderBackoff', lambda *_: gate)
    limit(gate)
    clock.advance(60)
    observed = []
    def once(output, **kwargs):
        observed.append(output)
        (output / 'invocation.json').write_text('{}')
        return {'trials': [{'exact_trial_task_matches': True,
                            'supervisor': {'provider_invocations': [invocation(failed=failed)]}}],
                'complete_tokens': None, 'complete_single_trial_receipt': not failed}
    monkeypatch.setattr(benchmark, '_execute_once', once)
    result = benchmark.execute(prepared)
    assert len(observed) == 1 and result['complete_tokens'] is None
    assert result['provider_backoff']['recovery_required'] is failed
    with pytest.raises(FileExistsError):
        benchmark.execute(prepared)
    assert len(observed) == 1


def test_validation_failure_releases_unused_recovery_probe(gate, clock, prepared, monkeypatch):
    import benchmarks.agent_supervisor.container_coding.terminal_provider_backoff as module
    monkeypatch.setattr(module, 'ProviderBackoff', lambda *_: gate)
    limit(gate)
    clock.advance(60)
    def fail(*args, **kwargs):
        raise ValueError('authored validation failure')
    monkeypatch.setattr(benchmark, '_execute_once', fail)
    with pytest.raises(ValueError):
        benchmark.execute(prepared)
    permit = gate.acquire('codex_cli')
    try:
        assert permit.allowed and permit.recovery_probe
    finally:
        permit.close()


def test_post_dispatch_missing_receipt_does_not_authorize_probe_replay(gate, clock, prepared, monkeypatch):
    import benchmarks.agent_supervisor.container_coding.terminal_provider_backoff as module
    monkeypatch.setattr(module, 'ProviderBackoff', lambda *_: gate)
    limit(gate)
    clock.advance(60)
    def fail(output, **kwargs):
        (output / 'invocation.json').write_text('{}')
        raise RuntimeError('authored collector failure')
    monkeypatch.setattr(benchmark, '_execute_once', fail)
    with pytest.raises(RuntimeError):
        benchmark.execute(prepared)
    assert gate.acquire('codex_cli').reason == 'settlement_unavailable'
    with pytest.raises(FileExistsError):
        benchmark.execute(prepared)


@pytest.mark.parametrize('mutation', ['extra_trial', 'foreign_task', 'duplicate_identity', 'foreign_model'])
def test_ambiguous_recovery_receipts_cannot_reopen_provider(gate, clock, prepared, monkeypatch, mutation):
    import benchmarks.agent_supervisor.container_coding.terminal_provider_backoff as module
    monkeypatch.setattr(module, 'ProviderBackoff', lambda *_: gate)
    limit(gate)
    clock.advance(60)
    trial = {'exact_trial_task_matches': True, 'supervisor': {'provider_invocations': [invocation()]}}
    result = {'trials': [trial]}
    if mutation == 'extra_trial':
        result['trials'].append(None)
    elif mutation == 'foreign_task':
        trial['exact_trial_task_matches'] = False
    elif mutation == 'duplicate_identity':
        trial['supervisor']['provider_invocations'].append(invocation())
    else:
        trial['supervisor']['provider_invocations'][0]['model'] = 'different-model'
    def once(output, **kwargs):
        (output / 'invocation.json').write_text('{}')
        return result
    monkeypatch.setattr(benchmark, '_execute_once', once)
    expected = 'recovery_unverified' if mutation == 'foreign_task' else 'provider_outcome_unknown'
    assert benchmark.execute(prepared)['provider_backoff']['reason'] == expected
    assert not gate.acquire('codex_cli').allowed


def test_real_benchmark_cli_collects_stop_then_defers_fresh_trial_before_harness(tmp_path):
    """Run the public CLI and real preparation checks with an authored harness.

    This is not a live benchmark score. Only the external Harbor process is
    replaced; native result collection, durable admission and CLI exit run.
    """
    from benchmarks.agent_supervisor.container_coding.test_terminal_task_selection import _prepared
    output, prepared, config = _prepared(tmp_path / 'first', benchmark)
    second, _, _ = _prepared(tmp_path / 'second', benchmark)
    job = output / 'jobs' / config['job_name']
    trial = job / 'authored-trial'
    native_result = {'task_name': prepared['task'], 'trial_name': trial.name,
        'config': {'task': config['tasks'][0], 'agent': config['agents'][0], 'verifier': config['verifier']},
        'verifier_result': {'rewards': {'reward': 0}}}
    report = {'provider_invocations': [invocation(failed=True)], 'task_completed': False}
    dispatches = tmp_path / 'physical-harness-dispatches'
    payload = tmp_path / 'authored-results.json'
    payload.write_text(json.dumps({'trial': str(trial), 'job': str(job),
                                   'native': native_result, 'report': report, 'dispatches': str(dispatches)}))
    script = tmp_path / 'authored-harbor.py'
    script.write_text('''import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
with Path(p['dispatches']).open('a') as f: f.write('dispatch\\n')
trial = Path(p['trial'])
(trial / 'agent').mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps(p['native']))
(trial / 'agent/supervisor-result.json').write_text(json.dumps(p['report']))
(Path(p['job']) / 'result.json').write_text('{}')
''')
    prepared['command'] = [sys.executable, str(script), str(payload)]
    (output / 'preparation.json').write_text(json.dumps(prepared))
    # No admission flag: the default gate is shared across these CLI processes.
    environment = dict(os.environ, XDG_STATE_HOME=str(tmp_path / 'host-state'))
    command = [sys.executable, '-m', benchmark.__name__, 'execute', '--output']
    first = subprocess.run([*command, str(output)], capture_output=True, text=True, timeout=30, env=environment)
    assert first.returncode == 0, first.stderr
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['trials'][0]['provider_stop']['state'] == 'rate_limited'
    assert 175 < receipt['provider_backoff']['retry_after_seconds'] <= 180
    assert receipt['dollar_cost'] is None
    assert dispatches.read_text().splitlines() == ['dispatch']
    first_marker = (output / 'invocation.json').read_bytes()
    deferred = subprocess.run([*command, str(second)], capture_output=True, text=True, timeout=30, env=environment)
    assert deferred.returncode == 75, deferred.stderr
    result = json.loads(deferred.stdout)
    assert result['state'] == 'provider_deferred' and result['reason'] == 'cooldown'
    assert result['provider_calls'] == result['containers_launched'] == 0
    assert not (second / 'invocation.json').exists()
    assert not (second / 'harbor.stdout').exists()
    assert (output / 'invocation.json').read_bytes() == first_marker
    assert dispatches.read_text().splitlines() == ['dispatch']


@pytest.mark.parametrize('case', ['missing_report_after_execution', 'unreceipted_attempt',
                                  'malformed_none', 'malformed_receipt', 'setup_only'])
def test_incomplete_result_after_possible_dispatch_blocks_but_setup_only_does_not(gate, prepared, monkeypatch, case):
    import benchmarks.agent_supervisor.container_coding.terminal_provider_backoff as module
    monkeypatch.setattr(module, 'ProviderBackoff', lambda *_: gate)
    def once(output, **kwargs):
        (output / 'invocation.json').write_text('{}')
        report = None
        if case == 'unreceipted_attempt':
            report = {'unreceipted_provider_attempt': {'phase': 'coding'}}
        elif case.startswith('malformed'):
            report = {'provider_invocations': [None if case == 'malformed_none' else {}]}
        return {'trials': [{'exact_trial_task_matches': True,
                           'native_agent_execution_started': case == 'missing_report_after_execution',
                           'supervisor': report}]}
    monkeypatch.setattr(benchmark, '_execute_once', once)
    result = benchmark.execute(prepared)
    if case == 'setup_only':
        assert result['provider_backoff']['reason'] == 'ready'
        assert gate.acquire('codex_cli').allowed
    else:
        assert result['provider_backoff']['reason'] == 'provider_outcome_unknown'
        assert not gate.acquire('codex_cli').allowed
