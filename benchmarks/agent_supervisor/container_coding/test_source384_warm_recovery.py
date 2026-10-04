"""Qualification retry boundaries; authored clocks are not measured PSI recovery."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source384_warm_recovery as recovery
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import EXTENDED_SOURCE384_PROFILE as PROFILE
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources


@pytest.fixture
def refusal(tmp_path):
    current = [ProofHostResources(8, 16384, 16384)]
    config = resources.ResourceSchedulerConfig.for_proof_host(state_path=tmp_path / 'scheduler.json',
        proof_resource_sampler=lambda: current[0], proof_resource_profile='local-benchmark@1',
        proof_memory_stall_percent=10., proof_recovery_enabled=True, proof_recovery_grants=1,
        lane_reservations={}, auto_renew_leases=False)
    scheduler = resources.GlobalResourceScheduler(config)
    current[0] = replace(current[0], memory_stall_percent=11.)
    with pytest.raises(resources.LeaseTimeoutError) as failed:
        scheduler.acquire('snapshot_evaluation', memory_mb=64, timeout=0)
    assert scheduler.snapshot()['active_lease_count'] == 0
    return failed.value


@pytest.fixture
def clock(monkeypatch):
    now, slept = [100.], []
    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(recovery, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    return now, slept


def observe(*, deadline=280., policy=recovery.POLICY, profile=PROFILE):
    return recovery.observe_warm_context(repository='authored', expected_receipt={'authored': True},
        profile=profile, policy=policy, deadline_monotonic=deadline)


def checked(value):
    digest = hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest()
    return recovery.validate_observation(value, profile=value['resource_profile'],
        policy=value['policy'], producer_sha256=digest)


def test_explicit_recovery_replays_same_receipt_without_renewing_deadline(clock, refusal, monkeypatch):
    now, slept = clock
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            now[0] += 90.
            raise refusal
        now[0] += 4.
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    result = checked(observe())
    assert [row['timeout_seconds'] for row in calls] == [90., 85.]
    assert calls[0]['expected_receipt'] == calls[1]['expected_receipt']
    assert calls[0]['expected_receipt'] is not calls[1]['expected_receipt']
    assert slept == [5.] and result['elapsed_seconds'] == 99.
    assert result['attempts'][0]['admission']['last_sample']['host']['memory_stall_percent'] == 11.
    assert result['attempts'][1]['status'] == result['status'] == 'validated'
    assert result['inference_replayed'] is result['admission_execution_split_measured'] is False


def test_default_observation_never_retries(clock, refusal, monkeypatch):
    now, slept = clock
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        now[0] += 90.
        raise refusal
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    with pytest.raises(resources.LeaseTimeoutError) as failed:
        observe(policy=None)
    result = checked(failed.value.source384_warm_observation)
    assert failed.value is refusal and result['max_attempts'] == 1
    assert result['selected_seconds'] == calls[0]['timeout_seconds'] == 90.
    assert len(calls) == 1 and slept == []


def test_persistent_pressure_stops_after_two_calls(clock, refusal, monkeypatch):
    now, slept = clock
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        now[0] += 5.
        raise refusal
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    with pytest.raises(resources.LeaseTimeoutError) as failed:
        observe()
    assert failed.value is refusal and len(calls) == 2 and slept == [5.]
    assert len(checked(failed.value.source384_warm_observation)['attempts']) == 2


@pytest.mark.parametrize('kind', ['cancelled', 'timeout', 'unattached', 'source', 'decoder',
    'malformed', 'contradictory', 'missing_host', 'wrong_threshold', 'cpu'])
def test_only_exact_native_memory_admission_timeout_can_retry(clock, refusal, monkeypatch, kind):
    if kind == 'cancelled':
        error = resources.LeaseCancelledError('authored cancellation')
    elif kind == 'timeout':
        error = TimeoutError('authored outer deadline')
    elif kind == 'unattached':
        error = resources.LeaseTimeoutError('no native observation')
    elif kind in {'source', 'decoder'}:
        error = ValueError('authored ' + kind + ' refusal')
    else:
        error = refusal
        observation = error.admission_observation = deepcopy(error.admission_observation)
        if kind == 'malformed': observation['PRIVATE'] = 'excluded body'
        elif kind == 'contradictory': observation['last_sample']['host']['memory_stall_percent'] = 1.
        elif kind == 'missing_host': observation['last_sample']['host'] = None
        elif kind == 'wrong_threshold': observation['last_sample']['thresholds']['memory_stall_percent'] = 2.
        else:
            observation['primary_gate']['reason'] = observation['last_sample']['reason'] = 'proof_cpu_stall'
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        raise error
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    with pytest.raises(type(error)) as failed:
        observe()
    assert failed.value is error and len(calls) == 1 and clock[1] == []
    assert checked(failed.value.source384_warm_observation)['status'] == 'failed'


@pytest.mark.parametrize('spent', [40., 45.])
def test_enclosing_deadline_expiry_in_call_or_backoff_prevents_retry(clock, refusal, monkeypatch, spent):
    now, slept = clock
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        now[0] += spent
        raise refusal
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    with pytest.raises(resources.LeaseTimeoutError) as failed:
        observe(deadline=145.)
    assert failed.value is refusal and len(calls) == 1
    assert calls[0]['timeout_seconds'] == 45.
    assert slept == ([5.] if spent == 40. else [])
    assert checked(failed.value.source384_warm_observation)['effective_seconds'] == 45.


def test_success_after_original_deadline_is_refused(clock, monkeypatch):
    monkeypatch.setattr(source, 'validate_source384_context', lambda **kwargs: clock[0].__setitem__(0, 146.))
    with pytest.raises(TimeoutError) as failed:
        observe(deadline=145.)
    assert checked(failed.value.source384_warm_observation)['status'] == 'failed'


@pytest.mark.parametrize('deadline', [None, True, float('nan'), float('inf'), 99., 701.])
def test_invalid_deadline_refused_before_validation(clock, monkeypatch, deadline):
    monkeypatch.setattr(source, 'validate_source384_context', lambda **kwargs: pytest.fail('validator entered'))
    with pytest.raises(ValueError):
        observe(deadline=deadline)


@pytest.mark.parametrize('profile,policy', [(None, recovery.POLICY), ('source384-5cpu-12gib@1', recovery.POLICY),
    (PROFILE, 'unknown'), (PROFILE, True), (PROFILE, '')])
def test_recovery_requires_explicit_profile_and_policy(clock, monkeypatch, profile, policy):
    monkeypatch.setattr(source, 'validate_source384_context', lambda **kwargs: pytest.fail('validator entered'))
    with pytest.raises(ValueError):
        observe(profile=profile, policy=policy)


@pytest.mark.parametrize('mutation', ['extra', 'policy', 'producer', 'count_bool', 'count_more', 'authority', 'attempt_extra'])
def test_closed_receipt_rejects_tampered_selection_or_bodies(clock, monkeypatch, mutation):
    monkeypatch.setattr(source, 'validate_source384_context', lambda **kwargs: None)
    result = observe()
    if mutation == 'extra': result['raw'] = 'excluded body'
    elif mutation == 'policy': result['policy'] = None
    elif mutation == 'producer': result['policy_source_sha256'] = '0' * 64
    elif mutation == 'count_bool': result['max_attempts'] = True
    elif mutation == 'count_more': result['max_attempts'] = 3
    elif mutation == 'authority': result['inference_replayed'] = True
    else: result['attempts'][0]['raw_body'] = 'excluded body'
    with pytest.raises(ValueError):
        recovery.validate_observation(result, profile=PROFILE, policy=recovery.POLICY,
            producer_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest())


def test_caller_receipt_mutation_during_backoff_cannot_change_selection(clock, refusal, monkeypatch):
    now, slept = clock
    original = {'authored': {'selected': 'old'}}
    calls = []
    def validate(**kwargs):
        calls.append(kwargs['expected_receipt'])
        if len(calls) == 1:
            now[0] += 90.
            raise refusal
        assert kwargs['expected_receipt'] == {'authored': {'selected': 'old'}}
    def sleep(seconds):
        original['authored']['selected'] = 'new'
        slept.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    result = recovery.observe_warm_context(repository='authored', expected_receipt=original,
        profile=PROFILE, policy=recovery.POLICY, deadline_monotonic=280.)
    assert checked(result)['status'] == 'validated'
    assert original['authored']['selected'] == 'new' and calls[0] == calls[1]


def test_interrupted_backoff_preserves_wait_and_cancellation(clock, refusal, monkeypatch):
    now, slept = clock
    def validate(**kwargs):
        now[0] += 90.
        raise refusal
    cancelled = KeyboardInterrupt('authored cancellation')
    def sleep(seconds):
        now[0] += 2.
        raise cancelled
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt) as failed:
        observe()
    result = checked(failed.value.source384_warm_observation)
    assert failed.value is cancelled and result['backoff_seconds'] == 2.
    assert result['backoff_requested_seconds'] == 5. and len(result['attempts']) == 1


@pytest.mark.parametrize('mutation', ['error_type', 'renewed_timeout', 'omitted_elapsed', 'missing_backoff'])
def test_receipt_rejects_contradictory_retry_timing_or_class(clock, refusal, monkeypatch, mutation):
    now, slept = clock
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            now[0] += 90.
            raise refusal
        now[0] += 1.
    monkeypatch.setattr(source, 'validate_source384_context', validate)
    value = observe()
    if mutation == 'error_type': value['attempts'][0]['error_type'] = 'ValueError'
    elif mutation == 'renewed_timeout': value['attempts'][1]['timeout_seconds'] = 90.
    elif mutation == 'omitted_elapsed': value['elapsed_seconds'] = 95.
    else: value['backoff_seconds'] = 0.
    with pytest.raises(ValueError):
        checked(value)
