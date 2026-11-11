"""Focused test SOURCE for isolated continuation boundaries; NOT EXECUTED.

Temporary fake unit/profile inputs below exercise local control flow only.
They cannot qualify historical custody, an original preflight issuer, Native,
Main identity or the future independent current-health consumer.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as executor


class LocalData:
    """Explicit test double: never represents a genuine unit/owner identity."""
    @staticmethod
    def _number(value):
        return type(value) in (int, float) and math.isfinite(value)

    @staticmethod
    def descriptor(path, **_):
        raw = Path(path).read_bytes()
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    @staticmethod
    def _document(ref):
        return json.loads(Path(ref['path']).read_bytes())


def test_clock_does_not_renew_at_check_or_accept_expired_future_or_nonfinite(monkeypatch):
    monkeypatch.setattr(executor, 'health_owner', lambda: LocalData())
    profile = {'max_age_seconds': 300}
    value = dict(recorded_at=1000, expires_at=1300)
    executor._closed_capacity_clock(value, profile, now=1299.999)
    assert value == dict(recorded_at=1000, expires_at=1300)
    for now in (999, 1300, 1301, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            executor._closed_capacity_clock(value, profile, now=now)
    with pytest.raises(ValueError):
        executor._closed_capacity_clock(dict(recorded_at=1000, expires_at=1301), profile, now=1001)


def test_incident_historical_pins_include_actual200_old37_and_aeb4():
    pins = executor.CLOSED_CAPACITY_INCIDENT_EVIDENCE
    assert pins['failed_stdout_ref'] == (3217, '0d7b1fe831f9b71a3ab2460fe068693830d4fa53d99b5cb6db077b7710e1cc9b')
    assert pins['prior_health_receipt_ref'] == (12498, '2d7591b0f4e3212791b88e958e0db2976d1a1366c4673e52b30c220f8d8482cf')
    assert pins['cleanup_observation_ref'] == (10632, 'aeb4dfb49e6120842061a55db940359643792113437c54adf81a2462e66b152f')


@pytest.fixture
def claim_case(tmp_path, monkeypatch):
    root = tmp_path/'proof'; root.mkdir(mode=0o700)
    lock_path = root/'scientific-executor.lock'; lock_path.write_bytes(b''); lock_path.chmod(0o600)
    profile = dict(proof_root=str(root), receipt_path=str(root/'new-health.json'),
        unit={'argv': [], 'name': 'fake-test-unit.service'})
    grant_path, ack_path = executor._closed_capacity_paths(profile)
    profile['unit']['argv'] = ['python', executor.CLOSED_CAPACITY_FLAG, str(grant_path)]
    value = dict(prior_main_identity={'pid': 101}, prior_health_receipt_ref={'old': '37'},
        global_hold_ref={'old': 'global-held'}, failed_cycle_hold_ref={'old': '200-held'},
        prior_reconciliation_ref={'old': 'catalogue-consumed'}, cleanup_observation_ref={'old': 'aeb4'},
        capacity_preflight_ref={'fresh': 'closed'}, recorded_at=1000, expires_at=1300)
    grant_path.write_text(json.dumps(value)); grant_path.chmod(0o400)
    reference = LocalData.descriptor(grant_path)
    monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV, reference['sha256'])
    monkeypatch.setattr(executor, 'health_owner', lambda: LocalData())
    identity = dict(pid=202, birth_ticks=444, invocation_id='a'*32)
    monkeypatch.setattr(executor, 'require_main_process', lambda *_: identity)
    monkeypatch.setattr(executor, '_validate_closed_capacity_continuation', lambda *_: (200, []))
    monkeypatch.setattr(executor, '_closed_capacity_final_freshness', lambda *_: None)
    return root, profile, reference, ack_path, identity, lock_path


def test_consumption_exclusive_new_Main_bound_and_never_reassigned(claim_case):
    root, profile, reference, ack_path, identity, lock_path = claim_case
    with lock_path.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert executor.reconcile_closed_capacity_continuation(reference, {'new': 'profile'}, profile,
            {}, identity, executor_lock=lock) == (200, [])
        original = ack_path.read_bytes(); ack = json.loads(original)
        assert ack['consuming_main_identity'] == identity
        assert ack['spent_cycle_count'] == 200 and ack['next_cycle_number'] == 201
        assert ack['next_health_sequence'] == 1 and ack['generation_number'] == ack['max_generations'] == 2
        assert ack['initial_consecutive_closed_negatives'] == 1 and ack['max_consecutive_closed_negatives'] == 3
        assert ack['state'] == 'claimed_before_first_original_owner'
        assert ack['old_health_republished'] is ack['health_credit_for_failed_cycle'] is False
        # Even a caller bypassing the validator cannot replace the O_EXCL claim.
        with pytest.raises(FileExistsError):
            executor.reconcile_closed_capacity_continuation(reference, {'new': 'profile'}, profile,
                {}, identity, executor_lock=lock)
        assert ack_path.read_bytes() == original


def test_partial_claim_stays_spent_instead_of_automatic_reissue(claim_case):
    _, profile, reference, ack_path, identity, lock_path = claim_case
    ack_path.write_bytes(b'{partial-unknown')
    original = ack_path.read_bytes()
    with lock_path.open('rb') as lock:
        with pytest.raises(FileExistsError):
            executor.reconcile_closed_capacity_continuation(reference, {'new': 'profile'}, profile,
                {}, identity, executor_lock=lock)
    assert ack_path.read_bytes() == original


def test_wrong_lock_inode_and_wrong_actual_Main_refuse_before_claim(claim_case, tmp_path):
    _, profile, reference, ack_path, identity, lock_path = claim_case
    other = tmp_path/'other.lock'; other.write_bytes(b''); other.chmod(0o600)
    with other.open('rb') as lock:
        with pytest.raises(ValueError, match='executor_lock_not_retained'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, identity, executor_lock=lock)
    with lock_path.open('rb') as lock:
        with pytest.raises(ValueError, match='not_real_distinct_Main'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, {'pid': 303}, executor_lock=lock)
    assert not ack_path.exists()


def test_wrong_environment_pin_and_writable_grant_refuse(claim_case, monkeypatch):
    _, profile, reference, ack_path, identity, lock_path = claim_case
    monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV, '0'*64)
    with lock_path.open('rb') as lock:
        with pytest.raises(ValueError, match='explicit_unit_grant'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, identity, executor_lock=lock)
    assert not ack_path.exists()
    monkeypatch.setenv(executor.CLOSED_CAPACITY_SHA_ENV, reference['sha256'])
    Path(reference['path']).chmod(0o600)
    with lock_path.open('rb') as lock:
        with pytest.raises(ValueError, match='explicit_unit_grant'):
            executor.reconcile_closed_capacity_continuation(reference, {}, profile, {}, identity, executor_lock=lock)
    assert not ack_path.exists()


@pytest.fixture
def finite_loop(tmp_path, monkeypatch):
    root = tmp_path/'proof'; root.mkdir(mode=0o700)
    profile = dict(proof_root=str(root), receipt_path=str(root/'new-unissued-health.json'))
    config_ref = {'path': 'fake-config'}; profile_ref = {'path': 'fake-profile'}
    recovery_ref = {'path': 'fake-capacity-grant'}
    config = dict(max_cycles=202, max_proof_bytes=2**30, poll_seconds=1)
    data = {config_ref['path']: config, recovery_ref['path']: {'schema': executor.CLOSED_CAPACITY_SCHEMA}}
    class FakeHealth:
        _path = staticmethod(Path)
        _document = staticmethod(lambda ref: data[ref['path']])
        _profile = staticmethod(lambda _: profile)
    class Stop:
        is_set = staticmethod(lambda: False)
        wait = staticmethod(lambda _: False)
    monkeypatch.setattr(executor, 'health_owner', lambda: FakeHealth())
    monkeypatch.setattr(executor, 'load_configuration', lambda _: (config, profile_ref, profile))
    monkeypatch.setattr(executor, 'require_main_process', lambda *_: {'fake': 'Main'})
    monkeypatch.setattr(executor, 'catalogue_parser_reference', lambda _: {'fake': 'parser'})
    monkeypatch.setattr(executor, 'load_explicit_catalogue_parser', lambda _: object())
    monkeypatch.setattr(executor, 'reconcile_closed_capacity_continuation', lambda *_, **__: (200, []))
    monkeypatch.setattr(executor, 'check_proof_budget', lambda *_, **__: None)
    monkeypatch.setattr(executor, 'observe_native', lambda *_, **__: {'fake': 'native'})
    monkeypatch.setattr(executor, 'select_template', lambda *_: ({}, 'genuine_scoped_idle'))
    holds = []; failures = []
    monkeypatch.setattr(executor, 'persistent_hold', lambda *args: holds.append(args))
    monkeypatch.setattr(executor, '_recovery_failure', lambda *args: failures.append(args))
    return config_ref, recovery_ref, config, Stop(), holds, failures


@pytest.mark.parametrize('errors', [
    (executor.CapacityAdmissionDeferred, executor.CapacityAdmissionDeferred),
    (executor.SourceAdmissionDeferred, executor.CapacityAdmissionDeferred),
    (executor.CatalogueRaceDeferred, executor.SourceAdmissionDeferred),
])
def test_original200_counts_toward_shared_three_negative_bound(finite_loop, monkeypatch, errors):
    config_ref, recovery_ref, _, stop, holds, failures = finite_loop
    attempts = []
    def cycle(*args, **options):
        attempts.append((options['cycle_number'], args[5], options['capacity_continuation_ref']))
        raise errors[len(attempts)-1]('/fake/cycle/negative.json')
    monkeypatch.setattr(executor, 'run_cycle', cycle)
    result = executor.serve(config_ref, stop, recovery_ref=recovery_ref, catalogue_parser_ref={'fake': 'parser'})
    assert result == executor.HOLD_EXIT
    assert attempts == [(201, 1, recovery_ref), (202, 1, recovery_ref)]
    assert holds[-1][-1] == 'closed_negative_consecutive_defer_bound'
    assert failures[-1][1:3] == (202, 1)


def test_only_positive_cycle_advances_new_health_and_resets_negative_counter(finite_loop, monkeypatch):
    config_ref, recovery_ref, config, stop, holds, _ = finite_loop
    config['max_cycles'] = 203
    attempts = []
    def cycle(*args, **options):
        attempts.append((options['cycle_number'], args[5]))
        if options['cycle_number'] == 202:
            raise executor.CapacityAdmissionDeferred('/fake/cycle/negative.json')
    monkeypatch.setattr(executor, 'run_cycle', cycle)
    assert executor.serve(config_ref, stop, recovery_ref=recovery_ref,
        catalogue_parser_ref={'fake': 'parser'}) == executor.HOLD_EXIT
    assert attempts == [(201, 1), (202, 2), (203, 2)]
    assert holds[-1][-1] == 'finite_cycle_budget_exhausted'
