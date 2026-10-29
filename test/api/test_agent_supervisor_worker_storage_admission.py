"""New worker storage admission preserves existing work and all retry budgets."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as d
from ipfs_accelerate_py.agent_supervisor.rescue import storage_pressure

spec = importlib.util.spec_from_file_location(
    '_worker_storage_rig', Path(__file__).with_name('test_agent_supervisor_repair_dispatcher.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
rig = fixtures.rig
MINIMUM = 2 * 1024**3


@pytest.fixture
def capacity(monkeypatch):
    free = [MINIMUM - 1]
    observed = []
    probes = []
    original = storage_pressure.check_storage_capacity
    durable = storage_pressure._durable_probe
    monkeypatch.setattr(storage_pressure.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=free[0], f_frsize=1))

    def probe(path):
        probes.append(str(path))
        return durable(path)

    def checked(paths, *, minimum_free_bytes):
        observed.append(minimum_free_bytes)
        return original(paths, minimum_free_bytes=minimum_free_bytes)

    monkeypatch.setattr(storage_pressure, '_durable_probe', probe)
    monkeypatch.setattr(d, 'check_storage_capacity', checked)
    monkeypatch.setattr(d, 'publish_emergency_status', lambda *_a, **_k: None)
    monkeypatch.setattr(d, 'clear_emergency_status', lambda *_a, **_k: None)
    return free, observed, probes


def test_low_capacity_does_not_reserve_attempts_cooldowns_or_launch(rig, capacity):
    config, now, lane, live, probe = rig
    free, observed, probes = capacity
    config.worker_minimum_free_bytes = MINIMUM
    live[200] = False
    state = {'attempts': {'earlier': {'count': 2, 'next_dispatch_at': now + 5}},
             'next_dispatch_at': now + 9, 'unresolved_signature': 'earlier'}
    before = copy.deepcopy(state)
    result = d.tick(config, state, apply=True, now=now, probe=probe,
                    worker=lambda *_: pytest.fail('low space launched a maintenance worker'))
    assert result['action'] == 'storage_blocked' and observed == [MINIMUM] and not probes
    assert result['storage_capacity']['observations'][0]['minimum_free_bytes'] == MINIMUM
    assert result['storage_capacity']['observations'][0]['free_bytes'] == MINIMUM - 1
    assert not state.get('spawn_intent') and not state.get('active_worker')
    for key in before:
        assert state[key] == before[key]


def test_exact_threshold_recovers_to_one_actual_worker_without_refunding_old_attempts(rig, capacity):
    config, now, lane, live, probe = rig
    free, observed, probes = capacity
    config.worker_minimum_free_bytes = MINIMUM
    config.worker_argv = [sys.executable, '-c', "from pathlib import Path;Path('actual-worker').write_text('started')"]
    config.worker_timeout = 3
    live[200] = False
    state = {'attempts': {'earlier': {'count': 2}}, 'unresolved_signature': 'earlier'}
    first = d.tick(config, state, apply=True, now=now, probe=probe)
    assert first['action'] == 'storage_blocked' and not (config.repo_root / 'actual-worker').exists()
    free[0] = MINIMUM
    second = d.tick(config, state, apply=True, now=now + 1, probe=probe)
    assert second['action'] == 'worker_finished' and second['worker']['exit_code'] == 0
    assert (config.repo_root / 'actual-worker').read_text() == 'started' and probes
    assert observed == [MINIMUM, MINIMUM]
    assert state['attempts']['earlier']['count'] == 2
    assert state['attempts'][second['signature']]['count'] == 1


def test_recovery_does_not_bypass_an_existing_cooldown(rig, capacity):
    config, now, lane, live, probe = rig
    free, observed, probes = capacity
    config.worker_minimum_free_bytes = MINIMUM
    live[200] = False
    signature = d.observe(config, {}, now=now, probe=probe)['signature']
    state = {'attempts': {signature: {'count': 2, 'next_dispatch_at': now + 10}},
             'next_dispatch_at': now + 20, 'unresolved_signature': signature}
    first = d.tick(config, state, apply=True, now=now, probe=probe)
    free[0] = MINIMUM
    second = d.tick(config, state, apply=True, now=now + 1, probe=probe,
                     worker=lambda *_: pytest.fail('restored capacity bypassed cooldown'))
    assert first['action'] == 'storage_blocked' and second['action'] == 'cooldown'
    assert state['attempts'][signature] == {'count': 2, 'next_dispatch_at': now + 10}
    assert state['next_dispatch_at'] == now + 20


@pytest.mark.parametrize('configured,manifest,expected', [(MINIMUM, MINIMUM * 2, MINIMUM * 2),
                                                        (1, 64 * 1024**2, 64 * 1024**2)])
def test_worker_margin_cannot_weaken_any_manifest_minimum(rig, capacity, configured, manifest, expected):
    config, now, lane, live, probe = rig
    free, observed, probes = capacity
    config.worker_minimum_free_bytes = configured
    value = json.loads(config.manifest_path.read_text())
    value['storage_minimum_free_bytes'] = manifest
    fixtures.write_json(config.manifest_path, value)
    free[0] = expected - 1
    result = d.tick(config, {}, now=now, probe=probe)
    assert result['action'] == 'storage_blocked' and observed == [expected]


def test_default_preserves_existing_64_mib_admission(rig, capacity):
    config, now, lane, live, probe = rig
    free, observed, probes = capacity
    assert config.worker_minimum_free_bytes == 64 * 1024**2
    assert d.WORKER_STORAGE_ADMISSION_VERSION == 1
    free[0] = 64 * 1024**2
    result = d.tick(config, {}, now=now, probe=probe)
    assert result['action'] == 'healthy' and observed == [64 * 1024**2] and probes


@pytest.mark.parametrize('expired', [False, True])
def test_actual_existing_worker_keeps_original_deadline_and_cleanup(rig, monkeypatch, expired):
    config, now, lane, live, probe = rig
    config.worker_minimum_free_bytes = MINIMUM
    monkeypatch.setattr(d, 'check_storage_capacity', lambda *_a, **_k: pytest.fail('active worker was subject to new admission'))
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], cwd=config.repo_root,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    identity = d.probe_process(child.pid)
    deadline = now - 1 if expired else now + 45
    active = {**identity, 'owned_session': True, 'started_at': now - 5, 'deadline': deadline}
    state = {'active_worker': active, 'attempts': {'earlier': {'count': 2}}}

    def actual(pid):
        return d.probe_process(pid) if pid == child.pid else probe(pid)

    try:
        result = d.tick(config, state, apply=True, now=now, probe=actual)
        assert state['attempts']['earlier']['count'] == 2
        if expired:
            assert result['action'] == 'adopted_worker_timed_out' and not state.get('active_worker')
            child.wait(timeout=3)
        else:
            assert result['action'] == 'worker_already_running' and child.poll() is None
            assert state['active_worker']['deadline'] == deadline
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)


@pytest.mark.parametrize('value', [1, 64 * 1024**2, MINIMUM, 2**63 - 1])
def test_cli_exact_positive_bounds_reach_config_without_writes(rig, monkeypatch, value):
    config, _, _, _, _ = rig
    captured = []

    def tick(config, state, **kwargs):
        captured.append(config.worker_minimum_free_bytes)
        return {'action': 'observational'}

    monkeypatch.setattr(d, 'tick', tick)
    monkeypatch.setattr(d, '_notify_control_watchdog', lambda: None)
    assert d.main(['--manifest-path', str(config.manifest_path), '--repo-root', str(config.repo_root),
                   '--once', '--worker-min-free-bytes', str(value)]) == 0
    assert captured == [value]


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'nan', 'True', str(2**63)])
def test_cli_invalid_bounds_refused_before_control_lock(rig, monkeypatch, value):
    config, _, _, _, _ = rig
    monkeypatch.setattr(d, 'singleton_lock', lambda *_: pytest.fail('invalid storage policy acquired control'))
    with pytest.raises(SystemExit) as caught:
        d.main(['--manifest-path', str(config.manifest_path), '--worker-min-free-bytes', value])
    assert caught.value.code == 2


@pytest.mark.parametrize('value', [0, -1, True, 1.5, None, 2**63])
def test_programmatic_invalid_policy_refuses_before_observation(tmp_path, value):
    with pytest.raises(ValueError, match='positive signed64-bit'):
        d.Config(tmp_path / 'manifest', tmp_path, tmp_path / 'state', worker_minimum_free_bytes=value)
