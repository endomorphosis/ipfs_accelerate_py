"""Explicit nonsecret budget-policy environment checks never mutate live actors."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as dispatcher

spec = importlib.util.spec_from_file_location(
    '_required_environment_fixture', Path(__file__).with_name('test_agent_supervisor_repair_dispatcher.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
rig = fixtures.rig
rotation_spec = importlib.util.spec_from_file_location(
    '_required_environment_rotation_fixture', Path(__file__).with_name('test_agent_supervisor_native_lease_rotation.py'))
rotation_fixtures = importlib.util.module_from_spec(rotation_spec)
rotation_spec.loader.exec_module(rotation_fixtures)
rotation = rotation_fixtures.rotation
KEY = 'FIXTURE_REFACTOR_LLM_BUDGET_POLICY'
VALUE = '/private-owned-fixture/policy.json'


@pytest.fixture
def child(monkeypatch):
    children = []
    monkeypatch.setenv(KEY, VALUE)

    def spawn(value=VALUE):
        env = {k: v for k, v in os.environ.items() if k != KEY}
        if value is not None:
            env[KEY] = value
        p = subprocess.Popen([sys.executable, '-c',
                              'import sys,time;print("ready",flush=True);time.sleep(30)'],
                             env=env, stdout=subprocess.PIPE, text=True)
        children.append(p)
        assert p.stdout.readline() == 'ready\n'
        identity = dispatcher.probe_process(p.pid)
        assert identity['alive']
        identity['identity_verified'] = True
        return p, identity

    yield spawn
    for p in children:
        p.terminate()
        p.wait(timeout=3)


def test_real_child_matching_policy_is_verified_without_values(child):
    p, identity = child()
    result = dispatcher._lane_environment(identity, (KEY, KEY))
    assert result['verified'] and result['required_keys'] == [KEY]
    assert p.poll() is None and VALUE not in json.dumps(result)


@pytest.mark.parametrize('value,reason', [(None, 'required_environment_missing'),
                                        ('/different-private-policy', 'required_environment_mismatch')])
def test_real_child_missing_or_different_policy_is_not_healthy(child, value, reason):
    p, identity = child(value)
    result = dispatcher._lane_environment(identity, (KEY,))
    assert not result['verified'] and result['reason'] == reason
    assert result['affected_keys'] == [KEY] and p.poll() is None
    assert VALUE not in json.dumps(result) and '/different-private-policy' not in json.dumps(result)


def test_configured_own_environment_drift_is_detected(child, monkeypatch):
    _, identity = child()
    monkeypatch.setenv(KEY, '/changed-dispatcher-policy')
    result = dispatcher._lane_environment(identity, (KEY,))
    assert result['reason'] == 'required_environment_mismatch'
    assert '/changed-dispatcher-policy' not in json.dumps(result)


def test_missing_expected_policy_is_typed_not_silently_disabled(child, monkeypatch):
    _, identity = child()
    monkeypatch.delenv(KEY)
    result = dispatcher._lane_environment(identity, (KEY,))
    assert result['reason'] == 'expected_environment_missing' and not result['verified']


def test_default_disabled_never_reads_an_environment(monkeypatch):
    monkeypatch.setattr(dispatcher, 'probe_process', lambda _: pytest.fail('unexpected process inspection'))
    assert dispatcher._lane_environment({'pid': 123}, ()) == {'enabled': False}


@pytest.mark.parametrize('keys', [('API_TOKEN',), ('PRIVATE_SECRET_VALUE',), ('bad_budget_policy',)])
def test_only_explicit_nonsecret_budget_policy_names_are_allowed(keys):
    result = dispatcher._lane_environment({'pid': 123}, keys)
    assert result['reason'] == 'invalid_required_key' and result['required_keys'] == []
    assert keys[0] not in json.dumps(result)


def test_stale_birth_identity_is_rejected_before_environment_read(child, monkeypatch):
    _, identity = child()
    identity['start_ticks'] += 1
    original = Path.read_bytes

    def read(path):
        assert path.name != 'environ', 'reused identity must not inspect a different process environment'
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    assert dispatcher._lane_environment(identity, (KEY,))['reason'] == 'lane_identity_changed'


@pytest.mark.parametrize('change', ['birth', 'uid', 'exec'])
def test_identity_change_after_read_rejects_apparent_match(child, monkeypatch, change):
    _, identity = child()
    original_probe = dispatcher.probe_process
    calls = []

    def probe(pid):
        result = original_probe(pid)
        calls.append(pid)
        if len(calls) == 2:
            if change == 'birth':
                result['start_ticks'] += 1
            elif change == 'exec':
                result['argv'] = ['/unrelated-exec']
        return result

    monkeypatch.setattr(dispatcher, 'probe_process', probe)
    if change == 'uid':
        uid_calls = []

        def uid():
            uid_calls.append(True)
            return os.stat('/proc/self').st_uid + (len(uid_calls) > 1)

        monkeypatch.setattr(dispatcher.os, 'getuid', uid)
    assert dispatcher._lane_environment(identity, (KEY,))['reason'] == 'lane_identity_changed'
    assert len(calls) == 2


def test_unreadable_environment_is_explicit_and_does_not_leak_exception(child, monkeypatch):
    _, identity = child()
    original = Path.read_bytes

    def read(path):
        if path.name == 'environ':
            raise PermissionError('this private diagnostic must not be published')
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    result = dispatcher._lane_environment(identity, (KEY,))
    assert result['reason'] == 'environment_unreadable'
    assert 'private diagnostic' not in json.dumps(result)


def test_duplicate_environment_key_is_not_a_verified_match(child, monkeypatch):
    _, identity = child()
    original = Path.read_bytes

    def read(path):
        return (KEY + '=' + VALUE + '\0' + KEY + '=' + VALUE + '\0').encode() if path.name == 'environ' else original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    assert dispatcher._lane_environment(identity, (KEY,))['reason'] == 'required_environment_mismatch'


def test_unverified_role_never_reads_environment(child, monkeypatch):
    _, identity = child()
    identity['identity_verified'] = False
    monkeypatch.setattr(dispatcher, 'probe_process', lambda _: pytest.fail('unexpected process inspection'))
    assert dispatcher._lane_environment(identity, (KEY,))['reason'] == 'lane_identity_unverified'


def test_observation_is_typed_and_preserves_other_gate_clocks(rig, child, monkeypatch):
    config, now, lane, live, probe = rig
    _, identity = child(None)
    config.required_lane_env = (KEY,)
    # Normal lane-role verification still runs before the separate /proc fence.
    original_helper = dispatcher._lane_environment

    def helper(process, keys):
        assert process['identity_verified']
        return original_helper(identity, keys)

    monkeypatch.setattr(dispatcher, '_lane_environment', helper)
    state = {}
    fixtures.status(lane, state='degraded', phase='gate_retry_wait', active_task_id='VOG-001',
                    gate_evidence_fingerprint='fixture', gate_recovery_needed=False, next_retry_at_ms=(now + 1000)*1000)
    result = dispatcher.tick(config, state, now=now, probe=probe)
    assert result['action'] != 'healthy'
    assert result['lane_environment_reviews'][0]['reason'] == 'required_environment_missing'
    assert any(item['kind'] == 'lane_environment_review_needed' for item in result['incidents'])
    assert VALUE not in json.dumps(result)
    before = copy.deepcopy(state['gate_observations'])
    second = dispatcher.tick(config, state, now=now + 5, probe=probe)
    assert result['signature'] == second['signature']
    assert state['gate_observations']['vog']['first_seen'] == before['vog']['first_seen']
    assert state['gate_observations']['vog']['signature'] == before['vog']['signature']


@pytest.mark.parametrize('value', [None, ''])
def test_missing_or_empty_dispatcher_policy_cannot_launch_a_repair(rig, monkeypatch, value):
    config, now, _, _, probe = rig
    config.required_lane_env = (KEY,)
    if value is None:
        monkeypatch.delenv(KEY, raising=False)
    else:
        monkeypatch.setenv(KEY, value)
    state = {}
    result = dispatcher.tick(config, state, now=now, probe=probe, apply=True,
                             worker=lambda *_: pytest.fail('invalid policy launched a repair'))
    assert result['action'] == 'lane_environment_configuration_error'
    assert not state.get('attempts') and not state.get('active_worker')


def test_cli_repeated_policy_keys_reach_config_without_values(rig, monkeypatch):
    config, _, _, _, _ = rig
    captured = []

    def tick(actual, state, **kwargs):
        captured.append(actual.required_lane_env)
        return {'action': 'observational'}

    monkeypatch.setattr(dispatcher, 'tick', tick)
    monkeypatch.setattr(dispatcher, '_notify_control_watchdog', lambda: None)
    assert dispatcher.main(['--manifest-path', str(config.manifest_path), '--repo-root', str(config.repo_root),
                            '--once', '--required-lane-env', KEY, '--required-lane-env', KEY]) == 0
    assert captured == [(KEY,)]


def test_cli_rejects_non_policy_key_before_control_files(rig, monkeypatch):
    config, _, _, _, _ = rig
    monkeypatch.setattr(dispatcher, 'singleton_lock', lambda _: pytest.fail('invalid policy acquired control'))
    with pytest.raises(SystemExit) as caught:
        dispatcher.main(['--manifest-path', str(config.manifest_path), '--required-lane-env', 'SECRET_TOKEN'])
    assert caught.value.code == 2


def test_configured_guard_preserves_real_watchdog_native_rotation_grace(rotation, monkeypatch):
    config, now, lane, live, probe, owner, fresh, spawn = rotation
    monkeypatch.setenv(KEY, VALUE)
    config.required_lane_env = (KEY,)
    state = {'attempts': {'earlier': {'count': 2}}, 'unresolved_signature': 'earlier',
             'next_dispatch_at': now + 500}
    first = dispatcher.tick(config, state, apply=True, now=now, probe=probe,
                            worker=lambda *_: pytest.fail('environment uncertainty bypassed native rotation grace'))
    assert first['action'] == 'native_lease_rotation_wait' and not first['incidents']
    assert first['lane_environment_reviews'][0]['reason'] == 'lane_identity_unverified'
    assert first['lanes'][0]['required_environment']['verified'] is False
    deadline = first['native_rotation_grace'][0]['deadline']
    assert deadline == pytest.approx(now + 180, abs=.001)
    assert state['attempts'] == {'earlier': {'count': 2}} and state['unresolved_signature'] == 'earlier'
    assert state['next_dispatch_at'] == now + 500 and owner.poll() is None
    state = json.loads(json.dumps(state))
    fresh(179)
    second = dispatcher.tick(config, state, apply=True, now=now + 179, probe=probe,
                             worker=lambda *_: pytest.fail('configured grace deadline changed'))
    assert second['action'] == 'native_lease_rotation_wait'
    assert second['native_rotation_grace'][0]['deadline'] == deadline
    fresh(181)
    third = dispatcher.tick(config, state, now=now + 181, probe=probe)
    assert not third['native_rotation_grace']
    assert fixtures.kinds(third) == {'lane_dead_pending'}


def test_configured_guard_does_not_replace_ordinary_crash_recovery(rotation, monkeypatch):
    config, now, lane, live, probe, owner, fresh, spawn = rotation
    monkeypatch.setenv(KEY, VALUE)
    config.required_lane_env = (KEY,)
    fixtures.status(lane, terminal_reason='unexpected_exit')
    result = dispatcher.tick(config, {}, now=now, probe=probe)
    assert result['action'] == 'would_dispatch'
    assert fixtures.kinds(result) == {'lane_dead_pending'}


def test_configured_guard_leaves_finished_idle_without_environment_repair(rotation, monkeypatch):
    config, now, lane, live, probe, owner, fresh, spawn = rotation
    monkeypatch.setenv(KEY, VALUE)
    config.required_lane_env = (KEY,)
    (config.repo_root / 'todo.md').write_text('## VOG-001 Done\n- Status: completed\n- Depends on:\n')
    result = dispatcher.tick(config, {}, apply=True, now=now, probe=probe,
                             worker=lambda *_: pytest.fail('finished lane environment repair'))
    assert result['action'] == 'healthy' and not result['incidents']
    assert result['lane_environment_reviews'][0]['reason'] == 'lane_identity_unverified'


def test_midread_identity_race_waits_without_dispatch_or_false_recovery(rig, monkeypatch):
    config, now, lane, live, probe = rig
    monkeypatch.setenv(KEY, VALUE)
    config.required_lane_env = (KEY,)
    monkeypatch.setattr(dispatcher, '_lane_environment', lambda process, keys:
                        {'enabled': True, 'verified': False, 'reason': 'lane_identity_changed',
                         'required_keys': list(keys), 'pid': process['pid'], 'start_ticks': process['start_ticks']})
    state = {'attempts': {'earlier': {'count': 2}}, 'unresolved_signature': 'earlier',
             'next_dispatch_at': now + 500}
    result = dispatcher.tick(config, state, apply=True, now=now, probe=probe,
                             worker=lambda *_: pytest.fail('identity race launched a repair'))
    assert result['action'] == 'lane_environment_identity_wait' and not result['incidents']
    assert state['unresolved_signature'] == 'earlier' and state['attempts']['earlier']['count'] == 2
    assert 'recovery_observed_at' not in state


def test_missing_expected_policy_still_blocks_during_normal_rotation(rotation, monkeypatch):
    config, now, lane, live, probe, owner, fresh, spawn = rotation
    monkeypatch.delenv(KEY, raising=False)
    config.required_lane_env = (KEY,)
    state = {}
    result = dispatcher.tick(config, state, apply=True, now=now, probe=probe,
                             worker=lambda *_: pytest.fail('missing expected policy launched a repair'))
    assert result['action'] == 'lane_environment_configuration_error'
    assert result['native_rotation_grace'] and not state.get('attempts')


@pytest.mark.parametrize('exit_at', ['environment_read', 'after_read_stat'])
def test_real_child_exit_during_environment_observation_cannot_dispatch(rig, child, monkeypatch, exit_at):
    config, now, lane, live, probe = rig
    p, identity = child()
    config.required_lane_env = (KEY,)
    original_helper = dispatcher._lane_environment
    original_read = Path.read_bytes
    original_stat = Path.stat
    stat_reads = []

    def exited():
        p.terminate()
        p.wait(timeout=3)
        return FileNotFoundError('allocated actor exited during private environment observation')

    def read(path):
        if path == Path('/proc') / str(p.pid) / 'environ' and exit_at == 'environment_read':
            raise exited()
        return original_read(path)

    def stat(path, *args, **kwargs):
        if path == Path('/proc') / str(p.pid):
            stat_reads.append(True)
            if len(stat_reads) == 2 and exit_at == 'after_read_stat':
                raise exited()
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(Path, 'stat', stat)
    monkeypatch.setattr(dispatcher, '_lane_environment', lambda process, keys: original_helper(identity, keys))
    state = {'attempts': {'earlier': {'count': 2}}, 'unresolved_signature': 'earlier',
             'next_dispatch_at': now + 500}
    result = dispatcher.tick(config, state, apply=True, now=now, probe=probe,
                             worker=lambda *_: pytest.fail('exited actor environment launched a repair'))
    assert p.poll() is not None and result['action'] == 'lane_environment_identity_wait'
    assert not result['incidents']
    assert result['lane_environment_reviews'][0]['reason'] == 'lane_identity_changed'
    assert state['attempts']['earlier']['count'] == 2 and state['unresolved_signature'] == 'earlier'
    assert 'recovery_observed_at' not in state
