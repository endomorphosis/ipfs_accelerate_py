"""An expired held adopted worker stays protected and explicitly overdue."""
import copy
import json
import os
from pathlib import Path
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as d

OVERDUE = ('reason', 'worker_deadline', 'worker_overdue_seconds', 'degraded')


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'discover_watchdogs', lambda *_: ([], []))
    root = tmp_path
    runtime = root / 'runtime'
    lane = runtime / 'lane'
    lane.mkdir(parents=True)
    now = time.time()
    manifest = runtime / 'bundle_lanes.json'
    manifest.write_text(json.dumps({'todo_path': 'todo.md', 'task_header_prefix': '## VOG-',
        'report_path': 'runtime/report.json',
        'lanes': [{'bundle_key': 'vog', 'state_dir': 'runtime/lane', 'state_prefix': 'vog'}]}))
    (root / 'todo.md').write_text('## VOG-001 Work\n- Status: pending\n- Depends on:\n')
    (runtime / 'watchdog.pid').write_text('100')
    (lane / 'vog_bundle_supervisor.pid').write_text('200')
    (lane / 'vog_status.json').write_text(json.dumps({'pid': 200, 'state': 'healthy',
        'phase': 'deterministic_command', 'active_task_id': 'VOG-001',
        'heartbeat_at_ms': now * 1000, 'last_task_transition_at_ms': now * 1000}))
    (runtime / 'report.json').write_text(json.dumps({'reports': [{'bundle_key': 'vog', 'action': 'none'}]}))
    (lane / 'supervisor.log').write_text('fixture only\n')
    config = d.Config(manifest, root, runtime / 'dispatcher', hold_paths=(lane / 'HOLD',),
                      repair_lock_path=runtime / 'fixer.lock', cooldown_seconds=10)
    birth = {'pid': 300, 'alive': True, 'start_ticks': 3000, 'owned_session': True,
             'pgid': 300, 'session_id': 300, 'started_at': now - 100, 'deadline': now - 1}
    marker = {'schema': d.SCHEMA, 'pid': 300, 'start_ticks': 3000, 'token': 'private-fixture'}
    config.repair_lock_path.write_text(json.dumps(marker))
    info = config.repair_lock_path.stat()
    birth['repair_marker'] = {'path': str(config.repair_lock_path), 'device': info.st_dev,
        'inode': info.st_ino, 'pid': 300, 'start_ticks': 3000, 'token': marker['token']}
    live = {100: True, 200: True, 300: True}

    def probe(pid):
        argv = (['python', '-m', 'ipfs_accelerate_py.agent_supervisor.rescue.lane_resume',
            '--repo-root', str(root), '--todo-path', 'todo.md', '--bundle-key', 'vog',
            '--status-path', str(lane / 'vog_status.json'),
            '--pid-path', str(lane / 'vog_bundle_supervisor.pid')] if pid == 200 else
            ['python', '-m', 'ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog',
             '--manifest-path', str(manifest), '--repo-root', str(root)])
        return {'pid': pid, 'alive': live.get(pid, False), 'age_seconds': 5000,
                'start_ticks': pid * 10, 'pgid': pid, 'session_id': pid,
                'argv': argv, 'cwd': str(root)}

    stopped = []
    def stop(identity):
        stopped.append(copy.deepcopy(identity))
        return True
    monkeypatch.setattr(d, 'stop_adopted_worker', stop)
    return config, now, lane, birth, live, probe, stopped


def hold(config, lane, kind):
    if kind == 'native_shaped':
        phase = config.repo_root / 'private-phase'
        phase.mkdir()
        marker = phase / 'hold-marker'
        marker.write_text('private native-checker observation fixture\n')
        path = lane / 'HOLD'
        os.link(marker, path)
    elif kind == 'manual':
        path = config.repo_root / 'HOLD'
        path.write_text('operator fixture\n')
    else:
        config.state_dir.mkdir()
        path = config.state_dir / 'hold'
        path.write_text('dispatch queue fixture\n')
    return path


def held_state(birth):
    return {'active_worker': copy.deepcopy(birth),
            'attempts': {'fixed-incident': {'count': 2, 'next_dispatch_at': 100}},
            'next_dispatch_at': 100}


def assert_no_overdue(value):
    assert all(key not in value for key in OVERDUE)


@pytest.mark.parametrize('kind', ['native_shaped', 'manual', 'queue'])
@pytest.mark.parametrize('apply', [False, True])
def test_expired_held_worker_is_explicit_without_signals_or_state_extension(rig, kind, apply):
    config, now, lane, birth, live, probe, stopped = rig
    marker = hold(config, lane, kind)
    identity = marker.stat()
    content = marker.read_bytes()
    state = held_state(birth)
    before = copy.deepcopy(state)
    value = d.tick(config, state, apply=apply, now=now, probe=probe)
    assert value['action'] == 'held' and value['held'] is True
    assert value.get('reason') == 'adopted_worker_deadline_exceeded_while_held'
    assert value.get('worker_deadline') == birth['deadline']
    assert value.get('worker_overdue_seconds') == 1
    assert value.get('degraded') is True
    assert value['completion_authority'] is value['work_complete'] is False
    assert not stopped
    assert state['active_worker'] == before['active_worker']
    assert state['attempts'] == before['attempts'] and state['next_dispatch_at'] == before['next_dispatch_at']
    after = marker.stat()
    assert (identity.st_dev, identity.st_ino, content) == (after.st_dev, after.st_ino, marker.read_bytes())
    assert json.loads((config.state_dir / 'status.json').read_text())['worker_deadline'] == birth['deadline']


@pytest.mark.parametrize('overdue', [0, 60, 86400])
def test_boundary_and_repeated_observation_retain_original_deadline(rig, overdue):
    config, now, lane, birth, live, probe, stopped = rig
    hold(config, lane, 'manual')
    birth['deadline'] = now
    state = held_state(birth)
    for seconds in (overdue, overdue + 3):
        value = d.tick(config, state, apply=True, now=now + seconds, probe=probe)
        assert value['action'] == 'held'
        assert value.get('reason') == 'adopted_worker_deadline_exceeded_while_held'
        assert value.get('worker_deadline') == now
        assert value.get('worker_overdue_seconds') == seconds
        assert value.get('degraded') is True
        assert state['active_worker']['deadline'] == now and not stopped


@pytest.mark.parametrize('kind', ['native_shaped', 'manual', 'queue'])
def test_held_worker_before_deadline_keeps_existing_behavior(rig, kind):
    config, now, lane, birth, live, probe, stopped = rig
    hold(config, lane, kind)
    birth['deadline'] = now + 100
    state = held_state(birth)
    value = d.tick(config, state, apply=True, now=now, probe=probe)
    assert value['action'] == 'worker_already_running' and value['held'] is True
    assert_no_overdue(value)
    assert state['active_worker']['deadline'] == now + 100 and not stopped


@pytest.mark.parametrize('apply', [False, True])
def test_expired_unheld_worker_retains_timeout_and_observational_semantics(rig, apply):
    config, now, lane, birth, live, probe, stopped = rig
    state = held_state(birth)
    value = d.tick(config, state, apply=apply, now=now, probe=probe)
    assert value['action'] == ('adopted_worker_timed_out' if apply else 'would_timeout_adopted_worker')
    assert len(stopped) == int(apply)
    assert bool(state.get('active_worker')) is not apply
    assert_no_overdue(value)


@pytest.mark.parametrize('apply', [False, True])
def test_missing_deadline_remains_attention_required_without_inventing_one(rig, apply):
    config, now, lane, birth, live, probe, stopped = rig
    hold(config, lane, 'manual')
    del birth['deadline']
    state = held_state(birth)
    value = d.tick(config, state, apply=apply, now=now, probe=probe)
    assert value['action'] == 'attention_required'
    assert value['reason'] == 'adopted_worker_missing_deadline'
    assert all(key not in value for key in ('worker_deadline', 'worker_overdue_seconds', 'degraded'))
    assert 'deadline' not in state['active_worker'] and not stopped


@pytest.mark.parametrize('invalid', ['missing_session_authority', 'marker_token_mismatch', 'marker_path_mismatch'])
def test_overdue_telemetry_grants_no_authority_for_malformed_saved_ownership(rig, invalid):
    config, now, lane, birth, live, probe, stopped = rig
    marker = hold(config, lane, 'native_shaped')
    if invalid == 'missing_session_authority':
        birth['owned_session'] = False
    elif invalid == 'marker_token_mismatch':
        birth['repair_marker']['token'] = 'untrusted-replacement'
    else:
        birth['repair_marker']['path'] = str(config.repo_root / 'unrelated-marker')
    state = held_state(birth)
    before = copy.deepcopy(state['active_worker'])
    value = d.tick(config, state, apply=True, now=now, probe=probe)
    assert value['action'] == 'held' and value.get('degraded') is True
    assert value.get('worker_deadline') == birth['deadline']
    assert state['active_worker'] == before and marker.exists() and not stopped
    assert config.repair_lock_path.exists()


@pytest.mark.parametrize('condition', ['dead', 'reused'])
def test_dead_or_reused_process_does_not_get_live_worker_overdue_telemetry(rig, condition):
    config, now, lane, birth, live, probe, stopped = rig
    hold(config, lane, 'manual')
    if condition == 'dead':
        live[300] = False
    def changed(pid):
        value = probe(pid)
        if pid == 300 and condition == 'reused':
            value['start_ticks'] += 1
        return value
    state = held_state(birth)
    value = d.tick(config, state, apply=True, now=now, probe=changed)
    assert value['action'] == 'held' and not state.get('active_worker') and not stopped
    assert_no_overdue(value)


@pytest.mark.parametrize('transition', ['hold_released', 'worker_exited'])
def test_overdue_details_do_not_linger_after_state_changes(rig, transition):
    config, now, lane, birth, live, probe, stopped = rig
    marker = hold(config, lane, 'manual')
    state = held_state(birth)
    first = d.tick(config, state, apply=True, now=now, probe=probe)
    assert first.get('reason') == 'adopted_worker_deadline_exceeded_while_held'
    if transition == 'hold_released':
        marker.unlink()
    else:
        live[300] = False
    second = d.tick(config, state, apply=True, now=now + 1, probe=probe)
    assert second['action'] == ('adopted_worker_timed_out' if transition == 'hold_released' else 'held')
    assert len(stopped) == int(transition == 'hold_released') and not state.get('active_worker')
    assert_no_overdue(second)
    assert_no_overdue(json.loads((config.state_dir / 'status.json').read_text()))
