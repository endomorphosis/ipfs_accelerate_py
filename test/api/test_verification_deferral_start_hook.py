import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.verification_deferral_start_hook import main


@pytest.mark.parametrize('hold_kind', ['inventory', 'fleet', 'HOLD', 'OPERATOR_STOP', 'broken_symlink', None])
def test_all_holds_and_live_owner_skip_loader(tmp_path, monkeypatch, hold_kind):
    root = tmp_path / 'repo'
    root.mkdir()
    runtime = root / 'runtime'
    runtime.mkdir()
    owner = runtime / 'owner.json'
    owner.write_text(json.dumps({'lifecycle': 'ready' if hold_kind is None else 'stopped'}))
    hold = runtime / (hold_kind or 'absent')
    if hold_kind == 'broken_symlink':
        hold.symlink_to(runtime / 'missing')
    elif hold_kind:
        hold.touch()
    board = {'id': 'spar', 'cwd': str(root), 'runtime_root': str(runtime), 'owner_status_path': str(owner),
             'hold_paths': [str(hold)] if hold_kind == 'inventory' else []}
    managed = {'id': 'spar', 'cwd': str(root), 'hold_files': [str(hold)] if hold_kind in ('fleet', 'broken_symlink') else []}
    inventory, fleet = tmp_path / 'inventory.json', tmp_path / 'fleet.json'
    inventory.write_text(json.dumps({'schema': 'ipfs_accelerate_py/taskboard-fleet-inventory@1', 'boards': [board]}))
    fleet.write_text(json.dumps({'schema': 'agent-supervisor/fleet-watchdog-config@1', 'boards': [managed]}))
    def forbidden(*args):
        raise AssertionError('held/live board dispatched recovery')
    monkeypatch.setattr('os.execv', forbidden)
    assert main(['--loader', str(tmp_path / 'nonexistent'), '--loader-sha256', 'unused', '--helper-sha256', 'unused',
                 '--source-root', str(root), '--inventory', str(inventory), '--fleet-config', str(fleet), '--board', 'spar']) == 0
