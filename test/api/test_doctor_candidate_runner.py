"""The actual proved Doctor candidate writes only its allocated native worktree."""
import base64
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import _prepare, _provers, SOURCE
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    execute_doctor_task_repair, prepare_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate


@pytest.fixture
def candidate(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    result = execute_doctor_task_repair(prepare_doctor_task_repair(**inputs))
    assert result['status'] == 'candidate_ready'
    workspace = tmp_path / 'allocated'
    subprocess.run(['git', '-C', str(scenario['repository']), 'worktree', 'add', '--detach',
        str(workspace), result['handoff']['base_commit']], check=True, capture_output=True)
    request = dict(artifact=Path(result['handoff_path']), expected_sha256=result['handoff_sha256'],
        task_cid=result['handoff']['task_cid'], prompt=json.dumps({'objective_id': result['handoff']['task_id']}),
        workspace=workspace)
    return inputs, result, request


def test_real_proved_candidate_materializes_without_publication_or_completion(candidate, scenario):
    inputs, result, request = candidate
    before = scenario['intent'].get_task(inputs['task_cid'])
    observed = materialize_doctor_candidate(**request)
    assert observed['status'] == 'candidate_materialized'
    assert observed['publication_authority'] is observed['completion_authority'] is False
    expected = base64.b64decode(result['handoff']['edits'][0]['after_bytes_base64'])
    assert (request['workspace'] / 'answer.py').read_bytes() == expected
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
    assert subprocess.check_output(['git', '-C', str(request['workspace']), 'rev-parse', 'HEAD'], text=True).strip() == result['handoff']['base_commit']
    assert subprocess.check_output(['git', '-C', str(request['workspace']), 'diff', '--name-only'], text=True).splitlines() == ['answer.py']
    assert scenario['intent'].get_task(inputs['task_cid']) == before


def test_bound_candidate_refuses_tamper_foreign_task_drift_and_symlinks(candidate, scenario, tmp_path):
    _, result, request = candidate
    workspace = request['workspace']
    target = workspace / 'answer.py'
    original = target.read_bytes()
    corrupt = tmp_path / 'corrupt-handoff.json'
    corrupt.write_bytes(request['artifact'].read_bytes() + b' ')
    linked = tmp_path / 'linked-handoff.json'
    linked.symlink_to(request['artifact'])
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    subprocess.run(['git', '-C', str(foreign), 'init', '-q'], check=True)
    for changed in (
        {'artifact': corrupt}, {'artifact': linked}, {'task_cid': 'foreign-task'},
        {'prompt': json.dumps({'objective_id': 'foreign-task'})},
        {'workspace': scenario['repository']}, {'workspace': foreign},
    ):
        with pytest.raises((ValueError, OSError, subprocess.CalledProcessError)):
            materialize_doctor_candidate(**{**request, **changed})
        assert target.read_bytes() == original
    target.write_bytes(original + b'\n# drift\n')
    with pytest.raises(ValueError, match='preimage drifted'):
        materialize_doctor_candidate(**request)
    assert target.read_bytes() == original + b'\n# drift\n'
    target.unlink()
    target.symlink_to(scenario['repository'] / 'answer.py')
    with pytest.raises(OSError):
        materialize_doctor_candidate(**request)
    assert target.is_symlink()
    assert (scenario['repository'] / 'answer.py').read_bytes() == original
    assert scenario['intent'].get_task(request['task_cid'])['status'] == 'ready'


def test_delegated_inode_is_preserved_and_partial_write_never_returns_success(candidate, scenario, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_candidate_runner as module
    _, result, request = candidate
    target = request['workspace'] / 'answer.py'
    before = target.stat()
    original = target.read_bytes()
    # Actual distinct-UID sticky-directory permissions are qualified in Docker.
    # Here select that same branch to exercise inode and interruption contracts.
    monkeypatch.setattr(module.os, 'geteuid', lambda: before.st_uid + 1)
    observed = module.materialize_doctor_candidate(**request)
    assert observed['write_mode'] == 'preserved_owner_inode'
    assert target.stat().st_ino == before.st_ino
    assert target.stat().st_uid == before.st_uid
    assert target.read_bytes() == base64.b64decode(result['handoff']['edits'][0]['after_bytes_base64'])
    target.write_bytes(original)
    actual_write = module.os.write
    calls = []
    def interrupted(fd, data):
        if calls:
            raise OSError('authored interrupted write')
        calls.append(True)
        return actual_write(fd, data[:5])
    monkeypatch.setattr(module.os, 'write', interrupted)
    with pytest.raises(OSError, match='interrupted write'):
        module.materialize_doctor_candidate(**request)
    assert calls and target.stat().st_ino == before.st_ino
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
    assert scenario['intent'].get_task(request['task_cid'])['status'] == 'ready'
