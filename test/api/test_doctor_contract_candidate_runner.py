"""Pinned contract candidates remain isolated until native supervisor gates."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_candidate_runner as runner


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


def _save(candidate, request):
    candidate['artifact_cid'] = content_identity({k: v for k, v in candidate.items() if k != 'artifact_cid'})
    raw = json.dumps(candidate, sort_keys=True).encode()
    artifact = request['artifact']
    if artifact.exists():
        artifact.chmod(0o600)
    artifact.write_bytes(raw)
    artifact.chmod(0o444)
    request['expected_sha256'] = hashlib.sha256(raw).hexdigest()


@pytest.fixture
def candidate(tmp_path):
    repository = tmp_path / 'repository'
    repository.mkdir()
    _git(repository, 'init', '-q')
    (repository / 'source.py').write_bytes(b'value = 1\n')
    _git(repository, 'add', 'source.py')
    _git(repository, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
         'commit', '-qm', 'authored public source')
    baseline = _git(repository, 'rev-parse', 'HEAD')
    workspace = tmp_path / 'allocated'
    _git(repository, 'worktree', 'add', '--detach', str(workspace), baseline)
    edits = []
    for name, effect, raw in [('source.py', 'modify', b'value = 2\n'),
                              ('report.jsonl', 'create', b'{"kind":"authored-fixture"}\n')]:
        edits.append({'path': name, 'effect': effect,
            'before_sha256': hashlib.sha256(b'value = 1\n').hexdigest() if effect == 'modify' else None,
            'after_sha256': hashlib.sha256(raw).hexdigest(),
            'after_bytes_base64': base64.b64encode(raw).decode()})
    payload = {'schema': runner.SCHEMA, 'repository': str(repository), 'baseline_commit': baseline,
        'task_cid': 'task:authored', 'task_id': 'TASK', 'task_revision': 1,
        'manifest_cid': 'manifest:authored', 'proof_receipt_id': 'proof:fixture-only',
        'proof_scope': 'Authored worker fixture; no theorem qualification claimed.',
        'analysis_cid': 'analysis:authored', 'edits': edits,
        'permitted_outputs': [{'path': edit['path'], 'effect': edit['effect'], 'media_type': 'text/plain'} for edit in edits],
        'provider_calls': 0, 'publication_authority': False, 'completion_authority': False}
    request = {'artifact': tmp_path / 'candidate.json', 'task_cid': payload['task_cid'],
        'prompt': json.dumps({'objective_id': 'TASK'}), 'workspace': workspace}
    _save(payload, request)
    return repository, payload, request


def test_modify_and_create_materialize_without_canonical_edit_or_commit(candidate):
    repository, payload, request = candidate
    result = runner.materialize_doctor_contract_candidate(**request)
    assert result['status'] == 'candidate_materialized'
    assert result['provider_calls'] == 0
    assert result['publication_authority'] is result['completion_authority'] is False
    assert [row['write_mode'] for row in result['writes']] == ['atomic_replace', 'exclusive_create']
    for edit in payload['edits']:
        assert (request['workspace'] / edit['path']).read_bytes() == base64.b64decode(edit['after_bytes_base64'])
    assert (repository / 'source.py').read_bytes() == b'value = 1\n'
    assert not (repository / 'report.jsonl').exists()
    assert _git(repository, 'rev-parse', 'HEAD') == payload['baseline_commit']
    assert _git(request['workspace'], 'rev-parse', 'HEAD') == payload['baseline_commit']


@pytest.mark.parametrize('kind', ['exists', 'symlink', 'directory'])
def test_all_preimages_and_creations_checked_before_any_write(candidate, kind):
    repository, _, request = candidate
    target = request['workspace'] / 'report.jsonl'
    if kind == 'exists':
        target.write_text('preexisting')
    elif kind == 'symlink':
        target.symlink_to(repository / 'source.py')
    else:
        target.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        runner.materialize_doctor_contract_candidate(**request)
    assert (request['workspace'] / 'source.py').read_bytes() == b'value = 1\n'
    assert (repository / 'source.py').read_bytes() == b'value = 1\n'


@pytest.mark.parametrize('kind', ['digest', 'writable', 'task', 'prompt', 'canonical', 'linked-artifact'])
def test_pin_identity_and_immutable_artifact_fail_closed(candidate, tmp_path, kind):
    repository, _, request = candidate
    if kind == 'digest':
        request['expected_sha256'] = '0' * 64
    elif kind == 'writable':
        request['artifact'].chmod(0o644)
    elif kind == 'task':
        request['task_cid'] = 'foreign'
    elif kind == 'prompt':
        request['prompt'] = '{"objective_id":"foreign"}'
    elif kind == 'canonical':
        request['workspace'] = repository
    else:
        linked = tmp_path / 'linked'
        linked.symlink_to(request['artifact'])
        request['artifact'] = linked
    with pytest.raises((ValueError, OSError)):
        runner.materialize_doctor_contract_candidate(**request)
    assert (repository / 'source.py').read_bytes() == b'value = 1\n'
    assert (request['workspace'] / 'source.py').read_bytes() == b'value = 1\n'
    assert not (request['workspace'] / 'report.jsonl').exists()


@pytest.mark.parametrize('kind', ['extra-field', 'duplicate-edit', 'traversal', 'permission', 'authority', 'bytes', 'baseline'])
def test_closed_contract_rejects_even_digest_pinned_malformed_payload(candidate, kind):
    _, payload, request = candidate
    if kind == 'extra-field':
        payload['transaction_id'] = 'fabricated'
    elif kind == 'duplicate-edit':
        payload['edits'].append(dict(payload['edits'][0]))
    elif kind == 'traversal':
        payload['edits'][1]['path'] = '../escape'
    elif kind == 'permission':
        payload['permitted_outputs'][1]['effect'] = 'modify'
    elif kind == 'authority':
        payload['completion_authority'] = True
    elif kind == 'bytes':
        payload['edits'][1]['after_sha256'] = '0' * 64
    else:
        payload['baseline_commit'] = '0' * 40
    _save(payload, request)
    with pytest.raises(ValueError):
        runner.materialize_doctor_contract_candidate(**request)
    assert (request['workspace'] / 'source.py').read_bytes() == b'value = 1\n'
    assert not (request['workspace'] / 'report.jsonl').exists()


@pytest.mark.parametrize('kind', ['drift', 'symlink', 'hardlink'])
def test_modified_source_must_be_current_regular_single_link(candidate, tmp_path, kind):
    repository, _, request = candidate
    target = request['workspace'] / 'source.py'
    if kind == 'drift':
        target.write_bytes(b'value = 9\n')
    elif kind == 'symlink':
        target.unlink()
        target.symlink_to(repository / 'source.py')
    else:
        (tmp_path / 'extra-link').hardlink_to(target)
    with pytest.raises((ValueError, OSError)):
        runner.materialize_doctor_contract_candidate(**request)
    assert not (request['workspace'] / 'report.jsonl').exists()
    assert (repository / 'source.py').read_bytes() == b'value = 1\n'


def test_owner_inode_preserved_and_interrupted_write_has_no_success(candidate, monkeypatch):
    repository, _, request = candidate
    target = request['workspace'] / 'source.py'
    before = target.stat()
    monkeypatch.setattr(runner.os, 'geteuid', lambda: before.st_uid + 1)
    result = runner.materialize_doctor_contract_candidate(**request)
    assert result['writes'][0]['write_mode'] == 'preserved_owner_inode'
    assert target.stat().st_ino == before.st_ino
    target.write_bytes(b'value = 1\n')
    (request['workspace'] / 'report.jsonl').unlink()
    actual = runner.os.write
    calls = []
    def partial(fd, raw):
        if calls:
            raise OSError('authored interrupted write')
        calls.append(True)
        return actual(fd, raw[:3])
    monkeypatch.setattr(runner.os, 'write', partial)
    with pytest.raises(OSError, match='interrupted write'):
        runner.materialize_doctor_contract_candidate(**request)
    assert target.stat().st_ino == before.st_ino
    assert not (request['workspace'] / 'report.jsonl').exists()
    assert (repository / 'source.py').read_bytes() == b'value = 1\n'


def test_foreign_git_worktree_is_not_an_allocated_candidate(candidate, tmp_path):
    _, _, request = candidate
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    _git(foreign, 'init', '-q')
    with pytest.raises(ValueError, match='foreign repository'):
        runner.materialize_doctor_contract_candidate(**{**request, 'workspace': foreign})
    assert not (foreign / 'source.py').exists()
