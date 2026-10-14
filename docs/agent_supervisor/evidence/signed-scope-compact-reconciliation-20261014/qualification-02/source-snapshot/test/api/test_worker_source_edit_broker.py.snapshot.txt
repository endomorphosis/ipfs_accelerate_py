"""Real mapping/socket/client/write/refresh with authored UID admission.

These local tests replace container peer/descriptor custody only. They do not
claim actual cross-UID deployment or provider-native token usage.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_edit_client as edit_client
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    selection = getattr(request, 'param', ('python-comments@1', False))
    profile, shared = selection[:2]
    mode = selection[2] if len(selection) > 2 else 'mapped-span@1'
    extra_program = len(selection) > 3 and selection[3]
    root = tmp_path / 'canonical'; root.mkdir()
    raw = (b'def answer():\n    value = 1  # ordinary explanation\n    return value\n'
           + b'# ordinary fixture explanation retained privately by the owner\n' * 40)
    sources = {'mod.py': raw, 'public.txt': b'Authored public input.\n'}
    program_paths = ['mod.py']
    if extra_program:
        sources['helper.py'] = b'def bonus():\n    value = 4  # another private explanation\n    return value\n'
        program_paths.append('helper.py')
        program_paths.sort()
    for name, body in sources.items():
        (root / name).write_bytes(body)
    git(root, 'init', '-q'); git(root, 'add', '--all')
    git(root, '-c', 'user.name=Qualification', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline')
    workspace = tmp_path / 'allocated'
    git(root, 'worktree', 'add', '--detach', '-q', str(workspace))
    output = root / '.runtime/semantic'
    prepare_semantic_context(repository=root, paths=sorted(sources), program_paths=program_paths,
        required_raw_paths=['public.txt'], objective='Set the answer to two.', task_id='TASK-EDIT', output=output)
    artifact = output / 'worker-context.json'
    refs = build_text_context_references(artifact.read_text(), reference_prefix='semantic-context',
        kind='semantic-context', path=artifact.relative_to(root).as_posix(), repository_id='repo:test',
        tree_id='tree:test', required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id='repo:test', tree_id='tree:test', objective_id='TASK-EDIT',
        objective_revision='sha256:task', policy_id='policy:test', policy_revision='sha256:policy',
        caller='supervisor:test', stage='implementation', goal={'id': 'TASK-EDIT'},
        authority={'mode': 'candidate_only', 'completion_authority': False},
        scope={'allowed_paths': program_paths}, acceptance={'criteria': ['pending native validation']}, evidence=refs)
    prompt = render_context_capsule(compiled.capsule)
    monkeypatch.chdir(workspace)
    with tempfile.TemporaryDirectory(prefix='compact-edit-') as directory:
        boundary = Path(directory) / 'boundary.json'; boundary.write_text('authored boundary')
        (boundary.parent / 'worker-observations').mkdir(mode=0o750)
        monkeypatch.setattr(guard, '_owner_boundary', lambda _: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), 'b' * 64))
        monkeypatch.setattr(guard, '_protected_directory', lambda *a, **k: None)
        policy = SourceViewPolicy.from_mapping(profile=profile) if shared else None
        with_args = dict(view_policy=policy) if shared else dict(profile=profile)
        value = broker.WorkerSourceBroker(repository=root, workspace=workspace, boundary=boundary,
            prompt=prompt, timeout_seconds=90, source_edit_mode=mode, **with_args)
        monkeypatch.setattr(value, '_peer', lambda _: None)
        monkeypatch.setattr(source.transport, '_verify_peer', lambda *a: None)
        monkeypatch.setattr(source.transport, '_socket_identity', lambda *a: None)
        def load(path, digest, *, workspace=None):
            assert Path(path) == value.descriptor_path and digest == value.descriptor_sha256
            return source.validate_descriptor(json.loads(source.canonical_json(value.descriptor)))
        monkeypatch.setattr(source, 'load_descriptor', load)
        try:
            yield dict(service=value, root=root, workspace=workspace, sources=sources,
                profile=profile, shared=shared, mode=mode, prompt=prompt, boundary=boundary)
        finally:
            value.close()


def read_working(case, *, path='mod.py', before='1', after='2'):
    value = case['service']
    response = source.request_page(value.descriptor_path, value.descriptor_sha256,
        path=path, working=True, max_bytes=8192)
    assert response['ok'] and response['page']['exhausted']
    page = response['page']; raw = page['content'].encode()
    start = raw.index(('value = ' + before).encode()) + len(b'value = ')
    edit = dict(path=path, expected_workspace_generation=page['workspace_generation'],
        expected_source_sha256=page['source_sha256'], expected_view_sha256=page['view_sha256'],
        expected_manifest_sha256=page['manifest_sha256'], view_start=start, view_end=start+len(before.encode()),
        expected_view_preimage=before, replacement=after)
    return response, edit


@pytest.mark.parametrize('case', [('identity@1', False), ('python-comments@1', False),
    ('python-comments@1', True)], indirect=True)
def test_actual_socket_edit_preserves_hidden_source_and_refreshes(case, record_property):
    value = case['service']; response, edit = read_working(case)
    expected = case['sources']['mod.py'].replace(b'value = 1', b'value = 2', 1)
    result = edit_client.edit_source(value.descriptor_path, value.descriptor_sha256, edit=edit)
    assert result['ok'] and result['status'] == 'applied_validation_pending', result
    assert result['owner_postimage_verified'] is True and result['validation_pending'] is True
    assert result['source_sha256'] == sha(expected)
    assert result['workspace_generation'] != edit['expected_workspace_generation']
    assert (case['workspace'] / 'mod.py').read_bytes() == expected
    assert all((case['root'] / path).read_bytes() == raw for path, raw in case['sources'].items())
    assert (case['workspace'] / 'public.txt').read_bytes() == case['sources']['public.txt']
    subprocess.run([sys.executable, '-B', '-c',
        "import runpy; assert runpy.run_path('mod.py')['answer']() == 2"], check=True, capture_output=True)
    refreshed = source.request_page(value.descriptor_path, value.descriptor_sha256,
        path='mod.py', working=True)
    assert refreshed['ok'] and refreshed['page']['workspace_generation'] == result['workspace_generation']
    with pytest.raises(ValueError):
        value.views.working_sources(expected_workspace_generation=edit['expected_workspace_generation'])
    repeated = edit_client.edit_source(value.descriptor_path, value.descriptor_sha256, edit=edit)
    assert repeated['ok'] is False and repeated['write_attempted'] is False
    assert (case['workspace'] / 'mod.py').read_bytes() == expected
    model_read = source.canonical_json(source.render_page_for_model(response, include_edit_pins=True))
    visible = (source.guidance(case['profile'], source_edit_mode='mapped-span@1').encode(),
        model_read, source.canonical_json(edit), source.canonical_json(result))
    observation = dict(profile=case['profile'], shared=case['shared'], original_bytes=len(case['sources']['mod.py']),
        compact_content_bytes=len(response['page']['content'].encode()),
        guidance_bytes=len(visible[0]), read_response_bytes=len(visible[1]), edit_json_bytes=len(visible[2]),
        result_bytes=len(visible[3]), visible_component_bytes=sum(map(len, visible)),
        original_expansion_requests=0, original_body_returned_in_edit=False,
        real_socket=True, authored_uid_admission=True, external_model_calls=0,
        provider_token_usage_measured=False, full_supervisor_start_qualified=False,
        public_smoke_exit=0, validation_pending=True)
    record_property('compact_edit_observation', json.dumps(observation, sort_keys=True))
    value.close()
    audit = json.loads(value.audit_path.read_text())
    assert audit['service_stopped'] and audit['accounting_complete']
    assert sum(row.get('source_edit_operation') == 'confirm' for row in audit['requests']) == 1
    assert all(row['owner_writes_performed'] is False for row in audit['requests'])


@pytest.mark.parametrize('failure', ['stale_pin', 'other_source', 'peer', 'policy'])
def test_prepare_refuses_without_worker_write(case, monkeypatch, failure):
    value = case['service']; _, edit = read_working(case)
    if failure == 'stale_pin': edit['expected_view_sha256'] = '0' * 64
    elif failure == 'other_source': (case['workspace'] / 'public.txt').write_bytes(b'concurrent edit\n')
    elif failure == 'peer':
        def refuse(_): raise ValueError('authored peer refusal')
        monkeypatch.setattr(value, '_peer', refuse)
    else: value._session_binding['view_policy_binding_sha256'] = '0' * 64
    result = edit_client.edit_source(value.descriptor_path, value.descriptor_sha256, edit=edit)
    assert result['ok'] is False and result['write_attempted'] is False
    assert (case['workspace'] / 'mod.py').read_bytes() == case['sources']['mod.py']


def test_concurrent_other_file_after_write_is_not_confirmed(case, monkeypatch):
    value = case['service']; _, edit = read_working(case)
    original = edit_client._write_candidate
    def concurrent(**kwargs):
        outcome = original(**kwargs)
        (case['workspace'] / 'public.txt').write_bytes(b'concurrent edit\n')
        return outcome
    monkeypatch.setattr(edit_client, '_write_candidate', concurrent)
    result = edit_client.edit_source(value.descriptor_path, value.descriptor_sha256, edit=edit)
    assert result['ok'] is False and result['status'] == 'written_confirmation_unknown'
    assert result['write_attempted'] is True and result['automatic_retry_performed'] is False
    assert (case['workspace'] / 'mod.py').read_bytes() != case['sources']['mod.py']
    assert (case['root'] / 'mod.py').read_bytes() == case['sources']['mod.py']


@pytest.mark.parametrize('case', [
    ('python-comments@1', False, 'mapped-span@2', True),
    ('python-comments@1', True, 'mapped-span@2', True)], indirect=True)
def test_repeated_socket_edits_refresh_all_files_and_refuse_old_confirmation(case, monkeypatch):
    service = case['service']
    generations, proposals, results = [], [], []
    expected = dict(case['sources'])
    original_write = edit_client._write_candidate

    def write_after_replay_check(**kwargs):
        if proposals:
            request = edit_client._request(service.descriptor, operation='confirm',
                proposal_sha256=proposals[-1])
            stale = edit_client._exchange(service.descriptor_path, service.descriptor, request)
            assert not stale['ok']
        return original_write(**kwargs)

    monkeypatch.setattr(edit_client, '_write_candidate', write_after_replay_check)
    for path, before, after in [('mod.py', '1', '2'), ('helper.py', '4', '5'), ('mod.py', '2', '3')]:
        response, edit = read_working(case, path=path, before=before, after=after)
        if generations:
            assert response['page']['workspace_generation'] == generations[-1]
        service.candidate_context.page(artifact='summary',
            expected_workspace_generation=edit['expected_workspace_generation'])
        assert service.candidate_context._generation == edit['expected_workspace_generation']
        summary = json.loads(service.candidate_context._artifacts['summary'])
        assert summary['workspace_generation'] == edit['expected_workspace_generation']
        assert summary['source_hashes'] == {name: sha(body) for name, body in expected.items()}
        result = edit_client.edit_source(service.descriptor_path, service.descriptor_sha256, edit=edit)
        assert result['ok'] and result['validation_pending']
        assert result['workspace_generation'] not in generations + [edit['expected_workspace_generation']]
        assert result['owner_postimage_verified'] is True
        generations.append(result['workspace_generation'])
        proposals.append(result['proposal_sha256'])
        results.append(result)
        expected[path] = expected[path].replace(('value = ' + before).encode(), ('value = ' + after).encode(), 1)
        assert all((case['workspace'] / name).read_bytes() == body for name, body in expected.items())
        assert all((case['root'] / name).read_bytes() == body for name, body in case['sources'].items())
        assert service.candidate_context._generation is None
    assert len(set(proposals)) == 3
    subprocess.run([sys.executable, '-B', '-c',
        "import runpy; assert runpy.run_path('mod.py')['answer']() == 3; "
        "assert runpy.run_path('helper.py')['bonus']() == 5"], check=True, capture_output=True)
    with pytest.raises(ValueError):
        service.candidate_context.page(artifact='summary', expected_workspace_generation=generations[0])
    stale = source.request_page(service.descriptor_path, service.descriptor_sha256,
        path='mod.py', working=True, expected_workspace_generation=generations[0])
    assert not stale['ok']
    refreshed, _ = read_working(case, before='3', after='6')
    assert refreshed['page']['workspace_generation'] not in generations[:-1]
    service.close()
    audit = json.loads(service.audit_path.read_text())
    assert audit['service_stopped'] and audit['accounting_complete']
    assert sum(row.get('source_edit_operation') == 'prepare' for row in audit['requests']) == 3
    assert sum(row.get('source_edit_operation') == 'confirm' for row in audit['requests']) == 3
    assert all(row['owner_writes_performed'] is False for row in audit['requests'])


@pytest.mark.parametrize('case', [('python-comments@1', False, 'mapped-span@2')], indirect=True)
def test_repeated_mode_blocks_more_mapping_after_uncertain_matching_confirmation(case, monkeypatch):
    service = case['service']; _, edit = read_working(case)
    original_write = edit_client._write_candidate
    def concurrent(**kwargs):
        outcome = original_write(**kwargs)
        (case['workspace'] / 'public.txt').write_bytes(b'concurrent change\n')
        return outcome
    monkeypatch.setattr(edit_client, '_write_candidate', concurrent)
    result = edit_client.edit_source(service.descriptor_path, service.descriptor_sha256, edit=edit)
    assert result['status'] == 'written_confirmation_unknown' and result['write_attempted']
    monkeypatch.setattr(edit_client, '_write_candidate', original_write)
    _, next_edit = read_working(case, before='2', after='3')
    refused = edit_client.edit_source(service.descriptor_path, service.descriptor_sha256, edit=next_edit)
    assert not refused['ok'] and not refused['write_attempted'] and not refused['automatic_retry_performed']
    assert (case['workspace'] / 'mod.py').read_bytes() == case['sources']['mod.py'].replace(b'value = 1', b'value = 2', 1)


@pytest.mark.parametrize('case', [('python-comments@1', False, 'mapped-span@2')], indirect=True)
@pytest.mark.parametrize('failure', ['before_context', 'after_context', 'after_peer', 'send'])
def test_matching_confirmation_owner_refusal_or_send_failure_blocks_new_edits(case, monkeypatch, failure):
    service = case['service']; _, edit = read_working(case)
    original_context, original_peer, original_send = service._current_context, service._peer, source.send_frame
    def context(*, working=False, generation=None):
        if service._edit_confirmation_request is not None:
            if failure == 'before_context' and generation is None or failure == 'after_context' and generation is not None:
                raise ValueError('authored surrounding context failure')
        return original_context(working=working, generation=generation)
    def peer(channel):
        if failure == 'after_peer' and service._edit_confirmation_request is not None:
            raise ValueError('authored final peer failure')
        return original_peer(channel)
    def send(channel, value):
        if (failure == 'send' and value.get('ok') is True
                and value['page'].get('schema') == edit_client.CONFIRMED_SCHEMA):
            raise OSError('authored owner send failure')
        return original_send(channel, value)
    monkeypatch.setattr(service, '_current_context', context)
    monkeypatch.setattr(service, '_peer', peer)
    monkeypatch.setattr(source, 'send_frame', send)
    outcome = edit_client.edit_source(service.descriptor_path, service.descriptor_sha256, edit=edit)
    assert outcome['status'] == 'written_confirmation_unknown' and outcome['write_attempted']
    assert not outcome['automatic_retry_performed']
    monkeypatch.setattr(service, '_current_context', original_context)
    monkeypatch.setattr(service, '_peer', original_peer)
    monkeypatch.setattr(source, 'send_frame', original_send)
    _, next_edit = read_working(case, before='2', after='3')
    refused = edit_client.edit_source(service.descriptor_path, service.descriptor_sha256, edit=next_edit)
    assert not refused['ok'] and not refused['write_attempted']
    assert service._edit_confirmation_blocked is True
    assert (case['workspace'] / 'mod.py').read_bytes() == case['sources']['mod.py'].replace(b'value = 1', b'value = 2', 1)
