"""Actual router/socket and deployed argument wiring; provider and UID admission authored.

No external model call or kernel cross-UID qualification is claimed here.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as context_audit
from benchmarks.agent_supervisor.container_coding import terminal_source_view_observation as observer
from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
from benchmarks.agent_supervisor.container_coding.test_source_formalization_worker_deployment import _run as worker_run
from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import evidence, _typed_audit  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_edit_client as edit_client
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from test.api.test_worker_source_edit_broker import case, read_working  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401

MODE = 'mapped-span@1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _run_router(case, **options):
    service = case['service']
    return runner.run(prompt=case['prompt'], provider='codex_cli', model='pinned', timeout=1,
        max_output_tokens=128, semantic_repository=case['root'],
        container_boundary=case['boundary'], container_boundary_sha256='b'*64,
        source_view_profile=case['profile'], source_view_descriptor=service.descriptor_path,
        source_view_descriptor_sha256=service.descriptor_sha256, **options)


def _boundary(case, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(case['workspace'].parent / 'empty-codex-home'))
    monkeypatch.setattr(boundary, 'verify_container_worker_boundary', lambda **kwargs:
        dict(manifest_sha256='b'*64, owner_uid=os.getuid(), worker_uid=os.getuid()+1))


@pytest.mark.parametrize("metadata_mode", ["legacy", "common-bindings@1"])
@pytest.mark.parametrize('case', [('python-comments@1', False, MODE),
    ('python-comments@1', False, 'mapped-span@2')], indirect=True)
def test_selected_mode_reaches_shared_router_then_real_socket_edit_and_closed_observer(case, provider, monkeypatch, metadata_mode):
    _boundary(case, monkeypatch)
    state, calls = provider
    service = case['service']; edits = []
    mode = case['mode']
    edit_count = 2 if mode == 'mapped-span@2' else 1
    encoded = codec.encode_semantic_router_prompt(prompt=case['prompt'], repository=case['root'])
    def perform_edit():
        for ordinal in range(edit_count):
            response, edit = read_working(case, before=str(ordinal+1), after=str(ordinal+2))
            assert response['page']['view'] == 'compact'
            result = edit_client.edit_source(service.descriptor_path, service.descriptor_sha256, edit=edit)
            assert result['ok'] is True
            edits.append(result)
    state['after_generate'] = perform_edit
    output, receipt = _run_router(case, source_edit_mode=mode, semantic_metadata_view=metadata_mode)
    assert output == 'literal summary' and len(calls) == 1 and len(edits) == edit_count
    expected_router = encoded.provider_prompt + source.guidance(case['profile'], source_edit_mode=mode)
    expected_model, _ = runner.render_model_prompt(prompt=expected_router, purpose='coding',
        workspace=case['workspace'], semantic_transport=True)
    if metadata_mode == 'legacy':
        assert calls[0][0] == expected_model
    else:
        assert receipt['representation_displacement']['requested_mode'] == metadata_mode
    rebuilt, _, checks = context_audit._model_projection(rendered=case['prompt'], receipt=receipt,
        repository=case['root'])
    assert rebuilt == calls[0][0] and all(checks.values()), checks
    changed_mode = deepcopy(receipt)
    changed_mode['source_view_retrieval']['source_edit_mode'] = 'mapped-span@2' if mode == MODE else MODE
    if metadata_mode == 'legacy':
        _, _, changed_checks = context_audit._model_projection(rendered=case['prompt'], receipt=changed_mode,
            repository=case['root'])
        assert not all(changed_checks.values())
    else:
        with pytest.raises(ValueError):
            context_audit._model_projection(rendered=case['prompt'], receipt=changed_mode,
                repository=case['root'])
    for mutation in ('missing_mode', 'disabled_mode', 'unknown_mode', 'wrong_schema'):
        edited = deepcopy(receipt)
        observed = edited['source_view_retrieval']
        if mutation == 'missing_mode': observed.pop('source_edit_mode')
        elif mutation == 'disabled_mode': observed['source_edit_mode'] = 'disabled'
        elif mutation == 'unknown_mode': observed['source_edit_mode'] = 'unknown'
        else: observed['schema'] = 'worker-source-dispatch-observation@2'
        with pytest.raises(ValueError):
            context_audit._model_projection(rendered=case['prompt'], receipt=edited, repository=case['root'])
    assert case['sources']['mod.py'].decode() not in expected_model
    assert 'worker_source_edit_client.py' in expected_model
    dispatch = receipt['source_view_retrieval']
    assert dispatch['schema'] == 'worker-source-dispatch-observation@3'
    assert dispatch['source_edit_mode'] == mode and dispatch['requests_observed'] is False
    assert receipt['router_prompt_sha256'] == sha(expected_router.encode())
    assert receipt['model_prompt_sha256'] == sha(calls[0][0].encode())
    assert receipt['completion_authority'] is False
    assert source.DESCRIPTOR_ENV not in os.environ and source.DIGEST_ENV not in os.environ
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    receipt['phase'] = 'coding'
    report = dict(arm='full', worker_cleanup_returncode=0, provider_invocations=[receipt],
        source_view_profile=case['profile'], source_edit_mode=mode)
    with tempfile.TemporaryDirectory(prefix='mapped-observer-', dir=Path.home()) as name:
        parent = Path(name); parent.chmod(0o750)
        retained = parent / service.audit_path.name
        retained.write_bytes(source.canonical_json(audit)); retained.chmod(0o400)
        observed = observer.collect(parent=parent, scope={'task_alias': 'TASK-EDIT'}, report=report)
    assert observed['status'] == 'observed', observed
    assert observed['source_edit_counts'] == dict(prepared_mapping_pages=edit_count, delivered_mapping_pages=edit_count,
        prepared_confirmation_pages=edit_count, delivered_confirmation_pages=edit_count)
    assert observed['counts']['accepted_requests'] == 3 * edit_count
    assert observed['counts']['delivered_source_bytes'] < len(case['sources']['mod.py'])
    assert observed['owner_writes_performed'] is observed['task_validation_observed'] is False
    assert observed['provider_tokens'] is None and observed['total_token_savings_measured'] is False
    closed = source.canonical_json(observed)
    for forbidden in (b'ordinary fixture explanation', b'expected_view_preimage', b'replacement',
                      str(case['workspace']).encode(), service.descriptor['session_id'].encode()):
        assert forbidden not in closed


@pytest.mark.parametrize('failure', ['omitted_mode', 'unknown_mode', 'descriptor_mode', 'other_mode'])
def test_mismatched_edit_mode_refuses_before_shared_router_generation(case, provider, monkeypatch, failure):
    _boundary(case, monkeypatch)
    _, calls = provider
    if failure == 'descriptor_mode':
        descriptor = deepcopy(case['service'].descriptor); descriptor['source_edit_mode'] = 'disabled'
        monkeypatch.setattr(source, 'load_descriptor', lambda *a, **k: descriptor)
    with pytest.raises(ValueError):
        _run_router(case, **({} if failure == 'omitted_mode' else
            {'source_edit_mode': 'unknown' if failure == 'unknown_mode' else 'mapped-span@2' if failure == 'other_mode' else MODE}))
    assert calls == []


def _source_args():
    return ['--semantic-repository', '/app', '--source-view-profile', 'python-comments@1',
        '--source-view-descriptor', '/owner/source.json', '--source-view-descriptor-sha256', 'a'*64]


@pytest.mark.parametrize('mode', ['disabled', MODE, 'mapped-span@2'])
def test_deployed_worker_parser_forwards_selection_to_router_without_losing_pins(tmp_path, mode):
    result = worker_run(tmp_path, [*_source_args(), '--source-edit-mode', mode])
    assert result.returncode == 0, result.stderr.decode()
    argv = json.loads(result.stdout)['argv']
    for flag in ('--source-view-profile', '--source-view-descriptor', '--source-view-descriptor-sha256'):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag)+1] == _source_args()[_source_args().index(flag)+1]
    if mode != 'disabled':
        assert argv.count('--source-edit-mode') == 1 and argv[argv.index('--source-edit-mode')+1] == mode
    else:
        assert '--source-edit-mode' not in argv


@pytest.mark.parametrize('extra', [[], ['--purpose', 'planning'], ['--preflight'], ['--semantic-metadata-census-only']])
def test_deployed_worker_refuses_missing_or_non_coding_source_route(tmp_path, extra):
    args = [] if not extra else _source_args() + extra
    result = worker_run(tmp_path, [*args, '--source-edit-mode', MODE])
    assert result.returncode != 0 and not result.stdout


@pytest.mark.parametrize('mode', [MODE, 'mapped-span@2'])
def test_signed_implementation_command_preserves_mode_and_default_bytes(mode):
    options = dict(router=Path('/owner/router'), model='pinned', reasoning='high', timeout=60,
        semantic_repository=Path('/app'), source_view_profile='python-comments@1', doctor=None)
    old = implementation_argv(**options)
    disabled = implementation_argv(**options, source_edit_mode='disabled')
    selected = implementation_argv(**options, source_edit_mode=mode)
    assert old == disabled and '--source-edit-mode' not in old
    assert selected == old + ['--source-edit-mode', mode]
    with pytest.raises(ValueError):
        implementation_argv(**{**options, 'semantic_repository': None}, source_edit_mode=mode)


@pytest.mark.parametrize('mode', [MODE, 'mapped-span@2'])
def test_driver_cli_forwards_literal_selection(monkeypatch, tmp_path, capsys, mode):
    calls = []
    monkeypatch.setattr(driver, 'run', lambda **kwargs: calls.append(kwargs) or
        dict(task_completed=False, arm='full', seconds=0, provider_invocations=[]))
    monkeypatch.setattr(sys, 'argv', ['driver', '--instruction', str(tmp_path/'instruction'),
        '--state', str(tmp_path/'state'), '--arm', 'full', '--source-view-profile', 'python-comments@1',
        '--source-edit-mode', mode])
    assert driver.main() == 1
    assert calls[0]['source_edit_mode'] == mode and calls[0]['source_view_profile'] == 'python-comments@1'
    capsys.readouterr()


def _editing(evidence):
    audit, row = evidence[3:5]
    _typed_audit(audit, row)
    descriptor = audit['session']
    descriptor.update(schema=source.DESCRIPTOR_SCHEMA_V3, source_edit_mode=MODE,
        view_policy_binding_sha256=None, source_scope_sha256='6'*64)
    digest = sha(source.canonical_json(descriptor)); audit['descriptor_sha256'] = digest
    audit['schema'] = 'supervisor-worker-source-session@4'
    advice = source.guidance(descriptor['profile'], source_edit_mode=MODE).encode()
    row['source_view_retrieval'].update(schema='worker-source-dispatch-observation@3',
        descriptor_sha256=digest, source_edit_mode=MODE, view_policy_binding_sha256=None,
        source_scope_sha256='6'*64, guidance_sha256=sha(advice), guidance_bytes=len(advice))
    for record in audit['requests']:
        record.update(source_edit_page=False, source_edit_operation=None, owner_writes_performed=False)
    mapping = audit['requests'][0]
    mapping.update(source_edit_page=True, source_edit_operation='prepare', returned_source_bytes=0)
    confirm = dict(mapping, ordinal=4, source_edit_operation='confirm', response_bytes=1024,
        request_sha256='5'*64, response_sha256='6'*64)
    audit['requests'].append(confirm); audit.update(accepted_requests=4, response_bytes=3305)
    evidence[2]['source_edit_mode'] = MODE
    evidence[5]()


def test_edit_session_accounting_retains_all_bytes_and_pending_status(evidence):
    _editing(evidence)
    result = observer.collect(parent=evidence[0], scope=evidence[1], report=evidence[2])
    assert result['status'] == 'observed'
    assert result['source_edit_counts'] == dict(prepared_mapping_pages=1, delivered_mapping_pages=1,
        prepared_confirmation_pages=1, delivered_confirmation_pages=1)
    assert result['counts']['accepted_requests'] == 4 and result['counts']['response_bytes'] == 3305
    assert result['counts']['prepared_source_bytes'] == 0
    assert result['task_validation_observed'] is result['owner_writes_performed'] is False


@pytest.mark.parametrize('change', ['owner_write', 'missing_flag', 'source_overlap', 'candidate_overlap',
    'discovery_overlap', 'bad_operation', 'bool_operation', 'missing_mode', 'changed_mode',
    'old_observation', 'old_audit', 'unknown_field', 'missing_discovery_pair'])
def test_edit_session_refuses_mixed_or_unbound_accounting(evidence, change):
    _editing(evidence)
    audit, row = evidence[3:5]; record = audit['requests'][0]
    if change == 'owner_write': record['owner_writes_performed'] = True
    elif change == 'missing_flag': record.pop('source_edit_page')
    elif change == 'source_overlap': record['returned_source_bytes'] = 1
    elif change == 'candidate_overlap': record.update(candidate_context_page=True, returned_candidate_context_bytes=1)
    elif change == 'discovery_overlap': record.update(discovery_page=True, returned_discovery_rows=1)
    elif change == 'bad_operation': record['source_edit_operation'] = 'apply'
    elif change == 'bool_operation': record['source_edit_operation'] = True
    elif change == 'missing_mode': evidence[2].pop('source_edit_mode')
    elif change == 'changed_mode': row['source_view_retrieval']['source_edit_mode'] = 'disabled'
    elif change == 'old_observation': row['source_view_retrieval']['schema'] = 'worker-source-dispatch-observation@2'
    elif change == 'old_audit': audit['schema'] = 'supervisor-worker-source-session@3'
    elif change == 'missing_discovery_pair':
        for item in audit['requests']:
            item.pop('discovery_page'); item.pop('returned_discovery_rows')
    else: record['literal_source'] = 'PRIVATE_LITERAL'
    evidence[5]()
    result = observer.collect(parent=evidence[0], scope=evidence[1], report=evidence[2])
    assert result['status'] == 'unavailable' and result['counts'] is None
    assert b'PRIVATE_LITERAL' not in source.canonical_json(result)
