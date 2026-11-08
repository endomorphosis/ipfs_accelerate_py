"""Unissued-template projection models; no native or recovery authority.

Only the selected projection function is exercised. The surrounding full
template validator is an explicit structural fixture; real cold/host/source
qualification and native operation remain separate gates.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as science


@pytest.fixture
def transition_case(tmp_path, monkeypatch):
    oldseed = tmp_path / 'old-seed'
    newseed = tmp_path / 'new-seed'
    oldseed.mkdir(mode=0o700)
    newseed.mkdir(mode=0o700)
    documents = []

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
        with path.open('xb') as stream:
            os.fchmod(stream.fileno(), 0o400)
            stream.write(body)
        documents.append(path)
        return {'path': str(path), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}

    def read(reference):
        body = Path(reference['path']).read_bytes()
        science.need(len(body) == reference['bytes'] and hashlib.sha256(body).hexdigest() == reference['sha256'],
                     'modeled_descriptor_changed')
        return json.loads(body)

    def keys(value, expected, reason):
        science.need(type(value) is dict and set(value) == set(expected), reason)

    monkeypatch.setattr(science, 'health_owner', lambda: SimpleNamespace(
        _document=read, _keys=keys, _integer=lambda value, lo=0, hi=2**63-1:
            type(value) is int and lo <= value <= hi))
    calls = []

    def structural_template_only(template, profile, kind):
        # Explicitly models the already-selected wrapper validator. The real
        # policy/argv/database/source comparison inside the subject is intact.
        assert template['schema'] == 'agent-supervisor/scientific-executor-cycle-template@1'
        assert kind == 'genuine_scoped_idle'
        calls.append((copy.deepcopy(template), copy.deepcopy(profile), kind))

    monkeypatch.setattr(science, 'validate_template', structural_template_only)
    transition = {
        'prior_sha256': '4662bd8fe6caf6f2c048e3d7cff7abb4d80e5c875b21e02e204b1b89371b8286',
        'target_sha256': 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'}
    driver = {'path': str(tmp_path / 'retained-H/native_embedded_observation_driver.py'),
              'bytes': 1, 'sha256': '1' * 64}
    database = str(tmp_path / 'uscode.db')
    olddb = {'path': database, 'bytes': 101, 'sha256': 'a' * 64}
    newdb = {'path': database, 'bytes': 102, 'sha256': 'b' * 64}
    profile = {'unit': {'argv': ['/model/python']}, 'driver_source': driver}
    prior = copy.deepcopy(profile)
    owner_ref = write(tmp_path / 'new-owner-commitment.json', {'source_only_modeled_owner': True})
    owner = {'scheduler_policy_transition': transition}
    owner_paths = [tmp_path / 'OW/ipfs_accelerate_py/__init__.py',
        tmp_path / 'OW/ipfs_accelerate_py/agent_supervisor/__init__.py',
        tmp_path / 'OW/ipfs_accelerate_py/agent_supervisor/rescue/__init__.py',
        tmp_path / 'OW/ipfs_accelerate_py/agent_supervisor/rescue/original_embedded_observation_owner.py']
    owner_pins = {str(path): {'path': str(path), 'bytes': 1, 'sha256': str(index + 2) * 64}
                  for index, path in enumerate(owner_paths)}
    prior['owner_source'] = {'path': str(tmp_path / 'retained-H/original_embedded_observation_owner.py'),
                             'bytes': 1, 'sha256': '6' * 64}
    profile['owner_source'] = owner_pins[str(owner_paths[-1])]

    def build(change=None):
        oldplan = {'schema': 'modeled-original-native-plan', 'task_scope': ['one'],
            'native_metadata_cli': {'attempt_directory': str(oldseed / 'attempt'),
                'observation_binding': {'database_preimage': olddb}, 'backend': 'same-retained-H'}}
        newplan = copy.deepcopy(oldplan)
        newplan['native_metadata_cli']['attempt_directory'] = str(newseed / 'attempt')
        newplan['native_metadata_cli']['observation_binding']['database_preimage'] = newdb
        oldplan_ref = write(oldseed / 'operation-plan.json', oldplan)
        newplan_ref = write(newseed / 'operation-plan.json', newplan)
        oldrequest = {'argv': ['/model/python', '-B', driver['path'], '--plan', oldplan_ref['path'],
                '--plan-sha256', oldplan_ref['sha256']],
            'attempt_directory': str(oldseed / 'attempt'), 'database_preimage': olddb,
            'expected_scheduler_policy_sha256': transition['prior_sha256'],
            'cpu_slots': 1, 'memory_mb': 1536, 'child_process_slots': 8,
            'storage_bytes': 384 * 1024**2, 'timeout_seconds': 300,
            'source_roots': ['/home/barberb/lift_coding/external/ipfs_datasets',
                             '/home/barberb/lift_coding/external/ipfs_accelerate'],
            'resource_ledger': str(science.ORIGINAL_RESOURCE_LEDGER),
            'scheduler_state_path': '/model/scheduler-state.json',
            'allowed_subprocess_argv': [],
            'source_sha256': {oldplan_ref['path']: oldplan_ref['sha256'], driver['path']: driver['sha256'],
                prior['owner_source']['path']: prior['owner_source']['sha256'], '/model/canonical-source.py': '7' * 64}}
        newrequest = copy.deepcopy(oldrequest)
        newrequest['argv'] = ['/model/python', '-B', driver['path'], '--plan', newplan_ref['path'],
                              '--plan-sha256', newplan_ref['sha256']]
        newrequest['attempt_directory'] = str(newseed / 'attempt')
        newrequest['database_preimage'] = newdb
        newrequest['expected_scheduler_policy_sha256'] = transition['target_sha256']
        newrequest['source_sha256'].pop(oldplan_ref['path'])
        newrequest['source_sha256'][newplan_ref['path']] = newplan_ref['sha256']
        newrequest['source_sha256'].update({path: pin['sha256'] for path, pin in owner_pins.items()})
        if change == 'wrong_plan_digest':
            newrequest['argv'][-1] = oldplan_ref['sha256']
        elif change == 'extra_interpreter_flag':
            newrequest['argv'].insert(1, '-O')
        elif change == 'extra_command_authority':
            newrequest['allowed_subprocess_argv'].append(['/model/unauthorized-command'])
        elif change is not None:
            raise AssertionError('unknown test-local model change')
        oldrequest_ref = write(oldseed / 'request.json', oldrequest)
        newrequest_ref = write(newseed / 'request.json', newrequest)
        oldtemplate = {'schema': 'agent-supervisor/scientific-executor-cycle-template@1',
            'kind': 'genuine_scoped_idle', 'seed_root': str(oldseed),
            'request_ref': oldrequest_ref, 'plan_ref': oldplan_ref, 'links': {'result': 'attempt/result.json'}}
        newtemplate = copy.deepcopy(oldtemplate)
        newtemplate.update(seed_root=str(newseed), request_ref=newrequest_ref, plan_ref=newplan_ref)
        oldtemplate_ref = write(oldseed / 'template.json', oldtemplate)
        newtemplate_ref = write(newseed / 'template.json', newtemplate)
        review = {'schema': 'scientific-native-owner-policy-transition-independent-review/v1',
            'accepted_explicit_owner_policy_transition': True, 'native_owner_source_commitment': owner_ref,
            'prior_template_ref': oldtemplate_ref, 'template_ref': newtemplate_ref,
            'scheduler_policy_transition': transition, 'runtime_or_task_completion_authority': False}
        review_ref = write(tmp_path / 'review.json', review)
        certificate = {'schema': 'agent-supervisor/scientific-native-owner-policy-transition@1',
            'native_owner_source_commitment': owner_ref, 'prior_template_ref': oldtemplate_ref,
            'template_ref': newtemplate_ref, 'scheduler_policy_transition': transition,
            'independent_review_ref': review_ref, 'runtime_activation_or_task_authority': False}
        certificate_ref = write(tmp_path / 'certificate.json', certificate)
        return dict(certificate=certificate, certificate_ref=certificate_ref, owner_ref=owner_ref, owner=owner,
            owner_pins=owner_pins, prior=prior, profile=profile,
            oldconfig={'idle_template_ref': oldtemplate_ref, 'catalogue': {'database_path': database}},
            config={'idle_template_ref': newtemplate_ref, 'catalogue': {'database_path': database}},
            oldplan_ref=oldplan_ref, newplan_ref=newplan_ref, oldrequest=oldrequest, newrequest=newrequest,
            documents=documents, calls=calls)

    return build


def project(case):
    return science._catalogue_race_template_transition(case['certificate_ref'], case['owner_ref'],
        case['owner'], case['owner_pins'], case['prior'], case['profile'], case['oldconfig'], case['config'])


def test_unissued_projection_accepts_genuine_new_plan_hash_without_mutating_either_request(transition_case):
    case = transition_case()
    assert case['oldplan_ref']['sha256'] != case['newplan_ref']['sha256']
    before = {str(path): path.read_bytes() for path in case['documents']}
    assert project(case) == case['certificate']
    assert {str(path): path.read_bytes() for path in case['documents']} == before
    assert case['newrequest']['argv'][-1] == case['newplan_ref']['sha256']
    assert case['oldrequest']['argv'][-1] == case['oldplan_ref']['sha256']
    assert len(case['calls']) == 2


def test_unissued_projection_refuses_an_unbound_new_plan_digest(transition_case):
    with pytest.raises(science.ExecutorHold, match='original_driver_argv_or_plan_digest_changed'):
        project(transition_case('wrong_plan_digest'))


def test_unissued_projection_refuses_extra_interpreter_flags(transition_case):
    with pytest.raises(science.ExecutorHold, match='original_driver_argv_or_plan_digest_changed'):
        project(transition_case('extra_interpreter_flag'))


def test_unissued_projection_refuses_additional_command_authority(transition_case):
    with pytest.raises(science.ExecutorHold, match='request_authority_changed'):
        project(transition_case('extra_command_authority'))
