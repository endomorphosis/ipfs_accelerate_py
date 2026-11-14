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


@pytest.fixture
def idle_source_case(tmp_path, monkeypatch):
    """Real profile readers over files; ELF header is fixture DATA, never executed."""
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
    monkeypatch.setattr(science, 'health_owner', lambda: health)
    source_root = tmp_path / 'R'
    monkeypatch.setattr(science, 'ORIGINAL_NAMED_ROOTS',
                        (source_root / 'workspace/test-logs',))
    proof = tmp_path / 'proof'; proof.mkdir(mode=0o700)

    def write(path, value, compact=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): path.chmod(0o600)
        raw = value if type(value) is bytes else (json.dumps(value, sort_keys=True,
            separators=(',', ':') if compact else None) + '\n').encode()
        path.write_bytes(raw); path.chmod(0o400)
        return health.descriptor(path)

    roles = {key: write(tmp_path / (key + '.py'), b'# source role fixture\n')
             for key in ('main_entrypoint', 'health_module', 'owner_source', 'driver_source')}
    tool = write(tmp_path / 'python-fixture', b'\x7fELFfixture-not-an-executable')
    relative = ('ipfs_datasets_py/logic/backends/process.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py',
        'scripts/ops/legal_ir/run_owned_validation.py')
    extra = write(tmp_path / 'extra-source.py', b'# unrelated source fixture\n')
    before = [write(source_root / path, b'# before source fixture\n') for path in relative]
    original = dict(schema=health.PROFILE_SCHEMA, **roles,
        catalogue=dict(board_namespace='fixture:science', plan_root_cid='fixture:plan',
            repository_tree_id='fixture:tree', population_sha256='a' * 64,
            database_uuid='fixture:db', store_generation=1),
        unit=dict(name='fixture-science.service', argv=[tool['path'], '-B', roles['main_entrypoint']['path']],
            interpreter=tool['path'], working_directory=str(tmp_path),
            environment={'PYTHONDONTWRITEBYTECODE': '1'}),
        source_descriptors=[*roles.values(), tool, extra, *before], task_scope=['fixture:task'],
        max_age_seconds=120, max_cycle_seconds=300, proof_root=str(proof),
        receipt_path=str(proof / 'current.json'),
        cycle_contract=dict(plan_schema='fixture-plan/v1', plan_operation='fixture-idle'))
    prior_ref = write(tmp_path / 'prior.json', original)
    snapshot = write(tmp_path / 'prior-frozen.json', Path(prior_ref['path']).read_bytes())
    after = [write(source_root / path, b'# after source fixture\n') for path in relative]
    profile = copy.deepcopy(original); profile['source_descriptors'][-5:] = after
    value = dict(schema='agent-supervisor/scientific-idle-source-preparation@1',
        prior_profile_ref=prior_ref, prior_profile_snapshot_ref=snapshot,
        profile_ref=write(tmp_path / 'current-profile.json', profile),
        replacements=[dict(slot=6 + i, before_ref=old, after_ref=new)
                      for i, (old, new) in enumerate(zip(before, after))])
    spec_path = tmp_path / 'comparison-spec.json'

    def freeze():
        value['profile_ref'] = write(tmp_path / 'current-profile.json', profile)
        return write(spec_path, value)

    return dict(health=health, write=write, freeze=freeze, value=value,
                profile=profile, original=original, spec_path=spec_path, after=after)


def test_idle_source_preparation_uses_current_profile_reader_without_activation(idle_source_case, monkeypatch):
    c = idle_source_case; reference = c['freeze']()
    with pytest.raises(c['health'].ExecutorHealthError, match='descriptor_changed'):
        c['health']._profile(c['value']['prior_profile_ref'])
    before = {str(p): p.read_bytes() for p in reference_path_files(c['spec_path'].parent)}
    def forbidden(*args, **kwargs):
        raise AssertionError('preparation must not write, start processes or observe a manager')
    monkeypatch.setattr(science, 'put', forbidden)
    monkeypatch.setattr(science.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(c['health'], '_manager', forbidden)
    result = science.prepare_idle_source_replacement(reference)
    assert result['changed_source_count'] == 5 and result['source_count'] == 11
    assert result['raw_inverse_verified'] is True and result['current_source_profile_validated'] is True
    assert all(result[key] is False for key in ('coherent_handover_validated', 'runtime_qualified',
        'activation_ready', 'dispatch_ready', 'hold_removal_authorized', 'authority_granted'))
    assert result['config_template_history_unresolved'] is True
    assert {str(p): p.read_bytes() for p in reference_path_files(c['spec_path'].parent)} == before


def reference_path_files(root):
    return [path for path in root.rglob('*') if path.is_file()]


@pytest.mark.parametrize('mutation', ['sixth_source', 'role', 'unit', 'environment',
    'task_scope', 'limits', 'bool_limit', 'float_limit', 'missing_row', 'duplicate_row',
    'bool_slot', 'float_slot', 'reverse_rows', 'extra_spec', 'wrong_schema', 'stale_snapshot',
    'foreign_canonical_path', 'old_ref_bool', 'source_population', 'reordered_sources',
    'prior_noncompact', 'missing_spec_field', 'old_ref_float', 'old_ref_zero',
    'old_ref_missing', 'old_ref_extra', 'old_ref_sha_upper', 'old_ref_sha_short',
    'old_ref_path_type', 'old_ref_bytes_string'])
def test_idle_source_preparation_refuses_broader_or_unbound_changes(idle_source_case, mutation):
    c = idle_source_case; value = c['value']; profile = c['profile']
    if mutation == 'sixth_source':
        old = profile['source_descriptors'][5]
        profile['source_descriptors'][5] = c['write'](Path(old['path']), b'# sixth source\n')
    elif mutation == 'role': profile['owner_source'] = profile['driver_source']
    elif mutation == 'unit': profile['unit']['name'] = 'other.service'
    elif mutation == 'environment': profile['unit']['environment']['PYTHONUNBUFFERED'] = '1'
    elif mutation == 'task_scope': profile['task_scope'].append('fixture:foreign')
    elif mutation == 'limits': profile['max_cycle_seconds'] = 299
    elif mutation == 'bool_limit': profile['max_age_seconds'] = True
    elif mutation == 'float_limit': profile['max_age_seconds'] = 120.0
    elif mutation == 'missing_row': value['replacements'].pop()
    elif mutation == 'duplicate_row': value['replacements'][-1] = value['replacements'][0]
    elif mutation == 'bool_slot': value['replacements'][0]['slot'] = True
    elif mutation == 'float_slot': value['replacements'][0]['slot'] = 5.0
    elif mutation == 'reverse_rows': value['replacements'].reverse()
    elif mutation == 'extra_spec': value['activation_ready'] = True
    elif mutation == 'wrong_schema': value['schema'] = 'wrong/v1'
    elif mutation == 'stale_snapshot':
        value['prior_profile_snapshot_ref'] = c['write'](Path(value['prior_profile_snapshot_ref']['path']),
                                                       dict(c['original'], max_age_seconds=119))
    elif mutation == 'foreign_canonical_path':
        slot = value['replacements'][0]['slot']
        value['replacements'][0]['after_ref'] = c['write'](c['spec_path'].parent / 'foreign.py', b'# foreign\n')
        profile['source_descriptors'][slot] = value['replacements'][0]['after_ref']
    elif mutation.startswith('old_ref_'):
        old = value['replacements'][0]['before_ref']
        if mutation == 'old_ref_bool': old['bytes'] = True
        elif mutation == 'old_ref_float': old['bytes'] = float(old['bytes'])
        elif mutation == 'old_ref_zero': old['bytes'] = 0
        elif mutation == 'old_ref_missing': old.pop('path')
        elif mutation == 'old_ref_extra': old['extra'] = False
        elif mutation == 'old_ref_sha_upper': old['sha256'] = 'A' * 64
        elif mutation == 'old_ref_sha_short': old['sha256'] = 'a' * 63
        elif mutation == 'old_ref_path_type': old['path'] = 7
        elif mutation == 'old_ref_bytes_string': old['bytes'] = str(old['bytes'])
        else: raise AssertionError('unknown malformed historical descriptor')
        # Bind the malformed historical row into both old documents as well.
        # The negative must be structural, not just a crossed snapshot join.
        value['prior_profile_ref'] = c['write'](Path(value['prior_profile_ref']['path']), c['original'])
        value['prior_profile_snapshot_ref'] = c['write'](
            Path(value['prior_profile_snapshot_ref']['path']), c['original'])
    elif mutation == 'source_population': profile['source_descriptors'].append(profile['source_descriptors'][0])
    elif mutation == 'prior_noncompact':
        value['prior_profile_ref'] = c['write'](Path(value['prior_profile_ref']['path']), c['original'], compact=False)
        value['prior_profile_snapshot_ref'] = c['write'](
            Path(value['prior_profile_snapshot_ref']['path']), c['original'], compact=False)
    elif mutation == 'missing_spec_field': value.pop('prior_profile_snapshot_ref')
    else: profile['source_descriptors'][-2:] = reversed(profile['source_descriptors'][-2:])
    with pytest.raises((science.ExecutorHold, c['health'].ExecutorHealthError)):
        science.prepare_idle_source_replacement(c['freeze']())


@pytest.mark.parametrize('field', ['spec', 'snapshot', 'profile'])
def test_idle_source_preparation_requires_frozen_control_data(idle_source_case, field):
    c = idle_source_case; reference = c['freeze']()
    refs = dict(spec=reference, snapshot=c['value']['prior_profile_snapshot_ref'],
                profile=c['value']['profile_ref'])
    Path(refs[field]['path']).chmod(0o600)
    with pytest.raises(science.ExecutorHold, match='control_not_frozen'):
        science.prepare_idle_source_replacement(reference)


def test_idle_source_preparation_detects_stale_current_source(idle_source_case):
    c = idle_source_case; reference = c['freeze']()
    c['write'](Path(c['after'][0]['path']), b'# later source\n')
    with pytest.raises(c['health'].ExecutorHealthError, match='descriptor_changed'):
        science.prepare_idle_source_replacement(reference)


def test_idle_source_preparation_rechecks_source_after_first_profile_read(idle_source_case, monkeypatch):
    c = idle_source_case; reference = c['freeze'](); actual = c['health']._profile; calls = []
    def observed(ref):
        result = actual(ref); calls.append(ref)
        if len(calls) == 1:
            c['write'](Path(c['after'][0]['path']), b'# changed during preparation\n')
        return result
    monkeypatch.setattr(c['health'], '_profile', observed)
    with pytest.raises(c['health'].ExecutorHealthError, match='descriptor_changed'):
        science.prepare_idle_source_replacement(reference)
    assert len(calls) == 1


# The metadata CLI must remain separate from the service parser and its effects.
@pytest.mark.parametrize('argv_kind', ['list', 'tuple'])
def test_idle_source_preparation_cli_real_profile_has_no_service_effects(idle_source_case, monkeypatch, capsys, argv_kind):
    c = idle_source_case
    reference = c['freeze']()
    def forbidden(*args, **kwargs):
        raise AssertionError('service or manager effect during preparation')
    monkeypatch.setattr(science, 'serve', forbidden)
    monkeypatch.setattr(science.threading, 'Event', forbidden)
    monkeypatch.setattr(science.signal, 'signal', forbidden)
    monkeypatch.setattr(c['health'], '_manager', forbidden)
    monkeypatch.setenv('IPFS_ACCEL_SKIP_CORE', '0')
    files = reference_path_files(c['spec_path'].parent)
    before = {str(path): path.read_bytes() for path in files}
    argv = ['prepare-idle-source-replacement', '--spec', reference['path'],
            '--spec-sha256', reference['sha256']]
    assert science.main(tuple(argv) if argv_kind == 'tuple' else argv) == 0
    output = capsys.readouterr()
    assert output.err == ''
    result = json.loads(output.out)
    assert result['schema'] == 'agent-supervisor/scientific-idle-source-comparison@1'
    assert result['source_count'] == 11 and result['changed_source_count'] == 5
    assert result['current_source_profile_validated'] is True
    assert result['raw_inverse_verified'] is True
    assert all(result[key] is False for key in ('coherent_handover_validated',
        'runtime_qualified', 'activation_ready', 'dispatch_ready',
        'hold_removal_authorized', 'authority_granted'))
    assert {str(path): path.read_bytes() for path in reference_path_files(c['spec_path'].parent)} == before
    assert os.environ['IPFS_ACCEL_SKIP_CORE'] == '1'


@pytest.mark.parametrize('extras', [
    ['--config', '/unselected/config'], ['--config-sha256', '0' * 64],
    ['--known-nonstarted-recovery', '/unselected/grant'],
    ['--known-closed-idle-recovery', '/unselected/grant'],
    ['--bounded-generation-handover', '/unselected/grant'],
    ['--known-source-admission-recovery', '/unselected/grant'],
    ['--known-closed-catalogue-race-recovery', '/unselected/grant'],
    ['--known-closed-capacity-recovery', '/unselected/grant'],
    ['--catalogue-parser-source', '/unselected/parser'],
    ['--catalogue-parser-sha256', '0' * 64]])
def test_idle_source_preparation_cli_rejects_service_flags_before_effects(monkeypatch, extras):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('effect before argument refusal')
    monkeypatch.setattr(science, 'health_owner', forbidden)
    monkeypatch.setattr(science, 'prepare_idle_source_replacement', forbidden)
    monkeypatch.setattr(science, 'serve', forbidden)
    monkeypatch.setattr(science.threading, 'Event', forbidden)
    monkeypatch.setattr(science.signal, 'signal', forbidden)
    monkeypatch.setenv('IPFS_ACCEL_SKIP_CORE', 'unchanged')
    with pytest.raises(SystemExit) as error:
        science.main(['prepare-idle-source-replacement', '--spec', '/unselected/spec',
                      '--spec-sha256', '0' * 64, *extras])
    assert error.value.code == 2 and calls == []
    assert os.environ['IPFS_ACCEL_SKIP_CORE'] == 'unchanged'


@pytest.mark.parametrize('argv', [
    ['prepare-idle-source-replacement'],
    ['prepare-idle-source-replacement', '--spec', '/unselected/spec'],
    ['prepare-idle-source-replacement', '--spec-sha256', '0' * 64],
    ['serve'], ['serve', '--config', '/unselected/config'],
    ['serve', '--config-sha256', '0' * 64]])
def test_idle_source_preparation_cli_preserves_both_required_argument_contracts(monkeypatch, argv):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('effect before required argument refusal')
    monkeypatch.setattr(science, 'health_owner', forbidden)
    monkeypatch.setattr(science, 'serve', forbidden)
    monkeypatch.setattr(science, 'prepare_idle_source_replacement', forbidden)
    monkeypatch.setattr(science.threading, 'Event', forbidden)
    monkeypatch.setattr(science.signal, 'signal', forbidden)
    monkeypatch.setenv('IPFS_ACCEL_SKIP_CORE', 'unchanged')
    with pytest.raises(SystemExit) as error:
        science.main(argv)
    assert error.value.code == 2 and calls == []
    assert os.environ['IPFS_ACCEL_SKIP_CORE'] == 'unchanged'


def test_idle_source_preparation_cli_digest_refuses_before_API(idle_source_case, monkeypatch, capsys):
    c = idle_source_case; reference = c['freeze'](); calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError('API before exact digest join')
    monkeypatch.setattr(science, 'prepare_idle_source_replacement', forbidden)
    assert science.main(['prepare-idle-source-replacement', '--spec', reference['path'],
                         '--spec-sha256', '0' * 64]) == science.HOLD_EXIT
    output = capsys.readouterr()
    assert output.out == '' and calls == []
    assert json.loads(output.err)['reason'] == 'idle_source_preparation_cli_digest'


def test_idle_source_preparation_cli_requires_bytecode_guard_before_imports(monkeypatch, capsys):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('import before bytecode guard')
    monkeypatch.setattr(science.sys, 'dont_write_bytecode', False)
    monkeypatch.setattr(science, 'health_owner', forbidden)
    monkeypatch.setenv('IPFS_ACCEL_SKIP_CORE', 'unchanged')
    assert science.main(['prepare-idle-source-replacement', '--spec', '/unselected/spec',
                         '--spec-sha256', '0' * 64]) == science.HOLD_EXIT
    output = capsys.readouterr()
    assert output.out == '' and calls == []
    assert json.loads(output.err)['reason'] == 'preparation_requires_B_and_assertions'
    assert os.environ['IPFS_ACCEL_SKIP_CORE'] == 'unchanged'


@pytest.mark.parametrize('field', ['coherent_handover_validated', 'runtime_qualified',
    'activation_ready', 'dispatch_ready', 'hold_removal_authorized', 'authority_granted'])
def test_idle_source_preparation_cli_never_publishes_authority(idle_source_case, monkeypatch, capsys, field):
    c = idle_source_case; reference = c['freeze']()
    result = {key: False for key in ('coherent_handover_validated', 'runtime_qualified',
        'activation_ready', 'dispatch_ready', 'hold_removal_authorized', 'authority_granted')}
    result[field] = True
    monkeypatch.setattr(science, 'prepare_idle_source_replacement', lambda *_: result)
    assert science.main(['prepare-idle-source-replacement', '--spec', reference['path'],
                         '--spec-sha256', reference['sha256']]) == science.HOLD_EXIT
    output = capsys.readouterr()
    assert output.out == ''
    assert json.loads(output.err)['reason'] == 'idle_source_preparation_cli_authority'


def test_idle_source_preparation_cli_preserves_original_failure_reason(idle_source_case, monkeypatch, capsys):
    c = idle_source_case; reference = c['freeze']()
    failure = PermissionError('fixture: exact original permission denial')
    def refused(*args):
        raise failure
    monkeypatch.setattr(science, 'prepare_idle_source_replacement', refused)
    assert science.main(['prepare-idle-source-replacement', '--spec', reference['path'],
                         '--spec-sha256', reference['sha256']]) == science.HOLD_EXIT
    output = capsys.readouterr()
    assert output.out == ''
    assert json.loads(output.err)['reason'] == str(failure)
    assert json.loads(output.err)['automatic_retry_admitted'] is False


def test_idle_source_preparation_cli_bounds_stdout_before_publication(idle_source_case, monkeypatch, capsys):
    c = idle_source_case; reference = c['freeze']()
    result = {key: False for key in ('coherent_handover_validated', 'runtime_qualified',
        'activation_ready', 'dispatch_ready', 'hold_removal_authorized', 'authority_granted')}
    result['oversized_fixture_payload'] = 'x' * c['health'].MAX_JSON
    monkeypatch.setattr(science, 'prepare_idle_source_replacement', lambda *_: result)
    assert science.main(['prepare-idle-source-replacement', '--spec', reference['path'],
                         '--spec-sha256', reference['sha256']]) == science.HOLD_EXIT
    output = capsys.readouterr()
    assert output.out == ''
    assert json.loads(output.err)['reason'] == 'idle_source_preparation_cli_output_bound'
