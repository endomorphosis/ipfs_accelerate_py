"""Expanded source grammar through actual indexed planning and native proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_repair_pipeline import (
    _git, container_umask,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import validate_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_doctor_alias_binding_coverage import SOURCE, DONOR, AFTER


def _prepared(tmp_path, *, donor=DONOR, task_data=False):
    root = tmp_path / 'repository'
    root.mkdir()
    (root / 'answer.py').write_text(SOURCE)
    (root / 'helpers.py').write_text(donor)
    if task_data:
        (root / 'settings.json').write_text('{"version": 1}\n')
    _git(root, 'init', '-q')
    _git(root, 'add', 'answer.py', 'helpers.py')
    if task_data:
        _git(root, 'add', 'settings.json')
    _git(root, '-c', 'user.name=Qualification', '-c', 'user.email=test@example.invalid',
         'commit', '-qm', 'authored documented Python alias mismatch')
    instruction = tmp_path / 'instruction.md'
    instruction.write_text('Repair the unique import alias mismatch in answer.py.\n')
    outputs = [dict(path='answer.py', effect='modify', media_type='text/x-python')]
    profile = dict(schema=profiles.SCHEMA,
        instruction_sha256=profiles.instruction_sha256(instruction.read_text()),
        input_paths=['answer.py', 'helpers.py'], outputs=outputs)
    if task_data:
        profile.update(schema=profiles.DATA_SCHEMA,
            input_paths=['answer.py', 'helpers.py', 'settings.json'],
            data_inputs=[{'path': 'settings.json', 'media_type': 'application/json'}])
    # Authored interpretation: this qualifies the join and proof, not learned
    # intent interpretation or a Terminal-Bench task reward.
    contract = _requirements(instruction, symbolic=True)
    for item in contract['requirements']:
        item['outputs'] = deepcopy(outputs)
    for item in contract['symbolic_operations']['operations']:
        item['outputs'] = deepcopy(outputs)
    contract = validate_intent_requirement_contract(contract, source_text=instruction.read_text())
    artifact = tmp_path / 'requirements.json'
    artifact.write_text(json.dumps(contract))
    state = tmp_path / 'state'
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, intent_requirement_contract=artifact, disable_intent_autoencoder=True)
    initial = prep.initial_context(state=state)
    assert initial['indexed_symbols'] == 2 and initial['provider_calls'] == 0
    planned = prep.plan(state, provider_callable=None)
    assert planned['qualified'], planned.get('failure')
    assert planned['planning_strategy'] == 'intent_symbolic' and planned['provider_calls'] == 0
    assert planned['goals'] == 2 and planned['tasks'] == 1
    assert prep.context(state=state)['initial_indexes_reused'] is True
    admission = json.loads((state / 'admission.json').read_text())
    task_cid, = planned['task_cids']
    return root, state, prepared, admission, task_cid


def test_richer_alias_grammar_publishes_real_proved_candidate_without_model(tmp_path):
    root, state, prepared, admission, task_cid = _prepared(tmp_path)
    with IntentRepository(state / 'intent.duckdb', install_schema=False) as intent:
        before = intent.get_task(task_cid)
    result = dispatch.prepare_terminal_doctor_dispatch(
        repository=root, state=state, admission=admission, task_cid=task_cid)
    assert result['status'] == 'candidate_ready', result
    assert result['provider_calls'] == 0 and result['route'] == 'doctor_candidate'
    proof = result['symbolic_capabilities']['proof']
    assert proof['local_contract_proof_reported'] is True and proof['whole_program_verified'] is False
    raw = Path(result['artifact']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['sha256']
    handoff = json.loads(raw)
    assert 'projected supplied parameters satisfy' in handoff['proof_scope']
    assert _git(root, 'show', handoff['candidate_commit'] + ':answer.py') == AFTER.strip()
    assert _git(root, 'diff', '--name-only', handoff['base_commit'], handoff['candidate_commit']) == 'answer.py'
    workspace = tmp_path / 'allocated'
    _git(root, 'worktree', 'add', '--detach', str(workspace), handoff['base_commit'])
    materialized = materialize_doctor_candidate(
        artifact=Path(result['artifact']), expected_sha256=result['sha256'], task_cid=task_cid,
        prompt=json.dumps({'objective_id': handoff['task_id']}), workspace=workspace)
    assert materialized['provider_calls'] == 0 and materialized['publication_authority'] is False
    assert (workspace / 'helpers.py').read_text() == DONOR
    subprocess.run(prepared['spec']['validations'][0]['argv'], cwd=workspace,
                   check=True, capture_output=True, timeout=10)
    # Additional authored behavior checks validate defaults and the positional
    # boundary after publication. These checks are not claimed as formal proof.
    subprocess.run(['python3', '-B', '-c',
        'from answer import answer; assert answer(3) == 6; assert answer(3, enabled=False) == 3'],
        cwd=workspace, check=True, capture_output=True, timeout=10)
    assert (root / 'answer.py').read_text() == SOURCE
    with IntentRepository(state / 'intent.duckdb', install_schema=False) as intent:
        assert intent.get_task(task_cid) == before


def test_signed_task_data_retains_explicit_unproved_consumer_contract(tmp_path):
    root, state, _, admission, task_cid = _prepared(tmp_path, task_data=True)
    result = dispatch.prepare_terminal_doctor_dispatch(
        repository=root, state=state, admission=admission, task_cid=task_cid)
    assert result['status'] == 'residual' and result['route'] == 'model_router'
    assert 'doctor_task_data_contract_unavailable' in result['reason_codes']
    assert result['provider_calls'] == 0 and result['completion_authority'] is False
    native = json.loads(Path(result['result_artifact']).read_text())
    assert native['source_hashes']['settings.json'] == hashlib.sha256(b'{"version": 1}\n').hexdigest()
    partition = native['source_partition']
    assert partition['program_paths'] == ['answer.py', 'helpers.py']
    assert {'path': 'settings.json', 'role': 'task_data',
            'sha256': native['source_hashes']['settings.json']} in partition['harness_support']
    assert native['plan_refill']['successors']
    assert not (root / '.runtime/doctor-handoffs').exists()


@pytest.mark.parametrize('failure', ['required_keyword', 'failed_prover', 'source_drift'])
def test_expanded_pipeline_never_publishes_unproved_or_stale_candidates(tmp_path, monkeypatch, failure):
    donor = DONOR.replace('enabled=True', 'enabled') if failure == 'required_keyword' else DONOR
    if failure == 'required_keyword':
        # Require an additional parameter absent from the actual signed call.
        donor = donor.replace('*, enabled', '*, missing, enabled')
    root, state, _, admission, task_cid = _prepared(tmp_path, donor=donor)
    if failure == 'failed_prover':
        solver, kernel = dispatch._installed_provers()
        monkeypatch.setattr(dispatch, '_installed_provers', lambda: (Path('/usr/bin/false'), kernel))
    if failure == 'source_drift':
        (root / 'helpers.py').write_text(DONOR.replace('value * scale', 'value + scale'))
        with pytest.raises(ValueError):
            dispatch.prepare_terminal_doctor_dispatch(
                repository=root, state=state, admission=admission, task_cid=task_cid)
    else:
        result = dispatch.prepare_terminal_doctor_dispatch(
            repository=root, state=state, admission=admission, task_cid=task_cid)
        assert result['status'] == 'residual' and result['route'] == 'model_router'
        assert result['provider_calls'] == 0 and result['completion_authority'] is False
    assert not (root / '.runtime/doctor-handoffs').exists()
    assert (root / 'answer.py').read_text() == SOURCE
