"""Real preparation, indexed symbolic planning and Doctor candidate handoff.

Intent interpretation is explicitly authored. These cases qualify the runtime
join, not learned interpretation quality or a Terminal-Bench score.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import validate_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_source_partition import terminal_doctor_source_partition
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_doctor_task_workflow import SOURCE
from test.api.test_doctor_imported_alias_workflow import SOURCE as ALIAS_SOURCE, DONOR, AFTER as ALIAS_AFTER


@pytest.fixture(autouse=True)
def container_umask():
    # The deployed driver establishes this before preparation. Keep the same
    # owner-controlled, worker-readable storage boundary in this local join.
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def prepared_pipeline(tmp_path, *, symbolic=True, alias=False):
    root = tmp_path / 'repository'
    root.mkdir()
    (root / 'bottle.py').write_text(ALIAS_SOURCE if alias else SOURCE)
    if alias:
        (root / 'helpers.py').write_text(DONOR)
    inputs = ['bottle.py', 'helpers.py'] if alias else ['bottle.py']
    _git(root, 'init', '-q')
    _git(root, 'add', *inputs)
    _git(root, '-c', 'user.name=Qualification', '-c', 'user.email=test@example.invalid',
         'commit', '-qm', 'authored public source')
    instruction = tmp_path / 'instruction.md'
    instruction.write_text('Repair the unique ' + ('import alias' if alias else 'keyword') + ' mismatch in bottle.py.\n')
    outputs = [dict(path='bottle.py', effect='modify', media_type='text/x-python')]
    profile = dict(schema=profiles.SCHEMA,
        instruction_sha256=profiles.instruction_sha256(instruction.read_text()),
        input_paths=inputs, outputs=outputs)
    # Reuse the reviewed fixture and bind its single operation to this exact
    # public output contract. No inferred formula authorizes its own meaning.
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
        task_profile=profile, intent_requirement_contract=artifact if symbolic else None,
        disable_intent_autoencoder=True)
    return root, state, prepared


@pytest.mark.parametrize('symbolic', [False, True])
def test_actual_preparation_acceptance_replays_in_doctor_partition(tmp_path, monkeypatch, symbolic):
    root, state, prepared = prepared_pipeline(tmp_path, symbolic=symbolic)
    provider = None
    if not symbolic:
        from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
        from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import _proposal_json
        _version(monkeypatch)
        proposal = json.loads(_proposal_json(prepared))
        proposal['tasks'][0]['predicted_files'] = ['bottle.py']
        provider = lambda *args, **kwargs: {'text': json.dumps(proposal), 'observation': {}, 'execution_receipt': None}
    planned = prep.plan(state, provider_callable=provider)
    assert planned['qualified'], planned.get('failure')
    # The direct arm uses one explicitly authored response fixture, never a
    # live model; the symbolic arm has no provider entry at all.
    assert planned['provider_calls'] == (0 if symbolic else 1)
    admission = json.loads((state / 'admission.json').read_text())
    task_cid, = planned['task_cids']
    assert prepared['spec']['acceptance'][0]['evidence_cids']
    partition = terminal_doctor_source_partition(repository=root, admission=admission, task_cid=task_cid)
    assert partition.program_paths == ('bottle.py',)
    partition.assert_current(root)


@pytest.mark.parametrize('alias', [False, True], ids=['keyword', 'import-alias'])
def test_indexed_symbolic_plan_reaches_real_doctor_proof_without_a_model(tmp_path, alias):
    root, state, prepared = prepared_pipeline(tmp_path, alias=alias)
    before_source = ALIAS_SOURCE if alias else SOURCE
    after_source = ALIAS_AFTER if alias else SOURCE.replace('count=2', 'amount=2')
    initial = prep.initial_context(state=state)
    assert initial['indexed_symbols'] == 2 and initial['provider_calls'] == 0
    planned = prep.plan(state, provider_callable=None)
    assert planned['qualified'], planned.get('failure')
    assert planned['planning_strategy'] == 'intent_symbolic'
    assert planned['provider_calls'] == 0 and planned['goals'] == 2 and planned['tasks'] == 1
    assert planned['initial_indexed_context'] and planned['requirement_coverage']['accepted']
    assert planned['completion_authority'] is False
    bound_context = prep.context(state=state)
    assert bound_context['initial_indexes_reused'] is True
    admission = json.loads((state / 'admission.json').read_text())
    task_cid, = planned['task_cids']
    with IntentRepository(state / 'intent.duckdb', install_schema=False) as intent:
        before = intent.get_task(task_cid)
    result = dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state,
        admission=admission, task_cid=task_cid)
    assert result['status'] == 'candidate_ready', result
    assert result['route'] == 'doctor_candidate' and result['provider_calls'] == 0
    assert result['symbolic_capabilities']['proof']['local_contract_proof_reported'] is True
    assert result['symbolic_capabilities']['proof']['whole_program_verified'] is False
    assert result['symbolic_capabilities']['operators']['selected_workflow'] == (
        'closed_imported_alias_call' if alias else 'closed_local_keyword_rename')
    raw = Path(result['artifact']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['sha256']
    handoff = json.loads(raw)
    assert _git(root, 'show', handoff['candidate_commit'] + ':bottle.py') == after_source.strip()
    assert _git(root, 'diff', '--name-only', handoff['base_commit'], handoff['candidate_commit']) == 'bottle.py'
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate
    workspace = tmp_path / 'allocated'
    _git(root, 'worktree', 'add', '--detach', str(workspace), handoff['base_commit'])
    observed = materialize_doctor_candidate(artifact=Path(result['artifact']), expected_sha256=result['sha256'],
        task_cid=task_cid, prompt=json.dumps({'objective_id': handoff['task_id']}), workspace=workspace)
    assert observed['status'] == 'candidate_materialized' and observed['provider_calls'] == 0
    assert observed['publication_authority'] is observed['completion_authority'] is False
    assert (workspace / 'bottle.py').read_text() == after_source
    if alias:
        assert (workspace / 'helpers.py').read_text() == DONOR
    assert _git(workspace, 'rev-parse', 'HEAD') == handoff['base_commit']
    subprocess.run(prepared['spec']['validations'][0]['argv'], cwd=workspace, check=True, capture_output=True, timeout=10)
    assert (root / 'bottle.py').read_text() == before_source
    with IntentRepository(state / 'intent.duckdb', install_schema=False) as intent:
        assert intent.get_task(task_cid) == before
    assert local.verify_local_benchmark_admission(admission, initial=True)['graph'].tasks[0].task_cid == task_cid
