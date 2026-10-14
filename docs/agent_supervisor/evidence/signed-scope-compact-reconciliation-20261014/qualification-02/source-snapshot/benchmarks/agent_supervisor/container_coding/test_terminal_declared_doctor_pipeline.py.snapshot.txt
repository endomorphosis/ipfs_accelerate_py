"""Real local declared plan, indexing and bounded Doctor proof; no model fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import DECLARED_PLANNING_MODE
from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_repair_pipeline import (
    prepared_pipeline, container_umask, _git, SOURCE, ALIAS_SOURCE, ALIAS_AFTER, DONOR,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import materialize_doctor_candidate
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _write_evidence(name, value):
    selected = os.environ.get('SYMBOLIC_EFFICIENCY_EVIDENCE_DIR')
    if selected:
        destination = Path(selected); destination.mkdir(parents=True, exist_ok=True)
        (destination / (name + '.json')).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def _prepare(tmp_path, monkeypatch, *, alias=False, failed_prover=False):
    from ipfs_accelerate_py import llm_router
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail('An actual model entrypoint was reached by the declared Doctor pipeline')
    monkeypatch.setattr(llm_router, 'generate_text', forbidden)
    z3 = shutil.which('z3')
    assert z3, 'real Z3 required for local Doctor qualification'
    real_which = shutil.which
    monkeypatch.setattr(dispatch.shutil, 'which', lambda name: None if name == 'z3' else real_which(name))
    selected = tmp_path / 'unavailable-z3' if failed_prover else Path(z3)
    monkeypatch.setenv('DOCTOR_COMPOSITION_Z3', str(selected))
    assert dispatch._installed_provers()[0] == selected
    root, state, prepared = prepared_pipeline(tmp_path, symbolic=False, alias=alias,
        planning_mode=DECLARED_PLANNING_MODE)
    initial = prep.initial_context(state=state)
    assert initial['indexed_symbols'] == 2 and initial['provider_calls'] == 0
    planned = prep.plan(state, provider_callable=None)
    assert planned['qualified'], planned.get('failure')
    assert planned['planning_strategy'] == 'declared_contract'
    assert planned['provider_calls'] == 0 and planned['goals'] == 2 and planned['tasks'] == 1
    assert planned['completion_authority'] is False and planned['initial_indexed_context']
    bound = prep.context(state=state)
    assert bound['initial_indexes_reused'] is True
    admission = json.loads((state / 'admission.json').read_text())
    task_cid, = planned['task_cids']
    assert local.verify_local_benchmark_admission(admission, initial=True)['graph'].tasks[0].task_cid == task_cid
    return root, state, prepared, initial, planned, admission, task_cid, calls


@pytest.mark.parametrize('alias', [False, True], ids=['keyword', 'import-alias'])
def test_declared_plan_indexes_proves_and_materializes_without_model(tmp_path, monkeypatch, alias):
    root, state, prepared, initial, planned, admission, task_cid, calls = _prepare(tmp_path, monkeypatch, alias=alias)
    before_source = ALIAS_SOURCE if alias else SOURCE
    after_source = ALIAS_AFTER if alias else SOURCE.replace('count=2', 'amount=2')
    before_head = _git(root, 'rev-parse', 'HEAD')
    with IntentRepository(state / 'intent.duckdb', install_schema=False) as intent:
        before_task = intent.get_task(task_cid)
    result = dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state, admission=admission, task_cid=task_cid)
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
    workspace = tmp_path / 'allocated'
    _git(root, 'worktree', 'add', '--detach', str(workspace), handoff['base_commit'])
    observed = materialize_doctor_candidate(artifact=Path(result['artifact']), expected_sha256=result['sha256'],
        task_cid=task_cid, prompt=json.dumps({'objective_id': handoff['task_id']}), workspace=workspace)
    assert observed['status'] == 'candidate_materialized' and observed['provider_calls'] == 0
    assert observed['publication_authority'] is observed['completion_authority'] is False
    assert (workspace / 'bottle.py').read_text() == after_source
    if alias: assert (workspace / 'helpers.py').read_text() == DONOR
    assert _git(workspace, 'rev-parse', 'HEAD') == handoff['base_commit']
    subprocess.run(prepared['spec']['validations'][0]['argv'], cwd=workspace, check=True, capture_output=True, timeout=10)
    assert (root / 'bottle.py').read_text() == before_source
    assert _git(root, 'rev-parse', 'HEAD') == before_head
    with IntentRepository(state / 'intent.duckdb', install_schema=False) as intent:
        assert intent.get_task(task_cid) == before_task
    assert calls == []
    _write_evidence('declared-doctor-' + ('import-alias' if alias else 'keyword'), dict(
        schema='declared-doctor-local-qualification@1', authored_source=True,
        planning_mode=DECLARED_PLANNING_MODE, indexed_symbols=initial['indexed_symbols'],
        independent_admission_qualified=planned['qualified'], initial_indexes_reused=True,
        declared_planning_provider_calls=planned['provider_calls'], doctor_provider_calls=result['provider_calls'],
        materialization_provider_calls=observed['provider_calls'], actual_model_entry_calls=len(calls),
        local_contract_proof_reported=True, whole_program_verified=False,
        doctor_route=result['route'], materialization_status=observed['status'],
        installed_z3_selected_off_path=True, authored_public_validation_exit=0,
        canonical_head_and_source_unchanged=True, task_record_unchanged=True,
        publication_authority=False, completion_authority=False, live_benchmark_executed=False))


def test_declared_doctor_missing_selected_prover_retains_model_route_without_dispatch(tmp_path, monkeypatch):
    root, state, prepared, initial, planned, admission, task_cid, calls = _prepare(
        tmp_path, monkeypatch, failed_prover=True)
    result = dispatch.prepare_terminal_doctor_dispatch(repository=root, state=state, admission=admission, task_cid=task_cid)
    assert result['route'] == 'model_router' and result['status'] != 'candidate_ready'
    assert result['provider_calls'] == 0 and result.get('artifact') is None and calls == []
    assert (root / 'bottle.py').read_text() == SOURCE
    _write_evidence('declared-doctor-missing-prover', dict(schema='declared-doctor-local-qualification@1',
        authored_source=True, planning_mode=DECLARED_PLANNING_MODE, independent_admission_qualified=planned['qualified'],
        doctor_route=result['route'], doctor_status=result['status'], candidate_ready=False,
        installed_z3_selected_off_path=True, selected_prover_missing=True,
        actual_model_entry_calls=len(calls), doctor_provider_calls=result['provider_calls'],
        residual_model_route_was_not_dispatched=True, canonical_source_unchanged=True,
        whole_program_verified=False, completion_authority=False, live_benchmark_executed=False))
