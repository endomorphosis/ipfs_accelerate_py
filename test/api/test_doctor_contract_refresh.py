"""Postpublication observations invalidate old local proofs without re-proving."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_contract_proof import _scoped, _prove
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_refresh as refresh
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_proof import persist_contract_world
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import ProgramWorldDatabase


def _workflow(scenario, tmp_path, *, prove=True):
    inputs, scoped = _scoped(scenario, tmp_path)
    if prove:
        report, proof = _prove(inputs, scoped, tmp_path / 'old-proof')
        assert proof is not None
    else:
        report, proof = {'status': 'unsupported'}, None
    catalog = persist_contract_world(scoped=scoped, analyses=[], proof_report=report,
        proof=proof, state=tmp_path / 'old-index', task_id=inputs['task_cid'])
    return {'repository': str(scenario['repository']), 'task_cid': inputs['task_cid'], 'analysis': scoped.report,
            'contract_index': catalog}, proof


@pytest.mark.parametrize('change', ['modified', 'deleted'])
def test_published_source_invalidates_actual_proof_and_hydrates_history(scenario, tmp_path, change):
    workflow, proof = _workflow(scenario, tmp_path)
    original_artifact = Path(workflow['contract_index']['artifact']).read_bytes()
    source = scenario['repository'] / 'answer.py'
    if change == 'modified':
        source.write_bytes(source.read_bytes() + b'\n# published change\n')
    else:
        source.unlink()
    state = tmp_path / 'successor'
    result = refresh.refresh_doctor_contract_index(repository=scenario['repository'],
        workflow=workflow, state=state)
    assert result['status'] == 'observed' and result['hydrated']
    assert result['changed_paths'] == ['answer.py']
    assert result['deleted_paths'] == (['answer.py'] if change == 'deleted' else [])
    assert result['active_receipt_ids'] == []
    assert result['invalidated_receipt_ids'] == [proof.content_id]
    assert result['historical_proof_receipt_ids'] == [proof.content_id]
    assert result['new_proof_receipts'] == result['provider_calls'] == 0
    assert result['observed_scope_root'] != result['parent_source_tree_id']
    assert result['whole_program_proved'] is result['whole_repository_freshness_checked'] is False
    assert result['completion_authority'] is result['publication_authority'] is False
    assert Path(workflow['contract_index']['artifact']).read_bytes() == original_artifact
    observed = ProgramWorldDatabase(state / 'contracts.duckdb').records_for_decision(
        task_id=workflow['task_cid'], operation='published_scoped_contract_invalidation')
    assert observed['n'] == 1
    assert observed['records'][0]['payload']['invalidated_receipt_ids'] == [proof.content_id]
    if os.environ.get('IPFS_ACCELERATE_RUN_LIVE_QUACK') == '1':
        assert result['world_record']['ducklake']['status'] == 'projected'
        assert result['metadata']['status'] == 'projected'


def test_unchanged_scope_does_not_claim_new_files_or_new_proofs(scenario, tmp_path):
    workflow, proof = _workflow(scenario, tmp_path)
    (scenario['repository'] / 'new_unindexed.py').write_text('raise RuntimeError("not executed")\n')
    result = refresh.refresh_doctor_contract_index(repository=scenario['repository'],
        workflow=workflow, state=tmp_path / 'unchanged')
    assert result['active_receipt_ids'] == [proof.content_id]
    assert result['invalidated_receipt_ids'] == []
    assert result['changed_paths'] == []
    assert set(result['source_hashes']) == {'answer.py'}
    assert result['scope_expanded'] is result['whole_repository_freshness_checked'] is False
    assert result['new_proof_receipts'] == 0


@pytest.mark.parametrize('tamper', ['artifact', 'analysis', 'inventory', 'task', 'repository'])
def test_prior_binding_tamper_refuses_successor(scenario, tmp_path, tamper):
    workflow, _ = _workflow(scenario, tmp_path, prove=False)
    if tamper == 'artifact':
        Path(workflow['contract_index']['artifact']).write_text('{}\n')
    elif tamper == 'analysis':
        workflow['analysis']['source_hashes']['answer.py'] = '0' * 64
    elif tamper == 'inventory':
        workflow['contract_index']['active_receipt_ids'] = ['fake-proof']
    elif tamper == 'task':
        workflow['task_cid'] = 'foreign-task'
    else:
        workflow['repository'] = str(tmp_path)
    state = tmp_path / 'refused'
    with pytest.raises(refresh.DoctorContractRefreshError):
        refresh.refresh_doctor_contract_index(repository=scenario['repository'], workflow=workflow, state=state)
    assert not state.exists()


def test_source_symlink_is_not_treated_as_a_missing_source(scenario, tmp_path):
    workflow, _ = _workflow(scenario, tmp_path, prove=False)
    source = scenario['repository'] / 'answer.py'
    source.unlink()
    source.symlink_to(tmp_path / 'nonexistent-outside-source')
    with pytest.raises(refresh.DoctorContractRefreshError, match='symlinked'):
        refresh.refresh_doctor_contract_index(repository=scenario['repository'],
            workflow=workflow, state=tmp_path / 'symlink-refused')


def test_concurrent_change_refuses_current_successor_receipt(scenario, tmp_path, monkeypatch):
    workflow, _ = _workflow(scenario, tmp_path, prove=False)
    capture = refresh._capture
    calls = 0
    def changed(repository, names):
        nonlocal calls
        calls += 1
        result = capture(repository, names)
        if calls == 1:
            source = repository / 'answer.py'
            source.write_bytes(source.read_bytes() + b'\n# changed during observation\n')
        return result
    monkeypatch.setattr(refresh, '_capture', changed)
    state = tmp_path / 'raced'
    with pytest.raises(refresh.DoctorContractRefreshError, match='during index reconstruction'):
        refresh.refresh_doctor_contract_index(repository=scenario['repository'], workflow=workflow, state=state)
    assert not state.exists()
