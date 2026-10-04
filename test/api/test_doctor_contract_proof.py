"""Real scoped sources, native proof checking, and observational lake hydration."""
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_header_contracts import PROGRAM, analyze
from test.api.test_doctor_task_workflow import _prepare, _provers

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.proof.proof_scope_index import (
    ProofScopeIndex, ProofScopeKey, ProofInputKind, update_proof_scope_index,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as proof_module
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_scoped_analysis import build_scoped_doctor_analysis
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import LocalPlanningError
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import ProgramWorldDatabase


def _scoped(scenario, tmp_path):
    inputs = _prepare(scenario, tmp_path, text=PROGRAM)
    scoped = build_scoped_doctor_analysis(repository=scenario['repository'],
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    return inputs, scoped


def _prove(inputs, scoped, state):
    inputs = _provers(inputs)
    candidate = analyze().candidate
    return proof_module.prove_header_contract(scoped=scoped, contract=candidate.contract,
        candidate_cid=content_identity(candidate.to_dict()), state=state,
        lean=inputs['kernel_executable'], z3=inputs['solver_executable'])


def test_actual_hammer_proves_local_guard_without_mutating_source_or_task(scenario, tmp_path):
    inputs, scoped = _scoped(scenario, tmp_path)
    before = scenario['intent'].get_task(inputs['task_cid'])
    report, proof = _prove(inputs, scoped, tmp_path / 'local-proof')
    assert report['status'] == 'proved_local_contract', report.get('reason_codes')
    assert proof is not None and proof.mutation_capable
    assert report['proof_receipt_id'] == proof.content_id
    assert report['provider_calls'] == 0
    assert report['whole_program_proved'] is False
    assert report['completion_authority'] is False
    assert report['proof_assumptions']
    assert report['candidate_path'] == 'answer.py'
    assert report['security_ir_bindings_verified'] is False
    assert report['security_ir_obligations_discharged'] is False
    assert report['implementation_bindings'] == proof_module._implementation_bindings()
    assert proof.roots.translator_id == 'translator:' + report['translator_cid']
    assert (scenario['repository'] / 'answer.py').read_text() == PROGRAM
    assert scenario['intent'].get_task(inputs['task_cid']) == before


def test_absent_prover_retains_explicit_unavailability_without_creating_state(scenario, tmp_path):
    _inputs, scoped = _scoped(scenario, tmp_path)
    state = tmp_path / 'missing-proof'
    candidate = analyze().candidate
    report, proof = proof_module.prove_header_contract(scoped=scoped, contract=candidate.contract,
        candidate_cid=content_identity(candidate.to_dict()), state=state,
        lean=Path('/unavailable/lean'), z3=Path('/unavailable/z3'))
    assert report['status'] == 'unavailable' and proof is None
    assert report['provider_calls'] == 0
    assert not state.exists()


def test_source_drift_refuses_before_prover_or_catalog_write(scenario, tmp_path):
    inputs, scoped = _scoped(scenario, tmp_path)
    (scenario['repository'] / 'answer.py').write_text(PROGRAM + '\n# changed\n')
    with pytest.raises(LocalPlanningError):
        _prove(inputs, scoped, tmp_path / 'stale-proof')
    with pytest.raises(LocalPlanningError):
        proof_module.persist_contract_world(scoped=scoped, analyses=[],
            proof_report={'status': 'unsupported'}, state=tmp_path / 'stale-world', task_id=inputs['task_cid'])
    assert not (tmp_path / 'stale-proof').exists()
    assert not (tmp_path / 'stale-world').exists()


def test_native_world_hydration_records_open_contract_without_active_proof(scenario, tmp_path):
    inputs, scoped = _scoped(scenario, tmp_path)
    state = tmp_path / 'open-world'
    result = proof_module.persist_contract_world(scoped=scoped,
        analyses=[{'status': 'unsupported', 'reason_codes': ['authored_unsupported_contract']}],
        proof_report={'status': 'unsupported'}, state=state, task_id=inputs['task_cid'])
    assert result['hydrated'] is True
    assert result['active_receipt_ids'] == []
    assert result['completion_authority'] is result['whole_program_proved'] is False
    observed = ProgramWorldDatabase(state / 'contracts.duckdb', state / 'contracts-lake').records_for_decision(
        task_id=inputs['task_cid'], operation='scoped_static_contracts')
    assert observed['n'] == 1
    record = observed['records'][0]['payload']
    assert record['analysis']['analysis_cid'] == scoped.report['analysis_cid']
    assert record['proof_status'] == 'unsupported'
    assert record['open_frontiers']
    assert record['proof_receipt_id'] is None
    assert record['completion_authority'] is record['whole_program_proved'] is False
    assert record['proof_index'] == json.loads(Path(result['artifact']).read_text())
    if os.environ.get('IPFS_ACCELERATE_RUN_LIVE_QUACK') == '1':
        assert result['world_record']['ducklake']['status'] == 'projected'
        assert result['world_record']['ducklake']['stored_records'] == 1
        assert result['metadata']['status'] == 'projected'
        assert result['metadata']['stored_catalogs'] == 2
        assert result['metadata']['stored_links'] == 2


def test_native_index_invalidates_local_proof_when_bound_source_changes(scenario, tmp_path):
    inputs, scoped = _scoped(scenario, tmp_path)
    report, proof = _prove(inputs, scoped, tmp_path / 'indexed-proof')
    assert proof is not None, report
    result = proof_module.persist_contract_world(scoped=scoped,
        analyses=[{'status': 'candidate', 'contract_cid': report['contract_cid']}],
        proof_report=report, proof=proof, state=tmp_path / 'proved-world', task_id=inputs['task_cid'])
    index = ProofScopeIndex.from_dict(json.loads(Path(result['artifact']).read_text()))
    assert index.active_receipt_ids == (proof.content_id,)
    for key in (
        ProofScopeKey(ProofInputKind.FILE, 'answer.py'),
        ProofScopeKey(ProofInputKind.POLICY, scoped.report['manifest_cid']),
        ProofScopeKey(ProofInputKind.PREMISE, report['contract_cid']),
        ProofScopeKey(ProofInputKind.PROGRAM_SNAPSHOT, report['candidate_cid']),
        ProofScopeKey(ProofInputKind.TOOLCHAIN, report['toolchain_cid']),
    ):
        changed = update_proof_scope_index(index, scope_blobs=index.blobs,
            obligations=index.obligations, receipts=index.receipts,
            root_id=index.root_id, changed_inputs=(key,))
        assert changed.active_receipt_ids == (), key
        assert any(item.subject_id == proof.content_id for item in changed.invalidations)
    assert report['whole_program_proved'] is False
    # A true abstract theorem cannot be rebound to different candidate bytes,
    # premises, or a substituted proof identity by editing report metadata.
    for field in ('candidate_cid', 'contract_cid', 'proof_receipt_id'):
        wrong = {**report, field: content_identity({'foreign': field})}
        state = tmp_path / ('rebound-' + field)
        with pytest.raises(ValueError, match='exact source-bound sealed native evidence'):
            proof_module.persist_contract_world(scoped=scoped, analyses=[], proof_report=wrong,
                proof=proof, state=state, task_id=inputs['task_cid'])
        assert not state.exists()


@pytest.mark.parametrize('status', ['whole_program_proved', 'proved_local_contract'])
def test_caller_assertions_cannot_create_active_proof(scenario, tmp_path, status):
    inputs, scoped = _scoped(scenario, tmp_path)
    state = tmp_path / 'asserted-proof'
    report = {'status': status, 'proof_receipt_id': content_identity({'claim': 'not proof'})}
    with pytest.raises(ValueError):
        proof_module.persist_contract_world(scoped=scoped, analyses=[], proof_report=report,
            state=state, task_id=inputs['task_cid'])
    assert not state.exists()


@pytest.mark.parametrize('kind', ['arbitrary', 'codepoints', 'source', 'assumption', 'extra',
                                  'candidate', 'foreign_candidate', 'ir_declaration_only',
                                  'ir_formalization_only', 'forged_ir_pair'])
def test_public_proof_helper_rejects_forged_contract_before_prover_lookup(scenario, tmp_path, kind):
    _inputs, scoped = _scoped(scenario, tmp_path)
    candidate = analyze().candidate
    contract = deepcopy(candidate.contract)
    candidate_cid = content_identity(candidate.to_dict())
    if kind == 'arbitrary':
        contract = {'claim': 'all arbitrary Python programs are secure'}
    elif kind == 'codepoints':
        contract['forbidden_codepoints'] = [10]
    elif kind == 'source':
        contract['source_sha256'] = '0' * 64
    elif kind == 'assumption':
        contract['assumptions'] = []
    elif kind == 'extra':
        contract['unreviewed_additional_claim'] = 'proved'
    elif kind == 'candidate':
        candidate_cid = content_identity({'unreviewed': 'candidate'})
    elif kind == 'foreign_candidate':
        candidate = analyze(PROGRAM.replace('clean_payload', 'other_payload')).candidate
        contract, candidate_cid = candidate.contract, content_identity(candidate.to_dict())
    elif kind == 'ir_declaration_only':
        contract['security_ir_declaration_cid'] = content_identity({'foreign': 'declaration'})
    elif kind == 'ir_formalization_only':
        contract['security_ir_formalization_cid'] = content_identity({'foreign': 'artifact'})
    else:
        contract.update(security_ir_declaration_cid=content_identity({'foreign': 'declaration'}),
                        security_ir_formalization_cid=content_identity({'foreign': 'artifact'}))
    state = tmp_path / 'forged-direct-proof'
    with pytest.raises(ValueError):
        proof_module.prove_header_contract(scoped=scoped, contract=contract,
            candidate_cid=candidate_cid, state=state,
            lean=Path('/unavailable/lean'), z3=Path('/unavailable/z3'))
    assert not state.exists()


def test_public_helper_does_not_select_first_of_two_independently_admitted_candidates(scenario, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptOutputRecord
    task = scenario['graph'].tasks[0]
    second = PromptOutputRecord(path='test_answer.py', effect='modify', media_type='text/x-python')
    scenario['graph'] = replace(scenario['graph'], tasks=(replace(task, outputs=(*task.outputs, second)),))
    scenario['manifest']['payload']['tasks'][0]['outputs'].append(
        {'path': second.path, 'effect': second.effect, 'media_type': second.media_type})
    (scenario['repository'] / 'test_answer.py').write_text(PROGRAM)
    subprocess.run(['git', '-C', str(scenario['repository']), 'add', 'test_answer.py'], check=True)
    inputs = _prepare(scenario, tmp_path, text=PROGRAM, external_consumer=True)
    scoped = build_scoped_doctor_analysis(repository=scenario['repository'],
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    candidate = analyze().candidate
    state = tmp_path / 'ambiguous-proof'
    with pytest.raises(ValueError, match='exactly one independently admitted'):
        proof_module.prove_header_contract(scoped=scoped, contract=candidate.contract,
            candidate_cid=content_identity(candidate.to_dict()), state=state,
            lean=Path('/unavailable/lean'), z3=Path('/unavailable/z3'))
    assert not state.exists()


def test_actual_helper_independently_recompiles_security_ir_bindings(scenario, tmp_path):
    inputs, scoped = _scoped(scenario, tmp_path)
    inputs = _provers(inputs)
    analysis = analyze()
    analyses = [{'path': 'answer.py', 'status': analysis.status,
        'reason_codes': list(analysis.reason_codes), 'source_sha256': analysis.source_sha256,
        'contracts': list(analysis.contracts), 'open_frontiers': list(analysis.open_frontiers)}]
    compilation = proof_module.security_ir_bridge.compile_header_security_ir(scoped=scoped, analyses=analyses)
    contract = {**analysis.candidate.contract,
        'security_ir_declaration_cid': compilation.report['declaration_cid'],
        'security_ir_formalization_cid': compilation.report['artifact_cid']}
    report, proof = proof_module.prove_header_contract(scoped=scoped, contract=contract,
        candidate_cid=content_identity(analysis.candidate.to_dict()), state=tmp_path / 'bound-ir-proof',
        lean=inputs['kernel_executable'], z3=inputs['solver_executable'])
    assert proof is not None and proof.mutation_capable
    assert report['security_ir_bindings_verified'] is True
    assert report['security_ir_obligations_discharged'] is False
    assert report['security_ir_declaration_cid'] == compilation.report['declaration_cid']
    for field in ('security_ir_declaration_cid', 'security_ir_formalization_cid'):
        state = tmp_path / ('wrong-' + field)
        with pytest.raises(ValueError, match='independently compiled'):
            proof_module.prove_header_contract(scoped=scoped,
                contract={**contract, field: content_identity({'rebound': field})},
                candidate_cid=content_identity(analysis.candidate.to_dict()), state=state,
                lean=Path('/unavailable/lean'), z3=Path('/unavailable/z3'))
        assert not state.exists()


def test_translator_implementation_change_rebinds_real_proof_roots_and_review(scenario, tmp_path, monkeypatch):
    inputs, scoped = _scoped(scenario, tmp_path)
    old_report, old_proof = _prove(inputs, scoped, tmp_path / 'before-translator-change')
    module = proof_module.header_contracts
    changed = tmp_path / 'reviewed-translator-next.py'
    changed.write_bytes(Path(module.__file__).read_bytes() + b'\n# Authored next translator revision.\n')
    # Change only the implementation artifact identity, without touching live
    # source or changing the semantics exercised by this regression.
    monkeypatch.setattr(module, '__file__', str(changed))
    new_report, new_proof = _prove(inputs, scoped, tmp_path / 'after-translator-change')
    assert old_proof is not None and new_proof is not None
    assert old_report['candidate_cid'] == new_report['candidate_cid']
    assert old_report['contract_cid'] == new_report['contract_cid']
    assert old_report['toolchain_cid'] != new_report['toolchain_cid']
    assert old_report['translator_cid'] != new_report['translator_cid']
    assert old_proof.roots.translator_id != new_proof.roots.translator_id
    assert old_proof.theorem.review_receipt_id != new_proof.theorem.review_receipt_id
    assert old_proof.content_id != new_proof.content_id


@pytest.mark.parametrize('module_name,key', [
    ('security_ir_bridge', 'security_ir_bridge_sha256'),
    ('native_security_ir', 'security_ir_compiler_sha256'),
])
def test_security_ir_implementation_artifacts_are_in_pinned_identity(tmp_path, monkeypatch, module_name, key):
    from ipfs_datasets_py.logic.security_ir import formalization_adapter
    module = formalization_adapter if module_name == 'native_security_ir' else proof_module.security_ir_bridge
    before = proof_module._implementation_bindings()
    changed = tmp_path / 'next-compiler.py'
    changed.write_bytes(Path(module.__file__).read_bytes() + b'\n# Authored next compiler revision.\n')
    monkeypatch.setattr(module, '__file__', str(changed))
    after = proof_module._implementation_bindings()
    assert before[key] != after[key]
    assert {k: v for k, v in before.items() if k != key} == {k: v for k, v in after.items() if k != key}


def test_implementation_drift_during_real_proof_is_not_returned_as_success(scenario, tmp_path, monkeypatch):
    inputs, scoped = _scoped(scenario, tmp_path)
    original = proof_module._implementation_bindings
    calls = 0
    def changing_bindings():
        nonlocal calls
        calls += 1
        value = original()
        if calls > 1:
            value['header_ast_translator_sha256'] = '0' * 64
        return value
    monkeypatch.setattr(proof_module, '_implementation_bindings', changing_bindings)
    state = tmp_path / 'drifting-proof'
    with pytest.raises(ValueError, match='changed during proof execution'):
        _prove(inputs, scoped, state)
    assert not (state / 'proof-report.json').exists()
