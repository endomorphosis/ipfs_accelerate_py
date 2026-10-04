"""Real SecurityIR formalization stays source-bound and declaration-only."""
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import SOURCE, _prepare
from test.api.test_doctor_header_contracts import PROGRAM, PROTOCOL
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_header_contracts import analyze_http_header_contracts
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_scoped_analysis import build_scoped_doctor_analysis
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_security_ir import compile_header_security_ir
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import LocalPlanningError


def setup(scenario, tmp_path, source=PROGRAM):
    inputs = _prepare(scenario, tmp_path, text=source)
    scoped = build_scoped_doctor_analysis(repository=scenario['repository'],
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    analysis = analyze_http_header_contracts(source, protocol=PROTOCOL)
    return scoped, [{'path': 'answer.py', 'status': analysis.status,
        'reason_codes': list(analysis.reason_codes), 'source_sha256': analysis.source_sha256,
        'contracts': list(analysis.contracts), 'open_frontiers': list(analysis.open_frontiers)}]


def test_native_declaration_sample_and_formalization_have_real_source_grounded_obligations(scenario, tmp_path):
    from ipfs_datasets_py.logic.security_ir.model import SecurityIR
    from ipfs_datasets_py.logic.formalization.samples import FormalizationSample
    from ipfs_datasets_py.logic.formalization.compiler import FormalizationArtifact
    scoped, analyses = setup(scenario, tmp_path)
    result = compile_header_security_ir(scoped=scoped, analyses=analyses, output=tmp_path / 'security-ir')
    assert isinstance(result.declaration, SecurityIR)
    assert isinstance(result.sample, FormalizationSample)
    assert isinstance(result.artifact, FormalizationArtifact)
    report = result.report
    assert report['status'] == 'compiled_local_declarations'
    assert report['normalizer_count'] == 2
    assert report['claim_count'] == report['obligation_count'] == 4
    assert report['formula_count'] > report['claim_count']
    assert report['declaration_schema'] == 'security-ir/v1'
    assert {formula.expression['kind'] for formula in result.artifact.formulas} >= {
        'threat_assumption', 'security_policy'}
    assert {source.content_sha256 for source in result.declaration.sources} == {
        hashlib.sha256(PROGRAM.encode()).hexdigest()}
    assert {asset.symbol for asset in result.declaration.assets} == {'clean_label', 'clean_payload'}
    assert all(claim.assumption_ids and claim.source_ids for claim in result.declaration.claims)
    assert all(policy.attributes['forbidden_codepoints'] == (0, 10, 13) for policy in result.declaration.policies)
    assert SecurityIR.from_dict(result.declaration.to_dict()).cid == result.declaration.cid
    assert report['provider_calls'] == 0
    for field in ('solver_executed', 'proof_created', 'whole_program_proved', 'mutation_authority', 'completion_authority'):
        assert report[field] is False
    forbidden_fields = {'proof_obligations', 'solver_results', 'runtime_traces', 'candidate',
                        'candidate_cid', 'proof_receipt_id', 'after_bytes_base64'}
    def check_declaration(item):
        if isinstance(item, dict):
            assert not forbidden_fields.intersection(item)
            for value in item.values():
                check_declaration(value)
        elif isinstance(item, list):
            for value in item:
                check_declaration(value)
    check_declaration(result.declaration.to_dict())
    for name, descriptor in report['artifacts'].items():
        raw = (tmp_path / 'security-ir' / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == descriptor['sha256']
        assert PROGRAM not in raw.decode()
    second = compile_header_security_ir(scoped=scoped, analyses=analyses)
    assert second.declaration.cid == result.declaration.cid
    assert second.sample.sample_cid == result.sample.sample_cid
    assert second.artifact.artifact_id == result.artifact.artifact_id


def test_unsupported_python_compiles_no_expected_claims_or_fabricated_solver_truth(scenario, tmp_path):
    scoped, analyses = setup(scenario, tmp_path, source=SOURCE)
    result = compile_header_security_ir(scoped=scoped, analyses=analyses)
    assert result.report['status'] == 'unsupported'
    assert result.report['normalizer_count'] == result.report['claim_count'] == result.report['obligation_count'] == 0
    assert not result.declaration.claims
    assert result.report['unmodeled_paths'] == ['answer.py']
    assert 'unsupported_python_security_semantics' in result.report['open_frontiers']


@pytest.mark.parametrize('kind', ['contract', 'source', 'status', 'candidate-field', 'duplicate', 'scope'])
def test_ir_converter_replays_structural_contract_instead_of_trusting_nominated_rows(scenario, tmp_path, kind):
    scoped, analyses = setup(scenario, tmp_path)
    rows = deepcopy(analyses)
    if kind == 'contract':
        rows[0]['contracts'][0]['forbidden_codepoints'] = [10]
    elif kind == 'source':
        rows[0]['source_sha256'] = '0' * 64
    elif kind == 'status':
        rows[0]['status'] = 'already_satisfied'
    elif kind == 'candidate-field':
        rows[0]['solver_results'] = {'proved': True}
    elif kind == 'duplicate':
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]['path'] = 'foreign.py'
    with pytest.raises(ValueError):
        compile_header_security_ir(scoped=scoped, analyses=rows)


def test_ir_compilation_rechecks_current_manifest_before_creating_artifacts(scenario, tmp_path):
    scoped, analyses = setup(scenario, tmp_path)
    (scenario['repository'] / 'answer.py').write_text(PROGRAM + '# changed\n')
    output = tmp_path / 'security-ir'
    with pytest.raises(LocalPlanningError):
        compile_header_security_ir(scoped=scoped, analyses=analyses, output=output)
    assert not output.exists()


def test_ir_artifacts_cannot_write_worker_repository_or_overwrite_prior_run(scenario, tmp_path):
    scoped, analyses = setup(scenario, tmp_path)
    for output in (scenario['repository'] / 'ir', tmp_path):
        with pytest.raises(ValueError, match='fresh external'):
            compile_header_security_ir(scoped=scoped, analyses=analyses, output=output)
