"""Native repair eligibility is source-bound abstention, never fabricated proof."""
import hashlib
import json

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario as scenario
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import (
    DoctorCompositionError, _inert_function_module, assess_doctor_repair_eligibility,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
    DoctorAuthorityRoots, DoctorSourceUnit, diagnose_repository,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters import materialize_runtime_diagnostics
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes


def _inputs(scenario):
    root = scenario['repository']
    admission = local.admit_local_benchmark_plan(graph=scenario['graph'], manifest=scenario['manifest'])
    names = sorted(scenario['manifest']['payload']['sources'])
    raw = {name: (root / name).read_bytes() for name in names}
    inventory = {name: {'sha256': hashlib.sha256(value).hexdigest(), 'source_cid': cid_for_bytes(value)}
                 for name, value in raw.items()}
    scope = cid_for_payload({'schema': 'supervisor-source-scope@1', 'sources': inventory})
    repository_id = cid_for_payload({'repository': str(root)})
    roots = DoctorAuthorityRoots(repository_id=repository_id, forest_id=scope, tree_id=scope,
        overlay_id=scope, file_root_id=scope, blob_root_id=scope,
        config_id=local.content_identity({'paths': names}),
        policy_id=local.content_identity({'mode': 'context_preparation_only'}))
    diagnostic = diagnose_repository([DoctorSourceUnit(path=name, source_bytes=value,
        blob_identity=inventory[name]['source_cid']) for name, value in raw.items()], authority_roots=roots)
    snapshot, findings, manifest = materialize_runtime_diagnostics(diagnostic, require_repository_id=repository_id)
    output = root / '.runtime/doctor.json'
    output.parent.mkdir()
    output.write_text(json.dumps({'snapshot': snapshot.to_dict(),
        'findings': [item.to_dict() for item in findings], 'manifest_cid': manifest}))
    return dict(repository=root, admission=admission, paths=names, diagnostic_artifact=output)


def test_inert_source_still_abstains_without_reviewed_inputs(scenario):
    inputs = _inputs(scenario)
    before = {name: (inputs['repository'] / name).read_bytes() for name in inputs['paths']}
    report = assess_doctor_repair_eligibility(**inputs)
    assert report['status'] == 'abstained' and not report['automatic_repair_eligible']
    assert report['diagnostics_replayed'] is True
    assert report['output_assessments'] == [{'task_key': 'LOCAL-TASK', 'path': 'answer.py',
        'module_shape_eligible': True, 'reason_codes': []}]
    assert report['reason_codes'] == ['reviewed_typed_operator_and_proof_inputs_unavailable']
    assert set(report['stages'].values()) == {'not_run'}
    assert report['source_edits'] == report['provider_calls'] == 0
    assert report['execution_authority'] is report['completion_authority'] is False
    assert report['report_cid'] == local.content_identity({k: v for k, v in report.items() if k != 'report_cid'})
    assert {name: (inputs['repository'] / name).read_bytes() for name in inputs['paths']} == before
    assert report['allowed_outputs'][0]['path'] == 'answer.py'
    assert 'test_answer.py' not in {row['path'] for row in report['output_assessments']}


@pytest.mark.parametrize('source', [
    'import os\n\ndef target():\n    pass\n',
    'class Target:\n    pass\n',
    '@decorator\ndef target():\n    pass\n',
    'target = lambda: None\n',
])
def test_live_operator_and_assessment_share_inert_module_gate(source):
    with pytest.raises(DoctorCompositionError, match='inert function-only'):
        _inert_function_module(source)


@pytest.mark.parametrize('tamper', ['findings', 'scope', 'manifest', 'source'])
def test_eligibility_rejects_unbound_diagnostics_and_signed_source(scenario, tamper):
    inputs = _inputs(scenario)
    artifact = inputs['diagnostic_artifact']
    if tamper == 'findings':
        data = json.loads(artifact.read_text())
        data['findings'] = [{'fake': 'diagnostic'}]
        artifact.write_text(json.dumps(data))
    elif tamper == 'scope':
        inputs['paths'] = ['answer.py']
    elif tamper == 'manifest':
        inputs['admission']['manifest']['payload']['tasks'][0]['outputs'][0]['path'] = 'test_answer.py'
    else:
        (inputs['repository'] / 'answer.py').write_text('def answer():\n    return 9\n')
    with pytest.raises((DoctorCompositionError, local.LocalPlanningError, ValueError)):
        assess_doctor_repair_eligibility(**inputs)
