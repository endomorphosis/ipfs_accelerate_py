"""Native repair eligibility is source-bound abstention, never fabricated proof."""
import hashlib
import json
from dataclasses import replace
import subprocess

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


def _signed_program_context(scenario, tmp_path, *, empty=False, noncanonical_smoke=False):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptAcceptanceRecord, PromptValidationRecord, PromptOutputRecord,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    root = scenario['repository']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('rm', '-q', 'test_answer.py')
    if empty:
        git('rm', '-q', 'answer.py')
    program = [] if empty else ['answer.py']
    instruction = 'Create the declared output.\n' if empty else 'Inspect the existing answer.\n'
    profile = profiles.validate_task_profile(dict(schema=profiles.SCHEMA,
        instruction_sha256=profiles.instruction_sha256(instruction), input_paths=program,
        outputs=[dict(path='new.py' if empty else 'answer.py', effect='create' if empty else 'modify',
            media_type='text/x-python')]))
    (root / profiles.INSTRUCTION).write_text(instruction)
    (root / profiles.PROFILE).write_bytes(profiles.task_profile_bytes(profile))
    (root / profiles.SMOKE).write_text('def signed_but_arbitrary_harness(): return 7\n'
        if noncanonical_smoke else profiles.task_profile_smoke(profile))
    git('add', '--', *profiles.task_profile_worker_inputs(profile))
    git('-c', 'user.name=Doctor eligibility', '-c', 'user.email=doctor@example.invalid',
        'commit', '-qm', 'signed task partition')
    profile_dir, lifecycle = tmp_path / 'program-profile', tmp_path / 'program-lifecycle'
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile_dir, lifecycle_dir=lifecycle)
    policy = local.content_identity(local.LOCAL_POLICY)
    spec = profiles.task_profile_spec(profile, policy_cid=policy)
    validation = PromptValidationRecord(**{**spec['validations'][0],
        'argv': tuple(spec['validations'][0]['argv']), 'expected_exit_codes': (0,)})
    acceptance = PromptAcceptanceRecord(**{**spec['acceptance'][0], 'evidence_cids': (),
        'validation_keys': tuple(spec['acceptance'][0]['validation_keys'])})
    roots = dict(request_cid=scenario['graph'].request_cid,
        scan_cid=local.content_identity({'sources': local._sources(root, profiles.task_profile_worker_inputs(profile))}),
        program_root=local.content_identity({'git_tree': git('rev-parse', 'HEAD^{tree}')}))
    goal = replace(scenario['graph'].goals[0], scope_paths=tuple(spec['scope_paths']), acceptance=(acceptance,))
    task = replace(scenario['graph'].tasks[0], task_key=spec['task_key'], goal_cid=goal.goal_cid,
        scope_paths=tuple(spec['scope_paths']), validations=(validation,), acceptance=(acceptance,),
        outputs=tuple(PromptOutputRecord(**row) for row in spec['outputs']),
        predicted_files=tuple(row['path'] for row in spec['outputs']))
    graph = replace(scenario['graph'], **roots, goals=(goal,), tasks=(task,))
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile_dir,
        lifecycle_dir=lifecycle, task_specs=[spec], planning_roots=roots)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    names = sorted(manifest['payload']['sources'])
    prepared = prepare_semantic_context(repository=root, paths=names,
        required_raw_paths=[profiles.INSTRUCTION], program_paths=program,
        objective=instruction.strip(), task_id=task.task_cid, output=root / '.runtime/program-semantic')
    return dict(repository=root, admission=admission, paths=names, program_paths=program,
        diagnostic_artifact=root / '.runtime/program-semantic/doctor.json'), prepared


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


@pytest.mark.parametrize('empty', [False, True])
def test_signed_program_subset_replays_native_doctor_without_support_code(scenario, tmp_path, empty):
    inputs, prepared = _signed_program_context(scenario, tmp_path, empty=empty)
    before = {name: (inputs['repository'] / name).read_bytes() for name in inputs['paths']}
    report = assess_doctor_repair_eligibility(**inputs)
    assert report['diagnostics_replayed'] is True
    assert report['scope_cid'] == prepared['scope_cid']
    assert report['program_paths'] == report['diagnostic_source_paths'] == inputs['program_paths']
    assert report['captured_source_count'] == len(inputs['paths'])
    assert report['program_source_count'] == (0 if empty else 1)
    assert report['support_source_count'] == 3
    assert set(report['source_hashes']) == set(inputs['paths'])
    assert report['findings_count'] == 0
    assert report['output_assessments'][0]['module_shape_eligible'] is not empty
    assert report['execution_authority'] is report['completion_authority'] is False
    assert report['automatic_repair_eligible'] is False
    assert {name: (inputs['repository'] / name).read_bytes() for name in inputs['paths']} == before


@pytest.mark.parametrize('tamper', ['empty_subset', 'include_smoke', 'partial_capture', 'support_drift', 'diagnostic'])
def test_doctor_explicit_subset_requires_signed_population_and_full_preimages(scenario, tmp_path, tamper):
    inputs, _ = _signed_program_context(scenario, tmp_path)
    from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
    if tamper == 'empty_subset':
        inputs['program_paths'] = []
    elif tamper == 'include_smoke':
        inputs['program_paths'] = sorted(['answer.py', profiles.SMOKE])
    elif tamper == 'partial_capture':
        inputs['paths'].remove(profiles.INSTRUCTION)
    elif tamper == 'support_drift':
        (inputs['repository'] / profiles.SMOKE).write_text('# changed signed support\n')
    else:
        data = json.loads(inputs['diagnostic_artifact'].read_text())
        data['findings'] = [{'invented': 'diagnostic'}]
        inputs['diagnostic_artifact'].write_text(json.dumps(data))
    with pytest.raises((DoctorCompositionError, local.LocalPlanningError)):
        assess_doctor_repair_eligibility(**inputs)


def test_signing_arbitrary_smoke_does_not_grant_program_partition(scenario, tmp_path):
    inputs, _ = _signed_program_context(scenario, tmp_path, noncanonical_smoke=True)
    with pytest.raises(DoctorCompositionError, match='partition'):
        assess_doctor_repair_eligibility(**inputs)


def test_explicit_subset_without_signed_profile_cannot_hide_source(scenario):
    inputs = _inputs(scenario)
    with pytest.raises(DoctorCompositionError, match='signed partition'):
        assess_doctor_repair_eligibility(**inputs, program_paths=[])
