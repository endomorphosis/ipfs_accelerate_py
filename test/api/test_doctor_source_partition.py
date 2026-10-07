"""Real signed harness inputs through Doctor proof, impact and candidate gates."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import SOURCE, _provers
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import DoctorMode
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptAcceptanceRecord, PromptValidationRecord, PromptOutputRecord
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import DoctorCompositionError
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_source_partition import (
    DoctorSourcePartitionError, terminal_doctor_source_partition,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    execute_doctor_task_repair, prepare_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import DeterministicDoctorPolicy


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def harness_inputs(scenario, tmp_path, *, extra=None, tamper=None, external_consumer=False, outputs=None,
                   materialize=True):
    root = scenario['repository']
    (root / 'answer.py').write_text(SOURCE)
    if not external_consumer:
        _git(root, 'rm', '-q', 'test_answer.py')
    paths = ['answer.py', *(['test_answer.py'] if external_consumer else [])]
    for name, text in (extra or {}).items():
        (root / name).write_text(text)
        paths.append(name)
    instruction = 'Repair the unique keyword mismatch in answer.py.\n'
    profile = profiles.validate_task_profile(dict(schema=profiles.SCHEMA,
        instruction_sha256=profiles.instruction_sha256(instruction), input_paths=paths,
        outputs=outputs if outputs is not None else [dict(path='answer.py', effect='modify', media_type='text/x-python')]))
    (root / profiles.INSTRUCTION).write_text(instruction)
    (root / profiles.PROFILE).write_bytes(profiles.task_profile_bytes(profile))
    (root / profiles.SMOKE).write_text(profiles.task_profile_smoke(profile))
    if tamper:
        tamper(root)
    _git(root, 'add', '--', *paths, profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE)
    _git(root, '-c', 'user.name=Doctor qualification', '-c', 'user.email=doctor@example.invalid',
         'commit', '-qm', 'authored generic task inputs')
    profile_dir, lifecycle = tmp_path / 'partition-profile', tmp_path / 'partition-lifecycle'
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile_dir, lifecycle_dir=lifecycle)
    policy = local.content_identity(local.LOCAL_POLICY)
    spec = profiles.task_profile_spec(profile, policy_cid=policy)
    validation = PromptValidationRecord(**{**spec['validations'][0],
        'argv': tuple(spec['validations'][0]['argv']), 'expected_exit_codes': (0,)})
    acceptance = PromptAcceptanceRecord(**{**spec['acceptance'][0], 'evidence_cids': (),
        'validation_keys': tuple(spec['acceptance'][0]['validation_keys'])})
    roots = dict(request_cid=scenario['graph'].request_cid,
        scan_cid=local.content_identity({'sources': local._sources(root, profiles.task_profile_worker_inputs(profile))}),
        program_root=local.content_identity({'git_tree': _git(root, 'rev-parse', 'HEAD^{tree}')}))
    goal = replace(scenario['graph'].goals[0], scope_paths=tuple(spec['scope_paths']), acceptance=(acceptance,))
    task = replace(scenario['graph'].tasks[0], task_key=spec['task_key'], goal_cid=goal.goal_cid,
        scope_paths=tuple(spec['scope_paths']), validations=(validation,), acceptance=(acceptance,),
        outputs=tuple(PromptOutputRecord(**row) for row in spec['outputs']))
    graph = replace(scenario['graph'], **roots, goals=(goal,), tasks=(task,))
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile_dir,
        lifecycle_dir=lifecycle, task_specs=[spec], planning_roots=roots)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    if materialize:
        local.materialize_local_benchmark_plan(admission=admission, intent=scenario['intent'])
    runtime = DeterministicDoctorRuntime(checkout_root=root, index_root=tmp_path / 'partition-index',
        policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.SANDBOX_AUTO))
    candidate = 'refs/heads/doctor/partition'
    _git(root, 'update-ref', candidate, _git(root, 'rev-parse', 'HEAD'))
    return dict(runtime=runtime, intent=scenario['intent'], admission=admission, task_cid=task.task_cid,
        state_root=tmp_path / 'partition-workflow', solver_executable=Path(shutil.which('z3') or '/unavailable/z3'),
        kernel_executable=Path('/unavailable/lean'), candidate_ref=candidate)


def test_generated_support_allows_real_proof_and_isolated_candidate(scenario, tmp_path):
    inputs = _provers(harness_inputs(scenario, tmp_path))
    root = scenario['repository']
    before = {name: (root / name).read_bytes() for name in (profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE)}
    task_before = scenario['intent'].get_task(inputs['task_cid'])
    prepared = prepare_doctor_task_repair(**inputs)
    assert prepared.report['status'] == 'prepared', prepared.report.get('reason_codes')
    assert set(prepared.inputs.source_hashes) == {'answer.py', *before}
    assert prepared.inputs.program_graph.roots.included_roots == ('answer.py',)
    partition = prepared.report['source_partition']
    assert {r['role'] for r in partition['harness_support']} == {'instruction', 'task_profile', 'structural_smoke'}
    assert partition['proof_authority'] is partition['execution_authority'] is False
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'candidate_ready', result.get('stages')
    assert result['provider_calls'] == 0 and result['transaction']['committed']
    assert prepared.runtime.composition_result.proof.mutation_capable
    assert result['handoff']['completion_authority'] is result['handoff']['publication_authority'] is False
    assert _git(root, 'diff', '--name-only', result['handoff']['base_commit'], result['handoff']['candidate_commit']) == 'answer.py'
    candidate = _git(root, 'show', result['handoff']['candidate_commit'] + ':answer.py')
    assert candidate == SOURCE.replace('count=2', 'amount=2').strip()
    assert (root / 'answer.py').read_text() == SOURCE
    assert {name: (root / name).read_bytes() for name in before} == before
    assert scenario['intent'].get_task(inputs['task_cid']) == task_before


@pytest.mark.parametrize('path', [profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE, 'answer.py'])
def test_any_signed_preimage_drift_refuses_before_proof(scenario, tmp_path, path):
    inputs = _provers(harness_inputs(scenario, tmp_path))
    prepared = prepare_doctor_task_repair(**inputs)
    root = scenario['repository']
    with (root / path).open('a') as f:
        f.write('\n# altered\n')
    with pytest.raises((local.LocalPlanningError, DoctorSourcePartitionError, DoctorCompositionError)):
        execute_doctor_task_repair(prepared)
    assert prepared.runtime.composition_result is None
    assert _git(root, 'rev-parse', inputs['candidate_ref']) == prepared.inputs.base_ref


@pytest.mark.parametrize('extra', [{'config.json': '{"amount": 2}\n'}, {'README.md': 'Real task documentation.\n'}])
def test_real_noncode_inputs_are_not_silently_excluded(scenario, tmp_path, extra):
    inputs = harness_inputs(scenario, tmp_path, extra=extra)
    result = execute_doctor_task_repair(prepare_doctor_task_repair(**inputs))
    assert result['status'] == 'residual'
    assert 'unsupported_or_incomplete_source_inventory' in result['reason_codes']
    assert set(extra) <= set(result['source_hashes'])
    assert set(extra) <= set(result['source_partition']['program_paths'])
    assert not inputs['state_root'].exists()


def test_supported_partition_still_requires_actual_provers(scenario, tmp_path):
    inputs = harness_inputs(scenario, tmp_path)
    result = prepare_doctor_task_repair(**inputs).report
    assert result['reason_codes'] == ['required_local_prover_unavailable']
    assert not inputs['state_root'].exists()


def test_large_valid_generated_smoke_is_bounded_by_its_exact_producer(scenario, tmp_path):
    # JSON escaping plus its Python literal wrapper expands these canonical
    # paths; stay within the native 32-create and per-component byte limits.
    outputs = [dict(path=('λ' * 80 + '/') * 4 + f'f{i:02d}.py',
        effect='create', media_type='text/x-python') for i in range(32)]
    inputs = harness_inputs(scenario, tmp_path, outputs=outputs, materialize=False)
    root = scenario['repository']
    assert (root / profiles.PROFILE).stat().st_size <= 65536
    assert (root / profiles.SMOKE).stat().st_size > 65536
    partition = terminal_doctor_source_partition(repository=root,
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    assert partition.program_paths == ('answer.py',)
    partition.assert_current(root)
    # Partition recognition grants no permission to bypass the independent
    # native pending-receipt limit for this unusually large task declaration.
    with pytest.raises(local.LocalPlanningError, match='pending summary exceeds bound'):
        local.materialize_local_benchmark_plan(admission=inputs['admission'], intent=inputs['intent'])
    assert not inputs['state_root'].exists()


def test_actual_external_python_consumer_stays_in_impact_frontier(scenario, tmp_path):
    inputs = _provers(harness_inputs(scenario, tmp_path, external_consumer=True))
    prepared = prepare_doctor_task_repair(**inputs)
    assert set(prepared.inputs.program_graph.roots.included_roots) == {'answer.py', 'test_answer.py'}
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual'
    assert 'uncovered_scc' in result['stages']['plan_compilation']['reason_codes']
    assert prepared.runtime.composition_transaction is None


@pytest.mark.parametrize('kind', ['smoke', 'instruction_binding', 'profile_scope', 'duplicate_profile_key'])
def test_signing_arbitrary_support_bytes_cannot_grant_harness_roles(scenario, tmp_path, kind):
    def tamper(root):
        if kind == 'smoke':
            (root / profiles.SMOKE).write_text('from answer import answer\nassert answer() == 2\n')
        elif kind == 'instruction_binding':
            (root / profiles.INSTRUCTION).write_text('A different instruction.\n')
        else:
            path = root / profiles.PROFILE
            if kind == 'duplicate_profile_key':
                path.write_text(path.read_text().replace('{', '{"schema":"duplicate",', 1))
            else:
                value = json.loads(path.read_text())
                value['outputs'][0]['path'] = 'new.py'
                value['outputs'][0]['effect'] = 'create'
                path.write_bytes(profiles.task_profile_bytes(value))
    if kind == 'duplicate_profile_key':
        # The owner rejects ambiguous JSON while authoring the manifest, before
        # a signed admission or Doctor source partition can be constructed.
        with pytest.raises(local.LocalPlanningError,
                           match='^reserved public task profile must be unambiguous JSON$') as refusal:
            harness_inputs(scenario, tmp_path, tamper=tamper)
        assert str(refusal.value.__cause__) == 'duplicate task profile key'
        assert not (tmp_path / 'partition-workflow').exists()
        assert not (tmp_path / 'partition-index').exists()
        return
    inputs = harness_inputs(scenario, tmp_path, tamper=tamper)
    with pytest.raises((DoctorSourcePartitionError, local.LocalPlanningError)):
        prepare_doctor_task_repair(**inputs)
    assert not inputs['state_root'].exists()


def test_caller_cannot_forge_partition_to_hide_program_or_validation(scenario, tmp_path):
    inputs = _provers(harness_inputs(scenario, tmp_path))
    prepared = prepare_doctor_task_repair(**inputs)
    actual = prepared.inputs.source_partition
    for forged in (replace(actual, program_paths=()), replace(actual, support_hashes=actual.support_hashes[:-1]),
                   replace(actual, profile_sha256='0' * 64)):
        with pytest.raises((DoctorSourcePartitionError, DoctorCompositionError)):
            replace(prepared.inputs, source_partition=forged)
    with pytest.raises(DoctorCompositionError):
        replace(prepared.inputs, source_partition=None)


def test_compatibility_exports_preserve_generated_profile_and_smoke_bytes():
    from benchmarks.agent_supervisor.container_coding import terminal_task_profile as old
    assert old.task_profile_smoke is profiles.task_profile_smoke
    assert old.task_profile_bytes is profiles.task_profile_bytes


def test_terminal_dispatch_uses_real_candidate_and_reports_symbolic_capabilities(scenario, tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import prepare_terminal_doctor_dispatch
    inputs = _provers(harness_inputs(scenario, tmp_path))
    result = prepare_terminal_doctor_dispatch(repository=scenario['repository'], state=tmp_path,
        admission=inputs['admission'], task_cid=inputs['task_cid'])
    assert result['route'] == 'doctor_candidate' and result['provider_calls'] == 0
    assert result['status'] == 'candidate_ready'
    artifact = Path(result['artifact'])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == result['sha256']
    capabilities = result['symbolic_capabilities']
    assert capabilities['inventory']['doctor_hash_binding'] == 'complete_hash_match'
    assert capabilities['inventory']['partition']['program_input_count'] == 1
    assert capabilities['inventory']['partition']['harness_support_count'] == 3
    assert capabilities['proof']['local_contract_proof_reported'] is True
    assert capabilities['proof']['whole_program_verified'] is False
    assert capabilities['execution_authority'] is capabilities['proof_authority'] is False
    assert 'semantic_source_coverage_incomplete' not in capabilities['gap_codes']
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
