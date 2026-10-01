"""Automatic AST selection, real proof/transaction, and native residual plans."""
from copy import deepcopy
from dataclasses import replace
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import DoctorMode
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import DoctorCompositionError
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    _candidates, execute_doctor_task_repair, prepare_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import DeterministicDoctorPolicy

SOURCE = 'def process(amount):\n    return amount\n\ndef answer():\n    return process(count=2)\n'


def _prepare(scenario, tmp_path, text=SOURCE, *, external_consumer=False):
    root = scenario['repository']
    (root / 'answer.py').write_text(text)
    paths = ['answer.py', 'test_answer.py'] if external_consumer else ['answer.py']
    if not external_consumer:
        subprocess.run(['git', '-C', str(root), 'rm', '-q', 'test_answer.py'], check=True)
    subprocess.run(['git', '-C', str(root), 'add', 'answer.py'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Doctor fixture', '-c',
        'user.email=doctor@example.invalid', 'commit', '-qm', 'public keyword mismatch'], check=True)
    baseline = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    scenario['profile'] = tmp_path / 'keyword-profile'
    scenario['lifecycle'] = tmp_path / 'keyword-lifecycle'
    Supervisor.init_local(repository=root, consent=True, profile_dir=scenario['profile'],
        lifecycle_dir=scenario['lifecycle'])
    roots = {
        'request_cid': scenario['graph'].request_cid,
        'scan_cid': local.content_identity({'sources': local._sources(root, paths)}),
        'program_root': local.content_identity({'git_tree': subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD^{tree}'], text=True).strip()}),
    }
    graph = replace(scenario['graph'], **roots)
    specs = deepcopy(scenario['manifest']['payload']['tasks'])
    if not external_consumer:
        # A separately signed public check exercises the single-file fixture;
        # no source consumer is excluded from the native graph inventory.
        validation = replace(graph.tasks[0].validations[0], argv=(
            'python3', '-B', '-c', 'from answer import answer; assert answer() == 2'))
        goal = replace(graph.goals[0], scope_paths=tuple(paths))
        task = replace(graph.tasks[0], goal_cid=goal.goal_cid, scope_paths=tuple(paths), validations=(validation,))
        graph = replace(graph, goals=(goal,), tasks=(task,))
        specs[0]['scope_paths'] = paths
        specs[0]['validations'][0]['argv'] = list(validation.argv)
    manifest = local.author_local_benchmark_manifest(repository=root,
        profile_dir=scenario['profile'], lifecycle_dir=scenario['lifecycle'],
        task_specs=specs, planning_roots=roots)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    local.materialize_local_benchmark_plan(admission=admission, intent=scenario['intent'])
    runtime = DeterministicDoctorRuntime(checkout_root=root, index_root=tmp_path / 'doctor-index',
        policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.SANDBOX_AUTO))
    candidate_ref = 'refs/heads/doctor/automatic'
    subprocess.run(['git', '-C', str(root), 'update-ref', candidate_ref, baseline], check=True)
    return dict(runtime=runtime, intent=scenario['intent'], admission=admission,
        task_cid=graph.tasks[0].task_cid, state_root=tmp_path / 'doctor-workflow',
        solver_executable=Path(shutil.which('z3') or '/unavailable/z3'),
        kernel_executable=Path('/unavailable/lean'), candidate_ref=candidate_ref)


def _provers(inputs):
    lean_pin = os.environ.get('DOCTOR_COMPOSITION_LEAN', '')
    if not shutil.which('z3') or (not lean_pin and not shutil.which('elan')):
        pytest.skip('real Lean and Z3 required')
    inputs['kernel_executable'] = Path(lean_pin or subprocess.check_output(['elan', 'which', 'lean'], text=True).strip())
    if not inputs['kernel_executable'].is_file() or not os.access(inputs['kernel_executable'], os.X_OK):
        pytest.skip('real Lean executable required')
    return inputs


def _prepare_analysis_guard_fixture(scenario, tmp_path):
    """Authored documentation example exercising the real native guard."""
    path = scenario['repository'] / 'docs/example.rst'
    path.parent.mkdir()
    path.write_text('Public example:\n\npassword = "AUTHORED-DUMMY-NOT-A-CREDENTIAL"\n')
    subprocess.run(['git', '-C', str(scenario['repository']), 'add', 'docs/example.rst'], check=True)
    return _prepare(scenario, tmp_path)


def test_automatic_source_selection_real_proof_transaction_and_bound_handoff(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    before = scenario['intent'].get_task(inputs['task_cid'])
    prepared = prepare_doctor_task_repair(**inputs)
    assert prepared.report['status'] == 'prepared'
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'candidate_ready', json.dumps({
        'stages': result.get('stages'), 'reason_codes': result.get('transaction', {}).get('reason_codes'),
        'refill_reasons': result.get('plan_refill', {}).get('residuals'),
    }, sort_keys=True)
    assert prepared.runtime.composition_result.proof.mutation_capable
    assert prepared.runtime.composition_result.synthesis.authoritative_proof
    assert prepared.runtime.composition_result.impact.mutation_admissible
    assert result['transaction']['committed']
    handoff = result['handoff']
    raw = Path(result['handoff_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['handoff_sha256']
    assert json.loads(raw) == handoff
    assert handoff['handoff_cid'] == local.content_identity({k:v for k,v in handoff.items() if k != 'handoff_cid'})
    candidate = base64.b64decode(handoff['edits'][0]['after_bytes_base64'], validate=True)
    assert candidate.decode() == SOURCE.replace('count=2', 'amount=2')
    assert handoff['base_commit'] != handoff['candidate_commit']
    assert handoff['task_cid'] == before['task_cid'] and handoff['task_id'] == before['task_alias']
    assert handoff['permitted_outputs'] == ['answer.py']
    assert handoff['publication_authority'] is handoff['completion_authority'] is False
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
    assert scenario['intent'].get_task(inputs['task_cid']) == before


@pytest.mark.parametrize('source,reason', [
    ('import os\n' + SOURCE, 'unsupported_module_or_signature_shape'),
    ('def process(amount=1, total=1):\n    return amount + total\n\ndef answer():\n    return process(count=2)\n', 'ambiguous_supported_repairs'),
    (SOURCE.replace('count=2', 'amount=2'), 'no_supported_keyword_mismatch'),
])
def test_unsupported_emits_append_only_native_refill_without_effects(scenario, tmp_path, source, reason):
    inputs = _prepare(scenario, tmp_path, source)
    before = scenario['intent'].get_task(inputs['task_cid'])
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual' and reason in result['reason_codes']
    assert result['plan_refill']['successors']
    assert not result['plan_refill']['completion_authority']
    assert not result['plan_refill']['mutation_authority']
    assert not inputs['state_root'].exists()
    assert scenario['intent'].get_task(inputs['task_cid']) == before
    assert (scenario['repository'] / 'answer.py').read_text() == source


@pytest.mark.parametrize('drift', ['source', 'task'])
def test_stale_source_or_task_refuses_before_proof_and_ref_movement(scenario, tmp_path, drift):
    inputs = _provers(_prepare(scenario, tmp_path))
    prepared = prepare_doctor_task_repair(**inputs)
    if drift == 'source':
        (scenario['repository'] / 'answer.py').write_text(SOURCE + '\n# changed\n')
    else:
        task = scenario['intent'].get_task(inputs['task_cid'])
        scenario['intent'].cas_task_status(task_cid=task['task_cid'], expected_revision=task['revision'], new_status='in_progress')
    with pytest.raises((DoctorCompositionError, local.LocalPlanningError)):
        execute_doctor_task_repair(prepared)
    assert prepared.runtime.composition_result is None
    assert subprocess.check_output(['git', '-C', str(scenario['repository']), 'rev-parse', inputs['candidate_ref']], text=True).strip() == prepared.inputs.base_ref


def test_keyword_selection_is_unique_and_byte_offset_aware():
    text = SOURCE.replace('return process(count=2)', 'return ("é", process(count=2))[1]')
    candidates = _candidates(text)
    assert len(candidates) == 1 and candidates[0]['replacement'] == 'amount'
    assert text[candidates[0]['offset']:].startswith('count=')
    assert not _candidates(SOURCE.replace('count=2', '**{"count": 2}'))


def test_unwritable_external_consumer_retains_native_compilation_abstention(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path, external_consumer=True))
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual'
    assert result['stages']['proof']['mutation_capable']
    assert result['stages']['impact']['mutation_admissible']
    assert 'uncovered_scc' in result['stages']['plan_compilation']['reason_codes']
    assert result['plan_refill']['successors']
    assert prepared.runtime.composition_transaction is None
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE


def test_missing_prover_emits_native_capability_gap_without_source_effect(scenario, tmp_path):
    inputs = _prepare(scenario, tmp_path)
    prepared = prepare_doctor_task_repair(**inputs)
    assert prepared.report['status'] == 'residual'
    assert prepared.report['reason_codes'] == ['required_local_prover_unavailable']
    assert prepared.report['plan_refill']['residuals'][0]['kind'] == 'capability_gap'
    assert not inputs['state_root'].exists()
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE


def test_failed_real_prover_never_starts_candidate_transaction(scenario, tmp_path):
    inputs = _provers(_prepare(scenario, tmp_path))
    inputs['solver_executable'] = Path('/usr/bin/false')
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual'
    assert not result['stages']['proof']['mutation_capable']
    assert prepared.runtime.composition_transaction is None
    assert not (inputs['state_root'] / 'candidate-handoff.json').exists()
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE


def test_actual_candidate_gate_failure_rolls_back_without_handoff(scenario, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import DoctorStepApplyResult, DoctorStepDisposition
    inputs = _provers(_prepare(scenario, tmp_path))
    prepared = prepare_doctor_task_repair(**inputs)
    def reject(_composition):
        return lambda *_: DoctorStepApplyResult(disposition=DoctorStepDisposition.FAILED,
            reason_codes=('candidate_gate_negative_control',), diagnostic_refs=('validation:negative-control',), static_replay=True)
    monkeypatch.setattr('ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition.composition_step_validator', reject)
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual' and not result['transaction']['committed']
    assert prepared.runtime.composition_result.proof.mutation_capable
    assert not (inputs['state_root'] / 'candidate-handoff.json').exists()
    assert subprocess.check_output(['git', '-C', str(scenario['repository']), 'rev-parse', inputs['candidate_ref']], text=True).strip() == prepared.inputs.base_ref
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE


def test_real_secret_screen_remains_strict_but_optional_analysis_proposes_capability_gap(scenario, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory import PlanningAnalysisSecretError
    inputs = _prepare_analysis_guard_fixture(scenario, tmp_path)
    before = scenario['intent'].get_task(inputs['task_cid'])
    baseline = subprocess.check_output(['git', '-C', str(scenario['repository']), 'rev-parse', 'HEAD'], text=True).strip()
    # The generic native analysis still refuses this authored fixture.
    with pytest.raises(PlanningAnalysisSecretError):
        inputs['runtime'].build_evidence(refresh=True)
    assert inputs['runtime'].evidence is None
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert prepared.inputs is None and inputs['runtime'].evidence is None
    assert result['status'] == 'residual' and result['analysis_status'] == 'unavailable'
    assert result['reason_codes'] == ['doctor_analysis_secret_screen_refused']
    assert result['evidence_id'] is result['diagnostic_snapshot_id'] is result['finding_count'] is None
    assert result['source_hashes'] is None and result['provider_calls'] == 0
    observation = result['analysis_observation']
    assert observation['observation_cid'] == local.content_identity({
        key: value for key, value in observation.items() if key != 'observation_cid'})
    assert observation['source_tree_id'] == local._tree(inputs['admission']['manifest']['payload']['sources'])
    assert observation['diagnostic_snapshot_created'] is observation['finding_created'] is observation['proof_created'] is False
    residual = result['plan_refill']['residuals'][0]
    assert residual['kind'] == 'capability_gap'
    assert residual['required_capability'] == 'doctor:screened-source-analysis'
    assert residual['evidence_refs'] == [observation['observation_cid']]
    assert len(result['plan_refill']['work_proposals']) == 1
    assert not result['plan_refill']['derived_runtime_admitted']
    serialized = json.dumps(result)
    assert 'docs/example.rst' not in serialized and 'AUTHORED-DUMMY' not in serialized
    assert scenario['intent'].get_task(inputs['task_cid']) == before
    assert (scenario['repository'] / 'answer.py').read_text() == SOURCE
    assert not inputs['state_root'].exists()
    assert inputs['runtime'].composition_result is inputs['runtime'].composition_transaction is None
    assert subprocess.check_output(['git', '-C', str(scenario['repository']), 'rev-parse', inputs['candidate_ref']], text=True).strip() == baseline


@pytest.mark.parametrize('kind', ['stability', 'path_escape', 'source_changed_during_screen', 'task_changed_during_screen'])
def test_optional_analysis_refusal_never_masks_other_integrity_failures(scenario, tmp_path, monkeypatch, kind):
    from ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory import (
        PlanningAnalysisSecretError, PlanningAnalysisStabilityError, PlanningAnalysisPathEscapeError,
    )
    inputs = _prepare(scenario, tmp_path)
    def failed(**kwargs):
        if kind == 'stability':
            raise PlanningAnalysisStabilityError('authored instability')
        if kind == 'path_escape':
            raise PlanningAnalysisPathEscapeError('authored path mismatch')
        if kind == 'source_changed_during_screen':
            (scenario['repository'] / 'answer.py').write_text(SOURCE + '# source changed\n')
        else:
            task = scenario['intent'].get_task(inputs['task_cid'])
            scenario['intent'].cas_task_status(task_cid=task['task_cid'], expected_revision=task['revision'], new_status='in_progress')
        raise PlanningAnalysisSecretError('AUTHORED-DUMMY-PRIVATE-DETAIL')
    monkeypatch.setattr(inputs['runtime'], 'build_evidence', failed)
    with pytest.raises((PlanningAnalysisStabilityError, PlanningAnalysisPathEscapeError,
                        DoctorCompositionError, local.LocalPlanningError)):
        prepare_doctor_task_repair(**inputs)
    assert not (scenario['repository'] / '.runtime/doctor-residuals').exists()
    assert not inputs['state_root'].exists()
    assert inputs['runtime'].composition_result is None
