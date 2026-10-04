"""Signed authored tasks through native header analysis, proof and candidate IO."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_header_contracts import PROGRAM, PROTOCOL
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptOutputRecord
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_header_workflow as workflow
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import materialize_doctor_contract_candidate


CHECK = '''from answer import clean_label, clean_payload
for normalize in (clean_label, clean_payload):
    for control in ('\\r', '\\n', '\\x00'):
        for value in ('prefix' + control + 'suffix', ('prefix' + control).encode()):
            try:
                normalize(value)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid input was accepted')
assert clean_label('x_key') == 'X-Key'
assert clean_payload('éλ') == 'éλ'
'''


def _prepare_headers(scenario, tmp_path, *, report=True, extra_output=False, callback='respond'):
    root = scenario['repository']
    source = PROGRAM.replace('respond', callback)
    (root / 'answer.py').write_text(source)
    check = CHECK
    if report:
        check += "import json\nfrom pathlib import Path\n"
        check += "assert json.loads(Path('report.jsonl').read_text()) == {'file_path': '/public/answer.py', 'cwe_id': ['cwe-93']}\n"
    (root / 'test_answer.py').write_text(check)
    subprocess.run(['git', '-C', str(root), 'add', 'answer.py', 'test_answer.py'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Header qualification', '-c',
                    'user.email=header@example.invalid', 'commit', '-qm', 'authored header contract fixture'], check=True)
    profile, lifecycle = tmp_path / 'header-profile', tmp_path / 'header-lifecycle'
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    outputs = [scenario['graph'].tasks[0].outputs[0]]
    if report:
        outputs.append(PromptOutputRecord(path='report.jsonl', effect='create', media_type='text/plain'))
    if extra_output:
        outputs.append(PromptOutputRecord(path='notes.txt', effect='create', media_type='text/plain'))
    scope = ('answer.py', 'test_answer.py', *(output.path for output in outputs if output.effect == 'create'))
    goal = replace(scenario['graph'].goals[0], scope_paths=scope)
    task = replace(scenario['graph'].tasks[0], goal_cid=goal.goal_cid, outputs=tuple(outputs),
                   scope_paths=scope, predicted_files=tuple(output.path for output in outputs))
    roots = {'request_cid': scenario['graph'].request_cid,
             'scan_cid': local.content_identity({'sources': local._sources(root, ['answer.py', 'test_answer.py'])}),
             'program_root': local.content_identity({'git_tree': subprocess.check_output(
                 ['git', '-C', str(root), 'rev-parse', 'HEAD^{tree}'], text=True).strip()})}
    graph = replace(scenario['graph'], **roots, goals=(goal,), tasks=(task,))
    specs = deepcopy(scenario['manifest']['payload']['tasks'])
    specs[0]['scope_paths'] = list(scope)
    specs[0]['outputs'] = [{'path': output.path, 'effect': output.effect, 'media_type': output.media_type} for output in outputs]
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    local.materialize_local_benchmark_plan(admission=admission, intent=scenario['intent'])
    return dict(repository=root, admission=admission, intent=scenario['intent'], task_cid=task.task_cid,
                state=tmp_path / 'header-workflow', protocol=replace(PROTOCOL, callback_parameter=callback),
                lean=Path('/unavailable/lean'), z3=Path('/unavailable/z3'),
                report_output=workflow.CweReportOutput('report.jsonl', '/public') if report else None)


def _real_provers(request):
    pin = os.environ.get('DOCTOR_COMPOSITION_LEAN')
    if not pin and shutil.which('elan'):
        pin = subprocess.check_output(['elan', 'which', 'lean'], text=True).strip()
    z3 = shutil.which('z3')
    if not pin or not Path(pin).is_file() or not z3:
        pytest.skip('actual installed Lean and Z3 required')
    return {**request, 'lean': Path(pin), 'z3': Path(z3)}


@pytest.mark.parametrize('report', [False, True])
def test_signed_task_actual_proof_hydration_and_independent_validation(scenario, tmp_path, report):
    request = _real_provers(_prepare_headers(scenario, tmp_path, report=report))
    root = scenario['repository']
    before_source = (root / 'answer.py').read_bytes()
    before_task = scenario['intent'].get_task(request['task_cid'])
    before_validation = subprocess.run([sys.executable, '-B', 'test_answer.py'], cwd=root, capture_output=True, text=True)
    assert before_validation.returncode == 1 and 'invalid input was accepted' in before_validation.stderr
    result = workflow.prepare_header_contract_repair(**request)
    assert result['status'] == 'candidate_ready', result['proof']
    assert result['proof']['status'] == 'proved_local_contract'
    assert result['proof']['proof']['disposition'] == 'verified'
    assert result['proof']['proof']['native_store_ref']
    assert result['proof']['proof']['kernel_store_ref']
    assert result['proof']['tactician']
    assert result['contract_index']['hydrated'] is True
    assert result['contract_index']['active_receipt_ids']
    assert result['synthesis']['edit_count'] == 2
    assert result['synthesis']['global_impact_closure'] is False
    assert result['provider_calls'] == result['canonical_source_edits'] == 0
    assert result['whole_program_proved'] is result['completion_authority'] is False
    artifact = Path(result['artifact'])
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['sha256']
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    body = json.loads(raw)
    assert len(body['edits']) == 1 + report
    assert body['proof_receipt_id'] == result['proof']['proof_receipt_id']
    workspace = tmp_path / 'allocated'
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(workspace),
                    body['baseline_commit']], check=True, capture_output=True)
    observed = materialize_doctor_contract_candidate(artifact=artifact, expected_sha256=result['sha256'],
        task_cid=request['task_cid'], prompt=json.dumps({'objective_id': before_task['task_alias']}), workspace=workspace)
    assert observed['status'] == 'candidate_materialized'
    assert len(observed['writes']) == 1 + report
    assert observed['completion_authority'] is observed['publication_authority'] is False
    after_validation = subprocess.run([sys.executable, '-B', 'test_answer.py'], cwd=workspace, capture_output=True, text=True)
    assert after_validation.returncode == 0, after_validation.stderr
    assert (root / 'answer.py').read_bytes() == before_source
    assert not (root / 'report.jsonl').exists()
    assert scenario['intent'].get_task(request['task_cid']) == before_task
    assert subprocess.check_output(['git', '-C', str(workspace), 'rev-parse', 'HEAD'], text=True).strip() == body['baseline_commit']


def test_missing_prover_has_hydrated_residual_without_artifact_or_fabricated_receipt(scenario, tmp_path):
    request = _prepare_headers(scenario, tmp_path)
    task = scenario['intent'].get_task(request['task_cid'])
    result = workflow.prepare_header_contract_repair(**request)
    assert result['status'] == 'residual'
    assert result['reason_codes'] == ['required_local_prover_unavailable']
    assert result['contract_index']['hydrated'] is True
    assert result['contract_index']['active_receipt_ids'] == []
    assert 'proof_receipt_id' not in result['proof'] and 'artifact' not in result
    assert not (scenario['repository'] / '.runtime/doctor-contract-candidates').exists()
    assert scenario['intent'].get_task(request['task_cid']) == task


@pytest.mark.parametrize('kind', ['source', 'task'])
def test_source_or_task_drift_during_analysis_refuses_before_proof(scenario, tmp_path, monkeypatch, kind):
    request = _prepare_headers(scenario, tmp_path)
    original = workflow.analyze_http_header_contracts
    def drifting(*args, **kwargs):
        result = original(*args, **kwargs)
        if kind == 'source':
            path = scenario['repository'] / 'answer.py'
            path.write_bytes(path.read_bytes() + b'\n# source changed\n')
        else:
            task = scenario['intent'].get_task(request['task_cid'])
            scenario['intent'].cas_task_status(task_cid=task['task_cid'], expected_revision=task['revision'], new_status='in_progress')
        return result
    monkeypatch.setattr(workflow, 'analyze_http_header_contracts', drifting)
    monkeypatch.setattr(workflow, 'prove_header_contract', lambda **_: pytest.fail('stale request must not invoke prover'))
    with pytest.raises(ValueError):
        workflow.prepare_header_contract_repair(**request)
    assert not (scenario['repository'] / '.runtime/doctor-contract-candidates').exists()


@pytest.mark.parametrize('kind', ['extra_output', 'undeclared_report_projection'])
def test_declared_output_coverage_required_without_implicit_report_generation(scenario, tmp_path, kind):
    request = _prepare_headers(scenario, tmp_path, extra_output=kind == 'extra_output')
    if kind == 'undeclared_report_projection':
        request['report_output'] = None
    result = workflow.prepare_header_contract_repair(**request)
    assert result['status'] == 'residual'
    assert result['reason_codes'] == ['local_operator_does_not_cover_declared_outputs']
    assert result['contract_index']['active_receipt_ids'] == []
    assert 'artifact' not in result
    assert not (scenario['repository'] / '.runtime/doctor-contract-candidates').exists()


def test_explicit_report_projection_cannot_choose_undeclared_output(scenario, tmp_path):
    request = _prepare_headers(scenario, tmp_path, report=False)
    request['report_output'] = workflow.CweReportOutput('report.jsonl', '/public')
    with pytest.raises(ValueError, match='not an independently declared output'):
        workflow.prepare_header_contract_repair(**request)
    assert not (scenario['repository'] / '.runtime/doctor-contract-candidates').exists()


@pytest.mark.parametrize('path,root', [('/absolute.jsonl', '/public'), ('../escape.jsonl', '/public'),
                                    ('.runtime/report.jsonl', '/public'), ('report.jsonl', 'relative')])
def test_report_projection_paths_are_closed(path, root):
    with pytest.raises(ValueError):
        workflow.CweReportOutput(path, root)


def test_real_dispatch_requires_explicit_profile_and_selects_contract_worker(scenario, tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
    request = _real_provers(_prepare_headers(scenario, tmp_path, callback='start_response'))
    monkeypatch.setattr(dispatch, '_installed_provers', lambda: (request['z3'], request['lean']))
    result = dispatch.prepare_terminal_doctor_dispatch(repository=scenario['repository'], state=tmp_path,
        admission=request['admission'], task_cid=request['task_cid'], contract_profile='wsgi-header-controls@1')
    assert result['status'] == 'candidate_ready' and result['route'] == 'doctor_contract_candidate'
    assert result['provider_calls'] == 0
    argv = dispatch.implementation_argv(router=Path('/router'), model='unused', reasoning='high', timeout=30,
        semantic_repository=scenario['repository'], doctor=result)
    assert '--doctor-contract-artifact' in argv and '--model' not in argv
    assert '--doctor-contract-sha256' in argv
    assert scenario['intent'].get_task(request['task_cid'])['status'] == 'ready'
    assert (scenario['repository'] / 'answer.py').read_text() == PROGRAM.replace('respond', 'start_response')
