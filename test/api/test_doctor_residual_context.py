"""Real Doctor residuals reach the existing model call without task authority."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import _prepare, _prepare_analysis_guard_fixture, SOURCE
from test.api.test_semantic_router_integration import provider  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_context as residual
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    execute_doctor_task_repair, prepare_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def retained(scenario, tmp_path, request):
    inputs = (_prepare_analysis_guard_fixture(scenario, tmp_path)
        if getattr(request, 'param', None) == 'secret_analysis'
        else _prepare(scenario, tmp_path, text='import os\n' + SOURCE))
    root = scenario['repository']
    with (root / '.git/info/exclude').open('a') as stream:
        stream.write('\n.runtime/\n')
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    context = residual.prepare_doctor_residual_context(prepared=prepared, result=result)
    workspace = tmp_path / 'allocated'
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', '-q', str(workspace)], check=True)
    task = scenario['intent'].get_task(inputs['task_cid'])
    request = dict(artifact=Path(context['artifact']), expected_sha256=context['sha256'], repository=root,
        task_cid=inputs['task_cid'], prompt=json.dumps({'objective_id': task['task_alias']}), workspace=workspace)
    return prepared, result, context, request


def test_typed_native_residual_capsule_keeps_scope_and_exact_historical_input(retained, scenario):
    _, result, context, request = retained
    before = scenario['intent'].get_task(request['task_cid'])
    text, receipt = residual.load_doctor_residual_advisory(**request)
    body = json.loads(request['artifact'].read_text())
    assert body['residuals'] == result['plan_refill']['residuals']
    assert body['capsule']['allowed_paths'] == ['answer.py']
    assert body['capsule']['repairable_record_ids'] == []
    assert body['capsule']['rejected_proposal_record_ids'] == []
    assert body['capsule']['completion_authority'] is False
    assert body['proposal_counts']['successors'] == 1
    assert 'unsupported_module_or_signature_shape' in text
    assert 'do not execute them' in text
    assert receipt['advisory_sha256'] == _sha(text) and receipt['extra_provider_calls'] == 0
    assert receipt['source_freshness_verified'] and not receipt['derived_runtime_admitted']
    source = request['repository'] / 'answer.py'
    source.write_text(source.read_text() + '\n# published after this invocation\n')
    with pytest.raises(ValueError, match='stale'):
        residual.load_doctor_residual_advisory(**request)
    historic, audit = residual.load_doctor_residual_advisory(**{**request, 'workspace': None,
        'require_current_source': False})
    assert historic == text and audit['historical_replay']
    assert audit['source_freshness_verified'] is False
    assert scenario['intent'].get_task(request['task_cid']) == before


@pytest.mark.parametrize('change', ['task', 'prompt', 'artifact', 'worker_source', 'foreign', 'symlink'])
def test_residual_context_refuses_tamper_and_foreign_or_stale_worker(retained, tmp_path, change):
    _, _, _, original = retained
    request = dict(original)
    if change == 'task':
        request['task_cid'] = 'foreign'
    elif change == 'prompt':
        request['prompt'] = json.dumps({'objective_id': 'foreign'})
    elif change == 'artifact':
        request['artifact'].chmod(0o644)
        request['artifact'].write_bytes(request['artifact'].read_bytes() + b' ')
        request['artifact'].chmod(0o444)
    elif change == 'worker_source':
        (request['workspace'] / 'answer.py').write_text('different source\n')
    elif change == 'foreign':
        request['workspace'] = request['repository']
    else:
        target = request['artifact'].with_name('symlink.json')
        target.symlink_to(request['artifact'])
        request['artifact'] = target
    with pytest.raises((ValueError, OSError)):
        residual.load_doctor_residual_advisory(**request)


@pytest.mark.parametrize('change', ['authority', 'scope', 'goal', 'memory'])
def test_owner_rejects_residual_rebinding_before_publication(scenario, tmp_path, change):
    inputs = _prepare(scenario, tmp_path, text='import os\n' + SOURCE)
    prepared = prepare_doctor_task_repair(**inputs)
    result = copy.deepcopy(execute_doctor_task_repair(prepared))
    if change == 'authority':
        result['plan_refill']['derived_runtime_admitted'] = True
    elif change == 'scope':
        result['plan_refill']['residuals'][0]['context_paths'] = ['outside.py']
    elif change == 'goal':
        result['plan_refill']['residuals'][0]['parent_goal_cid'] = 'foreign'
    else:
        result['plan_refill']['next_memory'] = {}
    with pytest.raises(ValueError):
        residual.prepare_doctor_residual_context(prepared=prepared, result=result)
    assert not (scenario['repository'] / '.runtime/doctor-residuals').exists()


@pytest.mark.parametrize('retained', ['unsupported', 'secret_analysis'], indirect=True)
def test_actual_router_consumes_semantic_and_residual_context_once(retained, provider, monkeypatch, scenario):
    _, _, context, request = retained
    root, workspace = request['repository'], request['workspace']
    task_id = scenario['intent'].get_task(request['task_cid'])['task_alias']
    output = root / '.runtime/semantic'
    prepare_semantic_context(repository=root, paths=['answer.py'], required_raw_paths=['answer.py'],
        objective='Repair the admitted answer', task_id=task_id, output=output)
    artifact = output / 'worker-context.json'
    refs = build_text_context_references(artifact.read_text(), reference_prefix='semantic-context',
        kind='semantic-context', path=artifact.relative_to(root).as_posix(), repository_id='repo:test',
        tree_id='tree:test', required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id='repo:test', tree_id='tree:test', objective_id=task_id,
        objective_revision='sha256:task', policy_id='policy:test', policy_revision='sha256:policy',
        caller='supervisor:test', stage='implementation', goal={'id':task_id},
        authority={'mode':'candidate_only','completion_authority':False}, scope={'allowed_paths':['answer.py']},
        acceptance={'criteria':['pending native validation']}, evidence=refs)
    prompt = render_context_capsule(compiled.capsule)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('CODEX_HOME', str(root.parent / 'empty-worker-home'))
    before = scenario['intent'].get_task(request['task_cid'])
    text, receipt = runner.run(prompt=prompt, provider='codex_cli', model='pinned', timeout=1,
        max_output_tokens=128, semantic_repository=root, doctor_residual_artifact=request['artifact'],
        doctor_residual_sha256=context['sha256'], doctor_residual_task_cid=request['task_cid'])
    _, observed = provider
    assert text == 'literal summary' and len(observed) == 1
    semantic = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    advisory, binding = residual.load_doctor_residual_advisory(**{**request, 'prompt': prompt})
    projected = semantic.provider_prompt + advisory
    # Finalization must reconstruct both projections, even after the original
    # worktree is gone or a completed publication has changed the source.
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
    parsed = audit._receipt({**receipt, 'phase': 'coding'}, workspace.parent)
    actual, _, checks = audit._model_projection(rendered=prompt, receipt=parsed, repository=root)
    assert actual == observed[0][0] and all(checks.values())
    actual, _ = observed[0]
    expected, workspace_advisory = runner.render_model_prompt(prompt=projected, purpose='coding',
        workspace=workspace, semantic_transport=True)
    assert actual == expected
    assert receipt['doctor_residual_context'] == binding
    assert receipt['native_prompt_sha256'] == _sha(prompt)
    assert receipt['router_prompt_sha256'] == _sha(projected)
    assert receipt['model_prompt_sha256'] == _sha(actual)
    assert receipt['workspace_advisory_sha256'] == _sha(workspace_advisory)
    assert receipt['usage']['prompt_tokens'] == 123
    assert scenario['intent'].get_task(request['task_cid']) == before


def test_worker_launcher_exposes_only_complete_residual_flag_set():
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    compile(WORKER_ENTRY, '<worker-entry>', 'exec')
    assert "not all(residual_values)" in WORKER_ENTRY
    assert "args.semantic_repository is None" in WORKER_ENTRY
    assert "'--doctor-residual-task-cid',args.doctor_residual_task_cid" in WORKER_ENTRY
