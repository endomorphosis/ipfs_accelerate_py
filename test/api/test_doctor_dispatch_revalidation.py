"""Native Doctor source checks at the last boundary before model dispatch."""
import json

import pytest

from test.api.test_doctor_residual_context import retained, scenario, provider  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context


@pytest.fixture
def admitted_request(retained, monkeypatch):
    _, _, _, request = retained
    root, workspace = request['repository'], request['workspace']
    task_id = json.loads(request['prompt'])['objective_id']
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
        caller='supervisor:test', stage='implementation', goal={'id': task_id},
        authority={'mode': 'candidate_only', 'completion_authority': False},
        scope={'allowed_paths': ['answer.py']}, acceptance={'criteria': ['pending native validation']},
        evidence=refs)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('CODEX_HOME', str(root.parent / 'empty-worker-home'))
    return dict(request, prompt=render_context_capsule(compiled.capsule))


def _run(request):
    return runner.run(prompt=request['prompt'], provider='codex_cli', model='pinned', timeout=1,
        max_output_tokens=128, semantic_repository=request['repository'],
        doctor_residual_artifact=request['artifact'], doctor_residual_sha256=request['expected_sha256'],
        doctor_residual_task_cid=request['task_cid'])


@pytest.mark.parametrize('change', ['workspace', 'repository', 'artifact'])
def test_late_doctor_change_refuses_before_provider_call(admitted_request, provider, monkeypatch, capsys, change):
    from ipfs_accelerate_py import llm_router
    request = admitted_request
    original = llm_router.get_llm_provider

    def initialize(*args, **kwargs):
        instance = original(*args, **kwargs)
        if change == 'artifact':
            request['artifact'].chmod(0o644)
            request['artifact'].write_bytes(request['artifact'].read_bytes() + b' ')
            request['artifact'].chmod(0o444)
        else:
            source = request[change] / 'answer.py'
            source.write_text(source.read_text() + '\n# changed during provider initialization\n')
        return instance

    monkeypatch.setattr(llm_router, 'get_llm_provider', initialize)
    with pytest.raises(ValueError, match='residual'):
        _run(request)
    assert provider[1] == []
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert receipt['router_calls'] == 0 and receipt['status'] == 'failed'
    assert receipt['failure_phase'] == 'doctor_residual'


def test_source_edit_during_actual_coding_is_not_a_preflight_failure(admitted_request, provider):
    state, calls = provider
    request = admitted_request
    source = request['workspace'] / 'answer.py'
    state['after_generate'] = lambda: source.write_text(source.read_text() + '\n# candidate edit\n')
    output, receipt = _run(request)
    assert output == 'literal summary' and len(calls) == 1
    assert receipt['router_calls'] == 1 and receipt['status'] == 'provider_returned'
    assert receipt['usage']['completion_tokens'] == 7
    assert receipt['doctor_residual_context']['source_freshness_verified'] is True
