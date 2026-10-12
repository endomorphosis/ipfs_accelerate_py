"""Query references compose with real admitted Doctor/public context."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_task_workflow import _prepare, SOURCE
from test.api.test_semantic_router_integration import provider
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    prepare_doctor_task_repair, execute_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import prepare_doctor_residual_context
from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import prepare_public_instruction_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding.terminal_source_input_experiment import _native_prompt
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit


@pytest.mark.parametrize('composition', ['reference', 'native_fallback'])
def test_real_doctor_public_reply_and_both_helpers_compose_with_query_reference(scenario, tmp_path, provider, monkeypatch, composition):
    root = scenario['repository']
    public_source = root / 'test_answer.py'
    public_source.write_text(public_source.read_text() + '\n' + '# Preserve the admitted public assertion and validate the answer behavior.\n' * 32)
    subprocess.run(['git', '-C', str(root), 'add', 'test_answer.py'], check=True)
    inputs = _prepare(scenario, tmp_path, text='import os\n' + SOURCE, external_consumer=True)
    with (root / '.git/info/exclude').open('a') as stream:
        stream.write('\n.runtime/\n')
    prepared = prepare_doctor_task_repair(**inputs)
    result = execute_doctor_task_repair(prepared)
    assert result['status'] == 'residual'
    residual = prepare_doctor_residual_context(prepared=prepared, result=result)
    public_bytes = (root / 'test_answer.py').read_bytes()
    public = prepare_public_instruction_context(repository=root, admission=inputs['admission'],
        task_cid=inputs['task_cid'], source_path='test_answer.py',
        expected_source_sha256=hashlib.sha256(public_bytes).hexdigest())
    task_id = scenario['intent'].get_task(inputs['task_cid'])['task_alias']
    artifact = root / '.runtime/semantic/worker-context.json'
    prepare_semantic_context(repository=root, paths=['answer.py', 'test_answer.py'],
        required_raw_paths=['test_answer.py'], program_paths=['answer.py'],
        objective='Repair admitted answer', task_id=task_id, output=artifact.parent,
        worker_query=public_bytes.decode(), worker_selector='duckdb-query-symbol-name@1',
        worker_capsule_limit=8, worker_max_bytes=32768, context_input_tokens=8192)
    prompt = _native_prompt(root, artifact, task_id, ['answer.py', 'test_answer.py'])
    workspace = tmp_path / 'allocated'
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', '-q', str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'unused-provider-home'))
    state, seen = provider
    state['reply'] = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
    def bind_schema():
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        schema = seen[-1][1]['codex_output_schema']
        raw = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        set_last_cli_observation('codex_cli', {'exit_code': 0, 'codex_output_schema_sha256': hashlib.sha256(raw).hexdigest(),
            'codex_output_schema_bytes': len(raw)})
    state['after_generate'] = bind_schema
    options = {}
    if composition == 'native_fallback':
        options['prompt_token_counter'] = runner.PromptTokenCounter(provider='codex_cli', model='pinned',
            tokenizer_id='authored', tokenizer_revision='equal', special_token_policy='none', count_tokens=lambda _: 1)
    _, receipt = runner.run(prompt=prompt, provider='codex_cli', model='pinned', timeout=1,
        max_output_tokens=128, semantic_repository=root, query_reference_policy="required-instruction-ref@1",
        source_view_policy="source-view-raw@1", tool_output_policy="artifact-preview@2",
        coding_reply_mode='ordinary-completion@1', doctor_residual_artifact=Path(residual['artifact']),
        doctor_residual_sha256=residual['sha256'], doctor_residual_task_cid=inputs['task_cid'],
        public_instruction_artifact=Path(public['artifact']), public_instruction_sha256=public['sha256'],
        public_instruction_task_cid=inputs['task_cid'], **options)
    assert len(seen) == 1 and public_bytes.decode() in seen[0][0]
    assert receipt['query_reference_view']['reference_count'] == 1
    assert receipt['query_reference_selected'] is (composition == 'reference')
    assert receipt['query_reference_dispatched'] is (composition == 'reference')
    assert receipt['source_view_policy']['usage']['helper_operations_observed'] is False
    assert receipt['tool_output_policy']['usage']['helper_operations_observed'] is False
    assert receipt['doctor_residual_context']['extra_provider_calls'] == 0
    assert receipt['public_instruction']['extra_provider_calls'] == 0
    if composition == 'native_fallback':
        assert receipt['semantic_translation'] is None
    parsed = audit._receipt({**receipt, 'phase': 'coding'}, workspace.parent)
    (root / 'answer.py').write_text('def answer(): return 7\n')
    subprocess.run(['git', '-C', str(root), 'worktree', 'remove', '--force', str(workspace)], check=True)
    monkeypatch.chdir(tmp_path)
    reconstructed, _, checks = audit._model_projection(rendered=prompt, receipt=parsed, repository=root)
    assert reconstructed == seen[0][0] and all(checks.values())
