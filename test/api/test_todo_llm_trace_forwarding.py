"""Execute the real generated child against fixture routers; no provider calls."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon import llm


@pytest.mark.parametrize('signature_style', ['named', 'kwargs', 'differently_named_kwargs'])
@pytest.mark.parametrize('trace_enabled', [True, False])
def test_generated_child_forwards_trace_flags_and_exact_directory(
    tmp_path, monkeypatch, signature_style, trace_enabled
):
    observed = {}
    trace_dir = tmp_path / 'actual-request-trace'
    trace_dir.mkdir()
    def capture(prompt, values):
        observed.update(values)
        assert prompt == 'fixture prompt'
        assert values['model_name'] == 'grok-4.7'
        assert values['provider'] == 'grok_cli'
        assert values['allow_local_fallback'] is False
        assert values['allow_cross_provider_fallback'] is False
        if values.get('trace'):
            Path(values['trace_dir'], 'grok_cli_trace.jsonl').write_text(
                json.dumps({'fixture_only': True, 'model': values['model_name']}) + '\n'
            )
        return 'fixture reply'
    if signature_style == 'named':
        def generate(prompt, *, model_name, provider, allow_local_fallback,
                     allow_cross_provider_fallback, timeout, max_new_tokens,
                     temperature, trace=False, trace_dir=None):
            return capture(prompt, {key: value for key, value in locals().items() if key != 'prompt'})
    elif signature_style == 'kwargs':
        def generate(prompt, **kwargs):
            return capture(prompt, kwargs)
    else:
        def generate(prompt, **options):
            return capture(prompt, options)
    monkeypatch.setattr(llm_router, 'generate_text', generate)
    monkeypatch.setattr(llm_router, 'get_last_generation_trace', lambda: {'effective_provider_name': 'grok_cli'})
    cfg = llm.LlmRouterInvocation(repo_root=tmp_path, provider='grok_cli', model_name='grok-4.7',
                                  env_prefix='TRACE_FIXTURE', trace=trace_enabled, trace_dir=trace_dir)
    prompt_path = tmp_path / 'prompt.txt'
    prompt_path.write_text('fixture prompt')
    values = {'PROMPT_FILE': str(prompt_path), 'MODEL_NAME': 'grok-4.7', 'PROVIDER': 'grok_cli',
              'TIMEOUT': '3', 'MAX_NEW_TOKENS': '32', 'TEMPERATURE': '0.0',
              'ALLOW_LOCAL_FALLBACK': '0', 'ALLOW_CROSS_PROVIDER_FALLBACK': '0',
              'TRACE': '1' if trace_enabled else '0', 'TRACE_DIR': str(trace_dir),
              'REQUIRED_EFFECTIVE_PROVIDERS': 'grok_cli', 'USAGE_MODE': 'off'}
    for suffix, value in values.items():
        monkeypatch.setenv(llm._env_name(cfg, suffix), value)
    for suffix in ('RESULT_FILE', 'ENVELOPE_FILE', 'ALLOCATION_SESSION_ID', 'ALLOCATION_PATH', 'CODEX_PLANNING_SCHEMA'):
        monkeypatch.delenv(llm._env_name(cfg, suffix), raising=False)
    before_path = sys.path[:]
    output = StringIO()
    try:
        with redirect_stdout(output):
            exec(compile(llm._llm_router_child_code(cfg), '<actual generated child>', 'exec'), {})
    finally:
        sys.path[:] = before_path
    assert output.getvalue() == 'fixture reply'
    assert observed['trace'] is trace_enabled
    assert observed['trace_dir'] == str(trace_dir)
    assert (trace_dir / 'grok_cli_trace.jsonl').exists() is trace_enabled


def test_generated_child_does_not_inject_trace_into_legacy_strict_signature(tmp_path, monkeypatch):
    observed = {}
    def generate(prompt, *, model_name, provider, allow_local_fallback,
                 allow_cross_provider_fallback, timeout, max_new_tokens, temperature):
        observed['called'] = True
        return 'legacy fixture'
    monkeypatch.setattr(llm_router, 'generate_text', generate)
    cfg = llm.LlmRouterInvocation(repo_root=tmp_path, env_prefix='TRACE_LEGACY')
    prompt = tmp_path / 'prompt.txt'; prompt.write_text('fixture')
    for key, value in {'PROMPT_FILE': str(prompt), 'MODEL_NAME': 'fixture', 'PROVIDER': 'fixture',
                       'TIMEOUT': '1', 'MAX_NEW_TOKENS': '16', 'TEMPERATURE': '0.0', 'TRACE': '1',
                       'TRACE_DIR': str(tmp_path)}.items():
        monkeypatch.setenv(llm._env_name(cfg, key), value)
    before_path = sys.path[:]
    try:
        with redirect_stdout(StringIO()):
            exec(compile(llm._llm_router_child_code(cfg), '<actual generated legacy child>', 'exec'), {})
    finally:
        sys.path[:] = before_path
    assert observed['called'] is True
