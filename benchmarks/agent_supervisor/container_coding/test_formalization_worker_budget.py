"""The shipped worker and shared router honor only explicit longer budgets."""
import argparse
import ast
import builtins
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_budget_policy import FORMALIZATION_BUDGET_PROFILE as PROFILE


def _shipped_route(argv):
    tree = ast.parse(WORKER_ENTRY)
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'parser' for target in node.targets))
    fake_sys = SimpleNamespace(argv=['worker-entry', *argv])
    events = []
    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            return super().parse_args(fake_sys.argv[1:] if args is None else args, namespace)
    def main():
        events.append(('router', list(fake_sys.argv)))
        return 0
    def custody(function):
        events.append(('custody',))
        return function()
    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'ipfs_accelerate_py.agent_supervisor.runtime':
            return SimpleNamespace(router_implementation_runner=runner)
        if name == 'ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner':
            return SimpleNamespace(main=main)
        if name == 'ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper':
            return SimpleNamespace(run_with_child_custody=custody)
        pytest.fail('unexpected provider or candidate import')
    namespace = {'__builtins__': {**vars(builtins), '__import__': imports},
        'argparse': SimpleNamespace(ArgumentParser=Parser), 'pathlib': __import__('pathlib'),
        'sys': fake_sys, 'selected_provider': 'codex_cli', 'artifact': Path('/fixture/boundary'), 'digest': '4' * 64}
    try:
        exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), '<shipped-worker>', 'exec'), namespace)
    except SystemExit as error:
        return error.code, events
    pytest.fail('worker did not terminate')


@pytest.mark.parametrize('profile, timeout, accepted', [(None, 600, True), (None, 601, False),
    (None, 1800, False), (PROFILE, 1800, True), (PROFILE, 1801, False)])
def test_shipped_worker_carries_profile_through_child_custody_and_refuses_excess(profile, timeout, accepted):
    argv = ['--timeout', str(timeout)]
    if profile is not None:
        argv += ['--budget-profile', profile]
    code, events = _shipped_route(argv)
    assert (code == 0) is accepted
    if accepted:
        assert events[0] == ('custody',) and events[1][0] == 'router'
        forwarded = events[1][1]
        assert forwarded[forwarded.index('--timeout') + 1] == str(timeout)
        assert ('--budget-profile' in forwarded) is (profile is not None)
        if profile is not None:
            assert forwarded[forwarded.index('--budget-profile') + 1] == PROFILE
    else:
        assert events == []


def test_shipped_worker_rejects_unreviewed_profile():
    with pytest.raises(ValueError, match='reviewed benchmark budget'):
        _shipped_route(['--timeout', '1800', '--budget-profile', 'unknown'])


@pytest.mark.parametrize('profile, timeout', [(None, 601), (None, 1800), (PROFILE, 1801),
    (PROFILE, True), ('unknown', 90)])
def test_actual_router_rejects_invalid_allowance_before_provider_discovery(monkeypatch, profile, timeout):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, 'discover_available_providers',
        lambda: pytest.fail('invalid allowance reached provider discovery'))
    with pytest.raises(ValueError):
        runner.run(prompt='authored fixture', provider='codex_cli', model='pinned', timeout=timeout,
            max_output_tokens=512, budget_profile=profile)


@pytest.mark.parametrize('profile, timeout', [(None, 600), (PROFILE, 1800)])
def test_actual_shared_router_forwards_selected_timeout_without_provider_fallback(tmp_path, monkeypatch, capsys,
        profile, timeout):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intelligence_index, 'discover_available_providers', lambda: ['codex_cli'])
    monkeypatch.setattr(intelligence_index, 'select_efficient_route', lambda **kwargs:
        SimpleNamespace(provider='codex_cli', model_name='pinned', reasoning_effort='high', catalog_revision='fixture'))
    monkeypatch.setattr(llm_router, 'get_llm_provider', lambda *args, **kwargs: object())
    calls = []
    def generate(prompt, **kwargs):
        calls.append(kwargs)
        set_last_cli_observation('codex_cli', {'exit_code': 0, 'prompt_tokens': 10, 'completion_tokens': 3})
        return 'authored completion'
    monkeypatch.setattr(llm_router, 'generate_text', generate)
    output, receipt = runner.run(prompt='authored fixture', provider='codex_cli', model='pinned',
        timeout=timeout, max_output_tokens=512, budget_profile=profile)
    assert output == 'authored completion' and receipt['timeout_seconds'] == timeout
    assert len(calls) == 1 and calls[0]['timeout'] == timeout
    assert calls[0]['allow_local_fallback'] is calls[0]['allow_cross_provider_fallback'] is False
    capsys.readouterr()
