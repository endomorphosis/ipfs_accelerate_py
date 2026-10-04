"""Diagnostic only: execute the archived original bind function on native fixtures."""
import ast
from pathlib import Path

def pytest_configure(config):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
    source = Path(__file__).resolve().parent / 'before/terminal_initial_context.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'bind_admitted_context']
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), initial.__dict__)
