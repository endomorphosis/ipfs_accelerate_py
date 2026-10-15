"""Verify the copied closed custody validator against pinned source by AST only."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/barberb/lift_coding/.worktrees/supervisor-planner-acceptance-20261015/ipfs_accelerate_py/agent_supervisor/todo_daemon/native_provider_custody_observation.py')
ORIGINAL = Path('/home/barberb/lift_coding/artifacts/doctor-public-validation-20261014/diagnostic_custody_contract.py')


def export():
    constants = {'SCHEMA', 'STAGE_REASONS', '_SUCCESS'}
    selected = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'validate_native_provider_custody_observation':
            selected.append(node)
    assert len(selected) == 4
    raw = ('"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""\n'
        'from __future__ import annotations\n\n' + ast.unparse(ast.Module(body=selected, type_ignores=[])) + '\n').encode()
    copied = ROOT / 'diagnostic_custody_contract.py'
    assert raw == copied.read_bytes() == ORIGINAL.read_bytes()
    def bind(path):
        value=path.read_bytes()
        return dict(path=str(path),sha256=hashlib.sha256(value).hexdigest(),bytes=len(value))
    return dict(schema='pure-custody-contract-export@1',source=bind(SOURCE),original=bind(ORIGINAL),copy=bind(copied),
        selected_constants=sorted(constants),selected_functions=['validate_native_provider_custody_observation'],
        ast_export_identical=True,runtime_modules_imported=False,provider_calls=0)


if __name__ == '__main__':
    with (ROOT/'custody-export-provenance.json').open('x') as handle:
        json.dump(export(),handle,indent=2,sort_keys=True);handle.write('\n')
