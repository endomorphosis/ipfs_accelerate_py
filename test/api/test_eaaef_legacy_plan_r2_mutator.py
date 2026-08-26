"""The historical run-v14 Plan-R2 experiment must remain fail-closed."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_run_v14_plan_r2_mutator_is_disabled() -> None:
    tree = ast.parse(
        (ROOT / "scripts/apply_eaaef_plan_r2.py").read_text(encoding="utf-8")
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first = main.body[0]
    assert isinstance(first, ast.Raise)
    assert isinstance(first.exc, ast.Call)
    assert isinstance(first.exc.func, ast.Name)
    assert first.exc.func.id == "SystemExit"
    assert len(first.exc.args) == 1
    assert isinstance(first.exc.args[0], ast.Name)
    assert first.exc.args[0].id == "LEGACY_MUTATOR_DISABLED_REASON"
