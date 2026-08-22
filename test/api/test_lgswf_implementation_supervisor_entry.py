from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"


def _module():
    spec = importlib.util.spec_from_file_location("lgswf_implementation_entry", ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_legacy_tuple_is_validated_and_consumed() -> None:
    module = _module()
    argv = [
        "--todo-path",
        "board.md",
        "--task-source-kind",
        "legacy-markdown",
        "--authority-mode",
        "legacy_markdown",
        "--state-failover-policy",
        "fail_closed",
        "--explicit-legacy-task-source",
        "--implement",
    ]
    assert module.normalize_configured_task_source_args(argv) == [
        "--todo-path",
        "board.md",
        "--implement",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["--task-source-kind", "duckdb", "--authority-mode", "legacy_markdown", "--state-failover-policy", "fail_closed", "--explicit-legacy-task-source"],
        ["--task-source-kind", "legacy-markdown", "--explicit-legacy-task-source"],
        ["--authority-mode", "legacy_markdown"],
        ["--explicit-legacy-task-source"],
    ],
)
def test_nonexact_or_incomplete_tuple_fails_closed(argv: list[str]) -> None:
    module = _module()
    with pytest.raises(ValueError):
        module.normalize_configured_task_source_args(argv)


def test_unrelated_direct_supervisor_args_are_unchanged() -> None:
    module = _module()
    argv = ["--todo-path", "board.md", "--implement"]
    assert module.normalize_configured_task_source_args(argv) == argv


def test_child_pythonpath_is_replaced_with_the_exact_checkout() -> None:
    module = _module()
    environment = {"PYTHONPATH": "/untrusted/ambient/path"}
    value = module.seal_child_pythonpath(environment)
    assert value == str(ROOT)
    assert environment == {"PYTHONPATH": str(ROOT)}
