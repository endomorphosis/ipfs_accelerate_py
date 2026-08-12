"""WPD-022: AnalyticalCloseExecutor hermetic mutation tests."""

from __future__ import annotations

import hashlib
import inspect

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.analytical_close_executor import (
    ANALYTICAL_CLOSE_EXECUTOR_INTERFACE,
    AnalyticalCloseExecutor,
    AnalyticalCloseMutationError,
    AnalyticalClosePathError,
    AnalyticalClosePlan,
    AnalyticalEdit,
    build_analytical_close_executor,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_interface_identity() -> None:
    assert ANALYTICAL_CLOSE_EXECUTOR_INTERFACE == "AnalyticalCloseExecutor@1"


def test_cold_source_has_no_llm_imports() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        analytical_close_executor as mod,
    )

    source = inspect.getsource(mod)
    for marker in ("openai", "anthropic", "litellm", "grok_cli"):
        assert marker not in source


def test_success_requires_real_byte_mutation(tmp_path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    original = "VALUE = 1\n"
    target.write_text(original, encoding="utf-8")
    executor = build_analytical_close_executor(tmp_path)
    plan = AnalyticalClosePlan(
        edits=(
            AnalyticalEdit(
                path="src/app.py",
                start=0,
                end=len(original),
                replacement="VALUE = 2\n",
                before_hash=_sha(original),
            ),
        ),
        expects_writes=True,
        plan_cid="plan:1",
        task_cid="task:1",
    )
    receipt = executor.apply(plan)
    assert receipt.applied is True
    assert receipt.mutated is True
    assert receipt.bytes_before == len(original.encode("utf-8"))
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert "src/app.py" in receipt.paths_touched


def test_fake_success_without_mutation_rejected(tmp_path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    original = "VALUE = 1\n"
    target.write_text(original, encoding="utf-8")
    executor = AnalyticalCloseExecutor(worktree_root=tmp_path)
    # Replacement identical to current span → no mutation.
    plan = AnalyticalClosePlan(
        edits=(
            AnalyticalEdit(
                path="src/app.py",
                start=0,
                end=len(original),
                replacement=original,
                before_hash=_sha(original),
            ),
        ),
        expects_writes=True,
    )
    with pytest.raises(AnalyticalCloseMutationError, match="no byte mutation"):
        executor.apply(plan)
    # Rollback leaves original content.
    assert target.read_text(encoding="utf-8") == original


def test_path_escape_rejected(tmp_path) -> None:
    executor = build_analytical_close_executor(tmp_path)
    with pytest.raises(AnalyticalClosePathError):
        AnalyticalEdit(
            path="../outside.py",
            start=0,
            end=0,
            replacement="x",
        )


def test_partial_failure_rolls_back(tmp_path) -> None:
    good = tmp_path / "a.py"
    good.write_text("A=1\n", encoding="utf-8")
    executor = build_analytical_close_executor(tmp_path)
    # Second edit has out-of-bounds span after first would write — validate all
    # first so no write occurs; simulate mid-apply failure via missing before_hash.
    plan = AnalyticalClosePlan(
        edits=(
            AnalyticalEdit(
                path="a.py",
                start=0,
                end=4,
                replacement="A=2\n",
                before_hash=_sha("A=1\n"),
            ),
            AnalyticalEdit(
                path="a.py",
                start=0,
                end=99,
                replacement="boom",
                before_hash=_sha("A=1\n"),
            ),
        ),
        expects_writes=True,
    )
    with pytest.raises(Exception):
        executor.apply(plan)
    assert good.read_text(encoding="utf-8") == "A=1\n"


def test_mapping_plan_round_trip(tmp_path) -> None:
    path = tmp_path / "m.py"
    path.write_text("x=0\n", encoding="utf-8")
    executor = build_analytical_close_executor(tmp_path)
    receipt = executor.apply(
        {
            "expects_writes": True,
            "edits": [
                {
                    "path": "m.py",
                    "start": 0,
                    "end": 4,
                    "replacement": "x=1\n",
                    "before_hash": _sha("x=0\n"),
                }
            ],
        }
    )
    assert receipt.mutated is True
    assert path.read_text(encoding="utf-8") == "x=1\n"
