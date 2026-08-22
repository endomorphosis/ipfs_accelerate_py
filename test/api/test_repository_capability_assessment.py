"""EAAEF-043: typed support outcomes; generic adapter does not admit mutation."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.project_adapters.assessment import (
    OUTCOMES,
    assess_repository,
)


def test_python_inventory_does_not_admit_mutation(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = assess_repository(tmp_path)
    assert result["outcome"] in OUTCOMES
    assert result["mutation_admitted"] is False
    assert "python" in result["languages"]


def test_empty_repo_is_preview_or_unsupported(tmp_path: Path) -> None:
    result = assess_repository(tmp_path)
    assert result["outcome"] in {"preview_only", "unsupported_language"}
    assert result["mutation_admitted"] is False
