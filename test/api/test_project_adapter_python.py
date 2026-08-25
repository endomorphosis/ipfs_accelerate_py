"""EAAEF-041: Python ProjectAdapter compiles structured argv, does not admit mutation."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.project_adapters.python import (
    PythonProjectAdapter,
    python_toolchain,
)


def test_python_project_compiles_pytest_argv(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    adapter = PythonProjectAdapter()
    support = adapter.inspect(tmp_path)
    assert "python" in support.languages
    argv = adapter.focused_test_argv(("test_x.py",))
    assert argv[:4] == ("python3.12", "-m", "pytest", "-q")
    assert "test_x.py" in argv
    assert adapter.mutation_admitted(support) is False
    assert "pytest" in python_toolchain()
