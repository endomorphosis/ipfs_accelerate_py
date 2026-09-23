"""The AST pytest seal reuses a pass only when file hashes still match."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.testing.pytest_ast_seal import (
    DuckDbQuackSealOracle,
    MemorySealOracle,
    PytestSealCatalogError,
    ast_closure,
    hash_of_hashes,
    seal_enabled,
    seal_files,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_hash_of_hashes_covers_ast_closure_and_ignores_mtime(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo / "pyproject.toml", "[project]\nname='seal'\n")
    _write(
        repo / "pkg" / "lib.py",
        "VALUE = 1\n",
    )
    _write(
        repo / "pkg" / "__init__.py",
        "",
    )
    test_file = repo / "test_sample.py"
    _write(test_file, "import pkg.lib\n\ndef test_one():\n    assert pkg.lib.VALUE == 1\n")
    files = ast_closure(repo, test_file)
    assert files is not None
    assert any(path.name == "lib.py" for path in files)
    oracle = MemorySealOracle()
    first = seal_files(files, oracle)
    assert first is not None and first.complete
    assert first.digest == hash_of_hashes((item.path, item.content_hash) for item in first.files)
    for path in files:
        path.touch()
    second = seal_files(files, oracle)
    assert second is not None
    assert second.digest == first.digest
    _write(repo / "pkg" / "lib.py", "VALUE = 2\n")
    changed = seal_files(files, oracle)
    assert changed is not None
    assert changed.digest != first.digest


def test_injected_oracle_returns_true_only_for_the_same_seal(tmp_path: Path) -> None:
    oracle = DuckDbQuackSealOracle(tmp_path / "pytest_ast_seal.duckdb")
    assert oracle.completion_authority is False
    assert oracle.reuse("test_sample.py::test_one", "abc") is False
    oracle.remember("test_sample.py::test_one", "abc", file_count=2)
    assert oracle.reuse("test_sample.py::test_one", "abc") is True
    assert oracle.reuse("test_sample.py::test_one", "def") is False
    assert oracle.reuse("test_sample.py::test_other", "abc") is False


def test_seal_is_opt_out_and_refuses_control_duckdb(tmp_path: Path) -> None:
    assert seal_enabled({}) is True
    assert seal_enabled({"IPFS_ACCELERATE_PYTEST_SEAL": "0"}) is False
    assert seal_enabled({"IPFS_ACCELERATE_PYTEST_SEAL": "off"}) is False
    with pytest.raises(PytestSealCatalogError):
        DuckDbQuackSealOracle(tmp_path / "control.duckdb")
