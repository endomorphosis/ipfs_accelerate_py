"""Deterministic tests for the safe generic ProjectAdapter (EAAEF-040)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import (
    ADAPTER_ID,
    INVENTORY_AUTHORIZES_MUTATION,
    ProjectAdapter,
    SupportOutcome,
    inspect_project,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.base import (
    GenericProjectAdapter,
    InventoryBounds,
    ProjectSupport,
)


def _python_project(root: Path) -> Path:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "example"\n'
        'version = "0.0.1"\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        "testpaths = [\"tests\"]\n"
        "\n"
        "[tool.ruff]\n"
        "line-length = 88\n",
        encoding="utf-8",
    )
    (root / "src" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_example.py").write_text(
        "from example import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    (root / "ruff.toml").write_text("line-length = 88\n", encoding="utf-8")
    (root / "README.md").write_text(
        "Run `pytest -q` or `python -m compileall .` after editing.\n",
        encoding="utf-8",
    )
    return root


def test_python_project_inventory_is_supported_without_mutation_argv(
    tmp_path: Path,
) -> None:
    root = tmp_path / "python-project"
    root.mkdir()
    _python_project(root)

    result = inspect_project(root)

    assert result.outcome is SupportOutcome.SUPPORTED_INVENTORY
    assert "python" in result.languages
    assert "python" in result.build_systems
    assert result.test_signals
    assert result.static_signals
    assert result.mutation_admitted is False
    assert result.mutation_argv == ()
    assert INVENTORY_AUTHORIZES_MUTATION is False
    assert result.adapter_id == ADAPTER_ID
    mapping = result.as_mapping()
    assert mapping["mutation_argv"] == ()
    assert mapping["mutation_admitted"] is False


def test_unknown_language_is_unsupported(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    (root / "blob.xyz").write_text("not a recognized language\n", encoding="utf-8")
    (root / "notes.txt").write_text("plain notes\n", encoding="utf-8")

    result = inspect_project(root)

    assert result.outcome is SupportOutcome.UNSUPPORTED_LANGUAGE
    assert result.languages == ()
    assert result.mutation_argv == ()


def test_empty_repo_is_preview_only(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")

    result = inspect_project(root)

    assert result.outcome is SupportOutcome.PREVIEW_ONLY
    assert result.languages == ()
    assert result.files_visited == 0
    assert ".git" in result.skipped_paths
    assert ".env" in result.skipped_paths
    assert result.mutation_argv == ()


def test_malicious_symlink_is_unsafe_repository(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("escape\n", encoding="utf-8")
    root = tmp_path / "linked"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    os.symlink(outside, root / "escape.py")

    result = inspect_project(root)

    assert result.outcome is SupportOutcome.UNSAFE_REPOSITORY
    assert "symlink" in result.reason
    assert result.mutation_argv == ()


def test_huge_tree_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "huge"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'huge'\n", encoding="utf-8")
    for index in range(12):
        (root / f"file_{index}.py").write_text(f"N = {index}\n", encoding="utf-8")

    result = inspect_project(root, max_files=8)

    assert result.outcome is SupportOutcome.UNSAFE_REPOSITORY
    assert "file count" in result.reason
    assert result.mutation_argv == ()


def test_language_without_build_system(tmp_path: Path) -> None:
    root = tmp_path / "src-only"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = inspect_project(root)

    assert result.outcome is SupportOutcome.UNSUPPORTED_BUILD_SYSTEM
    assert result.languages == ("python",)
    assert result.build_systems == ()


def test_build_without_validation_is_insufficient(tmp_path: Path) -> None:
    root = tmp_path / "no-checks"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'no-checks'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = inspect_project(root)

    assert result.outcome is SupportOutcome.INSUFFICIENT_VALIDATION
    assert "python" in result.languages
    assert "python" in result.build_systems
    assert result.test_signals == ()
    assert result.static_signals == ()


def test_generic_adapter_never_admits_mutation(tmp_path: Path) -> None:
    root = tmp_path / "python-project"
    root.mkdir()
    _python_project(root)
    adapter = GenericProjectAdapter()

    inventory = adapter.inspect(root)
    admitted = adapter.admit_mutation(inventory=inventory)

    assert inventory.outcome is SupportOutcome.SUPPORTED_INVENTORY
    assert admitted.outcome is SupportOutcome.MUTATION_NOT_ADMITTED
    assert adapter.mutation_commands(inventory) == ()
    assert admitted.mutation_argv == ()
    assert admitted.mutation_admitted is False


def test_skips_vcs_and_secret_paths_on_python_project(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    _python_project(root)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    (root / "hidden.pem").write_text("cert\n", encoding="utf-8")

    result = ProjectAdapter().inventory(root)

    assert result.outcome is SupportOutcome.SUPPORTED_INVENTORY
    assert ".git" in result.skipped_paths
    assert "id_rsa" in result.skipped_paths
    assert "hidden.pem" in result.skipped_paths
    relative_signals = {item.path for item in result.signals}
    assert "id_rsa" not in relative_signals
    assert "hidden.pem" not in relative_signals


def test_support_record_rejects_fabricated_mutation_argv() -> None:
    with pytest.raises(ValueError, match="mutation argv"):
        ProjectSupport(
            outcome=SupportOutcome.SUPPORTED_INVENTORY,
            mutation_argv=("pytest", "-q"),
        )
    with pytest.raises(ValueError, match="admit mutation"):
        ProjectSupport(
            outcome=SupportOutcome.SUPPORTED_INVENTORY,
            mutation_admitted=True,
        )


def test_inventory_bounds_reject_non_positive() -> None:
    with pytest.raises(ValueError, match="max_files"):
        InventoryBounds(max_files=0)
