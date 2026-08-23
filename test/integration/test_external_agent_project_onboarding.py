"""EAAEF-044: qualify onboarding against supported, unsupported and malicious fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.assessment import (
    assess_repository,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.base import (
    GenericProjectAdapter,
    SupportOutcome,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.python import (
    PythonProjectAdapter,
)
from ipfs_accelerate_py.agent_supervisor.security.repository_policy import (
    RepositoryPolicyError,
    admit_repository,
)


MUTATION_CANDIDATE_OUTCOMES = frozenset(
    {"supported_inventory", "mutation_not_admitted"}
)
UNSUPPORTED_OUTCOMES = frozenset(
    {
        "unsupported_language",
        "preview_only",
        "unsupported_build_system",
    }
)


def _python_with_tests(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.1'\n\n"
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        encoding="utf-8",
    )
    (root / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_demo.py").write_text(
        "def test_value() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    return root


def _python_without_tests(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'bare'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    (root / "src" / "bare.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def test_supported_python_fixture_is_inventory_not_live_mutation(tmp_path: Path) -> None:
    root = _python_with_tests(tmp_path / "python-ok")
    assessment = assess_repository(root)
    assert assessment["outcome"] == "supported_inventory"
    assert "python" in assessment["languages"]
    assert assessment["mutation_admitted"] is False
    support = GenericProjectAdapter().inspect(root)
    assert support.outcome is SupportOutcome.SUPPORTED_INVENTORY
    assert support.mutation_admitted is False
    adapter = PythonProjectAdapter()
    python_support = adapter.inspect(root)
    assert "python" in python_support.languages
    assert adapter.mutation_admitted(python_support) is False
    missing_tests = assess_repository(_python_without_tests(tmp_path / "python-bare"))
    assert missing_tests["outcome"] in {
        "supported_inventory",
        "mutation_not_admitted",
        "insufficient_validation",
    }
    assert missing_tests["mutation_admitted"] is False


def test_unsupported_language_is_not_mutation_admitted(tmp_path: Path) -> None:
    root = tmp_path / "rust-only"
    root.mkdir()
    (root / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    assessment = assess_repository(root)
    assert assessment["outcome"] in UNSUPPORTED_OUTCOMES
    assert assessment["mutation_admitted"] is False
    support = GenericProjectAdapter().inspect(root)
    assert support.outcome.value in UNSUPPORTED_OUTCOMES
    assert support.mutation_admitted is False
    assert PythonProjectAdapter().mutation_admitted(support) is False


def test_malicious_repository_is_unsafe_and_not_mutation_admitted(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks-root"
    hooks.mkdir()
    (hooks / ".git" / "hooks").mkdir(parents=True)
    (hooks / ".git" / "hooks" / "pre-commit").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(RepositoryPolicyError, match="hooks"):
        admit_repository(hooks)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("nope\n", encoding="utf-8")
    escape = tmp_path / "escape-root"
    escape.mkdir()
    os.symlink(outside / "secret", escape / "link")
    with pytest.raises(RepositoryPolicyError, match="symlink"):
        admit_repository(escape)
    escaped = assess_repository(escape)
    assert escaped["outcome"] == "unsafe_repository"
    assert escaped["mutation_admitted"] is False

    secrets = tmp_path / "secrets-root"
    secrets.mkdir()
    (secrets / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (secrets / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    with pytest.raises(RepositoryPolicyError):
        admit_repository(secrets)
    secret_support = GenericProjectAdapter().inspect(secrets)
    assert secret_support.mutation_admitted is False
    assert PythonProjectAdapter().mutation_admitted(secret_support) is False
