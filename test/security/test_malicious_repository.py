"""EAAEF-121: hostile repositories fail closed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.security.repository_policy import (
    RepositoryPolicyError,
    admit_repository,
)


def test_clean_tree_is_admitted(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("x = 1\n", encoding="utf-8")
    admit_repository(tmp_path)


def test_hook_and_symlink_escape_are_refused(tmp_path: Path) -> None:
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(RepositoryPolicyError, match="hooks"):
        admit_repository(tmp_path)
    other = tmp_path.parent / "outside"
    other.mkdir(exist_ok=True)
    (other / "secret").write_text("nope", encoding="utf-8")
    escape = tmp_path / "link"
    escape.symlink_to(other / "secret")
    # recreate a tree without hooks
    dirty = tmp_path / "escape-root"
    dirty.mkdir()
    (dirty / "link").symlink_to(other / "secret")
    with pytest.raises(RepositoryPolicyError, match="symlink"):
        admit_repository(dirty)
