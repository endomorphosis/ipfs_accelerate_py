#!/usr/bin/env python3
"""Smoke tests for stable package import surfaces in a repository checkout."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_module(name: str):
    return importlib.import_module(name)


def test_top_level_repo_packages_import_with_repo_root_on_syspath(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    scripts_generators = _import_module("scripts.generators")
    data_duckdb = _import_module("data.duckdb")

    assert Path(scripts_generators.__file__).is_file()
    assert Path(data_duckdb.__file__).is_file()


def test_canonical_package_import_surfaces_are_available(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    shared_pkg = _import_module("ipfs_accelerate_py.shared")
    github_cli_pkg = _import_module("ipfs_accelerate_py.github_cli")

    assert hasattr(shared_pkg, "SharedCore")
    assert hasattr(shared_pkg, "GitHubOperations")
    assert hasattr(github_cli_pkg, "GitHubCLI")
    assert hasattr(github_cli_pkg, "WorkflowManager")


def test_safe_subpackages_import_from_repo_checkout(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    generators_utils = _import_module("scripts.generators.utils")
    duckdb_schema = _import_module("data.duckdb.schema.creation")

    assert Path(generators_utils.__file__).is_file()
    assert Path(duckdb_schema.__file__).is_file()
    assert importlib.util.find_spec("data.duckdb.core") is not None
    assert importlib.util.find_spec("data.duckdb.utils") is not None