"""Packaging metadata and runtime __version__ stay aligned."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    assert match is not None, "pyproject.toml must declare version"
    return match.group(1)


def _read_setup_version() -> str:
    text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*,?\s*$', text)
    assert match is not None, "setup.py must declare version"
    return match.group(1)


def test_packaging_files_share_one_version() -> None:
    pyproject = _read_pyproject_version()
    setup = _read_setup_version()
    assert pyproject == setup
    assert pyproject  # non-empty


def test_runtime_version_matches_packaging_pin() -> None:
    """Cold import must report the same pin as packaging metadata."""
    import ipfs_accelerate_py

    packaging = _read_pyproject_version()
    assert getattr(ipfs_accelerate_py, "_PACKAGING_VERSION", None) == packaging
    assert ipfs_accelerate_py.__version__ == packaging
