"""Test API fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
repo_root_text = str(repo_root)

# Pytest may prepend this checkout's parent while importing ``test.api`` as a
# package.  That parent can contain sibling checkouts with the same top-level
# package names (notably ``ipfs_datasets_py``), so merely checking whether the
# local root is present leaves the sibling ahead of the nested submodule.
# Re-promote the local checkout deterministically before API tests are imported.
sys.path[:] = [entry for entry in sys.path if entry != repo_root_text]
sys.path.insert(0, repo_root_text)

# Pin the package while the local checkout is first.  Later imports from
# ipfs_accelerate_py/ipfs_kit_py may add sibling roots to ``sys.path``.
import ipfs_datasets_py as _ipfs_datasets_py  # noqa: E402,F401
