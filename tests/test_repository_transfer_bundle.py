"""Accelerator-owned copy of EAAEF-021 transfer tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

KIT_ROOT = Path(__file__).resolve().parents[1] / "ipfs_kit_py"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from ipfs_kit_py.repository_transfer.bundle import TransferError, admit_transfer


def test_refuses_host_paths_and_admits_aliases() -> None:
    req = admit_transfer(mode="managed_alias", locator="repos/core", alias="core")
    assert req.alias == "core"
    with pytest.raises(TransferError):
        admit_transfer(mode="git_bundle", locator="/var/lib/git/repo.git")
