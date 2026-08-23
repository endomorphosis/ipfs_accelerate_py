"""EAAEF-024: transfer modes reconstruct declared state or typed-refuse."""

from __future__ import annotations

from ipfs_kit_py.repository_transfer.bundle import TransferError, admit_transfer
import pytest


def test_admitted_modes_do_not_mutate_user_checkout(tmp_path) -> None:
    before = list(tmp_path.iterdir())
    req = admit_transfer(mode="managed_alias", locator="repos/core", alias="core")
    assert req.mode == "managed_alias"
    assert list(tmp_path.iterdir()) == before


def test_host_path_is_typed_refusal() -> None:
    with pytest.raises(TransferError, match="host paths"):
        admit_transfer(mode="git_bundle", locator="/home/user/src.git")
