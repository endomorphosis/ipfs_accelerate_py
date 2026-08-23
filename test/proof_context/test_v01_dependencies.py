"""PCCE-010: accelerator dependency loader."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from ipfs_accelerate_py.proof_context.compatibility import (
    CompatibilityError,
    frozen_matrix,
    reject_mock,
    reject_mutable_ref,
    reject_pseudo_cid,
)
from ipfs_accelerate_py.proof_context.dependencies import (
    DependencyUnavailable,
    require_production_capability,
    resolve_v01_surface,
)


def test_cold_import_creates_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    import ipfs_accelerate_py.proof_context as port
    import ipfs_accelerate_py.proof_context.dependencies as deps
    import ipfs_accelerate_py.proof_context.compatibility as compat

    assert port.SCHEMA.endswith("v0.1")
    after = set(tmp_path.rglob("*"))
    assert after == before
    assert deps.PORT_SCHEMA == port.SCHEMA
    assert "content_id" in compat.frozen_matrix()


def test_missing_ports_are_typed_unavailable_not_success() -> None:
    surface = resolve_v01_surface()
    for cap in surface:
        if not cap.available:
            with pytest.raises(DependencyUnavailable):
                require_production_capability(cap)


def test_production_rejects_mocks_pseudo_cids_and_mutable_refs() -> None:
    with pytest.raises(CompatibilityError):
        reject_mock(Mock())
    with pytest.raises(CompatibilityError):
        reject_pseudo_cid("sha256:deadbeef")
    with pytest.raises(CompatibilityError):
        reject_pseudo_cid("not-a-cid")
    with pytest.raises(CompatibilityError):
        reject_mutable_ref("main")
    reject_pseudo_cid("bafkreiapj52u5hi7pco5ebplvecv72olbnqglg2e7emwnmme4gguzsnpu4")
    pins = frozen_matrix()["repositories"]
    assert pins["endomorphosis/ipfs_datasets_py"]["commit"].startswith("b3669171")
