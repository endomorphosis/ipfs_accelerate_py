"""PCCE-012: accelerator delegates ContextPack construction to datasets."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
    ContextPacker,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    CONTEXT_PACK_V01_AUTHORITY,
)
from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable
from ipfs_accelerate_py.proof_context.semantic_bridge import (
    build_v01_context_pack,
    compatibility_packer_is_not_authority,
    datasets_is_v01_authority,
)


def test_legacy_packer_is_compatibility_only() -> None:
    assert ContextPacker.V01_PRODUCTION_AUTHORITY is False
    assert compatibility_packer_is_not_authority() is True
    assert datasets_is_v01_authority() == CONTEXT_PACK_V01_AUTHORITY
    assert "ipfs_datasets_py.proof_context.context_pack" in ContextPacker.V01_DELEGATE


def test_bridge_delegates_or_fails_closed_without_reconstructing() -> None:
    try:
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
        from ipfs_datasets_py.proof_context.context_pack import AUTHORITY
    except ImportError:
        with pytest.raises(DependencyUnavailable):
            build_v01_context_pack(task_id="PCCE-012")
        return

    def _cid(label: str) -> str:
        return cid_for_bytes(label.encode("utf-8"))

    record = build_v01_context_pack(
        repository_state_cid=_cid("repo-state"),
        task_id="PCCE-012",
        target_source_cid=_cid("target"),
        surrounding_source_cid=_cid("surround"),
        test_source_cid=_cid("test"),
        scanned_tree_oid="16ef68abe8a35a3033dfaf1ed4e8d6132600df8f",
        source_tree_oid="16ef68abe8a35a3033dfaf1ed4e8d6132600df8f",
    )
    again = ContextPacker().pack_v01(
        repository_state_cid=_cid("repo-state"),
        task_id="PCCE-012",
        target_source_cid=_cid("target"),
        surrounding_source_cid=_cid("surround"),
        test_source_cid=_cid("test"),
        scanned_tree_oid="16ef68abe8a35a3033dfaf1ed4e8d6132600df8f",
        source_tree_oid="16ef68abe8a35a3033dfaf1ed4e8d6132600df8f",
    )
    assert record.producer == AUTHORITY
    assert again.pack_cid == record.pack_cid
