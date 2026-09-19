"""SAWM-034 shadow_write."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_shadow_write import (
    ProgramWorldShadowWriter,
    ShadowWriteError,
    record_program_world_shadow_artifacts,
)


def test_shadow_cannot_be_authoritative_or_complete() -> None:
    writer = ProgramWorldShadowWriter()
    receipt = record_program_world_shadow_artifacts(
        {"artifact_id": "a1", "kind": "prediction", "cid": "bafy-1"},
        writer=writer,
    )
    assert receipt.authoritative is False
    assert receipt.influences_planning is False
    assert receipt.influences_routing is False
    assert receipt.completion_authority is False
    with pytest.raises(ShadowWriteError, match="authoritative"):
        record_program_world_shadow_artifacts(
            {"artifact_id": "a2", "authoritative": True}, writer=writer
        )
