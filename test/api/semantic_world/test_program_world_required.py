"""SAWM-037 required dispatch and completion gates."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_required import (
    admit_required_program_world_dispatch,
)
from ipfs_accelerate_py.agent_supervisor.validation.program_world_completion_gate import (
    REQUIRED_COMPLETION_RECEIPTS,
    admit_required_program_world_completion,
)


def test_dispatch_refuses_missing_receipts() -> None:
    result = admit_required_program_world_dispatch(
        {"task_id": "SAWM-037", "receipts": ["pre_root"]}
    )
    assert result["admitted"] is False
    assert "current_state" in result["missing"]
    assert result["completion_authority"] is False


def test_completion_never_cas_without_full_receipt_set() -> None:
    refused = admit_required_program_world_completion({"receipts": ["pre_root"]})
    assert refused["admitted"] is False
    assert refused["cas_completed"] is False
    admitted = admit_required_program_world_completion(
        {"receipts": list(REQUIRED_COMPLETION_RECEIPTS)}
    )
    assert admitted["admitted"] is True
    assert admitted["cas_completed"] is False
    assert admitted["completion_authority"] is False
