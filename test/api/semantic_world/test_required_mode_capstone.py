"""SAWM-043 required-mode capstone."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_negative_memory_explain import (
    RequiredModeCapstoneReceipt,
    explain_negative_memory_invalidation,
)


def test_negative_memory_explanation_does_not_complete_or_publish() -> None:
    result = explain_negative_memory_invalidation(
        {"reason_code": "stale_evidence_revoked", "mismatches": ["generation"]}
    )
    assert "stale_evidence_revoked" in result["explanation"]
    assert result["completion_authority"] is False
    assert result["generation_published"] is False
    receipt = RequiredModeCapstoneReceipt(explained=True)
    assert receipt.generation_published is False
    assert receipt.completion_authority is False
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_negative_memory_explain import (
        explain_negative_memory,
    )
    from pathlib import Path
    import json

    explained = explain_negative_memory({"id": "n1"}, tree_id="t", current_tree_id="t")
    assert explained["completion_authority"] is False
    capstone = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "artifacts/agent_supervisor/semantic_addressed_world_model/SAWM-043-capstone.json"
        ).read_text(encoding="utf-8")
    )
    assert capstone["self_hosted"] is False
    assert capstone["completion_authority"] is False
