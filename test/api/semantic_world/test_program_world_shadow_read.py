"""SAWM-035 shadow_read."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_shadow_read import (
    evaluate_program_world_shadow_reads,
)


def test_shadow_read_does_not_change_execution() -> None:
    result = evaluate_program_world_shadow_reads(
        {
            "query": "reuse?",
            "authoritative_hits": ["a"],
            "shadow_hits": ["a", "b"],
        }
    )
    assert result["execution_changed"] is False
    assert result["completion_authority"] is False
    assert result["comparison"].false_candidates == ("b",)
    assert result["decision"].would_reuse is False


def test_missing_binding_is_unavailable() -> None:
    result = evaluate_program_world_shadow_reads({"binding_missing": True})
    assert result["status"] == "unavailable"
    assert result["reason_code"] == "missing_binding"
    assert result["execution_changed"] is False
