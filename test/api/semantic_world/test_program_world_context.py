"""SAWM-022 semantic context capsules and compression."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.context.program_world_context import (
    ProgramWorldContextError,
    ProgramWorldContextPlanner,
    compile_program_world_context,
    explain_program_world_context,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _material(kind: str, *, required: bool = False, tokens: int = 4, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "identity_cid": _cid(kind),
        "kind": kind,
        "reason": extra.pop("reason", "declared"),
        "authority": extra.pop("authority", "program-world"),
        "freshness": extra.pop("freshness", "fresh"),
        "required": required,
        "tokens": tokens,
        "source_cid": _cid(f"source-{kind}"),
    }
    payload.update(extra)
    return payload


def test_assembly_order_and_inclusion_reasons() -> None:
    receipt = compile_program_world_context(
        {
            "token_budget": 100,
            "materials": [
                _material("proofs", required=True),
                _material("goal", required=True),
                _material("analogues"),
                _material("tests", required=True),
            ],
        }
    )
    assert [item.kind for item in receipt.included] == ["goal", "tests", "proofs", "analogues"]
    assert all(item.source_cid for item in receipt.included)
    assert receipt.token_budget == 100


def test_required_tests_proofs_policy_cannot_be_omitted() -> None:
    with pytest.raises(ProgramWorldContextError, match="cannot be omitted"):
        compile_program_world_context(
            {
                "token_budget": 100,
                "materials": [
                    {**_material("tests", required=True), "disposition": "omitted"},
                ],
            }
        )


def test_embeddings_cannot_suppress_required_material() -> None:
    with pytest.raises(ProgramWorldContextError, match="embeddings"):
        compile_program_world_context(
            {
                "token_budget": 100,
                "materials": [
                    _material("proofs", required=True, reason="ann similarity"),
                ],
            }
        )


def test_unresolved_question_forces_expansion() -> None:
    with pytest.raises(ProgramWorldContextError, match="unresolved questions"):
        compile_program_world_context(
            {
                "token_budget": 100,
                "unresolved_questions": ["what is the current obligation?"],
                "materials": [_material("goal", required=True)],
            }
        )


def test_raw_fallback_and_token_budget_omissions() -> None:
    receipt = compile_program_world_context(
        {
            "token_budget": 6,
            "materials": [
                _material("goal", required=True, tokens=2),
                _material("analogues", tokens=8),
                {**_material("raw_fallback", tokens=2), "disposition": "raw_fallback"},
            ],
        }
    )
    assert [item.kind for item in receipt.included] == ["goal"]
    assert [item.kind for item in receipt.omitted] == ["analogues"]
    assert [item.kind for item in receipt.raw_fallbacks] == ["raw_fallback"]
    assert receipt.fallback is True


def test_prefix_reuse_is_exact_bytes() -> None:
    planner = ProgramWorldContextPlanner(token_budget=32)
    prefix = {"goal": _cid("goal"), "state": _cid("state")}
    first = planner.reuse_prefix(prefix)
    second = planner.reuse_prefix(prefix)
    assert first.exact_bytes is True
    assert first.reused is False
    assert second.reused is True
    assert first.prefix_cid == second.prefix_cid


def test_context_compiler_owns_program_world_entry() -> None:
    compiler = ContextCompiler(ContextBudget(max_input_tokens=64))
    receipt = compiler.compile_program_world(
        {"token_budget": 64, "materials": [_material("goal", required=True)]}
    )
    assert receipt.included[0].kind == "goal"
    explanation = explain_program_world_context(
        {"token_budget": 64, "materials": [_material("goal", required=True)]}
    )
    assert explanation["included"] == ["goal"]
