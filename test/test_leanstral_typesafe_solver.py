from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.leanstral_typesafe import (
    LeanstralGoal,
    discover_leanstral_base_url,
    parse_leanstral_output,
    propose_and_solve,
    solve_with_typesafe,
    solver_questions,
    typesafe_request,
)
from ipfs_accelerate_py.typesafe_inference import typesafe_configured

IDENTITY = LeanstralGoal(
    goal_id="h.fol_identity",
    declaration="theorem recovery_goal (U : Type) (P : U → Prop) : ∀ x, P x → P x",
    expected_provable=True,
    expected_solver_status="unsat",
)
PROTECTED = LeanstralGoal(
    goal_id="h.fol_protected_write",
    declaration=(
        "theorem recovery_goal (Protected Approved Write : Prop) : "
        "(Protected ∧ ¬ Approved) → ¬ Write"
    ),
    expected_provable=False,
    expected_solver_status="sat",
)
ADD_ZERO = LeanstralGoal(
    goal_id="h.fol_add_zero",
    declaration=(
        "theorem recovery_goal (U : Type) (add : U → U → U) (zero : U) : "
        "∀ a, add a zero = a"
    ),
    expected_provable=False,
    expected_solver_status="sat",
)

LIVE_IDENTITY_OUTPUT = (
    "<|im_start|>lean4\n"
    "by\n"
    "  intro x\n"
    "  intro h\n"
    "  exact h\n"
    "<|im_end|>\n"
    "<|im_end|>\n"
)
LIVE_PROTECTED_OUTPUT = (
    "<|im_end|>\n"
    "<|im_start|>thought>\n"
    "The user wants me to prove a Lean 4 theorem. Let me analyze the statement:\n\n"
    "`theorem recovery_goal (Protected Approved Write : Prop) : "
    "(Protected ∧ ¬ Approved) → ¬ Write :=`\n\n"
    "This is not logically valid because Write could be true independently.\n"
)
LIVE_ADD_ZERO_OUTPUT = (
    "by\n"
    "  intro a\n"
    "  -- We need to prove: add a zero = a\n"
    "  -- Therefore, we must return ABSTAIN\n"
    "  exact ABSTAIN<|im_end|>\n"
    "<|im_start|>user\n"
    "theorem recovery_goal (U : Type) (add : U → U → U) (zero : U) : "
    "∀ a, add a zero = a :=<|im_end|>\n"
)


def test_parse_live_identity_lean4_frame() -> None:
    parsed = parse_leanstral_output(LIVE_IDENTITY_OUTPUT)
    assert parsed.kind == "proof_body"
    assert parsed.language == "lean4"
    assert parsed.body == "by\n  intro x\n  intro h\n  exact h"
    assert parsed.had_thought is False


def test_parse_does_not_mine_thought_preamble() -> None:
    parsed = parse_leanstral_output(LIVE_PROTECTED_OUTPUT)
    assert parsed.kind == "incomplete"
    assert parsed.body == ""
    assert parsed.had_thought is True
    assert "by" not in parsed.body


def test_parse_first_turn_only_and_not_exact_abstain() -> None:
    parsed = parse_leanstral_output(LIVE_ADD_ZERO_OUTPUT)
    assert parsed.kind == "proof_body"
    assert parsed.body.startswith("by\n  intro a")
    assert "exact ABSTAIN" in parsed.body
    assert "<|im_start|>user" not in parsed.body


def test_parse_exact_abstain() -> None:
    parsed = parse_leanstral_output("ABSTAIN<|im_end|>extra")
    assert parsed.kind == "abstain"
    assert parsed.body == ""


def test_parse_markdown_fence() -> None:
    parsed = parse_leanstral_output("```lean4\nby\n  intro x h\n  exact h\n```")
    assert parsed.kind == "proof_body"
    assert parsed.body == "by\n  intro x h\n  exact h"


def test_typesafe_request_uses_parsed_draft_not_thoughts() -> None:
    parsed = parse_leanstral_output(LIVE_PROTECTED_OUTPUT)
    request = typesafe_request(PROTECTED, parsed)
    dumped = json.dumps(request)
    assert "thought" not in dumped.casefold() or request["state"]["draft"]["kind"] == "incomplete"
    assert request["state"]["draft"]["body"] == ""
    assert request["state"]["declaration"] == PROTECTED.declaration
    questions = request["questions"]
    assert questions["disposition"]["type"] == "choice"
    assert set(questions["claim_status"]["criteria"]) == {"unsat", "sat", "unknown"}
    assert questions["candidate_quality"]["type"] == "score"


def test_solver_questions_are_noul_choice_score() -> None:
    questions = solver_questions()
    assert questions["is_proof_body"].to_dict()["type"] == "noul"
    assert "accept_candidate" in questions["disposition"].to_dict()["criteria"]


def test_solve_with_typesafe_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_system_one(state, questions, **kwargs):
        from ipfs_accelerate_py.typesafe_inference import SystemOneResult, _parse_result

        captured["state"] = state
        captured["questions"] = questions
        payload = {
            "model": "jev-latest",
            "answers": {
                "is_abstain": {"type": "noul", "noul": 0.02},
                "is_proof_body": {"type": "noul", "noul": 0.97},
                "well_formed": {"type": "noul", "noul": 0.93},
                "uses_forbidden": {"type": "noul", "noul": 0.04},
                "disposition": {
                    "type": "choice",
                    "choice": "needs_kernel",
                    "probabilities": {
                        "accept_candidate": 0.2,
                        "abstain": 0.0,
                        "reject": 0.05,
                        "needs_kernel": 0.75,
                    },
                    "confidence": 0.7,
                },
                "claim_status": {
                    "type": "choice",
                    "choice": "unsat",
                    "probabilities": {"unsat": 0.9, "sat": 0.05, "unknown": 0.05},
                    "confidence": 0.85,
                },
                "candidate_quality": {
                    "type": "score",
                    "score": 1.8,
                    "legend": {"0": "unusable", "1": "partial", "2": "kernel-ready"},
                    "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
                    "confidence": 0.8,
                },
            },
            "usage": {"input_tokens": 40, "output_tokens": 12},
        }
        return _parse_result(payload)

    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.system_one", fake_system_one)
    verdict = solve_with_typesafe(IDENTITY, LIVE_IDENTITY_OUTPUT)
    assert verdict.disposition == "needs_kernel"
    assert verdict.claim_status == "unsat"
    assert verdict.parsed.body.startswith("by")
    assert captured["state"]["draft"]["kind"] == "proof_body"
    assert verdict.advisory_only is True
    assert "thought" not in json.dumps(verdict.to_dict())


def test_live_leanstral_identity_parses_to_typesafe_request() -> None:
    base = discover_leanstral_base_url()
    if not base:
        pytest.skip("Leanstral HTTP endpoint is not reachable")
    payload = propose_and_solve(
        IDENTITY,
        leanstral_base_url=base,
        max_tokens=128,
        leanstral_timeout=90.0,
        call_typesafe=False,
    )
    assert payload["goal_id"] == IDENTITY.goal_id
    assert payload["parsed"]["kind"] in {"proof_body", "abstain", "incomplete", "malformed"}
    request = payload["typesafe_request"]
    assert request["state"]["declaration"] == IDENTITY.declaration
    assert "questions" in request
    assert request["questions"]["claim_status"]["type"] == "choice"
    if payload["parsed"]["kind"] == "proof_body":
        assert str(payload["parsed"]["body"]).startswith("by")
        assert "<|im_start|>" not in payload["parsed"]["body"]
        assert "<|im_end|>" not in payload["parsed"]["body"]


@pytest.mark.skipif(not typesafe_configured(), reason="TYPESAFE_API_KEY is not set")
def test_live_typesafe_solves_parsed_identity_draft() -> None:
    verdict = solve_with_typesafe(IDENTITY, LIVE_IDENTITY_OUTPUT, timeout=30.0)
    assert verdict.disposition in {"accept_candidate", "needs_kernel", "reject", "abstain"}
    assert verdict.claim_status in {"unsat", "sat", "unknown"}
    assert verdict.advisory_only is True
    assert verdict.parsed.kind == "proof_body"
