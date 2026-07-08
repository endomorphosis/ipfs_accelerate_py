#!/usr/bin/env python3
"""Tests for canonical risk scorer."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.risk_scorer import RiskGateError, RiskScorer, RiskScoringPolicy


class TestRiskScorer(unittest.TestCase):
    """Validate deterministic risk scoring and gate behavior."""

    def test_score_intent_returns_expected_components(self) -> None:
        scorer = RiskScorer(
            RiskScoringPolicy(
                tool_risk_overrides={"smoke.echo": 0.6},
                actor_trust_levels={"did:model:worker": 0.2},
            )
        )
        assessment = scorer.score_intent(
            tool="smoke.echo",
            actor="did:model:worker",
            params={"a": 1, "b": 2},
        )
        self.assertEqual(assessment.tool, "smoke.echo")
        self.assertEqual(assessment.actor, "did:model:worker")
        self.assertGreater(assessment.score, 0.0)
        self.assertIn("base_risk", assessment.factors)
        self.assertIn("complexity_penalty", assessment.factors)

    def test_score_and_gate_raises_for_high_risk(self) -> None:
        scorer = RiskScorer(
            RiskScoringPolicy(
                tool_risk_overrides={"danger.tool": 1.0},
                max_acceptable_risk=0.2,
            )
        )
        with self.assertRaises(RiskGateError):
            scorer.score_and_gate(tool="danger.tool", actor="did:model:risky", params={})


if __name__ == "__main__":
    unittest.main()
