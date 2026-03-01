#!/usr/bin/env python3
"""Deterministic tests for MCP++ temporal policy evaluation."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.mcplusplus.policy_engine import (
    evaluate_raw_policy,
    parse_policy_clauses,
)


class TestMCPServerMCPPlusPlusPolicy(unittest.TestCase):
    """Validate Profile D decision semantics."""

    def test_policy_denies_on_matching_prohibition(self) -> None:
        decision = evaluate_raw_policy(
            raw_clauses=[
                {
                    "clause_type": "prohibition",
                    "actor": "did:model:worker",
                    "action": "smoke.echo",
                }
            ],
            actor="did:model:worker",
            action="smoke.echo",
        )
        self.assertEqual(decision.decision, "deny")

    def test_policy_allows_with_obligations(self) -> None:
        decision = evaluate_raw_policy(
            raw_clauses=[
                {
                    "clause_type": "permission",
                    "actor": "did:model:worker",
                    "action": "smoke.echo",
                },
                {
                    "clause_type": "obligation",
                    "actor": "did:model:worker",
                    "action": "smoke.echo",
                    "obligation_deadline": "2030-01-01T00:00:00Z",
                },
            ],
            actor="did:model:worker",
            action="smoke.echo",
        )
        self.assertEqual(decision.decision, "allow_with_obligations")
        self.assertEqual(len(decision.obligations), 1)

    def test_parse_policy_clauses_tolerates_non_dict_entries(self) -> None:
        clauses = parse_policy_clauses([{"clause_type": "permission", "actor": "*", "action": "*"}, "bad", 123])
        self.assertEqual(len(clauses), 1)


if __name__ == "__main__":
    unittest.main()
