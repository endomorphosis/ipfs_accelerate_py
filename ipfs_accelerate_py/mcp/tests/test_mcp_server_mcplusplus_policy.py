#!/usr/bin/env python3
"""Deterministic tests for MCP++ temporal policy evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
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

    def test_obligation_lifecycle_progression_respects_validity_window(self) -> None:
        raw_clauses = [
            {
                "clause_type": "permission",
                "actor": "did:model:worker",
                "action": "smoke.echo",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2026-12-31T23:59:59Z",
            },
            {
                "clause_type": "obligation",
                "actor": "did:model:worker",
                "action": "smoke.echo",
                "obligation_deadline": "2026-06-01T00:00:00Z",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2026-12-31T23:59:59Z",
            },
        ]

        before = evaluate_raw_policy(
            raw_clauses=raw_clauses,
            actor="did:model:worker",
            action="smoke.echo",
            now=datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc),
        )
        self.assertEqual(before.decision, "deny")
        self.assertEqual(before.obligations, [])

        during = evaluate_raw_policy(
            raw_clauses=raw_clauses,
            actor="did:model:worker",
            action="smoke.echo",
            now=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(during.decision, "allow_with_obligations")
        self.assertEqual(len(during.obligations), 1)
        self.assertEqual(during.obligations[0].get("deadline"), "2026-06-01T00:00:00Z")

        after = evaluate_raw_policy(
            raw_clauses=raw_clauses,
            actor="did:model:worker",
            action="smoke.echo",
            now=datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(after.decision, "deny")
        self.assertEqual(after.obligations, [])

    def test_obligation_deadline_status_is_pending_then_overdue(self) -> None:
        raw_clauses = [
            {
                "clause_type": "permission",
                "actor": "did:model:worker",
                "action": "smoke.echo",
            },
            {
                "clause_type": "obligation",
                "actor": "did:model:worker",
                "action": "smoke.echo",
                "obligation_deadline": "2026-06-01T00:00:00Z",
            },
        ]

        pending = evaluate_raw_policy(
            raw_clauses=raw_clauses,
            actor="did:model:worker",
            action="smoke.echo",
            now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(pending.decision, "allow_with_obligations")
        self.assertEqual(pending.obligations[0].get("status"), "pending")

        overdue = evaluate_raw_policy(
            raw_clauses=raw_clauses,
            actor="did:model:worker",
            action="smoke.echo",
            now=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(overdue.decision, "allow_with_obligations")
        self.assertEqual(overdue.obligations[0].get("status"), "overdue")

    def test_fulfilled_obligation_no_longer_requires_outstanding_work(self) -> None:
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
                    "metadata": {
                        "fulfilled": True,
                        "fulfilled_at": "2026-02-01T00:00:00Z",
                        "ticket": "obl-7",
                    },
                },
            ],
            actor="did:model:worker",
            action="smoke.echo",
            now=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(decision.decision, "allow")
        self.assertEqual(decision.obligations, [])
        self.assertIn("already fulfilled", decision.justification)

    def test_policy_version_metadata_is_preserved_in_obligations(self) -> None:
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
                    "metadata": {"policy_version": "v2", "migration": "2026-q1"},
                },
            ],
            actor="did:model:worker",
            action="smoke.echo",
        )
        self.assertEqual(decision.decision, "allow_with_obligations")
        self.assertEqual(len(decision.obligations), 1)
        metadata = decision.obligations[0].get("metadata") or {}
        self.assertEqual(metadata.get("policy_version"), "v2")
        self.assertEqual(metadata.get("migration"), "2026-q1")


if __name__ == "__main__":
    unittest.main()
