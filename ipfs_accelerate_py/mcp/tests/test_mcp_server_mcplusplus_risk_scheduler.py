#!/usr/bin/env python3
"""Deterministic tests for MCP++ risk scheduler primitives."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.mcplusplus.risk_scheduler import RiskScheduler


class TestMCPServerMCPPlusPlusRiskScheduler(unittest.TestCase):
    """Validate risk scoring updates and frontier behavior."""

    def test_risk_record_updates_for_denials_and_obligations(self) -> None:
        scheduler = RiskScheduler()

        scheduler.record_outcome(actor="did:model:worker", allowed=False)
        scheduler.record_outcome(actor="did:model:worker", allowed=True, obligations=2, event_cid="cid-1")

        risk = scheduler.get_actor_risk("did:model:worker")
        self.assertEqual(risk["total_invocations"], 2)
        self.assertEqual(risk["denied_count"], 1)
        self.assertEqual(risk["obligation_count"], 2)
        self.assertEqual(risk["last_event_cid"], "cid-1")
        self.assertGreater(risk["score"], 0.0)

    def test_frontier_priority_orders_lower_risk_first(self) -> None:
        scheduler = RiskScheduler()

        # Actor A gets a denial first, increasing risk.
        scheduler.record_outcome(actor="actor-a", allowed=False)

        # Actor B has no denials.
        scheduler.record_outcome(actor="actor-b", allowed=True)

        scheduler.enqueue_frontier(
            event_cid="cid-a",
            actor="actor-a",
            expected_value=0.8,
            dependency_ready=True,
        )
        scheduler.enqueue_frontier(
            event_cid="cid-b",
            actor="actor-b",
            expected_value=0.8,
            dependency_ready=True,
        )

        first = scheduler.pop_next()
        second = scheduler.pop_next()

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.event_cid, "cid-b")
        self.assertEqual(second.event_cid, "cid-a")


if __name__ == "__main__":
    unittest.main()
