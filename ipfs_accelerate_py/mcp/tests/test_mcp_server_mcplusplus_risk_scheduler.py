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

    def test_frontier_deterministic_fifo_ties_under_load(self) -> None:
        scheduler = RiskScheduler()

        expected_order = []
        for index in range(50):
            event_cid = f"cid-{index:03d}"
            expected_order.append(event_cid)
            scheduler.enqueue_frontier(
                event_cid=event_cid,
                actor="actor-x",
                expected_value=0.8,
                dependency_ready=True,
            )

        popped = []
        while scheduler.frontier_size() > 0:
            next_item = scheduler.pop_next()
            self.assertIsNotNone(next_item)
            popped.append(next_item.event_cid)

        self.assertEqual(popped, expected_order)

    def test_frontier_retry_penalty_deprioritizes_retries(self) -> None:
        scheduler = RiskScheduler()

        scheduler.enqueue_frontier(
            event_cid="cid-retry",
            actor="actor-a",
            expected_value=0.9,
            dependency_ready=True,
            retry_count=3,
        )
        scheduler.enqueue_frontier(
            event_cid="cid-fresh",
            actor="actor-a",
            expected_value=0.9,
            dependency_ready=True,
            retry_count=0,
        )

        first = scheduler.pop_next()
        second = scheduler.pop_next()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.event_cid, "cid-fresh")
        self.assertEqual(second.event_cid, "cid-retry")

    def test_consensus_signal_is_optional_and_non_breaking(self) -> None:
        scheduler = RiskScheduler()

        baseline = scheduler.enqueue_frontier(
            event_cid="cid-baseline",
            actor="actor-a",
            expected_value=0.7,
            dependency_ready=True,
        )
        without_signal = scheduler.enqueue_frontier(
            event_cid="cid-no-consensus",
            actor="actor-a",
            expected_value=0.7,
            dependency_ready=True,
            consensus_signal={"confidence": 0.0, "disputed": True},
            enable_consensus_signal=False,
        )
        with_signal = scheduler.enqueue_frontier(
            event_cid="cid-with-consensus",
            actor="actor-a",
            expected_value=0.7,
            dependency_ready=True,
            consensus_signal={"confidence": 0.0, "disputed": True},
            enable_consensus_signal=True,
        )

        self.assertEqual(without_signal.priority, baseline.priority)
        self.assertGreater(with_signal.priority, without_signal.priority)

    def test_consensus_signal_confidence_can_prioritize(self) -> None:
        scheduler = RiskScheduler()

        scheduler.enqueue_frontier(
            event_cid="cid-low-confidence",
            actor="actor-a",
            expected_value=0.6,
            dependency_ready=True,
            consensus_signal={"confidence": 0.0},
            enable_consensus_signal=True,
        )
        scheduler.enqueue_frontier(
            event_cid="cid-high-confidence",
            actor="actor-a",
            expected_value=0.6,
            dependency_ready=True,
            consensus_signal={"confidence": 1.0},
            enable_consensus_signal=True,
        )

        first = scheduler.pop_next()
        self.assertIsNotNone(first)
        self.assertEqual(first.event_cid, "cid-high-confidence")


if __name__ == "__main__":
    unittest.main()
