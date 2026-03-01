#!/usr/bin/env python3
"""Tests for unified MCP policy audit logging."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.policy_audit_log import PolicyAuditLog


class TestPolicyAuditLog(unittest.TestCase):
    """Validate deterministic policy audit log behavior."""

    def test_record_and_stats_when_enabled(self) -> None:
        log = PolicyAuditLog(enabled=True, max_entries=10)

        log.record(decision="allow", tool="smoke.echo", actor="did:model:worker", intent_cid="cid-i")
        log.record(
            decision="policy_denied",
            tool="smoke.echo",
            actor="did:model:worker",
            intent_cid="cid-j",
            justification="policy denied",
        )

        stats = log.stats()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["total_recorded"], 2)
        self.assertEqual(stats["in_memory"], 2)
        self.assertEqual(stats["allow_count"], 1)
        self.assertEqual(stats["deny_count"], 1)
        self.assertEqual(stats["by_decision"]["policy_denied"], 1)

    def test_disabled_log_is_noop(self) -> None:
        log = PolicyAuditLog(enabled=False, max_entries=10)
        entry = log.record(decision="allow", tool="smoke.echo", actor="did:model:worker")
        self.assertIsNone(entry)
        stats = log.stats()
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["total_recorded"], 0)
        self.assertEqual(stats["in_memory"], 0)

    def test_ring_buffer_eviction(self) -> None:
        log = PolicyAuditLog(enabled=True, max_entries=2)
        log.record(decision="allow", tool="smoke.echo", actor="a", intent_cid="cid-1")
        log.record(decision="allow", tool="smoke.echo", actor="a", intent_cid="cid-2")
        log.record(decision="allow", tool="smoke.echo", actor="a", intent_cid="cid-3")

        entries = log.recent(5)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].intent_cid, "cid-2")
        self.assertEqual(entries[1].intent_cid, "cid-3")


if __name__ == "__main__":
    unittest.main()
