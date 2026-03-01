#!/usr/bin/env python3
"""Deterministic tests for MCP++ UCAN delegation validation."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
    parse_delegation_chain,
    validate_delegation_chain,
    validate_raw_delegation_chain,
)


class TestMCPServerMCPPlusPlusUCAN(unittest.TestCase):
    """Validate Profile C execution-time authorization checks."""

    def test_allows_valid_chain_for_leaf_actor_and_capability(self) -> None:
        raw_chain = [
            {
                "issuer": "did:user:alice",
                "audience": "did:model:planner",
                "capabilities": [{"resource": "*", "ability": "invoke"}],
            },
            {
                "issuer": "did:model:planner",
                "audience": "did:model:worker",
                "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
            },
        ]
        result = validate_raw_delegation_chain(
            raw_chain=raw_chain,
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")
        self.assertEqual(result.chain_length, 2)

    def test_denies_capability_escalation(self) -> None:
        raw_chain = [
            {
                "issuer": "did:user:alice",
                "audience": "did:model:planner",
                "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
            },
            {
                "issuer": "did:model:planner",
                "audience": "did:model:worker",
                "capabilities": [{"resource": "workflow.delete_workflow", "ability": "invoke"}],
            },
        ]
        result = validate_raw_delegation_chain(
            raw_chain=raw_chain,
            resource="workflow.delete_workflow",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertFalse(result.allowed)
        self.assertIn("capability_escalation", result.reason)

    def test_denies_expired_chain(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "*", "ability": "invoke"}],
                    "expiry": 1.0,
                }
            ]
        )
        result = validate_delegation_chain(
            chain=chain,
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            now=2.0,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "expired_at_hop_0")


if __name__ == "__main__":
    unittest.main()
