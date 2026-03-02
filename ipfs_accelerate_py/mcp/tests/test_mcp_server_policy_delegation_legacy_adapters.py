#!/usr/bin/env python3
"""Compatibility adapter tests for UNI-011 policy/delegation legacy surfaces."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.temporal_deontic_mcp_server import TemporalDeonticMCPServer
from ipfs_accelerate_py.mcp_server.temporal_policy import (
    PolicyEvaluator,
    make_simple_permission_policy,
)
from ipfs_accelerate_py.mcp_server.ucan_delegation import (
    Capability,
    Delegation,
    DelegationEvaluator,
)


class _DummyIntent:
    def __init__(self, tool: str, intent_cid: str = "") -> None:
        self.tool = tool
        self.intent_cid = intent_cid


class _DummyServer:
    def __init__(self) -> None:
        self.tools = {
            "tools_list_tools": {"function": self._list_tools},
            "tools_dispatch": {"function": self._dispatch},
            "echo": {
                "description": "echo tool",
                "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
            },
        }

    async def _list_tools(self):
        return [{"category": "demo", "name": "echo"}]

    async def _dispatch(self, category: str, tool_name: str, parameters: dict):
        return {"success": True, "category": category, "tool": tool_name, "parameters": parameters}


class TestMCPServerPolicyDelegationLegacyAdapters(unittest.TestCase):
    def test_ucan_delegation_evaluator_allows_matching_chain(self) -> None:
        evaluator = DelegationEvaluator()
        delegation = Delegation(
            cid="cid-leaf",
            issuer="did:user:alice",
            audience="did:model:worker",
            capabilities=[Capability(resource="mcp://tool/smoke.echo", ability="invoke")],
        )
        evaluator.add(delegation)

        allowed, reason = evaluator.can_invoke(
            leaf_cid="cid-leaf",
            resource="mcp://tool/smoke.echo",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "allowed")

    def test_temporal_policy_evaluator_uses_canonical_decision(self) -> None:
        policy = make_simple_permission_policy(actor="did:model:worker", action="smoke.echo")
        evaluator = PolicyEvaluator()

        result = evaluator.evaluate(
            _DummyIntent(tool="smoke.echo", intent_cid="cid-intent"),
            policy,
            actor="did:model:worker",
        )

        self.assertEqual(result["decision"], "allow")
        self.assertEqual(result["intent_cid"], "cid-intent")
        self.assertIn("justification", result)

    def test_temporal_deontic_server_facade_dispatches_via_meta_tools(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.temporal_deontic_mcp_server.create_server",
                return_value=_DummyServer(),
            ):
                server = TemporalDeonticMCPServer()
                result = await server.call_tool_direct("echo", {"value": "ok"})

            self.assertTrue(result["success"])
            self.assertEqual(result["category"], "demo")
            self.assertEqual(result["tool"], "echo")
            self.assertEqual(result["parameters"], {"value": "ok"})

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
