#!/usr/bin/env python3
"""UNI-008 runtime pipeline/context convergence tests."""

import os
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import HierarchicalToolManager
from ipfs_accelerate_py.mcp_server.mcp_interfaces import MCPServerProtocol, ToolManagerProtocol
from ipfs_accelerate_py.mcp_server.server_context import UnifiedServerContext


class _DummyServer:
    def __init__(self):
        self.tools = {}
        self.mcp = None

    def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
        self.tools[name] = {
            "function": function,
            "description": description,
            "input_schema": input_schema,
            "execution_context": execution_context,
            "tags": tags,
        }

    def validate_p2p_message(self, _message):
        return True


class TestUNI008RuntimeContext(unittest.TestCase):
    """Validate canonical runtime context and protocol contracts."""

    def test_unified_server_context_snapshot_is_stable(self) -> None:
        context = UnifiedServerContext(
            runtime_router=object(),
            tool_manager=object(),
            services={"a": object(), "b": object()},
            preloaded_categories=["ipfs", "workflow"],
            supported_profiles=["mcp++/profile-a-idl"],
            bootstrap_enabled=True,
        )

        snapshot = context.snapshot()
        self.assertTrue(snapshot.get("bootstrap_enabled"))
        self.assertEqual(snapshot.get("preloaded_categories"), ["ipfs", "workflow"])
        self.assertEqual(snapshot.get("supported_profiles"), ["mcp++/profile-a-idl"])
        self.assertEqual(snapshot.get("service_count"), 2)
        self.assertEqual(snapshot.get("services"), ["a", "b"])

        negotiation = snapshot.get("profile_negotiation") or {}
        self.assertTrue(negotiation.get("supports_profile_negotiation"))
        self.assertEqual(negotiation.get("mode"), "optional_additive")
        self.assertEqual(negotiation.get("profiles"), ["mcp++/profile-a-idl"])

        snapshot["preloaded_categories"].append("p2p")
        self.assertEqual(context.preloaded_categories, ["ipfs", "workflow"])

    def test_unified_bootstrap_attaches_context_snapshot(self) -> None:
        with patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", return_value=_DummyServer()):
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                    "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs,p2p",
                },
                clear=False,
            ):
                server = create_mcp_server(name="uni008-context")

        context = getattr(server, "_unified_server_context", None)
        snapshot = getattr(server, "_unified_server_context_snapshot", None)
        self.assertIsNotNone(context)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot, context.snapshot())
        self.assertEqual(snapshot.get("preloaded_categories"), ["ipfs", "p2p"])

    def test_protocol_runtime_checkable_contracts(self) -> None:
        server = _DummyServer()
        manager = HierarchicalToolManager()

        self.assertIsInstance(server, MCPServerProtocol)
        self.assertIsInstance(manager, ToolManagerProtocol)


if __name__ == "__main__":
    unittest.main()
