#!/usr/bin/env python3
"""Deterministic tests for MCP-IDL registry primitives and native tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import HierarchicalToolManager
from ipfs_accelerate_py.mcp_server.runtime_router import RuntimeRouter
from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    InterfaceDescriptorRegistry,
    build_descriptor,
    canonicalize_descriptor,
    compute_interface_cid,
)
from ipfs_accelerate_py.mcp_server.tools.idl import register_native_idl_tools


class TestMCPServerMCPPlusPlusIDL(unittest.TestCase):
    """Validate Profile A MCP-IDL deterministic behavior."""

    def test_descriptor_canonicalization_is_deterministic(self) -> None:
        descriptor_a = {
            "name": "demo",
            "namespace": "test.ns",
            "version": "1.0.0",
            "methods": [{"name": "x", "input_schema": {"type": "object"}}],
            "errors": [],
            "requires": ["mcp++/profile-a-idl"],
            "compatibility": {"compatible_with": [], "supersedes": []},
        }
        descriptor_b = {
            "version": "1.0.0",
            "namespace": "test.ns",
            "name": "demo",
            "compatibility": {"supersedes": [], "compatible_with": []},
            "requires": ["mcp++/profile-a-idl"],
            "errors": [],
            "methods": [{"input_schema": {"type": "object"}, "name": "x"}],
        }

        self.assertEqual(canonicalize_descriptor(descriptor_a), canonicalize_descriptor(descriptor_b))
        self.assertEqual(compute_interface_cid(descriptor_a), compute_interface_cid(descriptor_b))

    def test_registry_compatibility_missing_requirements(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["mcp++/profile-a-idl"])
        cid = registry.register_descriptor(
            build_descriptor(
                name="restricted",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "restricted.call", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["mcp++/profile-z-nonexistent"],
            )
        )
        verdict = registry.compat(cid)
        self.assertFalse(verdict.compatible)
        self.assertIn("missing_required_capabilities", verdict.reasons)
        self.assertIn("mcp++/profile-z-nonexistent", verdict.requires_missing)

    def test_native_idl_tools_dispatch_flow(self) -> None:
        async def _run() -> None:
            manager = HierarchicalToolManager(runtime_router=RuntimeRouter())
            register_native_idl_tools(
                manager,
                supported_capabilities=[
                    "mcp++/profile-a-idl",
                    "mcp++/profile-b-cid-artifacts",
                ],
            )

            listed = await manager.dispatch("idl", "interfaces_list", {})
            self.assertGreaterEqual(listed.get("count", 0), 1)

            interface_cid = listed["interface_cids"][0]
            descriptor_payload = await manager.dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
            self.assertTrue(descriptor_payload.get("found"))

            verdict = await manager.dispatch("idl", "interfaces_compat", {"interface_cid": interface_cid})
            self.assertTrue(verdict.get("compatible"))

            selected = await manager.dispatch("idl", "interfaces_select", {"task_hint_cid": "", "budget": 1})
            self.assertEqual(selected.get("count"), 1)
            self.assertEqual(len(selected.get("selected_interface_cids", [])), 1)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
