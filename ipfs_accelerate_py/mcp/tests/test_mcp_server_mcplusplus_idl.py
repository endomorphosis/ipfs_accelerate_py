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

    def _canonical_descriptor(self, *, name: str, methods: list[dict], requires: list[str]) -> dict:
        return {
            "name": name,
            "namespace": "test.ns",
            "version": "1.0.0",
            "methods": methods,
            "errors": [],
            "requires": requires,
            "compatibility": {"compatible_with": [], "supersedes": []},
        }

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

    def test_descriptor_cid_deterministic_for_stability_corpus(self) -> None:
        corpus = [
            (
                {
                    "name": "alpha",
                    "namespace": "test.ns",
                    "version": "1.0.0",
                    "methods": [{"name": "alpha/a", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                    "errors": [],
                    "requires": ["mcp++/profile-a-idl"],
                    "compatibility": {"compatible_with": [], "supersedes": []},
                },
                {
                    "compatibility": {"supersedes": [], "compatible_with": []},
                    "requires": ["mcp++/profile-a-idl"],
                    "errors": [],
                    "methods": [{"output_schema": {"type": "object"}, "input_schema": {"type": "object"}, "name": "alpha/a"}],
                    "version": "1.0.0",
                    "namespace": "test.ns",
                    "name": "alpha",
                },
            ),
            (
                {
                    "name": "beta",
                    "namespace": "test.ns",
                    "version": "2.1.0",
                    "methods": [
                        {"name": "beta/x", "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}}, "output_schema": {"type": "object"}},
                        {"name": "beta/y", "input_schema": {"type": "object", "properties": {"y": {"type": "string"}}}, "output_schema": {"type": "object"}},
                    ],
                    "errors": [{"name": "ValidationError"}],
                    "requires": ["mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"],
                    "compatibility": {"compatible_with": ["cidv1-sha256-demo"], "supersedes": []},
                },
                {
                    "requires": ["mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"],
                    "name": "beta",
                    "namespace": "test.ns",
                    "methods": [
                        {"output_schema": {"type": "object"}, "name": "beta/x", "input_schema": {"properties": {"x": {"type": "integer"}}, "type": "object"}},
                        {"input_schema": {"properties": {"y": {"type": "string"}}, "type": "object"}, "name": "beta/y", "output_schema": {"type": "object"}},
                    ],
                    "version": "2.1.0",
                    "errors": [{"name": "ValidationError"}],
                    "compatibility": {"supersedes": [], "compatible_with": ["cidv1-sha256-demo"]},
                },
            ),
            (
                {
                    "name": "gamma",
                    "namespace": "test.other",
                    "version": "0.9.3",
                    "methods": [{"name": "gamma/exec", "input_schema": {"type": "object", "properties": {}, "required": []}, "output_schema": {"type": "object"}}],
                    "errors": [],
                    "requires": [],
                    "compatibility": {"compatible_with": [], "supersedes": ["cidv1-sha256-prev"]},
                },
                {
                    "methods": [{"name": "gamma/exec", "output_schema": {"type": "object"}, "input_schema": {"required": [], "properties": {}, "type": "object"}}],
                    "namespace": "test.other",
                    "version": "0.9.3",
                    "name": "gamma",
                    "compatibility": {"supersedes": ["cidv1-sha256-prev"], "compatible_with": []},
                    "errors": [],
                    "requires": [],
                },
            ),
        ]

        cids: list[str] = []
        for descriptor_a, descriptor_b in corpus:
            self.assertEqual(canonicalize_descriptor(descriptor_a), canonicalize_descriptor(descriptor_b))
            cid_a = compute_interface_cid(descriptor_a)
            cid_b = compute_interface_cid(descriptor_b)
            self.assertEqual(cid_a, cid_b)
            cids.append(cid_a)

        self.assertEqual(len(cids), len(set(cids)))

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

    def test_registry_compatibility_normalizes_versioned_capabilities(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["mcp++/profile-a-idl@1.0.0"])
        cid = registry.register_descriptor(
            build_descriptor(
                name="version-tolerant",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "x", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["mcp++/profile-a-idl"],
            )
        )
        verdict = registry.compat(cid)
        self.assertTrue(verdict.compatible)
        self.assertEqual(verdict.requires_missing, [])

    def test_registry_compatibility_normalizes_case_and_whitespace(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["  MCP++/PROFILE-A-IDL  "])
        cid = registry.register_descriptor(
            build_descriptor(
                name="normalized-token",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "x", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["mcp++/profile-a-idl"],
            )
        )
        verdict = registry.compat(cid)
        self.assertTrue(verdict.compatible)
        self.assertEqual(verdict.requires_missing, [])

    def test_registry_compatibility_normalizes_case_whitespace_and_version_variants(self) -> None:
        registry = InterfaceDescriptorRegistry(
            supported_capabilities=[
                " mcp++/profile-a-idl@1.2.3 ",
                "MCP++/PROFILE-E-MCP-P2P@2.0.0",
            ]
        )
        cid = registry.register_descriptor(
            self._canonical_descriptor(
                name="variant-normalization",
                methods=[{"name": "x", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["  MCP++/PROFILE-A-IDL  ", "mcp++/profile-e-mcp-p2p"],
            )
        )
        verdict = registry.compat(cid)
        self.assertTrue(verdict.compatible)
        self.assertEqual(verdict.requires_missing, [])

    def test_registry_register_descriptor_isolated_from_external_nested_mutation(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["mcp++/profile-a-idl"])
        descriptor = build_descriptor(
            name="mutation-safe",
            namespace="test.ns",
            version="1.0.0",
            methods=[
                {
                    "name": "x",
                    "input_schema": {"type": "object", "properties": {"v": {"type": "string"}}},
                    "output_schema": {"type": "object"},
                }
            ],
            requires=["mcp++/profile-a-idl"],
        )
        cid = registry.register_descriptor(descriptor)

        descriptor["methods"][0]["input_schema"]["properties"]["v"]["type"] = "integer"
        descriptor["requires"].append("mcp++/profile-e-mcp-p2p")

        stored = registry.get_descriptor(cid) or {}
        method = ((stored.get("methods") or [{}])[0]) if isinstance(stored.get("methods"), list) else {}
        input_schema = method.get("input_schema") or {}
        properties = input_schema.get("properties") or {}
        self.assertEqual(((properties.get("v") or {}).get("type")), "string")
        self.assertEqual(stored.get("requires"), ["mcp++/profile-a-idl"])

    def test_registry_get_descriptor_returns_deep_copy(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["mcp++/profile-a-idl"])
        cid = registry.register_descriptor(
            build_descriptor(
                name="copy-safe",
                namespace="test.ns",
                version="1.0.0",
                methods=[
                    {
                        "name": "copy",
                        "input_schema": {"type": "object", "properties": {"flag": {"type": "boolean"}}},
                        "output_schema": {"type": "object"},
                    }
                ],
                requires=["mcp++/profile-a-idl"],
            )
        )

        first = registry.get_descriptor(cid) or {}
        first_methods = first.get("methods") or []
        first_methods[0]["input_schema"]["properties"]["flag"]["type"] = "string"
        first["requires"].append("mcp++/profile-e-mcp-p2p")

        second = registry.get_descriptor(cid) or {}
        second_method = ((second.get("methods") or [{}])[0]) if isinstance(second.get("methods"), list) else {}
        second_input_schema = second_method.get("input_schema") or {}
        second_properties = second_input_schema.get("properties") or {}

        self.assertEqual(((second_properties.get("flag") or {}).get("type")), "boolean")
        self.assertEqual(second.get("requires"), ["mcp++/profile-a-idl"])

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

    def test_native_idl_tools_include_loaded_category_descriptors(self) -> None:
        async def _run() -> None:
            manager = HierarchicalToolManager(runtime_router=RuntimeRouter())

            async def ipfs_echo(cid: str) -> dict:
                return {"cid": cid}

            manager.register_tool(
                category="ipfs",
                name="ipfs_echo",
                func=ipfs_echo,
                description="Echo CID",
                input_schema={
                    "type": "object",
                    "properties": {"cid": {"type": "string"}},
                    "required": ["cid"],
                },
                runtime="fastapi",
            )

            register_native_idl_tools(
                manager,
                supported_capabilities=[
                    "mcp++/profile-a-idl",
                    "mcp++/profile-e-mcp-p2p",
                ],
            )

            listed = await manager.dispatch("idl", "interfaces_list", {})
            self.assertGreaterEqual(listed.get("count", 0), 2)

            found_ipfs_descriptor = False
            for interface_cid in listed.get("interface_cids", []):
                payload = await manager.dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
                descriptor = (payload or {}).get("descriptor") or {}
                if descriptor.get("name") == "ipfs_tools":
                    found_ipfs_descriptor = True
                    methods = descriptor.get("methods", [])
                    self.assertTrue(any(m.get("name") == "ipfs/ipfs_echo" for m in methods if isinstance(m, dict)))
                    verdict = await manager.dispatch("idl", "interfaces_compat", {"interface_cid": interface_cid})
                    self.assertTrue(verdict.get("compatible"))
                    break

            self.assertTrue(found_ipfs_descriptor)

        anyio.run(_run)

    def test_native_idl_tools_cover_workflow_and_p2p_migrated_categories(self) -> None:
        async def _run() -> None:
            manager = HierarchicalToolManager(runtime_router=RuntimeRouter())

            async def workflow_ping(workflow_id: str) -> dict:
                return {"workflow_id": workflow_id}

            async def p2p_echo(peer_id: str) -> dict:
                return {"peer_id": peer_id}

            manager.register_tool(
                category="workflow",
                name="workflow_ping",
                func=workflow_ping,
                description="Workflow ping",
                input_schema={
                    "type": "object",
                    "properties": {"workflow_id": {"type": "string"}},
                    "required": ["workflow_id"],
                },
                runtime="fastapi",
            )
            manager.register_tool(
                category="p2p",
                name="p2p_echo",
                func=p2p_echo,
                description="P2P echo",
                input_schema={
                    "type": "object",
                    "properties": {"peer_id": {"type": "string"}},
                    "required": ["peer_id"],
                },
                runtime="fastapi",
            )

            register_native_idl_tools(
                manager,
                supported_capabilities=[
                    "mcp++/profile-a-idl",
                    "mcp++/profile-e-mcp-p2p",
                ],
            )

            listed = await manager.dispatch("idl", "interfaces_list", {})
            found = {"workflow_tools": False, "p2p_tools": False}

            for interface_cid in listed.get("interface_cids", []):
                payload = await manager.dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
                descriptor = (payload or {}).get("descriptor") or {}
                name = descriptor.get("name")
                if name not in found:
                    continue
                found[name] = True

                methods = [m for m in descriptor.get("methods", []) if isinstance(m, dict)]
                method_names = {m.get("name") for m in methods}
                requires = set(descriptor.get("requires", []))
                if name == "workflow_tools":
                    self.assertIn("workflow/workflow_ping", method_names)
                    self.assertEqual(requires, {"mcp++/profile-a-idl"})
                if name == "p2p_tools":
                    self.assertIn("p2p/p2p_echo", method_names)
                    self.assertEqual(requires, {"mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"})

            self.assertEqual(found, {"workflow_tools": True, "p2p_tools": True})

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
