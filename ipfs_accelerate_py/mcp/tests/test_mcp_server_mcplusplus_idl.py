#!/usr/bin/env python3
"""Deterministic tests for MCP-IDL registry primitives and native tools."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

    def test_descriptor_canonicalization_is_stable_across_runtime_environments(self) -> None:
        descriptor = {
            "name": "runtime-stable",
            "namespace": "test.ns",
            "version": "3.2.1",
            "methods": [
                {
                    "name": "runtime/stable",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "alpha": {"type": "string"},
                            "beta": {"type": "integer"},
                        },
                        "required": ["alpha"],
                    },
                    "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                }
            ],
            "errors": [{"name": "ValidationError"}],
            "requires": ["mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"],
            "compatibility": {"compatible_with": ["cidv1-sha256-prev"], "supersedes": []},
        }

        expected = {
            "canonical": canonicalize_descriptor(descriptor).decode("utf-8"),
            "cid": compute_interface_cid(descriptor),
        }

        code = (
            "import json, os;"
            "from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import canonicalize_descriptor, compute_interface_cid;"
            "descriptor=json.loads(os.environ['IDL_DESCRIPTOR_JSON']);"
            "print(json.dumps({'canonical': canonicalize_descriptor(descriptor).decode('utf-8'), 'cid': compute_interface_cid(descriptor)}, sort_keys=True))"
        )

        for hash_seed in ("1", "777"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hash_seed
            env["IDL_DESCRIPTOR_JSON"] = json.dumps(descriptor)

            result = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )

            observed = json.loads(result.stdout.strip())
            self.assertEqual(observed, expected)

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

    def test_registry_compatibility_corpus_is_deterministic(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["mcp++/profile-a-idl"])
        compatible_later = registry.register_descriptor(
            build_descriptor(
                name="compatible-later",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "later.call", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=[],
            )
        )
        incompatible = registry.register_descriptor(
            build_descriptor(
                name="incompatible",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "blocked.call", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["mcp++/profile-q-unavailable"],
            )
        )
        compatible_earlier = registry.register_descriptor(
            build_descriptor(
                name="compatible-earlier",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "earlier.call", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["mcp++/profile-a-idl"],
            )
        )
        target = registry.register_descriptor(
            build_descriptor(
                name="target",
                namespace="test.ns",
                version="1.0.0",
                methods=[{"name": "target.call", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}}],
                requires=["mcp++/profile-z-missing", "mcp++/profile-b-missing"],
            )
        )

        verdict = registry.compat(target)

        self.assertFalse(verdict.compatible)
        self.assertEqual(verdict.reasons, ["missing_required_capabilities"])
        self.assertEqual(verdict.requires_missing, ["mcp++/profile-b-missing", "mcp++/profile-z-missing"])
        self.assertEqual(verdict.suggested_alternatives, sorted([compatible_earlier, compatible_later]))
        self.assertNotIn(incompatible, verdict.suggested_alternatives)

    def test_registry_compatibility_returns_not_found_for_unknown_interface(self) -> None:
        registry = InterfaceDescriptorRegistry(supported_capabilities=["mcp++/profile-a-idl"])

        verdict = registry.compat("cidv1-sha256-does-not-exist")

        self.assertFalse(verdict.compatible)
        self.assertEqual(verdict.reasons, ["interface_not_found"])
        self.assertEqual(verdict.requires_missing, [])
        self.assertEqual(verdict.suggested_alternatives, [])

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

    def test_native_idl_tools_cover_all_loaded_migrated_categories_consistently(self) -> None:
        async def _run() -> None:
            manager = HierarchicalToolManager(runtime_router=RuntimeRouter())

            async def ipfs_echo(cid: str) -> dict:
                return {"cid": cid}

            async def workflow_ping(workflow_id: str) -> dict:
                return {"workflow_id": workflow_id}

            async def p2p_echo(peer_id: str) -> dict:
                return {"peer_id": peer_id}

            async def dataset_describe(dataset_name: str) -> dict:
                return {"dataset_name": dataset_name}

            registrations = [
                (
                    "ipfs",
                    "ipfs_echo",
                    ipfs_echo,
                    {"cid": {"type": "string"}},
                    ["cid"],
                    {"mcp++/profile-a-idl"},
                    "ipfs/ipfs_echo",
                    "ipfs_tools",
                ),
                (
                    "workflow",
                    "workflow_ping",
                    workflow_ping,
                    {"workflow_id": {"type": "string"}},
                    ["workflow_id"],
                    {"mcp++/profile-a-idl"},
                    "workflow/workflow_ping",
                    "workflow_tools",
                ),
                (
                    "p2p",
                    "p2p_echo",
                    p2p_echo,
                    {"peer_id": {"type": "string"}},
                    ["peer_id"],
                    {"mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"},
                    "p2p/p2p_echo",
                    "p2p_tools",
                ),
                (
                    "dataset",
                    "dataset_describe",
                    dataset_describe,
                    {"dataset_name": {"type": "string"}},
                    ["dataset_name"],
                    {"mcp++/profile-a-idl"},
                    "dataset/dataset_describe",
                    "dataset_tools",
                ),
            ]

            for category, name, func, properties, required, _requires, _method_name, _descriptor_name in registrations:
                manager.register_tool(
                    category=category,
                    name=name,
                    func=func,
                    description=f"{category} test tool",
                    input_schema={
                        "type": "object",
                        "properties": properties,
                        "required": required,
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
            self.assertEqual(listed.get("count"), 1 + len(registrations))

            descriptors_by_name = {}
            for interface_cid in listed.get("interface_cids", []):
                payload = await manager.dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
                descriptor = (payload or {}).get("descriptor") or {}
                if descriptor.get("name"):
                    descriptors_by_name[descriptor["name"]] = descriptor

            self.assertIn("interfaces", descriptors_by_name)
            for _category, _name, _func, _properties, _required, expected_requires, expected_method_name, descriptor_name in registrations:
                descriptor = descriptors_by_name.get(descriptor_name) or {}
                self.assertTrue(descriptor, msg=f"missing descriptor for {descriptor_name}")
                methods = [m for m in descriptor.get("methods", []) if isinstance(m, dict)]
                method_names = {m.get("name") for m in methods}
                self.assertEqual(method_names, {expected_method_name})
                self.assertEqual(set(descriptor.get("requires", [])), expected_requires)

                verdict = await manager.dispatch("idl", "interfaces_compat", {"interface_cid": descriptor.get("interface_cid")})
                self.assertTrue(verdict.get("compatible"), msg=f"unexpected incompatibility for {descriptor_name}: {verdict}")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
