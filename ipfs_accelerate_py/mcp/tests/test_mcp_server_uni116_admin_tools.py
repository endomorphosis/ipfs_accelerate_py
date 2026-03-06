#!/usr/bin/env python3
"""UNI-116 admin tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.admin_tools.native_admin_tools import (
    configure_system,
    manage_endpoints,
    register_native_admin_tools,
    system_health,
    system_maintenance,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI116AdminTools(unittest.TestCase):
    def test_register_includes_admin_tools(self) -> None:
        manager = _DummyManager()
        register_native_admin_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("manage_endpoints", names)
        self.assertIn("system_maintenance", names)
        self.assertIn("configure_system", names)
        self.assertIn("system_health", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_admin_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        endpoint_schema = by_name["manage_endpoints"]["input_schema"]
        self.assertIn("list", endpoint_schema["properties"]["action"].get("enum", []))
        self.assertEqual(endpoint_schema["properties"]["ctx_length"].get("minimum"), 1)

        config_schema = by_name["configure_system"]["input_schema"]
        self.assertEqual(config_schema["properties"]["validate_only"].get("default"), False)

        health_schema = by_name["system_health"]["input_schema"]
        self.assertEqual(health_schema["properties"]["component"].get("default"), "all")
        self.assertEqual(health_schema["properties"]["detailed"].get("default"), False)

    def test_manage_endpoints_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await manage_endpoints(action="bad")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_endpoints_rejects_missing_add_fields(self) -> None:
        async def _run() -> None:
            result = await manage_endpoints(action="add", model="m")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("required for add action", str(result.get("message", "")))

        anyio.run(_run)

    def test_system_maintenance_rejects_invalid_operation(self) -> None:
        async def _run() -> None:
            result = await system_maintenance(operation="reindex")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_system_maintenance_allows_status_alias(self) -> None:
        async def _run() -> None:
            result = await system_maintenance(operation="status")
            self.assertIn(result.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_configure_system_rejects_bad_settings_shape(self) -> None:
        async def _run() -> None:
            result = await configure_system(action="update", settings=["bad"])  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_system_health_rejects_empty_component(self) -> None:
        async def _run() -> None:
            result = await system_health(component="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty string", str(result.get("message", "")))

        anyio.run(_run)

    def test_system_health_rejects_non_boolean_detailed(self) -> None:
        async def _run() -> None:
            result = await system_health(component="all", detailed="yes")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be a boolean", str(result.get("message", "")))

        anyio.run(_run)

    def test_system_health_success_shape(self) -> None:
        async def _run() -> None:
            result = await system_health(component="all", detailed=True)
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("component", result)
                self.assertIn("health", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
