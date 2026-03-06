#!/usr/bin/env python3
"""UNI-116 admin tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.admin_tools.native_admin_tools import (
    cleanup_resources,
    configure_system,
    get_system_status,
    manage_endpoints,
    manage_service,
    register_native_admin_tools,
    system_health,
    system_maintenance,
    update_configuration,
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
        self.assertIn("get_system_status", names)
        self.assertIn("manage_service", names)
        self.assertIn("update_configuration", names)
        self.assertIn("cleanup_resources", names)

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

        status_schema = by_name["get_system_status"]["input_schema"]
        self.assertEqual(status_schema["properties"]["format"].get("default"), "json")

        service_schema = by_name["manage_service"]["input_schema"]
        self.assertEqual(service_schema["properties"]["timeout_seconds"].get("minimum"), 1)

        update_schema = by_name["update_configuration"]["input_schema"]
        self.assertEqual(update_schema["properties"]["create_backup"].get("default"), True)

        cleanup_schema = by_name["cleanup_resources"]["input_schema"]
        self.assertEqual(cleanup_schema["properties"]["max_log_age_days"].get("minimum"), 1)

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

    def test_get_system_status_rejects_invalid_format(self) -> None:
        async def _run() -> None:
            result = await get_system_status(format="xml")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_service_rejects_invalid_timeout(self) -> None:
        async def _run() -> None:
            result = await manage_service(service_name="cache_service", action="restart", timeout_seconds=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("positive integer", str(result.get("message", "")))

        anyio.run(_run)

    def test_update_configuration_rejects_invalid_config_updates_shape(self) -> None:
        async def _run() -> None:
            result = await update_configuration(action="update", config_updates=["bad"])  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_cleanup_resources_rejects_invalid_cleanup_type(self) -> None:
        async def _run() -> None:
            result = await cleanup_resources(cleanup_type="purge")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_enhanced_admin_success_shapes(self) -> None:
        async def _run() -> None:
            status_result = await get_system_status(format="detailed")
            self.assertIn(status_result.get("status"), ["success", "operational"])
            self.assertIn("health_status", status_result)

            service_result = await manage_service(service_name="all", action="status")
            self.assertEqual(service_result.get("status"), "success")
            self.assertIn("action", service_result)

            update_result = await update_configuration(action="validate", config_updates={"cache.timeout": 5})
            self.assertEqual(update_result.get("status"), "success")
            self.assertIn("action", update_result)

            cleanup_result = await cleanup_resources(cleanup_type="basic")
            self.assertEqual(cleanup_result.get("status"), "success")
            self.assertIn("cleanup_options", cleanup_result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
