#!/usr/bin/env python3
"""UNI-114 index-management tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.index_management_tools import native_index_management_tools
from ipfs_accelerate_py.mcp_server.tools.index_management_tools.native_index_management_tools import (
    load_index,
    manage_index_configuration,
    manage_shards,
    monitor_index_status,
    orchestrate_index_lifecycle,
    register_native_index_management_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI114IndexManagementTools(unittest.TestCase):
    def test_register_includes_index_management_tools(self) -> None:
        manager = _DummyManager()
        register_native_index_management_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("load_index", names)
        self.assertIn("manage_shards", names)
        self.assertIn("monitor_index_status", names)
        self.assertIn("manage_index_configuration", names)
        self.assertIn("orchestrate_index_lifecycle", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_index_management_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        monitor_schema = by_name["monitor_index_status"]["input_schema"]
        time_range = monitor_schema["properties"]["time_range"]
        self.assertEqual(time_range.get("default"), "24h")
        self.assertIn("7d", time_range.get("enum", []))

        config_schema = by_name["manage_index_configuration"]["input_schema"]
        optimization = config_schema["properties"]["optimization_level"]
        self.assertEqual(optimization.get("default"), 1)
        self.assertEqual(optimization.get("minimum"), 1)
        self.assertEqual(optimization.get("maximum"), 3)

        lifecycle_schema = by_name["orchestrate_index_lifecycle"]["input_schema"]
        self.assertEqual((lifecycle_schema["properties"]["dataset"]).get("minLength"), 1)
        self.assertIn("optimize", (lifecycle_schema["properties"]["action"]).get("enum", []))

    def test_load_index_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await load_index(action="bad_action")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_shards_rejects_non_positive_num_shards(self) -> None:
        async def _run() -> None:
            result = await manage_shards(action="create_shards", num_shards=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("positive integer", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_shards_rejects_non_integer_num_shards(self) -> None:
        async def _run() -> None:
            result = await manage_shards(action="create_shards", num_shards="bad")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("positive integer", str(result.get("message", "")))

        anyio.run(_run)

    def test_monitor_index_status_rejects_invalid_time_range(self) -> None:
        async def _run() -> None:
            result = await monitor_index_status(time_range="2h")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_index_configuration_rejects_invalid_optimization_level(self) -> None:
        async def _run() -> None:
            result = await manage_index_configuration(action="update_config", optimization_level=5)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("between 1 and 3", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_index_configuration_rejects_non_integer_optimization_level(self) -> None:
        async def _run() -> None:
            result = await manage_index_configuration(action="update_config", optimization_level="high")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("between 1 and 3", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_shards_success_shape(self) -> None:
        async def _run() -> None:
            result = await manage_shards(action="list_shards")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("action", result)

        anyio.run(_run)

    def test_orchestrate_index_lifecycle_rejects_missing_dataset(self) -> None:
        async def _run() -> None:
            result = await orchestrate_index_lifecycle(dataset="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("dataset is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_orchestrate_index_lifecycle_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await orchestrate_index_lifecycle(dataset="ds", action="delete")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("action must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_orchestrate_index_lifecycle_success_shape(self) -> None:
        async def _run() -> None:
            result = await orchestrate_index_lifecycle(dataset="demo-dataset", action="create")
            self.assertEqual(result.get("status"), "success")
            self.assertIn("load", result)
            self.assertIn("shards", result)
            self.assertIn("configuration", result)

        anyio.run(_run)

    def test_failed_delegate_payloads_infer_error_status(self) -> None:
        async def _failed(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_index_management_tools._API,
                {
                    "load_index": _failed,
                    "manage_shards": _failed,
                    "monitor_index_status": _failed,
                    "manage_index_configuration": _failed,
                },
                clear=False,
            ):
                loaded = await load_index(action="load")
                self.assertEqual(loaded.get("status"), "error")

                shards = await manage_shards(action="list_shards")
                self.assertEqual(shards.get("status"), "error")

                monitored = await monitor_index_status(time_range="24h")
                self.assertEqual(monitored.get("status"), "error")

                configured = await manage_index_configuration(action="get_config")
                self.assertEqual(configured.get("status"), "error")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
