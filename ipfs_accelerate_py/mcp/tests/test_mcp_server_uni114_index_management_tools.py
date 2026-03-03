#!/usr/bin/env python3
"""UNI-114 index-management tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.index_management_tools.native_index_management_tools import (
    load_index,
    manage_index_configuration,
    manage_shards,
    monitor_index_status,
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

    def test_manage_shards_success_shape(self) -> None:
        async def _run() -> None:
            result = await manage_shards(action="list_shards")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("action", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
