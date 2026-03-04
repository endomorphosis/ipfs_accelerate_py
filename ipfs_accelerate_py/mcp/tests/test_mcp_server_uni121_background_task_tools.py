#!/usr/bin/env python3
"""UNI-121 background-task tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.background_task_tools.native_background_task_tools import (
    check_task_status,
    manage_background_tasks,
    manage_task_queue,
    register_native_background_task_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI121BackgroundTaskTools(unittest.TestCase):
    def test_register_includes_background_task_tools(self) -> None:
        manager = _DummyManager()
        register_native_background_task_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("check_task_status", names)
        self.assertIn("manage_background_tasks", names)
        self.assertIn("manage_task_queue", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_background_task_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        status_schema = by_name["check_task_status"]["input_schema"]
        self.assertEqual(status_schema["properties"]["limit"].get("maximum"), 100)

        queue_schema = by_name["manage_task_queue"]["input_schema"]
        self.assertIn("clear_queue", queue_schema["properties"]["action"].get("enum", []))

    def test_check_task_status_rejects_invalid_task_type(self) -> None:
        async def _run() -> None:
            result = await check_task_status(task_type="etl")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("task_type must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_background_tasks_rejects_missing_task_id_for_cancel(self) -> None:
        async def _run() -> None:
            result = await manage_background_tasks(action="cancel")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("task_id is required for cancel action", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_task_queue_rejects_bad_max_concurrent(self) -> None:
        async def _run() -> None:
            result = await manage_task_queue(action="set_limits", max_concurrent=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_concurrent must be a positive integer", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_task_queue_rejects_missing_priority_for_clear(self) -> None:
        async def _run() -> None:
            result = await manage_task_queue(action="clear_queue")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("priority is required for clear_queue action", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_background_tasks_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await manage_background_tasks(action="get_stats")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("action"), "get_stats")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
