#!/usr/bin/env python3
"""UNI-121 background-task tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.background_task_tools import native_background_task_tools
from ipfs_accelerate_py.mcp_server.tools.background_task_tools.native_background_task_tools import (
    check_task_status,
    get_task_status,
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
        self.assertIn("get_task_status", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_background_task_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        status_schema = by_name["check_task_status"]["input_schema"]
        self.assertEqual(status_schema["properties"]["limit"].get("maximum"), 100)

        queue_schema = by_name["manage_task_queue"]["input_schema"]
        self.assertIn("clear_queue", queue_schema["properties"]["action"].get("enum", []))

        detailed_schema = by_name["get_task_status"]["input_schema"]
        self.assertEqual(detailed_schema["properties"]["log_limit"].get("maximum"), 500)
        self.assertEqual(detailed_schema["properties"]["include_logs"].get("default"), True)

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

    def test_get_task_status_rejects_invalid_log_limit(self) -> None:
        async def _run() -> None:
            result = await get_task_status(log_limit=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("between 1 and 500", str(result.get("message", "")))

        anyio.run(_run)

    def test_get_task_status_rejects_non_boolean_include_logs(self) -> None:
        async def _run() -> None:
            result = await get_task_status(include_logs="yes")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be a boolean", str(result.get("message", "")))

        anyio.run(_run)

    def test_check_task_status_success_defaults_with_minimal_payload(self) -> None:
        async def _minimal_status(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.background_task_tools.native_background_task_tools._API",
                {
                    "check_task_status": _minimal_status,
                    "manage_background_tasks": None,
                    "manage_task_queue": None,
                },
            ):
                result = await check_task_status(task_id="task-1", task_type="all", status_filter="all", limit=10)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("task_id"), "task-1")
            self.assertEqual(result.get("task_type"), "all")
            self.assertEqual(result.get("status_filter"), "all")
            self.assertEqual(result.get("limit"), 10)
            self.assertEqual(result.get("tasks"), [])
            self.assertEqual(result.get("count"), 0)

        anyio.run(_run)

    def test_manage_background_and_queue_success_defaults_with_minimal_payloads(self) -> None:
        async def _minimal_manage_background(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_manage_queue(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.background_task_tools.native_background_task_tools._API",
                {
                    "check_task_status": None,
                    "manage_background_tasks": _minimal_manage_background,
                    "manage_task_queue": _minimal_manage_queue,
                },
            ):
                list_result = await manage_background_tasks(action="list", task_type="create_embeddings", priority="high")
                stats_result = await manage_background_tasks(action="get_stats")
                queue_result = await manage_task_queue(action="get_stats")
                limits_result = await manage_task_queue(action="set_limits", max_concurrent=3)

            self.assertEqual(list_result.get("status"), "success")
            self.assertEqual(list_result.get("action"), "list")
            self.assertEqual(list_result.get("priority"), "high")
            self.assertEqual(list_result.get("task_type"), "create_embeddings")
            self.assertEqual(list_result.get("tasks"), [])
            self.assertEqual(list_result.get("count"), 0)

            self.assertEqual(stats_result.get("status"), "success")
            self.assertEqual(stats_result.get("statistics"), {})

            self.assertEqual(queue_result.get("status"), "success")
            self.assertEqual(queue_result.get("action"), "get_stats")
            self.assertEqual(queue_result.get("queue_statistics"), {})

            self.assertEqual(limits_result.get("status"), "success")
            self.assertEqual(limits_result.get("action"), "set_limits")
            self.assertEqual(limits_result.get("max_concurrent"), 3)

        anyio.run(_run)

    def test_get_task_status_success_defaults_with_minimal_payload(self) -> None:
        async def _minimal_get_status(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.background_task_tools.native_background_task_tools._API",
                {
                    "check_task_status": None,
                    "manage_background_tasks": None,
                    "manage_task_queue": None,
                    "get_task_status": _minimal_get_status,
                },
            ):
                summary_result = await get_task_status(log_limit=10)
                detailed_result = await get_task_status(
                    task_id="task-1",
                    include_logs=False,
                    include_system_status=True,
                    include_queue_status=True,
                    log_limit=5,
                )

            self.assertEqual(summary_result.get("status"), "success")
            self.assertEqual(summary_result.get("summary"), {"total_tasks": 0, "task_ids": []})

            self.assertEqual(detailed_result.get("status"), "success")
            self.assertEqual(detailed_result.get("task_id"), "task-1")
            self.assertEqual(detailed_result.get("include_logs"), False)
            self.assertEqual(detailed_result.get("system_status"), {})
            self.assertEqual(detailed_result.get("queue_status"), {})

        anyio.run(_run)

    def test_failed_delegate_payloads_infer_error_status(self) -> None:
        async def _failed(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_background_task_tools._API,
                {
                    "check_task_status": _failed,
                    "manage_background_tasks": _failed,
                    "manage_task_queue": _failed,
                    "get_task_status": _failed,
                },
                clear=False,
            ):
                self.assertEqual((await check_task_status()).get("status"), "error")
                self.assertEqual((await manage_background_tasks(action="list")).get("status"), "error")
                self.assertEqual((await manage_task_queue(action="get_stats")).get("status"), "error")
                self.assertEqual((await get_task_status()).get("status"), "error")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
