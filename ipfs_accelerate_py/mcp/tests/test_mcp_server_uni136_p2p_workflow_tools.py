#!/usr/bin/env python3
"""UNI-136 p2p_workflow_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools.native_p2p_workflow_tools import (
    get_assigned_workflows,
    get_next_p2p_workflow,
    get_p2p_scheduler_status,
    initialize_p2p_scheduler,
    register_native_p2p_workflow_tools,
    schedule_p2p_workflow,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI136P2PWorkflowTools(unittest.TestCase):
    def test_register_includes_p2p_workflow_tools(self) -> None:
        manager = _DummyManager()
        register_native_p2p_workflow_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("initialize_p2p_scheduler", names)
        self.assertIn("schedule_p2p_workflow", names)
        self.assertIn("get_next_p2p_workflow", names)
        self.assertIn("get_p2p_scheduler_status", names)
        self.assertIn("get_assigned_workflows", names)

    def test_initialize_p2p_scheduler_rejects_bad_peers_shape(self) -> None:
        async def _run() -> None:
            result = await initialize_p2p_scheduler(peers=["peer-1", ""])  # type: ignore[list-item]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty strings", str(result.get("message", "")))

        anyio.run(_run)

    def test_schedule_p2p_workflow_rejects_empty_tags(self) -> None:
        async def _run() -> None:
            result = await schedule_p2p_workflow(workflow_id="wf-1", name="workflow", tags=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty array", str(result.get("message", "")))

        anyio.run(_run)

    def test_schedule_p2p_workflow_rejects_invalid_priority(self) -> None:
        async def _run() -> None:
            result = await schedule_p2p_workflow(workflow_id="wf-1", name="workflow", tags=["p2p_eligible"], priority=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("positive number", str(result.get("message", "")))

        anyio.run(_run)

    def test_schedule_p2p_workflow_rejects_invalid_metadata_shape(self) -> None:
        async def _run() -> None:
            result = await schedule_p2p_workflow(
                workflow_id="wf-1",
                name="workflow",
                tags=["p2p_eligible"],
                metadata=["bad"],  # type: ignore[arg-type]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_get_status_wrappers_return_normalized_status(self) -> None:
        async def _run() -> None:
            for call in (get_next_p2p_workflow, get_p2p_scheduler_status, get_assigned_workflows):
                result = await call()
                self.assertIn(result.get("status"), ["success", "error"])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
