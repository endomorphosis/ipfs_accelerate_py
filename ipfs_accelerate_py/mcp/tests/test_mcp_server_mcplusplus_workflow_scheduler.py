#!/usr/bin/env python3
"""Unit tests for unified MCP++ workflow scheduler wrapper."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus import workflow_scheduler as ws


class _Context:
    workflow_scheduler: object | None = None


class TestWorkflowSchedulerWrapper(unittest.TestCase):
    """Validate wrapper behavior for scheduler availability and submission."""

    def test_unavailable_returns_none(self) -> None:
        with patch.object(ws, "HAVE_WORKFLOW_SCHEDULER", False):
            self.assertIsNone(ws.create_workflow_scheduler())
            self.assertIsNone(ws.get_scheduler())
            ws.reset_scheduler()

    def test_create_with_context_assigns_scheduler(self) -> None:
        ctx = _Context()
        scheduler = object()
        with (
            patch.object(ws, "HAVE_WORKFLOW_SCHEDULER", True),
            patch.object(ws, "_get_scheduler", MagicMock(return_value=scheduler)),
        ):
            created = ws.create_workflow_scheduler(context=ctx)
            self.assertIs(created, scheduler)
            self.assertIs(ctx.workflow_scheduler, scheduler)

    def test_get_scheduler_prefers_context(self) -> None:
        ctx = _Context()
        ctx.workflow_scheduler = "ctx-scheduler"
        self.assertEqual(ws.get_scheduler(context=ctx), "ctx-scheduler")

    def test_reset_invokes_underlying_reset(self) -> None:
        reset_mock = MagicMock()
        with (
            patch.object(ws, "HAVE_WORKFLOW_SCHEDULER", True),
            patch.object(ws, "_reset_scheduler", reset_mock),
        ):
            ws.reset_scheduler()
            reset_mock.assert_called_once_with()

    def test_submit_workflow_supports_sync_result(self) -> None:
        async def _run() -> None:
            scheduler = MagicMock()
            scheduler.submit_workflow.return_value = {"workflow_id": "wf-1"}
            with patch.object(ws, "get_scheduler", return_value=scheduler):
                workflow_id = await ws.submit_workflow("demo", [{"task": "one"}])
                self.assertEqual(workflow_id, "wf-1")

        anyio.run(_run)

    def test_submit_workflow_supports_async_result(self) -> None:
        async def _run() -> None:
            scheduler = MagicMock()
            scheduler.submit_workflow = AsyncMock(return_value="wf-async")
            with patch.object(ws, "get_scheduler", return_value=scheduler):
                workflow_id = await ws.submit_workflow("demo", [{"task": "one"}])
                self.assertEqual(workflow_id, "wf-async")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
