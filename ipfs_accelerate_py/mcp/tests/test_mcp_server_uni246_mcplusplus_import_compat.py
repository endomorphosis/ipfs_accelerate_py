#!/usr/bin/env python3
"""UNI-246 mcplusplus package import compatibility tests."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.mcplusplus import (
    PeerEngine,
    TaskQueueEngine,
    WorkflowEngine,
)
from ipfs_accelerate_py.mcp_server.tools.mcplusplus import compat_engines


class TestMCPServerUNI246McplusplusImportCompat(unittest.TestCase):
    def test_package_exports_compat_engine_classes(self) -> None:
        self.assertIs(TaskQueueEngine, compat_engines.TaskQueueEngine)
        self.assertIs(PeerEngine, compat_engines.PeerEngine)
        self.assertIs(WorkflowEngine, compat_engines.WorkflowEngine)

    def test_taskqueue_engine_delegates_to_native_wrapper(self) -> None:
        async def _run() -> None:
            expected = {"status": "success", "task_id": "task-1"}
            with patch.object(
                compat_engines.native_tools,
                "mcplusplus_taskqueue_get_status",
                new=AsyncMock(return_value=expected),
            ) as mock_get_status:
                result = await TaskQueueEngine().get_status(
                    "task-1",
                    include_logs=True,
                    include_metrics=True,
                )

            self.assertIs(result, expected)
            mock_get_status.assert_awaited_once_with(
                task_id="task-1",
                include_logs=True,
                include_metrics=True,
            )

        anyio.run(_run)

    def test_workflow_engine_delegates_to_native_wrapper(self) -> None:
        async def _run() -> None:
            expected = {"status": "success", "workflow_id": "wf-1"}
            with patch.object(
                compat_engines.native_tools,
                "mcplusplus_workflow_get_status",
                new=AsyncMock(return_value=expected),
            ) as mock_get_status:
                result = await WorkflowEngine().get_status(
                    "wf-1",
                    include_steps=False,
                    include_metrics=True,
                )

            self.assertIs(result, expected)
            mock_get_status.assert_awaited_once_with(
                workflow_id="wf-1",
                include_steps=False,
                include_metrics=True,
            )

        anyio.run(_run)

    def test_peer_engine_delegates_to_native_wrapper(self) -> None:
        async def _run() -> None:
            expected = {"status": "success", "peer_id": "peer-1"}
            with patch.object(
                compat_engines.native_tools,
                "mcplusplus_peer_connect",
                new=AsyncMock(return_value=expected),
            ) as mock_connect:
                result = await PeerEngine().connect(
                    "peer-1",
                    "/ip4/127.0.0.1/tcp/4001",
                    timeout=10,
                    retry_count=2,
                    persist=False,
                )

            self.assertIs(result, expected)
            mock_connect.assert_awaited_once_with(
                peer_id="peer-1",
                multiaddr="/ip4/127.0.0.1/tcp/4001",
                timeout=10,
                retry_count=2,
                persist=False,
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()