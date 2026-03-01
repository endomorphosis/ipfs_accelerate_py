#!/usr/bin/env python3
"""Unit tests for unified MCP++ task queue wrapper."""

import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus import task_queue as tq


class TestTaskQueueWrapper(unittest.TestCase):
    """Validate task queue wrapper behavior and compatibility helpers."""

    def test_create_task_queue_assigns_connection_hints(self) -> None:
        queue = tq.create_task_queue(queue_path="/tmp/q.db", peer_id="peer-a", multiaddr="/ip4/127.0.0.1/tcp/9000")
        self.assertEqual(queue.queue_path, "/tmp/q.db")
        self.assertEqual(queue.peer_id, "peer-a")
        self.assertEqual(queue.multiaddr, "/ip4/127.0.0.1/tcp/9000")

    def test_unavailable_wrapper_fails_softly(self) -> None:
        async def _run() -> None:
            with patch.object(tq, "HAVE_TASK_QUEUE", False):
                queue = tq.TaskQueueWrapper()
                self.assertIsNone(await queue.submit("demo", {}))
                self.assertIsNone(await queue.get_status("task-1"))
                self.assertFalse(await queue.cancel("task-1"))
                self.assertEqual(await queue.list(), [])

        anyio.run(_run)

    def test_submit_uses_client_submit_task_with_info(self) -> None:
        async def _run() -> None:
            mock_submit = AsyncMock(return_value={"task_id": "task-123", "peer_id": "peer-z"})
            with (
                patch.object(tq, "_build_remote", return_value="REMOTE") as mock_remote,
                patch.object(tq, "_client_submit_task_with_info", mock_submit),
            ):
                queue = tq.TaskQueueWrapper(peer_id="peer-z", multiaddr="/ip4/1.2.3.4/tcp/9999")
                task_id = await queue.submit("inference", {"prompt": "hello"}, priority=7, extra="yes")

                self.assertEqual(task_id, "task-123")
                mock_remote.assert_called_once_with(peer_id="peer-z", multiaddr="/ip4/1.2.3.4/tcp/9999")
                self.assertEqual(mock_submit.await_count, 1)
                called = mock_submit.await_args.kwargs
                self.assertEqual(called["remote"], "REMOTE")
                self.assertEqual(called["task_type"], "inference")
                self.assertEqual(called["model_name"], "default")
                self.assertEqual(called["payload"]["prompt"], "hello")
                self.assertEqual(called["payload"]["priority"], 7)
                self.assertEqual(called["payload"]["extra"], "yes")

        anyio.run(_run)

    def test_get_status_cancel_and_list_delegate_to_client(self) -> None:
        async def _run() -> None:
            mock_get = AsyncMock(return_value={"task_id": "task-9", "status": "pending"})
            mock_cancel = AsyncMock(return_value={"ok": True})
            mock_list = AsyncMock(return_value={"ok": True, "tasks": [{"task_id": "task-9"}]})

            with (
                patch.object(tq, "_build_remote", return_value="REMOTE"),
                patch.object(tq, "_client_get_task", mock_get),
                patch.object(tq, "_client_cancel_task", mock_cancel),
                patch.object(tq, "_client_list_tasks", mock_list),
            ):
                queue = tq.TaskQueueWrapper(peer_id="peer-x")

                status = await queue.get_status("task-9")
                self.assertEqual(status, {"task_id": "task-9", "status": "pending"})
                self.assertEqual(mock_get.await_args.kwargs["task_id"], "task-9")

                self.assertTrue(await queue.cancel("task-9", reason="no-longer-needed"))
                self.assertEqual(mock_cancel.await_args.kwargs["task_id"], "task-9")
                self.assertEqual(mock_cancel.await_args.kwargs["reason"], "no-longer-needed")

                tasks = await queue.list(status="pending", limit=5, task_types=["inference", "embedding"])
                self.assertEqual(tasks, [{"task_id": "task-9"}])
                self.assertEqual(mock_list.await_args.kwargs["status"], "pending")
                self.assertEqual(mock_list.await_args.kwargs["limit"], 5)
                self.assertEqual(mock_list.await_args.kwargs["task_types"], ["inference", "embedding"])

        anyio.run(_run)

    def test_module_level_helpers_delegate_via_wrapper(self) -> None:
        async def _run() -> None:
            queue = tq.TaskQueueWrapper()
            with (
                patch.object(tq, "create_task_queue", return_value=queue),
                patch.object(queue, "submit", AsyncMock(return_value="task-77")) as mock_submit,
                patch.object(queue, "get_status", AsyncMock(return_value={"status": "done"})) as mock_get,
                patch.object(queue, "cancel", AsyncMock(return_value=True)) as mock_cancel,
                patch.object(queue, "list", AsyncMock(return_value=[{"task_id": "task-77"}])) as mock_list,
            ):
                task_id = await tq.submit_task(task_type="embedding", payload={"text": "hi"})
                status = await tq.get_task_status(task_id="task-77")
                cancelled = await tq.cancel_task(task_id="task-77")
                listed = await tq.list_tasks(limit=10)

                self.assertEqual(task_id, "task-77")
                self.assertEqual(status, {"status": "done"})
                self.assertTrue(cancelled)
                self.assertEqual(listed, [{"task_id": "task-77"}])
                self.assertEqual(mock_submit.await_count, 1)
                self.assertEqual(mock_get.await_count, 1)
                self.assertEqual(mock_cancel.await_count, 1)
                self.assertEqual(mock_list.await_count, 1)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
