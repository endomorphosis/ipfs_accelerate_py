#!/usr/bin/env python3
"""Unit tests for unified MCP++ workflow engine primitive."""

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus.workflow_engine import (
    Task,
    TaskStatus,
    WorkflowEngine,
    WorkflowStatus,
)


class TestWorkflowEnginePrimitive(unittest.TestCase):
    """Validate workflow engine task execution and status transitions."""

    def test_execute_workflow_success(self) -> None:
        async def _run() -> None:
            engine = WorkflowEngine(max_concurrent_tasks=2)
            workflow = engine.create_workflow("wf-1", "demo")

            async def root_task() -> str:
                await anyio.sleep(0)
                return "ok-root"

            async def child_task() -> str:
                await anyio.sleep(0)
                return "ok-child"

            workflow.add_task(Task(task_id="t1", name="root", function=root_task))
            workflow.add_task(Task(task_id="t2", name="child", function=child_task, dependencies=["t1"]))

            result = await engine.execute_workflow("wf-1")
            self.assertEqual(result["status"], WorkflowStatus.COMPLETED.value)
            self.assertEqual(result["completed_tasks"], 2)
            self.assertEqual(result["failed_tasks"], 0)
            self.assertEqual(workflow.tasks["t1"].status, TaskStatus.COMPLETED)
            self.assertEqual(workflow.tasks["t2"].status, TaskStatus.COMPLETED)

        anyio.run(_run)

    def test_execute_workflow_with_unregistered_function_fails(self) -> None:
        async def _run() -> None:
            engine = WorkflowEngine(max_concurrent_tasks=1)
            workflow = engine.create_workflow("wf-2", "demo-fail")
            workflow.add_task(Task(task_id="t1", name="missing", function="not_registered"))

            result = await engine.execute_workflow("wf-2")
            self.assertEqual(result["status"], WorkflowStatus.FAILED.value)
            self.assertEqual(result["completed_tasks"], 0)
            self.assertEqual(result["failed_tasks"], 1)
            self.assertEqual(workflow.tasks["t1"].status, TaskStatus.FAILED)

        anyio.run(_run)

    def test_execute_workflow_retry_succeeds(self) -> None:
        async def _run() -> None:
            engine = WorkflowEngine(max_concurrent_tasks=1)
            workflow = engine.create_workflow("wf-3", "retry-demo")

            state = {"attempts": 0}

            async def flaky_task() -> str:
                state["attempts"] += 1
                if state["attempts"] == 1:
                    raise RuntimeError("first failure")
                return "ok-after-retry"

            workflow.add_task(
                Task(task_id="t1", name="flaky", function=flaky_task, max_retries=1)
            )

            result = await engine.execute_workflow("wf-3")
            self.assertEqual(result["status"], WorkflowStatus.COMPLETED.value)
            self.assertEqual(workflow.tasks["t1"].status, TaskStatus.COMPLETED)
            self.assertEqual(workflow.tasks["t1"].retry_count, 1)
            self.assertEqual(state["attempts"], 2)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
