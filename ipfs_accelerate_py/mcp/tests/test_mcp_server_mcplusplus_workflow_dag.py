#!/usr/bin/env python3
"""Unit tests for unified MCP++ workflow DAG primitives."""

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus.workflow_dag import (
    StepStatus,
    WorkflowDAG,
    WorkflowDAGExecutor,
    WorkflowStep,
)


class TestWorkflowDAGPrimitives(unittest.TestCase):
    """Validate DAG ordering, cycle detection, and executor behavior."""

    def test_validate_and_topological_sort(self) -> None:
        dag = WorkflowDAG()
        dag.add_step(WorkflowStep(step_id="a", action="root"))
        dag.add_step(WorkflowStep(step_id="b", action="child", depends_on=["a"]))
        dag.add_step(WorkflowStep(step_id="c", action="child", depends_on=["a"]))

        valid, error = dag.validate()
        self.assertTrue(valid)
        self.assertIsNone(error)

        levels = dag.topological_sort()
        self.assertEqual(levels[0], ["a"])
        self.assertCountEqual(levels[1], ["b", "c"])

    def test_cycle_detection(self) -> None:
        dag = WorkflowDAG()
        dag.add_step(WorkflowStep(step_id="a", action="x", depends_on=["b"]))
        dag.add_step(WorkflowStep(step_id="b", action="y", depends_on=["a"]))

        valid, error = dag.validate()
        self.assertFalse(valid)
        self.assertIn("cycle", str(error).lower())

    def test_executor_success_path(self) -> None:
        async def _run() -> None:
            steps = [
                {"step_id": "fetch", "action": "fetch"},
                {"step_id": "process", "action": "process", "depends_on": ["fetch"]},
            ]

            async def execute_step(step: WorkflowStep):
                await anyio.sleep(0)
                return {"step": step.step_id}

            executor = WorkflowDAGExecutor()
            result = await executor.execute_workflow(steps, execute_step)

            self.assertTrue(result["success"])
            self.assertEqual(result["steps_completed"], 2)
            self.assertEqual(result["steps_failed"], 0)
            self.assertEqual(result["results"]["fetch"]["status"], StepStatus.COMPLETED.value)
            self.assertEqual(result["results"]["process"]["status"], StepStatus.COMPLETED.value)

        anyio.run(_run)

    def test_executor_failure_marks_dependent_skipped(self) -> None:
        async def _run() -> None:
            steps = [
                {"step_id": "root", "action": "root"},
                {"step_id": "downstream", "action": "downstream", "depends_on": ["root"]},
            ]

            async def execute_step(step: WorkflowStep):
                if step.step_id == "root":
                    raise RuntimeError("boom")
                return {"step": step.step_id}

            executor = WorkflowDAGExecutor()
            result = await executor.execute_workflow(steps, execute_step)

            self.assertFalse(result["success"])
            self.assertEqual(result["steps_failed"], 1)
            self.assertEqual(result["steps_skipped"], 1)
            self.assertEqual(result["results"]["downstream"]["status"], StepStatus.SKIPPED.value)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
