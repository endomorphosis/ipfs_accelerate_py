"""Workflow DAG primitives for MCP++ runtime integration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import anyio

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents one step in a workflow DAG."""

    step_id: str
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def mark_ready(self) -> None:
        self.status = StepStatus.READY

    def mark_running(self) -> None:
        self.status = StepStatus.RUNNING

    def mark_completed(self, result: Any, execution_time_ms: float) -> None:
        self.status = StepStatus.COMPLETED
        self.result = result
        self.execution_time_ms = execution_time_ms

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error = error

    def mark_skipped(self, reason: str) -> None:
        self.status = StepStatus.SKIPPED
        self.error = reason


@dataclass
class WorkflowDAG:
    """Directed acyclic graph representation of a workflow."""

    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)
    reverse_adjacency_list: Dict[str, List[str]] = field(default_factory=dict)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step and update adjacency indexes."""
        self.steps[step.step_id] = step

        if step.step_id not in self.adjacency_list:
            self.adjacency_list[step.step_id] = []
        if step.step_id not in self.reverse_adjacency_list:
            self.reverse_adjacency_list[step.step_id] = []

        for dep in step.depends_on:
            if dep not in self.adjacency_list:
                self.adjacency_list[dep] = []
            self.adjacency_list[dep].append(step.step_id)

            if dep not in self.reverse_adjacency_list:
                self.reverse_adjacency_list[dep] = []
            self.reverse_adjacency_list[step.step_id].append(dep)

    def get_root_steps(self) -> List[str]:
        """Get step IDs with no dependencies."""
        return [step_id for step_id, step in self.steps.items() if not step.depends_on]

    def get_ready_steps(self) -> List[str]:
        """Get step IDs whose dependencies have completed."""
        ready: List[str] = []
        for step_id, step in self.steps.items():
            if step.status != StepStatus.PENDING:
                continue
            all_deps_completed = all(
                self.steps[dep].status == StepStatus.COMPLETED for dep in step.depends_on if dep in self.steps
            )
            if all_deps_completed:
                ready.append(step_id)
        return ready

    def detect_cycles(self) -> Optional[List[str]]:
        """Detect cycles using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.adjacency_list.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, path.copy())
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            rec_stack.remove(node)
            return None

        for step_id in self.steps:
            if step_id not in visited:
                cycle = dfs(step_id, [])
                if cycle:
                    return cycle

        return None

    def topological_sort(self) -> List[List[str]]:
        """Return parallelizable execution levels by topological order."""
        in_degree = {step_id: 0 for step_id in self.steps}
        for step_id, step in self.steps.items():
            in_degree[step_id] = len([dep for dep in step.depends_on if dep in self.steps])

        levels: List[List[str]] = []
        current_level = [step_id for step_id, degree in in_degree.items() if degree == 0]

        while current_level:
            levels.append(current_level)
            next_level: List[str] = []

            for step_id in current_level:
                for neighbor in self.adjacency_list.get(step_id, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)

            current_level = next_level

        if len([step for level in levels for step in level]) != len(self.steps):
            raise ValueError("Workflow contains cycles - cannot perform topological sort")

        return levels

    def get_execution_order(self) -> List[List[str]]:
        """Alias for topological execution levels."""
        return self.topological_sort()

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate dependency references and cycle constraints."""
        for step_id, step in self.steps.items():
            for dep in step.depends_on:
                if dep not in self.steps:
                    return False, f"Step '{step_id}' depends on non-existent step '{dep}'"

        cycle = self.detect_cycles()
        if cycle:
            cycle_str = " -> ".join(cycle)
            return False, f"Workflow contains cycle: {cycle_str}"

        return True, None


async def _gather(coros: List[Any]) -> List[Any]:
    """Run coroutines concurrently and return values/exceptions in order."""
    results: List[Any] = [None] * len(coros)

    async def _runner(idx: int, coro: Any) -> None:
        try:
            results[idx] = await coro
        except Exception as exc:  # noqa: BLE001 - intentionally surfaced as result element
            results[idx] = exc

    async with anyio.create_task_group() as tg:
        for idx, coro in enumerate(coros):
            tg.start_soon(_runner, idx, coro)

    return results


class WorkflowDAGExecutor:
    """Execute workflow DAG levels with parallel step execution."""

    def __init__(self, max_concurrent: int = 10, on_step_complete: Optional[callable] = None):
        self.max_concurrent = max_concurrent
        self.on_step_complete = on_step_complete
        self.dag: Optional[WorkflowDAG] = None

    async def execute_workflow(self, steps: List[Dict[str, Any]], step_executor: callable) -> Dict[str, Any]:
        """Execute steps using dependency-ordered levels."""
        self.dag = WorkflowDAG()

        for step_dict in steps:
            step = WorkflowStep(
                step_id=step_dict["step_id"],
                action=step_dict["action"],
                inputs=step_dict.get("inputs", {}),
                depends_on=step_dict.get("depends_on", []),
            )
            self.dag.add_step(step)

        is_valid, error_msg = self.dag.validate()
        if not is_valid:
            logger.error("Invalid workflow: %s", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "steps_completed": 0,
                "steps_failed": 0,
            }

        try:
            execution_levels = self.dag.get_execution_order()
        except ValueError as exc:
            logger.error("Failed to determine execution order: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "steps_completed": 0,
                "steps_failed": 0,
            }

        steps_completed = 0
        steps_failed = 0

        for level in execution_levels:
            tasks: List[Any] = []
            for step_id in level:
                step = self.dag.steps[step_id]
                step.mark_running()
                tasks.append(self._execute_step(step, step_executor))

            level_results = await _gather(tasks)

            for step_id, result in zip(level, level_results):
                step = self.dag.steps[step_id]
                if isinstance(result, Exception):
                    step.mark_failed(str(result))
                    steps_failed += 1
                else:
                    step.mark_completed(result["result"], result["execution_time_ms"])
                    steps_completed += 1
                    if self.on_step_complete:
                        self.on_step_complete(step_id, result)

            if steps_failed > 0:
                self._mark_skipped_steps()
                break

        return {
            "success": steps_failed == 0,
            "steps_completed": steps_completed,
            "steps_failed": steps_failed,
            "steps_skipped": len([s for s in self.dag.steps.values() if s.status == StepStatus.SKIPPED]),
            "total_steps": len(self.dag.steps),
            "results": {
                step_id: {
                    "status": step.status.value,
                    "result": step.result,
                    "error": step.error,
                    "execution_time_ms": step.execution_time_ms,
                }
                for step_id, step in self.dag.steps.items()
            },
        }

    async def _execute_step(self, step: WorkflowStep, step_executor: callable) -> Dict[str, Any]:
        """Execute a single workflow step with timing."""
        start_time = time.perf_counter()
        result = await step_executor(step)
        end_time = time.perf_counter()
        return {
            "success": True,
            "result": result,
            "execution_time_ms": (end_time - start_time) * 1000,
        }

    def _mark_skipped_steps(self) -> None:
        """Mark pending steps as skipped when they depend on failed steps."""
        if not self.dag:
            return

        for step_id, step in self.dag.steps.items():
            if step.status != StepStatus.PENDING:
                continue

            failed_deps = [dep for dep in step.depends_on if self.dag.steps[dep].status == StepStatus.FAILED]
            if failed_deps:
                step.mark_skipped(f"Depends on failed step(s): {', '.join(failed_deps)}")

    def get_workflow_graph(self) -> Dict[str, Any]:
        """Return graph representation suitable for visualization."""
        if not self.dag:
            return {"nodes": [], "edges": []}

        nodes = [
            {
                "id": step_id,
                "label": f"{step_id}\\n{step.action}",
                "status": step.status.value,
            }
            for step_id, step in self.dag.steps.items()
        ]

        edges = []
        for step_id, step in self.dag.steps.items():
            for dep in step.depends_on:
                edges.append({"from": dep, "to": step_id})

        return {"nodes": nodes, "edges": edges}


__all__ = [
    "StepStatus",
    "WorkflowStep",
    "WorkflowDAG",
    "WorkflowDAGExecutor",
]
