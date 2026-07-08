"""Workflow engine primitive for MCP++ runtime integration."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import anyio

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a workflow task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """Status of a workflow."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class Task:
    """Represents one task inside a workflow."""

    task_id: str
    name: str
    function: Any
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        if self.status != TaskStatus.PENDING:
            return False
        return all(dep in completed_tasks for dep in self.dependencies)

    def can_retry(self) -> bool:
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


@dataclass
class Workflow:
    """Represents a workflow DAG of executable tasks."""

    workflow_id: str
    name: str
    description: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists in workflow")
        self.tasks[task.task_id] = task

    def validate_dag(self) -> None:
        """Ensure task graph does not contain cycles."""

        def has_cycle(task_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = self.tasks.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id, visited, rec_stack):
                    raise ValueError(f"Workflow contains a cycle involving task {task_id}")

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """Return tasks that can be executed immediately."""
        return [task for task in self.tasks.values() if task.is_ready(completed_tasks)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "status": self.status.value,
            "created_time": self.created_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
        }


class WorkflowEngine:
    """Engine for dependency-aware workflow execution."""

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.workflows: Dict[str, Workflow] = {}
        self.task_functions: Dict[str, Callable] = {}
        self._running_tasks: Set[str] = set()
        self._semaphore = anyio.Semaphore(max_concurrent_tasks)

    def register_function(self, name: str, func: Callable) -> None:
        self.task_functions[name] = func

    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Workflow:
        if workflow_id in self.workflows:
            raise ValueError(f"Workflow {workflow_id} already exists")

        workflow = Workflow(workflow_id=workflow_id, name=name, description=description, metadata=metadata or {})
        self.workflows[workflow_id] = workflow
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self.workflows.get(workflow_id)

    async def execute_task(self, workflow: Workflow, task: Task) -> None:
        """Execute one task with timeout and failure handling."""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            self._running_tasks.add(task.task_id)

            try:
                if isinstance(task.function, str):
                    func = self.task_functions.get(task.function)
                    if not func:
                        raise ValueError(f"Function {task.function} not registered")
                else:
                    func = task.function

                try:
                    with anyio.fail_after(task.timeout):
                        if inspect.iscoroutinefunction(func):
                            task.result = await func(*task.args, **task.kwargs)
                        else:
                            task.result = await anyio.to_thread.run_sync(lambda: func(*task.args, **task.kwargs))

                    task.status = TaskStatus.COMPLETED
                except TimeoutError:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task timed out after {task.timeout}s"

            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
            finally:
                task.end_time = time.time()
                self._running_tasks.discard(task.task_id)

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow until completion or terminal failure."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow.validate_dag()

        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = time.time()

        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()

        try:
            while True:
                ready_tasks = workflow.get_ready_tasks(completed_tasks)
                for task in ready_tasks:
                    task.status = TaskStatus.READY

                if ready_tasks:
                    async with anyio.create_task_group() as tg:
                        for task in ready_tasks:
                            tg.start_soon(self.execute_task, workflow, task)

                    for task in ready_tasks:
                        if task.status == TaskStatus.COMPLETED:
                            completed_tasks.add(task.task_id)
                        elif task.status == TaskStatus.FAILED:
                            if task.can_retry():
                                task.retry_count += 1
                                task.status = TaskStatus.PENDING
                            else:
                                failed_tasks.add(task.task_id)
                else:
                    break

            all_completed = all(t.status == TaskStatus.COMPLETED for t in workflow.tasks.values())

            if all_completed:
                workflow.status = WorkflowStatus.COMPLETED
            elif failed_tasks:
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            workflow.status = WorkflowStatus.FAILED
            logger.error("Workflow %s execution failed: %s", workflow_id, exc)
        finally:
            workflow.end_time = time.time()

        return {
            "workflow_id": workflow_id,
            "status": workflow.status.value,
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "total_tasks": len(workflow.tasks),
            "execution_time": (
                workflow.end_time - workflow.start_time
                if workflow.end_time is not None and workflow.start_time is not None
                else 0
            ),
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a currently running workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow or workflow.status != WorkflowStatus.RUNNING:
            return False

        workflow.status = WorkflowStatus.CANCELLED
        workflow.end_time = time.time()

        for task in workflow.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED

        return True

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        return workflow.to_dict()


_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get singleton workflow engine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine


def reset_workflow_engine() -> None:
    """Reset singleton workflow engine instance."""
    global _workflow_engine
    _workflow_engine = None


__all__ = [
    "TaskStatus",
    "WorkflowStatus",
    "Task",
    "Workflow",
    "WorkflowEngine",
    "get_workflow_engine",
    "reset_workflow_engine",
]
