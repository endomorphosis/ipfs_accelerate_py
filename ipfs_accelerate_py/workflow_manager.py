"""
Workflow Management System for IPFS Accelerate MCP Server

This module provides workflow definition, execution, and management capabilities.
"""

import json
import sqlite3
import time
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow"""
    task_id: str
    name: str
    type: str  # 'inference', 'processing', 'custom'
    config: Dict[str, Any]
    status: str = TaskStatus.PENDING.value
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: List[str] = None  # List of task_ids that must complete first

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class Workflow:
    """Represents a complete workflow"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    status: str = WorkflowStatus.PENDING.value
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}

    def get_progress(self) -> Dict[str, int]:
        """Calculate workflow progress"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED.value)
        running = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING.value)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED.value)
        
        return {
            'total': total,
            'completed': completed,
            'running': running,
            'failed': failed,
            'pending': total - completed - running - failed
        }


class WorkflowStorage:
    """Handles workflow persistence using SQLite"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".ipfs_accelerate" / "workflows.db")
        
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    error TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    started_at REAL,
                    completed_at REAL,
                    dependencies TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            conn.commit()
    
    def save_workflow(self, workflow: Workflow):
        """Save or update a workflow"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows 
                (workflow_id, name, description, status, created_at, started_at, completed_at, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                workflow.status,
                workflow.created_at,
                workflow.started_at,
                workflow.completed_at,
                workflow.error,
                json.dumps(workflow.metadata)
            ))
            
            # Save tasks
            for task in workflow.tasks:
                conn.execute("""
                    INSERT OR REPLACE INTO tasks
                    (task_id, workflow_id, name, type, config, status, result, error, 
                     started_at, completed_at, dependencies)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    workflow.workflow_id,
                    task.name,
                    task.type,
                    json.dumps(task.config),
                    task.status,
                    json.dumps(task.result) if task.result else None,
                    task.error,
                    task.started_at,
                    task.completed_at,
                    json.dumps(task.dependencies)
                ))
            
            conn.commit()
    
    def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load a workflow by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Load workflow
            row = conn.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", 
                (workflow_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # Load tasks
            task_rows = conn.execute(
                "SELECT * FROM tasks WHERE workflow_id = ? ORDER BY task_id",
                (workflow_id,)
            ).fetchall()
            
            tasks = []
            for task_row in task_rows:
                tasks.append(WorkflowTask(
                    task_id=task_row['task_id'],
                    name=task_row['name'],
                    type=task_row['type'],
                    config=json.loads(task_row['config']),
                    status=task_row['status'],
                    result=json.loads(task_row['result']) if task_row['result'] else None,
                    error=task_row['error'],
                    started_at=task_row['started_at'],
                    completed_at=task_row['completed_at'],
                    dependencies=json.loads(task_row['dependencies'])
                ))
            
            return Workflow(
                workflow_id=row['workflow_id'],
                name=row['name'],
                description=row['description'],
                tasks=tasks,
                status=row['status'],
                created_at=row['created_at'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                error=row['error'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
    
    def list_workflows(self, status: Optional[str] = None, limit: int = 100) -> List[Workflow]:
        """List all workflows, optionally filtered by status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                rows = conn.execute(
                    "SELECT workflow_id FROM workflows WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT workflow_id FROM workflows ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            
            workflows = []
            for row in rows:
                workflow = self.load_workflow(row['workflow_id'])
                if workflow:
                    workflows.append(workflow)
            
            return workflows
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow and its tasks"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE workflow_id = ?", (workflow_id,))
            conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            conn.commit()


class WorkflowEngine:
    """Executes workflows and manages their lifecycle"""
    
    def __init__(self, storage: WorkflowStorage, ipfs_accelerate_instance=None):
        self.storage = storage
        self.ipfs_instance = ipfs_accelerate_instance
        self._running_workflows: Dict[str, asyncio.Task] = {}
    
    async def execute_task(self, workflow: Workflow, task: WorkflowTask) -> bool:
        """Execute a single task"""
        logger.info(f"Executing task {task.task_id}: {task.name}")
        
        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()
        self.storage.save_workflow(workflow)
        
        try:
            if task.type == "inference":
                # Execute inference task
                if self.ipfs_instance:
                    model = task.config.get('model', 'gpt2')
                    inputs = task.config.get('inputs', ['Hello world'])
                    
                    # Use the existing inference infrastructure
                    result = await self._run_inference(model, inputs)
                    task.result = result
                else:
                    # Simulated inference for testing
                    await asyncio.sleep(1)
                    task.result = {'output': 'Simulated output', 'model': task.config.get('model')}
                
            elif task.type == "processing":
                # Custom processing task
                await asyncio.sleep(0.5)
                task.result = {'processed': True}
                
            else:
                # Custom task type
                await asyncio.sleep(0.5)
                task.result = {'completed': True}
            
            task.status = TaskStatus.COMPLETED.value
            task.completed_at = time.time()
            logger.info(f"Task {task.task_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED.value
            task.error = str(e)
            task.completed_at = time.time()
            return False
        
        finally:
            self.storage.save_workflow(workflow)
    
    async def _run_inference(self, model: str, inputs: List[str]) -> Dict[str, Any]:
        """Run inference using the IPFS accelerate instance"""
        if not self.ipfs_instance:
            raise RuntimeError("IPFS accelerate instance not available")
        
        # This would integrate with the actual inference system
        # For now, return a simulated result
        return {
            'model': model,
            'inputs': inputs,
            'outputs': [f"Output for: {inp}" for inp in inputs],
            'timestamp': time.time()
        }
    
    async def execute_workflow(self, workflow_id: str):
        """Execute a complete workflow"""
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow.status not in [WorkflowStatus.PENDING.value, WorkflowStatus.PAUSED.value]:
            raise ValueError(f"Workflow {workflow_id} is not in a runnable state: {workflow.status}")
        
        logger.info(f"Starting workflow {workflow_id}: {workflow.name}")
        workflow.status = WorkflowStatus.RUNNING.value
        workflow.started_at = time.time()
        self.storage.save_workflow(workflow)
        
        try:
            # Track completed tasks for dependency resolution
            completed_tasks = set()
            
            while True:
                # Find tasks that are ready to run (dependencies met)
                runnable_tasks = []
                for task in workflow.tasks:
                    if task.status == TaskStatus.PENDING.value:
                        deps_met = all(dep in completed_tasks for dep in task.dependencies)
                        if deps_met:
                            runnable_tasks.append(task)
                
                if not runnable_tasks:
                    # Check if all tasks are done
                    all_done = all(
                        t.status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.SKIPPED.value]
                        for t in workflow.tasks
                    )
                    if all_done:
                        break
                    
                    # Check for paused state
                    workflow = self.storage.load_workflow(workflow_id)
                    if workflow.status == WorkflowStatus.PAUSED.value:
                        logger.info(f"Workflow {workflow_id} paused")
                        return
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(0.5)
                    continue
                
                # Execute runnable tasks (could be parallelized)
                for task in runnable_tasks:
                    success = await self.execute_task(workflow, task)
                    if success:
                        completed_tasks.add(task.task_id)
                
                # Reload workflow to get latest state
                workflow = self.storage.load_workflow(workflow_id)
            
            # Determine final status
            if any(t.status == TaskStatus.FAILED.value for t in workflow.tasks):
                workflow.status = WorkflowStatus.FAILED.value
                workflow.error = "One or more tasks failed"
            else:
                workflow.status = WorkflowStatus.COMPLETED.value
            
            workflow.completed_at = time.time()
            logger.info(f"Workflow {workflow_id} finished with status: {workflow.status}")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} error: {e}")
            workflow.status = WorkflowStatus.FAILED.value
            workflow.error = str(e)
            workflow.completed_at = time.time()
        
        finally:
            self.storage.save_workflow(workflow)
            if workflow_id in self._running_workflows:
                del self._running_workflows[workflow_id]
    
    def start_workflow(self, workflow_id: str):
        """Start a workflow in the background"""
        if workflow_id in self._running_workflows:
            raise ValueError(f"Workflow {workflow_id} is already running")
        
        task = asyncio.create_task(self.execute_workflow(workflow_id))
        self._running_workflows[workflow_id] = task
        return task
    
    def pause_workflow(self, workflow_id: str):
        """Pause a running workflow"""
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow.status != WorkflowStatus.RUNNING.value:
            raise ValueError(f"Workflow {workflow_id} is not running")
        
        workflow.status = WorkflowStatus.PAUSED.value
        self.storage.save_workflow(workflow)
        logger.info(f"Workflow {workflow_id} paused")
    
    def stop_workflow(self, workflow_id: str):
        """Stop a workflow"""
        workflow = self.storage.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow.status = WorkflowStatus.STOPPED.value
        workflow.completed_at = time.time()
        self.storage.save_workflow(workflow)
        
        # Cancel the task if it's running
        if workflow_id in self._running_workflows:
            self._running_workflows[workflow_id].cancel()
            del self._running_workflows[workflow_id]
        
        logger.info(f"Workflow {workflow_id} stopped")


class WorkflowManager:
    """High-level workflow management interface"""
    
    def __init__(self, storage: WorkflowStorage = None, ipfs_accelerate_instance=None):
        if storage is None:
            storage = WorkflowStorage()
        
        self.storage = storage
        self.engine = WorkflowEngine(storage, ipfs_accelerate_instance)
    
    def create_workflow(self, name: str, description: str, tasks: List[Dict[str, Any]]) -> Workflow:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Convert task dicts to WorkflowTask objects
        workflow_tasks = []
        for i, task_dict in enumerate(tasks):
            task_id = f"{workflow_id}-task-{i}"
            workflow_tasks.append(WorkflowTask(
                task_id=task_id,
                name=task_dict['name'],
                type=task_dict['type'],
                config=task_dict.get('config', {}),
                dependencies=task_dict.get('dependencies', [])
            ))
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks
        )
        
        self.storage.save_workflow(workflow)
        logger.info(f"Created workflow {workflow_id}: {name}")
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID"""
        return self.storage.load_workflow(workflow_id)
    
    def list_workflows(self, status: Optional[str] = None) -> List[Workflow]:
        """List all workflows"""
        return self.storage.list_workflows(status=status)
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow"""
        self.storage.delete_workflow(workflow_id)
        logger.info(f"Deleted workflow {workflow_id}")
    
    def start_workflow(self, workflow_id: str):
        """Start executing a workflow"""
        return self.engine.start_workflow(workflow_id)
    
    def pause_workflow(self, workflow_id: str):
        """Pause a workflow"""
        self.engine.pause_workflow(workflow_id)
    
    def stop_workflow(self, workflow_id: str):
        """Stop a workflow"""
        self.engine.stop_workflow(workflow_id)
