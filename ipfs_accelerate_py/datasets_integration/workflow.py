"""
Workflow Coordinator - P2P workflow scheduling and coordination

This module provides workflow scheduling and coordination for distributed
operations using ipfs_datasets_py's P2P workflow scheduler.
"""

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import json


class WorkflowCoordinator:
    """
    Coordinator for P2P workflow scheduling and task distribution.
    
    Provides a unified interface for:
    - Task submission and scheduling
    - Worker coordination
    - Distributed consensus
    - Task prioritization
    
    Uses ipfs_datasets_py's P2PWorkflowScheduler when available, falling back
    to local task queue otherwise.
    
    Attributes:
        enabled (bool): Whether P2P workflow scheduling is active
        workflow_scheduler: P2PWorkflowScheduler instance (if available)
        task_queue (List): Local task queue (fallback mode)
    
    Example:
        >>> coordinator = WorkflowCoordinator()
        >>> coordinator.submit_task("infer-001", "inference", {
        ...     "model": "bert-base",
        ...     "input": "text data"
        ... }, priority=8)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow coordinator.
        
        Args:
            config: Optional configuration dictionary
                - enable_p2p: Enable P2P mode (default: False for safety)
                - cache_dir: Directory for task cache
        """
        self.config = config or {}
        self.enabled = False
        self.workflow_scheduler = None
        self.task_queue = []
        
        # Set up cache directory
        cache_dir = self.config.get('cache_dir')
        if not cache_dir:
            cache_dir = Path.home() / '.cache' / 'ipfs_accelerate' / 'workflows'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Task queue file for fallback mode
        self.task_file = self.cache_dir / 'tasks.jsonl'
        
        # Load existing tasks
        self._load_tasks()
        
        # Try to initialize P2P workflow scheduler if enabled
        if self.config.get('enable_p2p', False):
            self._initialize()
    
    def _initialize(self):
        """Initialize P2PWorkflowScheduler if ipfs_datasets_py is available."""
        try:
            from ipfs_datasets_py.p2p_workflow_scheduler import get_scheduler
            
            self.workflow_scheduler = get_scheduler()
            self.enabled = True
            
        except (ImportError, Exception):
            # P2P not available - will use local task queue
            self.enabled = False
    
    def _load_tasks(self):
        """Load tasks from local task file."""
        if not self.task_file.exists():
            return
        
        with open(self.task_file, 'r') as f:
            for line in f:
                try:
                    task = json.loads(line)
                    if task.get('status') == 'pending':
                        self.task_queue.append(task)
                except json.JSONDecodeError:
                    continue
    
    def _save_task(self, task: Dict[str, Any]):
        """Save a task to local task file."""
        with open(self.task_file, 'a') as f:
            f.write(json.dumps(task) + '\n')
    
    def submit_task(self, task_id: str, task_type: str, 
                   data: Dict[str, Any], priority: int = 5,
                   tags: Optional[List[str]] = None) -> bool:
        """
        Submit a workflow task for execution.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task (e.g., "inference", "preprocessing", "training")
            data: Task data and parameters
            priority: Task priority (1-10, higher = more important)
            tags: Optional task tags (e.g., ["P2P_ELIGIBLE", "GPU_REQUIRED"])
        
        Returns:
            bool: True if submitted successfully
        
        Example:
            >>> coordinator.submit_task(
            ...     task_id="infer-001",
            ...     task_type="inference",
            ...     data={"model": "bert", "batch_size": 32},
            ...     priority=8,
            ...     tags=["P2P_ELIGIBLE", "GPU_PREFERRED"]
            ... )
        """
        task = {
            'task_id': task_id,
            'task_type': task_type,
            'data': data,
            'priority': priority,
            'tags': tags or [],
            'status': 'pending'
        }
        
        # Try to submit to P2P scheduler
        if self.enabled and self.workflow_scheduler:
            try:
                from ipfs_datasets_py.p2p_workflow_scheduler import WorkflowDefinition, WorkflowTag
                
                # Convert string tags to WorkflowTag enums
                workflow_tags = []
                if tags:
                    for tag in tags:
                        try:
                            workflow_tags.append(getattr(WorkflowTag, tag.upper()))
                        except AttributeError:
                            # Unknown tag, skip it
                            pass
                
                workflow = WorkflowDefinition(
                    id=task_id,
                    task_type=task_type,
                    data=data,
                    priority=priority,
                    tags=workflow_tags
                )
                
                self.workflow_scheduler.submit_workflow(workflow)
                task['status'] = 'submitted_p2p'
                self._save_task(task)
                return True
                
            except Exception:
                pass
        
        # Fallback: Add to local queue
        self.task_queue.append(task)
        self._save_task(task)
        return True
    
    def get_next_task(self, worker_id: Optional[str] = None,
                     capabilities: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next task to execute.
        
        Args:
            worker_id: Unique identifier for the worker
            capabilities: List of worker capabilities (e.g., ["GPU", "LARGE_MEM"])
        
        Returns:
            Optional[Dict]: Next task to execute, or None if queue is empty
        
        Example:
            >>> task = coordinator.get_next_task(
            ...     worker_id="worker-001",
            ...     capabilities=["GPU", "CUDA"]
            ... )
            >>> if task:
            ...     # Execute task
            ...     pass
        """
        if self.enabled and self.workflow_scheduler:
            try:
                return self.workflow_scheduler.get_next_task(
                    worker_id=worker_id,
                    capabilities=capabilities or []
                )
            except Exception:
                pass
        
        # Fallback: Get from local queue (highest priority first)
        if not self.task_queue:
            return None
        
        # Sort by priority (descending)
        self.task_queue.sort(key=lambda t: t.get('priority', 5), reverse=True)
        
        # Pop the highest priority task
        task = self.task_queue.pop(0)
        task['status'] = 'assigned'
        task['worker_id'] = worker_id
        return task
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: Unique task identifier
            result: Task result data
        
        Returns:
            bool: True if marked successfully
        
        Example:
            >>> coordinator.complete_task("infer-001", {
            ...     "status": "success",
            ...     "output_cid": "Qm...",
            ...     "duration_ms": 5000
            ... })
        """
        if self.enabled and self.workflow_scheduler:
            try:
                self.workflow_scheduler.complete_task(task_id, result)
                return True
            except Exception:
                pass
        
        # Fallback: Save completion to file
        completion = {
            'task_id': task_id,
            'status': 'completed',
            'result': result
        }
        self._save_task(completion)
        return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: Unique task identifier
            error: Error message
        
        Returns:
            bool: True if marked successfully
        
        Example:
            >>> coordinator.fail_task("infer-001", "Out of memory")
        """
        if self.enabled and self.workflow_scheduler:
            try:
                self.workflow_scheduler.fail_task(task_id, error)
                return True
            except Exception:
                pass
        
        # Fallback: Save failure to file
        failure = {
            'task_id': task_id,
            'status': 'failed',
            'error': error
        }
        self._save_task(failure)
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: Unique task identifier
        
        Returns:
            Optional[Dict]: Task status, or None if not found
        
        Example:
            >>> status = coordinator.get_task_status("infer-001")
            >>> print(f"Status: {status['status']}")
        """
        if self.enabled and self.workflow_scheduler:
            try:
                return self.workflow_scheduler.get_task_status(task_id)
            except Exception:
                pass
        
        # Fallback: Search in task file
        if not self.task_file.exists():
            return None
        
        with open(self.task_file, 'r') as f:
            for line in reversed(list(f)):  # Search from most recent
                try:
                    task = json.loads(line)
                    if task.get('task_id') == task_id:
                        return task
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def list_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        List all pending tasks.
        
        Returns:
            List of pending tasks
        
        Example:
            >>> tasks = coordinator.list_pending_tasks()
            >>> print(f"Pending tasks: {len(tasks)}")
        """
        if self.enabled and self.workflow_scheduler:
            try:
                return self.workflow_scheduler.list_pending_tasks()
            except Exception:
                pass
        
        # Fallback: Return local queue
        return [t for t in self.task_queue if t.get('status') == 'pending']
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Unique task identifier
        
        Returns:
            bool: True if cancelled successfully
        
        Example:
            >>> coordinator.cancel_task("infer-001")
        """
        if self.enabled and self.workflow_scheduler:
            try:
                self.workflow_scheduler.cancel_task(task_id)
                return True
            except Exception:
                pass
        
        # Fallback: Remove from local queue
        self.task_queue = [
            t for t in self.task_queue
            if t.get('task_id') != task_id
        ]
        
        # Mark as cancelled in file
        cancellation = {
            'task_id': task_id,
            'status': 'cancelled'
        }
        self._save_task(cancellation)
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of workflow coordinator.
        
        Returns:
            Dict with status information
        
        Example:
            >>> status = coordinator.get_status()
            >>> print(f"P2P enabled: {status['p2p_enabled']}")
        """
        return {
            'p2p_enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'pending_tasks': len(self.list_pending_tasks()),
            'workflow_scheduler': self.workflow_scheduler is not None,
        }
