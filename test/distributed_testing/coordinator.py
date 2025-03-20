#\!/usr/bin/env python3
"""
Distributed testing coordinator for IPFS Accelerate.

This module provides functionality for coordinating distributed test execution.
"""

import os
import sys
import json
import time
import uuid
import socket
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
import datetime
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization packages not available. Install matplotlib, seaborn, pandas to enable visualizations.")
    VISUALIZATION_AVAILABLE = False


class NodeRole(Enum):
    """Enum for node roles."""
    LEADER = auto()
    FOLLOWER = auto()
    CANDIDATE = auto()
    OFFLINE = auto()


class TaskStatus(Enum):
    """Enum for task status."""
    PENDING = auto()
    ASSIGNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class WorkerStatus(Enum):
    """Enum for worker status."""
    IDLE = auto()
    BUSY = auto()
    OFFLINE = auto()


@dataclass
class Task:
    """Class representing a test task."""
    id: str
    test_path: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    assigned_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher number = higher priority


@dataclass
class Worker:
    """Class representing a test worker."""
    id: str
    hostname: str
    ip_address: str
    capabilities: Dict[str, Any]
    status: WorkerStatus = WorkerStatus.IDLE
    current_task_id: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    total_execution_time: float = 0.0


@dataclass
class CoordinatorState:
    """Class representing the coordinator state."""
    id: str
    role: NodeRole
    tasks: Dict[str, Task]
    workers: Dict[str, Worker]
    start_time: float
    leader_id: Optional[str] = None
    term: int = 0  # For leader election
    last_applied: int = 0  # For state replication
    commit_index: int = 0  # For state replication
    last_status_update: float = field(default_factory=time.time)


class TaskQueue:
    """Priority queue for tasks."""
    
    def __init__(self):
        """Initialize the task queue."""
        self._queue = []
        self._lock = threading.Lock()
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to the queue.
        
        Args:
            task: The task to add
        """
        with self._lock:
            self._queue.append(task)
            # Sort by priority (high to low) and then by assignment time (oldest first)
            self._queue.sort(key=lambda x: (-x.priority, x.assigned_time or float('inf')))
    
    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task from the queue.
        
        Returns:
            The next task or None if the queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)
    
    def peek_next_task(self) -> Optional[Task]:
        """
        Peek at the next task without removing it.
        
        Returns:
            The next task or None if the queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]
    
    def remove_task(self, task_id: str) -> Optional[Task]:
        """
        Remove a task from the queue.
        
        Args:
            task_id: The ID of the task to remove
            
        Returns:
            The removed task or None if the task was not found
        """
        with self._lock:
            for i, task in enumerate(self._queue):
                if task.id == task_id:
                    return self._queue.pop(i)
            return None
    
    def __len__(self) -> int:
        """
        Get the length of the queue.
        
        Returns:
            The number of tasks in the queue
        """
        with self._lock:
            return len(self._queue)


class TestCoordinator:
    """
    Class for coordinating distributed test execution.
    """
    
    def __init__(self, 
                 host: str = '0.0.0.0', 
                 port: int = 5000, 
                 heartbeat_interval: int = 10,
                 worker_timeout: int = 30,
                 high_availability: bool = False):
        """
        Initialize the coordinator.
        
        Args:
            host: The host to bind to
            port: The port to bind to
            heartbeat_interval: The interval in seconds between heartbeats
            worker_timeout: The time in seconds after which a worker is considered offline
            high_availability: Whether to enable high availability mode
        """
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.worker_timeout = worker_timeout
        self.high_availability = high_availability
        
        # Initialize state
        self.id = str(uuid.uuid4())
        self.state = CoordinatorState(
            id=self.id,
            role=NodeRole.LEADER if not high_availability else NodeRole.CANDIDATE,
            tasks={},
            workers={},
            start_time=time.time()
        )
        
        # Initialize task queue
        self.task_queue = TaskQueue()
        
        # Initialize locks
        self.state_lock = threading.Lock()
        self.task_queue_lock = threading.Lock()
        
        # Initialize event for stopping threads
        self.stop_event = threading.Event()
        
        # Initialize threads
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.assignment_thread = threading.Thread(target=self._assignment_loop)
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        
        # Initialize statistics
        self.statistics = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'workers_registered': 0,
            'workers_active': 0
        }
        
        # Initialize logging
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # If high availability mode is enabled, start leadership election
        if high_availability:
            self.election_thread = threading.Thread(target=self._election_loop)
        else:
            self.election_thread = None
    
    def start(self) -> None:
        """Start the coordinator."""
        logger.info(f"Starting test coordinator at {self.host}:{self.port}")
        
        # Start threads
        self.heartbeat_thread.start()
        self.assignment_thread.start()
        self.cleanup_thread.start()
        
        if self.election_thread:
            self.election_thread.start()
        
        # Start API server (mock implementation)
        logger.info("Coordinator started")
    
    def stop(self) -> None:
        """Stop the coordinator."""
        logger.info("Stopping test coordinator")
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for threads to stop
        self.heartbeat_thread.join()
        self.assignment_thread.join()
        self.cleanup_thread.join()
        
        if self.election_thread:
            self.election_thread.join()
        
        logger.info("Coordinator stopped")
    
    def register_worker(self, hostname: str, ip_address: str, capabilities: Dict[str, Any]) -> str:
        """
        Register a new worker.
        
        Args:
            hostname: The hostname of the worker
            ip_address: The IP address of the worker
            capabilities: The capabilities of the worker
            
        Returns:
            The ID of the registered worker
        """
        with self.state_lock:
            worker_id = str(uuid.uuid4())
            worker = Worker(
                id=worker_id,
                hostname=hostname,
                ip_address=ip_address,
                capabilities=capabilities
            )
            self.state.workers[worker_id] = worker
            self.statistics['workers_registered'] += 1
            self.statistics['workers_active'] += 1
            
            logger.info(f"Registered worker {worker_id} ({hostname}, {ip_address})")
            
            return worker_id
    
    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker.
        
        Args:
            worker_id: The ID of the worker to unregister
            
        Returns:
            True if the worker was unregistered, False otherwise
        """
        with self.state_lock:
            if worker_id not in self.state.workers:
                logger.warning(f"Attempted to unregister unknown worker {worker_id}")
                return False
            
            worker = self.state.workers[worker_id]
            
            # If the worker has a current task, mark it as pending again
            if worker.current_task_id:
                task_id = worker.current_task_id
                if task_id in self.state.tasks:
                    task = self.state.tasks[task_id]
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    self.task_queue.add_task(task)
            
            # Remove the worker
            del self.state.workers[worker_id]
            self.statistics['workers_active'] -= 1
            
            logger.info(f"Unregistered worker {worker_id} ({worker.hostname}, {worker.ip_address})")
            
            return True
    
    def worker_heartbeat(self, worker_id: str, status: Dict[str, Any]) -> bool:
        """
        Process a heartbeat from a worker.
        
        Args:
            worker_id: The ID of the worker
            status: The status of the worker
            
        Returns:
            True if the heartbeat was processed, False otherwise
        """
        with self.state_lock:
            if worker_id not in self.state.workers:
                logger.warning(f"Received heartbeat from unknown worker {worker_id}")
                return False
            
            worker = self.state.workers[worker_id]
            worker.last_heartbeat = time.time()
            
            # Update worker status
            if 'status' in status:
                worker_status = status['status']
                if worker_status == 'idle':
                    worker.status = WorkerStatus.IDLE
                elif worker_status == 'busy':
                    worker.status = WorkerStatus.BUSY
                    
            # Update task status if the worker is working on a task
            if worker.current_task_id and 'task_status' in status:
                task_id = worker.current_task_id
                if task_id in self.state.tasks:
                    task = self.state.tasks[task_id]
                    task_status = status['task_status']
                    
                    if task_status == 'running':
                        task.status = TaskStatus.RUNNING
                        if 'start_time' in status:
                            task.start_time = status['start_time']
                    elif task_status == 'completed':
                        task.status = TaskStatus.COMPLETED
                        task.end_time = time.time()
                        
                        if 'result' in status:
                            task.result = status['result']
                        
                        # Update worker statistics
                        worker.total_tasks_completed += 1
                        worker.total_execution_time += (task.end_time - (task.start_time or task.assigned_time))
                        
                        # Update coordinator statistics
                        self.statistics['tasks_completed'] += 1
                        
                        # Clear worker's current task
                        worker.current_task_id = None
                        worker.status = WorkerStatus.IDLE
                    elif task_status == 'failed':
                        task.status = TaskStatus.FAILED
                        task.end_time = time.time()
                        
                        if 'result' in status:
                            task.result = status['result']
                        
                        # Update coordinator statistics
                        self.statistics['tasks_failed'] += 1
                        
                        # Clear worker's current task
                        worker.current_task_id = None
                        worker.status = WorkerStatus.IDLE
            
            return True
    
    def create_task(self, test_path: str, parameters: Dict[str, Any], priority: int = 0) -> str:
        """
        Create a new test task.
        
        Args:
            test_path: The path to the test to run
            parameters: Parameters for the test
            priority: Priority of the task (higher number = higher priority)
            
        Returns:
            The ID of the created task
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            test_path=test_path,
            parameters=parameters,
            priority=priority
        )
        
        with self.state_lock:
            self.state.tasks[task_id] = task
            self.statistics['tasks_created'] += 1
        
        with self.task_queue_lock:
            self.task_queue.add_task(task)
        
        logger.info(f"Created task {task_id} for test {test_path}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            The status of the task or None if the task was not found
        """
        with self.state_lock:
            if task_id not in self.state.tasks:
                return None
            
            task = self.state.tasks[task_id]
            return {
                'id': task.id,
                'test_path': task.test_path,
                'status': task.status.name,
                'worker_id': task.worker_id,
                'assigned_time': task.assigned_time,
                'start_time': task.start_time,
                'end_time': task.end_time,
                'result': task.result
            }
    
    def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a worker.
        
        Args:
            worker_id: The ID of the worker
            
        Returns:
            The status of the worker or None if the worker was not found
        """
        with self.state_lock:
            if worker_id not in self.state.workers:
                return None
            
            worker = self.state.workers[worker_id]
            return {
                'id': worker.id,
                'hostname': worker.hostname,
                'ip_address': worker.ip_address,
                'status': worker.status.name,
                'current_task_id': worker.current_task_id,
                'last_heartbeat': worker.last_heartbeat,
                'total_tasks_completed': worker.total_tasks_completed,
                'total_execution_time': worker.total_execution_time,
                'capabilities': worker.capabilities
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get coordinator statistics.
        
        Returns:
            A dictionary with coordinator statistics
        """
        with self.state_lock:
            stats = self.statistics.copy()
            stats['uptime'] = time.time() - self.state.start_time
            stats['tasks_pending'] = len(self.task_queue)
            stats['tasks_running'] = sum(1 for task in self.state.tasks.values() if task.status == TaskStatus.RUNNING)
            
            return stats
    
    def get_task_assignments(self) -> Dict[str, List[str]]:
        """
        Get current task assignments.
        
        Returns:
            A dictionary mapping worker IDs to lists of task IDs
        """
        with self.state_lock:
            assignments = {}
            
            for worker_id, worker in self.state.workers.items():
                if worker.current_task_id:
                    assignments[worker_id] = [worker.current_task_id]
                else:
                    assignments[worker_id] = []
            
            return assignments
    
    def _assign_tasks(self) -> int:
        """
        Assign tasks to available workers.
        
        Returns:
            The number of tasks assigned
        """
        with self.state_lock:
            # Find idle workers
            idle_workers = [worker for worker in self.state.workers.values() 
                          if worker.status == WorkerStatus.IDLE and worker.current_task_id is None]
            
            if not idle_workers:
                return 0
            
            assigned_count = 0
            
            # Assign tasks to idle workers
            for worker in idle_workers:
                task = self.task_queue.get_next_task()
                if not task:
                    break
                
                # Check if the worker can handle the task
                if not self._can_worker_handle_task(worker, task):
                    # Put the task back in the queue
                    self.task_queue.add_task(task)
                    continue
                
                # Assign the task to the worker
                task.status = TaskStatus.ASSIGNED
                task.worker_id = worker.id
                task.assigned_time = time.time()
                
                worker.status = WorkerStatus.BUSY
                worker.current_task_id = task.id
                
                assigned_count += 1
                
                logger.info(f"Assigned task {task.id} to worker {worker.id}")
            
            return assigned_count
    
    def _can_worker_handle_task(self, worker: Worker, task: Task) -> bool:
        """
        Check if a worker can handle a task.
        
        Args:
            worker: The worker to check
            task: The task to check
            
        Returns:
            True if the worker can handle the task, False otherwise
        """
        # Check hardware requirements
        if 'hardware_requirements' in task.parameters:
            requirements = task.parameters['hardware_requirements']
            
            for req, value in requirements.items():
                if req not in worker.capabilities:
                    return False
                
                if worker.capabilities[req] < value:
                    return False
        
        # Check software requirements
        if 'software_requirements' in task.parameters:
            requirements = task.parameters['software_requirements']
            
            for req, value in requirements.items():
                if req not in worker.capabilities.get('software', {}):
                    return False
                
                if worker.capabilities.get('software', {}).get(req) \!= value:
                    return False
        
        return True
    
    def _heartbeat_loop(self) -> None:
        """Loop for sending heartbeats to workers."""
        while not self.stop_event.is_set():
            try:
                # In a real implementation, this would send heartbeats to workers
                # through the API server. For this mock implementation, we'll just
                # log the heartbeat.
                with self.state_lock:
                    active_workers = sum(1 for worker in self.state.workers.values() 
                                       if worker.status \!= WorkerStatus.OFFLINE)
                    running_tasks = sum(1 for task in self.state.tasks.values() 
                                       if task.status == TaskStatus.RUNNING)
                    
                logger.debug(f"Heartbeat: {active_workers} active workers, {running_tasks} running tasks")
                
                # Update status
                with self.state_lock:
                    self.state.last_status_update = time.time()
                
                self.stop_event.wait(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                self.stop_event.wait(1)  # Wait a bit before retrying
    
    def _assignment_loop(self) -> None:
        """Loop for assigning tasks to workers."""
        while not self.stop_event.is_set():
            try:
                # Only assign tasks if we're the leader
                with self.state_lock:
                    if self.high_availability and self.state.role \!= NodeRole.LEADER:
                        self.stop_event.wait(1)
                        continue
                
                assigned = self._assign_tasks()
                
                if assigned > 0:
                    logger.debug(f"Assigned {assigned} tasks to workers")
                
                self.stop_event.wait(1)  # Check for new assignments every second
            except Exception as e:
                logger.error(f"Error in assignment loop: {e}")
                self.stop_event.wait(1)  # Wait a bit before retrying
    
    def _cleanup_loop(self) -> None:
        """Loop for cleaning up stale tasks and workers."""
        while not self.stop_event.is_set():
            try:
                with self.state_lock:
                    # Find workers that haven't sent a heartbeat recently
                    now = time.time()
                    stale_workers = [worker for worker in self.state.workers.values() 
                                   if now - worker.last_heartbeat > self.worker_timeout]
                    
                    for worker in stale_workers:
                        logger.warning(f"Worker {worker.id} has not sent a heartbeat in {now - worker.last_heartbeat:.1f} seconds")
                        
                        # Mark the worker as offline
                        worker.status = WorkerStatus.OFFLINE
                        
                        # Reassign the worker's task if it has one
                        if worker.current_task_id:
                            task_id = worker.current_task_id
                            if task_id in self.state.tasks:
                                task = self.state.tasks[task_id]
                                task.status = TaskStatus.PENDING
                                task.worker_id = None
                                self.task_queue.add_task(task)
                                
                                logger.info(f"Reassigned task {task_id} from offline worker {worker.id}")
                            
                            worker.current_task_id = None
                
                self.stop_event.wait(self.worker_timeout)  # Check for stale workers periodically
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                self.stop_event.wait(1)  # Wait a bit before retrying
    
    def _election_loop(self) -> None:
        """Loop for leader election in high availability mode."""
        while not self.stop_event.is_set():
            try:
                with self.state_lock:
                    # In a real implementation, this would implement the Raft
                    # leader election algorithm. For this mock implementation,
                    # we'll just make the current node the leader if there's no leader.
                    if self.state.role == NodeRole.CANDIDATE:
                        self.state.role = NodeRole.LEADER
                        self.state.leader_id = self.id
                        logger.info(f"Node {self.id} elected as leader")
                
                self.stop_event.wait(5)  # Check election status periodically
            except Exception as e:
                logger.error(f"Error in election loop: {e}")
                self.stop_event.wait(1)  # Wait a bit before retrying
    
    def generate_status_report(self) -> Dict[str, Any]:
        """
        Generate a status report.
        
        Returns:
            A dictionary with the status report
        """
        with self.state_lock:
            report = {
                'coordinator': {
                    'id': self.id,
                    'role': self.state.role.name,
                    'uptime': time.time() - self.state.start_time,
                    'term': self.state.term
                },
                'statistics': self.get_statistics(),
                'workers': {
                    worker_id: {
                        'hostname': worker.hostname,
                        'status': worker.status.name,
                        'tasks_completed': worker.total_tasks_completed
                    }
                    for worker_id, worker in self.state.workers.items()
                },
                'tasks': {
                    task_id: {
                        'status': task.status.name,
                        'worker_id': task.worker_id
                    }
                    for task_id, task in self.state.tasks.items()
                    if task.status \!= TaskStatus.COMPLETED  # Only include non-completed tasks
                }
            }
            
            return report
    
    def generate_visualization(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a visualization of the coordinator state.
        
        Args:
            output_path: Optional path to save the visualization to
            
        Returns:
            The path to the saved visualization or None if visualization failed
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization not available. Install matplotlib, seaborn, pandas.")
            return None
        
        try:
            # Get statistics
            with self.state_lock:
                stats = self.get_statistics()
                
                # Get task data
                task_data = []
                for task_id, task in self.state.tasks.items():
                    if task.start_time and task.end_time:
                        duration = task.end_time - task.start_time
                        task_data.append({
                            'id': task_id,
                            'test_path': task.test_path,
                            'status': task.status.name,
                            'duration': duration,
                            'worker_id': task.worker_id
                        })
                
                # Get worker data
                worker_data = []
                for worker_id, worker in self.state.workers.items():
                    worker_data.append({
                        'id': worker_id,
                        'hostname': worker.hostname,
                        'status': worker.status.name,
                        'tasks_completed': worker.total_tasks_completed,
                        'total_execution_time': worker.total_execution_time
                    })
            
            # Create a figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Task status pie chart
            task_status_counts = {
                'Pending': stats.get('tasks_pending', 0),
                'Running': stats.get('tasks_running', 0),
                'Completed': stats.get('tasks_completed', 0),
                'Failed': stats.get('tasks_failed', 0)
            }
            
            labels = list(task_status_counts.keys())
            sizes = list(task_status_counts.values())
            colors = ['#FFC107', '#2196F3', '#4CAF50', '#F44336']
            
            if sum(sizes) > 0:  # Avoid division by zero
                axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Task Status')
                axes[0, 0].axis('equal')
            
            # Plot 2: Worker status pie chart
            worker_status_counts = {
                'Idle': sum(1 for worker in worker_data if worker['status'] == 'IDLE'),
                'Busy': sum(1 for worker in worker_data if worker['status'] == 'BUSY'),
                'Offline': sum(1 for worker in worker_data if worker['status'] == 'OFFLINE')
            }
            
            labels = list(worker_status_counts.keys())
            sizes = list(worker_status_counts.values())
            colors = ['#4CAF50', '#2196F3', '#9E9E9E']
            
            if sum(sizes) > 0:  # Avoid division by zero
                axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Worker Status')
                axes[0, 1].axis('equal')
            
            # Plot 3: Task duration histogram
            if task_data:
                durations = [task['duration'] for task in task_data]
                
                sns.histplot(durations, kde=True, color='#2196F3', ax=axes[1, 0])
                axes[1, 0].set_title('Task Duration Distribution')
                axes[1, 0].set_xlabel('Duration (seconds)')
                axes[1, 0].set_ylabel('Count')
            
            # Plot 4: Worker performance bar chart
            if worker_data:
                worker_hostnames = [worker['hostname'] for worker in worker_data]
                tasks_completed = [worker['tasks_completed'] for worker in worker_data]
                
                # Truncate long hostnames
                worker_hostnames = [name[:20] if len(name) > 20 else name for name in worker_hostnames]
                
                y_pos = np.arange(len(worker_hostnames))
                
                axes[1, 1].barh(y_pos, tasks_completed, color='#673AB7')
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels(worker_hostnames)
                axes[1, 1].invert_yaxis()  # Labels read top-to-bottom
                axes[1, 1].set_title('Worker Performance')
                axes[1, 1].set_xlabel('Tasks Completed')
            
            # Add overall stats as text
            plt.figtext(0.5, 0.01, 
                      f"Total Tasks: {stats.get('tasks_created', 0)} | Completed: {stats.get('tasks_completed', 0)} | "
                      f"Failed: {stats.get('tasks_failed', 0)} | Workers: {stats.get('workers_active', 0)} | "
                      f"Uptime: {stats.get('uptime', 0):.1f} seconds",
                      ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Set title
            plt.suptitle(f"Distributed Testing Coordinator Status\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                        fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Save the figure
            if output_path:
                plt.savefig(output_path, dpi=300)
            else:
                # Generate a default output path
                os.makedirs('visualizations', exist_ok=True)
                output_path = f"visualizations/coordinator_status_{int(time.time())}.png"
                plt.savefig(output_path, dpi=300)
            
            plt.close()
            
            logger.info(f"Visualization saved to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='IPFS Accelerate Distributed Testing Coordinator')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--heartbeat-interval', type=int, default=10, help='Heartbeat interval in seconds')
    parser.add_argument('--worker-timeout', type=int, default=30, help='Worker timeout in seconds')
    parser.add_argument('--high-availability', action='store_true', help='Enable high availability mode')
    
    args = parser.parse_args()
    
    # Create and start the coordinator
    coordinator = TestCoordinator(
        host=args.host,
        port=args.port,
        heartbeat_interval=args.heartbeat_interval,
        worker_timeout=args.worker_timeout,
        high_availability=args.high_availability
    )
    
    try:
        coordinator.start()
        
        # For demo purposes, register some mock workers and create some mock tasks
        if os.environ.get('DEMO_MODE', '0') == '1':
            # Register workers
            coordinator.register_worker('worker1', '127.0.0.1', {'cpu': 4, 'memory': 8, 'software': {'transformers': '4.30.0'}})
            coordinator.register_worker('worker2', '127.0.0.2', {'cpu': 8, 'memory': 16, 'software': {'transformers': '4.30.0'}})
            
            # Create tasks
            coordinator.create_task('test_bert.py', {'batch_size': 8})
            coordinator.create_task('test_vit.py', {'batch_size': 4})
            
            # Generate a visualization
            coordinator.generate_visualization()
        
        # Wait for stop signal
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
    finally:
        coordinator.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
