"""
Test for the distributed testing coordinator.

This test verifies the functionality of the distributed testing framework,
including task distribution, result collection, and worker communication.
"""

import pytest
import os
import sys
import time
import json
import threading
import queue
import uuid
from typing import Dict, List, Any, Set, Optional
from pathlib import Path

# Add the root directory to the Python path
test_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))


class MockCoordinator:
    """Mock implementation of the distributed testing coordinator."""
    
    def __init__(self, worker_count: int = 4):
        self.worker_count = worker_count
        self.tasks = queue.Queue()
        self.results = {}
        self.workers = set()
        self.active_workers = set()
        self.worker_tasks = {}
        self.lock = threading.Lock()
        self.stopped = False
    
    def add_task(self, task_name: str, params: Dict[str, Any]):
        """Add a task to the queue."""
        task_id = str(uuid.uuid4())
        self.tasks.put({
            "task_id": task_id,
            "task_name": task_name,
            "params": params,
            "status": "pending"
        })
        return task_id
    
    def add_batch_tasks(self, tasks: List[Dict[str, Any]]):
        """Add multiple tasks to the queue."""
        task_ids = []
        for task in tasks:
            task_id = self.add_task(task["task_name"], task["params"])
            task_ids.append(task_id)
        return task_ids
    
    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]):
        """Register a worker with the coordinator."""
        with self.lock:
            self.workers.add(worker_id)
            self.active_workers.add(worker_id)
            print(f"Worker {worker_id} registered with capabilities: {capabilities}")
        return {"status": "registered", "worker_id": worker_id}
    
    def get_task(self, worker_id: str):
        """Get a task for the worker to execute."""
        if worker_id not in self.active_workers:
            return {"status": "error", "message": "Worker not registered"}
        
        try:
            task = self.tasks.get(block=False)
            with self.lock:
                self.worker_tasks[worker_id] = task["task_id"]
            return {"status": "task", "task": task}
        except queue.Empty:
            return {"status": "no_task"}
    
    def submit_result(self, worker_id: str, task_id: str, result: Dict[str, Any]):
        """Submit task result from a worker."""
        with self.lock:
            if worker_id not in self.active_workers:
                return {"status": "error", "message": "Worker not registered"}
            
            if worker_id in self.worker_tasks and self.worker_tasks[worker_id] == task_id:
                self.results[task_id] = result
                del self.worker_tasks[worker_id]
                return {"status": "success"}
            else:
                return {"status": "error", "message": "Task not assigned to worker"}
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker from the coordinator."""
        with self.lock:
            if worker_id in self.active_workers:
                self.active_workers.remove(worker_id)
                print(f"Worker {worker_id} unregistered")
            
            # Reassign any tasks from this worker
            if worker_id in self.worker_tasks:
                task_id = self.worker_tasks[worker_id]
                # Re-add the task to the queue
                for task in list(self.tasks.queue):
                    if task["task_id"] == task_id:
                        self.tasks.put(task)
                        break
                del self.worker_tasks[worker_id]
            
            return {"status": "unregistered"}
    
    def stop(self):
        """Stop the coordinator."""
        self.stopped = True
        return {"status": "stopped"}


class MockWorker:
    """Mock implementation of a distributed testing worker."""
    
    def __init__(self, coordinator, worker_id: Optional[str] = None, capabilities: Optional[Dict[str, Any]] = None):
        self.coordinator = coordinator
        self.worker_id = worker_id or str(uuid.uuid4())
        self.capabilities = capabilities or {
            "hardware": {
                "cpu": "mock_cpu",
                "gpu": "mock_gpu",
                "memory": 16384
            },
            "platforms": ["webgpu", "cuda"],
            "browser": {
                "chrome": True,
                "firefox": True
            }
        }
        self.registered = False
        self.stopped = False
        self.tasks_completed = 0
    
    def register(self):
        """Register with the coordinator."""
        response = self.coordinator.register_worker(self.worker_id, self.capabilities)
        self.registered = response["status"] == "registered"
        return self.registered
    
    def unregister(self):
        """Unregister from the coordinator."""
        response = self.coordinator.unregister_worker(self.worker_id)
        self.registered = False
        return response["status"] == "unregistered"
    
    def run(self, max_tasks: int = 5):
        """Run the worker, processing tasks until stopped or max_tasks reached."""
        if not self.registered:
            self.register()
        
        while not self.stopped and self.tasks_completed < max_tasks:
            response = self.coordinator.get_task(self.worker_id)
            
            if response["status"] == "task":
                task = response["task"]
                print(f"Worker {self.worker_id} processing task {task['task_id']}: {task['task_name']}")
                
                # Simulate task execution
                time.sleep(0.1)
                
                # Generate a result
                result = {
                    "status": "success",
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "result": f"Result for {task['task_name']} with params {task['params']}",
                    "worker_id": self.worker_id,
                    "timestamp": time.time()
                }
                
                # Submit the result
                submit_response = self.coordinator.submit_result(
                    self.worker_id, task["task_id"], result
                )
                
                if submit_response["status"] == "success":
                    self.tasks_completed += 1
                
            elif response["status"] == "no_task":
                # No tasks available, wait and try again
                time.sleep(0.1)
            
            elif response["status"] == "error":
                # Error occurred, stop the worker
                print(f"Worker {self.worker_id} error: {response.get('message')}")
                self.stopped = True
        
        if self.registered:
            self.unregister()
        
        return self.tasks_completed


@pytest.fixture
def coordinator():
    """Create a test coordinator."""
    return MockCoordinator(worker_count=4)


@pytest.fixture
def workers(coordinator):
    """Create test workers."""
    workers = []
    for i in range(4):
        capabilities = {
            "hardware": {
                "cpu": f"mock_cpu_{i}",
                "gpu": f"mock_gpu_{i}" if i % 2 == 0 else None,
                "memory": 8192 + i * 4096
            },
            "platforms": ["webgpu", "cuda"] if i % 2 == 0 else ["cpu"],
            "browser": {
                "chrome": i % 2 == 0,
                "firefox": i % 3 == 0
            }
        }
        workers.append(MockWorker(coordinator, f"worker-{i}", capabilities))
    return workers


@pytest.mark.integration
@pytest.mark.distributed
class TestDistributedCoordinator:
    """Test suite for the distributed testing coordinator."""
    
    def test_coordinator_initialization(self, coordinator):
        """Test that the coordinator initializes properly."""
        assert coordinator is not None
        assert coordinator.worker_count == 4
        assert len(coordinator.workers) == 0
        assert coordinator.tasks.empty()
    
    def test_worker_registration(self, coordinator, workers):
        """Test worker registration."""
        # Register all workers
        for worker in workers:
            assert worker.register()
        
        # Check that all workers are registered
        assert len(coordinator.workers) == 4
        assert len(coordinator.active_workers) == 4
    
    def test_task_assignment(self, coordinator, workers):
        """Test task assignment to workers."""
        # Add some tasks
        for i in range(10):
            coordinator.add_task(f"test_task_{i}", {"param1": i, "param2": f"value_{i}"})
        
        # Register all workers
        for worker in workers:
            worker.register()
        
        # Get tasks for workers
        for worker in workers:
            response = coordinator.get_task(worker.worker_id)
            assert response["status"] == "task"
            assert "task_id" in response["task"]
            assert "task_name" in response["task"]
            assert "params" in response["task"]
        
        # Check that tasks were assigned
        assert len(coordinator.worker_tasks) == 4
        assert coordinator.tasks.qsize() == 6  # 10 tasks - 4 assigned
    
    def test_result_submission(self, coordinator, workers):
        """Test result submission from workers."""
        # Add some tasks and register workers
        task_ids = []
        for i in range(4):
            task_id = coordinator.add_task(f"test_task_{i}", {"param": i})
            task_ids.append(task_id)
        
        for worker in workers:
            worker.register()
        
        # Get tasks and submit results
        for i, worker in enumerate(workers):
            response = coordinator.get_task(worker.worker_id)
            assert response["status"] == "task"
            task = response["task"]
            
            # Submit result
            result = {
                "status": "success",
                "result": f"Result for task {i}",
                "worker_id": worker.worker_id
            }
            
            submit_response = coordinator.submit_result(
                worker.worker_id, task["task_id"], result
            )
            assert submit_response["status"] == "success"
        
        # Check results
        assert len(coordinator.results) == 4
    
    def test_worker_execution(self, coordinator, workers):
        """Test worker execution with tasks."""
        # Add tasks
        for i in range(20):
            coordinator.add_task(f"test_task_{i}", {"param": i})
        
        # Start workers in threads
        threads = []
        for worker in workers:
            thread = threading.Thread(target=worker.run, args=(5,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(coordinator.results) == 20
        for worker in workers:
            assert worker.tasks_completed == 5
    
    def test_worker_failure_recovery(self, coordinator, workers):
        """Test recovery from worker failure."""
        # Add tasks
        for i in range(10):
            coordinator.add_task(f"test_task_{i}", {"param": i})
        
        # Register all workers
        for worker in workers:
            worker.register()
        
        # Assign tasks to all workers
        for worker in workers:
            response = coordinator.get_task(worker.worker_id)
            assert response["status"] == "task"
        
        # Simulate one worker failing
        failed_worker = workers[1]
        coordinator.unregister_worker(failed_worker.worker_id)
        
        # Verify the worker was unregistered
        assert failed_worker.worker_id not in coordinator.active_workers
        
        # Register a new worker to replace it
        new_worker = MockWorker(coordinator, "replacement-worker")
        new_worker.register()
        
        # The failed worker's task should be reassigned
        assert coordinator.tasks.qsize() == 7  # 10 - 4 + 1 (reassigned)
        
        # Start remaining workers
        threads = []
        remaining_workers = [w for w in workers if w != failed_worker]
        remaining_workers.append(new_worker)
        
        for worker in remaining_workers:
            thread = threading.Thread(target=worker.run, args=(3,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All tasks should be completed
        assert len(coordinator.results) == 10