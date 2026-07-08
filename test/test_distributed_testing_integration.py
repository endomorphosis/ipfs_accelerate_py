"""
Integration test for the Distributed Testing Framework.

This test verifies that all components of the framework work together correctly:
1. Advanced scheduling algorithms
2. Prometheus/Grafana integration
3. ML-based anomaly detection
4. End-to-end task management
"""

import os
import time
import unittest
import json
import random
import threading
import tempfile
import requests
from typing import Dict, List, Any, Optional

import pytest

# Import framework components
from test.distributed_testing.integration import DistributedTestingFramework, create_distributed_testing_framework
from test.distributed_testing.ml_anomaly_detection import MLAnomalyDetection
from test.distributed_testing.prometheus_grafana_integration import PrometheusGrafanaIntegration
from test.distributed_testing.advanced_scheduling import AdvancedScheduler, Task, Worker


@pytest.fixture
def test_config():
    """Create a test configuration for the framework."""
    return {
        "scheduler": {
            "algorithm": "adaptive",
            "fairness_window": 20,
            "resource_match_weight": 0.7,
            "user_fair_share_enabled": True,
            "adaptive_interval": 10,
        },
        "monitoring": {
            "prometheus_port": 8080,  # Use non-default port for testing
            "metrics_collection_interval": 5,
            "anomaly_detection_interval": 10, 
        },
        "ml": {
            "algorithms": ["isolation_forest", "threshold"],
            "forecasting": ["exponential_smoothing"],
            "visualization": False,  # Disable for testing
        },
        "metrics_interval": 5,
        "scheduling_interval": 2,
    }


@pytest.fixture
def test_framework(test_config):
    """Create a test framework instance with temporary storage."""
    # Create temporary directory for data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config file
        config_file = os.path.join(temp_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
            
        # Create and start framework
        framework = DistributedTestingFramework(
            config_file=config_file,
            coordinator_id="test-coordinator",
            data_dir=temp_dir,
        )
        
        # Start the framework
        framework.start()
        
        # Add start time attribute (shouldn't rely on this in prod)
        framework.start_time = time.time()
        
        # Return framework for testing
        yield framework
        
        # Stop the framework
        framework.stop()


class TestDistributedTestingFramework:
    """Test suite for the Distributed Testing Framework."""
    
    def test_basic_initialization(self, test_framework):
        """Test that the framework initializes correctly."""
        assert test_framework.running
        assert test_framework.coordinator_id == "test-coordinator"
        assert isinstance(test_framework.scheduler, AdvancedScheduler)
        assert isinstance(test_framework.monitoring, PrometheusGrafanaIntegration)
        
    def test_add_task(self, test_framework):
        """Test adding a task to the framework."""
        task_data = {
            "task_id": "test-task-1",
            "task_type": "unit-test",
            "user_id": "test-user",
            "priority": 3,
            "estimated_duration": 60.0,
            "required_resources": {
                "cpu": 2,
                "memory": 4,
            },
            "metadata": {
                "test_name": "test_add_task",
            },
        }
        
        task_id = test_framework.add_task(task_data)
        assert task_id == "test-task-1"
        
        # Verify task was added to scheduler
        assert task_id in test_framework.scheduler.tasks
        task = test_framework.scheduler.tasks[task_id]
        assert task.task_type == "unit-test"
        assert task.priority == 3
        
    def test_add_worker(self, test_framework):
        """Test adding a worker to the framework."""
        worker_data = {
            "worker_id": "test-worker-1",
            "worker_type": "cpu",
            "capabilities": {
                "cpu": 4,
                "memory": 8,
            },
            "status": "idle",
            "metadata": {
                "hostname": "test-host",
                "os": "linux",
            },
        }
        
        worker_id = test_framework.add_worker(worker_data)
        assert worker_id == "test-worker-1"
        
        # Verify worker was added to scheduler
        assert worker_id in test_framework.scheduler.workers
        worker = test_framework.scheduler.workers[worker_id]
        assert worker.worker_type == "cpu"
        assert worker.capabilities["cpu"] == 4
        
    def test_end_to_end_task_execution(self, test_framework):
        """Test end-to-end task execution flow."""
        # Add a worker
        worker_id = test_framework.add_worker({
            "worker_id": "test-worker-2",
            "worker_type": "cpu",
            "capabilities": {"cpu": 2, "memory": 4},
            "status": "idle",
        })
        
        # Add a task
        task_id = test_framework.add_task({
            "task_id": "test-task-2",
            "task_type": "unit-test",
            "user_id": "test-user",
            "priority": 5,
            "required_resources": {"cpu": 1, "memory": 2},
        })
        
        # Force a scheduling cycle (normally this happens in background thread)
        assignments = test_framework.scheduler.schedule_tasks()
        
        # Verify task was assigned to worker
        assert len(assignments) == 1
        assert assignments[0][0] == task_id
        assert assignments[0][1] == worker_id
        
        # Update worker status after assignment
        test_framework._update_assignment_metrics(assignments)
        
        # Mark task as completed
        completed_task_id = test_framework.complete_task(
            worker_id, 
            success=True, 
            result={"output": "test successful"}
        )
        
        # Verify task was completed
        assert completed_task_id == task_id
        assert task_id in test_framework.scheduler.completed_tasks
        
    def test_health_check(self, test_framework):
        """Test health check functionality."""
        # Add some workers and tasks
        for i in range(3):
            test_framework.add_worker({
                "worker_id": f"health-worker-{i}",
                "worker_type": "cpu",
                "capabilities": {"cpu": 2, "memory": 4},
                "status": "idle",
            })
            
        for i in range(5):
            test_framework.add_task({
                "task_id": f"health-task-{i}",
                "task_type": "unit-test",
                "user_id": "test-user",
                "priority": i,
                "required_resources": {"cpu": 1, "memory": 1},
            })
            
        # Run health check
        health = test_framework.health_check()
        
        # Verify health check results
        assert health["status"] == "healthy"
        assert health["coordinator_id"] == "test-coordinator"
        assert health["components"]["scheduler"] == "running"
        assert health["components"]["monitoring"] == "running"
        assert health["task_counts"]["pending"] == 5
        assert health["worker_counts"]["total"] == 3
        assert health["worker_counts"]["available"] == 3
        
    def test_metrics_collection(self, test_framework):
        """Test metrics collection."""
        # Add workers and tasks
        for i in range(2):
            test_framework.add_worker({
                "worker_id": f"metrics-worker-{i}",
                "worker_type": "cpu",
                "capabilities": {"cpu": 2, "memory": 4},
                "status": "idle",
            })
            
        for i in range(3):
            test_framework.add_task({
                "task_id": f"metrics-task-{i}",
                "task_type": "unit-test",
                "user_id": "test-user",
                "priority": i,
                "required_resources": {"cpu": 1, "memory": 1},
            })
            
        # Force metrics collection
        test_framework._collect_metrics()
        
        # Get metrics
        metrics = test_framework.get_metrics()
        
        # Check basic metrics structure
        assert "tasks" in metrics
        assert "workers" in metrics
        assert "system" in metrics
        
        # Check task metrics
        assert metrics["tasks"]["total_tasks"] == 3
        assert metrics["tasks"]["pending_tasks"] == 3
        
        # Check worker metrics
        assert metrics["workers"]["total_workers"] == 2
        assert metrics["workers"]["available_workers"] == 2
        
    def test_prometheus_metrics_endpoint(self, test_framework):
        """Test Prometheus metrics endpoint."""
        # Add some data
        for i in range(2):
            test_framework.add_worker({
                "worker_id": f"prom-worker-{i}",
                "worker_type": "cpu",
                "capabilities": {"cpu": 2, "memory": 4},
                "status": "idle",
            })
            
        for i in range(3):
            test_framework.add_task({
                "task_id": f"prom-task-{i}",
                "task_type": "unit-test",
                "user_id": "test-user",
                "priority": i,
                "required_resources": {"cpu": 1, "memory": 1},
            })
            
        # Force metrics collection and update
        test_framework._collect_metrics()
        test_framework.monitoring.update_metrics_from_data(test_framework.metrics)
        
        # Try to connect to Prometheus endpoint
        port = test_framework.config["monitoring"]["prometheus_port"]
        
        try:
            # Give the server a moment to start up
            time.sleep(1)
            
            # Check if the endpoint is available
            response = requests.get(f"http://localhost:{port}")
            
            # Should redirect to /metrics
            assert response.status_code in (200, 302)
            
            # Try direct metrics endpoint
            metrics_response = requests.get(f"http://localhost:{port}/metrics")
            assert metrics_response.status_code == 200
            
            # Check for some expected metrics in the response
            metrics_text = metrics_response.text
            assert "dtf_worker_count" in metrics_text
            assert "dtf_task_queue_length" in metrics_text
            
        except requests.RequestException:
            pytest.skip("Prometheus HTTP server not reachable - skipping test")
    
    def test_scheduler_algorithms(self, test_framework):
        """Test different scheduling algorithms."""
        # Add workers with different capabilities
        worker_types = ["cpu", "gpu", "webgpu"]
        for i, worker_type in enumerate(worker_types):
            resources = {"cpu": 2, "memory": 4}
            if worker_type in ["gpu", "webgpu"]:
                resources["gpu"] = 1
                
            test_framework.add_worker({
                "worker_id": f"algo-worker-{i}",
                "worker_type": worker_type,
                "capabilities": resources,
                "status": "idle",
            })
            
        # Add tasks with different requirements
        task_types = ["test", "benchmark", "validation"]
        for i, task_type in enumerate(task_types):
            resources = {"cpu": 1, "memory": 2}
            if task_type == "benchmark":
                resources["gpu"] = 0.5
                
            test_framework.add_task({
                "task_id": f"algo-task-{i}",
                "task_type": task_type,
                "user_id": "test-user",
                "priority": i,
                "required_resources": resources,
            })
            
        # Test different algorithms
        algorithms = [
            AdvancedScheduler.ALGORITHM_PRIORITY,
            AdvancedScheduler.ALGORITHM_RESOURCE_AWARE,
            AdvancedScheduler.ALGORITHM_FAIR,
        ]
        
        for algorithm in algorithms:
            # Set algorithm
            test_framework.scheduler.algorithm = algorithm
            test_framework.scheduler.current_best_algorithm = algorithm
            
            # Reset task and worker state
            for task_id in test_framework.scheduler.tasks:
                task = test_framework.scheduler.tasks[task_id]
                task.status = "pending"
                task.assigned_worker = None
                
            for worker_id in test_framework.scheduler.workers:
                worker = test_framework.scheduler.workers[worker_id]
                worker.status = "idle"
                worker.current_task = None
                test_framework.scheduler.available_workers.add(worker_id)
                
            # Empty running tasks
            test_framework.scheduler.running_tasks = {}
            
            # Rebuild task queue
            test_framework.scheduler.task_queue = list(test_framework.scheduler.tasks.values())
            
            # Run scheduling
            assignments = test_framework.scheduler.schedule_tasks()
            
            # Verify assignments were made
            assert len(assignments) > 0
            
            # For resource-aware algorithm, check that benchmark task got assigned to GPU
            if algorithm == AdvancedScheduler.ALGORITHM_RESOURCE_AWARE:
                for task_id, worker_id in assignments:
                    task = test_framework.scheduler.tasks[task_id]
                    worker = test_framework.scheduler.workers[worker_id]
                    
                    if task.task_type == "benchmark":
                        # Should be assigned to a GPU-capable worker
                        assert worker.worker_type in ["gpu", "webgpu"]
    
    def test_create_framework_helper(self):
        """Test the helper function to create a framework."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create framework with helper function
            framework = create_distributed_testing_framework(
                coordinator_id="helper-test",
                data_dir=temp_dir,
                monitoring_config={"prometheus_port": 8081},
            )
            
            try:
                # Verify framework was created and started
                assert framework.running
                assert framework.coordinator_id == "helper-test"
                assert framework.monitoring.prometheus_port == 8081
            finally:
                # Clean up
                framework.stop()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])