#!/usr/bin/env python3
"""
End-to-End Integration Test for the IPFS Accelerate Distributed Testing Framework.

This test verifies the complete system integration between:
1. Coordinator
2. Dynamic Resource Manager
3. Performance Trend Analyzer
4. Workers

It validates that all components work together properly:
- The coordinator can accept and distribute tasks
- The dynamic resource manager can provision/deprovision workers based on workload
- The performance trend analyzer can collect and analyze metrics
- Workers can execute tasks and report results
"""

import anyio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add the /test directory to the Python path so that the `distributed_testing`
# package resolves to `test/distributed_testing`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from distributed_testing.integration_mode import (
    integration_opt_in_message,
    real_integration_enabled,
    simulated_integration_enabled,
)

if not (real_integration_enabled() or simulated_integration_enabled()):
    pytest.skip(
        "E2E integrated system test requires integration mode; " + integration_opt_in_message(),
        allow_module_level=True,
    )

# Import the components to test
from distributed_testing.coordinator import TestCoordinator
from distributed_testing.dynamic_resource_manager import (
    DynamicResourceManager,
    ScalingStrategy,
    ProviderType,
    ResourceState,
    WorkerTemplate
)
from distributed_testing.performance_trend_analyzer import (
    PerformanceTrendAnalyzer,
    MetricsCollector,
    AnomalyDetector,
    TrendAnalyzer,
    Visualization
)
from distributed_testing.worker import Worker


class TestE2EIntegratedSystem(unittest.TestCase):
    """End-to-end integration test for the complete distributed testing framework."""

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger('e2e_test')

        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        cls.logger.info(f"Created temporary directory: {cls.temp_dir}")

        # Create configuration files
        cls._create_config_files()

    @classmethod
    def tearDownClass(cls):
        """Clean up shared resources."""
        # Remove the temporary directory
        shutil.rmtree(cls.temp_dir)
        cls.logger.info(f"Removed temporary directory: {cls.temp_dir}")

    @classmethod
    def _create_config_files(cls):
        """Create configuration files needed for the tests."""
        # Resource manager config
        resource_manager_config = {
            "min_workers": 2,
            "max_workers": 5,
            "polling_interval": 5,
            "metrics_window": 10,
            "cpu_threshold_high": 80.0,
            "cpu_threshold_low": 20.0,
            "memory_threshold_high": 80.0,
            "memory_threshold_low": 20.0,
            "queue_threshold_high": 5,
            "queue_threshold_low": 1,
            "scale_up_cooldown": 15,
            "scale_down_cooldown": 30,
            "forecast_horizon": 5,
            "worker_startup_buffer": 10,
            "cost_optimization_weight": 0.5,
            "enable_predictive_scaling": True,
            "enable_anomaly_detection": True,
            "preferred_providers": ["LOCAL"],
            "preferred_regions": ["us-east-1"],
            "scaling_strategy": "ADAPTIVE",
            "deprovision_on_stop": True
        }

        # Performance analyzer config
        performance_analyzer_config = {
            "polling_interval": 5,
            "history_window": 60,
            "anomaly_detection": {
                "z_score_threshold": 3.0,
                "isolation_forest": {
                    "enabled": True,
                    "contamination": 0.05,
                    "n_estimators": 100
                },
                "moving_average": {
                    "enabled": True,
                    "window_size": 3,
                    "threshold_factor": 2.0
                }
            },
            "trend_analysis": {
                "minimum_data_points": 5,
                "regression_confidence_threshold": 0.7,
                "trend_classification": {
                    "stable_threshold": 0.05,
                    "improvement_direction": {
                        "latency_ms": "decreasing",
                        "throughput_items_per_second": "increasing",
                        "memory_mb": "decreasing",
                        "cpu_percent": "decreasing",
                        "execution_time_seconds": "decreasing",
                        "task_processing_rate": "increasing"
                    }
                }
            },
            "reporting": {
                "generate_charts": True,
                "alert_thresholds": {
                    "latency_ms": {"warning": 1.5, "critical": 2.0},
                    "throughput_items_per_second": {"warning": 0.75, "critical": 0.5},
                    "memory_mb": {"warning": 1.2, "critical": 1.5},
                    "cpu_percent": {"warning": 1.3, "critical": 1.6},
                    "execution_time_seconds": {"warning": 1.5, "critical": 2.0},
                    "task_processing_rate": {"warning": 0.8, "critical": 0.6}
                },
                "email_alerts": False,
                "email_recipients": []
            },
            "metrics_to_track": [
                "latency_ms",
                "throughput_items_per_second",
                "memory_mb",
                "cpu_percent",
                "execution_time_seconds",
                "task_processing_rate",
                "queue_length"
            ],
            "metrics_grouping": [
                "model_name",
                "worker_id"
            ]
        }

        # Worker templates
        worker_templates = {
            "cpu-worker": {
                "provider": "LOCAL",
                "instance_type": "local",
                "startup_script": "python worker.py --coordinator {coordinator_url} --worker-type cpu",
                "resources": {
                    "cpu": 2,
                    "memory_gb": 4,
                    "disk_gb": 10
                },
                "capabilities": ["CPU", "ONNX"]
            },
            "gpu-worker": {
                "provider": "LOCAL",
                "instance_type": "local",
                "startup_script": "python worker.py --coordinator {coordinator_url} --worker-type gpu",
                "resources": {
                    "cpu": 4,
                    "memory_gb": 16,
                    "disk_gb": 50,
                    "gpu": 1
                },
                "capabilities": ["GPU", "CUDA", "ONNX"]
            }
        }

        # Test tasks
        test_tasks = {
            "tasks": [
                {
                    "type": "test",
                    "name": "CPU Backend Test",
                    "priority": 5,
                    "config": {
                        "test_file": "test_cpu_backend.py",
                        "hardware_requirements": {
                            "cpu": True,
                            "threads": 2
                        }
                    }
                },
                {
                    "type": "benchmark",
                    "name": "Small BERT Benchmark",
                    "priority": 10,
                    "config": {
                        "model": "prajjwal1/bert-tiny",
                        "batch_sizes": [1, 2],
                        "precision": "fp32",
                        "iterations": 3,
                        "hardware_requirements": {
                            "cpu": True
                        }
                    }
                }
            ]
        }

        # Write configurations to files
        cls.resource_manager_config_path = os.path.join(cls.temp_dir, "resource_manager_config.json")
        cls.performance_analyzer_config_path = os.path.join(cls.temp_dir, "performance_analyzer_config.json")
        cls.worker_templates_path = os.path.join(cls.temp_dir, "worker_templates.json")
        cls.test_tasks_path = os.path.join(cls.temp_dir, "test_tasks.json")

        with open(cls.resource_manager_config_path, "w") as f:
            json.dump(resource_manager_config, f, indent=2)
        
        with open(cls.performance_analyzer_config_path, "w") as f:
            json.dump(performance_analyzer_config, f, indent=2)
        
        with open(cls.worker_templates_path, "w") as f:
            json.dump(worker_templates, f, indent=2)
        
        with open(cls.test_tasks_path, "w") as f:
            json.dump(test_tasks, f, indent=2)

    async def _setup_coordinator(self):
        """Set up the coordinator for testing."""
        # Create a test coordinator with mock network interface
        coordinator = TestCoordinator(
            hostname="localhost",
            port=8080,
            work_dir=os.path.join(self.temp_dir, "coordinator"),
            enable_hardware_detection=True,
            enable_security=False
        )
        
        # Mock the network setup to avoid actual socket binding
        coordinator.setup_network = AsyncMock()
        coordinator.accept_connections = AsyncMock()
        
        # Initialize the coordinator
        await coordinator.initialize()
        
        return coordinator

    async def _setup_resource_manager(self, coordinator):
        """Set up the dynamic resource manager for testing."""
        # Create a resource manager with the test configuration
        resource_manager = DynamicResourceManager(
            config_path=self.resource_manager_config_path,
            worker_templates_path=self.worker_templates_path,
            coordinator_url="ws://localhost:8080",
            work_dir=os.path.join(self.temp_dir, "resource_manager")
        )
        
        # Mock the network connection to coordinator
        resource_manager._connect_to_coordinator = AsyncMock()
        resource_manager._coordinator_connection = AsyncMock()
        resource_manager._coordinator_connection.send = AsyncMock()
        resource_manager._coordinator_connection.receive = AsyncMock()
        
        # Connect the resource manager to the coordinator
        resource_manager._coordinator = coordinator
        
        # Initialize the resource manager
        await resource_manager.initialize()
        
        return resource_manager

    async def _setup_performance_analyzer(self, coordinator):
        """Set up the performance trend analyzer for testing."""
        # Create a performance analyzer with the test configuration
        performance_analyzer = PerformanceTrendAnalyzer(
            config_path=self.performance_analyzer_config_path,
            coordinator_url="ws://localhost:8080",
            output_dir=os.path.join(self.temp_dir, "performance_analyzer")
        )
        
        # Mock the network connection to coordinator
        performance_analyzer._connect_to_coordinator = AsyncMock()
        performance_analyzer._coordinator_connection = AsyncMock()
        performance_analyzer._coordinator_connection.send = AsyncMock()
        performance_analyzer._coordinator_connection.receive = AsyncMock()
        
        # Connect the performance analyzer to the coordinator
        performance_analyzer._coordinator = coordinator
        
        # Initialize the performance analyzer
        await performance_analyzer.initialize()
        
        return performance_analyzer

    async def _setup_mock_workers(self, coordinator, count=2):
        """Set up mock workers for testing."""
        workers = []
        for i in range(count):
            worker = Worker(
                coordinator_url="ws://localhost:8080",
                worker_id=f"worker-{i}",
                capabilities=["CPU", "ONNX"] if i % 2 == 0 else ["GPU", "CUDA", "ONNX"],
                work_dir=os.path.join(self.temp_dir, f"worker-{i}")
            )
            
            # Mock the network connection
            worker._connect_to_coordinator = AsyncMock()
            worker._coordinator_connection = AsyncMock()
            worker._coordinator_connection.send = AsyncMock()
            worker._coordinator_connection.receive = AsyncMock()
            
            # Connect the worker to the coordinator
            worker._coordinator = coordinator
            
            # Mock the task execution
            worker.execute_task = AsyncMock(return_value={
                "status": "success",
                "execution_time": 1.5,
                "metrics": {
                    "cpu_percent": 50,
                    "memory_mb": 200,
                    "latency_ms": 150,
                    "throughput_items_per_second": 100
                }
            })
            
            # Register the worker with the coordinator
            coordinator.register_worker(
                worker.worker_id,
                {"hardware": worker.capabilities},
            )
            workers.append(worker)
        
        return workers

    async def _submit_test_tasks(self, coordinator):
        """Submit test tasks to the coordinator."""
        # Load test tasks
        with open(self.test_tasks_path, "r") as f:
            task_data = json.load(f)
        
        # Submit each task to the coordinator
        task_ids = []
        for task in task_data["tasks"]:
            task_id = await coordinator.submit_task(task)
            task_ids.append(task_id)
            self.logger.info(f"Submitted task {task['name']} with ID {task_id}")
        
        return task_ids

    async def _run_integrated_test(self):
        """Run the integrated test with all components."""
        # Set up the coordinator
        self.logger.info("Setting up coordinator...")
        coordinator = await self._setup_coordinator()
        
        # Set up the dynamic resource manager
        self.logger.info("Setting up dynamic resource manager...")
        resource_manager = await self._setup_resource_manager(coordinator)
        
        # Set up the performance trend analyzer
        self.logger.info("Setting up performance trend analyzer...")
        performance_analyzer = await self._setup_performance_analyzer(coordinator)
        
        # Set up mock workers
        self.logger.info("Setting up mock workers...")
        workers = await self._setup_mock_workers(coordinator, count=3)
        
        async with anyio.create_task_group() as tg:
            # Start the components
            self.logger.info("Starting coordinator...")
            tg.start_soon(coordinator.run)
            
            self.logger.info("Starting dynamic resource manager...")
            tg.start_soon(resource_manager.run)
            
            self.logger.info("Starting performance trend analyzer...")
            tg.start_soon(performance_analyzer.run)
            
            # Wait for components to initialize
            await anyio.sleep(1)
            
            # Submit test tasks
            self.logger.info("Submitting test tasks...")
            task_ids = await self._submit_test_tasks(coordinator)
            
            # Give time for tasks to be processed
            self.logger.info("Waiting for tasks to be processed...")
            await anyio.sleep(5)
            
            # Verify that tasks were assigned to workers
            task_assignments = coordinator.get_task_assignments()
            self.logger.info(f"Task assignments: {task_assignments}")
            self.assertTrue(len(task_assignments) > 0, "No tasks were assigned to workers")
            
            # Verify that resource manager scaled appropriately
            worker_states = resource_manager.get_worker_states()
            self.logger.info(f"Worker states: {worker_states}")
            self.assertGreaterEqual(len(worker_states), 2, "Resource manager did not provision enough workers")
            
            # Simulate task completion
            for task_id in task_ids:
                for worker in workers:
                    if task_id in coordinator.get_worker_tasks(worker.worker_id):
                        await coordinator.mark_task_completed(
                            task_id=task_id,
                            worker_id=worker.worker_id,
                            result={
                                "status": "success",
                                "execution_time": 1.5,
                                "metrics": {
                                    "cpu_percent": 50,
                                    "memory_mb": 200,
                                    "latency_ms": 150,
                                    "throughput_items_per_second": 100
                                }
                            }
                        )
            
            # Give time for metrics to be collected
            await anyio.sleep(2)
            
            # Verify that performance analyzer collected metrics
            metrics = performance_analyzer.get_collected_metrics()
            self.logger.info(f"Collected metrics: {metrics}")
            self.assertTrue(len(metrics) > 0, "No metrics were collected by the performance analyzer")
            
            # Submit a large batch of tasks to trigger scaling
            self.logger.info("Submitting large batch of tasks to trigger scaling...")
            high_priority_tasks = []
            for i in range(10):
                task = {
                    "type": "benchmark",
                    "name": f"High Priority Benchmark {i}",
                    "priority": 20,
                    "config": {
                        "model": "prajjwal1/bert-tiny",
                        "batch_sizes": [1],
                        "precision": "fp32",
                        "iterations": 2,
                        "hardware_requirements": {
                            "cpu": True
                        }
                    }
                }
                task_id = await coordinator.submit_task(task)
                high_priority_tasks.append(task_id)
            
            # Give time for resource manager to detect high queue and scale up
            self.logger.info("Waiting for resource manager to scale up...")
            await anyio.sleep(10)
            
            # Verify that resource manager scaled up
            new_worker_states = resource_manager.get_worker_states()
            self.logger.info(f"New worker states after scaling: {new_worker_states}")
            self.assertGreater(len(new_worker_states), len(worker_states), 
                              "Resource manager did not scale up with high queue")
            
            # Complete all high priority tasks
            for task_id in high_priority_tasks:
                for worker in workers:
                    if task_id in coordinator.get_worker_tasks(worker.worker_id):
                        await coordinator.mark_task_completed(
                            task_id=task_id,
                            worker_id=worker.worker_id,
                            result={
                                "status": "success",
                                "execution_time": 0.8,
                                "metrics": {
                                    "cpu_percent": 70,
                                    "memory_mb": 300,
                                    "latency_ms": 100,
                                    "throughput_items_per_second": 150
                                }
                            }
                        )
            
            # Give time for metrics to be collected and resource manager to detect low queue
            self.logger.info("Waiting for resource manager to scale down...")
            await anyio.sleep(20)
            
            # Verify that resource manager eventually scaled down
            final_worker_states = resource_manager.get_worker_states()
            self.logger.info(f"Final worker states after scaling down: {final_worker_states}")
            self.assertLess(len(final_worker_states), len(new_worker_states), 
                           "Resource manager did not scale down with low queue")
            
            # Check for anomalies and trends
            anomalies = performance_analyzer.get_detected_anomalies()
            self.logger.info(f"Detected anomalies: {anomalies}")
            
            trends = performance_analyzer.get_identified_trends()
            self.logger.info(f"Identified trends: {trends}")
            
            # Stop all tasks
            self.logger.info("Stopping all components...")
            tg.cancel_scope.cancel()
            
            self.logger.info("End-to-end test completed successfully")

    def test_e2e_integrated_system(self):
        """Test the complete integrated system end-to-end."""
        anyio.run(self._run_integrated_test)


if __name__ == "__main__":
    unittest.main()