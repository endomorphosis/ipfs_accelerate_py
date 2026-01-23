#!/usr/bin/env python3
"""
Integration test for the Dynamic Resource Manager with Coordinator.

This test verifies that the Dynamic Resource Manager properly interacts with the
Coordinator to provision and deprovision workers based on workload.
"""

import anyio
import json
import os
import sys
import time
import unittest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import the components to test
from distributed_testing.dynamic_resource_manager import (
    DynamicResourceManager, 
    ScalingStrategy, 
    ProviderType, 
    ResourceState,
    WorkerTemplate
)
from distributed_testing.coordinator import TestCoordinator


class TestDynamicResourceManagerIntegration(unittest.TestCase):
    """Test integration between Dynamic Resource Manager and Coordinator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Create template configuration
        self.templates_path = os.path.join(self.test_dir.name, "templates.json")
        with open(self.templates_path, "w") as f:
            json.dump({
                "test-template": {
                    "provider": "LOCAL",
                    "instance_type": "test-instance",
                    "image_id": "test-image",
                    "startup_script": "echo 'Starting worker'",
                    "worker_config": {
                        "capabilities": {
                            "cpu": 4,
                            "memory": 8,
                            "software": {"transformers": "4.30.0"}
                        }
                    },
                    "costs": {"hourly": 0.0},
                    "startup_time_estimate": 1  # 1 second for fast testing
                }
            }, f)
        
        # Create provider configuration
        self.provider_config_path = os.path.join(self.test_dir.name, "providers.json")
        with open(self.provider_config_path, "w") as f:
            json.dump({
                "LOCAL": {
                    "regions": {
                        "local": {
                            "enabled": True
                        }
                    }
                }
            }, f)
        
        # Create resource manager configuration
        self.config_path = os.path.join(self.test_dir.name, "config.json")
        with open(self.config_path, "w") as f:
            json.dump({
                "min_workers": 1,
                "max_workers": 3,
                "polling_interval": 1,  # 1 second for fast testing
                "scale_up_cooldown": 2,  # 2 seconds for fast testing
                "scale_down_cooldown": 2,  # 2 seconds for fast testing
                "cpu_threshold_high": 70,
                "cpu_threshold_low": 20,
                "queue_threshold_high": 5,
                "queue_threshold_low": 1,
                "scaling_strategy": "STEPWISE"  # Use stepwise for predictable testing
            }, f)
        
        # Start the coordinator
        self.coordinator_port = 8765  # Use a non-standard port to avoid conflicts
        self.coordinator = TestCoordinator(
            host='localhost',
            port=self.coordinator_port,
            heartbeat_interval=1,
            worker_timeout=5,
            high_availability=False
        )
        self.coordinator.start()
        
        # Allow coordinator to initialize
        time.sleep(1)
        
        # Set up the dynamic resource manager
        self.resource_manager = DynamicResourceManager(
            coordinator_url=f"http://localhost:{self.coordinator_port}",
            config_path=self.config_path,
            templates_path=self.templates_path,
            provider_config_path=self.provider_config_path,
            strategy=ScalingStrategy.STEPWISE
        )
    
    def tearDown(self):
        """Clean up the test environment."""
        # Stop the coordinator
        self.coordinator.stop()
        
        # Clean up temporary directory
        self.test_dir.cleanup()
    
    async def async_setUp(self):
        """Set up async components."""
        # Start the resource manager
        await self.resource_manager.start()
    
    async def async_tearDown(self):
        """Clean up async components."""
        # Stop the resource manager
        await self.resource_manager.stop()

    async def test_resource_manager_provisions_initial_workers(self):
        """Test that the resource manager provisions initial workers."""
        # Set up
        await self.async_setUp()
        
        try:
            # Wait for initial provisioning to complete
            await anyio.sleep(2)
            
            # Check that the resource manager has provisioned the minimum number of workers
            self.assertEqual(
                sum(1 for r in self.resource_manager.resources.values() 
                    if r.state in [ResourceState.RUNNING, ResourceState.INITIALIZING]),
                self.resource_manager.config["min_workers"]
            )
        finally:
            await self.async_tearDown()

    async def test_resource_manager_scales_up_with_high_queue(self):
        """Test that the resource manager scales up when the task queue is high."""
        # Set up
        await self.async_setUp()
        
        try:
            # Wait for initial provisioning to complete
            await anyio.sleep(2)
            
            # Create tasks to fill the queue
            for _ in range(10):
                self.coordinator.create_task("test_task.py", {"param": "value"})
            
            # Mock the statistic response to simulate high queue pressure
            original_get = self.resource_manager.session.get
            
            async def mock_get(url, *args, **kwargs):
                if "/statistics" in url:
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={
                        "tasks_pending": 8,  # High queue value above threshold
                        "workers_active": 1,
                        "tasks_completed": 0,
                        "tasks_failed": 0,
                        "tasks_created": 10,
                        "resource_usage": {
                            "cpu_percent": 50.0,
                            "memory_percent": 40.0
                        }
                    })
                    return mock_response
                elif "/workers" in url:
                    # Return actual worker data
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    workers = []
                    for resource in self.resource_manager.resources.values():
                        if resource.state in [ResourceState.RUNNING, ResourceState.INITIALIZING]:
                            workers.append({
                                "id": resource.resource_id,
                                "instance_id": resource.instance_id,
                                "status": "idle",
                                "hardware_metrics": {
                                    "cpu_percent": 50.0,
                                    "memory_percent": 40.0
                                }
                            })
                    mock_response.json = AsyncMock(return_value={"workers": workers})
                    return mock_response
                else:
                    return await original_get(url, *args, **kwargs)
            
            # Apply the mock
            with patch.object(self.resource_manager.session, 'get', side_effect=mock_get):
                # Wait for scaling to occur (need to account for cooldown)
                await anyio.sleep(4)
                
                # Check that the resource manager has scaled up
                active_workers = sum(1 for r in self.resource_manager.resources.values() 
                                    if r.state in [ResourceState.RUNNING, ResourceState.INITIALIZING])
                
                # Should have scaled up at least one worker
                self.assertGreater(active_workers, self.resource_manager.config["min_workers"])
                
                # Should not exceed maximum workers
                self.assertLessEqual(active_workers, self.resource_manager.config["max_workers"])
        finally:
            await self.async_tearDown()

    async def test_resource_manager_scales_down_with_low_queue(self):
        """Test that the resource manager scales down when the task queue is low."""
        # Set up
        await self.async_setUp()
        
        try:
            # Force provisioning of maximum workers
            self.resource_manager.config["min_workers"] = self.resource_manager.config["max_workers"]
            await self._provision_initial_workers()
            
            # Reset min_workers to original value
            self.resource_manager.config["min_workers"] = 1
            
            # Mock the statistic response to simulate low queue pressure
            original_get = self.resource_manager.session.get
            
            async def mock_get(url, *args, **kwargs):
                if "/statistics" in url:
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={
                        "tasks_pending": 0,  # Low queue value below threshold
                        "workers_active": self.resource_manager.config["max_workers"],
                        "tasks_completed": 5,
                        "tasks_failed": 0,
                        "tasks_created": 5,
                        "resource_usage": {
                            "cpu_percent": 10.0,  # Low CPU usage
                            "memory_percent": 15.0  # Low memory usage
                        }
                    })
                    return mock_response
                elif "/workers" in url:
                    # Return actual worker data
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    workers = []
                    for resource in self.resource_manager.resources.values():
                        if resource.state in [ResourceState.RUNNING, ResourceState.INITIALIZING]:
                            workers.append({
                                "id": resource.resource_id,
                                "instance_id": resource.instance_id,
                                "status": "idle",
                                "hardware_metrics": {
                                    "cpu_percent": 10.0,
                                    "memory_percent": 15.0
                                }
                            })
                    mock_response.json = AsyncMock(return_value={"workers": workers})
                    return mock_response
                else:
                    return await original_get(url, *args, **kwargs)
            
            # Apply the mock
            with patch.object(self.resource_manager.session, 'get', side_effect=mock_get):
                # Wait for scaling to occur (need to account for cooldown)
                await anyio.sleep(4)
                
                # Check that the resource manager has scaled down
                active_workers = sum(1 for r in self.resource_manager.resources.values() 
                                    if r.state in [ResourceState.RUNNING, ResourceState.INITIALIZING])
                
                # Should have scaled down at least one worker
                self.assertLess(active_workers, self.resource_manager.config["max_workers"])
                
                # Should not go below minimum workers
                self.assertGreaterEqual(active_workers, self.resource_manager.config["min_workers"])
        finally:
            await self.async_tearDown()

    async def test_worker_registration_with_coordinator(self):
        """Test that workers provisioned by the resource manager properly register with the coordinator."""
        # Set up
        await self.async_setUp()
        
        try:
            # Wait for initial provisioning to complete
            await anyio.sleep(2)
            
            # Get the number of workers registered with the coordinator
            with patch('distributed_testing.worker.Worker.start', new_callable=AsyncMock) as mock_start:
                # Simulate worker registration
                for resource in self.resource_manager.resources.values():
                    if resource.state in [ResourceState.RUNNING, ResourceState.INITIALIZING]:
                        # Simulate worker registration with coordinator
                        worker_id = self.coordinator.register_worker(
                            hostname=f"worker-{resource.resource_id}",
                            ip_address="127.0.0.1",
                            capabilities={
                                "cpu": 4,
                                "memory": 8,
                                "software": {"transformers": "4.30.0"}
                            }
                        )
                        
                        # Update the worker's status
                        worker = self.coordinator.state.workers[worker_id]
                        worker.status = "idle"
                        worker.last_heartbeat = time.time()
                
                # Wait for the resource manager to update its state
                await anyio.sleep(2)
                
                # Check that the coordinator has the correct number of workers
                active_workers_in_coordinator = sum(1 for worker in self.coordinator.state.workers.values() 
                                                if worker.status == "idle")
                
                active_workers_in_manager = sum(1 for r in self.resource_manager.resources.values() 
                                            if r.state in [ResourceState.RUNNING, ResourceState.INITIALIZING])
                
                # The number should match or be close (allowing for timing differences)
                self.assertGreaterEqual(active_workers_in_coordinator, active_workers_in_manager - 1)
                self.assertLessEqual(active_workers_in_coordinator, active_workers_in_manager + 1)
        finally:
            await self.async_tearDown()

    async def test_anomaly_detection_and_recovery(self):
        """Test that the resource manager detects and recovers from worker anomalies."""
        # Set up
        await self.async_setUp()
        
        try:
            # Wait for initial provisioning to complete
            await anyio.sleep(2)
            
            # Get a worker to simulate anomaly
            test_worker = None
            for resource in self.resource_manager.resources.values():
                if resource.state == ResourceState.RUNNING:
                    test_worker = resource
                    break
            
            if not test_worker:
                self.fail("No running worker found to test anomaly detection")
            
            # Update the worker's metrics to simulate high CPU
            test_worker.metrics.cpu_percent = 98.0
            
            # Mock the statistic response to include the anomalous worker
            original_get = self.resource_manager.session.get
            
            async def mock_get(url, *args, **kwargs):
                if "/statistics" in url:
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(return_value={
                        "tasks_pending": 2,
                        "workers_active": sum(1 for r in self.resource_manager.resources.values() 
                                           if r.state in [ResourceState.RUNNING, ResourceState.INITIALIZING]),
                        "tasks_completed": 5,
                        "tasks_failed": 0,
                        "tasks_created": 7,
                        "resource_usage": {
                            "cpu_percent": 60.0,
                            "memory_percent": 40.0
                        }
                    })
                    return mock_response
                elif "/workers" in url:
                    # Return worker data with the anomaly
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    workers = []
                    for resource in self.resource_manager.resources.values():
                        if resource.state in [ResourceState.RUNNING, ResourceState.INITIALIZING]:
                            cpu_percent = 98.0 if resource.resource_id == test_worker.resource_id else 40.0
                            workers.append({
                                "id": resource.resource_id,
                                "instance_id": resource.instance_id,
                                "status": "idle",
                                "hardware_metrics": {
                                    "cpu_percent": cpu_percent,
                                    "memory_percent": 40.0
                                }
                            })
                    mock_response.json = AsyncMock(return_value={"workers": workers})
                    return mock_response
                else:
                    return await original_get(url, *args, **kwargs)
            
            # Enable anomaly detection
            self.resource_manager.config["enable_anomaly_detection"] = True
            
            # Apply the mock
            with patch.object(self.resource_manager.session, 'get', side_effect=mock_get):
                # Wait for anomaly detection to occur
                await anyio.sleep(3)
                
                # Check if the anomalous worker was detected
                self.assertIn(test_worker.resource_id, self.resource_manager.resources)
                worker_state = self.resource_manager.resources[test_worker.resource_id].state
                
                # The worker should be marked as ERROR or a replacement should be provisioned
                if worker_state == ResourceState.ERROR:
                    # Success case 1: Worker marked as error
                    pass
                else:
                    # Success case 2: A replacement was provisioned
                    # Count total workers to ensure we have at least min_workers healthy ones
                    healthy_workers = sum(1 for r in self.resource_manager.resources.values() 
                                       if r.state == ResourceState.RUNNING and r.resource_id != test_worker.resource_id)
                    self.assertGreaterEqual(healthy_workers, self.resource_manager.config["min_workers"])
        finally:
            await self.async_tearDown()

    async def _provision_initial_workers(self):
        """Helper method to provision initial workers and wait for completion."""
        await self.resource_manager._provision_initial_workers()
        await anyio.sleep(2)  # Wait for provisioning to complete


def run_tests():
    """Run the integration tests."""
    try:
        # Run async tests
        loop = # TODO: Remove event loop management - asyncio.get_event_loop()
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(TestDynamicResourceManagerIntegration('test_resource_manager_provisions_initial_workers'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_resource_manager_scales_up_with_high_queue'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_resource_manager_scales_down_with_low_queue'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_worker_registration_with_coordinator'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_anomaly_detection_and_recovery'))
        
        # Run tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
    except Exception as e:
        print(f"Error running tests: {e}")
    finally:
        # Close the event loop
        loop.close()


if __name__ == "__main__":
    run_tests()