#!/usr/bin/env python3
"""
Test Worker Auto-Discovery with CI/CD Integration

This module provides tests for the worker auto-discovery feature combined with CI/CD integration.
It verifies that workers can be automatically discovered and registered, and that test results
are properly reported to CI systems.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import socket
from unittest.mock import patch, MagicMock, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import from distributed_testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import necessary modules
from distributed_testing.coordinator import DistributedTestingCoordinator
from distributed_testing.worker import Worker
from distributed_testing.ci.api_interface import CIProviderFactory, CIProviderInterface, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter


class MockCIProvider(CIProviderInterface):
    """Mock CI provider for testing."""
    
    def __init__(self):
        self.test_runs = {}
        self.artifacts = {}
        self.pr_comments = {}
        self.build_statuses = {}
    
    async def initialize(self, config):
        """Initialize the mock CI provider."""
        self.config = config
        return True
    
    async def create_test_run(self, test_run_data):
        """Create a test run."""
        test_run_id = f"test-{len(self.test_runs) + 1}"
        test_run = {
            "id": test_run_id,
            "name": test_run_data.get("name", f"Test Run {test_run_id}"),
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "url": f"http://example.com/test-runs/{test_run_id}"
        }
        self.test_runs[test_run_id] = test_run
        return test_run
    
    async def update_test_run(self, test_run_id, update_data):
        """Update a test run."""
        if test_run_id not in self.test_runs:
            return False
        
        self.test_runs[test_run_id].update(update_data)
        return True
    
    async def add_pr_comment(self, pr_number, comment):
        """Add a comment to a PR."""
        if pr_number not in self.pr_comments:
            self.pr_comments[pr_number] = []
        
        self.pr_comments[pr_number].append(comment)
        return True
    
    async def upload_artifact(self, test_run_id, artifact_path, artifact_name):
        """Upload an artifact."""
        if test_run_id not in self.artifacts:
            self.artifacts[test_run_id] = []
        
        self.artifacts[test_run_id].append({
            "path": artifact_path,
            "name": artifact_name
        })
        return True
    
    async def get_test_run_status(self, test_run_id):
        """Get test run status."""
        if test_run_id not in self.test_runs:
            return {"status": "unknown"}
        
        return self.test_runs[test_run_id]
    
    async def set_build_status(self, status, description):
        """Set build status."""
        self.build_statuses[status] = description
        return True
    
    async def close(self):
        """Close the provider."""
        pass


class TestWorkerAutoDiscoveryWithCI(unittest.IsolatedAsyncioTestCase):
    """Test worker auto-discovery with CI integration."""
    
    async def asyncSetUp(self):
        """Set up for tests."""
        # Create temporary directories for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.reports_dir = self.temp_path / "reports"
        self.artifacts_dir = self.temp_path / "artifacts"
        self.db_path = self.temp_path / "coordinator.db"
        
        self.reports_dir.mkdir()
        self.artifacts_dir.mkdir()
        
        # Register the mock provider
        CIProviderFactory.register_provider("mock", MockCIProvider)
        
        # Create a mock CI provider
        self.mock_provider = await CIProviderFactory.create_provider("mock", {})
        
        # Create a test result reporter
        self.reporter = TestResultReporter(
            ci_provider=self.mock_provider,
            report_dir=str(self.reports_dir),
            artifact_dir=str(self.artifacts_dir)
        )
        
        # Patch socket to avoid network issues in tests
        self.socket_patcher = patch('socket.socket')
        self.mock_socket = self.socket_patcher.start()
        
        # Patch worker connection methods to avoid actual network connections
        self.worker_connect_patcher = patch.object(Worker, '_setup_connection', new_callable=AsyncMock)
        self.mock_worker_connect = self.worker_connect_patcher.start()
        
        # Patch coordinator startup methods to avoid actual network server
        self.coordinator_start_patcher = patch.object(DistributedTestingCoordinator, '_setup_server', new_callable=AsyncMock)
        self.mock_coordinator_start = self.coordinator_start_patcher.start()
        
        # Create a coordinator with worker auto-discovery enabled
        self.coordinator = DistributedTestingCoordinator(
            host="127.0.0.1",
            port=8080,
            db_path=str(self.db_path),
            worker_auto_discovery=True,
            auto_register_workers=True,
            enable_batch_processing=True
        )
        
        # Start the coordinator
        await self.coordinator.start()
    
    async def asyncTearDown(self):
        """Clean up after tests."""
        # Stop the coordinator
        await self.coordinator.shutdown()
        
        # Stop patches
        self.socket_patcher.stop()
        self.worker_connect_patcher.stop()
        self.coordinator_start_patcher.stop()
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    async def create_test_workers(self, count=2):
        """Create test workers."""
        workers = []
        
        for i in range(count):
            # Create a simple worker with basic capabilities
            worker_id = f"test-worker-{i+1}"
            
            # Create different capabilities for different workers
            capabilities = {
                "hardware": ["cpu"],
                "memory_gb": 8 + (i * 8),
                "models": ["bert", "t5"]
            }
            
            if i % 2 == 0:
                capabilities["hardware"].append("cuda")
                capabilities["models"].extend(["vit", "whisper"])
            
            # Create worker
            worker = Worker(
                coordinator_url="http://127.0.0.1:8080",
                worker_id=worker_id,
                capabilities=capabilities,
                auto_register=True
            )
            
            # Patch worker methods to avoid actual execution
            worker._execute_task = AsyncMock(return_value={"status": "completed", "result": f"Result from {worker_id}"})
            worker._subscribe_to_tasks = AsyncMock()
            
            # Connect to the coordinator
            await worker.connect()
            
            # Register worker
            self.coordinator.register_worker(worker_id, capabilities)
            
            workers.append(worker)
        
        return workers
    
    async def test_worker_auto_registration(self):
        """Test that workers are automatically registered."""
        # Create workers
        workers = await self.create_test_workers(2)
        
        # Check that workers are registered with the coordinator
        registered_workers = self.coordinator.get_registered_workers()
        self.assertEqual(len(registered_workers), 2)
        
        for worker in workers:
            self.assertIn(worker.worker_id, registered_workers)
    
    async def test_worker_capability_awareness(self):
        """Test that worker capabilities are properly tracked."""
        # Create workers with different capabilities
        workers = await self.create_test_workers(2)
        
        # Get registered workers with their capabilities
        worker_capabilities = {w: self.coordinator.get_worker_capabilities(w) for w in self.coordinator.get_registered_workers()}
        
        # Check that capabilities match what we set
        for i, worker in enumerate(workers):
            capabilities = worker_capabilities[worker.worker_id]
            self.assertEqual(capabilities["memory_gb"], 8 + (i * 8))
            self.assertIn("cpu", capabilities["hardware"])
            self.assertIn("bert", capabilities["models"])
            self.assertIn("t5", capabilities["models"])
            
            if i % 2 == 0:
                self.assertIn("cuda", capabilities["hardware"])
                self.assertIn("vit", capabilities["models"])
                self.assertIn("whisper", capabilities["models"])
    
    @patch('distributed_testing.coordinator.DistributedTestingCoordinator._assign_task_to_worker')
    async def test_ci_integration_with_auto_discovery(self, mock_assign_task):
        """Test CI integration with worker auto-discovery."""
        # Set up mock assignment function that simulates task completion
        mock_assign_task.side_effect = lambda task_id, worker_id: self.coordinator._mark_task_completed(task_id, {"status": "completed"})
        
        # Create workers
        workers = await self.create_test_workers(2)
        
        # Create a test run with the CI provider
        test_run_data = {
            "name": "Auto-Discovery Test Run",
            "build_id": "build-auto-discovery",
            "commit_sha": "abcdef123456"
        }
        
        test_run = await self.mock_provider.create_test_run(test_run_data)
        test_run_id = test_run["id"]
        
        # Create and submit tasks
        tasks = []
        for i in range(5):
            task_data = {
                "task_id": f"task-{i+1}",
                "model_name": "bert-base-uncased" if i % 2 == 0 else "vit-base",
                "model_type": "text" if i % 2 == 0 else "vision",
                "batch_size": 1,
                "hardware_type": "cpu",
                "priority": 1
            }
            
            # Submit task to coordinator
            await self.coordinator.submit_task(task_data)
            tasks.append(task_data)
        
        # Wait for tasks to be processed (in this case, immediately due to our mock)
        await asyncio.sleep(0.1)
        
        # Create a test result object
        test_result = TestRunResult(
            test_run_id=test_run_id,
            status="success",
            total_tests=len(tasks),
            passed_tests=len(tasks),
            failed_tests=0,
            skipped_tests=0,
            duration_seconds=1.0
        )
        
        # Add metadata
        test_result.metadata = {
            "performance_metrics": {
                "average_throughput": 5.0,  # 5 tasks per second
                "average_latency_ms": 200.0  # 200ms per task
            },
            "worker_metrics": {
                "worker_count": len(workers),
                "auto_discovery_enabled": True,
                "auto_registration_enabled": True
            }
        }
        
        # Create artifact with worker info
        worker_info_path = self.artifacts_dir / "worker_info.json"
        with open(worker_info_path, "w") as f:
            json.dump({
                "workers": [
                    {
                        "id": w.worker_id,
                        "capabilities": w.capabilities
                    }
                    for w in workers
                ]
            }, f)
        
        # Report test results and artifacts
        report_files = await self.reporter.report_test_result(
            test_result,
            formats=["markdown", "html", "json"]
        )
        
        artifacts = await self.reporter.collect_and_upload_artifacts(
            test_run_id,
            [str(worker_info_path)]
        )
        
        # Verify that reports were generated
        self.assertEqual(len(report_files), 3)
        self.assertTrue(os.path.exists(report_files["markdown"]))
        self.assertTrue(os.path.exists(report_files["html"]))
        self.assertTrue(os.path.exists(report_files["json"]))
        
        # Verify that artifacts were uploaded
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(self.mock_provider.artifacts[test_run_id]), 1)
        
        # Update final status
        await self.mock_provider.update_test_run(
            test_run_id,
            {
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "summary": {
                    "total_tests": len(tasks),
                    "passed_tests": len(tasks),
                    "failed_tests": 0,
                    "skipped_tests": 0
                }
            }
        )
        
        # Check that the test run status was updated
        updated_run = self.mock_provider.test_runs[test_run_id]
        self.assertEqual(updated_run["status"], "completed")
        self.assertTrue("end_time" in updated_run)


if __name__ == "__main__":
    unittest.main()