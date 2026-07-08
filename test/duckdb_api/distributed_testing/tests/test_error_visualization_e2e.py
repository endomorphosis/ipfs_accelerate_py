#!/usr/bin/env python3
"""
End-to-End Test for Error Visualization System.

This script tests the Error Visualization integration with real coordinator and worker instances,
ensuring that the system correctly collects, processes, and visualizes error data from distributed
test execution.

Usage:
    python -m duckdb_api.distributed_testing.tests.test_error_visualization_e2e
"""

import os
import sys
import time
import json
import logging
import anyio
import tempfile
import unittest
import random
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.duckdb.distributed_testing.coordinator import Coordinator
from data.duckdb.distributed_testing.dashboard.error_visualization_integration import ErrorVisualizationIntegration
from data.duckdb.distributed_testing.worker_error_reporting import EnhancedErrorReporter
from data.duckdb.distributed_testing.distributed_error_handler import ErrorCategory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_error_visualization_e2e")

class TestErrorVisualizationE2E(unittest.TestCase):
    """End-to-End Tests for the Error Visualization System."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create temporary directory for test output
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = cls.temp_dir.name
        
        # Create test database file
        cls.db_path = os.path.join(cls.output_dir, "test_error_visualization_e2e.duckdb")
        
        # Create directory for coordinator logs
        cls.log_dir = os.path.join(cls.output_dir, "logs")
        os.makedirs(cls.log_dir, exist_ok=True)
        
        # Create error schema
        cls._create_error_schema()
        
        # Start coordinator process
        cls.coordinator_port = 8000 + random.randint(1, 1000)  # Random port to avoid conflicts
        cls.coordinator_url = f"http://localhost:{cls.coordinator_port}"
        cls.coordinator_process = cls._start_coordinator(cls.coordinator_port, cls.log_dir, cls.db_path)
        
        # Give coordinator time to start
        time.sleep(2)
        
        # Start monitoring dashboard with error visualization
        cls.dashboard_port = 9000 + random.randint(1, 1000)  # Random port to avoid conflicts
        cls.dashboard_url = f"http://localhost:{cls.dashboard_port}"
        cls.dashboard_process = cls._start_dashboard(
            cls.dashboard_port, 
            cls.coordinator_url, 
            cls.db_path
        )
        
        # Give dashboard time to start
        time.sleep(2)
        
        # Create worker processes
        cls.worker_count = 3
        cls.worker_processes = []
        cls.worker_ids = []
        
        for i in range(cls.worker_count):
            worker_id = f"test-worker-{i+1}"
            cls.worker_ids.append(worker_id)
            worker_process = cls._start_worker(
                worker_id, 
                cls.coordinator_url, 
                cls.log_dir
            )
            cls.worker_processes.append(worker_process)
        
        # Give workers time to connect
        time.sleep(3)
        
        logger.info("Test environment setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Stop worker processes
        for process in cls.worker_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Stop dashboard process
        if cls.dashboard_process:
            try:
                cls.dashboard_process.terminate()
                cls.dashboard_process.wait(timeout=5)
            except:
                cls.dashboard_process.kill()
        
        # Stop coordinator process
        if cls.coordinator_process:
            try:
                cls.coordinator_process.terminate()
                cls.coordinator_process.wait(timeout=5)
            except:
                cls.coordinator_process.kill()
        
        # Clean up temporary directory
        cls.temp_dir.cleanup()
        
        logger.info("Test environment cleanup complete")
    
    @classmethod
    def _create_error_schema(cls):
        """Create the error reporting schema in the test database."""
        import duckdb
        
        # Get schema SQL
        schema_path = Path(__file__).parent.parent / "schema" / "error_reporting_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Create database and schema
        conn = duckdb.connect(cls.db_path)
        conn.execute(schema_sql)
        conn.close()
        
        logger.info(f"Created error schema in {cls.db_path}")
    
    @classmethod
    def _start_coordinator(cls, port: int, log_dir: str, db_path: str) -> subprocess.Popen:
        """Start a coordinator process for testing."""
        cmd = [
            sys.executable,
            "-m",
            "duckdb_api.distributed_testing.run_coordinator_server",
            "--host", "localhost",
            "--port", str(port),
            "--db-path", db_path,
            "--log-dir", log_dir
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Started coordinator on port {port}")
        return process
    
    @classmethod
    def _start_dashboard(cls, port: int, coordinator_url: str, db_path: str) -> subprocess.Popen:
        """Start a monitoring dashboard with error visualization for testing."""
        cmd = [
            sys.executable,
            "-m",
            "duckdb_api.distributed_testing.run_monitoring_dashboard_with_error_visualization",
            "--host", "localhost",
            "--port", str(port),
            "--coordinator-url", coordinator_url,
            "--db-path", db_path,
            "--dashboard-dir", cls.output_dir
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Started dashboard on port {port}")
        return process
    
    @classmethod
    def _start_worker(cls, worker_id: str, coordinator_url: str, log_dir: str) -> subprocess.Popen:
        """Start a worker process for testing."""
        cmd = [
            sys.executable,
            "-m",
            "duckdb_api.distributed_testing.run_enhanced_worker_with_error_reporting",
            "--worker-id", worker_id,
            "--coordinator-url", coordinator_url,
            "--log-dir", log_dir,
            "--hardware-types", "cpu,cuda",
            "--error-injection-rate", "0.3",  # Inject errors at a moderate rate
            "--error-types", "resource,network,hardware,worker,test"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Started worker {worker_id}")
        return process
    
    async def _submit_test_tasks(self, task_count: int) -> List[str]:
        """Submit test tasks to the coordinator."""
        import aiohttp
        task_ids = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(task_count):
                # Create a task
                task_data = {
                    "type": "test",
                    "model": f"test-model-{i%5+1}",
                    "hardware_requirements": ["cpu"] if i % 2 == 0 else ["cuda"],
                    "priority": i % 3 + 1,
                    "test_params": {
                        "batch_size": i % 4 + 1,
                        "sequence_length": (i % 3 + 1) * 128,
                        "iterations": 5
                    }
                }
                
                # Submit the task
                async with session.post(
                    f"{self.coordinator_url}/api/tasks",
                    json=task_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_ids.append(result.get("task_id"))
                    else:
                        logger.error(f"Error submitting task: {await response.text()}")
        
        logger.info(f"Submitted {len(task_ids)} test tasks")
        return task_ids
    
    async def _wait_for_task_completion(self, task_ids: List[str], timeout_seconds: int = 30) -> None:
        """Wait for tasks to complete."""
        import aiohttp
        start_time = time.time()
        completed_tasks = set()
        
        async with aiohttp.ClientSession() as session:
            while (time.time() - start_time) < timeout_seconds and len(completed_tasks) < len(task_ids):
                for task_id in task_ids:
                    if task_id in completed_tasks:
                        continue
                    
                    async with session.get(f"{self.coordinator_url}/api/tasks/{task_id}") as response:
                        if response.status == 200:
                            task_info = await response.json()
                            if task_info.get("status") in ["completed", "failed", "error"]:
                                completed_tasks.add(task_id)
                
                # Still tasks to complete
                if len(completed_tasks) < len(task_ids):
                    await anyio.sleep(1)
        
        logger.info(f"Completed {len(completed_tasks)} of {len(task_ids)} tasks")
    
    async def _get_error_data_from_api(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """Get error data from the dashboard API."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.dashboard_url}/api/errors?time_range={time_range_hours}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error getting error data: {await response.text()}")
                    return {}
    
    async def _get_error_count_from_db(self) -> int:
        """Get error count from the database."""
        import duckdb
        
        conn = duckdb.connect(self.db_path)
        result = conn.execute("SELECT COUNT(*) FROM worker_error_reports").fetchone()
        conn.close()
        
        return result[0] if result else 0
    
    def test_01_error_collection(self):
        anyio.run(self._test_01_error_collection)

    async def _test_01_error_collection(self):
        """Test that errors are collected during task execution."""
        # Submit tasks to generate errors
        task_count = 20
        task_ids = await self._submit_test_tasks(task_count)
        
        # Wait for tasks to complete
        await self._wait_for_task_completion(task_ids, timeout_seconds=30)
        
        # Check that errors were recorded in the database
        error_count = await self._get_error_count_from_db()
        
        # We should have some errors due to error injection
        self.assertGreater(error_count, 0, "No errors were recorded in the database")
        
        logger.info(f"Test 01: Found {error_count} errors in the database")
    
    def test_02_error_visualization_api(self):
        anyio.run(self._test_02_error_visualization_api)

    async def _test_02_error_visualization_api(self):
        """Test that error data is available through the API."""
        # Get error data from API
        error_data = await self._get_error_data_from_api(time_range_hours=1)
        
        # Verify structure of error data
        self.assertIn("summary", error_data, "Error data missing summary section")
        self.assertIn("timestamp", error_data, "Error data missing timestamp")
        self.assertIn("recent_errors", error_data, "Error data missing recent errors")
        self.assertIn("error_distribution", error_data, "Error data missing error distribution")
        self.assertIn("error_patterns", error_data, "Error data missing error patterns")
        
        # Verify error summary
        summary = error_data["summary"]
        self.assertIn("total_errors", summary, "Summary missing total_errors")
        self.assertGreater(summary["total_errors"], 0, "Summary shows no errors")
        
        logger.info(f"Test 02: API returned {summary['total_errors']} total errors")
    
    def test_03_error_pattern_detection(self):
        anyio.run(self._test_03_error_pattern_detection)

    async def _test_03_error_pattern_detection(self):
        """Test that error patterns are correctly detected."""
        # Get error data from API
        error_data = await self._get_error_data_from_api(time_range_hours=1)
        
        # Verify error patterns
        self.assertIn("error_patterns", error_data, "Error data missing error patterns")
        patterns = error_data["error_patterns"]
        
        self.assertIn("top_patterns", patterns, "Error patterns missing top_patterns")
        self.assertGreater(len(patterns["top_patterns"]), 0, "No error patterns detected")
        
        logger.info(f"Test 03: Detected {len(patterns['top_patterns'])} error patterns")
    
    def test_04_worker_error_analysis(self):
        anyio.run(self._test_04_worker_error_analysis)

    async def _test_04_worker_error_analysis(self):
        """Test worker error analysis."""
        # Get error data from API
        error_data = await self._get_error_data_from_api(time_range_hours=1)
        
        # Verify worker error analysis
        self.assertIn("worker_errors", error_data, "Error data missing worker_errors")
        worker_errors = error_data["worker_errors"]
        
        self.assertIn("worker_stats", worker_errors, "Worker errors missing worker_stats")
        worker_stats = worker_errors["worker_stats"]
        
        # Check that we have stats for all workers
        worker_ids_in_stats = {w["worker_id"] for w in worker_stats}
        for worker_id in self.worker_ids:
            self.assertIn(worker_id, worker_ids_in_stats, f"Worker {worker_id} missing from stats")
        
        logger.info(f"Test 04: Found statistics for {len(worker_stats)} workers")
    
    def test_05_hardware_error_analysis(self):
        anyio.run(self._test_05_hardware_error_analysis)

    async def _test_05_hardware_error_analysis(self):
        """Test hardware error analysis."""
        # Get error data from API
        error_data = await self._get_error_data_from_api(time_range_hours=1)
        
        # Verify hardware error analysis
        self.assertIn("hardware_errors", error_data, "Error data missing hardware_errors")
        hardware_errors = error_data["hardware_errors"]
        
        self.assertIn("hardware_status", hardware_errors, "Hardware errors missing hardware_status")
        hardware_status = hardware_errors["hardware_status"]
        
        # Check that we have stats for hardware types
        self.assertGreater(len(hardware_status), 0, "No hardware status information")
        
        # Should include at least CPU since all workers support it
        self.assertIn("cpu", hardware_status, "CPU hardware type missing from status")
        
        logger.info(f"Test 05: Found status for {len(hardware_status)} hardware types")
    
    def test_06_different_time_ranges(self):
        anyio.run(self._test_06_different_time_ranges)

    async def _test_06_different_time_ranges(self):
        """Test error data with different time ranges."""
        # Get error data for different time ranges
        data_1h = await self._get_error_data_from_api(time_range_hours=1)
        data_6h = await self._get_error_data_from_api(time_range_hours=6)
        data_24h = await self._get_error_data_from_api(time_range_hours=24)
        
        # All should have data
        self.assertTrue(data_1h["summary"], "1-hour data missing summary")
        self.assertTrue(data_6h["summary"], "6-hour data missing summary")
        self.assertTrue(data_24h["summary"], "24-hour data missing summary")
        
        # Counts should be consistent (24h >= 6h >= 1h)
        count_1h = data_1h["summary"]["total_errors"]
        count_6h = data_6h["summary"]["total_errors"]
        count_24h = data_24h["summary"]["total_errors"]
        
        self.assertGreaterEqual(count_6h, count_1h, "6-hour count should be >= 1-hour count")
        self.assertGreaterEqual(count_24h, count_6h, "24-hour count should be >= 6-hour count")
        
        logger.info(f"Test 06: Error counts: 1h={count_1h}, 6h={count_6h}, 24h={count_24h}")

    def test_07_error_classification(self):
        anyio.run(self._test_07_error_classification)

    async def _test_07_error_classification(self):
        """Test error classification (recurring, critical)."""
        # Submit more tasks to generate additional errors
        task_count = 10
        task_ids = await self._submit_test_tasks(task_count)
        await self._wait_for_task_completion(task_ids, timeout_seconds=30)
        
        # Get error data
        error_data = await self._get_error_data_from_api(time_range_hours=1)
        
        # Check for recurring and critical errors
        recent_errors = error_data["recent_errors"]
        self.assertGreater(len(recent_errors), 0, "No recent errors found")
        
        # Count recurring and critical errors
        recurring_count = sum(1 for e in recent_errors if e.get("is_recurring"))
        critical_count = sum(1 for e in recent_errors if e.get("is_critical"))
        
        # We should have some recurring errors due to error injection
        self.assertGreater(recurring_count, 0, "No recurring errors detected")
        
        logger.info(f"Test 07: Found {recurring_count} recurring errors and {critical_count} critical errors")

if __name__ == "__main__":
    unittest.main()