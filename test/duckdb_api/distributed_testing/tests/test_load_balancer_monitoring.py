#!/usr/bin/env python3
"""
End-to-End Integration Test for Load Balancer Monitoring

This module tests the complete integration of the Load Balancer Monitoring
Dashboard with the Distributed Testing Framework. It verifies that metrics
are properly collected, stored, and displayed in the dashboard.
"""

import os
import sys
import json
import time
import unittest
import tempfile
import threading
import anyio
import requests
import websocket
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from duckdb_api.distributed_testing.coordinator import CoordinatorServer
from duckdb_api.distributed_testing.coordinator_load_balancer_integration import CoordinatorLoadBalancerIntegration
from duckdb_api.distributed_testing.load_balancer.monitoring.integration import MonitoringIntegration
from duckdb_api.distributed_testing.run_test import generate_security_config


class LoadBalancerMonitoringIntegrationTest(unittest.TestCase):
    """
    End-to-End Integration Tests for Load Balancer Monitoring Dashboard.
    
    This test suite validates that the monitoring dashboard properly integrates
    with the load balancer and coordinator, collecting metrics and displaying
    them correctly.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with coordinator, load balancer, dashboard, and workers."""
        print("\nSetting up load balancer monitoring test environment...")
        
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {cls.temp_dir}")
        
        # Create a temporary database file
        cls.db_path = os.path.join(cls.temp_dir, "test_db.duckdb")
        cls.metrics_db_path = os.path.join(cls.temp_dir, "metrics.duckdb")
        print(f"Using database path: {cls.db_path}")
        print(f"Using metrics database path: {cls.metrics_db_path}")
        
        # Create security configuration
        cls.security_config_path = os.path.join(cls.temp_dir, "security_config.json")
        cls.security_config = generate_security_config(cls.security_config_path)
        cls.worker_api_key = cls.security_config["api_keys"]["worker"]
        print(f"Created security configuration with worker API key")
        
        # Set up coordinator
        cls.coordinator_host = "localhost"
        cls.coordinator_port = 8888  # Use non-standard port for testing
        cls.coordinator_url = f"ws://{cls.coordinator_host}:{cls.coordinator_port}"
        
        # Set up dashboard
        cls.dashboard_host = "localhost"
        cls.dashboard_port = 5555  # Different port for dashboard
        cls.dashboard_url = f"http://{cls.dashboard_host}:{cls.dashboard_port}"
        
        # Start the coordinator with load balancer in a separate thread
        cls.coordinator_started = anyio.Event()
        cls.coordinator_stopped = anyio.Event()
        cls.coordinator_thread = threading.Thread(
            target=cls._run_coordinator_with_load_balancer,
            daemon=True
        )
        cls.coordinator_thread.start()
        
        # Wait for coordinator to start
        timeout = 10
        start_time = time.time()
        while not cls.coordinator_started.is_set() and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        
        if not cls.coordinator_started.is_set():
            raise TimeoutError("Coordinator failed to start within timeout")
        
        print(f"Coordinator with load balancer started at {cls.coordinator_url}")
        
        # Start workers
        cls.worker_processes = []
        cls._start_workers(3)  # Start 3 workers with different capabilities
        
        # Wait for workers to connect and register
        time.sleep(5)
        print("Setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        print("\nCleaning up load balancer monitoring test environment...")
        
        # Stop all worker processes
        for worker_process in cls.worker_processes:
            print(f"Terminating worker process {worker_process.pid}")
            worker_process.terminate()
            worker_process.wait()
        
        # Stop the coordinator
        print("Stopping coordinator with load balancer")
        cls.coordinator_stopped.set()
        cls.coordinator_thread.join(timeout=5.0)
        
        # Clean up temporary directory
        if os.path.exists(cls.temp_dir):
            for file in os.listdir(cls.temp_dir):
                os.remove(os.path.join(cls.temp_dir, file))
            os.rmdir(cls.temp_dir)
            print(f"Removed temporary directory: {cls.temp_dir}")
    
    @classmethod
    def _run_coordinator_with_load_balancer(cls):
        """Run the coordinator with load balancer in a separate thread."""
        try:
            # Setup the event loop
            loop = # TODO: Remove event loop management - asyncio.new_event_loop()
            # TODO: Remove event loop management - asyncio.set_event_loop(loop)
            
            # Create coordinator with load balancer
            cls.coordinator = CoordinatorServer(
                host=cls.coordinator_host,
                port=cls.coordinator_port,
                db_path=cls.db_path,
                token_secret=cls.security_config["token_secret"],
                heartbeat_timeout=5,  # Short timeout for testing
                enable_load_balancer=True  # Enable load balancer
            )
            
            # Create monitoring integration
            cls.monitoring = MonitoringIntegration(
                coordinator=cls.coordinator,
                load_balancer=cls.coordinator.load_balancer,
                db_path=cls.metrics_db_path,
                dashboard_host=cls.dashboard_host,
                dashboard_port=cls.dashboard_port,
                collection_interval=1.0  # Collect metrics every second
            )
            
            # Start monitoring
            cls.monitoring.start()
            
            # Create helper function to know when coordinator is ready
            async def on_coordinator_start():
                cls.coordinator_started.set()
                await cls.coordinator_stopped.wait()
                # Stop monitoring integration
                cls.monitoring.stop()
                # Stop coordinator
                await cls.coordinator.stop()
            
            # Run both tasks: the coordinator and the signal handler
            loop.create_task(cls.coordinator.start())
            loop.run_until_complete(on_coordinator_start())
            
            # Cleanup
            loop.close()
        except Exception as e:
            print(f"Error in coordinator thread: {e}")
            import traceback
            traceback.print_exc()
    
    @classmethod
    def _start_workers(cls, count=3):
        """Start multiple worker processes with different capabilities."""
        for i in range(count):
            worker_id = f"test_worker_{i}"
            worker_dir = os.path.join(cls.temp_dir, f"worker_{i}")
            os.makedirs(worker_dir, exist_ok=True)
            
            # Add different capabilities for different workers
            capabilities = {}
            if i == 0:
                # Basic CPU worker
                capabilities = {
                    "hardware_types": ["cpu"]
                }
            elif i == 1:
                # CPU + GPU worker
                capabilities = {
                    "hardware_types": ["cpu", "cuda"],
                    "cuda_compute": 7.5,
                    "memory_gb": 16
                }
            else:
                # CPU + WebGPU (browser) worker
                capabilities = {
                    "hardware_types": ["cpu", "webgpu"],
                    "browsers": ["chrome", "firefox"],
                    "memory_gb": 8
                }
            
            capabilities_json = json.dumps(capabilities)
            
            worker_cmd = [
                sys.executable,
                os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
                "--coordinator", cls.coordinator_url,
                "--api-key", cls.worker_api_key,
                "--worker-id", worker_id,
                "--work-dir", worker_dir,
                "--reconnect-interval", "2",
                "--heartbeat-interval", "3",
                "--capabilities", capabilities_json
            ]
            
            print(f"Starting worker {worker_id} with capabilities: {capabilities}")
            process = subprocess.Popen(worker_cmd)
            cls.worker_processes.append(process)
    
    def test_01_monitoring_initialization(self):
        """Test that monitoring components are properly initialized."""
        # Check if monitoring is running
        self.assertTrue(self.monitoring.running)
        
        # Check if metrics collector is initialized
        self.assertIsNotNone(self.monitoring.metrics_collector)
        
        # Check if dashboard server is initialized
        self.assertIsNotNone(self.monitoring.dashboard_server)
        
        # Check if sources are properly set
        self.assertEqual(self.monitoring.load_balancer, self.coordinator.load_balancer)
        self.assertEqual(self.monitoring.coordinator, self.coordinator)
    
    def test_02_dashboard_api_access(self):
        """Test access to dashboard API endpoints."""
        # Check API status endpoint
        response = requests.get(f"{self.dashboard_url}/api/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("version", data)
        
        # Check dashboard metrics endpoint
        response = requests.get(f"{self.dashboard_url}/api/metrics/system")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        
        # Check workers endpoint
        response = requests.get(f"{self.dashboard_url}/api/workers")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("workers", data)
        self.assertGreaterEqual(len(data["workers"]), 3)  # Should have our 3 test workers
    
    def test_03_metrics_collection(self):
        """Test that metrics are being collected and stored."""
        # Add some tasks to generate metrics
        self._add_test_tasks()
        
        # Wait for metrics collection
        time.sleep(5)
        
        # Check system metrics
        response = requests.get(f"{self.dashboard_url}/api/metrics/system")
        data = response.json()
        
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        
        metrics = data["metrics"]
        self.assertIn("total_workers", metrics)
        self.assertIn("active_workers", metrics)
        self.assertIn("queued_tasks", metrics)
        self.assertIn("running_tasks", metrics)
        
        # Verify worker count
        self.assertEqual(metrics["total_workers"], 3)
        
        # Check worker metrics
        response = requests.get(f"{self.dashboard_url}/api/metrics/workers")
        data = response.json()
        
        self.assertTrue(data["success"])
        self.assertIn("workers", data)
        
        workers = data["workers"]
        self.assertEqual(len(workers), 3)
        
        for worker_id, worker_metrics in workers.items():
            self.assertIn("status", worker_metrics)
            self.assertIn("cpu_utilization", worker_metrics)
    
    def test_04_websocket_real_time_updates(self):
        """Test real-time updates via WebSocket connection."""
        # Connect to dashboard WebSocket
        ws_url = f"ws://{self.dashboard_host}:{self.dashboard_port}/ws"
        ws = websocket.create_connection(ws_url)
        
        # Subscribe to system metrics
        ws.send(json.dumps({
            "action": "subscribe",
            "channel": "system_metrics"
        }))
        
        # Add some tasks to generate metrics
        task_ids = self._add_test_tasks()
        
        # Wait for a response
        ws.settimeout(5)
        response = ws.recv()
        data = json.loads(response)
        
        # Verify response structure
        self.assertIn("channel", data)
        self.assertEqual(data["channel"], "system_metrics")
        self.assertIn("data", data)
        
        # Verify metrics data
        metrics = data["data"]
        self.assertIn("timestamp", metrics)
        self.assertIn("metrics", metrics)
        
        system_metrics = metrics["metrics"]
        self.assertIn("total_workers", system_metrics)
        self.assertIn("queued_tasks", system_metrics)
        
        # Close WebSocket connection
        ws.close()
    
    def test_05_anomaly_detection(self):
        """Test anomaly detection in metrics."""
        # Generate a large number of tasks to trigger anomalies
        for i in range(20):
            self.coordinator.add_task(
                "command",
                {"command": ["sleep", "10"]},
                {"hardware": ["cpu"]},
                priority=1
            )
        
        # Wait for anomaly detection
        time.sleep(5)
        
        # Check anomalies endpoint
        response = requests.get(f"{self.dashboard_url}/api/anomalies")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("anomalies", data)
        
        # Verify anomalies are detected (high queue time or worker overload)
        anomalies = data["anomalies"]
        self.assertGreater(len(anomalies), 0, "No anomalies detected")
        
        # Check for specific anomaly types
        anomaly_types = [anomaly["type"] for anomaly in anomalies]
        self.assertTrue(
            any(atype in anomaly_types for atype in ["system_high_queue_time", "worker_overload", "worker_queue_depth"]),
            "Expected anomaly types not found"
        )
    
    def test_06_task_tracking(self):
        """Test task tracking in the dashboard."""
        # Add some tasks
        task_ids = self._add_test_tasks()
        
        # Wait for tasks to be processed
        time.sleep(5)
        
        # Check tasks endpoint
        response = requests.get(f"{self.dashboard_url}/api/tasks")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("tasks", data)
        
        tasks = data["tasks"]
        
        # Verify our test tasks are tracked
        for task_id in task_ids:
            self.assertIn(task_id, [task["task_id"] for task in tasks])
        
        # Check task details endpoint for one task
        response = requests.get(f"{self.dashboard_url}/api/tasks/{task_ids[0]}")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("task", data)
        
        task = data["task"]
        self.assertEqual(task["task_id"], task_ids[0])
        self.assertIn("metrics", task)
        self.assertIn("status", task)
    
    def test_07_worker_performance_scoring(self):
        """Test worker performance scoring in the dashboard."""
        # Wait for enough metrics to be collected
        time.sleep(10)
        
        # Check worker performance endpoint
        response = requests.get(f"{self.dashboard_url}/api/workers/performance")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("workers", data)
        
        workers = data["workers"]
        self.assertEqual(len(workers), 3)
        
        for worker in workers:
            self.assertIn("worker_id", worker)
            self.assertIn("performance_score", worker)
            self.assertGreaterEqual(worker["performance_score"], 0)
            self.assertLessEqual(worker["performance_score"], 100)
    
    def test_08_historical_metrics(self):
        """Test retrieval of historical metrics."""
        # Add tasks and wait for metrics collection
        self._add_test_tasks()
        time.sleep(10)
        
        # Check historical system metrics endpoint
        response = requests.get(
            f"{self.dashboard_url}/api/metrics/history/system",
            params={
                "metrics": "queued_tasks,running_tasks,active_workers",
                "interval": "1s",
                "duration": "10s"
            }
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        
        metrics = data["metrics"]
        self.assertIn("queued_tasks", metrics)
        self.assertIn("running_tasks", metrics)
        self.assertIn("active_workers", metrics)
        
        # Each metric should have time-series data
        for metric_name, metric_data in metrics.items():
            self.assertGreater(len(metric_data), 0, f"No time-series data for {metric_name}")
            for point in metric_data:
                self.assertEqual(len(point), 2, "Time-series point should have timestamp and value")
    
    def test_09_load_balancer_monitoring_integration(self):
        """Test that load balancer events are properly monitored."""
        # Add tasks for different hardware
        cpu_task = self.coordinator.add_task(
            "command",
            {"command": ["echo", "CPU task"]},
            {"hardware": ["cpu"]},
            priority=1
        )
        
        gpu_task = self.coordinator.add_task(
            "command",
            {"command": ["echo", "GPU task"]},
            {"hardware": ["cuda"]},
            priority=1
        )
        
        browser_task = self.coordinator.add_task(
            "command",
            {"command": ["echo", "Browser task"]},
            {"hardware": ["webgpu"]},
            priority=1
        )
        
        # Wait for load balancer to process tasks
        time.sleep(5)
        
        # Check load balancer metrics endpoint
        response = requests.get(f"{self.dashboard_url}/api/metrics/load_balancer")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        
        # Verify load balancer metrics
        metrics = data["metrics"]
        self.assertIn("worker_utilization", metrics)
        self.assertIn("assignment_efficiency", metrics)
        self.assertIn("task_assignments", metrics)
        
        # Check task assignments
        response = requests.get(f"{self.dashboard_url}/api/tasks")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        tasks = data["tasks"]
        
        # Find task assignments
        cpu_task_data = next((t for t in tasks if t["task_id"] == cpu_task), None)
        gpu_task_data = next((t for t in tasks if t["task_id"] == gpu_task), None)
        browser_task_data = next((t for t in tasks if t["task_id"] == browser_task), None)
        
        # Verify assignments match worker capabilities
        self.assertIsNotNone(cpu_task_data)
        self.assertIsNotNone(gpu_task_data)
        self.assertIsNotNone(browser_task_data)
        
        if gpu_task_data.get("worker_id"):
            self.assertEqual(gpu_task_data["worker_id"], "test_worker_1", 
                          "GPU task should be assigned to worker with CUDA")
            
        if browser_task_data.get("worker_id"):
            self.assertEqual(browser_task_data["worker_id"], "test_worker_2", 
                          "WebGPU task should be assigned to worker with browser capabilities")
    
    def test_10_dashboard_html_interface(self):
        """Test that the dashboard HTML interface is accessible."""
        # Access the main dashboard page
        response = requests.get(f"{self.dashboard_url}/")
        self.assertEqual(response.status_code, 200)
        
        # Verify content type
        self.assertIn("text/html", response.headers["Content-Type"])
        
        # Check for key HTML elements
        html_content = response.text.lower()
        self.assertIn("dashboard", html_content)
        self.assertIn("metrics", html_content)
        self.assertIn("workers", html_content)
        self.assertIn("tasks", html_content)
    
    def _add_test_tasks(self):
        """
        Add test tasks to generate metrics.
        
        Returns:
            List of task IDs
        """
        task_ids = []
        
        # Add CPU tasks
        for i in range(5):
            task_id = self.coordinator.add_task(
                "command",
                {"command": ["echo", f"CPU task {i}"]},
                {"hardware": ["cpu"]},
                priority=i+1
            )
            task_ids.append(task_id)
        
        # Add GPU task
        task_id = self.coordinator.add_task(
            "command",
            {"command": ["echo", "GPU task"]},
            {"hardware": ["cuda"]},
            priority=1
        )
        task_ids.append(task_id)
        
        # Add WebGPU task
        task_id = self.coordinator.add_task(
            "command",
            {"command": ["echo", "WebGPU task"]},
            {"hardware": ["webgpu"]},
            priority=1
        )
        task_ids.append(task_id)
        
        # Add high memory task
        task_id = self.coordinator.add_task(
            "command",
            {"command": ["echo", "High memory task"]},
            {"hardware": ["cpu"], "min_memory_gb": 12},
            priority=1
        )
        task_ids.append(task_id)
        
        return task_ids


if __name__ == "__main__":
    unittest.main()